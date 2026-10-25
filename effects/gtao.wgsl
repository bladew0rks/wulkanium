//! effect: Ambient Occlusion (GTAO)
//! order: 100
//! texture: vdepth r32float depth
//! texture: term rg8unorm depth
//! texture: term2 rg8unorm depth
//! texture: ao r8unorm depth
//! pass: prefilter -> vdepth
//! pass: main -> term
//! pass: denoise -> term2
//! pass: denoise_last -> ao
//! pass: apply -> screen
//! param: radius 32 4 256 int
//! param: quality 2 0 3 int
//! param: final_power 2.2 0.5 5
//! param: max_distance 6000 256 20000 int
//! param: show 0 0 1 int

// a port of Intel's XeGTAO 1.30 (MIT, https://github.com/GameTechDev/XeGTAO), itself based on
// "Practical Real-Time Strategies for Accurate Indirect Occlusion" (Jimenez et al. 2016).
// differences from the original: fragment passes instead of compute, no depth mip chain (every
// sample reads full resolution depth), and the result multiplies the finished image. everything
// up to the apply pass runs at the depth buffer's resolution (below the screen's under dlss
// upscaling), so positions always match their depth texel.
//
// passes: linear depth -> occlusion term + depth edges -> two edge-aware denoise passes -> apply.

const PI: f32 = 3.1415926535897932;
const HALF_PI: f32 = 1.5707963267948966;
const RADIUS_MULTIPLIER: f32 = 1.457;
const FALLOFF_RANGE: f32 = 0.615;
const SAMPLE_DISTRIBUTION_POWER: f32 = 2.0;
// the raw term can overshoot 1 before denoising; stored scaled down to fit unorm
const OCCLUSION_TERM_SCALE: f32 = 1.5;
const DENOISE_BLUR_BETA: f32 = 1.2;
const SKY_DEPTH: f32 = 1e6;

// ---- view space as XeGTAO defines it: x right, y up, z = distance along the view axis ----

fn tan_half_fov() -> vec2f {
    return vec2f(1.0 / wk.proj[0][0], 1.0 / wk.proj[1][1]);
}

/// screen uv (pixel centre, depth-buffer jitter removed) and view z -> view position
fn view_pos(uv: vec2f, z: f32) -> vec3f {
    let t = tan_half_fov();
    let s = uv - vec2f(wk.jitter.x * 0.5, -wk.jitter.y * 0.5);
    return vec3f((vec2f(t.x * 2.0, -t.y * 2.0) * s + vec2f(-t.x, t.y)) * z, z);
}

fn size() -> vec2i {
    return vec2i(textureDimensions(vdepth));
}

fn load_z(p: vec2i) -> f32 {
    return textureLoad(vdepth, clamp(p, vec2i(0), size() - 1), 0).r;
}

fn centre_uv(p: vec2i) -> vec2f {
    return (vec2f(p) + 0.5) / vec2f(size());
}

// ---- pass 1: linear view depth ----

@fragment
fn prefilter(i: WkVsOut) -> @location(0) vec4f {
    let d = wk_depth(i.uv);
    if !wk_has_camera() || wk_is_sky(d) {
        return vec4f(SKY_DEPTH);
    }
    // the view axis distance, whatever the projection's handedness
    return vec4f(min(abs(wk_view_pos(wk_snap(i.uv), d).z), SKY_DEPTH));
}

// ---- pass 2: occlusion term and edges ----

fn calculate_edges(c: f32, l: f32, r: f32, t: f32, b: f32) -> vec4f {
    var e = vec4f(l, r, t, b) - c;
    let slope_lr = (e.y - e.x) * 0.5;
    let slope_tb = (e.w - e.z) * 0.5;
    let adjusted = e + vec4f(slope_lr, -slope_lr, slope_tb, -slope_tb);
    e = min(abs(e), abs(adjusted));
    return saturate(1.25 - e / (c * 0.011));
}

/// 2 bits per edge
fn pack_edges(e: vec4f) -> f32 {
    let q = round(saturate(e) * 2.9);
    return dot(q, vec4f(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0));
}

fn unpack_edges(v: f32) -> vec4f {
    let p = u32(v * 255.5);
    return saturate(vec4f(f32((p >> 6u) & 3u), f32((p >> 4u) & 3u), f32((p >> 2u) & 3u), f32(p & 3u)) / 3.0);
}

fn calculate_normal(e: vec4f, c: vec3f, l: vec3f, r: vec3f, t: vec3f, b: vec3f) -> vec3f {
    let accepted = saturate(vec4f(e.x * e.z, e.z * e.y, e.y * e.w, e.w * e.x) + 0.01);
    let ld = normalize(l - c);
    let rd = normalize(r - c);
    let td = normalize(t - c);
    let bd = normalize(b - c);
    let n = accepted.x * cross(ld, td) + accepted.y * cross(td, rd) + accepted.z * cross(rd, bd) + accepted.w * cross(bd, ld);
    return normalize(n);
}

fn fast_acos(x: f32) -> f32 {
    let a = abs(x);
    var res = -0.156583 * a + HALF_PI;
    res *= sqrt(max(1.0 - a, 0.0));
    return select(PI - res, res, x >= 0.0);
}

fn hilbert_index(px: u32, py: u32) -> u32 {
    var x = px & 63u;
    var y = py & 63u;
    var index = 0u;
    for (var level = 32u; level > 0u; level /= 2u) {
        let rx = select(0u, 1u, (x & level) > 0u);
        let ry = select(0u, 1u, (y & level) > 0u);
        index += level * level * ((3u * rx) ^ ry);
        if ry == 0u {
            if rx == 1u {
                x = 63u - x;
                y = 63u - y;
            }
            let tmp = x;
            x = y;
            y = tmp;
        }
    }
    return index;
}

/// hilbert curve driving the r2 sequence; no temporal part, since nothing accumulates frames
fn spatial_noise(p: vec2u) -> vec2f {
    let index = f32(hilbert_index(p.x, p.y));
    return fract(0.5 + index * vec2f(0.75487766624669276, 0.56984029099805327));
}

@fragment
fn main(i: WkVsOut) -> @location(0) vec4f {
    let p = vec2i(i.pos.xy);
    let z_raw = load_z(p);
    let zl = load_z(p + vec2i(-1, 0));
    let zr = load_z(p + vec2i(1, 0));
    let zt = load_z(p + vec2i(0, -1));
    let zb = load_z(p + vec2i(0, 1));
    let edges = calculate_edges(z_raw, zl, zr, zt, zb);
    let packed = pack_edges(edges);
    let depth = wk_depth(i.uv);
    if z_raw >= SKY_DEPTH || wk_is_viewmodel(depth) || z_raw > params.max_distance {
        return vec4f(1.0 / OCCLUSION_TERM_SCALE, packed, 0.0, 0.0);
    }
    let uv = centre_uv(p);
    let px = 1.0 / vec2f(size());
    let n = calculate_normal(edges, view_pos(uv, z_raw), view_pos(uv - vec2f(px.x, 0.0), zl), view_pos(uv + vec2f(px.x, 0.0), zr), view_pos(uv - vec2f(0.0, px.y), zt), view_pos(uv + vec2f(0.0, px.y), zb));

    // move the centre slightly towards the camera against depth imprecision (fp32 depth)
    let z = z_raw * 0.99999;
    let centre = view_pos(uv, z);
    let v = normalize(-centre);

    let q = u32(params.quality);
    let slices = array<f32, 4>(1.0, 2.0, 3.0, 9.0)[q];
    let steps = array<f32, 4>(2.0, 2.0, 3.0, 3.0)[q];
    let effect_radius = params.radius * RADIUS_MULTIPLIER;
    let falloff_range = FALLOFF_RANGE * effect_radius;
    let falloff_from = effect_radius * (1.0 - FALLOFF_RANGE);
    let falloff_mul = -1.0 / falloff_range;
    let falloff_add = falloff_from / falloff_range + 1.0;

    let noise = spatial_noise(vec2u(p));
    // view-space size of one pixel at this depth
    let t = tan_half_fov();
    let pixel_size = z * t.x * 2.0 * px.x;
    let screen_radius = effect_radius / pixel_size;
    var visibility = saturate((10.0 - screen_radius) / 100.0) * 0.5;
    // never sample this pixel itself
    let min_s = 1.3 / screen_radius;

    for (var slice = 0.0; slice < slices; slice += 1.0) {
        let phi = (slice + noise.x) / slices * PI;
        let cos_phi = cos(phi);
        let sin_phi = sin(phi);
        let omega = vec2f(cos_phi, -sin_phi) * screen_radius;
        let direction = vec3f(cos_phi, sin_phi, 0.0);
        let ortho = direction - dot(direction, v) * v;
        let axis = normalize(cross(ortho, v));
        let projected = n - axis * dot(n, axis);
        let sign_n = sign(dot(ortho, projected));
        var projected_len = length(projected);
        let cos_n = saturate(dot(projected, v) / projected_len);
        let nn = sign_n * fast_acos(cos_n);
        let low0 = cos(nn + HALF_PI);
        let low1 = cos(nn - HALF_PI);
        var horizon0 = low0;
        var horizon1 = low1;
        for (var step = 0.0; step < steps; step += 1.0) {
            let step_noise = fract(noise.y + (slice + step * steps) * 0.6180339887498948);
            var s = (step + step_noise) / steps;
            s = pow(s, SAMPLE_DISTRIBUTION_POWER) + min_s;
            // snap to pixel centres so the depth matches the position
            let offset = round(s * omega) * px;
            let uv0 = uv + offset;
            let uv1 = uv - offset;
            let pos0 = view_pos(uv0, load_z(vec2i(floor(uv0 * vec2f(size())))));
            let pos1 = view_pos(uv1, load_z(vec2i(floor(uv1 * vec2f(size())))));
            let delta0 = pos0 - centre;
            let delta1 = pos1 - centre;
            let dist0 = length(delta0);
            let dist1 = length(delta1);
            let weight0 = saturate(dist0 * falloff_mul + falloff_add);
            let weight1 = saturate(dist1 * falloff_mul + falloff_add);
            let shc0 = mix(low0, dot(delta0 / dist0, v), weight0);
            let shc1 = mix(low1, dot(delta1 / dist1, v), weight1);
            horizon0 = max(horizon0, shc0);
            horizon1 = max(horizon1, shc1);
        }
        // the original's fudge against slight over-darkening on steep slopes
        projected_len = mix(projected_len, 1.0, 0.05);
        let h0 = -fast_acos(horizon1);
        let h1 = fast_acos(horizon0);
        let iarc0 = (cos_n + 2.0 * h0 * sin(nn) - cos(2.0 * h0 - nn)) / 4.0;
        let iarc1 = (cos_n + 2.0 * h1 * sin(nn) - cos(2.0 * h1 - nn)) / 4.0;
        visibility += projected_len * (iarc0 + iarc1);
    }
    visibility /= slices;
    visibility = pow(visibility, params.final_power);
    visibility = max(0.03, visibility);
    return vec4f(saturate(visibility / OCCLUSION_TERM_SCALE), packed, 0.0, 0.0);
}

// ---- passes 3, 4: edge-aware denoise ----

fn load_term(tex: texture_2d<f32>, p: vec2i) -> vec2f {
    return textureLoad(tex, clamp(p, vec2i(0), size() - 1), 0).rg;
}

fn denoise_at(tex: texture_2d<f32>, p: vec2i, last: bool) -> vec2f {
    let blur = select(DENOISE_BLUR_BETA / 5.0, DENOISE_BLUR_BETA, last);
    let diag = 0.85 * 0.5;
    let c = load_term(tex, p);
    let l = load_term(tex, p + vec2i(-1, 0));
    let r = load_term(tex, p + vec2i(1, 0));
    let t = load_term(tex, p + vec2i(0, -1));
    let b = load_term(tex, p + vec2i(0, 1));
    let el = unpack_edges(l.y);
    let er = unpack_edges(r.y);
    let et = unpack_edges(t.y);
    let eb = unpack_edges(b.y);
    // a left edge here should match the right edge of the left neighbour, and so on
    var ec = unpack_edges(c.y) * vec4f(el.y, er.x, et.w, eb.z);
    // let a little ao leak in when there are 3 or 4 edges (thin features)
    let leak_threshold = 2.5;
    let leak_strength = 0.5;
    let edginess = (saturate(4.0 - leak_threshold - dot(ec, vec4f(1.0))) / (4.0 - leak_threshold)) * leak_strength;
    ec = saturate(ec + edginess);
    let w_tl = diag * (ec.x * el.z + ec.z * et.x);
    let w_tr = diag * (ec.z * et.y + ec.y * er.z);
    let w_bl = diag * (ec.w * eb.x + ec.x * el.w);
    let w_br = diag * (ec.y * er.w + ec.w * eb.y);
    var sum = c.x * blur;
    var weight = blur;
    sum += l.x * ec.x;
    sum += r.x * ec.y;
    sum += t.x * ec.z;
    sum += b.x * ec.w;
    sum += load_term(tex, p + vec2i(-1, -1)).x * w_tl;
    sum += load_term(tex, p + vec2i(1, -1)).x * w_tr;
    sum += load_term(tex, p + vec2i(-1, 1)).x * w_bl;
    sum += load_term(tex, p + vec2i(1, 1)).x * w_br;
    weight += ec.x + ec.y + ec.z + ec.w + w_tl + w_tr + w_bl + w_br;
    return vec2f(sum / weight, c.y);
}

@fragment
fn denoise(i: WkVsOut) -> @location(0) vec4f {
    return vec4f(denoise_at(term, vec2i(i.pos.xy), false), 0.0, 0.0);
}

@fragment
fn denoise_last(i: WkVsOut) -> @location(0) vec4f {
    return vec4f(saturate(denoise_at(term2, vec2i(i.pos.xy), true).x * OCCLUSION_TERM_SCALE));
}

// ---- pass 5: apply ----

@fragment
fn apply(i: WkVsOut) -> @location(0) vec4f {
    let color = textureSampleLevel(wk_color, wk_point, i.uv, 0.0);
    // ao is at depth resolution
    var occlusion = textureLoad(ao, wk_pixel(i.uv), 0).r;
    if !wk_has_camera() {
        occlusion = 1.0;
    }
    if params.show > 0.5 {
        return vec4f(vec3f(occlusion), 1.0);
    }
    return vec4f(color.rgb * occlusion, color.a);
}
