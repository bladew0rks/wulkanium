//! effect: Contact Shadows
//! order: 110
//! texture: mask r8unorm depth
//! pass: trace -> mask
//! pass: apply -> screen
//! param: strength 0.6 0 1
//! param: world 1 0 1 int
//! param: sun_thickness 0.005 0.0005 0.05
//! param: sun_contrast 4 1 8
//! param: all_casters 0 0 1 int
//! param: ignore_edges 0 0 1 int
//! param: viewmodel 1 0 1 int
//! param: vm_length 4 0.5 16
//! param: vm_thickness 1.5 0.1 8
//! param: bias 1 0 4
//! param: steps 24 4 64 int
//! param: blur 1 0 1 int
//! param: show 0 0 1 int

// short screen-space shadows (the usual "contact shadows" idea).
//
// world pixels use the sun shadow pass (wk_sun_shadow: bend studio's screen space shadows, run
// natively as compute): rays walk towards the sun's point on screen in depth-buffer space, one
// pixel at a time, up to 60 pixels. by default only moving things cast (props, players, npcs):
// brushes and static props are already shadowed in the lightmaps and would double up
// (all_casters 1 lets everything cast). sun_thickness is how thick each depth pixel is assumed
// to be, as a fraction of the depth left to the far plane; sun_contrast sharpens the edges.
//
// viewmodel pixels march towards the viewmodel's lights (blended into one direction, fading out when
// they come from opposite sides), in its own view space, since its depth lives in a different
// space. the two never shadow each other.
//
// the viewmodel uses no normals (rebuilt from depth they come out faceted on its curved
// surfaces); world pixels facing away from the sun fade out (flat brush faces make good normals).

/// interleaved gradient noise, moving each frame (dlss / taa average it out)
fn noise(p: vec2f) -> f32 {
    let q = p + f32(wk.frame % 64u) * vec2f(5.588238, 5.588238);
    return fract(52.9829189 * fract(dot(q, vec2f(0.06711056, 0.00583715))));
}

/// soft hit: the occluder is in front of the ray by more than the bias, but not by more than a
/// thin object would be
fn hit(gap: f32, bias: f32, thickness: f32) -> f32 {
    let front = saturate((gap - bias) / max(bias, 1e-3));
    let thin = saturate((thickness - gap) / (thickness * 0.5));
    return front * thin;
}

fn trace_world(px: vec2i, d: f32) -> f32 {
    if params.world < 0.5 || !wk_has_sun_shadow() || wk_is_sky(d) { return 0.0; }
    let uv = (vec2f(px) + 0.5) / wk.depth_size;
    // facing away from the sun: no sunlight to block
    let facing = saturate(dot(wk_world_normal(uv), wk_sun_dir()) * 5.0 + 0.5);
    return (1.0 - wk_sun_shadow(px)) * facing;
}

fn trace_viewmodel(px: vec2i, d: f32) -> f32 {
    if params.viewmodel < 0.5 { return 0.0; }
    let uv = (vec2f(px) + 0.5) / wk.depth_size;
    let p = wk_vm_view_pos(uv, d);
    let l = wk_vm_light_dir(p);
    if dot(l, l) < 0.5 { return 0.0; }
    let steps = i32(params.steps);
    let jitter = noise(vec2f(px));
    var occ = 0.0;
    for (var k = 0; k < steps; k++) {
        let f = (f32(k) + jitter) / f32(steps);
        let t = f * params.vm_length;
        let q = p + l * t;
        let s = wk_vm_project(q);
        if any(s.xy < vec2f(0.0)) || any(s.xy > vec2f(1.0)) { break; }
        let sd = wk_depth(s.xy);
        if !wk_is_viewmodel(sd) { continue; }
        let o = wk_vm_view_pos(wk_snap(s.xy), sd);
        // the bias grows a little with t so sloped surfaces don't shadow themselves through
        // pixel snapping
        let bias = 0.04 * params.bias * (1.0 + t);
        occ = max(occ, hit(abs(q.z) - abs(o.z), bias, params.vm_thickness) * (1.0 - f * f));
        if occ > 0.99 { break; }
    }
    // several lights from different sides: no single shadow direction to trust
    return occ * saturate((wk_vm_light_agreement() - 0.3) / 0.5);
}

@fragment
fn trace(i: WkVsOut) -> @location(0) vec4f {
    let px = wk_pixel(i.uv);
    let d = textureLoad(wk_depth_tex, px, 0);
    if wk_is_viewmodel(d) {
        if !wk_has_viewmodel() { return vec4f(1.0); }
        return vec4f(1.0 - trace_viewmodel(px, d));
    }
    return vec4f(1.0 - trace_world(px, d));
}

/// distance along the view axis, in whichever space the pixel lives
fn linear_z(px: vec2i, d: f32) -> f32 {
    let uv = (vec2f(px) + 0.5) / wk.depth_size;
    if wk_is_viewmodel(d) { return abs(wk_vm_view_pos(uv, d).z); }
    return abs(wk_view_pos(uv, d).z);
}

@fragment
fn apply(i: WkVsOut) -> @location(0) vec4f {
    let color = textureSampleLevel(wk_color, wk_point, i.uv, 0.0);
    let px = wk_pixel(i.uv);
    let d = textureLoad(wk_depth_tex, px, 0);
    // sky, and water drawn over the ground
    if wk_is_sky(d) || (!wk_is_viewmodel(d) && wk_covered(i.uv)) {
        return select(color, vec4f(1.0), params.show > 0.5);
    }
    var lit = textureLoad(mask, px, 0).r;
    if params.blur > 0.5 {
        // 5x5, only over pixels of the same kind (world / viewmodel) at a similar depth
        let size = vec2i(wk.depth_size) - 1;
        let vm = wk_is_viewmodel(d);
        let z = linear_z(px, d);
        var sum = 0.0;
        var wsum = 0.0;
        for (var y = -2; y <= 2; y++) {
            for (var x = -2; x <= 2; x++) {
                let q = clamp(px + vec2i(x, y), vec2i(0), size);
                let qd = textureLoad(wk_depth_tex, q, 0);
                if wk_is_viewmodel(qd) != vm || wk_is_sky(qd) { continue; }
                let w = saturate(1.0 - abs(linear_z(q, qd) - z) / (0.02 * z + 0.1)) * exp(-f32(x * x + y * y) / 4.0);
                sum += textureLoad(mask, q, 0).r * w;
                wsum += w;
            }
        }
        lit = sum / max(wsum, 1e-4);
    }
    lit = 1.0 - params.strength * (1.0 - lit);
    if params.show > 0.5 {
        return vec4f(vec3f(lit), 1.0);
    }
    return vec4f(color.rgb * lit, color.a);
}
