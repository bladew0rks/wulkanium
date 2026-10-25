//! effect: Sharpen (RCAS)
//! order: 900
//! pass: sharpen -> screen
//! param: sharpness 0.5 0 1

// robust contrast adaptive sharpening, after the algorithm in AMD FidelityFX FSR 1 (MIT).
// sharpens a cross of 5 pixels, limited so it never pushes a pixel past its neighbours.

const LIMIT: f32 = 0.1875; // 0.25 - 1/16

fn tap(uv: vec2f, dx: f32, dy: f32) -> vec3f {
    return textureSampleLevel(wk_color, wk_point, uv + vec2f(dx, dy) * wk.resolution.zw, 0.0).rgb;
}

fn luma(c: vec3f) -> f32 {
    return c.g + 0.5 * (c.r + c.b);
}

@fragment
fn sharpen(i: WkVsOut) -> @location(0) vec4f {
    let b = tap(i.uv, 0.0, -1.0);
    let d = tap(i.uv, -1.0, 0.0);
    let e = tap(i.uv, 0.0, 0.0);
    let f = tap(i.uv, 1.0, 0.0);
    let h = tap(i.uv, 0.0, 1.0);
    let mn = min(min(b, d), min(f, h));
    let mx = max(max(b, d), max(f, h));
    // the most negative lobe that keeps every channel in range
    let hit_min = mn / (4.0 * mx + 1e-5);
    let hit_max = (1.0 - mx) / (4.0 * mn - 4.0 - 1e-5);
    let lobe3 = max(-hit_min, hit_max);
    var lobe = max(-LIMIT, min(max(lobe3.r, max(lobe3.g, lobe3.b)), 0.0)) * params.sharpness;
    // back off on noise: a lone pixel against flat neighbours
    let bl = luma(b);
    let dl = luma(d);
    let el = luma(e);
    let fl = luma(f);
    let hl = luma(h);
    let range = max(max(max(bl, dl), max(fl, hl)), el) - min(min(min(bl, dl), min(fl, hl)), el);
    let nz = saturate(abs(0.25 * (bl + dl + fl + hl) - el) / max(range, 1e-5));
    lobe *= 1.0 - 0.5 * nz;
    let c = (lobe * (b + d + f + h) + e) / (4.0 * lobe + 1.0);
    return vec4f(c, 1.0);
}
