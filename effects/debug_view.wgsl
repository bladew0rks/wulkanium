//! effect: Debug View
//! order: 1000
//! pass: show -> screen
//! param: mode 0 0 4 int
//! param: grid 64 1 1024 int

// mode 0: distance (white = far), 1: world normal, 2: world-space grid lines every `grid` units,
// 3: motion vectors (dlss frames only), 4: the scene color untouched

@fragment
fn show(i: WkVsOut) -> @location(0) vec4f {
    let color = textureSampleLevel(wk_color, wk_point, i.uv, 0.0);
    let depth = wk_depth(i.uv);
    let mode = u32(params.mode + 0.5);
    if mode == 4u || !wk_has_camera() {
        return color;
    }
    if wk_is_sky(depth) {
        return vec4f(0.2, 0.3, 0.5, 1.0);
    }
    if mode == 0u {
        return vec4f(vec3f(1.0 - exp(-wk_distance(i.uv) / 1024.0)), 1.0);
    }
    if mode == 1u {
        return vec4f(wk_world_normal(i.uv) * 0.5 + 0.5, 1.0);
    }
    if mode == 2u {
        let p = wk_world_pos_at(i.uv) / params.grid;
        let f = abs(fract(p - 0.5) - 0.5) / fwidth(p);
        let line = 1.0 - min(min(f.x, f.y), f.z);
        return vec4f(mix(color.rgb * 0.5, vec3f(1.0, 0.8, 0.2), saturate(line)), 1.0);
    }
    let mv = wk_motion_uv(i.uv) * 50.0;
    return vec4f(abs(mv), 0.0, 1.0);
}
