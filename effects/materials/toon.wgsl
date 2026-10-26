//! shader: Toon
//! param: $bands float 3 "number of light bands"
//! param: $shadowtint color [0.35 0.35 0.5] "colour of the darkest band"
//! param: $rim float 0.4 "rim light strength"

// cel shading: source's diffuse lighting cut into flat bands, plus a rim light

fn shade(s: Surface) -> vec4f {
    let m = material();
    let base = base_texture(s.uv) * tint();
    let light = diffuse_lighting(s, s.normal);
    let level = max(max(light.r, light.g), light.b);
    let band = floor(level * m.bands + 0.5) / m.bands;
    let hue = light / max(level, 1e-4);
    let lit = mix(m.shadowtint, vec3f(1.0), saturate(band)) * hue * max(band, 0.15);
    let rim = pow(1.0 - saturate(dot(s.normal, s.view)), 4.0) * m.rim;
    return vec4f(base.rgb * lit + rim * level, base.a);
}
