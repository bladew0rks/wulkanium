//! shaders written here rather than translated from the game's bytecode: the ones the shader dll
//! (src/shaderdll.rs) sets, named `wk_*`. they use the same interface as translated shaders
//! (constant blocks, vertex inputs by usage, sampler bindings, clip / motion vector epilogue), so
//! the renderer treats them like any other.

use std::fmt::Write;

use crate::d3d2wgsl::{PS_FLOAT_CONSTS, Packed, PsOptions, Translated, VS_FLOAT_CONSTS};

pub const PREFIX: &str = "wk_";

/// SurfaceGGX static combo bits (vertex and pixel shader share them)
pub mod ggx {
    pub const BUMP: u32 = 1 << 0;
    /// $phongexponenttexture: inverted r roughness, g metalness, b fuzz mask
    pub const MASK: u32 = 1 << 1;
    /// $pbr + $masks: rgb f0, a roughness
    pub const PBR: u32 = 1 << 2;
    pub const METAL: u32 = 1 << 3;
    pub const ENVMAP: u32 = 1 << 4;
    pub const DETAIL_NORMAL: u32 = 1 << 5;
    pub const SSS: u32 = 1 << 6;
    pub const FUZZ: u32 = 1 << 7;
    /// output base alpha (translucent / alpha tested)
    pub const ALPHA: u32 = 1 << 8;

    pub const VS: &str = "wk_surfaceggx_vs";
    pub const PS: &str = "wk_surfaceggx_ps";

    /// pixel shader constant registers: c0 tint rgb (linear) + alpha, c1.x roughness multiplier,
    /// c2 eye position, c3 envmap tint, c4 (unused, fuzz strength, detail normal blend, detail
    /// normal scale), c5 diffusion weights + light count, c6 back diffusion weights, c7..c12
    /// ambient cube, c33..c52 lights (vertex shader layout), c53 fog params
    pub const C_TINT: i32 = 0;
    pub const C_SURFACE: i32 = 1;
    pub const C_EYE: i32 = 2;
    pub const C_ENV: i32 = 3;
    pub const C_EXTRA: i32 = 4;
    pub const C_DIFFUSION: i32 = 5;
    pub const C_DIFFUSION2: i32 = 6;
    pub const C_AMBIENT: i32 = 7;
    /// c29 fog color and c30 tonemap scale are kept by the shaderapi
    pub const C_LIGHTS: i32 = 33;
    pub const C_FOG: i32 = 53;
}

/// `name/static+dynamic` as the renderer labels shaders
fn parse_label(label: &str) -> (&str, u32) {
    let (name, rest) = label.split_once('/').unwrap_or((label, "0+0"));
    let st = rest.split('+').next().and_then(|v| v.parse().ok()).unwrap_or(0);
    (name, st)
}

const VCONSTS: &str = "struct VConsts {\n    i: array<vec4<i32>, 16>,\n    b: vec4<u32>,\n    hp: vec4f,\n    clip: array<vec4f, 6>,\n    mvm: array<vec4f, 4>,\n    pvp: array<vec4f, 4>,\n    obj: vec4f,\n    c: array<vec4f, VSN>,\n}\n@group(0) @binding(0) var<uniform> V: VConsts;\n";

pub fn translate_vs(label: &str, present: &dyn Fn(u8, u8) -> bool, packed: &dyn Fn(u8, u8) -> Packed, clip: bool, mv: bool) -> Result<Translated, String> {
    let (name, _st) = parse_label(label);
    if name != ggx::VS {
        return Err(format!("unknown built-in shader {name}"));
    }
    let mut out = String::new();
    if clip {
        out += "enable clip_distances;\n";
    }
    out += &VCONSTS.replace("VSN", &(VS_FLOAT_CONSTS + 3 * crate::api::MAX_BONES).to_string());
    // (name, usage, index, default)
    let wanted: [(&str, u8, u8, &str); 9] = [
        ("pos", 0, 0, "vec4f(0.0, 0.0, 0.0, 1.0)"),
        ("weights", 1, 0, "vec4f(1.0, 0.0, 0.0, 0.0)"),
        ("bones", 2, 0, "vec4f(0.0)"),
        ("normal", 3, 0, "vec4f(0.0, 0.0, 1.0, 0.0)"),
        ("uv", 5, 0, "vec4f(0.0, 0.0, 0.0, 1.0)"),
        ("tangent", 6, 0, "vec4f(1.0, 0.0, 0.0, 1.0)"),
        ("static_light", 10, 1, "vec4f(0.0)"),
        ("flex_pos", 0, 1, "vec4f(0.0)"),
        ("flex_normal", 3, 1, "vec4f(0.0)"),
    ];
    let mut inputs = Vec::new();
    let mut fields = String::new();
    let mut loads = String::new();
    // compressed vertices: normal and tangent share one ubyte4, weights are shorts
    let packed_normal = present(3, 0) && packed(3, 0) == Packed::UByte4;
    for (n, usage, index, def) in wanted {
        if present(usage, index) {
            let loc = inputs.len() as u32;
            inputs.push((loc, usage, index));
            let _ = writeln!(fields, "    @location({loc}) {n}: vec4f,");
            match (usage, index, packed(usage, index)) {
                (3, 0, Packed::UByte4) => {
                    let _ = writeln!(loads, "    let nt = unpack_normal_tangent(round(vin.{n} * 255.0));\n    let normal = vec4f(nt.n, 0.0);");
                }
                (1, 0, Packed::Short2) => {
                    let _ = writeln!(loads, "    let {n} = vec4f((round(vin.{n}.xy * 32767.0) + 1.0) / 32768.0, 0.0, 0.0);");
                }
                _ => {
                    let _ = writeln!(loads, "    let {n} = vin.{n};");
                }
            }
        } else if usage == 6 && packed_normal {
            let _ = writeln!(loads, "    let tangent = nt.t;");
        } else {
            let _ = writeln!(loads, "    let {n} = {def};");
        }
    }
    let skinned = present(2, 0);
    let has_static = present(10, 1);
    let flex = present(0, 1);
    let _ = write!(out, "\nstruct VIn {{\n{fields}    @builtin(vertex_index) vid: u32,\n}}\n\nstruct VOut {{\n    @builtin(position) pos: vec4f,\n");
    for l in 0..5 {
        let _ = writeln!(out, "    @location({l}) l{l}: vec4f,");
    }
    if clip {
        out += "    @builtin(clip_distances) clip: array<f32, 6>,\n";
    }
    if mv {
        out += "    @location(12) cur: vec4f,\n    @location(13) prev: vec4f,\n";
    }
    out += "}\n\n";
    out += r#"
struct NormalTangent {
    n: vec3f,
    t: vec4f,
}

/// source's compressed normal + tangent: one ubyte4, each vector as two octahedron-like bytes
/// with the signs folded in (as the stock shaders' DecompressVertex_NormalTangent)
fn unpack_normal_tangent(i: vec4f) -> NormalTangent {
    let zt_sign = select(vec4f(0.0), vec4f(1.0), i - 128.0 < vec4f(0.0));
    let xy_abs = abs(i - 128.0) - zt_sign;
    let xy_sign = select(vec4f(0.0), vec4f(1.0), xy_abs - 64.0 < vec4f(0.0));
    let nt = (abs(xy_abs - 64.0) - xy_sign) / 63.0;
    let xy_signs = 1.0 - 2.0 * xy_sign;
    let zt_signs = 1.0 - 2.0 * zt_sign;
    let n = normalize(vec3f(nt.xy, 1.0 - nt.x - nt.y));
    let t = normalize(vec3f(nt.zw, 1.0 - nt.z - nt.w));
    return NormalTangent(vec3f(n.xy * xy_signs.xy, n.z * zt_signs.x), vec4f(t.xy * xy_signs.zw, t.z * zt_signs.z, zt_signs.w));
}

fn model_row(bone: i32, r: i32) -> vec4f {
    return V.c[58 + 3 * bone + r];
}

fn xform(bone: i32, p: vec4f) -> vec3f {
    return vec3f(dot(p, model_row(bone, 0)), dot(p, model_row(bone, 1)), dot(p, model_row(bone, 2)));
}

@vertex
fn main(vin: VIn) -> VOut {
"#;
    out += &loads;
    out += "    var p = vec4f(pos.xyz, 1.0);\n    var n = vec4f(normal.xyz, 0.0);\n    let t = vec4f(tangent.xyz, 0.0);\n";
    if flex {
        // flex deltas, scaled like the stock shaders do (c3.x)
        out += "    p = vec4f(p.xyz + flex_pos.xyz * V.c[3].x, 1.0);\n    n = vec4f(n.xyz + flex_normal.xyz * V.c[3].x, 0.0);\n";
    }
    if skinned {
        // up to three bones; the indices are the raw bytes of a d3dcolor
        out += r#"    let idx = vec4<i32>(round(bones * 255.0));
    let w = vec3f(weights.x, weights.y, 1.0 - weights.x - weights.y);
    let world = xform(idx.x, p) * w.x + xform(idx.y, p) * w.y + xform(idx.z, p) * w.z;
    let wn = xform(idx.x, n) * w.x + xform(idx.y, n) * w.y + xform(idx.z, n) * w.z;
    let wt = xform(idx.x, t) * w.x + xform(idx.y, t) * w.y + xform(idx.z, t) * w.z;
"#;
    } else {
        out += "    let world = xform(0, p);\n    let wn = xform(0, n);\n    let wt = xform(0, t);\n";
    }
    out += r#"    let w4 = vec4f(world, 1.0);
    let clip_pos = vec4f(dot(w4, V.c[8]), dot(w4, V.c[9]), dot(w4, V.c[10]), dot(w4, V.c[11]));
    let proj_z = dot(w4, V.c[13]);
    let tc = vec4f(uv.xy, 0.0, 1.0);
    var out: VOut;
    out.pos = clip_pos;
    // base texture transform rows (c48, c49)
    out.l0 = vec4f(dot(tc, V.c[48]), dot(tc, V.c[49]), uv.xy);
    out.l1 = vec4f(world, proj_z);
    out.l2 = vec4f(normalize(wn), 0.0);
    out.l3 = vec4f(wt, select(1.0, -1.0, tangent.w < 0.0));
"#;
    if has_static {
        // baked per-vertex lighting (a d3dcolor, bgra in memory), 2x overbright like stock
        out += "    out.l4 = vec4f(static_light.zyx * 2.0, 1.0);\n";
    } else {
        out += "    out.l4 = vec4f(0.0);\n";
    }
    if clip {
        out += "    for (var i = 0; i < 6; i++) {\n        out.clip[i] = dot(V.clip[i], out.pos);\n    }\n";
    }
    if mv {
        out += "    out.cur = out.pos;\n    out.prev = vec4f(dot(out.pos, V.mvm[0]), dot(out.pos, V.mvm[1]), dot(out.pos, V.mvm[2]), dot(out.pos, V.mvm[3]));\n";
        // raw ubyte4 bone indices here, in order
        out += &crate::d3d2wgsl::object_motion("pos", skinned.then_some("vec3f(weights.x, weights.y, 1.0 - weights.x - weights.y)"), skinned.then_some("vec4<i32>(round(bones * 255.0)).xyz"));
    }
    // d3d9 pixel centres, plus dlss jitter (as translated shaders do)
    out += "    out.pos = vec4f(out.pos.x + (V.hp.x + V.hp.z) * out.pos.w, out.pos.y - (V.hp.y - V.hp.w) * out.pos.w, out.pos.zw);\n    return out;\n}\n";
    Ok(Translated { wgsl: out, samplers: Vec::new(), inputs })
}

/// samplers: 0 base, 1 bump, 2 mask, 3 envmap (cube), 4 detail normal
pub fn translate_ps(label: &str, opts: &PsOptions, mv: bool) -> Result<Translated, String> {
    use ggx::*;
    if crate::shaderdll::script::is_script_ps(label) {
        return crate::shaderdll::script::translate_ps(label, opts, mv);
    }
    let (name, st) = parse_label(label);
    if name != PS {
        return Err(format!("unknown built-in shader {name}"));
    }
    let on = |bit: u32| st & bit != 0;
    let mut samplers: Vec<(u32, u8)> = vec![(0, crate::d3d2wgsl::DIM_2D)];
    if on(BUMP) {
        samplers.push((1, crate::d3d2wgsl::DIM_2D));
    }
    if on(MASK) || on(PBR) {
        samplers.push((2, crate::d3d2wgsl::DIM_2D));
    }
    if on(ENVMAP) {
        samplers.push((3, crate::d3d2wgsl::DIM_CUBE));
    }
    if on(DETAIL_NORMAL) {
        samplers.push((4, crate::d3d2wgsl::DIM_2D));
    }
    let mut out = format!("diagnostic(off, derivative_uniformity);\nstruct PConsts {{\n    c: array<vec4f, {PS_FLOAT_CONSTS}>,\n    i: array<vec4<i32>, 16>,\n    b: vec4<u32>,\n}}\n@group(0) @binding(1) var<uniform> P: PConsts;\n");
    for &(n, dim) in &samplers {
        let ty = if dim == crate::d3d2wgsl::DIM_CUBE { "texture_cube<f32>" } else { "texture_2d<f32>" };
        let _ = writeln!(out, "@group(1) @binding({}) var tex{n}: {ty};\n@group(1) @binding({}) var smp{n}: sampler;", 2 * n, 2 * n + 1);
    }
    out += "fn srgb_to_linear(c: vec3f) -> vec3f { return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045)); }\n";
    out += "fn linear_to_srgb(c: vec3f) -> vec3f { let x = clamp(c, vec3f(0.0), vec3f(1.0)); return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3f(0.0031308)); }\n";
    for &(n, _) in &samplers {
        if opts.srgb_read & (1 << n) != 0 {
            let _ = writeln!(out, "fn fetch{n}(c: vec4f) -> vec4f {{ return vec4f(srgb_to_linear(c.rgb), c.a); }}");
        } else {
            let _ = writeln!(out, "fn fetch{n}(c: vec4f) -> vec4f {{ return c; }}");
        }
    }
    out += "\nstruct FIn {\n    @builtin(position) fpos: vec4f,\n    @builtin(front_facing) ff: bool,\n";
    for l in 0..5 {
        let _ = writeln!(out, "    @location({l}) l{l}: vec4f,");
    }
    if mv {
        out += "    @location(12) cur: vec4f,\n    @location(13) prev: vec4f,\n";
    }
    out += "}\n\nstruct FOut {\n    @location(0) color: vec4f,\n";
    if mv {
        out += "    @location(1) mv: vec2f,\n";
    }
    out += "}\n";
    let _ = write!(
        out,
        r#"
const PI: f32 = 3.14159265;

fn ambient(n: vec3f) -> vec3f {{
    let n2 = n * n;
    let x = select(P.c[{a} + 1].rgb, P.c[{a}].rgb, n.x >= 0.0);
    let y = select(P.c[{a} + 3].rgb, P.c[{a} + 2].rgb, n.y >= 0.0);
    let z = select(P.c[{a} + 5].rgb, P.c[{a} + 4].rgb, n.z >= 0.0);
    return n2.x * x + n2.y * y + n2.z * z;
}}

/// distance and spot attenuation, as the stock vertex shaders compute it
fn light_atten(i: i32, world: vec3f) -> f32 {{
    let base = {l} + 5 * i;
    let color = P.c[base];
    let dir = P.c[base + 1];
    let pos = P.c[base + 2];
    let spot = P.c[base + 3];
    let att = P.c[base + 4];
    let d = pos.xyz - world;
    let d2 = dot(d, d);
    let inv = inverseSqrt(max(d2, 1e-6));
    let dist = 1.0 / max(dot(att.xyz, vec3f(1.0, d2 * inv, d2)), 1e-6);
    let cos_theta = dot(dir.xyz, -d * inv);
    let spot_atten = clamp(pow(max((cos_theta - spot.z) * spot.w, 0.0001), spot.x), 0.0, 1.0);
    let a = mix(dist, dist * spot_atten, dir.w);
    return mix(a, 1.0, color.w);
}}

fn light_dir(i: i32, world: vec3f) -> vec3f {{
    let base = {l} + 5 * i;
    let to_light = normalize(P.c[base + 2].xyz - world);
    return normalize(mix(to_light, -P.c[base + 1].xyz, P.c[base].w));
}}

fn d_ggx(nh: f32, a2: f32) -> f32 {{
    let d = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}}

/// visibility from l.h alone (bounded at grazing angles), as the DX11 SurfaceGGX uses
fn vis_lh(lh: f32, a: f32) -> f32 {{
    let x = lh + sqrt(lh * lh * (1.0 - a) + a);
    return 1.0 / (x * x);
}}

/// the ambient cube as six lights through the same ggx lobe, weighted down for rough surfaces
fn ambient_spec(n: vec3f, v: vec3f, roughness: f32, a: f32, a2: f32) -> vec3f {{
    let k = roughness / min(roughness + 0.5, 1.0);
    var sum = vec3f(0.0);
    for (var i = 0; i < 6; i++) {{
        var d = vec3f(0.0);
        d[i / 2] = select(1.0, -1.0, (i & 1) == 1);
        let h = normalize(v + d);
        sum += P.c[{a} + i].rgb * d_ggx(clamp(dot(n, h), 0.0, 1.0), a2) * vis_lh(clamp(dot(d, h), 0.0, 1.0), a) * clamp(dot(n, d), 0.0, 1.0);
    }}
    return sum * k * k;
}}

fn f_schlick(f0: vec3f, vh: f32) -> vec3f {{
    return f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
}}

/// split-sum environment brdf, analytic fit (Karis 2014)
fn env_brdf(f0: vec3f, roughness: f32, nv: f32) -> vec3f {{
    let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4f(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
    let ab = vec2f(-1.04, 1.04) * a004 + r.zw;
    // the bias term fades out for f0 near zero (no reflection at all)
    return f0 * ab.x + clamp(f0.g * 50.0, 0.0, 1.0) * ab.y;
}}

@fragment
fn main(fin: FIn) -> FOut {{
    let mip_bias = bitcast<f32>(P.b.y);
    let uv = fin.l0.xy;
    let world = fin.l1.xyz;
    let base = fetch0(textureSampleBias(tex0, smp0, uv, mip_bias));
    let tint = P.c[{tint}];
    let albedo = base.rgb * tint.rgb;
    var n = normalize(fin.l2.xyz);
    var roughness = 1.0;
    var metal = 0.0;
    var fuzz = 0.0;
    let surface = P.c[{surf}];
"#,
        a = C_AMBIENT,
        l = C_LIGHTS,
        tint = C_TINT,
        surf = C_SURFACE,
    );
    if on(BUMP) {
        out += "    let bump = fetch1(textureSampleBias(tex1, smp1, uv, mip_bias));\n    var tn = bump.xyz * 2.0 - 1.0;\n    roughness = bump.a;\n";
        if on(DETAIL_NORMAL) {
            // c4: z detail normal blend, w its texcoord scale
            let _ = writeln!(out, "    let dn = fetch4(textureSampleBias(tex4, smp4, uv * P.c[{C_EXTRA}].w, mip_bias)).xyz * 2.0 - 1.0;\n    tn = normalize(vec3f(tn.xy + dn.xy * P.c[{C_EXTRA}].z, tn.z));");
        }
        // tangent space as the stock shaders build it
        out += "    let tg = normalize(fin.l3.xyz - n * dot(n, fin.l3.xyz));\n    let bt = cross(n, tg) * fin.l3.w;\n    n = normalize(tg * tn.x + bt * tn.y + n * tn.z);\n";
    }
    let mut f0 = "mix(vec3f(0.04), albedo, metal)".to_string();
    if on(PBR) {
        out += "    let masks = fetch2(textureSampleBias(tex2, smp2, uv, mip_bias));\n    roughness = pow(masks.a, 1.0 / 2.2);\n";
        f0 = "pow(masks.rgb, vec3f(2.2))".to_string();
    } else if on(MASK) {
        out += "    let mask = fetch2(textureSampleBias(tex2, smp2, uv, mip_bias));\n    roughness = 1.0 - mask.r;\n";
        if on(METAL) {
            out += "    metal = 1.0 - mask.g;\n";
        }
        if on(FUZZ) {
            let _ = writeln!(out, "    fuzz = (1.0 - mask.b) * P.c[{C_EXTRA}].y;");
        }
    }
    let _ = write!(
        out,
        r#"    roughness = max(clamp(roughness * surface.x, 0.0, 1.0), 0.03);
    let a = roughness * roughness;
    let a2 = a * a;
    let f0 = {f0};
    let diffuse_color = albedo * (1.0 - metal);
    let v = normalize(P.c[{eye}].xyz - world);
    let nv = max(dot(n, v), 1e-4);

    var diffuse = ambient(n);
    if fin.l4.w > 0.0 {{
        diffuse = fin.l4.rgb * fin.l4.rgb;
    }}
    // direct light: ggx with l.h visibility plus a (2 - F) / pi diffuse, the DX11 SurfaceGGX
    // model. source light colours are lambert-scaled (albedo * light * nl), so both terms get
    // pi / 2 to land the diffuse back on that
    var direct = vec3f(0.0);
    var specular = vec3f(0.0);
    let lights = i32(P.c[{diff}].w);
    for (var i = 0; i < 4; i++) {{
        if i >= lights {{ break; }}
        let l = light_dir(i, world);
        let radiance = P.c[{lt} + 5 * i].rgb * light_atten(i, world);
        let nl_raw = dot(n, l);
        let nl = max(nl_raw, 0.0);
"#,
        eye = C_EYE,
        diff = C_DIFFUSION,
        lt = C_LIGHTS,
    );
    out += r#"        let h = normalize(l + v);
        let f = f_schlick(f0, clamp(dot(v, h), 0.0, 1.0));
"#;
    if on(SSS) {
        // wrapped diffuse per channel, front and back
        let _ = writeln!(out, "        let wf = P.c[{C_DIFFUSION}].rgb;\n        let wrap = clamp((vec3f(nl_raw) + wf) / (1.0 + wf), vec3f(0.0), vec3f(1.0));");
    } else {
        out += "        let wrap = vec3f(nl);\n";
    }
    out += r#"        direct += diffuse_color * (1.0 - 0.5 * f) * wrap * radiance;
        let s = d_ggx(clamp(dot(n, h), 0.0, 1.0), a2) * vis_lh(clamp(dot(l, h), 0.0, 1.0), a) * f * (0.5 * PI) * nl * radiance;
        direct += s;
        specular += s;
    }
    var color = diffuse_color * diffuse + direct;
    var env_term = vec3f(0.0);
"#;
    if on(FUZZ) {
        out += "    color += fuzz * pow(1.0 - nv, 4.0) * albedo * ambient(n);\n";
    }
    if !on(ENVMAP) {
        // no cubemap: the ambient cube in the reflection direction stands in, so metals aren't black
        out += "    color += ambient_spec(n, v, roughness, a, a2) * env_brdf(f0, roughness, nv);\n";
    }
    if on(ENVMAP) {
        let _ = write!(
            out,
            r#"    let r = reflect(-v, n);
    let levels = f32(textureNumLevels(tex3));
    // c30.z: cubemap scale (16 under integer hdr, where cubemaps are stored scaled down)
    let cube = fetch3(textureSampleLevel(tex3, smp3, r, min((roughness - 0.001) * 8.0, levels - 1.0))).rgb * P.c[30].z;
    // rough surfaces lean on the ggx-filtered ambient cube, which carries local light the cubemap misses
    let env = mix(cube, ambient_spec(n, v, roughness, a, a2), roughness) * P.c[{env}].rgb;
    env_term = env * env_brdf(f0, roughness, nv);
    color += env_term;
"#,
            env = C_ENV
        );
    }
    let mut raw_cases = String::new();
    if on(MASK) || on(PBR) {
        raw_cases += "        case 9: { outc = vec4f(textureSample(tex2, smp2, uv).rgb, 1.0); }\n";
    }
    if on(BUMP) {
        raw_cases += "        case 10: { outc = vec4f(textureSample(tex1, smp1, uv).rgb, 1.0); }\n";
    }
    let alpha = if on(ALPHA) { "base.a * tint.a" } else { "tint.a" };
    let _ = write!(
        out,
        r#"    // stock FinalOutput: linear tonemap scale (c30), then range fog towards c29
    color *= P.c[30].x;
    let fog = P.c[{fog}];
    let fog_factor = clamp(min(fog.z, fin.l1.w * fog.w - fog.x), 0.0, 1.0);
    color = mix(color, P.c[29].rgb, fog_factor);
    var outc = vec4f(color, {alpha});
    // wk_ggx_debug (c4.x): 1 albedo, 2 normal, 3 roughness, 4 metal, 5 diffuse light, 6 direct specular, 7 reflection, 8 ambient cube, 9 raw mask, 10 raw bump, 11 uv
    switch i32(P.c[{extra}].x) {{
        case 1: {{ outc = vec4f(albedo, 1.0); }}
        case 2: {{ outc = vec4f(n * 0.5 + 0.5, 1.0); }}
        case 3: {{ outc = vec4f(vec3f(roughness), 1.0); }}
        case 4: {{ outc = vec4f(vec3f(metal), 1.0); }}
        case 5: {{ outc = vec4f(diffuse * P.c[30].x, 1.0); }}
        case 6: {{ outc = vec4f(specular * P.c[30].x, 1.0); }}
        case 7: {{ outc = vec4f(env_term * P.c[30].x, 1.0); }}
        case 8: {{ outc = vec4f(ambient(n) * P.c[30].x, 1.0); }}
{raw}        case 11: {{ outc = vec4f(fract(uv), 0.0, 1.0); }}
        default: {{}}
    }}
"#,
        fog = C_FOG,
        extra = C_EXTRA,
        raw = raw_cases
    );
    if let Some((func, r)) = opts.alpha_test {
        let a = "u32(round(clamp(outc.a, 0.0, 1.0) * 255.0))";
        let cond = match func {
            0 => "false".to_string(),
            1 => format!("{a} < {r}u"),
            2 => format!("{a} == {r}u"),
            3 => format!("{a} <= {r}u"),
            4 => format!("{a} > {r}u"),
            5 => format!("{a} != {r}u"),
            6 => format!("{a} >= {r}u"),
            _ => "true".to_string(),
        };
        let _ = writeln!(out, "    if (!({cond})) {{ discard; }}");
    }
    if opts.srgb_write {
        out += "    outc = vec4f(linear_to_srgb(outc.rgb), outc.a);\n";
    }
    out += "    var o: FOut;\n    o.color = outc;\n";
    if mv {
        out += "    o.mv = (fin.prev.xy / fin.prev.w - fin.cur.xy / fin.cur.w) * vec2f(0.5, -0.5);\n";
    }
    out += "    return o;\n}\n";
    Ok(Translated { wgsl: out, samplers, inputs: Vec::new() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(wgsl: &str) {
        let module = naga::front::wgsl::parse_str(wgsl).unwrap_or_else(|e| panic!("{}\n{wgsl}", e.emit_to_string(wgsl)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}\n{wgsl}", e.emit_to_string(wgsl)));
    }

    /// WK_DUMP_PS=<static combo> prints that pixel shader
    #[test]
    fn dump_ps() {
        let Ok(st) = std::env::var("WK_DUMP_PS") else { return };
        let opts = PsOptions { srgb_read: 0x8001, srgb_write: true, alpha_test: None, shadow_depth: 0 };
        println!("{}", translate_ps(&format!("{}/{st}+0", ggx::PS), &opts, false).unwrap().wgsl);
    }

    #[test]
    fn every_surface_ggx_combo_validates() {
        for st in 0..512u32 {
            let label = format!("{}/{st}+0", ggx::PS);
            let opts = PsOptions { srgb_read: 1 | 8, srgb_write: st % 2 == 0, alpha_test: (st % 3 == 0).then_some((6, 128)), shadow_depth: 0 };
            validate(&translate_ps(&label, &opts, st % 5 == 0).unwrap().wgsl);
        }
        for mask in 0..64u32 {
            let present = |u: u8, i: u8| match (u, i) {
                (0, 0) | (3, 0) | (5, 0) => true,
                (1, 0) | (2, 0) => mask & 1 != 0,
                (6, 0) => mask & 2 != 0,
                (10, 1) => mask & 4 != 0,
                (0, 1) | (3, 1) => mask & 8 != 0,
                _ => false,
            };
            let packed = |u: u8, _: u8| match (mask & 32 != 0, u) {
                (true, 3) => Packed::UByte4,
                (true, 1) => Packed::Short2,
                _ => Packed::No,
            };
            let t = translate_vs(&format!("{}/0+0", ggx::VS), &present, &packed, mask & 16 != 0, mask % 3 == 0).unwrap();
            validate(&t.wgsl);
        }
    }
}
