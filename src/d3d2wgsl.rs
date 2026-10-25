//! d3d9 shader bytecode (vs/ps 2.x, 3.0) to wgsl. every register is a vec4f; constants come from
//! per-draw uniform blocks (V / P). varyings use fixed locations: TEXCOORDn -> n, COLORn -> 8 + n,
//! FOG -> 10, POSITION -> 11.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;

use crate::d3dbc::*;

pub const VS_FLOAT_CONSTS: usize = 256;
// ps 3.0 allows 224, but stock shaders read at most c42
pub const PS_FLOAT_CONSTS: usize = 64;
pub const VARYINGS: u32 = 12;

// D3DDECLUSAGE
pub const USAGE_POSITION: u8 = 0;
pub const USAGE_BLENDWEIGHT: u8 = 1;
pub const USAGE_BLENDINDICES: u8 = 2;
pub const USAGE_NORMAL: u8 = 3;
pub const USAGE_TEXCOORD: u8 = 5;
pub const USAGE_TANGENT: u8 = 6;
pub const USAGE_BINORMAL: u8 = 7;
pub const USAGE_COLOR: u8 = 10;
pub const USAGE_FOG: u8 = 11;

pub const DIM_2D: u8 = 2;
pub const DIM_CUBE: u8 = 3;
pub const DIM_VOLUME: u8 = 4;
/// 2d sampler reading a shadow map with depth compare
pub const DIM_DEPTH: u8 = 5;

/// pixel-stage state baked into the shader
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PsOptions {
    pub srgb_read: u32,
    pub srgb_write: bool,
    /// ShaderAlphaFunc_t, ref 0..255
    pub alpha_test: Option<(i32, u8)>,
    /// samplers marked by SetShadowDepthFiltering. nvidia pcf: lit where z / w <= stored depth,
    /// filtered
    pub shadow_depth: u32,
}

pub struct Translated {
    pub wgsl: String,
    /// ps textures at binding 2 * reg (sampler +1), vs ones at 32 + 2 * reg
    pub samplers: Vec<(u32, u8)>,
    pub inputs: Vec<(u32, u8, u8)>,
}

pub fn varying_location(usage: u8, index: u8) -> Option<u32> {
    match (usage, index) {
        (USAGE_TEXCOORD, 0..=7) => Some(index as u32),
        (USAGE_COLOR, 0..=1) => Some(8 + index as u32),
        (USAGE_FOG, 0) => Some(10),
        (USAGE_POSITION, 0) => Some(11),
        _ => None,
    }
}

const SWZ: [char; 4] = ['x', 'y', 'z', 'w'];

fn fmt_f32(v: f32) -> String {
    if v.is_finite() {
        let s = format!("{v:?}");
        if s.contains('.') || s.contains('e') { s } else { format!("{s}.0") }
    } else if v.is_nan() {
        "0.0".into()
    } else if v > 0.0 {
        "3.4e38".into()
    } else {
        "-3.4e38".into()
    }
}

fn vec4_lit(v: [f32; 4]) -> String {
    format!("vec4f({}, {}, {}, {})", fmt_f32(v[0]), fmt_f32(v[1]), fmt_f32(v[2]), fmt_f32(v[3]))
}

struct Gen<'a> {
    s: &'a Shader,
    body: String,
    depth: usize,
    defs: HashMap<u32, [f32; 4]>,
    defi: HashMap<u32, [i32; 4]>,
    defb: HashMap<u32, bool>,
    inputs: BTreeMap<u32, (u8, u8)>,
    outputs: BTreeMap<u32, (u8, u8)>,
    samplers: BTreeMap<u32, u8>,
    shadow: u32,
    uses_rel_const: bool,
    uses_vpos: bool,
    uses_vface: bool,
    reps: usize,
}

impl Gen<'_> {
    fn line(&mut self, s: &str) {
        for _ in 0..self.depth + 1 {
            self.body.push_str("    ");
        }
        self.body.push_str(s);
        self.body.push('\n');
    }

    fn consts(&self) -> &'static str {
        if self.s.pixel { "P" } else { "V" }
    }

    fn rel(&self, r: &Rel) -> Result<String, String> {
        match r.reg.ty {
            REG_ADDR if !self.s.pixel => Ok(format!("a0.{}", SWZ[r.comp as usize])),
            REG_LOOP => Ok("aL".into()),
            t => Err(format!("relative addressing through register type {t}")),
        }
    }

    fn read(&mut self, r: Reg, rel: Option<&Rel>) -> Result<String, String> {
        let n = r.num;
        Ok(match r.ty {
            REG_TEMP => format!("r{n}"),
            REG_INPUT => format!("v{n}"),
            REG_CONST => match rel {
                Some(rel) => {
                    self.uses_rel_const = true;
                    format!("cv({} + {n})", self.rel(rel)?)
                }
                None => match self.defs.get(&n) {
                    Some(v) => vec4_lit(*v),
                    // past the uploaded block: reads as zero
                    None if self.s.pixel && n as usize >= PS_FLOAT_CONSTS => "vec4f(0.0)".into(),
                    None => format!("{}.c[{n}]", self.consts()),
                },
            },
            REG_ADDR if self.s.pixel => format!("t{n}"),
            REG_ADDR => "vec4f(a0)".into(),
            REG_CONSTINT => format!("vec4f({})", self.int_const(n)),
            REG_CONSTBOOL => format!("vec4f(f32({}))", self.bool_const(n)),
            REG_MISCTYPE if n == 0 => {
                self.uses_vpos = true;
                "vpos".into()
            }
            REG_MISCTYPE if n == 1 => {
                self.uses_vface = true;
                "vface".into()
            }
            REG_LOOP => "vec4f(f32(aL))".into(),
            t => return Err(format!("reading register type {t}")),
        })
    }

    fn int_const(&self, n: u32) -> String {
        match self.defi.get(&n) {
            Some(v) => format!("vec4<i32>({}, {}, {}, {})", v[0], v[1], v[2], v[3]),
            None => format!("{}.i[{n}]", self.consts()),
        }
    }

    fn bool_const(&self, n: u32) -> String {
        match self.defb.get(&n) {
            Some(v) => format!("{v}"),
            None => format!("((({}.b.x >> {n}u) & 1u) != 0u)", self.consts()),
        }
    }

    fn src(&mut self, s: &Src) -> Result<String, String> {
        let mut e = self.read(s.reg, s.rel.as_ref())?;
        if s.swizzle != 0xe4 {
            e.push('.');
            for i in 0..4 {
                e.push(SWZ[s.comp(i) as usize]);
            }
        }
        Ok(match s.modifier {
            0 => e,
            1 => format!("(-{e})"),
            2 => format!("({e} - vec4f(0.5))"),
            3 => format!("(vec4f(0.5) - {e})"),
            4 => format!("({e} * 2.0 - vec4f(1.0))"),
            5 => format!("(vec4f(1.0) - {e} * 2.0)"),
            6 => format!("(vec4f(1.0) - {e})"),
            7 => format!("({e} * 2.0)"),
            8 => format!("({e} * -2.0)"),
            9 => format!("({e} / {e}.z)"),
            10 => format!("({e} / {e}.w)"),
            11 => format!("abs({e})"),
            12 => format!("(-abs({e}))"),
            m => return Err(format!("source modifier {m}")),
        })
    }

    fn dst_name(&self, d: &Dst) -> Result<String, String> {
        if d.rel.is_some() {
            return Err("relative destination".into());
        }
        let n = d.reg.num;
        let sm3 = self.s.major >= 3;
        Ok(match d.reg.ty {
            REG_TEMP => format!("r{n}"),
            REG_ADDR if !self.s.pixel => "a0".into(),
            REG_RASTOUT => ["o_pos", "o_fog", "o_pts"].get(n as usize).ok_or("bad rastout")?.to_string(),
            REG_ATTROUT => format!("o_d{n}"),
            REG_OUTPUT if sm3 => format!("o{n}"),
            REG_OUTPUT => format!("o_t{n}"),
            REG_COLOROUT => format!("oc{n}"),
            REG_DEPTHOUT => "odepth".into(),
            t => return Err(format!("writing register type {t}")),
        })
    }

    /// honors write mask, shift and saturate
    fn write(&mut self, d: &Dst, value: String) -> Result<(), String> {
        let name = self.dst_name(d)?;
        let mut e = value;
        if d.shift != 0 {
            e = format!("({e}) * {}", fmt_f32(2f32.powi(d.shift as i32)));
        }
        if d.saturate {
            e = format!("clamp({e}, vec4f(0.0), vec4f(1.0))");
        }
        let addr = d.reg.ty == REG_ADDR && !self.s.pixel;
        if addr {
            e = format!("vec4<i32>(round({e}))");
        }
        if d.mask == 0xf {
            self.line(&format!("{name} = {e};"));
        } else {
            let ty = if addr { "vec4<i32>" } else { "vec4f" };
            let parts: Vec<String> = (0..4).map(|i| if d.mask & (1 << i) != 0 { format!("t.{}", SWZ[i]) } else { format!("{name}.{}", SWZ[i]) }).collect();
            self.line(&format!("{{ let t = {e}; {name} = {ty}({}); }}", parts.join(", ")));
        }
        Ok(())
    }

    fn cmp(ctrl: u8) -> Result<&'static str, String> {
        Ok(match ctrl & 7 {
            1 => ">",
            2 => "==",
            3 => ">=",
            4 => "<",
            5 => "!=",
            6 => "<=",
            c => return Err(format!("comparison {c}")),
        })
    }

    fn texture(&mut self, sampler: &Src, coord: String, op: u16, ctrl: u8, extra: &[String]) -> Result<String, String> {
        if sampler.reg.ty != REG_SAMPLER {
            return Err("texture op without sampler".into());
        }
        let n = sampler.reg.num;
        let dim = *self.samplers.entry(n).or_insert(DIM_2D);
        let (t, s) = if self.s.pixel { (format!("tex{n}"), format!("smp{n}")) } else { (format!("vtex{n}"), format!("vsmp{n}")) };
        if self.s.pixel && dim == DIM_2D && self.shadow & (1 << n) != 0 {
            // compare result in every channel
            let (uv, z) = if ctrl & 1 != 0 { (format!("({coord}.xy / {coord}.w)"), format!("({coord}.z / {coord}.w)")) } else { (format!("{coord}.xy"), format!("{coord}.z")) };
            return Ok(format!("vec4f(textureSampleCompareLevel({t}, {s}, {uv}, {z}))"));
        }
        let c = match dim {
            DIM_2D => format!("{coord}.xy"),
            _ => format!("{coord}.xyz"),
        };
        let projected = |c: &str| match dim {
            DIM_2D => format!("({coord}.xy / {coord}.w)"),
            _ => format!("({}.xyz / {coord}.w)", c.trim_end_matches(".xyz")),
        };
        let sample = match op {
            95 => format!("textureSampleLevel({t}, {s}, {c}, {coord}.w)"),
            93 => format!("textureSampleGrad({t}, {s}, {c}, {}, {})", sw(&extra[0], dim), sw(&extra[1], dim)),
            _ if !self.s.pixel => format!("textureSampleLevel({t}, {s}, {c}, 0.0)"),
            // per-draw mip bias (P.b.y, -1 under dlss)
            _ if ctrl & 1 != 0 => format!("textureSampleBias({t}, {s}, {}, mip_bias)", projected(&c)),
            _ if ctrl & 2 != 0 => format!("textureSampleBias({t}, {s}, {c}, {coord}.w + mip_bias)"),
            _ => format!("textureSampleBias({t}, {s}, {c}, mip_bias)"),
        };
        Ok(sample)
    }

    fn inst(&mut self, op: u16, ctrl: u8, dst: Option<&Dst>, src: &[Src]) -> Result<(), String> {
        let a = |g: &mut Self, i: usize| -> Result<String, String> { g.src(src.get(i).ok_or("missing operand")?) };
        let d = || dst.ok_or_else(|| "missing destination".to_string());
        let splat = |e: String| format!("vec4f({e})");
        match op {
            0 => {}
            1 => {
                let v = a(self, 0)?;
                self.write(d()?, v)?
            }
            46 => {
                let v = a(self, 0)?;
                self.write(d()?, v)?
            }
            2 => {
                let (x, y) = (a(self, 0)?, a(self, 1)?);
                self.write(d()?, format!("{x} + {y}"))?
            }
            3 => {
                let (x, y) = (a(self, 0)?, a(self, 1)?);
                self.write(d()?, format!("{x} - {y}"))?
            }
            4 => {
                let (x, y, z) = (a(self, 0)?, a(self, 1)?, a(self, 2)?);
                self.write(d()?, format!("m9({x}, {y}) + {z}"))?
            }
            5 => {
                let (x, y) = (a(self, 0)?, a(self, 1)?);
                self.write(d()?, format!("m9({x}, {y})"))?
            }
            6 => {
                let x = a(self, 0)?;
                self.write(d()?, splat(format!("1.0 / {x}.x")))?
            }
            7 => {
                let x = a(self, 0)?;
                self.write(d()?, splat(format!("inverseSqrt(abs({x}.x))")))?
            }
            8 => {
                let (x, y) = (a(self, 0)?, a(self, 1)?);
                self.write(d()?, splat(format!("dp3({x}, {y})")))?
            }
            9 => {
                let (x, y) = (a(self, 0)?, a(self, 1)?);
                self.write(d()?, splat(format!("dp4({x}, {y})")))?
            }
            10 => {
                let (x, y) = (a(self, 0)?, a(self, 1)?);
                self.write(d()?, format!("min({x}, {y})"))?
            }
            11 => {
                let (x, y) = (a(self, 0)?, a(self, 1)?);
                self.write(d()?, format!("max({x}, {y})"))?
            }
            12 => {
                let (x, y) = (a(self, 0)?, a(self, 1)?);
                self.write(d()?, format!("select(vec4f(0.0), vec4f(1.0), {x} < {y})"))?
            }
            13 => {
                let (x, y) = (a(self, 0)?, a(self, 1)?);
                self.write(d()?, format!("select(vec4f(0.0), vec4f(1.0), {x} >= {y})"))?
            }
            14 | 78 => {
                let x = a(self, 0)?;
                self.write(d()?, splat(format!("exp2({x}.x)")))?
            }
            15 | 79 => {
                let x = a(self, 0)?;
                self.write(d()?, splat(format!("log2(abs({x}.x))")))?
            }
            18 => {
                let (x, y, z) = (a(self, 0)?, a(self, 1)?, a(self, 2)?);
                self.write(d()?, format!("mix({z}, {y}, {x})"))?
            }
            19 => {
                let x = a(self, 0)?;
                self.write(d()?, format!("fract({x})"))?
            }
            32 => {
                let (x, y) = (a(self, 0)?, a(self, 1)?);
                self.write(d()?, splat(format!("pow9({x}.x, {y}.x)")))?
            }
            33 => {
                let (x, y) = (a(self, 0)?, a(self, 1)?);
                self.write(d()?, format!("vec4f(cross({x}.xyz, {y}.xyz), 0.0)"))?
            }
            34 => {
                let x = a(self, 0)?;
                self.write(d()?, format!("sign({x})"))?
            }
            35 => {
                let x = a(self, 0)?;
                self.write(d()?, format!("abs({x})"))?
            }
            36 => {
                let x = a(self, 0)?;
                self.write(d()?, format!("m9({x}, vec4f(inverseSqrt(dp3({x}, {x}))))"))?
            }
            37 => {
                let x = a(self, 0)?;
                self.write(d()?, format!("vec4f(cos({x}.x), sin({x}.x), 0.0, 0.0)"))?
            }
            88 => {
                let (x, y, z) = (a(self, 0)?, a(self, 1)?, a(self, 2)?);
                self.write(d()?, format!("select({z}, {y}, {x} >= vec4f(0.0))"))?
            }
            80 => {
                let (x, y, z) = (a(self, 0)?, a(self, 1)?, a(self, 2)?);
                self.write(d()?, format!("select({z}, {y}, {x} > vec4f(0.5))"))?
            }
            90 => {
                let (x, y, z) = (a(self, 0)?, a(self, 1)?, a(self, 2)?);
                self.write(d()?, splat(format!("dp4(vec4f({x}.xy, 0.0, 0.0), vec4f({y}.xy, 0.0, 0.0)) + {z}.x")))?
            }
            91 => {
                let x = a(self, 0)?;
                self.write(d()?, format!("dpdx({x})"))?
            }
            92 => {
                let x = a(self, 0)?;
                self.write(d()?, format!("dpdy({x})"))?
            }
            65 => {
                // operand is encoded as dst
                let dd = d()?;
                let v = self.read(dd.reg, None)?;
                let comps: Vec<String> = (0..4).filter(|i| dd.mask & (1 << i) != 0).map(|i| format!("{v}.{} < 0.0", SWZ[i])).collect();
                if !comps.is_empty() {
                    self.line(&format!("if ({}) {{ discard; }}", comps.join(" || ")));
                }
            }
            66 | 95 | 93 => {
                let coord = a(self, 0)?;
                let extra = if op == 93 { vec![a(self, 2)?, a(self, 3)?] } else { vec![] };
                let sampler = *src.get(1).ok_or("texld without sampler")?;
                let mut sample = self.texture(&sampler, coord, op, ctrl, &extra)?;
                if sampler.swizzle != 0xe4 {
                    sample.push('.');
                    for i in 0..4 {
                        sample.push(SWZ[sampler.comp(i) as usize]);
                    }
                }
                if self.s.pixel && op != 95 {
                    // srgb decode happens in `fetch`
                    let n = sampler.reg.num;
                    sample = format!("fetch{n}({sample})");
                }
                self.write(d()?, sample)?
            }
            40 => {
                let s = src.first().ok_or("if without operand")?;
                let cond = match s.reg.ty {
                    REG_CONSTBOOL => self.bool_const(s.reg.num),
                    t => return Err(format!("if on register type {t}")),
                };
                self.line(&format!("if ({cond}) {{"));
                self.depth += 1;
            }
            41 => {
                let (x, y) = (a(self, 0)?, a(self, 1)?);
                self.line(&format!("if ({x}.x {} {y}.x) {{", Self::cmp(ctrl)?));
                self.depth += 1;
            }
            42 => {
                self.depth = self.depth.saturating_sub(1);
                self.line("} else {");
                self.depth += 1;
            }
            43 | 39 | 29 => {
                self.depth = self.depth.saturating_sub(1);
                self.line("}");
            }
            38 => {
                let s = src.first().ok_or("rep without operand")?;
                let count = self.int_const(s.reg.num);
                let k = self.reps;
                self.reps += 1;
                self.line(&format!("for (var rep{k} = 0; rep{k} < min({count}.x, 255); rep{k}++) {{"));
                self.depth += 1;
            }
            27 => return Err("loop is not supported (unused by the stock shaders)".into()),
            44 => self.line("break;"),
            45 => {
                let (x, y) = (a(self, 0)?, a(self, 1)?);
                self.line(&format!("if ({x}.x {} {y}.x) {{ break; }}", Self::cmp(ctrl)?));
            }
            28 => {}
            _ => return Err(format!("unsupported opcode {op}")),
        }
        Ok(())
    }
}

fn sw(e: &str, dim: u8) -> String {
    if dim == DIM_2D { format!("{e}.xy") } else { format!("{e}.xyz") }
}

fn texture_type(dim: u8) -> &'static str {
    match dim {
        DIM_CUBE => "texture_cube<f32>",
        DIM_VOLUME => "texture_3d<f32>",
        DIM_DEPTH => "texture_depth_2d",
        _ => "texture_2d<f32>",
    }
}

/// derivatives in non-uniform control flow are fine in d3d9. m9/dp3/dp4/pow9 emulate d3d9's legacy
/// multiply where 0 * anything = 0.
const PRELUDE: &str = "diagnostic(off, derivative_uniformity);\n\
fn m9(a: vec4f, b: vec4f) -> vec4f { return select(a * b, vec4f(0.0), (a == vec4f(0.0)) | (b == vec4f(0.0))); }\n\
fn dp3(a: vec4f, b: vec4f) -> f32 { let p = m9(a, b); return p.x + p.y + p.z; }\n\
fn dp4(a: vec4f, b: vec4f) -> f32 { let p = m9(a, b); return p.x + p.y + p.z + p.w; }\n\
fn pow9(a: f32, b: f32) -> f32 { return exp2(m9(vec4f(b), vec4f(log2(abs(a)))).x); }\n\n";

fn consts_struct(name: &str, floats: usize) -> String {
    // hp: 1 / viewport size, then dlss jitter in ndc. clip: user clip planes. mvm: reprojection to
    // last frame's clip space.
    let tail = if name == "V" { "    hp: vec4f,\n    clip: array<vec4f, 6>,\n    mvm: array<vec4f, 4>,\n" } else { "" };
    format!("struct {name}Consts {{\n    c: array<vec4f, {floats}>,\n    i: array<vec4<i32>, 16>,\n    b: vec4<u32>,\n{tail}}}\n")
}

/// elements missing from the vertex data read as (0, 0, 0, 1). `mv` also outputs this and last
/// frame's clip position (locations 12 / 13).
pub fn translate_vs(s: &Shader, present: &dyn Fn(u8, u8) -> bool, clip: bool, mv: bool) -> Result<Translated, String> {
    if s.pixel {
        return Err("not a vertex shader".into());
    }
    let mut g = new_gen(s);
    let body = generate(&mut g)?;
    let mut out = String::new();
    if clip {
        out += "enable clip_distances;\n";
    }
    out += PRELUDE;
    out += &consts_struct("V", VS_FLOAT_CONSTS);
    out += "@group(0) @binding(0) var<uniform> V: VConsts;\n";
    for (&n, &dim) in &g.samplers {
        let _ = writeln!(out, "@group(1) @binding({}) var vtex{n}: {};\n@group(1) @binding({}) var vsmp{n}: sampler;", 32 + 2 * n, texture_type(dim), 33 + 2 * n);
    }
    if g.uses_rel_const {
        out += &rel_const_fn(&g, "V");
    }
    let mut inputs = Vec::new();
    out += "\nstruct VIn {\n";
    let mut loads = String::new();
    for (&reg, &(usage, index)) in &g.inputs {
        if present(usage, index) {
            let loc = inputs.len() as u32;
            inputs.push((loc, usage, index));
            let _ = writeln!(out, "    @location({loc}) a{loc}: vec4f,");
            // d3dcolor elements arrive as bgra
            let swz = if matches!(usage, USAGE_COLOR | USAGE_BLENDINDICES) { ".zyxw" } else { "" };
            let _ = writeln!(loads, "    let v{reg} = vin.a{loc}{swz};");
        } else {
            let def = if usage == USAGE_COLOR { "vec4f(1.0)" } else { "vec4f(0.0, 0.0, 0.0, 1.0)" };
            let _ = writeln!(loads, "    let v{reg} = {def};");
        }
    }
    out += "    @builtin(vertex_index) vid: u32,\n}\n\nstruct VOut {\n    @builtin(position) pos: vec4f,\n";
    for l in 0..VARYINGS {
        let _ = writeln!(out, "    @location({l}) l{l}: vec4f,");
    }
    if clip {
        out += "    @builtin(clip_distances) clip: array<f32, 6>,\n";
    }
    if mv {
        out += "    @location(12) cur: vec4f,\n    @location(13) prev: vec4f,\n";
    }
    out += "}\n\n@vertex\nfn main(vin: VIn) -> VOut {\n";
    out += &loads;
    out += &temps(&g);
    out += "    var a0 = vec4<i32>(0);\n    var aL = 0;\n";
    let sm3 = s.major >= 3;
    if sm3 {
        for &o in g.outputs.keys() {
            let _ = writeln!(out, "    var o{o} = vec4f(0.0);");
        }
    } else {
        out += "    var o_pos = vec4f(0.0);\n    var o_fog = vec4f(0.0);\n    var o_pts = vec4f(0.0);\n    var o_d0 = vec4f(0.0);\n    var o_d1 = vec4f(0.0);\n";
        for t in 0..8 {
            let _ = writeln!(out, "    var o_t{t} = vec4f(0.0);");
        }
    }
    out += &body;
    out += "    var out: VOut;\n";
    for l in 0..VARYINGS {
        let _ = writeln!(out, "    out.l{l} = vec4f(0.0);");
    }
    if sm3 {
        for (&o, &(usage, index)) in &g.outputs {
            if usage == USAGE_POSITION && index == 0 {
                let _ = writeln!(out, "    out.pos = o{o};");
            }
            if let Some(l) = varying_location(usage, index) {
                let _ = writeln!(out, "    out.l{l} = o{o};");
            }
        }
    } else {
        out += "    out.pos = o_pos;\n    out.l11 = o_pos;\n    out.l10 = o_fog;\n";
        out += "    out.l8 = clamp(o_d0, vec4f(0.0), vec4f(1.0));\n    out.l9 = clamp(o_d1, vec4f(0.0), vec4f(1.0));\n";
        for t in 0..8 {
            let _ = writeln!(out, "    out.l{t} = o_t{t};");
        }
    }
    if clip {
        // d3d clips where dot(plane, pos) < 0
        out += "    for (var i = 0; i < 6; i++) {\n        out.clip[i] = dot(V.clip[i], out.pos);\n    }\n";
    }
    if mv {
        out += "    out.cur = out.pos;\n    out.prev = vec4f(dot(out.pos, V.mvm[0]), dot(out.pos, V.mvm[1]), dot(out.pos, V.mvm[2]), dot(out.pos, V.mvm[3]));\n";
    }
    // d3d9 pixel centers sit at integer coords, d3d10+ at +0.5: shift by half a pixel (plus dlss
    // jitter)
    out += "    out.pos = vec4f(out.pos.x + (V.hp.x + V.hp.z) * out.pos.w, out.pos.y - (V.hp.y - V.hp.w) * out.pos.w, out.pos.zw);\n";
    out += "    return out;\n}\n";
    Ok(Translated { wgsl: out, samplers: g.samplers.into_iter().collect(), inputs })
}

/// `mv` also writes the motion vector to location 1
pub fn translate_ps(s: &Shader, opts: &PsOptions, mv: bool) -> Result<Translated, String> {
    if !s.pixel {
        return Err("not a pixel shader".into());
    }
    let mut g = new_gen(s);
    g.shadow = opts.shadow_depth;
    let body = generate(&mut g)?;
    for (&n, dim) in g.samplers.iter_mut() {
        if *dim == DIM_2D && opts.shadow_depth & (1 << n) != 0 {
            *dim = DIM_DEPTH;
        }
    }
    let sm3 = s.major >= 3;
    let mut out = String::from(PRELUDE);
    out += &consts_struct("P", PS_FLOAT_CONSTS);
    out += "@group(0) @binding(1) var<uniform> P: PConsts;\n";
    for (&n, &dim) in &g.samplers {
        let sampler = if dim == DIM_DEPTH { "sampler_comparison" } else { "sampler" };
        let _ = writeln!(out, "@group(1) @binding({}) var tex{n}: {};\n@group(1) @binding({}) var smp{n}: {sampler};", 2 * n, texture_type(dim), 2 * n + 1);
        if opts.srgb_read & (1 << n) != 0 && dim != DIM_DEPTH {
            let _ = writeln!(out, "fn fetch{n}(c: vec4f) -> vec4f {{ return vec4f(srgb_to_linear(c.rgb), c.a); }}");
        } else {
            let _ = writeln!(out, "fn fetch{n}(c: vec4f) -> vec4f {{ return c; }}");
        }
    }
    out += "fn srgb_to_linear(c: vec3f) -> vec3f { return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045)); }\n";
    out += "fn linear_to_srgb(c: vec3f) -> vec3f { let x = clamp(c, vec3f(0.0), vec3f(1.0)); return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3f(0.0031308)); }\n";
    if g.uses_rel_const {
        out += &rel_const_fn(&g, "P");
    }
    out += "\nstruct FIn {\n    @builtin(position) fpos: vec4f,\n    @builtin(front_facing) ff: bool,\n";
    let mut loads = String::new();
    let mut used = std::collections::BTreeSet::new();
    if sm3 {
        for (&reg, &(usage, index)) in &g.inputs {
            match varying_location(usage, index) {
                Some(l) => {
                    used.insert(l);
                    let _ = writeln!(loads, "    let v{reg} = fin.l{l};");
                }
                None => {
                    let _ = writeln!(loads, "    let v{reg} = vec4f(0.0);");
                }
            }
        }
    } else {
        // ps 2.x: v0/v1 are colors, t# texcoords
        for &reg in g.inputs.keys() {
            let l = 8 + reg.min(1);
            used.insert(l);
            let _ = writeln!(loads, "    let v{reg} = fin.l{l};");
        }
        for t in 0..8 {
            used.insert(t);
            let _ = writeln!(loads, "    let t{t} = fin.l{t};");
        }
    }
    for l in &used {
        let _ = writeln!(out, "    @location({l}) l{l}: vec4f,");
    }
    if mv {
        out += "    @location(12) cur: vec4f,\n    @location(13) prev: vec4f,\n";
    }
    out += "}\n\nstruct FOut {\n    @location(0) color: vec4f,\n";
    if mv {
        out += "    @location(1) mv: vec2f,\n";
    }
    let writes_depth = body.contains("odepth");
    if writes_depth {
        out += "    @builtin(frag_depth) depth: f32,\n";
    }
    out += "}\n\n@fragment\nfn main(fin: FIn) -> FOut {\n";
    out += "    let vpos = vec4f(fin.fpos.xy - vec2f(0.5), 0.0, 0.0);\n    let vface = select(-1.0, 1.0, fin.ff);\n    let mip_bias = bitcast<f32>(P.b.y);\n";
    out += &loads;
    out += &temps(&g);
    out += "    var a0 = vec4<i32>(0);\n    var aL = 0;\n    var oc0 = vec4f(0.0);\n    var oc1 = vec4f(0.0);\n    var oc2 = vec4f(0.0);\n    var oc3 = vec4f(0.0);\n    var odepth = vec4f(fin.fpos.z);\n";
    out += &body;
    out += "    var color = oc0;\n";
    if let Some((func, r)) = opts.alpha_test {
        let a = "u32(round(clamp(color.a, 0.0, 1.0) * 255.0))";
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
        out += "    color = vec4f(linear_to_srgb(color.rgb), color.a);\n";
    }
    out += "    var out: FOut;\n    out.color = color;\n";
    if mv {
        out += "    out.mv = (fin.prev.xy / fin.prev.w - fin.cur.xy / fin.cur.w) * vec2f(0.5, -0.5);\n";
    }
    if writes_depth {
        out += "    out.depth = clamp(odepth.x, 0.0, 1.0);\n";
    }
    out += "    return out;\n}\n";
    Ok(Translated { wgsl: out, samplers: g.samplers.into_iter().collect(), inputs: Vec::new() })
}

fn new_gen(s: &Shader) -> Gen<'_> {
    Gen {
        s,
        body: String::new(),
        depth: 0,
        defs: HashMap::new(),
        defi: HashMap::new(),
        defb: HashMap::new(),
        inputs: BTreeMap::new(),
        outputs: BTreeMap::new(),
        samplers: BTreeMap::new(),
        shadow: 0,
        uses_rel_const: false,
        uses_vpos: false,
        uses_vface: false,
        reps: 0,
    }
}

fn generate(g: &mut Gen) -> Result<String, String> {
    for i in &g.s.insts {
        match i {
            Inst::Def { reg, v } => {
                g.defs.insert(*reg, *v);
            }
            Inst::DefI { reg, v } => {
                g.defi.insert(*reg, *v);
            }
            Inst::DefB { reg, v } => {
                g.defb.insert(*reg, *v);
            }
            Inst::Dcl { usage, index, dst, sampler } => {
                if *sampler {
                    g.samplers.insert(dst.reg.num, *usage);
                } else if dst.reg.ty == REG_INPUT {
                    g.inputs.insert(dst.reg.num, (*usage, *index));
                } else if dst.reg.ty == REG_OUTPUT {
                    g.outputs.insert(dst.reg.num, (*usage, *index));
                }
            }
            _ => {}
        }
    }
    for i in &g.s.insts {
        if let Inst::Op { op, ctrl, predicate, dst, src } = i {
            if predicate.is_some() {
                return Err("predicated instructions are not supported".into());
            }
            g.inst(*op, *ctrl, dst.as_ref(), src)?;
        }
    }
    if g.depth != 0 {
        return Err("unbalanced control flow".into());
    }
    Ok(std::mem::take(&mut g.body))
}

fn temps(g: &Gen) -> String {
    let mut used = std::collections::BTreeSet::new();
    for i in &g.s.insts {
        if let Inst::Op { dst, src, .. } = i {
            for r in dst.iter().map(|d| d.reg).chain(src.iter().map(|s| s.reg)) {
                if r.ty == REG_TEMP {
                    used.insert(r.num);
                }
            }
        }
    }
    used.iter().map(|n| format!("    var r{n} = vec4f(0.0);\n")).collect()
}

/// float constant by computed index, `def` values first
fn rel_const_fn(g: &Gen, block: &str) -> String {
    let mut f = String::from("fn cv(i: i32) -> vec4f {\n    switch i {\n");
    let mut defs: Vec<_> = g.defs.iter().collect();
    defs.sort_by_key(|d| d.0);
    for (n, v) in defs {
        let _ = writeln!(f, "        case {n}: {{ return {}; }}", vec4_lit(*v));
    }
    let max = if block == "V" { VS_FLOAT_CONSTS } else { PS_FLOAT_CONSTS } - 1;
    let _ = writeln!(f, "        default: {{ return {block}.c[clamp(i, 0, {max})]; }}\n    }}\n}}");
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcs::VcsFile;

    fn validate(wgsl: &str) -> Result<(), String> {
        let module = naga::front::wgsl::parse_str(wgsl).map_err(|e| e.emit_to_string(wgsl))?;
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .map_err(|e| e.emit_to_string(wgsl))?;
        Ok(())
    }

    #[test]
    fn translate_real_shaders() {
        let Ok(dir) = std::env::var("SHADERS_DIR") else { return };
        let (mut ok, mut failed) = (0, 0);
        let mut first_errors = Vec::new();
        for e in std::fs::read_dir(dir).unwrap() {
            let path = e.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if name.contains("_ps1") || name.contains("_vs1") {
                continue;
            }
            let vcs = VcsFile::parse(std::fs::read(&path).unwrap()).unwrap();
            let step = (vcs.header.num_static_combos as usize / 3).max(1);
            for rec in (0..vcs.header.num_static_combos as usize - 1).step_by(step) {
                let id = vcs.records_for_test(rec) * vcs.header.dynamic_combos.max(1) as u32;
                for (combo, code) in vcs.static_combo(id).unwrap().into_iter().step_by(7) {
                    let s = parse(&code).unwrap();
                    let r = if s.pixel {
                        translate_ps(&s, &PsOptions { srgb_read: 1, srgb_write: true, alpha_test: Some((4, 128)), shadow_depth: 0 }, rec % 2 == 1).and_then(|t| {
                            let shadow = translate_ps(&s, &PsOptions { srgb_read: 1, srgb_write: true, alpha_test: None, shadow_depth: 0xffff }, false)?;
                            validate(&shadow.wgsl).map_err(|e| format!("shadow variant: {e}\n{}", shadow.wgsl))?;
                            Ok(t)
                        })
                    } else {
                        translate_vs(&s, &|u, _| u != USAGE_TANGENT, rec % 2 == 0, rec % 3 == 0)
                    };
                    match r.and_then(|t| validate(&t.wgsl).map_err(|e| format!("{e}\n{}", t.wgsl))) {
                        Ok(()) => ok += 1,
                        Err(e) => {
                            failed += 1;
                            if first_errors.len() < 3 {
                                first_errors.push(format!("{name} static {id} dynamic {combo}: {e}"));
                            }
                        }
                    }
                }
            }
        }
        for e in &first_errors {
            println!("{e}\n------");
        }
        println!("translated {ok}, failed {failed}");
        assert_eq!(failed, 0);
    }
}

#[cfg(test)]
mod dump {
    /// DUMP_SHADER=path/to/x.vcs prints the first combo's wgsl
    #[test]
    fn dump_shader() {
        let Ok(p) = std::env::var("DUMP_SHADER") else { return };
        let vcs = crate::vcs::VcsFile::parse(std::fs::read(&p).unwrap()).unwrap();
        let id = std::env::var("STATIC").map(|v| v.parse().unwrap()).unwrap_or(vcs.records_for_test(0) * vcs.header.dynamic_combos.max(1) as u32);
        let dynamic: u32 = std::env::var("DYNAMIC").map(|v| v.parse().unwrap()).unwrap_or(0);
        let code = vcs.static_combo(id).unwrap().into_iter().find(|c| c.0 == dynamic).unwrap().1;
        let s = crate::d3dbc::parse(&code).unwrap();
        let t = if s.pixel { super::translate_ps(&s, &Default::default(), false) } else { super::translate_vs(&s, &|_, _| true, false, false) }.unwrap();
        println!("{}", t.wgsl);
    }
}
