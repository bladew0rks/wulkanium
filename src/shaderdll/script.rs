//! material shaders from garrysmod/shaders/wulkanium/materials/*.wgsl, written while the game
//! runs. a file declares a shader name and its params in a header and supplies the shading in
//! wgsl; everything else (skinning, lighting inputs, fog, tonemap, motion vectors, render state
//! from $flags) is shared with SurfaceGGX:
//!
//! ```text
//! //! shader: Toon
//! //! param: $bands float 3 "number of light bands"
//! //! param: $shadowtint color [0.2 0.2 0.3]
//! //! param: $ramp texture
//!
//! fn shade(s: Surface) -> vec4f {
//!     let m = material();
//!     let base = base_texture(s.uv) * tint();
//!     let light = diffuse_lighting(s, s.normal);
//!     return vec4f(base.rgb * floor(light * m.bands) / m.bands, base.a);
//! }
//! ```
//!
//! editing the body hot-reloads it. a new shader name or a changed param list makes a new def,
//! and the shader dll registers again (see `super::reregister`) so materialsystem sees it;
//! materials already loaded keep their old shader until they reload.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::fmt::Write;
use std::path::PathBuf;
use std::time::{Instant, SystemTime};

use super::{Def, Param, ParamType, Params, add_def, shader_system, vfn};
use crate::builtin::ggx;
use crate::d3d2wgsl::{PS_FLOAT_CONSTS, PsOptions, Translated};
use crate::log;
use crate::vtables::{shader_api as api, shader_shadow as sh};

pub const DIR: &str = "garrysmod/shaders/wulkanium/materials";
/// pixel shader names: wk_script_<lowercase shader name>_ps
const PS_PREFIX: &str = "wk_script_";
/// script params live in c13..c28
const FIRST_REG: i32 = 13;
const MAX_REGS: usize = 16;
/// sampler 0 is $basetexture
const MAX_TEXTURES: usize = 7;
/// c4: time, light count
const C_EXTRA: i32 = 4;

#[derive(Clone, Debug, PartialEq)]
pub struct ScriptParam {
    /// with the $
    pub name: String,
    pub ty: ParamType,
    /// float, int, bool, vec2, vec3, vec4, color, texture, texture_srgb, cubemap
    pub kind: String,
    pub default: String,
    pub help: String,
    /// constant register, or sampler for textures
    pub slot: i32,
}

impl ScriptParam {
    fn field(&self) -> &str {
        self.name.trim_start_matches('$')
    }
    fn is_texture(&self) -> bool {
        matches!(self.kind.as_str(), "texture" | "texture_srgb" | "cubemap")
    }
}

#[derive(Clone, Debug)]
pub struct Script {
    pub name: String,
    pub params: Vec<ScriptParam>,
    pub body: String,
    pub path: PathBuf,
}

impl Script {
    /// the parts materialsystem sees; a change needs a new def
    fn signature(&self) -> (String, Vec<(String, String, String)>) {
        (self.name.to_lowercase(), self.params.iter().map(|p| (p.name.to_lowercase(), p.kind.clone(), p.default.clone())).collect())
    }
    fn uses_tangents(&self) -> bool {
        self.body.contains("tangent") || self.body.contains("binormal")
    }
}

/// `//! shader:` / `//! param:` header lines, then the body
pub fn parse(src: &str, path: PathBuf) -> Result<Script, String> {
    let mut name = None;
    let mut params: Vec<ScriptParam> = Vec::new();
    let (mut regs, mut samplers) = (0usize, 0usize);
    for (n, line) in src.lines().enumerate() {
        let Some(rest) = line.trim().strip_prefix("//!") else { continue };
        let rest = rest.trim();
        let err = |m: &str| format!("line {}: {m}", n + 1);
        if let Some(v) = rest.strip_prefix("shader:") {
            let v = v.trim();
            if v.is_empty() || !v.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return Err(err("shader name: letters, digits and _ only"));
            }
            name = Some(v.to_string());
        } else if let Some(v) = rest.strip_prefix("param:") {
            // $name kind [default...] ["help"]
            let (v, help) = match v.split_once('"') {
                Some((a, b)) => (a, b.trim_end().trim_end_matches('"').to_string()),
                None => (v, String::new()),
            };
            let mut it = v.split_whitespace();
            let pname = it.next().ok_or_else(|| err("param: missing name"))?;
            let kind = it.next().ok_or_else(|| err("param: missing type"))?.to_lowercase();
            let default = it.collect::<Vec<_>>().join(" ");
            let field = pname.strip_prefix('$').ok_or_else(|| err("param names start with $"))?;
            if field.is_empty() || !field.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') || field.starts_with(|c: char| c.is_ascii_digit()) {
                return Err(err("param name: letters, digits and _ only"));
            }
            if super::STANDARD.iter().any(|s| s.name.eq_ignore_ascii_case(pname)) || params.iter().any(|p| p.name.eq_ignore_ascii_case(pname)) {
                return Err(err(&format!("{pname} is already a param")));
            }
            let (ty, slot) = match kind.as_str() {
                "float" | "int" | "bool" | "vec2" | "vec3" | "vec4" | "color" => {
                    if regs == MAX_REGS {
                        return Err(err(&format!("at most {MAX_REGS} number params")));
                    }
                    regs += 1;
                    let ty = match kind.as_str() {
                        "float" => ParamType::Float,
                        "int" => ParamType::Integer,
                        "bool" => ParamType::Bool,
                        "vec2" => ParamType::Vec2,
                        "vec3" => ParamType::Vec3,
                        "vec4" => ParamType::Vec4,
                        _ => ParamType::Color,
                    };
                    (ty, FIRST_REG + regs as i32 - 1)
                }
                "texture" | "texture_srgb" | "cubemap" => {
                    if samplers == MAX_TEXTURES {
                        return Err(err(&format!("at most {MAX_TEXTURES} textures besides $basetexture")));
                    }
                    samplers += 1;
                    (ParamType::Texture, samplers as i32)
                }
                _ => return Err(err("type: float, int, bool, vec2, vec3, vec4, color, texture, texture_srgb or cubemap")),
            };
            params.push(ScriptParam { name: pname.to_string(), ty, kind, default, help, slot });
        }
    }
    let name = name.ok_or("missing `//! shader: <name>`")?;
    if !src.contains("fn shade(") {
        return Err("missing `fn shade(s: Surface) -> vec4f`".into());
    }
    Ok(Script { name, params, body: src.to_string(), path })
}

// ---- registry ----

struct Entry {
    script: Script,
    /// bumped on every accepted change; part of the pixel shader's cache key
    generation: u32,
    stamp: Option<SystemTime>,
}

#[derive(Default)]
struct Registry {
    /// by lowercase shader name
    entries: HashMap<String, Entry>,
    /// path -> lowercase shader name
    files: HashMap<PathBuf, String>,
    /// def index -> lowercase shader name
    defs: HashMap<usize, String>,
    last_scan: Option<Instant>,
}

static REGISTRY: parking_lot::Mutex<Option<Registry>> = parking_lot::Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut Registry) -> R) -> R {
    f(REGISTRY.lock().get_or_insert_with(Default::default))
}

fn ps_name(shader: &str) -> String {
    format!("{PS_PREFIX}{}_ps", shader.to_lowercase())
}

fn by_def(d: usize) -> Option<(Script, u32)> {
    with(|r| {
        let key = r.defs.get(&d)?;
        let e = r.entries.get(key)?;
        Some((e.script.clone(), e.generation))
    })
}

/// the script generation behind a pixel shader name, for the renderer's caches (0 otherwise)
pub fn generation(ps: &str) -> u32 {
    let Some(rest) = ps.strip_prefix(PS_PREFIX).and_then(|r| r.strip_suffix("_ps")) else { return 0 };
    with(|r| r.entries.get(rest).map_or(0, |e| e.generation))
}

pub fn is_script_ps(name: &str) -> bool {
    name.starts_with(PS_PREFIX)
}

/// (shader name, file) of every loaded script, for the overlay / lua
pub fn list() -> Vec<(String, PathBuf, u32)> {
    with(|r| {
        let mut v: Vec<_> = r.entries.values().map(|e| (e.script.name.clone(), e.script.path.clone(), e.generation)).collect();
        v.sort();
        v
    })
}

/// game thread, once a frame: pick up new and changed files (twice a second)
pub fn poll() {
    let due = with(|r| {
        let due = r.last_scan.is_none_or(|t| t.elapsed().as_millis() >= 500);
        if due {
            r.last_scan = Some(Instant::now());
        }
        due
    });
    if !due || super::shader_system() == 0 {
        return;
    }
    let files: Vec<PathBuf> = std::fs::read_dir(DIR).map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "wgsl")).collect()).unwrap_or_default();
    let mut register = Vec::new();
    for path in files {
        let stamp = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        let known = with(|r| r.files.get(&path).and_then(|k| r.entries.get(k)).map(|e| e.stamp));
        if known == Some(stamp) {
            continue;
        }
        let file = path.file_name().map_or_else(String::new, |f| f.to_string_lossy().into_owned());
        let src = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let script = match parse(&src, path.clone()).and_then(|s| validate(&s).map(|_| s)) {
            Ok(s) => s,
            Err(e) => {
                // keep the stamp so the same broken file isn't reported every scan
                with(|r| {
                    if let Some(e) = r.files.get(&path).and_then(|k| r.entries.get_mut(k)) {
                        e.stamp = stamp;
                    }
                });
                log::line(&format!("material shader {file}: {e}"));
                crate::overlay::notify(format!("{file}: {e}"), true);
                continue;
            }
        };
        let key = script.name.to_lowercase();
        let change = with(|r| {
            r.files.insert(path.clone(), key.clone());
            match r.entries.get_mut(&key) {
                Some(e) if e.script.signature() == script.signature() => {
                    e.script = script.clone();
                    e.generation += 1;
                    e.stamp = stamp;
                    false
                }
                _ => true,
            }
        });
        if change {
            register.push((script, stamp));
        } else {
            log::line(&format!("material shader {}: reloaded", script.name));
            crate::overlay::notify(format!("{}: reloaded", script.name), false);
        }
    }
    if register.is_empty() {
        return;
    }
    let mut names = Vec::new();
    for (script, stamp) in register {
        let key = script.name.to_lowercase();
        let d = add_def(make_def(&script));
        with(|r| {
            let generation = r.entries.get(&key).map_or(1, |e| e.generation + 1);
            r.defs.insert(d, key.clone());
            r.entries.insert(key.clone(), Entry { script: script.clone(), generation, stamp });
        });
        names.push(script.name);
    }
    match super::reregister() {
        Ok(()) => {
            for n in &names {
                let c = CString::new(n.as_str()).unwrap_or_default();
                let found = super::find_shader(&c);
                let ours = super::listed_shader(&c);
                let ok = found != 0 && found == ours;
                log::line(&format!("material shader {n}: registered, lookup {}", if ok { "returns it" } else { "returns something else" }));
                crate::overlay::notify(if ok { format!("{n}: registered (reload materials using it)") } else { format!("{n}: registered, but the game still finds another shader by that name") }, !ok);
            }
        }
        Err(e) => {
            log::line(&format!("material shaders: registering failed: {e}"));
            crate::overlay::notify(format!("material shaders: registering failed: {e}"), true);
        }
    }
}

/// compile the generated pixel shader once before accepting a file
fn validate(s: &Script) -> Result<(), String> {
    let opts = PsOptions { srgb_read: 1, srgb_write: true, alpha_test: None, shadow_depth: 0 };
    let wgsl = generate(s, &opts, true);
    let module = naga::front::wgsl::parse_str(&wgsl).map_err(|e| e.emit_to_string(&wgsl))?;
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).map_err(|e| format!("{e:?}"))?;
    Ok(())
}

// ---- the def ----

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

fn make_def(s: &Script) -> &'static Def {
    let name: &'static CStr = Box::leak(CString::new(s.name.as_str()).unwrap_or_default().into_boxed_c_str());
    let params: Vec<Param> = s
        .params
        .iter()
        .map(|p| Param {
            name: leak(&p.name),
            ty: p.ty,
            default: leak(if p.is_texture() { "" } else { &p.default }),
            help: leak(if p.help.is_empty() { &p.name } else { &p.help }),
            flags: 0,
        })
        .collect();
    Box::leak(Box::new(Def {
        names: Box::leak(vec![name].into_boxed_slice()),
        stock: None,
        params: Box::leak(params.into_boxed_slice()),
        supports: |_| true,
        init_params,
        init_instance,
        draw,
        translucent: |p| p.var("$flags").is_some_and(|v| v.int() & (TRANSLUCENT | ADDITIVE) != 0),
    }))
}

const TEXTUREFLAGS_SRGB: i32 = 0x40;
const VERTEX_POSITION: u32 = 1;
const VERTEX_NORMAL: u32 = 2;
/// compressed vertices are fine: the built-in vertex shader unpacks them
const VERTEX_FORMAT_COMPRESSED: u32 = 0x400;
// MaterialVarFlags_t
const ALPHATEST: i32 = 1 << 8;
const ADDITIVE: i32 = 1 << 7;
const NOCULL: i32 = 1 << 13;
const IGNOREZ: i32 = 1 << 15;
const DECAL: i32 = 1 << 16;
const TRANSLUCENT: i32 = 1 << 21;
// MaterialVarFlags2_t
const LIGHTING_VERTEX_LIT: i32 = 1 << 1;
const NEEDS_TANGENT_SPACES: i32 = 1 << 6;
const NEEDS_BAKED_LIGHTING_SNAPSHOTS: i32 = 1 << 9;
const SUPPORTS_HW_SKINNING: i32 = 1 << 18;
const TEXTURE_WHITE: i32 = 4;
const BLEND_ONE: i32 = 1;
const BLEND_SRC_ALPHA: i32 = 4;
const BLEND_ONE_MINUS_SRC_ALPHA: i32 = 5;

unsafe fn init_params(p: Params) {
    let Some((s, _)) = by_def(p.def) else { return };
    let mut bits = LIGHTING_VERTEX_LIT | SUPPORTS_HW_SKINNING | NEEDS_BAKED_LIGHTING_SNAPSHOTS;
    if s.uses_tangents() {
        bits |= NEEDS_TANGENT_SPACES;
    }
    for name in ["$flags2", "$flags_defined2"] {
        if let Some(v) = p.var(name) {
            v.set_int(v.int() | bits);
        }
    }
    // materialsystem zero-fills undefined params afterwards: set the defaults now
    for param in super::STANDARD.iter().chain(super::def(p.def).params) {
        if param.default.is_empty() || matches!(param.ty, ParamType::Texture) {
            continue;
        }
        let Some(v) = p.var(&param.name.to_lowercase()).filter(|v| !v.defined()) else { continue };
        let nums: Vec<f32> = param.default.trim_matches(['[', ']', '{', '}']).split_whitespace().filter_map(|x| x.parse().ok()).collect();
        match (param.ty, nums.as_slice()) {
            (ParamType::Integer | ParamType::Bool, [x, ..]) => v.set_int(*x as i32),
            (ParamType::Float, [x, ..]) => v.set_float(*x),
            (ParamType::Color | ParamType::Vec3 | ParamType::Vec4, [x, y, z, ..]) => v.set_vec3(*x, *y, *z),
            (ParamType::Vec2, [x, y, ..]) => v.set_vec3(*x, *y, 0.0),
            (ParamType::Color | ParamType::Vec3, [x]) => v.set_vec3(*x, *x, *x),
            _ => {}
        }
    }
}

/// IShaderInit: LoadTexture 0, LoadCubeMap 2
unsafe fn init_instance(p: Params, init: usize, group: *const c_char) {
    if init == 0 {
        return;
    }
    let Some((s, _)) = by_def(p.def) else { return };
    let load = |name: &str, flags: i32| {
        if let Some(v) = p.var(name).filter(|v| v.defined()) {
            unsafe { vfn::<unsafe extern "C" fn(usize, usize, *const c_char, i32)>(init, 0)(init, v.0, group, flags) };
        }
    };
    load("$basetexture", TEXTUREFLAGS_SRGB);
    for t in s.params.iter().filter(|t| t.is_texture()) {
        let name = t.name.to_lowercase();
        match t.kind.as_str() {
            "cubemap" => {
                if let Some(v) = p.var(&name).filter(|v| v.defined()) {
                    unsafe { vfn::<unsafe extern "C" fn(usize, *mut *mut c_void, usize, i32)>(init, 2)(init, p.ptr, v.0, 0) };
                }
            }
            "texture_srgb" => load(&name, TEXTUREFLAGS_SRGB),
            _ => load(&name, 0),
        }
    }
}

fn gamma(c: [f32; 3]) -> [f32; 3] {
    c.map(|x| x.max(0.0).powf(2.2))
}

/// seconds since the first script draw
fn time() -> f32 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f32()
}

unsafe fn draw(p: Params, shadow: usize, dynamic: usize) {
    let Some((s, _)) = by_def(p.def) else { return };
    let flags = p.var("$flags").map_or(0, |v| v.int());
    let sys = shader_system();
    if shadow != 0 {
        unsafe {
            let call0 = |slot: usize| vfn::<unsafe extern "C" fn(usize)>(shadow, slot)(shadow);
            let call_b = |slot: usize, v: bool| vfn::<unsafe extern "C" fn(usize, bool)>(shadow, slot)(shadow, v);
            let call_i = |slot: usize, v: i32| vfn::<unsafe extern "C" fn(usize, i32)>(shadow, slot)(shadow, v);
            let call_ib = |slot: usize, i: i32, v: bool| vfn::<unsafe extern "C" fn(usize, i32, bool)>(shadow, slot)(shadow, i, v);
            call0(sh::slot::SetDefaultState);
            if flags & IGNOREZ != 0 {
                call_b(sh::slot::EnableDepthTest, false);
                call_b(sh::slot::EnableDepthWrites, false);
            }
            if flags & DECAL != 0 {
                call_i(sh::slot::EnablePolyOffset, 1);
            }
            if flags & NOCULL != 0 {
                call_b(sh::slot::EnableCulling, false);
            }
            if flags & (TRANSLUCENT | ADDITIVE) != 0 {
                call_b(sh::slot::EnableBlending, true);
                let dst = if flags & ADDITIVE != 0 { BLEND_ONE } else { BLEND_ONE_MINUS_SRC_ALPHA };
                vfn::<unsafe extern "C" fn(usize, i32, i32)>(shadow, sh::slot::BlendFunc)(shadow, BLEND_SRC_ALPHA, dst);
                call_b(sh::slot::EnableDepthWrites, false);
            }
            if flags & ALPHATEST != 0 {
                call_b(sh::slot::EnableAlphaTest, true);
                let r = p.float_or("$alphatestreference", 0.0);
                vfn::<unsafe extern "C" fn(usize, i32, f32)>(shadow, sh::slot::AlphaFunc)(shadow, 6, if r > 0.0 { r } else { 0.5 });
            }
            call_ib(sh::slot::EnableTexture, 0, true);
            call_ib(sh::slot::EnableSRGBRead, 0, true);
            for t in s.params.iter().filter(|t| t.is_texture()) {
                call_ib(sh::slot::EnableTexture, t.slot, true);
                call_ib(sh::slot::EnableSRGBRead, t.slot, t.kind == "texture_srgb");
            }
            call_b(sh::slot::EnableSRGBWrite, true);
            let dims = [2i32];
            let user_data = if s.uses_tangents() { 4 } else { 0 };
            vfn::<unsafe extern "C" fn(usize, u32, i32, *const i32, i32)>(shadow, sh::slot::VertexShaderVertexFormat)(shadow, VERTEX_POSITION | VERTEX_NORMAL | VERTEX_FORMAT_COMPRESSED, 1, dims.as_ptr(), user_data);
            let vs = CString::new(ggx::VS).unwrap();
            let ps = CString::new(ps_name(&s.name)).unwrap_or_default();
            vfn::<unsafe extern "C" fn(usize, *const c_char, i32)>(shadow, sh::slot::SetVertexShader)(shadow, vs.as_ptr(), 0);
            vfn::<unsafe extern "C" fn(usize, *const c_char, i32)>(shadow, sh::slot::SetPixelShader)(shadow, ps.as_ptr(), 0);
            call_i(sh::slot::FogMode, if flags & ADDITIVE != 0 { 2 } else { 4 });
            if sys != 0 {
                // IShaderSystem::TakeSnapshot
                vfn::<unsafe extern "C" fn(usize)>(sys, 3)(sys);
            }
        }
        return;
    }
    if dynamic == 0 {
        return;
    }
    unsafe {
        // IShaderSystem::BindTexture(sampler, ITexture*, frame), slot 2
        let bind = |sampler: i32, tex: &str, frame: &str| -> bool {
            let Some(v) = p.var(tex).filter(|v| v.defined()) else { return false };
            let t = v.texture();
            if t == 0 || sys == 0 {
                return false;
            }
            let f = p.var(frame).filter(|v| v.defined()).map_or(0, |v| v.int());
            vfn::<unsafe extern "C" fn(usize, i32, usize, i32)>(sys, 2)(sys, sampler, t, f);
            true
        };
        let standard = |sampler: i32, id: i32| vfn::<unsafe extern "C" fn(usize, i32, i32)>(dynamic, api::slot::BindStandardTexture)(dynamic, sampler, id);
        if !bind(0, "$basetexture", "$frame") {
            standard(0, TEXTURE_WHITE);
        }
        for t in s.params.iter().filter(|t| t.is_texture()) {
            if !bind(t.slot, &t.name.to_lowercase(), "$frame") {
                standard(t.slot, TEXTURE_WHITE);
            }
        }
        // base texture transform rows -> c48, c49
        let m = p.var("$basetexturetransform").filter(|v| v.defined()).map_or([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], |v| v.matrix());
        vfn::<unsafe extern "C" fn(usize, i32, *const f32, i32, bool)>(dynamic, api::slot::SetVertexShaderConstant)(dynamic, 48, m.as_ptr(), 2, false);

        let ps = crate::api::builtin_set_ps_reg;
        let color = gamma(p.vec3_or("$color", [1.0; 3]));
        let color2 = gamma(p.vec3_or("$color2", [1.0; 3]));
        ps(ggx::C_TINT, [color[0] * color2[0], color[1] * color2[1], color[2] * color2[2], p.float_or("$alpha", 1.0)]);
        let eye = crate::api::builtin_eye_position();
        ps(ggx::C_EYE, [eye[0], eye[1], eye[2], 1.0]);
        let lights = crate::api::ps_lights_full(ggx::C_LIGHTS);
        ps(C_EXTRA, [time(), lights as f32, 0.0, 0.0]);
        for t in s.params.iter().filter(|t| !t.is_texture()) {
            let name = t.name.to_lowercase();
            let v = p.var(&name).filter(|v| v.defined());
            let reg = match t.ty {
                ParamType::Integer | ParamType::Bool => [v.map_or(0.0, |v| v.int() as f32), 0.0, 0.0, 0.0],
                ParamType::Float => [v.map_or(0.0, |v| v.float()), 0.0, 0.0, 0.0],
                ParamType::Color => {
                    let c = gamma(p.vec3_or(&name, [1.0; 3]));
                    [c[0], c[1], c[2], 1.0]
                }
                _ => v.map_or([0.0; 4], |v| v.vec()),
            };
            ps(t.slot, reg);
        }
        crate::api::builtin_ps_ambient_cube(ggx::C_AMBIENT);
        crate::api::builtin_ps_fog_params(ggx::C_FOG);
        vfn::<unsafe extern "C" fn(usize, i32)>(dynamic, api::slot::SetVertexShaderIndex)(dynamic, 0);
        vfn::<unsafe extern "C" fn(usize, i32)>(dynamic, api::slot::SetPixelShaderIndex)(dynamic, 0);
        if sys != 0 {
            // IShaderSystem::DrawSnapshot(true)
            vfn::<unsafe extern "C" fn(usize, bool)>(sys, 4)(sys, true);
        }
    }
}

// ---- wgsl ----

/// renderer entry: the pixel shader behind a wk_script_* name
pub fn translate_ps(name: &str, opts: &PsOptions, mv: bool) -> Result<Translated, String> {
    let key = name.split('/').next().unwrap_or(name).strip_prefix(PS_PREFIX).and_then(|r| r.strip_suffix("_ps")).ok_or_else(|| format!("bad script shader name {name}"))?;
    let script = with(|r| r.entries.get(key).map(|e| e.script.clone())).ok_or_else(|| format!("no material shader {key}"))?;
    let wgsl = generate(&script, opts, mv);
    let mut samplers: Vec<(u32, u8)> = vec![(0, crate::d3d2wgsl::DIM_2D)];
    for t in script.params.iter().filter(|t| t.is_texture()) {
        samplers.push((t.slot as u32, if t.kind == "cubemap" { crate::d3d2wgsl::DIM_CUBE } else { crate::d3d2wgsl::DIM_2D }));
    }
    Ok(Translated { wgsl, samplers, inputs: Vec::new() })
}

/// the full pixel shader: constants and textures, the Surface the script shades, helpers, the
/// script, then source's output (tonemap scale, fog, alpha test, srgb, motion vectors)
pub fn generate(s: &Script, opts: &PsOptions, mv: bool) -> String {
    let mut out = format!("diagnostic(off, derivative_uniformity);\nstruct PConsts {{\n    c: array<vec4f, {PS_FLOAT_CONSTS}>,\n    i: array<vec4<i32>, 16>,\n    b: vec4<u32>,\n}}\n@group(0) @binding(1) var<uniform> P: PConsts;\n");
    let textures: Vec<&ScriptParam> = s.params.iter().filter(|t| t.is_texture()).collect();
    let _ = writeln!(out, "@group(1) @binding(0) var tex0: texture_2d<f32>;\n@group(1) @binding(1) var smp0: sampler;");
    for t in &textures {
        let ty = if t.kind == "cubemap" { "texture_cube<f32>" } else { "texture_2d<f32>" };
        let _ = writeln!(out, "@group(1) @binding({}) var tex{}: {ty};\n@group(1) @binding({}) var smp{}: sampler;", 2 * t.slot, t.slot, 2 * t.slot + 1, t.slot);
    }
    out += "fn srgb_to_linear(c: vec3f) -> vec3f { return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045)); }\n";
    out += "fn linear_to_srgb(c: vec3f) -> vec3f { let x = clamp(c, vec3f(0.0), vec3f(1.0)); return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3f(0.0031308)); }\n";
    let fetch = |n: i32| if opts.srgb_read & (1 << n) != 0 { "vec4f(srgb_to_linear(c.rgb), c.a)" } else { "c" };
    let _ = writeln!(out, "fn base_texture(uv: vec2f) -> vec4f {{ let c = textureSampleBias(tex0, smp0, uv, bitcast<f32>(P.b.y)); return {}; }}", fetch(0));
    for t in &textures {
        let (arg, coord) = if t.kind == "cubemap" { ("dir: vec3f", "dir") } else { ("uv: vec2f", "uv") };
        let _ = writeln!(out, "fn sample_{}({arg}) -> vec4f {{ let c = textureSampleBias(tex{n}, smp{n}, {coord}, bitcast<f32>(P.b.y)); return {}; }}", t.field(), fetch(t.slot), n = t.slot);
        let _ = writeln!(out, "fn sample_{}_level({arg}, level: f32) -> vec4f {{ let c = textureSampleLevel(tex{n}, smp{n}, {coord}, level); return {}; }}", t.field(), fetch(t.slot), n = t.slot);
    }
    // the material's number params
    let numbers: Vec<&ScriptParam> = s.params.iter().filter(|t| !t.is_texture()).collect();
    if !numbers.is_empty() {
        out += "struct Material {\n";
        for t in &numbers {
            let ty = match t.kind.as_str() {
                "float" => "f32",
                "int" => "i32",
                "bool" => "bool",
                "vec2" => "vec2f",
                "vec3" | "color" => "vec3f",
                _ => "vec4f",
            };
            let _ = writeln!(out, "    {}: {ty},", t.field());
        }
        out += "}\nfn material() -> Material {\n    var m: Material;\n";
        for t in &numbers {
            let r = format!("P.c[{}]", t.slot);
            let v = match t.kind.as_str() {
                "float" => format!("{r}.x"),
                "int" => format!("i32({r}.x)"),
                "bool" => format!("{r}.x > 0.5"),
                "vec2" => format!("{r}.xy"),
                "vec3" | "color" => format!("{r}.xyz"),
                _ => r,
            };
            let _ = writeln!(out, "    m.{} = {v};", t.field());
        }
        out += "    return m;\n}\n";
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

/// what the script shades: all directions in world space, normalized
struct Surface {{
    /// $basetexturetransform applied, and the raw texcoord
    uv: vec2f,
    uv_raw: vec2f,
    world: vec3f,
    normal: vec3f,
    tangent: vec3f,
    binormal: vec3f,
    /// towards the eye
    view: vec3f,
    eye: vec3f,
    /// baked per-vertex lighting of static props
    vertex_light: vec3f,
    has_vertex_light: bool,
    /// pixel position
    screen: vec2f,
    front_facing: bool,
}}

/// $color * $color2 (linear) and $alpha
fn tint() -> vec4f {{ return P.c[{tint}]; }}
/// seconds, for animation
fn time() -> f32 {{ return P.c[{extra}].x; }}

/// the ambient light cube along a normal
fn ambient(n: vec3f) -> vec3f {{
    let n2 = n * n;
    let x = select(P.c[{a} + 1].rgb, P.c[{a}].rgb, n.x >= 0.0);
    let y = select(P.c[{a} + 3].rgb, P.c[{a} + 2].rgb, n.y >= 0.0);
    let z = select(P.c[{a} + 5].rgb, P.c[{a} + 4].rgb, n.z >= 0.0);
    return n2.x * x + n2.y * y + n2.z * z;
}}

fn light_count() -> i32 {{ return i32(P.c[{extra}].y); }}
fn light_color(i: i32) -> vec3f {{ return P.c[{l} + 5 * i].rgb; }}

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

/// towards light i
fn light_dir(i: i32, world: vec3f) -> vec3f {{
    let base = {l} + 5 * i;
    let to_light = normalize(P.c[base + 2].xyz - world);
    return normalize(mix(to_light, -P.c[base + 1].xyz, P.c[base].w));
}}

/// source-style diffuse lighting along n: baked vertex light or the ambient cube, plus the
/// dynamic lights (lambert)
fn diffuse_lighting(s: Surface, n: vec3f) -> vec3f {{
    var c = select(ambient(n), s.vertex_light, s.has_vertex_light);
    for (var i = 0; i < light_count(); i++) {{
        c += light_color(i) * light_atten(i, s.world) * max(dot(n, light_dir(i, s.world)), 0.0);
    }}
    return c;
}}

"#,
        tint = ggx::C_TINT,
        extra = C_EXTRA,
        a = ggx::C_AMBIENT,
        l = ggx::C_LIGHTS,
    );
    // the script, minus its header
    for line in s.body.lines() {
        if !line.trim_start().starts_with("//!") {
            out += line;
            out.push('\n');
        }
    }
    let _ = write!(
        out,
        r#"
@fragment
fn main(fin: FIn) -> FOut {{
    var s: Surface;
    s.uv = fin.l0.xy;
    s.uv_raw = fin.l0.zw;
    s.world = fin.l1.xyz;
    s.normal = normalize(fin.l2.xyz);
    s.tangent = normalize(fin.l3.xyz - s.normal * dot(fin.l3.xyz, s.normal) + vec3f(1e-6, 0.0, 0.0));
    s.binormal = cross(s.normal, s.tangent) * fin.l3.w;
    s.eye = P.c[{eye}].xyz;
    s.view = normalize(s.eye - s.world);
    s.vertex_light = fin.l4.rgb;
    s.has_vertex_light = fin.l4.w > 0.5;
    s.screen = fin.fpos.xy;
    s.front_facing = fin.ff;
    let shaded = shade(s);
    // stock FinalOutput: linear tonemap scale (c30), then range fog towards c29
    var color = shaded.rgb * P.c[30].x;
    let fog = P.c[{fog}];
    let fog_factor = clamp(min(fog.z, fin.l1.w * fog.w - fog.x), 0.0, 1.0);
    color = mix(color, P.c[29].rgb, fog_factor);
    var outc = vec4f(color, shaded.a);
"#,
        eye = ggx::C_EYE,
        fog = ggx::C_FOG,
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
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOON: &str = r#"//! shader: Toon
//! param: $bands float 3 "number of light bands"
//! param: $shadowtint color [0.2 0.2 0.3]
//! param: $ramp texture
//! param: $sky cubemap

fn shade(s: Surface) -> vec4f {
    let m = material();
    let base = base_texture(s.uv) * tint();
    let light = diffuse_lighting(s, s.normal);
    let banded = floor(light * m.bands) / m.bands;
    let r = sample_ramp(vec2f(banded.r, 0.5)).rgb + sample_sky(reflect(-s.view, s.normal)).rgb * 0.0;
    return vec4f(base.rgb * mix(m.shadowtint, vec3f(1.0), banded) * r, base.a);
}
"#;

    #[test]
    fn parses_header() {
        let s = parse(TOON, PathBuf::from("toon.wgsl")).unwrap();
        assert_eq!(s.name, "Toon");
        assert_eq!(s.params.len(), 4);
        assert_eq!((s.params[0].slot, s.params[1].slot), (FIRST_REG, FIRST_REG + 1));
        assert_eq!((s.params[2].slot, s.params[3].slot), (1, 2));
        assert_eq!(s.params[0].help, "number of light bands");
        assert!(parse("fn shade(s: Surface) -> vec4f { return vec4f(1.0); }", PathBuf::new()).is_err());
        assert!(parse("//! shader: X\n//! param: $alpha float 1\nfn shade(s: Surface) -> vec4f { return vec4f(1.0); }", PathBuf::new()).is_err());
    }

    #[test]
    fn shipped_material_shaders_validate() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("effects/materials");
        for e in std::fs::read_dir(dir).unwrap() {
            let path = e.unwrap().path();
            let s = parse(&std::fs::read_to_string(&path).unwrap(), path.clone()).unwrap();
            validate(&s).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        }
    }

    #[test]
    fn generated_shader_validates() {
        let s = parse(TOON, PathBuf::from("toon.wgsl")).unwrap();
        for (mv, alpha) in [(false, None), (true, Some((6, 128)))] {
            let opts = PsOptions { srgb_read: 1, srgb_write: true, alpha_test: alpha, shadow_depth: 0 };
            let wgsl = generate(&s, &opts, mv);
            let module = naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|e| panic!("{}\n{wgsl}", e.emit_to_string(&wgsl)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
        }
    }
}
