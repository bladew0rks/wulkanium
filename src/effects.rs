//! screen effects loaded from garrysmod/shaders/wulkanium/*.wgsl. they run after the 3d view
//! (and dlss) but before the hud, and see the scene color, the real depth buffer, the camera and
//! motion vectors. each file is one effect: a header of `//!` lines declares its textures, passes
//! and parameters, and the body is wgsl with a generated prelude in front (see EFFECTS.md).
//! files are polled for changes, so effects reload while the game runs.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Instant, SystemTime};

use crate::log;

pub const DIR: &str = "garrysmod/shaders/wulkanium";
/// for the lua addon: which effects and parameters exist
const MANIFEST: &str = "garrysmod/data/wulkanium_effects.json";

/// where a pass draws
#[derive(Clone, Debug, PartialEq)]
pub enum PassTarget {
    Screen,
    Texture(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pass {
    pub entry: String,
    pub target: PassTarget,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextureDef {
    pub name: String,
    pub format: wgpu::TextureFormat,
    /// of the screen size, or of the depth buffer's (`depth`)
    pub scale: f32,
    /// sized like the depth buffer: smaller than the screen under dlss upscaling
    pub depth: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: String,
    pub default: f32,
    pub min: f32,
    pub max: f32,
    /// whole numbers only
    pub int: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EffectDef {
    pub name: String,
    /// lowercase identifier for convars
    pub key: String,
    pub order: i32,
    pub enabled: bool,
    pub textures: Vec<TextureDef>,
    pub passes: Vec<Pass>,
    pub params: Vec<Param>,
    pub body: String,
}

fn ident(s: &str) -> bool {
    let mut c = s.chars();
    c.next().is_some_and(|f| f.is_ascii_alphabetic() || f == '_') && c.all(|x| x.is_ascii_alphanumeric() || x == '_') && !s.starts_with("wk_")
}

fn format(s: &str) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as F;
    Some(match s {
        "r8unorm" => F::R8Unorm,
        "rg8unorm" => F::Rg8Unorm,
        "rgba8unorm" => F::Rgba8Unorm,
        "r16float" => F::R16Float,
        "rg16float" => F::Rg16Float,
        "rgba16float" => F::Rgba16Float,
        "r32float" => F::R32Float,
        "rg32float" => F::Rg32Float,
        "rgba32float" => F::Rgba32Float,
        _ => return None,
    })
}

/// reads the `//!` header; errors name the line
pub fn parse(file_stem: &str, src: &str) -> Result<EffectDef, String> {
    let mut def = EffectDef {
        name: file_stem.to_string(),
        key: String::new(),
        order: 0,
        enabled: false,
        textures: Vec::new(),
        passes: Vec::new(),
        params: Vec::new(),
        body: src.to_string(),
    };
    // targets resolve after every texture is known
    let mut pass_targets: Vec<(usize, String, String)> = Vec::new();
    for (n, line) in src.lines().enumerate() {
        let n = n + 1;
        let Some(rest) = line.trim_start().strip_prefix("//!") else { continue };
        let Some((key, value)) = rest.split_once(':') else { return Err(format!("line {n}: expected `//! key: value`")) };
        let value = value.trim();
        let words: Vec<&str> = value.split_whitespace().collect();
        let num = |i: usize, what: &str| -> Result<f32, String> { words.get(i).ok_or(format!("line {n}: missing {what}"))?.parse::<f32>().map_err(|_| format!("line {n}: {what} is not a number")) };
        match key.trim() {
            "effect" => def.name = value.to_string(),
            "order" => def.order = num(0, "order")? as i32,
            "enabled" => def.enabled = matches!(value, "1" | "on" | "true" | "yes"),
            "texture" => {
                let name = words.first().ok_or(format!("line {n}: missing texture name"))?;
                if !ident(name) {
                    return Err(format!("line {n}: bad texture name '{name}'"));
                }
                let fmt = words.get(1).ok_or(format!("line {n}: missing texture format"))?;
                let format = format(fmt).ok_or(format!("line {n}: unknown format '{fmt}'"))?;
                let depth = words.get(3).or(words.get(2)) == Some(&"depth");
                let scale = if words.len() > 2 && words[2] != "depth" { num(2, "scale")? } else { 1.0 };
                if !(0.0..=4.0).contains(&scale) || scale == 0.0 {
                    return Err(format!("line {n}: scale must be in (0, 4]"));
                }
                def.textures.push(TextureDef { name: name.to_string(), format, scale, depth });
            }
            "pass" => {
                let (entry, target) = value.split_once("->").ok_or(format!("line {n}: expected `pass: entry -> target`"))?;
                let (entry, target) = (entry.trim(), target.trim());
                if !ident(entry) {
                    return Err(format!("line {n}: bad entry point '{entry}'"));
                }
                pass_targets.push((n, entry.to_string(), target.to_string()));
            }
            "param" => {
                let name = words.first().ok_or(format!("line {n}: missing param name"))?;
                if !ident(name) {
                    return Err(format!("line {n}: bad param name '{name}'"));
                }
                let int = words.last() == Some(&"int");
                let words = &words[..words.len() - int as usize];
                let num = |i: usize, what: &str| -> Result<f32, String> { words.get(i).ok_or(format!("line {n}: missing {what}"))?.parse::<f32>().map_err(|_| format!("line {n}: {what} is not a number")) };
                let default = num(1, "default")?;
                let (min, max) = if words.len() > 3 { (num(2, "min")?, num(3, "max")?) } else { (default.min(0.0), default.max(1.0)) };
                def.params.push(Param { name: name.to_string(), default, min, max, int });
            }
            other => return Err(format!("line {n}: unknown key '{other}'")),
        }
    }
    for (n, entry, target) in pass_targets {
        let target = if target == "screen" {
            PassTarget::Screen
        } else {
            PassTarget::Texture(def.textures.iter().position(|t| t.name == target).ok_or(format!("line {n}: unknown target '{target}'"))?)
        };
        def.passes.push(Pass { entry, target });
    }
    if def.passes.is_empty() {
        return Err("no `//! pass:` lines".into());
    }
    let mut names: Vec<&str> = def.textures.iter().map(|t| t.name.as_str()).chain(def.params.iter().map(|p| p.name.as_str())).collect();
    names.sort();
    if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
        return Err(format!("'{}' is declared twice", w[0]));
    }
    def.key = def.name.to_ascii_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    Ok(def)
}

/// per-frame values every effect sees (matches `WkGlobals` in the prelude)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Globals {
    view: [f32; 16],
    proj: [f32; 16],
    view_proj: [f32; 16],
    inv_view: [f32; 16],
    inv_proj: [f32; 16],
    inv_view_proj: [f32; 16],
    prev_view_proj: [f32; 16],
    camera_pos: [f32; 4],
    /// w, h, 1/w, 1/h
    resolution: [f32; 4],
    time: f32,
    delta: f32,
    frame: u32,
    /// bit 0 camera known, bit 1 motion vectors valid, bit 2 dynamic object mask valid, bit 3
    /// sun shadow mask valid
    flags: u32,
    /// the depth buffer's sub-pixel jitter (dlss), in ndc
    jitter: [f32; 2],
    /// the part of the depth / motion textures the scene covers (smaller under dlss upscaling)
    depth_size: [f32; 2],
    /// the viewmodel's projection and its inverse (see wk_vm_view_pos)
    vm_proj: [f32; 16],
    vm_inv_proj: [f32; 16],
    /// viewmodel depth range max, min, 1 when there is a viewmodel
    vm: [f32; 4],
    /// the viewmodel's lights blended into one direction towards them, in its view space (w = 1);
    /// w = 0 without lights. vm_light_color.w: how much the lights agree on that direction
    vm_light: [f32; 4],
    vm_light_color: [f32; 4],
    /// world-space direction towards the sun (w = 1), w = 0 without one
    sun: [f32; 4],
    sun_color: [f32; 4],
}

pub const PRELUDE: &str = r#"
struct WkGlobals {
    view: mat4x4f,
    proj: mat4x4f,
    view_proj: mat4x4f,
    inv_view: mat4x4f,
    inv_proj: mat4x4f,
    inv_view_proj: mat4x4f,
    prev_view_proj: mat4x4f,
    camera_pos: vec4f,
    resolution: vec4f,
    time: f32,
    delta: f32,
    frame: u32,
    flags: u32,
    jitter: vec2f,
    depth_size: vec2f,
    vm_proj: mat4x4f,
    vm_inv_proj: mat4x4f,
    vm: vec4f,
    vm_light: vec4f,
    vm_light_color: vec4f,
    sun: vec4f,
    sun_color: vec4f,
}
@group(0) @binding(0) var<uniform> wk: WkGlobals;
@group(0) @binding(1) var wk_color: texture_2d<f32>;
@group(0) @binding(2) var wk_depth_tex: texture_depth_2d;
@group(0) @binding(3) var wk_motion: texture_2d<f32>;
@group(0) @binding(4) var wk_linear: sampler;
@group(0) @binding(5) var wk_point: sampler;
@group(0) @binding(7) var wk_final_depth_tex: texture_depth_2d;
@group(0) @binding(8) var wk_stencil: texture_2d<u32>;
@group(0) @binding(9) var wk_sun_shadow_tex: texture_2d<f32>;

struct WkVsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn wk_vs(@builtin(vertex_index) i: u32) -> WkVsOut {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    var o: WkVsOut;
    o.pos = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    o.uv = uv;
    return o;
}

fn wk_has_camera() -> bool { return (wk.flags & 1u) != 0u; }
fn wk_has_motion() -> bool { return (wk.flags & 2u) != 0u; }

/// the depth pixel under a screen uv (under dlss upscaling depth covers only part of its texture)
fn wk_pixel(uv: vec2f) -> vec2i {
    return clamp(vec2i(floor(uv * wk.depth_size)), vec2i(0), vec2i(wk.depth_size) - 1);
}

/// the centre of that pixel, as a screen uv: reconstruct positions there, not at the raw uv, or
/// points land beside the surface they were read from
fn wk_snap(uv: vec2f) -> vec2f {
    return (vec2f(wk_pixel(uv)) + 0.5) / wk.depth_size;
}

fn wk_depth(uv: vec2f) -> f32 {
    return textureLoad(wk_depth_tex, wk_pixel(uv), 0);
}

fn wk_is_sky(depth: f32) -> bool { return depth >= 1.0; }

/// something drawn after the solid scene (water) covers this pixel: wk_depth is what's under it
fn wk_covered(uv: vec2f) -> bool {
    let p = wk_pixel(uv);
    return textureLoad(wk_final_depth_tex, p, 0) < textureLoad(wk_depth_tex, p, 0) - 1e-6;
}

/// the viewmodel draws with its depth squeezed in front of the world
fn wk_is_viewmodel(depth: f32) -> bool { return depth < 0.1; }

/// ndc of a depth-buffer position, without the depth buffer's jitter
fn wk_ndc(uv: vec2f, depth: f32) -> vec4f {
    return vec4f(uv.x * 2.0 - 1.0 - wk.jitter.x, 1.0 - uv.y * 2.0 - wk.jitter.y, depth, 1.0);
}

fn wk_view_pos(uv: vec2f, depth: f32) -> vec3f {
    let v = wk.inv_proj * wk_ndc(uv, depth);
    return v.xyz / v.w;
}

fn wk_world_pos(uv: vec2f, depth: f32) -> vec3f {
    let w = wk.inv_view_proj * wk_ndc(uv, depth);
    return w.xyz / w.w;
}

/// view / world position of the surface at a uv (snapped to its pixel)
fn wk_view_pos_at(uv: vec2f) -> vec3f {
    let s = wk_snap(uv);
    return wk_view_pos(s, wk_depth(s));
}

fn wk_world_pos_at(uv: vec2f) -> vec3f {
    let s = wk_snap(uv);
    return wk_world_pos(s, wk_depth(s));
}

/// world position -> (uv, depth), unjittered like wk_color
fn wk_project(world: vec3f) -> vec3f {
    let c = wk.view_proj * vec4f(world, 1.0);
    let n = c.xyz / c.w;
    return vec3f(n.x * 0.5 + 0.5, 0.5 - n.y * 0.5, n.z);
}

/// distance from the camera in world units
fn wk_distance(uv: vec2f) -> f32 { return length(wk_view_pos_at(uv)); }

/// view-space normal rebuilt from depth, facing the camera
fn wk_view_normal(uv: vec2f) -> vec3f {
    let px = 1.0 / wk.depth_size;
    let s = wk_snap(uv);
    let c = wk_view_pos_at(s);
    let l = wk_view_pos_at(s - vec2f(px.x, 0.0));
    let r = wk_view_pos_at(s + vec2f(px.x, 0.0));
    let u = wk_view_pos_at(s - vec2f(0.0, px.y));
    let d = wk_view_pos_at(s + vec2f(0.0, px.y));
    // the smaller difference on each axis avoids smearing across edges
    let dx = select(r - c, c - l, length(c - l) < length(r - c));
    let dy = select(d - c, c - u, length(c - u) < length(d - c));
    var n = normalize(cross(dx, dy));
    if dot(n, -c) < 0.0 { n = -n; }
    return n;
}

/// world-space normal rebuilt from depth, facing the camera
fn wk_world_normal(uv: vec2f) -> vec3f {
    return normalize((wk.inv_view * vec4f(wk_view_normal(uv), 0.0)).xyz);
}

fn wk_has_viewmodel() -> bool { return wk.vm.z > 0.5; }

/// a viewmodel pixel's position in the viewmodel's own view space (its projection, its depth
/// range); wk_view_pos is only right for the world
fn wk_vm_view_pos(uv: vec2f, depth: f32) -> vec3f {
    let nz = (depth - wk.vm.y) / max(wk.vm.x - wk.vm.y, 1e-6);
    let v = wk.vm_inv_proj * wk_ndc(uv, nz);
    return v.xyz / v.w;
}

/// viewmodel view position -> (uv, raw depth), jittered like the depth buffer
fn wk_vm_project(p: vec3f) -> vec3f {
    let c = wk.vm_proj * vec4f(p, 1.0);
    let n = c.xyz / c.w;
    return vec3f((n.x + 1.0 + wk.jitter.x) * 0.5, (1.0 - n.y - wk.jitter.y) * 0.5, wk.vm.y + n.z * (wk.vm.x - wk.vm.y));
}

/// towards the viewmodel's strongest light from a viewmodel view position (zero without one)
/// the map has a sun (light_environment) that models have seen recently
/// the surface at a depth pixel is a moving object (prop, player, npc, brush entity), not part of
/// what the map's lightmaps baked; true for everything when unknown (msaa)
fn wk_is_dynamic(p: vec2i) -> bool {
    if (wk.flags & 4u) == 0u { return true; }
    return (textureLoad(wk_stencil, p, 0).r & 0x80u) != 0u;
}

/// sun shadow at a depth pixel from the screen-space shadow pass (bend studio's), 1 = lit, 0 =
/// shadowed; 1 everywhere without a sun. using it in an effect runs the pass
fn wk_sun_shadow(p: vec2i) -> f32 {
    if (wk.flags & 8u) == 0u { return 1.0; }
    return textureLoad(wk_sun_shadow_tex, p, 0).r;
}

fn wk_has_sun_shadow() -> bool { return (wk.flags & 8u) != 0u; }

fn wk_has_sun() -> bool { return wk.sun.w > 0.5; }

/// world-space direction towards the sun, zero without one
fn wk_sun_dir() -> vec3f { return wk.sun.xyz; }

/// how much the viewmodel's lights agree on one direction: 1 = a single light, towards 0 when
/// lights from opposite sides cancel out
fn wk_vm_light_agreement() -> f32 { return wk.vm_light_color.w; }

fn wk_vm_light_dir(p: vec3f) -> vec3f {
    if wk.vm_light.w < 0.5 { return vec3f(0.0); }
    if wk.vm_light.w < 1.5 { return normalize(wk.vm_light.xyz); }
    return normalize(wk.vm_light.xyz - p);
}

/// screen-space motion of this pixel since last frame, in uv (zero without motion vectors)
fn wk_motion_uv(uv: vec2f) -> vec2f {
    if !wk_has_motion() { return vec2f(0.0); }
    // stored in uv units already, at depth resolution
    return textureLoad(wk_motion, wk_pixel(uv), 0).xy;
}
"#;

/// the prelude plus this effect's own textures and params
fn source(def: &EffectDef) -> String {
    let mut s = String::from(PRELUDE);
    s.push_str("\nstruct WkParams {\n");
    for p in &def.params {
        s.push_str(&format!("    {}: f32,\n", p.name));
    }
    if def.params.is_empty() {
        s.push_str("    _unused: f32,\n");
    }
    s.push_str("}\n@group(0) @binding(6) var<uniform> params: WkParams;\n");
    for (i, t) in def.textures.iter().enumerate() {
        s.push_str(&format!("@group(1) @binding({i}) var {}: texture_2d<f32>;\n", t.name));
    }
    // keep the user's line numbers readable in errors: the body starts on a known line
    s.push_str("// ---- effect ----\n");
    s.push_str(&def.body);
    s
}

/// lines before the effect body, to map error lines back to the file
fn prelude_lines(def: &EffectDef) -> usize {
    source(def).lines().count() - def.body.lines().count()
}

/// the lua side's view of the settings: (enabled, param values) by effect key
pub type Settings = HashMap<String, (bool, Vec<f32>)>;

static INFO: parking_lot::Mutex<Vec<(String, bool, Vec<Param>)>> = parking_lot::Mutex::new(Vec::new());
static SETTINGS: parking_lot::Mutex<Option<Settings>> = parking_lot::Mutex::new(None);

/// game thread, once a frame: read the effects' convars (`wk_fx_<key>`, `wk_fx_<key>_<param>`)
pub fn poll_settings() {
    static CVARS: parking_lot::Mutex<Option<crate::cvar::Cache>> = parking_lot::Mutex::new(None);
    let info = INFO.lock();
    if info.is_empty() {
        return;
    }
    let mut cvars = CVARS.lock();
    let cvars = cvars.get_or_insert_with(Default::default);
    let mut out = Settings::new();
    for (key, enabled, params) in info.iter() {
        let base = format!("wk_fx_{key}");
        let on = cvars.get(&base).map_or(*enabled, |c| c.int() != 0);
        let values = params
            .iter()
            .map(|p| {
                let v = cvars.get(&format!("{base}_{}", p.name.to_ascii_lowercase())).map_or(p.default, |c| c.float()).clamp(p.min.min(p.max), p.max.max(p.min));
                if p.int { v.round() } else { v }
            })
            .collect();
        out.insert(key.clone(), (on, values));
    }
    *SETTINGS.lock() = Some(out);
}

/// any effect switched on (so the renderer marks where the scene ends)
pub fn any_enabled() -> bool {
    SETTINGS.lock().as_ref().is_some_and(|s| s.values().any(|v| v.0)) || (SETTINGS.lock().is_none() && INFO.lock().iter().any(|i| i.1))
}

fn write_manifest(defs: &[EffectDef]) {
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let mut j = String::from("[\n");
    for (i, d) in defs.iter().enumerate() {
        j.push_str(&format!("  {{\"name\": \"{}\", \"key\": \"{}\", \"enabled\": {}, \"params\": [", esc(&d.name), d.key, d.enabled));
        for (k, p) in d.params.iter().enumerate() {
            j.push_str(&format!("{}{{\"name\": \"{}\", \"default\": {}, \"min\": {}, \"max\": {}, \"int\": {}}}", if k > 0 { ", " } else { "" }, esc(&p.name.to_ascii_lowercase()), p.default, p.min, p.max, p.int));
        }
        j.push_str(&format!("]}}{}\n", if i + 1 < defs.len() { "," } else { "" }));
    }
    j.push_str("]\n");
    if let Err(e) = std::fs::write(MANIFEST, j) {
        log::line(&format!("effects: can't write {MANIFEST}: {e}"));
    }
}

/// a compiled effect
struct Loaded {
    def: EffectDef,
    layout1: wgpu::BindGroupLayout,
    pipelines: Vec<wgpu::RenderPipeline>,
    params: wgpu::Buffer,
    /// (texture view, size) per declared texture, made at first use / on resize
    textures: Vec<Option<(wgpu::TextureView, (u32, u32))>>,
}

/// an enabled effect reads `wk_is_dynamic`: scene draws mark moving objects in stencil bit 0x80
pub static DYNAMIC_MASK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// what the renderer hands the effects at the end of the scene
pub struct Inputs<'a> {
    pub color: &'a wgpu::Texture,
    pub color_view: &'a wgpu::TextureView,
    /// depth aspect of the back buffer's depth
    pub depth: &'a wgpu::TextureView,
    pub motion: Option<&'a wgpu::TextureView>,
    /// dlss jitter of the depth buffer, in its pixels
    pub jitter: Option<[f32; 2]>,
    /// the part of depth / motion the scene covers (None = all of it)
    pub depth_size: Option<(u32, u32)>,
    /// (view, proj) of the main 3d view, row-vector convention like the engine's
    pub camera: Option<([f32; 16], [f32; 16])>,
    /// the viewmodel's (view, proj, depth range) and its strongest light
    pub viewmodel: Option<([f32; 16], [f32; 16], [f32; 2])>,
    pub viewmodel_lights: &'a [crate::api::Light],
    pub sun: Option<crate::api::Light>,
    /// stencil aspect of the depth buffer (see `DYNAMIC_MASK`)
    pub stencil: Option<&'a wgpu::TextureView>,
}

/// the debug overlay's buffer views, in order
pub const PREVIEWS: [&str; 3] = ["depth", "normals", "motion"];
const PREVIEW_WIDTH: u32 = 320;

static PREVIEWS_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// the overlay asks for buffer previews while it's open
pub fn set_previews(on: bool) {
    PREVIEWS_ON.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn previews_on() -> bool {
    PREVIEWS_ON.load(std::sync::atomic::Ordering::Relaxed)
}

const PREVIEW_BODY: &str = r#"
@fragment
fn pv_depth(i: WkVsOut) -> @location(0) vec4f {
    let d = wk_depth(i.uv);
    if wk_is_sky(d) { return vec4f(0.04, 0.05, 0.08, 1.0); }
    if !wk_has_camera() { return vec4f(vec3f(1.0 - d), 1.0); }
    // distance on a log scale, 16 to 8192 units
    let dist = length(wk_view_pos(wk_snap(i.uv), d));
    let t = clamp(log2(max(dist, 1.0) / 16.0) / 9.0, 0.0, 1.0);
    if wk_is_viewmodel(d) { return vec4f(vec3f(1.0, 0.65, 0.3) * (1.0 - 0.5 * t), 1.0); }
    return vec4f(vec3f(1.0 - t), 1.0);
}

@fragment
fn pv_normals(i: WkVsOut) -> @location(0) vec4f {
    let d = wk_depth(i.uv);
    if wk_is_sky(d) { return vec4f(0.04, 0.05, 0.08, 1.0); }
    var n = wk_view_normal(i.uv);
    if wk_has_camera() { n = wk_world_normal(i.uv); }
    return vec4f(n * 0.5 + 0.5, 1.0);
}

fn hue(h: f32) -> vec3f {
    let k = fract(vec3f(h) + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return clamp(abs(k - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
}

@fragment
fn pv_motion(i: WkVsOut) -> @location(0) vec4f {
    if !wk_has_motion() { return vec4f(0.15, 0.15, 0.15, 1.0); }
    // pixels moved: direction as hue, length as brightness (full at 16 px)
    let m = wk_motion_uv(i.uv) * wk.depth_size;
    let len = length(m);
    if len < 0.01 { return vec4f(0.0, 0.0, 0.0, 1.0); }
    let a = atan2(m.y, m.x) / 6.2831853 + 0.5;
    return vec4f(hue(a) * clamp(len / 16.0, 0.0, 1.0), 1.0);
}
"#;

/// the buffer views the overlay shows
struct Previews {
    pipelines: Vec<wgpu::RenderPipeline>,
    params: wgpu::Buffer,
    textures: Vec<(wgpu::Texture, wgpu::TextureView)>,
    /// bumped when the textures are recreated, so the overlay re-registers them
    generation: u32,
}

pub struct Effects {
    previews: Option<Previews>,
    /// the scene depth before and after its water, and the solid depth made from them and the
    /// final depth (see snapshot_depth, solid_depth)
    snapshots: [Option<(wgpu::Texture, wgpu::TextureView)>; 3],
    /// both snapshots were taken this frame
    snapped: [bool; 2],
    depth_copy: Option<(wgpu::RenderPipeline, wgpu::BindGroupLayout)>,
    compose: Option<(wgpu::RenderPipeline, wgpu::BindGroupLayout)>,
    /// the solid depth is ready for this frame's effects
    solid_now: bool,
    stamps: Vec<(PathBuf, Option<SystemTime>)>,
    scanned: Option<Instant>,
    loaded: Vec<Loaded>,
    layout0: wgpu::BindGroupLayout,
    globals: wgpu::Buffer,
    linear: wgpu::Sampler,
    point: wgpu::Sampler,
    dummy: wgpu::TextureView,
    dummy_stencil: wgpu::TextureView,
    /// screen-space sun shadows, made when an effect first asks (see `wk_sun_shadow`)
    sss: Option<crate::sss::ScreenShadows>,
    /// the mask holds this frame's shadows
    sun_shadow_ok: bool,
    /// copy of the back buffer each effect reads as wk_color
    copy: Option<(wgpu::Texture, wgpu::TextureView)>,
    start: Instant,
    last: Instant,
    frame: u32,
    prev_view_proj: Option<[f32; 16]>,
    /// the viewmodel's blended light direction (view space), smoothed over frames
    vm_light_dir: Option<[f32; 3]>,
    msaa: Option<MsaaHelpers>,
}

impl Effects {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Effects {
        use wgpu::{BindGroupLayoutEntry as E, BindingType as B, ShaderStages as S};
        let tex = |binding, sample_type| E { binding, visibility: S::FRAGMENT, ty: B::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None };
        let uniform = |binding| E { binding, visibility: S::FRAGMENT | S::VERTEX, ty: B::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let layout0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("effects globals"),
            entries: &[
                uniform(0),
                tex(1, wgpu::TextureSampleType::Float { filterable: true }),
                tex(2, wgpu::TextureSampleType::Depth),
                tex(3, wgpu::TextureSampleType::Float { filterable: true }),
                E { binding: 4, visibility: S::FRAGMENT, ty: B::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                E { binding: 5, visibility: S::FRAGMENT, ty: B::Sampler(wgpu::SamplerBindingType::NonFiltering), count: None },
                uniform(6),
                tex(7, wgpu::TextureSampleType::Depth),
                tex(8, wgpu::TextureSampleType::Uint),
                tex(9, wgpu::TextureSampleType::Float { filterable: false }),
            ],
        });
        let globals = device.create_buffer(&wgpu::BufferDescriptor { label: Some("effects globals"), size: std::mem::size_of::<Globals>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let clamp = wgpu::AddressMode::ClampToEdge;
        let linear = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("effects linear"), address_mode_u: clamp, address_mode_v: clamp, mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        let point = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("effects point"), address_mode_u: clamp, address_mode_v: clamp, ..Default::default() });
        let dummy_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("effects dummy"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(dummy_tex.as_image_copy(), &[0u8; 8], wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8), rows_per_image: None }, wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 });
        let dummy_stencil = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("effects dummy stencil"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let now = Instant::now();
        Effects {
            dummy_stencil: dummy_stencil.create_view(&Default::default()),
            sss: None,
            sun_shadow_ok: false,
            previews: None,
            snapshots: [None, None, None],
            snapped: [false; 2],
            depth_copy: None,
            compose: None,
            solid_now: false,
            stamps: Vec::new(),
            scanned: None,
            loaded: Vec::new(),
            layout0,
            globals,
            linear,
            point,
            dummy: dummy_tex.create_view(&Default::default()),
            copy: None,
            start: now,
            last: now,
            frame: 0,
            vm_light_dir: None,
            prev_view_proj: None,
            msaa: None,
        }
    }

    /// rescan the folder about once a second; reload everything when a file changed
    pub fn poll(&mut self, device: &wgpu::Device, screen_format: wgpu::TextureFormat) {
        if self.scanned.is_some_and(|t| t.elapsed().as_secs_f32() < 1.0) {
            return;
        }
        self.scanned = Some(Instant::now());
        let mut files: Vec<PathBuf> = std::fs::read_dir(DIR).map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "wgsl")).collect()).unwrap_or_default();
        files.sort();
        let stamps: Vec<(PathBuf, Option<SystemTime>)> = files.iter().map(|p| (p.clone(), std::fs::metadata(p).and_then(|m| m.modified()).ok())).collect();
        if stamps == self.stamps {
            return;
        }
        // only files that changed (or are new) get a notice; the first scan stays quiet on success
        let first = self.stamps.is_empty() && self.loaded.is_empty();
        let changed: Vec<PathBuf> = stamps.iter().filter(|s| !self.stamps.contains(s)).map(|s| s.0.clone()).collect();
        self.stamps = stamps;
        let mut defs = Vec::new();
        for path in &files {
            let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            match std::fs::read_to_string(path).map_err(|e| e.to_string()).and_then(|src| parse(&stem, &src)) {
                Ok(d) => defs.push((d, changed.contains(path))),
                Err(e) => {
                    log::line(&format!("effects: {}: {e}", path.display()));
                    crate::overlay::notify(format!("{stem}.wgsl: {e}"), true);
                }
            }
        }
        defs.sort_by(|a, b| a.0.order.cmp(&b.0.order).then(a.0.name.cmp(&b.0.name)));
        let mut loaded = Vec::new();
        for (def, changed) in defs {
            // a broken edit keeps the last working version
            match self.compile(device, &def, screen_format) {
                Ok(l) => {
                    log::line(&format!("effects: loaded '{}' ({} passes)", def.name, def.passes.len()));
                    if changed && !first {
                        crate::overlay::notify(format!("{}: reloaded", def.name), false);
                    }
                    loaded.push(l);
                }
                Err(e) => {
                    log::line(&format!("effects: '{}' failed: {e}", def.name));
                    // the first error line is the useful one; the log has the rest
                    let short = e.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim().to_string();
                    let at = e.lines().find(|l| l.contains("line ")).map(|l| l.trim().trim_start_matches(['┌', '─', ' ']).to_string()).unwrap_or_default();
                    crate::overlay::notify(format!("{} failed: {short} {at}", def.name), true);
                    if let Some(i) = self.loaded.iter().position(|l| l.def.key == def.key) {
                        loaded.push(self.loaded.remove(i));
                    }
                }
            }
        }
        self.loaded = loaded;
        let defs: Vec<EffectDef> = self.loaded.iter().map(|l| l.def.clone()).collect();
        *INFO.lock() = defs.iter().map(|d| (d.key.clone(), d.enabled, d.params.clone())).collect();
        write_manifest(&defs);
    }

    fn compile(&self, device: &wgpu::Device, def: &EffectDef, screen_format: wgpu::TextureFormat) -> Result<Loaded, String> {
        let src = source(def);
        // naga first: it gives readable errors and never touches the device
        let module = naga::front::wgsl::parse_str(&src).map_err(|e| {
            let offset = prelude_lines(def);
            let msg = e.emit_to_string(&src);
            // point at the user's file: the prelude shifts every line
            msg.lines().map(|l| shift_line_numbers(l, offset)).collect::<Vec<_>>().join("\n")
        })?;
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
            .validate(&module)
            .map_err(|e| e.emit_to_string(&src))?;
        for p in &def.passes {
            if !module.entry_points.iter().any(|e| e.name == p.entry && e.stage == naga::ShaderStage::Fragment) {
                return Err(format!("no @fragment fn {}", p.entry));
            }
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(&def.name), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let layout1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("effect textures"),
            entries: &def
                .textures
                .iter()
                .enumerate()
                .map(|(binding, t)| wgpu::BindGroupLayoutEntry {
                    binding: binding as u32,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    // 32-bit float formats can't be filtered: textureLoad / wk_point only
                    ty: wgpu::BindingType::Texture { sample_type: t.format.sample_type(None, None).unwrap_or(wgpu::TextureSampleType::Float { filterable: false }), view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(&def.name), bind_group_layouts: &[Some(&self.layout0), Some(&layout1)], immediate_size: 0 });
        let pipelines = def
            .passes
            .iter()
            .map(|p| {
                let format = match p.target {
                    PassTarget::Screen => screen_format,
                    PassTarget::Texture(i) => def.textures[i].format,
                };
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(&p.entry),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState { module: &shader, entry_point: Some("wk_vs"), compilation_options: Default::default(), buffers: &[] },
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some(&p.entry), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })] }),
                    multiview_mask: None,
                    cache: None,
                })
            })
            .collect();
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(e.to_string());
        }
        let size = (def.params.len().max(1) * 4).next_multiple_of(16) as u64;
        let params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("effect params"), size, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        Ok(Loaded { def: def.clone(), layout1, pipelines, params, textures: vec![None; def.textures.len()] })
    }

    /// globals, scene copy, depth (the pre-water snapshot when there is one), motion, samplers,
    /// params, final depth
    fn group0(&self, device: &wgpu::Device, color: &wgpu::TextureView, depth: &wgpu::TextureView, motion: Option<&wgpu::TextureView>, stencil: Option<&wgpu::TextureView>, params: &wgpu::Buffer) -> wgpu::BindGroup {
        let solid = self.snapshots[2].as_ref().filter(|_| self.solid_now).map_or(depth, |o| &o.1);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.globals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(color) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(solid) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(motion.unwrap_or(&self.dummy)) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.linear) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.point) },
                wgpu::BindGroupEntry { binding: 6, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(stencil.unwrap_or(&self.dummy_stencil)) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(self.sss.as_ref().and_then(|x| x.out.as_ref()).filter(|_| self.sun_shadow_ok).map_or(&self.dummy, |o| &o.1)) },
            ],
        })
    }

    /// copy the scene depth (sample 0 under msaa) before water draws over it
    pub fn snapshot_depth(&mut self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder, depth: &wgpu::TextureView, samples: u32, size: (u32, u32), after: bool) {
        if self.depth_copy.is_none() {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("depth snapshot"),
                entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None }],
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("depth snapshot"), source: wgpu::ShaderSource::Wgsl(DEPTH_COPY_WGSL.into()) });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("depth snapshot"),
                layout: Some(&pl),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: Some(true), depth_compare: Some(wgpu::CompareFunction::Always), stencil: Default::default(), bias: Default::default() }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("copy"), compilation_options: Default::default(), targets: &[] }),
                multiview_mask: None,
                cache: None,
            });
            self.depth_copy = Some((pipeline, layout));
        }
        let slot = after as usize;
        let target = Self::depth_target(device, &mut self.snapshots[slot], size, "effects depth snapshot");
        // multisampled sources go through the msaa resolve, single-sampled through the copy
        let (pipeline, group) = if samples > 1 {
            let format = self.copy.as_ref().map_or(wgpu::TextureFormat::Rgba8Unorm, |c| c.0.format());
            if self.msaa.as_ref().is_none_or(|m| m.samples != samples) {
                self.msaa = Some(MsaaHelpers::new(device, samples, format));
            }
            let m = self.msaa.as_ref().unwrap();
            (m.resolve.clone(), device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &m.resolve_layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth) }] }))
        } else {
            let (p, l) = self.depth_copy.as_ref().unwrap();
            (p.clone(), device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: l, entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth) }] }))
        };
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("depth snapshot"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &target, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(&pipeline);
        rp.set_bind_group(0, &group, &[]);
        rp.draw(0..3, 0..1);
        drop(rp);
        self.snapped[slot] = true;
    }

    fn depth_target(device: &wgpu::Device, slot: &mut Option<(wgpu::Texture, wgpu::TextureView)>, size: (u32, u32), label: &str) -> wgpu::TextureView {
        if slot.as_ref().is_none_or(|(t, _)| (t.width(), t.height()) != size) {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let v = t.create_view(&Default::default());
            *slot = Some((t, v));
        }
        slot.as_ref().unwrap().1.clone()
    }

    /// where the final depth is still the water's, the depth from before it; elsewhere (land,
    /// a viewmodel over the water) the final depth
    fn solid_depth(&mut self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder, final_depth: &wgpu::TextureView) {
        let ready = self.snapped == [true, true];
        self.snapped = [false; 2];
        self.solid_now = false;
        if !ready {
            return;
        }
        let Some(size) = self.snapshots[0].as_ref().map(|t| (t.0.width(), t.0.height())) else { return };
        if self.compose.is_none() {
            let entry = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None };
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("solid depth"), entries: &[entry(0), entry(1), entry(2)] });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("solid depth"), source: wgpu::ShaderSource::Wgsl(SOLID_DEPTH_WGSL.into()) });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("solid depth"),
                layout: Some(&pl),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: Some(true), depth_compare: Some(wgpu::CompareFunction::Always), stencil: Default::default(), bias: Default::default() }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("solid"), compilation_options: Default::default(), targets: &[] }),
                multiview_mask: None,
                cache: None,
            });
            self.compose = Some((pipeline, layout));
        }
        let target = Self::depth_target(device, &mut self.snapshots[2], size, "effects solid depth");
        let (pipeline, layout) = self.compose.as_ref().unwrap();
        let view = |i: usize| wgpu::BindingResource::TextureView(&self.snapshots[i].as_ref().unwrap().1);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: view(0) },
                wgpu::BindGroupEntry { binding: 1, resource: view(1) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(final_depth) },
            ],
        });
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("solid depth"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &target, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(pipeline);
        rp.set_bind_group(0, &group, &[]);
        rp.draw(0..3, 0..1);
        drop(rp);
        self.solid_now = true;
    }

    /// run the enabled effects onto the back buffer
    pub fn run(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, inp: &Inputs) {
        let settings = SETTINGS.lock().clone();
        let on = |l: &Loaded| settings.as_ref().map_or(l.def.enabled, |s| s.get(&l.def.key).is_some_and(|v| v.0));
        let previews = previews_on();
        DYNAMIC_MASK.store(self.loaded.iter().any(|l| on(l) && (l.def.body.contains("wk_is_dynamic") || l.def.body.contains("wk_sun_shadow"))), std::sync::atomic::Ordering::Relaxed);
        if !self.loaded.iter().any(on) && !previews {
            self.snapped = [false; 2];
            return;
        }
        self.solid_depth(device, enc, inp.depth);
        let size = (inp.color.width(), inp.color.height());
        let format = inp.color.format();
        if self.copy.as_ref().is_none_or(|(t, _)| (t.width(), t.height()) != size || t.format() != format) {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("effects scene copy"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let v = t.create_view(&Default::default());
            self.copy = Some((t, v));
        }
        let sun_user = self.loaded.iter().filter(|l| on(l)).find(|l| l.def.body.contains("wk_sun_shadow")).map(|l| {
            let values = settings.as_ref().and_then(|s| s.get(&l.def.key)).map_or_else(|| l.def.params.iter().map(|p| p.default).collect(), |v| v.1.clone());
            l.def.params.iter().zip(values).map(|(p, v)| (p.name.clone(), v)).collect::<HashMap<_, _>>()
        });
        self.sun_shadows(device, queue, enc, inp, size, sun_user);
        self.write_globals(queue, inp, size);
        let (copy_tex, copy_view) = self.copy.clone().unwrap();
        let mut loaded = std::mem::take(&mut self.loaded);
        for l in &mut loaded {
            let Some((true, values)) = settings.as_ref().map_or(Some((l.def.enabled, l.def.params.iter().map(|p| p.default).collect())), |s| s.get(&l.def.key).cloned()) else { continue };
            let mut bytes = vec![0f32; (l.params.size() / 4) as usize];
            let n = values.len().min(bytes.len());
            bytes[..n].copy_from_slice(&values[..n]);
            queue.write_buffer(&l.params, 0, bytemuck::cast_slice(&bytes));
            for (i, t) in l.def.textures.iter().enumerate() {
                let base = if t.depth { inp.depth_size.unwrap_or(size) } else { size };
                let want = (((base.0 as f32 * t.scale) as u32).max(1), ((base.1 as f32 * t.scale) as u32).max(1));
                if l.textures[i].as_ref().is_none_or(|(_, s)| *s != want) {
                    let tex = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some(&t.name),
                        size: wgpu::Extent3d { width: want.0, height: want.1, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: t.format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });
                    l.textures[i] = Some((tex.create_view(&Default::default()), want));
                }
            }
            // each effect sees the result of the ones before it
            enc.copy_texture_to_texture(inp.color.as_image_copy(), copy_tex.as_image_copy(), wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 });
            let g0 = self.group0(device, &copy_view, inp.depth, inp.motion, inp.stencil, &l.params);
            for (pass, pipeline) in l.def.passes.iter().zip(&l.pipelines) {
                // a texture can't be read while it's the target
                let views: Vec<&wgpu::TextureView> = l.textures.iter().enumerate().map(|(i, t)| if pass.target == PassTarget::Texture(i) { &self.dummy } else { &t.as_ref().unwrap().0 }).collect();
                let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &l.layout1,
                    entries: &views.iter().enumerate().map(|(i, v)| wgpu::BindGroupEntry { binding: i as u32, resource: wgpu::BindingResource::TextureView(v) }).collect::<Vec<_>>(),
                });
                let (target, tsize) = match pass.target {
                    PassTarget::Screen => (inp.color_view, size),
                    PassTarget::Texture(i) => {
                        let t = l.textures[i].as_ref().unwrap();
                        (&t.0, t.1)
                    }
                };
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(&pass.entry),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: target, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rp.set_pipeline(pipeline);
                rp.set_bind_group(0, &g0, &[]);
                rp.set_bind_group(1, &g1, &[]);
                rp.set_viewport(0.0, 0.0, tsize.0 as f32, tsize.1 as f32, 0.0, 1.0);
                rp.draw(0..3, 0..1);
            }
        }
        self.loaded = loaded;
        if previews {
            self.draw_previews(device, enc, inp, size, &copy_view);
        }
        self.solid_now = false;
    }

    /// depth, normals and motion into small textures for the overlay
    fn draw_previews(&mut self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder, inp: &Inputs, size: (u32, u32), color: &wgpu::TextureView) {
        const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
        if self.previews.is_none() {
            let def = EffectDef { body: PREVIEW_BODY.to_string(), ..EffectDef::default() };
            let src = source(&def);
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("buffer previews"), source: wgpu::ShaderSource::Wgsl(src.into()) });
            let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[] });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("buffer previews"), bind_group_layouts: &[Some(&self.layout0), Some(&empty)], immediate_size: 0 });
            let pipelines = PREVIEWS
                .iter()
                .map(|name| {
                    let entry = format!("pv_{name}");
                    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some(&entry),
                        layout: Some(&layout),
                        vertex: wgpu::VertexState { module: &shader, entry_point: Some("wk_vs"), compilation_options: Default::default(), buffers: &[] },
                        primitive: Default::default(),
                        depth_stencil: None,
                        multisample: Default::default(),
                        fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some(&entry), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })] }),
                        multiview_mask: None,
                        cache: None,
                    })
                })
                .collect();
            let params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("preview params"), size: 16, usage: wgpu::BufferUsages::UNIFORM, mapped_at_creation: false });
            self.previews = Some(Previews { pipelines, params, textures: Vec::new(), generation: 0 });
        }
        let g0 = self.group0(device, color, inp.depth, inp.motion, inp.stencil, &self.previews.as_ref().unwrap().params);
        let pv = self.previews.as_mut().unwrap();
        let want = (PREVIEW_WIDTH, (PREVIEW_WIDTH * size.1 / size.0.max(1)).max(1));
        if pv.textures.first().is_none_or(|t| (t.0.width(), t.0.height()) != want) {
            pv.textures = PREVIEWS
                .iter()
                .map(|name| {
                    let t = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some(name),
                        size: wgpu::Extent3d { width: want.0, height: want.1, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: FORMAT,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });
                    let v = t.create_view(&Default::default());
                    (t, v)
                })
                .collect();
            pv.generation += 1;
        }
        let empty_layout = pv.pipelines[0].get_bind_group_layout(1);
        let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &empty_layout, entries: &[] });
        for (pipeline, (_, view)) in pv.pipelines.iter().zip(&pv.textures) {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("buffer preview"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(pipeline);
            rp.set_bind_group(0, &g0, &[]);
            rp.set_bind_group(1, &g1, &[]);
            rp.draw(0..3, 0..1);
        }
    }

    /// (name, on, passes) of every loaded effect, for the overlay
    pub fn status(&self) -> Vec<(String, bool, usize)> {
        let settings = SETTINGS.lock();
        self.loaded.iter().map(|l| (l.def.name.clone(), settings.as_ref().map_or(l.def.enabled, |s| s.get(&l.def.key).is_some_and(|v| v.0)), l.def.passes.len())).collect()
    }

    /// the preview textures (in PREVIEWS order) and their generation
    pub fn previews(&self) -> Option<(u32, Vec<&wgpu::TextureView>)> {
        self.previews.as_ref().filter(|p| !p.textures.is_empty()).map(|p| (p.generation, p.textures.iter().map(|t| &t.1).collect()))
    }

    /// under msaa: resolve the multisampled depth (sample 0) for the effects, run them, then copy
    /// the result into every sample of the msaa color so later resolves (the hud) keep it
    pub fn run_msaa(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, inp: &Inputs, msaa_color: &wgpu::TextureView, msaa_depth: &wgpu::TextureView, samples: u32) {
        let settings_on = SETTINGS.lock().as_ref().is_some_and(|s| s.values().any(|v| v.0));
        if !settings_on && !self.loaded.iter().any(|l| l.def.enabled) && !previews_on() {
            self.snapped = [false; 2];
            return;
        }
        let size = (inp.color.width(), inp.color.height());
        let format = inp.color.format();
        if self.msaa.as_ref().is_none_or(|m| m.samples != samples || m.format != format) {
            self.msaa = Some(MsaaHelpers::new(device, samples, format));
        }
        let m = self.msaa.as_mut().unwrap();
        if m.depth.as_ref().is_none_or(|(t, _)| (t.width(), t.height()) != size) {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("effects resolved depth"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let v = t.create_view(&Default::default());
            m.depth = Some((t, v));
        }
        let depth_view = m.depth.as_ref().unwrap().1.clone();
        {
            let g = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &m.resolve_layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(msaa_depth) }] });
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("effects depth resolve"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &depth_view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&m.resolve);
            rp.set_bind_group(0, &g, &[]);
            rp.draw(0..3, 0..1);
        }
        let write_back = (m.write_back.clone(), m.write_layout.clone());
        let resolved = Inputs { color: inp.color, color_view: inp.color_view, depth: &depth_view, motion: None, jitter: None, depth_size: None, camera: inp.camera, viewmodel: inp.viewmodel, viewmodel_lights: inp.viewmodel_lights, sun: inp.sun, stencil: None };
        self.run(device, queue, enc, &resolved);
        let g = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &write_back.1, entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(inp.color_view) }] });
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("effects msaa write back"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: msaa_color, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(&write_back.0);
        rp.set_bind_group(0, &g, &[]);
        rp.draw(0..3, 0..1);
    }

    /// bend's screen-space shadows towards the sun, for effects that read `wk_sun_shadow`. tuned
    /// by that effect's params when it has them: sun_thickness, sun_contrast, all_casters,
    /// ignore_edges
    fn sun_shadows(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, inp: &Inputs, size: (u32, u32), params: Option<HashMap<String, f32>>) {
        self.sun_shadow_ok = false;
        let (Some(params), Some((view, proj)), Some(sun)) = (params, inp.camera, inp.sun) else { return };
        let get = |k: &str, d: f32| params.get(k).copied().unwrap_or(d);
        let depth = inp.depth_size.unwrap_or(size);
        let vp = crate::api::mat_mul(&view, &proj);
        // towards the sun, (dir, 0) through view x projection (row vectors)
        let n = (sun.dir[0] * sun.dir[0] + sun.dir[1] * sun.dir[1] + sun.dir[2] * sun.dir[2]).sqrt().max(1e-6);
        let d = [-sun.dir[0] / n, -sun.dir[1] / n, -sun.dir[2] / n];
        let light: [f32; 4] = std::array::from_fn(|c| d[0] * vp[c] + d[1] * vp[4 + c] + d[2] * vp[8 + c]);
        let (w, h) = (depth.0 as i32, depth.1 as i32);
        let list = crate::sss::dispatch_list(light, [w, h], [0, 0], [w, h], 1.0);
        let mask = inp.stencil.is_some() && get("all_casters", 0.0) < 0.5;
        let base = crate::sss::Params {
            light: [0.0; 4],
            wave_offset: [0, 0],
            size: [w, h],
            surface_thickness: get("sun_thickness", 0.005).max(1e-5),
            bilinear_threshold: 0.02,
            contrast: get("sun_contrast", 4.0).max(1.0),
            flags: mask as u32 | ((get("ignore_edges", 0.0) > 0.5) as u32) << 1,
            viewmodel_depth: 0.1,
            _pad: [0.0; 3],
        };
        let sss = self.sss.get_or_insert_with(|| crate::sss::ScreenShadows::new(device));
        sss.run(device, queue, enc, inp.depth, inp.stencil.unwrap_or(&self.dummy_stencil), depth, &list, base);
        self.sun_shadow_ok = true;
    }

    fn write_globals(&mut self, queue: &wgpu::Queue, inp: &Inputs, size: (u32, u32)) {
        let now = Instant::now();
        let depth = inp.depth_size.unwrap_or(size);
        let mut g = Globals {
            resolution: [size.0 as f32, size.1 as f32, 1.0 / size.0 as f32, 1.0 / size.1 as f32],
            time: (now - self.start).as_secs_f32(),
            delta: (now - self.last).as_secs_f32(),
            frame: self.frame,
            flags: inp.motion.is_some() as u32 * 2 | (inp.stencil.is_some() && DYNAMIC_MASK.load(std::sync::atomic::Ordering::Relaxed)) as u32 * 4 | self.sun_shadow_ok as u32 * 8,
            // the scene's vertex shaders add this to clip space (see vs_extra)
            jitter: inp.jitter.map_or([0.0, 0.0], |[x, y]| [2.0 * x / depth.0 as f32, -2.0 * y / depth.1 as f32]),
            depth_size: [depth.0 as f32, depth.1 as f32],
            ..bytemuck::Zeroable::zeroed()
        };
        self.last = now;
        self.frame = self.frame.wrapping_add(1);
        if let Some((view, proj)) = inp.camera {
            let vp = crate::api::mat_mul(&view, &proj);
            let inv_view = crate::api::mat_inverse(&view).unwrap_or(crate::api::IDENTITY);
            g.view = view;
            g.proj = proj;
            g.view_proj = vp;
            g.inv_view = inv_view;
            g.inv_proj = crate::api::mat_inverse(&proj).unwrap_or(crate::api::IDENTITY);
            g.inv_view_proj = crate::api::mat_inverse(&vp).unwrap_or(crate::api::IDENTITY);
            g.prev_view_proj = self.prev_view_proj.unwrap_or(vp);
            g.camera_pos = [inv_view[12], inv_view[13], inv_view[14], 1.0];
            g.flags |= 1;
            self.prev_view_proj = Some(vp);
        }
        if let Some((view, proj, range)) = inp.viewmodel {
            g.vm_proj = proj;
            g.vm_inv_proj = crate::api::mat_inverse(&proj).unwrap_or(crate::api::IDENTITY);
            g.vm = [range[1], range[0], 1.0, 0.0];
            match blend_lights(&view, inp.viewmodel_lights) {
                Some((dir, color, dominance)) => {
                    // ease towards the new direction so lights fading in and out don't snap it
                    let k = 1.0 - (-g.delta.min(0.1) * 12.0).exp();
                    let d = match self.vm_light_dir {
                        Some(p) => normalize3(std::array::from_fn(|i| p[i] + (dir[i] - p[i]) * k)),
                        None => dir,
                    };
                    self.vm_light_dir = Some(d);
                    g.vm_light = [d[0], d[1], d[2], 1.0];
                    g.vm_light_color = [color[0], color[1], color[2], dominance];
                }
                None => self.vm_light_dir = None,
            }
        }
        if let Some(l) = inp.sun {
            let n = (l.dir[0] * l.dir[0] + l.dir[1] * l.dir[1] + l.dir[2] * l.dir[2]).sqrt().max(1e-6);
            g.sun = [-l.dir[0] / n, -l.dir[1] / n, -l.dir[2] / n, 1.0];
            g.sun_color = [l.color[0], l.color[1], l.color[2], 0.0];
        }
        queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(&g));
    }
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / n, v[1] / n, v[2] / n]
}

/// the viewmodel's lights as one direction towards them (its view space), weighted by brightness
/// and distance falloff; plus their summed color and how much they agree (1 = one direction, 0 =
/// lights from opposite sides cancel out)
fn blend_lights(view: &[f32; 16], lights: &[crate::api::Light]) -> Option<([f32; 3], [f32; 3], f32)> {
    // the camera, in world space: the viewmodel sits right in front of it
    let inv = crate::api::mat_inverse(view)?;
    let eye = [inv[12], inv[13], inv[14]];
    let (mut sum, mut total, mut color) = ([0.0f32; 3], 0.0f32, [0.0f32; 3]);
    for l in lights {
        let luma = l.color[0] * 0.3 + l.color[1] * 0.59 + l.color[2] * 0.11;
        let (d, w) = if l.kind == 2 {
            (normalize3([-l.dir[0], -l.dir[1], -l.dir[2]]), luma)
        } else {
            let v = [l.pos[0] - eye[0], l.pos[1] - eye[1], l.pos[2] - eye[2]];
            let dist = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let falloff = l.atten[0] + l.atten[1] * dist + l.atten[2] * dist * dist;
            (normalize3(v), luma / if falloff > 1e-6 { falloff } else { 1.0 })
        };
        if !(w > 0.0) {
            continue;
        }
        for i in 0..3 {
            sum[i] += d[i] * w;
            color[i] += l.color[i];
        }
        total += w;
    }
    let len = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
    if total <= 0.0 || len <= 1e-6 * total {
        return None;
    }
    // to view space (row vectors: v * view)
    let d = [sum[0] / len, sum[1] / len, sum[2] / len];
    let vd: [f32; 3] = std::array::from_fn(|c| d[0] * view[c] + d[1] * view[4 + c] + d[2] * view[8 + c]);
    Some((normalize3(vd), color, len / total))
}

/// fullscreen passes for msaa: depth resolve and color write back
struct MsaaHelpers {
    samples: u32,
    format: wgpu::TextureFormat,
    resolve: wgpu::RenderPipeline,
    resolve_layout: wgpu::BindGroupLayout,
    write_back: wgpu::RenderPipeline,
    write_layout: wgpu::BindGroupLayout,
    depth: Option<(wgpu::Texture, wgpu::TextureView)>,
}

const DEPTH_COPY_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_depth_2d;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}

@fragment
fn copy(@builtin(position) p: vec4f) -> @builtin(frag_depth) f32 {
    return textureLoad(src, vec2i(p.xy), 0);
}
"#;

const SOLID_DEPTH_WGSL: &str = r#"
@group(0) @binding(0) var before: texture_depth_2d;
@group(0) @binding(1) var after: texture_depth_2d;
@group(0) @binding(2) var last: texture_depth_2d;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}

@fragment
fn solid(@builtin(position) p: vec4f) -> @builtin(frag_depth) f32 {
    let q = vec2i(p.xy);
    let a = textureLoad(before, q, 0);
    let b = textureLoad(after, q, 0);
    let c = textureLoad(last, q, 0);
    // still the water's depth: what's under it
    return select(c, a, c == b && b < a);
}
"#;

const MSAA_WGSL: &str = r#"
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}
@group(0) @binding(0) var msaa_depth: texture_depth_multisampled_2d;
@fragment
fn resolve(@builtin(position) p: vec4f) -> @builtin(frag_depth) f32 {
    return textureLoad(msaa_depth, vec2i(p.xy), 0);
}
"#;

const WRITE_BACK_WGSL: &str = r#"
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}
@group(0) @binding(0) var color: texture_2d<f32>;
@fragment
fn copy(@builtin(position) p: vec4f) -> @location(0) vec4f {
    return textureLoad(color, vec2i(p.xy), 0);
}
"#;

impl MsaaHelpers {
    fn new(device: &wgpu::Device, samples: u32, format: wgpu::TextureFormat) -> MsaaHelpers {
        let layout = |label, sample_type, multisampled| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled }, count: None }],
            })
        };
        let resolve_layout = layout("effects depth resolve", wgpu::TextureSampleType::Depth, true);
        let write_layout = layout("effects msaa write back", wgpu::TextureSampleType::Float { filterable: false }, false);
        let m1 = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("effects depth resolve"), source: wgpu::ShaderSource::Wgsl(MSAA_WGSL.into()) });
        let m2 = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("effects msaa write back"), source: wgpu::ShaderSource::Wgsl(WRITE_BACK_WGSL.into()) });
        let pl = |l: &wgpu::BindGroupLayout| device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(l)], immediate_size: 0 });
        let resolve = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("effects depth resolve"),
            layout: Some(&pl(&resolve_layout)),
            vertex: wgpu::VertexState { module: &m1, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: Some(true), depth_compare: Some(wgpu::CompareFunction::Always), stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState { module: &m1, entry_point: Some("resolve"), compilation_options: Default::default(), targets: &[] }),
            multiview_mask: None,
            cache: None,
        });
        let write_back = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("effects msaa write back"),
            layout: Some(&pl(&write_layout)),
            vertex: wgpu::VertexState { module: &m2, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
            fragment: Some(wgpu::FragmentState { module: &m2, entry_point: Some("copy"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })] }),
            multiview_mask: None,
            cache: None,
        });
        MsaaHelpers { samples, format, resolve, resolve_layout, write_back, write_layout, depth: None }
    }
}

/// "wgsl:123:4" style positions minus the prelude
fn shift_line_numbers(line: &str, offset: usize) -> String {
    // naga reports `┌─ wgsl:LINE:COL`
    if let Some(i) = line.find("wgsl:") {
        let rest = &line[i + 5..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = digits.parse::<usize>() {
            let shifted = if n > offset { format!("line {}", n - offset) } else { format!("prelude line {n}") };
            return format!("{}{}{}", &line[..i], shifted, &rest[digits.len()..]);
        }
    }
    line.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = "//! effect: Test Effect\n//! texture: half rgba16float 0.5\n//! pass: first -> half\n//! pass: second -> screen\n//! param: strength 0.5 0 2\n@fragment fn first(i: WkVsOut) -> @location(0) vec4f { return textureSample(wk_color, wk_linear, i.uv) * params.strength; }\n@fragment fn second(i: WkVsOut) -> @location(0) vec4f { return textureSample(half, wk_linear, i.uv) + vec4f(wk_world_normal(i.uv), wk_depth(i.uv)); }\n";

    #[test]
    fn header_parses() {
        let d = parse("file", EXAMPLE).unwrap();
        assert_eq!(d.name, "Test Effect");
        assert_eq!(d.key, "test_effect");
        assert_eq!(d.textures, vec![TextureDef { name: "half".into(), format: wgpu::TextureFormat::Rgba16Float, scale: 0.5, depth: false }]);
        let t = parse("f", "//! texture: z r32float depth\n//! texture: h r8unorm 0.5 depth\n//! pass: a -> screen\n").unwrap().textures;
        assert_eq!((t[0].scale, t[0].depth, t[1].scale, t[1].depth), (1.0, true, 0.5, true));
        assert_eq!(d.passes, vec![Pass { entry: "first".into(), target: PassTarget::Texture(0) }, Pass { entry: "second".into(), target: PassTarget::Screen }]);
        assert_eq!(d.params, vec![Param { name: "strength".into(), default: 0.5, min: 0.0, max: 2.0, int: false }]);
        let p = parse("f", "//! pass: a -> screen\n//! param: mode 1 0 4 int\n").unwrap().params;
        assert_eq!(p, vec![Param { name: "mode".into(), default: 1.0, min: 0.0, max: 4.0, int: true }]);
    }

    #[test]
    fn header_errors_name_the_line() {
        assert!(parse("f", "//! pass: a -> nowhere\n").unwrap_err().contains("line 1"));
        assert!(parse("f", "//! texture: t bogus\n//! pass: a -> screen\n").unwrap_err().contains("unknown format"));
        assert!(parse("f", "//! param: x 1\n//! texture: x r8unorm\n//! pass: a -> x\n").unwrap_err().contains("twice"));
        assert!(parse("f", "nothing here\n").is_err());
    }

    #[test]
    fn depth_helpers_validate() {
        for src in [DEPTH_COPY_WGSL, SOLID_DEPTH_WGSL] {
            let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty()).validate(&module).unwrap();
        }
    }

    #[test]
    fn buffer_previews_validate() {
        let src = source(&EffectDef { body: PREVIEW_BODY.to_string(), ..EffectDef::default() });
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty()).validate(&module).unwrap();
        for name in PREVIEWS {
            assert!(module.entry_points.iter().any(|e| e.name == format!("pv_{name}")));
        }
    }

    #[test]
    fn prelude_and_example_validate() {
        let d = parse("file", EXAMPLE).unwrap();
        let src = source(&d);
        let module = naga::front::wgsl::parse_str(&src).map_err(|e| e.emit_to_string(&src)).unwrap();
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty()).validate(&module).unwrap();
    }

    #[test]
    fn shipped_effects_validate() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/effects");
        for e in std::fs::read_dir(dir).unwrap() {
            let path = e.unwrap().path();
            // material shaders live in a subfolder, with their own test
            if path.extension().is_none_or(|x| x != "wgsl") {
                continue;
            }
            let src = std::fs::read_to_string(&path).unwrap();
            let d = parse("x", &src).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let full = source(&d);
            let module = naga::front::wgsl::parse_str(&full).unwrap_or_else(|e| panic!("{}: {}", path.display(), e.emit_to_string(&full)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{}: {}", path.display(), e.emit_to_string(&full)));
        }
    }
}
