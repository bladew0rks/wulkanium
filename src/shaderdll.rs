//! a shader dll inside the shaderapi: materialsystem loads the dll the hardware config names
//! (GetHWSpecificShaderDLLName), asks it for "ShaderDLL004" and registers its shaders. this
//! serves SurfaceGGX (also under the names other dx11 renderers use) so materials written for it
//! load. vtable layouts were read from gmod's stdshader_dx9.dll (CShaderDLL, CBaseShader).

use std::ffi::{c_char, c_void};
use std::sync::OnceLock;

use crate::Interface;
use crate::log;

/// ShaderParamType_t
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(i32)]
pub enum ParamType {
    Texture = 0,
    Integer = 1,
    Color = 2,
    Vec2 = 3,
    Vec3 = 4,
    Vec4 = 5,
    Float = 7,
    Bool = 8,
    Matrix = 10,
    Material = 11,
}

/// SHADER_PARAM_NOT_EDITABLE
const NOT_EDITABLE: i32 = 1;

pub struct Param {
    pub name: &'static str,
    pub ty: ParamType,
    pub default: &'static str,
    pub help: &'static str,
    pub flags: i32,
}

const fn p(name: &'static str, ty: ParamType, default: &'static str, help: &'static str) -> Param {
    Param { name, ty, default, help, flags: 0 }
}

use ParamType::*;

/// the engine's standard params come first, in this order (materialsystem reads them by index)
pub const STANDARD: [Param; 13] = [
    Param { name: "$flags", ty: Integer, default: "0", help: "flags", flags: NOT_EDITABLE },
    Param { name: "$flags_defined", ty: Integer, default: "0", help: "flags_defined", flags: NOT_EDITABLE },
    Param { name: "$flags2", ty: Integer, default: "0", help: "flags2", flags: NOT_EDITABLE },
    Param { name: "$flags_defined2", ty: Integer, default: "0", help: "flags2_defined", flags: NOT_EDITABLE },
    p("$color", Color, "[1 1 1]", "color"),
    p("$alpha", Float, "1.0", "alpha"),
    p("$basetexture", Texture, "shadertest/BaseTexture", "base texture (albedo)"),
    p("$frame", Integer, "0", "animation frame"),
    p("$basetexturetransform", Matrix, "center .5 .5 scale 1 1 rotate 0 translate 0 0", "base texture texcoord transform"),
    Param { name: "$flashlighttexture", ty: Texture, default: "effects/flashlight001", help: "flashlight spotlight shape texture", flags: NOT_EDITABLE },
    Param { name: "$flashlighttextureframe", ty: Integer, default: "0", help: "animation frame for $flashlight", flags: NOT_EDITABLE },
    p("$color2", Color, "[1 1 1]", "color2"),
    p("$srgbtint", Color, "[1 1 1]", "tint applied on srgb parts"),
];

/// SurfaceGGX's own params (names, types and defaults as materials written for it expect)
pub const SURFACE_GGX: &[Param] = &[
    // common extras those materials use
    p("$alpha2", Float, "0", "second alpha"),
    p("$alphatestreference", Float, "0", "alpha test threshold"),
    p("$allowdiffusemodulation", Bool, "1", "$color2 / render color tints the material"),
    p("$notint", Bool, "0", "ignore $color2 / render color"),
    p("$receiveprojectedtextures", Bool, "0", "receive projected textures"),
    p("$receiveprojectedtextureshadows", Bool, "1", "receive projected texture shadows"),
    p("$projectedtexturenolambert", Bool, "0", "no n.l on projected textures"),
    // surface
    p("$bumpmap", Texture, "", "normal map; alpha is the roughness source without a mask"),
    p("$bumpframe", Integer, "0", "frame of $bumpmap"),
    p("$bumptransform", Matrix, "", "texcoord transform of $bumpmap"),
    p("$ssbump", Bool, "0", "$bumpmap is self-shadowing"),
    p("$ssbumpmathfix", Bool, "0", "brightness fix for ssbumps"),
    p("$normaltexture", Texture, "", "internal"),
    p("$phongexponenttexture", Texture, "", "material mask, inverted: r roughness, g metalness, b fuzz / detail"),
    p("$roughnessmultiplier", Float, "1.0", "roughness multiplier"),
    p("$metal", Bool, "0", "metalness workflow from $phongexponenttexture green"),
    p("$pbr", Bool, "0", "$masks workflow: rgb specular f0, alpha roughness"),
    p("$masks", Texture, "", "rgb specular f0 (gamma), alpha roughness"),
    // detail
    p("$detail", Texture, "", "detail texture"),
    p("$detailframe", Integer, "0", "frame of $detail"),
    p("$detailscale", Float, "4", "detail texture scale"),
    p("$detailblendmode", Integer, "0", "detail blend mode 0..11"),
    p("$detailblendfactor", Float, "", "detail blend amount"),
    p("$detailtint", Color, "", "detail tint"),
    p("$detailtexturetransform", Matrix, "", "texcoord transform of $detail"),
    p("$detailnormal", Texture, "", "detail normal map"),
    p("$detailnormalblendfactor", Float, "1.0", "detail normal blend amount"),
    p("$detailroughnessmultiplier", Float, "1.0", "scales detail colour before its luminance becomes roughness"),
    p("$detailroughnessinvertmix", Float, "0.0", "0 bright detail is rough, 1 dark detail is rough"),
    p("$detailroughnessblendfactor", Float, "0.0", "blend mask roughness towards detail roughness"),
    // subsurface / fuzz
    p("$subsurfacescattering", Bool, "0", "wrapped diffuse subsurface term"),
    p("$subsurfaceintensity", Float, "1.0", "unused"),
    p("$diffusionweights", Vec3, "[1 0.9 0.5]", "front-facing subsurface wrap per channel"),
    p("$diffusionweights2", Vec3, "[0.5 0.2 0.1]", "back-facing subsurface wrap per channel"),
    p("$fuzz", Bool, "0", "velvet rim term"),
    p("$fuzzmultiplier", Float, "0", "velvet rim strength"),
    // environment map
    p("$envmap", Texture, "", "cubemap"),
    p("$envmapframe", Integer, "", "frame of $envmap"),
    p("$envmaptint", Color, "", "cubemap tint"),
    p("$envmapcontrast", Float, "", "cubemap contrast"),
    p("$envmapsaturation", Float, "", "cubemap saturation"),
    p("$envmapfresnel", Float, "", "fresnel on the cubemap"),
    p("$envmapfresnelminmaxexp", Vec3, "", "fresnel min, max, exponent"),
    p("$envmaplightscale", Float, "", "scale the cubemap by lighting"),
    p("$envmapmaskflip", Bool, "", "flip the envmap mask"),
    p("$envmapmaxmip", Float, "6.0", "unused"),
    p("$disableenvmap", Bool, "0", "no cubemap reflections"),
    p("$fresnelreflection", Float, "", "1 mirror, 0 water"),
    // tints, lightwarp
    p("$blendtintbybasealpha", Bool, "", "base alpha blends in the tint"),
    p("$blendtintcoloroverbase", Float, "", "tint as multiplier vs replacement"),
    p("$desaturatewithbasealpha", Float, "", "base alpha desaturates the base"),
    p("$lightwarptexture", Texture, "", "lightwarp ramp"),
    p("$lightwarptextureframe", Integer, "", "frame of $lightwarptexture"),
    p("$lightwarpnobump", Bool, "0", "lightwarp with a forced bumpmap"),
    p("$linearwrite", Bool, "", "no srgb conversion of the result"),
    // tree sway
    p("$treesway", Integer, "", "0 off, 1 classic, 2 radial"),
    p("$treeswayheight", Float, "1000", "height where sway is full"),
    p("$treeswaystartheight", Float, "0.1", "height where sway starts"),
    p("$treeswayradius", Float, "300", "radius where sway is full"),
    p("$treeswaystartradius", Float, "0.2", "radius where sway starts"),
    p("$treeswayspeed", Float, "1", "trunk speed"),
    p("$treeswayspeedhighwindmultiplier", Float, "2", "speed multiplier in gusts"),
    p("$treeswaystrength", Float, "10", "trunk strength"),
    p("$treeswayscrumblespeed", Float, "5", "leaf speed"),
    p("$treeswayscrumblestrength", Float, "10", "leaf strength"),
    p("$treeswayscrumblefrequency", Float, "12", "leaf frequency"),
    p("$treeswayfalloffexp", Float, "1.5", "trunk falloff"),
    p("$treeswayscrumblefalloffexp", Float, "1", "leaf falloff"),
    p("$treeswayspeedlerpstart", Float, "3", "wind speed where gusts start"),
    p("$treeswayspeedlerpend", Float, "6", "wind speed where gusts are full"),
    p("$treeswaystatic", Bool, "", "static wind instead of env_wind"),
    p("$treeswaystaticvalues", Vec2, "", "the static wind"),
];

/// IShaderSystem ("ShaderSystem002"), set on Connect
static SHADER_SYSTEM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn shader_system() -> usize {
    SHADER_SYSTEM.load(std::sync::atomic::Ordering::Relaxed)
}

/// what materialsystem sees: the standard params, then the shader's own
fn param(i: i32) -> Option<&'static Param> {
    let i = usize::try_from(i).ok()?;
    STANDARD.get(i).or_else(|| SURFACE_GGX.get(i - STANDARD.len().min(i)).filter(|_| i >= STANDARD.len()))
}

/// nul-terminated copies of the param strings (they're returned as char pointers)
fn cstr(s: &'static str) -> *const c_char {
    static CACHE: parking_lot::Mutex<Option<std::collections::HashMap<usize, std::ffi::CString>>> = parking_lot::Mutex::new(None);
    let mut c = CACHE.lock();
    let c = c.get_or_insert_with(Default::default);
    c.entry(s.as_ptr() as usize).or_insert_with(|| std::ffi::CString::new(s).unwrap_or_default()).as_ptr()
}

// ---- IShaderDLLInternal (6 slots) ----

type Factory = unsafe extern "C" fn(*const c_char, *mut i32) -> usize;

unsafe extern "C" fn dll_connect(_t: *mut c_void, factory: Factory, _is_material_system: bool) -> bool {
    let sys = unsafe { factory(c"ShaderSystem002".as_ptr(), std::ptr::null_mut()) };
    SHADER_SYSTEM.store(sys, std::sync::atomic::Ordering::Relaxed);
    log::line(&format!("shader dll: connected (ShaderSystem002 {})", if sys != 0 { "found" } else { "missing" }));
    true
}
unsafe extern "C" fn dll_disconnect(_t: *mut c_void, _is_material_system: bool) {}
unsafe extern "C" fn dll_connect_plain(t: *mut c_void, factory: Factory) -> bool {
    unsafe { dll_connect(t, factory, false) }
}
unsafe extern "C" fn dll_disconnect_plain(_t: *mut c_void) {}
unsafe extern "C" fn dll_shader_count(_t: *mut c_void) -> i32 {
    shaders().len() as i32
}
unsafe extern "C" fn dll_get_shader(_t: *mut c_void, i: i32) -> *mut c_void {
    usize::try_from(i).ok().and_then(|i| shaders().get(i)).map_or(std::ptr::null_mut(), |s| s.as_ptr())
}

pub fn dll_interface() -> Interface {
    let f = |p: *const ()| p as usize;
    Interface::new(
        [0usize; 6],
        &[
            (0, f(dll_connect as *const ())),
            (1, f(dll_disconnect as *const ())),
            (2, f(dll_shader_count as *const ())),
            (3, f(dll_get_shader as *const ())),
            (4, f(dll_connect_plain as *const ())),
            (5, f(dll_disconnect_plain as *const ())),
        ],
    )
}

// ---- IShader (16 slots, plus CBaseShader's 4 that materialsystem never calls) ----

macro_rules! named {
    ($fn_name:ident, $s:literal) => {
        unsafe extern "C" fn $fn_name(_t: *mut c_void) -> *const c_char {
            $s.as_ptr()
        }
    };
}
named!(name_surface_ggx, c"SurfaceGGX");
named!(name_lux_surface_ggx, c"LUX_SurfaceGGX");
named!(name_surface_ggx_dx9, c"SurfaceGGX_DX9");

unsafe extern "C" fn fallback(_t: *mut c_void, _params: *mut *mut c_void) -> *const c_char {
    std::ptr::null()
}
unsafe extern "C" fn num_params(_t: *mut c_void) -> i32 {
    (STANDARD.len() + SURFACE_GGX.len()) as i32
}
unsafe extern "C" fn init_params(_t: *mut c_void, params: *mut *mut c_void, _material: *const c_char) {
    unsafe { surface_ggx::init_params(Params(params)) }
}
unsafe extern "C" fn init_instance(_t: *mut c_void, params: *mut *mut c_void, init: *mut c_void, _material: *const c_char, group: *const c_char) {
    unsafe { surface_ggx::init_instance(Params(params), init as usize, group) }
}
unsafe extern "C" fn draw_elements(_t: *mut c_void, params: *mut *mut c_void, _modulation: i32, shadow: *mut c_void, dynamic: *mut c_void, _compression: i32, _context: *mut *mut c_void) {
    unsafe { surface_ggx::draw(Params(params), shadow as usize, dynamic as usize) }
}
unsafe extern "C" fn param_name(_t: *mut c_void, i: i32) -> *const c_char {
    param(i).map_or(std::ptr::null(), |p| cstr(p.name))
}
unsafe extern "C" fn param_help(_t: *mut c_void, i: i32) -> *const c_char {
    param(i).map_or(std::ptr::null(), |p| cstr(p.help))
}
unsafe extern "C" fn param_type(_t: *mut c_void, i: i32) -> i32 {
    param(i).map_or(Integer as i32, |p| p.ty as i32)
}
unsafe extern "C" fn param_default(_t: *mut c_void, i: i32) -> *const c_char {
    param(i).map_or(std::ptr::null(), |p| cstr(p.default))
}
unsafe extern "C" fn modulation_flags(_t: *mut c_void, _params: *mut *mut c_void, _api: *mut c_void) -> i32 {
    0
}
unsafe extern "C" fn no(_t: *mut c_void) -> bool {
    false
}
unsafe extern "C" fn param_flags(_t: *mut c_void, i: i32) -> i32 {
    param(i).map_or(0, |p| p.flags)
}
unsafe extern "C" fn zero(_t: *mut c_void) -> i32 {
    0
}
unsafe extern "C" fn nothing(_t: *mut c_void) {}

fn shader(name: unsafe extern "C" fn(*mut c_void) -> *const c_char) -> Interface {
    let f = |p: *const ()| p as usize;
    let nothing = f(nothing as *const ());
    Interface::new(
        [nothing; 20],
        &[
            (0, f(name as *const ())),
            (1, f(fallback as *const ())),
            (2, f(num_params as *const ())),
            (3, f(init_params as *const ())),
            (4, f(init_instance as *const ())),
            (5, f(draw_elements as *const ())),
            (6, f(param_name as *const ())),
            (7, f(param_help as *const ())),
            (8, f(param_type as *const ())),
            (9, f(param_default as *const ())),
            (10, f(modulation_flags as *const ())),
            (11, f(no as *const ())),
            (12, f(no as *const ())),
            (13, f(no as *const ())),
            (14, f(param_flags as *const ())),
            (15, f(zero as *const ())),
        ],
    )
}

fn shaders() -> &'static [Interface] {
    static ALL: OnceLock<Vec<Interface>> = OnceLock::new();
    ALL.get_or_init(|| vec![shader(name_surface_ggx), shader(name_lux_surface_ggx), shader(name_surface_ggx_dx9)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_follow_the_standard_ones_without_duplicates() {
        assert_eq!(param(0).unwrap().name, "$flags");
        assert_eq!(param(12).unwrap().name, "$srgbtint");
        assert_eq!(param(13).unwrap().name, SURFACE_GGX[0].name);
        assert!(param((STANDARD.len() + SURFACE_GGX.len()) as i32).is_none());
        assert!(param(-1).is_none());
        let mut names: Vec<&str> = STANDARD.iter().chain(SURFACE_GGX).map(|p| p.name).collect();
        names.sort();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n);
        assert!(names.iter().all(|n| n.starts_with('$') && n.to_lowercase() == *n));
    }
}

// ---- calling into the engine ----

/// virtual call `slot` on `obj` (x64: `this` first)
unsafe fn vfn<F: Copy>(obj: usize, slot: usize) -> F {
    unsafe {
        let vt = *(obj as *const *const usize);
        std::mem::transmute_copy(&*vt.add(slot))
    }
}

/// a material's param array (IMaterialVar**), in our param order
#[derive(Clone, Copy)]
pub struct Params(*mut *mut c_void);

/// gmod's IMaterialVar: GetTextureValue slot 0, SetIntValue 4, IsDefined 17, GetMatrixValue 20;
/// int value at +0x10, float / vector at +0x14 (read from stdshader_dx9's accessors)
#[derive(Clone, Copy)]
pub struct Var(usize);

impl Params {
    fn index(name: &str) -> Option<usize> {
        static MAP: OnceLock<std::collections::HashMap<&'static str, usize>> = OnceLock::new();
        MAP.get_or_init(|| STANDARD.iter().chain(SURFACE_GGX).enumerate().map(|(i, p)| (p.name, i)).collect()).get(name).copied()
    }

    pub fn var(self, name: &str) -> Option<Var> {
        let i = Self::index(name)?;
        let v = unsafe { *self.0.add(i) } as usize;
        (v != 0).then_some(Var(v))
    }

    /// defined and non-zero
    pub fn flag(self, name: &str) -> bool {
        self.var(name).is_some_and(|v| v.defined() && v.int() != 0)
    }

    pub fn defined(self, name: &str) -> bool {
        self.var(name).is_some_and(|v| v.defined())
    }

    pub fn float_or(self, name: &str, default: f32) -> f32 {
        self.var(name).filter(|v| v.defined()).map_or(default, |v| v.float())
    }

    pub fn vec3_or(self, name: &str, default: [f32; 3]) -> [f32; 3] {
        self.var(name).filter(|v| v.defined()).map_or(default, |v| {
            let x = v.vec();
            [x[0], x[1], x[2]]
        })
    }
}

impl Var {
    pub fn defined(self) -> bool {
        unsafe { vfn::<unsafe extern "C" fn(usize) -> bool>(self.0, 17)(self.0) }
    }
    pub fn int(self) -> i32 {
        unsafe { *((self.0 + 0x10) as *const i32) }
    }
    pub fn float(self) -> f32 {
        unsafe { *((self.0 + 0x14) as *const f32) }
    }
    pub fn vec(self) -> [f32; 4] {
        unsafe { *((self.0 + 0x14) as *const [f32; 4]) }
    }
    pub fn texture(self) -> usize {
        unsafe { vfn::<unsafe extern "C" fn(usize) -> usize>(self.0, 0)(self.0) }
    }
    pub fn set_int(self, v: i32) {
        unsafe { vfn::<unsafe extern "C" fn(usize, i32)>(self.0, 4)(self.0, v) }
    }
    /// VMatrix, row major
    pub fn matrix(self) -> [f32; 16] {
        unsafe { *vfn::<unsafe extern "C" fn(usize) -> *const [f32; 16]>(self.0, 20)(self.0) }
    }
}

mod surface_ggx {
    use super::*;
    use crate::builtin::ggx;
    use crate::vtables::{shader_api as api, shader_shadow as sh};

    const TEXTUREFLAGS_SRGB: i32 = 0x40;
    const VERTEX_POSITION: u32 = 1;
    const VERTEX_NORMAL: u32 = 2;
    // $flags
    const ALPHATEST: i32 = 1 << 8;
    const ADDITIVE: i32 = 1 << 7;
    const NOCULL: i32 = 1 << 13;
    const IGNOREZ: i32 = 1 << 15;
    const DECAL: i32 = 1 << 16;
    const TRANSLUCENT: i32 = 1 << 21;
    const ENVMAPMODE: i32 = 1 << 25;
    // $flags2
    const LIGHTING_VERTEX_LIT: i32 = 1 << 1;
    const USES_ENV_CUBEMAP: i32 = 1 << 5;
    const NEEDS_TANGENT_SPACES: i32 = 1 << 6;
    const NEEDS_BAKED_LIGHTING_SNAPSHOTS: i32 = 1 << 9;
    const SUPPORTS_HW_SKINNING: i32 = 1 << 18;
    // StandardTextureId_t
    const TEXTURE_WHITE: i32 = 4;
    // ShaderBlendFactor_t
    const BLEND_ONE: i32 = 1;
    const BLEND_SRC_ALPHA: i32 = 4;
    const BLEND_ONE_MINUS_SRC_ALPHA: i32 = 5;

    fn set_flags2(p: Params, bits: i32) {
        for name in ["$flags2", "$flags_defined2"] {
            if let Some(v) = p.var(name) {
                v.set_int(v.int() | bits);
            }
        }
    }

    /// what the engine prepares for this material: lighting, skinning, tangents, cubemaps
    pub unsafe fn init_params(p: Params) {
        let mut bits = LIGHTING_VERTEX_LIT | SUPPORTS_HW_SKINNING | NEEDS_BAKED_LIGHTING_SNAPSHOTS;
        if p.defined("$bumpmap") {
            bits |= NEEDS_TANGENT_SPACES;
        }
        let flags = p.var("$flags").map_or(0, |v| v.int());
        if p.defined("$envmap") && flags & ENVMAPMODE == 0 {
            bits |= USES_ENV_CUBEMAP;
        }
        set_flags2(p, bits);
    }

    /// IShaderInit: LoadTexture 0, LoadBumpMap 1, LoadCubeMap 2
    pub unsafe fn init_instance(p: Params, init: usize, group: *const c_char) {
        if init == 0 {
            return;
        }
        let load = |name: &str, flags: i32| {
            if let Some(v) = p.var(name).filter(|v| v.defined()) {
                unsafe { vfn::<unsafe extern "C" fn(usize, usize, *const c_char, i32)>(init, 0)(init, v.0, group, flags) };
            }
        };
        let load_bump = |name: &str| {
            if let Some(v) = p.var(name).filter(|v| v.defined()) {
                unsafe { vfn::<unsafe extern "C" fn(usize, usize, *const c_char)>(init, 1)(init, v.0, group) };
            }
        };
        load("$basetexture", TEXTUREFLAGS_SRGB);
        load("$phongexponenttexture", 0);
        load("$masks", 0);
        load_bump("$bumpmap");
        load_bump("$detailnormal");
        if let Some(v) = p.var("$envmap").filter(|v| v.defined()) {
            let srgb = if crate::hwconfig::hdr_type() == 0 { TEXTUREFLAGS_SRGB } else { 0 };
            unsafe { vfn::<unsafe extern "C" fn(usize, *mut *mut c_void, usize, i32)>(init, 2)(init, p.0, v.0, srgb) };
        }
    }

    fn combo(p: Params) -> u32 {
        let mut st = 0;
        let flags = p.var("$flags").map_or(0, |v| v.int());
        if p.defined("$bumpmap") {
            st |= ggx::BUMP;
            if p.defined("$detailnormal") {
                st |= ggx::DETAIL_NORMAL;
            }
        }
        if p.flag("$pbr") && p.defined("$masks") {
            st |= ggx::PBR;
        } else if p.defined("$phongexponenttexture") {
            st |= ggx::MASK;
            if p.flag("$metal") {
                st |= ggx::METAL;
            }
            if p.flag("$fuzz") {
                st |= ggx::FUZZ;
            }
        }
        if p.defined("$envmap") && !p.flag("$disableenvmap") {
            st |= ggx::ENVMAP;
        }
        if p.flag("$subsurfacescattering") {
            st |= ggx::SSS;
        }
        if flags & (TRANSLUCENT | ADDITIVE | ALPHATEST) != 0 {
            st |= ggx::ALPHA;
        }
        st
    }

    fn gamma(c: [f32; 3]) -> [f32; 3] {
        c.map(|x| x.max(0.0).powf(2.2))
    }

    pub unsafe fn draw(p: Params, shadow: usize, dynamic: usize) {
        let st = combo(p);
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
                    // SHADER_POLYOFFSET_DECAL
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
                    // SHADER_ALPHAFUNC_GEQUAL
                    vfn::<unsafe extern "C" fn(usize, i32, f32)>(shadow, sh::slot::AlphaFunc)(shadow, 6, if r > 0.0 { r } else { 0.5 });
                }
                call_ib(sh::slot::EnableTexture, 0, true);
                call_ib(sh::slot::EnableSRGBRead, 0, true);
                if st & ggx::BUMP != 0 {
                    call_ib(sh::slot::EnableTexture, 1, true);
                }
                if st & (ggx::MASK | ggx::PBR) != 0 {
                    call_ib(sh::slot::EnableTexture, 2, true);
                }
                if st & ggx::ENVMAP != 0 {
                    call_ib(sh::slot::EnableTexture, 3, true);
                    call_ib(sh::slot::EnableSRGBRead, 3, crate::hwconfig::hdr_type() == 0);
                }
                if st & ggx::DETAIL_NORMAL != 0 {
                    call_ib(sh::slot::EnableTexture, 4, true);
                }
                call_b(sh::slot::EnableSRGBWrite, true);
                let dims = [2i32];
                let user_data = if st & ggx::BUMP != 0 { 4 } else { 0 };
                vfn::<unsafe extern "C" fn(usize, u32, i32, *const i32, i32)>(shadow, sh::slot::VertexShaderVertexFormat)(shadow, VERTEX_POSITION | VERTEX_NORMAL, 1, dims.as_ptr(), user_data);
                let vs = std::ffi::CString::new(ggx::VS).unwrap();
                let ps = std::ffi::CString::new(ggx::PS).unwrap();
                vfn::<unsafe extern "C" fn(usize, *const c_char, i32)>(shadow, sh::slot::SetVertexShader)(shadow, vs.as_ptr(), 0);
                vfn::<unsafe extern "C" fn(usize, *const c_char, i32)>(shadow, sh::slot::SetPixelShader)(shadow, ps.as_ptr(), st as i32);
                // SHADER_FOGMODE_BLACK for additive, else FOGCOLOR
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
            // IShaderSystem::BindTexture(sampler, ITexture*, frame) is slot 2 in gmod's build
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
            if st & ggx::BUMP != 0 {
                bind(1, "$bumpmap", "$bumpframe");
            }
            if st & ggx::PBR != 0 {
                bind(2, "$masks", "$frame");
            } else if st & ggx::MASK != 0 {
                bind(2, "$phongexponenttexture", "$frame");
            }
            if st & ggx::ENVMAP != 0 {
                bind(3, "$envmap", "$envmapframe");
            }
            if st & ggx::DETAIL_NORMAL != 0 {
                bind(4, "$detailnormal", "$frame");
            }
            // base texture transform rows -> c48, c49
            let m = p.var("$basetexturetransform").filter(|v| v.defined()).map_or([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], |v| v.matrix());
            vfn::<unsafe extern "C" fn(usize, i32, *const f32, i32, bool)>(dynamic, api::slot::SetVertexShaderConstant)(dynamic, 48, m.as_ptr(), 2, false);

            let ps = crate::api::builtin_set_ps_reg;
            let color = gamma(p.vec3_or("$color", [1.0; 3]));
            let color2 = if p.flag("$notint") { [1.0; 3] } else { gamma(p.vec3_or("$color2", [1.0; 3])) };
            ps(ggx::C_TINT, [color[0] * color2[0], color[1] * color2[1], color[2] * color2[2], p.float_or("$alpha", 1.0)]);
            ps(ggx::C_SURFACE, [p.float_or("$roughnessmultiplier", 1.0), 0.0, 0.0, 0.0]);
            let eye = crate::api::builtin_eye_position();
            ps(ggx::C_EYE, [eye[0], eye[1], eye[2], 1.0]);
            let env_tint = gamma(p.vec3_or("$envmaptint", [1.0; 3]));
            ps(ggx::C_ENV, [env_tint[0], env_tint[1], env_tint[2], 0.0]);
            ps(ggx::C_EXTRA, [0.0, p.float_or("$fuzzmultiplier", 0.0), p.float_or("$detailnormalblendfactor", 1.0), p.float_or("$detailscale", 1.0)]);
            let lights = crate::api::ps_lights_full(ggx::C_LIGHTS);
            let w = p.vec3_or("$diffusionweights", [1.0, 0.9, 0.5]);
            ps(ggx::C_DIFFUSION, [w[0], w[1], w[2], lights as f32]);
            let w2 = p.vec3_or("$diffusionweights2", [0.5, 0.2, 0.1]);
            ps(ggx::C_DIFFUSION2, [w2[0], w2[1], w2[2], 0.0]);
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
}
