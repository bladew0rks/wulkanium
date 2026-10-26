//! SurfaceGGX: our own shader, under the names other dx11 renderers use too
use super::*;
use crate::builtin::ggx;
use crate::vtables::{shader_api as api, shader_shadow as sh};

pub static DEF: Def = Def {
    names: &[c"SurfaceGGX", c"LUX_SurfaceGGX", c"SurfaceGGX_DX9"],
    stock: None,
    params: PARAMS,
    supports: |_| true,
    init_params,
    init_instance,
    draw,
    translucent: |p| p.var("$flags").is_some_and(|v| v.int() & (TRANSLUCENT | ADDITIVE) != 0),
};

/// SurfaceGGX's own params (names, types and defaults as materials written for it expect)
pub const PARAMS: &[Param] = &[
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
    p("$envmaptint", Color, "[1 1 1]", "cubemap tint"),
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


const TEXTUREFLAGS_SRGB: i32 = 0x40;
const VERTEX_POSITION: u32 = 1;
const VERTEX_NORMAL: u32 = 2;
/// compressed vertices are fine: the built-in vertex shader unpacks them
const VERTEX_FORMAT_COMPRESSED: u32 = 0x400;
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
unsafe fn init_params(p: Params) {
    let mut bits = LIGHTING_VERTEX_LIT | SUPPORTS_HW_SKINNING | NEEDS_BAKED_LIGHTING_SNAPSHOTS;
    if p.defined("$bumpmap") {
        bits |= NEEDS_TANGENT_SPACES;
    }
    let flags = p.var("$flags").map_or(0, |v| v.int());
    if p.defined("$envmap") && flags & ENVMAPMODE == 0 {
        bits |= USES_ENV_CUBEMAP;
    }
    set_flags2(p, bits);
    apply_defaults(p);
}

/// materialsystem zero-fills every param still undefined after InitShaderParams, so the
/// defaults have to be set here (stock shaders do the same in their param init)
fn apply_defaults(p: Params) {
    for param in STANDARD.iter().chain(PARAMS) {
        if param.default.is_empty() || matches!(param.ty, Texture) {
            continue;
        }
        let Some(v) = p.var(param.name).filter(|v| !v.defined()) else { continue };
        let nums: Vec<f32> = param.default.trim_matches(['[', ']', '{', '}']).split_whitespace().filter_map(|x| x.parse().ok()).collect();
        match (param.ty, nums.as_slice()) {
            (Integer | Bool, [x, ..]) => v.set_int(*x as i32),
            (Float, [x, ..]) => v.set_float(*x),
            (Color | Vec3 | Vec4, [x, y, z, ..]) => v.set_vec3(*x, *y, *z),
            (Color | Vec3, [x]) => v.set_vec3(*x, *x, *x),
            _ => {}
        }
    }
}

/// IShaderInit: LoadTexture 0, LoadBumpMap 1, LoadCubeMap 2
unsafe fn init_instance(p: Params, init: usize, group: *const c_char) {
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
        unsafe { vfn::<unsafe extern "C" fn(usize, *mut *mut c_void, usize, i32)>(init, 2)(init, p.ptr, v.0, srgb) };
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

unsafe fn draw(p: Params, shadow: usize, dynamic: usize) {
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
            vfn::<unsafe extern "C" fn(usize, u32, i32, *const i32, i32)>(shadow, sh::slot::VertexShaderVertexFormat)(shadow, VERTEX_POSITION | VERTEX_NORMAL | VERTEX_FORMAT_COMPRESSED, 1, dims.as_ptr(), user_data);
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
        ps(ggx::C_EXTRA, [debug_mode(), p.float_or("$fuzzmultiplier", 0.0), p.float_or("$detailnormalblendfactor", 1.0), p.float_or("$detailscale", 1.0)]);
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

/// wk_ggx_debug: shows one surfaceggx input instead of the shaded result
fn debug_mode() -> f32 {
    static CV: parking_lot::Mutex<Option<crate::cvar::Cache>> = parking_lot::Mutex::new(None);
    CV.lock().get_or_insert_with(Default::default).get("wk_ggx_debug").map_or(0.0, |c| c.int() as f32)
}

