//! hardware caps: a dx 9.5 / sm3 device with integer hdr (rgba8 frame buffer, tonemapping in the
//! shaders)

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::device::DX_LEVEL;

macro_rules! constant {
    ($name:ident, $t:ty, $v:expr) => {
        unsafe extern "C" fn $name(_t: *mut c_void) -> $t {
            $v
        }
    };
}

constant!(yes, bool, true);
constant!(no, bool, false);
constant!(four, i32, 4);
constant!(eight, i32, 8);
constant!(sixteen, i32, 16);
constant!(zero, i32, 0);
constant!(one, i32, 1);
constant!(tex_max, i32, 16384);
constant!(tex_memory, i32, 1 << 30);
constant!(vs_constants, i32, 256);
constant!(ps_constants, i32, 224);
constant!(blend_matrix_indices, i32, 256);
constant!(vs_blend_matrices, i32, 53);
constant!(clip_planes, i32, 6);
constant!(dx_level, i32, DX_LEVEL);
constant!(vertex_texture_dim, i32, 4096);
constant!(tex_depth, i32, 2048);
constant!(null_ptr, usize, 0);

unsafe extern "C" fn shader_dll_name(_t: *mut c_void) -> *const u8 {
    c"DEFAULT".as_ptr() as *const u8
}
pub const HDR_INTEGER: i32 = 1;

/// the engine turns this on when a map with hdr lighting loads
static HDR_ENABLED: AtomicBool = AtomicBool::new(false);
/// 0 = not found yet
static HDR_LEVEL_CVAR: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn supports_hdr_mode(_t: *mut c_void, mode: i32) -> bool {
    mode == 0 || mode == HDR_INTEGER
}
unsafe extern "C" fn get_hdr_enabled(_t: *mut c_void) -> bool {
    HDR_ENABLED.load(Ordering::Relaxed)
}
unsafe extern "C" fn set_hdr_enabled(_t: *mut c_void, on: bool) {
    if HDR_ENABLED.swap(on, Ordering::Relaxed) != on {
        crate::log::line(&format!("SetHDREnabled({on}): mat_hdr_level {}", hdr_level()));
    }
}
unsafe extern "C" fn get_hdr_type(_t: *mut c_void) -> i32 {
    hdr_type()
}
unsafe extern "C" fn hardware_hdr_type(_t: *mut c_void) -> i32 {
    HDR_INTEGER
}

/// integer hdr while the map has hdr lighting and mat_hdr_level >= 2
pub fn hdr_type() -> i32 {
    if HDR_ENABLED.load(Ordering::Relaxed) && hdr_level() >= 2 { HDR_INTEGER } else { 0 }
}

/// defaults to 2 when the cvar can't be read
fn hdr_level() -> i32 {
    let mut cv = HDR_LEVEL_CVAR.load(Ordering::Relaxed);
    if cv == 0 {
        cv = find_hdr_level_cvar();
        HDR_LEVEL_CVAR.store(cv, Ordering::Relaxed);
    }
    if cv == 0 || cv == usize::MAX {
        return 2;
    }
    // x64 ConVar: name @24, parent @56, string @72, int value @88
    unsafe {
        let parent = *((cv + 56) as *const usize);
        *((parent + 88) as *const i32)
    }
}

fn find_hdr_level_cvar() -> usize {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        let module = GetModuleHandleA(c"vstdlib.dll".as_ptr() as *const u8);
        if module.is_null() {
            return 0;
        }
        let Some(create) = GetProcAddress(module, c"CreateInterface".as_ptr() as *const u8) else { return 0 };
        let create: unsafe extern "C" fn(*const i8, *mut i32) -> usize = std::mem::transmute(create);
        let icvar = create(c"VEngineCvar007".as_ptr(), std::ptr::null_mut());
        if icvar == 0 {
            return 0;
        }
        // ICvar::FindVar is slot 16 (checked against CCvar's vtable in vstdlib.dll)
        let find: unsafe extern "C" fn(usize, *const i8) -> usize = std::mem::transmute(*(*(icvar as *const *const usize)).add(16));
        let cv = find(icvar, c"mat_hdr_level".as_ptr());
        if cv == 0 {
            return 0;
        }
        let name_is = |p: usize| {
            let n = *((p + 24) as *const *const std::ffi::c_char);
            !n.is_null() && std::ffi::CStr::from_ptr(n).to_bytes() == b"mat_hdr_level"
        };
        let parent = *((cv + 56) as *const usize);
        let ok = name_is(cv) && parent != 0 && name_is(parent) && {
            let s = *((parent + 72) as *const *const std::ffi::c_char);
            !s.is_null() && std::ffi::CStr::from_ptr(s).to_str().ok().and_then(|s| s.trim().parse::<f32>().ok()).map(|v| v as i32) == Some(*((parent + 88) as *const i32))
        };
        crate::log::line(&format!("mat_hdr_level ConVar at {cv:#x}: {}", if ok { "layout ok" } else { "unexpected layout, assuming 2" }));
        if ok { cv } else { usize::MAX }
    }
    #[cfg(not(windows))]
    0
}
unsafe extern "C" fn ignore(_t: *mut c_void) {}

pub fn interface() -> crate::Interface {
    use crate::vtables::hardware_config::{slot as s, stubs};
    let f = |p: *const ()| p as usize;
    crate::Interface::new(
        stubs(),
        &[
            (s::HasDestAlphaBuffer, f(yes as *const ())),
            (s::HasStencilBuffer, f(yes as *const ())),
            (s::GetFrameBufferColorDepth, f(four as *const ())),
            (s::GetSamplerCount, f(sixteen as *const ())),
            (s::HasSetDeviceGammaRamp, f(yes as *const ())),
            (s::SupportsCompressedTextures, f(yes as *const ())),
            (s::SupportsCompressedVertices, f(zero as *const ())),
            (s::SupportsNormalMapCompression, f(no as *const ())),
            (s::SupportsVertexAndPixelShaders, f(yes as *const ())),
            (s::SupportsPixelShaders_1_4, f(yes as *const ())),
            (s::SupportsStaticControlFlow, f(yes as *const ())),
            (s::SupportsPixelShaders_2_0, f(yes as *const ())),
            (s::SupportsVertexShaders_2_0, f(yes as *const ())),
            (s::MaximumAnisotropicLevel, f(sixteen as *const ())),
            (s::MaxTextureWidth, f(tex_max as *const ())),
            (s::MaxTextureHeight, f(tex_max as *const ())),
            (s::TextureMemorySize, f(tex_memory as *const ())),
            (s::SupportsOverbright, f(yes as *const ())),
            (s::SupportsCubeMaps, f(yes as *const ())),
            (s::SupportsMipmappedCubemaps, f(yes as *const ())),
            (s::SupportsNonPow2Textures, f(yes as *const ())),
            (s::GetTextureStageCount, f(four as *const ())),
            (s::NumVertexShaderConstants, f(vs_constants as *const ())),
            (s::NumPixelShaderConstants, f(ps_constants as *const ())),
            (s::MaxNumLights, f(four as *const ())),
            (s::SupportsHardwareLighting, f(no as *const ())),
            (s::MaxBlendMatrices, f(four as *const ())),
            (s::MaxBlendMatrixIndices, f(blend_matrix_indices as *const ())),
            (s::MaxTextureAspectRatio, f(tex_max as *const ())),
            (s::MaxVertexShaderBlendMatrices, f(vs_blend_matrices as *const ())),
            (s::MaxUserClipPlanes, f(clip_planes as *const ())),
            (s::UseFastClipping, f(no as *const ())),
            (s::GetDXSupportLevel, f(dx_level as *const ())),
            (s::GetShaderDLLName, f(shader_dll_name as *const ())),
            (s::ReadPixelsFromFrontBuffer, f(no as *const ())),
            (s::PreferDynamicTextures, f(no as *const ())),
            (s::SupportsHDR, f(no as *const ())),
            (s::HasProjectedBumpEnv, f(no as *const ())),
            (s::SupportsSpheremapping, f(no as *const ())),
            (s::NeedsAAClamp, f(no as *const ())),
            (s::NeedsATICentroidHack, f(no as *const ())),
            (s::SupportsColorOnSecondStream, f(yes as *const ())),
            (s::SupportsStaticPlusDynamicLighting, f(yes as *const ())),
            (s::PreferReducedFillrate, f(no as *const ())),
            (s::GetMaxDXSupportLevel, f(dx_level as *const ())),
            (s::SpecifiesFogColorInLinearSpace, f(no as *const ())),
            (s::SupportsSRGB, f(yes as *const ())),
            (s::FakeSRGBWrite, f(no as *const ())),
            (s::CanDoSRGBReadFromRTs, f(yes as *const ())),
            (s::SupportsGLMixedSizeTargets, f(no as *const ())),
            (s::IsAAEnabled, f(crate::device::is_aa_enabled as *const ())),
            (s::GetVertexTextureCount, f(four as *const ())),
            (s::GetMaxVertexTextureDimension, f(vertex_texture_dim as *const ())),
            (s::MaxTextureDepth, f(tex_depth as *const ())),
            (s::GetHDRType, f(get_hdr_type as *const ())),
            (s::GetHardwareHDRType, f(hardware_hdr_type as *const ())),
            (s::SupportsPixelShaders_2_b, f(yes as *const ())),
            (s::SupportsStreamOffset, f(yes as *const ())),
            (s::StencilBufferBits, f(eight as *const ())),
            (s::MaxViewports, f(one as *const ())),
            (s::OverrideStreamOffsetSupport, f(ignore as *const ())),
            (s::GetShadowFilterMode, f(zero as *const ())),
            (s::NeedsShaderSRGBConversion, f(zero as *const ())),
            (s::UsesSRGBCorrectBlending, f(no as *const ())), // blending happens in gamma space, like dx9
            (s::SupportsShaderModel_3_0, f(yes as *const ())),
            (s::HasFastVertexTextures, f(no as *const ())),
            (s::MaxHWMorphBatchCount, f(zero as *const ())),
            (s::ActuallySupportsPixelShaders_2_b, f(yes as *const ())),
            (s::SupportsHDRMode, f(supports_hdr_mode as *const ())),
            (s::GetHDREnabled, f(get_hdr_enabled as *const ())),
            (s::SetHDREnabled, f(set_hdr_enabled as *const ())),
            (s::SupportsBorderColor, f(yes as *const ())),
            (s::SupportsFetch4, f(no as *const ())),
            (s::GetHWSpecificShaderDLLName, f(null_ptr as *const ())),
        ],
    )
}
