//! hardware caps: a dx 9.5 / sm3 device with integer hdr (rgba8 frame buffer, tonemapping in the
//! shaders)

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

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

unsafe extern "C" fn shader_dll_name(_t: *mut c_void) -> *const u8 {
    c"DEFAULT".as_ptr() as *const u8
}
pub const HDR_INTEGER: i32 = 1;

/// the engine turns this on when a map with hdr lighting loads
static HDR_ENABLED: AtomicBool = AtomicBool::new(false);
/// 0 = not found yet

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
    static CV: parking_lot::Mutex<Option<crate::cvar::Cache>> = parking_lot::Mutex::new(None);
    CV.lock().get_or_insert_with(Default::default).get("mat_hdr_level").map_or(2, |c| c.int())
}
unsafe extern "C" fn ignore(_t: *mut c_void) {}
/// materialsystem loads this as an extra shader dll: ourselves, for SurfaceGGX (see shaderdll)
unsafe extern "C" fn hw_shader_dll_name(_t: *mut c_void) -> *const std::ffi::c_char {
    crate::log::line("GetHWSpecificShaderDLLName -> shaderapiwgpu.dll");
    c"shaderapiwgpu.dll".as_ptr()
}

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
            // VERTEX_COMPRESSION_ON: model materials then carry VERTEX_FORMAT_COMPRESSED, which gmod
            // requires before it applies Entity:SetMaterial (see abi::compute_vertex_desc)
            (s::SupportsCompressedVertices, f(compression_on as *const ())),
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
            (s::GetHWSpecificShaderDLLName, f(hw_shader_dll_name as *const ())),
        ],
    )
}

/// VertexCompressionType_t VERTEX_COMPRESSION_ON
unsafe extern "C" fn compression_on(_t: *mut c_void) -> i32 {
    1
}
