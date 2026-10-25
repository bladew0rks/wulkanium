//! `IShaderShadow`: per-pass static state; TakeSnapshot turns it into an index into a table of
//! distinct states

use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_void};
use std::sync::Mutex;

/// a shader name as a small id, so draws carry no refcounted strings (their refcounts would
/// bounce between the game and render threads)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShaderName(u32);

static NAMES: std::sync::RwLock<Vec<std::sync::Arc<str>>> = std::sync::RwLock::new(Vec::new());

impl ShaderName {
    /// the depth-only null pixel shader
    pub const NONE: ShaderName = ShaderName(u32::MAX);

    pub fn intern(name: &str) -> ShaderName {
        let mut names = NAMES.write().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = names.iter().position(|n| &**n == name) {
            return ShaderName(i as u32);
        }
        names.push(name.into());
        ShaderName(names.len() as u32 - 1)
    }

    pub fn is_none(self) -> bool {
        self == ShaderName::NONE
    }

    pub fn get(self) -> std::sync::Arc<str> {
        if self.is_none() {
            return "".into();
        }
        NAMES.read().unwrap_or_else(|e| e.into_inner())[self.0 as usize].clone()
    }
}

impl std::fmt::Display for ShaderName {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.get())
    }
}

use crate::abi::*;

/// enum values stay raw, as the material system passes them
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShadowState {
    pub depth_func: i32,
    pub depth_test: bool,
    pub depth_write: bool,
    pub poly_offset: i32,
    pub color_write: bool,
    pub alpha_write: bool,
    pub blend: bool,
    pub blend_src: i32,
    pub blend_dst: i32,
    pub blend_op: i32,
    pub blend_separate_alpha: bool,
    pub blend_src_alpha: i32,
    pub blend_dst_alpha: i32,
    pub blend_op_alpha: i32,
    pub alpha_test: bool,
    pub alpha_func: i32,
    pub alpha_ref_bits: u32,
    pub cull: bool,
    pub vertex_format: VertexFormat,
    pub morph_format: u64,
    pub vertex_shader: Option<ShaderName>,
    pub vertex_shader_static: i32,
    pub pixel_shader: Option<ShaderName>,
    pub pixel_shader_static: i32,
    pub srgb_write: bool,
    pub srgb_read: u32,
    pub textures_enabled: u32,
    pub shadow_filtering: u32,
    pub fog_mode: i32,
    pub disable_fog_gamma: bool,
    pub alpha_to_coverage: bool,
    pub stencil: bool,
    pub stencil_func: i32,
    pub stencil_pass: i32,
    pub stencil_fail: i32,
    pub stencil_depth_fail: i32,
    pub stencil_ref: i32,
    pub stencil_mask: i32,
    pub stencil_write_mask: i32,
    /// 0 point, 1 line, 2 fill
    pub poly_mode: i32,
}

const DEPTHFUNC_NEAREROREQUAL: i32 = 3;
const BLEND_ZERO: i32 = 0;
const BLEND_ONE: i32 = 1;

impl Default for ShadowState {
    /// stock defaults
    fn default() -> Self {
        ShadowState {
            depth_func: DEPTHFUNC_NEAREROREQUAL,
            depth_test: true,
            depth_write: true,
            poly_offset: 0,
            color_write: true,
            alpha_write: false,
            blend: false,
            blend_src: BLEND_ONE,
            blend_dst: BLEND_ZERO,
            blend_op: 0,
            blend_separate_alpha: false,
            blend_src_alpha: BLEND_ONE,
            blend_dst_alpha: BLEND_ZERO,
            blend_op_alpha: 0,
            alpha_test: false,
            // dx9 default: alpha test GEQUAL 0.7
            alpha_func: 6,
            alpha_ref_bits: 0.7f32.to_bits(),
            cull: true,
            vertex_format: 0,
            morph_format: 0,
            vertex_shader: None,
            vertex_shader_static: 0,
            pixel_shader: None,
            pixel_shader_static: 0,
            srgb_write: false,
            srgb_read: 0,
            textures_enabled: 0,
            shadow_filtering: 0,
            fog_mode: 0,
            disable_fog_gamma: false,
            alpha_to_coverage: false,
            stencil: false,
            stencil_func: 0,
            stencil_pass: 0,
            stencil_fail: 0,
            stencil_depth_fail: 0,
            stencil_ref: 0,
            stencil_mask: -1,
            stencil_write_mask: -1,
            poly_mode: 2,
        }
    }
}

pub struct Shadow {
    pub current: ShadowState,
    pub snapshots: Vec<std::sync::Arc<ShadowState>>,
    index: HashMap<ShadowState, i16>,
}

pub static SHADOW: Mutex<Option<Shadow>> = Mutex::new(None);

pub fn with<R>(f: impl FnOnce(&mut Shadow) -> R) -> R {
    let _shader = crate::lock::guard();
    let mut g = SHADOW.lock().unwrap_or_else(|e| e.into_inner());
    let s = g.get_or_insert_with(|| Shadow { current: ShadowState::default(), snapshots: Vec::new(), index: HashMap::new() });
    f(s)
}

impl Shadow {
    pub fn take(&mut self) -> i16 {
        if let Some(&id) = self.index.get(&self.current) {
            return id;
        }
        let id = self.snapshots.len().min(i16::MAX as usize) as i16;
        self.snapshots.push(std::sync::Arc::new(self.current.clone()));
        self.index.insert(self.current.clone(), id);
        id
    }

    pub fn get(&self, id: i16) -> Option<&std::sync::Arc<ShadowState>> {
        self.snapshots.get(id as usize)
    }
}

fn set(f: impl FnOnce(&mut ShadowState)) {
    with(|s| f(&mut s.current));
}

fn cstr(p: *const c_char) -> Option<String> {
    (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
}

unsafe extern "C" fn set_default_state(_t: *mut c_void) {
    set(|s| *s = ShadowState::default());
}
unsafe extern "C" fn depth_func(_t: *mut c_void, v: i32) {
    set(|s| s.depth_func = v);
}
unsafe extern "C" fn enable_depth_writes(_t: *mut c_void, v: bool) {
    set(|s| s.depth_write = v);
}
unsafe extern "C" fn enable_depth_test(_t: *mut c_void, v: bool) {
    set(|s| s.depth_test = v);
}
unsafe extern "C" fn enable_poly_offset(_t: *mut c_void, v: i32) {
    set(|s| s.poly_offset = v);
}
unsafe extern "C" fn poly_mode(_t: *mut c_void, _face: i32, mode: i32) {
    set(|s| s.poly_mode = mode);
}
unsafe extern "C" fn enable_stencil(_t: *mut c_void, v: bool) {
    set(|s| s.stencil = v);
}
unsafe extern "C" fn stencil_func(_t: *mut c_void, v: i32) {
    set(|s| s.stencil_func = v);
}
unsafe extern "C" fn stencil_pass_op(_t: *mut c_void, v: i32) {
    set(|s| s.stencil_pass = v);
}
unsafe extern "C" fn stencil_fail_op(_t: *mut c_void, v: i32) {
    set(|s| s.stencil_fail = v);
}
unsafe extern "C" fn stencil_depth_fail_op(_t: *mut c_void, v: i32) {
    set(|s| s.stencil_depth_fail = v);
}
unsafe extern "C" fn stencil_reference(_t: *mut c_void, v: i32) {
    set(|s| s.stencil_ref = v);
}
unsafe extern "C" fn stencil_mask(_t: *mut c_void, v: i32) {
    set(|s| s.stencil_mask = v);
}
unsafe extern "C" fn stencil_write_mask(_t: *mut c_void, v: i32) {
    set(|s| s.stencil_write_mask = v);
}
unsafe extern "C" fn enable_color_writes(_t: *mut c_void, v: bool) {
    set(|s| s.color_write = v);
}
unsafe extern "C" fn enable_alpha_writes(_t: *mut c_void, v: bool) {
    set(|s| s.alpha_write = v);
}
unsafe extern "C" fn enable_blending(_t: *mut c_void, v: bool) {
    set(|s| s.blend = v);
}
unsafe extern "C" fn blend_func(_t: *mut c_void, src: i32, dst: i32) {
    set(|s| {
        s.blend_src = src;
        s.blend_dst = dst;
    });
}
unsafe extern "C" fn enable_alpha_test(_t: *mut c_void, v: bool) {
    set(|s| s.alpha_test = v);
}
unsafe extern "C" fn alpha_func(_t: *mut c_void, f: i32, r: f32) {
    set(|s| {
        s.alpha_func = f;
        s.alpha_ref_bits = r.to_bits();
    });
}
unsafe extern "C" fn enable_culling(_t: *mut c_void, v: bool) {
    set(|s| s.cull = v);
}
unsafe extern "C" fn vertex_shader_vertex_format(_t: *mut c_void, flags: u32, texcoord_count: i32, dims: *const i32, user_data: i32) {
    let flags = (flags as u64 & !VERTEX_BONE_INDEX) | VERTEX_FORMAT_VERTEX_SHADER;
    let n = texcoord_count.clamp(0, MAX_TEXCOORDS as i32) as usize;
    let dims = (!dims.is_null()).then(|| unsafe { std::slice::from_raw_parts(dims, n) });
    let mut fmt = compute_vertex_format(flags, n, dims, 0, user_data.max(0) as usize);
    if vertex_size(fmt) <= 16 {
        // dx9 pads narrow formats with one 2d texcoord
        fmt = compute_vertex_format(flags | VERTEX_POSITION, 1, None, 0, user_data.max(0) as usize);
    }
    set(|s| s.vertex_format = fmt);
}
unsafe extern "C" fn set_vertex_shader(_t: *mut c_void, name: *const c_char, static_index: i32) {
    let name = cstr(name);
    set(|s| {
        s.vertex_shader = name.map(|n| ShaderName::intern(&n));
        s.vertex_shader_static = static_index;
    });
}
unsafe extern "C" fn set_pixel_shader(_t: *mut c_void, name: *const c_char, static_index: i32) {
    let name = cstr(name);
    set(|s| {
        s.pixel_shader = name.map(|n| ShaderName::intern(&n));
        s.pixel_shader_static = static_index;
    });
}
unsafe extern "C" fn enable_srgb_write(_t: *mut c_void, v: bool) {
    set(|s| s.srgb_write = v);
}
unsafe extern "C" fn enable_srgb_read(_t: *mut c_void, sampler: i32, v: bool) {
    set(|s| s.srgb_read = bit(s.srgb_read, sampler, v));
}
unsafe extern "C" fn enable_texture(_t: *mut c_void, sampler: i32, v: bool) {
    set(|s| s.textures_enabled = bit(s.textures_enabled, sampler, v));
}
unsafe extern "C" fn enable_blending_separate_alpha(_t: *mut c_void, v: bool) {
    set(|s| s.blend_separate_alpha = v);
}
unsafe extern "C" fn blend_func_separate_alpha(_t: *mut c_void, src: i32, dst: i32) {
    set(|s| {
        s.blend_src_alpha = src;
        s.blend_dst_alpha = dst;
    });
}
unsafe extern "C" fn fog_mode(_t: *mut c_void, v: i32) {
    set(|s| s.fog_mode = v);
}
unsafe extern "C" fn set_morph_format(_t: *mut c_void, v: u64) {
    set(|s| s.morph_format = v);
}
unsafe extern "C" fn disable_fog_gamma_correction(_t: *mut c_void, v: bool) {
    set(|s| s.disable_fog_gamma = v);
}
unsafe extern "C" fn enable_alpha_to_coverage(_t: *mut c_void, v: bool) {
    set(|s| s.alpha_to_coverage = v);
}
unsafe extern "C" fn set_shadow_depth_filtering(_t: *mut c_void, sampler: i32) {
    set(|s| s.shadow_filtering = bit(s.shadow_filtering, sampler, true));
}
unsafe extern "C" fn blend_op(_t: *mut c_void, v: i32) {
    set(|s| s.blend_op = v);
}
unsafe extern "C" fn blend_op_separate_alpha(_t: *mut c_void, v: i32) {
    set(|s| s.blend_op_alpha = v);
}
/// fixed-function calls, nothing to record
unsafe extern "C" fn ignore(_t: *mut c_void) {}

fn bit(bits: u32, i: i32, on: bool) -> u32 {
    if !(0..32).contains(&i) {
        return bits;
    }
    if on { bits | (1 << i) } else { bits & !(1 << i) }
}

pub fn interface() -> crate::Interface {
    use crate::vtables::shader_shadow::{slot as s, stubs};
    let f = |p: *const ()| p as usize;
    crate::Interface::new(
        stubs(),
        &[
            (s::SetDefaultState, f(set_default_state as *const ())),
            (s::DepthFunc, f(depth_func as *const ())),
            (s::EnableDepthWrites, f(enable_depth_writes as *const ())),
            (s::EnableDepthTest, f(enable_depth_test as *const ())),
            (s::EnablePolyOffset, f(enable_poly_offset as *const ())),
            (s::EnableStencil, f(enable_stencil as *const ())),
            (s::StencilFunc, f(stencil_func as *const ())),
            (s::StencilPassOp, f(stencil_pass_op as *const ())),
            (s::StencilFailOp, f(stencil_fail_op as *const ())),
            (s::StencilDepthFailOp, f(stencil_depth_fail_op as *const ())),
            (s::StencilReference, f(stencil_reference as *const ())),
            (s::StencilMask, f(stencil_mask as *const ())),
            (s::StencilWriteMask, f(stencil_write_mask as *const ())),
            (s::EnableColorWrites, f(enable_color_writes as *const ())),
            (s::EnableAlphaWrites, f(enable_alpha_writes as *const ())),
            (s::EnableBlending, f(enable_blending as *const ())),
            (s::BlendFunc, f(blend_func as *const ())),
            (s::EnableAlphaTest, f(enable_alpha_test as *const ())),
            (s::AlphaFunc, f(alpha_func as *const ())),
            (s::PolyMode, f(ignore as *const ())),
            (s::EnableCulling, f(enable_culling as *const ())),
            (s::EnableConstantColor, f(ignore as *const ())),
            (s::VertexShaderVertexFormat, f(vertex_shader_vertex_format as *const ())),
            (s::PolyMode, f(poly_mode as *const ())),
            (s::SetVertexShader, f(set_vertex_shader as *const ())),
            (s::SetPixelShader, f(set_pixel_shader as *const ())),
            (s::EnableLighting, f(ignore as *const ())),
            (s::EnableSpecular, f(ignore as *const ())),
            (s::EnableSRGBWrite, f(enable_srgb_write as *const ())),
            (s::EnableSRGBRead, f(enable_srgb_read as *const ())),
            (s::EnableVertexBlend, f(ignore as *const ())),
            (s::OverbrightValue, f(ignore as *const ())),
            (s::EnableTexture, f(enable_texture as *const ())),
            (s::EnableTexGen, f(ignore as *const ())),
            (s::TexGen, f(ignore as *const ())),
            (s::EnableCustomPixelPipe, f(ignore as *const ())),
            (s::CustomTextureStages, f(ignore as *const ())),
            (s::CustomTextureOperation, f(ignore as *const ())),
            (s::DrawFlags, f(ignore as *const ())),
            (s::EnableAlphaPipe, f(ignore as *const ())),
            (s::EnableConstantAlpha, f(ignore as *const ())),
            (s::EnableVertexAlpha, f(ignore as *const ())),
            (s::EnableTextureAlpha, f(ignore as *const ())),
            (s::EnableBlendingSeparateAlpha, f(enable_blending_separate_alpha as *const ())),
            (s::BlendFuncSeparateAlpha, f(blend_func_separate_alpha as *const ())),
            (s::FogMode, f(fog_mode as *const ())),
            (s::SetDiffuseMaterialSource, f(ignore as *const ())),
            (s::SetMorphFormat, f(set_morph_format as *const ())),
            (s::DisableFogGammaCorrection, f(disable_fog_gamma_correction as *const ())),
            (s::EnableAlphaToCoverage, f(enable_alpha_to_coverage as *const ())),
            (s::SetShadowDepthFiltering, f(set_shadow_depth_filtering as *const ())),
            (s::BlendOp, f(blend_op as *const ())),
            (s::BlendOpSeparateAlpha, f(blend_op_separate_alpha as *const ())),
        ],
    )
}
