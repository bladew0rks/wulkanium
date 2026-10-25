//! `IShaderDeviceMgr` and `IShaderDevice`

use std::ffi::{c_char, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::abi::*;
use crate::log;
use crate::mesh::Mesh;

#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub hwnd: usize,
    pub info: ShaderDeviceInfo,
}

pub static WINDOW: Mutex<Option<Window>> = Mutex::new(None);
static FACTORY: Mutex<usize> = Mutex::new(0);

/// dx 9.5 / sm3: the material system loads stdshader_dx9
pub const DX_LEVEL: i32 = 95;

/// msaa samples from mat_antialias (1 = off)
pub fn aa_samples() -> u32 {
    WINDOW.lock().unwrap_or_else(|e| e.into_inner()).map_or(1, |w| w.info.aa_samples.max(1) as u32)
}

pub unsafe extern "C" fn is_aa_enabled(_t: *mut c_void) -> bool {
    aa_samples() > 1
}

pub fn back_buffer_size() -> (i32, i32) {
    let w = *WINDOW.lock().unwrap_or_else(|e| e.into_inner());
    #[cfg(windows)]
    if let Some(w) = w.filter(|w| w.hwnd != 0) {
        use windows_sys::Win32::Foundation::RECT;
        use windows_sys::Win32::UI::WindowsAndMessaging::GetClientRect;
        let mut r = RECT { left: 0, top: 0, right: 0, bottom: 0 };
        if unsafe { GetClientRect(w.hwnd as _, &mut r) } != 0 && r.right > r.left && r.bottom > r.top {
            return (r.right - r.left, r.bottom - r.top);
        }
    }
    if let Some(w) = w.filter(|w| w.info.display_mode.width > 0) {
        return (w.info.display_mode.width, w.info.display_mode.height);
    }
    desktop_size()
}

pub fn desktop_size() -> (i32, i32) {
    #[cfg(windows)]
    {
        use windows_sys::Win32::UI::WindowsAndMessaging::{GetSystemMetrics, SM_CXSCREEN, SM_CYSCREEN};
        let (w, h) = unsafe { (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)) };
        if w > 0 && h > 0 {
            return (w, h);
        }
    }
    (1920, 1080)
}


type CreateInterfaceFn = unsafe extern "C" fn(*const c_char, *mut i32) -> *mut c_void;

unsafe extern "C" fn mgr_connect(_t: *mut c_void, factory: usize) -> bool {
    *FACTORY.lock().unwrap_or_else(|e| e.into_inner()) = factory;
    // the material system's IShaderUtil; dx9 calls back into it on every draw
    if factory != 0 {
        let f: CreateInterfaceFn = unsafe { std::mem::transmute(factory) };
        let util = unsafe { f(c"VShaderUtil001".as_ptr(), std::ptr::null_mut()) };
        crate::shader_util::set(util as usize);
        log::line(&format!("ShaderDeviceMgr::Connect (VShaderUtil001 {})", if util.is_null() { "missing" } else { "ok" }));
    }
    true
}
unsafe extern "C" fn mgr_disconnect(_t: *mut c_void) {}
unsafe extern "C" fn mgr_query_interface(_t: *mut c_void, name: *const c_char) -> *mut c_void {
    let name = if name.is_null() { String::new() } else { unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy().into_owned() };
    let r = crate::find_interface(&name).unwrap_or(std::ptr::null_mut());
    log::line(&format!("ShaderDeviceMgr::QueryInterface({name}) -> {}", if r.is_null() { "null" } else { "ok" }));
    r
}
unsafe extern "C" fn mgr_init(_t: *mut c_void) -> i32 {
    log::line("ShaderDeviceMgr::Init");
    1 // INIT_OK
}
unsafe extern "C" fn mgr_shutdown(_t: *mut c_void) {
    log::line("ShaderDeviceMgr::Shutdown");
    crate::render::shutdown();
}
unsafe extern "C" fn mgr_get_dependencies(_t: *mut c_void) -> usize {
    0
}
unsafe extern "C" fn mgr_get_tier(_t: *mut c_void) -> i32 {
    4 // APP_SYSTEM_TIER_OTHER
}
unsafe extern "C" fn mgr_reconnect(_t: *mut c_void, factory: usize, _name: *const c_char) {
    *FACTORY.lock().unwrap_or_else(|e| e.into_inner()) = factory;
}
unsafe extern "C" fn yes(_t: *mut c_void) -> bool {
    true
}
unsafe extern "C" fn mgr_adapter_count(_t: *mut c_void) -> i32 {
    1
}
unsafe extern "C" fn mgr_adapter_info(_t: *mut c_void, adapter: i32, info: *mut MaterialAdapterInfo) {
    let info = unsafe { &mut *info };
    *info = unsafe { std::mem::zeroed() };
    let name = b"wgpu (shaderapiwgpu)";
    info.driver_name[..name.len()].copy_from_slice(name);
    info.dx_support_level = DX_LEVEL;
    info.max_dx_support_level = DX_LEVEL;
    log::line(&format!("ShaderDeviceMgr::GetAdapterInfo({adapter})"));
}
unsafe extern "C" fn mgr_recommended_config(_t: *mut c_void, _adapter: i32, _dx: i32, _kv: *mut c_void) -> bool {
    // filling this needs tier1's KeyValues; dxsupport.cfg defaults apply instead
    true
}
/// common sizes up to the desktop, plus the desktop
fn modes() -> Vec<(i32, i32)> {
    let (dw, dh) = desktop_size();
    let mut m: Vec<(i32, i32)> = [(800, 600), (1024, 768), (1280, 720), (1280, 1024), (1366, 768), (1600, 900), (1920, 1080), (2560, 1440), (3840, 2160)]
        .into_iter()
        .filter(|&(w, h)| w <= dw && h <= dh)
        .collect();
    if !m.contains(&(dw, dh)) {
        m.push((dw, dh));
    }
    m
}
fn fill_mode(p: *mut ShaderDisplayMode, (w, h): (i32, i32)) {
    if p.is_null() {
        return;
    }
    unsafe {
        *p = ShaderDisplayMode { version: 1, width: w, height: h, format: IMAGE_FORMAT_BGRA8888, refresh_num: 60, refresh_den: 1 };
    }
}
unsafe extern "C" fn mgr_mode_count(_t: *mut c_void, _adapter: i32) -> i32 {
    modes().len() as i32
}
unsafe extern "C" fn mgr_mode_info(_t: *mut c_void, info: *mut ShaderDisplayMode, _adapter: i32, mode: i32) {
    let m = modes();
    fill_mode(info, m.get(mode.max(0) as usize).copied().unwrap_or(*m.last().unwrap()));
}
unsafe extern "C" fn mgr_current_mode_info(_t: *mut c_void, info: *mut ShaderDisplayMode, _adapter: i32) {
    fill_mode(info, desktop_size());
}
unsafe extern "C" fn mgr_set_adapter(_t: *mut c_void, adapter: i32, flags: i32) -> bool {
    log::line(&format!("ShaderDeviceMgr::SetAdapter({adapter}, {flags:#x})"));
    true
}
pub fn set_mode(hwnd: usize, info: &ShaderDeviceInfo) {
    *WINDOW.lock().unwrap_or_else(|e| e.into_inner()) = Some(Window { hwnd, info: *info });
}

pub fn change_video_mode(info: &ShaderDeviceInfo) {
    let mut w = WINDOW.lock().unwrap_or_else(|e| e.into_inner());
    let hwnd = w.map_or(0, |w| w.hwnd);
    *w = Some(Window { hwnd, info: *info });
}

unsafe extern "C" fn mgr_set_mode(_t: *mut c_void, hwnd: usize, adapter: i32, info: *const ShaderDeviceInfo) -> CreateInterfaceFn {
    let info = unsafe { *info };
    log::line(&format!("ShaderDeviceMgr::SetMode(hwnd {hwnd:#x}, adapter {adapter}, {info:?})"));
    set_mode(hwnd, &info);
    crate::CreateInterface
}
unsafe extern "C" fn ignore(_t: *mut c_void) {}

pub fn manager_interface() -> crate::Interface {
    use crate::vtables::shader_device_mgr::{slot as s, stubs};
    let f = |p: *const ()| p as usize;
    crate::Interface::new(
        stubs(),
        &[
            (s::Connect, f(mgr_connect as *const ())),
            (s::Disconnect, f(mgr_disconnect as *const ())),
            (s::QueryInterface, f(mgr_query_interface as *const ())),
            (s::Init, f(mgr_init as *const ())),
            (s::Shutdown, f(mgr_shutdown as *const ())),
            (s::GetDependencies, f(mgr_get_dependencies as *const ())),
            (s::GetTier, f(mgr_get_tier as *const ())),
            (s::Reconnect, f(mgr_reconnect as *const ())),
            (s::IsSingleton, f(yes as *const ())),
            (s::GetAdapterCount, f(mgr_adapter_count as *const ())),
            (s::GetAdapterInfo, f(mgr_adapter_info as *const ())),
            (s::GetRecommendedConfigurationInfo, f(mgr_recommended_config as *const ())),
            (s::GetModeCount, f(mgr_mode_count as *const ())),
            (s::GetModeInfo, f(mgr_mode_info as *const ())),
            (s::GetCurrentModeInfo, f(mgr_current_mode_info as *const ())),
            (s::SetAdapter, f(mgr_set_adapter as *const ())),
            (s::SetMode, f(mgr_set_mode as *const ())),
            (s::AddModeChangeCallback, f(ignore as *const ())),
            (s::RemoveModeChangeCallback, f(ignore as *const ())),
        ],
    )
}


static NEXT_SHADER: AtomicU32 = AtomicU32::new(1);

unsafe extern "C" fn dev_back_buffer_format(_t: *mut c_void) -> i32 {
    IMAGE_FORMAT_BGRA8888
}
unsafe extern "C" fn dev_back_buffer_dims(_t: *mut c_void, w: *mut i32, h: *mut i32) {
    let (bw, bh) = back_buffer_size();
    unsafe {
        *w = bw;
        *h = bh;
    }
}
unsafe extern "C" fn zero(_t: *mut c_void) -> usize {
    0
}
unsafe extern "C" fn dev_stencil_bits(_t: *mut c_void) -> i32 {
    8
}
unsafe extern "C" fn dev_present(_t: *mut c_void) {
    crate::api::present();
}
unsafe extern "C" fn dev_create_shader(_t: *mut c_void, _buf: *mut c_void) -> usize {
    NEXT_SHADER.fetch_add(1, Ordering::Relaxed) as usize
}
unsafe extern "C" fn dev_create_static_mesh(_t: *mut c_void, fmt: u64, _group: *const c_char, _material: *mut c_void) -> *mut c_void {
    Box::leak(Mesh::new(fmt, false)).as_ptr()
}
unsafe extern "C" fn dev_destroy_static_mesh(_t: *mut c_void, mesh: *mut c_void) {
    if !mesh.is_null() {
        drop(unsafe { Box::from_raw(mesh as *mut Mesh) });
    }
}
fn is_static(kind: i32) -> bool {
    kind == 0 || kind == 2
}
unsafe extern "C" fn dev_create_vertex_buffer(_t: *mut c_void, kind: i32, fmt: u64, _count: i32, _group: *const c_char) -> *mut c_void {
    Box::leak(Mesh::new(fmt, !is_static(kind))).as_ptr()
}
unsafe extern "C" fn dev_destroy_vertex_buffer(_t: *mut c_void, vb: *mut c_void) {
    if !vb.is_null() {
        drop(unsafe { Box::from_raw(vb as *mut Mesh) });
    }
}
unsafe extern "C" fn dev_create_index_buffer(_t: *mut c_void, kind: i32, _fmt: i32, _count: i32, _group: *const c_char) -> *mut c_void {
    Box::leak(Mesh::new(0, !is_static(kind))).as_index_buffer()
}
unsafe extern "C" fn dev_destroy_index_buffer(_t: *mut c_void, ib: *mut c_void) {
    if !ib.is_null() {
        drop(unsafe { Box::from_raw((ib as *mut u8).sub(8) as *mut Mesh) });
    }
}
unsafe extern "C" fn dev_dynamic_vertex_buffer(_t: *mut c_void, _stream: i32, fmt: u64, _buffered: bool) -> *mut c_void {
    let m = crate::api::dynamic_vertex_buffer();
    m.set_format(fmt);
    m.as_ptr()
}
unsafe extern "C" fn dev_dynamic_index_buffer(_t: *mut c_void, _fmt: i32, _buffered: bool) -> *mut c_void {
    crate::api::dynamic_index_buffer().as_index_buffer()
}

pub fn device_interface() -> crate::Interface {
    use crate::vtables::shader_device::{slot as s, stubs};
    let f = |p: *const ()| p as usize;
    crate::Interface::new(
        stubs(),
        &[
            (s::ReleaseResources, f(ignore as *const ())),
            (s::ReacquireResources, f(ignore as *const ())),
            (s::GetBackBufferFormat, f(dev_back_buffer_format as *const ())),
            (s::GetBackBufferDimensions, f(dev_back_buffer_dims as *const ())),
            (s::GetCurrentAdapter, f(zero as *const ())),
            (s::IsUsingGraphics, f(yes as *const ())),
            (s::SpewDriverInfo, f(ignore as *const ())),
            (s::StencilBufferBits, f(dev_stencil_bits as *const ())),
            (s::IsAAEnabled, f(is_aa_enabled as *const ())),
            (s::Present, f(dev_present as *const ())),
            (s::GetWindowSize, f(dev_back_buffer_dims as *const ())),
            (s::SetHardwareGammaRamp, f(ignore as *const ())),
            (s::AddView, f(yes as *const ())),
            (s::RemoveView, f(ignore as *const ())),
            (s::SetView, f(ignore as *const ())),
            (s::CompileShader, f(zero as *const ())),
            (s::CreateVertexShader, f(dev_create_shader as *const ())),
            (s::DestroyVertexShader, f(ignore as *const ())),
            (s::CreateGeometryShader, f(dev_create_shader as *const ())),
            (s::DestroyGeometryShader, f(ignore as *const ())),
            (s::CreatePixelShader, f(dev_create_shader as *const ())),
            (s::DestroyPixelShader, f(ignore as *const ())),
            (s::CreateStaticMesh, f(dev_create_static_mesh as *const ())),
            (s::DestroyStaticMesh, f(dev_destroy_static_mesh as *const ())),
            (s::CreateVertexBuffer, f(dev_create_vertex_buffer as *const ())),
            (s::DestroyVertexBuffer, f(dev_destroy_vertex_buffer as *const ())),
            (s::CreateIndexBuffer, f(dev_create_index_buffer as *const ())),
            (s::DestroyIndexBuffer, f(dev_destroy_index_buffer as *const ())),
            (s::GetDynamicVertexBuffer, f(dev_dynamic_vertex_buffer as *const ())),
            (s::GetDynamicIndexBuffer, f(dev_dynamic_index_buffer as *const ())),
            (s::EnableNonInteractiveMode, f(ignore as *const ())),
            (s::RefreshFrontBufferNonInteractive, f(ignore as *const ())),
            (s::HandleThreadEvent, f(ignore as *const ())),
        ],
    )
}
