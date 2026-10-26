//! drop-in shaderapi dll for garry's mod x64, rendering through wgpu. vtable layouts come from
//! layout/gmod_x64_vtables.json; every slot starts as a logging stub and real implementations
//! replace them.

use std::ffi::{CStr, c_char, c_void};
use std::sync::OnceLock;

pub mod abi;
pub mod builtin;
pub mod cvar;
pub mod api;
pub mod device;
pub mod hwconfig;
pub mod lock;
pub mod log;
pub mod mesh;
pub mod render;
pub mod sampler;
pub mod shader_util;
pub mod shaderdll;
pub mod shadow;
pub mod stock;
pub mod texconv;
pub mod texlist;
pub mod vcs;
pub mod d3dbc;
pub mod d3d2wgsl;
pub mod shaders;
pub mod vpk;
pub mod overlay;
pub mod dlss;
pub mod effects;
pub mod sss;
pub mod gameui;
pub mod vtables;
pub mod vtf76;

#[repr(C)]
pub struct Interface {
    vtable: *const usize,
}

unsafe impl Sync for Interface {}
unsafe impl Send for Interface {}

impl Interface {
    /// vtable = `stubs` with the given overrides
    pub fn new<const N: usize>(mut stubs: [usize; N], overrides: &[(usize, usize)]) -> Interface {
        for &(slot, f) in overrides {
            stubs[slot] = f;
        }
        // leaked: vtables live for the whole process
        let table: &'static [usize; N] = Box::leak(Box::new(stubs));
        Interface { vtable: table.as_ptr() }
    }

    pub fn as_ptr(&'static self) -> *mut c_void {
        self as *const Interface as *mut c_void
    }

    pub fn vtable(&self) -> *const usize {
        self.vtable
    }
}

const IFACE_OK: i32 = 0;
const IFACE_FAILED: i32 = 1;

fn interfaces() -> &'static [(&'static str, Interface)] {
    static ALL: OnceLock<Vec<(&'static str, Interface)>> = OnceLock::new();
    ALL.get_or_init(|| {
        vec![
            ("ShaderDeviceMgr001", device::manager_interface()),
            ("ShaderApi030", api::interface()),
            ("ShaderDevice001", device::device_interface()),
            ("ShaderShadow010", shadow::interface()),
            ("MaterialSystemHardwareConfig012", hwconfig::interface()),
            ("DebugTextureInfo001", api::debug_texture_info_interface()),
            // materialsystem loads this dll again as a shader dll (see hwconfig)
            ("ShaderDLL004", shaderdll::dll_interface()),
        ]
    })
}

pub fn find_interface(name: &str) -> Option<*mut c_void> {
    interfaces().iter().find(|(n, _)| *n == name).map(|(_, i)| i.as_ptr())
}

/// tier1 factory exported by every source module.
///
/// # Safety
/// `name` must be nul-terminated; `return_code` null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn CreateInterface(name: *const c_char, return_code: *mut i32) -> *mut c_void {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        // a panic in an extern "C" fn aborts the game, so log it first
        std::panic::set_hook(Box::new(|info| log::line(&format!("PANIC: {info}"))));
        log::line(&format!("--- shaderapiwgpu {} loaded (build {}) ---", env!("CARGO_PKG_VERSION"), env!("BUILD_UNIX_TIME")));
        // materialsystem (which loads us) reads vtf 7.5 at most
        vtf76::install();
    });
    let name = if name.is_null() { "" } else { unsafe { CStr::from_ptr(name) }.to_str().unwrap_or("") };
    let found = find_interface(name);
    log::line(&format!("CreateInterface({name}) -> {}", if found.is_some() { "ok" } else { "not found" }));
    if !return_code.is_null() {
        unsafe { *return_code = if found.is_some() { IFACE_OK } else { IFACE_FAILED } };
    }
    found.unwrap_or(std::ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtables::*;

    #[test]
    fn layouts_have_the_slot_counts_of_gmods_binaries() {
        assert_eq!(shader_api::LEN, 258);
        assert_eq!(shader_shadow::LEN, 52);
        assert_eq!(shader_device::LEN, 33);
        assert_eq!(shader_device_mgr::LEN, 19);
        assert_eq!(hardware_config::LEN, 74);
        assert_eq!(vtables::mesh::LEN, 28);
        assert_eq!(mesh_index_buffer::LEN, 13);
        // anchors checked against materialsystem.dll and shaderapiempty.dll
        assert_eq!(shader_api::slot::MatrixMode, 6);
        assert_eq!(shader_api::slot::GMOD_SamplerBorderClamp, 99);
        assert_eq!(shader_api::slot::ClearBuffers, 100);
        assert_eq!(shader_api::slot::Draw, 230);
        assert_eq!(shader_api::slot::OverrideBlendSeparateAlpha, 256);
        assert_eq!(shader_device_mgr::slot::SetMode, 16);
        assert_eq!(shader_device::slot::CreateStaticMesh, 22);
        assert_eq!(hardware_config::slot::GetDXSupportLevel, 32);
    }

    #[test]
    fn factory_serves_interfaces_whose_vtables_hold_logging_stubs() {
        let mut rc = -1;
        let p = unsafe { CreateInterface(c"ShaderApi030".as_ptr(), &mut rc) };
        assert!(!p.is_null() && rc == IFACE_OK);
        assert!(unsafe { CreateInterface(c"Nope001".as_ptr(), &mut rc) }.is_null() && rc == IFACE_FAILED);
        // call a stub slot the way c++ would
        let vt = unsafe { *(p as *const *const usize) };
        let f: unsafe extern "C" fn(*mut c_void) -> usize = unsafe { std::mem::transmute(*vt.add(shader_api::slot::Rotate)) };
        assert_eq!(unsafe { f(p) }, 0);
        assert!(log::counts(shader_api::IFACE).contains(&(shader_api::slot::Rotate, 1)));
        // reference-returning stubs hand out zeroed memory
        let r: unsafe extern "C" fn(*mut c_void) -> *const u8 = unsafe { std::mem::transmute(shader_api::stubs()[shader_api::slot::GetToneMappingScaleLinear]) };
        assert_eq!(unsafe { *r(p) }, 0);
    }
}
