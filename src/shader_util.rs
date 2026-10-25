//! calls back into the material system's IShaderUtil, like dx9 does

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

static UTIL: AtomicUsize = AtomicUsize::new(0);

// IAppSystem has 9 slots, then IShaderUtil
const BIND_STANDARD_TEXTURE: usize = 13;
const ON_DRAW_MESH_PRIMS: usize = 24;
const ON_DRAW_MESH: usize = 25;
const SYNC_MATRICES: usize = 30;
const IN_FLASHLIGHT_MODE: usize = 18;
/// checked against CMaterialSystem's vtable
const DRAW_CLEAR_BUFFER_QUAD: usize = 23;
/// gmod has no ReadBackBuffer, so slots after DrawClearBufferQuad are one lower than sdk2013
const GET_CURRENT_COLOR_CORRECTION: usize = 35;

pub fn set(util: usize) {
    UTIL.store(util, Ordering::SeqCst);
}

fn slot(i: usize) -> Option<(usize, usize)> {
    let u = UTIL.load(Ordering::SeqCst);
    (u != 0).then(|| (u, unsafe { *(*(u as *const *const usize)).add(i) }))
}

/// false = don't draw
pub fn on_draw_mesh(mesh: *mut c_void, first: i32, count: i32) -> bool {
    slot(ON_DRAW_MESH).is_none_or(|(u, f)| unsafe {
        let f: unsafe extern "C" fn(usize, *mut c_void, i32, i32) -> bool = std::mem::transmute(f);
        f(u, mesh, first, count)
    })
}

pub fn on_draw_mesh_prims(mesh: *mut c_void, prims: *const c_void, n: i32) -> bool {
    slot(ON_DRAW_MESH_PRIMS).is_none_or(|(u, f)| unsafe {
        let f: unsafe extern "C" fn(usize, *mut c_void, *const c_void, i32) -> bool = std::mem::transmute(f);
        f(u, mesh, prims, n)
    })
}

pub fn sync_matrices() {
    if let Some((u, f)) = slot(SYNC_MATRICES) {
        unsafe {
            let f: unsafe extern "C" fn(usize) = std::mem::transmute(f);
            f(u)
        }
    }
}

pub fn bind_standard_texture(sampler: i32, id: i32) {
    if let Some((u, f)) = slot(BIND_STANDARD_TEXTURE) {
        unsafe {
            let f: unsafe extern "C" fn(usize, i32, i32) = std::mem::transmute(f);
            f(u, sampler, id)
        }
    }
}

/// draws a quad with the clear material, so it obeys the current stencil state
#[allow(clippy::too_many_arguments)]
pub fn draw_clear_buffer_quad(r: u8, g: u8, b: u8, a: u8, color: bool, alpha: bool, depth: bool) {
    if let Some((u, f)) = slot(DRAW_CLEAR_BUFFER_QUAD) {
        unsafe {
            let f: unsafe extern "C" fn(usize, u8, u8, u8, u8, bool, bool, bool) = std::mem::transmute(f);
            f(u, r, g, b, a, color, alpha, depth)
        }
    }
}

pub fn in_flashlight_mode() -> bool {
    slot(IN_FLASHLIGHT_MODE).is_some_and(|(u, f)| unsafe {
        let f: unsafe extern "C" fn(usize) -> bool = std::mem::transmute(f);
        f(u)
    })
}

/// the material system fills in the active lookups. zeroed first (64 bytes on x64), so nothing
/// filled means disabled.
pub fn get_current_color_correction(info: *mut c_void) {
    if info.is_null() {
        return;
    }
    unsafe { std::ptr::write_bytes(info as *mut u8, 0, 64) };
    if let Some((u, f)) = slot(GET_CURRENT_COLOR_CORRECTION) {
        unsafe {
            let f: unsafe extern "C" fn(usize, *mut c_void) = std::mem::transmute(f);
            f(u, info)
        }
    }
}
