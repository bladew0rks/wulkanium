//! the video options' "Hardware DirectX level" names come from GameUI's format strings; once
//! GameUI.dll is loaded they're rewritten to name the wgpu backend instead

use std::sync::OnceLock;
#[cfg(windows)]
use std::sync::atomic::{AtomicBool, Ordering};

/// set by the renderer once it picked an adapter
pub static BACKEND: OnceLock<String> = OnceLock::new();
#[cfg(windows)]
static DONE: AtomicBool = AtomicBool::new(false);

/// game thread, once a frame until it has patched
#[cfg(not(windows))]
pub fn tick() {}

/// game thread, once a frame until it has patched
#[cfg(windows)]
pub fn tick() {
    if DONE.load(Ordering::Relaxed) {
        return;
    }
    let Some(backend) = BACKEND.get() else { return };
    let base = unsafe { windows_sys::Win32::System::LibraryLoader::GetModuleHandleA(c"GameUI.dll".as_ptr() as *const u8) } as usize;
    if base == 0 {
        return;
    }
    DONE.store(true, Ordering::Relaxed);
    let name = format!("wgpu ({backend})");
    let image = unsafe { module_image(base) };
    let mut n = 0;
    for needle in [&b"DirectX v%.1f\0"[..], &b"DirectX v9.0+\0"[..]] {
        if let Some(at) = image.windows(needle.len()).position(|w| w == needle) {
            unsafe { overwrite(base + at, needle.len(), name.as_bytes()) };
            n += 1;
        }
    }
    crate::log::line(&format!("gameui: dx level names -> '{name}' ({n} of 2 strings)"));
}

#[cfg(windows)]
/// the whole mapped image, from the pe header's SizeOfImage
unsafe fn module_image(base: usize) -> &'static [u8] {
    unsafe {
        let pe = base + *((base + 0x3c) as *const u32) as usize;
        let size = *((pe + 0x18 + 0x38) as *const u32) as usize;
        std::slice::from_raw_parts(base as *const u8, size)
    }
}

#[cfg(windows)]
/// writes `text` (cut to fit, nul padded) over a `len` byte string
unsafe fn overwrite(at: usize, len: usize, text: &[u8]) {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let mut bytes = vec![0u8; len];
    let n = text.len().min(len - 1);
    bytes[..n].copy_from_slice(&text[..n]);
    unsafe {
        let mut old = 0;
        if VirtualProtect(at as *const _, len, PAGE_READWRITE, &mut old) == 0 {
            return;
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), at as *mut u8, len);
        VirtualProtect(at as *const _, len, old, &mut old);
    }
}
