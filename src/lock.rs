//! the shader api mutex, reentrant. the main thread holds it across
//! ShaderLock / ShaderUnlock so the render thread waits.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

use lock_api::RawReentrantMutex;
use parking_lot::{RawMutex, RawThreadId};

static MUTEX: RawReentrantMutex<RawMutex, RawThreadId> = RawReentrantMutex::INIT;
static ENABLED: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// locks taken through ShaderLock, so ShaderUnlock releases only those
    static HELD: Cell<u32> = const { Cell::new(0) };
}

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::SeqCst);
}

/// no-op while the mutex is disabled
pub struct Guard(bool);

impl Drop for Guard {
    fn drop(&mut self) {
        if self.0 {
            unsafe { MUTEX.unlock() };
        }
    }
}

pub fn guard() -> Guard {
    if ENABLED.load(Ordering::SeqCst) {
        MUTEX.lock();
        Guard(true)
    } else {
        Guard(false)
    }
}

pub fn shader_lock() {
    if ENABLED.load(Ordering::SeqCst) {
        MUTEX.lock();
        HELD.with(|h| h.set(h.get() + 1));
    }
}

pub fn shader_unlock() {
    HELD.with(|h| {
        if h.get() > 0 {
            h.set(h.get() - 1);
            unsafe { MUTEX.unlock() };
        }
    });
}
