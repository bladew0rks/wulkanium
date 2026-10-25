//! the shader api mutex, reentrant. the main thread holds it across
//! ShaderLock / ShaderUnlock so the render thread waits.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

use std::num::NonZeroUsize;

use lock_api::{GetThreadId, RawReentrantMutex};
use parking_lot::RawMutex;

/// os thread id; parking_lot's own goes through a thread_local, which is slow on windows
pub struct OsThreadId;

unsafe impl GetThreadId for OsThreadId {
    const INIT: Self = OsThreadId;
    fn nonzero_thread_id(&self) -> NonZeroUsize {
        #[cfg(windows)]
        let id = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() } as usize;
        #[cfg(not(windows))]
        let id = {
            thread_local!(static K: u8 = const { 0 });
            K.with(|k| k as *const u8 as usize)
        };
        NonZeroUsize::new(id).expect("thread id is never 0")
    }
}

static MUTEX: RawReentrantMutex<RawMutex, OsThreadId> = RawReentrantMutex::INIT;
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

/// ns spent waiting for the mutex while another thread held it (api calls / ShaderLock)
pub static WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LOCK_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// lock, timing only the contended case
fn lock_timed(counter: &std::sync::atomic::AtomicU64) {
    if !MUTEX.try_lock() {
        let t = std::time::Instant::now();
        MUTEX.lock();
        counter.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

pub fn guard() -> Guard {
    if ENABLED.load(Ordering::SeqCst) {
        lock_timed(&WAIT_NS);
        Guard(true)
    } else {
        Guard(false)
    }
}

pub fn shader_lock() {
    if ENABLED.load(Ordering::SeqCst) {
        lock_timed(&LOCK_WAIT_NS);
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
