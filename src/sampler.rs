//! sampling profiler (Shift+Home): records the instruction pointer of the game's drawing thread
//! and of our render thread about a thousand times a second, then logs where they
//! were, per module and per address.

/// threads the sampler follows
#[derive(Clone, Copy)]
pub enum Target {
    /// the thread that calls present (follows queued rendering)
    Game = 0,
    /// our render thread
    Render = 1,
}

#[cfg(windows)]
mod imp {
    use super::Target;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, HMODULE};
    use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, CONTEXT_CONTROL_AMD64, GetThreadContext};
    use windows_sys::Win32::System::LibraryLoader::{GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW, GetModuleHandleExW};
    use windows_sys::Win32::System::Threading::{GetCurrentThreadId, OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_SUSPEND_RESUME};

    const NAMES: [&str; 2] = ["game", "render"];
    static TIDS: [AtomicU32; 2] = [const { AtomicU32::new(0) }; 2];
    /// top of each thread's stack, so stack reads stay inside it
    static TOPS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    const MAX_SAMPLES: usize = 1 << 20;
    static RUNNING: AtomicBool = AtomicBool::new(false);
    /// per thread: (ip, nearest return address into this dll when ip is outside it, else 0)
    static SAMPLES: Mutex<[Vec<(u64, u64)>; 2]> = Mutex::new([Vec::new(), Vec::new()]);

    /// this dll's image range
    fn own_range() -> (u64, u64) {
        let (base, _) = module_of(own_range as fn() -> (u64, u64) as usize as u64);
        if base == 0 {
            return (0, 0);
        }
        // SAFETY: base is our own mapped pe image
        unsafe {
            let pe = base + *((base + 0x3c) as *const u32) as u64;
            (base, base + *((pe + 24 + 56) as *const u32) as u64) // SizeOfImage
        }
    }

    /// remember the calling thread as `t`
    pub fn note_thread(t: Target) {
        let tid = unsafe { GetCurrentThreadId() };
        if TIDS[t as usize].swap(tid, Ordering::Relaxed) != tid {
            // NT_TIB.StackBase, read from this thread's teb
            let teb: u64;
            unsafe { std::arch::asm!("mov {}, gs:[0x30]", out(reg) teb, options(nostack, readonly, preserves_flags)) };
            TOPS[t as usize].store(unsafe { *((teb + 8) as *const u64) }, Ordering::Relaxed);
        }
    }

    /// one sample: suspend, read ip (and a caller for code outside this dll), resume
    unsafe fn sample(h: HANDLE, top: u64, lo: u64, hi: u64) -> Option<(u64, u64)> {
        unsafe {
            if SuspendThread(h) == u32::MAX {
                return None;
            }
            let mut ctx: CONTEXT = std::mem::zeroed();
            ctx.ContextFlags = CONTEXT_CONTROL_AMD64;
            let mut out = None;
            if GetThreadContext(h, &mut ctx) != 0 {
                let mut caller = 0;
                if !(lo..hi).contains(&ctx.Rip) && ctx.Rsp & 7 == 0 && ctx.Rsp < top {
                    // the thread is suspended, so its stack is stable
                    let sp = ctx.Rsp as *const u64;
                    for i in 0..64.min((top - ctx.Rsp) as usize / 8) {
                        let v = *sp.add(i);
                        if (lo..hi).contains(&v) {
                            caller = v;
                            break;
                        }
                    }
                }
                out = Some((ctx.Rip, caller));
            }
            ResumeThread(h);
            out
        }
    }

    pub fn start() {
        if RUNNING.swap(true, Ordering::SeqCst) {
            return;
        }
        let targets: Vec<(usize, u32, u64)> =
            (0..2).map(|i| (i, TIDS[i].load(Ordering::Relaxed), TOPS[i].load(Ordering::Relaxed))).filter(|t| t.1 != 0).collect();
        let _ = std::thread::Builder::new().name("wulkanium sampler".into()).spawn(move || {
            let (lo, hi) = own_range();
            let handles: Vec<(usize, HANDLE, u64)> = targets
                .iter()
                .map(|&(i, tid, top)| (i, unsafe { OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, 0, tid) }, top))
                .filter(|h| !h.1.is_null())
                .collect();
            // never grows: allocating while a target is suspended can deadlock on its heap lock
            let mut local = [Vec::with_capacity(MAX_SAMPLES), Vec::with_capacity(MAX_SAMPLES)];
            while RUNNING.load(Ordering::Relaxed) && local.iter().all(|v| v.len() < MAX_SAMPLES) {
                std::thread::sleep(std::time::Duration::from_micros(1000));
                for &(i, h, top) in &handles {
                    // SAFETY: win32 calls on a handle we opened; stack reads stay below its top
                    if let Some(s) = unsafe { sample(h, top, lo, hi) } {
                        local[i].push(s);
                    }
                }
            }
            for (_, h, _) in handles {
                unsafe { CloseHandle(h) };
            }
            *SAMPLES.lock().unwrap_or_else(|e| e.into_inner()) = local;
        });
    }

    fn module_of(addr: u64) -> (u64, String) {
        let mut m: HMODULE = std::ptr::null_mut();
        // SAFETY: from-address lookup without touching the refcount
        let ok = unsafe { GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, addr as usize as *const u16, &mut m) };
        if ok == 0 || m.is_null() {
            return (0, "?".into());
        }
        let mut buf = [0u16; 260];
        let n = unsafe { GetModuleFileNameW(m, buf.as_mut_ptr(), buf.len() as u32) } as usize;
        let path = String::from_utf16_lossy(&buf[..n]);
        (m as u64, path.rsplit(['\\', '/']).next().unwrap_or(&path).to_string())
    }

    /// stop and log, per thread: samples per module, the hottest addresses (module + offset),
    /// and for samples outside this dll the return address they will come back to
    pub fn stop() {
        if !RUNNING.swap(false, Ordering::SeqCst) {
            return;
        }
        // the sampler hands its samples over when it notices
        std::thread::sleep(std::time::Duration::from_millis(5));
        let all = std::mem::take(&mut *SAMPLES.lock().unwrap_or_else(|e| e.into_inner()));
        let (lo, _) = own_range();
        for (samples, label) in all.iter().zip(NAMES) {
            if samples.is_empty() {
                continue;
            }
            let mut modules: std::collections::HashMap<u64, (String, u64)> = Default::default();
            let mut addrs: std::collections::HashMap<u64, u64> = Default::default();
            let mut callers: std::collections::HashMap<u64, u64> = Default::default();
            for &(a, c) in samples {
                *addrs.entry(a).or_default() += 1;
                if c != 0 {
                    *callers.entry(c).or_default() += 1;
                }
            }
            let mut by_addr: Vec<(u64, u64, String, u64)> = Vec::new();
            for (&a, &n) in &addrs {
                // modules are looked up once per distinct address
                let (base, name) = module_of(a);
                modules.entry(base).or_insert((name.clone(), 0)).1 += n;
                by_addr.push((a, n, name, base));
            }
            let total = samples.len() as f64;
            let pct = |n: u64| n as f64 / total * 100.0;
            let mut mods: Vec<_> = modules.into_values().collect();
            mods.sort_by(|a, b| b.1.cmp(&a.1));
            crate::log::line(&format!("sampler {label}: {} samples", samples.len()));
            for (name, n) in mods.iter().take(20) {
                crate::log::line(&format!("sampler {label} module {:5.1}% {name}", pct(*n)));
            }
            by_addr.sort_by(|a, b| b.1.cmp(&a.1));
            for (a, n, name, base) in by_addr.iter().take(200) {
                crate::log::line(&format!("sampler {label} addr {:5.2}% {name}+{:#x}", pct(*n), a - base));
            }
            let mut by_caller: Vec<_> = callers.into_iter().collect();
            by_caller.sort_by(|a, b| b.1.cmp(&a.1));
            for (c, n) in by_caller.iter().take(60) {
                crate::log::line(&format!("sampler {label} caller {:5.2}% shaderapiwgpu.dll+{:#x}", pct(*n), c - lo));
            }
        }
    }
}

#[cfg(windows)]
pub use imp::{note_thread, start, stop};

#[cfg(not(windows))]
pub fn note_thread(_: Target) {}
#[cfg(not(windows))]
pub fn start() {}
#[cfg(not(windows))]
pub fn stop() {}
