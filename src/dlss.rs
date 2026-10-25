//! dlss (dlaa) through dlss_wgpu, `dlss` feature. scene draws are jittered and write motion
//! vectors; dlss resolves the back buffer after the last scene draw, before the hud. End toggles.

use std::sync::atomic::{AtomicBool, Ordering};

/// scene draws get jitter and motion vectors
static ACTIVE: AtomicBool = AtomicBool::new(false);
/// F7 / F8 flip the jitter signs reported to dlss
pub static FLIP_X: AtomicBool = AtomicBool::new(false);
pub static FLIP_Y: AtomicBool = AtomicBool::new(false);

/// ngx's ctrl+alt+f debug keys switch vts on linux, so inject them via SendInput: Delete -> F6,
/// Pause -> F10. held a few frames so ngx sees them.
pub fn poll_debug_keys() {
    #[cfg(windows)]
    {
        use std::sync::atomic::AtomicU32;
        use windows_sys::Win32::UI::Input::KeyboardAndMouse::*;
        static DOWN: [AtomicBool; 5] = [const { AtomicBool::new(false) }; 5];
        // (vk << 8) | frames left, 0 = idle
        static HELD: AtomicU32 = AtomicU32::new(0);
        let send = |vk: u16, up: bool| {
            let input = INPUT {
                r#type: INPUT_KEYBOARD,
                Anonymous: INPUT_0 { ki: KEYBDINPUT { wVk: vk, wScan: 0, dwFlags: if up { KEYEVENTF_KEYUP } else { 0 }, time: 0, dwExtraInfo: 0 } },
            };
            unsafe { SendInput(1, &input, std::mem::size_of::<INPUT>() as i32) };
        };
        let held = HELD.load(Ordering::Relaxed);
        if held != 0 {
            let (vk, left) = ((held >> 8) as u16, held & 0xff);
            if left <= 1 {
                send(vk, true);
                send(VK_MENU, true);
                send(VK_CONTROL, true);
                HELD.store(0, Ordering::Relaxed);
            } else {
                HELD.store(held - 1, Ordering::Relaxed);
            }
            return;
        }
        let map = [(VK_DELETE, VK_F6), (VK_PAUSE, VK_F10)];
        for (i, (pad, f)) in map.into_iter().enumerate() {
            let down = unsafe { GetAsyncKeyState(pad as i32) } as u16 & 0x8000 != 0;
            if down && !DOWN[i].swap(true, Ordering::Relaxed) {
                send(VK_CONTROL, false);
                send(VK_MENU, false);
                send(f, false);
                HELD.store(((f as u32) << 8) | 4, Ordering::Relaxed);
                crate::log::line(&format!("dlss: injected Ctrl+Alt+F{}", f - VK_F1 + 1));
                return;
            } else if !down {
                DOWN[i].store(false, Ordering::Relaxed);
            }
        }
    }
}

pub fn poll_flip_keys() {
    #[cfg(windows)]
    {
        use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_F7, VK_F8};
        static DOWN: [AtomicBool; 2] = [AtomicBool::new(false), AtomicBool::new(false)];
        for (i, (key, flag)) in [(VK_F7, &FLIP_X), (VK_F8, &FLIP_Y)].into_iter().enumerate() {
            let down = unsafe { GetAsyncKeyState(key as i32) } as u16 & 0x8000 != 0;
            if down && !DOWN[i].swap(true, Ordering::Relaxed) {
                let v = !flag.load(Ordering::Relaxed);
                flag.store(v, Ordering::Relaxed);
                crate::log::line(&format!("dlss: jitter {} sign {}", if i == 0 { "x" } else { "y" }, if v { "flipped" } else { "normal" }));
            } else if !down {
                DOWN[i].store(false, Ordering::Relaxed);
            }
        }
    }
}

fn halton(mut index: u32, base: u32) -> f32 {
    let (mut f, mut r) = (1.0, 0.0);
    while index > 0 {
        f /= base as f32;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

/// Page Down cycles: 0 halton (32 phases), 1 none, 2-5 one fixed quadrant (guide 8.4.2)
pub static JITTER_MODE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn jitter_mode_name() -> &'static str {
    match JITTER_MODE.load(Ordering::Relaxed) {
        0 => "Halton",
        1 => "none",
        2 => "quadrant +x+y",
        3 => "quadrant -x+y",
        4 => "quadrant -x-y",
        _ => "quadrant +x-y",
    }
}

pub fn poll_jitter_mode_key() {
    #[cfg(windows)]
    {
        use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_NEXT};
        static DOWN: AtomicBool = AtomicBool::new(false);
        let down = unsafe { GetAsyncKeyState(VK_NEXT as i32) } as u16 & 0x8000 != 0;
        if down && !DOWN.swap(true, Ordering::Relaxed) {
            let m = (JITTER_MODE.load(Ordering::Relaxed) + 1) % 6;
            JITTER_MODE.store(m, Ordering::Relaxed);
            crate::log::line(&format!("dlss: jitter pattern {}", jitter_mode_name()));
        } else if !down {
            DOWN.store(false, Ordering::Relaxed);
        }
    }
}

/// pixels, -0.5..0.5, +y down
pub fn jitter(frame: u64) -> Option<[f32; 2]> {
    ACTIVE.load(Ordering::Relaxed).then(|| match JITTER_MODE.load(Ordering::Relaxed) {
        0 => {
            let i = (frame % 32) as u32 + 1;
            [halton(i, 2) - 0.5, halton(i, 3) - 0.5]
        }
        1 => [0.0, 0.0],
        2 => [0.25, 0.25],
        3 => [-0.25, 0.25],
        4 => [-0.25, -0.25],
        _ => [0.25, -0.25],
    })
}

pub fn set_active(on: bool) {
    ACTIVE.store(on, Ordering::Relaxed);
}

pub fn toggle_pressed() -> bool {
    #[cfg(windows)]
    {
        use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_END};
        static DOWN: AtomicBool = AtomicBool::new(false);
        let down = unsafe { GetAsyncKeyState(VK_END as i32) } as u16 & 0x8000 != 0;
        let was = DOWN.swap(down, Ordering::Relaxed);
        down && !was
    }
    #[cfg(not(windows))]
    false
}

/// uv-space offsets to last frame
pub const MV_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

/// Page Up cycles: 0 off, 1 motion vectors, 2 depth
pub static DEBUG_VIEW: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn poll_debug_view_key() {
    #[cfg(windows)]
    {
        use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_PRIOR};
        static DOWN: AtomicBool = AtomicBool::new(false);
        let down = unsafe { GetAsyncKeyState(VK_PRIOR as i32) } as u16 & 0x8000 != 0;
        if down && !DOWN.swap(true, Ordering::Relaxed) {
            let v = (DEBUG_VIEW.load(Ordering::Relaxed) + 1) % 3;
            DEBUG_VIEW.store(v, Ordering::Relaxed);
        } else if !down {
            DOWN.store(false, Ordering::Relaxed);
        }
    }
}

/// motion vectors as color around grey (1 px = 0.05), or depth
pub const DEBUG_VIEW_WGSL: &str = r#"
@group(0) @binding(0) var mv: texture_2d<f32>;
@group(0) @binding(1) var depth: texture_depth_2d;
struct Params { mode: u32, w: f32, h: f32, pad: f32 }
@group(0) @binding(2) var<uniform> p: Params;

@vertex
fn vs(@builtin(vertex_index) v: u32) -> @builtin(position) vec4<f32> {
    let c = vec2<f32>(f32(v & 1u), f32(v >> 1u));
    return vec4<f32>(c.x * 2.0 - 1.0, 1.0 - c.y * 2.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(pos.xy);
    if (p.mode == 1u) {
        let m = textureLoad(mv, px, 0).xy * vec2<f32>(p.w, p.h);
        return vec4<f32>(clamp(vec2<f32>(0.5) + m * 0.05, vec2<f32>(0.0), vec2<f32>(1.0)), 0.5, 1.0);
    }
    let d = textureLoad(depth, px, 0);
    let v = pow(1.0 - d, 0.25);
    return vec4<f32>(v, v, v, 1.0);
}
"#;

#[cfg(feature = "dlss")]
pub use imp::*;

#[cfg(feature = "dlss")]
mod imp {
    use std::sync::{Arc, Mutex};

    use dlss_wgpu::super_resolution::{DlssSuperResolution, DlssSuperResolutionExposure, DlssSuperResolutionRenderParameters};
    use dlss_wgpu::{DlssFeatureFlags, DlssPerfQualityMode, DlssSdk};

    use crate::log;

    pub const PROJECT_ID: uuid::Uuid = uuid::uuid!("6d1c0c55-7a0e-4a52-9b0e-2b8f6f2a6c1d");

    /// ngx loads `<game>/dlss/lib/Windows_x86_64/dev/nvngx_dlss.dll` (dlss_wgpu reads `$DLSS_SDK`)
    pub fn prepare_environment() {
        if let Ok(cwd) = std::env::current_dir() {
            let dir = cwd.join("dlss");
            if dir.join("lib/Windows_x86_64/dev/nvngx_dlss.dll").exists() {
                // SAFETY: runs before other threads read the environment
                unsafe { std::env::set_var("DLSS_SDK", &dir) };
            } else {
                log::line(&format!("dlss: {} has no lib/Windows_x86_64/dev/nvngx_dlss.dll", dir.display()));
            }
        }
    }

    pub struct Dlss {
        sdk: Arc<Mutex<DlssSdk>>,
        sr: Option<DlssSuperResolution>,
        size: (u32, u32),
        pub mv: Option<(wgpu::Texture, wgpu::TextureView)>,
        pub output: Option<(wgpu::Texture, wgpu::TextureView)>,
        pub enabled: bool,
        /// else the history is reset
        ran: bool,
    }

    impl Dlss {
        pub fn new(device: &wgpu::Device) -> Option<Dlss> {
            match DlssSdk::new(PROJECT_ID, device.clone()) {
                Ok(sdk) => {
                    log::line("dlss: SDK initialized");
                    Some(Dlss { sdk, sr: None, size: (0, 0), mv: None, output: None, enabled: true, ran: false })
                }
                Err(e) => {
                    log::line(&format!("dlss: SDK unavailable: {e:?}"));
                    None
                }
            }
        }

        pub fn ensure(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, size: (u32, u32)) -> bool {
            if self.sr.is_some() && self.size == size {
                return true;
            }
            self.sr = None;
            let flags = DlssFeatureFlags::LowResolutionMotionVectors | DlssFeatureFlags::AutoExposure;
            match DlssSuperResolution::new([size.0, size.1], DlssPerfQualityMode::Dlaa, flags, self.sdk.clone(), device, queue) {
                Ok(sr) => self.sr = Some(sr),
                Err(e) => {
                    log::line(&format!("dlss: can't create DLAA for {}x{}: {e:?}", size.0, size.1));
                    self.enabled = false;
                    return false;
                }
            }
            let tex = |label, format, usage| {
                let t = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                });
                let v = t.create_view(&Default::default());
                (t, v)
            };
            self.mv = Some(tex("dlss motion vectors", super::MV_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING));
            self.output = Some(tex("dlss output", wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING));
            self.size = size;
            self.ran = false;
            log::line(&format!("dlss: DLAA at {}x{}", size.0, size.1));
            true
        }

        /// returns the command buffer to submit after `encoder`'s
        pub fn evaluate(&mut self, encoder: &mut wgpu::CommandEncoder, adapter: &wgpu::Adapter, color: &wgpu::TextureView, depth: &wgpu::TextureView, jitter: [f32; 2]) -> Option<wgpu::CommandBuffer> {
            let (sr, mv, out) = (self.sr.as_mut()?, &self.mv.as_ref()?.1, &self.output.as_ref()?.1);
            let (w, h) = self.size;
            let params = DlssSuperResolutionRenderParameters {
                color,
                depth,
                motion_vectors: mv,
                exposure: DlssSuperResolutionExposure::Automatic,
                bias: None,
                dlss_output: out,
                reset: !self.ran,
                // guide 3.7.3: the projection offset in motion vector pixel space (+y down)
                jitter_offset: [
                    if super::FLIP_X.load(std::sync::atomic::Ordering::Relaxed) { -jitter[0] } else { jitter[0] },
                    if super::FLIP_Y.load(std::sync::atomic::Ordering::Relaxed) { -jitter[1] } else { jitter[1] },
                ],
                partial_texture_size: None,
                // motion vectors are in uv units
                motion_vector_scale: Some([w as f32, h as f32]),
            };
            match sr.render(params, encoder, adapter) {
                Ok(cb) => {
                    self.ran = true;
                    Some(cb)
                }
                Err(e) => {
                    log::line(&format!("dlss: evaluate failed: {e:?}"));
                    self.enabled = false;
                    None
                }
            }
        }

        pub fn size(&self) -> (u32, u32) {
            self.size
        }

        pub fn skipped(&mut self) {
            self.ran = false;
        }
    }
}
