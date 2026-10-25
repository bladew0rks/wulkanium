//! the wgpu side: draws the recorded frame with the game's translated shaders

use std::collections::HashMap;

use rustc_hash::FxHashMap;
use std::sync::{Arc, Mutex};

use crate::abi::{ElementType, vertex_elements};
use crate::api::{Cmd, DrawCmd, Frame, Geometry, PS_BLOCK, PipeKey, VS_BLOCK};
use crate::d3d2wgsl::{self, DIM_2D, DIM_CUBE, DIM_DEPTH, DIM_VOLUME, PsOptions};
use crate::log;
use crate::shaders::{Bytecode, ShaderStore};
use crate::texconv::Image;

/// quad over the destination viewport sampling a source rect
const BLIT: &str = r#"
@group(0) @binding(0) var<storage, read> rects: array<vec4<f32>>;
@group(1) @binding(0) var src: texture_2d<f32>;
@group(1) @binding(1) var smp: sampler;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VOut {
    let c = vec2<f32>(f32(v & 1u), f32(v >> 1u));
    let r = rects[i];
    var o: VOut;
    o.pos = vec4<f32>(c.x * 2.0 - 1.0, 1.0 - c.y * 2.0, 0.0, 1.0);
    o.uv = mix(r.xy, r.zw, c);
    return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(src, smp, i.uv, 0.0);
}
"#;

/// d3d9's d24s8
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24PlusStencil8;

const PIPELINE_CACHE_FILE: &str = "shaderapiwgpu.pipelinecache";

const MAX_QUERIES: u32 = 4096;

/// run the recorded part of the frame without presenting (ReadPixels)
/// ns the render thread spent drawing frames, since the last periodic log line
pub static RENDER_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// parts of RENDER_NS: waiting for a swapchain image, and the present call
pub static ACQUIRE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PRESENT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// gpu time of the last measured frame
pub static GPU_US: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// work for the render thread, in order
enum Job {
    Present(Frame),
    /// mid-frame flush; replies when done
    Flush(Frame, std::sync::mpsc::Sender<()>),
    /// replies once everything queued before it is done
    Sync(std::sync::mpsc::Sender<()>),
}

/// the render thread's queue: one frame in flight, so the game runs at most a frame ahead
static QUEUE: Mutex<Option<(std::sync::mpsc::SyncSender<Job>, std::thread::JoinHandle<()>)>> = Mutex::new(None);

fn queue() -> Option<std::sync::mpsc::SyncSender<Job>> {
    if SHUT_DOWN.load(std::sync::atomic::Ordering::SeqCst) {
        return None;
    }
    let mut q = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if q.is_none() {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Job>(1);
        let thread = std::thread::Builder::new()
            .name("wulkanium render".into())
            .spawn(move || {
                crate::sampler::note_thread(crate::sampler::Target::Render);
                for job in rx {
                    match job {
                        Job::Present(frame) => present_now(frame),
                        Job::Flush(frame, done) => {
                            flush_now(frame);
                            let _ = done.send(());
                        }
                        Job::Sync(done) => {
                            let _ = done.send(());
                        }
                    }
                }
            })
            .ok()?;
        *q = Some((tx, thread));
    }
    q.as_ref().map(|q| q.0.clone())
}

/// hand a finished frame to the render thread
pub fn present(frame: Frame) {
    if let Some(q) = queue() {
        let t = std::time::Instant::now();
        let _ = q.send(Job::Present(frame));
        crate::api::PRESENT_WAIT_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

/// run the recorded part of a frame now and wait for it
pub fn flush(frame: Frame) {
    let (tx, rx) = std::sync::mpsc::channel();
    if queue().is_some_and(|q| q.send(Job::Flush(frame, tx)).is_ok()) {
        let _ = rx.recv();
    }
}

/// wait until the render thread has done everything queued so far
pub fn sync() {
    let (tx, rx) = std::sync::mpsc::channel();
    if queue().is_some_and(|q| q.send(Job::Sync(tx)).is_ok()) {
        let _ = rx.recv();
    }
}

fn flush_now(frame: Frame) {
    let mut g = RENDERER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(Ok(r)) = g.as_mut() {
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| r.draw(frame, false))).is_err() {
            log::line("renderer: panicked while flushing; rendering disabled");
            *g = Some(Err("panicked".into()));
        }
    }
}

pub fn read_target(id: i32, x: u32, y: u32, w: u32, h: u32) -> Option<Vec<u8>> {
    sync();
    let g = RENDERER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Ok(r)) = g.as_ref() else { return None };
    let t = r.targets.get(&id)?;
    if w == 0 || h == 0 || x + w > t.size.0 || y + h > t.size.1 {
        return None;
    }
    let row = (w * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buf = r.device.create_buffer(&wgpu::BufferDescriptor { label: Some("read pixels"), size: (row * h) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut enc = r.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &t.texture, mip_level: 0, origin: wgpu::Origin3d { x, y, z: 0 }, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    r.queue.submit([enc.finish()]);
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = done.clone();
    buf.slice(..).map_async(wgpu::MapMode::Read, move |res| flag.store(res.is_ok(), std::sync::atomic::Ordering::SeqCst));
    let _ = r.device.poll(wgpu::PollType::wait_indefinitely());
    if !done.load(std::sync::atomic::Ordering::SeqCst) {
        return None;
    }
    let bgra = matches!(r.config.format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    {
        let data = buf.slice(..).get_mapped_range().ok()?;
        for y in 0..h as usize {
            for p in data[y * row as usize..y * row as usize + w as usize * 4].chunks_exact(4) {
                out.extend_from_slice(&if bgra { [p[2], p[1], p[0], p[3]] } else { [p[0], p[1], p[2], p[3]] });
            }
        }
    }
    buf.unmap();
    Some(out)
}

pub fn wait_for_queries() {
    sync();
    let g = RENDERER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(Ok(r)) = g.as_ref() {
        let _ = r.device.poll(wgpu::PollType::wait_indefinitely());
    }
}

/// bit n = n samples, 0 until the renderer is up
pub static MSAA_SUPPORTED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// multisampled back buffer, resolved at the end of every pass so copies and reads see it
struct BackMsaa {
    samples: u32,
    size: (u32, u32),
    color: wgpu::TextureView,
    depth: wgpu::TextureView,
}

/// Scroll Lock: last shadow map in the bottom-left corner
static SHADOW_VIEW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn poll_shadow_view_key() {
    #[cfg(windows)]
    {
        use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_SCROLL};
        static DOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let down = unsafe { GetAsyncKeyState(VK_SCROLL as i32) } as u16 & 0x8000 != 0;
        if down && !DOWN.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let on = !SHADOW_VIEW.load(std::sync::atomic::Ordering::Relaxed);
            SHADOW_VIEW.store(on, std::sync::atomic::Ordering::Relaxed);
            log::line(&format!("shadow map view {}", if on { "on" } else { "off" }));
        } else if !down {
            DOWN.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

const SHADOW_VIEW_WGSL: &str = r#"
@group(0) @binding(0) var t: texture_depth_2d;
struct V { @builtin(position) p: vec4f, @location(0) uv: vec2f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> V {
    let uv = vec2f(f32(i & 1u), f32(i >> 1u));
    return V(vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0), uv);
}
@fragment fn fs(v: V) -> @location(0) vec4f {
    let dim = vec2f(textureDimensions(t));
    let d = textureLoad(t, vec2i(clamp(v.uv * dim, vec2f(0.0), dim - 1.0)), 0);
    // flashlight near plane is 4: z = 4 / (1 - depth)
    let z = 4.0 / max(1.0 - d, 1e-6);
    return vec4f(vec3f(clamp(z / 1000.0, 0.0, 1.0)), 1.0);
}
"#;

const BACK: i32 = -1;
/// scratch target for render targets we don't know
const DISCARD: i32 = i32::MIN;

struct Target {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
}

/// shadow depth texture (NV_DST24), sampled with depth compare
struct ShadowMap {
    size: (u32, u32),
    view: wgpu::TextureView,
    sample_view: wgpu::TextureView,
    /// bind group cache tag
    serial: u64,
}

struct GpuTexture {
    version: u64,
    texture: wgpu::Texture,
    /// rgba8 2d or cube: dirty rects can be written in place
    partial_ok: bool,
    /// face 0 of a cube
    view: wgpu::TextureView,
    cube_view: Option<wgpu::TextureView>,
    volume_view: Option<wgpu::TextureView>,
}

/// d3d9 sampler state of a binding
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SamplerKey {
    min: bool,
    mag: bool,
    mips: bool,
    mip_linear: bool,
    wrap: [u8; 3],
    aniso: u16,
}

impl SamplerKey {
    /// `force`: GMOD_ForceFilterMode
    fn new(filter: [i32; 2], wrap: [i32; 3], force: Option<i32>, aniso: i32) -> SamplerKey {
        let (mut min, mips, mip_linear, mut aniso_on) = match filter[0] {
            0 => (false, false, false, false),
            1 => (true, false, false, false),
            2 => (false, true, false, false),
            3 => (true, true, false, false),
            4 => (false, true, true, false),
            6 => (true, true, true, true),
            _ => (true, true, true, false),
        };
        let mut mag = filter[1] != 0;
        if let Some(f) = force {
            min = f >= 2;
            mag = f >= 2;
            aniso_on = f == 3;
        }
        let aniso = if aniso_on && min && mag && mip_linear { aniso.clamp(1, 16) as u16 } else { 1 };
        SamplerKey { min, mag, mips, mip_linear, wrap: wrap.map(|w| w.clamp(0, 2) as u8), aniso }
    }
}

struct VsModule {
    module: wgpu::ShaderModule,
    inputs: Vec<(u32, u8, u8)>,
    samplers: Vec<(u32, u8)>,
}

struct PsModule {
    module: wgpu::ShaderModule,
    samplers: Vec<(u32, u8)>,
}

/// (binding, dimension), sorted
type SamplerSig = Vec<(u32, u8)>;

#[derive(Clone, PartialEq, Eq, Hash)]
struct PipelineId {
    vs: usize,
    format: u64,
    stride: u32,
    ps: usize,
    ps_opts: PsOptions,
    key: PipeKey,
    /// (stride, element offset)
    color: Option<(u32, u32)>,
    clip: bool,
    flex: bool,
    /// dlss scene passes have a second color target
    mv: bool,
    samples: u32,
}

/// a pipeline compiling, compiled or rejected. draws skip it until it's ready (pop-in instead of
/// a hitch).
enum PipeSlot {
    Ready(Pipeline),
    Pending(Arc<SamplerSig>, Arc<Mutex<Option<Option<wgpu::RenderPipeline>>>>),
    Failed,
}

/// an owned pipeline description, compiled on a worker thread
struct PipelineJob {
    label: String,
    layout: wgpu::PipelineLayout,
    vs: wgpu::ShaderModule,
    ps: wgpu::ShaderModule,
    /// (stride, attributes) of streams 0..2
    buffers: [Option<(u64, Vec<wgpu::VertexAttribute>)>; 3],
    primitive: wgpu::PrimitiveState,
    depth_stencil: wgpu::DepthStencilState,
    multisample: wgpu::MultisampleState,
    targets: [Option<wgpu::ColorTargetState>; 2],
    cache: Option<wgpu::PipelineCache>,
}

impl PipelineJob {
    fn create(self, device: &wgpu::Device) -> Option<wgpu::RenderPipeline> {
        let buffers: Vec<Option<wgpu::VertexBufferLayout>> = self.buffers.iter().map(|b| b.as_ref().map(|(stride, attributes)| wgpu::VertexBufferLayout { array_stride: *stride, step_mode: wgpu::VertexStepMode::Vertex, attributes })).collect();
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&self.layout),
            vertex: wgpu::VertexState { module: &self.vs, entry_point: Some("main"), compilation_options: Default::default(), buffers: &buffers },
            primitive: self.primitive,
            depth_stencil: Some(self.depth_stencil.clone()),
            multisample: self.multisample,
            fragment: Some(wgpu::FragmentState { module: &self.ps, entry_point: Some("main"), compilation_options: Default::default(), targets: &self.targets }),
            multiview_mask: None,
            cache: self.cache.as_ref(),
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            log::line(&format!("renderer: pipeline {} rejected: {e}", self.label));
            return None;
        }
        Some(pipeline)
    }
}

/// worker threads that compile pipelines (the driver compile takes 10-300 ms)
struct Compiler {
    tx: std::sync::mpsc::Sender<(PipelineJob, Arc<Mutex<Option<Option<wgpu::RenderPipeline>>>>)>,
}

impl Compiler {
    fn new(device: &wgpu::Device) -> Compiler {
        let (tx, rx) = std::sync::mpsc::channel::<(PipelineJob, Arc<Mutex<Option<Option<wgpu::RenderPipeline>>>>)>();
        let rx = Arc::new(Mutex::new(rx));
        let workers = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(1, 4));
        for i in 0..workers {
            let (rx, device) = (rx.clone(), device.clone());
            let _ = std::thread::Builder::new().name(format!("wulkanium pipelines {i}")).spawn(move || {
                loop {
                    let job = rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                    let Ok((job, cell)) = job else { break };
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.create(&device))).unwrap_or(None);
                    *cell.lock().unwrap_or_else(|e| e.into_inner()) = Some(result);
                }
            });
        }
        Compiler { tx }
    }

    fn compile(&self, job: PipelineJob, cell: Arc<Mutex<Option<Option<wgpu::RenderPipeline>>>>) {
        let _ = self.tx.send((job, cell));
    }
}

struct Pipeline {
    pipeline: wgpu::RenderPipeline,
    sig: Arc<SamplerSig>,
}

struct Renderer {
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    shaders: ShaderStore,
    vs_modules: FxHashMap<(usize, u64, bool, bool, bool, bool), Option<Arc<VsModule>>>,
    ps_modules: FxHashMap<(usize, PsOptions, bool), Option<Arc<PsModule>>>,
    pipelines: FxHashMap<PipelineId, PipeSlot>,
    /// background pipeline compiles
    compiler: Compiler,
    /// keyed by the snapshot's shader name pointer
    code: FxHashMap<(crate::shadow::ShaderName, i32, i32), Option<Bytecode>>,
    /// 0 = unavailable
    gpu_us: Arc<std::sync::atomic::AtomicU32>,
    timestamps: Option<wgpu::QuerySet>,
    pipeline_cache: Option<wgpu::PipelineCache>,
    unsaved_pipelines: usize,
    const_layout: wgpu::BindGroupLayout,
    tex_layouts: FxHashMap<Arc<SamplerSig>, (wgpu::BindGroupLayout, wgpu::PipelineLayout)>,
    bind_groups: FxHashMap<(Arc<SamplerSig>, Vec<(i32, u64, SamplerKey)>), wgpu::BindGroup>,
    blit_pipeline: wgpu::RenderPipeline,
    blit_rect_layout: wgpu::BindGroupLayout,
    blit_tex_layout: wgpu::BindGroupLayout,
    samplers: FxHashMap<SamplerKey, wgpu::Sampler>,
    border: bool,
    /// mat_forceaniso
    anisotropy: i32,
    clip_distances: bool,
    textures: HashMap<i32, GpuTexture>,
    /// 2d white, cube black, volume white
    dummies: [wgpu::TextureView; 3],
    bc: bool,
    lines: bool,
    lost: Arc<std::sync::atomic::AtomicBool>,
    meshes: HashMap<u64, (wgpu::Buffer, wgpu::Buffer)>,
    targets: HashMap<i32, Target>,
    depths: HashMap<i32, ((u32, u32), wgpu::TextureView)>,
    /// not retried until the data changes
    failed: HashMap<i32, u64>,
    back_msaa: Option<BackMsaa>,
    shadow_view: Option<(wgpu::RenderPipeline, wgpu::BindGroupLayout)>,
    shadow_maps: HashMap<i32, ShadowMap>,
    shadow_serial: u64,
    /// depth 1.0, everything lit
    dummy_depth: wgpu::TextureView,
    /// nvidia pcf
    compare_sampler: wgpu::Sampler,
    overlay: crate::overlay::Overlay,
    effects: crate::effects::Effects,
    #[cfg_attr(not(feature = "dlss"), allow(dead_code))]
    wgpu_adapter: wgpu::Adapter,
    #[cfg(feature = "dlss")]
    dlss: Option<crate::dlss::Dlss>,
    #[cfg(feature = "dlss")]
    debug_view: Option<(wgpu::RenderPipeline, wgpu::BindGroupLayout)>,
    /// the inputs as dlss saw them
    #[cfg(feature = "dlss")]
    debug_snapshot: Option<(wgpu::Texture, wgpu::TextureView)>,
    query_set: wgpu::QuerySet,
    adapter: String,
    backend: String,
}

static RENDERER: Mutex<Option<Result<Renderer, String>>> = Mutex::new(None);
static SHUT_DOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// tear down while the window exists: a live vulkan device can keep the process from exiting
pub fn shutdown() {
    SHUT_DOWN.store(true, std::sync::atomic::Ordering::SeqCst);
    // closing the queue ends the render thread once it's done
    if let Some((tx, thread)) = QUEUE.lock().unwrap_or_else(|e| e.into_inner()).take() {
        drop(tx);
        let _ = thread.join();
    }
    let mut g = RENDERER.lock().unwrap_or_else(|e| e.into_inner());
    match g.take() {
        Some(Ok(mut r)) if !r.lost.load(std::sync::atomic::Ordering::SeqCst) => {
            r.save_pipeline_cache(false);
            let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let _ = r.device.poll(wgpu::PollType::wait_indefinitely());
                drop(r);
            }))
            .is_ok();
            log::line(if ok { "renderer: shut down" } else { "renderer: panicked while shutting down" });
        }
        Some(Ok(r)) => std::mem::forget(r),
        _ => {}
    }
}

fn present_now(frame: Frame) {
    let mut g = RENDERER.lock().unwrap_or_else(|e| e.into_inner());
    // lost devices and panicked renderers get rebuilt
    let rebuild = match g.as_ref() {
        Some(Ok(r)) => r.lost.load(std::sync::atomic::Ordering::SeqCst),
        Some(Err(_)) => true,
        None => false,
    };
    if rebuild {
        static REBUILDS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if REBUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 5 {
            log::line("renderer: rebuilding after device loss / failure");
            // dropping a renderer on a lost device can panic in wgpu-hal, so leak it
            if let Some(old) = g.take() {
                std::mem::forget(old);
            }
            crate::api::RENDERER_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    if g.is_none() {
        let dropped = || crate::api::RENDERER_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let Some(win) = *crate::device::WINDOW.lock().unwrap_or_else(|e| e.into_inner()) else {
            dropped();
            return;
        };
        if win.hwnd == 0 {
            dropped();
            return;
        }
        let r = std::panic::catch_unwind(|| Renderer::new(win.hwnd)).unwrap_or_else(|_| Err("panicked while starting".into()));
        match &r {
            Ok(r) => log::line(&format!("renderer: line polygons {}, BC textures {}, surface {:?} {}x{}", r.lines, r.bc, r.config.format, r.config.width, r.config.height)),
            Err(e) => log::line(&format!("renderer: failed to start: {e}")),
        }
        *g = Some(r);
    }
    if let Some(Ok(r)) = g.as_mut() {
        // a panic must not unwind into the game
        let t = std::time::Instant::now();
        let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| r.draw(frame, true))).is_ok();
        RENDER_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        if !ok {
            log::line("renderer: panicked while drawing; rendering disabled");
            *g = Some(Err("panicked".into()));
        }
    }
}

fn level_bytes(l: &Image) -> usize {
    match l {
        Image::Rgba8 { data, .. } | Image::Rgba16f { data, .. } | Image::Bc { data, .. } => data.len(),
    }
}

/// first couple of thousand distinct lines
fn trace(msg: impl FnOnce() -> String) {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    static LAST: Mutex<String> = Mutex::new(String::new());
    let m = msg();
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if *last != m && N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 2000 {
        log::line(&m);
        *last = m;
    }
}

#[cfg(windows)]
fn window_handles(hwnd: usize) -> Result<wgpu::SurfaceTargetUnsafe, String> {
    use raw_window_handle::{RawDisplayHandle, RawWindowHandle, Win32WindowHandle, WindowsDisplayHandle};
    use windows_sys::Win32::UI::WindowsAndMessaging::{GWLP_HINSTANCE, GetWindowLongPtrW};
    let mut h = Win32WindowHandle::new(std::num::NonZeroIsize::new(hwnd as isize).ok_or("null hwnd")?);
    h.hinstance = std::num::NonZeroIsize::new(unsafe { GetWindowLongPtrW(hwnd as _, GWLP_HINSTANCE) });
    Ok(wgpu::SurfaceTargetUnsafe::RawHandle { raw_display_handle: Some(RawDisplayHandle::Windows(WindowsDisplayHandle::new())), raw_window_handle: RawWindowHandle::Win32(h) })
}

#[cfg(not(windows))]
fn window_handles(_hwnd: usize) -> Result<wgpu::SurfaceTargetUnsafe, String> {
    Err("window surfaces are only supported on Windows".into())
}

fn texture_entry(binding: u32, dim: wgpu::TextureViewDimension, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: dim, multisampled: false },
        count: None,
    }
}

fn sampler_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None }
}

fn view_dimension(dim: u8) -> wgpu::TextureViewDimension {
    match dim {
        DIM_CUBE => wgpu::TextureViewDimension::Cube,
        DIM_VOLUME => wgpu::TextureViewDimension::D3,
        _ => wgpu::TextureViewDimension::D2,
    }
}

fn dim_slot(dim: u8) -> usize {
    match dim {
        DIM_CUBE => 1,
        DIM_VOLUME => 2,
        _ => 0,
    }
}

impl Renderer {
    fn new(hwnd: usize) -> Result<Renderer, String> {
        let desc = wgpu::InstanceDescriptor::new_without_display_handle();
        // dlss needs ngx's vulkan extensions
        #[cfg(feature = "dlss")]
        let mut dlss_support = dlss_wgpu::FeatureSupport::default();
        #[cfg(feature = "dlss")]
        let instance = {
            crate::dlss::prepare_environment();
            match dlss_wgpu::create_instance(crate::dlss::PROJECT_ID, &desc, &mut dlss_support) {
                Ok(i) => i,
                Err(e) => {
                    log::line(&format!("dlss: instance creation failed ({e:?}), continuing without"));
                    dlss_support.super_resolution_supported = false;
                    wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle())
                }
            }
        };
        #[cfg(not(feature = "dlss"))]
        let instance = wgpu::Instance::new(desc);
        let surface = unsafe { instance.create_surface_unsafe(window_handles(hwnd)?) }.map_err(|e| e.to_string())?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
        let info = adapter.get_info();
        log::line(&format!("renderer: adapter {:?} ({:?})", info.name, info.backend));
        let bc = adapter.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC);
        let lines = adapter.features().contains(wgpu::Features::POLYGON_MODE_LINE);
        let mut features = wgpu::Features::empty();
        if bc {
            features |= wgpu::Features::TEXTURE_COMPRESSION_BC;
        }
        if lines {
            features |= wgpu::Features::POLYGON_MODE_LINE;
        }
        let border = adapter.features().contains(wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER);
        if border {
            features |= wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER;
        }
        let ts = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let has_timestamps = adapter.features().contains(ts);
        let has_cache = adapter.features().contains(wgpu::Features::PIPELINE_CACHE);
        if has_cache {
            features |= wgpu::Features::PIPELINE_CACHE;
        }
        if has_timestamps {
            features |= ts;
        }
        let clip_distances = adapter.features().contains(wgpu::Features::CLIP_DISTANCES);
        if clip_distances {
            features |= wgpu::Features::CLIP_DISTANCES;
        }
        // sample counts other than 1 and 4
        let format_features = adapter.features().contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES);
        if format_features {
            features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        }
        let device_desc = wgpu::DeviceDescriptor {
            label: Some("shaderapiwgpu"),
            required_features: features,
            required_limits: adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        };
        #[cfg(feature = "dlss")]
        let (device, queue) = match dlss_support.super_resolution_supported.then(|| dlss_wgpu::request_device(crate::dlss::PROJECT_ID, &adapter, &device_desc, &mut dlss_support, None)) {
            Some(Ok(dq)) => dq,
            other => {
                if let Some(Err(e)) = other {
                    log::line(&format!("dlss: device creation failed ({e:?}), continuing without"));
                }
                dlss_support.super_resolution_supported = false;
                pollster::block_on(adapter.request_device(&device_desc)).map_err(|e| e.to_string())?
            }
        };
        #[cfg(not(feature = "dlss"))]
        let (device, queue) = pollster::block_on(adapter.request_device(&device_desc)).map_err(|e| e.to_string())?;
        #[cfg(feature = "dlss")]
        let dlss = if dlss_support.super_resolution_supported { crate::dlss::Dlss::new(&device) } else {
            log::line("dlss: not supported on this system");
            None
        };
        device.on_uncaptured_error(Arc::new(|e: wgpu::Error| trace(|| format!("wgpu error: {e}"))));
        let lost = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = lost.clone();
        device.set_device_lost_callback(move |reason, msg| {
            log::line(&format!("wgpu device lost ({reason:?}): {msg}"));
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        let (w, h) = crate::device::back_buffer_size();
        let mut config = surface.get_default_config(&adapter, w.max(1) as u32, h.max(1) as u32).ok_or("surface not supported by the adapter")?;
        // the game writes gamma-space colors, present them as is
        let caps = surface.get_capabilities(&adapter);
        if let Some(f) = caps.formats.iter().copied().find(|f| !f.is_srgb()) {
            config.format = f;
        }
        config.present_mode = present_mode();
        config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST;
        surface.configure(&device, &config);
        // color and depth must both support the count
        let mut msaa = 0u32;
        for n in [2u32, 4, 8, 16] {
            let ok = if format_features {
                adapter.get_texture_format_features(config.format).flags.sample_count_supported(n) && adapter.get_texture_format_features(DEPTH_FORMAT).flags.sample_count_supported(n)
            } else {
                n == 4
            };
            if ok {
                msaa |= 1 << n;
            }
        }
        MSAA_SUPPORTED.store(msaa, std::sync::atomic::Ordering::Relaxed);
        log::line(&format!("renderer: MSAA sample counts {:?}", (1..=16).filter(|n| msaa & (1 << n) != 0).collect::<Vec<_>>()));

        let uniform = |binding, visibility, size: usize| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: std::num::NonZeroU64::new(size as u64) },
            count: None,
        };
        let const_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("constants"),
            entries: &[uniform(0, wgpu::ShaderStages::VERTEX, VS_BLOCK), uniform(1, wgpu::ShaderStages::FRAGMENT, PS_BLOCK)],
        });

        let blit_rect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit rects"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let blit_tex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit source"),
            entries: &[texture_entry(0, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT), sampler_entry(1, wgpu::ShaderStages::FRAGMENT)],
        });
        let blit_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("blit"), source: wgpu::ShaderSource::Wgsl(BLIT.into()) });
        let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("blit"), bind_group_layouts: &[Some(&blit_rect_layout), Some(&blit_tex_layout)], immediate_size: 0 });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&blit_layout),
            vertex: wgpu::VertexState { module: &blit_module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &blit_module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let dummy = |dim: wgpu::TextureDimension, layers: u32, view_dim: wgpu::TextureViewDimension, texel: [u8; 4]| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("dummy"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: layers },
                mip_level_count: 1,
                sample_count: 1,
                dimension: dim,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let data: Vec<u8> = texel.iter().copied().cycle().take(4 * layers as usize).collect();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &data,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: layers },
            );
            tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(view_dim), ..Default::default() })
        };
        let dummies = [
            dummy(wgpu::TextureDimension::D2, 1, wgpu::TextureViewDimension::D2, [255; 4]),
            dummy(wgpu::TextureDimension::D2, 6, wgpu::TextureViewDimension::Cube, [0, 0, 0, 255]),
            dummy(wgpu::TextureDimension::D3, 1, wgpu::TextureViewDimension::D3, [255; 4]),
        ];

        let dummy_depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dummy depth"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        {
            let view = dummy_depth_tex.create_view(&Default::default());
            let mut enc = device.create_command_encoder(&Default::default());
            enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear dummy depth"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0), store: wgpu::StoreOp::Store }) }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            queue.submit([enc.finish()]);
        }
        let dummy_depth = dummy_depth_tex.create_view(&wgpu::TextureViewDescriptor { aspect: wgpu::TextureAspect::DepthOnly, ..Default::default() });
        let compare_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow compare"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let overlay = crate::overlay::Overlay::new(&device, config.format);
        let effects = crate::effects::Effects::new(&device, &queue);
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("occlusion"), ty: wgpu::QueryType::Occlusion, count: MAX_QUERIES });
        // vulkan pipeline cache persisted in the game directory
        let pipeline_cache = has_cache.then(|| {
            let data = std::fs::read(PIPELINE_CACHE_FILE).ok();
            unsafe { device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor { label: Some("pipelines"), data: data.as_deref(), fallback: true }) }
        });
        let timestamps = has_timestamps.then(|| device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("frame time"), ty: wgpu::QueryType::Timestamp, count: 2 }));
        let compiler = Compiler::new(&device);
        Ok(Renderer {
            #[cfg(feature = "dlss")]
            dlss,
            #[cfg(feature = "dlss")]
            debug_view: None,
            #[cfg(feature = "dlss")]
            debug_snapshot: None,
            wgpu_adapter: adapter,
            overlay,
            effects,
            query_set,
            adapter: info.name.clone(),
            backend: format!("{:?}", info.backend),
            _instance: instance,
            surface,
            device,
            queue,
            config,
            shaders: ShaderStore::default(),
            vs_modules: FxHashMap::default(),
            ps_modules: FxHashMap::default(),
            pipelines: FxHashMap::default(),
            compiler,
            code: FxHashMap::default(),
            gpu_us: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            timestamps,
            pipeline_cache,
            unsaved_pipelines: 0,
            const_layout,
            tex_layouts: FxHashMap::default(),
            bind_groups: FxHashMap::default(),
            blit_pipeline,
            blit_rect_layout,
            blit_tex_layout,
            samplers: FxHashMap::default(),
            border,
            anisotropy: 1,
            clip_distances,
            textures: HashMap::new(),
            dummies,
            bc,
            lines,
            lost,
            meshes: HashMap::new(),
            targets: HashMap::new(),
            depths: HashMap::new(),
            failed: HashMap::new(),
            back_msaa: None,
            shadow_view: None,
            shadow_maps: HashMap::new(),
            shadow_serial: 0,
            dummy_depth,
            compare_sampler,
        })
    }

    fn buffer(&self, label: &str, data: &[u8], min: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let size = (data.len().max(min).max(256) as u64).next_multiple_of(4);
        let b = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        if !data.is_empty() {
            self.queue.write_buffer(&b, 0, data);
        }
        b
    }

    fn sampler(&mut self, key: SamplerKey) -> wgpu::Sampler {
        let border = self.border;
        self.samplers
            .entry(key)
            .or_insert_with(|| {
                let f = |linear: bool| if linear { wgpu::FilterMode::Linear } else { wgpu::FilterMode::Nearest };
                let address = |w: u8| match w {
                    1 => wgpu::AddressMode::Repeat,
                    2 if border => wgpu::AddressMode::ClampToBorder,
                    _ => wgpu::AddressMode::ClampToEdge,
                };
                self.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: None,
                    address_mode_u: address(key.wrap[0]),
                    address_mode_v: address(key.wrap[1]),
                    address_mode_w: address(key.wrap[2]),
                    mag_filter: f(key.mag),
                    min_filter: f(key.min),
                    mipmap_filter: if key.mip_linear { wgpu::MipmapFilterMode::Linear } else { wgpu::MipmapFilterMode::Nearest },
                    // D3DTEXF_NONE mip filter: top level only
                    lod_max_clamp: if key.mips { 32.0 } else { 0.0 },
                    anisotropy_clamp: key.aniso.max(1),
                    border_color: (key.wrap.contains(&2) && border).then_some(wgpu::SamplerBorderColor::TransparentBlack),
                    ..Default::default()
                })
            })
            .clone()
    }

    /// mips are the leading run of same-kind levels shared by every face
    fn create_texture(&mut self, faces: &[Vec<Option<Image>>], signed: bool, version: u64) -> Option<GpuTexture> {
        let base = faces.first()?.first()?.as_ref()?;
        let (w, h) = base.size();
        let (format, block) = match base {
            Image::Rgba8 { .. } if signed => (wgpu::TextureFormat::Rgba8Snorm, None),
            Image::Rgba8 { .. } => (wgpu::TextureFormat::Rgba8Unorm, None),
            Image::Rgba16f { .. } => (wgpu::TextureFormat::Rgba16Float, None),
            Image::Bc { kind, .. } if self.bc && w % 4 == 0 && h % 4 == 0 => (
                match kind {
                    1 => wgpu::TextureFormat::Bc1RgbaUnorm,
                    2 => wgpu::TextureFormat::Bc2RgbaUnorm,
                    7 => wgpu::TextureFormat::Bc7RgbaUnorm,
                    _ => wgpu::TextureFormat::Bc3RgbaUnorm,
                },
                Some(if *kind == 1 { 8u32 } else { 16 }),
            ),
            Image::Bc { .. } => return None,
        };
        let cube = faces.len() == 6 && w == h && faces.iter().all(|f| f.first().is_some_and(|l| l.is_some()));
        let faces = if cube { faces } else { &faces[..1] };
        let mut count = u32::MAX;
        for levels in faces {
            let mut n = 0u32;
            for (i, l) in levels.iter().enumerate() {
                let Some(l) = l else { break };
                let expect = ((w >> i).max(1), (h >> i).max(1));
                let same = std::mem::discriminant(base) == std::mem::discriminant(l);
                if !same || l.size() != expect {
                    break;
                }
                n += 1;
            }
            count = count.min(n);
        }
        let layers = faces.len() as u32;
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: layers },
            mip_level_count: count.max(1),
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (layer, levels) in faces.iter().enumerate() {
            for (i, l) in levels.iter().take(count as usize).enumerate() {
                let l = l.as_ref().unwrap();
                let (lw, lh) = l.size();
                let (data, bytes_per_row, rows) = match l {
                    Image::Rgba8 { data, .. } => (data, lw * 4, lh),
                    Image::Rgba16f { data, .. } => (data, lw * 8, lh),
                    Image::Bc { data, .. } => (data, lw.div_ceil(4) * block.unwrap(), lh.div_ceil(4)),
                };
                // bc mips smaller than a block copy at the rounded-up size
                let extent = if block.is_some() { wgpu::Extent3d { width: lw.div_ceil(4) * 4, height: lh.div_ceil(4) * 4, depth_or_array_layers: 1 } } else { wgpu::Extent3d { width: lw, height: lh, depth_or_array_layers: 1 } };
                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: i as u32, origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 }, aspect: wgpu::TextureAspect::All },
                    data,
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bytes_per_row), rows_per_image: Some(rows) },
                    extent,
                );
            }
        }
        let view = tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), base_array_layer: 0, array_layer_count: Some(1), ..Default::default() });
        let cube_view = cube.then(|| tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::Cube), ..Default::default() }));
        let partial_ok = block.is_none();
        Some(GpuTexture { version, texture: tex, partial_ok, view, cube_view, volume_view: None })
    }

    fn create_volume(&mut self, slices: &[Option<Image>], version: u64) -> Option<GpuTexture> {
        let mut data = Vec::new();
        let mut size = None;
        for s in slices {
            let Some(Image::Rgba8 { width, height, data: d }) = s else { return None };
            if size.is_some_and(|s| s != (*width, *height)) {
                return None;
            }
            size = Some((*width, *height));
            data.extend_from_slice(d);
        }
        let (w, h) = size?;
        let extent = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: slices.len() as u32 };
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            extent,
        );
        let view = self.dummies[0].clone();
        let volume_view = Some(tex.create_view(&Default::default()));
        Some(GpuTexture { version, texture: tex, partial_ok: false, view, cube_view: None, volume_view })
    }

    /// up to 256 MiB for textures the frame uses, then 48 MiB of unused ones so turning around
    /// doesn't upload hundreds at once
    fn sync_textures(&mut self, used: &[i32]) -> (usize, usize) {
        const BUDGET: usize = 256 << 20;
        const EAGER_BUDGET: usize = 48 << 20;
        enum Job {
            Full { id: i32, version: u64, faces: Vec<Vec<Option<Image>>>, signed: bool, volume: bool, desc: String },
            Partial { id: i32, version: u64, rects: Vec<(u32, u32, [u32; 4], Vec<u8>)> },
        }
        let jobs: Vec<Job> = (|| {
            let mut table = crate::api::texture_table();
            let mut bytes = 0usize;
            let mut seen = rustc_hash::FxHashSet::default();
            let mut jobs = Vec::new();
            let eager: Vec<i32> = table.keys().copied().collect();
            let mut queue = |id: i32, budget: usize, bytes: &mut usize| {
                if id == 0 || !seen.insert(id) || *bytes > budget {
                    return;
                }
                let Some(t) = table.get_mut(&id) else { return };
                let gpu = self.textures.get(&id);
                let stale = gpu.is_none_or(|g| g.version != t.version) && self.failed.get(&id) != Some(&t.version);
                if !stale || t.version == 0 {
                    return;
                }
                if gpu.is_some_and(|g| g.partial_ok && !t.full_dirty && !t.rects.is_empty()) {
                    let mut rects = Vec::new();
                    for (face, level, [x, y, w, h]) in t.rects.drain(..) {
                        let levels = if face == 0 { &t.levels } else { &t.faces[face as usize - 1] };
                        let Some(Some(img)) = levels.get(level as usize) else { continue };
                        let (Image::Rgba8 { width, data, .. } | Image::Rgba16f { width, data, .. }) = img else { continue };
                        let bpp = img.pixel_bytes() as u32;
                        let mut px = Vec::with_capacity((w * h * bpp) as usize);
                        for row in y..y + h {
                            let o = ((row * width + x) * bpp) as usize;
                            px.extend_from_slice(&data[o..o + (w * bpp) as usize]);
                        }
                        *bytes += px.len();
                        rects.push((face as u32, level as u32, [x, y, w, h], px));
                    }
                    jobs.push(Job::Partial { id, version: t.version, rects });
                    return;
                }
                let volume = t.depth > 1;
                let mut faces = if volume { vec![t.slices.clone()] } else { vec![t.levels.clone()] };
                if t.is_cube() && !volume {
                    faces.extend(t.faces.iter().cloned());
                }
                t.full_dirty = false;
                t.rects.clear();
                *bytes += faces.iter().flatten().flatten().map(level_bytes).sum::<usize>();
                let desc = format!("'{}' {}x{}x{} fmt {} mips {} flags {:#x}", t.name, t.width, t.height, t.depth, t.format, t.mips, t.flags);
                jobs.push(Job::Full { id, version: t.version, faces, signed: crate::texconv::is_signed(t.format), volume, desc });
            };
            for &id in used {
                queue(id, BUDGET, &mut bytes);
            }
            let mut eager_bytes = 0usize;
            for id in eager {
                queue(id, EAGER_BUDGET, &mut eager_bytes);
            }
            jobs
        })();
        let n = jobs.len();
        let mut tex_bytes = 0usize;
        for job in jobs {
            match job {
                Job::Partial { id, version, rects } => {
                    let Some(g) = self.textures.get_mut(&id) else { continue };
                    for (face, level, [x, y, w, h], px) in rects {
                        if level >= g.texture.mip_level_count() || x + w > (g.texture.width() >> level).max(1) || y + h > (g.texture.height() >> level).max(1) {
                            continue;
                        }
                        // rgba8 vs half floats
                        let bpp = g.texture.format().block_copy_size(None).unwrap_or(0);
                        if bpp == 0 || px.len() as u32 != w * h * bpp {
                            continue;
                        }
                        tex_bytes += px.len();
                        self.queue.write_texture(
                            wgpu::TexelCopyTextureInfo { texture: &g.texture, mip_level: level, origin: wgpu::Origin3d { x, y, z: face }, aspect: wgpu::TextureAspect::All },
                            &px,
                            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * bpp), rows_per_image: Some(h) },
                            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                        );
                    }
                    g.version = version;
                }
                Job::Full { id, version, faces, signed, volume, desc } => {
                    tex_bytes += faces.iter().flatten().flatten().map(level_bytes).sum::<usize>();
                    let created = if volume { self.create_volume(&faces[0], version) } else { self.create_texture(&faces, signed, version) };
                    match created {
                        Some(t) => {
                            self.textures.insert(id, t);
                            self.failed.remove(&id);
                        }
                        None => {
                            static FAILS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                            if FAILS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 40 {
                                log::line(&format!("renderer: texture {id} not uploadable: {desc}"));
                            }
                            self.failed.insert(id, version);
                        }
                    }
                }
            }
        }
        (n, tex_bytes)
    }

    fn module(&self, label: &str, wgsl: &str) -> Option<wgpu::ShaderModule> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let m = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(wgsl.into()) });
        match pollster::block_on(scope.pop()) {
            None => Some(m),
            Some(e) => {
                log::line(&format!("renderer: shader {label} rejected: {e}"));
                None
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn vs_module(&mut self, code: &Bytecode, format: u64, color_stream: bool, clip: bool, flex: bool, mv: bool, name: &str) -> Option<Arc<VsModule>> {
        let key = (Arc::as_ptr(code) as usize, format, color_stream, clip, flex, mv);
        if let Some(m) = self.vs_modules.get(&key) {
            return m.clone();
        }
        let elements = vertex_elements(format);
        let present = |usage: u8, index: u8| (color_stream && usage == 10 && index == 1) || (flex && index == 1 && (usage == 0 || usage == 3)) || elements.iter().any(|e| e.usage == usage && e.index == index);
        let translated = if name.starts_with(crate::builtin::PREFIX) { crate::builtin::translate_vs(name, &present, clip, mv) } else { crate::d3dbc::parse(code).and_then(|s| d3d2wgsl::translate_vs(&s, &present, clip, mv)) };
        let m = match translated {
            Ok(t) => self.module(name, &t.wgsl).map(|module| Arc::new(VsModule { module, inputs: t.inputs, samplers: t.samplers })),
            Err(e) => {
                log::line(&format!("renderer: can't translate {name}: {e}"));
                None
            }
        };
        self.vs_modules.insert(key, m.clone());
        m
    }

    fn ps_module(&mut self, code: &Bytecode, opts: PsOptions, mv: bool, name: &str) -> Option<Arc<PsModule>> {
        let key = (Arc::as_ptr(code) as usize, opts, mv);
        if let Some(m) = self.ps_modules.get(&key) {
            return m.clone();
        }
        let translated = if name.starts_with(crate::builtin::PREFIX) { crate::builtin::translate_ps(name, &opts, mv) } else { crate::d3dbc::parse(code).and_then(|s| d3d2wgsl::translate_ps(&s, &opts, mv)) };
        let m = match translated {
            Ok(t) => self.module(name, &t.wgsl).map(|module| Arc::new(PsModule { module, samplers: t.samplers })),
            Err(e) => {
                log::line(&format!("renderer: can't translate {name}: {e}"));
                None
            }
        };
        self.ps_modules.insert(key, m.clone());
        m
    }

    /// d3d9's null pixel shader (depth-only passes)
    fn null_ps(&mut self, mv: bool) -> Option<Arc<PsModule>> {
        let key = (0usize, PsOptions::default(), mv);
        if let Some(m) = self.ps_modules.get(&key) {
            return m.clone();
        }
        let wgsl = if mv {
            "struct O { @location(0) c: vec4f, @location(1) mv: vec2f }\n@fragment fn main() -> O { return O(vec4f(0.0), vec2f(0.0)); }\n"
        } else {
            "@fragment fn main() -> @location(0) vec4f { return vec4f(0.0); }\n"
        };
        let m = self.module("null pixel shader", wgsl).map(|module| Arc::new(PsModule { module, samplers: Vec::new() }));
        self.ps_modules.insert(key, m.clone());
        m
    }

    fn tex_layout(&mut self, sig: &Arc<SamplerSig>) -> &(wgpu::BindGroupLayout, wgpu::PipelineLayout) {
        if !self.tex_layouts.contains_key(sig) {
            let vis = wgpu::ShaderStages::VERTEX_FRAGMENT;
            let entries: Vec<wgpu::BindGroupLayoutEntry> = sig
                .iter()
                .flat_map(|&(b, dim)| {
                    if dim == DIM_DEPTH {
                        [
                            wgpu::BindGroupLayoutEntry { binding: b, visibility: vis, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                            wgpu::BindGroupLayoutEntry { binding: b + 1, visibility: vis, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison), count: None },
                        ]
                    } else {
                        [texture_entry(b, view_dimension(dim), vis), sampler_entry(b + 1, vis)]
                    }
                })
                .collect();
            let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("textures"), entries: &entries });
            let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&self.const_layout), Some(&bgl)], immediate_size: 0 });
            self.tex_layouts.insert(sig.clone(), (bgl, pl));
        }
        &self.tex_layouts[sig]
    }

    #[cfg(feature = "dlss")]
    fn draw_debug_view(&mut self, enc: &mut wgpu::CommandEncoder, out: &wgpu::TextureView, mv: &wgpu::TextureView, depth: &wgpu::TextureView, mode: u32, size: (u32, u32)) {
        if self.debug_view.is_none() {
            let vis = wgpu::ShaderStages::FRAGMENT;
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dlss debug view"),
                entries: &[
                    wgpu::BindGroupLayoutEntry { binding: 0, visibility: vis, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 1, visibility: vis, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 2, visibility: vis, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                ],
            });
            let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("dlss debug view"), source: wgpu::ShaderSource::Wgsl(crate::dlss::DEBUG_VIEW_WGSL.into()) });
            let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
            let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("dlss debug view"),
                layout: Some(&pl),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: self.config.format, blend: None, write_mask: wgpu::ColorWrites::ALL })] }),
                multiview_mask: None,
                cache: None,
            });
            self.debug_view = Some((pipeline, layout));
        }
        let (pipeline, layout) = self.debug_view.as_ref().unwrap();
        let params: [u32; 4] = [mode, (size.0 as f32).to_bits(), (size.1 as f32).to_bits(), 0];
        let buf = self.buffer("debug view params", bytemuck::cast_slice(&params), 16, wgpu::BufferUsages::UNIFORM);
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(mv) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 2, resource: buf.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dlss debug view"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: out, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..4, 0..1);
    }

    fn save_pipeline_cache(&mut self, background: bool) {
        if self.unsaved_pipelines == 0 {
            return;
        }
        if let Some(data) = self.pipeline_cache.as_ref().and_then(|c| c.get_data()) {
            self.unsaved_pipelines = 0;
            let write = move || {
                let tmp = format!("{PIPELINE_CACHE_FILE}.tmp");
                if std::fs::write(&tmp, &data).is_ok() {
                    let _ = std::fs::rename(&tmp, PIPELINE_CACHE_FILE);
                }
            };
            if background {
                std::thread::spawn(write);
            } else {
                write();
            }
        }
    }

    fn bytecode(&mut self, r: &crate::api::ShaderRef) -> Option<Bytecode> {
        let key = (r.name, r.static_index, r.dynamic);
        if let Some(c) = self.code.get(&key) {
            return c.clone();
        }
        let name = r.name.get();
        // built-in shaders have no bytecode: a unique handle per combo keeps the caches keyed
        let c = if name.starts_with(crate::builtin::PREFIX) { Some(Arc::new(vec![r.static_index as u32, r.dynamic as u32])) } else { self.shaders.get(&name, r.static_index, r.dynamic) };
        self.code.insert(key, c.clone());
        c
    }

    fn pipeline(&mut self, d: &DrawCmd, mv: bool, samples: u32) -> Option<&Pipeline> {
        let vs_code = self.bytecode(&d.vs)?;
        let ps_code = if d.ps.name.is_none() { None } else { Some(self.bytecode(&d.ps)?) };
        let id = PipelineId { vs: Arc::as_ptr(&vs_code) as usize, format: d.format, stride: d.stride, ps: ps_code.as_ref().map_or(0, |c| Arc::as_ptr(c) as usize), ps_opts: d.ps_opts, key: d.key, color: d.color.map(|c| (c.2, c.3)), clip: d.clip && self.clip_distances, flex: d.flex.is_some(), mv, samples };
        if !self.pipelines.contains_key(&id) {
            let slot = match self.build_pipeline(d, &vs_code, ps_code.as_ref(), mv, samples) {
                Some((job, sig)) => {
                    let cell = Arc::new(Mutex::new(None));
                    self.compiler.compile(job, cell.clone());
                    PipeSlot::Pending(sig, cell)
                }
                None => {
                    trace(|| format!("renderer: no pipeline for {}/{}+{} + {}/{}+{} (format {:#x})", d.vs.name, d.vs.static_index, d.vs.dynamic, d.ps.name, d.ps.static_index, d.ps.dynamic, d.format));
                    PipeSlot::Failed
                }
            };
            self.pipelines.insert(id.clone(), slot);
        }
        let slot = self.pipelines.get_mut(&id)?;
        if let PipeSlot::Pending(sig, cell) = slot {
            let done = cell.lock().unwrap_or_else(|e| e.into_inner()).take();
            if let Some(result) = done {
                *slot = match result {
                    Some(pipeline) => {
                        self.unsaved_pipelines += 1;
                        PipeSlot::Ready(Pipeline { pipeline, sig: sig.clone() })
                    }
                    None => PipeSlot::Failed,
                };
            }
        }
        match slot {
            PipeSlot::Ready(p) => Some(p),
            _ => None,
        }
    }

    /// everything up to the driver compile, which runs on a worker (see `PipelineJob`)
    fn build_pipeline(&mut self, d: &DrawCmd, vs_code: &Bytecode, ps_code: Option<&Bytecode>, mv: bool, samples: u32) -> Option<(PipelineJob, Arc<SamplerSig>)> {
        let clip = d.clip && self.clip_distances;
        let vs = self.vs_module(vs_code, d.format, d.color.is_some(), clip, d.flex.is_some(), mv, &format!("{}/{}+{}", d.vs.name, d.vs.static_index, d.vs.dynamic))?;
        // a2c replaces the alpha test, like the driver does for d3d9's atoc
        let a2c = samples > 1 && d.key.alpha_to_coverage;
        let ps_opts = if a2c { PsOptions { alpha_test: None, ..d.ps_opts } } else { d.ps_opts };
        let ps = match ps_code {
            Some(code) => self.ps_module(code, ps_opts, mv, &format!("{}/{}+{}", d.ps.name, d.ps.static_index, d.ps.dynamic))?,
            None => self.null_ps(mv)?,
        };
        let mut sig: SamplerSig = ps.samplers.iter().map(|&(n, dim)| (2 * n, dim)).chain(vs.samplers.iter().map(|&(n, dim)| (32 + 2 * n, dim))).collect();
        sig.sort();
        let sig = Arc::new(sig);
        let elements = vertex_elements(d.format);
        let color_attr: Vec<wgpu::VertexAttribute> = match d.color {
            Some((_, _, _, offset)) => vs.inputs.iter().filter(|i| i.1 == 10 && i.2 == 1).map(|i| wgpu::VertexAttribute { format: wgpu::VertexFormat::Unorm8x4, offset: offset as u64, shader_location: i.0 }).collect(),
            None => Vec::new(),
        };
        // POSITION1 = delta + wrinkle, NORMAL1 at +16
        let flex_attr: Vec<wgpu::VertexAttribute> = if d.flex.is_some() {
            vs.inputs
                .iter()
                .filter_map(|&(loc, usage, index)| match (usage, index) {
                    (0, 1) => Some(wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: loc }),
                    (3, 1) => Some(wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 16, shader_location: loc }),
                    _ => None,
                })
                .collect()
        } else {
            Vec::new()
        };
        let attributes: Vec<wgpu::VertexAttribute> = vs
            .inputs
            .iter()
            .filter(|i| d.color.is_none() || !(i.1 == 10 && i.2 == 1))
            .filter(|i| d.flex.is_none() || !(i.2 == 1 && (i.1 == 0 || i.1 == 3)))
            .filter_map(|&(loc, usage, index)| {
                let e = elements.iter().find(|e| e.usage == usage && e.index == index)?;
                let format = match e.ty {
                    ElementType::Color => wgpu::VertexFormat::Unorm8x4,
                    ElementType::Float(1) => wgpu::VertexFormat::Float32,
                    ElementType::Float(2) => wgpu::VertexFormat::Float32x2,
                    ElementType::Float(3) => wgpu::VertexFormat::Float32x3,
                    ElementType::Float(_) => wgpu::VertexFormat::Float32x4,
                };
                Some(wgpu::VertexAttribute { format, offset: e.offset as u64, shader_location: loc })
            })
            .collect();
        let k = d.key;
        let factor = |(src, dst, op): (i32, i32, i32)| wgpu::BlendComponent { src_factor: blend_factor(src), dst_factor: blend_factor(dst), operation: blend_op(op) };
        let blend = k.blend.map(|c| wgpu::BlendState { color: factor(c), alpha: factor(k.blend_alpha.unwrap_or(c)) });
        let mut write_mask = wgpu::ColorWrites::empty();
        if k.color_write {
            write_mask |= wgpu::ColorWrites::COLOR;
        }
        if k.alpha_write {
            write_mask |= wgpu::ColorWrites::ALPHA;
        }
        let format = self.config.format;
        let layout = self.tex_layout(&sig).1.clone();
        let job = PipelineJob {
            label: format!("{}/{}+{} + {}/{}+{}", d.vs.name, d.vs.static_index, d.vs.dynamic, d.ps.name, d.ps.static_index, d.ps.dynamic),
            layout,
            vs: vs.module.clone(),
            ps: ps.module.clone(),
            buffers: [
                Some((d.stride as u64, attributes)),
                d.color.map(|c| (c.2 as u64, color_attr)),
                d.flex.map(|_| (28, flex_attr)),
            ],
            primitive: wgpu::PrimitiveState {
                topology: match k.topology {
                    0 => wgpu::PrimitiveTopology::PointList,
                    1 => wgpu::PrimitiveTopology::LineList,
                    _ => wgpu::PrimitiveTopology::TriangleList,
                },
                // d3d9 culls by screen-space winding
                front_face: wgpu::FrontFace::Cw,
                cull_mode: match k.cull {
                    1 => Some(wgpu::Face::Back),
                    2 => Some(wgpu::Face::Front),
                    _ => None,
                },
                polygon_mode: match k.poly_mode {
                    0 | 1 if self.lines => wgpu::PolygonMode::Line,
                    _ => wgpu::PolygonMode::Fill,
                },
                ..Default::default()
            },
            depth_stencil: wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(k.depth_write && k.depth_test),
                depth_compare: Some(if k.depth_test { depth_func(k.depth_func) } else { wgpu::CompareFunction::Always }),
                stencil: match k.stencil {
                    Some((func, fail, zfail, pass, read_mask, write_mask)) => {
                        let face = wgpu::StencilFaceState { compare: stencil_func(func), fail_op: stencil_op(fail), depth_fail_op: stencil_op(zfail), pass_op: stencil_op(pass) };
                        wgpu::StencilState { front: face, back: face, read_mask: read_mask as u32, write_mask: write_mask as u32 }
                    }
                    None => Default::default(),
                },
                // decal bias from mat_*_decal; shadow bias from SetShadowDepthBiasFactors (d24 has
                // 2^24 steps)
                bias: match k.poly_offset {
                    1 => wgpu::DepthBiasState { constant: -64, slope_scale: -2.0, clamp: 0.0 },
                    2 => wgpu::DepthBiasState { constant: (f32::from_bits(k.shadow_bias.1) as f64 * 16777216.0).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32, slope_scale: f32::from_bits(k.shadow_bias.0), clamp: 0.0 },
                    _ => Default::default(),
                },
            },
            multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: a2c },
            // blended draws keep the motion vectors behind them
            targets: [
                Some(wgpu::ColorTargetState { format, blend, write_mask }),
                mv.then_some(wgpu::ColorTargetState { format: crate::dlss::MV_FORMAT, blend: None, write_mask: if blend.is_some() { wgpu::ColorWrites::empty() } else { wgpu::ColorWrites::ALL } }),
            ],
            cache: self.pipeline_cache.clone(),
        };
        Some((job, sig))
    }

    /// cached by texture ids, versions and sampler states
    fn bind_group(&mut self, sig: &Arc<SamplerSig>, d: &DrawCmd, target: i32, states: &HashMap<i32, ([i32; 2], [i32; 3])>) -> wgpu::BindGroup {
        let aniso = self.anisotropy;
        // (texture id or 0, version tag, sampler state)
        let resolve = |r: &Self, binding: u32, dim: u8| -> (i32, u64, SamplerKey) {
            let reg = if binding >= 32 { 16 + (binding - 32) / 2 } else { binding / 2 } as usize;
            let id = d.textures.get(reg).copied().unwrap_or(0);
            let mut state = states.get(&id).copied().unwrap_or(([5, 1], [1, 1, 1]));
            // GMOD_SamplerBorderClamp
            if binding < 32 && d.border & (1 << reg) != 0 {
                state.1 = [2, 2, 2];
            }
            let key = SamplerKey::new(state.0, state.1, d.filter_override, aniso);
            if id == 0 || id == target {
                return (0, 0, key);
            }
            if dim == DIM_DEPTH {
                return r.shadow_maps.get(&id).map_or((0, 0, key), |m| (id, (1 << 62) | m.serial, key));
            }
            if let Some(t) = r.targets.get(&id) {
                return if dim == DIM_2D { (id, (1 << 63) | ((t.size.0 as u64) << 32) | t.size.1 as u64, key) } else { (0, 0, key) };
            }
            match r.textures.get(&id) {
                Some(t) if (dim == DIM_2D && t.volume_view.is_none()) || (dim == DIM_CUBE && t.cube_view.is_some()) || (dim == DIM_VOLUME && t.volume_view.is_some()) => (id, t.version, key),
                _ => (0, 0, key),
            }
        };
        let ids: Vec<(i32, u64, SamplerKey)> = sig.iter().map(|&(b, dim)| resolve(self, b, dim)).collect();
        let key = (sig.clone(), ids);
        if let Some(bg) = self.bind_groups.get(&key) {
            return bg.clone();
        }
        if self.bind_groups.len() > 8192 {
            self.bind_groups.clear();
        }
        let mut resources: Vec<(u32, wgpu::TextureView, wgpu::Sampler)> = Vec::new();
        for (&(b, dim), &(id, _, skey)) in sig.iter().zip(key.1.iter()) {
            if dim == DIM_DEPTH {
                let view = self.shadow_maps.get(&id).map_or_else(|| self.dummy_depth.clone(), |m| m.sample_view.clone());
                resources.push((b, view, self.compare_sampler.clone()));
                continue;
            }
            let sampler = self.sampler(skey);
            let view = if id == 0 {
                self.dummies[dim_slot(dim)].clone()
            } else if let Some(t) = self.targets.get(&id) {
                t.view.clone()
            } else {
                let t = &self.textures[&id];
                match dim {
                    DIM_CUBE => t.cube_view.clone().unwrap(),
                    DIM_VOLUME => t.volume_view.clone().unwrap(),
                    _ => t.view.clone(),
                }
            };
            resources.push((b, view, sampler));
        }
        let entries: Vec<wgpu::BindGroupEntry> = resources
            .iter()
            .flat_map(|(b, v, s)| [wgpu::BindGroupEntry { binding: *b, resource: wgpu::BindingResource::TextureView(v) }, wgpu::BindGroupEntry { binding: b + 1, resource: wgpu::BindingResource::Sampler(s) }])
            .collect();
        let layout = self.tex_layout(sig).0.clone();
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries });
        self.bind_groups.insert(key, bg.clone());
        bg
    }

    fn target(&mut self, id: i32, size: (u32, u32)) {
        let format = self.config.format;
        if self.targets.get(&id).is_some_and(|t| t.size != size) {
            self.targets.remove(&id);
        }
        self.targets.entry(id).or_insert_with(|| {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("render target"),
                size: wgpu::Extent3d { width: size.0.max(1), height: size.1.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            Target { texture, view, size }
        });
    }

    /// one per target: sharing one let water reflection depth occlude the main view
    fn depth_for(&mut self, id: i32, size: (u32, u32)) -> wgpu::TextureView {
        if self.depths.get(&id).is_none_or(|(s, _)| *s != size) {
            let d = depth_view(&self.device, size.0.max(1), size.1.max(1));
            self.depths.insert(id, (size, d));
        }
        self.depths[&id].1.clone()
    }

    /// `present` = false is a mid-frame flush for ReadPixels
    fn draw(&mut self, frame: Frame, present: bool) {
        let cpu_start = std::time::Instant::now();
        let pipelines_before = self.pipelines.len();
        let (w, h) = crate::device::back_buffer_size();
        let (w, h) = (w.max(1) as u32, h.max(1) as u32);
        let mode = present_mode();
        if (w, h) != (self.config.width, self.config.height) || mode != self.config.present_mode {
            self.config.width = w;
            self.config.height = h;
            self.config.present_mode = mode;
            self.surface.configure(&self.device, &self.config);
        }
        for id in crate::api::take_destroyed() {
            self.meshes.remove(&id);
        }
        if present {
            self.effects.poll(&self.device, self.config.format);
        }
        let mut upload_bytes = 0usize;
        for u in &frame.uploads {
            let vb = self.buffer("mesh vertices", &u.vertices, 0, wgpu::BufferUsages::VERTEX);
            let ib = self.buffer("mesh indices", bytemuck::cast_slice(&u.indices), 0, wgpu::BufferUsages::INDEX);
            upload_bytes += u.vertices.len() + u.indices.len() * 4;
            self.meshes.insert(u.id, (vb, ib));
        }
        if !frame.uploads.is_empty() {
            trace(|| format!("renderer: uploaded {} static meshes ({:.1} MiB), {} resident", frame.uploads.len(), upload_bytes as f64 / 1048576.0, self.meshes.len()));
        }
        let t = std::time::Instant::now();
        let acquired = present.then(|| self.surface.get_current_texture());
        ACQUIRE_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let surface_tex = if let Some(acquired) = acquired {
            match acquired {
                wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Some(t),
                _ => {
                    self.surface.configure(&self.device, &self.config);
                    return;
                }
            }
        } else {
            None
        };

        let t_acquire = cpu_start.elapsed().as_secs_f32() * 1000.0;
        let mut ids: Vec<i32> = Vec::new();
        let mut used: Vec<i32> = Vec::new();
        let mut depth_ids: Vec<i32> = Vec::new();
        let mut copy_srcs: Vec<i32> = Vec::new();
        for c in &frame.cmds {
            match c {
                Cmd::Target { color, depth } => {
                    if *color != BACK {
                        ids.push(*color);
                    }
                    if *depth > 0 {
                        depth_ids.push(*depth);
                    }
                }
                Cmd::Copy { dst, .. } => ids.push(*dst),
                Cmd::CopyIn { src, .. } => {
                    used.push(*src);
                    copy_srcs.push(*src);
                }
                Cmd::Draw(d) => used.extend(d.textures.iter().filter(|t| **t != 0)),
                _ => {}
            }
        }
        let (src_sizes, sizes, depth_sizes) = {
            let table = crate::api::texture_table();
            let size = |id: &i32| table.get(id).map(|t| (*id, (t.width.max(1) as u32, t.height.max(1) as u32)));
            let src: HashMap<i32, (u32, u32)> = copy_srcs.iter().filter_map(size).collect();
            let rts: HashMap<i32, (u32, u32)> = ids.iter().filter_map(size).collect();
            let depth: Vec<(i32, (u32, u32))> = depth_ids.iter().filter(|id| table.get(id).is_some_and(|t| crate::api::is_depth_format(t.format))).filter_map(size).collect();
            (src, rts, depth)
        };
        for (id, size) in depth_sizes {
            if self.shadow_maps.get(&id).is_none_or(|m| m.size != size) {
                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("shadow depth"),
                    size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: DEPTH_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                self.shadow_serial += 1;
                let view = texture.create_view(&Default::default());
                let sample_view = texture.create_view(&wgpu::TextureViewDescriptor { aspect: wgpu::TextureAspect::DepthOnly, ..Default::default() });
                self.shadow_maps.insert(id, ShadowMap { size, view, sample_view, serial: self.shadow_serial });
            }
        }
        let (tex_uploads, tex_bytes) = self.sync_textures(&used);
        let states: HashMap<i32, ([i32; 2], [i32; 3])> = {
            let table = crate::api::texture_table();
            used.iter().filter_map(|id| table.get(id).map(|t| (*id, (t.filter, t.wrap)))).collect()
        };
        let aniso = crate::api::ANISOTROPY.load(std::sync::atomic::Ordering::Relaxed);
        self.anisotropy = aniso;
        let t_sync = cpu_start.elapsed().as_secs_f32() * 1000.0;
        self.target(BACK, (w, h));
        self.target(DISCARD, (w, h));
        let want = crate::device::aa_samples();
        let samples = if want > 1 && MSAA_SUPPORTED.load(std::sync::atomic::Ordering::Relaxed) & (1 << want.min(31)) != 0 { want } else { 1 };
        if samples == 1 {
            self.back_msaa = None;
        } else if self.back_msaa.as_ref().is_none_or(|m| m.samples != samples || m.size != (w, h)) {
            let make = |format: wgpu::TextureFormat, label: &str| {
                self.device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some(label),
                        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: samples,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        // readable: screen effects resolve the depth
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
                    .create_view(&Default::default())
            };
            self.back_msaa = Some(BackMsaa { samples, size: (w, h), color: make(self.config.format, "back buffer msaa"), depth: make(DEPTH_FORMAT, "back buffer msaa depth") });
            log::line(&format!("renderer: MSAA {samples}x"));
        }
        for (id, size) in &sizes {
            self.target(*id, *size);
        }

        let limits = self.device.limits();
        if frame.vertices.len() as u64 > limits.max_buffer_size || frame.indices.len() as u64 * 4 > limits.max_buffer_size {
            log::line("renderer: frame geometry exceeds the buffer size limit, dropped");
            return;
        }
        let vbuf = self.buffer("vertices", &frame.vertices, 0, wgpu::BufferUsages::VERTEX);
        let ibuf = self.buffer("indices", bytemuck::cast_slice(&frame.indices), 0, wgpu::BufferUsages::INDEX);
        // padded so a full-size binding at the last (short) block stays in bounds
        let vs_consts = self.buffer("vs constants", &frame.vs_blocks, frame.vs_blocks.len() + VS_BLOCK, wgpu::BufferUsages::UNIFORM);
        let ps_consts = self.buffer("ps constants", &frame.ps_blocks, PS_BLOCK, wgpu::BufferUsages::UNIFORM);
        let const_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.const_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &vs_consts, offset: 0, size: std::num::NonZeroU64::new(VS_BLOCK as u64) }) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &ps_consts, offset: 0, size: std::num::NonZeroU64::new(PS_BLOCK as u64) }) },
            ],
        });

        // the scene ends at the last perspective back buffer draw; passes up to there write motion
        // vectors
        #[cfg(feature = "dlss")]
        // dlss needs single-sampled inputs
        let dlss_frame = present && samples == 1 && frame.jitter.is_some() && self.dlss.as_mut().is_some_and(|d| d.enabled && d.ensure(&self.device, &self.queue, (w, h)));
        #[cfg(not(feature = "dlss"))]
        let dlss_frame = false;
        let scene_end: Option<usize> = if dlss_frame { find_scene_end(&frame.cmds) } else { None };
        // dlss upscaling: back buffer work up to the scene's end goes into the top-left render-size
        // part (draws, scissors and copies scaled by this)
        #[cfg(feature = "dlss")]
        let up: Option<[f32; 2]> = self
            .dlss
            .as_ref()
            .filter(|_| dlss_frame)
            .map(|d| [d.render_size().0 as f32 / w as f32, d.render_size().1 as f32 / h as f32])
            .filter(|s| *s != [1.0, 1.0]);
        #[cfg(not(feature = "dlss"))]
        let up: Option<[f32; 2]> = None;
        let scaled = |draws_before: usize| up.filter(|_| scene_end.is_some_and(|e| draws_before <= e));

        let mut rects: Vec<[f32; 4]> = Vec::new();
        let mut target = BACK;
        let mut blit_info: Vec<(i32, i32, [f32; 4], bool)> = Vec::new(); // src, dst, viewport, src is a texture
        let mut draws_before = 0usize;
        for c in &frame.cmds {
            match c {
                Cmd::Draw(_) => draws_before += 1,
                Cmd::Target { color, .. } => target = if *color == BACK { BACK } else if sizes.contains_key(color) { *color } else { DISCARD },
                Cmd::Copy { dst, src_rect, dst_rect } => {
                    let src_size = self.targets.get(&target).map_or((w, h), |t| t.size);
                    let (sw, sh) = (src_size.0 as f32, src_size.1 as f32);
                    let mut r = src_rect.map_or([0.0, 0.0, 1.0, 1.0], |r| [r[0] as f32 / sw, r[1] as f32 / sh, (r[0] + r[2]) as f32 / sw, (r[1] + r[3]) as f32 / sh]);
                    if let (BACK, Some([sx, sy])) = (target, scaled(draws_before)) {
                        r = [r[0] * sx, r[1] * sy, r[2] * sx, r[3] * sy];
                    }
                    rects.push(r);
                    let size = sizes.get(dst).copied().unwrap_or((w, h));
                    let vp = dst_rect.map_or([0.0, 0.0, size.0 as f32, size.1 as f32], |r| [r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32]);
                    blit_info.push((target, *dst, vp, false));
                }
                Cmd::CopyIn { src, src_rect, dst_rect } => {
                    let (sw, sh) = src_sizes.get(src).map_or((1.0, 1.0), |s| (s.0 as f32, s.1 as f32));
                    rects.push(src_rect.map_or([0.0, 0.0, 1.0, 1.0], |r| [r[0] as f32 / sw, r[1] as f32 / sh, (r[0] + r[2]) as f32 / sw, (r[1] + r[3]) as f32 / sh]));
                    let size = if target == BACK { (w, h) } else { sizes.get(&target).copied().unwrap_or((w, h)) };
                    let mut vp = dst_rect.map_or([0.0, 0.0, size.0 as f32, size.1 as f32], |r| [r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32]);
                    if let (BACK, Some([sx, sy])) = (target, scaled(draws_before)) {
                        vp = [vp[0] * sx, vp[1] * sy, vp[2] * sx, vp[3] * sy];
                    }
                    blit_info.push((*src, target, vp, true));
                }
                _ => {}
            }
        }

        // screen effects run after the same draw (single-sampled only: they read the depth buffer)
        let fx_end: Option<usize> = if present && crate::effects::any_enabled() { find_scene_end(&frame.cmds) } else { None };
        rects.push([0.0, 0.0, 1.0, 1.0]);
        let full_rect = rects.len() as u32 - 1;
        #[cfg(not(feature = "dlss"))]
        let _ = full_rect;
        let mut target = BACK;
        let mut draw_index = 0usize;
        let mut prepared: Vec<Option<(wgpu::RenderPipeline, wgpu::BindGroup)>> = Vec::new();
        struct Last<'a> {
            d: &'a DrawCmd,
            mv: bool,
            samples: u32,
            target: i32,
            pipe: Option<(wgpu::RenderPipeline, Arc<SamplerSig>)>,
            bind: Option<wgpu::BindGroup>,
        }
        let mut last: Option<Last> = None;
        let mut failed_draws = 0usize;
        for c in &frame.cmds {
            match c {
                Cmd::Target { color, .. } => target = if *color == BACK { BACK } else if sizes.contains_key(color) { *color } else { DISCARD },
                Cmd::Draw(d) => {
                    let mv = target == BACK && scene_end.is_some_and(|e| draw_index <= e);
                    draw_index += 1;
                    let draw_samples = if target == BACK { samples } else { 1 };
                    // consecutive draws mostly share pipeline and textures: reuse without hashing
                    let same_pipe = last.as_ref().is_some_and(|l: &Last| l.mv == mv && l.samples == draw_samples && same_pipeline(l.d, d));
                    let p = if same_pipe {
                        last.as_ref().unwrap().pipe.clone()
                    } else {
                        self.pipeline(d, mv, draw_samples).map(|p| (p.pipeline.clone(), p.sig.clone()))
                    };
                    let same_tex = same_pipe && last.as_ref().is_some_and(|l| l.target == target && l.d.textures == d.textures && l.d.border == d.border && l.d.filter_override == d.filter_override);
                    let entry = match &p {
                        Some((pipeline, sig)) => Some((pipeline.clone(), if same_tex { last.as_ref().unwrap().bind.clone().unwrap() } else { self.bind_group(sig, d, target, &states) })),
                        None => None,
                    };
                    last = Some(Last { d, mv, samples: draw_samples, target, pipe: p, bind: entry.as_ref().map(|e| e.1.clone()) });
                    prepared.push(entry);
                    if prepared.last().unwrap().is_none() {
                        failed_draws += 1;
                        if frame.capture {
                            log::line(&format!("  renderer: draw {} has no pipeline ({}/{}+{} + {}/{}+{})", prepared.len() - 1, d.vs.name, d.vs.static_index, d.vs.dynamic, d.ps.name, d.ps.static_index, d.ps.dynamic));
                        }
                    }
                }
                _ => {}
            }
        }
        let t_prepare = cpu_start.elapsed().as_secs_f32() * 1000.0;
        drop(last);
        let rect_buf = self.buffer("blit rects", bytemuck::cast_slice(&rects), 0, wgpu::BufferUsages::STORAGE);
        let rect_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.blit_rect_layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: rect_buf.as_entire_binding() }] });
        trace(|| format!("renderer: frame {} draws ({} without a pipeline), {} copies, {} targets, {} pipelines", prepared.len(), failed_draws, rects.len(), sizes.len(), self.pipelines.len()));

        let mut depth_by_target: HashMap<i32, wgpu::TextureView> = HashMap::new();
        let all: Vec<(i32, (u32, u32))> = self.targets.iter().map(|(id, t)| (*id, t.size)).collect();
        for (id, size) in all {
            let d = self.depth_for(id, size);
            depth_by_target.insert(id, d);
        }

        // a target switch, clear or copy ends the current pass
        enum Item<'a> {
            Draw(usize, &'a DrawCmd),
            QueryBegin(u32),
            QueryEnd,
        }
        enum Step<'a> {
            /// shadow map as depth attachment
            Pass { target: i32, depth_tex: Option<i32>, color: wgpu::LoadOp<wgpu::Color>, depth: wgpu::LoadOp<f32>, stencil: wgpu::LoadOp<u32>, mv: bool, draws: Vec<Item<'a>> },
            Blit { blit: usize },
            Dlss,
            Effects,
        }
        let mut dlss_pending = scene_end.is_some();
        let mut query_slots: Vec<(u64, u64)> = Vec::new();
        let mut open_query: Option<u64> = None;
        let mut steps: Vec<Step> = Vec::new();
        let mut target = BACK;
        // only when its size matches the color target
        let mut depth_tex: Option<i32> = None;
        let (mut draw_no, mut blit_no) = (0usize, 0usize);
        for c in &frame.cmds {
            match c {
                // unknown targets go to scratch, never the screen
                Cmd::Target { color, depth } => {
                    target = if *color == BACK { BACK } else if sizes.contains_key(color) { *color } else { DISCARD };
                    let tsize = if target == BACK { (w, h) } else { sizes.get(&target).copied().unwrap_or((w, h)) };
                    depth_tex = self.shadow_maps.get(depth).filter(|m| m.size == tsize && target != BACK).map(|_| *depth);
                }
                Cmd::Clear { color, depth, stencil } => {
                    let cl = color.map_or(wgpu::LoadOp::Load, |c| wgpu::LoadOp::Clear(wgpu::Color { r: c[0] as f64 / 255.0, g: c[1] as f64 / 255.0, b: c[2] as f64 / 255.0, a: c[3] as f64 / 255.0 }));
                    let dl = if *depth { wgpu::LoadOp::Clear(1.0) } else { wgpu::LoadOp::Load };
                    let sl = stencil.map_or(wgpu::LoadOp::Load, |v| wgpu::LoadOp::Clear(v as u32));
                    steps.push(Step::Pass { target, depth_tex, color: cl, depth: dl, stencil: sl, mv: target == BACK && dlss_pending, draws: Vec::new() });
                }
                Cmd::Draw(d) => {
                    if !matches!(steps.last(), Some(Step::Pass { target: t, depth_tex: dt, .. }) if *t == target && *dt == depth_tex) {
                        steps.push(Step::Pass { target, depth_tex, color: wgpu::LoadOp::Load, depth: wgpu::LoadOp::Load, stencil: wgpu::LoadOp::Load, mv: target == BACK && dlss_pending, draws: Vec::new() });
                    }
                    if let Some(Step::Pass { draws, .. }) = steps.last_mut() {
                        draws.push(Item::Draw(draw_no, d));
                    }
                    if Some(draw_no) == scene_end {
                        steps.push(Step::Dlss);
                        dlss_pending = false;
                    }
                    if Some(draw_no) == fx_end {
                        steps.push(Step::Effects);
                    }
                    draw_no += 1;
                }
                Cmd::QueryBegin { id, issue } => {
                    if query_slots.len() < MAX_QUERIES as usize {
                        if !matches!(steps.last(), Some(Step::Pass { target: t, depth_tex: dt, .. }) if *t == target && *dt == depth_tex) {
                            steps.push(Step::Pass { target, depth_tex, color: wgpu::LoadOp::Load, depth: wgpu::LoadOp::Load, stencil: wgpu::LoadOp::Load, mv: target == BACK && dlss_pending, draws: Vec::new() });
                        }
                        if let Some(Step::Pass { draws, .. }) = steps.last_mut() {
                            draws.push(Item::QueryBegin(query_slots.len() as u32));
                        }
                        query_slots.push((*id, *issue));
                        open_query = Some(*id);
                    }
                }
                Cmd::QueryEnd { id } => {
                    if open_query == Some(*id) {
                        if let Some(Step::Pass { draws, .. }) = steps.last_mut() {
                            draws.push(Item::QueryEnd);
                        }
                        open_query = None;
                    }
                }
                Cmd::Copy { dst, .. } => {
                    if sizes.contains_key(dst) && *dst != target {
                        steps.push(Step::Blit { blit: blit_no });
                    }
                    blit_no += 1;
                }
                Cmd::CopyIn { src, .. } => {
                    if src_sizes.contains_key(src) && *src != target && target != DISCARD {
                        steps.push(Step::Blit { blit: blit_no });
                    }
                    blit_no += 1;
                }
            }
        }

        let query_set = (!query_slots.is_empty()).then(|| self.query_set.clone());
        let mut enc = self.device.create_command_encoder(&Default::default());
        if let Some(ts) = &self.timestamps {
            enc.write_timestamp(ts, 0);
        }
        // dlss records its own buffer between ours
        let mut cmd_bufs: Vec<wgpu::CommandBuffer> = Vec::new();
        #[cfg(feature = "dlss")]
        let mv_view = if dlss_frame { self.dlss.as_ref().and_then(|d| d.mv.as_ref()).map(|m| m.1.clone()) } else { None };
        #[cfg(not(feature = "dlss"))]
        let mv_view: Option<wgpu::TextureView> = None;
        let mut mv_cleared = false;
        for step in &steps {
            match step {
                #[cfg(feature = "dlss")]
                Step::Dlss => {
                    let color = self.targets[&BACK].view.clone();
                    let depth = depth_by_target[&BACK].texture().create_view(&wgpu::TextureViewDescriptor { aspect: wgpu::TextureAspect::DepthOnly, ..Default::default() });
                    let jitter = frame.jitter.unwrap_or([0.0, 0.0]);
                    // capture inputs now, later passes may clear depth
                    crate::dlss::poll_debug_view_key();
                    let mode = crate::dlss::DEBUG_VIEW.load(std::sync::atomic::Ordering::Relaxed);
                    let mv = self.dlss.as_ref().and_then(|d| d.mv.as_ref()).map(|m| m.1.clone());
                    if let (true, Some(mv)) = (mode != 0, mv) {
                        let size = self.targets[&BACK].size;
                        if self.debug_snapshot.as_ref().is_none_or(|(t, _)| (t.width(), t.height()) != size) {
                            let t = self.device.create_texture(&wgpu::TextureDescriptor {
                                label: Some("dlss debug snapshot"),
                                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: wgpu::TextureDimension::D2,
                                format: self.config.format,
                                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                                view_formats: &[],
                            });
                            let v = t.create_view(&Default::default());
                            self.debug_snapshot = Some((t, v));
                        }
                        let out = self.debug_snapshot.as_ref().unwrap().1.clone();
                        self.draw_debug_view(&mut enc, &out, &mv, &depth, mode, size);
                    }
                    let (Some(dlss), adapter) = (self.dlss.as_mut(), &self.wgpu_adapter) else { continue };
                    let Some(cb) = dlss.evaluate(&mut enc, adapter, &color, &depth, jitter) else { continue };
                    let out = dlss.output.as_ref().unwrap().1.clone();
                    let finished = std::mem::replace(&mut enc, self.device.create_command_encoder(&Default::default()));
                    cmd_bufs.push(finished.finish());
                    cmd_bufs.push(cb);
                    let sampler = self.sampler(SamplerKey::new([0, 0], [0, 0, 0], None, 1));
                    let src_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.blit_tex_layout,
                        entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&out) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) }],
                    });
                    let back = &self.targets[&BACK];
                    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("dlss resolve"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &back.view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &depth_by_target[&BACK], depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }), stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }) }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(&self.blit_pipeline);
                    pass.set_bind_group(0, &rect_group, &[]);
                    pass.set_bind_group(1, &src_group, &[]);
                    pass.set_viewport(0.0, 0.0, back.size.0 as f32, back.size.1 as f32, 0.0, 1.0);
                    pass.draw(0..4, full_rect..full_rect + 1);
                }
                #[cfg(not(feature = "dlss"))]
                Step::Dlss => {}
                Step::Effects => {
                    let back = &self.targets[&BACK];
                    let (color, color_view) = (back.texture.clone(), back.view.clone());
                    let depth = depth_by_target[&BACK].texture().create_view(&wgpu::TextureViewDescriptor { aspect: wgpu::TextureAspect::DepthOnly, ..Default::default() });
                    let inputs = crate::effects::Inputs { color: &color, color_view: &color_view, depth: &depth, motion: mv_view.as_ref(), jitter: frame.jitter.filter(|_| dlss_frame), depth_size: up.map(|[sx, sy]| (((w as f32 * sx).round() as u32).max(1), ((h as f32 * sy).round() as u32).max(1))), camera: frame.camera };
                    match &self.back_msaa {
                        Some(m) => {
                            let (mc, md, n) = (m.color.clone(), m.depth.texture().create_view(&wgpu::TextureViewDescriptor { aspect: wgpu::TextureAspect::DepthOnly, ..Default::default() }), m.samples);
                            self.effects.run_msaa(&self.device, &self.queue, &mut enc, &inputs, &mc, &md, n);
                        }
                        None => self.effects.run(&self.device, &self.queue, &mut enc, &inputs),
                    }
                }
                Step::Pass { target, depth_tex, color, depth, stencil, mv, draws } => {
                    let t = &self.targets[target];
                    let msaa = self.back_msaa.as_ref().filter(|_| *target == BACK);
                    let depth_view = match (msaa, depth_tex.and_then(|id| self.shadow_maps.get(&id))) {
                        (_, Some(m)) => &m.view,
                        (Some(m), None) => &m.depth,
                        (None, None) => &depth_by_target[target],
                    };
                    let (color_view, resolve) = match msaa {
                        Some(m) => (&m.color, Some(&t.view)),
                        None => (&t.view, None),
                    };
                    let mv_attachment = mv_view.as_ref().filter(|_| *mv).map(|v| {
                        let load = if mv_cleared { wgpu::LoadOp::Load } else { wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT) };
                        mv_cleared = true;
                        wgpu::RenderPassColorAttachment { view: v, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load, store: wgpu::StoreOp::Store } }
                    });
                    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: color_view, depth_slice: None, resolve_target: resolve, ops: wgpu::Operations { load: *color, store: wgpu::StoreOp::Store } }), mv_attachment],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: depth_view, depth_ops: Some(wgpu::Operations { load: *depth, store: wgpu::StoreOp::Store }), stencil_ops: Some(wgpu::Operations { load: *stencil, store: wgpu::StoreOp::Store }) }),
                        timestamp_writes: None,
                        occlusion_query_set: query_set.as_ref(),
                        multiview_mask: None,
                    });
                    let (tw, th) = (t.size.0 as i32, t.size.1 as i32);
                    // a query left open only counts its pass
                    let mut active = false;
                    // state already set in this pass, so unchanged state isn't re-sent
                    #[derive(Default)]
                    struct Cur<'a> {
                        pipeline: Option<&'a wgpu::RenderPipeline>,
                        consts: Option<(u32, u32)>,
                        textures: Option<&'a wgpu::BindGroup>,
                        stencil_ref: Option<u32>,
                        viewport: Option<([i32; 4], u32, u32)>,
                        scissor: Option<[u32; 4]>,
                        vertex: Option<u64>,
                        color: Option<(u64, u64)>,
                        index: Option<u64>,
                    }
                    let mut cur = Cur::default();
                    for item in draws {
                        let (n, d) = match item {
                            Item::QueryBegin(slot) => {
                                if active {
                                    pass.end_occlusion_query();
                                }
                                pass.begin_occlusion_query(*slot);
                                active = true;
                                continue;
                            }
                            Item::QueryEnd => {
                                if active {
                                    pass.end_occlusion_query();
                                    active = false;
                                }
                                continue;
                            }
                            Item::Draw(n, d) => (n, d),
                        };
                        let Some((pipeline, textures)) = &prepared[*n] else { continue };
                        // frame vertices start at a multiple of their stride, so the frame buffer
                        // stays bound and draws pass a base vertex
                        let (vb_key, ib_key, base_vertex) = match d.geometry {
                            Geometry::Frame { vertex_offset } => (0u64, 0u64, (vertex_offset / d.stride.max(1) as u64) as i32),
                            Geometry::StaticVertices { mesh } => (mesh, 0, 0),
                            Geometry::Static { mesh } => (mesh, mesh, 0),
                        };
                        if vb_key != 0 && !self.meshes.contains_key(&vb_key) {
                            continue;
                        }
                        if cur.pipeline != Some(pipeline) {
                            pass.set_pipeline(pipeline);
                            cur.pipeline = Some(pipeline);
                        }
                        let consts = (d.vs_block, d.ps_block);
                        if cur.consts != Some(consts) {
                            pass.set_bind_group(0, &const_group, &[d.vs_block * 256, (d.ps_block as usize * PS_BLOCK) as u32]);
                            cur.consts = Some(consts);
                        }
                        if cur.textures != Some(textures) {
                            pass.set_bind_group(1, textures, &[]);
                            cur.textures = Some(textures);
                        }
                        if d.key.stencil.is_some() && cur.stencil_ref != Some(d.stencil_ref) {
                            pass.set_stencil_reference(d.stencil_ref);
                            cur.stencil_ref = Some(d.stencil_ref);
                        }
                        let mut viewport = d.viewport;
                        let mut scissor = d.scissor;
                        // the part of the target this draw may cover
                        let (mut fw, mut fh) = (tw, th);
                        if let (BACK, Some([sx, sy])) = (*target, scaled(*n)) {
                            let s = |v: i32, k: f32| (v as f32 * k).round() as i32;
                            viewport = [s(viewport[0], sx), s(viewport[1], sy), s(viewport[2], sx), s(viewport[3], sy)];
                            scissor = Some(scissor.map_or([0, 0, s(tw, sx), s(th, sy)], |[l, t, r, b]| [s(l, sx), s(t, sy), s(r, sx), s(b, sy)]));
                            (fw, fh) = (s(tw, sx), s(th, sy));
                        }
                        let [vx, vy, vw, vh] = viewport;
                        let (x0, y0) = (vx.clamp(0, tw), vy.clamp(0, th));
                        let (x1, y1) = ((vx + vw).clamp(0, tw), (vy + vh).clamp(0, th));
                        let vp = if vw > 0 && vh > 0 && x1 > x0 && y1 > y0 { [x0, y0, x1 - x0, y1 - y0] } else { [0, 0, fw, fh] };
                        let vp_key = (vp, d.depth[0].to_bits(), d.depth[1].to_bits());
                        if cur.viewport != Some(vp_key) {
                            pass.set_viewport(vp[0] as f32, vp[1] as f32, vp[2] as f32, vp[3] as f32, d.depth[0], d.depth[1]);
                            cur.viewport = Some(vp_key);
                        }
                        let sc = match scissor {
                            Some([l, t, r, b]) => {
                                let (l, t) = (l.clamp(0, tw), t.clamp(0, th));
                                let (r, b) = (r.clamp(l, tw), b.clamp(t, th));
                                [l as u32, t as u32, (r - l) as u32, (b - t) as u32]
                            }
                            None => [0, 0, tw as u32, th as u32],
                        };
                        if cur.scissor != Some(sc) {
                            pass.set_scissor_rect(sc[0], sc[1], sc[2], sc[3]);
                            cur.scissor = Some(sc);
                        }
                        if cur.vertex != Some(vb_key) {
                            if vb_key == 0 {
                                pass.set_vertex_buffer(0, vbuf.slice(..));
                            } else {
                                pass.set_vertex_buffer(0, self.meshes[&vb_key].0.slice(..));
                            }
                            cur.vertex = Some(vb_key);
                        }
                        if let Some(offset) = d.flex {
                            pass.set_vertex_buffer(2, vbuf.slice(offset..));
                        }
                        if let Some((mesh, offset, _, _)) = d.color {
                            if cur.color != Some((mesh, offset)) {
                                let Some((cb, _)) = self.meshes.get(&mesh) else { continue };
                                pass.set_vertex_buffer(1, cb.slice(offset..));
                                cur.color = Some((mesh, offset));
                            }
                        }
                        if cur.index != Some(ib_key) {
                            let ib = if ib_key == 0 { &ibuf } else { &self.meshes[&ib_key].1 };
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            cur.index = Some(ib_key);
                        }
                        pass.draw_indexed(d.first_index..d.first_index + d.index_count, base_vertex, 0..1);
                    }
                    if active {
                        pass.end_occlusion_query();
                    }
                }
                Step::Blit { blit } => {
                    let (src, dst, vp, from_texture) = blit_info[*blit];
                    let sampler = self.sampler(SamplerKey::new([1, 1], [0, 0, 0], None, 1));
                    let src_view = match (self.targets.get(&src), from_texture) {
                        (Some(t), _) => t.view.clone(),
                        (None, true) => match self.textures.get(&src) {
                            Some(t) if t.volume_view.is_none() => t.view.clone(),
                            _ => continue,
                        },
                        (None, false) => continue,
                    };
                    let src_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.blit_tex_layout,
                        entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) }],
                    });
                    let d = &self.targets[&dst];
                    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("copy"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &d.view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &depth_by_target[&dst], depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }), stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }) }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    let (dw, dh) = (d.size.0 as f32, d.size.1 as f32);
                    let (x0, y0) = (vp[0].clamp(0.0, dw), vp[1].clamp(0.0, dh));
                    let (x1, y1) = ((vp[0] + vp[2]).clamp(0.0, dw), (vp[1] + vp[3]).clamp(0.0, dh));
                    if x1 > x0 && y1 > y0 {
                        pass.set_pipeline(&self.blit_pipeline);
                        pass.set_bind_group(0, &rect_group, &[]);
                        pass.set_bind_group(1, &src_group, &[]);
                        pass.set_viewport(x0, y0, x1 - x0, y1 - y0, 0.0, 1.0);
                        pass.draw(0..4, *blit as u32..*blit as u32 + 1);
                    }
                }
            }
        }
        let ts_read = self.timestamps.as_ref().filter(|_| present).map(|ts| {
            enc.write_timestamp(ts, 1);
            let resolve = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("ts resolve"), size: 16, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let read = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("ts read"), size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            enc.resolve_query_set(ts, 0..2, &resolve, 0);
            enc.copy_buffer_to_buffer(&resolve, 0, &read, 0, 16);
            read
        });
        let readback = query_set.as_ref().map(|qs| {
            let n = query_slots.len() as u64;
            let resolve = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("query resolve"), size: n * 8, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let read = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("query readback"), size: n * 8, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            enc.resolve_query_set(qs, 0..n as u32, &resolve, 0);
            enc.copy_buffer_to_buffer(&resolve, 0, &read, 0, n * 8);
            read
        });
        poll_shadow_view_key();
        let last_shadow = steps.iter().rev().find_map(|st| if let Step::Pass { depth_tex: Some(id), .. } = st { Some(*id) } else { None });
        if let (true, Some(id)) = (present && SHADOW_VIEW.load(std::sync::atomic::Ordering::Relaxed), last_shadow) {
            if self.shadow_view.is_none() {
                let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("shadow view"), source: wgpu::ShaderSource::Wgsl(SHADOW_VIEW_WGSL.into()) });
                let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("shadow view"),
                    entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None }],
                });
                let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
                let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("shadow view"),
                    layout: Some(&pl),
                    vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                    primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
                    depth_stencil: None,
                    multisample: Default::default(),
                    fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: self.config.format, blend: None, write_mask: wgpu::ColorWrites::ALL })] }),
                    multiview_mask: None,
                    cache: None,
                });
                self.shadow_view = Some((pipeline, layout));
            }
            let (pipeline, layout) = self.shadow_view.as_ref().unwrap();
            let map = &self.shadow_maps[&id];
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&map.sample_view) }] });
            let back = &self.targets[&BACK];
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow view"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &back.view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let side = (back.size.1 as f32 * 0.6).min(back.size.0 as f32 * 0.5);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.set_viewport(0.0, back.size.1 as f32 - side, side, side, 0.0, 1.0);
            pass.draw(0..4, 0..1);
        }
        if let Some(surface_tex) = &surface_tex {
            let back = &self.targets[&BACK];
            enc.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo { texture: &back.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyTextureInfo { texture: &surface_tex.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::Extent3d { width: back.size.0.min(self.config.width), height: back.size.1.min(self.config.height), depth_or_array_layers: 1 },
            );
        }
        let submit_ms = cpu_start.elapsed().as_secs_f32() * 1000.0;
        let new_pipelines = self.pipelines.len() - pipelines_before;
        if present && (submit_ms > 8.0 || frame.record_ms > 8.0) {
            static SLOW: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SLOW.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
                log::line(&format!(
                    "slow frame: record {:.1} ms, submit {submit_ms:.1} ms (acquire {t_acquire:.1}, textures {:.1}, pipelines/bind groups {:.1}, encode {:.1}); {new_pipelines} new pipelines, {tex_uploads} textures ({:.1} MiB), {} meshes ({:.1} MiB), {} draws",
                    frame.record_ms,
                    t_sync - t_acquire,
                    t_prepare - t_sync,
                    submit_ms - t_prepare, tex_bytes as f64 / 1048576.0, frame.uploads.len(), upload_bytes as f64 / 1048576.0, prepared.len()
                ));
            }
        }
        if present && self.unsaved_pipelines >= 32 {
            self.save_pipeline_cache(true);
        }
        let mut targets_drawn: Vec<i32> = steps.iter().filter_map(|s| if let Step::Pass { target, .. } = s { Some(*target) } else { None }).collect();
        targets_drawn.sort();
        targets_drawn.dedup();
        let stats = crate::overlay::Stats {
            cpu_ms: cpu_start.elapsed().as_secs_f32() * 1000.0,
            record_ms: frame.record_ms,
            dlss: {
                #[cfg(feature = "dlss")]
                let status = match &self.dlss {
                    Some(d) if dlss_frame => {
                        let j = frame.jitter.unwrap_or([0.0, 0.0]);
                        let flips = match (crate::dlss::FLIP_X.load(std::sync::atomic::Ordering::Relaxed), crate::dlss::FLIP_Y.load(std::sync::atomic::Ordering::Relaxed)) {
                            (false, false) => "",
                            (true, false) => " · x flipped",
                            (false, true) => " · y flipped",
                            (true, true) => " · x+y flipped",
                        };
                        format!("{} {}×{} → {}×{} · {} ({:+.2}, {:+.2}){flips}", crate::dlss::mode_name(d.mode()), d.render_size().0, d.render_size().1, d.size().0, d.size().1, crate::dlss::jitter_mode_name(), j[0], j[1])
                    }
                    Some(d) if d.enabled => format!("{} on, waiting for a scene", crate::dlss::mode_name(crate::dlss::MODE.load(std::sync::atomic::Ordering::Relaxed))),
                    Some(_) => "off (End toggles)".to_string(),
                    None => "unavailable (see log)".to_string(),
                };
                #[cfg(not(feature = "dlss"))]
                let status = "not built".to_string();
                status
            },
            gpu_ms: self.gpu_us.load(std::sync::atomic::Ordering::Relaxed) as f32 / 1000.0,
            vertex_bytes: frame.vertices.len(),
            indices: frame.indices.len(),
            vs_blocks: frame.vs_block_count,
            const_bytes: frame.vs_blocks.len() + frame.ps_blocks.len(),
            ps_blocks: frame.ps_blocks.len() / PS_BLOCK,
            copies: rects.len(),
            mesh_uploads: frame.uploads.len(),
            mesh_upload_bytes: upload_bytes,
            tex_uploads,
            tex_bytes,
            new_pipelines: self.pipelines.len() - pipelines_before,
            bind_groups: self.bind_groups.len(),
            targets_drawn: targets_drawn.len(),
            skipped: frame.skipped,
            no_shader: frame.skipped_no_shader,
            adapter: self.adapter.clone(),
            backend: self.backend.clone(),
            present_mode: format!("{:?}", self.config.present_mode),
            size: (self.config.width, self.config.height),
            draws: prepared.len(),
            failed_draws,
            passes: steps.len(),
            pipelines: self.pipelines.len(),
            shader_modules: self.vs_modules.len() + self.ps_modules.len(),
            textures: self.textures.len(),
            meshes: self.meshes.len(),
            targets: self.targets.len(),
        };
        #[cfg(feature = "dlss")]
        if let (Some(surface_tex), true, Some((snap, _))) = (&surface_tex, dlss_frame && crate::dlss::DEBUG_VIEW.load(std::sync::atomic::Ordering::Relaxed) != 0, &self.debug_snapshot) {
            let (sw, sh) = (snap.width().min(self.config.width), snap.height().min(self.config.height));
            enc.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo { texture: snap, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyTextureInfo { texture: &surface_tex.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::Extent3d { width: sw, height: sh, depth_or_array_layers: 1 },
            );
        }
        if let Some(surface_tex) = &surface_tex {
            let surface_view = surface_tex.texture.create_view(&Default::default());
            self.overlay.draw(&self.device, &self.queue, &mut enc, &surface_view, &stats);
        }
        cmd_bufs.push(enc.finish());
        self.queue.submit(cmd_bufs);
        // takes effect next frame
        #[cfg(feature = "dlss")]
        if present {
            if let Some(d) = self.dlss.as_mut() {
                crate::dlss::poll_flip_keys();
                crate::dlss::poll_debug_keys();
                crate::dlss::poll_jitter_mode_key();
                if crate::dlss::toggle_pressed() {
                    d.enabled = !d.enabled;
                    log::line(&format!("dlss: {}", if d.enabled { "on" } else { "off" }));
                }
                if !dlss_frame {
                    d.skipped();
                }
                crate::dlss::set_active(d.enabled);
            }
        }
        if let Some(read) = ts_read {
            let (buf, out, period) = (read.clone(), self.gpu_us.clone(), self.queue.get_timestamp_period());
            read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                if r.is_ok() {
                    if let Ok(data) = buf.slice(..).get_mapped_range() {
                        let t: &[u64] = bytemuck::cast_slice(&data);
                        let us = (t[1].saturating_sub(t[0]) as f64 * period as f64 / 1000.0) as u32;
                        out.store(us, std::sync::atomic::Ordering::Relaxed);
                        GPU_US.store(us, std::sync::atomic::Ordering::Relaxed);
                    }
                    buf.unmap();
                }
            });
        }
        if let Some(read) = readback {
            let buf = read.clone();
            read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                if r.is_ok() {
                    if let Ok(data) = buf.slice(..).get_mapped_range() {
                        let counts: &[u64] = bytemuck::cast_slice(&data);
                        for ((id, issue), n) in query_slots.iter().zip(counts) {
                            crate::api::query_result(*id, *issue, *n);
                        }
                    }
                    buf.unmap();
                }
            });
        }
        if let Some(surface_tex) = surface_tex {
            let t = std::time::Instant::now();
            self.queue.present(surface_tex);
            PRESENT_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        let _ = self.device.poll(wgpu::PollType::Poll);
    }
}

/// the draw (index among draws) that ends the 3d scene: the last viewmodel draw (the 3d view
/// always ends with it; lua passes after it, like halos, redraw models and must not count), else
/// the last perspective back buffer draw
fn find_scene_end(cmds: &[Cmd]) -> Option<usize> {
    let scene = || cmds.iter().filter_map(|c| if let Cmd::Draw(d) = c { Some(d) } else { None }).enumerate().filter(|(_, d)| d.scene);
    // viewmodels draw with their depth range squeezed to [0, 0.1]
    scene().filter(|(_, d)| d.depth[1] < 0.15 && d.depth[1] > 0.02).map(|(i, _)| i).last().or_else(|| scene().map(|(i, _)| i).last())
}

/// ShaderDeviceInfo_t vsync flag (bit 4)
/// the fields `PipelineId` is built from (minus mv / samples, compared separately)
fn same_pipeline(a: &DrawCmd, b: &DrawCmd) -> bool {
    let shader = |x: &crate::api::ShaderRef, y: &crate::api::ShaderRef| x == y;
    shader(&a.vs, &b.vs)
        && shader(&a.ps, &b.ps)
        && a.format == b.format
        && a.stride == b.stride
        && a.ps_opts == b.ps_opts
        && a.key == b.key
        && a.color.map(|c| (c.2, c.3)) == b.color.map(|c| (c.2, c.3))
        && a.clip == b.clip
        && a.flex.is_some() == b.flex.is_some()
}

fn present_mode() -> wgpu::PresentMode {
    let vsync = crate::device::WINDOW.lock().unwrap_or_else(|e| e.into_inner()).is_some_and(|w| w.info.flags & 0x10 != 0);
    if vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync }
}

fn depth_view(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn blend_factor(f: i32) -> wgpu::BlendFactor {
    use wgpu::BlendFactor as B;
    match f {
        0 => B::Zero,
        1 => B::One,
        2 => B::Dst,
        3 => B::OneMinusDst,
        4 => B::SrcAlpha,
        5 => B::OneMinusSrcAlpha,
        6 => B::DstAlpha,
        7 => B::OneMinusDstAlpha,
        8 => B::SrcAlphaSaturated,
        9 => B::Src,
        10 => B::OneMinusSrc,
        _ => B::One,
    }
}

fn blend_op(op: i32) -> wgpu::BlendOperation {
    match op {
        1 => wgpu::BlendOperation::Subtract,
        2 => wgpu::BlendOperation::ReverseSubtract,
        3 => wgpu::BlendOperation::Min,
        4 => wgpu::BlendOperation::Max,
        _ => wgpu::BlendOperation::Add,
    }
}

fn stencil_func(f: i32) -> wgpu::CompareFunction {
    use wgpu::CompareFunction as C;
    match f {
        1 => C::Never,
        2 => C::Less,
        3 => C::Equal,
        4 => C::LessEqual,
        5 => C::Greater,
        6 => C::NotEqual,
        7 => C::GreaterEqual,
        _ => C::Always,
    }
}

fn stencil_op(op: i32) -> wgpu::StencilOperation {
    use wgpu::StencilOperation as S;
    match op {
        2 => S::Zero,
        3 => S::Replace,
        4 => S::IncrementClamp,
        5 => S::DecrementClamp,
        6 => S::Invert,
        7 => S::IncrementWrap,
        8 => S::DecrementWrap,
        _ => S::Keep,
    }
}

fn depth_func(f: i32) -> wgpu::CompareFunction {
    use wgpu::CompareFunction as C;
    match f {
        0 => C::Never,
        1 => C::Less,
        2 => C::Equal,
        3 => C::LessEqual,
        4 => C::Greater,
        5 => C::NotEqual,
        6 => C::GreaterEqual,
        _ => C::Always,
    }
}
