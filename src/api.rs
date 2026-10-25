//! `IShaderAPI` + `IShaderDynamicAPI`. draws flow like dx9: IMesh::Draw -> the material's DrawMesh
//! -> the shader's RenderPass per pass, which records a [`DrawCmd`].

use std::collections::HashMap;
use std::ffi::{c_char, c_void};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::abi::*;
use crate::log;
use crate::mesh::Mesh;
use crate::shadow;
use crate::texconv::{self, Image};

/// slot 72 of gmod's CMaterial vtable (checked via its "No bound shader" string)
const MATERIAL_DRAW_MESH: usize = 72;
const MATERIAL_GET_VERTEX_FORMAT: usize = 20;
const MATERIAL_GET_NAME: usize = 0;

/// one pass of a material on a mesh range
#[derive(Clone, Debug)]
pub struct DrawCall {
    pub snapshot: i16,
    pub mesh: usize,
    pub first_index: i32,
    pub index_count: i32,
    pub vs_index: i32,
    pub ps_index: i32,
}

#[derive(Default, Debug)]
pub struct Texture {
    pub width: i32,
    pub height: i32,
    pub depth: i32,
    pub format: i32,
    pub mips: i32,
    pub copies: i32,
    pub flags: i32,
    pub depth_target: bool,
    pub name: String,
    /// texture group (TEXTURE_GROUP_*), for mat_texture_list
    pub group: String,
    pub levels: Vec<Option<Image>>,
    /// faces 1..5 of a cube
    pub faces: Vec<Vec<Option<Image>>>,
    /// mip 0 slices of a volume
    pub slices: Vec<Option<Image>>,
    /// full re-upload, or just these rects (lightmap updates touch small parts of big pages)
    pub full_dirty: bool,
    pub rects: Vec<(u8, u8, [u32; 4])>,
    pub version: u64,
    /// min / mag filter and s / t / u wrap
    pub filter: [i32; 2],
    pub wrap: [i32; 3],
}

impl Texture {
    pub fn is_cube(&self) -> bool {
        self.flags & 1 != 0
    }

    /// 0 for 2d textures
    pub fn face_mut(&mut self, face: i32) -> Option<&mut Vec<Option<Image>>> {
        match face {
            0 => Some(&mut self.levels),
            1..=5 if self.is_cube() => {
                if self.faces.len() < 5 {
                    self.faces.resize(5, Vec::new());
                }
                Some(&mut self.faces[face as usize - 1])
            }
            _ => None,
        }
    }

    fn set_level(&mut self, face: i32, level: i32, img: Image) {
        let Some(levels) = self.face_mut(face) else { return };
        let l = level.max(0) as usize;
        if levels.len() <= l {
            levels.resize(l + 1, None);
        }
        levels[l] = Some(img);
        self.version += 1;
        self.full_dirty = true;
    }

    fn touch(&mut self, face: i32, level: usize, rect: [u32; 4]) {
        self.version += 1;
        if self.rects.len() >= 64 {
            self.full_dirty = true;
        } else {
            self.rects.push((face as u8, level as u8, rect));
        }
    }
}

/// fixed-function state of a draw
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PipeKey {
    /// 0 points, 1 lines, 2 triangles
    pub topology: u8,
    pub blend: Option<(i32, i32, i32)>,
    /// else the color factors apply to alpha too
    pub blend_alpha: Option<(i32, i32, i32)>,
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_func: i32,
    pub color_write: bool,
    pub alpha_write: bool,
    /// 0 none, 1 cull ccw, 2 cull cw
    pub cull: u8,
    /// 1 = decal, 2 = shadow bias
    pub poly_offset: i32,
    /// slope, constant (f32 bits)
    pub shadow_bias: (u32, u32),
    /// 0 point, 1 line, 2 fill
    pub poly_mode: i32,
    /// (compare, fail, depth fail, pass, test mask, write mask), d3d enums
    pub stencil: Option<(i32, i32, i32, i32, u8, u8)>,
    /// under msaa, coverage replaces the alpha test
    pub alpha_to_coverage: bool,
}

/// the shadow state's name, its static index and the dynamic index at draw time
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShaderRef {
    pub name: shadow::ShaderName,
    pub static_index: i32,
    pub dynamic: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Geometry {
    Frame { vertex_offset: u64 },
    /// world batches: static vertices, frame indices
    StaticVertices { mesh: u64 },
    Static { mesh: u64 },
}

#[derive(Clone, Debug)]
pub struct DrawCmd {
    pub geometry: Geometry,
    pub first_index: u32,
    pub index_count: u32,
    pub format: VertexFormat,
    pub stride: u32,
    pub key: PipeKey,
    pub vs: ShaderRef,
    pub ps: ShaderRef,
    pub ps_opts: crate::d3d2wgsl::PsOptions,
    /// vs block offset in 256-byte units, ps block index
    pub vs_block: u32,
    pub ps_block: u32,
    /// 0 = none; vertex texture samplers in 16..20
    pub textures: [i32; 20],
    pub viewport: [i32; 4],
    pub depth: [f32; 2],
    pub stencil_ref: u32,
    /// left, top, right, bottom
    pub scissor: Option<[i32; 4]>,
    /// 1 point, 2 linear, 3 anisotropic
    pub filter_override: Option<i32>,
    /// GMOD_SamplerBorderClamp
    pub border: u32,
    pub clip: bool,
    /// perspective draw onto the back buffer: dlss jitters it, and the last one marks where the hud
    /// begins
    pub scene: bool,
    /// stream 2 (stride 28) copied into the frame's vertex data at this offset
    pub flex: Option<u64>,
    /// stream 1: (mesh id, byte offset, stride, element offset)
    pub color: Option<(u64, u64, u32, u32)>,
}

#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Cmd {
    /// render target 0 (-1 = back buffer) and its depth (-1 = own, else a shadow map)
    Target { color: i32, depth: i32 },
    /// copy the current target into a texture
    Copy { dst: i32, src_rect: Option<[i32; 4]>, dst_rect: Option<[i32; 4]> },
    /// copy a texture into the current target
    CopyIn { src: i32, src_rect: Option<[i32; 4]>, dst_rect: Option<[i32; 4]> },
    Clear { color: Option<[u8; 4]>, depth: bool, stencil: Option<u8> },
    Draw(DrawCmd),
    /// (query handle, issue number)
    QueryBegin { id: u64, issue: u64 },
    QueryEnd { id: u64 },
}

/// uniform offsets must be 256-aligned
pub const VS_BLOCK: usize = 4608;
pub const PS_BLOCK: usize = 1536;

pub struct MeshUpload {
    pub id: u64,
    pub vertices: Vec<u8>,
    pub indices: Vec<u32>,
}

/// handle -> (issue, samples passed)
pub static QUERY_RESULTS: Mutex<Option<HashMap<u64, (u64, u64)>>> = Mutex::new(None);

pub fn query_result(id: u64, issue: u64, samples: u64) {
    let mut g = QUERY_RESULTS.lock().unwrap_or_else(|e| e.into_inner());
    let m = g.get_or_insert_with(HashMap::new);
    if m.get(&id).is_none_or(|r| r.0 <= issue) {
        m.insert(id, (issue, samples));
    }
}

static DESTROYED: Mutex<Vec<u64>> = Mutex::new(Vec::new());
/// bumped when the renderer is rebuilt so static meshes upload again
pub static RENDERER_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub fn mesh_destroyed(id: u64) {
    DESTROYED.lock().unwrap_or_else(|e| e.into_inner()).push(id);
}
pub fn take_destroyed() -> Vec<u64> {
    std::mem::take(&mut *DESTROYED.lock().unwrap_or_else(|e| e.into_inner()))
}

#[derive(Default)]
pub struct Frame {
    /// None = dlss off this frame
    pub jitter: Option<[f32; 2]>,
    /// (view, proj) of the main 3d view (see `note_camera`)
    pub camera: Option<([f32; 16], [f32; 16])>,
    pub record_ms: f32,
    pub capture: bool,
    /// frame hit its budget
    pub skipped: usize,
    pub skipped_no_shader: usize,
    pub vertices: Vec<u8>,
    pub indices: Vec<u32>,
    pub cmds: Vec<Cmd>,
    pub uploads: Vec<MeshUpload>,
    /// consecutive duplicates are shared
    pub vs_blocks: Vec<u8>,
    pub ps_blocks: Vec<u8>,
    pub vs_block_count: usize,
}

/// dynamic overrides of the snapshot state, incl. gmod's render.OverrideBlend / PushFilter*
#[derive(Clone, Copy, Debug, Default)]
pub struct Overrides {
    pub depth_func_equal: bool,
    pub depth_write: Option<bool>,
    pub alpha_write: Option<bool>,
    pub color_write: Option<bool>,
    /// (separate alpha, src, dst, op)
    pub blend: Option<(bool, i32, i32, i32)>,
    pub blend_alpha: Option<(i32, i32, i32)>,
    pub filter: Option<i32>,
}

#[derive(Clone, Copy, Debug)]
pub struct Stencil {
    pub enable: bool,
    pub func: i32,
    pub fail: i32,
    pub zfail: i32,
    pub pass: i32,
    pub reference: i32,
    pub test_mask: u32,
    pub write_mask: u32,
}

/// d3dlight9 fields from SetLight
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    /// 0 spot, 1 point, 2 directional
    pub kind: i32,
    pub color: [f32; 3],
    pub pos: [f32; 3],
    pub dir: [f32; 3],
    pub falloff: f32,
    pub atten: [f32; 3],
    pub theta: f32,
    pub phi: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fog {
    pub start: f32,
    pub end: f32,
    pub z: f32,
    pub max_density: f32,
    /// 0 none, 1 linear, 2 linear below fog z
    pub mode: i32,
    pub color: [u8; 3],
}

pub struct State {
    material: usize,
    snapshot: i16,
    mesh: usize,
    first_index: i32,
    index_count: i32,
    vs_index: i32,
    ps_index: i32,
    pub draws: Vec<DrawCall>,
    pub frame: Frame,
    pub frames: u64,
    pub render_target: i32,
    /// -1 = the target's own
    pub render_depth: i32,
    /// dx9 defaults: 16, 0.00008
    pub shadow_bias: [f32; 2],
    next_texture: i32,
    modify_texture: i32,
    pub bound_textures: [i32; 16],
    /// cleared when a different texture binds (dx9 then applies that texture's wrap modes)
    pub border_clamp: u32,
    pub vs_constants: Vec<[f32; 4]>,
    pub ps_constants: Vec<[f32; 4]>,
    pub viewport: [i32; 4],
    matrix_mode: usize,
    /// 0 view, 1 projection, 2..=9 texture, 10 model
    matrices: [Vec<[f32; 16]>; 11],
    pub clear_color: [u8; 4],
    /// 3x4
    pub bones: Vec<[f32; 12]>,
    pub cull_mode: i32,
    /// by StandardTextureId_t
    pub standard_textures: Vec<i32>,
    pub bones_version: u64,
    pub num_bones: i32,
    pub viewport_depth: [f32; 2],
    /// u64::MAX = none
    pub trace_from: u64,
    pub vs_ints: [[i32; 4]; 16],
    pub ps_ints: [[i32; 4]; 16],
    pub vs_bools: u32,
    pub ps_bools: u32,
    pub fog: Fog,
    pub overbright: f32,
    /// +x, -x, +y, -y, +z, -z
    pub ambient_cube: [[f32; 4]; 6],
    pub lights: [Option<Light>; 4],
    pub stencil: Stencil,
    pub ov: Overrides,
    pub int_params: [i32; 20],
    pub float_params: [f32; 20],
    pub vector_params: [[f32; 3]; 20],
    pub scissor: Option<[i32; 4]>,
    /// world space ax + by + cz + d
    pub clip_planes: [[f32; 4]; 6],
    pub clip_enabled: u32,
    pub clip_transform: Option<[f32; 16]>,
    pub clip_transform_override: bool,
    pub vs_dirty: bool,
    pub ps_dirty: bool,
    last_extra: [[f32; 4]; 11],
    last_mip_bias: f32,
    /// inputs of the last standard vs constants / vs block tail; same inputs = nothing to redo
    vs_std_key: Option<Box<VsStdKey>>,
    extra_key: Option<Box<(ExtraKey, [[f32; 4]; 11])>>,
    /// (format, color element offset) of the last color mesh
    color_elem: (u64, Option<u32>),
    /// (tonemap scale bits, hdr type) -> c30
    tonemap_cache: ([u32; 3], i32, [f32; 4]),
    /// snapshots by id (they never change)
    snap_cache: Vec<Option<std::sync::Arc<shadow::ShadowState>>>,
    /// cameras (view, view x proj) of this frame's / last frame's scene draws
    vp_cur: Vec<([f32; 16], [f32; 16])>,
    vp_prev: Vec<([f32; 16], [f32; 16])>,
    reproj_cache: Option<([f32; 16], [f32; 16])>,
    /// handle -> (last issue, frame)
    pub queries: HashMap<u64, (u64, u64)>,
    next_query: u64,
    pub lighting_origin: [f32; 3],
    /// for sharing identical blocks
    last_vs_block: Option<u32>,
    /// registers the last vs block holds
    last_vs_regs: usize,
    /// highest vs register the engine wrote directly (sizes every block)
    vs_high: usize,
    last_ps_block: Option<u32>,
    /// (view, proj, scene draws) since the back buffer's depth was last cleared
    cameras: Vec<([f32; 16], [f32; 16], u32)>,
    camera_best: u32,
}

pub(crate) const IDENTITY: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

/// the api state. like dx9's, it's guarded by the shader mutex alone: the engine either calls
/// from one thread or enables that mutex, and the render thread never touches it.
struct StateCell(std::cell::UnsafeCell<Option<State>>);
// SAFETY: access only happens inside `with`, under the shader mutex (see above)
unsafe impl Sync for StateCell {}
static STATE: StateCell = StateCell(std::cell::UnsafeCell::new(None));

/// textures live outside the state: the render thread reads them while the game records
static TEXTURES: parking_lot::Mutex<Option<HashMap<i32, Texture>>> = parking_lot::Mutex::new(None);

pub fn texture_table() -> parking_lot::MappedMutexGuard<'static, HashMap<i32, Texture>> {
    parking_lot::MutexGuard::map(TEXTURES.lock(), |t| t.get_or_insert_with(HashMap::new))
}

/// mat_forceaniso, read by the render thread
pub static ANISOTROPY: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(1);

/// api calls since the last periodic log line (profiling only)
static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// ns the game thread waited for the render thread in present
pub static PRESENT_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    if profiling() {
        CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let _shader = crate::lock::guard();
    // tests call from parallel threads without the engine's guarantee
    #[cfg(test)]
    let _test = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    #[cfg(debug_assertions)]
    let _busy = Busy::enter();
    // SAFETY: serialized by the shader mutex (or the engine's single-threaded mode); `with` is
    // never re-entered (checked in debug builds)
    let g = unsafe { &mut *STATE.0.get() };
    let s = g.get_or_insert_with(|| State {
        material: 0,
        snapshot: -1,
        mesh: 0,
        first_index: 0,
        index_count: 0,
        vs_index: 0,
        ps_index: 0,
        draws: Vec::new(),
        frame: Frame::default(),
        frames: 0,
        render_target: -1,
        render_depth: -1,
        shadow_bias: [16.0, 0.00008],
        next_texture: 1,
        modify_texture: 0,
        bound_textures: [0; 16],
        border_clamp: 0,
        vs_constants: vec![[0.0; 4]; 256],
        ps_constants: vec![[0.0; 4]; 224],
        viewport: [0, 0, 0, 0],
        matrix_mode: 0,
        matrices: std::array::from_fn(|_| vec![IDENTITY]),
        clear_color: [0; 4],
        bones: Vec::new(),
        cull_mode: 0,
        standard_textures: Vec::new(),
        bones_version: 0,
        num_bones: 0,
        viewport_depth: [0.0, 1.0],
        trace_from: u64::MAX,
        vs_ints: [[0; 4]; 16],
        ps_ints: [[0; 4]; 16],
        vs_bools: 0,
        ps_bools: 0,
        fog: Fog { start: 0.0, end: 0.0, z: 0.0, max_density: 1.0, mode: 0, color: [0; 3] },
        overbright: 2.0,
        ambient_cube: [[0.0; 4]; 6],
        lights: [None; 4],
        ov: Overrides::default(),
        int_params: [0; 20],
        float_params: [0.0; 20],
        vector_params: [[0.0; 3]; 20],
        scissor: None,
        clip_planes: [[0.0; 4]; 6],
        clip_enabled: 0,
        clip_transform: None,
        clip_transform_override: false,
        vs_dirty: true,
        ps_dirty: true,
        last_extra: [[0.0; 4]; 11],
        last_mip_bias: 0.0,
        snap_cache: Vec::new(),
        vs_std_key: None,
        extra_key: None,
        color_elem: (u64::MAX, None),
        tonemap_cache: ([0; 3], -1, [0.0; 4]),
        vp_cur: Default::default(),
        vp_prev: Default::default(),
        reproj_cache: None,
        queries: HashMap::new(),
        next_query: 1,
        stencil: Stencil { enable: false, func: 8, fail: 1, zfail: 1, pass: 1, reference: 0, test_mask: 0xffff_ffff, write_mask: 0xffff_ffff },
        lighting_origin: [0.0; 3],
        last_vs_block: None,
        last_vs_regs: 0,
        vs_high: 0,
        last_ps_block: None,
        cameras: Vec::new(),
        camera_best: 0,
    });
    f(s)
}

#[cfg(test)]
static TEST_LOCK: Mutex<()> = Mutex::new(());

/// debug builds: catches nested or overlapping `with` (which would alias the state)
#[cfg(debug_assertions)]
struct Busy;
#[cfg(debug_assertions)]
static BUSY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[cfg(debug_assertions)]
impl Busy {
    fn enter() -> Busy {
        assert!(!BUSY.swap(true, std::sync::atomic::Ordering::Acquire), "api state entered twice");
        Busy
    }
}
#[cfg(debug_assertions)]
impl Drop for Busy {
    fn drop(&mut self) {
        BUSY.store(false, std::sync::atomic::Ordering::Release);
    }
}

fn leak(m: Box<Mesh>) -> usize {
    Box::leak(m) as *mut Mesh as usize
}
fn dynamic(cell: &'static OnceLock<usize>) -> &'static Mesh {
    unsafe { &*(*cell.get_or_init(|| leak(Mesh::new(VERTEX_POSITION, true))) as *const Mesh) }
}
pub fn dynamic_mesh() -> &'static Mesh {
    static M: OnceLock<usize> = OnceLock::new();
    dynamic(&M)
}
pub fn dynamic_vertex_buffer() -> &'static Mesh {
    static M: OnceLock<usize> = OnceLock::new();
    dynamic(&M)
}
pub fn dynamic_index_buffer() -> &'static Mesh {
    static M: OnceLock<usize> = OnceLock::new();
    dynamic(&M)
}
fn flex_mesh() -> &'static Mesh {
    static M: OnceLock<usize> = OnceLock::new();
    dynamic(&M)
}

unsafe fn vcall<R>(obj: usize, slot: usize) -> R
where
    R: Copy,
{
    unsafe {
        let vt = *(obj as *const *const usize);
        let f: unsafe extern "C" fn(usize) -> R = std::mem::transmute_copy(&*vt.add(slot));
        f(obj)
    }
}

fn material_name(material: usize) -> String {
    if material == 0 {
        return "<null>".into();
    }
    let p: *const c_char = unsafe { vcall(material, MATERIAL_GET_NAME) };
    if p.is_null() { "<unnamed>".into() } else { unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned() }
}

/// dynamic mesh format: explicit format wins, else the material's; bone weights follow the
/// bone count
fn dynamic_format(material: usize, explicit: u64, bones: i32) -> u64 {
    let mut fmt = if explicit != 0 {
        explicit
    } else if material != 0 {
        (unsafe { vcall::<u64>(material, MATERIAL_GET_VERTEX_FORMAT) }) & !VERTEX_FORMAT_COMPRESSED
    } else {
        VERTEX_POSITION
    };
    let bones = bones.max(if explicit != 0 { num_bone_weights(explicit) as i32 } else { 0 });
    fmt &= !VERTEX_BONE_WEIGHT_MASK;
    if bones > 0 {
        fmt |= vertex_boneweight(2) | VERTEX_BONE_INDEX;
    }
    fmt
}

unsafe extern "C" fn get_dynamic_mesh(_t: *mut c_void, material: usize, bones: i32, _buffered: bool, v_override: *mut c_void, i_override: *mut c_void) -> *mut c_void {
    unsafe { get_dynamic_mesh_ex(_t, material, 0, bones, _buffered, v_override, i_override) }
}
unsafe extern "C" fn get_dynamic_mesh_ex(_t: *mut c_void, material: usize, fmt: u64, bones: i32, _buffered: bool, v_override: *mut c_void, i_override: *mut c_void) -> *mut c_void {
    let m = dynamic_mesh();
    if v_override.is_null() {
        m.set_format(dynamic_format(material, fmt, bones));
    }
    let d = m.d();
    d.vertex_override = (!v_override.is_null()).then_some(v_override as *mut Mesh);
    d.index_override = (!i_override.is_null()).then_some(i_override as *mut Mesh);
    m.as_ptr()
}
/// ps 2.b flex format, exactly 28 bytes
pub const FLEX_FORMAT: u64 = VERTEX_POSITION | VERTEX_NORMAL | VERTEX_WRINKLE | VERTEX_FORMAT_USE_EXACT_FORMAT;
unsafe extern "C" fn get_flex_mesh(_t: *mut c_void) -> *mut c_void {
    let m = flex_mesh();
    m.set_format(FLEX_FORMAT);
    m.as_ptr()
}


pub fn draw_mesh(mesh: *mut Mesh, first: i32, count: i32) {
    // one api call including the material callback
    let _shader = crate::lock::guard();
    let material = with(|s| {
        s.mesh = mesh as usize;
        s.first_index = first;
        s.index_count = count;
        s.material
    });
    if material == 0 {
        log::line("draw with no bound material");
        return;
    }
    let t = profiling().then(std::time::Instant::now);
    unsafe {
        let vt = *(material as *const *const usize);
        let f: unsafe extern "C" fn(usize, i32) = std::mem::transmute(*vt.add(MATERIAL_DRAW_MESH));
        f(material, 0); // VERTEX_COMPRESSION_NONE
    }
    if let Some(t) = t {
        DRAW_MESH_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    }
    with(|s| s.mesh = 0);
}

unsafe extern "C" fn bind(_t: *mut c_void, material: usize) {
    with(|s| s.material = material);
}
unsafe extern "C" fn begin_pass(_t: *mut c_void, snapshot: i16) {
    with(|s| s.snapshot = snapshot);
}
unsafe extern "C" fn render_pass(_t: *mut c_void, _pass: i32, _count: i32) {
    with(|s| {
        if s.mesh != 0 && s.frames < 6 && s.draws.len() < 60 {
            let vp = s.viewport;
            let rt = s.render_target;
            log::line(&format!("  RenderPass rt {rt} viewport {vp:?} material {}", material_name(s.material)));
        }
        if s.mesh != 0 {
            let call = DrawCall { snapshot: s.snapshot, mesh: s.mesh, first_index: s.first_index, index_count: s.index_count, vs_index: s.vs_index, ps_index: s.ps_index };
            s.draws.push(call);
            record_draw(s);
        }
    });
}

/// d3dx row-major
pub(crate) fn mat_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut o = [0.0; 16];
    for r in 0..4 {
        for c in 0..4 {
            o[r * 4 + c] = (0..4).map(|k| a[r * 4 + k] * b[k * 4 + c]).sum();
        }
    }
    o
}

/// how dx9 uploads matrices to constants
fn column(m: &[f32; 16], k: usize) -> [f32; 4] {
    [m[k], m[4 + k], m[8 + k], m[12 + k]]
}

/// -t R^T
fn camera_position(v: &[f32; 16]) -> [f32; 3] {
    let t = [v[12], v[13], v[14]];
    std::array::from_fn(|i| -(t[0] * v[i * 4] + t[1] * v[i * 4 + 1] + t[2] * v[i * 4 + 2]))
}

/// vs constants dx9 maintains itself: c0-c3, fog c2/c16, transforms c4-c13, model / bones from c58
fn standard_vs_constants(s: &mut State, skinned: bool, flex: bool) {
    let (model, view, proj) = (*s.matrices[10].last().unwrap(), *s.matrices[0].last().unwrap(), *s.matrices[1].last().unwrap());
    let vp = mat_mul(&view, &proj);
    let mvp = mat_mul(&model, &vp);
    let eye = camera_position(&view);
    let fog = s.fog;
    let oo_range = if fog.end != fog.start { 1.0 / (fog.end - fog.start) } else { 1.0 };
    let ob = s.overbright;
    let lights = sorted_lights(s);
    let mut dirty = false;
    let c = &mut s.vs_constants;
    // only writes what changed
    let mut put = |i: usize, v: [f32; 4]| {
        if c[i] != v {
            c[i] = v;
            dirty = true;
        }
    };
    put(0, [0.0, 1.0, 2.0, 0.5]);
    put(1, [1.0 / 2.2, ob, 1.0 / 3.0, 1.0 / ob]);
    put(2, [eye[0], eye[1], eye[2], fog.z]);
    // cFlexScale
    put(3, if flex { [1.0, 1.0, 0.0, 0.0] } else { [0.0; 4] });
    for k in 0..4 {
        put(4 + k, column(&mvp, k));
        put(8 + k, column(&vp, k));
    }
    put(12, column(&mvp, 2));
    put(13, column(&vp, 2));
    put(16, [oo_range * fog.end, 1.0, 1.0 - fog.max_density.clamp(0.0, 1.0), oo_range]);
    for k in 0..3 {
        put(58 + k, column(&model, k));
    }
    // lights sorted by type, five registers each from c27
    for (i, l) in lights.iter().enumerate() {
        let spot = l.kind == 0;
        put(27 + 5 * i, [l.color[0], l.color[1], l.color[2], if l.kind == 2 { 1.0 } else { 0.0 }]);
        put(28 + 5 * i, [l.dir[0], l.dir[1], l.dir[2], if spot { 1.0 } else { 0.0 }]);
        put(29 + 5 * i, [l.pos[0], l.pos[1], l.pos[2], 1.0]);
        put(30 + 5 * i, if spot {
            let (stop, stop2) = ((l.theta * 0.5).cos(), (l.phi * 0.5).cos());
            [l.falloff, stop, stop2, if stop > stop2 { 1.0 / (stop - stop2) } else { 0.0 }]
        } else {
            [0.0, 1.0, 1.0, 1.0]
        });
        put(31 + 5 * i, [l.atten[0], l.atten[1], l.atten[2], 0.0]);
    }
    if skinned {
        for (i, b) in s.bones.iter().enumerate().skip(1).take(52) {
            for r in 0..3 {
                put(58 + 3 * i + r, [b[r * 4], b[r * 4 + 1], b[r * 4 + 2], b[r * 4 + 3]]);
            }
        }
    }
    let ints = [lights.len() as i32, 0, 1, 0];
    let bools = (s.vs_bools & !0xf) | ((1u32 << lights.len()) - 1);
    if s.vs_ints[0] != ints || s.vs_bools != bools {
        s.vs_ints[0] = ints;
        s.vs_bools = bools;
        dirty = true;
    }
    s.vs_dirty |= dirty;
}

/// spot, point, directional
fn sorted_lights(s: &State) -> Lights {
    let mut v = Lights { list: [None; 4], len: 0 };
    for l in s.lights.iter().flatten() {
        v.list[v.len] = Some(*l);
        v.len += 1;
    }
    v.list[..v.len].sort_by_key(|l| l.map(|l| l.kind));
    v
}

/// up to four lights without allocating
struct Lights {
    list: [Option<Light>; 4],
    len: usize,
}

impl Lights {
    fn iter(&self) -> impl Iterator<Item = &Light> {
        self.list[..self.len].iter().flatten()
    }
    fn len(&self) -> usize {
        self.len
    }
}

/// 2.2 curve dx9 uses for fog colors
fn gamma_to_linear(x: f32) -> f32 {
    x.max(0.0).powf(2.2)
}

/// ps constants dx9 maintains: fog color (c29), tonemap scale (c30)
fn standard_ps_constants(s: &mut State, snap: &shadow::ShadowState) {
    let fog = s.fog;
    let col = fog.color.map(|v| v as f32 / 255.0);
    let mut rgb = match fog.mode {
        1 if snap.srgb_write => col.map(gamma_to_linear),
        1 => col,
        2 => col.map(gamma_to_linear),
        _ => [0.0; 3],
    };
    let bits = TONEMAP.each_ref().map(|a| a.load(std::sync::atomic::Ordering::Relaxed));
    let hdr = crate::hwconfig::hdr_type();
    if s.tonemap_cache.0 != bits || s.tonemap_cache.1 != hdr {
        s.tonemap_cache = (bits, hdr, tonemap_constant());
    }
    let tonemap = s.tonemap_cache.2;
    // integer hdr tonemaps in the shaders, so fog gets scaled too
    if crate::hwconfig::hdr_type() == crate::hwconfig::HDR_INTEGER && !snap.disable_fog_gamma {
        rgb = rgb.map(|v| v * tonemap[0]);
    }
    let fog_color = [rgb[0], rgb[1], rgb[2], 1.0 / 192.0];
    if s.ps_constants[29] != fog_color || s.ps_constants[30] != tonemap {
        s.ps_constants[29] = fog_color;
        s.ps_constants[30] = tonemap;
        s.ps_dirty = true;
    }
}

/// ldr: GammaToLinearFullRange(2.0)
pub const LIGHTMAP_SCALE: f32 = 4.594_794;

/// integer hdr lightmaps hold value / 16
pub fn lightmap_scale_factor() -> f32 {
    if crate::hwconfig::hdr_type() == crate::hwconfig::HDR_INTEGER { 16.0 } else { LIGHTMAP_SCALE }
}

/// f32 bits; GetToneMappingScaleLinear returns a reference to it
static TONEMAP: [std::sync::atomic::AtomicU32; 3] = [const { std::sync::atomic::AtomicU32::new(0x3f80_0000) }; 3];

/// c30: output scale, lightmap scale, reflection scale, output scale in gamma
fn tonemap_constant() -> [f32; 4] {
    let hdr = crate::hwconfig::hdr_type() == crate::hwconfig::HDR_INTEGER;
    let x = if hdr { f32::from_bits(TONEMAP[0].load(std::sync::atomic::Ordering::Relaxed)) } else { 1.0 };
    [x, lightmap_scale_factor(), if hdr { 16.0 } else { 1.0 }, x.max(0.0).powf(1.0 / 2.2)]
}

unsafe extern "C" fn set_tonemap_scale(_t: *mut c_void, scale: *const f32) {
    if scale.is_null() {
        return;
    }
    let v = unsafe { std::slice::from_raw_parts(scale, 3) };
    for (a, x) in TONEMAP.iter().zip(v) {
        a.store(x.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }
}
unsafe extern "C" fn get_tonemap_scale(_t: *mut c_void) -> *const f32 {
    TONEMAP.as_ptr() as *const f32
}

/// no fog when the scene has none or the pass disables it
fn pixel_fog_params(s: &State) -> [f32; 4] {
    let fog = s.fog;
    let disabled = shadow::with(|sh| sh.get(s.snapshot).is_some_and(|x| x.fog_mode == 0));
    if fog.mode != 0 && !disabled {
        let oo = if fog.end != fog.start { 1.0 / (fog.end - fog.start) } else { 1.0 };
        if fog.mode == 2 { [0.0, fog.z, 1.0, oo] } else { [fog.start * oo, fog.z, fog.max_density.clamp(0.0, 1.0), oo] }
    } else {
        [0.0, fog.z, 1.0, 0.0]
    }
}

/// row-major, None when singular
pub(crate) fn mat_inverse(m: &[f32; 16]) -> Option<[f32; 16]> {
    mat_inverse_f64(m).map(|v| v.map(|x| x as f32))
}

fn mat_inverse_f64(m: &[f32; 16]) -> Option<[f64; 16]> {
    let m: [f64; 16] = m.map(|v| v as f64);
    let mut inv = [0f64; 16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if det.abs() < 1e-30 {
        return None;
    }
    Some(inv.map(|v| v / det))
}

/// perspective (m[2][3] != 0) onto the back buffer
fn is_scene(s: &State) -> bool {
    s.render_target == -1 && s.matrices[1].last().unwrap()[11] != 0.0
}

/// this frame's clip space to last frame's. each camera matches last frame's nearest view matrix
/// (2d sky, 3d skybox and world use different ones). f64 because world coords are large.
fn reprojection(s: &mut State) -> [f32; 16] {
    let view = *s.matrices[0].last().unwrap();
    let proj = *s.matrices[1].last().unwrap();
    let vp = mat_mul(&view, &proj);
    if let Some((k, m)) = &s.reproj_cache {
        if *k == vp {
            return *m;
        }
    }
    if !s.vp_cur.iter().any(|(_, v)| *v == vp) {
        s.vp_cur.push((view, vp));
    }
    let dist = |a: &[f32; 16]| a.iter().zip(&view).map(|(x, y)| (x - y).abs() as f64).sum::<f64>();
    let prev = s.vp_prev.iter().min_by(|a, b| dist(&a.0).total_cmp(&dist(&b.0))).map(|p| p.1);
    let m = match prev {
        Some(p) if p == vp => IDENTITY,
        Some(p) => mat_inverse_f64(&vp).map(|inv| {
            let mut o = [0f32; 16];
            for r in 0..4 {
                for c in 0..4 {
                    o[r * 4 + c] = (0..4).map(|k| inv[r * 4 + k] * p[k * 4 + c] as f64).sum::<f64>() as f32;
                }
            }
            o
        }).unwrap_or(IDENTITY),
        None => IDENTITY,
    };
    s.reproj_cache = Some((vp, m));
    m
}

/// vs block tail: half-pixel offset, user clip planes in clip space (disabled = zero, never clips),
/// reprojection
/// what `standard_vs_constants` reads
#[derive(PartialEq)]
struct VsStdKey {
    model: [f32; 16],
    view: [f32; 16],
    proj: [f32; 16],
    fog: Fog,
    overbright: f32,
    lights: [Option<Light>; 4],
    flex: bool,
    skinned: bool,
    bones: u64,
}

/// what `vs_extra` reads
#[derive(PartialEq)]
struct ExtraKey {
    viewport: [i32; 4],
    depth: [f32; 2],
    frames: u64,
    jitter: Option<[f32; 2]>,
    scale: [f32; 2],
    render_target: i32,
    view: [f32; 16],
    proj: [f32; 16],
    clip_enabled: u32,
    clip_planes: [[f32; 4]; 6],
    clip_transform: Option<[f32; 16]>,
    clip_override: bool,
}

fn cached_vs_extra(s: &mut State) -> [[f32; 4]; 11] {
    let key = ExtraKey {
        viewport: s.viewport,
        depth: s.viewport_depth,
        frames: s.frames,
        jitter: crate::dlss::jitter(s.frames),
        scale: crate::dlss::render_scale(),
        render_target: s.render_target,
        view: *s.matrices[0].last().unwrap(),
        proj: *s.matrices[1].last().unwrap(),
        clip_enabled: s.clip_enabled,
        clip_planes: s.clip_planes,
        clip_transform: s.clip_transform,
        clip_override: s.clip_transform_override,
    };
    if let Some(c) = s.extra_key.as_deref() {
        if c.0 == key {
            return c.1;
        }
    }
    let out = vs_extra(s);
    s.extra_key = Some(Box::new((key, out)));
    out
}

fn vs_extra(s: &mut State) -> [[f32; 4]; 11] {
    let mut out = [[0.0f32; 4]; 11];
    let (mut w, mut h) = (s.viewport[2], s.viewport[3]);
    if w <= 0 || h <= 0 {
        let b = crate::device::back_buffer_size();
        (w, h) = (b.0.max(1), b.1.max(1));
    }
    let scene = is_scene(s);
    let jitter = if scene { crate::dlss::jitter(s.frames) } else { None };
    let [jx, jy] = jitter.unwrap_or([0.0, 0.0]);
    // under dlss upscaling the scene is drawn into a smaller viewport: offsets are in its pixels
    let [sx, sy] = if jitter.is_some() { crate::dlss::render_scale() } else { [1.0, 1.0] };
    let (w, h) = (w as f32 * sx, h as f32 * sy);
    out[0] = [1.0 / w, 1.0 / h, 2.0 * jx / w, -2.0 * jy / h];
    let m = if jitter.is_some() && s.viewport_depth[1] > 0.15 { reprojection(s) } else { IDENTITY };
    for k in 0..4 {
        out[7 + k] = column(&m, k);
    }
    if s.clip_enabled != 0 {
        let view = if s.clip_transform_override { s.clip_transform.unwrap_or(*s.matrices[0].last().unwrap()) } else { *s.matrices[0].last().unwrap() };
        if let Some(inv) = mat_inverse(&mat_mul(&view, s.matrices[1].last().unwrap())) {
            for i in 0..6 {
                if s.clip_enabled & (1 << i) != 0 {
                    let p = s.clip_planes[i];
                    // q_j = sum_i inv[j][i] p_i
                    out[1 + i] = std::array::from_fn(|j| (0..4).map(|k| inv[j * 4 + k] * p[k]).sum());
                }
            }
        }
    }
    out
}

/// shares the previous block when equal
/// vs block: ints, bools, the extra tail, then `regs` float registers, padded to 256 bytes.
/// returns its offset in 256-byte units.
fn push_vs_block(s: &mut State, extra: &[[f32; 4]; 11], regs: usize) -> u32 {
    let b = &mut s.frame.vs_blocks;
    let start = b.len();
    b.extend_from_slice(bytemuck::cast_slice(&s.vs_ints));
    b.extend_from_slice(bytemuck::cast_slice(&[s.vs_bools, 0, 0, 0]));
    b.extend_from_slice(bytemuck::cast_slice(extra));
    b.extend_from_slice(bytemuck::cast_slice(&s.vs_constants[..regs]));
    b.resize(b.len().next_multiple_of(256), 0);
    s.frame.vs_block_count += 1;
    s.last_vs_block = Some((start / 256) as u32);
    s.last_vs_regs = regs;
    (start / 256) as u32
}

#[allow(clippy::too_many_arguments)]
fn push_block(blocks: &mut Vec<u8>, last: &mut Option<u32>, consts: &[[f32; 4]], ints: &[[i32; 4]; 16], bools: [u32; 4], extra: &[[f32; 4]], size: usize) -> u32 {
    let start = blocks.len();
    debug_assert!(consts.len() * 16 + 256 + 16 + extra.len() * 16 <= size);
    blocks.extend_from_slice(bytemuck::cast_slice(consts));
    blocks.extend_from_slice(bytemuck::cast_slice(ints));
    blocks.extend_from_slice(bytemuck::cast_slice(&bools));
    blocks.extend_from_slice(bytemuck::cast_slice(extra));
    blocks.resize(start + size, 0);
    let idx = (start / size) as u32;
    *last = Some(idx);
    idx
}

/// queue a non-draw command (logged during a capture)
fn push_cmd(s: &mut State, c: Cmd) {
    if s.frames == s.trace_from {
        log::line(&format!("  {c:?}"));
    }
    s.frame.cmds.push(c);
}

fn push_draw(s: &mut State, c: Cmd) {
    if s.frames == s.trace_from {
        if let Cmd::Draw(d) = &c {
            let table = texture_table();
            let tex: Vec<String> = d.textures.iter().enumerate().filter(|t| *t.1 != 0).map(|(i, h)| format!("s{i}={h}:{}", table.get(h).map_or("?", |t| t.name.as_str()))).collect();
            log::line(&format!(
                "  draw '{}' vs {}/{}+{} ps {}/{}+{} {:?} first {} count {} format {:#x} stride {} vp {:?} depth {:?} vsblk {} psblk {} ps {:?} key {:?} color {:?} [{}]",
                material_name(s.material), d.vs.name, d.vs.static_index, d.vs.dynamic, d.ps.name, d.ps.static_index, d.ps.dynamic,
                d.geometry, d.first_index, d.index_count, d.format, d.stride, d.viewport, d.depth, d.vs_block, d.ps_block, d.ps_opts, d.key, d.color, tex.join(" ")
            ));
            // shadow receivers: log the flashlight constants
            if d.ps_opts.shadow_depth != 0 {
                let c: Vec<String> = s.ps_constants[..32].iter().enumerate().map(|(i, v)| format!("c{i}={v:?}")).collect();
                log::line(&format!("    ps constants: {}", c.join(" ")));
            }
        }
    }
    s.frame.cmds.push(c);
}

fn record_draw(s: &mut State) {
    // a clock read costs ~40 ns under wine, so only time draws while profiling
    if !profiling() {
        return record_draw_inner(s);
    }
    let start = std::time::Instant::now();
    record_draw_inner(s);
    s.frame.record_ms += start.elapsed().as_secs_f32() * 1000.0;
}

/// record-path profile: ns per section, summed until the next periodic log line
static PROF: [std::sync::atomic::AtomicU64; 9] = [const { std::sync::atomic::AtomicU64::new(0) }; 9];
const PROF_NAMES: [&str; 9] = ["snapshot", "debug", "flex", "vs consts", "ps consts", "vs extra", "blocks", "draw cmd", "geometry"];

/// WULKANIUM_PROFILE=1: time the record path (the timers themselves cost ~20 ns each)
/// 0 = not decided yet, 1 = on, 2 = off
static PROFILE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

fn profiling() -> bool {
    match PROFILE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on = std::env::var("WULKANIUM_PROFILE").is_ok_and(|v| v != "0");
            PROFILE.store(if on { 1 } else { 2 }, std::sync::atomic::Ordering::Relaxed);
            on
        }
    }
}

/// ns inside IMesh::Draw (the whole material draw: shader dll code plus our api calls)
static DRAW_MESH_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct Lap(Option<std::time::Instant>);
impl Lap {
    fn lap(&mut self, i: usize) {
        if let Some(t) = self.0.as_mut() {
            let now = std::time::Instant::now();
            PROF[i].fetch_add((now - *t).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            *t = now;
        }
    }
}

static PROF_DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// "section ns/draw ..." since the last call, then reset
fn take_profile() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let n = PROF_DRAWS.swap(0, Relaxed).max(1);
    PROF_NAMES.iter().zip(&PROF).map(|(name, v)| format!("{name} {}", v.swap(0, Relaxed) / n)).collect::<Vec<_>>().join(", ")
}

fn record_draw_inner(s: &mut State) {
    PROF_DRAWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut lap = Lap(profiling().then(std::time::Instant::now));
    // whatever runs after the last lap counts as geometry
    struct Tail(Option<std::time::Instant>);
    impl Drop for Tail {
        fn drop(&mut self) {
            if let Some(t) = self.0 {
                PROF[8].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    let mesh = unsafe { &*(s.mesh as *const Mesh) };
    let md = mesh.r();
    let vsrc = md.vertex_override.map(|m| unsafe { &*m }).unwrap_or(mesh).r();
    let isrc = md.index_override.map(|m| unsafe { &*m }).unwrap_or(mesh).r();
    const MAX_FRAME_VERTEX_BYTES: usize = 512 << 20;
    if s.frame.vertices.len() > MAX_FRAME_VERTEX_BYTES {
        s.frame.skipped += 1;
        return;
    }
    // snapshots never change, so keep them per id without the shadow lock
    let id = s.snapshot.max(-1) as usize;
    let cached = if s.snapshot >= 0 { s.snap_cache.get(id).cloned().flatten() } else { None };
    let snap = cached.or_else(|| {
        let a = shadow::with(|sh| sh.get(s.snapshot).cloned())?;
        if s.snap_cache.len() <= id {
            s.snap_cache.resize(id + 1, None);
        }
        s.snap_cache[id] = Some(a.clone());
        Some(a)
    });
    let Some(snap) = snap else {
        why_skipped(s, "no snapshot");
        return;
    };
    // vs without ps is a depth-only pass (DepthWrite): "" stands for the null pixel shader
    let (Some(vs_name), ps_name) = (snap.vertex_shader, snap.pixel_shader.unwrap_or(shadow::ShaderName::NONE)) else {
        s.frame.skipped_no_shader += 1;
        why_skipped(s, "no shaders");
        return;
    };
    let stride = vsrc.stride;
    if stride == 0 || vsrc.vertex_count == 0 {
        why_skipped(s, &format!("no vertices (stride {stride}, {} vertices, format {:#x})", vsrc.vertex_count, vsrc.format));
        return;
    }
    lap.lap(0);
    debug_material(s, &snap, vsrc.format, md.color_mesh.is_some());
    lap.lap(1);
    let skinned = s.num_bones > 0;
    // stream 2 + cFlexScale
    let flex = md.flex_mesh.and_then(|(fm, offset)| {
        let f = unsafe { &*fm }.r();
        let bytes = vsrc.vertex_count * 28;
        let src = f.vertices.get(offset..offset + bytes)?;
        let at = s.frame.vertices.len() as u64;
        s.frame.vertices.extend_from_slice(src);
        Some(at)
    });
    lap.lap(2);
    let key = VsStdKey {
        model: *s.matrices[10].last().unwrap(),
        view: *s.matrices[0].last().unwrap(),
        proj: *s.matrices[1].last().unwrap(),
        fog: s.fog,
        overbright: s.overbright,
        lights: s.lights,
        flex: flex.is_some(),
        skinned,
        bones: if skinned { s.bones_version } else { 0 },
    };
    if s.vs_std_key.as_deref() != Some(&key) {
        standard_vs_constants(s, skinned, flex.is_some());
        s.vs_std_key = Some(Box::new(key));
    }
    lap.lap(3);
    standard_ps_constants(s, &snap);
    lap.lap(4);
    let extra = cached_vs_extra(s);
    if extra != s.last_extra {
        s.last_extra = extra;
        s.vs_dirty = true;
    }
    lap.lap(5);
    // unskinned shaders read up to the model matrix (c58..c60), skinned ones the bones after it
    let regs = if skinned { 58 + 3 * s.bones.len().max(1) } else { 61 }.max(s.vs_high + 1).min(s.vs_constants.len());
    let vs_block = match s.last_vs_block {
        Some(b) if !s.vs_dirty && regs <= s.last_vs_regs => b,
        _ => push_vs_block(s, &extra, regs),
    };
    s.vs_dirty = false;
    // -1 for the scene under dlss (guide 3.5)
    // guide 3.5: log2(render / display) - 1 for the scene under dlss
    let mip_bias: f32 = if is_scene(s) && crate::dlss::jitter(s.frames).is_some() { crate::dlss::mip_bias() } else { 0.0 };
    if mip_bias != s.last_mip_bias {
        s.last_mip_bias = mip_bias;
        s.ps_dirty = true;
    }
    let ps_block = match s.last_ps_block {
        Some(b) if !s.ps_dirty => b,
        _ => push_block(&mut s.frame.ps_blocks, &mut s.last_ps_block, &s.ps_constants[..crate::d3d2wgsl::PS_FLOAT_CONSTS], &s.ps_ints, [s.ps_bools, mip_bias.to_bits(), 0, 0], &[], PS_BLOCK),
    };
    s.ps_dirty = false;
    lap.lap(6);

    // (-1, _) = whole mesh
    let (first, count) = if s.first_index < 0 || s.index_count <= 0 { (0usize, isrc.index_count) } else { (s.first_index as usize, s.index_count as usize) };
    let cull_mode = s.cull_mode;
    // snapshot stencil wins over the dynamic one
    let st = s.stencil;
    let (stencil, stencil_ref) = if snap.stencil {
        (Some((snap.stencil_func, snap.stencil_fail, snap.stencil_depth_fail, snap.stencil_pass, snap.stencil_mask as u8, snap.stencil_write_mask as u8)), snap.stencil_ref as u32)
    } else if st.enable {
        (Some((st.func, st.fail, st.zfail, st.pass, st.test_mask as u8, st.write_mask as u8)), st.reference as u32)
    } else {
        (None, 0)
    };
    let ov = s.ov;
    let (blend, blend_alpha) = match ov.blend {
        Some((separate, src, dst, op)) => (Some((src, dst, op)), if separate { ov.blend_alpha } else { None }),
        None => (snap.blend.then_some((snap.blend_src, snap.blend_dst, snap.blend_op)), (snap.blend && snap.blend_separate_alpha).then_some((snap.blend_src_alpha, snap.blend_dst_alpha, snap.blend_op_alpha))),
    };
    let shadow_bias = if snap.poly_offset == 2 { (s.shadow_bias[0].to_bits(), s.shadow_bias[1].to_bits()) } else { (0, 0) };
    let key = |topology| PipeKey {
        topology,
        blend,
        blend_alpha,
        depth_test: snap.depth_test || ov.depth_write.is_some(),
        depth_write: ov.depth_write.unwrap_or(snap.depth_write),
        // SHADER_DEPTHFUNC_EQUAL
        depth_func: if ov.depth_func_equal { 2 } else { snap.depth_func },
        color_write: ov.color_write.unwrap_or(snap.color_write),
        alpha_write: ov.alpha_write.unwrap_or(snap.alpha_write),
        // 0 = ccw (cull back faces), 1 = cw
        cull: if snap.cull { if cull_mode == 1 { 2 } else { 1 } } else { 0 },
        poly_offset: snap.poly_offset,
        shadow_bias,
        poly_mode: snap.poly_mode,
        stencil,
        alpha_to_coverage: snap.alpha_to_coverage && snap.alpha_test && blend.is_none(),
    };
    // only samplers the snapshot enables count
    let mut textures = [0i32; 20];
    for (i, t) in textures.iter_mut().enumerate().take(16) {
        if snap.textures_enabled & (1 << i) != 0 {
            *t = s.bound_textures[i];
        }
    }
    let ps_opts = crate::d3d2wgsl::PsOptions {
        srgb_read: snap.srgb_read,
        srgb_write: snap.srgb_write,
        alpha_test: snap.alpha_test.then(|| (snap.alpha_func, (f32::from_bits(snap.alpha_ref_bits).clamp(0.0, 1.0) * 255.0) as u8)),
        shadow_depth: snap.shadow_filtering & 0xffff,
    };
    let vs = ShaderRef { name: vs_name, static_index: snap.vertex_shader_static, dynamic: s.vs_index };
    let ps = ShaderRef { name: ps_name, static_index: snap.pixel_shader_static, dynamic: s.ps_index };
    let (viewport, depth, format, scissor, clip) = (s.viewport, s.viewport_depth, vsrc.format, s.scissor, s.clip_enabled != 0);
    let scene = is_scene(s);
    if scene && depth[1] > 0.15 {
        note_camera(s);
    }
    // baked vertex lighting, stream 1
    let color = md.color_mesh.and_then(|(cm, first)| {
        let cm = unsafe { &*cm };
        let (fmt, cstride, id) = { let d = cm.r(); (d.format, d.stride, d.id) };
        if s.color_elem.0 != fmt {
            s.color_elem = (fmt, vertex_elements(fmt).into_iter().find(|e| e.usage == 10).map(|e| e.offset as u32));
        }
        let e_offset = s.color_elem.1?;
        if s.frames == s.trace_from {
            // capture: mean bgra of the color mesh this draw uses
            let d = cm.r();
            let (mut sum, mut n) = ([0u64; 4], 0u64);
            for v in d.vertices.chunks_exact(cstride.max(1)).skip(first).take(4096) {
                if let Some(c) = v.get(e_offset as usize..e_offset as usize + 4) {
                    for i in 0..4 {
                        sum[i] += c[i] as u64;
                    }
                    n += 1;
                }
            }
            log::line(&format!("    color mesh {id}: {} verts total, first {first}, stride {cstride}, mean bytes {:?} over {n}", d.vertex_count, sum.map(|x| if n > 0 { x / n } else { 0 })));
        }
        ensure_uploaded(s, cm);
        Some((id, (first * cstride) as u64, cstride as u32, e_offset))
    });
    let border = s.border_clamp;
    lap.lap(7);
    let _tail = Tail(profiling().then(std::time::Instant::now));
    let draw = |geometry, first_index: usize, index_count: usize, topology| {
        Cmd::Draw(DrawCmd {
            geometry,
            first_index: first_index as u32,
            index_count: index_count as u32,
            format,
            stride: stride as u32,
            key: key(topology),
            vs: vs.clone(),
            ps: ps.clone(),
            ps_opts,
            vs_block,
            ps_block,
            textures,
            viewport,
            depth,
            stencil_ref,
            scissor,
            filter_override: ov.filter,
            border,
            clip,
            scene,
            flex,
            color,
        })
    };

    // static meshes upload once per version
    let is_static = !md.dynamic && md.vertex_override.is_none() && md.index_override.is_none() && matches!(md.primitive, 2 | 6 | 7) && md.index_count > 0;
    if is_static {
        let (id, index_count) = (md.id, md.index_count);
        ensure_uploaded(s, mesh);
        let end = (first + count).min(index_count);
        if end > first {
            let c = draw(Geometry::Static { mesh: id }, first, end - first, 2);
            push_draw(s, c);
        }
        return;
    }

    let raw: Vec<u32> = if isrc.index_count > 0 {
        isrc.indices.get(first..(first + count).min(isrc.indices.len())).unwrap_or(&[]).iter().map(|&i| i as u32).collect()
    } else {
        (0..vsrc.vertex_count as u32).collect()
    };
    let (topology, list) = match md.primitive {
        0 => (0u8, raw),
        1 => (1, raw),
        3 => (2, strip_to_list(&raw)),
        4 => (1, raw.windows(2).flat_map(|w| [w[0], w[1]]).collect()),
        5 => (1, raw.iter().zip(raw.iter().cycle().skip(1)).flat_map(|(a, b)| [*a, *b]).collect()),
        _ => (2, raw),
    };
    if list.is_empty() {
        return;
    }
    let max = *list.iter().max().unwrap() as usize;
    let min = *list.iter().min().unwrap() as usize;
    if max >= vsrc.vertex_count || (max + 1) * stride > vsrc.vertices.len() {
        why_skipped(s, &format!("index {max} out of range ({} vertices, {} bytes)", vsrc.vertex_count, vsrc.vertices.len()));
        return;
    }
    // world batches: static vertices stay on the gpu, only indices go into the frame
    if let Some(vm) = md.vertex_override.map(|m| unsafe { &*m }).filter(|m| !m.r().dynamic) {
        ensure_uploaded(s, vm);
        let first_index = s.frame.indices.len();
        s.frame.indices.extend(list.iter().copied());
        let c = draw(Geometry::StaticVertices { mesh: vm.r().id }, first_index, list.len(), topology);
        push_draw(s, c);
        return;
    }
    // copy just the used vertex range, indices rebased; start at a multiple of the stride so
    // the renderer can draw it with a base vertex
    let pad = (stride - s.frame.vertices.len() % stride) % stride;
    s.frame.vertices.resize(s.frame.vertices.len() + pad, 0);
    let vertex_offset = s.frame.vertices.len() as u64;
    s.frame.vertices.extend_from_slice(&vsrc.vertices[min * stride..(max + 1) * stride]);
    let first_index = s.frame.indices.len();
    s.frame.indices.extend(list.iter().map(|i| i - min as u32));
    let mut c = draw(Geometry::Frame { vertex_offset }, first_index, list.len(), topology);
    // the flex stream covers the whole mesh, so offset it to the copied range
    if let Cmd::Draw(d) = &mut c {
        if let Some(f) = d.flex.as_mut() {
            *f += (min * 28) as u64;
        }
    }
    push_draw(s, c);
}

/// counts scene draws per camera. the main camera is the one most draws used since the screen's
/// depth was last cleared (the world, not a late sprite or lua view with its own matrices). it's
/// settled as the scene draws come in, so clears after the scene (the hud) don't undo it.
fn note_camera(s: &mut State) {
    let (view, proj) = (*s.matrices[0].last().unwrap(), *s.matrices[1].last().unwrap());
    let n = match s.cameras.iter_mut().find(|c| c.0 == view && c.1 == proj) {
        Some(c) => {
            c.2 += 1;
            c.2
        }
        None => {
            s.cameras.push((view, proj, 1));
            1
        }
    };
    if n > s.camera_best {
        s.camera_best = n;
        s.frame.camera = Some((view, proj));
    }
}

/// once per material and reason
fn why_skipped(s: &State, why: &str) {
    static SEEN: Mutex<Option<std::collections::HashSet<(usize, String)>>> = Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let seen = seen.get_or_insert_with(Default::default);
    if s.frames == s.trace_from || (seen.len() < 300 && seen.insert((s.material, why.to_string()))) {
        log::line(&format!("draw of '{}' skipped: {why}", material_name(s.material)));
    }
}

fn ensure_uploaded(s: &mut State, mesh: &Mesh) {
    let generation = RENDERER_GENERATION.load(std::sync::atomic::Ordering::Relaxed);
    let m = mesh.r();
    if m.uploaded == (m.version, generation) {
        // filled through a kept pointer: no version bump tells us, so refresh every 60 frames
        // for the first 10 seconds
        let recent = m.unlocked_at.is_some_and(|t| t.elapsed().as_secs() < 10);
        if !recent || s.frames.saturating_sub(m.refreshed) < 60 {
            return;
        }
    }
    mesh.d().refreshed = s.frames;
    let m = mesh.r();
    let vertices = m.vertices[..(m.vertex_count * m.stride).min(m.vertices.len())].to_vec();
    let last = m.vertex_count.saturating_sub(1) as u32;
    let indices: Vec<u32> = m.indices[..m.index_count.min(m.indices.len())].iter().map(|&i| (i as u32).min(last)).collect();
    let (id, version) = (m.id, m.version);
    s.frame.uploads.push(MeshUpload { id, vertices, indices });
    mesh.d().uploaded = (version, generation);
}

/// once per material
fn debug_material(s: &State, snap: &shadow::ShadowState, format: VertexFormat, color_mesh: bool) {
    static FULL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if FULL.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    static SEEN: Mutex<Option<rustc_hash::FxHashSet<(usize, [i32; 16])>>> = Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let seen = seen.get_or_insert_with(Default::default);
    let mut bound = [0i32; 16];
    for (i, b) in bound.iter_mut().enumerate() {
        if snap.textures_enabled & (1 << i) != 0 {
            *b = s.bound_textures[i];
        }
    }
    if seen.len() >= 600 {
        FULL.store(true, std::sync::atomic::Ordering::Relaxed);
        return;
    }
    if !seen.insert((s.material, bound)) || material_name(s.material).starts_with("engine/preload") {
        return;
    }
    let table = texture_table();
    let bound: Vec<String> = (0..16)
        .filter(|i| snap.textures_enabled & (1 << i) != 0)
        .map(|i| {
            let h = s.bound_textures[i];
            format!("s{i}={h}:{}{}", table.get(&h).map_or("?", |t| t.name.as_str()), if table.get(&h).is_some_and(|t| t.version == 0) { "(empty)" } else { "" })
        })
        .collect();
    log::line(&format!(
        "material {}: vs {:?}/{} ps {:?}/{} samplers {:#x} srgb r {:#x} w {} format {format:#x} color mesh {color_mesh} [{}]",
        material_name(s.material),
        snap.vertex_shader,
        snap.vertex_shader_static,
        snap.pixel_shader,
        snap.pixel_shader_static,
        snap.textures_enabled,
        snap.srgb_read,
        snap.srgb_write,
        bound.join(" ")
    ));
}

fn strip_to_list(idx: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    for i in 0..idx.len().saturating_sub(2) {
        let (a, b, c) = if i % 2 == 0 { (idx[i], idx[i + 1], idx[i + 2]) } else { (idx[i + 1], idx[i], idx[i + 2]) };
        if a != b && b != c && a != c {
            out.extend([a, b, c]);
        }
    }
    out
}

/// recording continues on the same target
fn flush_recorded() {
    let _shader = crate::lock::guard();
    let frame = with(|s| {
        let frame = std::mem::take(&mut s.frame);
        s.last_vs_block = None;
        s.last_ps_block = None;
        if s.render_target != -1 || s.render_depth != -1 {
            s.frame.cmds.push(Cmd::Target { color: s.render_target, depth: s.render_depth });
        }
        frame
    });
    crate::render::flush(frame);
}

/// `stride` 0 = tight
fn write_pixels(src: &[u8], w: usize, h: usize, dst: *mut u8, fmt: i32, stride: usize) {
    let bpp = match fmt {
        2 | 3 => 3,
        0 | 1 | 11 | 12 | 16 => 4,
        _ => return,
    };
    let stride = if stride == 0 { w * bpp } else { stride };
    for y in 0..h {
        for x in 0..w {
            let p = &src[(y * w + x) * 4..(y * w + x) * 4 + 4];
            let (r, g, b, a) = (p[0], p[1], p[2], p[3]);
            let px: [u8; 4] = match fmt {
                0 => [r, g, b, a],
                1 => [a, b, g, r],
                2 => [r, g, b, 0],
                3 => [b, g, r, 0],
                11 => [a, r, g, b],
                12 => [b, g, r, a],
                _ => [b, g, r, 255],
            };
            unsafe { std::ptr::copy_nonoverlapping(px.as_ptr(), dst.add(y * stride + x * bpp), bpp) };
        }
    }
}

fn read_pixels_impl(x: i32, y: i32, w: i32, h: i32, data: *mut u8, fmt: i32, stride: usize) {
    if data.is_null() || w <= 0 || h <= 0 || x < 0 || y < 0 {
        return;
    }
    flush_recorded();
    let target = with(|s| s.render_target);
    if let Some(px) = crate::render::read_target(target, x as u32, y as u32, w as u32, h as u32) {
        write_pixels(&px, w as usize, h as usize, data, fmt, stride);
    }
}
unsafe extern "C" fn read_pixels(_t: *mut c_void, x: i32, y: i32, w: i32, h: i32, data: *mut u8, fmt: i32) {
    read_pixels_impl(x, y, w, h, data, fmt, 0);
}
/// no scaling
unsafe extern "C" fn read_pixels_rect(_t: *mut c_void, src: *const i32, dst: *const i32, data: *mut u8, fmt: i32, stride: i32) {
    let Some(sr) = rect(src) else { return };
    let dr = rect(dst).unwrap_or([0, 0, sr[2], sr[3]]);
    let bpp = match fmt {
        2 | 3 => 3,
        _ => 4,
    };
    let stride = if stride > 0 { stride as usize } else { dr[2].max(0) as usize * bpp };
    let base = unsafe { data.add(dr[1].max(0) as usize * stride + dr[0].max(0) as usize * bpp) };
    read_pixels_impl(sr[0], sr[1], sr[2].min(dr[2]), sr[3].min(dr[3]), base, fmt, stride);
}

fn capture_key() -> bool {
    #[cfg(windows)]
    {
        use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_HOME};
        static DOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let down = unsafe { GetAsyncKeyState(VK_HOME as i32) } as u16 & 0x8000 != 0;
        let was = DOWN.swap(down, std::sync::atomic::Ordering::Relaxed);
        let pressed = down && !was;
        // Shift+Home toggles profiling instead
        let shift = unsafe { GetAsyncKeyState(windows_sys::Win32::UI::Input::KeyboardAndMouse::VK_SHIFT as i32) } as u16 & 0x8000 != 0;
        if pressed && shift {
            // the sampler only: the record-path timers cost more than they measure under wine
            static SAMPLING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            let on = !SAMPLING.fetch_xor(true, std::sync::atomic::Ordering::Relaxed);
            log::line(&format!("sampler {}", if on { "on" } else { "off" }));
            if on { crate::sampler::start() } else { crate::sampler::stop() }
            return false;
        }
        pressed
    }
    #[cfg(not(windows))]
    false
}

pub fn present() {
    // lock order: shader mutex, then renderer
    let _shader = crate::lock::guard();
    crate::sampler::note_thread(crate::sampler::Target::Game);
    let capture = capture_key();
    let (frame_no, draws, mat, frame) = with(|s| {
        if s.frames == s.trace_from {
            log::line("--- end of frame capture ---");
        }
        s.frames += 1;
        crate::effects::poll_settings();
        crate::dlss::poll_mode();
        if crate::texlist::begin_export(s.frames) {
            crate::texlist::export(s.frames, &texture_table(), crate::device::back_buffer_size());
        }
        if capture {
            s.trace_from = s.frames;
            log::line(&format!("--- frame capture (frame {}) ---", s.frames));
        }
        let n = s.draws.len();
        s.draws.clear();
        let mut frame = std::mem::take(&mut s.frame);
        s.cameras.clear();
        s.camera_best = 0;
        // start the next frame at this one's sizes, so its buffers don't regrow
        s.frame.vertices.reserve(frame.vertices.len());
        s.frame.indices.reserve(frame.indices.len());
        s.frame.vs_blocks.reserve(frame.vs_blocks.len());
        s.frame.ps_blocks.reserve(frame.ps_blocks.len());
        s.frame.cmds.reserve(frame.cmds.len());
        frame.capture = s.frames == s.trace_from;
        // the counter was just advanced
        frame.jitter = crate::dlss::jitter(s.frames - 1);
        s.vp_prev = std::mem::take(&mut s.vp_cur);
        s.reproj_cache = None;
        s.last_vs_block = None;
        s.last_ps_block = None;
        if s.render_target != -1 || s.render_depth != -1 {
            s.frame.cmds.push(Cmd::Target { color: s.render_target, depth: s.render_depth });
        }
        (s.frames, n, s.material, frame)
    });
    if frame_no <= 5 || frame_no % 600 == 0 || (frame.skipped > 0 && frame_no % 60 == 0) {
        let drawn = frame.cmds.iter().filter(|c| matches!(c, Cmd::Draw(_))).count();
        log::line(&format!(
            "frame {frame_no}: {draws} draw passes ({drawn} recorded, {} vertex bytes, {} vs / {} ps constant blocks, {} skipped, {} without shaders), last material {}",
            frame.vertices.len(),
            frame.vs_block_count,
            frame.ps_blocks.len() / PS_BLOCK,
            frame.skipped,
            frame.skipped_no_shader,
            material_name(mat)
        ));
        if profiling() {
            log::line(&format!("record ns/draw: {}", take_profile()));
        }
        // averages since the last line (600 frames, or 1 for the first ones)
        let span = if frame_no <= 5 { 1 } else { 600 };
        let calls = CALLS.swap(0, std::sync::atomic::Ordering::Relaxed) / span;
        let wait = PRESENT_WAIT_NS.swap(0, std::sync::atomic::Ordering::Relaxed) / span;
        let render = crate::render::RENDER_NS.swap(0, std::sync::atomic::Ordering::Relaxed) / span;
        let draw_mesh = DRAW_MESH_NS.swap(0, std::sync::atomic::Ordering::Relaxed) / span;
        let mutex = crate::lock::WAIT_NS.swap(0, std::sync::atomic::Ordering::Relaxed) / span;
        let lock = crate::lock::LOCK_WAIT_NS.swap(0, std::sync::atomic::Ordering::Relaxed) / span;
        log::line(&format!(
            "per frame: render thread {:.3} ms (acquire {:.3}, present {:.3}, gpu {:.3}), game waited {:.3} ms, shader mutex waits {:.3} ms (calls) / {:.3} ms (ShaderLock){}",
            render as f64 / 1e6,
            crate::render::ACQUIRE_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / span as f64 / 1e6,
            crate::render::PRESENT_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / span as f64 / 1e6,
            crate::render::GPU_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3,
            wait as f64 / 1e6,
            mutex as f64 / 1e6,
            lock as f64 / 1e6,
            if profiling() { format!(", {calls} api calls, {:.3} ms in IMesh::Draw", draw_mesh as f64 / 1e6) } else { String::new() }
        ));
    }
    crate::render::present(frame);
}


unsafe extern "C" fn take_snapshot(_t: *mut c_void) -> i16 {
    shadow::with(|s| s.take())
}
fn snapshot_flag(id: i16, f: impl Fn(&shadow::ShadowState) -> bool) -> bool {
    shadow::with(|s| s.get(id).is_some_and(|x| f(x)))
}
unsafe extern "C" fn is_translucent(_t: *mut c_void, id: i16) -> bool {
    snapshot_flag(id, |s| s.blend)
}
unsafe extern "C" fn is_alpha_tested(_t: *mut c_void, id: i16) -> bool {
    snapshot_flag(id, |s| s.alpha_test)
}
unsafe extern "C" fn uses_shaders(_t: *mut c_void, id: i16) -> bool {
    snapshot_flag(id, |s| s.vertex_shader.is_some())
}
unsafe extern "C" fn is_depth_write_enabled(_t: *mut c_void, id: i16) -> bool {
    snapshot_flag(id, |s| s.depth_write)
}
/// `ComputeVertexUsage`: merge the passes' formats
unsafe extern "C" fn compute_vertex_format(_t: *mut c_void, n: i32, ids: *const i16) -> u64 {
    if n <= 0 || ids.is_null() {
        return 0;
    }
    let ids = unsafe { std::slice::from_raw_parts(ids, n as usize) };
    shadow::with(|s| {
        let fmts: Vec<u64> = ids.iter().filter_map(|&i| s.get(i).map(|x| x.vertex_format)).collect();
        if fmts.len() == 1 {
            return fmts[0];
        }
        let (mut flags, mut bones, mut user, mut tc) = (0u64, 0usize, 0usize, [0i32; MAX_TEXCOORDS]);
        for f in fmts {
            flags |= vertex_flags(f);
            bones = bones.max(num_bone_weights(f));
            user = user.max(user_data_size(f));
            for (j, t) in tc.iter_mut().enumerate() {
                *t = (*t).max(texcoord_size(j, f) as i32);
            }
        }
        crate::abi::compute_vertex_format(flags, MAX_TEXCOORDS, Some(&tc), bones, user)
    })
}
unsafe extern "C" fn compare_snapshots(_t: *mut c_void, a: i16, b: i16) -> i32 {
    a as i32 - b as i32
}


#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn create_texture(_t: *mut c_void, w: i32, h: i32, depth: i32, fmt: i32, mips: i32, copies: i32, flags: i32, name: *const c_char, group: *const c_char) -> i32 {
    let str = |p: *const c_char| if p.is_null() { String::new() } else { unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned() };
    let (name, group) = (str(name), str(group));
    static CREATED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if flags & 0xa != 0 || CREATED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 40 {
        log::line(&format!("CreateTexture '{name}' {w}x{h}x{depth} fmt {fmt} mips {mips} copies {copies} flags {flags:#x}"));
    }
    with(|s| {
        let id = s.next_texture;
        s.next_texture += 1;
        texture_table().insert(id, Texture { width: w, height: h, depth, format: fmt, mips, copies, flags, name, group, filter: [1, 1], ..Default::default() });
        id
    })
}
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn create_textures(t: *mut c_void, handles: *mut i32, count: i32, w: i32, h: i32, depth: i32, fmt: i32, mips: i32, copies: i32, flags: i32, name: *const c_char, group: *const c_char) {
    for i in 0..count.max(0) as usize {
        let id = unsafe { create_texture(t, w, h, depth, fmt, mips, copies, flags, name, group) };
        unsafe { *handles.add(i) = id };
    }
}
unsafe extern "C" fn create_depth_texture(_t: *mut c_void, fmt: i32, w: i32, h: i32, name: *const c_char, _texture: bool) -> i32 {
    let name = if name.is_null() { String::new() } else { unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy().into_owned() };
    with(|s| {
        let id = s.next_texture;
        s.next_texture += 1;
        texture_table().insert(id, Texture { width: w, height: h, depth: 1, format: fmt, mips: 1, copies: 1, depth_target: true, name, group: "RenderTargets".into(), ..Default::default() });
        id
    })
}
unsafe extern "C" fn delete_texture(_t: *mut c_void, id: i32) {
    texture_table().remove(&id);
}
unsafe extern "C" fn is_texture(_t: *mut c_void, id: i32) -> bool {
    texture_table().contains_key(&id)
}
unsafe extern "C" fn modify_texture(_t: *mut c_void, id: i32) {
    with(|s| s.modify_texture = id);
}
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn tex_image_2d(_t: *mut c_void, level: i32, face: i32, _dst_fmt: i32, z: i32, w: i32, h: i32, src_fmt: i32, _tiled: bool, data: *const u8) {
    if data.is_null() {
        return;
    }
    let bytes = image_bytes(src_fmt, w, h);
    let img = texconv::convert(src_fmt, w as u32, h as u32, unsafe { std::slice::from_raw_parts(data, bytes) });
    let id = with(|s| s.modify_texture);
    {
        let mut table = texture_table();
        let Some(t) = table.get_mut(&id) else { return };
        let Some(img) = img else { return };
        if t.depth > 1 {
            // one mip 0 slice per call
            if level == 0 && (0..t.depth).contains(&z) {
                if t.slices.len() < t.depth as usize {
                    t.slices.resize(t.depth as usize, None);
                }
                t.slices[z as usize] = Some(img);
                t.version += 1;
                t.full_dirty = true;
            }
        } else if z == 0 {
            t.set_level(face, level, img);
        }
        // diagnosing repeated full uploads: big top mips per texture
        if level == 0 && w * h >= 1024 * 1024 {
            static UPLOADS: Mutex<Option<HashMap<i32, u32>>> = Mutex::new(None);
            let mut u = UPLOADS.lock().unwrap_or_else(|e| e.into_inner());
            let n = u.get_or_insert_with(HashMap::new).entry(id).or_default();
            *n += 1;
            if *n <= 2 || n.is_power_of_two() {
                log::line(&format!("upload #{n} of '{}' ({id}) {w}x{h} fmt {src_fmt} face {face}", t.name));
            }
        }
    }
}
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn tex_sub_image_2d(_t: *mut c_void, level: i32, face: i32, x: i32, y: i32, z: i32, w: i32, h: i32, src_fmt: i32, stride: i32, _tiled: bool, data: *const u8) {
    if z != 0 || data.is_null() || w <= 0 || h <= 0 {
        return;
    }
    let bpp = texconv::bytes_per_pixel(src_fmt);
    let row = if stride > 0 { stride as usize } else { w as usize * bpp };
    let src = unsafe { std::slice::from_raw_parts(data, row * (h as usize - 1) + w as usize * bpp) };
    let id = with(|s| s.modify_texture);
    {
        let mut table = texture_table();
        let Some(t) = table.get_mut(&id) else { return };
        let (tw, th, fmt) = (t.width, t.height, t.format);
        let Some(levels) = t.face_mut(face) else { return };
        let l = level.max(0) as usize;
        if levels.len() <= l {
            levels.resize(l + 1, None);
        }
        let created = levels[l].is_none();
        if created {
            let (lw, lh) = ((tw >> l).max(1) as u32, (th >> l).max(1) as u32);
            levels[l] = Some(Image::blank(fmt, lw, lh));
        }
        if texconv::blit(levels[l].as_mut().unwrap(), x as u32, y as u32, w as u32, h as u32, src_fmt, src, row) {
            if created {
                t.version += 1;
                t.full_dirty = true;
            } else {
                t.touch(face, l, [x as u32, y as u32, w as u32, h as u32]);
            }
        }
    }
}

/// staging memory: rgba8888, or rgba16161616f for 16-bit textures
struct Locked {
    id: i32,
    level: i32,
    rect: [i32; 4],
    data: Vec<u8>,
    wide: bool,
}
static LOCKED: Mutex<Option<Locked>> = Mutex::new(None);

/// hand out staging memory through the inline CPixelWriter
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn tex_lock(_t: *mut c_void, level: i32, face: i32, x: i32, y: i32, w: i32, h: i32, writer: *mut u8) -> bool {
    if writer.is_null() || face != 0 || w <= 0 || h <= 0 || w > 16384 || h > 16384 {
        return false;
    }
    let id = with(|s| s.modify_texture);
    if !texture_table().contains_key(&id) {
        return false;
    }
    // 16-bit textures (hdr lightmaps) get a half float writer, so lightmaps keep their precision
    let wide = texture_table().get(&id).is_some_and(|t| texconv::is_wide(t.format));
    let bpp: usize = if wide { 8 } else { 4 };
    // like a d3d lock, starts with the current texels
    let mut data = vec![0u8; w as usize * h as usize * bpp];
    // copy the current texels in (skipped when there are none)
    (|| {
        let table = texture_table();
        let Some(Some(img)) = table.get(&id).and_then(|t| t.levels.get(level.max(0) as usize)) else { return };
        let (Image::Rgba8 { width, height, data: src } | Image::Rgba16f { width, height, data: src }) = img else { return };
        if img.pixel_bytes() != bpp {
            return;
        }
        for row in 0..h as usize {
            let sy = y as usize + row;
            if sy >= *height as usize || x < 0 || x as u32 >= *width {
                continue;
            }
            let n = (w as usize).min(*width as usize - x as usize) * bpp;
            let so = (sy * *width as usize + x as usize) * bpp;
            let d = row * w as usize * bpp;
            data[d..d + n].copy_from_slice(&src[so..so + n]);
        }
    })();
    let p = data.as_mut_ptr();
    unsafe {
        // CPixelWriter layout: base, bits, bytes per row, size, flags, rgba shifts, rgba masks
        std::ptr::write_unaligned(writer as *mut *mut u8, p);
        std::ptr::write_unaligned(writer.add(8) as *mut *mut u8, p);
        std::ptr::write_unaligned(writer.add(16) as *mut u16, (w as usize * bpp) as u16);
        *writer.add(18) = bpp as u8;
        // float | 16-bit float
        *writer.add(19) = if wide { 3 } else { 0 };
        let (step, mask) = if wide { (16i16, 0xffffu32) } else { (8, 0xff) };
        for i in 0..4 {
            std::ptr::write_unaligned(writer.add(20 + 2 * i) as *mut i16, step * i as i16);
            std::ptr::write_unaligned(writer.add(28 + 4 * i) as *mut u32, mask);
        }
    }
    *LOCKED.lock().unwrap_or_else(|e| e.into_inner()) = Some(Locked { id, level, rect: [x, y, w, h], data, wide });
    true
}
unsafe extern "C" fn tex_unlock(_t: *mut c_void) {
    let Some(l) = LOCKED.lock().unwrap_or_else(|e| e.into_inner()).take() else { return };
    {
        let mut table = texture_table();
        let Some(t) = table.get_mut(&l.id) else { return };
        let lv = l.level.max(0) as usize;
        if t.levels.len() <= lv {
            t.levels.resize(lv + 1, None);
        }
        let created = t.levels[lv].is_none();
        if created {
            let (lw, lh) = ((t.width >> lv).max(1) as u32, (t.height >> lv).max(1) as u32);
            t.levels[lv] = Some(Image::blank(t.format, lw, lh));
        }
        let [x, y, w, h] = l.rect;
        // rgba8888 / rgba16161616f
        let (src_fmt, bpp) = if l.wide { (24, 8) } else { (0, 4) };
        if texconv::blit(t.levels[lv].as_mut().unwrap(), x as u32, y as u32, w as u32, h as u32, src_fmt, &l.data, w as usize * bpp) {
            if created {
                t.version += 1;
                t.full_dirty = true;
            } else {
                t.touch(0, lv, [x as u32, y as u32, w as u32, h as u32]);
            }
        }
    }
}

// gmod x64 IVTFTexture slots, checked against CVTFTexture's vtable
const VTF_WIDTH: usize = 14;
const VTF_HEIGHT: usize = 15;
const VTF_MIP_COUNT: usize = 17;
const VTF_FORMAT: usize = 20;
const VTF_IMAGE_DATA_FRAME_FACE_MIP: usize = 38;

/// every mip of `frame`; vtf cube faces are in d3d order
unsafe extern "C" fn tex_image_from_vtf(_t: *mut c_void, vtf: usize, frame: i32) {
    if vtf == 0 {
        return;
    }
    let (w, h, mips, fmt) = unsafe { (vcall::<i32>(vtf, VTF_WIDTH), vcall::<i32>(vtf, VTF_HEIGHT), vcall::<i32>(vtf, VTF_MIP_COUNT), vcall::<i32>(vtf, VTF_FORMAT)) };
    let id = with(|s| s.modify_texture);
    let (cube, depth) = texture_table().get(&id).map_or((false, 1), |t| (t.is_cube(), t.depth));
    if depth > 1 {
        // mip 0 holds the slices back to back
        let data: *const u8 = unsafe {
            let vt = *(vtf as *const *const usize);
            let f: unsafe extern "C" fn(usize, i32, i32, i32) -> *const u8 = std::mem::transmute(*vt.add(VTF_IMAGE_DATA_FRAME_FACE_MIP));
            f(vtf, frame, 0, 0)
        };
        if data.is_null() {
            return;
        }
        let slice = image_bytes(fmt, w, h);
        let slices: Vec<Option<Image>> = (0..depth as usize).map(|z| texconv::convert(fmt, w as u32, h as u32, unsafe { std::slice::from_raw_parts(data.add(z * slice), slice) })).collect();
        if let Some(t) = texture_table().get_mut(&id) {
            t.slices = slices;
            t.version += 1;
            t.full_dirty = true;
        }
        return;
    }
    let faces = if cube { 6 } else { 1 };
    let mut out: Vec<Vec<Option<Image>>> = Vec::new();
    let mut why = String::new();
    for face in 0..faces {
        let mut levels = Vec::new();
        for mip in 0..mips.max(1) {
            let (mw, mh) = ((w >> mip).max(1), (h >> mip).max(1));
            let data: *const u8 = unsafe {
                let vt = *(vtf as *const *const usize);
                let f: unsafe extern "C" fn(usize, i32, i32, i32) -> *const u8 = std::mem::transmute(*vt.add(VTF_IMAGE_DATA_FRAME_FACE_MIP));
                f(vtf, frame, face, mip)
            };
            if data.is_null() {
                why = format!("ImageData({frame}, {face}, {mip}) returned null");
                break;
            }
            let bytes = image_bytes(fmt, mw, mh);
            let img = texconv::convert(fmt, mw as u32, mh as u32, unsafe { std::slice::from_raw_parts(data, bytes) });
            if img.is_none() && why.is_empty() {
                why = format!("format {fmt} not converted ({bytes} bytes for {mw}x{mh})");
            }
            levels.push(img);
        }
        out.push(levels);
    }
    if out.first().and_then(|l| l.first()).is_none_or(|l| l.is_none()) {
        static FAILS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if FAILS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 40 {
            let name = texture_table().get(&id).map(|t| t.name.clone()).unwrap_or_default();
            log::line(&format!("TexImageFromVTF(texture {id} '{name}', frame {frame}): {w}x{h} fmt {fmt} mips {mips}: no usable mip 0: {why}"));
        }
    }
    {
        if let Some(t) = texture_table().get_mut(&id) {
            for (face, levels) in out.into_iter().enumerate() {
                for (i, img) in levels.into_iter().enumerate() {
                    if let Some(img) = img {
                        t.set_level(face as i32, i as i32, img);
                    }
                }
            }
        }
    }
}
fn with_modify_texture(f: impl FnOnce(&mut Texture)) {
    let id = with(|s| s.modify_texture);
    if let Some(t) = texture_table().get_mut(&id) {
        f(t);
    }
}
unsafe extern "C" fn tex_min_filter(_t: *mut c_void, mode: i32) {
    with_modify_texture(|t| t.filter[0] = mode);
}
unsafe extern "C" fn tex_mag_filter(_t: *mut c_void, mode: i32) {
    with_modify_texture(|t| t.filter[1] = mode);
}
unsafe extern "C" fn tex_wrap(_t: *mut c_void, coord: i32, mode: i32) {
    with_modify_texture(|t| {
        if let Some(w) = t.wrap.get_mut(coord as usize) {
            *w = mode;
        }
    });
}
unsafe extern "C" fn clear_buffers(_t: *mut c_void, color: bool, depth: bool, stencil: bool, w: i32, h: i32) {
    trace(|| format!("  ClearBuffers(color {color}, depth {depth}, {w}x{h})"));
    with(|s| {
        let c = color.then_some(s.clear_color);
        if depth && s.render_target == -1 {
            s.cameras.clear();
            s.camera_best = 0;
        }
        push_cmd(s, Cmd::Clear { color: c, depth, stencil: stencil.then_some(0) });
    });
}
fn set_stencil(f: impl FnOnce(&mut Stencil)) {
    with(|s| f(&mut s.stencil));
}
unsafe extern "C" fn stencil_enable(_t: *mut c_void, v: bool) {
    set_stencil(|s| s.enable = v);
}
unsafe extern "C" fn stencil_fail(_t: *mut c_void, v: i32) {
    set_stencil(|s| s.fail = v);
}
unsafe extern "C" fn stencil_zfail(_t: *mut c_void, v: i32) {
    set_stencil(|s| s.zfail = v);
}
unsafe extern "C" fn stencil_pass(_t: *mut c_void, v: i32) {
    set_stencil(|s| s.pass = v);
}
unsafe extern "C" fn stencil_func(_t: *mut c_void, v: i32) {
    set_stencil(|s| s.func = v);
}
unsafe extern "C" fn stencil_ref(_t: *mut c_void, v: i32) {
    set_stencil(|s| s.reference = v);
}
unsafe extern "C" fn stencil_test_mask(_t: *mut c_void, v: u32) {
    set_stencil(|s| s.test_mask = v);
}
unsafe extern "C" fn stencil_write_mask(_t: *mut c_void, v: u32) {
    set_stencil(|s| s.write_mask = v);
}
/// treated as a whole-target stencil clear
unsafe extern "C" fn clear_stencil_rect(_t: *mut c_void, _x0: i32, _y0: i32, _x1: i32, _y1: i32, value: i32) {
    with(|s| push_cmd(s, Cmd::Clear { color: None, depth: false, stencil: Some(value as u8) }));
}

/// per pass: texture matrices to identity, model matrix mode, default shader indices
unsafe extern "C" fn dynamic_default_state(_t: *mut c_void) {
    with(|s| {
        for m in 2..6 {
            *s.matrices[m].last_mut().unwrap() = IDENTITY;
        }
        s.matrix_mode = 10;
        s.vs_index = 0;
        s.ps_index = 0;
    });
}

/// dx9: user clip planes off, then DrawClearBufferQuad
unsafe extern "C" fn clear_obey_stencil_ex(_t: *mut c_void, color: bool, alpha: bool, depth: bool) {
    if !color && !alpha && !depth {
        return;
    }
    let (c, clip) = with(|s| (s.clear_color, std::mem::take(&mut s.clip_enabled)));
    crate::shader_util::draw_clear_buffer_quad(c[0], c[1], c[2], c[3], color, alpha, depth);
    with(|s| s.clip_enabled = clip);
}
unsafe extern "C" fn clear_obey_stencil(t: *mut c_void, color: bool, depth: bool) {
    unsafe { clear_obey_stencil_ex(t, color, color, depth) };
}
unsafe extern "C" fn full_screen_stencil_operation(_t: *mut c_void) {
    let clip = with(|s| std::mem::take(&mut s.clip_enabled));
    crate::shader_util::draw_clear_buffer_quad(0, 0, 0, 0, false, false, false);
    with(|s| s.clip_enabled = clip);
}

// occlusion queries: brackets in the command stream, read back asynchronously (usually a frame
// late)

unsafe extern "C" fn create_query(_t: *mut c_void) -> u64 {
    with(|s| {
        let id = s.next_query;
        s.next_query += 1;
        s.queries.insert(id, (0, 0));
        id
    })
}
unsafe extern "C" fn destroy_query(_t: *mut c_void, id: u64) {
    with(|s| s.queries.remove(&id));
}
unsafe extern "C" fn begin_query(_t: *mut c_void, id: u64) {
    with(|s| {
        let frame = s.frames;
        if let Some(q) = s.queries.get_mut(&id) {
            q.0 += 1;
            q.1 = frame;
            let issue = q.0;
            push_cmd(s, Cmd::QueryBegin { id, issue });
        }
    });
}
unsafe extern "C" fn end_query(_t: *mut c_void, id: u64) {
    with(|s| {
        if s.queries.contains_key(&id) {
            push_cmd(s, Cmd::QueryEnd { id });
        }
    });
}
/// latest issue's count, else PENDING (-1). a flushed query from the frame still being recorded
/// can't finish, so it reports ERROR (-2) to end the material system's wait loop.
unsafe extern "C" fn query_pixels(_t: *mut c_void, id: u64, flush: bool) -> i32 {
    let Some((issue, frame, now)) = with(|s| s.queries.get(&id).map(|q| (q.0, q.1, s.frames))) else { return -2 };
    if issue == 0 {
        return -2;
    }
    let lookup = || QUERY_RESULTS.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| m.get(&id).copied());
    if let Some((i, n)) = lookup() {
        if i == issue {
            return n.min(i32::MAX as u64) as i32;
        }
    }
    if !flush {
        return -1;
    }
    if frame >= now {
        return -2;
    }
    crate::render::wait_for_queries();
    match lookup() {
        Some((i, n)) if i == issue => n.min(i32::MAX as u64) as i32,
        _ => -2,
    }
}

// flashlight state: stored and handed back to the shaders

/// FlashlightState_t (136 bytes in sdk2013, room for gmod's additions), world-to-texture matrix,
/// depth texture
struct Flashlight {
    state: [u8; 256],
    world_to_texture: [f32; 16],
    depth: usize,
}
static FLASHLIGHT: Mutex<Flashlight> = Mutex::new(Flashlight { state: [0; 256], world_to_texture: [0.0; 16], depth: 0 });

unsafe extern "C" fn set_flashlight_state_ex(_t: *mut c_void, state: *const u8, m: *const f32, depth: usize) {
    if state.is_null() || m.is_null() {
        return;
    }
    let mut f = FLASHLIGHT.lock().unwrap_or_else(|e| e.into_inner());
    unsafe {
        std::ptr::copy_nonoverlapping(state, f.state.as_mut_ptr(), 256);
        std::ptr::copy_nonoverlapping(m, f.world_to_texture.as_mut_ptr(), 16);
    }
    f.depth = depth;
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3 {
        // m_bEnableShadows is at offset 84
        log::line(&format!("SetFlashlightStateEx: shadows {} depth texture {depth:#x}", f.state[84]));
    }
}
unsafe extern "C" fn set_flashlight_state(t: *mut c_void, state: *const u8, m: *const f32) {
    unsafe { set_flashlight_state_ex(t, state, m, 0) };
}
unsafe extern "C" fn get_flashlight_state_ex(_t: *mut c_void, m: *mut f32, depth: *mut usize) -> *const u8 {
    let f = FLASHLIGHT.lock().unwrap_or_else(|e| e.into_inner());
    unsafe {
        if !m.is_null() {
            std::ptr::copy_nonoverlapping(f.world_to_texture.as_ptr(), m, 16);
        }
        if !depth.is_null() {
            *depth = f.depth;
        }
    }
    f.state.as_ptr()
}
unsafe extern "C" fn get_flashlight_state(t: *mut c_void, m: *mut f32) -> *const u8 {
    unsafe { get_flashlight_state_ex(t, m, std::ptr::null_mut()) }
}
unsafe extern "C" fn get_current_color_correction(_t: *mut c_void, info: *mut c_void) {
    crate::shader_util::get_current_color_correction(info);
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) && !info.is_null() {
        let b = unsafe { std::slice::from_raw_parts(info as *const u8, 64) };
        let i = |o: usize| i32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let fl = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        crate::log::line(&format!("GetCurrentColorCorrection: enabled {} lookups {} default weight {} weights {:?}", b[0], i(4), fl(8), [fl(48), fl(52), fl(56), fl(60)]));
    }
}

unsafe extern "C" fn in_flashlight_mode(_t: *mut c_void) -> bool {
    crate::shader_util::in_flashlight_mode()
}

fn ov(f: impl FnOnce(&mut Overrides)) {
    with(|s| f(&mut s.ov));
}
unsafe extern "C" fn force_depth_func_equals(_t: *mut c_void, v: bool) {
    ov(|o| o.depth_func_equal = v);
}
unsafe extern "C" fn override_depth_enable(_t: *mut c_void, enable: bool, write: bool) {
    ov(|o| o.depth_write = enable.then_some(write));
}
unsafe extern "C" fn override_alpha_write(_t: *mut c_void, enable: bool, write: bool) {
    ov(|o| o.alpha_write = enable.then_some(write));
}
unsafe extern "C" fn override_color_write(_t: *mut c_void, enable: bool, write: bool) {
    ov(|o| o.color_write = enable.then_some(write));
}
/// gmod render.OverrideBlend
unsafe extern "C" fn override_blend(_t: *mut c_void, enable: bool, separate: bool, src: i32, dst: i32, op: i32) {
    ov(|o| o.blend = enable.then_some((separate, src, dst, op)));
}
unsafe extern "C" fn override_blend_alpha(_t: *mut c_void, enable: bool, _separate: bool, src: i32, dst: i32, op: i32) {
    ov(|o| o.blend_alpha = enable.then_some((src, dst, op)));
}
/// gmod render.PushFilterMag / Min
unsafe extern "C" fn force_filter_mode(_t: *mut c_void, force: bool, filter: i32) {
    ov(|o| o.filter = force.then_some(filter));
}
unsafe extern "C" fn set_scissor_rect(_t: *mut c_void, l: i32, t: i32, r: i32, b: i32, enable: bool) {
    with(|s| s.scissor = enable.then_some([l, t, r, b]));
}
unsafe extern "C" fn set_int_param(_t: *mut c_void, n: i32, v: i32) {
    with(|s| {
        if let Some(p) = s.int_params.get_mut(n as usize) {
            *p = v;
        }
    });
}
unsafe extern "C" fn set_float_param(_t: *mut c_void, n: i32, v: f32) {
    with(|s| {
        if let Some(p) = s.float_params.get_mut(n as usize) {
            *p = v;
        }
    });
}
unsafe extern "C" fn set_vector_param(_t: *mut c_void, n: i32, v: *const f32) {
    if v.is_null() {
        return;
    }
    let v = unsafe { [*v, *v.add(1), *v.add(2)] };
    with(|s| {
        if let Some(p) = s.vector_params.get_mut(n as usize) {
            *p = v;
        }
    });
}
unsafe extern "C" fn get_int_param(_t: *mut c_void, n: i32) -> i32 {
    with(|s| s.int_params.get(n as usize).copied().unwrap_or(0))
}
unsafe extern "C" fn get_float_param(_t: *mut c_void, n: i32) -> f32 {
    with(|s| s.float_params.get(n as usize).copied().unwrap_or(0.0))
}
/// returned through a hidden pointer (msvc x64)
unsafe extern "C" fn get_vector_param(_t: *mut c_void, out: *mut f32, n: i32) -> *mut f32 {
    let v = with(|s| s.vector_params.get(n as usize).copied().unwrap_or([0.0; 3]));
    if !out.is_null() {
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), out, 3) };
    }
    out
}
unsafe extern "C" fn should_write_depth_to_dest_alpha(_t: *mut c_void) -> bool {
    with(|s| s.fog.mode != 2 && s.int_params[10] != 0)
}
unsafe extern "C" fn set_anisotropic_level(_t: *mut c_void, n: i32) {
    ANISOTROPY.store(n.clamp(1, 16), std::sync::atomic::Ordering::Relaxed);
}
/// world space ax + by + cz = d, stored with d negated
unsafe extern "C" fn set_clip_plane(_t: *mut c_void, i: i32, p: *const f32) {
    if p.is_null() || !(0..6).contains(&i) {
        return;
    }
    let v = unsafe { [*p, *p.add(1), *p.add(2), -*p.add(3)] };
    with(|s| s.clip_planes[i as usize] = v);
}
unsafe extern "C" fn enable_clip_plane(_t: *mut c_void, i: i32, enable: bool) {
    if (0..6).contains(&i) {
        with(|s| if enable { s.clip_enabled |= 1 << i } else { s.clip_enabled &= !(1 << i) });
    }
}
unsafe extern "C" fn enable_user_clip_transform_override(_t: *mut c_void, enable: bool) {
    with(|s| s.clip_transform_override = enable);
}
/// VMatrix is the transpose of the d3dx layout
unsafe extern "C" fn user_clip_transform(_t: *mut c_void, m: *const f32) {
    if m.is_null() {
        return;
    }
    let v = unsafe { std::slice::from_raw_parts(m, 16) };
    let t: [f32; 16] = std::array::from_fn(|i| v[(i % 4) * 4 + i / 4]);
    with(|s| s.clip_transform = Some(t));
}

/// first calls of the early frames
fn trace(msg: impl FnOnce() -> String) {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if with(|s| s.frames) < 6 && N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 400 {
        log::line(&msg());
    }
}
unsafe extern "C" fn set_render_target(_t: *mut c_void, color: i32, depth: i32) {
    trace(|| format!("  SetRenderTarget(color {color}, depth {depth})"));
    set_target(color, depth);
}
unsafe extern "C" fn set_render_target_ex(_t: *mut c_void, id: i32, color: i32, depth: i32) {
    trace(|| format!("  SetRenderTargetEx({id}, color {color}, depth {depth})"));
    if id == 0 {
        set_target(color, depth);
    }
}
/// -1 / -2 keep the target's own depth; a shadow depth texture becomes the attachment
fn set_target(color: i32, depth: i32) {
    // 0 also means the back buffer
    let color = if color == 0 { -1 } else { color };
    with(|s| {
        let depth = if depth > 0 && texture_table().get(&depth).is_some_and(|t| is_depth_format(t.format)) { depth } else { -1 };
        if s.render_target != color || s.render_depth != depth {
            s.render_target = color;
            s.render_depth = depth;
            push_cmd(s, Cmd::Target { color, depth });
        }
    });
}

/// NV_DST16 .. ATI_DST24
pub fn is_depth_format(fmt: i32) -> bool {
    (IMAGE_FORMAT_NV_DST16..=IMAGE_FORMAT_ATI_DST24).contains(&fmt)
}
const IMAGE_FORMAT_NV_DST16: i32 = 30;
const IMAGE_FORMAT_NV_DST24: i32 = 31;
const IMAGE_FORMAT_ATI_DST24: i32 = 35;
const IMAGE_FORMAT_NV_NULL: i32 = 36;

/// nvidia DST24 with hardware pcf (GetShadowFilterMode 0)
unsafe extern "C" fn shadow_depth_format(_t: *mut c_void) -> i32 {
    IMAGE_FORMAT_NV_DST24
}
unsafe extern "C" fn null_texture_format(_t: *mut c_void) -> i32 {
    IMAGE_FORMAT_NV_NULL
}
/// known once the renderer is up, before that 2 / 4 / 8
unsafe extern "C" fn supports_msaa_mode(_t: *mut c_void, samples: i32) -> bool {
    let known = crate::render::MSAA_SUPPORTED.load(std::sync::atomic::Ordering::Relaxed);
    (2..=16).contains(&samples) && if known == 0 { matches!(samples, 2 | 4 | 8) } else { known & (1 << samples) != 0 }
}
/// nvidia's d3d9 csaa doesn't exist under vulkan
unsafe extern "C" fn supports_csaa_mode(_t: *mut c_void, _samples: i32, _quality: i32) -> bool {
    false
}
unsafe extern "C" fn supports_shadow_depth_textures(_t: *mut c_void) -> bool {
    true
}
unsafe extern "C" fn set_shadow_depth_bias(_t: *mut c_void, slope: f32, bias: f32) {
    with(|s| s.shadow_bias = [slope, bias]);
}
fn rect(p: *const i32) -> Option<[i32; 4]> {
    (!p.is_null()).then(|| unsafe { [*p, *p.add(1), *p.add(2), *p.add(3)] })
}
unsafe extern "C" fn copy_rt_to_texture(_t: *mut c_void, dst: i32) {
    with(|s| push_cmd(s, Cmd::Copy { dst, src_rect: None, dst_rect: None }));
}
/// only target 0 is ever bound
unsafe extern "C" fn copy_texture_to_rt_ex(_t: *mut c_void, id: i32, src: i32, src_rect: *const i32, dst_rect: *const i32) {
    if id == 0 && src > 0 {
        let (src_rect, dst_rect) = (rect(src_rect), rect(dst_rect));
        with(|s| push_cmd(s, Cmd::CopyIn { src, src_rect, dst_rect }));
    }
}
unsafe extern "C" fn copy_rt_to_texture_ex(_t: *mut c_void, dst: i32, id: i32, src: *const i32, dst_rect: *const i32) {
    if id == 0 {
        let (src_rect, dst_rect) = (rect(src), rect(dst_rect));
        with(|s| push_cmd(s, Cmd::Copy { dst, src_rect, dst_rect }));
    }
}
unsafe extern "C" fn bind_texture(_t: *mut c_void, sampler: i32, id: i32) {
    with(|s| {
        crate::texlist::note_bind(id);
        if let Some(b) = s.bound_textures.get_mut(sampler as usize) {
            if *b != id {
                s.border_clamp &= !(1 << sampler);
            }
            *b = id;
        }
    });
}
/// gmod's dx9 sets border addressing on the sampler; border color stays transparent black
unsafe extern "C" fn sampler_border_clamp(_t: *mut c_void, sampler: i32) {
    if (0..16).contains(&sampler) {
        with(|s| s.border_clamp |= 1 << sampler);
    }
}

/// 0 when unknown
pub fn image_bytes(fmt: i32, w: i32, h: i32) -> usize {
    let (w, h) = (w.max(0) as usize, h.max(0) as usize);
    let block = |bytes: usize| w.div_ceil(4).max(1) * h.div_ceil(4).max(1) * bytes;
    match fmt {
        0 | 1 | 11 | 12 | 16 => w * h * 4,
        2 | 3 | 9 | 10 => w * h * 3,
        4 | 17 | 18 | 19 | 6 => w * h * 2,
        5 | 7 | 8 => w * h,
        13 => block(8),
        14 | 15 => block(16),
        20 => block(8),
        21 | 22 => w * h * 2,
        23 => w * h * 4,
        24 => w * h * 8,
        25 => w * h * 8,
        26 => w * h * 4,
        27 => w * h * 4,
        28 => w * h * 12,
        29 => w * h * 16,
        _ => 0,
    }
}


unsafe extern "C" fn set_vs_constant(_t: *mut c_void, reg: i32, v: *const f32, n: i32, _force: bool) {
    set_constants(reg, v, n, true);
}
unsafe extern "C" fn set_ps_constant(_t: *mut c_void, reg: i32, v: *const f32, n: i32, _force: bool) {
    set_constants(reg, v, n, false);
}
fn set_constants(reg: i32, v: *const f32, n: i32, vertex: bool) {
    if v.is_null() || reg < 0 || n <= 0 {
        return;
    }
    let src = unsafe { std::slice::from_raw_parts(v, n as usize * 4) };
    with(|s| {
        if vertex {
            let (lo, hi) = (reg as usize, reg as usize + n as usize);
            s.vs_high = s.vs_high.max(hi - 1);
            // registers standard_vs_constants owns: redo it next draw
            if lo < 17 || (lo < 48 && hi > 27) || hi > 58 {
                s.vs_std_key = None;
            }
        }
        let dst = if vertex { &mut s.vs_constants } else { &mut s.ps_constants };
        let mut changed = false;
        for (i, c) in src.chunks_exact(4).enumerate() {
            if let Some(d) = dst.get_mut(reg as usize + i) {
                if d[..] != *c {
                    d.copy_from_slice(c);
                    changed = true;
                }
            }
        }
        // only real changes start a new constant block
        if changed {
            if vertex { s.vs_dirty = true } else { s.ps_dirty = true }
        }
    });
}
unsafe extern "C" fn set_vs_index(_t: *mut c_void, i: i32) {
    with(|s| s.vs_index = i);
}
unsafe extern "C" fn set_ps_index(_t: *mut c_void, i: i32) {
    with(|s| s.ps_index = i);
}
unsafe extern "C" fn set_viewports(_t: *mut c_void, n: i32, vp: *const i32) {
    if n > 0 && !vp.is_null() {
        // version, x, y, width, height, min z, max z
        let v = unsafe { std::slice::from_raw_parts(vp, 5) };
        let z = unsafe { std::slice::from_raw_parts(vp.add(5) as *const f32, 2) };
        let depth = [z[0].clamp(0.0, 1.0), z[1].clamp(0.0, 1.0)];
        trace(|| format!("  SetViewports({n}, version {} x {} y {} w {} h {} z {depth:?})", v[0], v[1], v[2], v[3], v[4]));
        with(|s| {
            s.viewport = [v[1], v[2], v[3], v[4]];
            s.viewport_depth = if depth[1] > depth[0] { depth } else { [0.0, 1.0] };
        });
    }
}
unsafe extern "C" fn get_viewports(_t: *mut c_void, vp: *mut i32, max: i32) -> i32 {
    if max > 0 && !vp.is_null() {
        let v = with(|s| s.viewport);
        unsafe {
            *vp = 1;
            for i in 0..4 {
                *vp.add(1 + i) = v[i];
            }
            *(vp.add(5) as *mut f32) = 0.0;
            *(vp.add(6) as *mut f32) = 1.0;
        }
    }
    1
}
unsafe extern "C" fn matrix_mode(_t: *mut c_void, mode: i32) {
    with(|s| s.matrix_mode = mode.clamp(0, 10) as usize);
}
unsafe extern "C" fn push_matrix(_t: *mut c_void) {
    with(|s| {
        let st = &mut s.matrices[s.matrix_mode];
        let top = *st.last().unwrap();
        st.push(top);
    });
}
unsafe extern "C" fn pop_matrix(_t: *mut c_void) {
    with(|s| {
        let st = &mut s.matrices[s.matrix_mode];
        if st.len() > 1 {
            st.pop();
        }
    });
}
unsafe extern "C" fn load_matrix(_t: *mut c_void, m: *const f32) {
    if m.is_null() {
        return;
    }
    let mut v = [0.0; 16];
    v.copy_from_slice(unsafe { std::slice::from_raw_parts(m, 16) });
    with(|s| *s.matrices[s.matrix_mode].last_mut().unwrap() = v);
}
unsafe extern "C" fn load_identity(_t: *mut c_void) {
    with(|s| *s.matrices[s.matrix_mode].last_mut().unwrap() = IDENTITY);
}
unsafe extern "C" fn get_matrix(_t: *mut c_void, mode: i32, dst: *mut f32) {
    if dst.is_null() {
        return;
    }
    let m = with(|s| *s.matrices[mode.clamp(0, 10) as usize].last().unwrap());
    unsafe { std::ptr::copy_nonoverlapping(m.as_ptr(), dst, 16) };
}
unsafe extern "C" fn clear_color3(_t: *mut c_void, r: u8, g: u8, b: u8) {
    with(|s| s.clear_color = [r, g, b, 255]);
}
unsafe extern "C" fn clear_color4(_t: *mut c_void, r: u8, g: u8, b: u8, a: u8) {
    with(|s| s.clear_color = [r, g, b, a]);
}


unsafe extern "C" fn current_time(_t: *mut c_void) -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}
unsafe extern "C" fn back_buffer_dims(_t: *mut c_void, w: *mut i32, h: *mut i32) {
    let (bw, bh) = crate::device::back_buffer_size();
    unsafe {
        *w = bw;
        *h = bh;
    }
}
unsafe extern "C" fn lightmap_dims(_t: *mut c_void, w: *mut i32, h: *mut i32) {
    unsafe {
        *w = 1;
        *h = 1;
    }
}
unsafe extern "C" fn same_format(_t: *mut c_void, fmt: i32) -> i32 {
    fmt
}
unsafe extern "C" fn max_vertices(_t: *mut c_void) -> i32 {
    32768
}
unsafe extern "C" fn max_to_render(_t: *mut c_void, _mesh: *mut c_void, _until_flush: bool, v: *mut i32, i: *mut i32) {
    unsafe {
        if !v.is_null() {
            *v = 32768;
        }
        if !i.is_null() {
            *i = 32768;
        }
    }
}
unsafe extern "C" fn dynamic_vb_size(_t: *mut c_void) -> i32 {
    2 * 1024 * 1024
}
unsafe extern "C" fn dx_level_defaults(_t: *mut c_void, max: *mut u32, rec: *mut u32) {
    unsafe {
        *max = crate::device::DX_LEVEL as u32;
        *rec = crate::device::DX_LEVEL as u32;
    }
}
unsafe extern "C" fn yes(_t: *mut c_void) -> bool {
    true
}
unsafe extern "C" fn no(_t: *mut c_void) -> bool {
    false
}
unsafe extern "C" fn four(_t: *mut c_void) -> i32 {
    4
}
unsafe extern "C" fn ignore(_t: *mut c_void) {}

/// queued rendering builds dynamic meshes in the material system's own memory
unsafe extern "C" fn compute_vertex_description(_t: *mut c_void, buf: *mut u8, fmt: u64, desc: *mut MeshDesc) {
    if desc.is_null() {
        return;
    }
    let d = unsafe { &mut (*desc).vertex };
    compute_vertex_desc(buf, fmt, d);
    d.first_vertex = 0;
    d.offset = 0;
}
unsafe extern "C" fn get_scene_fog_color(_t: *mut c_void, rgb: *mut u8) {
    if !rgb.is_null() {
        let c = with(|s| s.fog.color);
        unsafe { std::ptr::copy_nonoverlapping(c.as_ptr(), rgb, 3) };
    }
}
unsafe extern "C" fn get_scene_fog_mode(_t: *mut c_void) -> i32 {
    with(|s| s.fog.mode)
}
/// PIXELFOGTYPE is the fog mode shifted down one
unsafe extern "C" fn get_pixel_fog_combo(_t: *mut c_void) -> i32 {
    with(|s| (s.fog.mode - 1).max(0))
}
unsafe extern "C" fn fog_start(_t: *mut c_void, v: f32) {
    with(|s| s.fog.start = v);
}
unsafe extern "C" fn fog_end(_t: *mut c_void, v: f32) {
    with(|s| s.fog.end = v);
}
unsafe extern "C" fn set_fog_z(_t: *mut c_void, v: f32) {
    with(|s| s.fog.z = v);
}
unsafe extern "C" fn fog_max_density(_t: *mut c_void, v: f32) {
    with(|s| s.fog.max_density = v);
}
unsafe extern "C" fn scene_fog_mode(_t: *mut c_void, mode: i32) {
    with(|s| s.fog.mode = mode.clamp(0, 2));
}
unsafe extern "C" fn scene_fog_color(_t: *mut c_void, r: u8, g: u8, b: u8) {
    with(|s| s.fog.color = [r, g, b]);
}
unsafe extern "C" fn set_standard_vs_constants(_t: *mut c_void, overbright: f32) {
    with(|s| s.overbright = if overbright > 0.0 { overbright } else { 2.0 });
}
unsafe extern "C" fn lightmap_scale(_t: *mut c_void) -> f32 {
    lightmap_scale_factor()
}
unsafe extern "C" fn set_ambient_light_cube(_t: *mut c_void, cube: *const f32) {
    if cube.is_null() {
        return;
    }
    let v = unsafe { std::slice::from_raw_parts(cube, 24) };
    with(|s| {
        for i in 0..6 {
            s.ambient_cube[i].copy_from_slice(&v[i * 4..i * 4 + 4]);
        }
    });
}
/// c21..c26
fn vs_ambient_cube() {
    with(|s| {
        let cube = s.ambient_cube;
        if s.vs_constants[21..27] != cube {
            s.vs_constants[21..27].copy_from_slice(&cube);
            s.vs_dirty = true;
        }
    });
}
unsafe extern "C" fn set_vs_ambient_cube(_t: *mut c_void) {
    vs_ambient_cube();
}
fn ps_ambient_cube(reg: i32, black: bool) {
    with(|s| {
        let cube = if black { [[0.0; 4]; 6] } else { s.ambient_cube };
        for (i, c) in cube.iter().enumerate() {
            if let Some(d) = s.ps_constants.get_mut(reg as usize + i) {
                if d != c {
                    *d = *c;
                    s.ps_dirty = true;
                }
            }
        }
    });
}
unsafe extern "C" fn set_ps_ambient_cube(_t: *mut c_void, reg: i32, black: bool) {
    ps_ambient_cube(reg, black);
}
/// for built-in shaders (SurfaceGGX): the lights in the vertex shader layout (color + directional
/// flag, direction + spot flag, position, spot params, attenuation; five registers each) as pixel
/// shader constants from `reg`. returns how many.
pub(crate) fn ps_lights_full(reg: i32) -> usize {
    let regs = with(|s| {
        let lights = sorted_lights(s);
        let mut out = [[0.0f32; 4]; 20];
        let mut n = 0;
        for (i, l) in lights.iter().enumerate().take(4) {
            let spot = l.kind == 0;
            out[5 * i] = [l.color[0], l.color[1], l.color[2], if l.kind == 2 { 1.0 } else { 0.0 }];
            out[5 * i + 1] = [l.dir[0], l.dir[1], l.dir[2], if spot { 1.0 } else { 0.0 }];
            out[5 * i + 2] = [l.pos[0], l.pos[1], l.pos[2], 1.0];
            out[5 * i + 3] = if spot {
                let (stop, stop2) = ((l.theta * 0.5).cos(), (l.phi * 0.5).cos());
                [l.falloff, stop, stop2, if stop > stop2 { 1.0 / (stop - stop2) } else { 0.0 }]
            } else {
                [0.0, 1.0, 1.0, 1.0]
            };
            out[5 * i + 4] = [l.atten[0], l.atten[1], l.atten[2], 0.0];
            n = i + 1;
        }
        (out, n)
    });
    for (i, v) in regs.0.iter().enumerate() {
        set_ps_reg(reg + i as i32, *v);
    }
    regs.1
}
pub(crate) fn builtin_ps_ambient_cube(reg: i32) {
    ps_ambient_cube(reg, false);
}
pub(crate) fn builtin_ps_fog_params(reg: i32) {
    ps_fog_params(reg);
}
pub(crate) fn builtin_eye_position() -> [f32; 3] {
    with(|s| camera_position(s.matrices[0].last().unwrap()))
}
pub(crate) fn builtin_set_ps_reg(reg: i32, v: [f32; 4]) {
    set_ps_reg(reg, v);
}
fn set_ps_reg(reg: i32, v: [f32; 4]) {
    with(|s| {
        if let Some(d) = s.ps_constants.get_mut(reg.max(0) as usize) {
            if *d != v {
                *d = v;
                s.ps_dirty = true;
            }
        }
    });
}
fn ps_fog_params(reg: i32) {
    let v = with(|s| pixel_fog_params(s));
    set_ps_reg(reg, v);
}
unsafe extern "C" fn set_ps_fog_params(_t: *mut c_void, reg: i32) {
    ps_fog_params(reg);
}
fn ps_eye_pos(reg: i32) {
    let e = with(|s| camera_position(s.matrices[0].last().unwrap()));
    set_ps_reg(reg, [e[0], e[1], e[2], 1.0]);
}
/// up to three lights in six registers, the fourth spread over their w; directional lights become
/// far points
fn ps_lighting(reg: i32) {
    let regs = with(|s| {
        let mut st = [[0.0f32; 4]; 6];
        let point = |l: &Light| if l.kind == 2 { std::array::from_fn(|k| s.lighting_origin[k] - l.dir[k] * 10000.0) } else { l.pos };
        for (i, l) in sorted_lights(s).iter().enumerate() {
            let p: [f32; 3] = point(l);
            if i < 3 {
                st[2 * i] = [l.color[0], l.color[1], l.color[2], 0.0];
                st[2 * i + 1] = [p[0], p[1], p[2], 0.0];
            } else {
                for k in 0..3 {
                    st[k][3] = l.color[k];
                    st[3 + k][3] = p[k];
                }
            }
        }
        st
    });
    for (i, v) in regs.iter().enumerate() {
        set_ps_reg(reg + i as i32, *v);
    }
}
unsafe extern "C" fn set_light(_t: *mut c_void, n: i32, desc: *const u8) {
    if desc.is_null() || !(0..4).contains(&n) {
        return;
    }
    let f = |o: usize| unsafe { std::ptr::read_unaligned(desc.add(o) as *const f32) };
    let v = |o: usize| [f(o), f(o + 4), f(o + 8)];
    let ty = unsafe { std::ptr::read_unaligned(desc as *const i32) };
    // LightType_t -> VertexShaderLightTypes_t
    let kind = match ty {
        1 => 1,
        2 => 2,
        3 => 0,
        _ => -1,
    };
    let light = (kind >= 0).then(|| {
        let phi = f(64).min(std::f32::consts::PI);
        let mut theta = f(60);
        if theta - phi > -1e-3 {
            theta = phi - 1e-3;
        }
        Light { kind, color: v(4), pos: v(16), dir: v(28), falloff: f(44), atten: [f(48), f(52), f(56)], theta, phi }
    });
    unsafe { std::ptr::copy_nonoverlapping(desc, LIGHT_DESCS.lock().unwrap_or_else(|e| e.into_inner())[n as usize].as_mut_ptr(), 88) };
    with(|s| s.lights[n as usize] = light);
}
unsafe extern "C" fn disable_all_local_lights(_t: *mut c_void) {
    with(|s| s.lights = [None; 4]);
    LIGHT_DESCS.lock().unwrap_or_else(|e| e.into_inner()).iter_mut().for_each(|l| *l = [0; 88]);
}
/// 88 bytes on x64, so GetLight can return a reference
static LIGHT_DESCS: Mutex<[[u8; 88]; 4]> = Mutex::new([[0; 88]; 4]);
/// MATERIAL_LIGHT_DISABLE (0) when unset
unsafe extern "C" fn get_light(_t: *mut c_void, n: i32) -> *const u8 {
    let g = LIGHT_DESCS.lock().unwrap_or_else(|e| e.into_inner());
    g[(n.clamp(0, 3)) as usize].as_ptr()
}
unsafe extern "C" fn ambient_cube_luminance(_t: *mut c_void) -> f32 {
    with(|s| s.ambient_cube.iter().map(|c| 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]).sum::<f32>() / 6.0)
}
/// srgb curves, as dx9 uses on pc
unsafe extern "C" fn gamma_to_linear_hw(_t: *mut c_void, x: f32) -> f32 {
    if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }
}
unsafe extern "C" fn linear_to_gamma_hw(_t: *mut c_void, x: f32) -> f32 {
    if x <= 0.0031308 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}
unsafe extern "C" fn set_ps_near_far(t: *mut c_void, reg: i32) {
    let m = with(|s| *s.matrices[1].last().unwrap());
    // m[2][2] = f/(n-f), m[3][2] = nf/(n-f)
    let n = m[14] / m[10];
    let f = (m[14] * n) / (n + m[14]);
    let v = [n, f, 0.0, 0.0];
    unsafe { set_ps_constant(t, reg, v.as_ptr(), 1, false) };
}
unsafe extern "C" fn set_lighting_origin(_t: *mut c_void, v: *const f32) {
    if !v.is_null() {
        let o = unsafe { [*v, *v.add(1), *v.add(2)] };
        with(|s| s.lighting_origin = o);
    }
}
unsafe extern "C" fn get_dx9_light_state(_t: *mut c_void, st: *mut u8) {
    if st.is_null() {
        return;
    }
    let (n, ambient) = with(|s| (s.lights.iter().flatten().count() as i32, s.ambient_cube.iter().any(|c| c[0] != 0.0 || c[1] != 0.0 || c[2] != 0.0)));
    unsafe {
        std::ptr::write_unaligned(st as *mut i32, n);
        *st.add(4) = ambient as u8;
        // sdk2013 / gmod order: ambient, static vertex, static texel. static vertex = the mesh has
        // a color mesh.
        *st.add(5) = with(|s| s.mesh != 0 && (*(s.mesh as *const Mesh)).r().color_mesh.is_some()) as u8;
        *st.add(6) = 0;
    }
}
unsafe extern "C" fn commit_ps_lighting(_t: *mut c_void, reg: i32) {
    ps_lighting(reg);
}
unsafe extern "C" fn set_depth_feathering(_t: *mut c_void, reg: i32, scale: f32) {
    set_ps_reg(reg, [192.0 / scale, 0.0, 0.0, 0.0]);
}
unsafe extern "C" fn set_bool_vs_constant(_t: *mut c_void, reg: i32, v: *const i32, n: i32, _force: bool) {
    set_bools(reg, v, n, true);
}
unsafe extern "C" fn set_bool_ps_constant(_t: *mut c_void, reg: i32, v: *const i32, n: i32, _force: bool) {
    set_bools(reg, v, n, false);
}
fn set_bools(reg: i32, v: *const i32, n: i32, vertex: bool) {
    if v.is_null() || reg < 0 || n <= 0 {
        return;
    }
    let src = unsafe { std::slice::from_raw_parts(v, n as usize) };
    with(|s| {
        if vertex {
            s.vs_std_key = None;
        }
        let before = if vertex { s.vs_bools } else { s.ps_bools };
        let bits = if vertex { &mut s.vs_bools } else { &mut s.ps_bools };
        for (i, &b) in src.iter().enumerate() {
            let r = reg as usize + i;
            if r < 32 {
                if b != 0 { *bits |= 1 << r } else { *bits &= !(1 << r) }
            }
        }
        if *bits != before {
            if vertex { s.vs_dirty = true } else { s.ps_dirty = true }
        }
    });
}
unsafe extern "C" fn set_int_vs_constant(_t: *mut c_void, reg: i32, v: *const i32, n: i32, _force: bool) {
    set_ints(reg, v, n, true);
}
unsafe extern "C" fn set_int_ps_constant(_t: *mut c_void, reg: i32, v: *const i32, n: i32, _force: bool) {
    set_ints(reg, v, n, false);
}
fn set_ints(reg: i32, v: *const i32, n: i32, vertex: bool) {
    if v.is_null() || reg < 0 || n <= 0 {
        return;
    }
    let src = unsafe { std::slice::from_raw_parts(v, n as usize * 4) };
    with(|s| {
        if vertex {
            s.vs_std_key = None;
        }
        let dst = if vertex { &mut s.vs_ints } else { &mut s.ps_ints };
        let mut changed = false;
        for (i, c) in src.chunks_exact(4).enumerate() {
            if let Some(d) = dst.get_mut(reg as usize + i) {
                if d[..] != *c {
                    d.copy_from_slice(c);
                    changed = true;
                }
            }
        }
        if changed {
            if vertex { s.vs_dirty = true } else { s.ps_dirty = true }
        }
    });
}
unsafe extern "C" fn get_world_space_camera_position(_t: *mut c_void, pos: *mut f32) {
    if pos.is_null() {
        return;
    }
    let p = with(|s| camera_position(s.matrices[0].last().unwrap()));
    unsafe { std::ptr::copy_nonoverlapping(p.as_ptr(), pos, 3) };
}
unsafe extern "C" fn get_fog_distances(_t: *mut c_void, start: *mut f32, end: *mut f32, z: *mut f32) {
    let f = with(|s| s.fog);
    for (p, v) in [(start, f.start), (end, f.end), (z, f.z)] {
        if !p.is_null() {
            unsafe { *p = v };
        }
    }
}
unsafe extern "C" fn load_bone_matrix(_t: *mut c_void, index: i32, m: *const f32) {
    if m.is_null() || !(0..53).contains(&index) {
        return;
    }
    let mut b = [0.0f32; 12];
    b.copy_from_slice(unsafe { std::slice::from_raw_parts(m, 12) });
    with(|s| {
        let i = index as usize;
        if s.bones.len() <= i {
            s.bones.resize(i + 1, [0.0; 12]);
        }
        s.bones[i] = b;
        s.bones_version += 1;
        if i == 0 {
            // bone 0 doubles as the model matrix
            let mut m = IDENTITY;
            for r in 0..3 {
                for c in 0..4 {
                    m[c * 4 + r] = b[r * 4 + c];
                }
            }
            *s.matrices[10].last_mut().unwrap() = m;
        }
    });
}
unsafe extern "C" fn get_current_num_bones(_t: *mut c_void) -> i32 {
    with(|s| s.num_bones)
}
unsafe extern "C" fn set_num_bone_weights(_t: *mut c_void, n: i32) {
    with(|s| s.num_bones = n);
}
unsafe extern "C" fn cull_mode(_t: *mut c_void, mode: i32) {
    with(|s| s.cull_mode = mode);
}

unsafe extern "C" fn enable_shader_mutex(_t: *mut c_void, on: bool) {
    log::line(&format!("EnableShaderShaderMutex({on})"));
    crate::lock::set_enabled(on);
}
unsafe extern "C" fn shader_lock(_t: *mut c_void) {
    crate::lock::shader_lock();
}
unsafe extern "C" fn shader_unlock(_t: *mut c_void) {
    crate::lock::shader_unlock();
}

/// dx9 shaders record per-draw state into these: int command, then operands (pointers are 8 bytes)
unsafe extern "C" fn execute_command_buffer(_t: *mut c_void, buf: *const u8) {
    unsafe { run_command_buffer(buf, 0) };
}

unsafe fn run_command_buffer(mut p: *const u8, depth: u32) {
    if p.is_null() || depth > 16 {
        return;
    }
    let int = |p: *const u8, i: usize| unsafe { std::ptr::read_unaligned((p as *const i32).add(i)) };
    let ptr = |p: *const u8| unsafe { std::ptr::read_unaligned(p.add(4) as *const *const u8) };
    for _ in 0..100_000 {
        let cmd = int(p, 0);
        match cmd {
            0 => return,
            1 => p = ptr(p),
            2 => {
                unsafe { run_command_buffer(ptr(p), depth + 1) };
                p = unsafe { p.add(4 + 8) };
            }
            256 | 257 => {
                // first reg, count, values
                let (first, n) = (int(p, 1), int(p, 2));
                let values = unsafe { p.add(12) } as *const f32;
                set_constants(first, values, n, cmd == 257);
                p = unsafe { p.add(12 + n.max(0) as usize * 16) };
            }
            258 => {
                // first reg, count, pointer to values
                let (first, n) = (int(p, 1), int(p, 2));
                let values = unsafe { std::ptr::read_unaligned(p.add(12) as *const *const f32) };
                set_constants(first, values, n, true);
                p = unsafe { p.add(12 + 8) };
            }
            259..=262 => {
                // reg
                let reg = int(p, 1);
                match cmd {
                    259 => ps_fog_params(reg),
                    260 => ps_eye_pos(reg),
                    261 => ps_lighting(reg),
                    _ => ps_ambient_cube(reg, false),
                }
                p = unsafe { p.add(8) };
            }
            263 => {
                vs_ambient_cube();
                p = unsafe { p.add(4) };
            }
            264 => {
                // reg, blend scale
                let scale = f32::from_bits(int(p, 2) as u32);
                unsafe { set_depth_feathering(std::ptr::null_mut(), int(p, 1), scale) };
                p = unsafe { p.add(12) };
            }
            512 => {
                // sampler, standard texture id
                let (sampler, id) = (int(p, 1), int(p, 2));
                bind_standard(sampler, id);
                p = unsafe { p.add(12) };
            }
            513 => {
                // sampler, 8-byte handle
                let (sampler, handle) = (int(p, 1), int(p, 2));
                unsafe { bind_texture(std::ptr::null_mut(), sampler, handle) };
                p = unsafe { p.add(16) };
            }
            1024 => {
                unsafe { set_ps_index(std::ptr::null_mut(), int(p, 1)) };
                p = unsafe { p.add(8) };
            }
            1025 => {
                unsafe { set_vs_index(std::ptr::null_mut(), int(p, 1)) };
                p = unsafe { p.add(8) };
            }
            2000 => p = unsafe { p.add(8) }, // vertex flashlight state
            2001 => p = unsafe { p.add(20) }, // pixel flashlight state: 3 regs + sampler
            2002 | 2003 => p = unsafe { p.add(4) },
            other => {
                static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    log::line(&format!("ExecuteCommandBuffer: unknown command {other}, stopping"));
                }
                return;
            }
        }
    }
}

fn bind_standard(sampler: i32, id: i32) {
    let handle = with(|s| s.standard_textures.get(id as usize).copied().unwrap_or(0));
    if handle != 0 {
        unsafe { bind_texture(std::ptr::null_mut(), sampler, handle) };
    } else {
        // the material system knows the rest (lightmap pages, ...)
        crate::shader_util::bind_standard_texture(sampler, id);
    }
}
unsafe extern "C" fn set_standard_texture_handle(_t: *mut c_void, id: i32, handle: i32) {
    with(|s| {
        if (0..256).contains(&id) {
            let i = id as usize;
            if s.standard_textures.len() <= i {
                s.standard_textures.resize(i + 1, 0);
            }
            s.standard_textures[i] = handle;
        }
    });
}
unsafe extern "C" fn bind_standard_texture(_t: *mut c_void, sampler: i32, id: i32) {
    bind_standard(sampler, id);
}

unsafe extern "C" fn set_mode(_t: *mut c_void, hwnd: usize, adapter: i32, info: *const ShaderDeviceInfo) -> bool {
    let info = unsafe { *info };
    log::line(&format!("IShaderAPI::SetMode(hwnd {hwnd:#x}, adapter {adapter}, {info:?})"));
    crate::device::set_mode(hwnd, &info);
    true
}
unsafe extern "C" fn change_video_mode(_t: *mut c_void, info: *const ShaderDeviceInfo) {
    let info = unsafe { *info };
    log::line(&format!("IShaderAPI::ChangeVideoMode({info:?})"));
    crate::device::change_video_mode(&info);
}

unsafe extern "C" fn texture_list(_t: *mut c_void) -> *mut c_void {
    crate::texlist::list()
}
unsafe extern "C" fn texture_list_fresh(_t: *mut c_void, frames: i32) -> bool {
    crate::texlist::fresh(frames)
}
/// returns the previous state
unsafe extern "C" fn set_debug_texture_rendering(_t: *mut c_void, on: bool) -> bool {
    crate::texlist::RENDERING.swap(on, std::sync::atomic::Ordering::Relaxed)
}
unsafe extern "C" fn enable_get_all_textures(_t: *mut c_void, on: bool) {
    crate::texlist::ALL.store(on, std::sync::atomic::Ordering::Relaxed);
}
/// memory of one texture with its mips, faces and slices
pub fn texture_bytes(t: &Texture) -> u64 {
    let base = image_bytes(t.format, t.width, t.height) as u64 * t.depth.max(1) as u64;
    let faces = if t.flags & 0x1 != 0 { 6 } else { 1 };
    let chain = if t.mips > 1 { base * 4 / 3 } else { base };
    chain * faces
}
/// bytes of all loaded textures (also used for bound-last-frame), or picmip estimates
unsafe extern "C" fn texture_memory_used(_t: *mut c_void, kind: i32) -> i32 {
    let total: u64 = {
        texture_table()
            .values()
            .map(texture_bytes)
            .sum()
    };
    let v = match kind {
        3 => total / 4,
        4 => total / 16,
        _ => total,
    };
    v.min(i32::MAX as u64) as i32
}
unsafe extern "C" fn enable_debug_texture_list(_t: *mut c_void, on: bool) {
    crate::texlist::ENABLED.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn debug_texture_info_interface() -> crate::Interface {
    use crate::vtables::debug_texture_info::{slot as s, stubs};
    let f = |p: *const ()| p as usize;
    crate::Interface::new(
        stubs(),
        &[
            (s::EnableDebugTextureList, f(enable_debug_texture_list as *const ())),
            (s::EnableGetAllTextures, f(enable_get_all_textures as *const ())),
            (s::GetDebugTextureList, f(texture_list as *const ())),
            (s::GetTextureMemoryUsed, f(texture_memory_used as *const ())),
            (s::IsDebugTextureListFresh, f(texture_list_fresh as *const ())),
            (s::SetDebugTextureRendering, f(set_debug_texture_rendering as *const ())),
        ],
    )
}

pub fn interface() -> crate::Interface {
    use crate::vtables::shader_api::{slot as s, stubs};
    let f = |p: *const ()| p as usize;
    crate::Interface::new(
        stubs(),
        &[
            (s::SetViewports, f(set_viewports as *const ())),
            (s::GetViewports, f(get_viewports as *const ())),
            (s::CurrentTime, f(current_time as *const ())),
            (s::GetLightmapDimensions, f(lightmap_dims as *const ())),
            (s::MatrixMode, f(matrix_mode as *const ())),
            (s::PushMatrix, f(push_matrix as *const ())),
            (s::PopMatrix, f(pop_matrix as *const ())),
            (s::LoadMatrix, f(load_matrix as *const ())),
            (s::GetMatrix, f(get_matrix as *const ())),
            (s::LoadIdentity, f(load_identity as *const ())),
            (s::SetVertexShaderConstant, f(set_vs_constant as *const ())),
            (s::SetPixelShaderConstant, f(set_ps_constant as *const ())),
            (s::SetVertexShaderIndex, f(set_vs_index as *const ())),
            (s::SetPixelShaderIndex, f(set_ps_index as *const ())),
            (s::GetBackBufferDimensions, f(back_buffer_dims as *const ())),
            (s::GetMaxLights, f(four as *const ())),
            (s::GetLight, f(get_light as *const ())),
            (s::CopyTextureToRenderTargetEx, f(copy_texture_to_rt_ex as *const ())),
            (s::GetAmbientLightCubeLuminance, f(ambient_cube_luminance as *const ())),
            (s::GammaToLinear_HardwareSpecific, f(gamma_to_linear_hw as *const ())),
            (s::LinearToGamma_HardwareSpecific, f(linear_to_gamma_hw as *const ())),
            (s::SetPSNearAndFarZ, f(set_ps_near_far as *const ())),
            (s::GMOD_SamplerBorderClamp, f(sampler_border_clamp as *const ())),
            (s::GetLightMapScaleFactor, f(lightmap_scale as *const ())),
            (s::SetToneMappingScaleLinear, f(set_tonemap_scale as *const ())),
            (s::GetToneMappingScaleLinear, f(get_tonemap_scale as *const ())),
            (s::GetSceneFogMode, f(get_scene_fog_mode as *const ())),
            (s::GetPixelFogCombo, f(get_pixel_fog_combo as *const ())),
            (s::FogStart, f(fog_start as *const ())),
            (s::FogEnd, f(fog_end as *const ())),
            (s::SetFogZ, f(set_fog_z as *const ())),
            (s::FogMaxDensity, f(fog_max_density as *const ())),
            (s::SceneFogMode, f(scene_fog_mode as *const ())),
            (s::SceneFogColor3ub, f(scene_fog_color as *const ())),
            (s::SetStandardVertexShaderConstants, f(set_standard_vs_constants as *const ())),
            (s::SetAmbientLightCube, f(set_ambient_light_cube as *const ())),
            (s::SetLight, f(set_light as *const ())),
            (s::DisableAllLocalLights, f(disable_all_local_lights as *const ())),
            (s::SetLightingOrigin, f(set_lighting_origin as *const ())),
            (s::GetDX9LightState, f(get_dx9_light_state as *const ())),
            (s::SetVertexShaderStateAmbientLightCube, f(set_vs_ambient_cube as *const ())),
            (s::SetPixelShaderStateAmbientLightCube, f(set_ps_ambient_cube as *const ())),
            (s::SetPixelShaderFogParams, f(set_ps_fog_params as *const ())),
            (s::CommitPixelShaderLighting, f(commit_ps_lighting as *const ())),
            (s::SetDepthFeatheringPixelShaderConstant, f(set_depth_feathering as *const ())),
            (s::SetBooleanVertexShaderConstant, f(set_bool_vs_constant as *const ())),
            (s::SetIntegerVertexShaderConstant, f(set_int_vs_constant as *const ())),
            (s::SetBooleanPixelShaderConstant, f(set_bool_ps_constant as *const ())),
            (s::SetIntegerPixelShaderConstant, f(set_int_ps_constant as *const ())),
            (s::GetDXLevelDefaults, f(dx_level_defaults as *const ())),
            (s::ClearColor3ub, f(clear_color3 as *const ())),
            (s::ClearColor4ub, f(clear_color4 as *const ())),
            (s::TakeSnapshot, f(take_snapshot as *const ())),
            (s::SetMode, f(set_mode as *const ())),
            (s::ExecuteCommandBuffer, f(execute_command_buffer as *const ())),
            (s::SetStandardTextureHandle, f(set_standard_texture_handle as *const ())),
            (s::BindStandardTexture, f(bind_standard_texture as *const ())),
            (s::EnableShaderShaderMutex, f(enable_shader_mutex as *const ())),
            (s::ShaderLock, f(shader_lock as *const ())),
            (s::ShaderUnlock, f(shader_unlock as *const ())),
            (s::ComputeVertexDescription, f(compute_vertex_description as *const ())),
            (s::GetSceneFogColor, f(get_scene_fog_color as *const ())),
            (s::GetWorldSpaceCameraPosition, f(get_world_space_camera_position as *const ())),
            (s::GetFogDistances, f(get_fog_distances as *const ())),
            (s::LoadBoneMatrix, f(load_bone_matrix as *const ())),
            (s::CullMode, f(cull_mode as *const ())),
            (s::GetCurrentNumBones, f(get_current_num_bones as *const ())),
            (s::SetNumBoneWeights, f(set_num_bone_weights as *const ())),
            (s::ChangeVideoMode, f(change_video_mode as *const ())),
            (s::Bind, f(bind as *const ())),
            (s::FlushBufferedPrimitives, f(ignore as *const ())),
            (s::GetDynamicMesh, f(get_dynamic_mesh as *const ())),
            (s::GetDynamicMeshEx, f(get_dynamic_mesh_ex as *const ())),
            (s::IsTranslucent, f(is_translucent as *const ())),
            (s::IsAlphaTested, f(is_alpha_tested as *const ())),
            (s::UsesVertexAndPixelShaders, f(uses_shaders as *const ())),
            (s::IsDepthWriteEnabled, f(is_depth_write_enabled as *const ())),
            (s::ComputeVertexFormat, f(compute_vertex_format as *const ())),
            (s::ComputeVertexUsage, f(compute_vertex_format as *const ())),
            (s::BeginPass, f(begin_pass as *const ())),
            (s::RenderPass, f(render_pass as *const ())),
            (s::GetNearestSupportedFormat, f(same_format as *const ())),
            (s::GetNearestRenderTargetFormat, f(same_format as *const ())),
            (s::DoRenderTargetsNeedSeparateDepthBuffer, f(no as *const ())),
            (s::CreateTexture, f(create_texture as *const ())),
            (s::CreateTextures, f(create_textures as *const ())),
            (s::DeleteTexture, f(delete_texture as *const ())),
            (s::CreateDepthTexture, f(create_depth_texture as *const ())),
            (s::IsTexture, f(is_texture as *const ())),
            (s::IsTextureResident, f(yes as *const ())),
            (s::ModifyTexture, f(modify_texture as *const ())),
            (s::TexImage2D, f(tex_image_2d as *const ())),
            (s::TexSubImage2D, f(tex_sub_image_2d as *const ())),
            (s::TexLock, f(tex_lock as *const ())),
            (s::TexUnlock, f(tex_unlock as *const ())),
            (s::TexImageFromVTF, f(tex_image_from_vtf as *const ())),
            (s::TexMinFilter, f(tex_min_filter as *const ())),
            (s::TexMagFilter, f(tex_mag_filter as *const ())),
            (s::TexWrap, f(tex_wrap as *const ())),
            (s::ClearBuffers, f(clear_buffers as *const ())),
            (s::SetStencilEnable, f(stencil_enable as *const ())),
            (s::SetDefaultState, f(dynamic_default_state as *const ())),
            (s::SetFlashlightState, f(set_flashlight_state as *const ())),
            (s::SetFlashlightStateEx, f(set_flashlight_state_ex as *const ())),
            (s::GetFlashlightState, f(get_flashlight_state as *const ())),
            (s::GetFlashlightStateEx, f(get_flashlight_state_ex as *const ())),
            (s::InFlashlightMode, f(in_flashlight_mode as *const ())),
            (s::GetCurrentColorCorrection, f(get_current_color_correction as *const ())),
            (s::ReadPixels, f(read_pixels_rect as *const ())),
            (s::ReadPixels_1, f(read_pixels as *const ())),
            (s::CreateOcclusionQueryObject, f(create_query as *const ())),
            (s::DestroyOcclusionQueryObject, f(destroy_query as *const ())),
            (s::BeginOcclusionQueryDrawing, f(begin_query as *const ())),
            (s::EndOcclusionQueryDrawing, f(end_query as *const ())),
            (s::OcclusionQuery_GetNumPixelsRendered, f(query_pixels as *const ())),
            (s::ClearBuffersObeyStencil, f(clear_obey_stencil as *const ())),
            (s::ClearBuffersObeyStencilEx, f(clear_obey_stencil_ex as *const ())),
            (s::PerformFullScreenStencilOperation, f(full_screen_stencil_operation as *const ())),
            (s::ForceDepthFuncEquals, f(force_depth_func_equals as *const ())),
            (s::OverrideDepthEnable, f(override_depth_enable as *const ())),
            (s::OverrideAlphaWriteEnable, f(override_alpha_write as *const ())),
            (s::OverrideColorWriteEnable, f(override_color_write as *const ())),
            (s::OverrideBlend, f(override_blend as *const ())),
            (s::OverrideBlendSeparateAlpha, f(override_blend_alpha as *const ())),
            (s::GMOD_ForceFilterMode, f(force_filter_mode as *const ())),
            (s::SetScissorRect, f(set_scissor_rect as *const ())),
            (s::SetIntRenderingParameter, f(set_int_param as *const ())),
            (s::SetFloatRenderingParameter, f(set_float_param as *const ())),
            (s::SetVectorRenderingParameter, f(set_vector_param as *const ())),
            (s::GetIntRenderingParameter, f(get_int_param as *const ())),
            (s::GetFloatRenderingParameter, f(get_float_param as *const ())),
            (s::GetVectorRenderingParameter, f(get_vector_param as *const ())),
            (s::ShouldWriteDepthToDestAlpha, f(should_write_depth_to_dest_alpha as *const ())),
            (s::SetAnisotropicLevel, f(set_anisotropic_level as *const ())),
            (s::SetClipPlane, f(set_clip_plane as *const ())),
            (s::EnableClipPlane, f(enable_clip_plane as *const ())),
            (s::EnableUserClipTransformOverride, f(enable_user_clip_transform_override as *const ())),
            (s::UserClipTransform, f(user_clip_transform as *const ())),
            (s::SetStencilFailOperation, f(stencil_fail as *const ())),
            (s::SetStencilZFailOperation, f(stencil_zfail as *const ())),
            (s::SetStencilPassOperation, f(stencil_pass as *const ())),
            (s::SetStencilCompareFunction, f(stencil_func as *const ())),
            (s::SetStencilReferenceValue, f(stencil_ref as *const ())),
            (s::SetStencilTestMask, f(stencil_test_mask as *const ())),
            (s::SetStencilWriteMask, f(stencil_write_mask as *const ())),
            (s::ClearStencilBufferRectangle, f(clear_stencil_rect as *const ())),
            (s::SetRenderTarget, f(set_render_target as *const ())),
            (s::SetRenderTargetEx, f(set_render_target_ex as *const ())),
            (s::CopyRenderTargetToTexture, f(copy_rt_to_texture as *const ())),
            (s::CopyRenderTargetToTextureEx, f(copy_rt_to_texture_ex as *const ())),
            (s::BeginFrame, f(ignore as *const ())),
            (s::EndFrame, f(ignore as *const ())),
            (s::BindTexture, f(bind_texture as *const ())),
            (s::CanDownloadTextures, f(yes as *const ())),
            (s::GetCurrentDynamicVBSize, f(dynamic_vb_size as *const ())),
            (s::GetMaxToRender, f(max_to_render as *const ())),
            (s::GetMaxVerticesToRender, f(max_vertices as *const ())),
            (s::GetMaxIndicesToRender, f(max_vertices as *const ())),
            (s::CompareSnapshots, f(compare_snapshots as *const ())),
            (s::GetFlexMesh, f(get_flex_mesh as *const ())),
            (s::OwnGPUResources, f(no as *const ())),
            (s::GetShadowDepthTextureFormat, f(shadow_depth_format as *const ())),
            (s::GetNullTextureFormat, f(null_texture_format as *const ())),
            (s::SupportsShadowDepthTextures, f(supports_shadow_depth_textures as *const ())),
            (s::SupportsMSAAMode, f(supports_msaa_mode as *const ())),
            (s::SupportsCSAAMode, f(supports_csaa_mode as *const ())),
            (s::SetShadowDepthBiasFactors, f(set_shadow_depth_bias as *const ())),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_buffers_bind_textures_and_set_constants() {
        let mut sub: Vec<u8> = Vec::new();
        for v in [513i32, 3, 77, 0, 0] {
            sub.extend(v.to_le_bytes());
        }
        let mut main: Vec<u8> = Vec::new();
        for v in [257i32, 10, 1] {
            main.extend(v.to_le_bytes());
        }
        for f in [1.0f32, 2.0, 3.0, 4.0] {
            main.extend(f.to_le_bytes());
        }
        main.extend(2i32.to_le_bytes());
        main.extend((sub.as_ptr() as u64).to_le_bytes());
        for v in [1024i32, 5, 0] {
            main.extend(v.to_le_bytes());
        }
        unsafe { execute_command_buffer(std::ptr::null_mut(), main.as_ptr()) };
        with(|s| {
            assert_eq!(s.vs_constants[10], [1.0, 2.0, 3.0, 4.0]);
            assert_eq!(s.bound_textures[3], 77);
            assert_eq!(s.ps_index, 5);
        });
    }
}
