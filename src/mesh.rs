//! `IMesh` with its `IIndexBuffer` vtable at +8. like dx9, `IMesh::Draw` asks the material to draw,
//! and the shader's `RenderPass` calls are what get recorded.

use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::sync::OnceLock;

use crate::abi::*;
use crate::vtables::{mesh, mesh_index_buffer};

/// two vtable pointers: IMesh/IVertexBuffer, then IIndexBuffer
#[repr(C)]
pub struct Mesh {
    vt: *const usize,
    vt_index: *const usize,
    pub data: UnsafeCell<MeshData>,
}

unsafe impl Sync for Mesh {}
unsafe impl Send for Mesh {}

#[derive(Default)]
pub struct MeshData {
    pub format: VertexFormat,
    pub dynamic: bool,
    pub primitive: i32,
    pub stride: usize,
    pub vertices: Vec<u8>,
    pub vertex_count: usize,
    pub indices: Vec<u16>,
    pub index_count: usize,
    /// GetDynamicMesh overrides
    pub vertex_override: Option<*mut Mesh>,
    pub index_override: Option<*mut Mesh>,
    /// baked vertex lighting (stream 1) and its first vertex
    pub color_mesh: Option<(*mut Mesh, usize)>,
    /// flex deltas (stream 2) and this mesh's byte offset in them
    pub flex_mesh: Option<(*mut Mesh, usize)>,
    /// gpu caches key on the id; version bumps on every change
    pub id: u64,
    pub version: u64,
    /// (version, renderer generation) last uploaded
    pub uploaded: (u64, u64),
    /// vertices of the last lock_mesh
    pub locked_vertices: usize,
    /// locked and unlocked with nothing written: the engine keeps the pointer and fills it later
    /// (the static prop color mesh pool), so the contents can change without a lock
    pub written_outside_lock: bool,
    /// frame of the last upload
    pub refreshed: u64,
    /// when such a mesh was unlocked; it's refreshed for a while after (the engine fills the
    /// lighting while the map loads)
    pub unlocked_at: Option<std::time::Instant>,
}

static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl Drop for Mesh {
    fn drop(&mut self) {
        crate::api::mesh_destroyed(self.d().id);
    }
}

pub const MATERIAL_TRIANGLES: i32 = 2;

impl Mesh {
    pub fn new(format: VertexFormat, dynamic: bool) -> Box<Mesh> {
        let (vt, vt_index) = vtables();
        Box::new(Mesh {
            vt,
            vt_index,
            data: UnsafeCell::new(MeshData { format, dynamic, primitive: MATERIAL_TRIANGLES, stride: vertex_size(format), id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed), ..Default::default() }),
        })
    }

    /// # Safety
    ///
    /// `p` must point at a [`Mesh`] we created.
    pub unsafe fn from_ptr<'a>(p: *mut c_void) -> &'a Mesh {
        unsafe { &*(p as *const Mesh) }
    }

    /// secondary base at +8
    unsafe fn from_index_ptr<'a>(p: *mut c_void) -> &'a Mesh {
        unsafe { &*((p as *const u8).sub(8) as *const Mesh) }
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self as *const Mesh as *mut c_void
    }

    pub fn as_index_buffer(&self) -> *mut c_void {
        unsafe { (self as *const Mesh as *mut u8).add(8) as *mut c_void }
    }

    pub fn r(&self) -> &MeshData {
        unsafe { &*self.data.get() }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn d(&self) -> &mut MeshData {
        unsafe { &mut *self.data.get() }
    }

    pub fn set_format(&self, format: VertexFormat) {
        let d = self.d();
        if d.format != format {
            d.format = format;
            d.stride = vertex_size(format);
            d.vertices.clear();
            d.vertex_count = 0;
        }
    }

    fn lock_vertices(&self, first: usize, count: usize, desc: &mut VertexDesc) {
        let d = self.d();
        if count == 0 || d.stride == 0 {
            compute_vertex_desc(std::ptr::null_mut(), d.format, desc);
            set_vertex_sizes_zero(desc);
        } else {
            let need = (first + count) * d.stride;
            if d.vertices.len() < need {
                d.vertices.resize(need, 0);
            }
            compute_vertex_desc(unsafe { d.vertices.as_mut_ptr().add(first * d.stride) }, d.format, desc);
        }
        desc.first_vertex = 0;
        desc.offset = (first * d.stride) as u32;
    }

    fn lock_indices(&self, first: usize, count: usize, desc: &mut IndexDesc) {
        let d = self.d();
        let need = first + count.max(1);
        if d.indices.len() < need {
            d.indices.resize(need, 0);
        }
        desc.indices = if count == 0 { scratch() as *mut u16 } else { unsafe { d.indices.as_mut_ptr().add(first) } };
        desc.offset = (first * 2) as u32;
        desc.first_index = first as u32;
        desc.index_size = if count == 0 { 0 } else { 1 };
    }
}

/// zero-vertex lock: everything goes to scratch and nothing advances
fn set_vertex_sizes_zero(desc: &mut VertexDesc) {
    desc.size_position = 0;
    desc.size_bone_weight = 0;
    desc.size_bone_matrix_index = 0;
    desc.size_normal = 0;
    desc.size_color = 0;
    desc.size_specular = 0;
    desc.size_texcoord = [0; MAX_TEXCOORDS];
    desc.size_tangent_s = 0;
    desc.size_tangent_t = 0;
    desc.size_wrinkle = 0;
    desc.size_user_data = 0;
    desc.actual_vertex_size = 0;
}


unsafe extern "C" fn vertex_count(this: *mut c_void) -> i32 {
    unsafe { Mesh::from_ptr(this) }.d().vertex_count as i32
}
unsafe extern "C" fn get_vertex_format(this: *mut c_void) -> u64 {
    unsafe { Mesh::from_ptr(this) }.d().format
}
unsafe extern "C" fn is_dynamic(this: *mut c_void) -> bool {
    unsafe { Mesh::from_ptr(this) }.d().dynamic
}
unsafe extern "C" fn room_remaining(this: *mut c_void) -> i32 {
    let m = unsafe { Mesh::from_ptr(this) };
    if m.d().dynamic { 32768 } else { 0 }
}
unsafe extern "C" fn vb_lock(this: *mut c_void, count: i32, append: bool, desc: *mut VertexDesc) -> bool {
    let _shader = crate::lock::guard();
    let m = unsafe { Mesh::from_ptr(this) };
    let first = if append { m.d().vertex_count } else { 0 };
    m.lock_vertices(first, count.max(0) as usize, unsafe { &mut *desc });
    true
}
unsafe extern "C" fn vb_unlock(this: *mut c_void, count: i32, desc: *mut VertexDesc) {
    let _shader = crate::lock::guard();
    let m = unsafe { Mesh::from_ptr(this) };
    let d = m.d();
    let offset = unsafe { (*desc).offset } as usize;
    let first = if d.stride > 0 { offset / d.stride } else { 0 };
    d.vertex_count = first + count.max(0) as usize;
    d.version += 1;
}
unsafe extern "C" fn set_primitive_type(this: *mut c_void, t: i32) {
    unsafe { Mesh::from_ptr(this) }.d().primitive = t;
}
unsafe extern "C" fn draw_prims(this: *mut c_void, lists: *const PrimList, n: i32) {
    if !crate::shader_util::on_draw_mesh_prims(this, lists as *const c_void, n) {
        return;
    }
    let lists = if lists.is_null() || n <= 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(lists, n as usize) } };
    for l in lists {
        crate::api::draw_mesh(this as *mut Mesh, l.first_index, l.num_indices);
    }
}
unsafe extern "C" fn set_color_mesh(this: *mut c_void, color: *mut c_void, offset: i32) {
    let m = unsafe { Mesh::from_ptr(this) };
    m.d().color_mesh = (!color.is_null()).then(|| (color as *mut Mesh, offset.max(0) as usize));
}
unsafe extern "C" fn set_flex_mesh(this: *mut c_void, flex: *mut c_void, offset_bytes: i32) {
    let m = unsafe { Mesh::from_ptr(this) };
    m.d().flex_mesh = (!flex.is_null()).then(|| (flex as *mut Mesh, offset_bytes.max(0) as usize));
}
unsafe extern "C" fn disable_flex_mesh(this: *mut c_void) {
    unsafe { Mesh::from_ptr(this) }.d().flex_mesh = None;
}
unsafe extern "C" fn draw(this: *mut c_void, first: i32, count: i32) {
    // dx9 lets the material system push its matrices first, and maybe veto the draw
    if !crate::shader_util::on_draw_mesh(this, first, count) {
        return;
    }
    crate::api::draw_mesh(this as *mut Mesh, first, count);
}
unsafe extern "C" fn lock_mesh(this: *mut c_void, nv: i32, ni: i32, desc: *mut MeshDesc) {
    if unsafe { Mesh::from_ptr(this) }.d().dynamic {
        crate::shader_util::sync_matrices();
    }
    let _shader = crate::lock::guard();
    let m = unsafe { Mesh::from_ptr(this) };
    let desc = unsafe { &mut *desc };
    m.lock_vertices(0, nv.max(0) as usize, &mut desc.vertex);
    m.lock_indices(0, ni.max(0) as usize, &mut desc.index);
    m.d().locked_vertices = nv.max(0) as usize;
}
unsafe extern "C" fn unlock_mesh(this: *mut c_void, nv: i32, ni: i32, _desc: *mut MeshDesc) {
    let _shader = crate::lock::guard();
    let d = unsafe { Mesh::from_ptr(this) }.d();
    // a static buffer keeps the size it was locked with, whatever was written this time
    d.written_outside_lock = !d.dynamic && nv <= 0 && d.locked_vertices > 0;
    d.unlocked_at = d.written_outside_lock.then(std::time::Instant::now);
    d.vertex_count = if d.written_outside_lock { d.locked_vertices } else { nv.max(0) as usize };
    d.index_count = ni.max(0) as usize;
    d.version += 1;
}
unsafe extern "C" fn modify_begin(this: *mut c_void, first_v: i32, nv: i32, first_i: i32, ni: i32, desc: *mut MeshDesc) {
    unsafe { modify_begin_ex(this, false, first_v, nv, first_i, ni, desc) }
}
unsafe extern "C" fn modify_begin_ex(this: *mut c_void, _read_only: bool, first_v: i32, nv: i32, first_i: i32, ni: i32, desc: *mut MeshDesc) {
    let _shader = crate::lock::guard();
    let m = unsafe { Mesh::from_ptr(this) };
    let desc = unsafe { &mut *desc };
    m.lock_vertices(first_v.max(0) as usize, nv.max(0) as usize, &mut desc.vertex);
    m.lock_indices(first_i.max(0) as usize, ni.max(0) as usize, &mut desc.index);
}
unsafe extern "C" fn modify_end(this: *mut c_void, _desc: *mut MeshDesc) {
    unsafe { Mesh::from_ptr(this) }.d().version += 1;
}
unsafe extern "C" fn get_mesh(this: *mut c_void) -> *mut c_void {
    this
}
unsafe extern "C" fn memory_used(this: *mut c_void) -> u32 {
    let d = unsafe { Mesh::from_ptr(this) }.d();
    (d.vertices.len() + 2 * d.indices.len()) as u32
}
unsafe extern "C" fn destructor(this: *mut c_void, _flags: u32) -> *mut c_void {
    this
}


unsafe extern "C" fn ib_index_count(this: *mut c_void) -> i32 {
    unsafe { Mesh::from_index_ptr(this) }.d().index_count as i32
}
unsafe extern "C" fn ib_index_format(_this: *mut c_void) -> i32 {
    1 // MATERIAL_INDEX_FORMAT_16BIT
}
unsafe extern "C" fn ib_is_dynamic(this: *mut c_void) -> bool {
    unsafe { Mesh::from_index_ptr(this) }.d().dynamic
}
unsafe extern "C" fn ib_room_remaining(this: *mut c_void) -> i32 {
    if unsafe { Mesh::from_index_ptr(this) }.d().dynamic { 32768 } else { 0 }
}
unsafe extern "C" fn ib_lock(this: *mut c_void, count: i32, append: bool, desc: *mut IndexDesc) -> bool {
    let _shader = crate::lock::guard();
    let m = unsafe { Mesh::from_index_ptr(this) };
    let first = if append { m.d().index_count } else { 0 };
    m.lock_indices(first, count.max(0) as usize, unsafe { &mut *desc });
    true
}
unsafe extern "C" fn ib_unlock(this: *mut c_void, count: i32, desc: *mut IndexDesc) {
    let _shader = crate::lock::guard();
    let d = unsafe { Mesh::from_index_ptr(this) }.d();
    d.index_count = unsafe { (*desc).first_index } as usize + count.max(0) as usize;
    d.version += 1;
}
unsafe extern "C" fn ib_modify_begin(this: *mut c_void, _read_only: bool, first: i32, count: i32, desc: *mut IndexDesc) {
    let _shader = crate::lock::guard();
    unsafe { Mesh::from_index_ptr(this) }.lock_indices(first.max(0) as usize, count.max(0) as usize, unsafe { &mut *desc });
}
unsafe extern "C" fn ib_modify_end(this: *mut c_void, _desc: *mut IndexDesc) {
    unsafe { Mesh::from_index_ptr(this) }.d().version += 1;
}
unsafe extern "C" fn ib_destructor(this: *mut c_void, _flags: u32) -> *mut c_void {
    this
}

fn vtables() -> (*const usize, *const usize) {
    static VT: OnceLock<(usize, usize)> = OnceLock::new();
    let (a, b) = *VT.get_or_init(|| {
        use mesh::slot as s;
        let primary = crate::Interface::new(
            mesh::stubs(),
            &[
                (s::Destructor_IVertexBuffer, destructor as *const () as usize),
                (s::VertexCount, vertex_count as *const () as usize),
                (s::GetVertexFormat, get_vertex_format as *const () as usize),
                (s::IsDynamic, is_dynamic as *const () as usize),
                (s::GetRoomRemaining, room_remaining as *const () as usize),
                (s::Lock, vb_lock as *const () as usize),
                (s::Unlock, vb_unlock as *const () as usize),
                (s::SetPrimitiveType, set_primitive_type as *const () as usize),
                // msvc puts Draw(CPrimList*, int) first
                (s::Draw, draw_prims as *const () as usize),
                (s::Draw_1, draw as *const () as usize),
                (s::LockMesh, lock_mesh as *const () as usize),
                (s::UnlockMesh, unlock_mesh as *const () as usize),
                (s::ModifyBegin, modify_begin as *const () as usize),
                (s::ModifyBeginEx, modify_begin_ex as *const () as usize),
                (s::ModifyEnd, modify_end as *const () as usize),
                (s::ComputeMemoryUsed, memory_used as *const () as usize),
                (s::SetColorMesh, set_color_mesh as *const () as usize),
                (s::SetFlexMesh, set_flex_mesh as *const () as usize),
                (s::DisableFlexMesh, disable_flex_mesh as *const () as usize),
                (s::Unknown27, get_mesh as *const () as usize),
            ],
        );
        use mesh_index_buffer::slot as i;
        let index = crate::Interface::new(
            mesh_index_buffer::stubs(),
            &[
                (i::Destructor_IIndexBuffer, ib_destructor as *const () as usize),
                (i::IndexCount, ib_index_count as *const () as usize),
                (i::IndexFormat, ib_index_format as *const () as usize),
                (i::IsDynamic, ib_is_dynamic as *const () as usize),
                (i::GetRoomRemaining, ib_room_remaining as *const () as usize),
                (i::Lock, ib_lock as *const () as usize),
                (i::Unlock, ib_unlock as *const () as usize),
                (i::ModifyBegin, ib_modify_begin as *const () as usize),
                (i::ModifyEnd, ib_modify_end as *const () as usize),
            ],
        );
        (primary.vtable() as usize, index.vtable() as usize)
    });
    (a as *const usize, b as *const usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locking_hands_out_strided_vertex_memory() {
        let fmt = VERTEX_POSITION | VERTEX_COLOR | vertex_texcoord_size(0, 2);
        let m = Mesh::new(fmt, false);
        let mut desc: MeshDesc = unsafe { std::mem::zeroed() };
        unsafe { lock_mesh(m.as_ptr(), 3, 6, &mut desc) };
        let stride = desc.vertex.actual_vertex_size as usize;
        assert_eq!(stride, 32);
        unsafe {
            let p = (desc.vertex.position as *mut u8).add(2 * desc.vertex.size_position as usize) as *mut f32;
            *p = 7.0;
            *desc.index.indices.add(5) = 2;
            unlock_mesh(m.as_ptr(), 3, 6, &mut desc);
        }
        let d = m.d();
        assert_eq!((d.vertex_count, d.index_count), (3, 6));
        assert_eq!(f32::from_le_bytes(d.vertices[2 * stride..2 * stride + 4].try_into().unwrap()), 7.0);
        assert_eq!(d.indices[5], 2);
        assert_eq!(unsafe { ib_index_count(m.as_index_buffer()) }, 6);
    }
}
