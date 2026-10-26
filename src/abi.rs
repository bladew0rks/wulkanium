//! material system structs (gmod x64 layouts) and dx9 vertex format rules

use std::ffi::c_void;

pub type VertexFormat = u64;

pub const VERTEX_POSITION: u64 = 0x0001;
pub const VERTEX_NORMAL: u64 = 0x0002;
pub const VERTEX_COLOR: u64 = 0x0004;
pub const VERTEX_SPECULAR: u64 = 0x0008;
pub const VERTEX_TANGENT_S: u64 = 0x0010;
pub const VERTEX_TANGENT_T: u64 = 0x0020;
pub const VERTEX_WRINKLE: u64 = 0x0040;
pub const VERTEX_BONE_INDEX: u64 = 0x0080;
pub const VERTEX_FORMAT_VERTEX_SHADER: u64 = 0x0100;
pub const VERTEX_FORMAT_USE_EXACT_FORMAT: u64 = 0x0200;
pub const VERTEX_FORMAT_COMPRESSED: u64 = 0x0400;
const VERTEX_LAST_BIT: u32 = 10;
const VERTEX_BONE_WEIGHT_BIT: u32 = VERTEX_LAST_BIT + 1;
const USER_DATA_SIZE_BIT: u32 = VERTEX_LAST_BIT + 4;
const TEX_COORD_SIZE_BIT: u32 = VERTEX_LAST_BIT + 7;
pub const VERTEX_BONE_WEIGHT_MASK: u64 = 0x7 << VERTEX_BONE_WEIGHT_BIT;
pub const MAX_TEXCOORDS: usize = 8;

pub const fn vertex_boneweight(n: u64) -> u64 {
    n << VERTEX_BONE_WEIGHT_BIT
}
pub const fn vertex_userdata_size(n: u64) -> u64 {
    n << USER_DATA_SIZE_BIT
}
pub const fn vertex_texcoord_size(index: usize, coords: u64) -> u64 {
    coords << (TEX_COORD_SIZE_BIT as usize + 3 * index)
}
pub fn vertex_flags(f: VertexFormat) -> u64 {
    f & ((1 << (VERTEX_LAST_BIT + 1)) - 1)
}
pub fn num_bone_weights(f: VertexFormat) -> usize {
    ((f >> VERTEX_BONE_WEIGHT_BIT) & 7) as usize
}
pub fn user_data_size(f: VertexFormat) -> usize {
    ((f >> USER_DATA_SIZE_BIT) & 7) as usize
}
pub fn texcoord_size(index: usize, f: VertexFormat) -> usize {
    ((f >> (TEX_COORD_SIZE_BIT as usize + 3 * index)) & 7) as usize
}

/// vertex format from flags, texcoord dims, bone weights and user data
pub fn compute_vertex_format(flags: u64, texcoord_count: usize, dims: Option<&[i32]>, bone_weights: usize, user_data: usize) -> VertexFormat {
    // VERTEX_FORMAT_COMPRESSED stays: a shader that takes compressed vertices says so, and the
    // engine only builds compressed meshes when we report support for them
    let mut fmt = flags & !VERTEX_FORMAT_USE_EXACT_FORMAT;
    if bone_weights > 0 {
        fmt |= vertex_boneweight(2);
    }
    fmt |= vertex_userdata_size(user_data.min(4) as u64);
    for i in 0..texcoord_count.min(MAX_TEXCOORDS) {
        let d = dims.map_or(2, |d| d.get(i).copied().unwrap_or(0).clamp(0, 4) as u64);
        fmt |= vertex_texcoord_size(i, d);
    }
    fmt
}

/// `VertexDesc_t`, 0xf0 bytes on x64
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VertexDesc {
    pub size_position: i32,
    pub size_bone_weight: i32,
    pub size_bone_matrix_index: i32,
    pub size_normal: i32,
    pub size_color: i32,
    pub size_specular: i32,
    pub size_texcoord: [i32; MAX_TEXCOORDS],
    pub size_tangent_s: i32,
    pub size_tangent_t: i32,
    pub size_wrinkle: i32,
    pub size_user_data: i32,
    pub actual_vertex_size: i32,
    pub compression_type: i32,
    pub num_bone_weights: i32,
    pub position: *mut f32,
    pub bone_weight: *mut f32,
    pub bone_matrix_index: *mut u8,
    pub normal: *mut f32,
    pub color: *mut u8,
    pub specular: *mut u8,
    pub texcoord: [*mut f32; MAX_TEXCOORDS],
    pub tangent_s: *mut f32,
    pub tangent_t: *mut f32,
    pub wrinkle: *mut f32,
    pub user_data: *mut f32,
    pub first_vertex: i32,
    pub offset: u32,
}

/// `IndexDesc_t`, 0x18 bytes
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IndexDesc {
    pub indices: *mut u16,
    pub offset: u32,
    pub first_index: u32,
    pub index_size: u8,
}

/// `MeshDesc_t`, 0x108 bytes
#[repr(C)]
pub struct MeshDesc {
    pub vertex: VertexDesc,
    pub index: IndexDesc,
}

const _: () = assert!(std::mem::size_of::<VertexDesc>() == 0xf0);
const _: () = assert!(std::mem::offset_of!(VertexDesc, position) == 0x58);
const _: () = assert!(std::mem::offset_of!(VertexDesc, user_data) == 0xe0);
const _: () = assert!(std::mem::offset_of!(VertexDesc, num_bone_weights) == 0x50);
const _: () = assert!(std::mem::size_of::<IndexDesc>() == 0x18);
const _: () = assert!(std::mem::size_of::<MeshDesc>() == 0x108);

/// scratch target for writes to absent elements (dx9 does the same)
#[repr(C, align(32))]
struct Scratch([u8; 1024]);
static mut SCRATCH: Scratch = Scratch([0; 1024]);
pub fn scratch() -> *mut u8 {
    std::ptr::addr_of_mut!(SCRATCH) as *mut u8
}

fn texcoord_bytes(n: usize) -> usize {
    4 * n
}

pub fn vertex_size(fmt: VertexFormat) -> usize {
    let mut desc: VertexDesc = unsafe { std::mem::zeroed() };
    compute_vertex_desc(std::ptr::null_mut(), fmt, &mut desc);
    desc.actual_vertex_size as usize
}

/// `ComputeVertexDesc`: every present element's size is the full stride (CMeshBuilder advances by
/// it)
pub fn compute_vertex_desc(buf: *mut u8, fmt: VertexFormat, desc: &mut VertexDesc) {
    let dummy = scratch() as *mut f32;
    let mut offset = 0usize;
    let mut present: Vec<*mut i32> = Vec::with_capacity(20);
    let at = |o: usize| if buf.is_null() { dummy as *mut u8 } else { unsafe { buf.add(o) } };
    // compressed formats (VERTEX_FORMAT_COMPRESSED): bone weights as short2, normal and tangent
    // packed together into one ubyte4 (source's "combined tangents" scheme), user data inside the
    // normal; everything else as uncompressed
    let compressed = fmt & VERTEX_FORMAT_COMPRESSED != 0;
    desc.compression_type = compressed as i32;
    if fmt & VERTEX_POSITION != 0 {
        desc.position = at(offset) as *mut f32;
        offset += 12;
        present.push(&mut desc.size_position);
        if fmt & VERTEX_WRINKLE != 0 {
            desc.wrinkle = at(offset) as *mut f32;
            offset += 4;
            present.push(&mut desc.size_wrinkle);
        } else {
            desc.wrinkle = dummy;
            desc.size_wrinkle = 0;
        }
    } else {
        desc.position = dummy;
        desc.size_position = 0;
        desc.wrinkle = dummy;
        desc.size_wrinkle = 0;
    }
    desc.num_bone_weights = num_bone_weights(fmt) as i32;
    if fmt & VERTEX_BONE_INDEX != 0 {
        if desc.num_bone_weights > 0 {
            desc.bone_weight = at(offset) as *mut f32;
            offset += if compressed { 4 } else { 8 };
            present.push(&mut desc.size_bone_weight);
        } else {
            desc.bone_weight = dummy;
            desc.size_bone_weight = 0;
        }
        desc.bone_matrix_index = at(offset);
        offset += 4;
        present.push(&mut desc.size_bone_matrix_index);
    } else {
        desc.bone_weight = dummy;
        desc.size_bone_weight = 0;
        desc.bone_matrix_index = dummy as *mut u8;
        desc.size_bone_matrix_index = 0;
    }
    if fmt & VERTEX_NORMAL != 0 {
        desc.normal = at(offset) as *mut f32;
        offset += if compressed { 4 } else { 12 };
        present.push(&mut desc.size_normal);
    } else {
        desc.normal = dummy;
        desc.size_normal = 0;
    }
    if fmt & VERTEX_COLOR != 0 {
        desc.color = at(offset);
        offset += 4;
        present.push(&mut desc.size_color);
    } else {
        desc.color = dummy as *mut u8;
        desc.size_color = 0;
    }
    if fmt & VERTEX_SPECULAR != 0 {
        desc.specular = at(offset);
        offset += 4;
        present.push(&mut desc.size_specular);
    } else {
        desc.specular = dummy as *mut u8;
        desc.size_specular = 0;
    }
    for i in 0..MAX_TEXCOORDS {
        let n = texcoord_size(i, fmt);
        if n != 0 {
            desc.texcoord[i] = at(offset) as *mut f32;
            offset += texcoord_bytes(n);
            present.push(&mut desc.size_texcoord[i]);
        } else {
            desc.texcoord[i] = dummy;
            desc.size_texcoord[i] = 0;
        }
    }
    if fmt & VERTEX_TANGENT_S != 0 {
        desc.tangent_s = at(offset) as *mut f32;
        offset += 12;
        present.push(&mut desc.size_tangent_s);
    } else {
        desc.tangent_s = dummy;
        desc.size_tangent_s = 0;
    }
    if fmt & VERTEX_TANGENT_T != 0 {
        desc.tangent_t = at(offset) as *mut f32;
        offset += 12;
        present.push(&mut desc.size_tangent_t);
    } else {
        desc.tangent_t = dummy;
        desc.size_tangent_t = 0;
    }
    let ud = if compressed { 0 } else { user_data_size(fmt) };
    if ud > 0 {
        desc.user_data = at(offset) as *mut f32;
        offset += 4 * ud;
        present.push(&mut desc.size_user_data);
    } else {
        desc.user_data = dummy;
        desc.size_user_data = 0;
    }
    if fmt & VERTEX_FORMAT_USE_EXACT_FORMAT == 0 && offset > 16 {
        offset = (offset + 15) & !15;
    }
    desc.actual_vertex_size = offset as i32;
    for p in present {
        unsafe { *p = offset as i32 };
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElementType {
    Float(u8),
    /// b g r a bytes
    Color,
    /// four bytes the shader reads as 0..255 (compressed normal + tangent)
    UByte4,
    /// two shorts the shader reads as -32768..32767 (compressed bone weights)
    Short2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Element {
    pub usage: u8,
    pub index: u8,
    pub offset: u32,
    pub ty: ElementType,
}

/// stream 0 declaration for a format
pub fn vertex_elements(fmt: VertexFormat) -> Vec<Element> {
    // lay out at a fake base address and read the offsets back
    const BASE: usize = 0x1000_0000;
    let mut d: VertexDesc = unsafe { std::mem::zeroed() };
    compute_vertex_desc(BASE as *mut u8, fmt, &mut d);
    let mut out = Vec::new();
    let mut push = |usage: u8, index: u8, p: usize, present: bool, ty: ElementType| {
        if present && p >= BASE {
            out.push(Element { usage, index, offset: (p - BASE) as u32, ty });
        }
    };
    push(0, 0, d.position as usize, fmt & VERTEX_POSITION != 0, ElementType::Float(3));
    let skin = fmt & VERTEX_BONE_INDEX != 0;
    let compressed = fmt & VERTEX_FORMAT_COMPRESSED != 0;
    push(1, 0, d.bone_weight as usize, skin && num_bone_weights(fmt) > 0, if compressed { ElementType::Short2 } else { ElementType::Float(2) });
    push(2, 0, d.bone_matrix_index as usize, skin, ElementType::Color);
    push(3, 0, d.normal as usize, fmt & VERTEX_NORMAL != 0, if compressed { ElementType::UByte4 } else { ElementType::Float(3) });
    push(10, 0, d.color as usize, fmt & VERTEX_COLOR != 0, ElementType::Color);
    push(10, 1, d.specular as usize, fmt & VERTEX_SPECULAR != 0, ElementType::Color);
    for i in 0..MAX_TEXCOORDS {
        let n = texcoord_size(i, fmt);
        push(5, i as u8, d.texcoord[i] as usize, n > 0, ElementType::Float(n as u8));
    }
    // tangent comes from user data when present (inside the normal when compressed)
    let ud = if compressed { 0 } else { user_data_size(fmt) };
    if ud > 0 {
        push(6, 0, d.user_data as usize, true, ElementType::Float(ud as u8));
    } else {
        push(6, 0, d.tangent_s as usize, fmt & VERTEX_TANGENT_S != 0, ElementType::Float(3));
    }
    push(7, 0, d.tangent_t as usize, fmt & VERTEX_TANGENT_T != 0, ElementType::Float(3));
    out
}

/// `MaterialAdapterInfo_t`, 0x220 bytes
#[repr(C)]
pub struct MaterialAdapterInfo {
    pub driver_name: [u8; 512],
    pub vendor_id: u32,
    pub device_id: u32,
    pub sub_sys_id: u32,
    pub revision: u32,
    pub dx_support_level: i32,
    pub max_dx_support_level: i32,
    pub driver_version_high: u32,
    pub driver_version_low: u32,
}
const _: () = assert!(std::mem::size_of::<MaterialAdapterInfo>() == 0x220);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ShaderDisplayMode {
    pub version: i32,
    pub width: i32,
    pub height: i32,
    pub format: i32,
    pub refresh_num: i32,
    pub refresh_den: i32,
}

/// `ShaderDeviceInfo_t`; flags bits: 0 windowed, 1 resizing, 2 stencil
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ShaderDeviceInfo {
    pub version: i32,
    pub display_mode: ShaderDisplayMode,
    pub back_buffer_count: i32,
    pub aa_samples: i32,
    pub aa_quality: i32,
    pub dx_level: i32,
    pub windowed_limit_w: i32,
    pub windowed_limit_h: i32,
    pub flags: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PrimList {
    pub first_index: i32,
    pub num_indices: i32,
}

pub const IMAGE_FORMAT_UNKNOWN: i32 = -1;
pub const IMAGE_FORMAT_BGRA8888: i32 = 12;

pub type Ptr = *mut c_void;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_layouts_follow_dx9() {
        // 36 bytes padded to 48
        let fmt = VERTEX_POSITION | VERTEX_NORMAL | VERTEX_COLOR | vertex_texcoord_size(0, 2);
        let mut buf = vec![0u8; 256];
        let mut d: VertexDesc = unsafe { std::mem::zeroed() };
        compute_vertex_desc(buf.as_mut_ptr(), fmt, &mut d);
        assert_eq!(d.actual_vertex_size, 48);
        assert_eq!(d.size_position, 48);
        assert_eq!(d.size_texcoord[0], 48);
        assert_eq!(d.size_texcoord[1], 0);
        let base = buf.as_mut_ptr() as usize;
        assert_eq!(d.normal as usize - base, 12);
        assert_eq!(d.color as usize - base, 24);
        assert_eq!(d.texcoord[0] as usize - base, 28);
        // 16 bytes or less: no padding
        assert_eq!(vertex_size(VERTEX_POSITION | VERTEX_COLOR), 16);
        let skinned = VERTEX_POSITION | VERTEX_BONE_INDEX | vertex_boneweight(2);
        assert_eq!(vertex_size(skinned | VERTEX_FORMAT_USE_EXACT_FORMAT), 24);
        assert_eq!(compute_vertex_format(VERTEX_POSITION, 2, Some(&[2, 3]), 0, 4), VERTEX_POSITION | vertex_texcoord_size(0, 2) | vertex_texcoord_size(1, 3) | vertex_userdata_size(4));
    }
}
