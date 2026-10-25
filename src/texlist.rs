//! mat_texture_list: the per-frame texture list the engine's panel reads. it's a KeyValues tree
//! the engine copies with its own KeyValues code, so the nodes copy gmod's x64 layout and get
//! their key names from vstdlib's KeyValuesSystem.

use std::collections::{HashMap, VecDeque};
use std::ffi::{CString, c_char, c_void};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering::Relaxed};

use crate::api::Texture;

/// EnableDebugTextureList / EnableGetAllTextures / SetDebugTextureRendering
pub static ENABLED: AtomicBool = AtomicBool::new(false);
pub static ALL: AtomicBool = AtomicBool::new(false);
pub static RENDERING: AtomicBool = AtomicBool::new(false);

static FRAME: AtomicU64 = AtomicU64::new(0);
static EXPORTED: AtomicU64 = AtomicU64::new(u64::MAX);
static ROOT: AtomicPtr<Kv> = AtomicPtr::new(std::ptr::null_mut());
/// recent lists stay alive: the engine may still be copying the previous one
static LISTS: parking_lot::Mutex<VecDeque<Tree>> = parking_lot::Mutex::new(VecDeque::new());
/// texture -> (binds this frame, most binds in one frame)
static BINDS: parking_lot::Mutex<Option<HashMap<i32, (u32, u32)>>> = parking_lot::Mutex::new(None);

/// counted only while the list is on; the panel's own preview draws don't count
pub fn note_bind(id: i32) {
    if ENABLED.load(Relaxed) && !RENDERING.load(Relaxed) && id >= 0 {
        let mut b = BINDS.lock();
        let e = b.get_or_insert_with(Default::default).entry(id).or_default();
        e.0 += 1;
        e.1 = e.1.max(e.0);
    }
}

pub fn fresh(frames_allowed: i32) -> bool {
    let (now, at) = (FRAME.load(Relaxed), EXPORTED.load(Relaxed));
    at <= now && at + frames_allowed.max(0) as u64 >= now
}

pub fn list() -> *mut c_void {
    ROOT.load(Relaxed) as *mut c_void
}

/// KeyValues as gmod's x64 tier1 lays it out (64 bytes)
#[repr(C)]
struct Kv {
    /// case-insensitive symbol in the low 24 bits, low byte of the case-sensitive one on top
    name: u32,
    string: *mut c_char,
    wstring: *mut u16,
    value: u64,
    kind: u8,
    escapes: u8,
    /// upper 16 bits of the case-sensitive symbol
    name_hi: u16,
    _pad: u32,
    peer: *mut Kv,
    sub: *mut Kv,
    chain: *mut Kv,
}
const _: () = assert!(std::mem::size_of::<Kv>() == 64);

const TYPE_STRING: u8 = 1;
const TYPE_INT: u8 = 2;

/// a whole list: the nodes and the strings they point at
#[derive(Default)]
struct Tree {
    /// boxed: nodes point at each other, so they must not move as the vec grows
    #[allow(clippy::vec_box)]
    nodes: Vec<Box<Kv>>,
    strings: Vec<CString>,
}
// only raw pointers into its own boxes
unsafe impl Send for Tree {}

/// (case-insensitive, case-sensitive) symbols: KeyValuesSystem()->GetSymbolForStringCaseSensitive
fn symbols(name: &str) -> (i32, i32) {
    static SYS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let sys = *SYS.get_or_init(|| {
        #[cfg(windows)]
        unsafe {
            use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
            let m = GetModuleHandleA(c"vstdlib.dll".as_ptr() as *const u8);
            if m.is_null() {
                return 0;
            }
            match GetProcAddress(m, c"KeyValuesSystem".as_ptr() as *const u8) {
                Some(f) => {
                    let f: unsafe extern "C" fn() -> usize = std::mem::transmute(f);
                    f()
                }
                None => 0,
            }
        }
        #[cfg(not(windows))]
        0
    });
    if sys == 0 {
        return (-1, -1);
    }
    let c = CString::new(name).unwrap_or_default();
    let mut insensitive = -1i32;
    // SAFETY: slot 9 of IKeyValuesSystem, as the engine's KeyValues constructor calls it
    let sensitive = unsafe {
        let vt = *(sys as *const *const usize);
        let f: unsafe extern "C" fn(usize, *mut i32, *const c_char, bool) -> i32 = std::mem::transmute(*vt.add(9));
        f(sys, &mut insensitive, c.as_ptr(), true)
    };
    (insensitive, sensitive)
}

impl Tree {
    fn node(&mut self, name: &str) -> *mut Kv {
        let (ins, cs) = symbols(name);
        let mut kv = Box::new(Kv {
            name: (ins as u32 & 0xff_ffff) | ((cs as u32 & 0xff) << 24),
            string: std::ptr::null_mut(),
            wstring: std::ptr::null_mut(),
            value: 0,
            kind: 0,
            escapes: 0,
            name_hi: (cs as u32 >> 8) as u16,
            _pad: 0,
            peer: std::ptr::null_mut(),
            sub: std::ptr::null_mut(),
            chain: std::ptr::null_mut(),
        });
        let p: *mut Kv = &mut *kv;
        self.nodes.push(kv);
        p
    }

    /// appends `child` to `parent`'s sub keys; `last` tracks the tail
    fn link(parent: *mut Kv, last: &mut *mut Kv, child: *mut Kv) {
        // SAFETY: both point into boxes owned by this tree
        unsafe {
            if last.is_null() {
                (*parent).sub = child;
            } else {
                (**last).peer = child;
            }
        }
        *last = child;
    }

    fn string(&mut self, parent: *mut Kv, last: &mut *mut Kv, key: &str, v: &str) {
        let n = self.node(key);
        let s = CString::new(v.replace('\0', "")).unwrap_or_default();
        unsafe {
            (*n).kind = TYPE_STRING;
            (*n).string = s.as_ptr() as *mut c_char;
        }
        self.strings.push(s);
        Self::link(parent, last, n);
    }

    fn int(&mut self, parent: *mut Kv, last: &mut *mut Kv, key: &str, v: i64) {
        let n = self.node(key);
        unsafe {
            (*n).kind = TYPE_INT;
            (*n).value = v.clamp(i32::MIN as i64, i32::MAX as i64) as i32 as u32 as u64;
        }
        Self::link(parent, last, n);
    }
}

/// ImageLoader::GetName
fn format_name(fmt: i32) -> &'static str {
    const NAMES: [&str; 39] = [
        "RGBA8888", "ABGR8888", "RGB888", "BGR888", "RGB565", "I8", "IA88", "P8", "A8", "RGB888_BLUESCREEN",
        "BGR888_BLUESCREEN", "ARGB8888", "BGRA8888", "DXT1", "DXT3", "DXT5", "BGRX8888", "BGR565", "BGRX5551",
        "BGRA4444", "DXT1_ONEBITALPHA", "BGRA5551", "UV88", "UVWQ8888", "RGBA16161616F", "RGBA16161616",
        "IMAGE_FORMAT_UVLX8888", "IMAGE_FORMAT_R32F", "IMAGE_FORMAT_RGB323232F", "IMAGE_FORMAT_RGBA32323232F",
        "NV_DST16", "NV_DST24", "NV_INTZ", "NV_RAWZ", "ATI_DST16", "ATI_DST24", "NV_NULL", "ATI2N", "ATI1N",
    ];
    usize::try_from(fmt).ok().and_then(|i| NAMES.get(i)).copied().unwrap_or("UNKNOWN")
}

/// called once per frame; true when the panel wants a new list
pub fn begin_export(frame: u64) -> bool {
    FRAME.store(frame, Relaxed);
    ENABLED.load(Relaxed)
}

/// rebuild the list
pub fn export(frame: u64, textures: &HashMap<i32, Texture>, back_buffer: (i32, i32)) {
    let all = ALL.load(Relaxed);
    let mut binds = BINDS.lock();
    let binds = binds.get_or_insert_with(Default::default);
    let mut t = Tree::default();
    let root = t.node("TextureList");
    let mut last = std::ptr::null_mut();
    let mut n = 0;
    let mut entry = |t: &mut Tree, fill: &mut dyn FnMut(&mut Tree, *mut Kv, &mut *mut Kv)| {
        n += 1;
        let e = t.node(&n.to_string());
        Tree::link(root, &mut last, e);
        let mut l = std::ptr::null_mut();
        fill(t, e, &mut l);
    };
    let mut ids: Vec<&i32> = textures.keys().collect();
    ids.sort();
    for id in ids {
        let tex = &textures[id];
        let (frame_binds, max_binds) = binds.get(id).copied().unwrap_or((0, 0));
        if !all && frame_binds == 0 {
            continue;
        }
        entry(&mut t, &mut |t, e, l| {
            t.string(e, l, "Name", &tex.name);
            t.string(e, l, "TexGroup", &tex.group);
            t.int(e, l, "Size", crate::api::texture_bytes(tex) as i64);
            if tex.copies > 1 {
                t.int(e, l, "Count", tex.copies as i64);
            }
            t.string(e, l, "Format", format_name(tex.format));
            t.int(e, l, "Width", tex.width as i64);
            t.int(e, l, "Height", tex.height as i64);
            t.int(e, l, "BindsMax", max_binds as i64);
            t.int(e, l, "BindsFrame", frame_binds as i64);
        });
    }
    let (w, h) = back_buffer;
    for (name, bytes) in [("BACKBUFFER", 4), ("FRONTBUFFER", 4), ("DEPTHBUFFER", 4)] {
        entry(&mut t, &mut |t, e, l| {
            t.string(e, l, "Name", name);
            t.string(e, l, "TexGroup", "RenderTargets");
            t.int(e, l, "Size", bytes * w as i64 * h as i64);
            t.string(e, l, "Format", "32 bit buffer (hack)");
            t.int(e, l, "Width", w as i64);
            t.int(e, l, "Height", h as i64);
            t.int(e, l, "BindsMax", 1);
            t.int(e, l, "BindsFrame", 1);
        });
    }
    // next frame counts from zero; the per-frame maximum stays
    binds.retain(|id, b| {
        b.0 = 0;
        textures.contains_key(id)
    });
    ROOT.store(root, Relaxed);
    EXPORTED.store(frame, Relaxed);
    let mut lists = LISTS.lock();
    lists.push_back(t);
    while lists.len() > 3 {
        lists.pop_front();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    #[test]
    fn keyvalues_match_gmods_layout() {
        // offsets read from the engine's KeyValues::GetInt / FindKey / constructor
        assert_eq!(offset_of!(Kv, string), 0x08);
        assert_eq!(offset_of!(Kv, wstring), 0x10);
        assert_eq!(offset_of!(Kv, value), 0x18);
        assert_eq!(offset_of!(Kv, kind), 0x20);
        assert_eq!(offset_of!(Kv, name_hi), 0x22);
        assert_eq!(offset_of!(Kv, peer), 0x28);
        assert_eq!(offset_of!(Kv, sub), 0x30);
        assert_eq!(offset_of!(Kv, chain), 0x38);
    }

    #[test]
    fn list_has_one_entry_per_bound_texture_plus_buffers() {
        let mut textures = HashMap::new();
        textures.insert(7, Texture { name: "dev/dev_measuregeneric01".into(), group: "World textures".into(), width: 512, height: 512, format: 13, ..Default::default() });
        textures.insert(8, Texture { name: "unbound".into(), ..Default::default() });
        ENABLED.store(true, Relaxed);
        note_bind(7);
        note_bind(7);
        assert!(begin_export(1));
        export(1, &textures, (1280, 720));
        ENABLED.store(false, Relaxed);
        let root = list() as *const Kv;
        let mut entries = vec![];
        unsafe {
            let mut e = (*root).sub;
            while !e.is_null() {
                let mut vals = vec![];
                let mut v = (*e).sub;
                while !v.is_null() {
                    vals.push(match (*v).kind {
                        TYPE_STRING => std::ffi::CStr::from_ptr((*v).string).to_string_lossy().into_owned(),
                        _ => ((*v).value as u32 as i32).to_string(),
                    });
                    v = (*v).peer;
                }
                entries.push(vals);
                e = (*e).peer;
            }
        }
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0], ["dev/dev_measuregeneric01", "World textures", "131072", "DXT1", "512", "512", "2", "2"]);
        assert_eq!(entries[3][0], "DEPTHBUFFER");
        assert!(fresh(1));
    }
}
