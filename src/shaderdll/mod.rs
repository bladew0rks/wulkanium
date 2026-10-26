//! a shader dll inside the shaderapi: materialsystem loads the dll the hardware config names
//! (GetHWSpecificShaderDLLName), asks it for "ShaderDLL004" and registers its shaders. it loads
//! after stdshader_dx9 and is searched first, so a shader here named like a stock one replaces it.
//! such a replacement is "stock-backed": it reports the stock shader's own param list (so indices
//! match) with ours appended, and hands every material it doesn't support to the stock shader.
//! vtable layouts were read from gmod's stdshader_dx9.dll (CShaderDLL, CBaseShader).

pub mod script;
mod surface_ggx;

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString, c_char, c_void};

use crate::Interface;
use crate::log;

/// ShaderParamType_t
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(i32)]
pub enum ParamType {
    Texture = 0,
    Integer = 1,
    Color = 2,
    Vec2 = 3,
    Vec3 = 4,
    Vec4 = 5,
    Float = 7,
    Bool = 8,
    Matrix = 10,
    Material = 11,
}

/// SHADER_PARAM_NOT_EDITABLE
const NOT_EDITABLE: i32 = 1;

pub struct Param {
    pub name: &'static str,
    pub ty: ParamType,
    pub default: &'static str,
    pub help: &'static str,
    pub flags: i32,
}

const fn p(name: &'static str, ty: ParamType, default: &'static str, help: &'static str) -> Param {
    Param { name, ty, default, help, flags: 0 }
}

use ParamType::*;

/// the engine's standard params come first, in this order (materialsystem reads them by index)
pub const STANDARD: [Param; 13] = [
    Param { name: "$flags", ty: Integer, default: "0", help: "flags", flags: NOT_EDITABLE },
    Param { name: "$flags_defined", ty: Integer, default: "0", help: "flags_defined", flags: NOT_EDITABLE },
    Param { name: "$flags2", ty: Integer, default: "0", help: "flags2", flags: NOT_EDITABLE },
    Param { name: "$flags_defined2", ty: Integer, default: "0", help: "flags2_defined", flags: NOT_EDITABLE },
    p("$color", Color, "[1 1 1]", "color"),
    p("$alpha", Float, "1.0", "alpha"),
    p("$basetexture", Texture, "shadertest/BaseTexture", "base texture (albedo)"),
    p("$frame", Integer, "0", "animation frame"),
    p("$basetexturetransform", Matrix, "center .5 .5 scale 1 1 rotate 0 translate 0 0", "base texture texcoord transform"),
    Param { name: "$flashlighttexture", ty: Texture, default: "effects/flashlight001", help: "flashlight spotlight shape texture", flags: NOT_EDITABLE },
    Param { name: "$flashlighttextureframe", ty: Integer, default: "0", help: "animation frame for $flashlight", flags: NOT_EDITABLE },
    p("$color2", Color, "[1 1 1]", "color2"),
    p("$srgbtint", Color, "[1 1 1]", "tint applied on srgb parts"),
];

/// a shader as the host serves it
pub struct Def {
    pub names: &'static [&'static CStr],
    /// stock shader whose params come first and which takes the materials `supports` turns down
    pub stock: Option<&'static str>,
    /// our params: after STANDARD for our own shaders, after the stock list for replacements
    pub params: &'static [Param],
    /// checked once per material, before its params are initialised
    pub supports: fn(Params) -> bool,
    pub init_params: unsafe fn(Params),
    pub init_instance: unsafe fn(Params, usize, *const c_char),
    pub draw: unsafe fn(Params, usize, usize),
    pub translucent: fn(Params) -> bool,
}

/// built-in defs first, then script defs as files appear; never removed (materials keep pointers
/// to their shaders), a changed script gets a new def
static DEFS: parking_lot::RwLock<Vec<&'static Def>> = parking_lot::RwLock::new(Vec::new());

fn defs() -> parking_lot::RwLockReadGuard<'static, Vec<&'static Def>> {
    {
        let d = DEFS.read();
        if !d.is_empty() {
            return d;
        }
    }
    let mut d = DEFS.write();
    if d.is_empty() {
        d.push(&surface_ggx::DEF);
    }
    drop(d);
    DEFS.read()
}

fn def(i: usize) -> &'static Def {
    defs()[i]
}

/// add a def and its shaders; returns its index. listed decides which shaders materialsystem
/// sees next time it reads the list (see `reregister`)
pub fn add_def(d: &'static Def) -> usize {
    let i = {
        let _ = defs();
        let mut all = DEFS.write();
        all.push(d);
        all.len() - 1
    };
    let mut sh = SHADERS.write();
    // a newer def under the same name takes over the listing
    for &name in d.names {
        sh.listed.retain(|s| s.name != name);
    }
    for &name in d.names {
        let s: &'static Shader = Box::leak(Box::new(Shader { iface: shader_vtable(), name, def: i }));
        sh.all.push(s);
        sh.listed.push(s);
    }
    i
}

/// IShaderSystem ("ShaderSystem002"), set on Connect
static SHADER_SYSTEM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn shader_system() -> usize {
    SHADER_SYSTEM.load(std::sync::atomic::Ordering::Relaxed)
}

/// virtual call `slot` on `obj` (x64: `this` first)
unsafe fn vfn<F: Copy>(obj: usize, slot: usize) -> F {
    unsafe {
        let vt = *(obj as *const *const usize);
        std::mem::transmute_copy(&*vt.add(slot))
    }
}

// ---- param tables ----

/// a param as materialsystem sees it
struct Meta {
    name: CString,
    help: CString,
    default: CString,
    ty: i32,
    flags: i32,
}

pub struct Table {
    metas: Vec<Meta>,
    /// lowercase name -> index
    index: HashMap<String, usize>,
    /// the stock IShader of a replacement, 0 when there is none
    stock: usize,
}

fn meta(p: &Param) -> Meta {
    let c = |s: &str| CString::new(s).unwrap_or_default();
    Meta { name: c(p.name), help: c(p.help), default: c(p.default), ty: p.ty as i32, flags: p.flags }
}

fn build_table(def: &Def) -> Table {
    let mut metas = Vec::new();
    let stock = def.stock.and_then(crate::stock::find).unwrap_or(0);
    if stock != 0 {
        unsafe {
            let s = |i: i32, slot: usize| {
                let p = vfn::<unsafe extern "C" fn(usize, i32) -> *const c_char>(stock, slot)(stock, i);
                if p.is_null() { CString::default() } else { CStr::from_ptr(p).to_owned() }
            };
            for i in 0..vfn::<unsafe extern "C" fn(usize) -> i32>(stock, 2)(stock) {
                metas.push(Meta {
                    name: s(i, 6),
                    help: s(i, 7),
                    default: s(i, 9),
                    ty: vfn::<unsafe extern "C" fn(usize, i32) -> i32>(stock, 8)(stock, i),
                    flags: vfn::<unsafe extern "C" fn(usize, i32) -> i32>(stock, 14)(stock, i),
                });
            }
        }
    } else {
        if let Some(name) = def.stock {
            log::line(&format!("shader dll: stock {name} not found, serving it alone"));
        }
        metas.extend(STANDARD.iter().map(meta));
    }
    let known: HashSet<String> = metas.iter().map(|m| m.name.to_string_lossy().to_lowercase()).collect();
    metas.extend(def.params.iter().filter(|p| !known.contains(&p.name.to_lowercase())).map(meta));
    let index = metas.iter().enumerate().map(|(i, m)| (m.name.to_string_lossy().to_lowercase(), i)).collect();
    Table { metas, index, stock }
}

fn table(d: usize) -> &'static Table {
    static TABLES: parking_lot::RwLock<Vec<Option<&'static Table>>> = parking_lot::RwLock::new(Vec::new());
    if let Some(Some(t)) = TABLES.read().get(d) {
        return t;
    }
    let mut all = TABLES.write();
    if all.len() <= d {
        all.resize(d + 1, None);
    }
    *all[d].get_or_insert_with(|| Box::leak(Box::new(build_table(def(d)))))
}

// ---- IShaderDLLInternal (6 slots) ----

type Factory = unsafe extern "C" fn(*const c_char, *mut i32) -> usize;

unsafe extern "C" fn dll_connect(_t: *mut c_void, factory: Factory, _is_material_system: bool) -> bool {
    let sys = unsafe { factory(c"ShaderSystem002".as_ptr(), std::ptr::null_mut()) };
    SHADER_SYSTEM.store(sys, std::sync::atomic::Ordering::Relaxed);
    log::line(&format!("shader dll: connected (ShaderSystem002 {})", if sys != 0 { "found" } else { "missing" }));
    static DUMPED: std::sync::Once = std::sync::Once::new();
    DUMPED.call_once(|| crate::stock::dump("wulkanium_stock_shaders.txt"));
    true
}
unsafe extern "C" fn dll_disconnect(_t: *mut c_void, _is_material_system: bool) {}
unsafe extern "C" fn dll_connect_plain(t: *mut c_void, factory: Factory) -> bool {
    unsafe { dll_connect(t, factory, false) }
}
unsafe extern "C" fn dll_disconnect_plain(_t: *mut c_void) {}
unsafe extern "C" fn dll_shader_count(_t: *mut c_void) -> i32 {
    shaders().listed.len() as i32
}
unsafe extern "C" fn dll_get_shader(_t: *mut c_void, i: i32) -> *mut c_void {
    usize::try_from(i).ok().and_then(|i| shaders().listed.get(i).copied()).map_or(std::ptr::null_mut(), |s| s.iface.as_ptr())
}

pub fn dll_interface() -> Interface {
    let f = |p: *const ()| p as usize;
    Interface::new(
        [0usize; 6],
        &[
            (0, f(dll_connect as *const ())),
            (1, f(dll_disconnect as *const ())),
            (2, f(dll_shader_count as *const ())),
            (3, f(dll_get_shader as *const ())),
            (4, f(dll_connect_plain as *const ())),
            (5, f(dll_disconnect_plain as *const ())),
        ],
    )
}

// ---- IShader (16 slots, plus CBaseShader's 4 that materialsystem never calls) ----

/// one registered name of a def
struct Shader {
    iface: Interface,
    name: &'static CStr,
    def: usize,
}

#[derive(Default)]
struct Shaders {
    /// every shader ever made (materials may hold any of them)
    all: Vec<&'static Shader>,
    /// what materialsystem gets: the newest shader per name
    listed: Vec<&'static Shader>,
}

static SHADERS: parking_lot::RwLock<Shaders> = parking_lot::RwLock::new(Shaders { all: Vec::new(), listed: Vec::new() });

fn shaders() -> parking_lot::RwLockReadGuard<'static, Shaders> {
    {
        let s = SHADERS.read();
        if !s.all.is_empty() {
            return s;
        }
    }
    let builtin: Vec<&'static Def> = defs().clone();
    let mut s = SHADERS.write();
    if s.all.is_empty() {
        for (d, def) in builtin.iter().enumerate() {
            for &name in def.names {
                let sh: &'static Shader = Box::leak(Box::new(Shader { iface: shader_vtable(), name, def: d }));
                s.all.push(sh);
                s.listed.push(sh);
            }
        }
    }
    drop(s);
    SHADERS.read()
}

fn this(t: *mut c_void) -> &'static Shader {
    shaders().all.iter().copied().find(|s| s.iface.as_ptr() == t).expect("unknown IShader")
}

/// materials (by param array) a replacement handed to the stock shader
fn stock_materials() -> parking_lot::MutexGuard<'static, Option<HashSet<usize>>> {
    static SET: parking_lot::Mutex<Option<HashSet<usize>>> = parking_lot::Mutex::new(None);
    SET.lock()
}

/// materials (by param array) each def serves itself
fn own_materials() -> parking_lot::MutexGuard<'static, Option<HashMap<usize, usize>>> {
    static MAP: parking_lot::Mutex<Option<HashMap<usize, usize>>> = parking_lot::Mutex::new(None);
    MAP.lock()
}

/// (shader, materials we draw, materials handed to the stock shader) for the overlay
pub fn material_counts() -> Vec<(String, usize, usize)> {
    let own = own_materials();
    let stock = stock_materials();
    defs()
        .iter()
        .enumerate()
        .map(|(d, def)| {
            let ours = own.as_ref().map_or(0, |m| m.values().filter(|&&x| x == d).count());
            // stock materials aren't tracked per def; only replacements have any
            let theirs = if def.stock.is_some() { stock.as_ref().map_or(0, |s| s.len()) } else { 0 };
            (def.names[0].to_string_lossy().into_owned(), ours, theirs)
        })
        .collect()
}

/// the stock IShader that owns this material, if any
fn stock_for(t: *mut c_void, params: *mut *mut c_void) -> Option<usize> {
    let stock = table(this(t).def).stock;
    (stock != 0 && stock_materials().as_ref().is_some_and(|s| s.contains(&(params as usize)))).then_some(stock)
}

fn params(t: *mut c_void, ptr: *mut *mut c_void) -> Params {
    let d = this(t).def;
    Params { ptr, table: table(d), def: d }
}

/// wk_shader_<name> 0 sends every material of a replacement to the stock shader (at material load)
fn replacement_enabled(name: &CStr) -> bool {
    static CV: parking_lot::Mutex<Option<crate::cvar::Cache>> = parking_lot::Mutex::new(None);
    let cv = format!("wk_shader_{}", name.to_string_lossy().to_lowercase());
    CV.lock().get_or_insert_with(Default::default).get(&cv).is_none_or(|c| c.int() != 0)
}

unsafe extern "C" fn get_name(t: *mut c_void) -> *const c_char {
    this(t).name.as_ptr()
}
unsafe extern "C" fn fallback(t: *mut c_void, params: *mut *mut c_void) -> *const c_char {
    let stock = table(this(t).def).stock;
    if stock != 0 {
        return unsafe { vfn::<unsafe extern "C" fn(usize, *mut *mut c_void) -> *const c_char>(stock, 1)(stock, params) };
    }
    std::ptr::null()
}
unsafe extern "C" fn num_params(t: *mut c_void) -> i32 {
    table(this(t).def).metas.len() as i32
}
unsafe extern "C" fn init_params(t: *mut c_void, ptr: *mut *mut c_void, material: *const c_char) {
    let sh = this(t);
    let def = def(sh.def);
    let stock = table(sh.def).stock;
    let p = params(t, ptr);
    let ours = stock == 0 || (replacement_enabled(sh.name) && (def.supports)(p));
    {
        let mut set = stock_materials();
        let set = set.get_or_insert_with(Default::default);
        if ours { set.remove(&(ptr as usize)) } else { set.insert(ptr as usize) };
        let mut own = own_materials();
        let own = own.get_or_insert_with(Default::default);
        if ours { own.insert(ptr as usize, sh.def) } else { own.remove(&(ptr as usize)) };
    }
    if ours {
        unsafe { (def.init_params)(p) }
    } else {
        unsafe { vfn::<unsafe extern "C" fn(usize, *mut *mut c_void, *const c_char)>(stock, 3)(stock, ptr, material) }
    }
}
unsafe extern "C" fn init_instance(t: *mut c_void, ptr: *mut *mut c_void, init: *mut c_void, material: *const c_char, group: *const c_char) {
    match stock_for(t, ptr) {
        Some(s) => unsafe { vfn::<unsafe extern "C" fn(usize, *mut *mut c_void, *mut c_void, *const c_char, *const c_char)>(s, 4)(s, ptr, init, material, group) },
        None => unsafe { (def(this(t).def).init_instance)(params(t, ptr), init as usize, group) },
    }
}
unsafe extern "C" fn draw_elements(t: *mut c_void, ptr: *mut *mut c_void, modulation: i32, shadow: *mut c_void, dynamic: *mut c_void, compression: i32, context: *mut *mut c_void) {
    match stock_for(t, ptr) {
        Some(s) => unsafe {
            vfn::<unsafe extern "C" fn(usize, *mut *mut c_void, i32, *mut c_void, *mut c_void, i32, *mut *mut c_void)>(s, 5)(s, ptr, modulation, shadow, dynamic, compression, context)
        },
        None => unsafe { (def(this(t).def).draw)(params(t, ptr), shadow as usize, dynamic as usize) },
    }
}
fn meta_at(t: *mut c_void, i: i32) -> Option<&'static Meta> {
    table(this(t).def).metas.get(usize::try_from(i).ok()?)
}
unsafe extern "C" fn param_name(t: *mut c_void, i: i32) -> *const c_char {
    meta_at(t, i).map_or(std::ptr::null(), |m| m.name.as_ptr())
}
unsafe extern "C" fn param_help(t: *mut c_void, i: i32) -> *const c_char {
    meta_at(t, i).map_or(std::ptr::null(), |m| m.help.as_ptr())
}
unsafe extern "C" fn param_type(t: *mut c_void, i: i32) -> i32 {
    meta_at(t, i).map_or(Integer as i32, |m| m.ty)
}
unsafe extern "C" fn param_default(t: *mut c_void, i: i32) -> *const c_char {
    meta_at(t, i).map_or(std::ptr::null(), |m| m.default.as_ptr())
}
unsafe extern "C" fn param_flags(t: *mut c_void, i: i32) -> i32 {
    meta_at(t, i).map_or(0, |m| m.flags)
}
unsafe extern "C" fn modulation_flags(t: *mut c_void, ptr: *mut *mut c_void, api: *mut c_void) -> i32 {
    match stock_for(t, ptr) {
        Some(s) => unsafe { vfn::<unsafe extern "C" fn(usize, *mut *mut c_void, *mut c_void) -> i32>(s, 10)(s, ptr, api) },
        None => 0,
    }
}
unsafe extern "C" fn needs_pow2_fb(t: *mut c_void, ptr: *mut *mut c_void, check: bool) -> bool {
    match stock_for(t, ptr) {
        Some(s) => unsafe { vfn::<unsafe extern "C" fn(usize, *mut *mut c_void, bool) -> bool>(s, 11)(s, ptr, check) },
        None => false,
    }
}
unsafe extern "C" fn needs_full_fb(t: *mut c_void, ptr: *mut *mut c_void, check: bool) -> bool {
    match stock_for(t, ptr) {
        Some(s) => unsafe { vfn::<unsafe extern "C" fn(usize, *mut *mut c_void, bool) -> bool>(s, 12)(s, ptr, check) },
        None => false,
    }
}
unsafe extern "C" fn is_translucent(t: *mut c_void, ptr: *mut *mut c_void) -> bool {
    match stock_for(t, ptr) {
        Some(s) => unsafe { vfn::<unsafe extern "C" fn(usize, *mut *mut c_void) -> bool>(s, 13)(s, ptr) },
        None => (def(this(t).def).translucent)(params(t, ptr)),
    }
}
unsafe extern "C" fn get_flags(t: *mut c_void) -> i32 {
    let stock = table(this(t).def).stock;
    if stock != 0 { unsafe { vfn::<unsafe extern "C" fn(usize) -> i32>(stock, 15)(stock) } } else { 0 }
}
unsafe extern "C" fn nothing(_t: *mut c_void) {}

fn shader_vtable() -> Interface {
    let f = |p: *const ()| p as usize;
    let nothing = f(nothing as *const ());
    Interface::new(
        [nothing; 20],
        &[
            (0, f(get_name as *const ())),
            (1, f(fallback as *const ())),
            (2, f(num_params as *const ())),
            (3, f(init_params as *const ())),
            (4, f(init_instance as *const ())),
            (5, f(draw_elements as *const ())),
            (6, f(param_name as *const ())),
            (7, f(param_help as *const ())),
            (8, f(param_type as *const ())),
            (9, f(param_default as *const ())),
            (10, f(modulation_flags as *const ())),
            (11, f(needs_pow2_fb as *const ())),
            (12, f(needs_full_fb as *const ())),
            (13, f(is_translucent as *const ())),
            (14, f(param_flags as *const ())),
            (15, f(get_flags as *const ())),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn own_shader_params_follow_the_standard_ones_without_duplicates() {
        let t = build_table(&surface_ggx::DEF);
        assert_eq!(t.metas[0].name.to_str().unwrap(), "$flags");
        assert_eq!(t.metas[12].name.to_str().unwrap(), "$srgbtint");
        assert_eq!(t.metas[13].name.to_str().unwrap(), surface_ggx::PARAMS[0].name);
        assert_eq!(t.metas.len(), STANDARD.len() + surface_ggx::PARAMS.len());
        assert_eq!(t.index.len(), t.metas.len());
        assert!(t.metas.iter().all(|m| m.name.to_str().unwrap().starts_with('$')));
    }
}

// ---- calling into the engine ----

/// a material's param array (IMaterialVar**), indexed through its shader's table
#[derive(Clone, Copy)]
pub struct Params {
    pub ptr: *mut *mut c_void,
    table: &'static Table,
    /// the def serving this material
    pub def: usize,
}

/// gmod's IMaterialVar: GetTextureValue slot 0, SetIntValue 4, IsDefined 17, GetMatrixValue 20;
/// int value at +0x10, float / vector at +0x14 (read from stdshader_dx9's accessors)
#[derive(Clone, Copy)]
pub struct Var(usize);

impl Params {
    pub fn var(self, name: &str) -> Option<Var> {
        let i = *self.table.index.get(name)?;
        let v = unsafe { *self.ptr.add(i) } as usize;
        (v != 0).then_some(Var(v))
    }

    /// defined and non-zero
    pub fn flag(self, name: &str) -> bool {
        self.var(name).is_some_and(|v| v.defined() && v.int() != 0)
    }

    pub fn defined(self, name: &str) -> bool {
        self.var(name).is_some_and(|v| v.defined())
    }

    pub fn float_or(self, name: &str, default: f32) -> f32 {
        self.var(name).filter(|v| v.defined()).map_or(default, |v| v.float())
    }

    pub fn vec3_or(self, name: &str, default: [f32; 3]) -> [f32; 3] {
        self.var(name).filter(|v| v.defined()).map_or(default, |v| {
            let x = v.vec();
            [x[0], x[1], x[2]]
        })
    }
}

impl Var {
    pub fn defined(self) -> bool {
        unsafe { vfn::<unsafe extern "C" fn(usize) -> bool>(self.0, 17)(self.0) }
    }
    pub fn int(self) -> i32 {
        unsafe { *((self.0 + 0x10) as *const i32) }
    }
    pub fn float(self) -> f32 {
        unsafe { *((self.0 + 0x14) as *const f32) }
    }
    pub fn vec(self) -> [f32; 4] {
        unsafe { *((self.0 + 0x14) as *const [f32; 4]) }
    }
    pub fn texture(self) -> usize {
        unsafe { vfn::<unsafe extern "C" fn(usize) -> usize>(self.0, 0)(self.0) }
    }
    /// slot 3 SetFloatValue
    pub fn set_float(self, v: f32) {
        unsafe { vfn::<unsafe extern "C" fn(usize, f32)>(self.0, 3)(self.0, v) }
    }
    /// slot 10 SetVecValue(x, y, z): msvc lists the overloads in reverse, unlike the sdk header
    pub fn set_vec3(self, x: f32, y: f32, z: f32) {
        unsafe { vfn::<unsafe extern "C" fn(usize, f32, f32, f32)>(self.0, 10)(self.0, x, y, z) }
    }
    pub fn set_int(self, v: i32) {
        unsafe { vfn::<unsafe extern "C" fn(usize, i32)>(self.0, 4)(self.0, v) }
    }
    /// VMatrix, row major
    pub fn matrix(self) -> [f32; 16] {
        unsafe { *vfn::<unsafe extern "C" fn(usize) -> *const [f32; 16]>(self.0, 20)(self.0) }
    }
}


// ---- registering again at runtime ----

/// materialsystem reads a shader dll's list once, when it loads the dll, and keeps the names in a
/// lookup. its shader system's LoadShaderDLL (internal interface slot 21) replaces an entry
/// loaded under the same name and reads the list again; calling it for ourselves makes new and
/// changed script shaders visible. the internal interface sits 8 bytes before ShaderSystem002
/// (CShaderSystem's second base), checked through its rtti before use.
pub fn reregister() -> Result<(), String> {
    let sys = shader_system();
    if sys == 0 {
        return Err("no ShaderSystem002".into());
    }
    let internal = sys - 8;
    unsafe {
        #[cfg(windows)]
        let base = windows_sys::Win32::System::LibraryLoader::GetModuleHandleA(c"materialsystem.dll".as_ptr() as *const u8) as usize;
        #[cfg(not(windows))]
        let base = 0usize;
        if base == 0 {
            return Err("materialsystem.dll not loaded".into());
        }
        let vt = *(internal as *const usize);
        // rtti: complete object locator before the vtable, its type descriptor's name at +16
        let col = *((vt - 8) as *const usize);
        let td = base + *((col + 12) as *const u32) as usize;
        let name = CStr::from_ptr((td + 16) as *const c_char).to_string_lossy();
        if name != ".?AVCShaderSystem@@" || *((col + 4) as *const u32) != 0 {
            return Err(format!("unexpected shader system layout ({name})"));
        }
        // m_ShaderDLLs: elements at +0x10, count at +0x20; 72 bytes each, name first, dll at +0x10,
        // mod flag at +0x1c
        let data = *((internal + 0x10) as *const usize);
        let count = *((internal + 0x20) as *const i32);
        let mut entry = None;
        for i in 0..count.max(0) as usize {
            let e = data + i * 72;
            let n = *(e as *const *const c_char);
            if n.is_null() {
                continue;
            }
            let n = CStr::from_ptr(n).to_owned();
            if n.to_string_lossy().to_lowercase().contains("shaderapiwgpu") {
                entry = Some((n, *((e + 0x1c) as *const bool)));
            }
        }
        let Some((name, is_mod)) = entry else { return Err("our shader dll entry not found".into()) };
        let load = vfn::<unsafe extern "C" fn(usize, *const c_char, *const c_char, bool) -> bool>(internal, 21);
        // a failed load leaves the old entry alone, so try the likely path ids in turn
        for path_id in [None, Some(c"EXECUTABLE_PATH"), Some(c"GAMEBIN")] {
            if load(internal, name.as_ptr(), path_id.map_or(std::ptr::null(), |p| p.as_ptr()), is_mod) {
                log::line(&format!("shader dll: registered again as {} ({})", name.to_string_lossy(), path_id.map_or("no path id".into(), |p| p.to_string_lossy())));
                return Ok(());
            }
        }
        Err(format!("LoadShaderDLL({}) failed", name.to_string_lossy()))
    }
}

/// IShaderSystemInternal::FindShader (slot 9): what a material naming `name` would get now
pub fn find_shader(name: &CStr) -> usize {
    let sys = shader_system();
    if sys == 0 {
        return 0;
    }
    unsafe { vfn::<unsafe extern "C" fn(usize, *const c_char) -> usize>(sys - 8, 9)(sys - 8, name.as_ptr()) }
}

/// our current IShader for a name, to compare with `find_shader`
pub fn listed_shader(name: &CStr) -> usize {
    shaders().listed.iter().find(|s| s.name == name).map_or(0, |s| s.iface.as_ptr() as usize)
}
