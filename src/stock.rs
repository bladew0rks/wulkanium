//! gmod's own shader dlls (stdshader_dx9, stdshader_dbg): reached through their ShaderDLL004
//! interface, so replacements can hand materials back to the stock shader of the same name

use std::ffi::{CStr, c_char};

#[cfg(windows)]
type CreateInterface = unsafe extern "C" fn(*const c_char, *mut i32) -> usize;

const DLLS: [&CStr; 2] = [c"stdshader_dx9.dll", c"stdshader_dbg.dll"];

unsafe fn vfn<F: Copy>(obj: usize, slot: usize) -> F {
    unsafe {
        let vt = *(obj as *const *const usize);
        std::mem::transmute_copy(&*vt.add(slot))
    }
}

/// IShaderDLLInternal of a loaded shader dll
#[cfg(windows)]
fn shader_dll(module: &CStr) -> Option<usize> {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    unsafe {
        let m = GetModuleHandleA(module.as_ptr() as *const u8);
        if m.is_null() {
            return None;
        }
        let f: CreateInterface = std::mem::transmute(GetProcAddress(m, c"CreateInterface".as_ptr() as *const u8)?);
        let dll = f(c"ShaderDLL004".as_ptr(), std::ptr::null_mut());
        (dll != 0).then_some(dll)
    }
}

#[cfg(not(windows))]
fn shader_dll(_module: &CStr) -> Option<usize> {
    None
}

/// every IShader of a shader dll (IShaderDLLInternal: 2 ShaderCount, 3 GetShader)
fn shaders_of(dll: usize) -> Vec<usize> {
    unsafe {
        let n = vfn::<unsafe extern "C" fn(usize) -> i32>(dll, 2)(dll);
        (0..n).map(|i| vfn::<unsafe extern "C" fn(usize, i32) -> usize>(dll, 3)(dll, i)).filter(|&s| s != 0).collect()
    }
}

unsafe fn str_at(p: *const c_char) -> String {
    if p.is_null() { String::new() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() }
}

/// IShader: 0 GetName
pub fn shader_name(s: usize) -> String {
    unsafe { str_at(vfn::<unsafe extern "C" fn(usize) -> *const c_char>(s, 0)(s)) }
}

/// the stock IShader called `name` (case-insensitive), stdshader_dx9 first
pub fn find(name: &str) -> Option<usize> {
    DLLS.iter().filter_map(|m| shader_dll(m)).flat_map(shaders_of).find(|&s| shader_name(s).eq_ignore_ascii_case(name))
}

/// shaders whose param queries crash
const SKIP: [&str; 1] = ["screenspace_general_8tex"];

/// writes every stock shader's params (name, type, default, flags, help) to a text file
pub fn dump(path: &str) {
    use std::fmt::Write;
    let mut out = String::new();
    for m in DLLS {
        let Some(dll) = shader_dll(m) else {
            let _ = writeln!(out, "# {} not loaded", m.to_string_lossy());
            continue;
        };
        for s in shaders_of(dll) {
            let name = shader_name(s);
            // reports more params than it describes
            if SKIP.contains(&name.as_str()) {
                continue;
            }
            unsafe {
                let n = vfn::<unsafe extern "C" fn(usize) -> i32>(s, 2)(s);
                let _ = writeln!(out, "shader {} ({}) {} params", shader_name(s), m.to_string_lossy(), n);
                for i in 0..n {
                    let name = str_at(vfn::<unsafe extern "C" fn(usize, i32) -> *const c_char>(s, 6)(s, i));
                    let help = str_at(vfn::<unsafe extern "C" fn(usize, i32) -> *const c_char>(s, 7)(s, i));
                    let ty = vfn::<unsafe extern "C" fn(usize, i32) -> i32>(s, 8)(s, i);
                    let def = str_at(vfn::<unsafe extern "C" fn(usize, i32) -> *const c_char>(s, 9)(s, i));
                    let flags = vfn::<unsafe extern "C" fn(usize, i32) -> i32>(s, 14)(s, i);
                    let _ = writeln!(out, "  {i:3} {name} type {ty} default \"{def}\" flags {flags} | {help}");
                }
            }
            // written as it goes, so a crash in a later shader keeps the rest
            let _ = std::fs::write(path, &out);
        }
    }
    let _ = std::fs::write(path, out);
}

