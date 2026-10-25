//! reading the engine's convars through vstdlib's ICvar

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char};

/// x64 ConVar offsets: name, parent, float and int value (read from ConVar's setters in engine.dll)
const NAME: usize = 24;
const PARENT: usize = 56;
const FLOAT: usize = 84;
const INT: usize = 88;

/// a found ConVar (its parent, which holds the values)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConVar(usize);

fn icvar() -> usize {
    static ICVAR: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ICVAR.get_or_init(|| {
        #[cfg(windows)]
        unsafe {
            use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
            let module = GetModuleHandleA(c"vstdlib.dll".as_ptr() as *const u8);
            if module.is_null() {
                return 0;
            }
            let Some(create) = GetProcAddress(module, c"CreateInterface".as_ptr() as *const u8) else { return 0 };
            let create: unsafe extern "C" fn(*const c_char, *mut i32) -> usize = std::mem::transmute(create);
            create(c"VEngineCvar007".as_ptr(), std::ptr::null_mut())
        }
        #[cfg(not(windows))]
        0
    })
}

/// ICvar::FindVar (slot 16, checked against CCvar's vtable in vstdlib.dll); None when missing or
/// when the object doesn't look like a ConVar
pub fn find(name: &str) -> Option<ConVar> {
    let icvar = icvar();
    if icvar == 0 {
        return None;
    }
    let c = CString::new(name).ok()?;
    // SAFETY: icvar is vstdlib's CCvar; the offsets are checked against the name before use
    unsafe {
        let find: unsafe extern "C" fn(usize, *const c_char) -> usize = std::mem::transmute(*(*(icvar as *const *const usize)).add(16));
        let cv = find(icvar, c.as_ptr());
        if cv == 0 {
            return None;
        }
        let name_is = |p: usize| {
            let n = *((p + NAME) as *const *const c_char);
            !n.is_null() && CStr::from_ptr(n).to_bytes().eq_ignore_ascii_case(name.as_bytes())
        };
        let parent = *((cv + PARENT) as *const usize);
        (name_is(cv) && parent != 0 && name_is(parent)).then_some(ConVar(parent))
    }
}

impl ConVar {
    /// the value fields hold their bits xor'd with the low half of the convar's address
    fn key(self) -> u32 {
        self.0 as u32
    }

    pub fn float(self) -> f32 {
        f32::from_bits(unsafe { *((self.0 + FLOAT) as *const u32) } ^ self.key())
    }

    pub fn int(self) -> i32 {
        (unsafe { *((self.0 + INT) as *const u32) } ^ self.key()) as i32
    }
}

/// convars by name, looked up again (at most once a second) while missing, since lua creates
/// them later than we first ask
#[derive(Default)]
pub struct Cache {
    found: HashMap<String, ConVar>,
    missing: HashMap<String, std::time::Instant>,
}

impl Cache {
    pub fn get(&mut self, name: &str) -> Option<ConVar> {
        if let Some(c) = self.found.get(name) {
            return Some(*c);
        }
        if self.missing.get(name).is_some_and(|t| t.elapsed().as_secs() < 1) {
            return None;
        }
        match find(name) {
            Some(c) => {
                crate::log::line(&format!("convar {name} = {} / {}", c.int(), c.float()));
                self.missing.remove(name);
                self.found.insert(name.to_string(), c);
                Some(c)
            }
            None => {
                self.missing.insert(name.to_string(), std::time::Instant::now());
                None
            }
        }
    }
}
