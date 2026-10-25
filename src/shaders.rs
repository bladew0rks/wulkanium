//! compiled game shaders: `shaders/fxc/<name>.vcs`, loose files first, then the vpk

use std::collections::HashMap;
use std::sync::Arc;

use crate::log;
use crate::vcs::VcsFile;
use crate::vpk::Vpk;

pub type Bytecode = Arc<Vec<u32>>;

#[derive(Default)]
pub struct ShaderStore {
    vpk: Option<Option<Vpk>>,
    files: HashMap<Arc<str>, Option<VcsFile>>,
    /// (shader, static record) -> dynamic combo -> bytecode
    combos: HashMap<(Arc<str>, u32), Arc<HashMap<u32, Bytecode>>>,
}

impl ShaderStore {
    fn file(&mut self, name: &Arc<str>) -> Option<&VcsFile> {
        if !self.files.contains_key(name) {
            let rel = format!("shaders/fxc/{name}.vcs");
            let data = std::fs::read(format!("garrysmod/{rel}")).ok().or_else(|| {
                let vpk = self.vpk.get_or_insert_with(|| match Vpk::open(std::path::Path::new("garrysmod/garrysmod_dir.vpk"), "shaders/") {
                    Ok(v) => Some(v),
                    Err(e) => {
                        log::line(&format!("shaders: can't open the game VPK: {e}"));
                        None
                    }
                });
                vpk.as_ref()?.read(&rel)
            });
            let parsed = match data.map(VcsFile::parse) {
                Some(Ok(v)) => Some(v),
                Some(Err(e)) => {
                    log::line(&format!("shaders: {name}: {e}"));
                    None
                }
                None => {
                    log::line(&format!("shaders: {name} not found"));
                    None
                }
            };
            self.files.insert(name.clone(), parsed);
        }
        self.files.get(name)?.as_ref()
    }

    /// `static_index` as the shadow state passes it, `dynamic` as SetVertexShaderIndex does
    /// (negative = 0)
    pub fn get(&mut self, name: &Arc<str>, static_index: i32, dynamic: i32) -> Option<Bytecode> {
        let ndyn = self.file(name)?.header.dynamic_combos.max(1) as u32;
        let static_index = static_index.max(0) as u32;
        let key = (name.clone(), static_index / ndyn);
        if !self.combos.contains_key(&key) {
            let file = self.files.get(name)?.as_ref()?;
            let decoded = match file.static_combo(static_index) {
                Ok(list) => list.into_iter().map(|(d, code)| (d, Arc::new(code))).collect(),
                Err(e) => {
                    log::line(&format!("shaders: {name} static combo {}: {e}", static_index / ndyn));
                    HashMap::new()
                }
            };
            self.combos.insert(key.clone(), Arc::new(decoded));
        }
        self.combos[&key].get(&(dynamic.max(0) as u32)).cloned()
    }
}
