//! read-only vpk v1/v2, enough to pull shaders out of garrysmod_dir.vpk

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

struct Entry {
    preload: Vec<u8>,
    archive: u16,
    offset: u32,
    length: u32,
}

pub struct Vpk {
    dir: PathBuf,
    /// data embedded in the dir file (archive 0x7fff)
    data_start: u64,
    entries: HashMap<String, Entry>,
}

impl Vpk {
    /// prefix is lowercase and '/' separated
    pub fn open(dir: &Path, prefix: &str) -> Result<Vpk, String> {
        let b = std::fs::read(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let u32_at = |o: usize| b.get(o..o + 4).map(|x| u32::from_le_bytes(x.try_into().unwrap())).ok_or("truncated vpk");
        if u32_at(0)? != 0x55aa_1234 {
            return Err("not a vpk".into());
        }
        let header = match u32_at(4)? {
            1 => 12,
            2 => 28,
            v => return Err(format!("vpk version {v}")),
        };
        let tree = u32_at(8)? as usize;
        let mut p = header;
        let cstr = |p: &mut usize| -> Result<String, String> {
            let end = b[*p..].iter().position(|&c| c == 0).ok_or("unterminated vpk string")?;
            let s = String::from_utf8_lossy(&b[*p..*p + end]).into_owned();
            *p += end + 1;
            Ok(s)
        };
        let mut entries = HashMap::new();
        loop {
            let ext = cstr(&mut p)?;
            if ext.is_empty() {
                break;
            }
            loop {
                let path = cstr(&mut p)?;
                if path.is_empty() {
                    break;
                }
                loop {
                    let name = cstr(&mut p)?;
                    if name.is_empty() {
                        break;
                    }
                    let preload_len = b.get(p + 4..p + 6).map(|x| u16::from_le_bytes([x[0], x[1]])).ok_or("truncated vpk")? as usize;
                    let archive = u16::from_le_bytes([b[p + 6], b[p + 7]]);
                    let offset = u32_at(p + 8)?;
                    let length = u32_at(p + 12)?;
                    p += 18;
                    let full = if path.trim() == "" { format!("{name}.{ext}") } else { format!("{path}/{name}.{ext}") }.to_ascii_lowercase();
                    if full.starts_with(prefix) {
                        let preload = b.get(p..p + preload_len).ok_or("truncated vpk")?.to_vec();
                        entries.insert(full, Entry { preload, archive, offset, length });
                    }
                    p += preload_len;
                }
            }
        }
        Ok(Vpk { dir: dir.to_path_buf(), data_start: (header + tree) as u64, entries })
    }

    pub fn read(&self, path: &str) -> Option<Vec<u8>> {
        let e = self.entries.get(&path.to_ascii_lowercase())?;
        let mut out = e.preload.clone();
        if e.length > 0 {
            let (file, base) = if e.archive == 0x7fff {
                (self.dir.clone(), self.data_start)
            } else {
                let name = self.dir.file_name()?.to_string_lossy().replace("_dir.vpk", &format!("_{:03}.vpk", e.archive));
                (self.dir.with_file_name(name), 0)
            };
            let mut f = std::fs::File::open(file).ok()?;
            f.seek(SeekFrom::Start(base + e.offset as u64)).ok()?;
            let mut data = vec![0; e.length as usize];
            f.read_exact(&mut data).ok()?;
            out.extend(data);
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    /// needs GMOD_DIR
    #[test]
    fn reads_gmod_vpk() {
        let Ok(root) = std::env::var("GMOD_DIR") else { return };
        let vpk = super::Vpk::open(&std::path::Path::new(&root).join("garrysmod/garrysmod_dir.vpk"), "shaders/").unwrap();
        let data = vpk.read("shaders/fxc/g_sky_ps20b.vcs").unwrap();
        crate::vcs::VcsFile::parse(data).unwrap();
    }
}
