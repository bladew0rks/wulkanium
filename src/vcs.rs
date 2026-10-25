//! .vcs v6 shader files. header, static combo records (last is
//! a sentinel), alias records, then per static combo a chain of blocks (bzip2 / lzma / stored)
//! holding (dynamic combo, size, bytecode) entries.

#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub version: i32,
    pub total_combos: i32,
    pub dynamic_combos: i32,
    pub flags: u32,
    pub centroid_mask: u32,
    pub num_static_combos: u32,
    pub source_crc: u32,
}

pub struct VcsFile {
    data: Vec<u8>,
    pub header: Header,
    /// (static combo id, file offset), last is a sentinel
    records: Vec<(u32, u32)>,
    /// (combo, source combo)
    aliases: Vec<(u32, u32)>,
}

fn u32_at(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}

impl VcsFile {
    pub fn parse(data: Vec<u8>) -> Result<VcsFile, String> {
        let h = |i: usize| u32_at(&data, i * 4).ok_or("truncated header");
        let header = Header {
            version: h(0)? as i32,
            total_combos: h(1)? as i32,
            dynamic_combos: h(2)? as i32,
            flags: h(3)?,
            centroid_mask: h(4)?,
            num_static_combos: h(5)?,
            source_crc: h(6)?,
        };
        if header.version != 6 {
            return Err(format!("unsupported vcs version {}", header.version));
        }
        let mut o = 28;
        let mut records = Vec::with_capacity(header.num_static_combos as usize);
        for _ in 0..header.num_static_combos {
            records.push((u32_at(&data, o).ok_or("truncated records")?, u32_at(&data, o + 4).ok_or("truncated records")?));
            o += 8;
        }
        let n_alias = u32_at(&data, o).ok_or("truncated alias count")?;
        o += 4;
        let mut aliases = Vec::with_capacity(n_alias as usize);
        for _ in 0..n_alias {
            aliases.push((u32_at(&data, o).ok_or("truncated aliases")?, u32_at(&data, o + 4).ok_or("truncated aliases")?));
            o += 8;
        }
        Ok(VcsFile { data, header, records, aliases })
    }

    #[cfg(test)]
    pub fn records_for_test(&self, i: usize) -> u32 {
        self.records[i].0
    }

    /// record index for a static combo id, aliases resolved
    fn find_static(&self, id: u32) -> Option<usize> {
        let id = self.aliases.binary_search_by_key(&id, |a| a.0).map_or(id, |i| self.aliases[i].1);
        self.records.binary_search_by_key(&id, |r| r.0).ok().filter(|&i| i + 1 < self.records.len())
    }

    /// all (dynamic combo, bytecode) pairs of the static combo `static_index` belongs to
    pub fn static_combo(&self, static_index: u32) -> Result<Vec<(u32, Vec<u32>)>, String> {
        let dynamic = self.header.dynamic_combos.max(1) as u32;
        let rec = self.find_static(static_index / dynamic).ok_or_else(|| format!("no static combo {}", static_index / dynamic))?;
        let (start, end) = (self.records[rec].1 as usize, self.records[rec + 1].1 as usize);
        let mut p = start;
        let mut out = Vec::new();
        while p + 4 <= end {
            let size = u32_at(&self.data, p).ok_or("truncated block")?;
            p += 4;
            if size == 0xffff_ffff {
                break;
            }
            let len = (size & 0x3fff_ffff) as usize;
            let raw = self.data.get(p..p + len).ok_or("truncated block data")?;
            let block = match size & 0xc000_0000 {
                0x4000_0000 => decompress_lzma(raw)?,
                0x8000_0000 => raw.to_vec(),
                _ => return Err("bzip2 shader blocks are not supported".into()),
            };
            p += len;
            let mut q = 0;
            while q + 8 <= block.len() {
                let (combo, n) = (u32_at(&block, q).unwrap(), u32_at(&block, q + 4).unwrap() as usize);
                q += 8;
                let code = block.get(q..q + n).ok_or("truncated shader")?;
                q += n;
                // v6 stores the dynamic combo, v5 the full combo
                let combo = if combo >= static_index { combo - static_index } else { combo };
                out.push((combo, code.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()));
            }
        }
        Ok(out)
    }
}

/// valve's lzma blocks: "LZMA", sizes, 5 property bytes, raw stream
fn decompress_lzma(b: &[u8]) -> Result<Vec<u8>, String> {
    if b.get(0..4) != Some(b"LZMA") {
        return Err("bad LZMA block id".into());
    }
    let actual = u32_at(b, 4).ok_or("truncated LZMA header")?;
    let props = b.get(12..17).ok_or("truncated LZMA header")?;
    // rebuild an lzma-alone header
    let mut alone = Vec::with_capacity(b.len() + 8);
    alone.extend_from_slice(props);
    alone.extend_from_slice(&(actual as u64).to_le_bytes());
    alone.extend_from_slice(&b[17..]);
    let mut out = Vec::with_capacity(actual as usize);
    lzma_rs::lzma_decompress(&mut std::io::Cursor::new(alone), &mut out).map_err(|e| format!("LZMA: {e}"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// needs SHADERS_DIR
    #[test]
    fn loads_real_shaders() {
        let Ok(dir) = std::env::var("SHADERS_DIR") else { return };
        let mut n = 0;
        for e in std::fs::read_dir(dir).unwrap() {
            let path = e.unwrap().path();
            let vcs = VcsFile::parse(std::fs::read(&path).unwrap()).unwrap();
            let first = vcs.records[0].0 * vcs.header.dynamic_combos.max(1) as u32;
            let combos = vcs.static_combo(first).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert!(!combos.is_empty(), "{}", path.display());
            for (_, code) in &combos {
                let kind = code[0] >> 16;
                assert!(kind == 0xfffe || kind == 0xffff, "{}: token {:#x}", path.display(), code[0]);
            }
            n += 1;
        }
        assert!(n > 0);
    }
}
