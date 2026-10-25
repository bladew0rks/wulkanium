//! VTF 7.6 (Strata Source: zstd / deflate compressed image data, BC6H and BC7) for gmod's
//! materialsystem, which reads up to 7.5. CVTFTexture::Unserialize is hooked: a 7.6 file is
//! rewritten in memory as the 7.5 file the engine understands (decompressed; BC6H decoded to
//! RGBA16161616F, BC5 / BC4 / R8 renamed to their gmod equivalents) and handed to the original.
//!
//! BC7 stays compressed: its blocks are the size of DXT5's, so the engine gets them labelled DXT5
//! and each mip is remembered here; when that exact data reaches TexImage2D (`bc7_upload`) the
//! renderer creates a real BC7 texture.

const HEADER: usize = 80;
const RES_LOW: [u8; 3] = [0x01, 0, 0];
const RES_HIGH: [u8; 3] = [0x30, 0, 0];
const RES_AXC: [u8; 3] = *b"AXC";
/// resource data lives in the entry itself
const RES_NO_DATA: u8 = 0x02;
const TEXTUREFLAGS_ENVMAP: u32 = 0x4000;

// strata image formats beyond gmod's
const BC7: i32 = 70;
const BC6H: i32 = 71;
const BC6H_UNSIGNED: i32 = 72;
const BC5: i32 = 73;
const BC4: i32 = 74;
const R8: i32 = 69;
// gmod's
const DXT5: i32 = 15;
const I8: i32 = 5;
const RGBA16161616F: i32 = 24;
const ATI2N: i32 = 37;
const ATI1N: i32 = 38;

/// bytes of one w x h image
fn image_size(fmt: i32, w: usize, h: usize) -> usize {
    let blocks = w.div_ceil(4).max(1) * h.div_ceil(4).max(1);
    match fmt {
        BC7 | BC6H | BC6H_UNSIGNED | BC5 => blocks * 16,
        BC4 => blocks * 8,
        R8 => w * h,
        _ => crate::api::image_bytes(fmt, w as i32, h as i32),
    }
}

/// what the engine gets instead
fn output_format(fmt: i32) -> i32 {
    match fmt {
        BC7 => DXT5,
        BC6H | BC6H_UNSIGNED => RGBA16161616F,
        BC5 => ATI2N,
        BC4 => ATI1N,
        R8 => I8,
        f => f,
    }
}

/// BC7 mips handed to the engine as DXT5, by (length, sampled hash)
static BC7_MIPS: parking_lot::Mutex<Option<std::collections::HashSet<(usize, u64)>>> = parking_lot::Mutex::new(None);

fn fingerprint(data: &[u8]) -> (usize, u64) {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    let n = data.len().min(256);
    data[..n].hash(&mut h);
    data[data.len() - n..].hash(&mut h);
    (data.len(), h.finish())
}

/// TexImage2D data that is really BC7 (it arrives as DXT5)
pub fn bc7_upload(data: &[u8]) -> bool {
    !data.is_empty() && BC7_MIPS.lock().as_ref().is_some_and(|s| s.contains(&fingerprint(data)))
}

/// kept for textures whose size breaks the gpu's 4x4 block rule
#[allow(dead_code)]
pub fn decode_bc7(data: &[u8], w: usize, h: usize) -> Vec<u8> {
    decode_image(BC7, data, w, h)
}

fn decode_image(fmt: i32, data: &[u8], w: usize, h: usize) -> Vec<u8> {
    match fmt {
        BC7 => {
            let mut out = vec![0u8; w * h * 4];
            for by in 0..h.div_ceil(4) {
                for bx in 0..w.div_ceil(4) {
                    let i = (by * w.div_ceil(4) + bx) * 16;
                    let Some(block) = data.get(i..i + 16) else { continue };
                    let mut px = [0u8; 64];
                    bcdec_rs::bc7(block, &mut px, 16);
                    copy_block(&px, 4, &mut out, w, h, bx, by);
                }
            }
            out
        }
        BC6H | BC6H_UNSIGNED => {
            let mut out = vec![0u8; w * h * 8];
            for by in 0..h.div_ceil(4) {
                for bx in 0..w.div_ceil(4) {
                    let i = (by * w.div_ceil(4) + bx) * 16;
                    let Some(block) = data.get(i..i + 16) else { continue };
                    let mut rgb = [0u16; 48];
                    bcdec_rs::bc6h_half(block, &mut rgb, 12, fmt == BC6H);
                    // rgb halfs -> rgba halfs, alpha 1.0
                    let mut px = [0u8; 128];
                    for p in 0..16 {
                        for c in 0..4 {
                            let v = if c == 3 { 0x3c00 } else { rgb[p * 3 + c] };
                            px[p * 8 + c * 2..p * 8 + c * 2 + 2].copy_from_slice(&v.to_le_bytes());
                        }
                    }
                    copy_block(&px, 8, &mut out, w, h, bx, by);
                }
            }
            out
        }
        _ => data.to_vec(),
    }
}

/// a decoded 4x4 block into the image, clipped at the edges
fn copy_block(px: &[u8], bpp: usize, out: &mut [u8], w: usize, h: usize, bx: usize, by: usize) {
    for y in 0..4 {
        let oy = by * 4 + y;
        if oy >= h {
            break;
        }
        let n = 4.min(w.saturating_sub(bx * 4));
        let dst = (oy * w + bx * 4) * bpp;
        out[dst..dst + n * bpp].copy_from_slice(&px[y * 4 * bpp..(y * 4 + n) * bpp]);
    }
}

fn decompress(chunk: &[u8], expected: usize) -> Option<Vec<u8>> {
    if chunk.len() == expected {
        return Some(chunk.to_vec());
    }
    let out = if chunk.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
        use std::io::Read;
        let mut d = ruzstd::decoding::StreamingDecoder::new(chunk).ok()?;
        let mut v = Vec::with_capacity(expected);
        d.read_to_end(&mut v).ok()?;
        v
    } else if chunk.first().is_some_and(|b| b & 0x0f == 8) && chunk.len() > 2 && u16::from_be_bytes([chunk[0], chunk[1]]) % 31 == 0 {
        miniz_oxide::inflate::decompress_to_vec_zlib(chunk).ok()?
    } else {
        miniz_oxide::inflate::decompress_to_vec(chunk).ok()?
    };
    (out.len() >= expected).then_some(out)
}

fn u16_at(d: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(d.get(o..o + 2)?.try_into().ok()?))
}
fn u32_at(d: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(d.get(o..o + 4)?.try_into().ok()?))
}

/// a 7.6 file as 7.5. with `header_only` (or when the image data isn't all there) only the
/// header and small resources are rewritten, sized for the converted image.
pub fn convert(d: &[u8], header_only: bool) -> Result<Vec<u8>, String> {
    if d.get(0..4) != Some(b"VTF\0") || u32_at(d, 4) != Some(7) || u32_at(d, 8) != Some(6) {
        return Err("not a 7.6 vtf".into());
    }
    let w = u16_at(d, 16).ok_or("short header")? as usize;
    let h = u16_at(d, 18).ok_or("short header")? as usize;
    let flags = u32_at(d, 20).ok_or("short header")?;
    let frames = (u16_at(d, 24).ok_or("short header")? as usize).max(1);
    let fmt = u32_at(d, 52).ok_or("short header")? as i32;
    let mips = (*d.get(56).ok_or("short header")? as usize).max(1);
    let depth = (u16_at(d, 63).unwrap_or(1) as usize).max(1);
    let nres = u32_at(d, 68).ok_or("short header")? as usize;
    if nres > 32 || w == 0 || h == 0 {
        return Err("bad header".into());
    }
    let faces = if flags & TEXTUREFLAGS_ENVMAP != 0 { 6 } else { 1 };
    let mut res = Vec::new();
    for i in 0..nres {
        let o = HEADER + 8 * i;
        let e = d.get(o..o + 8).ok_or("short resource dictionary")?;
        res.push(([e[0], e[1], e[2]], e[3], u32::from_le_bytes([e[4], e[5], e[6], e[7]])));
    }
    let find = |tag: [u8; 3]| res.iter().find(|r| r.0 == tag);
    let out_fmt = output_format(fmt);
    let dims = |m: usize| ((w >> m).max(1), (h >> m).max(1), (depth >> m).max(1));
    // the converted image: mips smallest first, then frames, faces, slices
    let mut high = Vec::new();
    let mut have_image = false;
    if !header_only {
        if let Some(&(_, _, off)) = find(RES_HIGH) {
            let axc = find(RES_AXC).filter(|r| r.1 & RES_NO_DATA == 0);
            let images_per_mip = |m: usize| frames * faces * dims(m).2;
            let mut chunks: Vec<(usize, usize)> = Vec::new();
            if let Some(&(_, _, aoff)) = axc {
                let aoff = aoff as usize;
                let len = u32_at(d, aoff).ok_or("short axc")? as usize;
                let n = len.saturating_sub(4) / 4;
                let mut at = off as usize;
                for i in 0..n {
                    let sz = u32_at(d, aoff + 8 + 4 * i).ok_or("short axc")? as usize;
                    chunks.push((at, sz));
                    at += sz;
                }
            }
            let mut ok = true;
            let mut raw_at = off as usize;
            let mut chunk_i = 0;
            'mips: for m in (0..mips).rev() {
                let (mw, mh, _) = dims(m);
                let one = image_size(fmt, mw, mh);
                let per_mip = chunks.len() == mips;
                let mip_data: Vec<u8> = if chunks.is_empty() {
                    let total = one * images_per_mip(m);
                    let Some(s) = d.get(raw_at..raw_at + total) else { ok = false; break };
                    raw_at += total;
                    s.to_vec()
                } else if per_mip {
                    let (at, sz) = chunks[chunk_i];
                    chunk_i += 1;
                    let Some(c) = d.get(at..at + sz) else { ok = false; break };
                    match decompress(c, one * images_per_mip(m)) {
                        Some(v) => v,
                        None if high.is_empty() => return Err(format!("mip {m} didn't decompress")),
                        None => break,
                    }
                } else {
                    let mut v = Vec::new();
                    for _ in 0..images_per_mip(m) {
                        let Some(&(at, sz)) = chunks.get(chunk_i) else { ok = false; break 'mips };
                        chunk_i += 1;
                        let Some(c) = d.get(at..at + sz) else { ok = false; break 'mips };
                        v.extend(decompress(c, one).ok_or(format!("mip {m} didn't decompress"))?);
                    }
                    v
                };
                for img in 0..images_per_mip(m) {
                    let s = mip_data.get(img * one..(img + 1) * one).ok_or("short mip")?;
                    if fmt == BC7 {
                        BC7_MIPS.lock().get_or_insert_with(Default::default).insert(fingerprint(s));
                        high.extend_from_slice(s);
                    } else {
                        high.extend(decode_image(fmt, s, mw, mh));
                    }
                }
            }
            // a partial read (the engine skipping top mips) still has the small ones, which come
            // first: keep what decoded
            let _ = ok;
            have_image = !high.is_empty();
        }
    }
    // rebuild: header, resource dictionary, resource data, low res image, high res image
    let keep: Vec<_> = res.iter().filter(|r| r.0 != RES_AXC).collect();
    let mut out = d[..HEADER].to_vec();
    out[8..12].copy_from_slice(&5u32.to_le_bytes());
    let header_size = (HEADER + 8 * keep.len()) as u32;
    out[12..16].copy_from_slice(&header_size.to_le_bytes());
    out[52..56].copy_from_slice(&out_fmt.to_le_bytes());
    out[68..72].copy_from_slice(&(keep.len() as u32).to_le_bytes());
    let dict_at = out.len();
    out.resize(header_size as usize, 0);
    let low_fmt = u32_at(d, 57).unwrap_or(u32::MAX) as i32;
    let low_size = if low_fmt >= 0 { image_size(low_fmt, d[61] as usize, d[62] as usize) } else { 0 };
    for (i, &&(tag, rflags, val)) in keep.iter().enumerate() {
        let new_val = if rflags & RES_NO_DATA != 0 {
            val
        } else if tag == RES_HIGH {
            // after everything else; past the end when the image isn't converted (header only)
            u32::MAX
        } else if tag == RES_LOW {
            let at = out.len() as u32;
            if let Some(s) = d.get(val as usize..val as usize + low_size) {
                out.extend_from_slice(s);
            }
            at
        } else {
            let at = out.len() as u32;
            let len = u32_at(d, val as usize).unwrap_or(0) as usize;
            if let Some(s) = d.get(val as usize..val as usize + 4 + len) {
                out.extend_from_slice(s);
            }
            at
        };
        let e = dict_at + 8 * i;
        out[e..e + 3].copy_from_slice(&tag);
        out[e + 3] = rflags;
        out[e + 4..e + 8].copy_from_slice(&new_val.to_le_bytes());
    }
    let high_at = out.len() as u32;
    for (i, r) in keep.iter().enumerate() {
        if r.0 == RES_HIGH {
            let e = dict_at + 8 * i;
            out[e + 4..e + 8].copy_from_slice(&high_at.to_le_bytes());
        }
    }
    if have_image {
        out.extend(high);
    }
    Ok(out)
}

#[cfg(windows)]
mod hook {
    use super::*;
    use crate::log;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static ORIGINAL: AtomicUsize = AtomicUsize::new(0);
    type Unserialize = unsafe extern "C" fn(usize, *mut u8, bool, i32) -> bool;

    /// CUtlBuffer (x64, from CUtlBuffer::Get in materialsystem): memory @0, allocation count @8,
    /// grow size @0xc, get @0x10, put @0x14, error @0x18, max put @0x20, offset @0x24, get
    /// overflow fn @0x28
    const BUF_SIZE: usize = 0x40;

    unsafe extern "C" fn no_overflow(_buf: usize, _size: i32) -> bool {
        false
    }

    unsafe extern "C" fn unserialize(this: usize, buf: *mut u8, header_only: bool, skip: i32) -> bool {
        let original: Unserialize = unsafe { std::mem::transmute(ORIGINAL.load(Ordering::Relaxed)) };
        unsafe {
            let mem = *(buf as *const *const u8);
            let get = *(buf.add(0x10) as *const i32);
            let max_put = *(buf.add(0x20) as *const i32);
            let offset = *(buf.add(0x24) as *const i32);
            let avail = (max_put - get).max(0) as usize;
            if !mem.is_null() && avail >= 12 {
                let data = std::slice::from_raw_parts(mem.add((get - offset).max(0) as usize), avail);
                if &data[..4] == b"VTF\0" && u32::from_le_bytes(data[8..12].try_into().unwrap()) == 6 {
                    static CALLS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    let call = CALLS.fetch_add(1, Ordering::Relaxed);
                    let result = convert(data, header_only);
                    if call < 64 || call.is_power_of_two() {
                        log::line(&format!("vtf 7.6: call {call}: header_only {header_only} skip {skip}, {avail} bytes in -> {}", match &result { Ok(v) => format!("{} bytes", v.len()), Err(e) => e.clone() }));
                    }
                    match result {
                        Ok(conv) => {
                            let mut fake = [0u8; BUF_SIZE];
                            std::ptr::copy_nonoverlapping(buf, fake.as_mut_ptr(), BUF_SIZE);
                            let f = fake.as_mut_ptr();
                            *(f as *mut *const u8) = conv.as_ptr();
                            *(f.add(0x8) as *mut i32) = conv.len() as i32;
                            *(f.add(0xc) as *mut i32) = 0;
                            *(f.add(0x10) as *mut i32) = 0;
                            *(f.add(0x14) as *mut i32) = conv.len() as i32;
                            *f.add(0x18) = 0;
                            *(f.add(0x20) as *mut i32) = conv.len() as i32;
                            *(f.add(0x24) as *mut i32) = 0;
                            *(f.add(0x28) as *mut usize) = no_overflow as *const () as usize;
                            let ok = original(this, f, header_only, skip);
                            // the caller's buffer counts as read
                            *(buf.add(0x10) as *mut i32) = max_put;
                            return ok;
                        }
                        Err(e) => log::line(&format!("vtf 7.6: {e}")),
                    }
                }
            }
            original(this, buf, header_only, skip)
        }
    }

    /// the vtable of `class` in a loaded module, found through its rtti
    unsafe fn find_vtable(module: usize, class: &[u8]) -> Option<*mut usize> {
        unsafe {
            let pe = module + *((module + 0x3c) as *const u32) as usize;
            let nsec = *((pe + 6) as *const u16) as usize;
            let opt = *((pe + 20) as *const u16) as usize;
            let mut data_secs = Vec::new();
            for i in 0..nsec {
                let s = pe + 24 + opt + 40 * i;
                let va = *((s + 12) as *const u32) as usize;
                let size = *((s + 8) as *const u32) as usize;
                let ch = *((s + 36) as *const u32);
                // initialized data, readable, not code
                if ch & 0x40 != 0 && ch & 0x4000_0000 != 0 && ch & 0x20 == 0 {
                    data_secs.push(std::slice::from_raw_parts((module + va) as *const u8, size));
                }
            }
            let name: Vec<u8> = [b".?AV".as_slice(), class, b"@@\0"].concat();
            let td = data_secs.iter().find_map(|s| s.windows(name.len()).position(|w| w == name.as_slice()).map(|p| s.as_ptr() as usize + p - 16))?;
            let td_rva = (td - module) as u32;
            // RTTICompleteObjectLocator: signature 1, offset 0, cd offset, type descriptor rva, ...,
            // self rva
            let col = data_secs.iter().find_map(|s| {
                (0..s.len().saturating_sub(24)).step_by(4).find_map(|i| {
                    let p = s.as_ptr() as usize + i;
                    let r = |o: usize| *((p + o) as *const u32);
                    (r(0) == 1 && r(4) == 0 && r(12) == td_rva && r(20) as usize == p - module).then_some(p)
                })
            })?;
            let needle = (col as u64).to_le_bytes();
            let meta = data_secs.iter().find_map(|s| (0..s.len().saturating_sub(8)).step_by(8).find(|&i| s[i..i + 8] == needle).map(|i| s.as_ptr() as usize + i))?;
            Some((meta + 8) as *mut usize)
        }
    }

    pub fn install() {
        use windows_sys::Win32::System::LibraryLoader::GetModuleHandleA;
        use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
        if ORIGINAL.load(Ordering::Relaxed) != 0 {
            return;
        }
        unsafe {
            let m = GetModuleHandleA(c"materialsystem.dll".as_ptr() as *const u8);
            if m.is_null() {
                log::line("vtf 7.6: materialsystem.dll not loaded");
                return;
            }
            let Some(vt) = find_vtable(m as usize, b"CVTFTexture") else {
                log::line("vtf 7.6: CVTFTexture vtable not found, 7.6 textures won't load");
                return;
            };
            // IVTFTexture::Unserialize
            let slot = vt.add(9);
            let mut old = 0;
            if VirtualProtect(slot as *const _, 8, PAGE_READWRITE, &mut old) == 0 {
                log::line("vtf 7.6: can't unprotect the vtable");
                return;
            }
            ORIGINAL.store(*slot, Ordering::Relaxed);
            *slot = unserialize as *const () as usize;
            VirtualProtect(slot as *const _, 8, old, &mut old);
            log::line(&format!("vtf 7.6: CVTFTexture::Unserialize hooked ({:#x})", ORIGINAL.load(Ordering::Relaxed)));
        }
    }
}

#[cfg(windows)]
pub use hook::install;
#[cfg(not(windows))]
pub fn install() {}

#[cfg(test)]
mod tests {
    use super::*;

    /// VTF_76_DIR=dir with 7.6 files converts each and checks the result is a sane 7.5 file
    #[test]
    fn converts_real_files() {
        let Ok(dir) = std::env::var("VTF_76_DIR") else { return };
        let mut n = 0;
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.extension().is_none_or(|x| x != "vtf") {
                continue;
            }
            let d = std::fs::read(&p).unwrap();
            if u32_at(&d, 8) != Some(6) {
                continue;
            }
            let out = convert(&d, false).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            assert_eq!(u32_at(&out, 8), Some(5));
            let (w, h, mips) = (u16_at(&out, 16).unwrap() as usize, u16_at(&out, 18).unwrap() as usize, out[56] as usize);
            let fmt = u32_at(&out, 52).unwrap() as i32;
            let total: usize = (0..mips).map(|m| image_size(fmt, (w >> m).max(1), (h >> m).max(1))).sum();
            let nres = u32_at(&out, 68).unwrap() as usize;
            let high = (0..nres).find(|&i| out[HEADER + 8 * i] == 0x30).map(|i| u32_at(&out, HEADER + 8 * i + 4).unwrap() as usize).unwrap();
            assert_eq!(out.len() - high, total, "{}", p.display());
            // the largest mip isn't blank
            let top = &out[out.len() - image_size(fmt, w, h)..];
            assert!(top.iter().any(|&b| b != 0 && b != 255), "{}", p.display());
            if let (Ok(dump), BC7) = (std::env::var("VTF_76_DUMP"), u32_at(&d, 52).unwrap() as i32) {
                let top = decode_bc7(&out[out.len() - image_size(fmt, w, h)..], w, h);
                assert!(bc7_upload(&out[out.len() - image_size(fmt, w, h)..]));
                // top mip as a ppm, to look at
                let mut ppm = format!("P6 {w} {h} 255\n").into_bytes();
                for px in top.chunks_exact(4) {
                    ppm.extend_from_slice(&px[..3]);
                }
                std::fs::write(format!("{dump}/{}.ppm", p.file_stem().unwrap().to_string_lossy()), ppm).unwrap();
            }
            let hdr = convert(&d, true).unwrap();
            assert_eq!(u32_at(&hdr, 8), Some(5));
            n += 1;
        }
        assert!(n > 0);
    }
}
