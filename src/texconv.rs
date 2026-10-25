//! material system images to gpu formats: rgba8, half floats for 16-bit formats, bc kept compressed

#[derive(Clone, Debug)]
pub enum Image {
    Rgba8 { width: u32, height: u32, data: Vec<u8> },
    /// unclamped
    Rgba16f { width: u32, height: u32, data: Vec<u8> },
    /// 1 = bc1 (dxt1), 2 = bc2 (dxt3), 3 = bc3 (dxt5), 7 = bc7
    Bc { kind: u8, width: u32, height: u32, data: Vec<u8> },
}

impl Image {
    pub fn size(&self) -> (u32, u32) {
        match self {
            Image::Rgba8 { width, height, .. } | Image::Rgba16f { width, height, .. } | Image::Bc { width, height, .. } => (*width, *height),
        }
    }

    pub fn blank(fmt: i32, width: u32, height: u32) -> Image {
        let n = (width * height) as usize;
        if is_wide(fmt) { Image::Rgba16f { width, height, data: vec![0; n * 8] } } else { Image::Rgba8 { width, height, data: vec![0; n * 4] } }
    }

    /// 0 for bc
    pub fn pixel_bytes(&self) -> usize {
        match self {
            Image::Rgba8 { .. } => 4,
            Image::Rgba16f { .. } => 8,
            Image::Bc { .. } => 0,
        }
    }
}

/// 16-bit channels stay half floats instead of rgba8
pub fn is_wide(fmt: i32) -> bool {
    matches!(fmt, RGBA16161616F | RGBA16161616)
}

// imageformat.h
const RGBA8888: i32 = 0;
const ABGR8888: i32 = 1;
const RGB888: i32 = 2;
const BGR888: i32 = 3;
const RGB565: i32 = 4;
const I8: i32 = 5;
const IA88: i32 = 6;
const A8: i32 = 8;
const RGB888_BLUESCREEN: i32 = 9;
const BGR888_BLUESCREEN: i32 = 10;
const ARGB8888: i32 = 11;
const BGRA8888: i32 = 12;
const DXT1: i32 = 13;
const DXT3: i32 = 14;
const DXT5: i32 = 15;
const BGRX8888: i32 = 16;
const BGR565: i32 = 17;
const BGRX5551: i32 = 18;
const BGRA4444: i32 = 19;
const DXT1_ONEBITALPHA: i32 = 20;
const BGRA5551: i32 = 21;
const UV88: i32 = 22;
const UVWQ8888: i32 = 23;
const RGBA16161616F: i32 = 24;
const RGBA16161616: i32 = 25;

/// 0 = unsupported
pub fn bytes_per_pixel(fmt: i32) -> usize {
    match fmt {
        RGBA8888 | ABGR8888 | ARGB8888 | BGRA8888 | BGRX8888 | UVWQ8888 => 4,
        RGBA16161616F | RGBA16161616 => 8,
        26 => 4,
        RGB888 | BGR888 | RGB888_BLUESCREEN | BGR888_BLUESCREEN => 3,
        RGB565 | BGR565 | BGRX5551 | BGRA4444 | BGRA5551 | IA88 | UV88 => 2,
        I8 | A8 => 1,
        _ => 0,
    }
}

fn pixel(fmt: i32, p: &[u8]) -> [u8; 4] {
    let e5 = |v: u16| ((v as u32 * 255 + 15) / 31) as u8;
    let e6 = |v: u16| ((v as u32 * 255 + 31) / 63) as u8;
    let e4 = |v: u16| (v as u8) * 17;
    match fmt {
        RGBA8888 => [p[0], p[1], p[2], p[3]],
        ABGR8888 => [p[3], p[2], p[1], p[0]],
        ARGB8888 => [p[1], p[2], p[3], p[0]],
        BGRA8888 => [p[2], p[1], p[0], p[3]],
        BGRX8888 => [p[2], p[1], p[0], 255],
        RGB888 | RGB888_BLUESCREEN => [p[0], p[1], p[2], 255],
        BGR888 | BGR888_BLUESCREEN => [p[2], p[1], p[0], 255],
        RGB565 => {
            let v = u16::from_le_bytes([p[0], p[1]]);
            [e5(v & 31), e6((v >> 5) & 63), e5(v >> 11), 255]
        }
        BGR565 => {
            let v = u16::from_le_bytes([p[0], p[1]]);
            [e5(v >> 11), e6((v >> 5) & 63), e5(v & 31), 255]
        }
        BGRX5551 | BGRA5551 => {
            let v = u16::from_le_bytes([p[0], p[1]]);
            let a = if fmt == BGRA5551 && v & 0x8000 == 0 { 0 } else { 255 };
            [e5((v >> 10) & 31), e5((v >> 5) & 31), e5(v & 31), a]
        }
        BGRA4444 => {
            let v = u16::from_le_bytes([p[0], p[1]]);
            [e4((v >> 8) & 15), e4((v >> 4) & 15), e4(v & 15), e4(v >> 12)]
        }
        I8 => [p[0], p[0], p[0], 255],
        IA88 => [p[0], p[0], p[0], p[1]],
        A8 => [0, 0, 0, p[0]],
        // signed formats keep their bytes (sampled as snorm)
        UV88 => [p[0], p[1], 0, 127],
        UVWQ8888 => [p[0], p[1], p[2], p[3]],
        RGBA16161616F => {
            // clamped to 0..1
            let h = |i: usize| (half_to_f32(u16::from_le_bytes([p[2 * i], p[2 * i + 1]])).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            [h(0), h(1), h(2), h(3)]
        }
        RGBA16161616 => [p[1], p[3], p[5], p[7]], // high bytes
        26 => [p[0], p[1], p[2], p[3]],
        _ => [255, 0, 255, 255],
    }
}

/// 8-bit formats scaled to 0..1
fn pixel16(fmt: i32, p: &[u8]) -> [u16; 4] {
    let c = |i: usize| u16::from_le_bytes([p[2 * i], p[2 * i + 1]]);
    match fmt {
        RGBA16161616F => [c(0), c(1), c(2), c(3)],
        RGBA16161616 => [0, 1, 2, 3].map(|i| f32_to_half(c(i) as f32 / 65535.0)),
        _ => pixel(fmt, p).map(|v| f32_to_half(v as f32 / 255.0)),
    }
}

/// round to nearest, overflow to inf, tiny values flush to zero
pub fn f32_to_half(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 255) as i32;
    let man = b & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - e);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    let h = ((e as u32) << 10) | (man >> 13);
    // a carry into the exponent is still correct
    (sign as u32 | (h + ((man >> 12) & 1))) as u16
}

fn half_to_f32(h: u16) -> f32 {
    let (sign, exp, man) = ((h >> 15) as u32, ((h >> 10) & 31) as u32, (h & 1023) as u32);
    let bits = match (exp, man) {
        (0, 0) => sign << 31,
        (0, m) => {
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 1024 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((m & 1023) << 13)
        }
        (31, m) => (sign << 31) | (255 << 23) | (m << 13),
        (e, m) => (sign << 31) | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// v8u8 / q8w8v8u8 / x8l8v8u8, sampled as snorm
pub fn is_signed(fmt: i32) -> bool {
    matches!(fmt, UV88 | UVWQ8888 | 26)
}

pub fn convert(fmt: i32, width: u32, height: u32, data: &[u8]) -> Option<Image> {
    let (w, h) = (width as usize, height as usize);
    let bc = match fmt {
        DXT1 | DXT1_ONEBITALPHA => Some((1u8, 8usize)),
        DXT3 => Some((2, 16)),
        // vtf 7.6 BC7 arrives labelled DXT5 (see vtf76)
        DXT5 if crate::vtf76::bc7_upload(&data[..(w.div_ceil(4).max(1) * h.div_ceil(4).max(1) * 16).min(data.len())]) => Some((7, 16)),
        DXT5 => Some((3, 16)),
        _ => None,
    };
    if let Some((kind, block)) = bc {
        let need = w.div_ceil(4).max(1) * h.div_ceil(4).max(1) * block;
        return (data.len() >= need).then(|| Image::Bc { kind, width, height, data: data[..need].to_vec() });
    }
    let bpp = bytes_per_pixel(fmt);
    if bpp == 0 || data.len() < w * h * bpp {
        return None;
    }
    if is_wide(fmt) {
        let mut out = Vec::with_capacity(w * h * 8);
        for p in data[..w * h * bpp].chunks_exact(bpp) {
            for c in pixel16(fmt, p) {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        return Some(Image::Rgba16f { width, height, data: out });
    }
    let mut out = Vec::with_capacity(w * h * 4);
    for p in data[..w * h * bpp].chunks_exact(bpp) {
        out.extend_from_slice(&pixel(fmt, p));
    }
    Some(Image::Rgba8 { width, height, data: out })
}

/// rows are `stride` bytes apart (0 = tight)
#[allow(clippy::too_many_arguments)]
pub fn blit(dst: &mut Image, x: u32, y: u32, w: u32, h: u32, fmt: i32, src: &[u8], stride: usize) -> bool {
    let wide = matches!(dst, Image::Rgba16f { .. });
    let (Image::Rgba8 { width, height, data } | Image::Rgba16f { width, height, data }) = dst else { return false };
    let dbpp = if wide { 8 } else { 4 };
    let bpp = bytes_per_pixel(fmt);
    if bpp == 0 {
        return false;
    }
    let stride = if stride == 0 { w as usize * bpp } else { stride };
    for row in 0..h as usize {
        let ty = y as usize + row;
        if ty >= *height as usize {
            break;
        }
        for col in 0..w as usize {
            let tx = x as usize + col;
            let s = row * stride + col * bpp;
            if tx >= *width as usize || s + bpp > src.len() {
                continue;
            }
            let d = (ty * *width as usize + tx) * dbpp;
            if wide {
                for (i, c) in pixel16(fmt, &src[s..s + bpp]).iter().enumerate() {
                    data[d + 2 * i..d + 2 * i + 2].copy_from_slice(&c.to_le_bytes());
                }
            } else {
                data[d..d + 4].copy_from_slice(&pixel(fmt, &src[s..s + bpp]));
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_and_blits() {
        let Some(Image::Rgba8 { data, .. }) = convert(BGRA8888, 1, 1, &[1, 2, 3, 4]) else { panic!() };
        assert_eq!(data, vec![3, 2, 1, 4]);
        let mut img = Image::Rgba8 { width: 4, height: 4, data: vec![0; 64] };
        assert!(blit(&mut img, 1, 2, 2, 1, I8, &[100, 200], 0));
        let Image::Rgba8 { data, .. } = img else { panic!() };
        assert_eq!(&data[(2 * 4 + 1) * 4..(2 * 4 + 3) * 4], &[100, 100, 100, 255, 200, 200, 200, 255]);
        assert!(matches!(convert(DXT1, 8, 8, &[0; 32]), Some(Image::Bc { kind: 1, .. })));
        // half floats pass through unclamped
        let Some(Image::Rgba16f { data, .. }) = convert(RGBA16161616F, 1, 1, &[0x00, 0x3c, 0x00, 0x38, 0x00, 0x40, 0x00, 0x00]) else { panic!() };
        assert_eq!(data, [0x00, 0x3c, 0x00, 0x38, 0x00, 0x40, 0x00, 0x00]);
        let Some(Image::Rgba16f { data, .. }) = convert(RGBA16161616, 1, 1, &[0xff, 0xff, 0, 0, 0, 0, 0, 0]) else { panic!() };
        assert_eq!(&data[..2], &0x3c00u16.to_le_bytes());
        for v in [0.0f32, 1.0, 0.5, 2.0, 16.0, 1.0 / 16.0, 0.3, 1e-6, 70000.0] {
            let back = half_to_f32(f32_to_half(v));
            assert!((back - v).abs() <= v * 1e-3 + 1e-7 || (v > 65504.0 && back.is_infinite()), "{v} -> {back}");
        }
        let mut img = Image::blank(RGBA16161616, 2, 1);
        assert!(blit(&mut img, 1, 0, 1, 1, I8, &[255], 0));
        let Image::Rgba16f { data, .. } = img else { panic!() };
        assert_eq!(&data[8..10], &0x3c00u16.to_le_bytes());
    }
}
