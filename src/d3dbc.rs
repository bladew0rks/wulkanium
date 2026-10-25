//! d3d9 shader bytecode decoder (vs/ps 2.x and 3.0)

pub const OP_DCL: u16 = 31;
pub const OP_DEF: u16 = 81;
pub const OP_DEFI: u16 = 48;
pub const OP_DEFB: u16 = 47;
const OP_COMMENT: u16 = 0xfffe;
const OP_END: u16 = 0xffff;

pub const REG_TEMP: u8 = 0;
pub const REG_INPUT: u8 = 1;
pub const REG_CONST: u8 = 2;
pub const REG_ADDR: u8 = 3; // texture in pixel shaders
pub const REG_RASTOUT: u8 = 4;
pub const REG_ATTROUT: u8 = 5;
pub const REG_OUTPUT: u8 = 6; // texcrdout before vs 3.0
pub const REG_CONSTINT: u8 = 7;
pub const REG_COLOROUT: u8 = 8;
pub const REG_DEPTHOUT: u8 = 9;
pub const REG_SAMPLER: u8 = 10;
pub const REG_CONSTBOOL: u8 = 14;
pub const REG_LOOP: u8 = 15;
pub const REG_MISCTYPE: u8 = 17;
pub const REG_LABEL: u8 = 18;
pub const REG_PREDICATE: u8 = 19;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reg {
    pub ty: u8,
    pub num: u32,
}

/// `[reg.component + offset]`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rel {
    pub reg: Reg,
    pub comp: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct Dst {
    pub reg: Reg,
    pub mask: u8,
    pub saturate: bool,
    pub shift: i8,
    pub rel: Option<Rel>,
}

#[derive(Clone, Copy, Debug)]
pub struct Src {
    pub reg: Reg,
    /// four 2-bit selectors, x first
    pub swizzle: u8,
    pub modifier: u8,
    pub rel: Option<Rel>,
}

impl Src {
    pub fn comp(&self, i: usize) -> u8 {
        (self.swizzle >> (2 * i)) & 3
    }
}

#[derive(Clone, Debug)]
pub enum Inst {
    /// `dcl_usageN reg`, or `dcl_2d s0` (usage 2 = 2d, 3 = cube, 4 = volume)
    Dcl { usage: u8, index: u8, dst: Dst, sampler: bool },
    Def { reg: u32, v: [f32; 4] },
    DefI { reg: u32, v: [i32; 4] },
    DefB { reg: u32, v: bool },
    Op { op: u16, ctrl: u8, predicate: Option<Src>, dst: Option<Dst>, src: Vec<Src> },
}

#[derive(Debug)]
pub struct Shader {
    pub pixel: bool,
    pub major: u8,
    pub minor: u8,
    pub insts: Vec<Inst>,
}

fn reg_of(t: u32) -> Reg {
    Reg { ty: (((t >> 28) & 7) | ((t >> 8) & 0x18)) as u8, num: t & 0x7ff }
}

/// dst / src operand counts (dcl and def* are handled separately)
fn arity(op: u16) -> Option<(usize, usize)> {
    Some(match op {
        0 | 28 | 29 | 39 | 42 | 43 | 44 => (0, 0),
        1 | 6 | 7 | 14 | 15 | 16 | 19 | 34 | 35 | 36 | 46 | 78 | 79 | 91 | 92 => (1, 1),
        2 | 3 | 5 | 8 | 9 | 10 | 11 | 12 | 13 | 17 | 20..=24 | 32 | 33 => (1, 2),
        4 | 18 | 80 | 88 | 90 => (1, 3),
        37 => (1, usize::MAX), // 1 src in 3.0, 3 in 2.x
        25 => (0, 1),
        26 => (0, 2),
        27 => (0, 2),
        30 => (0, 1),
        38 | 40 => (0, 1),
        41 | 45 => (0, 2),
        96 => (0, 1),
        65 => (1, 0), // operand is encoded as dst
        66 => (1, usize::MAX),
        93 => (1, 4),
        94 => (1, 2),
        95 => (1, 2),
        _ => return None,
    })
}

pub fn parse(code: &[u32]) -> Result<Shader, String> {
    let ver = *code.first().ok_or("empty shader")?;
    let pixel = match ver >> 16 {
        0xffff => true,
        0xfffe => false,
        _ => return Err(format!("bad version token {ver:#x}")),
    };
    let (major, minor) = (((ver >> 8) & 0xff) as u8, (ver & 0xff) as u8);
    if major < 2 {
        return Err(format!("shader model {major}.{minor} not supported"));
    }
    let mut insts = Vec::new();
    let mut p = 1;
    let tok = |i: usize| code.get(i).copied().ok_or_else(|| "truncated shader".to_string());
    loop {
        let t = tok(p)?;
        let op = (t & 0xffff) as u16;
        if op == OP_END {
            break;
        }
        if op == OP_COMMENT {
            p += 1 + ((t >> 16) & 0x7fff) as usize;
            continue;
        }
        let len = ((t >> 24) & 0xf) as usize;
        let ctrl = ((t >> 16) & 0xff) as u8;
        let predicated = t & (1 << 28) != 0;
        let end = p + 1 + len;
        let mut q = p + 1;
        let read_dst = |q: &mut usize| -> Result<Dst, String> {
            let d = tok(*q)?;
            *q += 1;
            let rel = if d & (1 << 13) != 0 {
                let r = tok(*q)?;
                *q += 1;
                Some(Rel { reg: reg_of(r), comp: ((r >> 16) & 3) as u8 })
            } else {
                None
            };
            let shift = ((d >> 24) & 0xf) as i8;
            Ok(Dst { reg: reg_of(d), mask: ((d >> 16) & 0xf) as u8, saturate: d & (1 << 20) != 0, shift: if shift > 7 { shift - 16 } else { shift }, rel })
        };
        let read_src = |q: &mut usize| -> Result<Src, String> {
            let s = tok(*q)?;
            *q += 1;
            // 2.0+ always has the relative-address token when bit 13 is set
            let rel = if s & (1 << 13) != 0 {
                let r = tok(*q)?;
                *q += 1;
                Some(Rel { reg: reg_of(r), comp: ((r >> 16) & 3) as u8 })
            } else {
                None
            };
            Ok(Src { reg: reg_of(s), swizzle: ((s >> 16) & 0xff) as u8, modifier: ((s >> 24) & 0xf) as u8, rel })
        };
        let inst = match op {
            OP_DCL => {
                let u = tok(q)?;
                q += 1;
                let dst = read_dst(&mut q)?;
                let sampler = dst.reg.ty == REG_SAMPLER;
                if sampler {
                    Inst::Dcl { usage: ((u >> 27) & 0xf) as u8, index: 0, dst, sampler }
                } else {
                    Inst::Dcl { usage: (u & 0x1f) as u8, index: ((u >> 16) & 0xf) as u8, dst, sampler }
                }
            }
            OP_DEF | OP_DEFI | OP_DEFB => {
                let reg = tok(q)? & 0x7ff;
                let v = [tok(q + 1)?, tok(q + 2).unwrap_or(0), tok(q + 3).unwrap_or(0), tok(q + 4).unwrap_or(0)];
                match op {
                    OP_DEF => Inst::Def { reg, v: v.map(f32::from_bits) },
                    OP_DEFI => Inst::DefI { reg, v: v.map(|x| x as i32) },
                    _ => Inst::DefB { reg, v: v[0] != 0 },
                }
            }
            _ => {
                let (nd, mut ns) = arity(op).ok_or_else(|| format!("unsupported opcode {op} at token {p}"))?;
                let mut dst = None;
                if nd == 1 {
                    dst = Some(read_dst(&mut q)?);
                }
                let mut predicate = None;
                if predicated {
                    predicate = Some(read_src(&mut q)?);
                }
                let mut src = Vec::new();
                if ns == usize::MAX {
                    ns = usize::MAX - 1;
                }
                while src.len() < ns && q < end {
                    src.push(read_src(&mut q)?);
                }
                Inst::Op { op, ctrl, predicate, dst, src }
            }
        };
        insts.push(inst);
        p = end;
    }
    Ok(Shader { pixel, major, minor, insts })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcs::VcsFile;
    use std::collections::BTreeMap;

    #[test]
    fn decode_real_shaders() {
        let Ok(dir) = std::env::var("SHADERS_DIR") else { return };
        let mut ops: BTreeMap<u16, usize> = BTreeMap::new();
        let mut regs: BTreeMap<(bool, u8), usize> = BTreeMap::new();
        let mut models: BTreeMap<(bool, u8, u8), usize> = BTreeMap::new();
        let mut dcl: BTreeMap<(bool, u8, u8, u8), usize> = BTreeMap::new();
        let mut max_const: BTreeMap<bool, (u32, String)> = BTreeMap::new();
        let mut rel_ps = 0;
        let mut vs_high: BTreeMap<String, u32> = BTreeMap::new();
        let mut n = 0;
        for e in std::fs::read_dir(dir).unwrap() {
            let path = e.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if name.contains("_ps1") || name.contains("_vs1") {
                continue; // sm1 is never used at dx level 9x
            }
            let vcs = VcsFile::parse(std::fs::read(&path).unwrap()).unwrap();
            let step = (vcs.header.num_static_combos as usize / 4).max(1);
            for rec in (0..vcs.header.num_static_combos as usize - 1).step_by(step) {
                let id = vcs.records_for_test(rec) * vcs.header.dynamic_combos.max(1) as u32;
                for (_, code) in vcs.static_combo(id).unwrap() {
                    let s = parse(&code).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                    n += 1;
                    *models.entry((s.pixel, s.major, s.minor)).or_default() += 1;
                    for i in &s.insts {
                        match i {
                            Inst::Op { op, dst, src, .. } => {
                                *ops.entry(*op).or_default() += 1;
                                for r in dst.iter().map(|d| d.reg).chain(src.iter().map(|s| s.reg)) {
                                    *regs.entry((s.pixel, r.ty)).or_default() += 1;
                                    if r.ty == REG_CONST {
                                        let m = max_const.entry(s.pixel).or_insert((0, String::new()));
                                        if r.num as u32 > m.0 {
                                            *m = (r.num as u32, name.clone());
                                        }
                                    }
                                }
                                if s.pixel && src.iter().any(|x| x.rel.is_some()) {
                                    rel_ps += 1;
                                }
                                if !s.pixel {
                                    for x in src.iter().filter(|x| x.reg.ty == REG_CONST && x.rel.is_none() && x.reg.num > 60) {
                                        vs_high.entry(name.clone()).or_insert(x.reg.num as u32);
                                    }
                                }
                            }
                            Inst::Dcl { usage, index, dst, .. } => *dcl.entry((s.pixel, dst.reg.ty, *usage, *index)).or_default() += 1,
                            _ => {}
                        }
                    }
                }
            }
        }
        println!("max const (pixel?): {max_const:?}, ps rel-addressed ops: {rel_ps}");
        println!("vs reading c61+ directly: {vs_high:?}");
        println!("{n} shaders\nmodels {models:?}\nops {ops:?}\nregs {regs:?}\ndcl {dcl:?}");
    }
}
