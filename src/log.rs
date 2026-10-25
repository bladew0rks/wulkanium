//! logs the first calls of every stub slot to shaderapiwgpu.log, with running counts

use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::vtables::{IFACE_NAMES, slot_names};

const LOG_FIRST: u32 = 3;
const MAX_SLOTS: usize = 320;

static COUNTS: [[AtomicU32; MAX_SLOTS]; IFACE_NAMES.len()] = [const { [const { AtomicU32::new(0) }; MAX_SLOTS] }; IFACE_NAMES.len()];
static FILE: Mutex<Option<std::fs::File>> = Mutex::new(None);

#[repr(align(16))]
pub struct Zeroes(pub [u8; 4096]);
pub static ZEROES: Zeroes = Zeroes([0; 4096]);

pub fn line(msg: &str) {
    let mut f = FILE.lock().unwrap_or_else(|e| e.into_inner());
    if f.is_none() {
        *f = std::fs::OpenOptions::new().create(true).append(true).open("shaderapiwgpu.log").ok();
    }
    if let Some(f) = f.as_mut() {
        let _ = writeln!(f, "{msg}");
    }
}

pub fn stub(iface: u8, slot: usize) {
    let n = COUNTS[iface as usize][slot.min(MAX_SLOTS - 1)].fetch_add(1, Ordering::Relaxed) + 1;
    if n <= LOG_FIRST {
        let name = slot_names(iface).get(slot).copied().unwrap_or("?");
        line(&format!("stub {}[{slot}] {name} (call {n})", IFACE_NAMES[iface as usize]));
    }
}

pub fn counts(iface: u8) -> Vec<(usize, u32)> {
    COUNTS[iface as usize].iter().enumerate().map(|(i, c)| (i, c.load(Ordering::Relaxed))).filter(|(_, c)| *c > 0).collect()
}
