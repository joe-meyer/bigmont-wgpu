//! Beacon's `G2C1` candidate files and `G2V1` verdict files (file IO only).
//!
//! Both start with a 16-byte little-endian header: magic `u32`, container bits `u32`, count `u64`.
//! A candidate body is `count` numbers of `ceil(bits / 32)` little-endian `u32` limbs; a verdict
//! body is `count` bytes (0 composite, 1 probable prime, 2 reserved for "not tested").

use num_bigint::BigUint;
use std::path::{Path, PathBuf};

/// `"G2C1"` read as a little-endian `u32`.
pub const CANDIDATE_MAGIC: u32 = 0x3143_3247;
/// `"G2V1"` read as a little-endian `u32`.
pub const VERDICT_MAGIC: u32 = 0x3156_3247;
const HEADER: usize = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateFile {
    /// Container width in bits (the numbers themselves may be narrower).
    pub bits: u32,
    pub numbers: Vec<BigUint>,
}

fn limbs_of(bits: u32) -> usize {
    bits.div_ceil(32) as usize
}

/// Validates the header against the file length and returns `(bits, count)`. All size arithmetic is
/// checked, so a hostile header cannot cause a huge allocation or an overflow.
fn parse_header(raw: &[u8], magic: u32, what: &str, record_bytes: impl Fn(u32) -> usize) -> Result<(u32, usize), String> {
    if raw.len() < HEADER {
        return Err(format!("{what} file is {} bytes, shorter than its {HEADER}-byte header", raw.len()));
    }
    let got = u32::from_le_bytes(raw[0..4].try_into().expect("4 bytes"));
    if got != magic {
        return Err(format!("{what} file has magic {got:#010x}, expected {magic:#010x}"));
    }
    let bits = u32::from_le_bytes(raw[4..8].try_into().expect("4 bytes"));
    let count = u64::from_le_bytes(raw[8..16].try_into().expect("8 bytes"));
    let body = (raw.len() - HEADER) as u64;
    let want = u64::try_from(record_bytes(bits)).ok().and_then(|r| r.checked_mul(count));
    if want != Some(body) {
        return Err(format!("{what} file header claims {count} entries at {bits} bits, which does not match its {body}-byte body"));
    }
    let count = usize::try_from(count).map_err(|_| format!("{what} file entry count {count} does not fit in memory"))?;
    Ok((bits, count))
}

pub fn read_candidates(path: &Path) -> Result<CandidateFile, String> {
    let raw = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let (bits, count) = parse_header(&raw, CANDIDATE_MAGIC, "candidate", |b| limbs_of(b) * 4).map_err(|e| format!("{}: {e}", path.display()))?;
    if bits == 0 {
        return Err(format!("{}: candidate file declares 0-bit numbers", path.display()));
    }
    let stride = limbs_of(bits) * 4;
    let numbers = (0..count)
        .map(|i| {
            let rec = &raw[HEADER + i * stride..HEADER + (i + 1) * stride];
            let limbs: Vec<u32> = rec.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes"))).collect();
            BigUint::new(limbs)
        })
        .collect();
    Ok(CandidateFile { bits, numbers })
}

fn header(magic: u32, bits: u32, count: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(HEADER);
    v.extend_from_slice(&magic.to_le_bytes());
    v.extend_from_slice(&bits.to_le_bytes());
    v.extend_from_slice(&(count as u64).to_le_bytes());
    v
}

/// Writes a candidate file with a `bits`-bit container. Errors if a number does not fit.
pub fn write_candidates(path: &Path, bits: u32, numbers: &[BigUint]) -> Result<(), String> {
    let limbs = limbs_of(bits);
    let mut out = header(CANDIDATE_MAGIC, bits, numbers.len());
    out.reserve(numbers.len().saturating_mul(limbs * 4));
    for (i, n) in numbers.iter().enumerate() {
        if n.bits() > u64::from(bits) {
            return Err(format!("candidate {i} has {} bits, more than the {bits}-bit container", n.bits()));
        }
        let mut digits = n.to_u32_digits();
        digits.resize(limbs, 0);
        for d in digits {
            out.extend_from_slice(&d.to_le_bytes());
        }
    }
    std::fs::write(path, out).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

pub fn read_verdicts(path: &Path) -> Result<(u32, Vec<u8>), String> {
    let raw = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let (bits, _) = parse_header(&raw, VERDICT_MAGIC, "verdict", |_| 1).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((bits, raw[HEADER..].to_vec()))
}

/// Writes the verdict file to `<path>.tmp` and renames it over `path`, so readers never see a
/// partial file.
pub fn write_verdicts(path: &Path, bits: u32, verdicts: &[u8]) -> Result<(), String> {
    let mut out = header(VERDICT_MAGIC, bits, verdicts.len());
    out.extend_from_slice(verdicts);
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, out).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("cannot rename {} to {}: {e}", tmp.display(), path.display())
    })
}
