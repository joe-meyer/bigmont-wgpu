//! In-driver early exit: candidates are grouped into "k"s by a sizes file and tested in waves; a k
//! that has produced a probable prime is retired and its remaining candidates are never tested.

use crate::engine::Engine;
use num_bigint::BigUint;

/// Wave boundaries `[0, b1, b2, ..., s_max]`: wave `w` covers positions `[b[w-1], b[w])` of every
/// k still alive. The first step is `chunk`; the step is multiplied by `growth` after each wave
/// when `growth > 1.0`, rounded half to even, clamped to at least 1, and the last boundary is
/// capped at `s_max` (the largest k size). `chunk <= 0` therefore gives steps of 1, and
/// `s_max == 0` gives `[0]` (nothing to test).
pub fn schedule(chunk: i32, growth: f64, s_max: u32) -> Vec<u32> {
    let s = i64::from(s_max);
    let mut bounds = vec![0u32];
    let mut at = 0i64;
    let mut c = f64::from(chunk);
    while at < s {
        // `as i64` saturates (and maps NaN to 0), so a runaway `c` cannot overflow.
        let step = (c.round_ties_even() as i64).max(1);
        at = at.saturating_add(step).min(s);
        bounds.push(u32::try_from(at).expect("at <= s_max"));
        if growth > 1.0 {
            c *= growth;
        }
    }
    bounds
}

/// Anything that can classify candidates: 0 for composite, 1 for probable prime, in input order.
pub trait VerdictSource {
    fn verdicts(&mut self, ns: &[BigUint]) -> Result<Vec<u8>, String>;
}

impl VerdictSource for Engine {
    fn verdicts(&mut self, ns: &[BigUint]) -> Result<Vec<u8>, String> {
        Engine::verdicts(self, ns)
    }
}

/// What a wave run produced.
#[derive(Debug, PartialEq, Eq)]
pub struct EeOutcome {
    /// One byte per candidate: the source's verdict if tested, `marker` if skipped.
    pub verdicts: Vec<u8>,
    /// Candidates sent to the source.
    pub tested: u64,
    /// Source calls that carried at least one candidate.
    pub waves: u32,
    /// Bytes equal to 1 in `verdicts`.
    pub prp: u64,
}

/// Per-k progress through the waves.
struct KState {
    /// Index of the k's first candidate.
    off: usize,
    size: u32,
    /// Positions `[0, tested)` have been tested.
    tested: u32,
    alive: bool,
}

impl KState {
    /// End (exclusive) of this k's positions in a wave that stops at `bound`. Never before
    /// `tested`, so boundaries that fail to increase cannot move a k backwards.
    fn wave_end(&self, bound: u32) -> u32 {
        self.size.min(bound).max(self.tested)
    }
}

/// Tests `numbers` in waves. `sizes[k]` consecutive candidates belong to k number `k`; wave `w`
/// (1-based) tests positions `[tested, min(size, boundaries[w]))` of every k still alive and then
/// retires each k that produced a hit (verdict 1, or 2 for a Gerbicz alarm) or has been tested to
/// its end. Skipped positions keep `marker` in the result. Stops at the first wave with nothing to
/// test. `sizes` must sum to `numbers.len()`.
pub fn run_waves(
    src: &mut impl VerdictSource,
    numbers: &[BigUint],
    sizes: &[u32],
    boundaries: &[u32],
    marker: u8,
) -> Result<EeOutcome, String> {
    let sum: u64 = sizes.iter().map(|&s| u64::from(s)).sum();
    if sum != numbers.len() as u64 {
        return Err(format!("sizes sum {sum} != count {}", numbers.len()));
    }
    let mut ks = Vec::with_capacity(sizes.len());
    let mut off = 0usize;
    for &size in sizes {
        ks.push(KState { off, size, tested: 0, alive: true });
        off += size as usize;
    }
    let mut verdicts = vec![marker; numbers.len()];
    let mut tested_total = 0u64;
    let mut waves = 0u32;
    let mut idx: Vec<usize> = Vec::new();
    let mut batch: Vec<BigUint> = Vec::new();
    for &bound in boundaries.iter().skip(1) {
        idx.clear();
        for k in ks.iter().filter(|k| k.alive) {
            let hi = k.wave_end(bound);
            idx.extend((k.tested..hi).map(|p| k.off + p as usize));
        }
        if idx.is_empty() {
            break;
        }
        batch.clear();
        batch.extend(idx.iter().map(|&i| numbers[i].clone()));
        let reply = src.verdicts(&batch)?;
        if reply.len() != idx.len() {
            return Err(format!(
                "verdict source returned {} verdicts for {} candidates",
                reply.len(),
                idx.len()
            ));
        }
        for (&i, &v) in idx.iter().zip(&reply) {
            verdicts[i] = v;
        }
        waves += 1;
        tested_total += idx.len() as u64;
        for k in ks.iter_mut().filter(|k| k.alive) {
            let hi = k.wave_end(bound);
            let hit = verdicts[k.off + k.tested as usize..k.off + hi as usize]
                .iter()
                .any(|&v| v == 1 || v == 2);
            k.tested = hi;
            if hit || k.tested >= k.size {
                k.alive = false;
            }
        }
    }
    let prp = verdicts.iter().filter(|&&v| v == 1).count() as u64;
    Ok(EeOutcome { verdicts, tested: tested_total, waves, prp })
}
