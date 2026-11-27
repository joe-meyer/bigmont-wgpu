//! WGSL generator for the symmetric squaring of the 13-bit limb scheme (`Squaring::Symmetric`):
//! cooperative Montgomery squaring that forms `x^2` once, with about three quarters of the limb
//! products of the fused loop in `unrolled.rs`, then reduces it. Same bindings, modes, targets and
//! result contract as `unrolled.rs` (`x <- x^2 R^-1 mod n`, result in [0, 2n), limbs fully normalised,
//! `R = 2^(13 L) > 16 n`). It needs no 64-bit integers, so it also runs in the browser. Multiply mode
//! keeps the fused generator.
//!
//! The schedule is the one of `limb32sq.rs` (cuda-fixnum's "rediagonalisation" with a lane's whole
//! limb run as the digit), adapted to redundant 13-bit limbs: the 32-bit kernel needs exact carry
//! chains, here every word is a 32-bit accumulator of 26- or 27-bit products and the carries are
//! deferred to light passes, as in the fused kernel. Every bound below is also checked when the
//! source is generated (`Bound` and the `assert!`s), for each lane count and limb count.
//!
//! ## Phase 1: `S = x^2` in two accumulators per lane
//!
//! Let `W = TPI`, block `A_l` lane `l`'s `LPL` limbs and `B = 2^(13 LPL)`. Lane `j` owns two output
//! blocks of `S`: block `j` in `lo0..` and block `W + j` in `hi0..` (32-bit accumulators, so `S` is
//! held in redundant form).
//!
//! In iteration `i` (`0 <= i < W/2`) lane `j` plays the role `l = (j - 2 i) mod W` of `limb32sq.rs`: it
//! multiplies blocks `(p, q) = (i, l + i)` if `l + i < W`, else `(l + i - W/2, W/2 + i)`, so
//! `p + q = l + 2 i`, which is `j` or `j + W`: the low half of the block product belongs to the lane
//! itself (to `lo` if `j >= 2 i`, else to `hi`), the high half to the next lane (one rotation of `LPL`
//! words). Over all lanes and iterations this visits every pair `p < q` once and the diagonals `(i, i)`
//! for `i < W/2` (role 0 is the only one with `p == q`). Cross products are needed twice, so the second
//! operand is doubled limbwise (limbs below `2^14`: in redundant form doubling needs no carry); role 0
//! uses the undoubled operand. The upper diagonals `(a, a)`, `a >= W/2`, are added at the end: lane `j`
//! squares block `W/2 + j/2` and keeps the low half (even `j`) or the high half (odd `j`), both of
//! which are its own block `W + j`.
//!
//! The block product is formed row by row straight onto the lane's own low-half block (`lo`, which
//! names whichever block receives it, see `phase1`) and an `LPL`-word high half (`ph`); every word gains
//! one product below `2^27` per row. A light pass over those `2 LPL` words (in-lane, every word keeps
//! its low 13 bits and passes the rest one word up, so the low block's earlier contents move up with the
//! right weight too; the top word keeps everything, and its bound is tracked like every other word's)
//! runs whenever the next row could overflow a word, and
//! once at the end, so every word is then below `2^13 + 2^19`. An output word gains at most one routed
//! high-half word per iteration on top of that, so it stays far below `2^32` for every width.
//!
//! Register pressure (measured on an RX 6700 XT): the iterations are forced to run in order and the
//! operand can be held two limbs per word; see `ORDER` and `PACK` on `phase1`.
//!
//! ## Phase 2: reduction of the low half, then the high half added
//!
//! `S = S_lo + S_hi R` with `S_lo` the value of `lo` and `S_hi` the value of `hi` (both redundant, so
//! `S_lo` may exceed `R`). The loop is the fused loop without the `x b` products: `m` zeroes the window's
//! limb 0, `t <- (t + n m) / 2^13`, starting from `t = lo`, and `M` is the sum of the `m` digits. Then
//! `S_lo + M n` is divisible by `R`, so `(S + M n) / R = (S_lo + M n) / R + S_hi`: the high half is just
//! added at the end, limb by limb (lane `j` holds both block `j` of the window and block `W + j` of
//! `S`), with no injection into the window and no extra shuffle per step.
//!
//! Nothing is lost at the top: the window after step `k` has the value `(S_lo + M_k n) / 2^(13(k+1))`,
//! below `S_lo / 2^13 + n < R` (every `lo` word is far below `2^26`, so `S_lo < 2^13 R`), and a window
//! whose value is below `R` has a top limb below `2^13` (all limbs are non-negative), so the carry the
//! light pass drops off the last lane's top limb is zero. The first light pass comes after at least one
//! step. The result is `(x^2 + M n) / R < 16 n^2 / R + n < 2 n` for `x < 4 n` (R > 16n, as for the fused
//! kernel), and the final full normalisation drops nothing either.
//!
//! Per step only the `n m` product (below `2^26`) and the lane-0 carry (below `2^19`) are added, so the
//! light passes are about three times rarer than in the fused loop (`PERIOD` there is 20); the period is
//! the largest that keeps every word below `2^32`, including the `hi` words added at the end.

use crate::kernel::{KernelConfig, Packing, SUBGROUP_HELPERS, Target};
use crate::unrolled::{
    Gen, Mode, conditional_double, full_normalize, hoists, light_pass, prologue, store,
};

/// Largest limb value (`MASK`).
const M: u64 = 0x1fff;
/// Largest accumulator value.
const LIMIT: u64 = u32::MAX as u64;
/// Operand mask that steers AMD's compiler to the 24-bit multiply-add.
const M24: &str = "0xffffffu";
/// Largest value a word has right after a light pass: its own low 13 bits plus the carry of a word
/// below `2^32`.
const AFTER_PASS: u64 = M + (LIMIT >> 13);

/// Emits the complete WGSL source for `cfg` (an unrolled 13-bit configuration with `tpi` 4, 8, 16 or
/// 32) in [`Mode::Squarings`] or [`Mode::Fermat`]. Bindings as in `unrolled::wgsl`.
pub(crate) fn wgsl(cfg: &KernelConfig, target: Target, mode: Mode) -> Result<String, String> {
    if !matches!(cfg.tpi, 4 | 8 | 16 | 32) {
        return Err(format!(
            "the symmetric generator supports tpi 4, 8, 16, 32 only (got {})",
            cfg.tpi
        ));
    }
    if mode == Mode::Mul {
        return Err("the symmetric squaring generator has no multiply mode".to_string());
    }
    let w = cfg.tpi as usize;
    let lpl = cfg.limbs_per_lane() as usize;
    let mut g = Gen::new();
    let prelude = target.prelude();
    if !prelude.is_empty() {
        g.raw(prelude.trim_end());
    }
    g.line(format!(
        "// Generated by bigmont-core unrolledsq.rs: TPI {w}, {lpl} limbs per lane, symmetric squaring then reduction."
    ));
    g.line(format!("const TPI: u32 = {w}u;     // lanes per number"));
    g.line(format!(
        "const LPL: u32 = {lpl}u;     // limbs per lane; TPI * LPL = number of limbs"
    ));
    g.line("const L: u32 = TPI * LPL;");
    g.line(format!("const WG: u32 = {}u;", cfg.wg));
    g.line("const B: u32 = 13u;");
    g.line("const MASK: u32 = 0x1fffu;");
    // Phase 1 first: phase 2's schedule depends on the bounds phase 1 leaves.
    let mut body = Gen::new();
    body.depth = 2;
    let out = phase1(&mut body, w, lpl, cfg.packing == Packing::Pairs);
    let period = reduce_period(out.lo, out.hi);
    let in_body = lpl > period;
    let k = if in_body { 1 } else { period / lpl };
    if in_body {
        g.line(format!(
            "// reduction: light carry passes inside the unrolled body every {period} steps (LPL {lpl} > {period})"
        ));
    } else {
        g.line(format!(
            "const K: u32 = {k}u;       // reduction owners between light carry passes (K * LPL <= {period} steps)"
        ));
    }
    g.line("");
    if mode == Mode::Fermat {
        g.line("@group(0) @binding(0) var<uniform> params: vec4<u32>;          // x = instance count, y = ladder iterations in this dispatch");
    } else {
        g.line("@group(0) @binding(0) var<uniform> params: vec4<u32>;          // x = instance count, y = squarings");
    }
    g.line("@group(0) @binding(1) var<storage, read> n_limbs: array<u32>;");
    g.line("@group(0) @binding(2) var<storage, read> ninv_buf: array<u32>;");
    g.line("@group(0) @binding(3) var<storage, read_write> x_limbs: array<u32>;");
    if mode == Mode::Fermat {
        g.line("@group(0) @binding(4) var<storage, read> e_words: array<u32>;   // n - 1, words per instance = fp.z");
        g.line("@group(0) @binding(5) var<uniform> fp: vec4<u32>;               // x = first iteration, y = nbits, z = words per instance");
    }
    g.raw(SUBGROUP_HELPERS);
    g.line("@compute @workgroup_size(WG)");
    g.line("fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) sg_in: u32) {");
    g.depth += 1;
    prologue(&mut g);
    for c in 0..lpl {
        g.line(format!("var x{c} = x_limbs[off + {c}u];"));
    }
    // Only limb 0 of n stays live across phase 1 (for the hoisted N0 and NN0): the other limbs are
    // loaded by phase 2, which is the only user, to keep phase 1's register pressure down.
    g.line("let n0 = n_limbs[off];");
    hoists(&mut g);
    g.line("for (var s = 0u; s < params.y; s++) {");
    g.depth += 1;
    g.lines.append(&mut body.lines);
    phase2(&mut g, lpl, in_body, period);
    for c in 0..lpl {
        g.line(format!("x{c} = t{c};"));
    }
    if mode == Mode::Fermat {
        conditional_double(&mut g, lpl as u32);
    }
    g.depth -= 1;
    g.line("}");
    store(&mut g, lpl as u32);
    g.depth -= 1;
    g.line("}");
    Ok(g.lines.join("\n") + "\n")
}

/// Generation-time upper bounds of the block product's accumulator words.
struct Bound(Vec<u64>);

impl Bound {
    /// Whether adding `add` to every word in `cols` keeps them all at most `LIMIT`.
    fn fits(&self, cols: impl IntoIterator<Item = (usize, u64)>) -> bool {
        cols.into_iter().all(|(c, add)| self.0[c] + add <= LIMIT)
    }
    fn add(&mut self, cols: impl IntoIterator<Item = (usize, u64)>) {
        for (c, add) in cols {
            self.0[c] += add;
            assert!(self.0[c] <= LIMIT, "accumulator bound exceeded");
        }
    }
    /// The bounds after `product_pass`.
    fn pass(&mut self) {
        let old = self.0.clone();
        let top = old.len() - 1;
        for c in 0..=top {
            let carry = if c == 0 { 0 } else { old[c - 1] >> 13 };
            let own = if c == top { old[c] } else { M };
            self.0[c] = own + carry;
            assert!(self.0[c] <= LIMIT, "accumulator bound exceeded");
        }
    }
    fn max(&self, range: std::ops::Range<usize>) -> u64 {
        self.0[range].iter().copied().max().unwrap_or(0)
    }
}

/// In-lane light pass over a `2 lpl`-word product held in the words `name(0..2 lpl)`: every word but the
/// top keeps its low 13 bits and passes the rest one word up; the top word keeps everything (its
/// bound is tracked in `bound` like every other word's).
fn product_pass(g: &mut Gen, lpl: usize, name: &dyn Fn(usize) -> String, bound: &mut Bound) {
    let top = 2 * lpl - 1;
    g.line("{");
    g.depth += 1;
    for c in 0..top {
        g.line(format!("let o{c} = {};", name(c)));
    }
    for c in 0..=top {
        let carry = if c == 0 {
            String::new()
        } else {
            format!(" + (o{} >> B)", c - 1)
        };
        let own = if c == top {
            name(c)
        } else {
            format!("(o{c} & MASK)")
        };
        g.line(format!("{} = {own}{carry};", name(c)));
    }
    g.depth -= 1;
    g.line("}");
    bound.pass();
}

/// Adds one row of products (`(column, bound)` pairs) to a product held in `name(..)`, with a light
/// pass first when the row could overflow a word.
fn row(
    g: &mut Gen,
    lpl: usize,
    name: &dyn Fn(usize) -> String,
    bound: &mut Bound,
    cols: &[(usize, u64)],
    emit: impl FnOnce(&mut Gen),
) {
    if !bound.fits(cols.iter().copied()) {
        product_pass(g, lpl, name, bound);
    }
    bound.add(cols.iter().copied());
    emit(g);
}

/// What phase 1 leaves: an upper bound of the `lo` and `hi` accumulator words.
struct Phase1 {
    lo: u64,
    hi: u64,
}

/// `let all = lo0 | hi0 | ...;`: a value that depends on every accumulator word, for `ORDER`.
fn all_words(g: &mut Gen, lpl: usize) {
    let words: Vec<String> = (0..lpl)
        .flat_map(|c| [format!("lo{c}"), format!("hi{c}")])
        .collect();
    g.line(format!("let all = {};", words.join(" | ")));
}

/// `let {name}0.. = ` the block of lane `{lane}` (one shuffle per limb, or per pair of limbs and an
/// unpack with `pairs`), every limb below 2^13.
fn fetch(g: &mut Gen, lpl: usize, pairs: bool, name: &str, packed: &str, lane: &str) {
    if pairs {
        for m in 0..lpl.div_ceil(2) {
            g.line(format!("let {packed}{m} = bcast(xp{m}, {lane}, base);"));
        }
        for k in 0..lpl {
            g.line(format!("let {name}{k} = {};", unpack(packed, k)));
        }
    } else {
        for k in 0..lpl {
            g.line(format!(
                "let {name}{k} = bcast(x{k}, {lane}, base) & {M24};"
            ));
        }
    }
}

/// Limb `k` of a block packed two limbs per word in `{prefix}0..` (`PACK`): the low or high 16 bits.
/// Both are below 2^13, which the mask or shift lets the compiler see (24-bit multiplies).
fn unpack(prefix: &str, k: usize) -> String {
    if k.is_multiple_of(2) {
        format!("({prefix}{} & MASK)", k / 2)
    } else {
        format!("({prefix}{} >> 16u)", k / 2)
    }
}

/// Swaps `lo` and `hi` on the lanes where `cond` holds.
fn swap(g: &mut Gen, lpl: usize, cond: &str) {
    g.line(format!("let sw = {cond};"));
    for c in 0..lpl {
        g.line(format!(
            "{{ let a = select(lo{c}, hi{c}, sw); hi{c} = select(hi{c}, lo{c}, sw); lo{c} = a; }}"
        ));
    }
}

/// Phase 1: defines `lo0..` (lane `j` holds block `j` of `x^2`) and `hi0..` (block `W + j`), in
/// redundant form. See the module comment for the schedule and the bounds.
///
/// Register use. During the cross iterations `lo` names the block that receives the lane's low half
/// (`ORDER` below): block `j` while `j >= 2 i`, block `W + j` after. The low half is then accumulated
/// straight into `lo` and only the high half needs its own `LPL` words (`ph`); the in-lane light pass
/// runs over `lo` and `ph` as one `2 LPL`-word number, which has exactly the right weights (`ph` is
/// the block above `lo`). The swap happens once per lane, at the iteration where its target changes,
/// and is undone after the last iteration.
///
/// `PACK` ([`Packing::Pairs`]): phase 1 reads `x` only as shuffle sources, so it can hold it packed two
/// 13-bit limbs per word (`xp`): half the registers, and one shuffle moves two limbs, for one AND or
/// shift per unpacked limb. Whether that pays depends on the compiler (it may repeat the unpack at
/// every use to save registers), so both forms are generated and the tuner picks.
///
/// `ORDER`: every block product depends only on `x`, so a compiler is free to overlap several
/// iterations to hide shuffle latency, which keeps several products live at once (on an RX 6700 XT
/// that spilled at TPI 4 and 8). Each iteration therefore starts with `all & params.z` added to its
/// lane index, where `all` depends on every accumulator word: `params.z` is always 0 (every host
/// writes `[count, iterations, 0, 0]`), but the compiler cannot know it, so the iteration's shuffles
/// wait until the previous iteration's results are in.
fn phase1(g: &mut Gen, w: usize, lpl: usize, pairs: bool) -> Phase1 {
    let half = w / 2;
    let words = 2 * lpl;
    if pairs {
        // x packed two limbs per word (`PACK`); x itself is dead until phase 2 assigns it.
        for m in 0..lpl.div_ceil(2) {
            if 2 * m + 1 < lpl {
                g.line(format!("let xp{m} = x{} | (x{} << 16u);", 2 * m, 2 * m + 1));
            } else {
                g.line(format!("let xp{m} = x{};", 2 * m));
            }
        }
    }
    for c in 0..lpl {
        g.line(format!("var lo{c} = 0u;"));
        g.line(format!("var hi{c} = 0u;"));
    }
    // One bound for every output word of every lane.
    let mut acc = 0u64;
    let cross = |c: usize| {
        if c < lpl {
            format!("lo{c}")
        } else {
            format!("ph{}", c - lpl)
        }
    };
    for i in 0..half {
        g.line("{");
        g.depth += 1;
        let shift = (w - 2 * i) % w;
        let dep = if i > 0 {
            all_words(g, lpl);
            " + (all & params.z)"
        } else {
            ""
        };
        g.line(format!(
            "let role = (lane + {shift}u{dep}) & (TPI - 1u);       // l = (lane - 2 i) mod TPI"
        ));
        g.line(format!("let wrap = role + {i}u >= TPI;"));
        g.line(format!(
            "let pi = select({i}u, role + {i}u - {half}u, wrap);       // block p of this lane's pair"
        ));
        g.line(format!(
            "let qi = select(role + {i}u, {}u, wrap);                  // block q; p + q = role + 2 i = lane (mod TPI)",
            half + i
        ));
        g.line("let single = role == 0u;                         // p == q only for role 0: that product must not be doubled");
        if i > 0 {
            // Lanes 2 i - 2 and 2 i - 1 now receive their low half in block lane + TPI.
            swap(
                g,
                lpl,
                &format!("lane >= {}u && lane < {}u", 2 * i - 2, 2 * i),
            );
        }
        fetch(g, lpl, pairs, "ap", "aw", "pi");
        for c in 0..lpl {
            g.line(format!("var ph{c} = 0u;"));
        }
        // Block product A_p * D_q, D_q = 2 A_q limbwise (A_q for role 0), added onto the low-half
        // block `lo`: every product below 2^27.
        let mut bound = Bound((0..words).map(|c| if c < lpl { acc } else { 0 }).collect());
        for j in 0..lpl {
            if pairs && j.is_multiple_of(2) {
                // One shuffle brings limbs j and j + 1 of A_q. In iteration 0 no role wraps, so q is
                // the lane itself and its limbs need no shuffle.
                let m = j / 2;
                if i == 0 {
                    g.line(format!("let bw{m} = xp{m};"));
                } else {
                    g.line(format!("let bw{m} = bcast(xp{m}, qi, base);"));
                }
            }
            let cols: Vec<(usize, u64)> = (0..lpl).map(|k| (j + k, 2 * M * M)).collect();
            row(g, lpl, &cross, &mut bound, &cols, |g| {
                g.line("{");
                g.depth += 1;
                let bq = if pairs {
                    unpack("bw", j)
                } else if i == 0 {
                    // In iteration 0 no role wraps, so q is the lane itself: no shuffle.
                    format!("x{j}")
                } else {
                    format!("bcast(x{j}, qi, base)")
                };
                g.line(format!("let bq = {bq};"));
                g.line(format!(
                    "let dj = select(bq << 1u, bq, single) & {M24};   // limb {j} of D_q, below 2^14"
                ));
                for k in 0..lpl {
                    g.line(format!("{} += ap{k} * dj;", cross(j + k)));
                }
                g.depth -= 1;
                g.line("}");
            });
        }
        product_pass(g, lpl, &cross, &mut bound);
        let (b_low, b_high) = (bound.max(0..lpl), bound.max(lpl..words));
        // High half: the next lane's block, so it moves up one lane (rotating within the instance). It
        // belongs to the current `lo` block on every lane but lane 2 i, where `lo` is block 2 i and
        // the high half belongs to block 2 i + 1 + ... = lane + TPI, i.e. `hi`.
        g.line("let below = (lane + TPI - 1u) & (TPI - 1u);       // the lane whose high half lands here");
        g.line(format!("let to_hi = lane == {}u;", 2 * i));
        for k in 0..lpl {
            g.line(format!(
                "{{ let v = bcast(ph{k}, below, base); lo{k} += select(v, 0u, to_hi); hi{k} += select(0u, v, to_hi); }}"
            ));
        }
        acc = b_low.max(acc) + b_high;
        // Below 2^25: the window argument of phase 2 needs S_lo < 2^13 R (see the module comment).
        assert!(acc < 1 << 25, "output bound exceeded");
        g.depth -= 1;
        g.line("}");
    }
    // Undo the swaps: after iteration W/2 - 1 every lane below W - 2 has `lo` and `hi` exchanged.
    swap(g, lpl, &format!("lane < {}u", w - 2));
    // Upper diagonals: lane j squares block TPI/2 + j/2 and keeps the half that is its block TPI + j.
    g.line("{");
    g.depth += 1;
    all_words(g, lpl);
    g.line(format!(
        "let ai = {half}u + (lane >> 1u) + (all & params.z);"
    ));
    fetch(g, lpl, pairs, "dg", "dw", "ai");
    for c in 0..words {
        g.line(format!("var pw{c} = 0u;"));
    }
    let diag = |c: usize| format!("pw{c}");
    let mut bound = Bound(vec![0; words]);
    for j in 0..lpl {
        // Row j: the square at column 2 j, the doubled cross products at j + k for k > j.
        let mut cols = vec![(2 * j, M * M)];
        cols.extend((j + 1..lpl).map(|k| (j + k, 2 * M * M)));
        row(g, lpl, &diag, &mut bound, &cols, |g| {
            g.line("{");
            g.depth += 1;
            g.line(format!("pw{} += dg{j} * dg{j};", 2 * j));
            if j + 1 < lpl {
                g.line(format!("let dj = (dg{j} << 1u) & {M24};"));
                for k in j + 1..lpl {
                    g.line(format!("pw{} += dg{k} * dj;", j + k));
                }
            }
            g.depth -= 1;
            g.line("}");
        });
    }
    product_pass(g, lpl, &diag, &mut bound);
    g.line("let odd = (lane & 1u) == 1u;");
    for k in 0..lpl {
        g.line(format!("hi{k} += select(pw{k}, pw{}, odd);", lpl + k));
    }
    let hi = acc + bound.max(0..words);
    assert!(hi < 1 << 25, "output bound exceeded");
    g.depth -= 1;
    g.line("}");
    Phase1 { lo: acc, hi }
}

/// The largest number of reduction steps between light passes (and between the start of the reduction
/// and the first pass, and after the last pass) that keeps every word at most `LIMIT`, for a window
/// that starts from `lo` words below `lo` and has `hi` words below `hi` added at the end.
///
/// A step adds `n m` (at most `M^2`) to every word and the lane-0 carry (`(T0 + N0 m) >> 13`) to one;
/// `T0 + N0 m` itself must fit.
fn reduce_period(lo: u64, hi: u64) -> usize {
    let start = lo.max(AFTER_PASS);
    let mut b = start;
    let mut steps = 0;
    loop {
        // One more step: T0 + N0 m must fit, then every word gains n m and possibly carry0.
        if b + M * M > LIMIT {
            break;
        }
        let carry0 = (b + M * M) >> 13;
        let next = b + M * M + carry0;
        if next + hi > LIMIT {
            break;
        }
        b = next;
        steps += 1;
    }
    assert!(steps >= 1, "no reduction step fits");
    steps
}

/// Phase 2: reduces `lo` into `t0..` and adds `hi` (see the module comment). Light passes every
/// `period` steps, placed as in the fused loop (`unrolled::wgsl`).
fn phase2(g: &mut Gen, lpl: usize, in_body: bool, period: usize) {
    for c in 1..lpl {
        g.line(format!("let n{c} = n_limbs[off + {c}u];"));
    }
    for c in 0..lpl {
        g.line(format!("var t{c} = lo{c};"));
    }
    g.line("for (var owner = 0u; owner < TPI; owner++) {");
    g.depth += 1;
    for w in 0..lpl {
        g.line("{");
        g.depth += 1;
        g.line("let T0 = bcast(t0, 0u, base);               // window limb 0 of lane 0");
        g.line("let TN = down1(t0, lane);                   // next lane's window limb 0 (0 for the last lane)");
        g.line("let m = ((T0 & MASK) * ninv) & MASK;        // Montgomery factor: only the low 13 bits matter (24-bit multiply)");
        g.line(format!(
            "let carry0 = (T0 + (N0 & {M24}) * m) >> B;  // exact: the low 13 bits are zero"
        ));
        for c in 0..lpl - 1 {
            let u = c + 1;
            let carry = if c == 0 { " + (carry0 & is0)" } else { "" };
            g.line(format!("t{c} = t{u} + (n{u} & {M24}) * m{carry};"));
        }
        let top = lpl - 1;
        let top_carry = if lpl == 1 { " + (carry0 & is0)" } else { "" };
        g.line(format!("t{top} = TN + (NN0 & {M24}) * m{top_carry};"));
        g.depth -= 1;
        g.line("}");
        if in_body && ((w + 1) % period == 0 || (w + 1 == lpl && !lpl.is_multiple_of(period))) {
            light_pass(g, lpl as u32);
        }
    }
    if !in_body {
        g.line("if ((owner % K) == K - 1u) {");
        g.depth += 1;
        light_pass(g, lpl as u32);
        g.depth -= 1;
        g.line("}");
    }
    g.depth -= 1;
    g.line("}");
    g.line("// + S_hi: lane j's block TPI + j of x^2 has the weight of the result's block j");
    for c in 0..lpl {
        g.line(format!("t{c} += hi{c};"));
    }
    full_normalize(g, lpl as u32);
}
