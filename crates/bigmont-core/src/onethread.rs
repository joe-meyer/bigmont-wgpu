//! EXPERIMENTAL: one-thread-per-number Montgomery squaring kernels, for comparing limb schemes.
//!
//! Every limb of a number lives in one thread's registers as a named scalar (`x0..`, `n0..`,
//! `t0..`); the code is fully unrolled over limbs, and there are no subgroup operations at all. This
//! is not the cooperative (TPI >= 4) kernel the library uses; it exists to measure the arithmetic
//! of two limb schemes in isolation (see the `scheme_bench` example):
//!
//! - [`Scheme::Limb13`] (A): 13-bit limbs in u32, `L = limbs_for_bits(bits)`, R = 2^(13 L). The same
//!   arithmetic as `unrolled.rs` with TPI = 1: per step `bl = x_i & 0xffffff`, `x * bl` and `n * m`
//!   accumulated with `& 0xffffff` operand masks (24-bit multiply-add), `m = ((p & MASK) * ninv) &
//!   MASK` with `ninv` masked at load, the one-limb shift fused into the accumulate, a light carry
//!   pass after every 20 steps, and at the end of each squaring one sequential carry pass (with a
//!   single thread that pass normalises completely). Result in [0, 2n), every limb below 2^13.
//! - [`Scheme::Wide32`] (B): 32-bit limbs, u64 products (needs `SHADER_INT64`), `W = ceil((bits +
//!   2) / 32)`, R = 2^(32 W). Separated CIOS: per multiplier limb a pass `t += x * x_i` with a
//!   sequential u64 carry, then `m = t0 * ninv32` and a pass `t = (t + n * m) / 2^32`, then a final
//!   conditional subtraction of n. Result in [0, n).
//! - [`Scheme::Wide32Split`] (B2): as B but the two passes are fused per limb (FIOS): one loop over
//!   `j` carries both chains (`x * x_i` and `n * m`), so `t` is read and written once per multiplier
//!   limb and the top-word fold of B disappears. Result in [0, n).
//!
//! Buffer layout and bindings are those of the other squaring kernels: `params` uniform (x =
//! instance count, y = squarings), `n_limbs`, `ninv_buf` (one word per instance: -n^-1 mod 2^13 for
//! A, mod 2^32 for B) and `x_limbs` (in/out), limbs of an instance contiguous. Threads past the last
//! instance compute on the last instance and skip the store.
//!
//! Why B's words never overflow: inputs are below n (results of the previous squaring are reduced),
//! and with `t < 2n` at the start of a step, `(t + x x_i + m n) / 2^32 < (2n + 2^33 n) / 2^32 < 2n +
//! 1`, so `t` stays below 2n < R and fits W words between steps; `t + x x_i < 2^(32 (W + 1))` needs
//! one extra word (`tw`) during the step. Every `u64(a) + u64(b) * u64(c) + u64(d)` with 32-bit a, b,
//! c, d is at most 2^64 - 1.

use crate::gpu::{Batch, Gpu, Run, guarded};
use bigmont_oracle::{LIMB_BITS, limbs_for_bits, neg_inv_limb, neg_inv_u32, pack_limbs_n, pack_u32_limbs, unpack_limbs, unpack_u32_limbs};
use num_bigint::BigUint;
use std::time::Instant;

/// Steps (single-limb multiply-accumulates) between light carry passes in scheme A, as in
/// `unrolled.rs` (the overflow bound there tolerates 24).
const PERIOD: usize = 20;

/// Operand mask that steers AMD's compiler to the 24-bit multiply-add.
const M24: &str = "0xffffffu";

/// The limb scheme of a [`OneThread`] kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    /// A: 13-bit limbs, 24-bit multiply-add, deferred carries (the project's scheme, TPI = 1).
    Limb13,
    /// B: 32-bit limbs, u64 products, separated CIOS with sequential carries. Needs SHADER_INT64.
    Wide32,
    /// B2: 32-bit limbs, u64 products, fused (FIOS) passes. Needs SHADER_INT64.
    Wide32Split,
    /// A-sqr: scheme A's radix and result contract, but a dedicated squaring (cross terms once,
    /// doubled, plus the diagonal, then a word-by-word Montgomery reduction). Issue #2.
    Limb13Sqr,
    /// B-sqr: scheme B's radix and result contract with the same dedicated squaring. Needs
    /// SHADER_INT64. Issue #2.
    Wide32Sqr,
}

impl Scheme {
    pub fn name(self) -> &'static str {
        match self {
            Scheme::Limb13 => "A (13-bit)",
            Scheme::Wide32 => "B (32-bit, u64)",
            Scheme::Wide32Split => "B2 (32-bit, u64, FIOS)",
            Scheme::Limb13Sqr => "A-sqr (13-bit, symmetric squaring)",
            Scheme::Wide32Sqr => "B-sqr (32-bit, u64, symmetric squaring)",
        }
    }

    /// True for the 13-bit radix (A and A-sqr).
    pub fn is_limb13(self) -> bool {
        matches!(self, Scheme::Limb13 | Scheme::Limb13Sqr)
    }

    /// True for the dedicated-squaring kernels.
    pub fn is_sqr(self) -> bool {
        matches!(self, Scheme::Limb13Sqr | Scheme::Wide32Sqr)
    }

    /// The same-radix baseline that does not use squaring symmetry.
    pub fn baseline(self) -> Scheme {
        match self {
            Scheme::Limb13Sqr => Scheme::Limb13,
            Scheme::Wide32Sqr => Scheme::Wide32,
            other => other,
        }
    }

    /// Limbs per number at `bits` bits.
    pub fn limbs(self, bits: u32) -> usize {
        if self.is_limb13() { limbs_for_bits(bits) } else { (bits + 2).div_ceil(32) as usize }
    }

    /// Bits per limb.
    pub fn limb_bits(self) -> u32 {
        if self.is_limb13() { LIMB_BITS } else { 32 }
    }

    pub fn needs_int64(self) -> bool {
        !self.is_limb13()
    }
}

/// One kernel: scheme, operand size in bits (moduli below 2^bits) and workgroup size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneThread {
    pub scheme: Scheme,
    pub bits: u32,
    pub wg: u32,
}

impl OneThread {
    pub fn new(scheme: Scheme, bits: u32, wg: u32) -> Self {
        Self { scheme, bits, wg }
    }

    pub fn limbs(&self) -> usize {
        self.scheme.limbs(self.bits)
    }

    /// Montgomery radix exponent: R = 2^r_bits.
    pub fn r_bits(&self) -> u32 {
        self.scheme.limb_bits() * self.limbs() as u32
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.bits < 8 || self.bits > 4096 {
            return Err(format!("bits {} out of range 8..=4096", self.bits));
        }
        if self.wg == 0 || self.wg > 256 {
            return Err(format!("workgroup size {} must be 1..=256", self.wg));
        }
        Ok(())
    }

    /// Packs (n, x) pairs (n odd, below 2^bits; x below n for scheme B, below 2n for A).
    pub fn batch(&self, values: &[(BigUint, BigUint)]) -> Batch {
        let limbs = self.limbs();
        let mut b = Batch { count: values.len() as u32, limbs, n: Vec::new(), ninv: Vec::new(), x: Vec::new() };
        for (n, x) in values {
            assert!(n.bits() <= u64::from(self.bits), "modulus wider than {} bits", self.bits);
            match self.scheme {
                Scheme::Limb13 | Scheme::Limb13Sqr => {
                    assert!(x < &(n * 2u32), "x must be below 2n");
                    b.n.extend(pack_limbs_n(n, limbs));
                    b.ninv.push(neg_inv_limb(n));
                    b.x.extend(pack_limbs_n(x, limbs));
                }
                Scheme::Wide32 | Scheme::Wide32Split | Scheme::Wide32Sqr => {
                    assert!(x < n, "x must be below n");
                    b.n.extend(pack_u32_limbs(n, limbs));
                    b.ninv.push(neg_inv_u32(n));
                    b.x.extend(pack_u32_limbs(x, limbs));
                }
            }
        }
        b
    }

    /// The value held by one instance's output limbs.
    pub fn decode(&self, limbs: &[u32]) -> BigUint {
        if self.scheme.is_limb13() { unpack_limbs(limbs) } else { unpack_u32_limbs(limbs) }
    }

    /// The complete WGSL source (native wgpu; scheme B needs a device with SHADER_INT64).
    pub fn wgsl(&self) -> String {
        let mut g = Gen { lines: Vec::new(), depth: 0 };
        let l = self.limbs();
        assert!(l >= 2, "at least two limbs");
        g.line(format!("// Generated by bigmont-core onethread.rs: scheme {}, {} bits, {l} limbs, one thread per number.", self.scheme.name(), self.bits));
        g.line(format!("const L: u32 = {l}u;"));
        g.line(format!("const WG: u32 = {}u;", self.wg));
        if self.scheme.is_limb13() {
            g.line("const B: u32 = 13u;");
            g.line("const MASK: u32 = 0x1fffu;");
        }
        g.line("");
        g.line("@group(0) @binding(0) var<uniform> params: vec4<u32>;          // x = instance count, y = squarings");
        g.line("@group(0) @binding(1) var<storage, read> n_limbs: array<u32>;");
        g.line("@group(0) @binding(2) var<storage, read> ninv_buf: array<u32>;");
        g.line("@group(0) @binding(3) var<storage, read_write> x_limbs: array<u32>;");
        g.line("");
        g.line("@compute @workgroup_size(WG)");
        g.line("fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {");
        g.depth += 1;
        g.line("let inst = wid.x * WG + li;");
        g.line("let live = inst < params.x;");
        g.line("let ic = min(inst, params.x - 1u);     // out-of-range threads compute on a valid instance; their store is skipped");
        g.line("let off = ic * L;");
        for c in 0..l {
            g.line(format!("var x{c} = x_limbs[off + {c}u];"));
        }
        for c in 0..l {
            g.line(format!("let n{c} = n_limbs[off + {c}u];"));
        }
        // The squaring kernels hold a 2L-word product and declare it inside the loop.
        if !self.scheme.is_sqr() {
            for c in 0..l {
                g.line(format!("var t{c} = 0u;"));
            }
        }
        if self.scheme.is_limb13() {
            g.line("let ninv = ninv_buf[ic] & MASK;   // already below 2^13; the mask lets the compiler see the m product fits 24 bits");
        } else {
            g.line("let ninv = ninv_buf[ic];            // -n^-1 mod 2^32");
        }
        g.line("for (var s = 0u; s < params.y; s++) {");
        g.depth += 1;
        if !self.scheme.is_sqr() {
            for c in 0..l {
                g.line(format!("t{c} = 0u;"));
            }
        }
        match self.scheme {
            Scheme::Limb13 => limb13_squaring(&mut g, l),
            Scheme::Wide32 => wide32_squaring(&mut g, l, false),
            Scheme::Wide32Split => wide32_squaring(&mut g, l, true),
            Scheme::Limb13Sqr => limb13_symmetric_squaring(&mut g, l),
            Scheme::Wide32Sqr => wide32_symmetric_squaring(&mut g, l),
        }
        g.depth -= 1;
        g.line("}");
        g.line("if (live) {");
        g.depth += 1;
        for c in 0..l {
            g.line(format!("x_limbs[off + {c}u] = x{c};"));
        }
        g.depth -= 1;
        g.line("}");
        g.depth -= 1;
        g.line("}");
        g.lines.join("\n") + "\n"
    }
}

struct Gen {
    lines: Vec<String>,
    depth: usize,
}

impl Gen {
    fn line(&mut self, s: impl AsRef<str>) {
        self.lines.push(format!("{}{}", "    ".repeat(self.depth), s.as_ref()));
    }
}

/// Scheme A: `unrolled.rs` with TPI = 1 (so `bcast` is the identity, `down1` and `up1` are 0 and
/// lane 0 owns every limb). Light passes after steps 20, 40, ... (not after the last step, which the
/// final sequential pass covers), so at most 20 steps run between passes.
fn limb13_squaring(g: &mut Gen, l: usize) {
    for w in 0..l {
        g.line("{");
        g.depth += 1;
        g.line(format!("let bl = x{w} & {M24};                      // multiplier limb {w}"));
        g.line(format!("let p = t0 + (x0 & {M24}) * bl;"));
        g.line("let m = ((p & MASK) * ninv) & MASK;        // Montgomery factor (24-bit multiply)");
        g.line(format!("let carry0 = (p + (n0 & {M24}) * m) >> B;  // exact: the low 13 bits are zero"));
        for c in 0..l - 1 {
            let u = c + 1;
            let carry = if c == 0 { " + carry0" } else { "" };
            g.line(format!("t{c} = t{u} + (x{u} & {M24}) * bl + (n{u} & {M24}) * m{carry};"));
        }
        g.line(format!("t{} = 0u;", l - 1));
        g.depth -= 1;
        g.line("}");
        if (w + 1) % PERIOD == 0 && w + 1 < l {
            // Light pass: each limb keeps its low 13 bits plus the carry of the old limb below it.
            g.line("{");
            g.depth += 1;
            g.line("var c_in = 0u;");
            for c in 0..l {
                g.line(format!("let v{c} = t{c};"));
                g.line(format!("t{c} = (v{c} & MASK) + c_in;"));
                g.line(format!("c_in = v{c} >> B;"));
            }
            g.depth -= 1;
            g.line("}");
        }
    }
    // Sequential carry pass: with one thread this normalises fully. The value is below 2n < R, so
    // the carry out of the top limb is zero (the top limb is stored unmasked so a bug would show).
    g.line("{");
    g.depth += 1;
    g.line("var c = 0u;");
    for c in 0..l {
        if c + 1 < l {
            g.line(format!("let v{c} = t{c} + c;"));
            g.line(format!("x{c} = v{c} & MASK;"));
            g.line(format!("c = v{c} >> B;"));
        } else {
            g.line(format!("x{c} = t{c} + c;"));
        }
    }
    g.depth -= 1;
    g.line("}");
}

/// Schemes B (`fused` false: separated CIOS) and B2 (`fused` true: FIOS), then the conditional
/// subtraction of n.
fn wide32_squaring(g: &mut Gen, w: usize, fused: bool) {
    for i in 0..w {
        g.line("{");
        g.depth += 1;
        g.line(format!("let xi = u64(x{i});                         // multiplier limb {i}"));
        g.line("var a: u64;");
        if !fused {
            g.line("var c = 0lu;");
            for j in 0..w {
                g.line(format!("a = u64(t{j}) + u64(x{j}) * xi + c;"));
                g.line(format!("t{j} = u32(a);"));
                g.line("c = a >> 32u;");
            }
            g.line("let tw = u32(c);                           // t < 2n < R between steps, so the top word was 0");
            g.line("let m = t0 * ninv;                         // u32 wrapping: t + m n = 0 mod 2^32");
            g.line("let mm = u64(m);");
            g.line("a = u64(t0) + u64(n0) * mm;");
            g.line("c = a >> 32u;                              // low word is zero");
            for j in 1..w {
                g.line(format!("a = u64(t{j}) + u64(n{j}) * mm + c;"));
                g.line(format!("t{} = u32(a);", j - 1));
                g.line("c = a >> 32u;");
            }
            g.line(format!("t{} = tw + u32(c);                         // no carry out: t < 2n < R", w - 1));
        } else {
            g.line("var b: u64;");
            g.line("a = u64(t0) + u64(x0) * xi;");
            g.line("var c1 = a >> 32u;");
            g.line("let m = u32(a) * ninv;                     // u32 wrapping: t + x x_i + m n = 0 mod 2^32");
            g.line("let mm = u64(m);");
            g.line("b = u64(u32(a)) + u64(n0) * mm;");
            g.line("var c2 = b >> 32u;                         // low word is zero");
            for j in 1..w {
                g.line(format!("a = u64(t{j}) + u64(x{j}) * xi + c1;"));
                g.line("c1 = a >> 32u;");
                g.line(format!("b = u64(u32(a)) + u64(n{j}) * mm + c2;"));
                g.line(format!("t{} = u32(b);", j - 1));
                g.line("c2 = b >> 32u;");
            }
            g.line(format!("t{} = u32(c1 + c2);                       // no carry out: t < 2n < R", w - 1));
        }
        g.depth -= 1;
        g.line("}");
    }
    // t < 2n: subtract n once if t >= n.
    g.line("{");
    g.depth += 1;
    g.line("var bw = 0lu;");
    g.line("var d: u64;");
    for j in 0..w {
        g.line(format!("d = u64(t{j}) - u64(n{j}) - bw;"));
        g.line(format!("let d{j} = u32(d);"));
        g.line("bw = d >> 63u;                             // 1 when this word borrowed");
    }
    g.line("let keep = bw == 1lu;                      // t < n: keep t");
    for j in 0..w {
        g.line(format!("x{j} = select(d{j}, t{j}, keep);"));
    }
    g.depth -= 1;
    g.line("}");
}

/// Rows of cross products between light carry passes in A-sqr. Each row adds one product (below
/// 2^13 * 2^14 = 2^27, the multiplicand is the doubled limb) to a column, so a column gains at most
/// `PERIOD_CROSS` * 2^27 between passes.
const PERIOD_CROSS: usize = 20;

/// Reduction rows between light carry passes in A-sqr. Each row adds one product `n_j * m` (below
/// 2^26) to a column.
const PERIOD_REDUCE: usize = 40;

/// A light carry pass over columns `lo..=hi`: each keeps its low 13 bits plus the carry of the column
/// below, and the carry out of `hi` goes to column `hi + 1`.
fn light_pass_window(g: &mut Gen, lo: usize, hi: usize) {
    g.line("{");
    g.depth += 1;
    g.line("var c_in = 0u;");
    for k in lo..=hi {
        g.line(format!("let v{k} = t{k};"));
        g.line(format!("t{k} = (v{k} & MASK) + c_in;"));
        g.line(format!("c_in = v{k} >> B;"));
    }
    g.line(format!("t{} = t{} + c_in;", hi + 1, hi + 1));
    g.depth -= 1;
    g.line("}");
}

/// Scheme A-sqr: a dedicated Montgomery squaring in the 13-bit radix, with the same limb contract as
/// [`limb13_squaring`] (input limbs below 2^13 and value below 2n, output limbs below 2^13 and value
/// below 2n, R = 2^(13 L) > 16 n).
///
/// 1. Cross terms `x_i * 2 x_j` (i < j) are accumulated by row into the 2L columns `t0..`. The
///    multiplicand is the doubled limb `xd_j = 2 x_j` (below 2^14), so every product is below
///    2^13 * 2^14 = 2^27 and the doubling of the cross sum costs nothing. A row adds one product to
///    each column it touches, so with a light carry pass every `PERIOD_CROSS` = 20 rows a column holds
///    at most 2^19 + 2^13 (left by the last pass) + 20 * 2^27 ~= 2.68e9 < 2^32.
/// 2. The diagonal `x_i^2` (below 2^26) is added to column `2 i`, then one sequential carry pass over
///    all 2L columns normalises every limb below 2^13 (a column is at most ~2.75e9 plus a carry below
///    2^20, so no u32 overflow). The value is x^2 < 4 n^2 < R^2 / 4, so nothing leaves the top limb.
/// 3. Word-by-word reduction: row `i` takes `m = ((t_i & MASK) * ninv) & MASK` from column `i`
///    (its low 13 bits; the rest is deferred carry, passed on as `carry0`) and adds `n_j * m` to
///    column `i + j`. Each row adds one product (below 2^26) to a column, so with a light pass every
///    `PERIOD_REDUCE` = 40 rows a column is at most 2^19 + 2^13 + 40 * 2^26 ~= 2.68e9, and `p +
///    n0 * m` at most 2.75e9, both below 2^32. Columns at or below the current row are dead (their
///    carry went up through `carry0`), so a pass only covers the live window `i + 1 ..= i + L - 1`.
/// 4. A final sequential pass over columns `L .. 2L` gives the normalised result
///    `(x^2 + M n) / R < (4 n^2 + R n) / R < 2 n`.
fn limb13_symmetric_squaring(g: &mut Gen, l: usize) {
    for k in 0..2 * l {
        g.line(format!("var t{k} = 0u;"));
    }
    // 1. cross terms, doubled by using 2 x_j as the multiplicand
    for j in 1..l {
        g.line(format!("let xd{j} = (x{j} << 1u) & {M24};            // 2 x_{j}, below 2^14"));
    }
    let mut group = 0usize; // first row since the last light pass
    for i in 0..l - 1 {
        g.line("{");
        g.depth += 1;
        g.line(format!("let xi = x{i} & {M24};"));
        for j in i + 1..l {
            g.line(format!("t{} = t{} + xi * xd{j};", i + j, i + j));
        }
        g.depth -= 1;
        g.line("}");
        if i + 1 - group == PERIOD_CROSS && i + 1 < l - 1 {
            // Rows `group..=i` touched columns `2 group + 1 ..= i + L - 1`.
            light_pass_window(g, 2 * group + 1, i + l - 1);
            group = i + 1;
        }
    }
    // 2. diagonal, then one sequential pass
    for i in 0..l {
        g.line(format!("t{} = t{} + (x{i} & {M24}) * (x{i} & {M24});", 2 * i, 2 * i));
    }
    sequential_pass(g, 0, 2 * l, None);
    // 3. word-by-word reduction
    let mut group = 0usize;
    for i in 0..l {
        g.line("{");
        g.depth += 1;
        g.line(format!("let p = t{i};"));
        g.line("let m = ((p & MASK) * ninv) & MASK;        // Montgomery factor (24-bit multiply)");
        g.line(format!("let carry0 = (p + (n0 & {M24}) * m) >> B;  // exact: the low 13 bits are zero"));
        g.line(format!("t{} = t{} + (n1 & {M24}) * m + carry0;", i + 1, i + 1));
        for j in 2..l {
            g.line(format!("t{} = t{} + (n{j} & {M24}) * m;", i + j, i + j));
        }
        g.depth -= 1;
        g.line("}");
        if i + 1 - group == PERIOD_REDUCE && i + 1 < l {
            light_pass_window(g, i + 1, i + l - 1);
            group = i + 1;
        }
    }
    // 4. normalised result from the upper half
    sequential_pass(g, l, 2 * l, Some("x"));
}

/// Sequential carry pass over columns `lo..hi`: each column becomes `(t + carry) & MASK` and the carry
/// moves up; the top column keeps the unmasked sum (a bug would show). With `out` set the result goes
/// to `{out}{k - lo}` instead of back into `t{k}`.
fn sequential_pass(g: &mut Gen, lo: usize, hi: usize, out: Option<&str>) {
    g.line("{");
    g.depth += 1;
    g.line("var c = 0u;");
    for k in lo..hi {
        let dst = match out {
            Some(name) => format!("{name}{}", k - lo),
            None => format!("t{k}"),
        };
        if k + 1 < hi {
            g.line(format!("let v{k} = t{k} + c;"));
            g.line(format!("{dst} = v{k} & MASK;"));
            g.line(format!("c = v{k} >> B;"));
        } else {
            g.line(format!("{dst} = t{k} + c;"));
        }
    }
    g.depth -= 1;
    g.line("}");
}

/// Scheme B-sqr: a dedicated Montgomery squaring in 32-bit limbs with u64 products, same contract as
/// scheme B (input value below n, output value below n, R = 2^(32 W) > 4 n).
///
/// 1. Cross terms `x_i x_j` (i < j) by row with a sequential u64 carry into the 2W words `t`
///    (W (W - 1) / 2 products). Row `i` ends by writing its carry to `t[i + W]`, a word no earlier
///    row has touched.
/// 2. The cross sum is doubled and the diagonal `x_i^2` added in one pass (`d_k = (t_k << 1) | the bit
///    shifted out of t_{k-1}`, plus the low or high half of `x_{k/2}^2`, plus the carry). The sum is
///    `x^2 < 2^(64 W)`, so neither the final carry nor the shifted-out bit is ever set.
/// 3. Word-by-word reduction (W^2 products): row `i` adds `m n`, `m = t_i * ninv` (u32 wrapping), at
///    offset `i`; the chain carry plus the carry bit `e` of the previous row go into `t[i + W]`. Every
///    sum `t + n_j m + c` is at most 2^64 - 1. The total `x^2 + M n < n^2 + R n < 2^(64 W)`, so `e`
///    is zero after the last row.
/// 4. The upper half is `(x^2 + M n) / R < n + n^2 / R < 2 n`: one conditional subtraction of n, as
///    in scheme B.
fn wide32_symmetric_squaring(g: &mut Gen, w: usize) {
    for k in 0..2 * w {
        g.line(format!("var t{k} = 0u;"));
    }
    // 1. cross terms
    for i in 0..w - 1 {
        g.line("{");
        g.depth += 1;
        g.line(format!("let xi = u64(x{i});"));
        g.line("var a: u64;");
        g.line("var c = 0lu;");
        for j in i + 1..w {
            g.line(format!("a = u64(t{}) + xi * u64(x{j}) + c;", i + j));
            g.line(format!("t{} = u32(a);", i + j));
            g.line("c = a >> 32u;");
        }
        g.line(format!("t{} = u32(c);", i + w));
        g.depth -= 1;
        g.line("}");
    }
    // 2. double and add the diagonal
    for i in 0..w {
        g.line(format!("let sq{i} = u64(x{i}) * u64(x{i});"));
    }
    g.line("{");
    g.depth += 1;
    g.line("var a: u64;");
    g.line("var c = 0lu;");
    g.line("var prev = 0u;                                     // bit shifted out of the word below");
    for k in 0..2 * w {
        let half = if k % 2 == 0 { format!("u64(u32(sq{}))", k / 2) } else { format!("(sq{} >> 32u)", k / 2) };
        g.line(format!("let d{k} = (t{k} << 1u) | prev;"));
        g.line(format!("prev = t{k} >> 31u;"));
        g.line(format!("a = u64(d{k}) + {half} + c;"));
        g.line(format!("t{k} = u32(a);"));
        g.line("c = a >> 32u;");
    }
    g.depth -= 1;
    g.line("}");
    // 3. reduction
    g.line("{");
    g.depth += 1;
    g.line("var a: u64;");
    g.line("var e = 0lu;                                       // carry bit into the next row's top word");
    for i in 0..w {
        g.line("{");
        g.depth += 1;
        g.line(format!("let m = t{i} * ninv;                         // u32 wrapping: t + m n = 0 mod 2^32"));
        g.line("let mm = u64(m);");
        g.line(format!("a = u64(t{i}) + u64(n0) * mm;"));
        g.line("var c = a >> 32u;                              // low word is zero");
        for j in 1..w {
            g.line(format!("a = u64(t{}) + u64(n{j}) * mm + c;", i + j));
            g.line(format!("t{} = u32(a);", i + j));
            g.line("c = a >> 32u;");
        }
        g.line(format!("a = u64(t{}) + c + e;", i + w));
        g.line(format!("t{} = u32(a);", i + w));
        g.line("e = a >> 32u;");
        g.depth -= 1;
        g.line("}");
    }
    g.depth -= 1;
    g.line("}");
    // 4. t[W..2W) < 2n: subtract n once if it is >= n.
    g.line("{");
    g.depth += 1;
    g.line("var bw = 0lu;");
    g.line("var d: u64;");
    for j in 0..w {
        g.line(format!("d = u64(t{}) - u64(n{j}) - bw;", w + j));
        g.line(format!("let d{j} = u32(d);"));
        g.line("bw = d >> 63u;                             // 1 when this word borrowed");
    }
    g.line("let keep = bw == 1lu;                      // t < n: keep t");
    for j in 0..w {
        g.line(format!("x{j} = select(d{j}, t{}, keep);", w + j));
    }
    g.depth -= 1;
    g.line("}");
}

/// Persistent buffers and pipeline for one kernel and batch, so a benchmark can time many
/// dispatches without recompiling.
pub struct Session<'a> {
    gpu: &'a Gpu,
    kernel: OneThread,
    batch_x: Vec<u32>,
    count: u32,
    groups: u32,
    params: wgpu::Buffer,
    x_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    words: u64,
}

impl<'a> Session<'a> {
    pub fn new(gpu: &'a Gpu, kernel: &OneThread, batch: &Batch) -> Result<Session<'a>, String> {
        kernel.validate()?;
        let count = batch.count as usize;
        let limbs = kernel.limbs();
        if count == 0 {
            return Err("batch is empty".into());
        }
        if batch.limbs != limbs || batch.x.len() != count * limbs || batch.n.len() != count * limbs || batch.ninv.len() != count {
            return Err(format!("batch does not match {kernel:?} ({limbs} limbs, {count} instances)"));
        }
        if kernel.scheme.needs_int64() && !gpu.has_int64 {
            return Err(format!("{kernel:?} needs SHADER_INT64, which this device lacks"));
        }
        let groups = batch.count.div_ceil(kernel.wg);
        if groups > 65535 {
            return Err(format!("{groups} workgroups exceeds the 65535 per dispatch limit"));
        }
        let (device, queue) = (&gpu.device, &gpu.queue);
        let words = (count * limbs) as u64;
        let (params, n_buf, ninv_buf, x_buf, staging) = pollster::block_on(guarded(device, "buffer creation", || {
            let storage = |label: &str, size: u64, extra: wgpu::BufferUsages| {
                device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra, mapped_at_creation: false })
            };
            let params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("params"), size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let n_buf = storage("n", words * 4, wgpu::BufferUsages::empty());
            let ninv_buf = storage("ninv", u64::from(batch.count) * 4, wgpu::BufferUsages::empty());
            let x_buf = storage("x", words * 4, wgpu::BufferUsages::COPY_SRC);
            let staging = device.create_buffer(&wgpu::BufferDescriptor { label: Some("staging"), size: words * 4, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            queue.write_buffer(&n_buf, 0, bytemuck::cast_slice(&batch.n));
            queue.write_buffer(&ninv_buf, 0, bytemuck::cast_slice(&batch.ninv));
            queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(&batch.x));
            (params, n_buf, ninv_buf, x_buf, staging)
        }))?;
        let module = pollster::block_on(guarded(device, &format!("shader module for {kernel:?}"), || {
            device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("onethread"), source: wgpu::ShaderSource::Wgsl(kernel.wgsl().into()) })
        }))?;
        let pipeline = pollster::block_on(guarded(device, &format!("compute pipeline for {kernel:?}"), || {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("onethread"), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
        }))?;
        let bind_group = pollster::block_on(guarded(device, "bind group creation", || {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: n_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ninv_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: x_buf.as_entire_binding() },
                ],
            })
        }))?;
        Ok(Session { gpu, kernel: *kernel, batch_x: batch.x.clone(), count: batch.count, groups, params, x_buf, staging, pipeline, bind_group, words })
    }

    pub fn kernel(&self) -> OneThread {
        self.kernel
    }

    /// Restores the batch's starting values.
    pub fn reset(&self) {
        self.gpu.queue.write_buffer(&self.x_buf, 0, bytemuck::cast_slice(&self.batch_x));
    }

    /// Runs `repeats` dispatches of `squarings` squarings each (each continues from the previous
    /// output) and returns the best wall-clock time of one dispatch (submit plus blocking poll).
    pub fn time(&self, squarings: u32, repeats: u32) -> Result<f64, String> {
        let (device, queue) = (&self.gpu.device, &self.gpu.queue);
        queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&[self.count, squarings, 0u32, 0u32]));
        let mut best = f64::MAX;
        for _ in 0..repeats.max(1) {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(self.groups, 1, 1);
            }
            let t = Instant::now();
            queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll failed: {e}"))?;
            best = best.min(t.elapsed().as_secs_f64());
        }
        Ok(best)
    }

    /// Reads the current x limbs back.
    pub fn read(&self) -> Result<Vec<u32>, String> {
        let (device, queue) = (&self.gpu.device, &self.gpu.queue);
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.x_buf, 0, &self.staging, 0, self.words * 4);
        queue.submit([enc.finish()]);
        let slice = self.staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("mapping readback buffer failed"));
        device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll failed: {e}"))?;
        let x = {
            let view = slice.get_mapped_range().expect("readback buffer not mapped");
            bytemuck::cast_slice::<u8, u32>(&view).to_vec()
        };
        self.staging.unmap();
        Ok(x)
    }
}

/// Uploads `batch`, runs `repeats` chained dispatches of `squarings` squarings and reads back.
pub fn run(gpu: &Gpu, kernel: &OneThread, batch: &Batch, squarings: u32, repeats: u32) -> Result<Run, String> {
    let session = Session::new(gpu, kernel, batch)?;
    let best_secs = session.time(squarings, repeats)?;
    Ok(Run { x: session.read()?, best_secs })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(src: &str, int64: bool) -> Result<(), String> {
        let module = naga::front::wgsl::parse_str(src).map_err(|e| e.emit_to_string(src))?;
        let caps = if int64 { naga::valid::Capabilities::SHADER_INT64 } else { naga::valid::Capabilities::empty() };
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps).validate(&module).map_err(|e| format!("{e:?}"))?;
        Ok(())
    }

    const SCHEMES: [Scheme; 5] = [Scheme::Limb13, Scheme::Wide32, Scheme::Wide32Split, Scheme::Limb13Sqr, Scheme::Wide32Sqr];

    #[test]
    fn every_kernel_validates_in_naga() {
        for scheme in SCHEMES {
            for bits in [64, 130, 256, 384, 512, 1024] {
                for wg in [64, 256] {
                    let k = OneThread::new(scheme, bits, wg);
                    validate(&k.wgsl(), scheme.needs_int64()).unwrap_or_else(|e| panic!("{k:?}: {e}"));
                }
            }
        }
    }

    #[test]
    fn wide_kernels_need_the_int64_capability_and_the_13_bit_kernel_does_not() {
        assert!(validate(&OneThread::new(Scheme::Wide32, 256, 64).wgsl(), false).is_err());
        assert!(validate(&OneThread::new(Scheme::Wide32Split, 256, 64).wgsl(), false).is_err());
        let a = OneThread::new(Scheme::Limb13, 256, 64).wgsl();
        validate(&a, false).unwrap();
        assert!(!a.contains("subgroup") && !a.contains("u64"));
    }

    #[test]
    fn limb_counts_and_radices() {
        let l = |s: Scheme, b| OneThread::new(s, b, 64).limbs();
        assert_eq!([l(Scheme::Limb13, 256), l(Scheme::Limb13, 384), l(Scheme::Limb13, 512), l(Scheme::Limb13, 1024)], [20, 30, 40, 80]);
        assert_eq!([l(Scheme::Wide32, 256), l(Scheme::Wide32, 384), l(Scheme::Wide32, 512), l(Scheme::Wide32, 1024)], [9, 13, 17, 33]);
        for bits in [64, 130, 256, 384, 512, 1024] {
            for s in SCHEMES {
                // R > 4n for every n below 2^bits.
                assert!(OneThread::new(s, bits, 64).r_bits() >= bits + 2, "{s:?} {bits}");
            }
        }
    }

    /// At most PERIOD steps between light passes (or between the squaring start and the first pass,
    /// or between the last pass and the final sequential pass).
    #[test]
    fn limb13_light_passes_every_20_steps() {
        for bits in [64, 130, 256, 384, 512, 1024] {
            let src = OneThread::new(Scheme::Limb13, bits, 64).wgsl();
            let mut run = 0usize;
            let mut max = 0usize;
            for line in src.lines() {
                if line.contains("let bl =") {
                    run += 1;
                    max = max.max(run);
                } else if line.contains("var c_in = 0u;") || line.contains("var c = 0u;") {
                    run = 0;
                }
            }
            assert!(max <= PERIOD && max > 0, "{bits}: {max} steps between passes");
        }
    }
}
