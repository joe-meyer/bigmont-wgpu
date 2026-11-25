//! Fully unrolled straight-line WGSL generator for the cooperative kernels (TPI 4, 8, 16, 32, any limb count).
//!
//! The rolled kernel (`mont.wgsl`) keeps limbs in private arrays and loops over them; the AMD driver
//! does not unroll those loops. This generator instead emits one `main` whose limb values are named
//! scalar locals (`x0..`, `n0..`, `t0..`), with the one-limb shift of the Montgomery step folded into
//! the accumulate, so there are no limb arrays and no loops over limbs. Only the loop over the owning
//! lane (`owner`), the squaring loop and the normalisation rounds stay as (uniform) loops.
//!
//! Two modes share the generator ([`Mode`]):
//! - Squarings: same bindings, entry point and arithmetic contract as the rolled kernel: per
//!   instance `x <- x^2 * R^-1 mod n` (R = 2^(13 * limbs), result in [0, 2n), limbs fully
//!   normalised), `params.y` times.
//! - Fermat: one left-to-right binary ladder step per iteration, for the base-2 Fermat test. After
//!   each squaring the exponent bit `i = fp.y - 2 - (fp.x + s)` of `e_words` (`n - 1`, `fp.z` words
//!   per instance) is read, `t = 2x` is formed limbwise and fully normalised, and
//!   `x = select(x, t, bit == 1)` per limb. The doubling is branch-free on purpose: instances of one
//!   subgroup take different bits, and the subgroup shuffles and `subgroupAny` in the next squaring
//!   (and in the normalisation of `t` itself) must stay in uniform control flow, so every lane
//!   always computes `t` and the bit only picks the result.
//!
//!   This needs R > 16n (`KernelConfig::for_bits` guarantees it: `13 * limbs >= bits + 4`). After a
//!   doubling x < 4n, and a Montgomery squaring of x < 4n gives `(x^2 + m n) / R < 16n^2 / R + n`,
//!   which is below 2n exactly when R > 16n; so x stays below 2n after every squaring and below 4n
//!   (far below R) after every doubling, and no carry ever leaves the top limb.

use crate::kernel::{KernelConfig, SUBGROUP_HELPERS, Target};

/// Iterations (single-limb multiply-accumulate steps) between light carry passes, for every width.
/// (The bound argument in `wgsl` tolerates up to 24; 20 is used so all widths behave the same.)
const PERIOD: u32 = 20;

/// Operand mask that steers AMD's compiler to the 24-bit multiply-add.
const M24: &str = "0xffffffu";

/// Which kernel the generator emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// `params.y` Montgomery squarings per dispatch.
    Squarings,
    /// `params.y` base-2 Fermat ladder steps per dispatch (squaring, then conditional doubling).
    Fermat,
}

/// Emits the complete unrolled WGSL source for `cfg` (`cfg.tpi` must be 4, 8, 16 or 32).
///
/// Where the light carry passes go. At most `PERIOD` = 20 single-limb steps ever run between two
/// passes (or between the start of a squaring and the first pass), at every width and limb count:
/// - `LPL` > `PERIOD`: a pass is emitted INSIDE the unrolled owner body after every `PERIOD`-th step,
///   plus an extra end-of-owner pass after the last step when `LPL % PERIOD != 0` (the remainder is
///   shorter than `PERIOD`), so every owner iteration ends with a pass.
/// - `LPL` <= `PERIOD`: a runtime `(owner % K) == K - 1` pass with `K = PERIOD / LPL` (floor, at
///   least 1), so a pass follows every `K` owner iterations, `K * LPL <= PERIOD` steps. `K` need not
///   divide `PERIOD` or `TPI`: the full normalisation at the end of each squaring resets the limbs,
///   and at most `K - 1` owner iterations (fewer than `PERIOD` steps) run after the last pass.
///
/// Why 32-bit accumulators never overflow: limbs entering a step (`x`, `n`) are below 2^13, so every
/// product is below 2^26. Each accumulator word gains at most two products (below 2^27) per step,
/// plus `carry0` (below 2^19) on lane 0's limb 0. After a light pass every limb equals
/// `(v & MASK) + c_in`, where `v & MASK` is below 2^13 and `c_in = v_below >> 13` is below 2^19 (the
/// limb below it was under 2^32), so it is below 2^13 + 2^19. With at most 20 steps between passes a limb
/// stays below about 2^19 + 2^13 + 20 * (2^27 + 2^19) ~= 2.70e9 < 2^32 (24 steps would give about
/// 3.23e9, still below 2^32). `p` and `p + N0 * m` are limb-0 values after one and two products and
/// obey the same bound. The value stays below 2n because R > 4n (as in the rolled kernel), so the
/// discarded carry out of the top lane's top limb is zero.
///
/// The Montgomery factor is `m = ((p & MASK) * ninv) & MASK`: only the low 13 bits of `p * ninv`
/// matter, and `(p & MASK) * ninv` is below 2^26 (both factors below 2^13), so it is exact and can
/// use the 24-bit multiply instead of a full 32-bit low multiply. `ninv` is masked once when it is
/// loaded (it is already below 2^13): without that mask the compiler cannot see that the second
/// factor is small and still emits a full 32-bit low multiply (measured in the ISA).
pub(crate) fn wgsl(cfg: &KernelConfig, target: Target, mode: Mode) -> String {
    assert!(
        matches!(cfg.tpi, 4 | 8 | 16 | 32),
        "unrolled generator supports tpi 4, 8, 16, 32 only"
    );
    let lpl = cfg.limbs_per_lane();
    // LPL > PERIOD: passes are emitted inside the body and every owner iteration ends with one.
    // Otherwise one pass every K owner iterations, K = floor(PERIOD / LPL) >= 1.
    let in_body = lpl > PERIOD;
    let k = if in_body { 1 } else { PERIOD / lpl };

    let mut g = Gen {
        lines: Vec::new(),
        depth: 0,
    };
    header(&mut g, cfg, target, mode, lpl, k, in_body);
    g.raw(SUBGROUP_HELPERS);
    g.line("@compute @workgroup_size(WG)");
    g.line("fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) sg_in: u32) {");
    g.depth += 1;
    prologue(&mut g);
    load(&mut g, lpl);
    hoists(&mut g);
    g.line("for (var s = 0u; s < params.y; s++) {");
    g.depth += 1;
    squaring_start(&mut g, lpl);
    g.line("for (var owner = 0u; owner < TPI; owner++) {");
    g.depth += 1;
    for w in 0..lpl {
        iteration(&mut g, lpl, w);
        // LPL > PERIOD: a pass every PERIOD steps inside the body, plus an extra end-of-owner pass
        // after the last step when LPL % PERIOD != 0 (otherwise the first rule already covers it).
        if in_body && ((w + 1) % PERIOD == 0 || (w + 1 == lpl && !lpl.is_multiple_of(PERIOD))) {
            light_pass(&mut g, lpl);
        }
    }
    if !in_body {
        g.line("if ((owner % K) == K - 1u) {");
        g.depth += 1;
        light_pass(&mut g, lpl);
        g.depth -= 1;
        g.line("}");
    }
    g.depth -= 1;
    g.line("}");
    full_normalize(&mut g, lpl);
    for c in 0..lpl {
        g.line(format!("x{c} = t{c};"));
    }
    if mode == Mode::Fermat {
        conditional_double(&mut g, lpl);
    }
    g.depth -= 1;
    g.line("}");
    store(&mut g, lpl);
    g.depth -= 1;
    g.line("}");
    g.lines.join("\n") + "\n"
}

/// Line collector with indentation.
struct Gen {
    lines: Vec<String>,
    depth: usize,
}

impl Gen {
    fn line(&mut self, s: impl AsRef<str>) {
        self.lines
            .push(format!("{}{}", "    ".repeat(self.depth), s.as_ref()));
    }
    fn raw(&mut self, s: &str) {
        self.lines.push(s.to_string());
    }
}

fn header(
    g: &mut Gen,
    cfg: &KernelConfig,
    target: Target,
    mode: Mode,
    lpl: u32,
    k: u32,
    in_body: bool,
) {
    // Native source has no prelude (naga 30 rejects `enable subgroups;`); browsers require it.
    let prelude = target.prelude();
    if !prelude.is_empty() {
        g.raw(prelude.trim_end());
    }
    g.line("// Generated by bigmont-core unrolled.rs: straight-line limb arithmetic, named scalar limbs.");
    g.line(format!(
        "const TPI: u32 = {}u;     // lanes per number",
        cfg.tpi
    ));
    g.line(format!(
        "const LPL: u32 = {lpl}u;     // limbs per lane; TPI * LPL = number of limbs"
    ));
    g.line("const L: u32 = TPI * LPL;");
    g.line(format!("const WG: u32 = {}u;", cfg.wg));
    g.line("const B: u32 = 13u;");
    g.line("const MASK: u32 = 0x1fffu;");
    if in_body {
        let tail = if !lpl.is_multiple_of(PERIOD) {
            format!(
                ", plus an extra one at the end of each owner (LPL % {PERIOD} = {})",
                lpl % PERIOD
            )
        } else {
            String::new()
        };
        g.line(format!("// light carry passes are emitted inside the unrolled body every {PERIOD} steps (LPL {lpl} > {PERIOD}){tail}"));
    } else {
        g.line(format!("const K: u32 = {k}u;       // owners between light carry passes (K * LPL <= {PERIOD} steps)"));
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
}

fn prologue(g: &mut Gen) {
    g.line("let sg = sg_in;");
    g.line("let lane = sg % TPI;");
    g.line("let base = sg - lane;");
    g.line("let inst = wid.x * (WG / TPI) + li / TPI;");
    g.line("let live = inst < params.x;");
    g.line("let ic = min(inst, params.x - 1u);     // out-of-range lanes compute on a valid instance; their store is skipped");
    g.line("let off = ic * L + lane * LPL;");
}

fn load(g: &mut Gen, lpl: u32) {
    for c in 0..lpl {
        g.line(format!("var x{c} = x_limbs[off + {c}u];"));
    }
    for c in 0..lpl {
        g.line(format!("let n{c} = n_limbs[off + {c}u];"));
    }
    for c in 0..lpl {
        g.line(format!("var t{c} = 0u;"));
    }
}

/// Values that do not change across squarings.
fn hoists(g: &mut Gen) {
    g.line("let ninv = ninv_buf[ic] & MASK;                   // already below 2^13; the explicit mask lets the compiler see the m product fits 24 bits");
    g.line("let is0 = select(0u, 0xffffffffu, lane == 0u);   // all ones on lane 0 only");
    g.line("let N0 = bcast(n0, 0u, base);                     // lane 0's limb 0 of n");
    g.line("let NN0 = down1(n0, lane);                        // next lane's limb 0 of n (0 for the last lane)");
}

fn squaring_start(g: &mut Gen, lpl: u32) {
    g.line("let X0 = bcast(x0, 0u, base);");
    g.line("let XN0 = down1(x0, lane);");
    for c in 0..lpl {
        g.line(format!("t{c} = 0u;"));
    }
}

/// One multiply-accumulate step: the multiplier limb is `x{w}` of lane `owner`.
fn iteration(g: &mut Gen, lpl: u32, w: u32) {
    g.line("{");
    g.depth += 1;
    g.line(format!(
        "let bl = bcast(x{w}, owner, base) & {M24};    // next limb of the multiplier"
    ));
    g.line("let T0 = bcast(t0, 0u, base);               // accumulator limb 0 of lane 0");
    g.line("let TN = down1(t0, lane);                   // next lane's accumulator limb 0 (0 for the last lane)");
    g.line(format!(
        "let p = T0 + (X0 & {M24}) * bl;            // lane 0's limb 0 after adding x*b"
    ));
    g.line("let m = ((p & MASK) * ninv) & MASK;        // Montgomery factor: only the low 13 bits matter, product below 2^26 (24-bit multiply)");
    g.line(format!(
        "let carry0 = (p + (N0 & {M24}) * m) >> B;  // exact: the low 13 bits are zero"
    ));
    g.line("// new t = old t shifted down one limb, plus x*b and n*m shifted the same way; each line reads the old value above it");
    for c in 0..lpl - 1 {
        let u = c + 1;
        let carry = if c == 0 { " + (carry0 & is0)" } else { "" };
        g.line(format!(
            "t{c} = t{u} + (x{u} & {M24}) * bl + (n{u} & {M24}) * m{carry};"
        ));
    }
    let top = lpl - 1;
    // With a single limb per lane the top limb is also limb 0, which receives the carry.
    let top_carry = if lpl == 1 { " + (carry0 & is0)" } else { "" };
    g.line(format!(
        "t{top} = TN + (XN0 & {M24}) * bl + (NN0 & {M24}) * m{top_carry};"
    ));
    g.depth -= 1;
    g.line("}");
}

/// Each limb keeps its low 13 bits; the carry moves one limb up (across lanes for the top limb).
fn light_pass(g: &mut Gen, lpl: u32) {
    g.line("{");
    g.depth += 1;
    g.line(format!(
        "var c_in = up1(t{} >> B, lane);   // read before any limb is modified",
        lpl - 1
    ));
    for c in 0..lpl {
        g.line(format!("let v{c} = t{c};"));
        g.line(format!("t{c} = (v{c} & MASK) + c_in;"));
        g.line(format!("c_in = v{c} >> B;"));
    }
    g.depth -= 1;
    g.line("}");
}

/// Light passes until every limb of every instance in the subgroup is below 2^13.
fn full_normalize(g: &mut Gen, lpl: u32) {
    g.line("for (var round = 0u; round < L + 2u; round++) {");
    g.depth += 1;
    let big: Vec<String> = (0..lpl).map(|c| format!("(t{c} >> B)")).collect();
    g.line(format!("let big = {};", big.join(" | ")));
    g.line("if (!any_nonzero(big)) { break; }");
    light_pass(g, lpl);
    g.depth -= 1;
    g.line("}");
}

/// Fermat mode: `x <- 2x` (fully normalised) when exponent bit `fp.y - 2 - (fp.x + s)` is set.
/// Every lane computes the doubling and the bit only selects the result, so the shuffles and
/// `subgroupAny` in `full_normalize` stay in uniform control flow.
fn conditional_double(g: &mut Gen, lpl: u32) {
    g.line("let gi = fp.y - 2u - (fp.x + s);");
    g.line("let ebit = (e_words[ic * fp.z + (gi >> 5u)] >> (gi & 31u)) & 1u;");
    for c in 0..lpl {
        g.line(format!("t{c} = x{c} << 1u;"));
    }
    full_normalize(g, lpl);
    for c in 0..lpl {
        g.line(format!("x{c} = select(x{c}, t{c}, ebit == 1u);"));
    }
}

fn store(g: &mut Gen, lpl: u32) {
    g.line("if (live) {");
    g.depth += 1;
    for c in 0..lpl {
        g.line(format!("x_limbs[off + {c}u] = x{c};"));
    }
    g.depth -= 1;
    g.line("}");
}
