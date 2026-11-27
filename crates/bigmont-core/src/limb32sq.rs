//! WGSL generator for the symmetric squaring of the 32-bit limb scheme (`Squaring::Symmetric`):
//! cooperative Montgomery squaring that forms `x^2` once, with about three quarters of the limb
//! products of the fused loop in `limb32.rs`, then reduces it. Same bindings, modes and result
//! contract as `limb32.rs` (`x <- x^2 R^-1 mod n`, result in [0, 2n), full 32-bit limbs,
//! `R = 2^(32 L) > 16 n`). It needs `SHADER_INT64` and subgroup operations, so like `limb32.rs` it is
//! never generated for the browser target. Multiply mode keeps the fused generator.
//!
//! Design notes: `docs/cooperative-squaring-design.md`. The schedule is the "rediagonalisation" of
//! cuda-fixnum's `sqr_wide_` (after Ozturk, Guilford and Gopal 2013) with one lane's whole limb run as
//! the digit (ideas only; no code is copied).
//!
//! ## Phase 1: `S = x^2` as `2 L` words
//!
//! Let `W = TPI` and block `A_l` be lane `l`'s `LPL` limbs, `B = 2^(32 LPL)`. In iteration `i`
//! (`0 <= i < W/2`) lane `l` multiplies blocks `(p, q) = (i, l + i)` if `l + i < W`, else
//! `(l + i - W/2, W/2 + i)`. Either way `p + q = l + 2 i`, so the product has block column `l + 2 i`:
//! lane `l`'s own position in a window that slides down one lane per half-iteration. Over all lanes and
//! iterations this visits every unordered pair `p < q` once, plus the diagonals `(i, i)` for `i < W/2`
//! (only lane 0 has `p == q`). The diagonals for `a >= W/2` are added at the end.
//!
//! Doubling: every cross term is needed twice, so the second operand is doubled before the block
//! product (`2 A_q`, an `LPL`-word shift plus one top bit, whose contribution `A_p B^LPL` is added
//! separately). Lane 0 uses the undoubled operand, so its `(i, i)` product is `A_i^2` exactly once.
//! The product is then at most `2 B^2 - 1`: its `2 LPL` words plus one carry bit `ext` of weight
//! `B^(2 LPL)`.
//!
//! Window and carries: per half-iteration the low (then high) `LPL` words of the product are added to
//! the lane's window words with a sequential u64 carry chain. The carry out of the chain's top word has
//! the weight of the next lane's word 0, which after the following one-lane shift is this lane's word
//! 0 again, so it stays in the lane (`cy`) and is the next chain's carry-in. `ext` has the same weight
//! as the high chain's carry-out and is added to `cy`. Chain bound: carry-in is at most 3, so a sum is
//! below `2 (2^32 - 1) + 3`, the carry out is at most 2, and `cy` (carry out plus `ext`) is at most 3.
//! After the low add, lane 0's window block is exact (every pair with `p + q <= 2 i` was added and its
//! carry-in came from the previous chain), so it is the finished block `2 i`; likewise after the high
//! add for block `2 i + 1`. Finished blocks go to lane `j` as `r` (one broadcast per word).
//!
//! After `W/2` iterations the window holds blocks `W .. 2 W` (lane `j` holds block `W + j`) and every
//! lane has a pending carry `cy` into its word 0. The upper diagonals are added (lane `l` squares block
//! `W/2 + l/2` with a symmetric in-lane triangle and keeps its low or high half by parity), then the
//! pending carries ripple up through the lanes (uniform `subgroupAny` rounds, as `resolve` in
//! `limb32.rs`). Result: `r` is the low half of `S`, `s` the high half, `S = x^2 < R^2` exactly, for
//! any `x < R`.
//!
//! ## Phase 2: reduction
//!
//! The window starts as `S`'s low half (`r`). Step `k` (the multiplier index of the fused loop) is the
//! fused step without the `x b` products: `m = t_0 * ninv`, `t <- (t + n m) / 2^32`, and the word
//! `S[L + k]` (broadcast from the block `s`) enters the top element of the last lane in place of the
//! multiplier broadcast. The top element is `q + inj + pend` with `q` at most `2^64 - 1`, `inj` below
//! `2^32` and `pend` below `2^34`: at most one of the two sums wraps (after a wrap the first sum is
//! below `2^32`, so adding `pend` cannot wrap again), so the new `pend = (s2 >> 32) + wrap * 2^32` is
//! below `2^33`, inside the `2^34` bound the next step relies on. The result is `(S + M n) / R <
//! x^2 / R + n < 2 n` for `x < 4 n`, resolved with the same pending-carry rounds as `limb32.rs`.

use crate::kernel::{KernelConfig, SUBGROUP_HELPERS};
use crate::limb32::{Gen, conditional_double, resolve};
use crate::unrolled::Mode;

/// Emits the complete native WGSL source for `cfg` (an unrolled 32-bit configuration with `tpi` 4, 8,
/// 16 or 32) in [`Mode::Squarings`] or [`Mode::Fermat`]. Bindings as in `limb32::wgsl`.
pub(crate) fn wgsl(cfg: &KernelConfig, mode: Mode) -> Result<String, String> {
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
    let mut g = Gen {
        lines: Vec::new(),
        depth: 0,
    };
    g.line(format!(
        "// Generated by bigmont-core limb32sq.rs: TPI {w}, {lpl} limbs per lane, symmetric squaring then reduction."
    ));
    g.line(format!("const TPI: u32 = {w}u;     // lanes per number"));
    g.line(format!("const LPL: u32 = {lpl}u;     // limbs per lane"));
    g.line("const L: u32 = TPI * LPL;");
    g.line(format!("const WG: u32 = {}u;", cfg.wg));
    g.line("");
    if mode == Mode::Fermat {
        g.line("@group(0) @binding(0) var<uniform> params: vec4<u32>;          // x = instance count, y = ladder iterations in this dispatch");
    } else {
        g.line("@group(0) @binding(0) var<uniform> params: vec4<u32>;          // x = instance count, y = squarings");
    }
    g.line("@group(0) @binding(1) var<storage, read> n_limbs: array<u32>;");
    g.line("@group(0) @binding(2) var<storage, read> ninv_buf: array<u32>;   // -n^-1 mod 2^32");
    g.line("@group(0) @binding(3) var<storage, read_write> x_limbs: array<u32>;");
    if mode == Mode::Fermat {
        g.line("@group(0) @binding(4) var<storage, read> e_words: array<u32>;   // n - 1, words per instance = fp.z");
        g.line("@group(0) @binding(5) var<uniform> fp: vec4<u32>;               // x = first iteration, y = nbits, z = words per instance");
    }
    g.raw(SUBGROUP_HELPERS);
    g.line("@compute @workgroup_size(WG)");
    g.line("fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) sg_in: u32) {");
    g.depth += 1;
    g.line("let lane = sg_in % TPI;");
    g.line("let base = sg_in - lane;");
    g.line("let inst = wid.x * (WG / TPI) + li / TPI;");
    g.line("let live = inst < params.x;");
    g.line("let ic = min(inst, params.x - 1u);     // out-of-range lanes compute on a valid instance; their store is skipped");
    g.line("let off = ic * L + lane * LPL;");
    for c in 0..lpl {
        g.line(format!("var x{c} = x_limbs[off + {c}u];"));
    }
    for c in 0..lpl {
        g.line(format!("let n{c} = n_limbs[off + {c}u];"));
    }
    g.line("let ninv = ninv_buf[ic];");
    g.line("let is0 = lane == 0u;");
    g.line("let N0 = bcast(n0, 0u, base);                     // lane 0's limb 0 of n");
    g.line("let NN0 = down1(n0, lane);                        // next lane's limb 0 of n (0 for the last lane)");
    g.line("for (var s = 0u; s < params.y; s++) {");
    g.depth += 1;
    phase1(&mut g, w, lpl);
    phase2(&mut g, lpl);
    for c in 0..lpl {
        g.line(format!("x{c} = t{c};"));
    }
    if mode == Mode::Fermat {
        conditional_double(&mut g, lpl);
    }
    g.depth -= 1;
    g.line("}");
    g.line("if (live) {");
    g.depth += 1;
    for c in 0..lpl {
        g.line(format!("x_limbs[off + {c}u] = x{c};"));
    }
    g.depth -= 1;
    g.line("}");
    g.depth -= 1;
    g.line("}");
    Ok(g.lines.join("\n") + "\n")
}

/// Adds the `lpl` words named by `src(k)` into the window words `s{k}` with a sequential u64 carry
/// chain starting from carry `c0`; the carry out is left in the variable `c`.
fn add_chain(g: &mut Gen, lpl: usize, c0: &str, src: impl Fn(usize) -> String) {
    g.line(format!("var c = u64({c0});"));
    for k in 0..lpl {
        g.line(format!("let v{k} = u64(s{k}) + u64({}) + c;", src(k)));
        g.line(format!("s{k} = u32(v{k});"));
        g.line(format!("c = v{k} >> 32u;"));
    }
}

/// Phase 1: defines `r0..` (low half of `x^2`, lane `j` holds block `j`) and `s0..` (high half, lane `j`
/// holds block `W + j`). See the module comment for the schedule and the carry argument.
fn phase1(g: &mut Gen, w: usize, lpl: usize) {
    let half = w / 2;
    for c in 0..lpl {
        g.line(format!("var r{c} = 0u;"));
        g.line(format!("var s{c} = 0u;"));
    }
    g.line("var cy = 0u;                                       // deferred carry into this lane's window word 0 (at most 3)");
    for i in 0..half {
        g.line("{");
        g.depth += 1;
        g.line(format!("let wrap = lane + {i}u >= TPI;"));
        g.line(format!(
            "let pi = select({i}u, lane + {i}u - {half}u, wrap);         // block p of this lane's pair"
        ));
        g.line(format!(
            "let qi = select(lane + {i}u, {}u, wrap);                     // block q; p + q = lane + 2 i",
            half + i
        ));
        for k in 0..lpl {
            g.line(format!("let ap{k} = bcast(x{k}, pi, base);"));
        }
        for k in 0..lpl {
            if i == 0 && w == 4 {
                // No lane wraps in iteration 0, so every lane's second block is its own. (Measured at
                // 2,029 bits on an RTX 4070 Laptop, Vulkan: +1.3 points at TPI 4, -0.6 at TPI 8, where
                // the compiler's schedule is better without it. Other vendors may differ.)
                g.line(format!("let aq{k} = x{k};"));
            } else {
                g.line(format!("let aq{k} = bcast(x{k}, qi, base);"));
            }
        }
        g.line("let single = lane == 0u;                          // p == q only on lane 0: that product must not be doubled");
        g.line(format!(
            "let topbit = select(aq{} >> 31u, 0u, single);          // bit 2^(32 LPL) of 2 A_q",
            lpl - 1
        ));
        // Block product P = A_p * (2 A_q mod B) (2 LPL words), operand scanning. The doubled limb is
        // formed in place (a register-pressure choice: keeping a doubled copy live cost occupancy).
        for k in 0..2 * lpl {
            g.line(format!("var pr{k} = 0u;"));
        }
        g.line("var a: u64;");
        for j in 0..lpl {
            g.line("{");
            g.depth += 1;
            let carry_in = if j == 0 {
                String::new()
            } else {
                format!(" | (aq{} >> 31u)", j - 1)
            };
            g.line(format!(
                "let bj = u64(select((aq{j} << 1u){carry_in}, aq{j}, single));"
            ));
            g.line("var c = 0lu;");
            for k in 0..lpl {
                g.line(format!("a = u64(pr{}) + u64(ap{k}) * bj + c;", j + k));
                g.line(format!("pr{} = u32(a);", j + k));
                g.line("c = a >> 32u;");
            }
            g.line(format!("pr{} = u32(c);", j + lpl));
            g.depth -= 1;
            g.line("}");
        }
        // + topbit * A_p * B^LPL, and the carry bit above the 2 LPL words.
        g.line("var ext = 0u;");
        g.line("{");
        g.depth += 1;
        g.line("var c = 0lu;");
        for k in 0..lpl {
            g.line(format!(
                "a = u64(pr{}) + u64(select(0u, ap{k}, topbit == 1u)) + c;",
                lpl + k
            ));
            g.line(format!("pr{} = u32(a);", lpl + k));
            g.line("c = a >> 32u;");
        }
        g.line("ext = u32(c);");
        g.depth -= 1;
        g.line("}");
        // Low half of the product into the window; lane 0's block is then the finished block 2 i.
        g.line("{");
        g.depth += 1;
        add_chain(g, lpl, "cy", |k| format!("pr{k}"));
        g.line("cy = u32(c);");
        g.depth -= 1;
        g.line("}");
        finish_and_shift(g, lpl, 2 * i);
        // High half, plus ext (same weight as the chain's carry out).
        g.line("{");
        g.depth += 1;
        add_chain(g, lpl, "cy", |k| format!("pr{}", lpl + k));
        g.line("cy = u32(c) + ext;");
        g.depth -= 1;
        g.line("}");
        finish_and_shift(g, lpl, 2 * i + 1);
        g.depth -= 1;
        g.line("}");
    }
    // Upper diagonals: lane l squares block W/2 + l/2 with a symmetric in-lane triangle.
    g.line("{");
    g.depth += 1;
    g.line(format!("let ai = {half}u + (lane >> 1u);"));
    for k in 0..lpl {
        g.line(format!("let dg{k} = bcast(x{k}, ai, base);"));
    }
    for k in 0..2 * lpl {
        g.line(format!("var q{k} = 0u;"));
    }
    g.line("var a: u64;");
    for j in 0..lpl - 1 {
        g.line("{");
        g.depth += 1;
        g.line(format!("let xj = u64(dg{j});"));
        g.line("var c = 0lu;");
        for k in j + 1..lpl {
            g.line(format!("a = u64(q{}) + xj * u64(dg{k}) + c;", j + k));
            g.line(format!("q{} = u32(a);", j + k));
            g.line("c = a >> 32u;");
        }
        g.line(format!("q{} = u32(c);", j + lpl));
        g.depth -= 1;
        g.line("}");
    }
    // double the cross sum, then add the diagonal squares
    g.line("var sc = 0lu;");
    g.line("var prev = 0u;");
    for k in 0..2 * lpl {
        g.line(format!("let d{k} = (q{k} << 1u) | prev;"));
        g.line(format!("prev = q{k} >> 31u;"));
        if k % 2 == 0 {
            g.line(format!(
                "let sq{} = u64(dg{}) * u64(dg{});",
                k / 2,
                k / 2,
                k / 2
            ));
            g.line(format!("a = u64(d{k}) + u64(u32(sq{})) + sc;", k / 2));
        } else {
            g.line(format!("a = u64(d{k}) + (sq{} >> 32u) + sc;", k / 2));
        }
        g.line(format!("let e{k} = u32(a);"));
        g.line("sc = a >> 32u;");
    }
    g.line("let odd = (lane & 1u) == 1u;");
    add_chain(g, lpl, "cy", |k| format!("select(e{k}, e{}, odd)", lpl + k));
    g.line("cy = up1(u32(c), lane);                            // the chain's carry out has the weight of the next lane's word 0");
    g.depth -= 1;
    g.line("}");
    // Ripple the pending carries up through the lanes.
    g.line("for (var round = 0u; round < TPI + 2u; round++) {");
    g.depth += 1;
    g.line("if (!any_nonzero(cy)) { break; }");
    g.line("var c = u64(cy);");
    for k in 0..lpl {
        g.line(format!("let u{k} = u64(s{k}) + c;"));
        g.line(format!("s{k} = u32(u{k});"));
        g.line(format!("c = u{k} >> 32u;"));
    }
    g.line("cy = up1(u32(c), lane);");
    g.depth -= 1;
    g.line("}");
}

/// After a chain add: lane 0's window block is the finished block `col`, stored in lane `col`'s `r`;
/// then the window shifts down one lane.
fn finish_and_shift(g: &mut Gen, lpl: usize, col: usize) {
    for k in 0..lpl {
        g.line(format!(
            "r{k} = select(r{k}, bcast(s{k}, 0u, base), lane == {col}u);"
        ));
    }
    for k in 0..lpl {
        g.line(format!("s{k} = down1(s{k}, lane);"));
    }
}

/// Phase 2: reduces `S = (r, s)` into `t0..` (see the module comment).
fn phase2(g: &mut Gen, lpl: usize) {
    for c in 0..lpl {
        g.line(format!("var t{c} = r{c};"));
    }
    g.line("var pend = 0lu;                                   // deferred carry into this lane's last limb (below 2^34)");
    g.line("for (var owner = 0u; owner < TPI; owner++) {");
    g.depth += 1;
    for w in 0..lpl {
        g.line("{");
        g.depth += 1;
        g.line(format!(
            "let inj = u64(bcast(s{w}, owner, base));          // word L + k of x^2"
        ));
        g.line("let T0 = bcast(t0, 0u, base);                   // window limb 0 of lane 0");
        g.line("let TN = down1(t0, lane);                       // next lane's window limb 0 (0 for the last lane)");
        g.line(
            "let m = T0 * ninv;                              // Montgomery factor (u32 wrapping)",
        );
        g.line("let mm = u64(m);");
        g.line("let b0 = u64(T0) + u64(N0) * mm;               // low word is zero");
        g.line("var c2 = select(0lu, b0 >> 32u, is0);");
        for c in 0..lpl - 1 {
            let u = c + 1;
            g.line(format!("let q{c} = u64(n{u}) * mm + u64(t{u}) + c2;"));
            g.line(format!("t{c} = u32(q{c});"));
            g.line(format!("c2 = q{c} >> 32u;"));
        }
        let top = lpl - 1;
        g.line("let qt = u64(NN0) * mm + u64(TN) + c2;");
        g.line("let injt = select(0lu, inj, lane == TPI - 1u); // the last lane's top element takes the next word of S");
        g.line("let s2a = qt + injt;");
        g.line("let w1 = select(0lu, 4294967296lu, s2a < qt);");
        g.line("let s2 = s2a + pend;                            // pend below 2^34");
        g.line("let w2 = select(0lu, 4294967296lu, s2 < s2a);");
        g.line(format!("t{top} = u32(s2);"));
        g.line(
            "pend = (s2 >> 32u) + w1 + w2;                  // below 2^33 (see the module comment)",
        );
        g.depth -= 1;
        g.line("}");
    }
    g.depth -= 1;
    g.line("}");
    resolve(g, lpl);
}
