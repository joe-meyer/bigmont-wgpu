//! Pure-Rust Montgomery base-2 Fermat test with 64-bit limbs (CIOS), used as the CPU fallback and
//! as an independent check on GPU verdicts. Not constant time. Setup (R mod n) uses num-bigint.
//!
//! [`fermat_base2_gec`] and [`fermat_batch_gec`] run the same test under the Gerbicz error check
//! (the algorithm of `bigmont_oracle::gerbicz_model` and of the GPU's checked runs), so a silent
//! arithmetic error on the CPU is reported as [`Checked::Alarm`] instead of a wrong verdict.

use bigmont_oracle::{GEC_BLOCK, gec_blocks, gec_window_sum_words};
use num_bigint::BigUint;
use num_traits::One;

/// Montgomery context for an odd modulus of `w` 64-bit limbs; R = 2^(64 w). Every result is fully
/// reduced, in `[0, n)`.
pub struct Mont {
    n: Vec<u64>,
    ninv: u64, // -n^-1 mod 2^64
    w: usize,
    kernel: Kernel,
}

impl Mont {
    /// The context for the odd modulus whose little-endian 64-bit limbs are `n` (the top limb may
    /// be zero).
    pub fn new(n: &[u64]) -> Mont {
        assert!(n[0] & 1 == 1, "modulus must be odd");
        let mut inv = n[0]; // correct to 3 bits
        for _ in 0..5 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(n[0].wrapping_mul(inv)));
        }
        Mont {
            n: n.to_vec(),
            ninv: inv.wrapping_neg(),
            w: n.len(),
            kernel: Kernel::detect(),
        }
    }

    /// Length of the scratch slice `t` that [`Mont::mul`] and [`Mont::sqr`] both accept: `w + 2`
    /// limbs are enough for `mul`, `2 w` for `sqr`, so this is the larger of the two. A longer
    /// slice is fine; only the first `scratch_len()` limbs are used.
    pub fn scratch_len(&self) -> usize {
        (2 * self.w).max(self.w + 2)
    }

    /// This context forced onto the portable safe-Rust kernel even where the hardware one is
    /// available. The results are identical; this is for tests and for benchmarking the two.
    pub fn portable(mut self) -> Mont {
        self.kernel = Kernel::Portable;
        self
    }

    /// Whether the squaring uses the BMI2 + ADX (`mulx`, `adcx`, `adox`) kernel: only on x86-64
    /// CPUs that report both features, and not after [`Mont::portable`].
    pub fn uses_adx(&self) -> bool {
        self.kernel != Kernel::Portable
    }

    /// out = a * b * R^-1 mod n for a, b < n. `t` is scratch of at least `w + 2` limbs.
    ///
    /// Operand scanning (CIOS). Every loop runs over sub-slices of one known length, so the
    /// compiler can drop the per-limb bounds checks without any `unsafe`.
    pub fn mul(&self, a: &[u64], b: &[u64], out: &mut [u64], t: &mut [u64]) {
        let w = self.w;
        let (a, b, n, out) = (&a[..w], &b[..w], &self.n[..w], &mut out[..w]);
        let t = &mut t[..w + 2];
        t.fill(0);
        for &bi in b {
            let mut c = 0u64;
            for (tj, &aj) in t[..w].iter_mut().zip(a) {
                (*tj, c) = mac(aj, bi, *tj, c);
            }
            let s = u128::from(t[w]) + u128::from(c);
            t[w] = s as u64;
            t[w + 1] = (s >> 64) as u64;
            let m = t[0].wrapping_mul(self.ninv);
            let (_, mut c) = mac(m, n[0], t[0], 0);
            for j in 1..w {
                (t[j - 1], c) = mac(m, n[j], t[j], c);
            }
            let s = u128::from(t[w]) + u128::from(c);
            t[w - 1] = s as u64;
            t[w] = t[w + 1] + (s >> 64) as u64;
        }
        out.copy_from_slice(&t[..w]);
        if t[w] != 0 || ge(out, n) {
            sub_in_place(out, n);
        }
    }

    /// out = a * a * R^-1 mod n for a < n. `t` is scratch of at least `2 w` limbs (see
    /// [`Mont::scratch_len`]); this is the operation the Fermat ladder spends nearly all its time in.
    ///
    /// A dedicated squaring followed by a word-by-word Montgomery reduction (SOS): the cross terms
    /// a_i a_j (i < j) are computed once and doubled, then the diagonal a_i^2 is added, which is
    /// about w^2 / 2 limb products against w^2 for `mul(a, a)`, and the reduction adds m_i n for
    /// each low limb. Like [`Mont::mul`] it is safe code written so the bounds checks disappear.
    pub fn sqr(&self, a: &[u64], out: &mut [u64], t: &mut [u64]) {
        let w = self.w;
        assert!(
            t.len() >= 2 * w,
            "Mont::sqr needs {} limbs of scratch, got {}",
            2 * w,
            t.len()
        );
        let (a, n, out) = (&a[..w], &self.n[..w], &mut out[..w]);
        let t = &mut t[..2 * w];
        t.fill(0);
        // Cross terms: row i adds a_i * a_j for j > i at limb i + j, and its carry lands at limb
        // i + w, which no earlier row has touched. The sum is below B^(2w) / 2.
        for (i, &ai) in a.iter().enumerate() {
            t[i + w] = addmul_row(self.kernel, &mut t[2 * i + 1..i + w], &a[i + 1..], ai);
        }
        // Double them; no bit leaves the top limb because the sum is below B^(2w) / 2.
        let mut sh = 0u64;
        for v in t.iter_mut() {
            let next = *v >> 63;
            *v = (*v << 1) | sh;
            sh = next;
        }
        // Add the diagonal a_i^2 at limbs 2i and 2i + 1; the total is a^2 < B^(2w), so no carry
        // leaves the top.
        let mut carry = 0u64;
        for (pair, &ai) in t.as_chunks_mut::<2>().0.iter_mut().zip(a) {
            let sq = u128::from(ai) * u128::from(ai);
            let s = u128::from(pair[0]) + u128::from(sq as u64) + u128::from(carry);
            pair[0] = s as u64;
            let s = u128::from(pair[1]) + (sq >> 64) + (s >> 64);
            pair[1] = s as u64;
            carry = (s >> 64) as u64;
        }
        // Reduce: row i adds m_i n (m_i = t_i * ninv) so that limb i becomes zero. Limb i is then
        // dead, so the row's carry out (weight B^(i + w)) is parked there and added to the high
        // half once at the end, which avoids propagating it through the rows.
        for i in 0..w {
            let m = t[i].wrapping_mul(self.ninv);
            t[i] = addmul_row(self.kernel, &mut t[i..i + w], n, m);
        }
        // (a^2 + M n) / B^w < 2 n: the high half plus the parked carries, with at most one carry
        // out, then at most one subtraction of n.
        let (parked, high) = t.split_at(w);
        let mut carry = false;
        for ((o, &h), &p) in out.iter_mut().zip(high).zip(parked) {
            let (s, c1) = h.overflowing_add(p);
            let (s, c2) = s.overflowing_add(u64::from(carry));
            *o = s;
            carry = c1 | c2;
        }
        if carry || ge(out, n) {
            sub_in_place(out, n);
        }
    }

    /// x = 2x mod n for x < n.
    pub fn double(&self, x: &mut [u64]) {
        let mut carry = 0u64;
        for v in x.iter_mut() {
            let next = *v >> 63;
            *v = (*v << 1) | carry;
            carry = next;
        }
        if carry != 0 || ge(x, &self.n) {
            sub_in_place(x, &self.n);
        }
    }
}

/// a * b + t + c as (low, high) limbs; it cannot overflow 128 bits: (2^64-1)^2 + 2 (2^64-1) = 2^128 - 1,
/// so the high limb plus the two carries below never wraps either.
///
/// Written as one widening multiply and two `overflowing_add`s on the low limb rather than as a
/// 128-bit sum of three terms: LLVM compiles the sum to about 10 instructions per limb product
/// (it builds each carry bit in a register), and this form to about 9, which measured about 10%
/// faster for the whole test (see `docs/results/2026-11-26-cpu-backend-vs-gmp.md`).
#[inline(always)]
fn mac(a: u64, b: u64, t: u64, c: u64) -> (u64, u64) {
    let p = u128::from(a) * u128::from(b);
    let (s, c1) = (p as u64).overflowing_add(c);
    let (s, c2) = s.overflowing_add(t);
    (s, ((p >> 64) as u64) + u64::from(c1) + u64::from(c2))
}

/// Which row routine a [`Mont`] uses. Chosen once, when the context is built.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kernel {
    /// Safe Rust everywhere.
    Portable,
    /// `mulx` with two carry chains (`adcx`, `adox`), x86-64 only. Constructed only by
    /// [`Kernel::detect`], after it has seen BMI2 and ADX on the running CPU.
    #[cfg(target_arch = "x86_64")]
    Adx,
}

impl Kernel {
    fn detect() -> Kernel {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("bmi2") && is_x86_feature_detected!("adx") {
            return Kernel::Adx;
        }
        Kernel::Portable
    }
}

#[cfg(target_arch = "x86_64")]
mod adx {
    use super::mac;
    use std::arch::asm;

    /// [`super::addmul_row_portable`] with the four-limb body in assembly: per limb one `mulx`
    /// (no flags touched, any destination registers), `adcx` adding the previous limb's high word
    /// (carry flag chain) and `adox` adding the accumulator word (overflow flag chain). Two
    /// independent chains replace the single add/adc chain that limited the portable loop; on
    /// Zen 3 and Zen 4 that is about 1.5x faster for the whole test.
    ///
    /// The rows are 32 limbs long here, so a row is `len / 4` passes of the asm plus a scalar tail
    /// of at most three limbs.
    ///
    /// # Safety
    /// The CPU must support BMI2 and ADX (`Kernel::Adx` is only built after both are detected).
    pub(super) unsafe fn addmul_row(t: &mut [u64], src: &[u64], m: u64) -> u64 {
        // The asm below reads and writes `4 * quads` limbs through raw pointers, so the length
        // match is a hard check, not a debug one.
        assert_eq!(t.len(), src.len());
        let quads = t.len() / 4;
        let mut c = 0u64;
        if quads > 0 {
            let (mut z, mut h1, mut l) = (0u64, 0u64, 0u64);
            let mut hi = 0u64;
            // SAFETY: `t` and `src` are distinct slices (one `&mut`, one `&`) of equal length, so
            // the asm touches only limbs `0 .. 4 * quads` of each, all in bounds. It reads and
            // writes nothing else, uses no stack, and `rdx` (the multiplier `mulx` needs) and
            // `rcx` (the pass counter `jrcxz` tests) are declared as the registers they are.
            // `lea` and `jrcxz` leave the flags alone, so CF and OF carry from one limb to the
            // next, also across passes; `xor` clears both before the first limb and the two final
            // adds fold the pending carries into `hi`, which cannot overflow because the whole
            // row sum t + m * src is below 2^(64 * (len + 1)). The CPU requirement is this
            // function's own safety contract.
            unsafe {
                asm!(
                    // Load-bearing: clears CF and OF as well as `z`, which `adcx`/`adox` start from.
                    "xor {z:e}, {z:e}",
                    "2:",
                    "mulx {h1}, {l}, qword ptr [{src}]",
                    "adcx {l}, {hi}",
                    "adox {l}, qword ptr [{t}]",
                    "mov qword ptr [{t}], {l}",
                    "mulx {hi}, {l}, qword ptr [{src}+8]",
                    "adcx {l}, {h1}",
                    "adox {l}, qword ptr [{t}+8]",
                    "mov qword ptr [{t}+8], {l}",
                    "mulx {h1}, {l}, qword ptr [{src}+16]",
                    "adcx {l}, {hi}",
                    "adox {l}, qword ptr [{t}+16]",
                    "mov qword ptr [{t}+16], {l}",
                    "mulx {hi}, {l}, qword ptr [{src}+24]",
                    "adcx {l}, {h1}",
                    "adox {l}, qword ptr [{t}+24]",
                    "mov qword ptr [{t}+24], {l}",
                    "lea {src}, [{src}+32]",
                    "lea {t}, [{t}+32]",
                    "lea rcx, [rcx-1]",
                    "jrcxz 3f",
                    "jmp 2b",
                    "3:",
                    "adcx {hi}, {z}",
                    "adox {hi}, {z}",
                    z = out(reg) z,
                    h1 = out(reg) h1,
                    l = out(reg) l,
                    hi = inout(reg) hi,
                    src = inout(reg) src.as_ptr() => _,
                    t = inout(reg) t.as_mut_ptr() => _,
                    inout("rcx") quads => _,
                    in("rdx") m,
                    options(nostack),
                );
            }
            let _ = (z, h1, l);
            c = hi;
        }
        for (tj, &sj) in t[4 * quads..].iter_mut().zip(&src[4 * quads..]) {
            (*tj, c) = mac(m, sj, *tj, c);
        }
        c
    }
}

/// t += m * src as one multi-limb number (`t` and `src` have the same length); returns the carry
/// limb out of the top. This is the inner loop of the squaring and of the reduction.
fn addmul_row(kernel: Kernel, t: &mut [u64], src: &[u64], m: u64) -> u64 {
    match kernel {
        Kernel::Portable => addmul_row_portable(t, src, m),
        // SAFETY: `Kernel::Adx` exists only after `Kernel::detect` saw BMI2 and ADX.
        #[cfg(target_arch = "x86_64")]
        Kernel::Adx => unsafe { adx::addmul_row(t, src, m) },
    }
}

fn addmul_row_portable(t: &mut [u64], src: &[u64], m: u64) -> u64 {
    let mut c = 0u64;
    for (tj, &sj) in t.iter_mut().zip(src) {
        (*tj, c) = mac(m, sj, *tj, c);
    }
    c
}

fn ge(a: &[u64], b: &[u64]) -> bool {
    for i in (0..a.len()).rev() {
        if a[i] != b[i] {
            return a[i] > b[i];
        }
    }
    true
}

fn sub_in_place(a: &mut [u64], b: &[u64]) {
    let mut borrow = 0u64;
    for i in 0..a.len() {
        let (d1, b1) = a[i].overflowing_sub(b[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        a[i] = d2;
        borrow = u64::from(b1 | b2);
    }
}

fn to_limbs(x: &BigUint, w: usize) -> Vec<u64> {
    let mut v = x.to_u64_digits();
    v.resize(w, 0);
    v
}

/// Base-2 Fermat test: 2^(n-1) = 1 (mod n); same semantics as `bigmont_oracle::fermat_base2`.
pub fn fermat_base2(n: &BigUint) -> bool {
    let two = BigUint::from(2u32);
    if n < &two {
        return false;
    }
    if n == &two {
        return true;
    }
    if !n.bit(0) {
        return false;
    }
    let w = n.to_u64_digits().len();
    let nl = to_limbs(n, w);
    let ctx = Mont::new(&nl);
    let tlen = ctx.scratch_len();
    let r = (BigUint::one() << (64 * w)) % n; // Montgomery form of 1
    let one_m = to_limbs(&r, w);
    let mut x = to_limbs(&((&r << 1u32) % n), w); // Montgomery form of 2 (top exponent bit)
    let e = n - 1u32;
    let nbits = e.bits();
    let mut t = vec![0u64; tlen];
    let mut y = vec![0u64; w];
    for i in (0..nbits - 1).rev() {
        ctx.sqr(&x, &mut y, &mut t);
        x.copy_from_slice(&y);
        if e.bit(i) {
            ctx.double(&mut x);
        }
    }
    x == one_m
}

/// Verdicts for every `n`, in order. `threads == 0` uses all cores.
pub fn fermat_batch(ns: &[BigUint], threads: usize) -> Vec<bool> {
    let threads = if threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        threads
    };
    let threads = threads.min(ns.len()).max(1);
    let per = ns.len().div_ceil(threads).max(1);
    let mut out = vec![false; ns.len()];
    std::thread::scope(|s| {
        for (chunk_in, chunk_out) in ns.chunks(per).zip(out.chunks_mut(per)) {
            s.spawn(move || {
                for (n, o) in chunk_in.iter().zip(chunk_out.iter_mut()) {
                    *o = fermat_base2(n);
                }
            });
        }
    });
    out
}

/// The verdict of a Gerbicz-checked run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Checked {
    Composite,
    Prp,
    /// The check failed: the computation went wrong somewhere and its result says nothing.
    Alarm,
}

/// The Montgomery arithmetic the checked ladder uses, so that tests can wrap [`Mont`] in an adapter
/// (for example one that corrupts an output) without any fault code in the algorithm. Every operand
/// and result is a fully reduced Montgomery value in `[0, n)`; `t` is scratch of [`Mont::scratch_len`] limbs.
pub trait MontOps {
    /// out = a * b * R^-1 mod n.
    fn mul(&mut self, a: &[u64], b: &[u64], out: &mut [u64], t: &mut [u64]);
    /// out = a * a * R^-1 mod n.
    fn sqr(&mut self, a: &[u64], out: &mut [u64], t: &mut [u64]);
    /// x = 2x mod n.
    fn double(&mut self, x: &mut [u64]);
}

impl MontOps for Mont {
    fn mul(&mut self, a: &[u64], b: &[u64], out: &mut [u64], t: &mut [u64]) {
        Mont::mul(self, a, b, out, t)
    }
    fn sqr(&mut self, a: &[u64], out: &mut [u64], t: &mut [u64]) {
        Mont::sqr(self, a, out, t)
    }
    fn double(&mut self, x: &mut [u64]) {
        Mont::double(self, x)
    }
}

/// Ladder iteration `i` of the base-2 Fermat ladder over `e` (`bits` bits): x <- x^2, then x <- 2x
/// if bit `bits - 2 - i` of `e` is set. `y` is scratch of length w.
fn ladder_step<M: MontOps>(
    ops: &mut M,
    e: &BigUint,
    bits: u32,
    i: u32,
    x: &mut [u64],
    y: &mut [u64],
    t: &mut [u64],
) {
    ops.sqr(x, y, t);
    x.copy_from_slice(y);
    if e.bit(u64::from(bits - 2 - i)) {
        ops.double(x);
    }
}

/// Base-2 Fermat test under the Gerbicz error check; the degenerate inputs follow
/// [`fermat_base2`] (below 2 or even is `Composite`, 2 is `Prp`). See [`fermat_base2_gec_with`].
pub fn fermat_base2_gec(n: &BigUint) -> Checked {
    fermat_base2_gec_with(n, |m| m)
}

/// [`fermat_base2_gec`] with its arithmetic done by `wrap(Mont::new(n))`; production passes the
/// identity, tests pass a faulty adapter.
///
/// The algorithm is the GPU path's on fully reduced values (so equal mod n means equal limbs): blocks
/// of `GEC_BLOCK` (40) ladder iterations, `A = B = 1~` and `y_prev = x = 2~` at the start, and at the
/// end of each block `A = A * y_prev`, `B = B * x`, `y_prev = x`. After the last full block the
/// remaining `(bits - 1) % 40` iterations (the whole run when there is no full block) run on `x`
/// and again on the copy in `y_prev`; the two must agree. With at least one block, `A` squared 40
/// times and multiplied by `2^(sum V)` (from `1~` by the same ladder over the bits of the window sum)
/// must equal `B`. Any mismatch is [`Checked::Alarm`]; otherwise the verdict is `Prp` iff `x == 1~`.
pub fn fermat_base2_gec_with<M: MontOps>(n: &BigUint, wrap: impl FnOnce(Mont) -> M) -> Checked {
    let two = BigUint::from(2u32);
    if n < &two {
        return Checked::Composite;
    }
    if n == &two {
        return Checked::Prp;
    }
    if !n.bit(0) {
        return Checked::Composite;
    }
    let w = n.to_u64_digits().len();
    let nl = to_limbs(n, w);
    let mont = Mont::new(&nl);
    let tlen = mont.scratch_len();
    let mut ops = wrap(mont);
    let r = (BigUint::one() << (64 * w)) % n; // Montgomery form of 1
    let one_m = to_limbs(&r, w);
    let two_m = to_limbs(&((&r << 1u32) % n), w); // Montgomery form of 2 (top exponent bit)
    let e = n - 1u32;
    let bits = u32::try_from(n.bits()).expect("candidate too large");
    let iters = bits - 1;
    let blocks = gec_blocks(bits, GEC_BLOCK);
    let full = blocks * GEC_BLOCK;
    let mut t = vec![0u64; tlen];
    let mut y = vec![0u64; w];
    let mut x = two_m.clone();
    let mut y_prev = two_m;
    let (mut a, mut b) = (one_m.clone(), one_m.clone());
    for i in 0..full {
        ladder_step(&mut ops, &e, bits, i, &mut x, &mut y, &mut t);
        if (i + 1) % GEC_BLOCK == 0 {
            ops.mul(&a, &y_prev, &mut y, &mut t);
            a.copy_from_slice(&y);
            ops.mul(&b, &x, &mut y, &mut t);
            b.copy_from_slice(&y);
            y_prev.copy_from_slice(&x);
        }
    }
    // The tail belongs to no block: compute it twice, on x and on the copy of it in y_prev.
    for i in full..iters {
        ladder_step(&mut ops, &e, bits, i, &mut x, &mut y, &mut t);
    }
    for i in full..iters {
        ladder_step(&mut ops, &e, bits, i, &mut y_prev, &mut y, &mut t);
    }
    if x != y_prev {
        return Checked::Alarm;
    }
    if blocks > 0 {
        // A^(2^40) * 2^(sum V) == B.
        for _ in 0..GEC_BLOCK {
            ops.sqr(&a, &mut y, &mut t);
            a.copy_from_slice(&y);
        }
        let sum = gec_window_sum_words(&e.to_u32_digits(), bits, GEC_BLOCK);
        let mut p = one_m.clone();
        for k in (0..64 - sum.leading_zeros()).rev() {
            ops.sqr(&p, &mut y, &mut t);
            p.copy_from_slice(&y);
            if (sum >> k) & 1 == 1 {
                ops.double(&mut p);
            }
        }
        ops.mul(&a, &p, &mut y, &mut t);
        if y != b {
            return Checked::Alarm;
        }
    }
    if x == one_m {
        Checked::Prp
    } else {
        Checked::Composite
    }
}

/// [`fermat_base2_gec`] for every `n`, in order, with the threading of [`fermat_batch`]
/// (`threads == 0` uses all cores).
pub fn fermat_batch_gec(ns: &[BigUint], threads: usize) -> Vec<Checked> {
    let threads = if threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        threads
    };
    let threads = threads.min(ns.len()).max(1);
    let per = ns.len().div_ceil(threads).max(1);
    let mut out = vec![Checked::Composite; ns.len()];
    std::thread::scope(|s| {
        for (chunk_in, chunk_out) in ns.chunks(per).zip(out.chunks_mut(per)) {
            s.spawn(move || {
                for (n, o) in chunk_in.iter().zip(chunk_out.iter_mut()) {
                    *o = fermat_base2_gec(n);
                }
            });
        }
    });
    out
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use bigmont_oracle::XorShift64;

    /// The assembly row against the portable row on every length from 0 to 40 (all remainders of
    /// the four-limb pass, empty and shorter-than-a-pass rows included) and on operands that push
    /// both carry chains to their extremes: all-ones limbs, alternating all-ones and zero, zero
    /// accumulators. Skips, saying so, on a CPU without BMI2 and ADX.
    #[test]
    fn the_adx_row_matches_the_portable_row() {
        if Kernel::detect() == Kernel::Portable {
            eprintln!("no BMI2 + ADX on this CPU: the assembly row was not tested");
            return;
        }
        let mut rng = XorShift64::new(99);
        let limbs = |kind: usize, len: usize, rng: &mut XorShift64| -> Vec<u64> {
            (0..len)
                .map(|i| match kind {
                    0 => rng.next_u64(),
                    1 => u64::MAX,
                    2 => {
                        if i % 2 == 0 {
                            u64::MAX
                        } else {
                            0
                        }
                    }
                    3 => 0,
                    _ => rng.next_u64() & 0xff,
                })
                .collect()
        };
        for len in 0..=40 {
            for kind_t in 0..5 {
                for kind_s in 0..5 {
                    for m in [0, 1, u64::MAX, 0x8000_0000_0000_0000, rng.next_u64()] {
                        let t = limbs(kind_t, len, &mut rng);
                        let src = limbs(kind_s, len, &mut rng);
                        let (mut t_portable, mut t_adx) = (t.clone(), t.clone());
                        let c_portable = addmul_row_portable(&mut t_portable, &src, m);
                        // SAFETY: `Kernel::detect` just reported BMI2 and ADX.
                        let c_adx = unsafe { adx::addmul_row(&mut t_adx, &src, m) };
                        assert_eq!(
                            (c_portable, &t_portable),
                            (c_adx, &t_adx),
                            "len={len} kinds=({kind_t},{kind_s}) m={m:x} t={t:x?} src={src:x?}"
                        );
                    }
                }
            }
        }
    }
}
