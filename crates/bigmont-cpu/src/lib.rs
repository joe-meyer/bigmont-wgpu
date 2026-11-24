//! Pure-Rust Montgomery base-2 Fermat test with 64-bit limbs (CIOS), used as the CPU fallback and
//! as an independent check on GPU verdicts. Not constant time. Setup (R mod n) uses num-bigint.

use num_bigint::BigUint;
use num_traits::One;

/// Montgomery context for an odd modulus of `w` 64-bit limbs; R = 2^(64 w).
pub struct Mont {
    n: Vec<u64>,
    ninv: u64, // -n^-1 mod 2^64
    w: usize,
}

impl Mont {
    pub fn new(n: &[u64]) -> Mont {
        assert!(n[0] & 1 == 1, "modulus must be odd");
        let mut inv = n[0]; // correct to 3 bits
        for _ in 0..5 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(n[0].wrapping_mul(inv)));
        }
        Mont { n: n.to_vec(), ninv: inv.wrapping_neg(), w: n.len() }
    }

    /// out = a * b * R^-1 mod n for a, b < n. `t` is scratch of length w + 2.
    pub fn mul(&self, a: &[u64], b: &[u64], out: &mut [u64], t: &mut [u64]) {
        let w = self.w;
        t.iter_mut().for_each(|v| *v = 0);
        for i in 0..w {
            let mut c: u128 = 0;
            for j in 0..w {
                let s = u128::from(t[j]) + u128::from(a[j]) * u128::from(b[i]) + c;
                t[j] = s as u64;
                c = s >> 64;
            }
            let s = u128::from(t[w]) + c;
            t[w] = s as u64;
            t[w + 1] = (s >> 64) as u64;
            let m = t[0].wrapping_mul(self.ninv);
            let s = u128::from(t[0]) + u128::from(m) * u128::from(self.n[0]);
            let mut c = s >> 64;
            for j in 1..w {
                let s = u128::from(t[j]) + u128::from(m) * u128::from(self.n[j]) + c;
                t[j - 1] = s as u64;
                c = s >> 64;
            }
            let s = u128::from(t[w]) + c;
            t[w - 1] = s as u64;
            t[w] = t[w + 1] + (s >> 64) as u64;
        }
        out.copy_from_slice(&t[..w]);
        if t[w] != 0 || ge(out, &self.n) {
            sub_in_place(out, &self.n);
        }
    }

    pub fn sqr(&self, a: &[u64], out: &mut [u64], t: &mut [u64]) {
        self.mul(a, a, out, t)
    }

    /// x = 2x mod n for x < n.
    fn double(&self, x: &mut [u64]) {
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
    let r = (BigUint::one() << (64 * w)) % n; // Montgomery form of 1
    let one_m = to_limbs(&r, w);
    let mut x = to_limbs(&((&r << 1u32) % n), w); // Montgomery form of 2 (top exponent bit)
    let e = n - 1u32;
    let nbits = e.bits();
    let mut t = vec![0u64; w + 2];
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
    let threads = if threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { threads };
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
