use bigmont_oracle::{XorShift64, random_odd_modulus, random_prp};
use num_bigint::BigUint;
use num_traits::One;

fn mersenne(p: u32) -> BigUint {
    (BigUint::one() << p) - 1u32
}

#[test]
fn matches_the_oracle_on_small_and_edge_values() {
    for v in [
        0u32,
        1,
        2,
        3,
        4,
        5,
        9,
        15,
        91,
        100,
        341,
        561,
        1105,
        65_537,
        4_294_967_295,
    ] {
        let n = BigUint::from(v);
        assert_eq!(
            bigmont_cpu::fermat_base2(&n),
            bigmont_oracle::fermat_base2(&n),
            "n = {v}"
        );
    }
}

#[test]
fn every_odd_number_below_8192_matches_the_oracle() {
    // The engine sends 2- to 12-bit numbers to the GPU by default, but the CPU stays the fallback
    // and the integrity re-check for them, so it must be right on every tiny modulus too.
    for v in (3u32..8192).step_by(2) {
        let n = BigUint::from(v);
        assert_eq!(
            bigmont_cpu::fermat_base2(&n),
            bigmont_oracle::fermat_base2(&n),
            "n = {v}"
        );
    }
    for k in [31u32, 32, 33, 63, 64, 65] {
        for n in [
            mersenne(k),
            (BigUint::one() << (k - 1)) + 1u32,
            (BigUint::one() << k) + 1u32,
        ] {
            assert_eq!(
                bigmont_cpu::fermat_base2(&n),
                bigmont_oracle::fermat_base2(&n),
                "n = {n}"
            );
        }
    }
}

#[test]
fn mersenne_numbers() {
    // For prime p, n = 2^p-1 has ord_n(2) = p, and p divides n-1 = 2^p-2 (Fermat's little theorem), so
    // 2^(n-1) = 1 mod n: every such n passes the base-2 Fermat test even when it is composite (2^67-1, 2^2029-1).
    // A composite exponent k gives n = 2^k-1 with n-1 = 2^k-2 not divisible by k in general, so these fail (2^2030-1 is checked against the oracle).
    for (p, passes) in [
        (61u32, true),
        (127, true),
        (521, true),
        (607, true),
        (1279, true),
        (2203, true),
        (67, true),    // 2^67-1 is composite but a base-2 pseudoprime
        (2029, true), // 2029 is prime, so 2^2029-1 is a base-2 pseudoprime (not Mersenne-prime-ness)
        (2030, false), // composite exponent: 2^k-1 with k=2030 fails base-2 Fermat
    ] {
        let n = mersenne(p);
        assert_eq!(bigmont_cpu::fermat_base2(&n), passes, "2^{p}-1");
        assert_eq!(
            bigmont_cpu::fermat_base2(&n),
            bigmont_oracle::fermat_base2(&n),
            "oracle 2^{p}-1"
        );
    }
}

#[test]
fn random_numbers_at_many_sizes() {
    let mut rng = XorShift64::new(77);
    for bits in [33u32, 64, 65, 127, 129, 512, 1019, 2029, 2203] {
        for _ in 0..6 {
            let n = random_odd_modulus(&mut rng, bits);
            assert_eq!(
                bigmont_cpu::fermat_base2(&n),
                bigmont_oracle::fermat_base2(&n),
                "{bits} bits"
            );
        }
        let p = random_prp(&mut rng, bits.max(40));
        assert!(bigmont_cpu::fermat_base2(&p), "random PRP at {bits} bits");
    }
}

#[test]
fn batch_preserves_order_for_any_thread_count() {
    let mut rng = XorShift64::new(4);
    let mut ns: Vec<BigUint> = (0..30).map(|_| random_odd_modulus(&mut rng, 600)).collect();
    ns.insert(7, random_prp(&mut rng, 600));
    ns.insert(20, random_prp(&mut rng, 600));
    let want: Vec<bool> = ns.iter().map(bigmont_oracle::fermat_base2).collect();
    for threads in [0usize, 1, 3, 64] {
        assert_eq!(
            bigmont_cpu::fermat_batch(&ns, threads),
            want,
            "threads {threads}"
        );
    }
}

// ---- the Gerbicz-checked ladder (`fermat_base2_gec`) ----

use bigmont_cpu::{
    Checked, Mont, MontOps, fermat_base2_gec, fermat_base2_gec_with, fermat_batch_gec,
};
use bigmont_oracle::GEC_BLOCK;
use std::cell::Cell;

fn plain(n: &BigUint) -> Checked {
    if bigmont_cpu::fermat_base2(n) {
        Checked::Prp
    } else {
        Checked::Composite
    }
}

#[test]
fn a_clean_checked_run_equals_the_plain_run_at_every_size() {
    let mut rng = XorShift64::new(31);
    for bits in [2u32, 3, 40, 41, 42, 81, 130, 300, 1019] {
        let mut ns: Vec<BigUint> = (0..6).map(|_| random_odd_modulus(&mut rng, bits)).collect();
        ns.push(random_prp(&mut rng, bits));
        for n in &ns {
            assert_eq!(fermat_base2_gec(n), plain(n), "{bits} bits: n = {n}");
        }
        let prp = ns.last().unwrap();
        assert_eq!(fermat_base2_gec(prp), Checked::Prp, "{bits}-bit PRP");
    }
    // Every odd number below 4096 (no full block below 42 bits: the whole run is the tail), and
    // edges where the last block ends exactly at the last iteration (41, 81, 121 bits).
    for v in (3u32..4096).step_by(2) {
        let n = BigUint::from(v);
        assert_eq!(fermat_base2_gec(&n), plain(&n), "n = {v}");
    }
    for k in [41u32, 64, 81, 121, 127, 521] {
        for n in [mersenne(k), (BigUint::one() << (k - 1)) + 1u32] {
            assert_eq!(fermat_base2_gec(&n), plain(&n), "n = {n}");
        }
    }
}

#[test]
fn degenerate_inputs_follow_the_plain_rules() {
    for (v, want) in [
        (0u32, Checked::Composite),
        (1, Checked::Composite),
        (2, Checked::Prp),
        (4, Checked::Composite),
        (100, Checked::Composite),
    ] {
        assert_eq!(fermat_base2_gec(&BigUint::from(v)), want, "n = {v}");
    }
    let even = (BigUint::one() << 1019u32) + 2u32;
    assert_eq!(fermat_base2_gec(&even), Checked::Composite);
}

#[test]
fn the_checked_batch_preserves_order_for_any_thread_count() {
    let mut rng = XorShift64::new(8);
    let mut ns: Vec<BigUint> = (0..30).map(|_| random_odd_modulus(&mut rng, 600)).collect();
    ns.insert(7, random_prp(&mut rng, 600));
    ns.insert(20, random_prp(&mut rng, 600));
    ns.extend([0u32, 1, 2, 3, 4].map(BigUint::from));
    let want: Vec<Checked> = ns.iter().map(fermat_base2_gec).collect();
    assert!(want.contains(&Checked::Prp));
    assert_eq!(
        want,
        ns.iter().map(plain).collect::<Vec<_>>(),
        "clean runs equal the plain verdicts"
    );
    for threads in [0usize, 1, 3, 64] {
        assert_eq!(fermat_batch_gec(&ns, threads), want, "threads {threads}");
    }
    assert!(fermat_batch_gec(&[], 4).is_empty());
}

/// A faulty `MontOps`: the real `Mont`, counting every operation; the output of operation number
/// `fault_at` (0-based) gets its lowest bit flipped and is reduced mod n, so it stays a valid
/// operand (below n) that differs from the right value mod n.
struct Faulty<'a> {
    inner: Mont,
    n: BigUint,
    fault_at: u64,
    ops: &'a Cell<u64>,
}

impl Faulty<'_> {
    fn after(&mut self, out: &mut [u64]) {
        let k = self.ops.get();
        self.ops.set(k + 1);
        if k != self.fault_at {
            return;
        }
        let v = out
            .iter()
            .rev()
            .fold(BigUint::ZERO, |acc, &l| (acc << 64u32) + l);
        let mut d = ((v ^ BigUint::one()) % &self.n).to_u64_digits();
        d.resize(out.len(), 0);
        out.copy_from_slice(&d);
    }
}

impl MontOps for Faulty<'_> {
    fn mul(&mut self, a: &[u64], b: &[u64], out: &mut [u64], t: &mut [u64]) {
        self.inner.mul(a, b, out, t);
        self.after(out);
    }
    fn sqr(&mut self, a: &[u64], out: &mut [u64], t: &mut [u64]) {
        self.inner.sqr(a, out, t);
        self.after(out);
    }
    fn double(&mut self, x: &mut [u64]) {
        self.inner.double(x);
        self.after(x);
    }
}

/// The checked run of `n` with a fault at operation `fault_at`, and how many operations it ran.
fn faulty_run(n: &BigUint, fault_at: u64) -> (Checked, u64) {
    let ops = Cell::new(0);
    let got = fermat_base2_gec_with(n, |m| Faulty {
        inner: m,
        n: n.clone(),
        fault_at,
        ops: &ops,
    });
    (got, ops.get())
}

/// Where each phase of the checked run starts, in operations, by the documented algorithm: the
/// ladder (a squaring per iteration plus a doubling per set exponent bit, and the `A`, `B`
/// multiplies after each block), the tail on `x`, the tail on the copy, the 40 squarings of `A`,
/// the ladder for `2^(sum V)` from `1~` and the final multiply.
struct Phases {
    block_start: Vec<u64>,
    tail_x: u64,
    tail_copy: u64,
    check_sqr: u64,
    pow: u64,
    final_mul: u64,
    total: u64,
}

fn phases(n: &BigUint) -> Phases {
    let bits = n.bits() as u32;
    let e = n - 1u32;
    let iter_ops = |i: u32| 1 + u64::from(e.bit(u64::from(bits - 2 - i)));
    let blocks = (bits - 1) / GEC_BLOCK;
    let mut k = 0u64;
    let mut block_start = Vec::new();
    for j in 0..blocks {
        block_start.push(k);
        k += (j * GEC_BLOCK..(j + 1) * GEC_BLOCK)
            .map(iter_ops)
            .sum::<u64>()
            + 2;
    }
    let tail: u64 = (blocks * GEC_BLOCK..bits - 1).map(iter_ops).sum();
    let tail_x = k;
    let tail_copy = k + tail;
    let check_sqr = tail_copy + tail;
    let (pow, final_mul, total) = if blocks == 0 {
        (check_sqr, check_sqr, check_sqr)
    } else {
        let sum = bigmont_oracle::gec_window_sum(&e, bits, GEC_BLOCK);
        let pow = check_sqr + u64::from(GEC_BLOCK);
        let pow_ops = u64::from(64 - sum.leading_zeros()) + u64::from(sum.count_ones());
        (pow, pow + pow_ops, pow + pow_ops + 1)
    };
    Phases {
        block_start,
        tail_x,
        tail_copy,
        check_sqr,
        pow,
        final_mul,
        total,
    }
}

#[test]
fn the_phase_model_counts_exactly_the_operations_of_a_clean_run() {
    let mut rng = XorShift64::new(12);
    for bits in [2u32, 20, 40, 41, 42, 81, 300, 1019] {
        let n = random_odd_modulus(&mut rng, bits);
        let (got, ops) = faulty_run(&n, u64::MAX);
        assert_eq!(got, plain(&n), "{bits} bits: a run with no fault is clean");
        assert_eq!(ops, phases(&n).total, "{bits} bits");
    }
}

#[test]
fn a_fault_in_any_phase_alarms_and_a_fault_past_the_end_does_not() {
    let mut rng = XorShift64::new(13);
    let cases = [
        random_prp(&mut rng, 1019),
        random_odd_modulus(&mut rng, 1019),
        random_prp(&mut rng, 300),
        random_odd_modulus(&mut rng, 302),
    ];
    for n in &cases {
        let p = phases(n);
        let blocks = p.block_start.len();
        assert!(blocks >= 7 && p.tail_x < p.tail_copy, "{} bits", n.bits());
        let mid = p.block_start[blocks / 2];
        let after_mid = p.block_start[blocks / 2 + 1];
        let last = p.block_start[blocks - 1];
        for (what, k) in [
            ("the first op of block 0", 0),
            ("inside block 0", 17),
            ("a middle block", mid + 5),
            ("the A multiply of a middle block", after_mid - 2),
            ("the B multiply of a middle block", after_mid - 1),
            ("the last block", last + 3),
            ("the last op of the last block", p.tail_x - 1),
            ("the tail on x", p.tail_x + 1),
            ("the tail on the copy", p.tail_copy + 1),
            ("the last tail op", p.check_sqr - 1),
            ("the squarings of A", p.check_sqr + 7),
            ("the last squaring of A", p.pow - 1),
            ("the 2^sumV ladder", p.pow + 3),
            ("the final multiply", p.final_mul),
        ] {
            assert_eq!(
                faulty_run(n, k).0,
                Checked::Alarm,
                "{}-bit n, fault at op {k} ({what})",
                n.bits()
            );
        }
        for k in [p.total, p.total + 1, u64::MAX] {
            assert_eq!(
                faulty_run(n, k),
                (plain(n), p.total),
                "fault at {k}: past the end"
            );
        }
    }
}

#[test]
fn every_single_fault_alarms_on_small_candidates_including_probable_primes() {
    // A 20-bit candidate has no full block: the tail recomputation covers the whole run. The others
    // sit on the block and tail boundaries: 41 bits (one block, no tail), 42 (one block and a
    // 1-iteration tail), 81 (two blocks), 130 (three blocks and a 9-iteration tail).
    let mut rng = XorShift64::new(14);
    for bits in [20u32, 41, 42, 81, 130] {
        for n in [
            random_prp(&mut rng, bits),
            random_odd_modulus(&mut rng, bits),
        ] {
            let total = phases(&n).total;
            for k in 0..total {
                assert_eq!(
                    faulty_run(&n, k).0,
                    Checked::Alarm,
                    "{bits}-bit n = {n}, fault at op {k} of {total}"
                );
            }
            assert_eq!(faulty_run(&n, total).0, plain(&n));
        }
    }
}

// ---- Montgomery multiply and square against their definition --------------------------------

fn to_big(x: &[u64]) -> BigUint {
    x.iter()
        .rev()
        .fold(BigUint::ZERO, |acc, &l| (acc << 64u32) + l)
}

fn to_limbs(x: &BigUint, w: usize) -> Vec<u64> {
    let mut v = x.to_u64_digits();
    v.resize(w, 0);
    v
}

/// Odd moduli of exactly `w` limbs that stress the carry paths: random full-width ones, one whose
/// top limb is 1, the all-ones modulus 2^(64 w) - 1, and 2^(64 w - 1) + 1.
fn moduli(w: usize, rng: &mut XorShift64) -> Vec<BigUint> {
    let full = BigUint::one() << (64 * w);
    let mut v = vec![
        random_odd_modulus(rng, 64 * w as u32),
        random_odd_modulus(rng, 64 * w as u32),
        random_odd_modulus(rng, 64 * w as u32 - 5),
        &full - 1u32,
        (BigUint::one() << (64 * w - 1)) + 1u32,
    ];
    if w >= 2 {
        v.push((BigUint::one() << (64 * (w - 1))) + 1u32);
    }
    v
}

/// Operands below `n` that sit on limb boundaries or are all ones, plus random ones.
fn operands(n: &BigUint, rng: &mut XorShift64) -> Vec<BigUint> {
    let w = n.to_u64_digits().len();
    let mut v = vec![
        BigUint::ZERO,
        BigUint::one(),
        n - 1u32,
        n - 2u32 + (n & BigUint::one()),
    ];
    for k in [1usize, 63, 64, 65, 64 * w - 1, 64 * w] {
        let x = (BigUint::one() << k) - 1u32;
        v.push(x % n);
        v.push((BigUint::one() << k) % n);
    }
    for _ in 0..6 {
        v.push(bigmont_oracle::random_below(rng, n));
    }
    v
}

/// `out` is the Montgomery product of `a` and `b`: below n and out * R = a * b (mod n).
fn check_product(n: &BigUint, w: usize, a: &BigUint, b: &BigUint, out: &[u64], what: &str) {
    let o = to_big(out);
    assert!(
        &o < n,
        "{what}: result not reduced (w={w}, n={n:x}, a={a:x}, b={b:x})"
    );
    assert_eq!(
        (&o << (64 * w)) % n,
        (a * b) % n,
        "{what} (w={w}, n={n:x}, a={a:x}, b={b:x})"
    );
}

#[test]
fn mul_and_sqr_match_the_definition_and_each_other() {
    let mut rng = XorShift64::new(4040);
    for w in [1usize, 2, 3, 4, 5, 8, 16, 31, 32, 33] {
        for n in moduli(w, &mut rng) {
            let m = Mont::new(&to_limbs(&n, w));
            let mut t = vec![0u64; m.scratch_len()];
            let ops = operands(&n, &mut rng);
            for a in &ops {
                let al = to_limbs(a, w);
                let mut sq = vec![0u64; w];
                m.sqr(&al, &mut sq, &mut t);
                check_product(&n, w, a, a, &sq, "sqr");
                let mut prod = vec![0u64; w];
                m.mul(&al, &al, &mut prod, &mut t);
                assert_eq!(sq, prod, "sqr != mul(a, a) (w={w}, n={n:x}, a={a:x})");
                for b in ops.iter().take(8) {
                    let mut out = vec![0u64; w];
                    m.mul(&al, &to_limbs(b, w), &mut out, &mut t);
                    check_product(&n, w, a, b, &out, "mul");
                }
            }
        }
    }
}

#[test]
fn mul_works_with_the_minimal_scratch_of_w_plus_2() {
    let mut rng = XorShift64::new(4041);
    for w in [1usize, 2, 7, 32] {
        let n = random_odd_modulus(&mut rng, 64 * w as u32);
        let m = Mont::new(&to_limbs(&n, w));
        let mut t = vec![0u64; w + 2];
        let (a, b) = (
            bigmont_oracle::random_below(&mut rng, &n),
            bigmont_oracle::random_below(&mut rng, &n),
        );
        let mut out = vec![0u64; w];
        m.mul(&to_limbs(&a, w), &to_limbs(&b, w), &mut out, &mut t);
        check_product(&n, w, &a, &b, &out, "mul with w + 2 scratch");
    }
}

// ---- the portable and the hardware (mulx + adcx/adox) kernels ------------------------------

/// Squarings of every operand under both kernels give the same limbs, which also equal the
/// definition. On a CPU without BMI2 and ADX both contexts run the portable kernel, so the test
/// only covers that one (it says so rather than claiming otherwise).
#[test]
fn the_portable_and_the_hardware_kernel_agree() {
    let mut rng = XorShift64::new(5150);
    let mut hardware_seen = false;
    // Every width from 1 to 12 covers all remainders of the 4-limb unrolled loop and rows shorter
    // than one iteration; 31 to 33 are the sizes the Fermat test actually uses.
    for w in (1usize..=12).chain([31, 32, 33]) {
        for n in moduli(w, &mut rng) {
            let limbs = to_limbs(&n, w);
            let auto = Mont::new(&limbs);
            let portable = Mont::new(&limbs).portable();
            assert!(!portable.uses_adx());
            hardware_seen |= auto.uses_adx();
            let mut t = vec![0u64; auto.scratch_len()];
            for a in operands(&n, &mut rng) {
                let al = to_limbs(&a, w);
                let (mut x, mut y) = (vec![0u64; w], vec![0u64; w]);
                auto.sqr(&al, &mut x, &mut t);
                portable.sqr(&al, &mut y, &mut t);
                assert_eq!(x, y, "kernels disagree (w={w}, n={n:x}, a={a:x})");
                check_product(&n, w, &a, &a, &x, "sqr");
            }
        }
    }
    eprintln!(
        "hardware kernel exercised: {hardware_seen} (false means this CPU has no BMI2 + ADX)"
    );
}

/// The whole Fermat ladder gives the same verdicts and the same Gerbicz verdicts under both
/// kernels (`fermat_base2` and the checked run use the auto-selected kernel; here they are compared
/// with the oracle, which never uses either).
#[test]
fn verdicts_match_the_oracle_at_the_sizes_the_kernel_unrolls_over() {
    let mut rng = XorShift64::new(5151);
    for bits in [60u32, 64, 65, 128, 200, 255, 256, 257, 511, 513, 1019, 2029] {
        for _ in 0..5 {
            let n = random_odd_modulus(&mut rng, bits);
            assert_eq!(
                bigmont_cpu::fermat_base2(&n),
                bigmont_oracle::fermat_base2(&n),
                "{bits} bits"
            );
        }
        let p = random_prp(&mut rng, bits.max(40));
        assert!(bigmont_cpu::fermat_base2(&p), "random PRP at {bits} bits");
        assert_eq!(
            fermat_base2_gec(&p),
            Checked::Prp,
            "checked PRP at {bits} bits"
        );
    }
}
