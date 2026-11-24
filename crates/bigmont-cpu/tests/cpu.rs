use bigmont_oracle::{XorShift64, random_odd_modulus, random_prp};
use num_bigint::BigUint;
use num_traits::One;

fn mersenne(p: u32) -> BigUint {
    (BigUint::one() << p) - 1u32
}

#[test]
fn matches_the_oracle_on_small_and_edge_values() {
    for v in [0u32, 1, 2, 3, 4, 5, 9, 15, 91, 100, 341, 561, 1105, 65_537, 4_294_967_295] {
        let n = BigUint::from(v);
        assert_eq!(bigmont_cpu::fermat_base2(&n), bigmont_oracle::fermat_base2(&n), "n = {v}");
    }
}

#[test]
fn mersenne_numbers() {
    // For prime p, n = 2^p-1 has ord_n(2) = p, and p divides n-1 = 2^p-2 (Fermat's little theorem), so
    // 2^(n-1) = 1 mod n: every such n passes the base-2 Fermat test even when it is composite (2^67-1, 2^2029-1).
    // A composite exponent k gives n = 2^k-1 with n-1 = 2^k-2 not divisible by k in general, so these fail (2^2030-1 is checked against the oracle).
    for (p, passes) in [
        (61u32, true), (127, true), (521, true), (607, true), (1279, true), (2203, true),
        (67, true),   // 2^67-1 is composite but a base-2 pseudoprime
        (2029, true), // 2029 is prime, so 2^2029-1 is a base-2 pseudoprime (not Mersenne-prime-ness)
        (2030, false), // composite exponent: 2^k-1 with k=2030 fails base-2 Fermat
    ] {
        let n = mersenne(p);
        assert_eq!(bigmont_cpu::fermat_base2(&n), passes, "2^{p}-1");
        assert_eq!(bigmont_cpu::fermat_base2(&n), bigmont_oracle::fermat_base2(&n), "oracle 2^{p}-1");
    }
}

#[test]
fn random_numbers_at_many_sizes() {
    let mut rng = XorShift64::new(77);
    for bits in [33u32, 64, 65, 127, 129, 512, 1019, 2029, 2203] {
        for _ in 0..6 {
            let n = random_odd_modulus(&mut rng, bits);
            assert_eq!(bigmont_cpu::fermat_base2(&n), bigmont_oracle::fermat_base2(&n), "{bits} bits");
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
        assert_eq!(bigmont_cpu::fermat_batch(&ns, threads), want, "threads {threads}");
    }
}
