//! Writes a Beacon candidate file for smoke tests: `COUNT` random odd 2,029-bit numbers in a
//! 2,048-bit container, then 3 probable primes (small-prime filter plus the CPU Fermat test).
//!
//!   cargo run --release -p bigmont-cli --example make_candidates -- COUNT SEED OUT.bin

use bigmont_oracle::{XorShift64, random_odd_modulus};
use num_traits::Zero;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [count, seed, out] = args.as_slice() else {
        eprintln!("usage: make_candidates COUNT SEED OUTPUT");
        std::process::exit(2);
    };
    let count: usize = count.parse().expect("COUNT must be a number");
    let seed: u64 = seed.parse().expect("SEED must be a number");
    let mut rng = XorShift64::new(seed);
    let mut numbers: Vec<_> = (0..count).map(|_| random_odd_modulus(&mut rng, 2029)).collect();

    let small: Vec<u32> = (3..1000u32).filter(|p| (2..*p).take_while(|d| d * d <= *p).all(|d| p % d != 0)).collect();
    let mut found = 0;
    while found < 3 {
        let n = random_odd_modulus(&mut rng, 2029);
        if small.iter().any(|&p| (&n % p).is_zero()) || !bigmont_cpu::fermat_base2(&n) {
            continue;
        }
        numbers.push(n);
        found += 1;
    }
    bigmont_cli::formats::write_candidates(Path::new(out), 2048, &numbers).expect("write candidate file");
    eprintln!("wrote {} candidates ({count} random + 3 probable primes) to {out}", numbers.len());
}
