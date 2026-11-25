//! Writes a `G2C1` candidate file for smoke tests: `COUNT` random odd numbers of `--bits N` bits
//! (default 2,029, in a 2,048-bit container; in general the next multiple of 32), then 3 probable
//! primes (small-prime filter plus the CPU Fermat test).
//!
//!   cargo run --release -p bigmont-cli --example make_candidates -- [--bits N] COUNT SEED OUT.bin

use bigmont_oracle::{XorShift64, random_odd_modulus};
use num_traits::Zero;
use std::path::Path;

fn main() {
    let mut raw: Vec<String> = std::env::args().skip(1).collect();
    let mut bits: u32 = 2029;
    if let Some(i) = raw.iter().position(|a| a == "--bits") {
        bits = raw.get(i + 1).and_then(|v| v.parse().ok()).filter(|&b| b >= 2).unwrap_or_else(|| {
            eprintln!("--bits needs a number of at least 2");
            std::process::exit(2);
        });
        raw.drain(i..=i + 1);
    }
    let [count, seed, out] = raw.as_slice() else {
        eprintln!("usage: make_candidates [--bits N] COUNT SEED OUTPUT");
        std::process::exit(2);
    };
    let count: usize = count.parse().expect("COUNT must be a number");
    let seed: u64 = seed.parse().expect("SEED must be a number");
    let mut rng = XorShift64::new(seed);
    let mut numbers: Vec<_> = (0..count).map(|_| random_odd_modulus(&mut rng, bits)).collect();

    let small: Vec<u32> = (3..1000u32).filter(|p| (2..*p).take_while(|d| d * d <= *p).all(|d| p % d != 0)).collect();
    let mut found = 0;
    while found < 3 {
        let n = random_odd_modulus(&mut rng, bits);
        if small.iter().any(|&p| (&n % p).is_zero()) || !bigmont_cpu::fermat_base2(&n) {
            continue;
        }
        numbers.push(n);
        found += 1;
    }
    let container = bits.div_ceil(32) * 32;
    bigmont_cli::formats::write_candidates(Path::new(out), container, &numbers).expect("write candidate file");
    eprintln!("wrote {} candidates ({count} random + 3 probable primes, {bits} bits, {container}-bit container) to {out}", numbers.len());
}
