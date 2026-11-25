//! Informational throughput: single-thread and all-core base-2 Fermat tests/s at 2,029 bits.
use bigmont_oracle::{XorShift64, random_odd_modulus};
use std::time::Instant;

fn main() {
    let mut rng = XorShift64::new(2029);
    let ns: Vec<_> = (0..40)
        .map(|_| random_odd_modulus(&mut rng, 2029))
        .collect();

    let t0 = Instant::now();
    let passed = ns.iter().filter(|n| bigmont_cpu::fermat_base2(n)).count();
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "1 thread : {} tests in {:.3}s = {:.1} tests/s ({passed} passed)",
        ns.len(),
        dt,
        ns.len() as f64 / dt
    );

    let many: Vec<_> = ns.iter().cycle().take(400).cloned().collect();
    let t0 = Instant::now();
    let v = bigmont_cpu::fermat_batch(&many, 0);
    let dt = t0.elapsed().as_secs_f64();
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!(
        "{cores} cores : {} tests in {:.3}s = {:.1} tests/s ({} passed)",
        many.len(),
        dt,
        many.len() as f64 / dt,
        v.iter().filter(|&&b| b).count()
    );
}
