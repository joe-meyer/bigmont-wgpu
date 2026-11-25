//! Compares the limb schemes (13-bit limbs, 32-bit limbs with u64 products) on the selected adapter
//! at one operand size (default the 2,029-bit class), each through `KernelConfig::for_bits_scheme`
//! and `gpu::run`.
//!
//!   BIGMONT_ADAPTER=4070 cargo run --release -p bigmont-core --example scheme_compare
//!   ... -- --bits 2029 --count 65536 --runs 3 --tpis 4,8,16 --wgs 64,128,256 --schemes 13,32
//!
//! Slope method: a dispatch of 2 squarings is timed (best of 5), the squarings per dispatch are
//! doubled while the predicted time stays within 0.4 s and the last one is under 0.2 s, and the rate
//! is instances x (S_long - 2) / (t_long - t_short). After every timed configuration a sample of a
//! separate 5-squaring run is compared with the oracle. A failing run prints no rate and the exit
//! status is nonzero. Schemes the adapter cannot run (see `KernelConfig::requires`) are skipped
//! with a note.

use bigmont_core::gpu::{Batch, Gpu, run};
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_core::scheme::LimbScheme;
use num_bigint::BigUint;

const BUDGET: f64 = 0.4;
const TARGET: f64 = 0.2;
const MAX_SQUARINGS: u32 = 1 << 14;
const VERIFY_SQUARINGS: u32 = 5;

fn arg<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    args.iter().position(|a| a == name).map(|i| args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or_else(|| panic!("{name} needs a value")))
}

fn list<T: std::str::FromStr>(args: &[String], name: &str, default: &str) -> Vec<T> {
    let v: String = arg(args, name).unwrap_or_else(|| default.to_string());
    v.split(',').map(|t| t.trim().parse().unwrap_or_else(|_| panic!("bad {name} entry '{t}'"))).collect()
}

fn sample_indices(count: u32, per_wg: u32) -> Vec<usize> {
    let count = count as usize;
    let mut set = std::collections::BTreeSet::new();
    set.extend(0..4.min(count));
    set.extend(count.saturating_sub(4)..count);
    set.extend((1..=16).map(|i| i * (count - 1) / 17));
    set.extend([per_wg as usize - 1, per_wg as usize, count / 2].into_iter().filter(|&i| i < count));
    set.into_iter().collect()
}

/// Slope method over a timing closure `time(squarings, repeats) -> seconds`.
/// Returns (slope in instance-squarings per second, long squarings, long seconds).
fn slope(count: u32, mut time: impl FnMut(u32, u32) -> Result<f64, String>) -> Result<(f64, u32, f64), String> {
    let short = time(2, 5)?;
    let (mut s, mut t) = (2u32, short);
    while t < TARGET {
        let next = s * 2;
        if next > MAX_SQUARINGS || t * f64::from(next) / f64::from(s) > BUDGET {
            break;
        }
        t = time(next, 3)?;
        s = next;
    }
    let extra = t - short;
    if s <= 2 || extra <= 0.02 * short {
        return Err(format!("long dispatch ({s} squarings, {:.3} ms) not measurably longer than the short one ({:.3} ms)", t * 1e3, short * 1e3));
    }
    Ok((f64::from(count) * f64::from(s - 2) / extra, s, t))
}

fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = if s.len() % 2 == 1 { s[s.len() / 2] } else { (s[s.len() / 2 - 1] + s[s.len() / 2]) / 2.0 };
    (s[0], med, s[s.len() - 1])
}

/// Runs `VERIFY_SQUARINGS` squarings and compares a sample of instances with the oracle chain.
fn verify(gpu: &Gpu, cfg: KernelConfig, values: &[(BigUint, BigUint)], batch: &Batch) -> Result<usize, String> {
    let out = run(gpu, cfg, batch, VERIFY_SQUARINGS, 1)?;
    let l = cfg.limbs as usize;
    let idx = sample_indices(values.len() as u32, cfg.instances_per_workgroup());
    let bad: Vec<String> = std::thread::scope(|s| {
        let hs: Vec<_> = idx
            .iter()
            .map(|&i| {
                let out = &out;
                s.spawn(move || {
                    let (n, x0) = &values[i];
                    let limbs = &out.x[i * l..(i + 1) * l];
                    let got = cfg.scheme.unpack(limbs);
                    if !cfg.scheme.normalised(limbs) {
                        Some(format!("{i} (limb out of range)"))
                    } else if got >= n * 2u32 {
                        Some(format!("{i} (not below 2n)"))
                    } else if &got % n != cfg.scheme.chain(n, x0, VERIFY_SQUARINGS, l) {
                        Some(format!("{i} (wrong value)"))
                    } else {
                        None
                    }
                })
            })
            .collect();
        hs.into_iter().filter_map(|h| h.join().unwrap()).collect()
    });
    if bad.is_empty() { Ok(idx.len()) } else { Err(format!("{} of {} sampled instances failed: {}", bad.len(), idx.len(), bad.join(", "))) }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let bits: u32 = arg(&args, "--bits").unwrap_or(2029);
    let count: u32 = arg(&args, "--count").unwrap_or(65536);
    let runs: usize = arg(&args, "--runs").unwrap_or(3);
    let tpis: Vec<u32> = list(&args, "--tpis", "4,8,16");
    let wgs: Vec<u32> = list(&args, "--wgs", "64,128,256");
    let schemes: Vec<LimbScheme> = list::<String>(&args, "--schemes", "13,32")
        .iter()
        .map(|s| LimbScheme::parse(s).unwrap_or_else(|| panic!("unknown scheme '{s}' (use 13 or 32)")))
        .collect();
    let gpu = Gpu::new().expect("no usable adapter");
    println!("adapter: {}", gpu.describe());
    println!("driver: {} {}", gpu.info.driver, gpu.info.driver_info);
    println!("SHADER_INT64: {}, subgroups: {}", gpu.has_int64, gpu.has_subgroups);
    println!("bits {bits}, instances {count}, runs {runs}");
    println!();
    let values = Batch::random_values(count, bits, 0xc00b_0000 + u64::from(bits));
    let mut failures = 0u32;
    // (scheme, tpi, best wg, median slope)
    let mut best: Vec<(LimbScheme, u32, u32, f64)> = Vec::new();

    println!("| scheme | limbs | tpi | wg | run | slope sq/s | long sq | long ms | verified |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for &scheme in &schemes {
        let name = format!("{}-bit", scheme.name());
        for &tpi in &tpis {
            let mut top: Option<(u32, Vec<f64>)> = None;
            for &wg in &wgs {
                let cfg = KernelConfig::for_bits_scheme(scheme, bits, tpi, wg, Variant::Unrolled);
                if let Err(e) = gpu.check_subgroups(tpi).and_then(|_| cfg.validate()).and_then(|_| cfg.check_caps(gpu.caps())) {
                    println!("| {name} | - | {tpi} | {wg} | - | skipped: {e} | | | |");
                    continue;
                }
                let batch = Batch::from_values_cfg(&values, &cfg);
                let mut slopes = Vec::new();
                for r in 0..runs {
                    match slope(count, |sq, rep| run(&gpu, cfg, &batch, sq, rep).map(|x| x.best_secs)) {
                        Ok((sl, s, t)) => match verify(&gpu, cfg, &values, &batch) {
                            Ok(v) => {
                                println!("| {name} | {} | {tpi} | {wg} | {r} | {sl:.4e} | {s} | {:.1} | {v} inst x {VERIFY_SQUARINGS} sq |", cfg.limbs, t * 1e3);
                                slopes.push(sl);
                            }
                            Err(e) => {
                                println!("| {name} | {} | {tpi} | {wg} | {r} | FAILED: {e} | | | |", cfg.limbs);
                                failures += 1;
                            }
                        },
                        Err(e) => {
                            println!("| {name} | {} | {tpi} | {wg} | {r} | FAILED: {e} | | | |", cfg.limbs);
                            failures += 1;
                        }
                    }
                }
                if slopes.len() == runs && top.as_ref().is_none_or(|(_, b)| stats(&slopes).1 > stats(b).1) {
                    top = Some((wg, slopes));
                }
            }
            if let Some((wg, s)) = top {
                best.push((scheme, tpi, wg, stats(&s).1));
            }
        }
    }
    println!();
    println!("Summary: median squarings/s at the best workgroup per (scheme, tpi).");
    println!();
    println!("| scheme | tpi | best wg | sq/s median | vs {}-bit at this tpi |", schemes[0].name());
    println!("|---|---|---|---|---|");
    for (scheme, tpi, wg, med) in &best {
        let ratio = if *scheme == schemes[0] { String::new() } else { best.iter().find(|r| r.0 == schemes[0] && r.1 == *tpi).map_or("n/a".to_string(), |r| format!("{:.2}", med / r.3)) };
        println!("| {}-bit | {tpi} | {wg} | {med:.4e} | {ratio} |", scheme.name());
    }
    let top = |s: LimbScheme| best.iter().filter(|r| r.0 == s).map(|r| r.3).fold(0.0, f64::max);
    for &scheme in schemes.iter().skip(1) {
        if top(schemes[0]) > 0.0 && top(scheme) > 0.0 {
            println!();
            println!("best {}-bit / best {}-bit: {:.2}", scheme.name(), schemes[0].name(), top(scheme) / top(schemes[0]));
        }
    }
    if failures > 0 {
        eprintln!("{failures} configuration run(s) failed");
        std::process::exit(1);
    }
}
