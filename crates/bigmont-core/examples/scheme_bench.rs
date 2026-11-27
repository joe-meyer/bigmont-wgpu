//! Limb-scheme experiment: squarings per second of the one-thread-per-number kernels
//! (`bigmont_core::onethread`), scheme A (13-bit limbs, 24-bit multiply-add) against scheme B
//! (32-bit limbs, u64 products) and B2 (B with fused passes), at 256, 384, 512 and 1024 bits.
//!
//!   BIGMONT_ADAPTER=4070 cargo run --release -p bigmont-core --example scheme_bench
//!   ... -- --runs 3 --count 262144 --sizes 256,384,512,1024 --wgs 64,128,256 --schemes A,B,B2
//!
//! Method (the slope method of the `spike` example): per (scheme, bits, workgroup) and run, the
//! starting values are re-uploaded, a dispatch of 2 squarings is timed (best of 5), then the
//! squarings per dispatch are doubled while the next dispatch is predicted (linear scaling) to stay
//! within 0.4 s and the current one is under 0.2 s (each best of 3). The rate is the slope between
//! the short and the longest dispatch: instances x (S_long - 2) / (t_long - t_short), which removes
//! the fixed per-dispatch cost. After every run a sample of the TIMED output (24 instances: first,
//! last, spread, workgroup boundaries) is compared with the oracle; a failing run prints no rate
//! and makes the exit status nonzero. Times are wall-clock around submit plus a blocking poll.
//!
//! The summary uses, per (scheme, bits), the workgroup with the best median slope, and prints min /
//! median / max over the runs at that workgroup; B/A is the ratio of medians.

use bigmont_core::gpu::{Batch, Gpu};
use bigmont_core::onethread::{OneThread, Scheme, Session};
use bigmont_oracle::{LIMB_MASK, mont_square_chain_bits};
use num_bigint::BigUint;

const BUDGET: f64 = 0.4;
const TARGET: f64 = 0.2;
const MAX_SQUARINGS: u32 = 1 << 16;

fn arg<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    args.iter().position(|a| a == name).map(|i| args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or_else(|| panic!("{name} needs a value")))
}

fn list<T: std::str::FromStr>(args: &[String], name: &str, default: &str) -> Vec<T> {
    let v: String = arg(args, name).unwrap_or_else(|| default.to_string());
    v.split(',').map(|t| t.trim().parse().unwrap_or_else(|_| panic!("bad {name} entry '{t}'"))).collect()
}

fn sample_indices(count: u32, wg: u32) -> Vec<usize> {
    let count = count as usize;
    let mut set = std::collections::BTreeSet::new();
    set.extend(0..4.min(count));
    set.extend(count.saturating_sub(4)..count);
    set.extend((1..=16).map(|i| i * (count - 1) / 17));
    set.extend([wg as usize - 1, wg as usize, count / 2].into_iter().filter(|&i| i < count));
    set.into_iter().collect()
}

/// Compares sampled instances of `out` with the oracle after `squarings` squarings.
fn verify(k: &OneThread, values: &[(BigUint, BigUint)], out: &[u32], squarings: u32) -> Result<usize, String> {
    let idx = sample_indices(values.len() as u32, k.wg);
    let l = k.limbs();
    let bad: Vec<String> = std::thread::scope(|s| {
        let handles: Vec<_> = idx
            .iter()
            .map(|&i| {
                s.spawn(move || {
                    let (n, x0) = &values[i];
                    let limbs = &out[i * l..(i + 1) * l];
                    let got = k.decode(limbs);
                    let why = if k.scheme.is_limb13() && limbs.iter().any(|&v| v > LIMB_MASK) {
                        Some("limbs not normalised")
                    } else if k.scheme.is_limb13() && got >= n * 2u32 {
                        Some("not below 2n")
                    } else if !k.scheme.is_limb13() && &got >= n {
                        Some("not below n")
                    } else if &got % n != mont_square_chain_bits(n, x0, squarings, k.r_bits()) {
                        Some("wrong value")
                    } else {
                        None
                    };
                    why.map(|w| format!("{i} ({w})"))
                })
            })
            .collect();
        handles.into_iter().filter_map(|h| h.join().unwrap()).collect()
    });
    if bad.is_empty() { Ok(idx.len()) } else { Err(format!("{} of {} sampled instances failed after {squarings} squarings: {}", bad.len(), idx.len(), bad.join(", "))) }
}

struct Measure {
    slope: f64,
    short_ms: f64,
    long_sq: u32,
    long_ms: f64,
    verified: usize,
    total_sq: u32,
}

/// One run of the slope method on a fresh copy of the starting values, then oracle verification.
fn measure(session: &Session, values: &[(BigUint, BigUint)]) -> Result<Measure, String> {
    session.reset();
    let count = values.len() as f64;
    let short = session.time(2, 5)?;
    let mut total = 10u32;
    let (mut s, mut t) = (2u32, short);
    while t < TARGET {
        let next = s * 2;
        if next > MAX_SQUARINGS || t * f64::from(next) / f64::from(s) > BUDGET {
            break;
        }
        t = session.time(next, 3)?;
        s = next;
        total += 3 * next;
    }
    if t > 0.5 {
        eprintln!("warning: {:?}: a dispatch took {:.3} s (> 0.5 s)", session.kernel(), t);
    }
    let extra = t - short;
    if s <= 2 || extra <= 0.02 * short {
        return Err(format!("long dispatch ({s} squarings, {:.3} ms) not measurably longer than the short one ({:.3} ms)", t * 1e3, short * 1e3));
    }
    let slope = count * f64::from(s - 2) / extra;
    let out = session.read()?;
    let verified = verify(&session.kernel(), values, &out, total)?;
    Ok(Measure { slope, short_ms: short * 1e3, long_sq: s, long_ms: t * 1e3, verified, total_sq: total })
}

fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = if s.len() % 2 == 1 { s[s.len() / 2] } else { (s[s.len() / 2 - 1] + s[s.len() / 2]) / 2.0 };
    (s[0], med, s[s.len() - 1])
}

fn parse_scheme(s: &str) -> Scheme {
    match s {
        "A" | "a" => Scheme::Limb13,
        "B" | "b" => Scheme::Wide32,
        "B2" | "b2" => Scheme::Wide32Split,
        "As" | "as" | "A-sqr" => Scheme::Limb13Sqr,
        "Bs" | "bs" | "B-sqr" => Scheme::Wide32Sqr,
        other => panic!("unknown scheme {other} (A, B, B2, As, Bs)"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let runs: usize = arg(&args, "--runs").unwrap_or(3);
    let count: u32 = arg(&args, "--count").unwrap_or(1 << 18);
    let sizes: Vec<u32> = list(&args, "--sizes", "256,384,512,1024");
    let wgs: Vec<u32> = list(&args, "--wgs", "64,128,256");
    let scheme_names: Vec<String> = list(&args, "--schemes", "A,B,B2");
    let schemes: Vec<Scheme> = scheme_names.iter().map(|s| parse_scheme(s)).collect();
    assert!(runs >= 1 && count >= 64);

    let gpu = Gpu::new().expect("no usable adapter");
    println!("adapter: {}", gpu.describe());
    println!("driver: {} {}", gpu.info.driver, gpu.info.driver_info);
    println!("SHADER_INT64: {}", gpu.has_int64);
    println!("instances: {count}, runs per configuration: {runs}, workgroups: {wgs:?}");
    println!();

    let mut failures = 0u32;
    // (scheme, bits) -> (best wg, limbs, slopes at best wg)
    let mut summary: Vec<(Scheme, u32, usize, Option<(u32, Vec<f64>)>)> = Vec::new();
    println!("| scheme | bits | limbs | wg | run | slope sq/s | short ms (2 sq) | long sq/dispatch | long ms | verified |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for &bits in &sizes {
        let values = Batch::random_values(count, bits, 0xbe9c_0000 + u64::from(bits));
        for &scheme in &schemes {
            let limbs = scheme.limbs(bits);
            if scheme.needs_int64() && !gpu.has_int64 {
                println!("| {} | {bits} | {limbs} | - | - | skipped: no SHADER_INT64 | | | | |", scheme.name());
                summary.push((scheme, bits, limbs, None));
                continue;
            }
            let mut best: Option<(u32, Vec<f64>)> = None;
            for &wg in &wgs {
                let k = OneThread::new(scheme, bits, wg);
                let batch = k.batch(&values);
                let session = match Session::new(&gpu, &k, &batch) {
                    Ok(s) => s,
                    Err(e) => {
                        println!("| {} | {bits} | {limbs} | {wg} | - | REJECTED: {e} | | | | |", scheme.name());
                        failures += 1;
                        continue;
                    }
                };
                let mut slopes = Vec::new();
                for r in 0..runs {
                    match measure(&session, &values) {
                        Ok(m) => {
                            println!(
                                "| {} | {bits} | {limbs} | {wg} | {r} | {:.4e} | {:.3} | {} | {:.1} | {} inst x {} sq |",
                                scheme.name(),
                                m.slope,
                                m.short_ms,
                                m.long_sq,
                                m.long_ms,
                                m.verified,
                                m.total_sq
                            );
                            slopes.push(m.slope);
                        }
                        Err(e) => {
                            println!("| {} | {bits} | {limbs} | {wg} | {r} | FAILED: {e} | | | | |", scheme.name());
                            failures += 1;
                        }
                    }
                }
                if slopes.len() == runs && best.as_ref().is_none_or(|(_, b)| stats(&slopes).1 > stats(b).1) {
                    best = Some((wg, slopes));
                }
            }
            summary.push((scheme, bits, limbs, best));
        }
    }

    println!();
    println!("Summary: best workgroup by median slope; squarings/s min / median / max over {runs} runs; ratios are of medians (vs scheme A at the same bits).");
    println!();
    println!("| scheme | bits | limbs | best wg | sq/s min | sq/s median | sq/s max | ratio vs A | note |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for &(scheme, bits, limbs, ref best) in &summary {
        let a_med = summary.iter().find(|r| r.0 == Scheme::Limb13 && r.1 == bits).and_then(|r| r.3.as_ref()).map(|(_, s)| stats(s).1);
        let note = if bits > 512 && scheme == Scheme::Limb13 { "13-bit limbs may spill at this size" } else { "" };
        match best {
            Some((wg, s)) => {
                let (lo, med, hi) = stats(s);
                let ratio = a_med.map_or("n/a".to_string(), |a| format!("{:.2}", med / a));
                println!("| {} | {bits} | {limbs} | {wg} | {lo:.4e} | {med:.4e} | {hi:.4e} | {ratio} | {note} |", scheme.name());
            }
            None => println!("| {} | {bits} | {limbs} | - | - | - | - | n/a | no successful configuration |", scheme.name()),
        }
    }
    // Per-run spread of the B/A ratio (min B / max A .. max B / min A) for the decision bar.
    println!();
    println!("| bits | B/A median | B/A worst case (min B / max A) | B/A best case (max B / min A) | B2/A median |");
    println!("|---|---|---|---|---|");
    for &bits in &sizes {
        let get = |s: Scheme| summary.iter().find(|r| r.0 == s && r.1 == bits).and_then(|r| r.3.as_ref()).map(|(_, v)| stats(v));
        let a = get(Scheme::Limb13);
        let fmt = |x: Option<f64>| x.map_or("n/a".to_string(), |v| format!("{v:.2}"));
        let (b, b2) = (get(Scheme::Wide32), get(Scheme::Wide32Split));
        let pair = |f: fn((f64, f64, f64), (f64, f64, f64)) -> f64, x: Option<(f64, f64, f64)>| a.zip(x).map(|(a, x)| f(a, x));
        println!(
            "| {bits} | {} | {} | {} | {} |",
            fmt(pair(|a, b| b.1 / a.1, b)),
            fmt(pair(|a, b| b.0 / a.2, b)),
            fmt(pair(|a, b| b.2 / a.0, b)),
            fmt(pair(|a, b| b.1 / a.1, b2))
        );
    }
    // Squaring symmetry (issue #2): each dedicated-squaring kernel against its same-radix baseline.
    println!();
    println!("| scheme | bits | baseline | sqr sq/s median | baseline sq/s median | speedup | worst (sqr min / base max) | best (sqr max / base min) |");
    println!("|---|---|---|---|---|---|---|---|");
    for &bits in &sizes {
        for &scheme in schemes.iter().filter(|s| s.is_sqr()) {
            let get = |s: Scheme| summary.iter().find(|r| r.0 == s && r.1 == bits).and_then(|r| r.3.as_ref()).map(|(_, v)| stats(v));
            match (get(scheme), get(scheme.baseline())) {
                (Some(s), Some(b)) => println!(
                    "| {} | {bits} | {} | {:.4e} | {:.4e} | {:.3} | {:.3} | {:.3} |",
                    scheme.name(),
                    scheme.baseline().name(),
                    s.1,
                    b.1,
                    s.1 / b.1,
                    s.0 / b.2,
                    s.2 / b.0
                ),
                _ => println!("| {} | {bits} | {} | n/a | n/a | n/a | n/a | n/a |", scheme.name(), scheme.baseline().name()),
            }
        }
    }
    if failures > 0 {
        eprintln!("{failures} configuration run(s) failed");
        std::process::exit(1);
    }
}
