//! What do the missing carry and widening-multiply primitives cost in WGSL? Runs big-integer
//! multiplication and Montgomery multiplication, one number per thread, in two forms that compute
//! the same bits (see `carry_kernels/mod.rs`):
//!
//! - `u64`: 64-bit products and carries (wgpu's native-only `SHADER_INT64`; not valid in a browser),
//! - `u32 emulation`: only `u32` operations, the high word of a product rebuilt from four 16x16
//!   products and carries found by comparing sums (valid WGSL everywhere, including WebGPU).
//!
//! Every kernel is first checked against exact big-integer arithmetic on the CPU (edge cases and
//! random operands); a kernel that fails prints no rate and the exit status is nonzero. The time of
//! the remaining kernels uses the slope method of `scheme_compare`: a dispatch of 2 repetitions
//! per thread is timed (best of 3), the repetitions are doubled until one dispatch takes about
//! `--budget` seconds, and the rate is threads x (R_long - 2) / (t_long - t_short).
//!
//!   BIGMONT_ADAPTER=4070 cargo run --release -p bigmont-core --example carry_bench
//!   ... -- --limbs 8,16,32 --ops mul,montmul --threads 65536 --budget 0.25
//!   ... -- --dump-wgsl DIR     (also writes every generated kernel as DIR/<op>_<limbs>_<form>.wgsl,
//!                               for feeding to another compiler, e.g. a browser's)
//!
//! "G mac/s" counts one `mac` step (x * y + t + c) as one unit: n^2 per product, 2 n^2 per Montgomery
//! multiplication, so it is comparable across sizes and between the two operations. The ratio column
//! is time per operation relative to the u64 form on the same device. This is one thread per number,
//! unlike the cooperative kernels the engine uses, so it measures the cost of the primitives, not the
//! speed of the engine.

mod carry_kernels;

use bigmont_core::gpu::Gpu;
use carry_kernels::{Form, Kernel, Op, bigint_kernel, make_inputs, verify};

/// Threads whose output is compared with the CPU (the first ones are the edge cases).
const CHECK: usize = 512;

fn arg<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    args.iter().position(|a| a == name).map(|i| {
        args.get(i + 1)
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("{name} needs a value"))
    })
}

fn list<T: std::str::FromStr>(args: &[String], name: &str, default: &str) -> Vec<T> {
    let v: String = arg(args, name).unwrap_or_else(|| default.to_string());
    v.split(',')
        .map(|t| {
            t.trim()
                .parse()
                .unwrap_or_else(|_| panic!("bad {name} entry '{t}'"))
        })
        .collect()
}

fn form_tag(form: Form) -> &'static str {
    match form {
        Form::U64 => "u64",
        Form::U32 => "u32",
    }
}

fn best_of(k: &Kernel, gpu: &Gpu, reps: u32, runs: usize) -> Result<f64, String> {
    let mut best = f64::MAX;
    for _ in 0..runs {
        best = best.min(k.run(gpu, reps)?);
    }
    Ok(best)
}

/// Operations per second per thread-repetition, by the slope method (see the module docs).
fn rate(k: &Kernel, gpu: &Gpu, threads: u32, budget: f64) -> Result<(f64, u32, f64), String> {
    const SHORT: u32 = 2;
    const MAX_REPS: u32 = 1 << 16;
    let t_short = best_of(k, gpu, SHORT, 3)?;
    let mut reps = 8;
    let mut t_long = best_of(k, gpu, reps, 1)?;
    while t_long < budget && reps < MAX_REPS {
        reps *= 2;
        t_long = best_of(k, gpu, reps, 1)?;
    }
    t_long = t_long.min(best_of(k, gpu, reps, 2)?);
    let dt = t_long - t_short;
    if dt <= 0.0 {
        return Err("timing too noisy: the long run was not slower than the short one".into());
    }
    Ok((
        f64::from(threads) * f64::from(reps - SHORT) / dt,
        reps,
        t_long,
    ))
}

fn main() {
    // The unrolled 1,024-bit kernels are large straight-line shaders; compiling one through DX12
    // overflowed the 1 MiB main-thread stack of a Windows process, so everything runs on a thread
    // with a bigger stack.
    let worker = std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(run)
        .expect("spawn");
    if worker.join().is_err() {
        std::process::exit(1);
    }
}

fn run() {
    let args: Vec<String> = std::env::args().collect();
    let limbs: Vec<usize> = list(&args, "--limbs", "8,16,32");
    let ops: Vec<String> = list(&args, "--ops", "mul,montmul");
    let threads: u32 = arg(&args, "--threads").unwrap_or(1 << 16);
    let budget: f64 = arg(&args, "--budget").unwrap_or(0.25);
    let dump: Option<String> = arg(&args, "--dump-wgsl");
    if let Some(dir) = &dump {
        std::fs::create_dir_all(dir).expect("create the dump directory");
    }
    let gpu = Gpu::new().unwrap_or_else(|e| panic!("no adapter: {e}"));
    println!(
        "{} [{:?}], SHADER_INT64: {}, {threads} threads, workgroup {}",
        gpu.info.name,
        gpu.info.backend,
        gpu.has_int64,
        carry_kernels::WG
    );
    println!(
        "{:<20} {:>5} {:<26} {:>9} {:>11} {:>9} {:>8}",
        "operation", "bits", "form", "ms (long)", "M ops/s", "G mac/s", "vs u64"
    );
    let mut failed = false;
    for name in &ops {
        let op = match name.as_str() {
            "mul" => Op::Mul,
            "montmul" => Op::MontMul,
            other => panic!("unknown op '{other}' (mul, montmul)"),
        };
        for &n in &limbs {
            let (inp, moduli) = make_inputs(op, n, threads, 0x5eed + n as u64);
            let mut u64_rate = None;
            for form in Form::ALL {
                let label = format!("{:<20} {:>5} {:<26}", op.name(), 32 * n, form.label());
                if form.needs_int64() && !gpu.has_int64 {
                    println!("{label} skipped (this device has no SHADER_INT64)");
                    continue;
                }
                let source = bigint_kernel(form, op, n);
                if let Some(dir) = &dump {
                    let file = format!("{dir}/{name}_{n}_{}.wgsl", form_tag(form));
                    std::fs::write(&file, &source).unwrap_or_else(|e| panic!("{file}: {e}"));
                }
                let kernel = match Kernel::new(&gpu, &source, &inp, threads, op.out_words(n)) {
                    Ok(k) => k,
                    Err(e) => {
                        println!("{label} {}", e.lines().next().unwrap_or("rejected"));
                        failed = true;
                        continue;
                    }
                };
                let checked = kernel
                    .run(&gpu, 1)
                    .and_then(|_| kernel.read_out(&gpu))
                    .and_then(|out| verify(op, n, &inp, &moduli, &out, CHECK));
                if let Err(e) = checked {
                    println!("{label} FAILED the CPU check: {e}");
                    failed = true;
                    continue;
                }
                match rate(&kernel, &gpu, threads, budget) {
                    Ok((ops_per_s, _reps, t_long)) => {
                        let gmac = ops_per_s * op.macs(n) as f64 / 1e9;
                        let rel = match (form, u64_rate) {
                            (Form::U64, _) => {
                                u64_rate = Some(ops_per_s);
                                "1.00x".to_string()
                            }
                            (_, Some(r)) => format!("{:.2}x", r / ops_per_s),
                            _ => "-".to_string(),
                        };
                        println!(
                            "{label} {:>9.1} {:>11.3} {:>9.2} {:>8}",
                            t_long * 1e3,
                            ops_per_s / 1e6,
                            gmac,
                            rel
                        );
                    }
                    Err(e) => {
                        println!("{label} {e}");
                        failed = true;
                    }
                }
            }
        }
    }
    println!();
    println!("vs u64 = time per operation relative to the u64 form (1.00x); above 1 is slower.");
    if failed {
        std::process::exit(1);
    }
}
