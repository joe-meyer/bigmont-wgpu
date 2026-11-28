//! Shared by the `carry_bench` example and the `carry_emulation` test (the test includes this file
//! with `#[path]`; Cargo does not treat `examples/<dir>/mod.rs` as an example of its own).
//!
//! It generates per-thread big-integer kernels in WGSL, built from two tiny primitives:
//!
//! - `mac(x, y, t, c)`: the multiply-accumulate-with-carry step `x * y + t + c` as a (low, high)
//!   pair of u32. It never overflows: `(2^32 - 1)^2 + 2 (2^32 - 1) = 2^64 - 1`.
//! - `adc(x, y)`: the add-with-carry step `x + y` as a (sum, carry) pair.
//!
//! Every GPU and every shader IR that WGSL is translated to (SPIR-V, DXIL, MSL) has these as one
//! instruction or two, but WGSL itself can only express them in one of two ways, so there are two
//! [`Form`]s of each primitive:
//!
//! - [`Form::U64`] uses `u64` arithmetic. It needs wgpu's native-only `SHADER_INT64` feature, which
//!   WebGPU does not have, so it cannot run in a browser.
//! - [`Form::U32`] uses only `u32` arithmetic, so it is valid WGSL everywhere: the high word of a
//!   32x32 product is rebuilt from four 16x16 products and every carry is recovered by comparing
//!   a sum with one of its operands.
//!
//! The two forms compute bit-identical results; what differs is how many instructions it takes. On
//! top of the primitives, [`bigint_kernel`] unrolls an `n`-limb schoolbook multiplication
//! ([`Op::Mul`], a full `2n`-limb product) and an `n`-limb Montgomery multiplication ([`Op::MontMul`],
//! CIOS, the same word-by-word structure as the engine's kernels, but one number per thread).

#![allow(dead_code)]

use bigmont_core::gpu::Gpu;
use bigmont_oracle::{XorShift64, neg_inv_u32, pack_u32_limbs, random_below, random_bits};
use num_bigint::BigUint;
use num_traits::One;
use std::fmt::Write as _;
use std::time::Instant;

/// Threads per workgroup in every generated kernel.
pub const WG: u32 = 64;

/// How the carry and widening-multiply primitives are written in WGSL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// `u64` arithmetic (native only: needs `wgpu::Features::SHADER_INT64`).
    U64,
    /// `u32` arithmetic only (valid WGSL on every backend, including browsers).
    U32,
}

impl Form {
    pub const ALL: [Form; 2] = [Form::U64, Form::U32];

    pub fn label(self) -> &'static str {
        match self {
            Form::U64 => "u64 (native only)",
            Form::U32 => "u32 emulation (web-valid)",
        }
    }

    pub fn needs_int64(self) -> bool {
        self == Form::U64
    }

    /// WGSL source of `mac` and `adc` in this form.
    pub fn helpers(self) -> &'static str {
        match self {
            Form::U64 => {
                "\
// (low, high) of x * y + t + c. At most (2^32-1)^2 + 2 (2^32-1) = 2^64 - 1, so it never overflows.
fn mac(x: u32, y: u32, t: u32, c: u32) -> vec2<u32> {
    let p = u64(x) * u64(y) + u64(t) + u64(c);
    return vec2<u32>(u32(p), u32(p >> 32u));
}
// (sum, carry) of x + y.
fn adc(x: u32, y: u32) -> vec2<u32> {
    let s = u64(x) + u64(y);
    return vec2<u32>(u32(s), u32(s >> 32u));
}
"
            }
            Form::U32 => {
                "\
// (low, high) of x * y + t + c, from u32 operations only.
fn mac(x: u32, y: u32, t: u32, c: u32) -> vec2<u32> {
    // x = xh 2^16 + xl and y = yh 2^16 + yl, so x*y = hh 2^32 + (lh + hl) 2^16 + ll.
    let xl = x & 0xffffu;
    let xh = x >> 16u;
    let yl = y & 0xffffu;
    let yh = y >> 16u;
    let ll = xl * yl;
    let lh = xl * yh;
    let hl = xh * yl;
    let hh = xh * yh;
    // The middle column (ll >> 16) + low halves of lh and hl is below 3 * 2^16; its overflow past
    // bit 16 is the carry into the high word.
    let mid = (ll >> 16u) + (lh & 0xffffu) + (hl & 0xffffu);
    var hi = hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
    // The low word is the wrapping product. Each addition below can wrap; a wrap is detected by the
    // sum being smaller than an operand, and counts 1 in the high word. The total never exceeds
    // 2^64 - 1, so the high word does not overflow.
    let lo = x * y;
    let lo1 = lo + t;
    hi += select(0u, 1u, lo1 < lo);
    let lo2 = lo1 + c;
    hi += select(0u, 1u, lo2 < lo1);
    return vec2<u32>(lo2, hi);
}
// (sum, carry) of x + y.
fn adc(x: u32, y: u32) -> vec2<u32> {
    let s = x + y;
    return vec2<u32>(s, select(0u, 1u, s < x));
}
"
            }
        }
    }
}

/// Which big-integer operation a kernel computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// `p = a * b`, a full `2n`-limb product (any `a`, `b`).
    Mul,
    /// `r = (a b + m n) / 2^(32 n)` with `m = -a b n^-1 mod 2^(32 n)`: Montgomery multiplication
    /// without the final conditional subtraction, so `r` is in `[0, 2n)` (`n` odd, `a`, `b` < `n`
    /// and `n` below `2^(32 limbs - 4)`).
    MontMul,
}

impl Op {
    pub fn name(self) -> &'static str {
        match self {
            Op::Mul => "multiply",
            Op::MontMul => "Montgomery multiply",
        }
    }

    /// u32 words of input per thread.
    pub fn in_words(self, n: usize) -> usize {
        match self {
            Op::Mul => 2 * n,
            Op::MontMul => 3 * n + 1,
        }
    }

    /// u32 words of output per thread.
    pub fn out_words(self, n: usize) -> usize {
        match self {
            Op::Mul => 2 * n,
            Op::MontMul => n,
        }
    }

    /// Number of `mac` steps one operation takes (`n^2` for the product, `2 n^2` for Montgomery).
    pub fn macs(self, n: usize) -> usize {
        match self {
            Op::Mul => n * n,
            Op::MontMul => 2 * n * n,
        }
    }
}

const BINDINGS: &str = "\
@group(0) @binding(0) var<uniform> params: vec4<u32>;               // x = threads, y = repetitions
@group(0) @binding(1) var<storage, read> inp: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
";

/// WGSL for one operation on `n` limbs per thread, repeated `params.y` times (see [`Kernel::run`]).
///
/// Each repetition feeds its result back into the left operand (`a ^= p` for the product, `a = r` for
/// the Montgomery product), so the compiler cannot hoist or merge repetitions. With one repetition the
/// output is exactly `a * b` (or the Montgomery product), which is what the tests and the benchmark's
/// check compare against the CPU.
pub fn bigint_kernel(form: Form, op: Op, n: usize) -> String {
    assert!(n >= 2, "at least two limbs");
    let mut s = String::new();
    let _ = writeln!(
        s,
        "// {} on {n} 32-bit limbs per thread, {}.",
        op.name(),
        form.label()
    );
    s.push_str(BINDINGS);
    s.push_str(form.helpers());
    let _ = writeln!(s, "const STRIDE: u32 = {}u;", op.in_words(n));
    let _ = writeln!(s, "const OUT: u32 = {}u;", op.out_words(n));
    let _ = writeln!(s, "@compute @workgroup_size({WG})");
    s.push_str("fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n");
    s.push_str("    let tid = gid.x;\n    if (tid >= params.x) { return; }\n");
    s.push_str("    let base = tid * STRIDE;\n");
    for i in 0..n {
        let _ = writeln!(s, "    var a{i} = inp[base + {i}u];");
    }
    for i in 0..n {
        let _ = writeln!(s, "    let b{i} = inp[base + {}u];", n + i);
    }
    let width = match op {
        Op::Mul => 2 * n,
        Op::MontMul => {
            for i in 0..n {
                let _ = writeln!(s, "    let n{i} = inp[base + {}u];", 2 * n + i);
            }
            let _ = writeln!(s, "    let ninv = inp[base + {}u];", 3 * n);
            n + 2
        }
    };
    for k in 0..width {
        let _ = writeln!(s, "    var t{k} = 0u;");
    }
    s.push_str("    for (var rep = 0u; rep < params.y; rep++) {\n");
    for k in 0..width {
        let _ = writeln!(s, "        t{k} = 0u;");
    }
    match op {
        Op::Mul => {
            // Operand scanning: row i adds a_i * b into t at offset i; the row's final carry is a
            // fresh limb (nothing has reached t[i + n] yet).
            for i in 0..n {
                let _ = writeln!(s, "        {{ var c = 0u;");
                for j in 0..n {
                    let _ = writeln!(
                        s,
                        "          {{ let r = mac(a{i}, b{j}, t{}, c); t{} = r.x; c = r.y; }}",
                        i + j,
                        i + j
                    );
                }
                let _ = writeln!(s, "          t{} = c; }}", i + n);
            }
            for i in 0..n {
                let _ = writeln!(s, "        a{i} = a{i} ^ t{i};");
            }
        }
        Op::MontMul => {
            // CIOS: for each limb b_i, t += a * b_i, then t = (t + m n) / 2^32 with m = t0 * ninv.
            for i in 0..n {
                let _ = writeln!(s, "        {{ var c = 0u;");
                for j in 0..n {
                    let _ = writeln!(
                        s,
                        "          {{ let r = mac(a{j}, b{i}, t{j}, c); t{j} = r.x; c = r.y; }}"
                    );
                }
                let _ = writeln!(
                    s,
                    "          {{ let q = adc(t{n}, c); t{n} = q.x; t{} = q.y; }}",
                    n + 1
                );
                let _ = writeln!(s, "          let m = t0 * ninv;");
                // Limb 0 of t + m n is zero by the choice of m; only its carry survives.
                let _ = writeln!(s, "          {{ let r = mac(m, n0, t0, 0u); c = r.y; }}");
                for j in 1..n {
                    let _ = writeln!(
                        s,
                        "          {{ let r = mac(m, n{j}, t{j}, c); t{} = r.x; c = r.y; }}",
                        j - 1
                    );
                }
                let _ = writeln!(
                    s,
                    "          {{ let q = adc(t{n}, c); t{} = q.x; t{n} = t{} + q.y; }} }}",
                    n - 1,
                    n + 1
                );
            }
            for i in 0..n {
                let _ = writeln!(s, "        a{i} = t{i};");
            }
        }
    }
    s.push_str("    }\n");
    // Stored after the loop. For Mul this is the product of the last repetition's operands; for
    // MontMul the new `a` (the last result).
    match op {
        Op::Mul => {
            for k in 0..2 * n {
                let _ = writeln!(s, "    out[tid * OUT + {k}u] = t{k};");
            }
        }
        Op::MontMul => {
            for k in 0..n {
                let _ = writeln!(s, "    out[tid * OUT + {k}u] = a{k};");
            }
        }
    }
    s.push_str("}\n");
    s
}

/// WGSL that runs `mac` and `adc` once per thread on four u32 inputs `x y t c` (words `4 tid ..`):
/// the output is `mac.lo, mac.hi, adc(x, t).sum, adc(x, t).carry`.
pub fn primitive_kernel(form: Form) -> String {
    format!(
        "{BINDINGS}{}\
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let tid = gid.x;
    if (tid >= params.x) {{ return; }}
    let i = tid * 4u;
    let m = mac(inp[i], inp[i + 1u], inp[i + 2u], inp[i + 3u]);
    let s = adc(inp[i], inp[i + 2u]);
    out[i] = m.x;
    out[i + 1u] = m.y;
    out[i + 2u] = s.x;
    out[i + 3u] = s.y;
}}
",
        form.helpers()
    )
}

/// Reference `mac` and `adc` on the CPU: `(lo, hi, sum, carry)` for `x y t c`.
pub fn primitive_reference(x: u32, y: u32, t: u32, c: u32) -> [u32; 4] {
    let p = u64::from(x) * u64::from(y) + u64::from(t) + u64::from(c);
    let s = u64::from(x) + u64::from(t);
    [p as u32, (p >> 32) as u32, s as u32, (s >> 32) as u32]
}

/// Inputs for `threads` operations, as the kernel reads them. The first threads are edge cases (all
/// ones, zeros, one, a modulus of all ones below the top four bits, operands `n - 1` and `n - 2`,
/// and a single high bit); the rest are random. Montgomery moduli are odd with the top four bits
/// clear. Returns the packed words and, for Montgomery, each thread's modulus.
pub fn make_inputs(op: Op, n: usize, threads: u32, seed: u64) -> (Vec<u32>, Vec<BigUint>) {
    let bits = 32 * n as u32;
    let mut rng = XorShift64::new(seed);
    let mut words = Vec::with_capacity(op.in_words(n) * threads as usize);
    let mut moduli = Vec::new();
    let full = (BigUint::one() << bits) - 1u32;
    let top = BigUint::one() << (bits - 1);
    for t in 0..threads {
        match op {
            Op::Mul => {
                let (a, b) = match t {
                    0 => (full.clone(), full.clone()),
                    1 => (BigUint::from(0u32), full.clone()),
                    2 => (BigUint::one(), full.clone()),
                    3 => (top.clone(), top.clone()),
                    4 => (full.clone(), BigUint::one()),
                    _ => (random_bits(&mut rng, bits), random_bits(&mut rng, bits)),
                };
                words.extend(pack_u32_limbs(&a, n));
                words.extend(pack_u32_limbs(&b, n));
            }
            Op::MontMul => {
                // n below 2^(bits - 4), odd; R = 2^bits > 16 n.
                let modulus = match t {
                    0 | 1 => (BigUint::one() << (bits - 4)) - 1u32,
                    2 => (BigUint::one() << (bits - 5)) + 1u32,
                    _ => random_bits(&mut rng, bits - 4) | BigUint::one(),
                };
                let (a, b) = match t {
                    0 => (&modulus - 1u32, &modulus - 1u32),
                    1 => (BigUint::from(0u32), &modulus - 1u32),
                    2 => (BigUint::one(), &modulus - 2u32),
                    _ => (
                        random_below(&mut rng, &modulus),
                        random_below(&mut rng, &modulus),
                    ),
                };
                words.extend(pack_u32_limbs(&a, n));
                words.extend(pack_u32_limbs(&b, n));
                words.extend(pack_u32_limbs(&modulus, n));
                words.push(neg_inv_u32(&modulus));
                moduli.push(modulus);
            }
        }
    }
    (words, moduli)
}

/// Checks the first `check` threads' output of one repetition against exact big-integer
/// arithmetic. Returns a description of the first mismatch.
pub fn verify(
    op: Op,
    n: usize,
    inp: &[u32],
    moduli: &[BigUint],
    out: &[u32],
    check: usize,
) -> Result<(), String> {
    let (is, os) = (op.in_words(n), op.out_words(n));
    for t in 0..check {
        let a = BigUint::new(inp[t * is..t * is + n].to_vec());
        let b = BigUint::new(inp[t * is + n..t * is + 2 * n].to_vec());
        let got = BigUint::new(out[t * os..(t + 1) * os].to_vec());
        match op {
            Op::Mul => {
                if got != &a * &b {
                    return Err(format!("thread {t}: product mismatch (a={a:x}, b={b:x})"));
                }
            }
            Op::MontMul => {
                let modulus = &moduli[t];
                // r = (a b + m n) / R with m = -a b n^-1 mod R, computed independently of the
                // kernel with big integers: r is the unique value in [0, 2n) congruent to a b R^-1.
                let r_bits = 32 * n as u32;
                let r = BigUint::one() << r_bits;
                let ninv = bigmont_oracle::inv_mod_pow2(modulus, r_bits);
                let ab = &a * &b;
                let m = (&r - ((&ab * &ninv) % &r)) % &r;
                let want = (ab + m * modulus) >> r_bits;
                if got != want {
                    return Err(format!(
                        "thread {t}: Montgomery mismatch (a={a:x}, b={b:x})"
                    ));
                }
            }
        }
    }
    Ok(())
}

/// A compiled kernel with its buffers, bound to one [`Gpu`].
pub struct Kernel {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    out: wgpu::Buffer,
    staging: wgpu::Buffer,
    threads: u32,
    out_bytes: u64,
}

impl Kernel {
    /// Compiles `source` and uploads `inp` (read as `array<u32>` at binding 1). The output buffer
    /// holds `out_words` u32 words per thread. A shader the device rejects is an `Err`, not a panic.
    pub fn new(
        gpu: &Gpu,
        source: &str,
        inp: &[u32],
        threads: u32,
        out_words: usize,
    ) -> Result<Kernel, String> {
        let device = &gpu.device;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("carry"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("carry"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(format!("shader rejected: {e}"));
        }
        let buffer = |size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let out_bytes = 4 * out_words as u64 * u64::from(threads);
        let params = buffer(
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let inp_buf = buffer(
            4 * inp.len() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let out = buffer(
            out_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = buffer(
            out_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        gpu.queue
            .write_buffer(&inp_buf, 0, bytemuck::cast_slice(inp));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: inp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
            ],
        });
        Ok(Kernel {
            pipeline,
            bind_group,
            params,
            out,
            staging,
            threads,
            out_bytes,
        })
    }

    /// Runs the kernel with `reps` repetitions per thread and returns the seconds from submit until
    /// the GPU is idle.
    pub fn run(&self, gpu: &Gpu, reps: u32) -> Result<f64, String> {
        gpu.queue.write_buffer(
            &self.params,
            0,
            bytemuck::cast_slice(&[self.threads, reps, 0, 0]),
        );
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.threads.div_ceil(WG), 1, 1);
        }
        let start = Instant::now();
        gpu.queue.submit([enc.finish()]);
        gpu.wait()?;
        Ok(start.elapsed().as_secs_f64())
    }

    /// The output buffer as u32 words.
    pub fn read_out(&self, gpu: &Gpu) -> Result<Vec<u32>, String> {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.out, 0, &self.staging, 0, self.out_bytes);
        gpu.queue.submit([enc.finish()]);
        gpu.read_back(&self.staging, self.out_bytes)
    }
}
