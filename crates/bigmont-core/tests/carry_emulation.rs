//! Why WGSL needs add-with-carry and an extended multiply, as a test script.
//!
//! Without them a 32-bit-limb big-integer kernel has two options: `u64` arithmetic (native wgpu only,
//! `SHADER_INT64`, not valid in a browser) or an emulation in plain `u32` operations (valid
//! everywhere). These tests show that
//!
//! 1. the `u32` emulation is bit-exact, so it is a correct fallback for the web (`primitives_*`,
//!    `bigint_*`: run on whatever adapter is selected, no `SHADER_INT64` needed for the emulation),
//! 2. it takes several times as many operations as the `u64` form for the same work
//!    (`emulation_needs_many_more_operations`: a static count on the compiled IR, no GPU needed), and
//!    the `carry_bench` example measures what that costs in time.
//!
//! GPU parts are strict like the other GPU tests (`BIGMONT_ALLOW_SKIP=1` turns a missing adapter, or a
//! missing `SHADER_INT64` for the `u64` form, into a loud skip).
#[path = "../examples/carry_kernels/mod.rs"]
mod carry_kernels;
mod common;

use bigmont_core::gpu::Gpu;
use carry_kernels::{
    Form, Kernel, Op, bigint_kernel, make_inputs, primitive_kernel, primitive_reference, verify,
};

fn open() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => {
            eprintln!("adapter: {} ({:?})", g.info.name, g.info.backend);
            Some(g)
        }
        Err(e) => {
            common::skip(format!("carry_emulation: no adapter: {e}"));
            None
        }
    }
}

/// Whether `form` can run on `gpu`; a missing `SHADER_INT64` is a (strict) skip.
fn usable(gpu: &Gpu, form: Form, what: &str) -> bool {
    if form.needs_int64() && !gpu.has_int64 {
        common::skip(format!(
            "{what}: the u64 form needs SHADER_INT64, which {} lacks",
            gpu.info.name
        ));
        return false;
    }
    true
}

/// Inputs that exercise every carry path of `mac` and `adc`: all combinations of limb values at the
/// edges of the range, then random words.
fn primitive_inputs() -> Vec<[u32; 4]> {
    let edge = [
        0u32,
        1,
        2,
        0xffff,
        0x1_0000,
        0x7fff_ffff,
        0x8000_0000,
        0x8000_0001,
        0xffff_0000,
        0xffff_fffe,
        0xffff_ffff,
    ];
    let mut v = Vec::new();
    for &x in &edge {
        for &y in &edge {
            for &t in &edge {
                for &c in &edge {
                    v.push([x, y, t, c]);
                }
            }
        }
    }
    let mut rng = bigmont_oracle::XorShift64::new(0xca77);
    for _ in 0..20_000 {
        let w = rng.next_u64();
        let w2 = rng.next_u64();
        v.push([w as u32, (w >> 32) as u32, w2 as u32, (w2 >> 32) as u32]);
    }
    v
}

fn check_primitives(form: Form) {
    let Some(gpu) = open() else { return };
    if !usable(&gpu, form, "primitives") {
        return;
    }
    let cases = primitive_inputs();
    let words: Vec<u32> = cases.iter().flatten().copied().collect();
    let threads = cases.len() as u32;
    let kernel = Kernel::new(&gpu, &primitive_kernel(form), &words, threads, 4)
        .unwrap_or_else(|e| panic!("{form:?}: {e}"));
    kernel.run(&gpu, 1).expect("run");
    let out = kernel.read_out(&gpu).expect("read back");
    for (i, case) in cases.iter().enumerate() {
        let [x, y, t, c] = *case;
        let want = primitive_reference(x, y, t, c);
        assert_eq!(
            out[4 * i..4 * i + 4],
            want,
            "{form:?}: mac({x:#x}, {y:#x}, {t:#x}, {c:#x}) and adc({x:#x}, {t:#x}) as [lo, hi, sum, carry]"
        );
    }
    eprintln!("{form:?}: {} primitive cases match", cases.len());
}

#[test]
fn primitives_u32_emulation_is_exact() {
    check_primitives(Form::U32);
}

#[test]
fn primitives_u64_form_is_exact() {
    check_primitives(Form::U64);
}

fn check_bigint(form: Form, op: Op, n: usize) {
    let Some(gpu) = open() else { return };
    if !usable(&gpu, form, "bigint") {
        return;
    }
    let threads = 256u32;
    let (inp, moduli) = make_inputs(op, n, threads, 0xb16 + n as u64);
    let kernel = Kernel::new(
        &gpu,
        &bigint_kernel(form, op, n),
        &inp,
        threads,
        op.out_words(n),
    )
    .unwrap_or_else(|e| panic!("{form:?} {op:?} {n} limbs: {e}"));
    kernel.run(&gpu, 1).expect("run");
    let out = kernel.read_out(&gpu).expect("read back");
    verify(op, n, &inp, &moduli, &out, threads as usize)
        .unwrap_or_else(|e| panic!("{form:?} {op:?} {n} limbs: {e}"));
    eprintln!("{form:?} {op:?} {n} limbs: {threads} operations match exact arithmetic");
}

#[test]
fn bigint_multiply_u32_emulation() {
    for n in [2, 8, 16] {
        check_bigint(Form::U32, Op::Mul, n);
    }
}

#[test]
fn bigint_montgomery_u32_emulation() {
    for n in [2, 8, 16] {
        check_bigint(Form::U32, Op::MontMul, n);
    }
}

#[test]
fn bigint_multiply_u64_form() {
    for n in [2, 8, 16] {
        check_bigint(Form::U64, Op::Mul, n);
    }
}

#[test]
fn bigint_montgomery_u64_form() {
    for n in [2, 8, 16] {
        check_bigint(Form::U64, Op::MontMul, n);
    }
}

/// Counts the arithmetic in `mac` on naga's IR (no GPU, no validation: the `u64` form needs a
/// capability the validator is not given here, and the front end alone is enough to count).
/// Returns `(multiplies, other binary operations, selects)`. Width conversions (`u64(x)`, `u32(p)`)
/// are not counted: they are register moves, not instructions, on every GPU.
fn ops_in_mac(form: Form) -> (usize, usize, usize) {
    let source = format!(
        "{}
@compute @workgroup_size(1) fn main() {{ let r = mac(1u, 2u, 3u, 4u); _ = r; }}",
        form.helpers()
    );
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{form:?}: {}", e.emit_to_string(&source)));
    let (_, mac) = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("mac"))
        .expect("mac function");
    let (mut multiplies, mut other, mut selects) = (0, 0, 0);
    for (_, e) in mac.expressions.iter() {
        match e {
            naga::Expression::Binary { op, .. } if *op == naga::BinaryOperator::Multiply => {
                multiplies += 1;
            }
            naga::Expression::Binary { .. } => other += 1,
            naga::Expression::Select { .. } => selects += 1,
            _ => {}
        }
    }
    (multiplies, other, selects)
}

/// The same work, with and without a native wide product: one `x * y + t + c` step is one 64-bit
/// multiply-add where `u64` exists, and several multiplies plus many other operations where it does
/// not. This is the static part of the argument; the `carry_bench` example measures the time.
#[test]
fn emulation_needs_many_more_operations() {
    let (m64, o64, s64) = ops_in_mac(Form::U64);
    let (m32, o32, s32) = ops_in_mac(Form::U32);
    eprintln!("mac, u64 form: {m64} multiplies, {o64} other binary operations, {s64} selects");
    eprintln!("mac, u32 emulation: {m32} multiplies, {o32} other binary operations, {s32} selects");
    assert_eq!(m64, 1, "the u64 form multiplies once");
    assert_eq!(
        m32, 5,
        "the emulation needs four 16x16 products plus the low word"
    );
    let (t64, t32) = (m64 + o64 + s64, m32 + o32 + s32);
    assert!(
        t32 >= 4 * t64,
        "expected the emulation to need at least 4x the operations ({t32} vs {t64})"
    );
}
