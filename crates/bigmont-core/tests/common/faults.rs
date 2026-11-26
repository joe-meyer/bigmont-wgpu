//! Simulated device faults: each helper takes a generated kernel source and returns a mutated one
//! that behaves like a faulty device (a flipped register, a flipped bit arriving from a load, a
//! faulty multiply unit, a corrupted memory cell). The runner, the other kernels and the host
//! comparison then run unchanged on top of it.
//!
//! `inst` is the instance slot within one dispatch (the candidate's index within its chunk), so a
//! fault keyed to `inst` hits that slot in every chunk. `fp.x` is the dispatch's first ladder
//! iteration and `fp.w` the runner's pass role (0 = the primary computation, 1 = the tail
//! recomputation's copy, 2 = the span check's own passes, which also run with `fp.x == 0`). Every anchor must occur exactly once, so a generator change fails loudly
//! instead of silently not injecting.
#![allow(dead_code)]
use bigmont_core::scheme::LimbScheme;

/// The kernels' Montgomery-factor load, the line every fault after the loads is inserted after.
const NINV_ANCHOR: &str = "let ninv = ninv_buf[ic]";
/// The first statement after a ladder iteration's squaring (before its optional doubling).
const AFTER_SQUARING_ANCHOR: &str = "let gi = fp.y - 2u - (fp.x + s);";

/// The integer bit every value fault flips.
const FAULT_BIT: u32 = 17;

fn only_once(src: &str, anchor: &str) -> usize {
    assert_eq!(
        src.matches(anchor).count(),
        1,
        "anchor {anchor:?} must occur exactly once in the kernel source"
    );
    src.find(anchor).unwrap()
}

/// Inserts `stmt` as a new line right after the line containing `anchor`.
fn insert_after_line(src: &str, anchor: &str, stmt: &str) -> String {
    let at = only_once(src, anchor);
    let eol = at + src[at..].find('\n').expect("anchor line ends") + 1;
    format!("{}    {stmt}\n{}", &src[..eol], &src[eol..])
}

/// Inserts `stmt` as a new line right before the line containing `anchor`, at its indentation.
fn insert_before_line(src: &str, anchor: &str, stmt: &str) -> String {
    let at = only_once(src, anchor);
    let bol = src[..at].rfind('\n').map_or(0, |i| i + 1);
    let indent = &src[bol..at];
    format!("{}{indent}{stmt}\n{}", &src[..bol], &src[bol..])
}

/// Replaces the unique `from` with `to`.
fn replace_once(src: &str, from: &str, to: &str) -> String {
    only_once(src, from);
    src.replacen(from, to, 1)
}

/// Limbs per lane, from the generated `const LPL: u32 = Nu;`.
fn limbs_per_lane(src: &str) -> u32 {
    let at = only_once(src, "const LPL: u32 = ") + "const LPL: u32 = ".len();
    let end = at + src[at..].find('u').expect("LPL literal");
    src[at..end].parse().expect("LPL value")
}

/// `(lane, statement)` flipping bit [`FAULT_BIT`] of the number held in the lane's named limbs
/// `x0, x1, ...`: limb `17 / limb_bits` holds it (13-bit scheme: limb 1 bit 4; 32-bit: limb 0
/// bit 17), and lane `limb / LPL` holds that limb as `x{limb % LPL}`.
fn value_flip(src: &str, scheme: LimbScheme) -> (u32, String) {
    let limb_bits = scheme.limb_bits();
    let limb = FAULT_BIT / limb_bits;
    let lpl = limbs_per_lane(src);
    let var = format!("x{}", limb % lpl);
    let mask = 1u32 << (FAULT_BIT % limb_bits);
    (limb / lpl, format!("{var} = {var} ^ {mask}u;"))
}

/// A register fault in the Fermat ladder: bit 17 of instance `inst`'s state flips right after the
/// squaring of 0-based ladder iteration `iteration`, in the primary computation only.
pub fn register_flip(src: &str, scheme: LimbScheme, inst: u32, iteration: u32) -> String {
    let (lane, flip) = value_flip(src, scheme);
    let stmt = format!(
        "if (inst == {inst}u && (fp.x + s) == {iteration}u && lane == {lane}u && fp.w == 0u) {{ {flip} }}"
    );
    insert_before_line(src, AFTER_SQUARING_ANCHOR, &stmt)
}

/// A memory-read fault in the Fermat ladder: bit 17 of instance `inst`'s state arrives flipped
/// from the load at the start of the dispatch whose first iteration is `first_iteration` (a block
/// start, or the tail start), in the primary computation only.
pub fn read_flip(src: &str, scheme: LimbScheme, inst: u32, first_iteration: u32) -> String {
    let (lane, flip) = value_flip(src, scheme);
    let stmt = format!(
        "if (inst == {inst}u && fp.x == {first_iteration}u && lane == {lane}u && fp.w == 0u) {{ {flip} }}"
    );
    insert_after_line(src, NINV_ANCHOR, &stmt)
}

/// A faulty multiply unit (multiply kernel): bit 17 of instance `inst`'s left operand flips in
/// every multiply (the multiply kernel has no iteration index).
pub fn mul_path_flip(src: &str, scheme: LimbScheme, inst: u32) -> String {
    let (lane, flip) = value_flip(src, scheme);
    let stmt = format!("if (inst == {inst}u && lane == {lane}u) {{ {flip} }}");
    insert_after_line(src, NINV_ANCHOR, &stmt)
}

/// Makes a read-only storage binding writable (so a fault can store into it) and stores `write`
/// once, from lane 0 of instance `inst` in the primary dispatch starting at `first_iteration`.
/// Later dispatches read the corrupted cell.
fn cell_flip(src: &str, binding: &str, inst: u32, first_iteration: u32, write: &str) -> String {
    let src = replace_once(
        src,
        &format!("var<storage, read> {binding}: array<u32>;"),
        &format!("var<storage, read_write> {binding}: array<u32>;"),
    );
    let stmt = format!(
        "if (inst == {inst}u && fp.x == {first_iteration}u && lane == 0u && fp.w == 0u) {{ {write} }}"
    );
    insert_after_line(&src, NINV_ANCHOR, &stmt)
}

/// A persistent corruption of the modulus in memory: bit 1 of instance `inst`'s limb 0 of `n`
/// flips (so `n` stays odd); every later dispatch, of every kernel, reads the wrong modulus.
pub fn cell_flip_n(src: &str, inst: u32, first_iteration: u32) -> String {
    cell_flip(
        src,
        "n_limbs",
        inst,
        first_iteration,
        "n_limbs[off] = n0 ^ 2u;",
    )
}

/// A persistent corruption of instance `inst`'s Montgomery factor `-n^-1` in memory (bit 2).
pub fn cell_flip_ninv(src: &str, inst: u32, first_iteration: u32) -> String {
    cell_flip(
        src,
        "ninv_buf",
        inst,
        first_iteration,
        "ninv_buf[ic] = ninv_buf[ic] ^ 4u;",
    )
}

/// A persistent corruption of instance `inst`'s exponent `n - 1` in memory: bit 0 of word 1
/// (exponent bit 32) flips.
pub fn cell_flip_e(src: &str, inst: u32, first_iteration: u32) -> String {
    cell_flip(
        src,
        "e_words",
        inst,
        first_iteration,
        "e_words[ic * fp.z + 1u] = e_words[ic * fp.z + 1u] ^ 1u;",
    )
}

/// A persistent fault in the path that loads the modulus, with the buffer itself intact: every
/// load of instance `inst`'s limb 0 of `n`, in every dispatch of the kernel, returns it with bit 1
/// flipped. Apply it to both the Fermat and the multiply source to make every kernel agree on the
/// wrong modulus.
pub fn bad_modulus_path(src: &str, inst: u32) -> String {
    replace_once(
        src,
        "let n0 = n_limbs[off + 0u];",
        &format!("let n0 = n_limbs[off + 0u] ^ select(0u, 2u, inst == {inst}u && lane == 0u);"),
    )
}
