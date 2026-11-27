//! Portable (wgpu) Montgomery arithmetic kernels for multi-limb numbers (13-bit limbs, or 32-bit
//! limbs with u64 products where the device has int64; several lanes per number, generated WGSL)
//! and a reusable base-2 Fermat-test runner built on them.
//!
//! - [`gpu`]: adapter selection and device setup, plus the one-shot squaring-chain harness.
//! - [`kernel`]: kernel configuration and WGSL generation (native and browser targets).
//! - [`runner`]: the persistent-buffer Fermat runner the `bigmont` CLI drives.
//!
//! The work is `async` (a browser cannot block): `Runner::new_async`, `Runner::fermat_async`,
//! `gpu::run_async` and the rest. Native callers run them with `pollster::block_on`; there are no
//! blocking twins.

pub mod gpu;
pub mod kernel;
mod limb32;
mod limb32sq;
pub mod runner;
/// Limb schemes (radix-dependent host helpers) and device capabilities.
pub mod scheme;
mod signal;
mod unrolled;
