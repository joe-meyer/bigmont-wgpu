//! Browser entry points for the bigmont engine: the Rust Fermat runner on the page's own WebGPU.
//!
//! Build with `cargo build -p bigmont-wasm --target wasm32-unknown-unknown --release`, then run
//! `wasm-bindgen --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/bigmont_wasm.wasm`
//! and open `web/rust.html` (see `web/README.md`).
//!
//! A browser has no 64-bit shader integers, so only the 13-bit limb scheme runs here. The kernel
//! shape is fixed by the caller (`tpi`, `wg`); there is no autotuner in the browser yet.
#![cfg(target_arch = "wasm32")]

use bigmont_core::gpu::{AdapterSelector, Gpu};
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_core::runner::Runner;
use bigmont_core::scheme::LimbScheme;
use num_bigint::BigUint;
use wasm_bindgen::prelude::*;

/// The most candidates one dispatch takes (the runner clamps it further to the device limits).
const CAPACITY: u32 = 4096;

fn js(e: String) -> JsError {
    JsError::new(&e)
}

/// An open WebGPU device.
#[wasm_bindgen]
pub struct Engine {
    gpu: Gpu,
}

#[wasm_bindgen]
impl Engine {
    /// Opens the page's WebGPU adapter and device (13-bit kernels need subgroups).
    pub async fn open() -> Result<Engine, JsError> {
        let gpu = Gpu::open_async(&AdapterSelector::default()).await.map_err(js)?;
        Ok(Engine { gpu })
    }

    /// What was opened: adapter, backend, subgroup sizes and the capabilities the kernels need.
    pub fn describe(&self) -> String {
        format!("{} | subgroups: {} | int64: {}", self.gpu.describe(), self.gpu.has_subgroups, self.gpu.has_int64)
    }

    /// Base-2 Fermat verdicts (1 = probable prime) for `candidates`, hex strings of odd numbers that
    /// each have exactly `bits` bits, using the 13-bit unrolled kernel with `tpi` lanes per number
    /// and workgroup size `wg`. Verdicts come back in input order. Errors (a lost device, a rejected
    /// kernel, a malformed candidate) are thrown as JS errors.
    pub async fn fermat_hex(&self, bits: u32, candidates: Vec<String>, tpi: u32, wg: u32) -> Result<Vec<u8>, JsError> {
        let ns = candidates
            .iter()
            .enumerate()
            .map(|(i, h)| BigUint::parse_bytes(h.as_bytes(), 16).ok_or_else(|| JsError::new(&format!("candidate {i} is not a hex number"))))
            .collect::<Result<Vec<_>, _>>()?;
        let cfg = KernelConfig::for_bits_scheme(LimbScheme::Limb13, bits, tpi, wg, Variant::Unrolled);
        let mut runner = Runner::new_async(&self.gpu, cfg, bits, CAPACITY).await.map_err(js)?;
        let verdicts = runner.fermat_async(&self.gpu, &ns).await.map_err(js)?;
        Ok(verdicts.into_iter().map(u8::from).collect())
    }
}
