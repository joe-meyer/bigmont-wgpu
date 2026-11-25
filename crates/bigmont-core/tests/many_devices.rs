//! Regression test for the static-TLS exhaustion: opening many `Gpu`s in one process must keep
//! finding the same adapter. Before the process-wide shared `Instance`, the 11th open on a Linux
//! machine with the NVIDIA driver lost the NVIDIA adapter (a named selection failed with "no
//! adapter matches"; an unnamed one silently picked a different adapter).

mod common;
use bigmont_core::gpu::Gpu;
use bigmont_core::kernel::{KernelConfig, Variant};

#[test]
fn opening_many_devices_keeps_the_same_adapter() {
    let Some(first) = common::open_gpu(KernelConfig::new(8, 64, Variant::Unrolled)) else { return };
    let (name, backend) = (first.info.name.clone(), first.info.backend);
    drop(first);
    for i in 1..40 {
        let gpu = Gpu::new().unwrap_or_else(|e| panic!("open #{i} failed: {e}"));
        assert_eq!((gpu.info.name.as_str(), gpu.info.backend), (name.as_str(), backend), "open #{i} picked a different adapter");
    }
}
