//! Kernel configuration and WGSL generation.

use bigmont_oracle::LIMBS;

const TEMPLATE: &str = include_str!("mont.wgsl");

/// Scalar helpers for TPI = 1: no cross-lane traffic at all.
const SCALAR_HELPERS: &str = "
fn bcast(v: u32, src_lane: u32, base: u32) -> u32 { return v; }
fn down1(v: u32, lane: u32) -> u32 { return 0u; }
fn up1(v: u32, lane: u32) -> u32 { return 0u; }
fn any_nonzero(v: u32) -> bool { return v != 0u; }
";

/// Subgroup helpers for TPI > 1; `base` is the subgroup lane id of this instance's lane 0.
const SUBGROUP_HELPERS: &str = "
fn bcast(v: u32, src_lane: u32, base: u32) -> u32 { return subgroupShuffle(v, base + src_lane); }
fn down1(v: u32, lane: u32) -> u32 { return select(0u, subgroupShuffleDown(v, 1u), lane != TPI - 1u); }
fn up1(v: u32, lane: u32) -> u32 { return select(0u, subgroupShuffleUp(v, 1u), lane != 0u); }
fn any_nonzero(v: u32) -> bool { return subgroupAny(v != 0u); }
";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelConfig {
    /// Lanes per number: 1 (one thread per number) or a cooperative width.
    pub tpi: u32,
    /// Workgroup size (invocations).
    pub wg: u32,
}

impl KernelConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.tpi, 1 | 8 | 16 | 32) {
            return Err(format!("tpi {} not supported (1, 8, 16, 32)", self.tpi));
        }
        if LIMBS as u32 % self.tpi != 0 {
            return Err(format!("{LIMBS} limbs not divisible by tpi {}", self.tpi));
        }
        if self.wg == 0 || self.wg % self.tpi != 0 || self.wg > 256 {
            return Err(format!("workgroup size {} must be a multiple of tpi {} and at most 256", self.wg, self.tpi));
        }
        Ok(())
    }

    pub fn needs_subgroups(&self) -> bool {
        self.tpi > 1
    }

    pub fn instances_per_workgroup(&self) -> u32 {
        self.wg / self.tpi
    }

    pub fn limbs_per_lane(&self) -> u32 {
        LIMBS as u32 / self.tpi
    }

    /// The complete WGSL source for this configuration.
    pub fn wgsl(&self) -> String {
        let (enable, sgparam, sglet, helpers) = if self.needs_subgroups() {
            // naga 30 rejects `enable subgroups;` as unimplemented; the subgroup built-ins need no
            // directive there and are gated by wgpu::Features::SUBGROUP at device creation.
            // Browsers REQUIRE `enable subgroups;` in WGSL, so the cooperative source emitted here
            // is not browser-valid as it stands; a web build must emit the directive (TODO(browser)).
            ("", ", @builtin(subgroup_invocation_id) sg_in: u32", "let sg = sg_in;", SUBGROUP_HELPERS)
        } else {
            ("", "", "let sg = 0u;", SCALAR_HELPERS)
        };
        TEMPLATE
            .replace("__ENABLE__", enable)
            .replace("__TPI__", &format!("{}u", self.tpi))
            .replace("__LPL__", &format!("{}u", self.limbs_per_lane()))
            .replace("__WG__", &format!("{}u", self.wg))
            .replace("__SGPARAM__", sgparam)
            .replace("__SGLET__", sglet)
            .replace("__SHUFFLES__", helpers)
    }
}
