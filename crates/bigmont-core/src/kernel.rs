//! Kernel configuration and WGSL generation.

use bigmont_oracle::{LIMBS, limbs_for_bits};

const TEMPLATE: &str = include_str!("mont.wgsl");

/// Scalar helpers for TPI = 1: no cross-lane traffic at all.
const SCALAR_HELPERS: &str = "
fn bcast(v: u32, src_lane: u32, base: u32) -> u32 { return v; }
fn down1(v: u32, lane: u32) -> u32 { return 0u; }
fn up1(v: u32, lane: u32) -> u32 { return 0u; }
fn any_nonzero(v: u32) -> bool { return v != 0u; }
";

/// Subgroup helpers for TPI > 1; `base` is the subgroup lane id of this instance's lane 0.
pub(crate) const SUBGROUP_HELPERS: &str = "
fn bcast(v: u32, src_lane: u32, base: u32) -> u32 { return subgroupShuffle(v, base + src_lane); }
fn down1(v: u32, lane: u32) -> u32 { return select(0u, subgroupShuffleDown(v, 1u), lane != TPI - 1u); }
fn up1(v: u32, lane: u32) -> u32 { return select(0u, subgroupShuffleUp(v, 1u), lane != 0u); }
fn any_nonzero(v: u32) -> bool { return subgroupAny(v != 0u); }
";

/// Which code generator produces the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// The `mont.wgsl` template: limb arrays and loops (the Phase 0 baseline).
    Rolled,
    /// Fully unrolled straight-line code with named scalar limbs (TPI 4, 8, 16, 32 only).
    Unrolled,
}

/// Where the generated WGSL will run. Browsers require `enable subgroups;`, and Chrome's uniformity
/// analysis cannot prove the loop exits uniform, so the browser prelude also switches that
/// diagnostic off. naga 30 (native wgpu) rejects `enable subgroups;` as unimplemented (the subgroup
/// built-ins need no directive there and are gated by `wgpu::Features::SUBGROUP` at device
/// creation), so native source has no prelude.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Native,
    Browser,
}

impl Target {
    pub(crate) fn prelude(self) -> &'static str {
        match self {
            Target::Native => "",
            Target::Browser => "enable subgroups;
diagnostic(off, subgroup_uniformity);
",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelConfig {
    /// Lanes per number: 1 (one thread per number) or a cooperative width.
    pub tpi: u32,
    /// Workgroup size (invocations).
    pub wg: u32,
    /// Code generator: rolled template or unrolled straight-line generator.
    pub variant: Variant,
    /// Limbs per number (13-bit limbs); a multiple of `tpi`.
    pub limbs: u32,
}

impl KernelConfig {
    /// The Phase 0 size class: 160 limbs (2,029-bit numbers).
    pub const fn new(tpi: u32, wg: u32, variant: Variant) -> Self {
        Self { tpi, wg, variant, limbs: LIMBS as u32 }
    }

    pub const fn with_limbs(mut self, limbs: u32) -> Self {
        self.limbs = limbs;
        self
    }

    /// Config for numbers of `bits` bits that is safe for the Fermat ladder (R > 16n): the smallest
    /// multiple of `tpi` that is at least `limbs_for_bits(bits)`.
    pub fn for_bits(bits: u32, tpi: u32, wg: u32, variant: Variant) -> Self {
        let need = limbs_for_bits(bits) as u32;
        Self::new(tpi, wg, variant).with_limbs(need.div_ceil(tpi) * tpi)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.tpi, 1 | 4 | 8 | 16 | 32) {
            return Err(format!("tpi {} not supported (1, 4, 8, 16, 32)", self.tpi));
        }
        if self.limbs < self.tpi || self.limbs % self.tpi != 0 || self.limbs > 1024 {
            return Err(format!("{} limbs must be a multiple of tpi {} and at most 1024", self.limbs, self.tpi));
        }
        if self.wg == 0 || self.wg % self.tpi != 0 || self.wg > 256 {
            return Err(format!("workgroup size {} must be a multiple of tpi {} and at most 256", self.wg, self.tpi));
        }
        if self.variant == Variant::Unrolled && self.tpi == 1 {
            return Err("the unrolled variant supports tpi 4, 8, 16, 32 only (tpi 1 is rolled-only)".to_string());
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
        self.limbs / self.tpi
    }

    /// The complete squaring-mode WGSL source for native wgpu.
    pub fn wgsl(&self) -> String {
        self.wgsl_for(Target::Native)
    }

    /// The complete squaring-mode WGSL source for this configuration.
    pub fn wgsl_for(&self, target: Target) -> String {
        if self.variant == Variant::Unrolled {
            return crate::unrolled::wgsl(self, target, crate::unrolled::Mode::Squarings);
        }
        let (enable, sgparam, sglet, helpers) = if self.needs_subgroups() {
            (target.prelude(), ", @builtin(subgroup_invocation_id) sg_in: u32", "let sg = sg_in;", SUBGROUP_HELPERS)
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

    /// The complete Fermat-ladder WGSL source (unrolled variant only): bindings 0..3 as in squaring
    /// mode plus `@binding(4) e_words` (`n - 1` as u32 words, read-only storage) and `@binding(5) fp`
    /// (uniform: first iteration, nbits, words per instance). Needs R > 16n (see `for_bits`).
    pub fn fermat_wgsl_for(&self, target: Target) -> String {
        assert_eq!(self.variant, Variant::Unrolled, "the Fermat ladder is generated for the unrolled variant only");
        crate::unrolled::wgsl(self, target, crate::unrolled::Mode::Fermat)
    }
}
