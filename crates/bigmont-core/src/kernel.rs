//! Kernel configuration and WGSL generation.

use crate::scheme::{Caps, LimbScheme};
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
    /// The target the current build runs on: [`Target::Browser`] for wasm32, otherwise native.
    pub const fn host() -> Target {
        if cfg!(target_arch = "wasm32") {
            Target::Browser
        } else {
            Target::Native
        }
    }

    /// What this target can run: a browser offers no 64-bit integers, native offers everything.
    pub fn offers(self) -> Caps {
        match self {
            Target::Native => Caps {
                subgroups: true,
                int64: true,
            },
            Target::Browser => Caps::BROWSER,
        }
    }

    pub(crate) fn prelude(self) -> &'static str {
        match self {
            Target::Native => "",
            Target::Browser => {
                "enable subgroups;
diagnostic(off, subgroup_uniformity);
"
            }
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
    /// Limbs per number; a multiple of `tpi`.
    pub limbs: u32,
    /// The limb representation (radix) of the kernel.
    pub scheme: LimbScheme,
}

impl KernelConfig {
    /// The Phase 0 size class: 160 limbs (2,029-bit numbers).
    pub const fn new(tpi: u32, wg: u32, variant: Variant) -> Self {
        Self {
            tpi,
            wg,
            variant,
            limbs: LIMBS as u32,
            scheme: LimbScheme::Limb13,
        }
    }

    pub const fn with_limbs(mut self, limbs: u32) -> Self {
        self.limbs = limbs;
        self
    }

    /// Config for numbers of `bits` bits that is safe for the Fermat ladder (R > 16n): the smallest
    /// multiple of `tpi` that is at least `limbs_for_bits(bits)`.
    pub fn for_bits(bits: u32, tpi: u32, wg: u32, variant: Variant) -> Self {
        Self::for_bits_scheme(LimbScheme::Limb13, bits, tpi, wg, variant)
    }

    /// [`KernelConfig::for_bits`] for a given limb scheme.
    pub fn for_bits_scheme(
        scheme: LimbScheme,
        bits: u32,
        tpi: u32,
        wg: u32,
        variant: Variant,
    ) -> Self {
        let need = scheme.limbs_for_bits(bits);
        Self {
            scheme,
            ..Self::new(tpi, wg, variant)
        }
        .with_limbs(need.div_ceil(tpi) * tpi)
    }

    /// The same scheme, lane count, workgroup size and variant, with the limb count for `bits`.
    pub fn retarget(&self, bits: u32) -> KernelConfig {
        Self::for_bits_scheme(self.scheme, bits, self.tpi, self.wg, self.variant)
    }

    /// `log2 R` of this configuration.
    pub fn r_bits(&self) -> u32 {
        self.scheme.r_bits(self.limbs)
    }

    /// What a device must offer to run this kernel.
    pub fn requires(&self) -> Caps {
        Caps {
            subgroups: self.needs_subgroups(),
            int64: self.scheme.requires().int64,
        }
    }

    /// `Ok` when a device offering `have` can run this kernel, else an error naming what is missing
    /// ("needs int64, which the adapter lacks").
    pub fn check_caps(&self, have: Caps) -> Result<(), String> {
        let missing = have.missing(self.requires());
        if missing.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{self:?} needs {}, which the adapter lacks",
                missing.join(" and ")
            ))
        }
    }

    /// `Err` when `target` cannot run this kernel's scheme (a 32-bit kernel needs int64, which a
    /// browser does not offer).
    fn check_target(&self, target: Target) -> Result<(), String> {
        let missing = target.offers().missing(self.scheme.requires());
        if missing.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{:?} needs {missing:?}, which the {target:?} target does not offer",
                self.scheme
            ))
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.tpi, 1 | 4 | 8 | 16 | 32) {
            return Err(format!("tpi {} not supported (1, 4, 8, 16, 32)", self.tpi));
        }
        if self.limbs < self.tpi || !self.limbs.is_multiple_of(self.tpi) || self.limbs > 1024 {
            return Err(format!(
                "{} limbs must be a multiple of tpi {} and at most 1024",
                self.limbs, self.tpi
            ));
        }
        if self.wg == 0 || !self.wg.is_multiple_of(self.tpi) || self.wg > 256 {
            return Err(format!(
                "workgroup size {} must be a multiple of tpi {} and at most 256",
                self.wg, self.tpi
            ));
        }
        if self.variant == Variant::Unrolled && self.tpi == 1 {
            return Err(
                "the unrolled variant supports tpi 4, 8, 16, 32 only (tpi 1 is rolled-only)"
                    .to_string(),
            );
        }
        if self.scheme == LimbScheme::Limb32 && (self.variant != Variant::Unrolled || self.tpi == 1)
        {
            return Err(
                "32-bit limbs are generated for the unrolled variant with tpi 4, 8, 16, 32 only"
                    .to_string(),
            );
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
        self.try_wgsl_for(target).unwrap_or_else(|e| panic!("{e}"))
    }

    /// [`KernelConfig::wgsl_for`], or the reason this configuration has no source for `target`.
    pub fn try_wgsl_for(&self, target: Target) -> Result<String, String> {
        self.check_target(target)?;
        if self.scheme == LimbScheme::Limb32 {
            if self.variant != Variant::Unrolled {
                return Err("32-bit limbs are generated for the unrolled variant only".to_string());
            }
            return crate::limb32::wgsl(self, crate::unrolled::Mode::Squarings);
        }
        if self.variant == Variant::Unrolled {
            return Ok(crate::unrolled::wgsl(
                self,
                target,
                crate::unrolled::Mode::Squarings,
            ));
        }
        let (enable, sgparam, sglet, helpers) = if self.needs_subgroups() {
            (
                target.prelude(),
                ", @builtin(subgroup_invocation_id) sg_in: u32",
                "let sg = sg_in;",
                SUBGROUP_HELPERS,
            )
        } else {
            ("", "", "let sg = 0u;", SCALAR_HELPERS)
        };
        Ok(TEMPLATE
            .replace("__ENABLE__", enable)
            .replace("__TPI__", &format!("{}u", self.tpi))
            .replace("__LPL__", &format!("{}u", self.limbs_per_lane()))
            .replace("__WG__", &format!("{}u", self.wg))
            .replace("__SGPARAM__", sgparam)
            .replace("__SGLET__", sglet)
            .replace("__SHUFFLES__", helpers))
    }

    /// The complete Fermat-ladder WGSL source (unrolled variant only): bindings 0..3 as in squaring
    /// mode plus `@binding(4) e_words` (`n - 1` as u32 words, read-only storage) and `@binding(5) fp`
    /// (uniform: first iteration, nbits, words per instance). Needs R > 16n (see `for_bits`).
    pub fn fermat_wgsl_for(&self, target: Target) -> String {
        self.try_fermat_wgsl_for(target)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// [`KernelConfig::fermat_wgsl_for`], or the reason this configuration has no source for `target`.
    pub fn try_fermat_wgsl_for(&self, target: Target) -> Result<String, String> {
        self.check_target(target)?;
        if self.variant != Variant::Unrolled {
            return Err("the Fermat ladder is generated for the unrolled variant only".to_string());
        }
        if self.scheme == LimbScheme::Limb32 {
            return crate::limb32::wgsl(self, crate::unrolled::Mode::Fermat);
        }
        Ok(crate::unrolled::wgsl(
            self,
            target,
            crate::unrolled::Mode::Fermat,
        ))
    }

    /// The Montgomery multiply source (unrolled variant only): bindings 0..3 as in squaring mode
    /// plus `@binding(4) b_limbs` (read-only). Per instance `x <- x * b * R^-1 mod n`, `params.y`
    /// times, with the result in `[0, 2n)` for `x, b < 4n` (needs R > 16n, see `for_bits`).
    pub fn try_mul_wgsl_for(&self, target: Target) -> Result<String, String> {
        self.check_target(target)?;
        if self.variant != Variant::Unrolled {
            return Err(
                "the multiply kernel is generated for the unrolled variant only".to_string(),
            );
        }
        if self.scheme == LimbScheme::Limb32 {
            return crate::limb32::wgsl(self, crate::unrolled::Mode::Mul);
        }
        Ok(crate::unrolled::wgsl(
            self,
            target,
            crate::unrolled::Mode::Mul,
        ))
    }
}
