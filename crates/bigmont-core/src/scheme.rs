//! Limb schemes and device capabilities: everything that depends on the radix of the multi-limb
//! representation (limb count for a bit length, R, host-side packing, the Montgomery factor) lives
//! here, so the runner and the tuner never name a limb width.

use bigmont_oracle::{LIMB_BITS, LIMB_MASK, limbs_for_bits, mont_square_chain_bits, neg_inv_limb, pack_limbs_n, unpack_limbs};
use num_bigint::BigUint;

/// What a device offers, or what a kernel needs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Caps {
    pub subgroups: bool,
    pub int64: bool,
}

impl Caps {
    /// What a browser (WebGPU) target offers: subgroups, no 64-bit integers.
    pub const BROWSER: Caps = Caps { subgroups: true, int64: false };

    /// Capabilities of `need` that `self` lacks, as words ("subgroups", "int64").
    pub fn missing(self, need: Caps) -> Vec<&'static str> {
        let mut out = Vec::new();
        if need.subgroups && !self.subgroups {
            out.push("subgroups");
        }
        if need.int64 && !self.int64 {
            out.push("int64");
        }
        out
    }

    /// Whether `self` offers everything `need` requires.
    pub fn satisfies(self, need: Caps) -> bool {
        self.missing(need).is_empty()
    }
}

/// How a multi-limb number is represented in the kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LimbScheme {
    /// 13-bit limbs, one per u32, products accumulated in u32.
    Limb13,
}

impl LimbScheme {
    pub const ALL: &'static [LimbScheme] = &[LimbScheme::Limb13];

    pub fn limb_bits(self) -> u32 {
        match self {
            LimbScheme::Limb13 => LIMB_BITS,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            LimbScheme::Limb13 => "13",
        }
    }

    pub fn parse(s: &str) -> Option<LimbScheme> {
        match s {
            "13" => Some(LimbScheme::Limb13),
            _ => None,
        }
    }

    /// What this scheme needs from the device by itself (subgroups are a property of the lane
    /// count, see `KernelConfig::requires`).
    pub fn requires(self) -> Caps {
        match self {
            LimbScheme::Limb13 => Caps::default(),
        }
    }

    /// `ceil((bits + 4) / limb_bits)`: the fewest limbs with R > 16n for every n below 2^bits.
    pub fn limbs_for_bits(self, bits: u32) -> u32 {
        match self {
            LimbScheme::Limb13 => limbs_for_bits(bits) as u32,
        }
    }

    /// `log2 R` for `limbs` limbs.
    pub fn r_bits(self, limbs: u32) -> u32 {
        self.limb_bits() * limbs
    }

    pub fn pack(self, x: &BigUint, limbs: usize) -> Vec<u32> {
        match self {
            LimbScheme::Limb13 => pack_limbs_n(x, limbs),
        }
    }

    pub fn unpack(self, limbs: &[u32]) -> BigUint {
        match self {
            LimbScheme::Limb13 => unpack_limbs(limbs),
        }
    }

    /// `-n^-1 mod 2^limb_bits` for odd `n`.
    pub fn neg_inv(self, n: &BigUint) -> u32 {
        match self {
            LimbScheme::Limb13 => neg_inv_limb(n),
        }
    }

    /// Whether every limb is below `2^limb_bits`.
    pub fn normalised(self, limbs: &[u32]) -> bool {
        match self {
            LimbScheme::Limb13 => limbs.iter().all(|&l| l <= LIMB_MASK),
        }
    }

    /// The oracle's Montgomery squaring chain at this radix with `limbs` limbs.
    pub fn chain(self, n: &BigUint, x0: &BigUint, squarings: u32, limbs: usize) -> BigUint {
        mont_square_chain_bits(n, x0, squarings, self.r_bits(limbs as u32))
    }
}
