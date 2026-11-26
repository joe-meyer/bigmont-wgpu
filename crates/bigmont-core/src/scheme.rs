//! Limb schemes and device capabilities: everything that depends on the radix of the multi-limb
//! representation (limb count for a bit length, R, host-side packing, the Montgomery factor) lives
//! here, so the runner and the tuner never name a limb width.

use bigmont_oracle::{
    LIMB_BITS, LIMB_MASK, limbs_for_bits, mont_square_chain_bits, neg_inv_limb, neg_inv_u32,
    pack_limbs_n, pack_u32_limbs, unpack_limbs, unpack_u32_limbs,
};
use num_bigint::BigUint;

/// What a device offers, or what a kernel needs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Caps {
    pub subgroups: bool,
    pub int64: bool,
}

impl Caps {
    /// What a browser (WebGPU) target offers: subgroups, no 64-bit integers.
    pub const BROWSER: Caps = Caps {
        subgroups: true,
        int64: false,
    };

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
    /// 32-bit limbs, one per u32, products in u64 (needs `SHADER_INT64`; never built for browsers).
    Limb32,
}

impl LimbScheme {
    pub const ALL: &'static [LimbScheme] = &[LimbScheme::Limb13, LimbScheme::Limb32];

    pub fn limb_bits(self) -> u32 {
        match self {
            LimbScheme::Limb13 => LIMB_BITS,
            LimbScheme::Limb32 => 32,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            LimbScheme::Limb13 => "13",
            LimbScheme::Limb32 => "32",
        }
    }

    pub fn parse(s: &str) -> Option<LimbScheme> {
        match s {
            "13" => Some(LimbScheme::Limb13),
            "32" => Some(LimbScheme::Limb32),
            _ => None,
        }
    }

    /// What this scheme needs from the device by itself (subgroups are a property of the lane
    /// count, see `KernelConfig::requires`).
    pub fn requires(self) -> Caps {
        match self {
            LimbScheme::Limb13 => Caps::default(),
            LimbScheme::Limb32 => Caps {
                subgroups: false,
                int64: true,
            },
        }
    }

    /// `ceil((bits + 4) / limb_bits)`: the fewest limbs with R > 16n for every n below 2^bits.
    pub fn limbs_for_bits(self, bits: u32) -> u32 {
        match self {
            LimbScheme::Limb13 => limbs_for_bits(bits) as u32,
            LimbScheme::Limb32 => (bits + 4).div_ceil(32),
        }
    }

    /// `log2 R` for `limbs` limbs.
    pub fn r_bits(self, limbs: u32) -> u32 {
        self.limb_bits() * limbs
    }

    pub fn pack(self, x: &BigUint, limbs: usize) -> Vec<u32> {
        match self {
            LimbScheme::Limb13 => pack_limbs_n(x, limbs),
            LimbScheme::Limb32 => pack_u32_limbs(x, limbs),
        }
    }

    /// Appends the `limbs` limbs that [`LimbScheme::pack`] returns for `x` to `out`, without
    /// allocating a vector for them (`pack` makes one, and an intermediate one for the digits, per
    /// call). Panics if `x` does not fit in `limbs` limbs, like `pack`.
    pub fn pack_into(self, x: &BigUint, limbs: usize, out: &mut Vec<u32>) {
        assert!(
            x.bits() <= u64::from(self.limb_bits()) * limbs as u64,
            "value does not fit in {limbs} limbs"
        );
        match self {
            LimbScheme::Limb13 => {
                // `acc` holds the `have` low bits not yet emitted; its bits above `have` are zero,
                // and `have + 32` stays below 64, so the shift below never loses bits.
                let mut digits = x.iter_u32_digits();
                let (mut acc, mut have) = (0u64, 0u32);
                for _ in 0..limbs {
                    if have < LIMB_BITS {
                        acc |= u64::from(digits.next().unwrap_or(0)) << have;
                        have += 32;
                    }
                    out.push(acc as u32 & LIMB_MASK);
                    acc >>= LIMB_BITS;
                    have -= LIMB_BITS;
                }
            }
            LimbScheme::Limb32 => {
                let start = out.len();
                out.extend(x.iter_u32_digits());
                out.resize(start + limbs, 0);
            }
        }
    }

    pub fn unpack(self, limbs: &[u32]) -> BigUint {
        match self {
            LimbScheme::Limb13 => unpack_limbs(limbs),
            LimbScheme::Limb32 => unpack_u32_limbs(limbs),
        }
    }

    /// `-n^-1 mod 2^limb_bits` for odd `n`.
    pub fn neg_inv(self, n: &BigUint) -> u32 {
        match self {
            LimbScheme::Limb13 => neg_inv_limb(n),
            LimbScheme::Limb32 => neg_inv_u32(n),
        }
    }

    /// Whether every limb is below `2^limb_bits`.
    pub fn normalised(self, limbs: &[u32]) -> bool {
        match self {
            LimbScheme::Limb13 => limbs.iter().all(|&l| l <= LIMB_MASK),
            LimbScheme::Limb32 => true,
        }
    }

    /// The oracle's Montgomery squaring chain at this radix with `limbs` limbs.
    pub fn chain(self, n: &BigUint, x0: &BigUint, squarings: u32, limbs: usize) -> BigUint {
        mont_square_chain_bits(n, x0, squarings, self.r_bits(limbs as u32))
    }
}
