
const TPI: u32 = 8u;     // lanes per number
const LPL: u32 = 20u;     // limbs per lane; TPI * LPL = 160
const L: u32 = TPI * LPL;
const WG: u32 = 64u;
const B: u32 = 13u;
const MASK: u32 = 0x1fffu;
const PERIOD: u32 = 16u;      // iterations between light carry passes (see plan: bound < 2^32)

@group(0) @binding(0) var<uniform> params: vec4<u32>;          // x = instance count, y = squarings
@group(0) @binding(1) var<storage, read> n_limbs: array<u32>;
@group(0) @binding(2) var<storage, read> ninv_buf: array<u32>;
@group(0) @binding(3) var<storage, read_write> x_limbs: array<u32>;

var<private> xx: array<u32, LPL>;
var<private> nn: array<u32, LPL>;
var<private> tt: array<u32, LPL>;


fn bcast(v: u32, src_lane: u32, base: u32) -> u32 { return subgroupShuffle(v, base + src_lane); }
fn down1(v: u32, lane: u32) -> u32 { return select(0u, subgroupShuffleDown(v, 1u), lane != TPI - 1u); }
fn up1(v: u32, lane: u32) -> u32 { return select(0u, subgroupShuffleUp(v, 1u), lane != 0u); }
fn any_nonzero(v: u32) -> bool { return subgroupAny(v != 0u); }


// Each limb keeps its low 13 bits; the carry moves one limb up (across lanes for the top limb).
fn light_pass(lane: u32) {
    var c_in = up1(tt[LPL - 1u] >> B, lane);   // read before any lane modifies its limbs
    for (var c = 0u; c < LPL; c++) {
        let v = tt[c];
        tt[c] = (v & MASK) + c_in;
        c_in = v >> B;
    }
}

// Repeat light passes until every limb of every instance in the subgroup is below 2^13.
fn full_normalize(lane: u32) {
    for (var round = 0u; round < L + 2u; round++) {
        var big = 0u;
        for (var c = 0u; c < LPL; c++) { big |= tt[c] >> B; }
        if (!any_nonzero(big)) { break; }
        light_pass(lane);
    }
}

// tt <- xx * xx * R^-1 (mod n), tt normalised, value below 2n.
fn mont_sqr(lane: u32, base: u32, ninv: u32) {
    for (var c = 0u; c < LPL; c++) { tt[c] = 0u; }
    for (var i = 0u; i < L; i++) {
        let bi = bcast(xx[i % LPL], i / LPL, base);
        for (var c = 0u; c < LPL; c++) {
            tt[c] += (xx[c] & 0xffffffu) * (bi & 0xffffffu);
        }
        let m = (bcast(tt[0], 0u, base) * ninv) & MASK;
        for (var c = 0u; c < LPL; c++) {
            tt[c] += (nn[c] & 0xffffffu) * m;
        }
        let carry0 = bcast(tt[0], 0u, base) >> B;   // limb 0 is now divisible by 2^13
        let next_first = down1(tt[0], lane);        // limb 0 of the next lane becomes our new top limb
        for (var c = 0u; c + 1u < LPL; c++) { tt[c] = tt[c + 1u]; }
        tt[LPL - 1u] = next_first;
        if (lane == 0u) { tt[0] += carry0; }
        if ((i % PERIOD) == PERIOD - 1u) { light_pass(lane); }
    }
    full_normalize(lane);
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32 , @builtin(subgroup_invocation_id) sg_in: u32) {
    let sg = sg_in;
    let lane = sg % TPI;
    let base = sg - lane;
    let inst = wid.x * (WG / TPI) + li / TPI;
    let live = inst < params.x;
    let ic = min(inst, params.x - 1u);     // out-of-range lanes compute on a valid instance; their store is skipped
    let off = ic * L + lane * LPL;
    for (var c = 0u; c < LPL; c++) {
        xx[c] = x_limbs[off + c];
        nn[c] = n_limbs[off + c];
    }
    let ninv = ninv_buf[ic];
    for (var s = 0u; s < params.y; s++) {
        mont_sqr(lane, base, ninv);
        for (var c = 0u; c < LPL; c++) { xx[c] = tt[c]; }
    }
    if (live) {
        for (var c = 0u; c < LPL; c++) { x_limbs[off + c] = xx[c]; }
    }
}
