#!/usr/bin/env python3
"""Compare bigmont's in-driver early exit with Beacon's CUDA g1_fermat_v3 on Beacon's own fixtures.

Runs both drivers on the same candidate file / sizes sidecar / chunk / growth / marker and compares the
verdict files byte for byte, plus the `tested=` and `waves=` figures from the stderr summary lines.
"""
import os, random, re, struct, subprocess, sys, time

P3 = os.path.expanduser("~/beacon-test/p3")
CUDA = os.path.join(P3, "g1_fermat_v3")
BIGMONT = os.path.expanduser(os.environ.get("BIGMONT", "~/bigmont-ee-target/release/bigmont"))
WORK = os.path.expanduser("~/ee-validate")
os.makedirs(WORK, exist_ok=True)
MAGIC_C = 0x31433247
MAGIC_S = 0x31533247


def read_c(path):
    raw = open(path, "rb").read()
    magic, bits, count = struct.unpack("<IIQ", raw[:16])
    assert magic == MAGIC_C
    stride = bits // 8
    return bits, [raw[16 + i * stride: 16 + (i + 1) * stride] for i in range(count)]


def write_c(path, bits, recs):
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQ", MAGIC_C, bits, len(recs)))
        for r in recs:
            f.write(r)


def write_sizes(path, sizes):
    with open(path, "wb") as f:
        f.write(struct.pack("<II", MAGIC_S, len(sizes)))
        f.write(struct.pack("<%dI" % len(sizes), *sizes))


def sizes_regular(total, per):
    s = [per] * (total // per)
    if total % per:
        s.append(total % per)
    return s


def sizes_random(total, lo, hi, seed):
    rnd = random.Random(seed)
    s = []
    left = total
    while left > 0:
        n = min(left, rnd.randint(lo, hi))
        s.append(n)
        left -= n
    return s


def summary(err):
    m = re.search(r"tested=(\d+) \(([\d.]+)\) waves=(\d+)", err)
    return (int(m.group(1)), int(m.group(3))) if m else None


def run(cmd):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stderr, time.time() - t


def compare(name, cand, sizes_path, extra, bins=False):
    out_c, out_b = os.path.join(WORK, name + ".cuda.v"), os.path.join(WORK, name + ".bigmont.v")
    for p in (out_c, out_b):
        if os.path.exists(p):
            os.remove(p)
    base = ["--ee", sizes_path] + extra + (["--bins"] if bins else [])
    rc1, e1, t1 = run([CUDA, cand, out_c] + base)
    rc2, e2, t2 = run([BIGMONT, cand, out_b, "--backend", "gpu", "--adapter", "4070", "--vk-backend", "Vulkan"] + base)
    same = os.path.exists(out_c) and os.path.exists(out_b) and open(out_c, "rb").read() == open(out_b, "rb").read()
    s1, s2 = summary(e1), summary(e2)
    ok = rc1 == 0 and rc2 == 0 and same and s1 == s2
    print(f"{'PASS' if ok else 'FAIL'} {name}: cuda rc={rc1} {s1} {t1:.2f}s | bigmont rc={rc2} {s2} {t2:.2f}s | identical={same}")
    if not ok:
        print("  cuda stderr tail:", e1[-300:].replace("\n", " | "))
        print("  bigmont stderr tail:", e2[-300:].replace("\n", " | "))
    return ok


def main():
    results = []
    canary = os.path.join(P3, "canary_2048.bin")
    big = os.path.join(P3, "big_2048.bin")
    cb, crecs = read_c(canary)
    bb, brecs = read_c(big)
    print(f"canary: {len(crecs)} candidates, {cb} bits; big: {len(brecs)} candidates, {bb} bits")
    layouts = [
        ("canary_reg8", canary, sizes_regular(len(crecs), 8)),
        ("canary_rand", canary, sizes_random(len(crecs), 1, 40, 11)),
        ("big_reg40", big, sizes_regular(len(brecs), 40)),
        ("big_rand", big, sizes_random(len(brecs), 5, 120, 23)),
        ("big_one_k", big, [len(brecs)]),
    ]
    params = [(6, 1.15, 3), (2, 1.0, 7), (1, 1.5, 3), (50, 1.0, 3)]
    for name, cand, sizes in layouts:
        sp = os.path.join(WORK, name + ".sizes")
        write_sizes(sp, sizes)
        for chunk, growth, marker in params:
            # A single k of 32,000 positions with chunk 2 or 50 needs more than 512 schedule boundaries; the CUDA
            # driver keeps them in `int bnd[512]` and crashes (stack smashing), so those two are not comparable.
            if name == "big_one_k" and (chunk, growth) in ((2, 1.0), (50, 1.0)):
                print(f"SKIP {name}_c{chunk}_g{growth}: the CUDA driver overflows its 512-boundary schedule array here")
                continue
            results.append(compare(f"{name}_c{chunk}_g{growth}_m{marker}", cand, sp,
                                   ["--chunk", str(chunk), "--growth", str(growth), "--marker", str(marker)]))
    # --bins: split the big file in two, list file with CRLF endings
    half = len(brecs) // 2 + 3
    a, b = os.path.join(WORK, "big_a.bin"), os.path.join(WORK, "big_b.bin")
    write_c(a, bb, brecs[:half]); write_c(b, bb, brecs[half:])
    lst = os.path.join(WORK, "big.list")
    open(lst, "w", newline="").write(a + "\r\n" + b + "\r\n")
    sp = os.path.join(WORK, "big_bins.sizes")
    write_sizes(sp, sizes_random(len(brecs), 5, 120, 29))
    results.append(compare("big_bins", lst, sp, ["--chunk", "6", "--growth", "1.15", "--marker", "3"], bins=True))
    print(f"\n{sum(results)}/{len(results)} comparisons passed")
    sys.exit(0 if all(results) else 1)


main()
