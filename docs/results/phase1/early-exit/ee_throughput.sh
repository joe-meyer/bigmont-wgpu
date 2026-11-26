#!/bin/bash
# Early-exit throughput on Beacon's 32,000-candidate fixture: in-driver EE (CUDA, bigmont) against the
# ee_drive.py wave path (plain CUDA driver, plain bigmont driver).
set -u
W=$HOME/ee-validate
P3=$HOME/beacon-test/p3
BM=$HOME/bigmont-ee-target/release/bigmont
ED=$HOME/beacon-test/kit/o1_kit/ee_drive.py
export BIGMONT_ADAPTER=4070 BIGMONT_BACKEND=Vulkan
cat > $W/bigmont_plain.sh <<EOF
#!/bin/bash
exec $BM "\$@" --backend gpu
EOF
chmod +x $W/bigmont_plain.sh
# input + manifest for ee_drive (sizes written by ee_validate.py: big_reg40.sizes is G2S1; build the json here)
python3 - <<PY
import struct, json, os
W = os.path.expanduser("~/ee-validate")
raw = open(W + "/big_reg40.sizes", "rb").read()
magic, nk = struct.unpack("<II", raw[:8])
sizes = list(struct.unpack("<%dI" % nk, raw[8:8 + 4 * nk]))
os.system("cp %s/big_2048.bin %s/big_reg40.bin" % (os.path.expanduser("~/beacon-test/p3"), W))
json.dump({"sizes": sizes}, open(W + "/big_reg40.manifest.json", "w"))
PY
run() { local label=$1; shift; local t0=$(date +%s.%N); "$@" > $W/$label.out 2> $W/$label.err; local rc=$?; local t1=$(date +%s.%N); printf "%-28s rc=%s %.2fs\n" "$label" $rc "$(echo "$t1 - $t0" | bc)"; }
for rep in 1 2; do
  echo "== repetition $rep"
  run "cuda_inDriverEE"   $P3/g1_fermat_v3 $W/big_reg40.bin $W/o_cuda_ee.v --ee $W/big_reg40.sizes --chunk 6 --growth 1.15 --marker 3
  run "bigmont_inDriverEE" $BM $W/big_reg40.bin $W/o_bm_ee.v --backend gpu --ee $W/big_reg40.sizes --chunk 6 --growth 1.15 --marker 3
  run "cuda_waves(ee_drive)" python3 $ED $W/big_reg40.bin $W/o_cuda_w.v --driver $P3/g1_fermat_v3 --gpu 0 --chunk 6 --growth 1.15 --marker 3
  run "bigmont_waves(ee_drive)" python3 $ED $W/big_reg40.bin $W/o_bm_w.v --driver $W/bigmont_plain.sh --gpu 0 --chunk 6 --growth 1.15 --marker 3
  cmp $W/o_cuda_ee.v $W/o_bm_ee.v && echo "in-driver EE: cuda == bigmont"
  cmp $W/o_cuda_w.v $W/o_bm_w.v && echo "wave path:    cuda == bigmont"
  cmp $W/o_cuda_ee.v $W/o_cuda_w.v && echo "cuda in-driver == cuda waves"
done
grep -h "waves=" $W/cuda_inDriverEE.err $W/bigmont_inDriverEE.err | sed -E 's/.*(g1_fermat\[EE\]:.*)/\1/' | cut -c1-200
tail -n 2 $W/bigmont_waves\(ee_drive\).err 2>/dev/null | cut -c1-200
