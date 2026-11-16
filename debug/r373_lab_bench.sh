#!/usr/bin/env bash
# EXPERIMENT R3.60 / R3.73 (not for merge): the lab's read of the x86 MXFP4
# decode. It builds debug/r360_expert_bench.c twice, against main's C decode
# and against the SIMD decode of branch r3.73-mxfp4-simd-decode (both pinned
# below), and runs them in ABBA order. No model file is read.
#
#   bash debug/r373_lab_bench.sh THREADS [ROUNDS] [MB_OF_WEIGHT_COPIES]
#   (on the lab: bash debug/r373_lab_bench.sh 26)
#
# What it decides:
#   R3.73 KILL (fixed beforehand): the SIMD decode under 2x faster per weight
#   than the C decode on this CPU. The reading is the MXFP4 "decode alone"
#   line, C over SIMD; the last lines print it per round and the verdict.
# What it informs:
#   R3.60: passes/tile per width with the SIMD decode. The tile's cost at 1
#   to 7 columns, against the native passes, is what the CPU variant
#   cb82532b paid at -b 64.
set -u
T=${1:?usage: r373_lab_bench.sh THREADS [ROUNDS] [MB]}
R=${2:-7}
MB=${3:-512}
C_SHA=1315c099c1fbe1a00a6bed8b04c4a4dbcbbabc09   # main: the C decode
S_SHA=80078a3a28c17e9c87e50b75c27fd20af1e34efa   # r3.73-mxfp4-simd-decode
BENCH=debug/r360_expert_bench.c
[ -f "$BENCH" ] || { echo "run from a checkout of experiment/r3.60-expert-bench"; exit 2; }
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
echo "== host $(date -u +%H:%M:%SZ)"
lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|Thread\(s\) per core' || true
echo "avx512: $(grep -o -w 'avx512[a-z0-9_]*' /proc/cpuinfo 2>/dev/null | sort -u | tr '\n' ' ')"
echo "load1: $(cut -d' ' -f1 /proc/loadavg 2>/dev/null)  compiler: $(${CC:-gcc} --version | head -1)"
for v in c simd; do
  sha=$C_SHA; [ $v = simd ] && sha=$S_SHA
  mkdir -p "$W/$v"
  git archive "$sha" src | tar -x -C "$W/$v" || { echo "cannot read $sha (git fetch origin first)"; exit 2; }
  ${CC:-gcc} -O3 -ffast-math -std=gnu11 -march=native -I "$W/$v/src" "$BENCH" "$W/$v/src/quants.c" \
      -o "$W/bench-$v" -lm -lpthread
  echo "build $v ($sha) exit $?"
done
n=0
for order in "c simd" "simd c"; do
  for v in $order; do
    n=$((n + 1))
    echo "== run $n: $v decode, $(date -u +%H:%M:%SZ), load1 $(cut -d' ' -f1 /proc/loadavg 2>/dev/null)"
    "$W/bench-$v" "$T" "$R" "$MB" | tee "$W/out-$n-$v.txt"
    echo "bench $v exit ${PIPESTATUS[0]}"
  done
done
echo "== R3.73: MXFP4 decode alone, ms a product (C, then SIMD)"
c=$(for f in "$W"/out-*-c.txt; do grep -m1 'decode alone' "$f" | sed -E 's/.*: ([0-9.]+) ms.*/\1/'; done | tr '\n' ' ')
s=$(for f in "$W"/out-*-simd.txt; do grep -m1 'decode alone' "$f" | sed -E 's/.*: ([0-9.]+) ms.*/\1/'; done | tr '\n' ' ')
echo "C: $c   SIMD: $s"
python3 - "$c" "$s" <<'PY'
import sys
c = [float(v) for v in sys.argv[1].split()]; s = [float(v) for v in sys.argv[2].split()]
r = min(c) / min(s)
print("C over SIMD, best of each: %.2fx per weight (KILL under 2.00x): %s" % (r, "holds" if r >= 2 else "KILLED"))
print("per run: " + ", ".join("%.2fx" % (a / b) for a, b in zip(c, s)))
PY
