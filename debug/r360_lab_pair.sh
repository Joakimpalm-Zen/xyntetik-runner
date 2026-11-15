#!/bin/bash
# R3.60 on the lab, the CPU-only variant (route by pass on the CPU, the GPU
# routes untouched). Run from a runner checkout:
#   debug/r360_lab_pair.sh MODEL.gguf [THREADS]
# (e) MoE identity FIRST: the variant at -b 64 and -b 512 must give the same 32
#     greedy tokens and the same K and V in every layer at position 10 and at
#     the last prompt position. The prompt is trimmed until -b 64 leaves a
#     one-row tail, so the base, which takes another route for that row, is a
#     live control: it must differ. If the variant differs, it dies here,
#     before any timing (exit 2).
# (d) the gain: three interleaved rounds of the variant at -b 64 and -b 512 and
#     the base at -b 64; kill if the variant's -b 512 over its -b 64 is under 1.3x.
set -eu
M=$1; T=${2:-26}
VAR=cb82532b; BASE=704b339a
git fetch -q origin experiment/r3.60-route-by-pass-cpu
[ -d ../r360-var ] || git worktree add -q ../r360-var $VAR
[ -d ../r360-base ] || git worktree add -q ../r360-base $BASE
(cd ../r360-var && make -j16 runner >/dev/null) && (cd ../r360-base && make -j16 runner >/dev/null)
mkdir -p r360-out && cd r360-out
run() {   # run ARM B POS OUT [N_GEN]
  RUNNER_TEST_KV_DUMP=$3 ../../r360-$1/runner -m "$M" --gpu off -t $T -c 4096 -b $2 \
    --temp 0 -n ${5:-32} -f prompt.txt > $4.out 2> $4.err
}
ntok() { grep -o '^prompt: [0-9]*' $1.err | grep -o '[0-9]*'; }
# the prompt: 1,601 words, trimmed until N % 64 == 1 (one probe is one prefill)
python3 - > words.txt <<'PY'
w = ("the quick brown fox jumps over a lazy dog while seven bright stars watch from far above "
     "and a quiet river runs through green fields toward the old stone bridge where children play "
     "every summer afternoon").split()
print("\n".join((w * 200)[:1601]))
PY
nw=1601
for probe in 1 2 3 4 5 6 7 8; do
  head -n $nw words.txt | tr '\n' ' ' | sed 's/ $//' > prompt.txt
  run base 64 0 probe 1; N=$(ntok probe); r=$((N % 64))
  echo "probe $probe: $nw words, $N tokens, tail $r at -b 64"
  [ $r -eq 1 ] && break
  if [ $r -eq 0 ]; then nw=$((nw - 1)); else nw=$((nw - (r - 1))); fi
done
[ $((N % 64)) -eq 1 ] || { echo "no prompt with a one-row tail at -b 64 after 8 probes"; exit 3; }
LAST=$((N - 1))
echo "== (e) MoE identity: $N tokens, K/V at positions 10 and $LAST"
for arm in base var; do for b in 64 512; do for pos in 10 $LAST; do run $arm $b $pos id-$arm-b$b-p$pos; done; done; done
same() {   # same ARM: -b 64 and -b 512 agree at both positions
  for pos in 10 $LAST; do
    cmp -s id-$1-b64-p$pos.out id-$1-b512-p$pos.out || return 1
    diff -q <(grep kvdump id-$1-b64-p$pos.err) <(grep kvdump id-$1-b512-p$pos.err) >/dev/null || return 1
  done
}
if same base; then echo "base: IDENTICAL at -b 64 and -b 512: the live control is NOT live on this model"; else echo "base: differs (the live control is live)"; fi
if same var; then echo "variant: IDENTICAL at -b 64 and -b 512 (32 tokens, every layer's K and V at 10 and $LAST)"
else echo "variant: DIFFERS at -b 64 and -b 512: MoE identity fails, the variant dies before any timing"; exit 2; fi
echo "== (d) the gain: 3 interleaved rounds"
for round in 1 2 3; do
  run var 64 0 t-var-b64-r$round; run var 512 0 t-var-b512-r$round; run base 64 0 t-base-b64-r$round
done
for f in t-*.err; do echo "$f $(grep -h '^prompt:' $f)"; done
python3 - <<'PY'
import glob, re, statistics
def med(p): return statistics.median(float(re.search(r'prompt: \d+ tok, ([\d.]+) tok/s', open(f).read()).group(1)) for f in glob.glob(p))
v64, v512, b64 = med('t-var-b64-r*.err'), med('t-var-b512-r*.err'), med('t-base-b64-r*.err')
print('median prompt tok/s: variant -b 64 %.1f, variant -b 512 %.1f, base -b 64 %.1f' % (v64, v512, b64))
print('GAIN variant -b 512 / -b 64: %.3fx (kill under 1.3x): %s' % (v512 / v64, 'KILL' if v512 / v64 < 1.3 else 'PASS'))
print('route cost on this MoE, variant -b 64 / base -b 64: %.3fx' % (v64 / b64))
PY
