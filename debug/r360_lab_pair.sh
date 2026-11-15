#!/bin/bash
# R3.60 on the lab: does a prefill that takes the batch route at every width
# keep -b 512 bit-identical to -b 64 on a CPU MoE, and what does -b 512 gain?
#   lab_pair.sh MODEL.gguf [THREADS]       (run from a runner checkout)
set -eu
M=$1; T=${2:-26}
git fetch -q origin experiment/r3.60-route-by-pass
[ -d ../r360-exp ] || git worktree add -q ../r360-exp 599d97a9    # the experiment
[ -d ../r360-base ] || git worktree add -q ../r360-base 704b339a  # its base: the same tree, routes unchanged
(cd ../r360-exp && make -j16 runner >/dev/null) && (cd ../r360-base && make -j16 runner >/dev/null)
mkdir -p r360-out && cd r360-out
python3 - > prompt.txt <<'PY'
w = ("the quick brown fox jumps over a lazy dog while seven bright stars watch from far above "
     "and a quiet river runs through green fields toward the old stone bridge where children play "
     "every summer afternoon").split()
print(" ".join((w * 200)[:1601]))
PY
for round in 1 2 3; do
  for arm in base exp; do
    for b in 64 512; do
      RUNNER_TEST_KV_DUMP=1000 ../../r360-$arm/runner -m "$M" --gpu off -t $T -c 4096 -b $b \
        --temp 0 -n 32 -f prompt.txt > $arm-b$b-r$round.out 2> $arm-b$b-r$round.err
    done
  done
done
for f in *-r*.err; do echo "$f $(grep -h '^prompt:' $f)"; done
for arm in base exp; do
  if cmp -s $arm-b64-r1.out $arm-b512-r1.out && diff -q <(grep kvdump $arm-b64-r1.err) <(grep kvdump $arm-b512-r1.err) >/dev/null
  then echo "$arm: -b 64 and -b 512 IDENTICAL (32 tokens, every layer's K and V at position 1000)"
  else echo "$arm: -b 64 and -b 512 DIFFER"; fi
done
