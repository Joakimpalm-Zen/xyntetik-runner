#!/bin/bash
# R4.50.6 (olmo2 admission), the real file: allenai/Olmo-3-7B-Think (revision d97e442d, BF16 safetensors, YaRN
# factor 8 over 8192, attention_factor 1.2079 = the default 1 + 0.1 ln 8, sliding window 4096) against Runner's CPU route
# at main f3ff8e84 on unsloth/Olmo-3-7B-Think-GGUF's Q8_0 (@1dcda0d5, sha256 3a4d7dda...). The standard read of
# docs/measure-a-quantized-file.md: gold-corpus-2k.txt, --score 20 (solo), gold-logits.py --max-positions 2000
# --stride 2 (it reads the corpus's first 4009 tokens, scored at -c 4096, inside the original 8192). The reference:
# transformers-main 046bdb8, BF16 weights cast to float32 per block (--upcast-at-use: the checkpoint is BF16), CPU,
# 16 threads. usage: run_r4506_olmo3_7b.sh [KV] (default f16). One model process at a time.
set -u
KV=${1:-f16}; SUF=$([ "$KV" = f16 ] && echo "" || echo "-kv$KV")
R=$HOME/r448v; W=$R/wt-main-f3ff8e84; O=$R/results/r4506/olmo3-7b-think; mkdir -p $O
HF=$R/files/olmo/allenai_Olmo-3-7B-Think; G=$R/files/olmo/unsloth_Olmo-3-7B-Think-GGUF/Olmo-3-7B-Think-Q8_0.gguf
TF=$R/src/transformers-main/src; PY=$R/venv/bin/python; CORPUS=$W/tests/fixtures/gold-corpus-2k.txt
stop() { echo "STOP: $* $(date -u +%FT%TZ)"; exit 5; }
echo "== start $(date -u +%FT%TZ)"
[ "$(git -C $W rev-parse --short=8 HEAD)" = f3ff8e84 ] && [ -z "$(git -C $W status --porcelain --untracked-files=no)" ] || stop "build tree"
[ "$(git -C $R/src/transformers-main rev-parse --short=7 HEAD)" = 046bdb8 ] || stop "transformers-main"
[ "$(sha256sum $G | cut -c1-8)" = 3a4d7dda ] || stop "the Q8_0 GGUF's sha256"
(cd $HF && for f in model-0000[1-3]-of-00003.safetensors; do grep -q "$(sha256sum $f | cut -c1-64)  $f" SHA256SUMS || exit 1; done) || stop "the checkpoint's sha256"
for p in $(pgrep -f "inkling2-read-bin|read_timed|runner --serve|runner -m|test-inkling2|ref_firstlight|llama-server|gold-logits"); do
  grep -qa pgrep /proc/$p/cmdline 2>/dev/null && continue
  a=$$; while [ "$a" -gt 1 ] && [ "$a" != "$p" ]; do a=$(awk '{print $4}' /proc/$a/stat 2>/dev/null || echo 1); done
  [ "$a" = "$p" ] && continue   # an ancestor of this script (its caller's command line), not a model process
  stop "another model process: $p"; done
echo "runner $(sha256sum $W/runner | cut -c1-16) build f3ff8e84; corpus $(sha256sum $CORPUS | cut -c1-16)"
echo "== score $(date -u +%FT%TZ)"
$W/runner -m $G --score 20 -f $CORPUS --score-tokens 4009 -c 4096 -t 16 --gpu off --kv $KV --score-out $O/q8_0$SUF.score.json \
  > $O/q8_0$SUF.score.stdout 2> $O/q8_0$SUF.score.stderr || stop "score rc $? ($(tail -n 2 $O/q8_0$SUF.score.stderr | tr '\n' ' '))"
grep -i "olmo2\|admitted\|yarn\|ctx\|route" $O/q8_0$SUF.score.stderr | head -5 | cut -c1-220
echo "== reference $(date -u +%FT%TZ)"
env PYTHONPATH=$TF OMP_NUM_THREADS=16 $PY $W/scripts/gold-logits.py --hf $HF --corpus $CORPUS --score-a $O/q8_0$SUF.score.json \
  --gguf-a $G --repo-a unsloth/Olmo-3-7B-Think-GGUF@1dcda0d5 --max-positions 2000 --stride 2 --threads 16 --upcast-at-use \
  --out $O/gold-olmo3-7b-q8_0$SUF.json > $O/gold$SUF.stdout 2> $O/gold$SUF.stderr &
pid=$!; echo "r4506 olmo3 gold pid $pid $(date -u +%FT%TZ)" >> $R/logs/pids.txt
peak=0
while kill -0 $pid 2>/dev/null; do
  rss=$(awk '/^VmRSS:/{print $2*1024}' /proc/$pid/status 2>/dev/null || echo 0)
  [ "${rss:-0}" -gt "$peak" ] && peak=$rss
  if [ "${rss:-0}" -gt 62000000000 ]; then echo "GUARD: RSS $rss > 62 GB, killing recorded pid $pid"; kill $pid; fi
  sleep 2
done
wait $pid || stop "gold-logits rc $? ($(tail -n 3 $O/gold$SUF.stderr | tr '\n' ' '))"
echo "reference peak RSS $peak"
grep -E '"mean_kld_vs_gold"|"max_kld_vs_gold"|"top1_agreement_pct"|"margin_qualified_top1_pct"|"matched_by_id_positions"|"id_spelling_mismatches"' $O/gold$SUF.stdout
echo "== done $(date -u +%FT%TZ)"
