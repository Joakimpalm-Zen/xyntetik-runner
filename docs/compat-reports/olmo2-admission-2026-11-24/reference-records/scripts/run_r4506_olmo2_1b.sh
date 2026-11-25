#!/bin/bash
# R4.50.6 (olmo2 admission): the FIRST olmo2 reference read, on the small allenai/OLMo-2-0425-1B (revision a1847dff,
# float32 safetensors) against Runner's CPU route on the publisher's own F16 GGUF (allenai/OLMo-2-0425-1B-GGUF
# d4cd762e, sha256 452d42f1...), at main f3ff8e84 (#1017, olmo2 CPU). The standard read of
# docs/measure-a-quantized-file.md: tests/fixtures/gold-corpus-2k.txt, Runner's --score 20 pass (solo, the decode
# arithmetic), gold-logits.py --max-positions 2000 --stride 2. The model's training context is 4096
# (max_position_embeddings): the corpus is 7869 tokens on its tokenizer; gold-logits reads its first 4009 (2000
# positions x stride 2, + its 8-token minimum prefix, + 1) and checks each id against the score file's, so the score
# covers exactly those 4009 tokens at -c 4096. The reference: transformers-main 046bdb8 (the ruling of record), float32, CPU, 16 threads.
# One model process at a time; the score first, then the reference.
# usage: run_r4506_olmo2_1b.sh [KV] (default f16, Runner's default KV cache; f32 for the arithmetic alone, files
# suffixed -kvf32)
set -u
KV=${1:-f16}; SUF=$([ "$KV" = f16 ] && echo "" || echo "-kv$KV")
R=$HOME/r448v; W=$R/wt-main-f3ff8e84; O=$R/results/r4506/olmo2-1b; mkdir -p $O
HF=$R/files/olmo/allenai_OLMo-2-0425-1B; G=$R/files/olmo/allenai_OLMo-2-0425-1B-GGUF/OLMo-2-0425-1B-F16.gguf
TF=$R/src/transformers-main/src; PY=$R/venv/bin/python; CORPUS=$W/tests/fixtures/gold-corpus-2k.txt
stop() { echo "STOP: $* $(date -u +%FT%TZ)"; exit 5; }
echo "== start $(date -u +%FT%TZ)"
[ "$(git -C $W rev-parse --short=8 HEAD)" = f3ff8e84 ] && [ -z "$(git -C $W status --porcelain --untracked-files=no)" ] || stop "build tree"
[ "$(git -C $R/src/transformers-main rev-parse --short=7 HEAD)" = 046bdb8 ] || stop "transformers-main"
[ "$(sha256sum $G | cut -c1-8)" = 452d42f1 ] || stop "the F16 GGUF's sha256"
(cd $HF && for f in model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do grep -q "$(sha256sum $f | cut -c1-64)  $f" SHA256SUMS || exit 1; done) || stop "the checkpoint's sha256"
for p in $(pgrep -f "inkling2-read-bin|read_timed|runner --serve|runner -m|test-inkling2|ref_firstlight|llama-server|gold-logits"); do
  grep -qa pgrep /proc/$p/cmdline 2>/dev/null && continue; stop "another model process: $p"; done
echo "runner $(sha256sum $W/runner | cut -c1-16) build f3ff8e84; corpus $(sha256sum $CORPUS | cut -c1-16)"
echo "== score $(date -u +%FT%TZ)"
$W/runner -m $G --score 20 -f $CORPUS --score-tokens 4009 -c 4096 -t 16 --gpu off --kv $KV --score-out $O/f16$SUF.score.json \
  > $O/f16$SUF.score.stdout 2> $O/f16$SUF.score.stderr || stop "score rc $? ($(tail -n 2 $O/f16$SUF.score.stderr | tr '\n' ' '))"
grep -i "olmo2\|admitted\|route\|ctx" $O/f16$SUF.score.stderr | head -5 | cut -c1-200
echo "== reference $(date -u +%FT%TZ)"
env PYTHONPATH=$TF OMP_NUM_THREADS=16 $PY $W/scripts/gold-logits.py --hf $HF --corpus $CORPUS --score-a $O/f16$SUF.score.json \
  --gguf-a $G --repo-a allenai/OLMo-2-0425-1B-GGUF@d4cd762e --max-positions 2000 --stride 2 --threads 16 \
  --out $O/gold-olmo2-1b-f16$SUF.json > $O/gold$SUF.stdout 2> $O/gold$SUF.stderr || stop "gold-logits rc $? ($(tail -n 3 $O/gold$SUF.stderr | tr '\n' ' '))"
grep -E '"mean_kld_vs_gold"|"max_kld_vs_gold"|"top1_agreement_pct"|"margin_qualified_top1_pct"|"matched_by_id_positions"' $O/gold$SUF.stdout
echo "== done $(date -u +%FT%TZ)"
