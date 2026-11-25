#!/bin/bash
# R4.50.6 (the admission bar set 2026-11-24 before this column: on each file, Runner's mean KL vs the publisher's
# float32 NOT WORSE than llama.cpp 56eea9e4's by separated intervals): llama.cpp's column by the SAME standard read:
# gold-corpus-2k.txt, 2000 positions at stride 2 (its first 4009 tokens), -c 4096, llama.cpp's default KV (f16, as
# Runner's), served by mainline's build-cpu-server (56eea9e4, -ngl 0, 16 threads), gold-logits.py's endpoint side
# (one cached prefix, one decoded token a position: the decode arithmetic, as Runner's solo --score). The reference
# as Runner's run: transformers-main 046bdb8, float32 (the 7B: --upcast-at-use). usage: run_r4506_llama.sh 1b|7b
set -u
WHICH=$1
R=$HOME/r448v; W=$R/wt-main-f3ff8e84; L=$R/src/llama.cpp; S=$L/build-cpu-server/bin/llama-server
TF=$R/src/transformers-main/src; PY=$R/venv/bin/python; CORPUS=$W/tests/fixtures/gold-corpus-2k.txt; PORT=58641
case $WHICH in
  1b) HF=$R/files/olmo/allenai_OLMo-2-0425-1B; G=$R/files/olmo/allenai_OLMo-2-0425-1B-GGUF/OLMo-2-0425-1B-F16.gguf
      REPO=allenai/OLMo-2-0425-1B-GGUF@d4cd762e; GS=452d42f1; O=$R/results/r4506/olmo2-1b; UP=""; TAG=f16 ;;
  1b-instruct) # R4.50.6.1: the 1B INSTRUCT's own column (allenai/OLMo-2-0425-1B-Instruct @48d788ec; its publisher's F16 GGUF)
      HF=$R/files/olmo/allenai_OLMo-2-0425-1B-Instruct; G=$R/files/olmo/allenai_OLMo-2-0425-1B-Instruct-GGUF/OLMo-2-0425-1B-Instruct-F16.gguf
      REPO=allenai/OLMo-2-0425-1B-Instruct-GGUF@62f8c199; GS=376249df; O=$R/results/r45061/olmo2-1b-instruct; UP=""; TAG=f16 ;;
  7b) HF=$R/files/olmo/allenai_Olmo-3-7B-Think; G=$R/files/olmo/unsloth_Olmo-3-7B-Think-GGUF/Olmo-3-7B-Think-Q8_0.gguf
      REPO=unsloth/Olmo-3-7B-Think-GGUF@1dcda0d5; GS=3a4d7dda; O=$R/results/r4506/olmo3-7b-think; UP="--upcast-at-use"; TAG=q8_0 ;;
  7b-af1) # (A), information only: a COPY of the 7B Q8_0 with olmo2.rope.scaling.attn_factor = 1.0 (gguf_set_metadata.py;
      # was 1.2079441547): does llama.cpp's gap close when its own mscale is no longer multiplied by the file's factor?
      HF=$R/files/olmo/allenai_Olmo-3-7B-Think; G=$R/work/af1/Olmo-3-7B-Think-Q8_0-af1.gguf
      REPO=unsloth/Olmo-3-7B-Think-GGUF@1dcda0d5-attn_factor-1.0-copy; GS=c3774832; O=$R/results/r4506/olmo3-7b-think; UP="--upcast-at-use"; TAG=q8_0-af1 ;;
  *) echo "usage: $0 1b|7b|7b-af1"; exit 2 ;;
esac
mkdir -p $O
stop() { echo "STOP: $* $(date -u +%FT%TZ)"; [ -n "${SP:-}" ] && kill $SP 2>/dev/null; exit 5; }
echo "== start $WHICH $(date -u +%FT%TZ)"
[ "$(git -C $L rev-parse --short=8 HEAD)" = 56eea9e4 ] || stop "llama.cpp not at 56eea9e4"
[ "$(git -C $L status --short --untracked-files=no | wc -l)" = 0 ] || stop "llama.cpp tracked changes"
[ "$(sha256sum $G | cut -c1-8)" = $GS ] || stop "the GGUF's sha256"
for p in $(pgrep -f "inkling2-read-bin|read_timed|runner --serve|runner -m|test-inkling2|ref_firstlight|llama-server|gold-logits"); do
  grep -qa pgrep /proc/$p/cmdline 2>/dev/null && continue
  a=$$; while [ "$a" -gt 1 ] && [ "$a" != "$p" ]; do a=$(awk '{print $4}' /proc/$a/stat 2>/dev/null || echo 1); done
  [ "$a" = "$p" ] && continue   # an ancestor of this script (its caller's command line), not a model process
  stop "another model process: $p"; done
echo "llama-server $(sha256sum $S | cut -c1-16) $($S --version 2>&1 | grep -i version | head -1)"
$S -m $G -c 4096 -t 16 -ngl 0 -np 1 --port $PORT > $O/llama-$TAG.server.log 2>&1 &
SP=$!; echo "r4506 llama-server $WHICH pid $SP $(date -u +%FT%TZ)" >> $R/logs/pids.txt
for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 1; done
curl -sf http://127.0.0.1:$PORT/health > /dev/null || stop "the server never came up"
echo "== server up $(date -u +%FT%TZ); reference + endpoint"
env PYTHONPATH=$TF OMP_NUM_THREADS=16 $PY $W/scripts/gold-logits.py --hf $HF --corpus $CORPUS \
  --endpoint-a http://127.0.0.1:$PORT --model-name-a $(basename $G) --gguf-a $G --repo-a $REPO \
  --max-positions 2000 --stride 2 --threads 16 $UP --out $O/gold-llama-$TAG.json > $O/gold-llama-$TAG.stdout 2> $O/gold-llama-$TAG.stderr
rc=$?
kill $SP; wait $SP 2>/dev/null; SP=""
[ $rc -eq 0 ] || stop "gold-logits rc $rc ($(tail -n 3 $O/gold-llama-$TAG.stderr | tr '\n' ' '))"
grep -E '"mean_kld_vs_gold"|"max_kld_vs_gold"|"top1_agreement_pct"|"margin_qualified_top1_pct"|"matched_by_id_positions"|"id_spelling_mismatches"|"positions_with_dropped_mass"' $O/gold-llama-$TAG.stdout
echo "== done $(date -u +%FT%TZ)"
