#!/bin/bash
# The golden pass for 1.6.0 (the lab box, 2026-10-24/25), per docs/golden-pass-2026-10-22.md and its gold1022-run.sh: the same
# 23 rows (12 models x bf16/served), corpus (gold-corpus-2k.txt, sha256 56e9fcba...), 2,000 positions, scorer
# (gold-logits.py, sha256 0be651b4...), llama.cpp 73a43d1 (flash attention off at bf16, on at the served file), servers at
# -t 16 -c 4096, the reference transformers 5.16.1 float32 on the CPU (venv gold-tf516; Apertus-8B through goldref-bf16.py
# as on 10-22), fresh servers per pass. Side a is Runner built with clang from runner main at or after 33c2a08f (the v1.6.0 merge) (the
# commit is fixed at start and logged). Two passes per row: vs llama.cpp (the README claim) and vs the published 1.5.0
# asset that 10-22 scored (sha256 4dfc4684..., kept from that run), so a move against 10-22 shows per position.
# Differences from 10-22, all host-side: no taskset (the host's CPUs carry outside tenants; the container has a 26-CPU
# quota), files read from NVMe copies (gold-stage.sh, read-back verified), the GPU hidden from every server
# (CUDA_VISIBLE_DEVICES empty; every route is --gpu off / a CPU-only llama.cpp build), and gemma-4-E2B scored on the
# 10-22 pair itself (bf16 converted by llama.cpp 73a43d1, q4_0 written from it by 1.5.0).
set -u
G=~/workspace/golden; W=$G/run-1025; mkdir -p $W; cd $W || exit 9
PY=python3; LS=~/workspace/llama-ref/build/bin/llama-server
ST=~/gold-stage; M=$ST/models; HF=$ST/hf; E2B=$ST/e2b; OLD=~/workspace/gold-1.5.0
NPOS=2000; STRIDE=1; PA=58741; PB=58742; PC=58743
LOGF=$W/gold1025.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOGF; }
export CUDA_VISIBLE_DEVICES=

until grep -q 'gold-stage done (0 failed)' $G/stage.log; do grep -q 'gold-stage done' $G/stage.log && { log "STAGING_FAILED"; exit 2; }; sleep 60; done
for p in $PA $PB $PC; do (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null && { log "PORT_BUSY $p"; exit 2; }; done

# side a: runner main, built clean with clang; the commit is fixed here
if [ ! -x $W/runner ]; then
  cd ~/runner && git fetch -q origin main && C=$(git rev-parse origin/main) || { log FETCH_FAIL; exit 2; }
  git merge-base --is-ancestor 33c2a08f $C || { log "MAIN_TOO_OLD $C"; exit 2; }   # the v1.6.0 merge
  WT=~/runner-gold-${C:0:8}; rm -rf $WT; git worktree prune; git worktree add -q --detach $WT $C || { log CHECKOUT_FAIL; exit 2; }
  (cd $WT && make -j16 runner CC=clang PYTHON=python3 > $W/build.log 2>&1) || { log BUILD_FAIL; exit 2; }
  cp $WT/runner $W/runner; echo $C > $W/RUNNER-COMMIT
  cp $WT/scripts/gold-logits.py $WT/tests/fixtures/gold-corpus-2k.txt $WT/docs/golden-pass-evidence/gold-2k/goldref-bf16.py $W/
  cd $W
fi
# side b of pass 2: the published 1.5.0 asset 10-22 scored
[ -x $W/runner150 ] || cp $OLD/runner $W/runner150
[ "$(sha256sum runner150 | cut -c1-16)" = 4dfc46845050c73a ] || { log "ASSET150_SHA_MISMATCH $(sha256sum runner150 | cut -c1-16)"; exit 3; }
[ "$(sha256sum gold-logits.py | cut -c1-16)" = 0be651b41c8e6e36 ] || { log "SCORER_CHANGED $(sha256sum gold-logits.py | cut -c1-16)"; exit 4; }
[ "$(sha256sum gold-corpus-2k.txt | cut -c1-16)" = 56e9fcba658f72ae ] || { log "CORPUS_CHANGED"; exit 4; }
log "runner $(./runner --version 2>&1 | head -1) commit $(cat RUNNER-COMMIT) sha256 $(sha256sum runner | cut -c1-16) (clang)"
log "baseline $(./runner150 --version 2>&1 | head -1) sha256 $(sha256sum runner150 | cut -c1-16) (the published 1.5.0 asset, 10-22's side a)"
log "llama.cpp $($LS --version 2>&1 | grep -i version | head -1)"
log "scorer gold-logits.py sha256 $(sha256sum gold-logits.py | cut -c1-16); corpus sha256 $(sha256sum gold-corpus-2k.txt | cut -c1-16)"
log "reference $($PY -c 'import sys,torch,transformers,tokenizers;print("python",sys.version.split()[0],"torch",torch.__version__,"transformers",transformers.__version__,"tokenizers",tokenizers.__version__)')"
CORPUS=$W/gold-corpus-2k.txt

stop() { for p in "$@"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done; sleep 3; for p in "$@"; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done; }

summ() {
  $PY -c "
import json,sys
d=json.load(open(sys.argv[1])); s=d.get('summary',d); a=s['sides']['a']; b=s['sides'].get('b',{}); g=s.get('significance',{})
print('  %-7s %-11s n=%4d  a kld %7.4f mq %6.2f%%  |  b kld %7.4f mq %6.2f%%  |  McNemar p=%.3f  Wilcoxon p=%.2e  %s' % (
  sys.argv[2], sys.argv[3], s['positions'], a['mean_kld_vs_gold'], a['margin_qualified_top1_pct'],
  b.get('mean_kld_vs_gold',-1), b.get('margin_qualified_top1_pct',-1),
  g.get('mcnemar_exact_p',-1), g.get('kld_wilcoxon_p',-1) or -1, g.get('verdict','?')))
" "$1" "$2" "$3" | tee -a $LOGF
}

run_pass() {  # $1 = pass label, $2 = side b port, $3 = side b model name, $4 = out json, rest = extra
  lbl=$1; pb=$2; nb=$3; out=$4; shift 4
  rm -f "$out"; t0=$(date +%s)
  $PY $WRAP gold-logits.py --hf "$hf" --corpus $CORPUS \
     --endpoint-a http://127.0.0.1:$PA --model-name-a $(basename "$g") \
     --endpoint-b http://127.0.0.1:$pb --model-name-b "$nb" \
     --max-positions $NPOS --stride $STRIDE --threads 32 "$@" --out "$out" > "${out%.json}.txt" 2> "${out%.json}.err"
  if [ -s "$out" ]; then summ "$out" "$tier" "$lbl"; log "  ($(( $(date +%s) - t0 ))s)"
  else log "  $tier $lbl FAILED: $(tail -2 "${out%.json}.err" | tr '\n' ' ' | cut -c1-160)"; fi
}

# fresh servers per pass, as on 09-07 and 10-22 (a warm runner server's prefix cache changes the last digits)
serve() {  # $1 = binary, $2 = port, $3 = log; prints the PID
  if [ "$1" = llama ]; then $LS -m "$g" --port $2 -t 16 -c 4096 --no-warmup -fa $fa $LX > $3 2>&1 &
  else $1 -m "$g" --serve --no-tray --port $2 --gpu off -t 16 -c 4096 > $3 2>&1 & fi
  echo $!
}
ready() {  # $1 = port
  if [ $1 = $PB ]; then curl -s http://127.0.0.1:$1/health 2>/dev/null | grep -q ok
  else curl -s http://127.0.0.1:$1/v1/models >/dev/null 2>&1; fi
}
up() {  # $1 $2 = ports, $3 $4 = their PIDs; 0 when both answer, 1 when one dies or 30 min pass
  for i in $(seq 1 360); do
    ready $1 && ready $2 && return 0
    kill -0 $3 2>/dev/null && kill -0 $4 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

gold() {
  name=$1; hf=$2; bf=$3; sv=$4; extra=$5; tag=$6; lxbf=${7:-}
  WRAP=""; [ "$name" = apertus-8b ] && WRAP=goldref-bf16.py
  log "=== $name$tag"
  for pair in "bf16:$bf:off" "served:$sv:on"; do
    tier=${pair%%:*}; rest=${pair#*:}; g=${rest%:*}; fa=${rest##*:}
    LX=""; [ "$tier" = bf16 ] && LX=$lxbf
    [ -n "$g" ] && [ -s "$g" ] || { log "  $tier: no file"; continue; }
    base=$W/gold1025-$name$tag-$tier
    [ -s $base-vs-llama.json ] && [ -s $base-vs-150.json ] && { log "  $tier: kept from an earlier start"; continue; }
    RP=$(serve ./runner $PA $base-rmain-p1.log); LP=$(serve llama $PB $base-llama.log)
    if up $PA $PB $RP $LP; then run_pass "vs-llama" $PB "$(basename "$g")" $base-vs-llama.json $extra
    else log "  $tier vs-llama FAILED, servers did not come up"; fi
    stop $RP $LP
    RP=$(serve ./runner $PA $base-rmain-p2.log); BP=$(serve ./runner150 $PC $base-r150.log)
    if up $PA $PC $RP $BP; then run_pass "vs-1.5.0" $PC "$(basename "$g")" $base-vs-150.json $extra
    else log "  $tier vs-1.5.0 FAILED, servers did not come up"; fi
    stop $RP $BP
    [ -s $base-vs-llama.json ] && [ -s $base-vs-150.json ] && $PY -c "
import json,sys
x=[json.load(open(f)) for f in sys.argv[1:3]]
a=[d.get('summary',d)['sides']['a'] for d in x]
same=all(a[0][k]==a[1][k] for k in ('mean_kld_vs_gold','max_kld_vs_gold','top1_agreement_pct','margin_qualified_top1_pct'))
print('  side a repeat across passes:', 'IDENTICAL' if same else 'DIFFERS %r' % ([(k,a[0][k],a[1][k]) for k in a[0] if a[0][k]!=a[1][k]],))
" $base-vs-llama.json $base-vs-150.json 2>&1 | tee -a $LOGF
  done
}

gold smollm2-135m        $HF/smollm2-135m        $M/gold-bf16/smollm2-135m-bf16.gguf        $M/SmolLM2-135M-Instruct-Q8_0/SmolLM2-135M-Instruct-Q8_0.gguf "" ""
gold qwen2.5-0.5b        $HF/qwen2.5-0.5b        $M/gold-bf16/qwen2.5-0.5b-bf16.gguf        $M/Qwen2.5-0.5B-Instruct-Q4_K_M/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf "" ""
gold qwen3-0.6b          $HF/qwen3-0.6b          $M/Qwen3-0.6B-bf16/Qwen3-0.6B-bf16.gguf    $M/Qwen3-0.6B-q4_0/Qwen3-0.6B-q4_0.gguf "" ""
gold qwen3.5-0.8b        $HF/qwen3.5-0.8b        $M/gold-bf16/qwen3.5-0.8b-bf16.gguf        $M/Qwen3.5-0.8B/Qwen3.5-0.8B-Q4_K_M.gguf "" ""
gold stablelm-2-1.6b     $HF/stablelm-2-1.6b     $M/gold-bf16/stablelm-2-1.6b-bf16.gguf     "" "" ""
gold granite-4.0-h-micro $HF/granite-4.0-h-micro $M/gold-bf16/granite-4.0-h-micro-bf16.gguf $M/granite-4.0-h-micro/granite-4.0-h-micro-Q4_K_M.gguf "" ""
gold granite-4.2-3b      $HF/granite-4.2-3b      $M/granite-4.2/granite-4.2-3b-bf16.gguf    $M/granite-4.2/granite-4.2-3b-Q4_K_M.gguf "" ""
gold trinity-nano        $HF/trinity-nano        $M/gold-bf16/trinity-nano-bf16.gguf        $M/trinity-nano/Trinity-Nano-Preview-Q8_0.gguf "" ""
gold gemma-3-4b-it       $HF/gemma-3-4b-it       $M/gold-bf16/gemma-3-4b-it-bf16.gguf       $M/google_gemma-3-4b-it-Q4_K_M/google_gemma-3-4b-it-Q4_K_M.gguf "" ""
gold gemma-4-e2b-it      $HF/gemma-4-e2b-it      $E2B/gemma-4-E2B-bf16.gguf                 $E2B/gemma-4-E2B-q4_0.gguf "--force-bos" "-forcebos"
gold qwen3-4b            $HF/qwen3-4b            $M/Qwen3-4B-tooluse/Qwen_Qwen3-4B-bf16.gguf $M/Qwen3-4B-Q4_K_M/Qwen3-4B-Q4_K_M.gguf "" ""
gold apertus-8b          $HF/apertus-8b          $M/gold-bf16/apertus-8b-bf16.gguf          $M/Apertus-8B-Instruct-2509-Q4_K_M/Apertus-8B-Instruct-2509-Q4_K_M.gguf "" "" "--no-jinja"
log GOLD_DONE
