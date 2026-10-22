#!/bin/bash
# The README's comparison rows rerun on the published runner 1.5.0 Linux
# asset, 2026-10-22 (docs/golden-pass-2026-10-22.md). 2,000 positions against
# the publisher's float32 model, llama.cpp 73a43d1 beside Runner on the same
# file, CPU route (--gpu off) as on 2026-09-07. Each row runs two passes on
# one reference: 1.5.0 vs llama.cpp (the README claim) and 1.5.0 vs the
# published 1.4.2 (the release's own effect). Side a is a fresh 1.5.0 server
# in both passes. Apertus-8B bf16: llama.cpp 73a43d1 refuses that file's chat
# template, so its llama-server gets --no-jinja (templating only; the scorer
# reads raw positions). Apertus-8B's reference runs through goldref-bf16.py
# (bfloat16 storage widened to float32 at use, bitwise identical to the
# float32 load), the host's memory limit on the day being 32 GiB.
# This is the script that ran, with host paths replaced by ~/workspace and
# the resume logic of the day (passes kept across a host restart) left out.
set -u
W=$HOME/workspace/gold-1.5.0; mkdir -p $W; cd $W || exit 9
CPUS=40-55,104-119   # 16 cores, 32 hardware threads: every server and the scorer
PY=/opt/conda/bin/python3; LS=$HOME/workspace/llama-ref/build/bin/llama-server
M=$HOME/workspace/models; HF=$HOME/workspace/hf
NPOS=2000; STRIDE=1; PA=58731; PB=58732; PC=58733
LOGF=$W/gold1022.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOGF; }

# the runner: the published 1.5.0 asset, checked against the release's SHA256SUMS
if [ ! -x $W/runner ]; then
  curl -fsSL -o runner https://github.com/Joakimpalm-Zen/xyntetik-runner/releases/download/v1.5.0/runner-linux-x86_64 || { log ASSET_FETCH_FAIL; exit 2; }
  curl -fsSL -o SHA256SUMS https://github.com/Joakimpalm-Zen/xyntetik-runner/releases/download/v1.5.0/SHA256SUMS || { log SUMS_FETCH_FAIL; exit 2; }
  want=$(grep ' runner-linux-x86_64$' SHA256SUMS | awk '{print $1}'); have=$(sha256sum runner | awk '{print $1}')
  [ -n "$want" ] && [ "$want" = "$have" ] || { log "ASSET_SHA_MISMATCH want=$want have=$have"; rm -f runner; exit 3; }
  chmod +x runner
fi
if [ ! -x $W/runner142 ]; then
  curl -fsSL -o runner142 https://github.com/Joakimpalm-Zen/xyntetik-runner/releases/download/v1.4.2/runner-linux-x86_64 || { log ASSET142_FETCH_FAIL; exit 2; }
  curl -fsSL -o SHA256SUMS-142 https://github.com/Joakimpalm-Zen/xyntetik-runner/releases/download/v1.4.2/SHA256SUMS || { log SUMS142_FETCH_FAIL; exit 2; }
  want=$(grep ' runner-linux-x86_64$' SHA256SUMS-142 | awk '{print $1}'); have=$(sha256sum runner142 | awk '{print $1}')
  [ -n "$want" ] && [ "$want" = "$have" ] || { log "ASSET142_SHA_MISMATCH want=$want have=$have"; rm -f runner142; exit 3; }
  chmod +x runner142
fi
log "runner $(./runner --version 2>&1 | head -1) sha256 $(sha256sum runner | cut -c1-16)"
log "baseline $(./runner142 --version 2>&1 | head -1) sha256 $(sha256sum runner142 | cut -c1-16)"
log "llama.cpp $($LS --version 2>&1 | head -1)"
# the scoring script with the top-1 fix (sha256 0be651b4...), the corpus from the release tag
[ "$(sha256sum gold-logits.py | cut -c1-16)" = 0be651b41c8e6e36 ] || { log "SCORER_NOT_FIXED $(sha256sum gold-logits.py | cut -c1-16)"; exit 4; }
[ -s gold-corpus-2k.txt ] || { log NO_CORPUS; exit 2; }
log "scorer gold-logits.py sha256 $(sha256sum gold-logits.py | cut -c1-16) (PR #354)"
CORPUS=$W/gold-corpus-2k.txt

# gemma-4-E2B: the 2026-09-07 files are gone from the host. bf16 is rebuilt
# from the reference weights with the same llama.cpp checkout (73a43d1); the
# served file is a Q4_0 written by this runner from that bf16 (provenance
# differs from the original q4_0, so that row is new, not a rerun).
E2B=$W/e2b; mkdir -p $E2B
if [ ! -s $E2B/gemma-4-E2B-bf16.gguf ]; then
  taskset -c $CPUS $PY $HOME/workspace/llama-ref/convert_hf_to_gguf.py $HF/gemma-4-e2b-it --outtype bf16 --outfile $E2B/gemma-4-E2B-bf16.gguf > $E2B/convert.log 2>&1
  log "e2b bf16 convert rc=$? $(tail -1 $E2B/convert.log | cut -c1-120)"
fi
if [ -s $E2B/gemma-4-E2B-bf16.gguf ] && [ ! -s $E2B/gemma-4-E2B-q4_0.gguf ]; then
  taskset -c $CPUS ./runner -m $E2B/gemma-4-E2B-bf16.gguf --quantize $E2B/gemma-4-E2B-q4_0.gguf --quant q4_0 > $E2B/quant.log 2>&1
  log "e2b q4_0 quantize rc=$? $(tail -1 $E2B/quant.log | cut -c1-120)"
fi

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
  taskset -c $CPUS $PY $WRAP gold-logits.py --hf "$hf" --corpus $CORPUS \
     --endpoint-a http://127.0.0.1:$PA --model-name-a $(basename "$g") \
     --endpoint-b http://127.0.0.1:$pb --model-name-b "$nb" \
     --max-positions $NPOS --stride $STRIDE --threads 32 "$@" --out "$out" > "${out%.json}.txt" 2> "${out%.json}.err"
  if [ -s "$out" ]; then summ "$out" "$tier" "$lbl"; log "  ($(( $(date +%s) - t0 ))s)"
  else log "  $tier $lbl FAILED: $(tail -2 "${out%.json}.err" | tr '\n' ' ' | cut -c1-160)"; fi
}

# Every pass starts from fresh servers. A runner server's answer depends on
# what its prefix cache held: on the CPU route a prompt read in batches of 8
# or more takes the f32 tile (docs/kernel-routes.md) and a cached prefix the
# one-column dot, so a warm server's rows differ from a fresh one's in the last
# digits (measured in the pre-flight: every Q8_0 position moved, bf16 none).
# The 2026-09-07 pass started fresh servers per row; so does each pass here.
serve() {  # $1 = binary, $2 = port, $3 = log; prints the PID
  if [ "$1" = llama ]; then taskset -c $CPUS $LS -m "$g" --port $2 -t 16 -c 4096 --no-warmup -fa $fa $LX > $3 2>&1 &
  else taskset -c $CPUS $1 -m "$g" --serve --no-tray --port $2 --gpu off -t 16 -c 4096 > $3 2>&1 & fi
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
    base=$W/gold1022-$name$tag-$tier
    RP=$(serve ./runner $PA $base-r150-p1.log); LP=$(serve llama $PB $base-llama.log)
    if up $PA $PB $RP $LP; then run_pass "vs-llama" $PB "$(basename "$g")" $base-vs-llama.json $extra
    else log "  $tier vs-llama FAILED, servers did not come up"; fi
    stop $RP $LP
    RP=$(serve ./runner $PA $base-r150-p2.log); BP=$(serve ./runner142 $PC $base-r142.log)
    if up $PA $PC $RP $BP; then run_pass "vs-1.4.2" $PC "$(basename "$g")" $base-vs-142.json $extra
    else log "  $tier vs-1.4.2 FAILED, servers did not come up"; fi
    stop $RP $BP
    [ -s $base-vs-llama.json ] && [ -s $base-vs-142.json ] && $PY -c "
import json,sys
x=[json.load(open(f)) for f in sys.argv[1:3]]
a=[d.get('summary',d)['sides']['a'] for d in x]
same=all(a[0][k]==a[1][k] for k in ('mean_kld_vs_gold','max_kld_vs_gold','top1_agreement_pct','margin_qualified_top1_pct'))
print('  side a repeat across passes:', 'IDENTICAL' if same else 'DIFFERS %r' % ([(k,a[0][k],a[1][k]) for k in a[0] if a[0][k]!=a[1][k]],))
" $base-vs-llama.json $base-vs-142.json 2>&1 | tee -a $LOGF
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
