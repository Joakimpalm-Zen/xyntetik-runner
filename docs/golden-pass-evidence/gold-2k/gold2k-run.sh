#!/bin/sh
cd ~/workspace/runner-gold || exit 9
export PATH=$HOME/.conda/envs/ccbuild/bin:$PATH
unset CFLAGS
PY=/opt/conda/bin/python3; LS=~/workspace/llama-ref/build/bin/llama-server
M=~/workspace/models; HF=~/workspace/hf; E=~/workspace/gold2k-s1
CORPUS=tests/fixtures/gold-corpus-2k.txt
NPOS=2000
STRIDE=1
mkdir -p $E
PA=58711; PB=58712

gold() {
  name=$1; hf=$2; bf=$3; sv=$4; extra=$5; tag=$6
  echo "=== $name$tag"
  for pair in "bf16:$bf:off" "served:$sv:on"; do
    tier=$(echo $pair | cut -d: -f1); g=$(echo $pair | cut -d: -f2); fa=$(echo $pair | cut -d: -f3)
    [ -n "$g" ] && [ -s "$g" ] || { echo "  $tier: no file"; continue; }
    out=$E/gold2k-$name$tag-$tier.json; rm -f "$out"
    ./runner -m "$g" --serve --no-tray --port $PA --gpu off -c 4096 > $E/rsrv-$name$tag-$tier.log 2>&1 &
    $LS -m "$g" --port $PB -t 24 -c 4096 --no-warmup -fa $fa > $E/llsrv-$name$tag-$tier.log 2>&1 &
    ok=0
    for i in $(seq 1 360); do
      curl -s http://127.0.0.1:$PA/v1/models >/dev/null 2>&1 && curl -s http://127.0.0.1:$PB/health 2>/dev/null | grep -q "ok" && { ok=1; break; }
      sleep 5
    done
    if [ $ok -eq 1 ]; then
      t0=$(date +%s)
      $PY scripts/gold-logits.py --hf "$hf" --corpus $CORPUS \
         --endpoint-a http://127.0.0.1:$PA --model-name-a $(basename "$g") \
         --endpoint-b http://127.0.0.1:$PB --model-name-b $(basename "$g") \
         --max-positions $NPOS --stride $STRIDE --threads 48 $extra --out "$out" > $E/gold2k-$name$tag-$tier.txt 2> $E/gold2k-$name$tag-$tier.err
      if [ -s "$out" ]; then
        $PY -c "
import json
d=json.load(open('$out')); s=d['summary']; a=s['sides']['a']; b=s['sides'].get('b',{}); g=s.get('significance',{})
print('  %-7s fa=$fa n=%4d  runner kld %7.4f mq %6.2f%%  |  llama kld %7.4f mq %6.2f%%  |  McNemar p=%.3f  Wilcoxon p=%.2e  %s' % (
  '$tier', s['positions'], a['mean_kld_vs_gold'], a['margin_qualified_top1_pct'],
  b.get('mean_kld_vs_gold',-1), b.get('margin_qualified_top1_pct',-1),
  g.get('mcnemar_exact_p',-1), g.get('kld_wilcoxon_p',-1) or -1, g.get('verdict','?')))
"
        echo "  ($(( $(date +%s) - t0 ))s)"
      else echo "  $tier FAILED: $(tail -2 $E/gold2k-$name$tag-$tier.err | tr "\n" " " | sed "s/\x1b\[[0-9;]*m//g" | cut -c1-140)"; fi
    else echo "  $tier FAILED, servers did not come up"; fi
    pkill -f "port $PA" 2>/dev/null; pkill -f "port $PB" 2>/dev/null; sleep 3
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
gold gemma-4-e2b-it      $HF/gemma-4-e2b-it      $M/gemma-4-E2B-bf16/gemma-4-E2B-bf16.gguf  $M/gemma-4-E2B-q4_0/gemma-4-E2B-q4_0.gguf "--force-bos" "-forcebos"
gold qwen3-4b            $HF/qwen3-4b            $M/Qwen3-4B-tooluse/Qwen_Qwen3-4B-bf16.gguf $M/Qwen3-4B-Q4_K_M/Qwen3-4B-Q4_K_M.gguf "" ""
gold apertus-8b          $HF/apertus-8b          $M/gold-bf16/apertus-8b-bf16.gguf          $M/Apertus-8B-Instruct-2509-Q4_K_M/Apertus-8B-Instruct-2509-Q4_K_M.gguf "" ""
gold nemotron-nano-9b-v2 $HF/nemotron-nano-9b-v2 $M/nemotron-nano-9b-v2/nvidia_NVIDIA-Nemotron-Nano-9B-v2-bf16.gguf $M/nemotron-nano-9b-v2/NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf "" ""
echo GOLD2K-DONE
