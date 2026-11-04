#!/bin/sh
# R3.30 step 1: one bench config a call, so each run is its own foreground
# job (bench.sh's loop was stopped by the host for low memory, 2026-11-04).
# Usage: bench1.sh RUNNER MODEL OUTDIR ROUND MODE
RUNNER=$1 M=$2 O=$3 R=$4 MODE=$5
mkdir -p "$O"
P="Write four paragraphs about how reservoirs and water towers keep a city's water pressure steady through the day."
"$RUNNER" -m "$M" --gpu 99 --cpu-moe $MODE -c 4096 --bench-json -p "$P" -n 128 --no-tray \
  > "$O/r$R-$MODE.json" 2> "$O/r$R-$MODE.err"
python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
r = d.get('route') or {}
print(sys.argv[2], sys.argv[3], d.get('gen_tok_s'), d.get('prompt_tok_s'),
      {k: r.get(k) for k in ('gpu_layers', 'experts_on_host', 'moe_layers')})
" "$O/r$R-$MODE.json" $R $MODE
