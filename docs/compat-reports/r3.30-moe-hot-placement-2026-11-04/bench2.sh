#!/bin/sh
# R3.30 step 2: one config a call. MODE auto = --cpu-moe auto; hot = --cpu-moe
# 99 (every bank on the host) with RUNNER_MOE_HOT=HOTFILE.
# Usage: bench2.sh RUNNER MODEL OUTDIR ROUND MODE [HOTFILE]
RUNNER=$1 M=$2 O=$3 R=$4 MODE=$5 HOT=$6
mkdir -p "$O"
P="Write four paragraphs about how reservoirs and water towers keep a city's water pressure steady through the day."
if [ "$MODE" = hot ]; then
  RUNNER_MOE_HOT_PROF=1 RUNNER_MOE_HOT="$HOT" "$RUNNER" -m "$M" --gpu 99 --cpu-moe 99 -c 4096 \
    --bench-json -p "$P" -n 128 --no-tray > "$O/r$R-$MODE.json" 2> "$O/r$R-$MODE.err"
else
  "$RUNNER" -m "$M" --gpu 99 --cpu-moe $MODE -c 4096 --bench-json -p "$P" -n 128 --no-tray \
    > "$O/r$R-$MODE.json" 2> "$O/r$R-$MODE.err"
fi
python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
r = d.get('route') or {}
print(sys.argv[2], sys.argv[3], d.get('gen_tok_s'), d.get('prompt_tok_s'),
      {k: r.get(k) for k in ('experts_on_host', 'moe_layers', 'moe_hot')})
" "$O/r$R-$MODE.json" $R $MODE
