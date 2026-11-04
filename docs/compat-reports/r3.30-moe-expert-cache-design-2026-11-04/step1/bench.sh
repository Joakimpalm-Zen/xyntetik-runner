#!/bin/sh
# R3.30 step 1: today's placement on the card and the host-expert slope.
# Decode tok/s at --cpu-moe auto and at explicit host-bank counts, interleaved,
# three rounds, with each run's route record.
# Usage: bench.sh RUNNER MODEL OUTDIR
RUNNER=$1 M=$2 O=$3
mkdir -p "$O"
P="Write four paragraphs about how reservoirs and water towers keep a city's water pressure steady through the day."
for round in 1 2 3; do
  for mode in auto 28 24 20 16; do
    "$RUNNER" -m "$M" --gpu 99 --cpu-moe $mode -c 4096 --bench-json -p "$P" -n 128 --no-tray \
      > "$O/r$round-$mode.json" 2> "$O/r$round-$mode.err"
    python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
r = d.get('route') or {}
print(sys.argv[2], sys.argv[3], d.get('gen_tok_s'), d.get('prompt_tok_s'),
      {k: r.get(k) for k in ('gpu_layers', 'experts_on_host', 'moe_layers', 'fallback')})
" "$O/r$round-$mode.json" $round $mode
  done
done
