#!/bin/sh
# One bench run as its own foreground job, as step 2's bench2.sh, with a fixed
# seed and the hard page reads logged beside it.
# Usage: one.sh TAG SEED HOTFILE|auto [TRACEFILE]
TAG=$1 SEED=$2 HOT=$3 TR=$4
R=${RUNNER_BIN}
M=$R330_MODEL
P="Write four paragraphs about how reservoirs and water towers keep a city's water pressure steady through the day."
O=$R330_DIR/out; mkdir -p $O
rm -f $O/$TAG.stop
python $R330_DIR/pagein.py $O/$TAG.pagein.json $O/$TAG.stop &
if [ -n "$TR" ]; then rm -f "$TR"; export RUNNER_MOE_TRACE="$TR"; fi
if [ "$HOT" = auto ]; then
  "$R" -m "$M" --gpu 99 --cpu-moe auto -c 4096 --bench-json -p "$P" -n 128 -s $SEED --no-tray > $O/$TAG.json 2> $O/$TAG.err
else
  RUNNER_MOE_HOT_PROF=1 RUNNER_MOE_HOT="$HOT" "$R" -m "$M" --gpu 99 --cpu-moe 99 -c 4096 --bench-json -p "$P" -n 128 -s $SEED --no-tray > $O/$TAG.json 2> $O/$TAG.err
fi
rc=$?
touch $O/$TAG.stop; wait
python - "$O/$TAG" "$rc" <<'PY'
import json,re,sys
b=sys.argv[1]
try: d=json.load(open(b+'.json'))
except Exception as e: print(b,'rc',sys.argv[2],'no json',e); raise SystemExit
t=open(b+'.err',errors='replace').read()
m=re.search(r'all layers.*hit ([\d.]+)',t); w=re.search(r'only ([\d.]+) GB of RAM',t)
p=json.load(open(b+'.pagein.json'))
print(b.rsplit('/',1)[1],'tok/s',d.get('gen_tok_s'),'gen_s',d.get('gen_s'),'prompt_s',d.get('prompt_s'),'hit',m and m[1],'ramavail',w and w[1],'pagein_total',sum(p),'tail10s',p[-10:])
PY
