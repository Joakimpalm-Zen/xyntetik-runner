#!/bin/bash
# A two-file, two-engine baseline for a derived model artifact: the original
# file and the derived one, each through Runner and through llama.cpp, with
# four outcomes kept apart: it loads, the tool protocol behaves, a fixed task
# set scores, and the numbers stay close to the parent.
#
#   scripts/artifact-baseline.sh OUT_DIR
#
# It is a driver, not a framework: every measurement is an existing harness
# (tool-protocol-check, agent-torture, eval-tooluse-shifted, kld-compare-raw,
# token_divergence) run with its own record under OUT_DIR, and a step that
# fails is recorded and the run goes on. Nothing here decides whether the
# derived file is better; it produces the matched records a person compares.
#
# Configuration is environment, or VAR=value lines in the file BL_ENV names
# (default: artifact-baseline.env beside this script):
#   BL_ORIGINAL     the source GGUF                         (required)
#   BL_DERIVED      the derived GGUF                        (required)
#   BL_PARENT       a higher-precision GGUF of the same model, the
#                   reference of the fidelity rows          (optional)
#   BL_LLAMA_SERVER llama.cpp's llama-server                (optional)
#   BL_LLAMA_VERSION the llama.cpp revision, for the records
#   BL_RUNNER       runner binary        (default ./runner)
#   BL_PYTHON       python               (default python3)
#   BL_GPU          auto | off for the Runner arm (default auto)
#   BL_THREADS      CPU threads for the GPU arm's server (default 4)
#   BL_CPU_THREADS  threads for each CPU server (default 8)
#   BL_CTX          context for the served arms (default 8192)
#   BL_CORPUS       text for the fidelity rows (default tests/fixtures/mixed-corpus.txt)
#   BL_POSITIONS    scored positions per fidelity row (default 300)
#   BL_PIN          a command prefix for every process, e.g. "taskset -c 0-23"
#   BL_STEP_TIMEOUT seconds any one step may take (default 2400)
#   BL_SKIP         comma list of steps to skip: protocol,torture,task,
#                   fidelity,divergence,llama-protocol
#   BL_LLAMA_TORTURE  set to also run the 120-request agent-torture matrix
#                   against llama.cpp (hours on a CPU build)
set -u
BL_ENV=${BL_ENV:-$(dirname "$0")/artifact-baseline.env}
# shellcheck disable=SC1090
[ -f "$BL_ENV" ] && { set -a; . "$BL_ENV"; set +a; }
OUT=${1:?usage: artifact-baseline.sh OUT_DIR}
: "${BL_ORIGINAL:?set BL_ORIGINAL}" "${BL_DERIVED:?set BL_DERIVED}"
RUNNER=${BL_RUNNER:-./runner}
PY=${BL_PYTHON:-python3}
GPU=${BL_GPU:-auto}
THREADS=${BL_THREADS:-4}
CPU_THREADS=${BL_CPU_THREADS:-8}
CTX=${BL_CTX:-8192}
CORPUS=${BL_CORPUS:-tests/fixtures/mixed-corpus.txt}
POS=${BL_POSITIONS:-300}
PIN=${BL_PIN:-}
TMO=${BL_STEP_TIMEOUT:-2400}
SKIP=",${BL_SKIP:-},"
mkdir -p "$OUT" || exit 2
STEPLOG=$OUT/steps.tsv
: > "$STEPLOG"

note() { echo "[baseline $(date -u +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
skipped() { case "$SKIP" in *",$1,"*) return 0 ;; esac; return 1; }
step() {  # label logfile command...
  local label=$1 log=$2; shift 2
  local t0=$SECONDS rc
  note "start $label"
  # shellcheck disable=SC2086
  $PIN timeout "$TMO" "$@" > "$log" 2>&1; rc=$?
  printf '%s\t%s\t%s\n' "$label" "$rc" "$((SECONDS - t0))" >> "$STEPLOG"
  note "end   $label rc=$rc $((SECONDS - t0))s"
  return $rc
}
wait_port() {  # port seconds
  local i=0
  until "$PY" -c "import socket,sys; s=socket.create_connection(('127.0.0.1',$1),1)" 2>/dev/null; do
    i=$((i + 1)); [ "$i" -ge "$2" ] && return 1; sleep 1
  done
}
port_busy() {
  "$PY" -c "import socket,sys; s=socket.create_connection(('127.0.0.1',$1),1)" 2>/dev/null
}
serve() {  # name port gpu threads model -> pid in SRV_PID
  local name=$1 port=$2 gpu=$3 threads=$4 model=$5
  SRV_PID=
  # Something already listening here would be scored in this model's name.
  if port_busy "$port"; then
    note "$name: port $port is already in use; not started, its steps are skipped"
    printf '%s\t%s\t%s\n' "$name/port-$port-busy" 98 0 >> "$STEPLOG"
    return 1
  fi
  # shellcheck disable=SC2086
  $PIN "$RUNNER" --serve --no-tray -m "$model" -c "$CTX" --port "$port" \
      --gpu "$gpu" -t "$threads" > "$OUT/$name.server.log" 2>&1 &
  SRV_PID=$!
  if ! wait_port "$port" 900; then
    note "$name: server did not come up (see $OUT/$name.server.log)"
    kill "$SRV_PID" 2>/dev/null; SRV_PID=
    return 1
  fi
  # the listener has to be the process started here, not one that won the port
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    note "$name: the server started here exited, yet port $port answers; skipped"
    SRV_PID=
    return 1
  fi
}
stop() { [ -n "${1:-}" ] && kill "$1" 2>/dev/null && wait "$1" 2>/dev/null; }

note "runner $("$RUNNER" --version 2>&1 | head -1)"
"$RUNNER" --caps > "$OUT/caps.json" 2>/dev/null

# ---- identity: what the two files are, and that both load
for tag in original derived; do
  f=$BL_ORIGINAL; [ $tag = derived ] && f=$BL_DERIVED
  { sha256sum "$f" 2>/dev/null || shasum -a 256 "$f"; } > "$OUT/$tag.sha256"
  step "$tag/fit" "$OUT/$tag.fit.log" "$RUNNER" --fit "$f"
  step "$tag/tool-info" "$OUT/$tag.tool-info.json" "$RUNNER" -m "$f" --tool-info --gpu off
done

# ---- the fidelity reference, on the CPU, shared by both files
PARENT_PID=
if [ -n "${BL_PARENT:-}" ] && ! skipped fidelity; then
  serve parent 18311 off "$CPU_THREADS" "$BL_PARENT" && PARENT_PID=$SRV_PID
fi

# ---- the llama.cpp arm runs beside the Runner arm, on its own threads
llama_arm() {
  for tag in original derived; do
    f=$BL_ORIGINAL; [ $tag = derived ] && f=$BL_DERIVED
    if ! skipped divergence && { port_busy 18201 || port_busy 18202; }; then
      note "$tag: token_divergence's ports 18201/18202 are in use; step skipped"
    elif ! skipped divergence; then
      step "$tag/divergence-vs-llama.cpp" "$OUT/$tag.divergence.log" \
        "$PY" scripts/token_divergence.py --model "$f" \
          --reference "$BL_LLAMA_SERVER" --runner "$RUNNER" \
          --threads "$CPU_THREADS" --ctx "$CTX" --tokens 64
    fi
    if ! skipped llama-protocol; then
      if port_busy 18321; then
        note "$tag: port 18321 is already in use; llama.cpp protocol step skipped"
        continue
      fi
      # shellcheck disable=SC2086
      $PIN "$BL_LLAMA_SERVER" -m "$f" -c "$CTX" --port 18321 --host 127.0.0.1 \
          -t "$CPU_THREADS" --jinja > "$OUT/$tag.llama-server.log" 2>&1 &
      lp=$!
      if wait_port 18321 900 && kill -0 "$lp" 2>/dev/null; then
        step "$tag/tool-protocol-llama.cpp" "$OUT/$tag.tool-protocol-llama.log" \
          "$PY" scripts/tool-protocol-check.py --base-url http://127.0.0.1:18321 \
            --model-file "$f" --label "$tag llama.cpp ${BL_LLAMA_VERSION:-}" \
            --out "$OUT/$tag.tool-protocol-llama.json"
        # the full matrix is 120 requests: minutes on a GPU, hours on a CPU
        if [ -n "${BL_LLAMA_TORTURE:-}" ]; then
          step "$tag/agent-torture-llama.cpp" "$OUT/$tag.torture-llama.log" \
            "$PY" scripts/agent-torture.py --endpoint 127.0.0.1:18321 \
              --runtime llama.cpp --runtime-version "${BL_LLAMA_VERSION:-unknown}" \
              --model "$f" --out "$OUT/$tag.torture-llama"
        fi
      else
        note "$tag: llama-server did not come up"
      fi
      stop "$lp"
    fi
  done
}
LLAMA_ARM=
if [ -n "${BL_LLAMA_SERVER:-}" ]; then llama_arm & LLAMA_ARM=$!; fi

# ---- the Runner arm: one server per file, every served check against it
for tag in original derived; do
  f=$BL_ORIGINAL; [ $tag = derived ] && f=$BL_DERIVED
  if serve "$tag" 18301 "$GPU" "$THREADS" "$f"; then
    pid=$SRV_PID
    skipped protocol || step "$tag/tool-protocol" "$OUT/$tag.tool-protocol.log" \
      "$PY" scripts/tool-protocol-check.py --base-url http://127.0.0.1:18301 \
        --model-file "$f" --runner "$RUNNER" --label "$tag" \
        --out "$OUT/$tag.tool-protocol.json"
    skipped torture || step "$tag/agent-torture" "$OUT/$tag.torture.log" \
      "$PY" scripts/agent-torture.py --endpoint 127.0.0.1:18301 \
        --runtime runner --model "$f" --out "$OUT/$tag.torture"
    if [ -n "$PARENT_PID" ]; then
      step "$tag/fidelity-vs-parent" "$OUT/$tag.fidelity.log" \
        "$PY" scripts/kld-compare-raw.py \
          --endpoint-a http://127.0.0.1:18301 --model-name-a "$(basename "$f")" \
          --endpoint-b http://127.0.0.1:18311 --model-name-b "$(basename "$BL_PARENT")" \
          --corpus "$CORPUS" --max-positions "$POS" \
          --out "$OUT/$tag.fidelity.json"
    fi
    stop "$pid"
  fi
  skipped task || step "$tag/task-set" "$OUT/$tag.task.log" \
    "$PY" scripts/eval-tooluse-shifted.py --runner "$RUNNER" --model "$f" \
      --gpu "$GPU" --threads "$THREADS" --set v2 --legs raw,native \
      --out "$OUT/$tag.task.json"
done

stop "$PARENT_PID"
[ -n "$LLAMA_ARM" ] && wait "$LLAMA_ARM"
note "done; step results in $STEPLOG"
awk -F'\t' '$2 != 0 { bad = 1 } END { exit bad }' "$STEPLOG"
