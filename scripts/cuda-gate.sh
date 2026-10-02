#!/bin/bash
# The scheduled CUDA gate (R6.9): the device checks hosted CI cannot run, as
# one script a lab box runs from a fresh build of main.
#
#   scripts/cuda-gate.sh BUILD_DIR [OUT_DIR]
#
# BUILD_DIR is a checkout whose `runner` is already built with CUDA. Exit 0
# when every step passed, 1 when a step failed, 2 when the gate could not run
# (no CUDA device, no model, no python): a gate that cannot run is not a pass.
#
# Steps, each with its own log under OUT_DIR/cuda-gate-<stamp>/:
#   caps         --caps reports a CUDA device
#   tc-overflow  make test-tc-overflow (fp16 range of the tensor-core operands)
#   iquants      make test-cuda-iquants (device codebook decoders vs the CPU)
#   cpu-cuda     scripts/cpu_cuda_check.py on one small model (byte identity)
# and a summary cuda-gate-<stamp>.json plus, on a pass, the device-evidence
# row written by scripts/device-evidence.py into a COPY of the ledger, so the
# checkout is never modified. A person or a pull request carries the row into
# docs/device-evidence.json.
#
# Everything a box differs in is an environment variable:
#   GATE_CLASS       device-evidence class id (default by uname)
#   GATE_PYTHON      python with pytest (default python3)
#   GATE_MODEL       GGUF for the cpu-cuda step
#   GATE_MAKE_ARGS   extra make arguments, e.g. CC=... or OS=Windows_NT
#   RUNNER_IQ_FIXTURES / RUNNER_LLAMA_CPP_BIN   as tests/test_iquants.py reads
#   GATE_LOCK        lock file to hold while running (optional)
#   GATE_SKIP_IF     shell command; when it exits 0 the gate SKIPS, logged
# A box keeps its values in a file beside its copy of this script
# (cuda-gate.env, plain VAR=value lines) or in the file GATE_ENV names.
set -u
GATE_ENV=${GATE_ENV:-$(dirname "$0")/cuda-gate.env}
# shellcheck disable=SC1090
[ -f "$GATE_ENV" ] && { set -a; . "$GATE_ENV"; set +a; }
BUILD=${1:?usage: cuda-gate.sh BUILD_DIR [OUT_DIR]}
OUT=${2:-${GATE_OUT:-$BUILD/cuda-gate-out}}
cd "$BUILD" || exit 2
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
DIR=$OUT/cuda-gate-$STAMP
mkdir -p "$DIR" || exit 2
SUMMARY=$OUT/cuda-gate-$STAMP.json

case "$(uname -s)" in
  Linux)            DEF_CLASS=linux-x86_64-cuda; EXE=./runner ;;
  MINGW*|MSYS*|CYGWIN*) DEF_CLASS=windows-x86_64-cuda; EXE=./runner.exe ;;
  *)                DEF_CLASS=none; EXE=./runner ;;
esac
CLASS=${GATE_CLASS:-$DEF_CLASS}
PY=${GATE_PYTHON:-python3}
MAKE_ARGS=${GATE_MAKE_ARGS:-}

MODEL=${GATE_MODEL:-}

note() { echo "[cuda-gate $(date -u +%H:%M:%S)] $*"; }
STEPS=""
add_step() { STEPS="$STEPS${STEPS:+,}{\"step\":\"$1\",\"result\":\"$2\",\"seconds\":$3}"; }
finish() {  # result exit_code
  printf '{"schema":"xyntetik.runner.cuda-gate.v1","stamp":"%s","class":"%s","commit":"%s","result":"%s","steps":[%s]}\n' \
    "$STAMP" "$CLASS" "$(git rev-parse HEAD 2>/dev/null || echo unknown)" "$1" "$STEPS" > "$SUMMARY"
  note "result: $1 ($SUMMARY)"
  [ -n "${GATE_LOCK:-}" ] && [ -n "${LOCKED:-}" ] && rm -f "$GATE_LOCK"
  exit "$2"
}

if [ -n "${GATE_SKIP_IF:-}" ] && bash -c "$GATE_SKIP_IF" > "$DIR/skip-check.log" 2>&1; then
  note "skipped: GATE_SKIP_IF held (see $DIR/skip-check.log)"
  finish skipped 0
fi
if [ -n "${GATE_LOCK:-}" ]; then
  if [ -e "$GATE_LOCK" ]; then note "skipped: $GATE_LOCK exists"; finish skipped 0; fi
  echo $$ > "$GATE_LOCK" && LOCKED=1
fi

run_step() {  # name command...
  local name=$1; shift
  local t0=$SECONDS
  note "$name: $*"
  if "$@" > "$DIR/$name.log" 2>&1; then
    add_step "$name" pass $((SECONDS - t0)); note "$name: pass"
    return 0
  fi
  add_step "$name" fail $((SECONDS - t0)); note "$name: FAIL (tail of $DIR/$name.log)"
  tail -n 25 "$DIR/$name.log"
  return 1
}

[ -x "$EXE" ] || { note "no $EXE in $BUILD"; finish not-run 2; }
"$PY" -c 'import pytest' 2>/dev/null || { note "$PY has no pytest"; finish not-run 2; }
[ -n "$MODEL" ] && [ -f "$MODEL" ] || { note "no model for the cpu-cuda step (set GATE_MODEL)"; finish not-run 2; }

"$EXE" --caps > "$DIR/caps.json" 2> "$DIR/caps.err"
if ! "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'cuda' else 1)" "$DIR/caps.json"; then
  note "caps reports no CUDA device"; add_step caps fail 0; finish not-run 2
fi
add_step caps pass 0

FAIL=0
# shellcheck disable=SC2086
run_step tc-overflow make $MAKE_ARGS PYTHON="$PY" test-tc-overflow || FAIL=1
# shellcheck disable=SC2086
run_step iquants make $MAKE_ARGS PYTHON="$PY" test-cuda-iquants || FAIL=1
run_step cpu-cuda "$PY" scripts/cpu_cuda_check.py "$MODEL" --runner "$EXE" --tokens 64 || FAIL=1

if grep -q "skipped (no CUDA device)" "$DIR/tc-overflow.log" 2>/dev/null; then
  note "tc-overflow skipped itself: the device was not used"; FAIL=1
fi

[ "$FAIL" = 0 ] || finish fail 1

cp docs/device-evidence.json "$DIR/device-evidence.json"
if "$PY" scripts/device-evidence.py --json "$DIR/device-evidence.json" \
     --record "$CLASS" --result pass --caps "$DIR/caps.json" \
     --ran "scripts/cuda-gate.sh: test-tc-overflow, test-cuda-iquants, cpu_cuda_check ($(basename "$MODEL"), 64 tokens)" \
     > "$DIR/record.log" 2>&1; then
  note "device-evidence row written to $DIR/device-evidence.json"
else
  note "the gates passed but the row could not be recorded (see $DIR/record.log)"
  finish fail 1
fi
finish pass 0
