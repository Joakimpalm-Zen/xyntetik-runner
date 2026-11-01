#!/bin/bash
# The scheduled CUDA gate (R6.9): the device checks hosted CI cannot run, as
# one script a CUDA host runs from a fresh build of main.
#
#   scripts/cuda-gate.sh BUILD_DIR [OUT_DIR]
#
# BUILD_DIR is a checkout whose `runner` is already built with CUDA. Exit 0
# when every step passed, 1 when a step failed, 2 when the gate could not run
# (no CUDA device, no model, no python, no llama.cpp directory): a gate that
# cannot run is not a pass.
#
# Steps, each with its own log under OUT_DIR/cuda-gate-<stamp>/:
#   caps         --caps reports a CUDA device
#   tc-overflow  make test-tc-overflow (fp16 range of the tensor-core operands)
#   iquants      make test-cuda-iquants (device codebook decoders vs the CPU)
#   cuda-gated   make test-cuda-gated (the pytest cases that skip without CUDA,
#                none of them allowed to skip here for want of it)
#   kv-fp4       make test-cuda-kv-fp4 (the device fp4 KV store vs the CPU codec)
#   nvfp4        make test-cuda-nvfp4 (NVFP4 kernels and their scale companion)
#   qwen4exp     make test-cuda-qwen4exp (the family's placements, the shared
#                split GGUF, the MTP head, the BF16 split-K decode route)
#   q3k          make test-cuda-q3k (the Q3_K decode GEMV on a Q3_K fixture)
#   tc-kq        make test-cuda-tc-kq (the tensor-core prefill of Q5_K, Q3_K,
#                Q2_K, Q4_1, Q5_0 and Q5_1, teacher-forced tolerance)
#   moe-legacy   make test-cuda-moe-legacy (MoE experts in Q5_0, Q4_1 and
#                Q5_1 on the fused device routing, against the eager path)
#   varff        make test-cuda-varff (per-layer FFN widths, the gemma-4 E2B shape)
#   moe-twin     make test-cuda-moe-twin (by-expert MoE prefill == the fused path)
#   keepn        make test-cuda-keepn (pruned MoE: per-layer expert counts)
#   moe-replay   make test-cuda-moe-replay (expert trace and replay on the device routers)
#   batch-id     make test-cuda-batch-identity (the decode microbatch == sequential
#                decode, byte for byte, on the F32, Q8_0 and BF16 fixtures)
#   spec-id      make test-cuda-spec-identity (a speculative verify round's rows
#                == solo decode steps, byte for byte, dense, MoE and hybrid)
#   cpu-cuda     scripts/cpu_cuda_check.py on one small model (byte identity)
# Every make target the steps name is in DEVICE_STEPS below, and every
# test-cuda-* target in the Makefile has to be there (tests/test_ci_contracts.py):
# a device target no step runs is a gate nothing runs. A step whose log says
# it skipped itself for want of a device or driver fails, since it did not run.
# and a summary cuda-gate-<stamp>.json plus, on a pass, the device-evidence
# row written by scripts/device-evidence.py into a COPY of the ledger, so the
# checkout is never modified. A person or a pull request carries the row into
# docs/device-evidence.json.
#
# Everything a box differs in is an environment variable:
#   GATE_CLASS       device-evidence class id (default by uname)
#   GATE_PYTHON      python with pytest (default python3)
#   GATE_MODEL       GGUF for the cpu-cuda step
#   GATE_MAKE_ARGS   extra make arguments, e.g. CC=... or -j4
#   RUNNER_IQ_FIXTURES / RUNNER_LLAMA_CPP_BIN   as tests/test_iquants.py reads
#                    (moe-legacy needs llama-quantize there, so the gate does
#                    not run without RUNNER_LLAMA_CPP_BIN)
#   GATE_LOCK        lock file to hold while running (optional)
#   GATE_SKIP_IF     shell command; when it exits 0 the gate SKIPS, logged
#   GATE_ALLOW_DEGRADED_ENV=1   run in a degraded launch environment (below)
#                    with a warning instead of refusing
# A box keeps its values in a file beside its copy of this script
# (cuda-gate.env, plain VAR=value lines with LF line endings) or in the file
# GATE_ENV names. A gate that cannot run for want of a setting says which env
# file it looked for, whether it loaded it, the settings it ended up with and
# the shell running it: a variable exported in another shell does not reach
# this one (on the Windows CUDA host, a Git Bash export never reached the
# MSYS2 bash the gate ran under), and the env file is the one place that does.
#
# Before its first step the gate also asks its own python (GATE_PYTHON) for
# its temp directory and the user, and finishes not-run when the launch
# environment is degraded: a temp directory under the Windows directory, or
# on POSIX one that is / or not writable, or a TMPDIR/TEMP/TMP python could
# not use; no user name at all, an empty USERNAME (Windows) or USER (POSIX)
# and a getuser() that raises (an empty variable alone is a warning: a Linux
# container without USER still has a user python resolves); under MSYS2, no
# MSYSTEM, or no HOME outside a login shell. On the
# Windows CUDA host a tool shell that started MSYS2 bash without USERNAME,
# TEMP or TMP sent every fixture through C:\Windows\Temp (python's last
# resort) and the gate ran about 27 times slower, with subprocess timeouts
# that were not hangs. The summary JSON records the temp directory, whether a
# user resolved (never the name: the file is committed as device evidence)
# and the shell the run had.
set -u
GATE_ENV=${GATE_ENV:-$(dirname "$0")/cuda-gate.env}
GATE_ENV_STATE="not found"
if [ -f "$GATE_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  if . "$GATE_ENV"; then GATE_ENV_STATE="found and loaded"
  else GATE_ENV_STATE="found, but sourcing it failed"; fi
  set +a
fi
BUILD=${1:?usage: cuda-gate.sh BUILD_DIR [OUT_DIR]}
OUT=${2:-${GATE_OUT:-$BUILD/cuda-gate-out}}
cd "$BUILD" || exit 2
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
DIR=$OUT/cuda-gate-$STAMP
mkdir -p "$DIR" || exit 2
SUMMARY=$OUT/cuda-gate-$STAMP.json

UNAME=$(uname -s)
case "$UNAME" in
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
json_str() { printf '"%s"' "$(printf '%s' "$1" | tr -d '\000-\037' | sed 's/[\\"]/\\&/g')"; }
ENV_TMP="" ENV_USER="" ENV_DEGRADED="" ENV_WARNED="" ENV_LOGIN=false
shopt -q login_shell && ENV_LOGIN=true
finish() {  # result exit_code
  # the summary is committed as device evidence, so it says whether a user
  # resolved, never who: the name stays in the console notes
  local user_ok=false; [ -n "$ENV_USER" ] && user_ok=true
  printf '{"schema":"xyntetik.runner.cuda-gate.v2","stamp":"%s","class":"%s","commit":"%s","result":"%s","env":{"tempdir":%s,"user_resolved":%s,"shell":%s,"login_shell":%s,"msystem":%s,"degraded":[%s],"warned":[%s]},"steps":[%s]}\n' \
    "$STAMP" "$CLASS" "$(git rev-parse HEAD 2>/dev/null || echo unknown)" "$1" \
    "$(json_str "$ENV_TMP")" "$user_ok" "$(json_str "${BASH:-}")" "$ENV_LOGIN" \
    "$(json_str "${MSYSTEM:-}")" "$ENV_DEGRADED" "$ENV_WARNED" "$STEPS" > "$SUMMARY"
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

# Everything the gate needs before a step runs, every gap reported at once,
# then where the settings came from.
SETTINGS="GATE_MODEL GATE_PYTHON RUNNER_LLAMA_CPP_BIN RUNNER_IQ_FIXTURES"
env_report() {
  local v val list="" crs="" root=""
  note "env file: $GATE_ENV ($GATE_ENV_STATE)"
  for v in $SETTINGS; do
    val=${!v:-}
    case $val in
      "") list="$list, $v unset"; [ "$v" = GATE_PYTHON ] && list="$list ($PY assumed)" ;;
      *$'\r') list="$list, $v=${val%$'\r'}\\r"; crs="$crs $v" ;;
      *) list="$list, $v=$val" ;;
    esac
  done
  note "settings:${list#,}"
  [ -z "$crs" ] || note "${crs# } end in a carriage return: an env file with CRLF line endings ends every value with one; save it with LF"
  command -v cygpath > /dev/null 2>&1 && root=", root $(cygpath -m / 2> /dev/null)"
  note "shell: ${BASH:-unknown} (bash ${BASH_VERSION:-?}${MSYSTEM:+, MSYSTEM=$MSYSTEM}$root)"
  note "a variable exported in another shell does not reach this one: set it in $GATE_ENV"
}
PROBLEMS=0
problem() { note "not run: $*"; PROBLEMS=$((PROBLEMS + 1)); }

# The launch environment, as the gate's own python sees it (the header says
# why). Each failed check is a problem, or with GATE_ALLOW_DEGRADED_ENV=1 a
# warning, and the summary JSON lists it under "degraded"; a check that only
# warns is listed under "warned".
listed() { case ",$1," in *",\"$2\","*) echo "$1" ;; *) echo "$1${1:+,}\"$2\"" ;; esac; }
degraded() {  # check detail
  ENV_DEGRADED=$(listed "$ENV_DEGRADED" "$1")
  if [ "${GATE_ALLOW_DEGRADED_ENV:-}" = 1 ]; then note "warning: $1: $2 (GATE_ALLOW_DEGRADED_ENV=1: going on)"
  else problem "$1: $2"; fi
}
warned() { ENV_WARNED=$(listed "$ENV_WARNED" "$1"); note "warning: $1: $2"; }
# Six lines, from python's stdin (no multi-line argument for MSYS2 to quote
# for a Windows python): the temp directory, the TMPDIR/TEMP/TMP (the first
# set, in python's order) it did not use, whether it is writable, the user,
# why getuser() raised, and SYSTEMROOT as a child process gets it.
ENV_PROBE='import getpass, os, sys, tempfile
try: t = tempfile.gettempdir()
except Exception: t = ""
named = [(k, os.environ[k]) for k in ("TMPDIR", "TEMP", "TMP") if os.environ.get(k)][:1]
unused = ""
if named and os.path.normcase(os.path.abspath(named[0][1])) != os.path.normcase(t):
    unused = "%s=%s" % named[0]
try: u, why = getpass.getuser(), ""
except Exception as e: u, why = "", "%s: %s" % (type(e).__name__, e)
root = os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR") or ""
w = "1" if t and os.access(t, os.W_OK) else "0"
lines = (str(x).replace("\n", " ") for x in (t, unused, w, u, why, root))
sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8", "replace"))
'
path_key() { printf '%s' "$1" | tr '\\' / | tr '[:upper:]' '[:lower:]' | sed 's:/*$::'; }
env_check() {
  local probe unused writable why root win="" uvar=USER uval login
  case $UNAME in MINGW*|MSYS*|CYGWIN*) win=1; uvar=USERNAME ;; esac
  if ! probe=$(printf '%s\n' "$ENV_PROBE" | "$PY" - 2> /dev/null); then
    problem "$PY could not report its temp directory and user"; return
  fi
  { IFS= read -r ENV_TMP; IFS= read -r unused; IFS= read -r writable
    IFS= read -r ENV_USER; IFS= read -r why; IFS= read -r root; } <<< "$probe"
  if [ -z "$ENV_TMP" ]; then degraded "temp directory" "python finds no usable one"
  elif [ -n "$win" ]; then
    # compared without regard to case or slash direction: C:\WINDOWS holds
    # C:\Windows\Temp
    [ -z "$root" ] || case "$(path_key "$ENV_TMP")/" in
      "$(path_key "$root")"/*) degraded "temp directory" "$ENV_TMP is under the Windows directory $root" ;;
    esac
  elif [ "$ENV_TMP" = / ]; then degraded "temp directory" "it is /"
  elif [ "$writable" != 1 ]; then degraded "temp directory" "$ENV_TMP is not writable"
  fi
  [ -z "$unused" ] || degraded "temp directory" "$unused is missing or not writable, so python uses $ENV_TMP"
  uval=${!uvar:-}
  if [ -z "$uval" ] && [ -n "$why" ]; then degraded "user name" "$uvar is empty and python's getpass.getuser() raises ($why)"
  elif [ -z "$uval" ]; then warned "user name" "$uvar is empty, python's getpass.getuser() resolves $ENV_USER"
  elif [ -n "$why" ]; then warned "user name" "python's getpass.getuser() raises ($why)"
  fi
  case $UNAME in MINGW*|MSYS*)
    [ -n "${MSYSTEM:-}" ] || degraded "MSYS2 shell" "MSYSTEM is not set"
    [ "$ENV_LOGIN" = true ] || [ -n "${HOME:-}" ] || degraded "MSYS2 shell" "not a login shell and HOME is not set" ;;
  esac
  [ -n "$ENV_DEGRADED$ENV_WARNED" ] || return 0
  login="not a login shell"; [ "$ENV_LOGIN" = true ] && login="login shell"
  note "launch environment: temp directory ${ENV_TMP:-none}, user ${ENV_USER:-none} ($uvar ${uval:-empty}), shell ${BASH:-unknown} ($login), MSYSTEM ${MSYSTEM:-unset}"
  if [ -n "$win" ]; then
    note "remedy: launch from a login shell (PowerShell: C:\\msys64\\usr\\bin\\bash.exe -lc '...') or set TEMP/TMP and USERNAME"
  else
    note "remedy: launch from a login shell (bash -lc '...') or set TMPDIR and USER"
  fi
}

[ -x "$EXE" ] || problem "no $EXE in $BUILD"
if ! command -v "$PY" > /dev/null 2>&1; then problem "no $PY on PATH (GATE_PYTHON)"
else
  "$PY" -c 'import pytest' 2> /dev/null || problem "$PY has no pytest (GATE_PYTHON)"
  env_check
fi
if [ -z "$MODEL" ]; then problem "GATE_MODEL is not set: the cpu-cuda step needs a GGUF"
elif [ ! -f "$MODEL" ]; then problem "GATE_MODEL $MODEL does not exist"; fi
if [ -z "${RUNNER_LLAMA_CPP_BIN:-}" ]; then
  problem "RUNNER_LLAMA_CPP_BIN is not set: moe-legacy quantizes its fixtures with the llama-quantize there"
elif [ ! -d "$RUNNER_LLAMA_CPP_BIN" ]; then
  problem "RUNNER_LLAMA_CPP_BIN $RUNNER_LLAMA_CPP_BIN is not a directory"
fi
[ "$PROBLEMS" = 0 ] || { env_report; finish not-run 2; }

"$EXE" --caps > "$DIR/caps.json" 2> "$DIR/caps.err"
if ! "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'cuda' else 1)" "$DIR/caps.json"; then
  note "caps reports no CUDA device"; add_step caps fail 0; finish not-run 2
fi
add_step caps pass 0

FAIL=0
DEVICE_STEPS="tc-overflow:test-tc-overflow iquants:test-cuda-iquants cuda-gated:test-cuda-gated kv-fp4:test-cuda-kv-fp4 nvfp4:test-cuda-nvfp4 qwen4exp:test-cuda-qwen4exp q3k:test-cuda-q3k tc-kq:test-cuda-tc-kq moe-legacy:test-cuda-moe-legacy varff:test-cuda-varff tc2-bytes:test-cuda-tc2-bytes moe-twin:test-cuda-moe-twin keepn:test-cuda-keepn moe-replay:test-cuda-moe-replay batch-id:test-cuda-batch-identity spec-id:test-cuda-spec-identity"
for pair in $DEVICE_STEPS; do
  name=${pair%%:*}; target=${pair#*:}
  # shellcheck disable=SC2086
  run_step "$name" make $MAKE_ARGS PYTHON="$PY" "$target" || FAIL=1
  if grep -qE "skipped[ :(]+(no CUDA|driver API|cuInit)" "$DIR/$name.log" 2>/dev/null; then
    note "$name skipped itself: the device was not used"; FAIL=1
  fi
done
run_step cpu-cuda "$PY" scripts/cpu_cuda_check.py "$MODEL" --runner "$EXE" --tokens 64 || FAIL=1

[ "$FAIL" = 0 ] || finish fail 1

cp docs/device-evidence.json "$DIR/device-evidence.json"
if "$PY" scripts/device-evidence.py --json "$DIR/device-evidence.json" \
     --record "$CLASS" --result pass --caps "$DIR/caps.json" \
     --ran "scripts/cuda-gate.sh: $(for pair in $DEVICE_STEPS; do printf '%s, ' "${pair#*:}"; done)cpu_cuda_check ($(basename "$MODEL"), 64 tokens)" \
     > "$DIR/record.log" 2>&1; then
  note "device-evidence row written to $DIR/device-evidence.json"
else
  note "the gates passed but the row could not be recorded (see $DIR/record.log)"
  finish fail 1
fi
finish pass 0
