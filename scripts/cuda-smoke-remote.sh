#!/bin/sh
# Drive scripts/cuda-smoke.py on a remote CUDA box and bring the evidence back.
#
# Written for ZEN-GAMING (Windows 11, RTX 3070), whose SSH session runs cmd.exe
# and DOES NOT FORWARD STDERR from the remote process. Silence there means
# nothing, so the remote run redirects everything into a file as the sole
# command of its ssh invocation (redirects do not survive `&` chaining) and the
# file is fetched afterwards. The remote exit code IS trustworthy and is what
# this script exits with. See docs/windows-remote-checks.md.
#
#   scripts/cuda-smoke-remote.sh \
#       --binary C:/Users/zen/cuda-smoke/runner.exe \
#       --model  C:/Users/zen/qwen3-0.6b-q8_0.gguf \
#       --expect-version 0.4.3 \
#       --report docs/compat-reports/cuda-smoke-0.4.3-2026-08-29-rtx3070.json
#
# Exit 0 means the gate passed on real hardware.
set -eu

HOST=${HOST:-zen@192.168.1.123}
REMOTE_DIR=${REMOTE_DIR:-C:/Users/zen/cuda-smoke}
BINARY=""
MODEL=""
EXPECT_VERSION=""
REPORT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --host) HOST=$2; shift 2 ;;
        --remote-dir) REMOTE_DIR=$2; shift 2 ;;
        --binary) BINARY=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        --expect-version) EXPECT_VERSION=$2; shift 2 ;;
        --report) REPORT=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ -n "$BINARY" ] || { echo "--binary is required" >&2; exit 2; }
[ -n "$MODEL" ] || { echo "--model is required" >&2; exit 2; }

HERE=$(dirname "$0")
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"

# cmd.exe's mkdir rejects forward slashes even though everything downstream
# (scp, python) accepts them, so the directory is created with backslashes.
REMOTE_DIR_WIN=$(printf '%s' "$REMOTE_DIR" | tr '/' '\\')
$SSH "$HOST" "mkdir $REMOTE_DIR_WIN" >/dev/null 2>&1 || true
scp -o BatchMode=yes -q "$HERE/cuda-smoke.py" "$HOST:$REMOTE_DIR/cuda-smoke.py"

VERSION_ARG=""
[ -n "$EXPECT_VERSION" ] && VERSION_ARG="--expect-version $EXPECT_VERSION"

# Sole command, everything redirected. Do not add `&` clauses here.
rc=0
$SSH "$HOST" "python $REMOTE_DIR/cuda-smoke.py --binary $BINARY --model $MODEL --out $REMOTE_DIR/report.json $VERSION_ARG > $REMOTE_DIR/log.txt 2>&1" || rc=$?

TMPLOG=$(mktemp)
trap 'rm -f "$TMPLOG"' EXIT
if scp -o BatchMode=yes -q "$HOST:$REMOTE_DIR/log.txt" "$TMPLOG"; then
    cat "$TMPLOG"
else
    echo "could not fetch the remote log; the run may not have started" >&2
fi

if [ -n "$REPORT" ]; then
    if scp -o BatchMode=yes -q "$HOST:$REMOTE_DIR/report.json" "$REPORT"; then
        echo "report: $REPORT"
    else
        echo "could not fetch the remote report" >&2
        [ "$rc" -eq 0 ] && rc=1
    fi
fi

exit "$rc"
