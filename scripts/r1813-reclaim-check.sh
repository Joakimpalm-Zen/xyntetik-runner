#!/usr/bin/env bash
# R18.13 real-file Linux cgroup v2 memory.high reclaim check.
#
# Runs the pre-registered cell matrix on a real validated model (default:
# LiquidAI/LFM2.5-8B-A1B-GGUF Q4_K_M, 5.16 GB public MoE from
# docs/model-tiers.json, pinned by repo@revision + sha256):
#
#   Warm cells (page cache dropped first, then warmed INSIDE the MemoryHigh
#   cgroup; streaming pressure applied BETWEEN load and first prompt):
#     1. main               (expected RED: evicted, high prompt-1 majflt)
#     2. branch off (0)     (expected RED: live control)
#     3. branch cached      (expected GREEN: resident >= 0.990, low majflt)
#     4. branch full        (expected GREEN: resident >= 0.990, low majflt)
#   Repeated in both MGLRU and classic two-list LRU when sudo can toggle
#   /sys/kernel/mm/lru_gen/enabled.
#
#   Additional branch cells:
#     5. branch cached --parallel 2 (warm, MemoryHigh): confirms 1x VmPTE +
#        reclaim protection with one populated mapping
#     6. branch cached cold (drop_caches): reports TTFT and load read_bytes
#        (must NOT read the file through during load)
#     7. branch full cold   (drop_caches): reports TTFT and load read_bytes
#     8. branch cached root-owned read-only (chown root:root, chmod 0444,
#        non-root caller, drop_caches): mincore trap (kernel 134fca9063ad)
#        must skip cached pass, stay lazy, and NOT read the file through.
#
# Usage:
#   scripts/r1813-reclaim-check.sh \
#     --base-runner /path/to/main/runner \
#     --branch-runner /path/to/branch/runner \
#     [--model /path/to/model.gguf] \
#     [--out-dir /path/to/artifacts]

set -euo pipefail

# ---- Pre-registered BAR (written before any run) ---------------------------
BAR_GREEN_MIN_RESIDENT="0.990"     # warm green cells: resident after pressure >= 0.990
BAR_GREEN_MAX_MAJFLT="64"          # warm green cells: prompt-1 major faults <= 64
BAR_RED_MAX_RESIDENT="0.900"       # warm red cells: resident drops < 0.900 ...
BAR_RED_MIN_MAJFLT="256"           # ... OR prompt-1 major faults >= 256
BAR_COLD_CACHED_MAX_READ_FRAC="0.10"  # cached cold: load read_bytes < 10% of file
BAR_COLD_FULL_MIN_READ_FRAC="0.80"    # full cold:   load read_bytes >= 80% of file
BAR_TRAP_MAX_READ_FRAC="0.10"         # root-owned 0444 trap: load read_bytes < 10% of file

# Pinned validated public MoE from docs/model-tiers.json (5.16 GB)
DEFAULT_REPO="LiquidAI/LFM2.5-8B-A1B-GGUF"
DEFAULT_REV="49c14831707011e64d70b2ebd8462ba08d608434"
DEFAULT_FILE="LFM2.5-8B-A1B-Q4_K_M.gguf"
DEFAULT_SHA256="4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0"

BASE_RUNNER=""
BRANCH_RUNNER=""
MODEL=""
OUT_DIR=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --base-runner)   BASE_RUNNER="$2"; shift 2 ;;
        --branch-runner) BRANCH_RUNNER="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --out-dir)       OUT_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,36p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [ "$(uname -s)" != "Linux" ]; then
    echo "ERROR: r1813-reclaim-check.sh requires Linux with cgroup v2" >&2
    exit 2
fi
if [ -z "$BASE_RUNNER" ] || [ -z "$BRANCH_RUNNER" ]; then
    echo "usage: $0 --base-runner <main-runner> --branch-runner <branch-runner> [--model <gguf>] [--out-dir <dir>]" >&2
    exit 2
fi
BASE_RUNNER="$(readlink -f "$BASE_RUNNER")"
BRANCH_RUNNER="$(readlink -f "$BRANCH_RUNNER")"
[ -x "$BASE_RUNNER" ] || { echo "ERROR: base runner not executable: $BASE_RUNNER" >&2; exit 2; }
[ -x "$BRANCH_RUNNER" ] || { echo "ERROR: branch runner not executable: $BRANCH_RUNNER" >&2; exit 2; }

HEADROOM_MIB="${HEADROOM_MIB:-1280}"
PRESSURE_MIB="${PRESSURE_MIB:-3072}"
PRESSURE_PASSES="${PRESSURE_PASSES:-3}"
CTX="${CTX:-256}"
RUN_UID="$(id -u)"
RUN_GID="$(id -g)"

SCRATCH_BASE="${SCRATCH_DIR:-/var/tmp}"
[ -d "$SCRATCH_BASE" ] && [ -w "$SCRATCH_BASE" ] || SCRATCH_BASE="${TMPDIR:-/tmp}"
WORKDIR="$(mktemp -d "$SCRATCH_BASE/r1813-reclaim.XXXXXX")"
[ -n "$OUT_DIR" ] || OUT_DIR="$WORKDIR/out"
mkdir -p "$OUT_DIR"

ORIG_LRU_GEN=""
MODEL_TEMP=0
MODEL_ABS=""
SCRATCH="$WORKDIR/pressure.bin"

cleanup() {
    if [ -n "$ORIG_LRU_GEN" ] && [ -e /sys/kernel/mm/lru_gen/enabled ]; then
        echo "$ORIG_LRU_GEN" | sudo tee /sys/kernel/mm/lru_gen/enabled >/dev/null 2>&1 || true
    fi
    if [ -n "$MODEL_ABS" ] && [ -e "$MODEL_ABS" ] && [ "$RUN_UID" -ne 0 ]; then
        sudo chown "$RUN_UID:$RUN_GID" "$MODEL_ABS" 2>/dev/null || true
        sudo chmod 0644 "$MODEL_ABS" 2>/dev/null || true
    fi
    if [ "$MODEL_TEMP" -eq 1 ] && [ -n "$MODEL_ABS" ]; then
        rm -f "$MODEL_ABS"
    fi
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

# Print environment & pre-registered BAR before any run
KERNEL_VER="$(uname -r)"
LRU_GEN_PATH="/sys/kernel/mm/lru_gen/enabled"
if [ -r "$LRU_GEN_PATH" ]; then
    ORIG_LRU_GEN="$(cat "$LRU_GEN_PATH")"
else
    ORIG_LRU_GEN="absent"
fi

echo "=== R18.13 Real-File Reclaim Check ==="
echo "kernel: $KERNEL_VER"
echo "lru_gen/enabled (initial): $ORIG_LRU_GEN"
echo "uid/gid: $RUN_UID/$RUN_GID"
echo "BAR: warm-green resident>=$BAR_GREEN_MIN_RESIDENT majflt<=$BAR_GREEN_MAX_MAJFLT | warm-red resident<$BAR_RED_MAX_RESIDENT or majflt>=$BAR_RED_MIN_MAJFLT | cold-cached read_frac<$BAR_COLD_CACHED_MAX_READ_FRAC | cold-full read_frac>=$BAR_COLD_FULL_MIN_READ_FRAC | trap read_frac<$BAR_TRAP_MAX_READ_FRAC"

# Fetch and verify pinned model if --model was not supplied
if [ -z "$MODEL" ]; then
    MODEL="$WORKDIR/$DEFAULT_FILE"
    MODEL_TEMP=1
    URL="https://huggingface.co/${DEFAULT_REPO}/resolve/${DEFAULT_REV}/${DEFAULT_FILE}"
    echo "model: downloading pinned validated MoE ${DEFAULT_REPO}@${DEFAULT_REV}/${DEFAULT_FILE}..."
    curl -fsSL --retry 3 -o "$MODEL" "$URL"
    GOT_SHA="$(sha256sum "$MODEL" | awk '{print $1}')"
    if [ "$GOT_SHA" != "$DEFAULT_SHA256" ]; then
        echo "ERROR: sha256 mismatch for $DEFAULT_FILE: got $GOT_SHA want $DEFAULT_SHA256" >&2
        exit 2
    fi
    echo "model: verified sha256=$GOT_SHA"
fi

MODEL_ABS="$(readlink -f "$MODEL")"
MODEL_BYTES="$(stat -c %s "$MODEL_ABS")"
HIGH_BYTES=$(( MODEL_BYTES + HEADROOM_MIB * 1048576 ))
echo "model: $MODEL_ABS ($(( MODEL_BYTES / 1048576 )) MiB), MemoryHigh=$(( HIGH_BYTES / 1048576 )) MiB, pressure=${PRESSURE_PASSES}x${PRESSURE_MIB} MiB"

# Create disk-backed scratch file for streaming reclaim pressure
dd if=/dev/urandom of="$SCRATCH" bs=1M count="$PRESSURE_MIB" status=none
sync

drop_caches() {
    sync
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
}

# Write helper scripts used inside each scope
cat > "$WORKDIR/warm_cell.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

CELL="$1"
RUNNER_BIN="$2"
MODEL_ABS="$3"
POP_ENV="$4"
PARALLEL="$5"
CTX="$6"
SCRATCH="$7"
PRESSURE_PASSES="$8"
OUT_ENV="$9"
ERR_LOG="${10}"

PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"

CG_REL="$(awk -F: '$1==0{print $3}' /proc/self/cgroup)"
CG_PATH="/sys/fs/cgroup${CG_REL}"
[ -f "$CG_PATH/memory.stat" ] || CG_PATH="/sys/fs/cgroup"

read_pgsteal() {
    python3 - "$CG_PATH/memory.stat" <<'PY'
import sys
vals = {}
with open(sys.argv[1]) as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 2:
            vals[parts[0]] = int(parts[1])
print(vals.get("pgsteal", vals.get("pgsteal_kswapd", 0) + vals.get("pgsteal_direct", 0)))
PY
}

mincore_frac() {
    python3 - "$MODEL_ABS" <<'PY'
import ctypes, mmap, os, sys
path = sys.argv[1]
fd = os.open(path, os.O_RDONLY)
sz = os.fstat(fd).st_size
ps = os.sysconf("SC_PAGE_SIZE")
npages = (sz + ps - 1) // ps
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
addr = libc.mmap(None, sz, mmap.PROT_READ, mmap.MAP_PRIVATE, fd, 0)
os.close(fd)
vec = (ctypes.c_ubyte * npages)()
rc = libc.mincore(addr, sz, vec)
libc.munmap(addr, sz)
if rc != 0:
    raise OSError(ctypes.get_errno(), "mincore failed")
res = sum(1 for b in vec if b & 1)
print(f"{res / npages:.6f}")
PY
}

read_proc_metrics() {
    python3 - "$1" "$MODEL_ABS" <<'PY'
import sys
pid, target = sys.argv[1], sys.argv[2]
rss_kb = 0
match = False
with open(f"/proc/{pid}/smaps") as f:
    for line in f:
        if not line[0].isspace() and "-" in line:
            parts = line.strip().split(None, 5)
            match = (len(parts) == 6 and parts[5] == target)
        elif match and line.startswith("Rss:"):
            rss_kb += int(line.split()[1])
vmpte_kb = 0
with open(f"/proc/{pid}/status") as f:
    for line in f:
        if line.startswith("VmPTE:"):
            vmpte_kb = int(line.split()[1])
with open(f"/proc/{pid}/stat") as f:
    s = f.read()
majflt = int(s[s.rfind(")") + 2:].split()[9])
print(f"{rss_kb/1024.0:.1f} {vmpte_kb} {majflt}")
PY
}

# CRITICAL (c): warm INSIDE the MemoryHigh cgroup so page cache charges here!
cat "$MODEL_ABS" > /dev/null
RES_PRE="$(mincore_frac)"

# Start runner --serve with the requested RUNNER_POPULATE setting
if [ "$POP_ENV" = "__UNSET__" ]; then
    unset RUNNER_POPULATE
else
    export RUNNER_POPULATE="$POP_ENV"
fi

T0="$(python3 -c 'import time; print(time.monotonic())')"
"$RUNNER_BIN" --serve --port "$PORT" --parallel "$PARALLEL" --gpu off -c "$CTX" -m "$MODEL_ABS" \
    >/dev/null 2>"$ERR_LOG" &
SRV_PID=$!
trap 'kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true' EXIT

READY=0
for _ in $(seq 1 1200); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
        echo "ERROR [$CELL]: runner exited early" >&2
        tail -n 30 "$ERR_LOG" >&2
        exit 2
    fi
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 0.25
done
[ "$READY" -eq 1 ] || { echo "ERROR [$CELL]: /health timeout" >&2; exit 2; }
LOAD_S="$(python3 - "$T0" <<'PY'
import sys, time
print(f"{time.monotonic() - float(sys.argv[1]):.3f}")
PY
)"

read -r SMAPS_LOAD VMPTE_LOAD MAJFLT_BEFORE <<<"$(read_proc_metrics "$SRV_PID")"
PGSTEAL_BEFORE="$(read_pgsteal)"

# CRITICAL (b): apply pressure BETWEEN load and FIRST prompt!
for _ in $(seq 1 "$PRESSURE_PASSES"); do
    cat "$SCRATCH" > /dev/null
done

PGSTEAL_AFTER="$(read_pgsteal)"
RES_PRESSURE="$(mincore_frac)"
read -r SMAPS_PRESSURE VMPTE_PRESSURE MAJFLT_PRE_P1 <<<"$(read_proc_metrics "$SRV_PID")"

# Send FIRST prompt and count its major faults + TTFT
TP0="$(python3 -c 'import time; print(time.monotonic())')"
curl -sf -X POST "http://127.0.0.1:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"The capital of France is","max_tokens":4,"temperature":0}' >/dev/null
P1_S="$(python3 - "$TP0" <<'PY'
import sys, time
print(f"{time.monotonic() - float(sys.argv[1]):.3f}")
PY
)"

read -r SMAPS_AFTER VMPTE_AFTER MAJFLT_AFTER <<<"$(read_proc_metrics "$SRV_PID")"

kill "$SRV_PID" 2>/dev/null || true
wait "$SRV_PID" 2>/dev/null || true
trap - EXIT

PGSTEAL_DELTA=$(( PGSTEAL_AFTER - PGSTEAL_BEFORE ))
MAJFLT_DELTA=$(( MAJFLT_AFTER - MAJFLT_PRE_P1 ))
POP_LINE="$(grep -m1 '^populate:' "$ERR_LOG" || echo '(none)')"

cat > "$OUT_ENV" <<RES
RES_PRE=$RES_PRE
RES_PRESSURE=$RES_PRESSURE
LOAD_S=$LOAD_S
P1_S=$P1_S
SMAPS_LOAD=$SMAPS_LOAD
SMAPS_PRESSURE=$SMAPS_PRESSURE
VMPTE_LOAD=$VMPTE_LOAD
VMPTE_PRESSURE=$VMPTE_PRESSURE
PGSTEAL_DELTA=$PGSTEAL_DELTA
MAJFLT_DELTA=$MAJFLT_DELTA
POP_LINE=$(printf '%q' "$POP_LINE")
RES
EOF
chmod +x "$WORKDIR/warm_cell.sh"

cat > "$WORKDIR/cold_cell.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

CELL="$1"
RUNNER_BIN="$2"
MODEL_ABS="$3"
POP_ENV="$4"
CTX="$5"
OUT_ENV="$6"
ERR_LOG="$7"

PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"

if [ "$POP_ENV" = "__UNSET__" ]; then
    unset RUNNER_POPULATE
else
    export RUNNER_POPULATE="$POP_ENV"
fi

T0="$(python3 -c 'import time; print(time.monotonic())')"
"$RUNNER_BIN" --serve --port "$PORT" --gpu off -c "$CTX" -m "$MODEL_ABS" \
    >/dev/null 2>"$ERR_LOG" &
SRV_PID=$!
trap 'kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true' EXIT

READY=0
for _ in $(seq 1 1200); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
        echo "ERROR [$CELL]: runner exited early" >&2
        tail -n 30 "$ERR_LOG" >&2
        exit 2
    fi
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 0.1
done
[ "$READY" -eq 1 ] || { echo "ERROR [$CELL]: /health timeout" >&2; exit 2; }
LOAD_S="$(python3 - "$T0" <<'PY'
import sys, time
print(f"{time.monotonic() - float(sys.argv[1]):.3f}")
PY
)"

# Read /proc/$SRV_PID/io read_bytes (block-layer bytes read during load), smaps Rss, VmPTE
read -r READ_BYTES SMAPS_LOAD VMPTE_LOAD MAJFLT_LOAD <<<"$(python3 - "$SRV_PID" "$MODEL_ABS" <<'PY'
import sys
pid, target = sys.argv[1], sys.argv[2]
rbytes = 0
with open(f"/proc/{pid}/io") as f:
    for line in f:
        if line.startswith("read_bytes:"):
            rbytes = int(line.split()[1])
rss_kb = 0
match = False
with open(f"/proc/{pid}/smaps") as f:
    for line in f:
        if not line[0].isspace() and "-" in line:
            parts = line.strip().split(None, 5)
            match = (len(parts) == 6 and parts[5] == target)
        elif match and line.startswith("Rss:"):
            rss_kb += int(line.split()[1])
vmpte_kb = 0
with open(f"/proc/{pid}/status") as f:
    for line in f:
        if line.startswith("VmPTE:"):
            vmpte_kb = int(line.split()[1])
with open(f"/proc/{pid}/stat") as f:
    s = f.read()
majflt = int(s[s.rfind(")") + 2:].split()[9])
print(f"{rbytes} {rss_kb/1024.0:.1f} {vmpte_kb} {majflt}")
PY
)"

# Time to first token (max_tokens=1)
TT0="$(python3 -c 'import time; print(time.monotonic())')"
curl -sf -X POST "http://127.0.0.1:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"The capital of France is","max_tokens":1,"temperature":0}' >/dev/null
TTFT_S="$(python3 - "$TT0" <<'PY'
import sys, time
print(f"{time.monotonic() - float(sys.argv[1]):.3f}")
PY
)"

kill "$SRV_PID" 2>/dev/null || true
wait "$SRV_PID" 2>/dev/null || true
trap - EXIT

POP_LINE="$(grep -m1 '^populate:' "$ERR_LOG" || echo '(none)')"

cat > "$OUT_ENV" <<RES
LOAD_S=$LOAD_S
TTFT_S=$TTFT_S
READ_BYTES=$READ_BYTES
SMAPS_LOAD=$SMAPS_LOAD
VMPTE_LOAD=$VMPTE_LOAD
MAJFLT_LOAD=$MAJFLT_LOAD
POP_LINE=$(printf '%q' "$POP_LINE")
RES
EOF
chmod +x "$WORKDIR/cold_cell.sh"

run_warm_in_scope() {
    local cell="$1" runner_bin="$2" pop_env="$3" parallel="$4" out_env="$5" err_log="$6"
    drop_caches
    sudo systemd-run --scope --quiet \
        --uid="$RUN_UID" --gid="$RUN_GID" \
        -p MemoryAccounting=yes -p "MemoryHigh=$HIGH_BYTES" \
        "$WORKDIR/warm_cell.sh" "$cell" "$runner_bin" "$MODEL_ABS" "$pop_env" \
        "$parallel" "$CTX" "$SCRATCH" "$PRESSURE_PASSES" "$out_env" "$err_log"
}

OVERALL_FAIL=0

check_warm_cell() {
    local lru_tag="$1" cell="$2" expect="$3" runner_bin="$4" pop_env="$5" parallel="$6"
    local out_env="$OUT_DIR/${lru_tag}-${cell}.env"
    local err_log="$OUT_DIR/${lru_tag}-${cell}.err"
    run_warm_in_scope "${lru_tag}:${cell}" "$runner_bin" "$pop_env" "$parallel" "$out_env" "$err_log"
    # shellcheck disable=SC1090
    source "$out_env"
    local status
    status="$(python3 - "$expect" "$RES_PRESSURE" "$MAJFLT_DELTA" "$PGSTEAL_DELTA" \
        "$BAR_GREEN_MIN_RESIDENT" "$BAR_GREEN_MAX_MAJFLT" "$BAR_RED_MAX_RESIDENT" "$BAR_RED_MIN_MAJFLT" <<'PY'
import sys
expect, res, maj, pgs = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
g_res, g_maj, r_res, r_maj = float(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7]), int(sys.argv[8])
if pgs <= 0:
    print("FAIL(no-pgsteal)")
elif expect == "green":
    print("PASS" if (res >= g_res and maj <= g_maj) else "FAIL")
else:
    print("PASS" if (res < r_res or maj >= r_maj) else "FAIL")
PY
)"
    local pg_mb
    pg_mb="$(python3 -c "import os; print(f'{$PGSTEAL_DELTA * os.sysconf(\"SC_PAGE_SIZE\") / 1e6:.1f}')")"
    echo "CELL [$lru_tag] $cell (expect=$expect): $status | res_pre=$RES_PRE res_after_pressure=$RES_PRESSURE pgsteal_mb=$pg_mb p1_majflt=$MAJFLT_DELTA p1_s=$P1_S smaps_mb=${SMAPS_LOAD}->${SMAPS_PRESSURE} vmpte_kb=${VMPTE_LOAD}->${VMPTE_PRESSURE} load_s=$LOAD_S | $POP_LINE"
    if [ "$status" != "PASS" ]; then
        OVERALL_FAIL=1
    fi
}

# Build list of LRU modes to run (both MGLRU and classic two-list if sudo can toggle)
LRU_MODES=("initial")
if [ -w "$LRU_GEN_PATH" ] || sudo test -w "$LRU_GEN_PATH" 2>/dev/null; then
    if [ "$ORIG_LRU_GEN" != "0x0000" ] && [ "$ORIG_LRU_GEN" != "0" ]; then
        LRU_MODES=("mglru:$ORIG_LRU_GEN" "classic:0")
    else
        LRU_MODES=("classic:$ORIG_LRU_GEN" "mglru:0x0007")
    fi
fi

for entry in "${LRU_MODES[@]}"; do
    tag="${entry%%:*}"
    val="${entry#*:}"
    if [ "$tag" != "initial" ] && [ -e "$LRU_GEN_PATH" ]; then
        if echo "$val" | sudo tee "$LRU_GEN_PATH" >/dev/null 2>&1; then
            echo "--- LRU mode: $tag (lru_gen/enabled=$(cat "$LRU_GEN_PATH")) ---"
        else
            echo "--- LRU mode: $tag (toggle refused; skipping) ---"
            continue
        fi
    else
        echo "--- LRU mode: $tag (lru_gen/enabled=$ORIG_LRU_GEN) ---"
    fi

    check_warm_cell "$tag" "main"          "red"   "$BASE_RUNNER"   "__UNSET__" "1"
    check_warm_cell "$tag" "branch-off"    "red"   "$BRANCH_RUNNER" "0"         "1"
    check_warm_cell "$tag" "branch-cached" "green" "$BRANCH_RUNNER" "__UNSET__" "1"
    check_warm_cell "$tag" "branch-full"   "green" "$BRANCH_RUNNER" "full"      "1"
done

# Restore initial LRU mode before parallel / cold / trap cells
if [ -n "$ORIG_LRU_GEN" ] && [ "$ORIG_LRU_GEN" != "absent" ] && [ -e "$LRU_GEN_PATH" ]; then
    echo "$ORIG_LRU_GEN" | sudo tee "$LRU_GEN_PATH" >/dev/null 2>&1 || true
fi

# Cell 5: --parallel 2 VmPTE + reclaim protection check
echo "--- Parallel 2 VmPTE check ---"
check_warm_cell "par2" "branch-cached-p2" "green" "$BRANCH_RUNNER" "__UNSET__" "2"

# Cell 6: cold cache, cached (default) -> must NOT read file through during load
echo "--- Cold-cache cells ---"
drop_caches
"$WORKDIR/cold_cell.sh" "cold-cached" "$BRANCH_RUNNER" "$MODEL_ABS" "__UNSET__" "$CTX" \
    "$OUT_DIR/cold-cached.env" "$OUT_DIR/cold-cached.err"
# shellcheck disable=SC1090
source "$OUT_DIR/cold-cached.env"
ST_CC="$(python3 - "$READ_BYTES" "$MODEL_BYTES" "$BAR_COLD_CACHED_MAX_READ_FRAC" <<'PY'
import sys
rb, mb, cap = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
print("PASS" if (rb / mb) < cap else "FAIL")
PY
)"
echo "CELL [cold] branch-cached: $ST_CC | load_s=$LOAD_S ttft_s=$TTFT_S load_read_mb=$(python3 -c "print(f'{$READ_BYTES/1e6:.1f}')") ($(( READ_BYTES * 100 / MODEL_BYTES ))%) smaps_mb=$SMAPS_LOAD vmpte_kb=$VMPTE_LOAD | $POP_LINE"
[ "$ST_CC" = "PASS" ] || OVERALL_FAIL=1

# Cell 7: cold cache, full -> reads full model during load
drop_caches
"$WORKDIR/cold_cell.sh" "cold-full" "$BRANCH_RUNNER" "$MODEL_ABS" "full" "$CTX" \
    "$OUT_DIR/cold-full.env" "$OUT_DIR/cold-full.err"
# shellcheck disable=SC1090
source "$OUT_DIR/cold-full.env"
ST_CF="$(python3 - "$READ_BYTES" "$MODEL_BYTES" "$BAR_COLD_FULL_MIN_READ_FRAC" <<'PY'
import sys
rb, mb, floor = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
print("PASS" if (rb / mb) >= floor else "FAIL")
PY
)"
echo "CELL [cold] branch-full:   $ST_CF | load_s=$LOAD_S ttft_s=$TTFT_S load_read_mb=$(python3 -c "print(f'{$READ_BYTES/1e6:.1f}')") ($(( READ_BYTES * 100 / MODEL_BYTES ))%) smaps_mb=$SMAPS_LOAD vmpte_kb=$VMPTE_LOAD | $POP_LINE"
[ "$ST_CF" = "PASS" ] || OVERALL_FAIL=1

# Cell 8: root-owned read-only mincore trap (kernel 134fca9063ad), cold cache
echo "--- Root-owned read-only mincore trap cell ---"
if [ "$RUN_UID" -eq 0 ]; then
    echo "CELL [trap] root-owned-ro: FAIL (must run as non-root user with sudo to test kernel 134fca9063ad)"
    OVERALL_FAIL=1
else
    sudo chown root:root "$MODEL_ABS"
    sudo chmod 0444 "$MODEL_ABS"
    drop_caches
    "$WORKDIR/cold_cell.sh" "mincore-trap" "$BRANCH_RUNNER" "$MODEL_ABS" "__UNSET__" "$CTX" \
        "$OUT_DIR/mincore-trap.env" "$OUT_DIR/mincore-trap.err"
    sudo chown "$RUN_UID:$RUN_GID" "$MODEL_ABS"
    sudo chmod 0644 "$MODEL_ABS"
    # shellcheck disable=SC1090
    source "$OUT_DIR/mincore-trap.env"
    ST_TRAP="$(python3 - "$READ_BYTES" "$MODEL_BYTES" "$BAR_TRAP_MAX_READ_FRAC" "$OUT_DIR/mincore-trap.err" <<'PY'
import sys
rb, mb, cap, errp = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
err = open(errp).read()
ok_io = (rb / mb) < cap
ok_msg = "mincore untrusted" in err
print("PASS" if (ok_io and ok_msg) else f"FAIL(io={ok_io},msg={ok_msg})")
PY
)"
    echo "CELL [trap] root-owned-ro: $ST_TRAP | load_s=$LOAD_S ttft_s=$TTFT_S load_read_mb=$(python3 -c "print(f'{$READ_BYTES/1e6:.1f}')") ($(( READ_BYTES * 100 / MODEL_BYTES ))%) smaps_mb=$SMAPS_LOAD vmpte_kb=$VMPTE_LOAD | $POP_LINE"
    [ "$ST_TRAP" = "PASS" ] || OVERALL_FAIL=1
fi

if [ "$OVERALL_FAIL" -eq 0 ]; then
    echo "PASS: r1813-reclaim-check all cells passed (kernel=$KERNEL_VER lru_gen=$ORIG_LRU_GEN)"
    exit 0
else
    echo "FAIL: r1813-reclaim-check one or more cells failed (kernel=$KERNEL_VER lru_gen=$ORIG_LRU_GEN)"
    exit 1
fi
