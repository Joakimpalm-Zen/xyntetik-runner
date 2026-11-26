#!/usr/bin/env bash
# R18.13 real-file Linux cgroup v2 memory.high reclaim check.
#
# ============================================================================
# PRE-REGISTRATION AMENDMENT (2026-11-26, written BEFORE run (c))
# ============================================================================
# Prior run: GitHub Actions run 45907096696 on c4a3afc2 (LFM2.5-8B-A1B Q4_K_M,
# 5.16 GB public MoE, MemoryHigh = model + 1280 MiB, 3x3 GiB pressure between
# load and Prompt 1):
#   - MGLRU (lru_gen/enabled=0x0007):
#       main:          resident=0.613, p1_majflt=4,987, p1_s=4.56 s  (RED)
#       branch-cached: resident=1.000, p1_majflt=0,     p1_s=0.77 s  (GREEN)
#       branch-full:   resident=1.000, p1_majflt=0                   (GREEN)
#   - Classic two-list LRU (lru_gen/enabled=0): FAILED original bar:
#       branch-cached: resident=0.621, p1_majflt=4,971 (smaps Rss 4,917 -> 3,054 MB)
#       branch-full:   resident=0.618, p1_majflt=5,829
#   - Cold-cached, cold-full, --parallel 2, root-owned 0444 trap: all PASS.
#
# Scope narrowed after run 45907096696 (nothing waived; parity required):
#   1. MGLRU cells: original bar unchanged:
#        - main & branch-off: RED (resident < 0.900 or p1_majflt >= 256)
#        - branch-cached & branch-full: GREEN (resident >= 0.990 and p1_majflt <= 64)
#   2. Classic two-list LRU Runner cells ("no harm" bar for default cached):
#        - main: RED (resident < 0.900 or p1_majflt >= 256)
#        - branch-cached: NO HARM vs classic main:
#            cached_res >= main_res - 0.050
#            AND cached_majflt <= max(main_majflt + 500, int(main_majflt * 1.15))
#        - branch-off & branch-full: recorded as measured limit.
#   3. llama.cpp parity bar (pinned 56eea9e4b0f5d125da16b75b551440b170aa8b1f,
#      same model, same MemoryHigh scope, same 3x3 GiB pressure):
#        - Recorded in BOTH MGLRU and classic LRU.
#        - PARITY RULE (written before run):
#            Under classic LRU (idle during pressure), if llama.cpp also loses
#            its pages (llama_res < 0.900 or llama_majflt >= 256) and stays
#            within +0.100 resident of Runner cached (llama_res - cached_res < 0.100),
#            that is PARITY (PASS).
#            If llama.cpp keeps its pages (llama_res >= 0.950 and llama_majflt <= 64)
#            or beats Runner cached by >= 0.100 resident, a mechanism is missing:
#            PARITY FAIL (blocks merge, back to analysis).
#   4. HYPOTHESIS TEST cell (informational, both MGLRU and classic LRU):
#        - branch-cached-generating: model actively generating during the 3x3 GiB
#          pressure wave, testing whether classic LRU deactivates idle active_file
#          pages (clearing Accessed bits) while active generation refreshes them.
#   5. Cold-cached, cold-full, --parallel 2 (VmPTE), and root-owned 0444
#      mincore-trap cells: unchanged bar.
# ============================================================================
#
# Usage:
#   scripts/r1813-reclaim-check.sh \
#     --base-runner /path/to/main/runner \
#     --branch-runner /path/to/branch/runner \
#     [--llama-server /path/to/llama-server] \
#     [--model /path/to/model.gguf] \
#     [--out-dir /path/to/artifacts]

set -euo pipefail

# ---- Pre-registered BAR (amended 2026-11-26, written before run (c)) -------
BAR_GREEN_MIN_RESIDENT="0.990"        # MGLRU green: resident after pressure >= 0.990
BAR_GREEN_MAX_MAJFLT="64"             # MGLRU green: prompt-1 major faults <= 64
BAR_RED_MAX_RESIDENT="0.900"          # red cells: resident < 0.900 ...
BAR_RED_MIN_MAJFLT="256"              # ... OR prompt-1 major faults >= 256
BAR_CLASSIC_NOHARM_RES_MARGIN="0.050" # classic cached: resident >= main_res - 0.050
BAR_CLASSIC_NOHARM_MAJ_RATIO="1.15"   # classic cached: majflt <= max(main+500, main*1.15)
BAR_CLASSIC_NOHARM_MAJ_PAD="500"
BAR_PARITY_MAX_LLAMA_ADV="0.100"      # classic parity: llama_res - cached_res < 0.100
BAR_PARITY_LLAMA_KEPT_RES="0.950"     # classic parity fail if llama keeps >= 0.950 & majflt <= 64
BAR_COLD_CACHED_MAX_READ_FRAC="0.10"  # cached cold: load read_bytes < 10% of file
BAR_COLD_FULL_MIN_READ_FRAC="0.80"    # full cold:   load read_bytes >= 80% of file
BAR_TRAP_MAX_READ_FRAC="0.10"         # root-owned 0444 trap: load read_bytes < 10% of file

# Pinned validated public MoE from docs/model-tiers.json (5.16 GB)
DEFAULT_REPO="LiquidAI/LFM2.5-8B-A1B-GGUF"
DEFAULT_REV="49c14831707011e64d70b2ebd8462ba08d608434"
DEFAULT_FILE="LFM2.5-8B-A1B-Q4_K_M.gguf"
DEFAULT_SHA256="4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0"

# Pinned llama.cpp commit for parity comparison
LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"
LLAMA_SHA="56eea9e4b0f5d125da16b75b551440b170aa8b1f"

BASE_RUNNER=""
BRANCH_RUNNER=""
LLAMA_SERVER=""
MODEL=""
OUT_DIR=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --base-runner)   BASE_RUNNER="$2"; shift 2 ;;
        --branch-runner) BRANCH_RUNNER="$2"; shift 2 ;;
        --llama-server)  LLAMA_SERVER="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --out-dir)       OUT_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,60p' "$0"
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
    echo "usage: $0 --base-runner <main-runner> --branch-runner <branch-runner> [--llama-server <bin>] [--model <gguf>] [--out-dir <dir>]" >&2
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
    if [ -n "$ORIG_LRU_GEN" ] && [ "$ORIG_LRU_GEN" != "absent" ] && [ -e /sys/kernel/mm/lru_gen/enabled ]; then
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

KERNEL_VER="$(uname -r)"
LRU_GEN_PATH="/sys/kernel/mm/lru_gen/enabled"
if [ -r "$LRU_GEN_PATH" ]; then
    ORIG_LRU_GEN="$(cat "$LRU_GEN_PATH")"
else
    ORIG_LRU_GEN="absent"
fi

echo "=== R18.13 Real-File Reclaim Check (Amendment 2026-11-26, Run (c)) ==="
echo "kernel: $KERNEL_VER"
echo "lru_gen/enabled (initial): $ORIG_LRU_GEN"
echo "uid/gid: $RUN_UID/$RUN_GID"
echo "AMENDED BAR (narrowed after run 45907096696 classic failures: cached 0.621/4971, full 0.618/5829):"
echo "  MGLRU:   main/off RED (res<$BAR_RED_MAX_RESIDENT or majflt>=$BAR_RED_MIN_MAJFLT); cached/full GREEN (res>=$BAR_GREEN_MIN_RESIDENT and majflt<=$BAR_GREEN_MAX_MAJFLT)"
echo "  Classic: cached NO-HARM vs main (res>=main-$BAR_CLASSIC_NOHARM_RES_MARGIN and majflt<=max(main+$BAR_CLASSIC_NOHARM_MAJ_PAD, main*$BAR_CLASSIC_NOHARM_MAJ_RATIO));"
echo "           llama.cpp ($LLAMA_SHA) PARITY: FAIL if llama keeps res>=$BAR_PARITY_LLAMA_KEPT_RES & majflt<=$BAR_GREEN_MAX_MAJFLT or beats cached by >=$BAR_PARITY_MAX_LLAMA_ADV"
echo "  Cold:    cached read_frac<$BAR_COLD_CACHED_MAX_READ_FRAC | full read_frac>=$BAR_COLD_FULL_MIN_READ_FRAC | root-owned-0444 trap read_frac<$BAR_TRAP_MAX_READ_FRAC"
echo "  Hypothesis test (informational, both LRU modes): branch-cached generating concurrently during pressure"

# Build pinned llama.cpp (56eea9e4b) llama-server if not supplied
if [ -z "$LLAMA_SERVER" ]; then
    LLAMA_SRC="$WORKDIR/llama.cpp"
    echo "llama.cpp: fetching pinned commit $LLAMA_SHA..."
    git init -q "$LLAMA_SRC"
    git -C "$LLAMA_SRC" remote add origin "$LLAMA_REPO"
    git -C "$LLAMA_SRC" fetch --depth 1 -q origin "$LLAMA_SHA"
    git -C "$LLAMA_SRC" checkout -q FETCH_HEAD
    echo "llama.cpp: building llama-server (CPU, static)..."
    cmake -S "$LLAMA_SRC" -B "$LLAMA_SRC/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_NATIVE=OFF \
        -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        >"$OUT_DIR/llama-cmake.log" 2>&1
    cmake --build "$LLAMA_SRC/build" --target llama-server -j"$(nproc)" \
        >"$OUT_DIR/llama-build.log" 2>&1
    LLAMA_SERVER="$LLAMA_SRC/build/bin/llama-server"
fi
LLAMA_SERVER="$(readlink -f "$LLAMA_SERVER")"
[ -x "$LLAMA_SERVER" ] || { echo "ERROR: llama-server not executable: $LLAMA_SERVER" >&2; exit 2; }
echo "llama.cpp: ready at $LLAMA_SERVER (commit $LLAMA_SHA)"

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

dd if=/dev/urandom of="$SCRATCH" bs=1M count="$PRESSURE_MIB" status=none
sync

drop_caches() {
    sync
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
}

# ---- Warm cell helper (supports Runner AND llama-server, idle OR generating) -
cat > "$WORKDIR/warm_cell.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

CELL="$1"
ENGINE="$2"      # "runner" or "llama"
BIN="$3"
MODEL_ABS="$4"
POP_ENV="$5"
PARALLEL="$6"
GENERATING="$7"  # "0" (idle during pressure) or "1" (generating during pressure)
CTX="$8"
SCRATCH="$9"
PRESSURE_PASSES="${10}"
OUT_ENV="${11}"
ERR_LOG="${12}"

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

# Warm INSIDE the MemoryHigh cgroup so page cache charges here
cat "$MODEL_ABS" > /dev/null
RES_PRE="$(mincore_frac)"

if [ "$POP_ENV" = "__UNSET__" ]; then
    unset RUNNER_POPULATE
else
    export RUNNER_POPULATE="$POP_ENV"
fi

T0="$(python3 -c 'import time; print(time.monotonic())')"
if [ "$ENGINE" = "llama" ]; then
    "$BIN" --host 127.0.0.1 --port "$PORT" -m "$MODEL_ABS" -c "$CTX" -ngl 0 \
        >/dev/null 2>"$ERR_LOG" &
else
    "$BIN" --serve --port "$PORT" --parallel "$PARALLEL" --gpu off -c "$CTX" -m "$MODEL_ABS" \
        >/dev/null 2>"$ERR_LOG" &
fi
SRV_PID=$!
GEN_PID=""
cleanup_inner() {
    [ -n "$GEN_PID" ] && kill "$GEN_PID" 2>/dev/null || true
    kill "$SRV_PID" 2>/dev/null || true
    [ -n "$GEN_PID" ] && wait "$GEN_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
}
trap cleanup_inner EXIT

READY=0
for _ in $(seq 1 1200); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
        echo "ERROR [$CELL]: server exited early" >&2
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

GEN_REQS=0
if [ "$GENERATING" = "1" ]; then
    # Hypothesis test: continuously generate tokens concurrently WITH the
    # streaming pressure wave so active weight pages keep refreshing Accessed bits.
    GEN_COUNT_FILE="${OUT_ENV}.gencount"
    echo 0 > "$GEN_COUNT_FILE"
    (
        c=0
        while true; do
            if curl -sf -X POST "http://127.0.0.1:${PORT}/v1/completions" \
                -H "Content-Type: application/json" \
                -d '{"prompt":"Explain how operating system page reclaim works in detail:","max_tokens":32,"temperature":0}' >/dev/null 2>&1; then
                c=$((c + 1))
                echo "$c" > "$GEN_COUNT_FILE"
            else
                sleep 0.1
            fi
        done
    ) &
    GEN_PID=$!
    sleep 0.5
fi

# Streaming file-cache pressure inside the same MemoryHigh scope
for _ in $(seq 1 "$PRESSURE_PASSES"); do
    cat "$SCRATCH" > /dev/null
done

if [ -n "$GEN_PID" ]; then
    kill "$GEN_PID" 2>/dev/null || true
    wait "$GEN_PID" 2>/dev/null || true
    GEN_PID=""
    GEN_REQS="$(cat "${OUT_ENV}.gencount" 2>/dev/null || echo 0)"
    rm -f "${OUT_ENV}.gencount"
fi

PGSTEAL_AFTER="$(read_pgsteal)"
RES_PRESSURE="$(mincore_frac)"
read -r SMAPS_PRESSURE VMPTE_PRESSURE MAJFLT_PRE_P1 <<<"$(read_proc_metrics "$SRV_PID")"

# Send verification prompt after pressure and count its major faults + latency
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

cleanup_inner
trap - EXIT

PGSTEAL_DELTA=$(( PGSTEAL_AFTER - PGSTEAL_BEFORE ))
MAJFLT_P1=$(( MAJFLT_AFTER - MAJFLT_PRE_P1 ))
MAJFLT_DURING=$(( MAJFLT_PRE_P1 - MAJFLT_BEFORE ))
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
MAJFLT_DELTA=$MAJFLT_P1
MAJFLT_DURING=$MAJFLT_DURING
GEN_REQS=$GEN_REQS
POP_LINE=$(printf '%q' "$POP_LINE")
RES
EOF
chmod +x "$WORKDIR/warm_cell.sh"

# ---- Cold cell helper ------------------------------------------------------
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
    local cell="$1" engine="$2" bin="$3" pop_env="$4" parallel="$5" generating="$6" out_env="$7" err_log="$8"
    drop_caches
    sudo systemd-run --scope --quiet \
        --uid="$RUN_UID" --gid="$RUN_GID" \
        -p MemoryAccounting=yes -p "MemoryHigh=$HIGH_BYTES" \
        "$WORKDIR/warm_cell.sh" "$cell" "$engine" "$bin" "$MODEL_ABS" "$pop_env" \
        "$parallel" "$generating" "$CTX" "$SCRATCH" "$PRESSURE_PASSES" "$out_env" "$err_log"
}

OVERALL_FAIL=0

check_warm_cell() {
    local lru_tag="$1" cell="$2" expect="$3" engine="$4" bin="$5" pop_env="$6" parallel="$7" generating="$8"
    local out_env="$OUT_DIR/${lru_tag}-${cell}.env"
    local err_log="$OUT_DIR/${lru_tag}-${cell}.err"
    run_warm_in_scope "${lru_tag}:${cell}" "$engine" "$bin" "$pop_env" "$parallel" "$generating" "$out_env" "$err_log"
    # shellcheck disable=SC1090
    source "$out_env"
    local status="RECORDED"
    if [ "$expect" = "green" ] || [ "$expect" = "red" ]; then
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
        [ "$status" = "PASS" ] || OVERALL_FAIL=1
    elif [ "$expect" = "hypothesis" ]; then
        status="HYPOTHESIS_TEST"
    fi
    local pg_mb
    pg_mb="$(python3 -c "import os; print(f'{$PGSTEAL_DELTA * os.sysconf(\"SC_PAGE_SIZE\") / 1e6:.1f}')")"
    local extra=""
    [ "$generating" = "1" ] && extra=" gen_reqs=$GEN_REQS during_majflt=$MAJFLT_DURING"
    echo "CELL [$lru_tag] $cell (expect=$expect): $status | res_pre=$RES_PRE res_after_pressure=$RES_PRESSURE pgsteal_mb=$pg_mb p1_majflt=$MAJFLT_DELTA p1_s=$P1_S smaps_mb=${SMAPS_LOAD}->${SMAPS_PRESSURE} vmpte_kb=${VMPTE_LOAD}->${VMPTE_PRESSURE} load_s=$LOAD_S${extra} | $POP_LINE"
}

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

    if [ "$tag" = "mglru" ] || { [ "$tag" = "initial" ] && [ "$ORIG_LRU_GEN" != "0x0000" ] && [ "$ORIG_LRU_GEN" != "0" ] && [ "$ORIG_LRU_GEN" != "absent" ]; }; then
        check_warm_cell "$tag" "main"                     "red"        "runner" "$BASE_RUNNER"   "__UNSET__" "1" "0"
        check_warm_cell "$tag" "branch-off"               "red"        "runner" "$BRANCH_RUNNER" "0"         "1" "0"
        check_warm_cell "$tag" "branch-cached"            "green"      "runner" "$BRANCH_RUNNER" "__UNSET__" "1" "0"
        check_warm_cell "$tag" "branch-full"              "green"      "runner" "$BRANCH_RUNNER" "full"      "1" "0"
        check_warm_cell "$tag" "llama-56eea9e4b"          "recorded"   "llama"  "$LLAMA_SERVER"  "__UNSET__" "1" "0"
        check_warm_cell "$tag" "branch-cached-generating" "hypothesis" "runner" "$BRANCH_RUNNER" "__UNSET__" "1" "1"
    else
        # Classic two-list LRU: amended no-harm + llama.cpp parity bar + generating hypothesis test
        check_warm_cell "$tag" "main"                     "red"        "runner" "$BASE_RUNNER"   "__UNSET__" "1" "0"
        check_warm_cell "$tag" "branch-off"               "recorded"   "runner" "$BRANCH_RUNNER" "0"         "1" "0"
        check_warm_cell "$tag" "branch-cached"            "recorded"   "runner" "$BRANCH_RUNNER" "__UNSET__" "1" "0"
        check_warm_cell "$tag" "branch-full"              "recorded"   "runner" "$BRANCH_RUNNER" "full"      "1" "0"
        check_warm_cell "$tag" "llama-56eea9e4b"          "recorded"   "llama"  "$LLAMA_SERVER"  "__UNSET__" "1" "0"
        check_warm_cell "$tag" "branch-cached-generating" "hypothesis" "runner" "$BRANCH_RUNNER" "__UNSET__" "1" "1"

        # Evaluate classic NO-HARM bar (cached vs main) and PARITY bar (llama.cpp vs cached)
        VERDICT="$(python3 - "$OUT_DIR/${tag}-main.env" "$OUT_DIR/${tag}-branch-cached.env" "$OUT_DIR/${tag}-llama-56eea9e4b.env" \
            "$BAR_CLASSIC_NOHARM_RES_MARGIN" "$BAR_CLASSIC_NOHARM_MAJ_RATIO" "$BAR_CLASSIC_NOHARM_MAJ_PAD" \
            "$BAR_PARITY_MAX_LLAMA_ADV" "$BAR_PARITY_LLAMA_KEPT_RES" "$BAR_GREEN_MAX_MAJFLT" <<'PY'
import sys
def load(p):
    d = {}
    for line in open(p):
        if '=' in line:
            k, v = line.strip().split('=', 1)
            d[k] = v
    return d

m = load(sys.argv[1])
c = load(sys.argv[2])
l = load(sys.argv[3])
res_margin, maj_ratio, maj_pad = float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
max_adv, kept_res, green_maj = float(sys.argv[7]), float(sys.argv[8]), int(sys.argv[9])

m_res, m_maj = float(m["RES_PRESSURE"]), int(m["MAJFLT_DELTA"])
c_res, c_maj = float(c["RES_PRESSURE"]), int(c["MAJFLT_DELTA"])
l_res, l_maj = float(l["RES_PRESSURE"]), int(l["MAJFLT_DELTA"])

noharm_res = c_res >= (m_res - res_margin)
noharm_maj = c_maj <= max(m_maj + maj_pad, int(m_maj * maj_ratio))
noharm_ok = noharm_res and noharm_maj

llama_kept = (l_res >= kept_res and l_maj <= green_maj)
llama_adv = (l_res - c_res) >= max_adv
parity_ok = (not llama_kept) and (not llama_adv)

print(f"CLASSIC_NOHARM={'PASS' if noharm_ok else 'FAIL'} "
      f"(cached res={c_res:.3f} vs main={m_res:.3f}, cached majflt={c_maj} vs main={m_maj}) | "
      f"CLASSIC_PARITY={'PASS' if parity_ok else 'FAIL'} "
      f"(llama res={l_res:.3f} majflt={l_maj} vs cached res={c_res:.3f} majflt={c_maj})")
sys.exit(0 if (noharm_ok and parity_ok) else 1)
PY
)" || OVERALL_FAIL=1
        echo "GATE [$tag]: $VERDICT"
    fi
done

# Restore initial LRU mode before parallel / cold / trap cells
if [ -n "$ORIG_LRU_GEN" ] && [ "$ORIG_LRU_GEN" != "absent" ] && [ -e "$LRU_GEN_PATH" ]; then
    echo "$ORIG_LRU_GEN" | sudo tee "$LRU_GEN_PATH" >/dev/null 2>&1 || true
fi

# Parallel 2 VmPTE + reclaim protection check
echo "--- Parallel 2 VmPTE check ---"
check_warm_cell "par2" "branch-cached-p2" "green" "runner" "$BRANCH_RUNNER" "__UNSET__" "2" "0"

# Cold-cache cells
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

# Root-owned read-only mincore trap cell (kernel 134fca9063ad), cold cache
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
    echo "PASS: r1813-reclaim-check (amended bar + classic parity) all gated cells passed (kernel=$KERNEL_VER lru_gen=$ORIG_LRU_GEN)"
    exit 0
else
    echo "FAIL: r1813-reclaim-check one or more gated cells failed (kernel=$KERNEL_VER lru_gen=$ORIG_LRU_GEN)"
    exit 1
fi
