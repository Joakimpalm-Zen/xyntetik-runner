#!/usr/bin/env bash
# R18.13 real-file Linux cgroup v2 memory.high reclaim check.
#
# One-command red-then-green check:
#   scripts/r1813-reclaim-check.sh <model.gguf> <runner-bin> [extra runner args...]
#
# Expected results on a real model (e.g. Inkling-2-Mini-UD-Q8_K_XL.gguf):
#   - On main (before R18.13): FAIL (resident_during drops below 0.990 under pgsteal)
#   - On R18.13 branch:        PASS (resident_during >= 0.990 under pgsteal)
#   - On R18.13 with RUNNER_POPULATE=0: FAIL (control)
#
# Optional environment knobs:
#   HEADROOM_MIB=1536      MiB added above model file size for memory.high
#   PRESSURE_MIB=4096      Size of streaming scratch file (MiB)
#   PRESSURE_PASSES=3      Number of sequential passes over scratch file
#   MIN_RESIDENT=0.990     Minimum resident fraction during reclaim to PASS
#   CTX=512                Context length passed to runner --serve (-c)
#   SCRATCH_DIR=/var/tmp   Directory for disk-backed pressure scratch file
#   CGROUP_DIR=""          Optional pre-created writable cgroup v2 directory

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <model.gguf> <runner-bin> [extra runner args...]" >&2
    exit 2
fi

if [ "$(uname -s)" != "Linux" ]; then
    echo "ERROR: r1813-reclaim-check.sh requires Linux with cgroup v2" >&2
    exit 2
fi

MODEL="$1"
RUNNER="$2"
shift 2

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model file not found: $MODEL" >&2
    exit 2
fi
if [ ! -x "$RUNNER" ]; then
    echo "ERROR: runner binary not executable: $RUNNER" >&2
    exit 2
fi

HEADROOM_MIB="${HEADROOM_MIB:-1536}"
PRESSURE_MIB="${PRESSURE_MIB:-4096}"
PRESSURE_PASSES="${PRESSURE_PASSES:-3}"
MIN_RESIDENT="${MIN_RESIDENT:-0.990}"
CTX="${CTX:-512}"
SCRATCH_BASE="${SCRATCH_DIR:-/var/tmp}"
[ -d "$SCRATCH_BASE" ] && [ -w "$SCRATCH_BASE" ] || SCRATCH_BASE="${TMPDIR:-/tmp}"

MODEL_ABS="$(readlink -f "$MODEL")"
RUNNER_ABS="$(readlink -f "$RUNNER")"
MODEL_BYTES="$(stat -c %s "$MODEL_ABS")"
HIGH_BYTES=$(( MODEL_BYTES + HEADROOM_MIB * 1048576 ))

WORKDIR="$(mktemp -d "$SCRATCH_BASE/r1813-reclaim.XXXXXX")"
SCRATCH="$WORKDIR/pressure.bin"
RESULT_FILE="$WORKDIR/result.env"
CHILD_CG=""

cleanup() {
    if [ -n "$CHILD_CG" ] && [ -d "$CHILD_CG" ]; then
        rmdir "$CHILD_CG" 2>/dev/null || true
    fi
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"

echo "r1813-reclaim: model=$MODEL_ABS ($(( MODEL_BYTES / 1048576 )) MiB)"
echo "r1813-reclaim: memory.high=$(( HIGH_BYTES / 1048576 )) MiB (headroom=${HEADROOM_MIB} MiB), pressure=${PRESSURE_PASSES}x${PRESSURE_MIB} MiB"

# Create disk-backed scratch file for streaming reclaim pressure
dd if=/dev/urandom of="$SCRATCH" bs=1M count="$PRESSURE_MIB" status=none
sync

# Pre-warm the model file into page cache (same state as Claim 2 pre())
echo "r1813-reclaim: pre-warming model into page cache..."
cat "$MODEL_ABS" > /dev/null

# Helper: measure mincore(2) resident fraction of $MODEL_ABS
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
if addr == ctypes.c_void_p(-1).value:
    raise OSError(ctypes.get_errno(), "mmap failed")
vec = (ctypes.c_ubyte * npages)()
rc = libc.mincore(addr, sz, vec)
libc.munmap(addr, sz)
if rc != 0:
    raise OSError(ctypes.get_errno(), "mincore failed")
res = sum(1 for b in vec if b & 1)
print(f"{res / npages:.6f}")
PY
}

PRE_FRAC="$(mincore_frac)"
echo "r1813-reclaim: resident_pre=$PRE_FRAC"

# Inner payload executed inside the memory.high-capped cgroup
cat > "$WORKDIR/inner.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

MODEL_ABS="$1"
RUNNER_ABS="$2"
PORT="$3"
CTX="$4"
SCRATCH="$5"
PRESSURE_PASSES="$6"
WORKDIR="$7"
RESULT_FILE="$8"
shift 8

CG_REL="$(awk -F: '$1==0{print $3}' /proc/self/cgroup)"
CG_PATH="/sys/fs/cgroup${CG_REL}"
if [ ! -f "$CG_PATH/memory.stat" ] && [ -f "/sys/fs/cgroup/memory.stat" ]; then
    CG_PATH="/sys/fs/cgroup"
fi

read_pgsteal() {
    python3 - "$CG_PATH/memory.stat" <<'PY'
import sys
vals = {}
with open(sys.argv[1]) as f:
    for line in f:
        k, v = line.split()[:2]
        vals[k] = int(v)
if "pgsteal" in vals:
    print(vals["pgsteal"])
else:
    print(vals.get("pgsteal_kswapd", 0) + vals.get("pgsteal_direct", 0))
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

smaps_model_rss_mb() {
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
print(f"{rss_kb / 1024.0:.1f}")
PY
}

read_majflt() {
    python3 - "$1" <<'PY'
import sys
with open(f"/proc/{sys.argv[1]}/stat") as f:
    s = f.read()
# comm is inside (...), strip up to last ')'
rest = s[s.rfind(")") + 2:].split()
# field 12 in /proc/pid/stat is index 9 after comm+state? No: field 3 is state (index 0), so field 12 is index 9.
print(int(rest[9]))
PY
}

echo "r1813-reclaim: cgroup=$CG_PATH memory.high=$(cat "$CG_PATH/memory.high" 2>/dev/null || echo unknown)"

"$RUNNER_ABS" --serve --port "$PORT" --gpu off -c "$CTX" -m "$MODEL_ABS" "$@" \
    >"$WORKDIR/runner.out" 2>"$WORKDIR/runner.err" &
SRV_PID=$!
trap 'kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true' EXIT

# Wait up to 300 s for /health
READY=0
for _ in $(seq 1 1200); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
        echo "ERROR: runner exited prematurely:" >&2
        tail -n 30 "$WORKDIR/runner.err" >&2
        exit 2
    fi
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 0.25
done
if [ "$READY" -ne 1 ]; then
    echo "ERROR: timed out waiting for runner /health" >&2
    tail -n 30 "$WORKDIR/runner.err" >&2
    exit 2
fi

# Warm prompt 1 (touches hot matrices + a small subset of embedding/expert rows)
curl -sf -X POST "http://127.0.0.1:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"The capital of France is","max_tokens":4,"temperature":0}' >/dev/null

SMAPS_BEFORE="$(smaps_model_rss_mb "$SRV_PID")"
PGSTEAL_BEFORE="$(read_pgsteal)"
MAJFLT_BEFORE="$(read_majflt "$SRV_PID")"

# Drive streaming file-cache pressure inside the same memory.high cgroup
for _ in $(seq 1 "$PRESSURE_PASSES"); do
    cat "$SCRATCH" > /dev/null
done

PGSTEAL_AFTER="$(read_pgsteal)"
RESIDENT_DURING="$(mincore_frac)"
SMAPS_DURING="$(smaps_model_rss_mb "$SRV_PID")"

# Prompt 2 with fresh tokens to observe major faults on cold rows
curl -sf -X POST "http://127.0.0.1:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Quantum electrodynamics describes how light and matter interact","max_tokens":4,"temperature":0}' >/dev/null

MAJFLT_AFTER="$(read_majflt "$SRV_PID")"

# Stop server cleanly before measuring post-exit residency
kill "$SRV_PID" 2>/dev/null || true
wait "$SRV_PID" 2>/dev/null || true
trap - EXIT

RESIDENT_AFTER="$(mincore_frac)"
PGSTEAL_DELTA=$(( PGSTEAL_AFTER - PGSTEAL_BEFORE ))
MAJFLT_DELTA=$(( MAJFLT_AFTER - MAJFLT_BEFORE ))

cat > "$RESULT_FILE" <<RES
SMAPS_BEFORE=$SMAPS_BEFORE
SMAPS_DURING=$SMAPS_DURING
RESIDENT_DURING=$RESIDENT_DURING
RESIDENT_AFTER=$RESIDENT_AFTER
PGSTEAL_DELTA=$PGSTEAL_DELTA
MAJFLT_DELTA=$MAJFLT_DELTA
RES
EOF
chmod +x "$WORKDIR/inner.sh"

# Launch inner.sh inside a memory.high-capped cgroup v2:
#   1) explicit CGROUP_DIR if provided
#   2) direct child cgroup v2 if current cgroup permits subtree delegation
#   3) systemd-run --user --scope
#   4) systemd-run --scope (root)
run_in_cgroup() {
    if [ -n "${CGROUP_DIR:-}" ]; then
        echo "$HIGH_BYTES" > "$CGROUP_DIR/memory.high"
        ( echo "$BASHPID" > "$CGROUP_DIR/cgroup.procs"
          exec "$WORKDIR/inner.sh" "$MODEL_ABS" "$RUNNER_ABS" "$PORT" "$CTX" \
               "$SCRATCH" "$PRESSURE_PASSES" "$WORKDIR" "$RESULT_FILE" "$@" )
        return
    fi

    local cur_rel cur_cg
    cur_rel="$(awk -F: '$1==0{print $3}' /proc/self/cgroup 2>/dev/null || true)"
    cur_cg="/sys/fs/cgroup${cur_rel}"
    if [ -w "$cur_cg" ] && [ -f "$cur_cg/cgroup.controllers" ]; then
        if grep -qw memory "$cur_cg/cgroup.subtree_control" 2>/dev/null || \
           echo "+memory" > "$cur_cg/cgroup.subtree_control" 2>/dev/null; then
            CHILD_CG="$cur_cg/r1813-repro-$$"
            if mkdir "$CHILD_CG" 2>/dev/null; then
                echo "$HIGH_BYTES" > "$CHILD_CG/memory.high"
                ( echo "$BASHPID" > "$CHILD_CG/cgroup.procs"
                  exec "$WORKDIR/inner.sh" "$MODEL_ABS" "$RUNNER_ABS" "$PORT" "$CTX" \
                       "$SCRATCH" "$PRESSURE_PASSES" "$WORKDIR" "$RESULT_FILE" "$@" )
                return
            fi
        fi
    fi

    if command -v systemd-run >/dev/null 2>&1; then
        if systemd-run --user --scope --quiet true 2>/dev/null; then
            systemd-run --user --scope --quiet \
                -p MemoryAccounting=yes -p "MemoryHigh=$HIGH_BYTES" \
                "$WORKDIR/inner.sh" "$MODEL_ABS" "$RUNNER_ABS" "$PORT" "$CTX" \
                "$SCRATCH" "$PRESSURE_PASSES" "$WORKDIR" "$RESULT_FILE" "$@"
            return
        fi
        if [ "$(id -u)" -eq 0 ] && systemd-run --scope --quiet true 2>/dev/null; then
            systemd-run --scope --quiet \
                -p MemoryAccounting=yes -p "MemoryHigh=$HIGH_BYTES" \
                "$WORKDIR/inner.sh" "$MODEL_ABS" "$RUNNER_ABS" "$PORT" "$CTX" \
                "$SCRATCH" "$PRESSURE_PASSES" "$WORKDIR" "$RESULT_FILE" "$@"
            return
        fi
    fi

    echo "ERROR: could not create a cgroup v2 with memory.high (tried direct cgroup v2 and systemd-run). Pass CGROUP_DIR=/sys/fs/cgroup/<delegated-dir>." >&2
    exit 2
}

run_in_cgroup "$@"

# Evaluate verdict
# shellcheck disable=SC1090
source "$RESULT_FILE"

python3 - "$RESIDENT_DURING" "$RESIDENT_AFTER" "$MIN_RESIDENT" "$PGSTEAL_DELTA" "$SMAPS_BEFORE" "$SMAPS_DURING" "$MAJFLT_DELTA" <<'PY'
import os, sys
res_dur = float(sys.argv[1])
res_aft = float(sys.argv[2])
min_res = float(sys.argv[3])
pgsteal = int(sys.argv[4])
smaps_b = sys.argv[5]
smaps_d = sys.argv[6]
majflt  = int(sys.argv[7])
pg_mb   = pgsteal * os.sysconf("SC_PAGE_SIZE") / 1e6

if pgsteal <= 0:
    print(f"FAIL: r1813-reclaim no pgsteal observed (pgsteal_pages=0); pressure did not trigger cgroup reclaim")
    sys.exit(1)

ok = res_dur >= min_res
tag = "PASS" if ok else "FAIL"
cmp = ">=" if ok else "<"
print(f"{tag}: r1813-reclaim resident_during={res_dur:.6f} ({cmp} {min_res:.3f}) "
      f"resident_after={res_aft:.6f} smaps_rss_mb={smaps_b}->{smaps_d} "
      f"pgsteal_mb={pg_mb:.1f} majflt_delta={majflt}")
sys.exit(0 if ok else 1)
PY
