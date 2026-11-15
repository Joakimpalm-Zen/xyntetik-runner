#!/bin/bash
# R3.60, the profiling read before a third variant (the owner's pick,
# 2026-11-15: one more variant, after a profile). R3.57's prefill profiler
# (597ad72f) on the base 704b339a and on the killed CPU variant cb82532b,
# each at -b 64 and -b 512, three repeats, the order flipping each repeat.
# Run from a runner checkout on the lab:
#   debug/r360_profile.sh MODEL.gguf [THREADS]
# Builds both arms in ../r360p-base and ../r360p-var (CC and CFLAGS from the
# environment, as the pair's runs were built). Writes the trimmed prompt (the
# pair's 1,601 words, 1,601 tokens on gpt-oss-120b, a one-row tail at -b 64).
# Checks the model file's page-cache residency before every run and reads the
# file whole again under 0.99 (RESIDENT="cmd" replaces the built-in check).
# Prints each run's readings and the medians, and judges the prediction
# written on R3.60 before the profile.
set -eu
M=$1; T=${2:-26}
git fetch -q origin experiment/r3.60-profile-base experiment/r3.60-profile-cpu-variant
[ -d ../r360p-base ] || git worktree add -q --detach ../r360p-base origin/experiment/r3.60-profile-base
[ -d ../r360p-var ] || git worktree add -q --detach ../r360p-var origin/experiment/r3.60-profile-cpu-variant
(cd ../r360p-base && make -j16 runner >/dev/null) && (cd ../r360p-var && make -j16 runner >/dev/null)
mkdir -p r360p-out && cd r360p-out
python3 - > prompt.txt <<'PY'
import sys
w = ("the quick brown fox jumps over a lazy dog while seven bright stars watch from far above "
     "and a quiet river runs through green fields toward the old stone bridge where children play "
     "every summer afternoon").split()
sys.stdout.write(" ".join((w * 200)[:1601]))
PY
resident() {
  if [ -n "${RESIDENT:-}" ]; then $RESIDENT "$1"; return; fi
  python3 -I - "$1" <<'PY'
import ctypes, ctypes.util, mmap, os, sys
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
path = sys.argv[1]; size = os.path.getsize(path); fd = os.open(path, os.O_RDONLY)
addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
if addr is None or addr == ctypes.c_void_p(-1).value: sys.exit("mmap failed")
pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
vec = ctypes.create_string_buffer(pages)
if libc.mincore(ctypes.c_void_p(addr), size, vec) != 0: sys.exit("mincore failed")
print("%.4f" % ((pages - vec.raw[:pages].count(b"\0")) / pages))
libc.munmap(ctypes.c_void_p(addr), size); os.close(fd)
PY
}
run() {   # run ARM B OUT
  local r; r=$(resident "$M")
  if awk -v r="$r" 'BEGIN{exit !(r < 0.99)}'; then
    echo "$(date -u +%FT%TZ) $3 resident $r under 0.99: read whole again" >> runlog.txt
    dd if="$M" of=/dev/null bs=16M status=none; r=$(resident "$M")
  fi
  echo "$(date -u +%FT%TZ) $3 load1 $(cut -d' ' -f1 /proc/loadavg) resident $r" >> runlog.txt
  RUNNER_CPU_PROF=prefill ../../r360p-$1/runner -m "$M" --gpu off -t "$T" -c 4096 -b "$2" \
    --temp 0 -n 1 -f prompt.txt > "$3.out" 2> "$3.err"
}
for rep in 1 2 3; do
  if [ $rep -eq 2 ]; then arms="var base"; else arms="base var"; fi
  for arm in $arms; do for b in 64 512; do run $arm $b p-$arm-b$b-r$rep; done; done
done
python3 - <<'PY'
import glob, re, statistics
def read(path):
    t = open(path).read()
    st = re.search(r"^cpu-prof: (\d+) steps, ([\d.]+) ms a step", t, re.M)
    pr = re.search(r"^prompt: (\d+) tok, ([\d.]+) tok/s", t, re.M)
    if not st or not pr: raise SystemExit("%s: no cpu-prof report or prompt line" % path)
    steps = int(st.group(1))
    exp = oth = 0.0
    for m in re.finditer(r"^cpu-prof: run \S+ ([0-9a-f]{16}) ([\d.]+) ([\d.]+) ", t, re.M):
        if (int(m.group(1), 16) >> 60) & 1: exp += float(m.group(3))
        else: oth += float(m.group(3))
    rg = re.search(r"^cpu-prof: region experts-loop-wall.* ([\d.]+) ([\d.]+) ([\d.]+)$", t, re.M)
    return dict(tokens=int(pr.group(1)), tps=float(pr.group(2)), steps=steps,
                total=steps * float(st.group(2)), experts=steps * exp, other=steps * oth,
                loop=steps * float(rg.group(2)) if rg else float("nan"))
rows = {}
for f in sorted(glob.glob("p-*-r*.err")):
    arm, b = re.match(r"p-(\w+)-b(\d+)-r\d+\.err", f.split("/")[-1]).groups()
    r = read(f); rows.setdefault((arm, int(b)), []).append(r)
    print("%-22s tokens %d, %.1f tok/s; %d profiled passes: total %.0f ms, expert runs (bit 60) %.0f ms, "
          "expert loop wall %.0f ms, other runs %.0f ms" % (f, r["tokens"], r["tps"], r["steps"], r["total"],
          r["experts"], r["loop"], r["other"]))
med = {k: {f: statistics.median(r[f] for r in v) for f in ("tps", "total", "experts", "loop", "other")}
       for k, v in rows.items()}
print("\nMEDIANS of 3 (ms over the profiled passes of one prefill; the 1-row tail at -b 64 is a one-token pass, not profiled)")
for k in sorted(med):
    m = med[k]
    print("  %-4s -b %-3d  %.1f tok/s  total %.0f  expert runs %.0f  expert loop wall %.0f  other runs %.0f"
          % (k[0], k[1], m["tps"], m["total"], m["experts"], m["loop"], m["other"]))
b, v = med[("base", 64)], med[("var", 64)]
extra = v["total"] - b["total"]
share = (v["experts"] - b["experts"]) / extra if extra else float("nan")
oth = abs(v["other"] - b["other"]) / b["other"]
b5, v5 = med[("base", 512)], med[("var", 512)]
e512 = abs(v5["experts"] - b5["experts"]) / b5["experts"]
print("\nPREDICTION (written on R3.60 before the profile):")
print("  (1) -b 64: the variant's extra time %.0f ms; the expert runs carry %.0f%% of it (predicted 80%% or more): %s"
      % (extra, 100 * share, "HOLDS" if share >= 0.8 else "FAILS"))
print("  (2) -b 64: the other runs differ by %.1f%% (predicted under 5%%): %s" % (100 * oth, "HOLDS" if oth < 0.05 else "FAILS"))
print("  (3) -b 512: the expert runs of the two arms differ by %.1f%% (predicted within 5%%): %s"
      % (100 * e512, "HOLDS" if e512 < 0.05 else "FAILS"))
PY
