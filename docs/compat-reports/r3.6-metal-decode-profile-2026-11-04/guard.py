"""Run a command under the laptop's swap guard: wait (up to 5 min) for wired
< 2.2 GB and free+inactive > 4 GB, record free/wired/swap, then run it,
sampling swap each second; a run whose swap use grows more than 64 MB is
killed and reported STOPPED (exit 99). Usage: guard.py CMD..."""
import os, re, subprocess, sys, time, datetime
AVAIL = float(os.environ.get('GUARD_MIN_AVAIL', 4000)); WAIT = float(os.environ.get('GUARD_MAX_WAIT', 300))
def vm():
    o = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
    g = lambda k: int(re.search(k + r':\s+(\d+)', o).group(1)) * 16384 / 1e6
    return g('Pages free'), g('Pages inactive'), g('Pages wired down')
def swap():
    o = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout
    return float(re.search(r'used = ([0-9.]+)M', o).group(1))
now = lambda: datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%SZ')
t0 = time.time()
while True:
    f, i, w = vm()
    if (w < 2200 and f + i > AVAIL) or time.time() - t0 > WAIT: break
    time.sleep(2)
f, i, w = vm(); s0 = swap()
print(f'guard: start {now()} after {time.time()-t0:.0f} s | free {f:.0f} inactive {i:.0f} wired {w:.0f} MB, swap {s0:.0f} MB', file=sys.stderr, flush=True)
p = subprocess.Popen(sys.argv[1:]); peak = s0
while p.poll() is None:
    peak = max(peak, swap())
    if peak - s0 > 64:
        p.kill(); p.wait()
        print(f'guard: STOPPED {now()}: swap grew {s0:.0f} -> {peak:.0f} MB', file=sys.stderr, flush=True); sys.exit(99)
    time.sleep(1)
print(f'guard: end {now()} exit {p.returncode} | swap {s0:.0f} -> {peak:.0f} MB', file=sys.stderr, flush=True)
sys.exit(p.returncode)
