"""R3.6 Metal decode profile: Qwen3-4B Q4_K_M on the R3.28.2 build with
profile-only drop levers, the 512-token prompt, 128 decode tokens, --mlock.
Per arm: decode GPU ms a step (median of RUNNER_METAL_TIMING=1 n=1 spans,
steps 5..128) and CPU encode ms; three interleaved rounds under guard.py."""
import os, re, statistics as st, subprocess, sys, datetime, collections
W, M, P = sys.argv[1], sys.argv[2], sys.argv[3]
ARMS = {'all': {}, 'noQ4K': {'R36_SKIP': '12'}, 'noQ6K': {'R36_SKIP': '14'}, 'noHead': {'R36_SKIP_HEAD': '1'},
        'noAttn': {'R36_SKIP_ATTN': '1'}, 'rest': {'R36_SKIP': '12,14', 'R36_SKIP_HEAD': '1', 'R36_SKIP_ATTN': '1'}}
now = lambda: datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%SZ')
def run(arm):
    e = dict(os.environ, RUNNER_METAL_TIMING='1', GUARD_MIN_AVAIL='3000', GUARD_MAX_WAIT='120', **ARMS[arm])
    p = subprocess.run(['python3', "guard.py", W + '/runner', '-m', M, '-f', P, '-n', '128',
                        '--temp', '0', '--ignore-eos', '--mlock', '--no-tray'], env=e, cwd=W,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if 'STOPPED' in p.stderr or p.returncode: return None
    rows = [(float(m.group(1)), float(m.group(2))) for l in p.stderr.splitlines()
            for m in [re.search(r'metal-timing n=1 encode=([0-9.]+)ms .*gpu=([0-9.]+)ms', l)] if m][4:]
    return st.median(r[1] for r in rows), st.median(r[0] for r in rows)
res = collections.defaultdict(list)
print('== start', now(), flush=True)
for r in range(3):
    for arm in (list(ARMS) if r % 2 == 0 else list(ARMS)[::-1]):
        v = run(arm); print(f'-- r{r} {arm} {now()} -> {v}', flush=True)
        if v: res[arm].append(v)
print('== end', now())
for a, v in res.items():
    print(f'{a:7s} gpu {[round(x[0],2) for x in v]} median {st.median(x[0] for x in v):.2f} ms | encode median {st.median(x[1] for x in v):.2f} ms')
