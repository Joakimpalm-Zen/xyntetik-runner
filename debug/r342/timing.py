"""R3.42 timing, as written on the item before the run.

One binary (the experiment). Arm metal: no flags. Arm cpu: --gpu off.
Two models, two prompts (R3.60's 129- and 1025-token texts), 128 decoded tokens
with --ignore-eos at temperature 0. One discarded warm-up per cell, then five
rounds of fresh processes, the arm order flipping each round. Start gate: the
1-minute load under 1.5 and memory pressure level 1. Median per cell; the
ratio metal/cpu per cell; kill if any of the 8 ratios is under 1.3.

    python3 debug/r342/timing.py ./runner LFM2.5-1.2B-Instruct-Q4_K_M.gguf LFM2.5-350M-Q4_K_M.gguf

The prompts are debug/r342/p129.txt and p1025.txt (129 and 1025 tokens in
SmolLM2's tokenizer, R3.60's texts; 130 and 1026 in LFM2's). The run of
2026-11-15 on the experiment's head is debug/r342/timing-2026-11-15.log, its
per-run numbers runs.json.
"""
import json, os, pathlib, re, statistics, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
BIN = sys.argv[1]
MODELS = {'1.2b': sys.argv[2], '350m': sys.argv[3]}
PROMPTS = {k: (HERE / ('%s.txt' % k)).read_text() for k in ('p129', 'p1025')}
OUT = pathlib.Path(os.environ.get('TMPDIR', '/tmp')) / 'r342-timing'
OUT.mkdir(parents=True, exist_ok=True)
LINE = re.compile(r'^prompt: (\d+) tok, ([\d.]+) tok/s \| gen: (\d+) tok, ([\d.]+) tok/s', re.M)


def pressure():
    return int(subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                              capture_output=True, text=True).stdout.strip() or 0)


def load1(): return os.getloadavg()[0]


def run(arm, model, prompt):
    p = OUT / ('%s.txt' % prompt)
    p.write_text(PROMPTS[prompt])
    args = [BIN, '-m', MODELS[model], '-c', '4096', '--temp', '0', '-n', '128', '--ignore-eos', '-f', str(p)]
    if arm == 'cpu': args += ['--gpu', 'off']
    r = subprocess.run(args, capture_output=True, timeout=900)
    err = r.stderr.decode('utf-8', 'replace')
    m = LINE.search(err)
    if r.returncode or not m: sys.exit('%s %s %s failed rc %d: %s' % (arm, model, prompt, r.returncode, err[-400:]))
    gpu = 'Metal backend' in err
    if gpu != (arm == 'metal'): sys.exit('%s %s %s: Metal backend line %s' % (arm, model, prompt, gpu))
    thr = re.search(r'\| (\d+) threads \|', err)
    return {'ptok': int(m.group(1)), 'pps': float(m.group(2)), 'gtok': int(m.group(3)), 'gps': float(m.group(4)),
            'threads': int(thr.group(1)) if thr else None}


t0 = time.time()
checks = 0
while load1() >= 1.5 or pressure() != 1:     # the start gate, bounded by 360 checks
    checks += 1
    if checks > 360: sys.exit('the start gate never held (load %.2f, pressure %d)' % (load1(), pressure()))
    time.sleep(10)
print('start %s, gate checks %d, load %.2f, pressure %d' % (time.strftime('%H:%M:%SZ', time.gmtime()), checks, load1(),
                                                             pressure()), flush=True)
cells = [(m, p) for m in MODELS for p in PROMPTS]
for m, p in cells:                            # the discarded warm-up
    for arm in ('metal', 'cpu'): run(arm, m, p)
print('warm-up done %s' % time.strftime('%H:%M:%SZ', time.gmtime()), flush=True)
res = {(m, p, a): [] for m, p in cells for a in ('metal', 'cpu')}
for rnd in range(5):
    print('round %d load %.2f pressure %d %s' % (rnd + 1, load1(), pressure(), time.strftime('%H:%M:%SZ', time.gmtime())),
          flush=True)
    for m, p in cells:
        for arm in (('metal', 'cpu') if rnd % 2 == 0 else ('cpu', 'metal')):
            res[(m, p, arm)].append(run(arm, m, p))
json.dump({'|'.join(k): v for k, v in res.items()}, open(OUT / 'runs.json', 'w'), indent=1)
ratios = []
print('\ncell              tokens  metal tok/s   cpu tok/s   ratio  (medians of 5)')
for m, p in cells:
    M, C = res[(m, p, 'metal')], res[(m, p, 'cpu')]
    for kind, tk, key in (('prompt', 'ptok', 'pps'), ('decode', 'gtok', 'gps')):
        mm, cm = statistics.median(r[key] for r in M), statistics.median(r[key] for r in C)
        ratios.append((mm / cm, m, p, kind))
        print('%-5s %-6s %-6s %5d  %9.2f  %9.2f  %6.3f   metal %s  cpu %s' % (
            m, p, kind, M[0][tk], mm, cm, mm / cm, [r[key] for r in M], [r[key] for r in C]))
print('threads: metal %s, cpu %s' % (sorted({r['threads'] for k, v in res.items() if k[2] == 'metal' for r in v}),
                                     sorted({r['threads'] for k, v in res.items() if k[2] == 'cpu' for r in v})))
lo = min(ratios)
print('\nLOWEST RATIO %.3f (%s %s %s); kill line 1.3: %s' % (lo[0], lo[1], lo[2], lo[3],
                                                          'KILL' if lo[0] < 1.3 else 'PASS'))
print('end %s, load %.2f' % (time.strftime('%H:%M:%SZ', time.gmtime()), load1()))
