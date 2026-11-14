import re, sys
t = open(sys.argv[2]).read()
fw = [tuple(map(float, m)) for m in re.findall(r"metal-timing n=1 encode=([\d.]+)ms submit\+wait=([\d.]+)ms gpu=([\d.]+)ms total=([\d.]+)ms", t)][2:]
g = re.search(r"gen: (\d+) tok, ([\d.]+) tok/s", t)
if not g or not fw: print(sys.argv[1], "no data:", t.strip().splitlines()[-1][:120]); sys.exit()
tps = float(g.group(2)); tok = 1000.0 / tps
enc = sum(f[0] for f in fw) / len(fw); gpu = sum(f[2] for f in fw) / len(fw); tot = sum(f[3] for f in fw) / len(fw)
print(f"{sys.argv[1]:26s} {len(fw)} fwd: token {tok:.2f} ms | forward {tot:.2f} (encode {enc:.2f}, gpu {gpu:.2f}, wait-gpu {tot-enc-gpu:.2f}) | outside {tok - tot:.2f} ms = {100 * (tok - tot) / tok:.1f}%")
