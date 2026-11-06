"""R3.30 step 3: the capped LRU's real admission stream for a trace, as file
offsets, for the side load. Usage: adm_list.py MODEL HOTFILE TRACE PT R OUT
Writes one line per upload: offset bytes (three per admitted expert)."""
import collections, json, struct, sys
MODEL, HOT, TRACE, PT, R, OUT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
f = open(MODEL, "rb")
def rd(fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
def rstr():
    return f.read(rd("Q")).decode("utf-8", "replace")
SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def rval(t):
    if t == 8: return rstr()
    if t == 9:
        et, n = rd("I"), rd("Q")
        return [rval(et) for _ in range(n)] if et in (8, 9) else f.read(SZ[et] * n) and None
    return f.read(SZ[t]) if t != 4 else rd("I")
assert f.read(4) == b"GGUF"
ver, nt, nkv = rd("I"), rd("Q"), rd("Q")
align = 32
for _ in range(nkv):
    k = rstr(); t = rd("I"); v = rval(t)
    if k == "general.alignment": align = v
ten = {}
for _ in range(nt):
    n = rstr(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; ty = rd("I"); off = rd("Q")
    ten[n] = (dims, ty, off)
base = (f.tell() + align - 1) // align * align
def slices(l, e):
    out = []
    for nm in ("gate", "up", "down"):
        dims, ty, off = ten[f"blk.{l}.ffn_{nm}_exps.weight"]
        assert ty == 8 and dims[2] == 64, (dims, ty)
        b = dims[0] * dims[1] // 32 * 34
        out.append((base + off + e * b, b))
    return out
assert sum(b for _, b in slices(0, 0)) == 6580224
hot = [tuple(map(int, l.split())) for l in open(HOT) if l.strip() and not l.startswith("#")]
tok = collections.defaultdict(dict)
for line in open(TRACE):
    r = json.loads(line); tok[r["pos"]][r["layer"]] = r["experts"]
slots = collections.OrderedDict((k, 1) for k in hot); ghost = collections.OrderedDict()
adm = []; B = len(hot); nd = 0
for p in sorted(tok):
    if p < PT: continue
    nd += 1; budget = R
    for l in sorted(tok[p]):
        for e in tok[p][l]:
            k = (l, e)
            if k in slots: slots.move_to_end(k); continue
            if k in ghost and budget > 0:
                del ghost[k]; slots.popitem(last=False); slots[k] = 1; budget -= 1; adm.append(k)
            else:
                ghost[k] = 1; ghost.move_to_end(k)
                if len(ghost) > B: ghost.popitem(last=False)
if R == 0:  # the prime list: every expert this trace routes that the hot file lacks
    adm = sorted({(l, e) for p in tok for l in tok[p] for e in tok[p][l]} - set(hot))
with open(OUT, "w") as o:
    for l, e in adm:
        for off, b in slices(l, e): o.write(f"{off} {b}\n")
print(OUT, "admissions", len(adm), "per token", round(len(adm)/nd, 2), "distinct", len(set(adm)))
