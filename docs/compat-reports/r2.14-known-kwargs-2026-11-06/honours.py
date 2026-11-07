"""Which renderers honour each known key, measured on Runner alone, so it covers
every renderer, whether or not its reference template is on this host."""
import importlib.util
import json
import os
import sys

ROOT = os.environ["R214_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sp = importlib.util.spec_from_file_location("tc", os.path.join(ROOT, "scripts", "template-conformance.py"))
tc = importlib.util.module_from_spec(sp)
sp.loader.exec_module(tc)
BIN = sys.argv[1]
KEYS = {"enable_thinking": [True, False], "preserve_thinking": [True, False],
        "reasoning_effort": ["low", "medium", "high"], "reasoning_strength": ["low", "medium", "high"]}
H = [{"role": "user", "content": "First?"},
     {"role": "assistant", "content": "One.", "reasoning_content": "first thought"},
     {"role": "user", "content": "Second?"},
     {"role": "assistant", "content": "Two.", "reasoning_content": "second thought"},
     {"role": "user", "content": "Third?"}]
names = sorted({f.runner for f in tc.FAMILIES.values() if f.runner})
out = {k: [] for k in KEYS}
errs = {}
for t in names:
    cases = [{"id": "base", "template": t, "request": {"model": "x", "messages": H}}]
    for k, vs in KEYS.items():
        for v in vs:
            cases.append({"id": f"{k}={v}", "template": t,
                          "request": {"model": "x", "messages": H, "chat_template_kwargs": {k: v}}})
    g = tc.render_runner(BIN, cases)
    base = g["base"][0]
    if g["base"][1]:
        errs[t] = g["base"][1][:70]
        continue
    for k, vs in KEYS.items():
        moved = any(g[f"{k}={v}"][1] or g[f"{k}={v}"][0] != base for v in vs)
        if moved:
            out[k].append(t)
print(len(names), "renderers:", names)
for k, v in out.items():
    print(f"\n{k} ({len(v)}): {v}")
print("\nbase errors:", errs)
print("not honouring enable_thinking:", [n for n in names if n not in out["enable_thinking"] and n not in errs])
