"""R2.14: for every family and every known key, who changes when the key is set?

Renders one history with the key absent and with each value, through the family's
reference template and through Runner's real request path, and compares the two
deltas. Reference moves and Runner does not: the key is silently dropped. Runner
moves and the reference does not: Runner invents a difference. Reads
scripts/template-conformance.py, changes nothing in the tree.

Usage: audit.py ORACLE_DIR RENDERER OUT.json
"""
import importlib.util
import json
import os
import sys

ROOT = os.environ["R214_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sp = importlib.util.spec_from_file_location("tc", os.path.join(ROOT, "scripts", "template-conformance.py"))
tc = importlib.util.module_from_spec(sp)
sp.loader.exec_module(tc)

ODIR, BIN, OUT = sys.argv[1:4]
KEYS = {
    "enable_thinking": [True, False],
    "preserve_thinking": [True, False],
    "reasoning_effort": ["low", "medium", "high", "xhigh"],
    "reasoning_strength": ["low", "medium", "high"],
    "bare_recipients": [True, False],
}
HISTORY = [
    {"role": "user", "content": "First question?"},
    {"role": "assistant", "content": "First answer.", "reasoning_content": "first thought"},
    {"role": "user", "content": "Second question?"},
    {"role": "assistant", "content": "Second answer.", "reasoning_content": "second thought"},
    {"role": "user", "content": "Third question?"},
]


def ref(env, tmpl, meta, kw, add_gen=True):
    args = dict(messages=HISTORY, add_generation_prompt=add_gen, bos_token=meta.get("bos_token") or "",
                eos_token=meta.get("eos_token") or "")
    args.update(kw)
    try:
        return env.from_string(tmpl).render(**args), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:90]}"


env = tc.jinja_env()
rows = {}
not_checked = {}
for name, fam in tc.FAMILIES.items():
    if not fam.source or fam.source[0] not in ("hf", "gguf"):
        not_checked[name] = f"no jinja source ({fam.source})"
        continue
    path = os.path.join(ODIR, name + ".jinja")
    if not os.path.exists(path):
        ok, err = tc.fetch_oracle(name, fam, ODIR)
        if not ok:
            not_checked[name] = (err or "")[:110]
            continue
    tmpl = open(path, encoding="utf-8").read()
    meta = json.load(open(os.path.join(ODIR, name + ".json"), encoding="utf-8"))
    cases = [{"id": "base", "template": fam.runner, "chat_template": tmpl,
              "request": {"model": "x", "messages": HISTORY}}]
    want = {}
    for k, vals in KEYS.items():
        for v in vals:
            cid = f"{k}={v}"
            want[cid] = (k, v)
            cases.append({"id": cid, "template": fam.runner, "chat_template": tmpl,
                          "request": {"model": "x", "messages": HISTORY, "chat_template_kwargs": {k: v}}})
    try:
        got = tc.render_runner(BIN, cases)
    except SystemExit as e:
        not_checked[name] = f"renderer: {str(e)[:100]}"
        continue
    rbase, rerr = ref(env, tmpl, meta, {})
    mbase, merr = got["base"][0], got["base"][1]
    res = {"base_ref_error": rerr, "base_runner_error": merr, "keys": {}}
    if rerr or merr:
        rows[name] = res
        continue
    for cid, (k, v) in want.items():
        r, re_ = ref(env, tmpl, meta, {k: v})
        m, me = got[cid][0], got[cid][1]
        ref_moved = r != rbase if r is not None else None
        run_moved = m != mbase if not me else None
        res["keys"][cid] = {"ref_moved": ref_moved, "run_moved": run_moved, "ref_error": re_, "run_error": me,
                            "same_text": r == m if (r is not None and not me) else None}
    # add_generation_prompt false
    r, re_ = ref(env, tmpl, meta, {}, add_gen=False)
    res["add_gen_false_ref_moved"] = (r != rbase) if r is not None else None
    res["add_gen_false_ref_error"] = re_
    rows[name] = res

json.dump({"rows": rows, "not_checked": not_checked}, open(OUT, "w"), indent=1)
print(len(rows), "families read,", len(not_checked), "not checked")
for n, why in not_checked.items():
    print("  NOT CHECKED", n, why[:100])

silent, invented, refuses = [], [], []
for n, r in rows.items():
    for cid, d in r["keys"].items():
        if d["ref_moved"] and d["run_moved"] is False:
            silent.append((n, cid))
        elif d["ref_moved"] is False and d["run_moved"]:
            invented.append((n, cid))
        elif d["ref_moved"] is None and d["run_error"] is None:
            refuses.append((n, cid, (d["ref_error"] or "")[:60]))
print("\nreference moves, Runner does not (silently dropped):")
for s in silent:
    print("  ", *s)
print("\nRunner moves, reference does not (invented):")
for s in invented:
    print("  ", *s)
print("\nreference raises, Runner renders:")
for s in refuses:
    print("  ", *s)
g = [n for n, r in rows.items() if r.get("add_gen_false_ref_moved")]
print("\nadd_generation_prompt false moves the reference on", len(g), "of", len(rows), "families")
print("base render failed:", [(n, r["base_ref_error"] or r["base_runner_error"]) for n, r in rows.items()
                              if r["base_ref_error"] or r["base_runner_error"]])
