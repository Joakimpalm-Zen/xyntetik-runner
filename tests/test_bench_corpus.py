"""The workload and shape corpus (plan P2) holds the shapes it promises, and
its driver produces records that say what they measured.

scripts/bench-corpus.json is the fixed set of cases a performance change is
measured on: one-row decode, prefill on both sides of tile boundaries, long
prompts, speculative verification at every width 3 to 9 (odd ones too), a
partial and a resident split, a tool-result turn that reuses its call's
prefix, one and two requests in flight, and the idle lifecycle. The driver
runs them through the existing tools; here it runs a few on the fixture.
"""
import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "scripts" / "bench-corpus.json").read_text(encoding="utf-8"))


def test_the_corpus_holds_every_shape_it_promises():
    ids = [c["id"] for c in MANIFEST["cases"]]
    assert len(ids) == len(set(ids)), "case ids repeat"
    for c in MANIFEST["cases"]:
        assert c["kind"] in ("bench", "server", "idle"), c
        assert c["measure"] and c["why"], c
        if c["kind"] == "bench":
            assert c["prompt_tokens"] > 0 and c["n_predict"] > 0, c
    assert "decode-1row" in ids and "idle-lifecycle" in ids and "tool-result-reuse" in ids
    for n in (31, 32, 33, 511, 512, 513):
        assert f"prefill-{n}" in ids
    widths = sorted(c["draft_k"] + 1 for c in MANIFEST["cases"] if c.get("draft") == "self")
    assert widths[0] == 3 and widths[-1] == 9 and any(w % 2 for w in widths), widths
    assert {"split-partial", "split-resident", "concurrent-1", "concurrent-2"} <= set(ids)


def test_the_driver_records_what_it_measured(tmp_path):
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner not built")
    model = tmp_path / "m.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(model)],
                   check=True, stdout=subprocess.DEVNULL)
    out = tmp_path / "r.json"
    subprocess.run([sys.executable, ROOT / "scripts/bench-corpus.py", "--runner", str(exe),
                    "--model", str(model), "--ctx", "4096", "--repeat", "2",
                    "--cases", "^(decode-1row|prefill-33|verify-k3|concurrent-1|tool-result-reuse)$",
                    "--out", str(out)], check=True, timeout=600, cwd=ROOT)
    rec = json.loads(out.read_text())
    assert rec["model"]["size"] == model.stat().st_size and rec["runner"]["version"]
    rows = {(r["id"], r["attempt"]): r for r in rec["cases"]}
    assert ("decode-1row", "first") in rows and ("decode-1row", "repeat") in rows
    for (cid, _), r in rows.items():
        res = r["result"]
        assert "error" not in res, (cid, res)
        if cid in ("decode-1row", "prefill-33", "verify-k3"):
            assert res["route"]["path"] == "solo" and res["gen_tok_s"] > 0, (cid, res)
            assert 0 < res["prompt_tokens"] <= r["prompt_tokens_target"], (cid, res)
    # the model drafting for itself, greedy: every round verifies the full width
    spec = rows[("verify-k3", "first")]["result"]["spec"]
    assert re.search(r"\(4\.00 tok/round\)", spec), spec
    assert rows[("tool-result-reuse", "first")]["result"]["prompt_cached_tokens"] > 0
    assert rows[("concurrent-1", "first")]["result"]["tok_s"] > 0
