"""Public-interface gates for scripts/tool-choice-boundary.py, the unlabeled
tool-choice decision-boundary lane over choice_logprobs."""

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tool-choice-boundary.py"
BANK = ROOT / "scripts" / "tool-choice-boundary-bank.jsonl"

# Published by the bank's author in its MANIFEST.sha256.json (v1.0). The file
# is carried byte-identical so this external value keeps pinning it.
UPSTREAM_BANK_SHA256 = \
    "c7a3099460a7d35c374c148ee73f3de855b1f036a4a2f2704975c930c6d6d084"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bank_is_the_upstream_bank_byte_for_byte():
    data = BANK.read_bytes()
    assert hashlib.sha256(data).hexdigest() == UPSTREAM_BANK_SHA256
    rows = [json.loads(l) for l in data.decode().splitlines() if l.strip()]
    assert len(rows) == 36
    assert len({r["id"] for r in rows}) == 36
    assert {r["family"] for r in rows} == {
        "config_browse", "list_vs_search", "read_vs_search",
        "read_vs_write", "tool_vs_none"}
    assert all(set(r) == {"id", "family", "prompt"} for r in rows)


def _rec(index, alts):
    return {"index": index, "n_legal": len(alts), "coverage": 0.999,
            "alternatives": [{"token": t, "id": i, "prob": 0.0, "logprob": lp}
                             for i, (t, lp) in enumerate(alts)]}


def test_tool_decision_groups_prefix_pieces_under_their_tool():
    m = _load("tcb", SCRIPT)
    recs = [
        _rec(0, [("{", -0.01), (" {", -4.6)]),            # not a tool branch
        _rec(4, [("none", 0.0), ("no", -16.165), ("non", -17.088),
                 ("search", -19.107), ("n", -19.962), ("read", -20.503)]),
    ]
    d = m.tool_decision(recs, m.TOOLS)
    assert d["index"] == 4
    assert d["top1"] == "none"
    # the runner-up is the best alternative that leads to a DIFFERENT tool,
    # not a shorter tokenization of the same one
    assert d["top2"] == "search_files"
    assert d["margin_nat"] == pytest.approx(19.107, abs=1e-6)
    assert d["by_tool"]["none"] == 0.0
    assert d["by_tool"]["read_file"] == pytest.approx(-20.503)
    assert "write_file" not in d["by_tool"]


def test_tool_decision_is_none_when_no_branch_was_probed():
    m = _load("tcb", SCRIPT)
    assert m.tool_decision([_rec(0, [("{", -0.01), (" {", -4.6)])],
                           m.TOOLS) is None


def _study_constants():
    """TOOLS and SYSTEM from make-tooluse-data.py without running it (its
    module level generates data files)."""
    import ast
    src = (ROOT / "scripts" / "make-tooluse-data.py").read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in ("TOOLS", "SYSTEM")
                    for t in n.targets)]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "study", "exec"), ns)
    return ns


def test_prompt_contract_is_the_study_template_verbatim():
    """The lane must send exactly the prompt the published adapters were
    trained against, so its decisions are the study's decisions."""
    m = _load("tcb", SCRIPT)
    study = _study_constants()
    assert m.SYSTEM == study["SYSTEM"]
    assert m.TOOL_SIGNATURES == study["TOOLS"]


def _record(i, fam, cond, adapter, top1, top2, margin, call):
    m = _load("tcb", SCRIPT)
    alts = [(top1.split("_")[0], 0.0), (top2.split("_")[0], -margin)]
    rec = _rec(4, alts)
    return {"record": m.RECORD_VERSION, "id": i, "family": fam,
            "prompt": "p-" + i, "condition": cond, "adapter": adapter,
            "bank_sha256": "b", "full_text": json.dumps({"tool": call}),
            "full_call": {"tool": call, "args": {}}, "n_decisions": 1,
            "all_decisions": [rec], "decision": m.tool_decision([rec], m.TOOLS)}


def test_summary_reports_adapter_disagreement_and_carries_no_labels(tmp_path):
    m = _load("tcb", SCRIPT)
    rows = [
        _record("a", "tool_vs_none", "base", None, "read_file", "search_files", 0.4, "read_file"),
        _record("a", "tool_vs_none", "bf16", "x.gguf", "read_file", "none", 0.4, "read_file"),
        _record("a", "tool_vs_none", "q4", "y.gguf", "none", "read_file", 2.4, "none"),
        _record("b", "config_browse", "base", None, "list_dir", "search_files", 5.0, "list_dir"),
        _record("b", "config_browse", "bf16", "x.gguf", "list_dir", "search_files", 3.0, "list_dir"),
        _record("b", "config_browse", "q4", "y.gguf", "list_dir", "search_files", 4.0, "list_dir"),
    ]
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = m.summarize(m.load_records(path))
    assert s["unlabeled"] is True
    assert s["adapters"] == ["bf16", "q4"] and s["bases"] == ["base"]
    assert [d["id"] for d in s["adapter_disagreements"]] == ["a"]
    assert s["adapter_disagreements"][0]["call_disagrees"] is True
    assert s["adapter_disagreement_families"] == {"tool_vs_none": 1}
    assert s["margins"]["q4"]["n_le_3"] == 1
    assert s["tightest_adapter"] == {"bf16": 2}
    # the lane is unlabeled by construction: nothing in the report may look
    # like a score, so it can never be mistaken for a calibration input
    def keys(x):
        if isinstance(x, dict):
            for k, v in x.items():
                yield k
                yield from keys(v)
        elif isinstance(x, list):
            for v in x:
                yield from keys(v)
    for k in keys(s):
        if k == "unlabeled":
            continue
        for word in ("accuracy", "correct", "gold", "label", "score"):
            assert word not in k.lower(), k
    out = subprocess.run([sys.executable, SCRIPT, "summarize", str(path)],
                         capture_output=True, text=True, check=True).stdout
    assert "UNLABELED" in out and "**a**" in out and "calls DISAGREE" in out


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model():
    path = ROOT / "test.gguf"
    if not path.exists():
        pytest.skip("test.gguf not built")
    return path


def test_run_records_every_prompt_per_condition(runner_bin, model, tmp_path):
    """End to end through the public interface: the script launches the
    runner, probes a bank under one condition, and writes one versioned
    record per prompt with the decision trace, or an explicit None when the
    synthetic model's random logits left no tool piece among the probed
    legal alternatives."""
    m = _load("tcb", SCRIPT)
    bank = tmp_path / "bank.jsonl"
    bank.write_text("\n".join(json.dumps(r) for r in [
        {"id": "t1", "family": "sanity", "prompt": "read notes.txt"},
        {"id": "t2", "family": "sanity", "prompt": "list docs"},
    ]) + "\n")
    # the synthetic model's context is far too small for the study prompt;
    # a custom template is recorded by sha so its records cannot be mistaken
    # for study records
    template = tmp_path / "template.txt"
    template.write_text("T: %s\nJSON:")
    out = tmp_path / "records.jsonl"
    r = subprocess.run([sys.executable, SCRIPT, "run", "--runner", runner_bin,
                        "--model", model, "--condition", "base",
                        "--bank", bank, "--out", out, "--max-tokens", "6",
                        "--probe", "64", "--ready-timeout", "120",
                        "--template", template],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    recs = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert [x["id"] for x in recs] == ["t1", "t2"]
    for x in recs:
        assert x["record"] == m.RECORD_VERSION
        assert x["condition"] == "base" and x["adapter"] is None
        assert x["model"] == "test.gguf"
        assert x["runner_version"].startswith("runner ")
        assert x["tools"] == m.TOOLS
        assert x["template_sha256"] == hashlib.sha256(
            template.read_bytes()).hexdigest()
        assert isinstance(x["all_decisions"], list)
        assert x["n_decisions"] == len(x["all_decisions"])
        assert "decision" in x and "full_call" in x
        assert "correct_id" not in x
    assert "wrote 2 records" in r.stderr
    # summarize accepts what run wrote
    s = subprocess.run([sys.executable, SCRIPT, "summarize", out, "--json"],
                       capture_output=True, text=True, check=True)
    assert json.loads(s.stdout)["records"] == 2
