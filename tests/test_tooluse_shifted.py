"""Tests for the tooluse-shifted evaluation (R8.4).

Catalog, the two prompt sets, the frozen labels, the scorer's four legs
(mocked endpoints), the paired comparison and the calibration certificate.
"""
import hashlib
import importlib.util
import io
import json
import math
import os
import urllib.error
import urllib.request

import pytest

EVAL_DIR = "evals/tooluse-shifted"
SETS = {"v1": ("set-v1.jsonl", "LABELS.sha256"), "v2": ("set-v2.jsonl", "LABELS-v2.sha256")}


def load_catalog():
    with open(os.path.join(EVAL_DIR, "catalog-v1.json")) as f:
        return json.load(f)


def load_set(version="v1"):
    rows = []
    with open(os.path.join(EVAL_DIR, SETS[version][0])) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_frozen_hash(version="v1"):
    with open(os.path.join(EVAL_DIR, SETS[version][1])) as f:
        return f.read().strip()


def canonical(rows, version):
    labels = []
    for row in rows:
        if version == "v1":
            labels.append([row["id"], row["gold_tool"], row["gold_args"]])
        else:
            labels.append([row["id"], row["gold_tool"], row["gold_args"],
                           row.get("also_ok", []), row.get("gold_next")])
    labels.sort(key=lambda x: x[0])
    return "\n".join(json.dumps(l, ensure_ascii=False) for l in labels)


def compute_labels_hash(version="v1"):
    return hashlib.sha256(canonical(load_set(version), version).encode()).hexdigest()


def load_script(name):
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", name)
    spec = importlib.util.spec_from_file_location(name.replace("-", "_").replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_scorer():
    return load_script("eval-tooluse-shifted.py")


class TestCatalog:
    def test_catalog_has_tools(self):
        assert len(load_catalog()) >= 20

    def test_catalog_has_none_tool(self):
        assert "none" in [t["name"] for t in load_catalog()]

    def test_catalog_tool_structure(self):
        for t in load_catalog():
            assert {"name", "description", "args", "required"} <= set(t)
            assert set(t["required"]) <= set(t["args"])


@pytest.mark.parametrize("version", ["v1", "v2"])
class TestSets:
    """Both sets: structure, labels against the catalog, frozen hash."""

    def test_minimum_prompts(self, version):
        assert len(load_set(version)) >= {"v1": 150, "v2": 250}[version]

    def test_required_fields_and_unique_ids(self, version):
        rows = load_set(version)
        required = {"id", "prompt", "gold_tool", "gold_args", "category", "source"}
        for row in rows:
            assert required <= set(row), row.get("id")
            assert row["source"] == "handwritten" or row["source"].startswith("bank-copy:")
        ids = [r["id"] for r in rows]
        assert len(set(ids)) == len(ids)
        assert len({r["prompt"] for r in rows}) == len(rows), "duplicate prompt"

    def test_gold_tools_and_args_in_catalog(self, version):
        tools = {t["name"]: t for t in load_catalog()}
        for row in load_set(version):
            t = tools[row["gold_tool"]]
            if row["gold_tool"] == "none":
                assert row["gold_args"] == {}
            else:
                assert set(row["gold_args"]) <= set(t["args"]), row["id"]
                if version == "v2":
                    assert set(row["gold_args"]) == set(t["required"]), row["id"]
                    for k, v in row["gold_args"].items():
                        want = {"string": str, "integer": int}[t["args"][k]["type"]]
                        assert isinstance(v, want) and not isinstance(v, bool), (row["id"], k)

    def test_categories(self, version):
        by = {}
        for r in load_set(version):
            by.setdefault(r["category"], []).append(r["id"])
        expected = {"paraphrase", "multi_intent", "underspecified", "near_miss", "none"}
        if version == "v2":
            expected |= {"arg_shift"}
        assert set(by) == expected
        for cat, ids in by.items():
            assert len(ids) >= 20, cat

    def test_labels_hash_matches(self, version):
        assert compute_labels_hash(version) == load_frozen_hash(version)
        assert len(load_frozen_hash(version)) == 64

    def test_scorer_agrees_on_the_hash(self, version):
        mod = load_scorer()
        rows, h = mod.load_set(version)
        assert h == load_frozen_hash(version)
        assert all("gold" in r for r in rows)


class TestSetV2Labels:
    """The v2 labelling rules: synonyms share a signature, second calls are
    valid, directory defaults are '.', file paths never default."""

    def test_also_ok_shares_the_signature(self):
        tools = {t["name"]: t for t in load_catalog()}
        for row in load_set("v2"):
            for alt in row.get("also_ok", []):
                assert set(tools[alt]["required"]) == set(tools[row["gold_tool"]]["required"]), row["id"]

    def test_multi_intent_rows_carry_a_valid_second_call(self):
        tools = {t["name"]: t for t in load_catalog()}
        rows = load_set("v2")
        for row in rows:
            if row["category"] == "multi_intent":
                nxt = row["gold_next"]
                assert nxt["tool"] in tools, row["id"]
                assert set(nxt["args"]) == set(tools[nxt["tool"]]["required"]), row["id"]
            else:
                assert "gold_next" not in row, row["id"]

    def test_underspecified_and_none_are_labelled_none(self):
        for row in load_set("v2"):
            if row["category"] in ("underspecified", "none"):
                assert row["gold_tool"] == "none", row["id"]

    def test_frozen_hash_detects_a_label_change(self):
        rows = load_set("v2")
        rows[0] = dict(rows[0], gold_tool="modified_tool")
        h = hashlib.sha256(canonical(rows, "v2").encode()).hexdigest()
        assert h != load_frozen_hash("v2")

    def test_frozen_hash_detects_a_synonym_or_next_change(self):
        rows = load_set("v2")
        alt = [dict(r) for r in rows]
        alt[0]["also_ok"] = ["grep_text"]
        assert hashlib.sha256(canonical(alt, "v2").encode()).hexdigest() != load_frozen_hash("v2")
        alt = [dict(r) for r in rows]
        m = next(i for i, r in enumerate(alt) if r["category"] == "multi_intent")
        alt[m]["gold_next"] = {"tool": "none", "args": {}} if alt[m]["gold_next"]["tool"] != "none" \
            else {"tool": "read_file", "args": {"path": "x"}}
        assert hashlib.sha256(canonical(alt, "v2").encode()).hexdigest() != load_frozen_hash("v2")


class TestFieldScoring:
    def test_norm_value_paths_and_integers(self):
        mod = load_scorer()
        assert mod.norm_value("./", ".") == "."
        assert mod.norm_value("src/", "src") == "src"
        assert mod.norm_value("./src", "src") == "src"
        assert mod.norm_value("10", 10) == 10
        assert mod.norm_value(10.0, 10) == 10
        assert mod.norm_value("ten", 10) == "ten"
        assert mod.norm_value(" x.py ", "x.py") == "x.py"
        assert mod.norm_value(True, 1) is True   # a boolean is never an integer here

    def test_score_args_counts_fields(self):
        mod = load_scorer()
        gold = {"path": "main.py", "start": 40, "end": 60}
        req = {"path", "start", "end"}
        s = mod.score_args({"path": "main.py", "start": "40", "end": 61}, gold, req)
        assert (s["keys_ok"], s["fields_total"], s["fields_ok"], s["exact"]) == (True, 3, 2, False)
        s = mod.score_args({"path": "main.py", "start": 40, "end": 60, "extra": 1}, gold, req)
        assert s["keys_ok"] is False and s["fields_ok"] == 3 and s["exact"] is False
        s = mod.score_args({"path": "main.py", "start": 40, "end": 60}, gold, req)
        assert s["exact"] is True
        assert mod.score_args("junk", gold, req)["fields_ok"] == 0

    def test_score_call_synonyms_and_none(self):
        mod = load_scorer()
        schemas = {"find_files": {"pattern", "path"}, "search_files": {"pattern", "path"}, "none": set()}
        gold = {"tool": "find_files", "args": {"pattern": "*.py", "path": "."}, "also_ok": ["search_files"]}
        v = mod.score_call("search_files", {"pattern": "*.py", "path": "./"}, gold, schemas)
        assert v["tool_ok"] and v["exact"]
        v = mod.score_call("grep_text", {"pattern": "*.py", "path": "."}, gold, schemas)
        assert not v["tool_ok"] and not v["exact"]
        none = {"tool": "none", "args": {}, "also_ok": []}
        assert mod.score_call(None, None, none, schemas)["exact"]
        assert mod.score_call("none", {}, none, schemas)["exact"]
        assert not mod.score_call("none", {"x": 1}, none, schemas)["exact"]
        assert not mod.score_call("read_file", {"path": "x"}, none, schemas)["tool_ok"]

    def test_limit_rows_takes_the_first_of_each_category(self):
        mod = load_scorer()
        rows = [{"id": i, "category": c} for i, c in enumerate("aabbbc")]
        assert [r["id"] for r in mod.limit_rows(rows, 2)] == [0, 1, 2, 3, 5]


class R(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestServerCommand:
    def _capture(self, monkeypatch, mod):
        seen = []

        class P:
            def __init__(self, cmd, **kw):
                seen.append(cmd)

        monkeypatch.setattr(mod.subprocess, "Popen", P)
        return seen

    def test_serve_without_adapter(self, monkeypatch):
        mod = load_scorer()
        seen = self._capture(monkeypatch, mod)

        class Args:
            runner, model, threads, lora_scale, gpu = "./runner", "m.gguf", 4, 1.0, "off"

        mod.serve(Args(), None, 9000)
        assert "--lora" not in seen[0] and "--lora-scale" not in seen[0]
        assert seen[0][:3] == ["./runner", "-m", "m.gguf"]
        assert seen[0][seen[0].index("--gpu") + 1] == "off"

    def test_serve_with_adapter_passes_the_scale_and_gpu(self, monkeypatch):
        mod = load_scorer()
        seen = self._capture(monkeypatch, mod)

        class Args:
            runner, model, threads, lora_scale, gpu = "./runner", "m.gguf", 4, 0.5, "auto"

        mod.serve(Args(), "a.gguf", 9000)
        assert seen[0][seen[0].index("--lora") + 1] == "a.gguf"
        assert seen[0][seen[0].index("--lora-scale") + 1] == "0.5"
        assert seen[0][seen[0].index("--gpu") + 1] == "auto"


class TestNativeLeg:
    """The chat leg names the served model and never scores a refusal as a match."""

    CATALOG = [{"name": "read_file", "args": {"path": {"type": "string"}}, "required": ["path"]},
               {"name": "none", "args": {}, "required": []}]
    ROWS = [{"id": "a", "category": "none", "prompt": "hello", "gold": {"tool": "none", "args": {}}},
            {"id": "b", "category": "paraphrase", "prompt": "open x", "gold": {"tool": "read_file", "args": {"path": "x"}}}]

    def test_served_model_id_reads_v1_models(self, monkeypatch):
        mod = load_scorer()
        monkeypatch.setattr(mod.urllib.request, "urlopen",
                            lambda url, timeout=0: R(json.dumps({"data": [{"id": "served-1"}]}).encode()))
        assert mod.served_model_id(1) == "served-1"

    def test_refused_requests_are_failures_not_none_matches(self, monkeypatch):
        mod = load_scorer()
        bodies = []

        def refuse(req, timeout=0):
            bodies.append(json.loads(req.data))
            raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, io.BytesIO(b'{"error":"model_not_found"}'))

        monkeypatch.setattr(mod.urllib.request, "urlopen", refuse)
        out = mod.score_native_leg(self.ROWS, 1, self.CATALOG, "served-1")
        assert all(b["model"] == "served-1" for b in bodies)
        assert all(b["enable_thinking"] is False and b["max_tokens"] >= 256 for b in bodies)
        assert out["refusals"] == 2
        assert out["tool_ok"] == 0 and out["exact_match"] == 0
        assert all(r["refusal"] for r in out["rows"])

    def test_a_none_answer_matches_only_when_the_request_succeeded(self, monkeypatch):
        mod = load_scorer()
        monkeypatch.setattr(mod.urllib.request, "urlopen",
                            lambda req, timeout=0: R(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]}).encode()))
        out = mod.score_native_leg(self.ROWS[:1], 1, self.CATALOG, "served-1")
        assert out["refusals"] == 0 and out["tool_ok"] == 1 and out["empty_outputs"] == 0
        assert out["rows"][0]["finish_reason"] == "stop" and out["rows"][0]["content_head"] == "hi"

    def test_an_empty_answer_is_counted(self, monkeypatch):
        mod = load_scorer()
        monkeypatch.setattr(mod.urllib.request, "urlopen",
                            lambda req, timeout=0: R(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}).encode()))
        out = mod.score_native_leg(self.ROWS[1:], 1, self.CATALOG, "served-1")
        assert out["empty_outputs"] == 1 and out["tool_ok"] == 0

    def test_a_call_in_the_message_is_scored(self, monkeypatch):
        mod = load_scorer()
        resp = {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
            {"function": {"name": "read_file", "arguments": "{\"path\": \"x\"}"}}]}}]}
        monkeypatch.setattr(mod.urllib.request, "urlopen", lambda req, timeout=0: R(json.dumps(resp).encode()))
        out = mod.score_native_leg(self.ROWS[1:], 1, self.CATALOG, "served-1")
        assert out["tool_ok"] == 1 and out["exact_match"] == 1
        assert out["fields_ok"] == 1 and out["fields_total"] == 1


class TestRawLeg:
    CATALOG = [{"name": "read_file", "args": {"path": {"type": "string"}}, "required": ["path"]},
               {"name": "none", "args": {}, "required": []}]

    def test_unparsed_output_is_wrong_even_for_none(self, monkeypatch):
        mod = load_scorer()
        monkeypatch.setattr(mod.urllib.request, "urlopen",
                            lambda req, timeout=0: R(json.dumps({"choices": [{"text": "no json"}]}).encode()))
        rows = [{"id": "a", "category": "none", "prompt": "hello", "gold": {"tool": "none", "args": {}}}]
        out = mod.score_raw_leg(rows, 1, mod.render_system_prompt(self.CATALOG), self.CATALOG,
                                {"read_file": {"path"}, "none": set()})
        assert out["json_parses"] == 0 and out["exact_match"] == 0 and out["right_tool"] == 0

    def test_empty_output_counted(self, monkeypatch):
        mod = load_scorer()
        monkeypatch.setattr(mod.urllib.request, "urlopen",
                            lambda req, timeout=0: R(json.dumps({"choices": [{"text": ""}]}).encode()))
        rows = [{"id": "a", "category": "x", "prompt": "open x", "gold": {"tool": "read_file", "args": {"path": "x"}}}]
        out = mod.score_raw_leg(rows, 1, mod.render_system_prompt(self.CATALOG), self.CATALOG,
                                {"read_file": {"path"}, "none": set()})
        assert out["empty_outputs"] == 1 and out["refusals"] == 0


class TestDecideLeg:
    CATALOG = [{"name": "read_file", "args": {}, "required": ["path"]},
               {"name": "cat_file", "args": {}, "required": ["path"]},
               {"name": "grep_text", "args": {}, "required": []},
               {"name": "none", "args": {}, "required": []}]
    ROWS = [{"id": "a", "category": "paraphrase", "prompt": "open x",
             "gold": {"tool": "read_file", "args": {"path": "x"}, "also_ok": ["cat_file"]}},
            {"id": "b", "category": "none", "prompt": "sing",
             "gold": {"tool": "none", "args": {}, "also_ok": []}}]

    def _serve(self, monkeypatch, mod, probs_by_id):
        bodies = []

        def fake(req, timeout=0):
            body = json.loads(req.data)
            bodies.append(body)
            q = body["questions"][0]
            p = probs_by_id[q["id"]]
            return R(json.dumps({"decisions": [{"id": q["id"], "options": q["options"], "probs": p,
                                                "logprobs": [math.log(max(x, 1e-12)) for x in p],
                                                "argmax": p.index(max(p)), "n_tokens": [1] * len(p)}]}).encode())

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake)
        return bodies

    def test_state_rendering_and_acceptable_mass(self, monkeypatch):
        mod = load_scorer()
        bodies = self._serve(monkeypatch, mod, {"a": [0.5, 0.3, 0.2, 0.0], "b": [0.6, 0.1, 0.1, 0.2]})
        sp = mod.render_system_prompt(self.CATALOG)
        out = mod.score_decide_leg(self.ROWS, 1, sp, self.CATALOG, "served-1", target=0.9)
        assert all(b["rendering"] == "continuation-v1" and b["state"].endswith('{"tool": "') for b in bodies)
        assert bodies[0]["questions"][0]["options"] == ["read_file", "cat_file", "grep_text", "none"]
        a, b = out["rows"]
        assert a["top_ok"] and abs(a["p_acceptable"] - 0.8) < 1e-9
        assert abs(a["nll"] - (-math.log(0.8))) < 1e-9
        assert not b["top_ok"] and abs(b["p_acceptable"] - 0.2) < 1e-9
        assert out["top1_ok"] == 1 and out["refusals"] == 0
        assert out["certificate"]["target"] == 0.9
        # the most confident decision (b, 0.6) is wrong, so no slice certifies
        assert out["certificate"]["threshold"] is None
        assert out["by_category"]["paraphrase"]["top_ok"] == 1

    def test_refused_rows_are_counted_not_scored(self, monkeypatch):
        mod = load_scorer()

        def refuse(req, timeout=0):
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b'{"error":"x"}'))

        monkeypatch.setattr(mod.urllib.request, "urlopen", refuse)
        out = mod.score_decide_leg(self.ROWS, 1, mod.render_system_prompt(self.CATALOG), self.CATALOG, "served-1")
        assert out["refusals"] == 2 and out["top1_ok"] == 0 and out["mean_nll"] is None

    def test_decisions_file_for_cl_calibration(self, monkeypatch, tmp_path):
        mod = load_scorer()
        self._serve(monkeypatch, mod, {"a": [0.5, 0.3, 0.2, 0.0], "b": [0.1, 0.1, 0.1, 0.7]})
        out = mod.score_decide_leg(self.ROWS, 1, mod.render_system_prompt(self.CATALOG), self.CATALOG, "served-1")
        p = tmp_path / "d.jsonl"
        mod.write_decisions(out, str(p))
        cal = load_script("cl-calibration.py")
        with open(p) as f:
            decisions, skipped = cal.read_decisions(f)
        assert skipped == 0 and len(decisions) == 2
        assert [d[1] for d in decisions] == [True, True]   # synonym mass counts as the truth
        s = cal.summarize(decisions)
        assert s["accuracy"] == 1.0 and 0 <= s["ece"] <= 1


class TestNextLeg:
    CATALOG = [{"name": "read_file", "args": {"path": {"type": "string"}}, "required": ["path"]},
               {"name": "append_file", "args": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"]},
               {"name": "none", "args": {}, "required": []}]
    ROWS = [{"id": "m1", "category": "multi_intent", "prompt": "read notes.txt then append 'x' to it",
             "gold": {"tool": "read_file", "args": {"path": "notes.txt"}, "also_ok": ["cat_file"],
                      "next": {"tool": "append_file", "args": {"path": "notes.txt", "content": "x"}}}},
            {"id": "m2", "category": "multi_intent", "prompt": "read notes.txt and summarise it",
             "gold": {"tool": "read_file", "args": {"path": "notes.txt"}, "also_ok": [],
                      "next": {"tool": "none", "args": {}}}},
            {"id": "p1", "category": "paraphrase", "prompt": "open x",
             "gold": {"tool": "read_file", "args": {"path": "x"}, "also_ok": [], "next": None}}]

    def test_teacher_forced_transcript_and_scoring(self, monkeypatch):
        mod = load_scorer()
        bodies = []

        def fake(req, timeout=0):
            body = json.loads(req.data)
            bodies.append(body)
            if body["messages"][0]["content"].endswith("to it"):
                resp = {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
                    {"function": {"name": "append_file", "arguments": "{\"path\": \"notes.txt\", \"content\": \"x\"}"}}]}}]}
            else:
                resp = {"choices": [{"finish_reason": "stop", "message": {"content": "Summary: three lines."}}]}
            return R(json.dumps(resp).encode())

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake)
        out = mod.score_next_leg(self.ROWS, 1, self.CATALOG, "served-1")
        assert out["n"] == 2 and len(bodies) == 2          # the paraphrase row has no second call
        m = bodies[0]["messages"]
        assert [x["role"] for x in m] == ["user", "assistant", "tool"]
        assert m[1]["tool_calls"][0]["function"]["name"] == "read_file"
        assert json.loads(m[1]["tool_calls"][0]["function"]["arguments"]) == {"path": "notes.txt"}
        assert m[2]["tool_call_id"] == m[1]["tool_calls"][0]["id"]
        assert "choice_logprobs" not in bodies[0]
        assert out["tool_ok"] == 2 and out["exact"] == 2 and out["fields_ok"] == 2


class TestCompare:
    def _record(self, name, raw_exact, nll):
        return {"adapter_file": name, "set_version": "v2", "model_sha256": "m" * 64,
                "adapter_sha256": None if name is None else "a" * 64, "host": "h", "threads": 8, "gpu": "off",
                "raw_leg": {"rows": [{"id": f"r{i}", "category": "c", "exact": e} for i, e in enumerate(raw_exact)]},
                "decide_leg": {"rows": [{"id": f"r{i}", "category": "c", "nll": v} for i, v in enumerate(nll)]}}

    def test_paired_counts_and_bootstrap_direction(self):
        cmpm = load_script("tooluse-shifted-compare.py")
        n = 60
        base = self._record(None, [i % 3 == 0 for i in range(n)], [1.0 + (i % 5) * 0.1 for i in range(n)])
        good = self._record("good.gguf", [i % 3 != 1 for i in range(n)], [0.5 + (i % 5) * 0.1 for i in range(n)])
        same = self._record("same.gguf", [i % 3 == 0 for i in range(n)], [1.0 + (i % 5) * 0.1 for i in range(n)])
        for r in (base, good, same):
            r["_name"] = r["adapter_file"] or "base"
        c = cmpm.compare(base, good, seed=1, resamples=500)
        raw = c["legs"]["raw_leg"]
        assert raw["n"] == n and raw["b_only"] == 20 and raw["a_only"] == 0 and raw["z"] < -2
        d = c["legs"]["decide_leg"]
        assert d["mean_diff"] > 0 and d["excludes_zero"] and d["b_better"] == n
        assert cmpm.apart(c) and cmpm.apart_any(c) and cmpm.direction(c) == "B"
        c2 = cmpm.compare(base, same, seed=1, resamples=500)
        assert c2["legs"]["raw_leg"]["z"] == 0 and not c2["legs"]["decide_leg"]["excludes_zero"]
        assert not cmpm.apart(c2) and not cmpm.apart_any(c2) and cmpm.direction(c2) == "none"
        text = cmpm.render(base, [good, same], [c, c2], [cmpm.compare(good, same, 1, 500)])
        assert "R8.4.3 as written" in text and "as an instrument question" in text
        assert "NOT PASSED" in text.split("## Verdict")[1]   # `same` is not told apart from base
        text1 = cmpm.render(base, [good], [c], [])
        assert "adapters clearly ABOVE the base (raw z <= -2 and decide CI below zero): good.gguf" in text1


class TestCalibrationCertificate:
    def test_certificate_is_a_contiguous_top_slice(self):
        cal = load_script("cl-calibration.py")
        d = [(0.95, True, 0), (0.9, True, 0), (0.85, False, 0), (0.8, True, 0), (0.5, False, 0)]
        c = cal.certificate(d, 0.9)
        assert c["threshold"] == 0.9 and c["n_covered"] == 2 and c["coverage"] == 0.4
        c = cal.certificate(d, 0.6)
        # 0.85 fails (2/3 = 0.667 >= 0.6 passes), 0.8 -> 3/4, 0.5 -> 3/5 = 0.6 passes
        assert c["threshold"] == 0.5 and c["coverage"] == 1.0
        assert cal.certificate([(0.9, False, 0)], 0.5)["threshold"] is None
        assert cal.certificate([], 0.5)["threshold"] is None

    def test_summarize_ece_of_a_perfect_and_an_overconfident_set(self):
        cal = load_script("cl-calibration.py")
        perfect = [(1.0, True, 0.0)] * 10
        assert cal.summarize(perfect)["ece"] == 0.0
        over = [(0.99, False, 1.0)] * 10
        s = cal.summarize(over)
        assert abs(s["ece"] - 0.99) < 1e-9 and s["accuracy"] == 0.0

    def test_read_decisions_accepts_a_set(self):
        cal = load_script("cl-calibration.py")
        lines = [json.dumps({"alternatives": [{"id": 1, "prob": 0.6}, {"id": 2, "prob": 0.4}],
                             "correct_id": 2, "acceptable_ids": [1, 2]}),
                 json.dumps({"alternatives": [{"id": 1, "prob": 0.6}, {"id": 2, "prob": 0.4}],
                             "correct_id": 2}),
                 "not json"]
        d, skipped = cal.read_decisions(io.StringIO("\n".join(lines)))
        assert skipped == 1 and [x[1] for x in d] == [True, False]
        assert abs(d[0][2] - 0.0) < 1e-9 and abs(d[1][2] - (0.36 + 0.36)) < 1e-9


class TestRescore:
    def test_path_spelling_is_canonical_on_both_sides(self):
        mod = load_scorer()
        s = mod.score_args({"path": "./logs/app.log", "n": 100}, {"path": "./logs/app.log", "n": 100}, {"path", "n"})
        assert s["exact"]
        s = mod.score_args({"path": "logs/app.log", "n": 100}, {"path": "./logs/app.log", "n": 100}, {"path", "n"})
        assert s["exact"]

    def test_next_gold_takes_the_synonym_rule(self):
        mod = load_scorer()
        row = {"gold": {"tool": "run_pytest", "args": {}, "next": {"tool": "read_file", "args": {"path": "x"}}}}
        assert mod.next_gold(row)["also_ok"] == ["cat_file"]
        rows = load_set("v2")
        for r in rows:
            assert r.get("also_ok", []) == mod.SYNONYMS.get(r["gold_tool"], []), r["id"]

    def test_rescore_rebuilds_verdicts_from_stored_outputs(self, tmp_path):
        mod = load_scorer()
        catalog, schemas = mod.load_catalog(os.path.join(EVAL_DIR, "catalog-v1.json"))
        rows, h = mod.load_set("v2")
        by_id = {r["id"]: r for r in rows}
        a = by_id["arg_shift_28"]          # head of ./logs/app.log, 100 lines
        m = next(r for r in rows if r["category"] == "multi_intent" and r["gold_next"]["tool"] == "read_file")
        record = {
            "set_version": "v2", "labels_sha256": h,
            "raw_leg": {"n": 1, "rows": [{"id": a["id"], "category": a["category"], "prompt": a["prompt"],
                                          "output": '{"tool": "head_file", "args": {"path": "./logs/app.log", "n": 100}}',
                                          "parsed": None, "exact": False, "refusal": None}]},
            "next_leg": {"n": 1, "rows": [{"id": m["id"], "category": "multi_intent", "prompt": m["prompt"],
                                           "emitted_calls": [{"function": {"name": "cat_file",
                                                                           "arguments": json.dumps(m["gold_next"]["args"])}}],
                                           "refusal": None, "content_head": None, "exact": False}]},
            "decide_leg": {"n": 1, "options": ["head_file", "read_file", "none"],
                           "rows": [{"id": a["id"], "category": a["category"], "gold_tool": "head_file",
                                     "probs": [0.7, 0.2, 0.1]}]},
        }
        p = tmp_path / "r.json"
        p.write_text(json.dumps(record))
        mod.rescore_files([str(p)])
        out = json.loads(p.read_text())
        assert out["raw_leg"]["exact_match"] == 1 and out["raw_leg"]["rows"][0]["json_parse"]
        assert out["next_leg"]["exact"] == 1
        d = out["decide_leg"]
        assert d["top1_ok"] == 1 and abs(d["rows"][0]["nll"] - (-math.log(0.7))) < 1e-9
        assert d["certificate"]["threshold"] == 0.7 and out["rescored_with"].startswith("xyntetik")

    def test_rescore_refuses_a_record_with_other_labels(self, tmp_path):
        mod = load_scorer()
        p = tmp_path / "r.json"
        p.write_text(json.dumps({"set_version": "v2", "labels_sha256": "0" * 64}))
        with pytest.raises(SystemExit):
            mod.rescore_files([str(p)])
