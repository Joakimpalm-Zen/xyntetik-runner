"""`shadow adapt`: the ledger's own commits as verified-SFT data at the
function unit, the held-out gate, and the model pick at install. No model:
the served endpoint is scripted, training is a stand-in that writes an
adapter file, and the verifier is the real one on real scratch worktrees."""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from xyntetik_runner.shadow import adapt, edits
from xyntetik_runner.shadow.cli import main
from xyntetik_runner.shadow.importer import Episode
from xyntetik_runner.shadow.install import (Candidate, fit_verdict, read_config, suggest_model,
                                            write_config)
from xyntetik_runner.shadow.tasks import RepairTask, admit

FIXTURE = Path(__file__).parent / "fixtures" / "repair_task_v1"
BUGGY = (FIXTURE / "workspace" / "calc" / "money.py").read_text(encoding="utf-8")
VISIBLE = (FIXTURE / "workspace" / "tests" / "test_money.py").read_text(encoding="utf-8")
SOLUTION_TESTS = (FIXTURE / "protected" / "tests" / "test_money.py").read_text(encoding="utf-8")
# a fix that lives entirely inside the function: no import, so the unit is the whole change
CORRECT_FN = '''def parse_amount(text: str) -> int:
    """Return the amount in integer cents for a decimal money string."""
    cleaned = "".join(ch for ch in text if ch.isdigit() or ch in "-.")
    whole, _, frac = cleaned.partition(".")
    sign = -1 if whole.startswith("-") else 1
    whole = whole.lstrip("-") or "0"
    return sign * (int(whole) * 100 + int((frac + "00")[:2]))'''
CORRECT = '"""Money parsing for the ledger importer."""\n\n\n' + CORRECT_FN + "\n"
BUGGY_FN = BUGGY.split("\n\n\n", 1)[1].rstrip("\n")


def git(repo: Path, *args: str, date: str = "2026-09-01T12:00:00+00:00") -> str:
    env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date,
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@x"}
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          env=env, check=True).stdout.strip()


def make_repo(root: Path, name: str, *, tag: str) -> Path:
    """Buggy at 11:00, fixed inside parse_amount at 12:00; ``tag`` varies the
    content so two repositories do not share commit ids."""
    r = root / name
    (r / "calc").mkdir(parents=True)
    (r / "tests").mkdir()
    git(root, "init", "-q", "-b", "main", str(r))
    (r / "calc" / "__init__.py").write_text(f"# {tag}\n", encoding="utf-8")
    (r / "calc" / "money.py").write_text(BUGGY, encoding="utf-8")
    (r / "tests" / "test_money.py").write_text(VISIBLE, encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "buggy", date="2026-09-01T11:00:00+00:00")
    (r / "calc" / "money.py").write_text(CORRECT, encoding="utf-8")
    (r / "tests" / "test_money.py").write_text(SOLUTION_TESTS, encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "fix parse_amount", date="2026-09-01T12:00:00+00:00")
    return r


def admitted(root: Path, out: Path, names: tuple[str, ...]) -> list[RepairTask]:
    tasks = []
    for i, name in enumerate(names):
        repo = make_repo(root, name, tag=name)
        ep = Episode(source="claude_code", session_id=f"s{i}", turn=1, cwd=str(repo),
                     started_at="2026-09-01T11:30:00Z", ended_at="2026-09-01T12:10:00Z",
                     request="make parse_amount handle thousands separators, currency and rounding",
                     request_sha256=str(i) * 64)
        t = admit(ep, out_dir=out, python=sys.executable)
        assert isinstance(t, RepairTask), t
        tasks.append(t)
    return tasks


def write_gguf(path: Path, kvs: list[tuple[str, Any]]) -> None:
    """A GGUF v3 header with no tensors: strings, a uint32 and one string array."""
    def s(x: str) -> bytes:
        b = x.encode("utf-8")
        return struct.pack("<Q", len(b)) + b
    body = b""
    for k, v in kvs:
        body += s(k)
        if isinstance(v, str):
            body += struct.pack("<I", 8) + s(v)
        elif isinstance(v, int):
            body += struct.pack("<I", 4) + struct.pack("<I", v)
        else:
            body += struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(v))
            for item in v:
                body += s(item)
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, len(kvs)) + body)


def test_gguf_header_and_template_family(tmp_path: Path) -> None:
    m = tmp_path / "m.gguf"
    write_gguf(m, [("general.architecture", "qwen2"), ("qwen2.block_count", 28),
                   ("tokenizer.ggml.tokens", ["<|im_start|>", "a", "b"]),
                   ("tokenizer.chat_template", "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}")])
    meta = adapt.gguf_meta(m)
    assert meta["general.architecture"] == "qwen2"
    assert adapt.template_family(str(meta["tokenizer.chat_template"])) == "chatml"
    assert adapt.template_family("<|start_header_id|>system<|end_header_id|>") == "llama3"
    assert adapt.template_family("[INST] {{ x }} [/INST]") is None
    assert adapt.render_prompt("chatml", "S", "U") == "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"
    with pytest.raises(ValueError):
        adapt.gguf_meta(tmp_path / "not.gguf") if (tmp_path / "not.gguf").write_bytes(b"nope") else None


def test_function_unit_self_check_and_split(tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out / "tasks").mkdir(parents=True)
    tasks = admitted(tmp_path, out / "tasks", ("alpha", "beta"))
    u = adapt.function_unit(tasks[0])
    assert isinstance(u, adapt.FunctionUnit), u
    assert u.name == "parse_amount" and u.rel == "calc/money.py" and u.base_fn == BUGGY_FN
    assert u.sol_fn == CORRECT_FN
    # the human's function alone is verified; the buggy one is not
    assert adapt.judge(tasks[0], u, u.sol_fn, python=sys.executable).verified
    j = adapt.judge(tasks[0], u, u.base_fn, python=sys.executable)
    assert not j.verified and j.fixed == 0 and j.failing >= 1
    ds = adapt.build_dataset(tasks, family="chatml", ctx=4096, python=sys.executable, protocol="whole")
    assert len(ds.dev) == 1 and len(ds.holdout) == 1 and len(ds.examples) == 1 and not ds.dropped
    ex = ds.examples[0]
    assert str(ex["prompt"]).startswith("<|im_start|>system\n") and str(ex["prompt"]).endswith("<|im_start|>assistant\n")
    assert "Current source of parse_amount in calc/money.py" in str(ex["prompt"])
    assert ex["completion"] == CORRECT_FN and ex["weight"] == 1.0 and ex["end_of_turn"] is True
    # the default protocol trains on the human's change as anchored edits, the prompt asking for them
    ds = adapt.build_dataset(tasks, family="chatml", ctx=4096, python=sys.executable)
    assert ds.protocol == "edits" and len(ds.examples) == 1 and not ds.dropped
    ex = ds.examples[0]
    assert ex["completion"] == edits.render(edits.derive(BUGGY_FN, CORRECT_FN).edits)
    assert str(ex["prompt"]).startswith("<|im_start|>system\n" + edits.SYSTEM) and edits.ASK in str(ex["prompt"])
    # a task whose fix is not one function is named, not silently dropped
    two = tmp_path / "gamma"
    make_repo(tmp_path, "gamma", tag="g")
    (two / "calc" / "money.py").write_text(CORRECT + "\n\ndef other() -> int:\n    return 1\n", encoding="utf-8")
    git(two, "commit", "-q", "-am", "more", date="2026-09-01T12:30:00+00:00")
    ep = Episode(source="claude_code", session_id="s9", turn=1, cwd=str(two),
                 started_at="2026-09-01T11:30:00Z", ended_at="2026-09-01T12:40:00Z",
                 request="x", request_sha256="9" * 64)
    t3 = admit(ep, out_dir=out / "tasks", python=sys.executable)
    if isinstance(t3, RepairTask):
        why = adapt.function_unit(t3)
        assert isinstance(why, str) and ("functions changed" in why or "outside any function" in why or "new at the solution" in why)


def test_adapt_end_to_end_keeps_the_adapter_only_on_a_held_out_rise(tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
    from xyntetik_runner.shadow import cli
    home = tmp_path / "home"
    out = tmp_path / "out"
    (out / "tasks").mkdir(parents=True)
    admitted(tmp_path, out / "tasks", ("alpha", "beta"))
    model = tmp_path / "coder.gguf"
    write_gguf(model, [("tokenizer.chat_template", "<|im_start|>x")])
    write_config(home, model=str(model), runner="fake-runner", ctx=4096, gpu="auto", threads=0, out=str(out))
    served: dict[str, Any] = {"lora": False}

    class FakeManaged:
        def __init__(self, launch: Any, **k: Any) -> None:
            served["lora"] = "--lora" in launch.extra_args
            served["extra"] = launch.extra_args
            self.base_url = "http://fake"

        def start(self, **k: Any) -> bool:
            return True

        def stop(self, **k: Any) -> None:
            pass

    behaviour = {"adapter_answer": CORRECT_FN}

    class FakeEndpoint:
        def __init__(self, url: str, **k: Any) -> None:
            self.base_url = url

        def capabilities(self, **k: Any) -> dict[str, Any]:
            return {"models": [{"id": "coder.gguf"}]}

        def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            assert path == "/v1/chat/completions" and payload["messages"][0]["content"] == adapt.SYSTEM
            answer = behaviour["adapter_answer"] if served["lora"] else BUGGY_FN
            return {"choices": [{"message": {"content": "```python\n" + answer + "\n```"}}]}

    def fake_train(runner: str, model_path: str, data: Path, adapter: Path, **k: Any) -> adapt.TrainResult:
        assert runner == "fake-runner" and Path(data).is_file()
        lines = Path(data).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1 and json.loads(lines[0])["completion"] == CORRECT_FN
        adapter.write_bytes(b"GGUF-adapter")
        return adapt.TrainResult(adapter, k["steps"], 0, k["log_path"], 12.5, 0.5, 0.1)
    monkeypatch.setattr(cli, "ManagedRunner", FakeManaged)
    monkeypatch.setattr(cli, "RunnerEndpoint", FakeEndpoint)
    monkeypatch.setattr(adapt, "train", fake_train)
    common = ["adapt", "--out", str(out), "--bank", "", "--home", str(home), "--python", sys.executable, "--k", "2",
              "--protocol", "whole"]
    # the plan first, nothing trained
    assert main(common + ["--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "units: 2 (development 1, held out 1)" in text and "dry run: nothing trained" in text
    assert not list(adapt.adapters_dir(home).glob("*")) if adapt.adapters_dir(home).is_dir() else True
    # no terminal, no --yes: refused
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(""))
    assert main(common) == 2
    # the run: base answers the buggy function, the adapter the human's -> held-out rises -> kept
    assert main(common + ["--yes"]) == 0
    text = capsys.readouterr().out
    assert "kept: held-out verified rose 0 -> 2 of 2 samples" in text, text
    cfg = read_config(home)
    assert Path(str(cfg["adapter"])).is_file() and served["extra"] == ("--lora", str(cfg["adapter"]))
    runs = sorted(adapt.adapters_dir(home).glob("*/run.json"))
    assert len(runs) == 1
    rec = json.loads(runs[0].read_text(encoding="utf-8"))
    assert rec["verdict"]["promoted"] is True and rec["base_holdout"]["summary"]["verified_samples"] == 0
    assert rec["adapter_holdout"]["summary"]["verified_samples"] == 2 and rec["dataset"]["examples"] == 1
    assert (runs[0].parent / "train.jsonl").is_file() and (runs[0].parent / "manifest.json").is_file()
    # a second run whose adapter is no better is recorded and not kept; the config keeps the first
    behaviour["adapter_answer"] = BUGGY_FN
    assert main(common + ["--yes"]) == 0
    text = capsys.readouterr().out
    assert "not kept: held-out verified 0 -> 0: no rise" in text, text
    assert read_config(home)["adapter"] == cfg["adapter"]
    assert main(["adapt", "--home", str(home), "--status"]) == 0
    status = capsys.readouterr().out
    assert status.count("| yes |") == 1 and status.count("| no |") == 1
    # delegate serves the kept adapter by itself
    assert main(["delegate", "--repo", str(tmp_path / "alpha"), "--request", "x", "--home", str(home),
                 "--python", sys.executable, "--max-turns", "1"]) in (0, 1)
    assert served["lora"] is True


def test_adapt_refuses_without_a_gate_or_a_renderable_template(tmp_path: Path, capsys: Any) -> None:
    home = tmp_path / "home"
    out = tmp_path / "out"
    (out / "tasks").mkdir(parents=True)
    admitted(tmp_path, out / "tasks", ("solo",))
    model = tmp_path / "m.gguf"
    write_gguf(model, [("tokenizer.chat_template", "<|im_start|>")])
    write_config(home, model=str(model), runner="r", ctx=4096, gpu="auto", threads=0, out=str(out))
    assert main(["adapt", "--out", str(out), "--bank", "", "--home", str(home), "--python", sys.executable, "--dry-run"]) == 1
    assert "too few units to adapt" in capsys.readouterr().out
    write_gguf(model, [("tokenizer.chat_template", "[INST]")])
    assert main(["adapt", "--out", str(out), "--bank", "", "--home", str(home), "--python", sys.executable, "--dry-run"]) == 2
    assert "chat template is not one adapt can render" in capsys.readouterr().err
    assert "no adaptation runs yet" in adapt.render_status(home)


def test_install_picks_the_largest_gguf_that_fits(tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
    from xyntetik_runner.shadow import cli, install as inst
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    (home / "models").mkdir()
    (home / "models" / "small.gguf").write_bytes(b"0" * 10)
    (home / "models" / "big.gguf").write_bytes(b"0" * 20)
    (home / "models" / "huge.gguf").write_bytes(b"0" * 30)
    verdicts = {"huge.gguf": "PAGES", "big.gguf": "FITS WITH --kv q8", "small.gguf": "FITS"}
    monkeypatch.setattr(inst, "fit_verdict", lambda runner, m, ctx: verdicts[m.name])
    best, cands = suggest_model("runner", home)
    assert best is not None and best.path.name == "big.gguf" and [c.path.name for c in cands] == ["huge.gguf", "big.gguf", "small.gguf"]
    monkeypatch.setattr(cli, "suggest_model", lambda runner, home, ctx=8192: suggest_model(runner, home, ctx=ctx))
    assert main(["install", "--home", str(home), "--yes"]) == 0
    text = capsys.readouterr().out
    assert "big.gguf (found on disk" in text and "huge.gguf" in text and "PAGES" in text
    assert read_config(home)["model"] == str((home / "models" / "big.gguf").resolve())
    # nothing fits: installed without a model, said plainly
    verdicts.update({"big.gguf": "PAGES", "small.gguf": "PAGES"})
    home2 = tmp_path / "home2"
    (home2 / ".codex").mkdir(parents=True)
    (home2 / "models").mkdir()
    (home2 / "models" / "small.gguf").write_bytes(b"0" * 10)
    assert main(["install", "--home", str(home2), "--yes"]) == 0
    text = capsys.readouterr().out
    assert "no model for offloading yet: none of the GGUF files found fits" in text
    assert read_config(home2) == {}
    # the verdict parser reads the runner's own line
    monkeypatch.setattr(inst.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, "fit: x\n  weights 1 GiB\n  verdict       FITS — 2.37 GiB to spare at ctx 8192\n", ""))
    assert fit_verdict("runner", Path("x.gguf"), 8192) == "FITS"
    assert Candidate(Path("x"), 1, "FITS WITH --kv q8").fits and not Candidate(Path("x"), 1, "PAGES").fits


BUGGY_LINE = "return int(float(text) * 100)"


def test_edits_protocol_schema_apply_and_derive() -> None:
    # the schema names only the function's own distinct lines; the buggy line is one of them
    sch = edits.schema(BUGGY_FN)
    anchors = sch["properties"]["edits"]["items"]["properties"]["line"]["enum"]
    assert BUGGY_LINE in anchors and "" not in anchors
    assert sch["properties"]["edits"]["items"]["required"] == ["line", "until", "mode", "text"]
    # the human's change derives to edits that reproduce the human's function exactly
    d = edits.derive(BUGGY_FN, CORRECT_FN)
    assert d.reason == "" and len(d.edits) == 1 and d.edits[0]["mode"] == "replace"
    assert d.edits[0]["line"] == BUGGY_LINE and d.text == CORRECT_FN
    got, why = edits.apply(BUGGY_FN, edits.render(d.edits))
    assert why == "" and got == CORRECT_FN
    # an anchor that is not in the function, an until before its line, and overlapping edits are refused
    assert edits.apply(BUGGY_FN, json.dumps({"edits": [{"line": "return 0", "until": "return 0",
                                                        "mode": "replace", "text": "x"}]}))[1].startswith("line occurs 0")
    first = BUGGY_FN.split("\n")[0].strip()
    backwards = edits.apply(BUGGY_FN, json.dumps({"edits": [{"line": BUGGY_LINE, "until": first, "mode": "replace",
                                                             "text": "x"}]}))
    assert backwards == ("x", "")  # the range named backwards is the same range
    assert edits.apply(BUGGY_FN, json.dumps({"edits": [
        {"line": first, "until": BUGGY_LINE, "mode": "replace", "text": "x"},
        {"line": BUGGY_LINE, "until": BUGGY_LINE, "mode": "replace", "text": "y"}]}))[1] == "edits overlap"
    assert edits.apply(BUGGY_FN, "not json")[1] == "reply is not the edits object"
    # dedented text is re-indented to the anchor; text with its own indentation is taken as written
    fn = "def f(x):\n    if x:\n        return 1\n    return 2"
    got, why = edits.apply(fn, json.dumps({"edits": [{"line": "return 2", "until": "return 2", "mode": "replace",
                                                     "text": "if x < 0:\n    return -1\nreturn 2"}]}))
    assert why == "" and got == "def f(x):\n    if x:\n        return 1\n    if x < 0:\n        return -1\n    return 2"
    got, why = edits.apply(fn, json.dumps({"edits": [{"line": "if x:", "until": "if x:", "mode": "insert_after",
                                                     "text": "x = int(x)"}]}))
    assert why == "" and got == "def f(x):\n    if x:\n        x = int(x)\n        return 1\n    return 2"
    # a deletion, and an insertion after a repeated line reaching through its neighbours
    got, why = edits.apply(fn, json.dumps({"edits": [{"line": "if x:", "until": "return 1", "mode": "replace",
                                                     "text": ""}]}))
    assert why == "" and got == "def f(x):\n    return 2"
    dup = "def g(x):\n    if x:\n        return\n    if x > 1:\n        return\n    return 3"
    d = edits.derive(dup, "def g(x):\n    if x:\n        return\n    if x > 1:\n        return\n    x += 1\n    return 3")
    assert d.reason == "" and len(d.edits) == 1 and d.edits[0]["line"] == "if x > 1:" and d.edits[0]["until"] == "return 3"
    assert d.text.endswith("    x += 1\n    return 3")
    assert edits.derive(fn, fn).reason == "no change"


def test_adapt_edits_protocol_trains_on_derived_edits_and_decodes_under_the_schema(
        tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
    from xyntetik_runner.shadow import cli
    home = tmp_path / "home"
    out = tmp_path / "out"
    (out / "tasks").mkdir(parents=True)
    admitted(tmp_path, out / "tasks", ("alpha", "beta"))
    model = tmp_path / "coder.gguf"
    write_gguf(model, [("tokenizer.chat_template", "<|im_start|>x")])
    write_config(home, model=str(model), runner="fake-runner", ctx=4096, gpu="auto", threads=0, out=str(out))
    served: dict[str, Any] = {"lora": False}
    seen: list[dict[str, Any]] = []

    class FakeManaged:
        def __init__(self, launch: Any, **k: Any) -> None:
            served["lora"] = "--lora" in launch.extra_args
            self.base_url = "http://fake"

        def start(self, **k: Any) -> bool:
            return True

        def stop(self, **k: Any) -> None:
            pass

    fix = edits.render(edits.derive(BUGGY_FN, CORRECT_FN).edits)
    # the base answers a well-formed edit that changes nothing (the runner's decoder could never
    # produce an invented anchor), except its second sample, which is not applicable
    noop = json.dumps({"edits": [{"line": BUGGY_LINE, "until": BUGGY_LINE, "mode": "replace", "text": BUGGY_LINE}]})

    class FakeEndpoint:
        def __init__(self, url: str, **k: Any) -> None:
            self.base_url = url

        def capabilities(self, **k: Any) -> dict[str, Any]:
            return {"models": [{"id": "coder.gguf"}]}

        def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            assert path == "/v1/chat/completions" and payload["messages"][0]["content"] == edits.SYSTEM
            schema = payload["response_format"]["json_schema"]["schema"]
            assert BUGGY_LINE in schema["properties"]["edits"]["items"]["properties"]["line"]["enum"]
            seen.append(payload)
            if served["lora"]:
                answer = fix
            else:
                answer = noop if payload["seed"] == 1000 else "{\"edits\": []}"
            return {"choices": [{"message": {"content": answer}}]}

    def fake_train(runner: str, model_path: str, data: Path, adapter: Path, **k: Any) -> adapt.TrainResult:
        lines = Path(data).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        ex = json.loads(lines[0])
        assert ex["completion"] == fix and ex["end_of_turn"] is True
        assert ex["prompt"].endswith(edits.ASK + "<|im_end|>\n<|im_start|>assistant\n")
        adapter.write_bytes(b"GGUF-adapter")
        return adapt.TrainResult(adapter, k["steps"], 0, k["log_path"], 12.5, 0.5, 0.1)
    monkeypatch.setattr(cli, "ManagedRunner", FakeManaged)
    monkeypatch.setattr(cli, "RunnerEndpoint", FakeEndpoint)
    monkeypatch.setattr(adapt, "train", fake_train)
    common = ["adapt", "--out", str(out), "--bank", "", "--home", str(home), "--python", sys.executable, "--k", "2"]
    assert main(common + ["--dry-run"]) == 0
    assert "edits protocol" in capsys.readouterr().out
    assert main(common + ["--yes"]) == 0
    text = capsys.readouterr().out
    assert "kept: held-out verified rose 0 -> 2 of 2 samples" in text, text
    assert "rejected (no edits)" in text
    rec = json.loads(sorted(adapt.adapters_dir(home).glob("*/run.json"))[0].read_text(encoding="utf-8"))
    assert rec["dataset"]["protocol"] == "edits" and rec["dataset"]["examples"] == 1
    assert rec["base_holdout"]["summary"]["rejected"] == 1 and rec["adapter_holdout"]["summary"]["rejected"] == 0
    assert rec["base_holdout"]["summary"]["verified_samples"] == 0
    assert rec["adapter_holdout"]["summary"]["verified_samples"] == 2
    assert all("response_format" in p for p in seen)
