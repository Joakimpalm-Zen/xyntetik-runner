"""Tandem: the prompt hook starts a background delegation only where the
ledger shows verified successes for this repository and model, the harness
is told, the stop hook surfaces a verified result once, and every
delegation joins the ledger with the class its diff had."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from xyntetik_runner.shadow import tandem
from xyntetik_runner.shadow.cli import main
from xyntetik_runner.shadow.evidence import Disposition, EpisodeEvidence
from xyntetik_runner.shadow.install import read_config, write_config

FIXTURE = Path(__file__).parent / "fixtures" / "repair_task_v1"
BUGGY = (FIXTURE / "workspace" / "calc" / "money.py").read_text(encoding="utf-8")
VISIBLE = (FIXTURE / "workspace" / "tests" / "test_money.py").read_text(encoding="utf-8")
CORRECT_FN = '''def parse_amount(text: str) -> int:
    """Return the amount in integer cents for a decimal money string."""
    cleaned = "".join(ch for ch in text if ch.isdigit() or ch in "-.")
    whole, _, frac = cleaned.partition(".")
    sign = -1 if whole.startswith("-") else 1
    whole = whole.lstrip("-") or "0"
    return sign * (int(whole) * 100 + int((frac + "00")[:2]))'''
CORRECT = '"""Money parsing for the ledger importer."""\n\n\n' + CORRECT_FN + "\n"


def git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@x"}
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env,
                          check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "proj"
    (r / "calc").mkdir(parents=True)
    (r / "tests").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    (r / "calc" / "__init__.py").write_text("", encoding="utf-8")
    (r / "calc" / "money.py").write_text(BUGGY, encoding="utf-8")
    (r / "tests" / "test_money.py").write_text(VISIBLE, encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "buggy")
    return r


def record(project: str, sha: str, verified: bool, task_class: str = "function") -> EpisodeEvidence:
    from xyntetik_runner.shadow.evidence import Identity, VerifierOutcome
    ident = Identity(project=project, task_class=task_class, context_band="s", tool_set=(), verifier_id="v",
                     environment_id="e", model_sha256=sha, quant="q4", template_sha256="t", runner_build="b",
                     backend="cpu", harness_version="h", scaffold_sha256="base")
    v = VerifierOutcome("v", verified, 3, 3 if verified else 2, 0 if verified else 1, 0, 0)
    return EpisodeEvidence(episode_id=f"e{time.time_ns()}", source="capture", observed_at="now",
                           disposition=Disposition.VERIFIED_LOCAL_ATTEMPT if verified else Disposition.LOCAL_FAILED,
                           identity=ident, baseline_sha256="", patch_sha256=None, changed_paths=("x",),
                           verifier=v, wall_s=1.0)


def test_gate_opens_only_on_this_repository_and_model() -> None:
    recs = [record("proj", "m1", True), record("proj", "m1", True), record("proj", "m1", False),
            record("other", "m1", True), record("other", "m1", True), record("other", "m1", True),
            record("proj", "m2", True), record("proj", "m2", True), record("proj", "m2", True)]
    r = tandem.qualifies(recs, "proj", "m1")
    assert r is not None and (r.attempted, r.verified) == (3, 2)
    assert tandem.qualifies(recs, "proj", "m3") is None
    assert tandem.qualifies(recs[:2], "proj", "m1") is None, "two attempts are not evidence"
    assert not tandem.looks_like_a_task("/shadow") and not tandem.looks_like_a_task("yes")
    assert tandem.looks_like_a_task("fix parse_amount so thousands separators work")


def test_strong_record_asks_for_runner_first_and_wait_returns_the_verdict(repo: Path, tmp_path: Path) -> None:
    from xyntetik_runner.shadow.routes import Route
    assert Route("function", "m", 5, 4).strong and not Route("function", "m", 5, 3).strong
    assert not Route("function", "m", 4, 4).strong and Route("function", "m", 4, 4).qualifies
    home = tmp_path / "home"
    spawned: list[list[str]] = []
    recs = [record("proj", "sha1", True)] * 4 + [record("proj", "sha1", False)]
    out = tandem.hook_prompt(home, session_id="s1", cwd=str(repo), prompt="make parse_amount handle currency",
                             repo=repo, records=recs, model="/m/coder.gguf", model_sha256="sha1", python="py",
                             out=str(tmp_path / "out"), spawn=spawned.append)
    assert out is not None
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "runner first" in ctx and "delegations --wait" in ctx and "4 verified of 5" in ctx
    st = tandem.states(home)[0]
    # wait: the delegation ends while waiting; the verdict comes back and counts as surfaced
    ticks: list[float] = []

    def sleep(sec: float) -> None:
        ticks.append(sec)
        st.status, st.tests_exit, st.changed_paths, st.patch_path = "done", 0, ("calc/money.py",), "/p/y.patch"
        st.save(home)
    done = tandem.wait_for(home, st.id, sleep=sleep)
    assert done is not None and done.verified and done.surfaced and len(ticks) == 1
    assert tandem.wait_for(home, "nope") is None
    assert tandem.hook_stop(home, session_id="s1") is None, "already surfaced by the wait"


def test_prompt_hook_starts_a_background_attempt_and_the_stop_hook_surfaces_it_once(repo: Path, tmp_path: Path) -> None:
    home = tmp_path / "home"
    spawned: list[list[str]] = []
    recs = [record("proj", "sha1", True)] * 3
    common = dict(session_id="s1", cwd=str(repo), repo=repo, records=recs, model="/m/coder.gguf",
                  model_sha256="sha1", python="py", out=str(tmp_path / "out"), spawn=spawned.append)
    # a short or slash prompt starts nothing
    assert tandem.hook_prompt(home, prompt="/shadow", **common) is None and not spawned
    # a task in a qualifying repository starts one and tells the harness
    out = tandem.hook_prompt(home, prompt="make parse_amount handle thousands separators", **common)
    assert out is not None and out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "attempting this request in the background" in ctx and "3 verified of 3" in ctx
    assert len(spawned) == 1 and spawned[0][:4] == ["py", "-m", "xyntetik_runner.shadow", "delegate"]
    assert "--record" in spawned[0] and "--background" not in spawned[0]
    st = tandem.states(home)
    assert len(st) == 1 and st[0].running and st[0].session_id == "s1"
    # while one runs, a second prompt starts nothing more; the stop hook says nothing
    assert tandem.hook_prompt(home, prompt="and also handle currency prefixes please", **common) is None
    assert len(spawned) == 1
    assert tandem.hook_stop(home, session_id="s1") is None
    # the delegation ends verified: the stop hook surfaces it exactly once
    st[0].status, st[0].tests_exit, st[0].changed_paths = "done", 0, ("calc/money.py",)
    st[0].patch_path, st[0].verdict, st[0].ended_at = "/p/x.patch", "tests passed on the scratch copy", time.time()
    st[0].save(home)
    stop = tandem.hook_stop(home, session_id="s1")
    assert stop is not None and stop["decision"] == "block" and "git apply /p/x.patch" in stop["reason"]
    assert tandem.hook_stop(home, session_id="s1") is None, "surfaced once"
    nxt = tandem.hook_prompt(home, prompt="next request, long enough to count", **common) or {}
    assert "ended verified" not in nxt.get("hookSpecificOutput", {}).get("additionalContext", ""), "not surfaced twice"
    # a failed one is never surfaced as a block, and another session's result is not this session's
    st2 = tandem.DelegationState(id="b", session_id="s2", request="r", repo=str(repo), started_at=time.time() - 5,
                                 status="done", tests_exit=1, changed_paths=("x",), verdict="tests failed")
    st2.save(home)
    assert tandem.hook_stop(home, session_id="s1") is None
    assert tandem.hook_stop(home, session_id="s2") is None
    assert "| done | tests failed |" in tandem.render_delegations(home)


def test_capture_hook_emits_json_only_when_tandem_applies(repo: Path, tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
    from xyntetik_runner.shadow import cli
    home = tmp_path / "home"
    out = tmp_path / "out"
    out.mkdir()
    model = tmp_path / "coder.gguf"
    model.write_bytes(b"GGUF")
    write_config(home, model=str(model), runner="r", ctx=4096, gpu="auto", threads=0, out=str(out))
    cap = home / ".xyntetik" / "shadow" / "capture.jsonl"
    spawned: list[list[str]] = []
    monkeypatch.setattr(tandem, "spawn_detached", spawned.append)
    monkeypatch.setattr(cli, "_model_sha", lambda m: "sha1")
    stdin = lambda d: monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(d)))  # noqa: E731
    # no evidence yet: the hook records and prints nothing
    stdin({"session_id": "s1", "cwd": str(repo), "prompt": "make parse_amount handle thousands separators"})
    assert main(["capture", "--event", "prompt", "--file", str(cap), "--home", str(home)]) == 0
    assert capsys.readouterr().out == "" and not spawned
    # with evidence for this repository and model: starts, and prints one JSON line
    with (out / "evidence.jsonl").open("w", encoding="utf-8") as f:
        for r in [record("proj", "sha1", True)] * 3:
            f.write(r.to_json() + "\n")
    stdin({"session_id": "s1", "cwd": str(repo), "prompt": "make parse_amount handle thousands separators"})
    assert main(["capture", "--event", "prompt", "--file", str(cap), "--home", str(home)]) == 0
    line = capsys.readouterr().out.strip()
    assert line and json.loads(line)["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert len(spawned) == 1
    # tandem off: nothing starts
    assert main(["tandem", "off", "--home", str(home)]) == 0
    assert capsys.readouterr().out.strip() == "tandem off"
    assert read_config(home)["tandem"] is False
    stdin({"session_id": "s1", "cwd": str(repo), "prompt": "another request that is long enough"})
    assert main(["capture", "--event", "prompt", "--file", str(cap), "--home", str(home)]) == 0
    assert capsys.readouterr().out == "" and len(spawned) == 1


def test_recorded_delegation_joins_the_ledger_with_its_class(repo: Path, tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
    from xyntetik_runner.shadow import cli
    home = tmp_path / "home"
    out = tmp_path / "out"
    model = tmp_path / "coder.gguf"
    model.write_bytes(b"GGUF")
    write_config(home, model=str(model), runner="r", ctx=4096, gpu="auto", threads=0, out=str(out))

    class Fixer:
        def __init__(self, url: str, **k: Any) -> None:
            self.base_url = url
            self.n = 0

        def capabilities(self, **k: Any) -> dict[str, Any]:
            return {"models": [{"id": "coder.gguf"}], "version": "t", "backend": "cpu"}

        def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            self.n += 1
            if self.n == 1:
                return {"choices": [{"message": {"content": "", "tool_calls": [
                    {"id": "1", "function": {"name": "write_file", "arguments": json.dumps(
                        {"path": "calc/money.py", "content": CORRECT})}}]}}]}
            return {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "2", "function": {"name": "finish", "arguments": "{}"}}]}}]}
    monkeypatch.setattr(cli, "RunnerEndpoint", Fixer)
    st = tandem.DelegationState(id="d1", session_id="s1", request="fix parse_amount", repo=str(repo),
                                started_at=time.time())
    st.save(home)
    rc = main(["delegate", "--repo", str(repo), "--request", "fix parse_amount", "--endpoint", "http://x",
               "--python", sys.executable, "--home", str(home), "--out", str(out), "--record", "d1"])
    text = capsys.readouterr().out
    assert rc == 0, text
    done = tandem.states(home)[0]
    assert done.status == "done" and done.verified and done.task_class == "function"
    recs = [EpisodeEvidence.from_json(l) for l in (out / "evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 1 and recs[0].disposition is Disposition.VERIFIED_LOCAL_ATTEMPT
    assert recs[0].identity.project == "proj" and recs[0].identity.task_class == "function"
    assert recs[0].identity.verifier_id == "repo-tests" and recs[0].source == "delegation"
    # the background entry point refuses without evidence, and starts when it has some
    monkeypatch.setattr(cli, "_model_sha", lambda m: recs[0].identity.model_sha256)
    spawned: list[list[str]] = []
    monkeypatch.setattr(cli, "spawn_detached", spawned.append)
    assert main(["delegate", "--repo", str(repo), "--request", "another change, long enough to count",
                 "--home", str(home), "--out", str(out), "--background", "--python", "py"]) == 1
    assert "not started: no task class qualifies" in capsys.readouterr().out
    with (out / "evidence.jsonl").open("a", encoding="utf-8") as f:
        for r in [record("proj", recs[0].identity.model_sha256, True)] * 2:
            f.write(r.to_json() + "\n")
    assert main(["delegate", "--repo", str(repo), "--request", "another change, long enough to count",
                 "--home", str(home), "--out", str(out), "--background", "--python", "py", "--session", "s9"]) == 0
    assert "started delegation" in capsys.readouterr().out and len(spawned) == 1 and "--record" in spawned[0]
    assert main(["delegations", "--home", str(home), "--session", "s9"]) == 0
    assert "| running |" in capsys.readouterr().out
    assert main(["delegations", "--home", str(home), "--wait", "d1"]) == 0
    assert "git apply" in capsys.readouterr().out
    assert main(["delegations", "--home", str(home), "--wait", "missing"]) == 2


def test_delegation_writes_a_signed_receipt_when_a_key_exists(repo: Path, tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
    from xyntetik_runner.shadow import cli, receipt
    home = tmp_path / "home"
    out = tmp_path / "out"
    model = tmp_path / "coder.gguf"
    model.write_bytes(b"GGUF")
    write_config(home, model=str(model), runner="fake-runner", ctx=4096, gpu="auto", threads=0, out=str(out))

    class Fixer:
        def __init__(self, url: str, **k: Any) -> None:
            self.base_url = url
            self.n = 0

        def capabilities(self, **k: Any) -> dict[str, Any]:
            return {"models": [{"id": "coder.gguf"}], "version": "t", "backend": "cpu"}

        def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            self.n += 1
            if self.n == 1:
                return {"choices": [{"message": {"content": "", "tool_calls": [
                    {"id": "1", "function": {"name": "write_file", "arguments": json.dumps(
                        {"path": "calc/money.py", "content": CORRECT})}}]}}]}
            return {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "2", "function": {"name": "finish", "arguments": "{}"}}]}}]}
    monkeypatch.setattr(cli, "RunnerEndpoint", Fixer)
    # no key: no receipt, and nothing said about one
    rc = main(["delegate", "--repo", str(repo), "--request", "fix parse_amount", "--endpoint", "http://x",
               "--python", sys.executable, "--home", str(home), "--out", str(out)])
    assert rc == 0 and "receipt:" not in capsys.readouterr().out
    assert not receipt.receipts_dir(home).exists()
    # a key: the runner would sign; here a stand-in appends what the runner appends
    key = home / receipt.SIGNKEY_REL
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps({"algo": "ed25519", "public_key": "ab" * 32, "seed": "00" * 32}), encoding="utf-8")
    signed_calls: list[tuple[str, Path, Path, Path | None]] = []

    def fake_sign(runner: str, path: Path, k: Path, prev: Path | None) -> receipt.Signed:
        signed_calls.append((runner, path, k, prev))
        rec = json.loads(path.read_text(encoding="utf-8"))
        body = json.dumps(rec, sort_keys=True)
        chain = receipt.sha256_text(body)
        rec["chain"] = {"algo": "sha256", "prev": json.loads(prev.read_text(encoding="utf-8"))["chain"]["hash"] if prev else "", "hash": chain}
        rec["signature"] = {"algo": "ed25519", "public_key": "ab" * 32, "sig": "cd" * 64}
        path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        return receipt.Signed(path, chain, "ab" * 32)
    monkeypatch.setattr(receipt, "sign_with_runner", fake_sign)
    rc = main(["delegate", "--repo", str(repo), "--request", "fix parse_amount", "--endpoint", "http://x",
               "--python", sys.executable, "--home", str(home), "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and "receipt:" in text, text
    assert len(signed_calls) == 1 and signed_calls[0][0] == "fake-runner" and signed_calls[0][3] is None
    files = sorted(receipt.receipts_dir(home).glob("*.json"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text(encoding="utf-8"))
    assert rec["_type"] == receipt.STATEMENT_TYPE and rec["predicateType"] == receipt.PREDICATE_TYPE
    assert rec["subject"][1]["digest"]["gitCommit"] == git(repo, "rev-parse", "HEAD")
    assert rec["predicate"]["verdict"] == "tests passed on the scratch copy"
    assert rec["predicate"]["request_sha256"] == receipt.sha256_text("fix parse_amount")
    assert "fix parse_amount" not in files[0].read_text(encoding="utf-8"), "the request is never in the receipt"
    assert rec["predicate"]["budget"] == {"turns": 12, "wall_s": 900.0}
    # the second receipt links to the first
    rc = main(["delegate", "--repo", str(repo), "--request", "fix parse_amount again", "--endpoint", "http://x",
               "--python", sys.executable, "--home", str(home), "--out", str(out)])
    assert rc == 0 and signed_calls[1][3] == files[0]
    assert main(["receipts", "--home", str(home)]) == 0
    listing = capsys.readouterr().out
    assert listing.count("| proj |") == 2 and rec["chain"]["hash"][:12] in listing
