"""Evaluation as a recorded step (R17.2): scripts/eval-record.py and --require-eval.

An evaluation writes <model>.eval.<kind>.json about one file (its sha256),
with the method, the numbers, the thresholds and whether they passed;
--sign-key signs it. --lineage shows a file's evaluations (STALE when made
for another file), and --require-eval refuses a model without a passing,
current one (signed by --trust-key when that is given).
"""
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval-record.py"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    if not (ROOT / "test.gguf").exists():
        pytest.skip("run `make test.gguf` first")
    return exe


def _run(cmd, cwd):
    return subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True,
                          text=True, timeout=600)


@pytest.fixture()
def work(runner_bin, tmp_path):
    shutil.copy(ROOT / "test.gguf", tmp_path / "base.gguf")
    assert _run([runner_bin, "-m", "base.gguf", "--quantize", "q.gguf", "--quant", "q8_0"],
                tmp_path).returncode == 0
    (tmp_path / "corpus.txt").write_text(
        (ROOT / "tests" / "fixtures" / "gold-corpus-2k.txt").read_text()[:600])
    assert _run([runner_bin, "--keygen", "key.json"], tmp_path).returncode == 0
    return tmp_path


def _fidelity(runner_bin, d, *extra):
    return _run([sys.executable, SCRIPT, "fidelity", "--model", "q.gguf", "--reference",
                 "base.gguf", "--corpus", "corpus.txt", "--max-positions", "16",
                 "--runner", runner_bin, *extra], d)


def test_a_fidelity_eval_is_recorded_signed_and_shown(runner_bin, work):
    p = _fidelity(runner_bin, work, "--max-mean-kld", "1", "--sign-key", "key.json")
    assert p.returncode == 0, p.stderr
    rec = json.loads((work / "q.gguf.eval.fidelity.json").read_text())
    assert rec["schema_version"] == "xyntetik.runner.eval.v1" and rec["pass"] is True
    assert rec["metrics"]["positions_scored"] == 16
    assert rec["subject"]["record"]["path"].endswith("q.gguf.quant.json")
    assert "signature" in rec
    w = _run([runner_bin, "--lineage", "q.gguf"], work)
    # the signed evaluation verifies; the fixture's quantize record is unsigned
    assert w.returncode == 1 and "CONSISTENT, NOT ALL SIGNED" in w.stdout, w.stdout
    assert "eval fidelity: vs base.gguf" in w.stdout and "pass  VERIFIED (signed" in w.stdout


def test_a_failed_threshold_is_recorded_and_exits_1(runner_bin, work):
    p = _fidelity(runner_bin, work, "--min-top1", "101")
    assert p.returncode == 1, p.stderr
    rec = json.loads((work / "q.gguf.eval.fidelity.json").read_text())
    assert rec["pass"] is False and rec["checks"] == {"min_top1_pct": False}


def test_require_eval_gates_the_load(runner_bin, work):
    gen = ["-p", "hi", "-n", "2", "--temp", "0", "--gpu", "off"]
    # no evaluation yet
    p = _run([runner_bin, "-m", "q.gguf", "--require-eval", "fidelity", *gen], work)
    assert p.returncode != 0 and "no fidelity evaluation" in p.stderr
    # a passing, unsigned one: accepted without --trust-key, refused with one
    assert _fidelity(runner_bin, work, "--max-mean-kld", "1").returncode == 0
    p = _run([runner_bin, "-m", "q.gguf", "--require-eval", "fidelity", *gen], work)
    assert p.returncode == 0, p.stderr
    pub = json.loads((work / "key.json").read_text())["public_key"]
    p = _run([runner_bin, "-m", "q.gguf", "--require-eval", "fidelity",
              "--trust-key", pub, *gen], work)
    assert p.returncode != 0 and "unsigned" in p.stderr
    # signed by that key: accepted
    assert _fidelity(runner_bin, work, "--max-mean-kld", "1",
                     "--sign-key", "key.json").returncode == 0
    p = _run([runner_bin, "-m", "q.gguf", "--require-eval", "fidelity",
              "--trust-key", pub, *gen], work)
    assert p.returncode == 0, p.stderr
    # a failing one is refused
    assert _fidelity(runner_bin, work, "--min-top1", "101").returncode == 1
    p = _run([runner_bin, "-m", "q.gguf", "--require-eval", "fidelity", *gen], work)
    assert p.returncode != 0 and "did not pass" in p.stderr


def test_an_eval_for_another_file_is_stale(runner_bin, work):
    assert _fidelity(runner_bin, work, "--max-mean-kld", "1").returncode == 0
    # the file changes after its evaluation: the record no longer speaks for it
    assert _run([runner_bin, "-m", "base.gguf", "--quantize", "q.gguf", "--quant", "q4_0"],
                work).returncode == 0
    w = _run([runner_bin, "--lineage", "q.gguf"], work)
    assert "STALE (made for another file)" in w.stdout, w.stdout
    p = _run([runner_bin, "-m", "q.gguf", "--require-eval", "fidelity", "-p", "hi", "-n",
              "2", "--gpu", "off"], work)
    assert p.returncode != 0 and "made for another file" in p.stderr


def test_an_agent_eval_summarizes_a_bank_run(runner_bin, work):
    ev = work / "evidence.jsonl"
    rows = [{"disposition": "verified_local_attempt", "wall_s": 100.0, "reasons": ["ok"],
             "resources": {"turns": 4, "tool_calls": 6}, "identity": {"harness_version": "shadow-0.1"}},
            {"disposition": "rejected", "wall_s": 50.0,
             "reasons": ["attempt: model error: HTTP 500"], "resources": {"turns": 2, "tool_calls": 2}}]
    ev.write_text("".join(json.dumps(r) + "\n" for r in rows))
    att = work / "attempts" / "task1"
    att.mkdir(parents=True)
    (att / "x.transcript.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        {"role": "assistant", "tool_calls": [{"function": {"name": "a"}}, {"function": {"name": "b"}}]},
        {"role": "assistant", "tool_calls": [{"function": {"name": "c"}}]},
    ]))
    p = _run([sys.executable, SCRIPT, "agent", "--model", "q.gguf", "--evidence", ev,
              "--attempts", work / "attempts", "--min-verified", "1", "--runner", runner_bin], work)
    assert p.returncode == 0, p.stderr
    m = json.loads((work / "q.gguf.eval.agent.json").read_text())["metrics"]
    assert (m["tasks_attempted"], m["tasks_verified"], m["engine_error_aborts"]) == (2, 1, 1)
    assert (m["turns"], m["tool_calls"], m["multi_call_turns"]) == (6, 8, 1)
    w = _run([runner_bin, "--lineage", "q.gguf"], work)
    assert "eval agent: 1 of 2 tasks verified, 1 aborts; pass" in w.stdout, w.stdout
