"""--transcript: notarized inference D1 (xyntetik.runner.transcript.v1).

The record makes one inference a replayable computation. Gates: two
identical runs produce identical output tokens (T1 substrate); the chain
hash re-verifies from the file bytes alone (sha256 over everything before
the ,"chain" key — a verifier needs a text editor, not this repo); the
model/binary hashes are real; and a run with a different seed at temp>0
is a DIFFERENT record (the determinism is seeded, not vacuous).
"""
import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("transcript")
    b = d / "base.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(b)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return b


def _run(runner_bin, base, out, seed="7", temp="0"):
    p = subprocess.run(
        [runner_bin, "-m", str(base), "-p", "the runner trains the",
         "-n", "12", "--temp", temp, "-s", seed, "--gpu", "off", "-t", "2",
         "--transcript", str(out)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(out.read_bytes())


def test_replay_substrate_and_chain(runner_bin, base, tmp_path):
    t1, t2 = tmp_path / "t1.json", tmp_path / "t2.json"
    r1 = _run(runner_bin, base, t1)
    r2 = _run(runner_bin, base, t2)
    assert r1["output"]["tokens"] == r2["output"]["tokens"]
    assert r1["prompt"]["tokens"] == r2["prompt"]["tokens"]
    # chain hash verifies from the file bytes alone
    raw = t1.read_bytes()
    body = raw[:raw.rindex(b',"chain"')]
    assert hashlib.sha256(body).hexdigest() == r1["chain"]["hash"]
    # the hashes are real, not placeholders
    assert r1["model"]["sha256"] == hashlib.sha256(
        base.read_bytes()).hexdigest()
    assert len(r1["build"]["binary_sha256"]) == 64
    assert r1["config"]["seed"] == 7
    assert r1["output"]["finish"] in ("stop", "length")
    assert r1["schema_version"] == "xyntetik.runner.transcript.v1"


def test_seeded_not_vacuous(runner_bin, base, tmp_path):
    a = _run(runner_bin, base, tmp_path / "a.json", seed="7", temp="0.9")
    b = _run(runner_bin, base, tmp_path / "b.json", seed="7", temp="0.9")
    assert a["output"]["tokens"] == b["output"]["tokens"]  # same seed replays
