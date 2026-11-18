"""R17.9 step 4: a --save-every run's record commits to its checkpoints, so an auditor can re-run one segment.

The record lists every numbered checkpoint the run wrote ({step, cursor, adapter_sha256, optim_sha256}) and a chain
over them, bound to the record itself:
  h_0 = sha256("xyntetik.runner.train.chain.v1|" + sha256(the record's bytes before the chain fields, plus "}"))
  h_i = sha256("h_{i-1}|step|cursor|adapter_sha256|optim_sha256"), each as lowercase hex text
so a chain cannot be moved to another run: any change to the base's or the data's hash, the recipe, the overrides or
the environment changes h_0. The gate under it: in a T3 build, a run resumed from a checkpoint is the uninterrupted
run byte for byte at every later checkpoint (adapter and .optim), measured at several cut points.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROWS = [{"prompt": "question %d about item %d" % (i, i * 7 % 5), "completion": " answer %d" % (i * 3)}
        for i in range(5)]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def _env():
    return {k: v for k, v in os.environ.items() if not k.startswith("RUNNER_") and not k.startswith("XR_")}


def _run(exe, args):
    p = subprocess.run([str(exe), *map(str, args)], cwd=ROOT, capture_output=True, text=True, timeout=600,
                       env=_env())
    assert p.returncode == 0, p.stderr[-800:]
    return p


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _ck(out, k):
    return pathlib.Path("%s.step-%05d" % (out, k))


@pytest.fixture
def fixture(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), str(tmp_path / "base.gguf")],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    return tmp_path


def _recipe(d, steps, every):
    return ["-m", d / "base.gguf", "--train", d / "data.jsonl", "--train-steps", steps, "--save-every", every,
            "--lr", "3e-3", "--train-batch", 2, "--gpu", "off"]


def chain_of(text, rec):
    body = text[:text.index(',"checkpoints":')] + "}"
    h = hashlib.sha256(("xyntetik.runner.train.chain.v1|" +
                        hashlib.sha256(body.encode()).hexdigest()).encode()).hexdigest()
    h0 = h
    for c in rec["checkpoints"]:
        h = hashlib.sha256(("%s|%d|%d|%s|%s" % (h, c["step"], c["cursor"], c["adapter_sha256"],
                                                 c["optim_sha256"])).encode()).hexdigest()
    return h0, h


def test_a_save_every_run_records_its_checkpoints_and_a_chain_bound_to_the_record(runner_bin, fixture):
    out = fixture / "a.gguf"
    _run(runner_bin, [*_recipe(fixture, 6, 2), "-t", 2, "--train-out", out])
    text = pathlib.Path(str(out) + ".train.json").read_text()
    rec = json.loads(text)
    cks = rec.get("checkpoints")
    assert [c["step"] for c in cks or []] == [2, 4, 6], rec.keys()
    for c in cks:
        assert c["cursor"] == 2 * c["step"], c                        # batch 2: rows consumed
        assert c["adapter_sha256"] == _sha(_ck(out, c["step"])), c
        assert c["optim_sha256"] == _sha(str(_ck(out, c["step"])) + ".optim"), c
    h0, head = chain_of(text, rec)
    assert rec["checkpoint_chain"] == {"basis": "xyntetik.runner.train.chain.v1", "h0": h0, "head": head}
    # the binding's live control: the same chain over a record whose lr differs is another h_0
    moved = text.replace('"lr":', '"lr":1', 1)
    assert chain_of(moved, json.loads(moved))[0] != h0


def test_a_run_without_checkpoints_records_none(runner_bin, fixture):
    out = fixture / "b.gguf"
    _run(runner_bin, ["-m", fixture / "base.gguf", "--train", fixture / "data.jsonl", "--train-steps", 2,
                      "--lr", "3e-3", "-t", 2, "--gpu", "off", "--train-out", out])
    rec = json.loads(pathlib.Path(str(out) + ".train.json").read_text())
    assert "checkpoints" not in rec and "checkpoint_chain" not in rec


def test_resumed_runs_are_the_straight_run_byte_for_byte_in_a_t3_build(runner_bin, fixture):
    """The kill switch of segment replay, as a gate: if a resumed run were not byte-exact, segments could not be
    chained. Straight at -t 2; resumed at -t 4 from steps 1, 3 and 6 of 8, with warmup and a cosine schedule."""
    v = subprocess.run([str(runner_bin), "--version"], capture_output=True, text=True, timeout=20).stdout
    if "(t3)" not in v:
        pytest.skip("a T3 build's gate (make T3=1): a default build is exact on its own binary only")
    sched = ["--lr-warmup", 2, "--lr-schedule", "cosine", "--lr-floor", "0.1"]
    straight = fixture / "s.gguf"
    _run(runner_bin, [*_recipe(fixture, 8, 1), *sched, "-t", 2, "--train-out", straight])
    for cut in (1, 3, 6):
        out = fixture / ("r%d.gguf" % cut)
        _run(runner_bin, [*_recipe(fixture, 8, 1), *sched, "-t", 4, "--train-resume", _ck(straight, cut),
                          "--train-out", out])
        for k in range(cut + 1, 9):
            assert _sha(_ck(out, k)) == _sha(_ck(straight, k)), (cut, k, "adapter")
            assert _sha(str(_ck(out, k)) + ".optim") == _sha(str(_ck(straight, k)) + ".optim"), (cut, k, "optim")
        assert _sha(out) == _sha(straight), (cut, "final")
    ctl = fixture / "c.gguf"            # the live control: another lr is another adapter
    _run(runner_bin, [*[x if x != "3e-3" else "2e-3" for x in _recipe(fixture, 8, 1)], *sched, "-t", 2,
                      "--train-out", ctl])
    assert _sha(ctl) != _sha(straight)
