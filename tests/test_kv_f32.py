"""--kv f32 (R6.32): K and V rows kept f32 in the CPU inference attention, a measurement instrument.

The default cache rounds K and V to f16. R6.31 needed Runner's distance from a float32 reference with that rounding
taken out. The CPU attention already read f32 rows for training (--train-kv-f32, R8.25.2), so `--kv f32` gives any
CPU run the same cache. It doubles the KV memory against f16, and it is not a speed or quality default.

What is checked, with the anchors outside the code under test where there are any:
- **A run that could land on a GPU is refused, with the reason.** The GPU attention kernels read f16, q8_0 and fp4
  rows only.
- **Every place that names the KV type says f32:** the -v line, --info doctor and the receipt. --verify replays such
  a record.
- **The option does something:** an f32 and an f16 cache score the same text differently.
- **The f32 cache is the training forward's f32 cache, bit for bit.** Rollouts sampled by a `--kv f32 -b 1` server are
  scored again by `--train ... --lr 0 --train-kv-f32` on their token ids, and every logprob is equal. This is the
  R17.5.1 instrument; the anchor is the trainer's own forward.
- **Session images and KV snapshots carry f32.** An image written straight through and one written suspended and
  resumed are the same bytes. A snapshot names kv_type f32, and a server with another KV type refuses it by code.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from test_rollout_identity import CTX, _fixture, _prompt_ids, _rollouts, _run  # noqa: E402
from test_kv_snapshots import MEMORY, _post, _srv  # noqa: E402

TEXT = "the quick brown fox jumps over the lazy dog and the dog sleeps in the warm sun all day"


def _model(tmp_path, name="m.gguf", extra=()):
    path = tmp_path / name
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(path), *extra], check=True,
                   cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


def _cli(runner_bin, *args, ok=True):
    p = subprocess.run([str(runner_bin), *map(str, args)], cwd=ROOT, capture_output=True, text=True,
                       errors="replace", timeout=300)
    if ok:
        assert p.returncode == 0, p.stderr[-1500:]
    return p


def test_a_run_that_could_use_a_gpu_is_refused_with_the_reason(runner_bin, tmp_path):
    m = _model(tmp_path)
    p = _cli(runner_bin, "-m", m, "-p", "hi", "-n", "2", "--kv", "f32", ok=False)
    assert p.returncode != 0
    assert "--kv f32 runs on the CPU path only" in p.stderr and "--gpu off" in p.stderr, p.stderr[-800:]
    p = _cli(runner_bin, "-m", m, "-p", "hi", "-n", "2", "--kv", "f64", ok=False)
    assert p.returncode != 0 and "f16, f32, q8, k8v4 or fp4" in p.stderr, p.stderr[-800:]


def test_every_place_that_names_the_kv_type_says_f32_and_verify_replays_it(runner_bin, tmp_path):
    m = _model(tmp_path)
    p = _cli(runner_bin, "-m", m, "-p", "hello", "-n", "3", "--temp", "0", "--kv", "f32", "--gpu", "off",
             "-v", "--record", tmp_path / "r.json")
    assert any(l.startswith("kv cache") and l.rstrip().endswith("(f32)") for l in p.stderr.splitlines()), \
        p.stderr[-1500:]
    rec = json.loads((tmp_path / "r.json").read_text())
    assert rec["profile"]["kv"] == "f32" and rec["profile"]["gpu"] is False
    # doctor exits non-zero for what it finds on a bare fixture (no chat template); its JSON is the answer
    doc = json.loads(_cli(runner_bin, "-m", m, "--kv", "f32", "--gpu", "off", "--info", "doctor", ok=False).stdout)
    assert doc["context"]["kv"] == "f32"
    v = _cli(runner_bin, "--verify", tmp_path / "r.json", "-m", m)
    assert '"verdict":"VERIFIED"' in v.stdout, (v.stdout, v.stderr[-800:])
    # a record that claims an f32 cache on a GPU cannot have come from a run
    rec["profile"]["gpu"] = True
    (tmp_path / "forged.json").write_text(json.dumps(rec))
    v = _cli(runner_bin, "--verify", tmp_path / "forged.json", "-m", m, ok=False)
    assert "UNVERIFIABLE" in v.stdout + v.stderr


def test_an_f32_cache_scores_differently_from_f16(runner_bin, tmp_path):
    m = _model(tmp_path)
    lps = {}
    for kv in ("f16", "f32"):
        p = _cli(runner_bin, "-m", m, "-p", TEXT, "--score", "1", "--gpu", "off", "-t", "2", "--kv", kv)
        lps[kv] = json.loads(p.stdout)["logprobs"]
    assert len(lps["f16"]) == len(lps["f32"]) > 8
    assert lps["f16"] != lps["f32"], "an f32 cache read the f16 rows"


def _trained_f32(runner_bin, d, rollouts):
    data = d / "ids.jsonl"
    data.write_text("".join(json.dumps({"prompt_tokens": p, "completion_tokens": c, "weight": 1.0}) + "\n"
                            for p, c, _ in rollouts))
    lp = d / "trained.jsonl"
    _run(runner_bin, ["-m", str(d / "base.gguf"), "--train", str(data), "--train-steps", str(len(rollouts)),
                      "--lr", "0", "--train-out", str(d / "lr0.gguf"), "--train-logprobs", str(lp),
                      "--train-kv-f32", "--gpu", "off", "--lora-rank", "8", "-t", "2", "-c", CTX])
    lines = [json.loads(x) for x in lp.read_text().splitlines()]
    assert [x["step"] for x in lines] == list(range(1, len(rollouts) + 1))
    return [x["logprobs"] for x in lines]


@pytest.mark.parametrize("family", ["f32", "q4_k", "gemma4", "e-series"])
def test_the_f32_cache_is_the_training_forwards_f32_cache_bit_for_bit(runner_bin, tmp_path, family):
    d = _fixture(runner_bin, tmp_path, family)
    base = d / "base.gguf"
    pids = _prompt_ids(runner_bin, base)
    rolls = _rollouts(runner_bin, base, None, ["--kv", "f32", "-b", "1"], prompt_ids=pids)
    trained = _trained_f32(runner_bin, d, rolls)
    for (p, c, served), got in zip(rolls, trained):
        assert served == got, (family, p, c)
    # and the f16 cache's rollouts are not the f32 trainer's: the comparison has power
    rolls16 = _rollouts(runner_bin, base, None, ["-b", "1"], prompt_ids=pids)
    assert any(r[2] != t for r, t in zip(rolls16, _trained_f32(runner_bin, d, rolls16))), family


def test_a_session_image_carries_f32(runner_bin, tmp_path):
    m = _model(tmp_path)
    gen = ["-p", "hello there", "-n", "12", "-s", "7", "--temp", "0.8", "--ignore-eos", "--kv", "f32",
           "--gpu", "off"]
    _cli(runner_bin, "-m", m, *gen, "--session-out", tmp_path / "end.img")
    _cli(runner_bin, "-m", m, *gen, "--suspend-after", "5", "--session-out", tmp_path / "mid.img")
    _cli(runner_bin, "-m", m, "--resume", tmp_path / "mid.img", "--session-out", tmp_path / "mid-end.img")
    assert (tmp_path / "end.img").read_bytes() == (tmp_path / "mid-end.img").read_bytes()
    p = _cli(runner_bin, "-m", m, "--kv", "f16", "--resume", tmp_path / "mid.img", ok=False)
    assert p.returncode != 0 and "--kv" in p.stderr, p.stderr[-800:]


def test_a_kv_snapshot_carries_f32_and_another_type_refuses_it(runner_bin, tmp_path):
    m, snaps = _model(tmp_path), tmp_path / "snaps"
    with _srv(runner_bin, m, snaps, "--kv", "f32") as srv:
        assert _post(srv, "/v1/runner/contexts", {"id": "w", "prompt": MEMORY})[0] == 200
        st, snap = _post(srv, "/v1/runner/contexts/w/snapshot", {"name": "w32"})
        assert st == 200, snap
    assert json.loads((snaps / "w32.kv.json").read_text())["kv_type"] == "f32"
    with _srv(runner_bin, m, snaps, "--kv", "f32") as srv:
        st, body = _post(srv, "/v1/runner/contexts", {"id": "again", "snapshot": "w32"})
        assert st == 200, body
    with _srv(runner_bin, m, snaps) as srv:
        st, body = _post(srv, "/v1/runner/contexts", {"id": "x", "snapshot": "w32"})
        assert st == 409 and body.get("error", {}).get("code") == "snapshot_kv_type_mismatch", body
