"""A qwen35-family Gated DeltaNet layer reserves no KV rows (R3.44).

Qwen3.5, Qwen3.5/3.6 MoE (qwen35moe), Qwen3.8-Flash-Next (qwen4exp) and Clef
run three Gated DeltaNet layers to one attention layer. A DeltaNet layer keeps
its state in the recurrent fold and never reads or writes a KV row, yet until
R3.44 every layer was given rows of the attention layers' shape: about 4x the
KV a 1-in-4 hybrid needs, on the host, on the device, in every prefix-cache
entry and in every saved state. The loader now gives those layers 0 KV heads.

What is checked, on the CPU, against RUNNER_KV_RECURRENT_ROWS=1 (the prior
layout, kept to resume what an earlier runner saved; checked once against the
prior runner's binary, whose images it resumes to the same tokens, state and
logits, the header's binary digest aside):
- the rows were dead: greedy text and teacher-forced scores are byte-identical
  under either layout;
- the bytes saved: a session image's state shrinks by exactly
  tokens x recurrent layers x the fixture's K+V row (2 KV heads x 8 wide x
  f16 x 2), the fixture's declared geometry being the anchor; --fit and the
  loader's KV line are held to that same geometry, a declared NextN/MTP
  block charged only under --mtp (R3.45);
- a state saved under the prior layout is refused by name, never misread: a
  session image and a KV snapshot each say what wrote them and how to read
  them, and under the knob the image resumes to the straight run's bytes.
"""
import json
import os
import pathlib
import re
import struct
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

KNOB = "RUNNER_KV_RECURRENT_ROWS"
KV_HEADS, HEAD_DIM, F16 = 2, 8, 2      # every fixture below
ROW = 2 * KV_HEADS * HEAD_DIM * F16     # K + V, one layer, one position
N, HALF = 40, 20

# (id, generator, its environment, layers); full_attention_interval is 4 in all
FAMILIES = [
    ("qwen35", "make-test-ornith.py", {}, 4),
    ("qwen35-8", "make-test-ornith.py", {"ORNITH_TEST_LAYERS": "8"}, 8),
    ("qwen35moe", "make-test-ornith.py", {"ORNITH_TEST_MOE": "1"}, 4),
    ("clef", "make-test-clef.py", {}, 4),
    ("qwen4exp", "make-test-qwen4exp.py", {}, 4),
]


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp("q35kv")
    out = {"d": d}
    for name, gen, env, _ in FAMILIES:
        path = d / f"{name}.gguf"
        subprocess.run([sys.executable, ROOT / "scripts" / gen, str(path)], check=True,
                       cwd=ROOT, stdout=subprocess.DEVNULL, env={**os.environ, **env})
        out[name] = path
    # the 8-layer fixture with its declared NextN/MTP block's tensors, so --mtp binds it
    path = d / "qwen35-8-mtp.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", str(path)], check=True,
                   cwd=ROOT, stdout=subprocess.DEVNULL,
                   env={**os.environ, "ORNITH_TEST_LAYERS": "8", "ORNITH_TEST_MTP": "1"})
    out["qwen35-8-mtp"] = path
    return out


def _env(prior):
    env = {k: v for k, v in os.environ.items() if k != KNOB}
    if prior:
        env[KNOB] = "1"
    return env


def _run(runner_bin, prior, *args):
    r = subprocess.run([runner_bin, "--gpu", "off", "-t", "2", *map(str, args)], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
                       env=_env(prior))
    return r.returncode, r.stdout, r.stderr.decode(errors="replace")


def _state_bytes(path):
    b = pathlib.Path(path).read_bytes()
    off = 16
    (hl,) = struct.unpack_from("<I", b, off); off += 4 + hl
    (nt,) = struct.unpack_from("<I", b, off); off += 4 + 4 * nt
    (sl,) = struct.unpack_from("<Q", b, off)
    return nt, sl


@pytest.mark.parametrize("fam", FAMILIES, ids=[f[0] for f in FAMILIES])
def test_the_rows_were_dead_and_are_gone(runner_bin, fx, fam):
    name, _, _, layers = fam
    model, d = fx[name], fx["d"] / name
    d.mkdir()
    texts, scores, states = [], [], []
    for prior in (False, True):
        rc, out, err = _run(runner_bin, prior, "-m", model, "-p", "hello", "-n", N,
                            "--temp", "0", "--ignore-eos", "--session-out", d / f"{prior}.img")
        assert rc == 0, err[-800:]
        texts.append(out)
        states.append(_state_bytes(d / f"{prior}.img"))
        rc, out, err = _run(runner_bin, prior, "-m", model, "-p", "the quick brown fox " * 6,
                            "--score", "3")
        assert rc == 0, err[-800:]
        scores.append(out)
    assert texts[0] == texts[1], f"{name}: greedy text depends on the recurrent layers' rows"
    assert scores[0] == scores[1], f"{name}: scores depend on the recurrent layers' rows"
    (nt, now), (nt_prior, prior) = states
    assert nt == nt_prior
    recurrent = layers - layers // 4
    assert prior - now == nt * recurrent * ROW, (name, nt, prior, now)


@pytest.mark.parametrize("mtp", [False, True], ids=["plain", "mtp"])
@pytest.mark.parametrize("prior", [False, True], ids=["now", "prior-layout"])
def test_fit_and_the_loader_charge_the_same_rows(runner_bin, fx, prior, mtp):
    """--fit reads the header by the loader's rules: the DeltaNet layers own
    no rows (R3.44), and the declared NextN/MTP block past the trunk is no
    layer and owns rows only when --mtp binds it (R3.45). Anchor: the
    fixture's declared geometry, 8 trunk layers of which 2 attend (every
    4th), and one predictor block of the attention shape."""
    model = fx["qwen35-8-mtp"]
    rows = (8 if prior else 2) + (1 if mtp else 0)
    flag = ["--mtp"] if mtp else []
    big, mid = 1 << 22, 1 << 16
    rc, out, err = _run(runner_bin, prior, "--fit", model, "-c", big, *flag)
    assert rc == 0, err[-600:]
    text = out.decode()
    assert re.search(r"model\s+qwen35, 8 layers, 1 NextN/MTP block "
                     + ("charged" if mtp else "not charged"), text), text
    fit_gib = float(re.search(r"kv cache\s+([\d.]+) GiB", text).group(1))
    assert abs(fit_gib - rows * ROW * big / (1 << 30)) <= 0.006, text
    rc, _, err = _run(runner_bin, prior, "-m", model, "-v", "-p", "hi", "-n", 1, "-c", mid, *flag)
    assert rc == 0, err[-600:]
    load_mb = float(re.search(r"kv cache\s+([\d.]+) MB", err).group(1))
    assert abs(load_mb - rows * ROW * mid / 1e6) <= 0.06, err[-600:]


def test_a_prior_layout_image_is_named_and_resumes_under_the_knob(runner_bin, fx):
    model, d = fx["qwen35"], fx["d"] / "prior-image"
    d.mkdir()
    gen = ["-m", model, "-p", "hello", "-n", N, "-s", "5", "--temp", "0.9", "--ignore-eos"]
    assert _run(runner_bin, True, *gen, "--session-out", d / "straight.img")[0] == 0
    rc, _, err = _run(runner_bin, True, *gen, "--suspend-after", HALF,
                      "--session-out", d / "half.img")
    assert rc == 0, err[-800:]

    rc, out, err = _run(runner_bin, False, "-m", model, "--resume", d / "half.img",
                        "--session-out", d / "refused.img")
    assert rc == 1 and out == b"", err[-600:]
    assert "an earlier runner wrote it" in err and f"{KNOB}=1" in err, err[-600:]
    assert not (d / "refused.img").exists()

    rc, _, err = _run(runner_bin, True, "-m", model, "--resume", d / "half.img",
                      "--session-out", d / "resumed.img")
    assert rc == 0, err[-800:]
    assert (d / "resumed.img").read_bytes() == (d / "straight.img").read_bytes()

    # and the other way: an image of the new layout is not the prior one's
    assert _run(runner_bin, False, *gen, "--suspend-after", HALF,
                "--session-out", d / "new.img")[0] == 0
    rc, _, err = _run(runner_bin, True, "-m", model, "--resume", d / "new.img")
    assert rc == 1 and "the model's identity differs" in err, err[-600:]


def _post(srv, path, payload):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as r:
        return r.code, json.load(r)


def _srv(runner_bin, model, snaps, prior):
    return RunnerServer(runner_bin, model, ctx=256, env=_env(prior), extra_args=[
        "--gpu", "off", "-t", "2", "--kv-snapshots", str(snaps)])


def test_a_prior_layout_snapshot_is_named_and_loads_under_the_knob(runner_bin, fx):
    model, snaps = fx["qwen35"], fx["d"] / "snaps"
    memory = "the key is under the mat; the owl is white; the cat is grey."
    with _srv(runner_bin, model, snaps, True) as srv:
        assert _post(srv, "/v1/runner/contexts", {"id": "mem", "prompt": memory})[0] == 200
        st, snap = _post(srv, "/v1/runner/contexts/mem/snapshot", {})
        assert st == 200, snap
    with _srv(runner_bin, model, snaps, False) as srv:
        st, body = _post(srv, "/v1/runner/contexts", {"id": "x", "snapshot": "mem"})
        err = body.get("error", {})
        assert st == 409 and err.get("code") == "snapshot_kv_layout", body
        assert f"{KNOB}=1" in err.get("message", ""), err
    with _srv(runner_bin, model, snaps, True) as srv:
        st, ctx = _post(srv, "/v1/runner/contexts", {"id": "y", "snapshot": "mem"})
        assert st == 200, ctx
