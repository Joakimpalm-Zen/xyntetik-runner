"""RUNNER_MOE_HOT, a host layer's experts in a device hot bank (R3.30 step 2).

The hot file is "layer expert" pairs, one a line; the runner refuses a malformed file, a model out of
v1's scope (the plain top-k SiLU expert FFN) and the knob without --cpu-moe, on any build. On a CUDA
device, the identity gates on the toy MoE (scripts/make-test-moe.py's moe4: top-2 of 4 experts, a
real router, two layers), F32 and Q8_0, scored through RUNNER_SCORE_CHUNKED so every row goes
through the layers' FFN:

1. Every expert in the bank: the mixed path runs the fused path's own kernels on the bank's slots,
   so the scores are --cpu-moe 0's bytes.
2. Two experts of each layer's four in the bank: the other two run on the host, by the CPU path's
   own expert function. The run is the same bytes twice, and its perplexity is within 1e-3 of the
   all-host placement's; a mixed layer's sum is neither placement's bytes.
"""
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from score_record import body_bytes  # noqa: E402

EXE = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
TEXT = " ".join(["the quick brown fox jumps over the lazy dog"] * 12)
N_LAYER = 2    # make-test-moe.py's fixtures


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    if not EXE.exists():
        pytest.skip("runner binary not built")
    d = tmp_path_factory.mktemp("moehot")
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(d / "m")], check=True,
                   cwd=ROOT, stdout=subprocess.PIPE)
    subprocess.run([str(EXE), "-m", str(d / "m.moe4.gguf"), "--quantize", str(d / "m-q8.gguf"),
                    "--quant", "q8_0"], check=True, cwd=ROOT, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE)
    return d


def _run(args, hot=None, tmp=None, env=None):
    e = dict(os.environ, **(env or {}))
    e.pop("RUNNER_MOE_HOT", None)
    if hot is not None:
        f = tmp / "hot.txt"
        f.write_text(hot)
        e["RUNNER_MOE_HOT"] = str(f)
    return subprocess.run([str(EXE), *args, "--no-tray"], cwd=ROOT, env=e, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=600)


def _gen(model, extra=()):
    return ["-m", str(model), "-p", "hello", "-n", "2", "--temp", "0", *extra]


@pytest.mark.parametrize("hot,why", [
    ("0 x\n", 'line 1: a line is not "layer expert"'),
    ("0 1\n0 9\n", "line 2: the expert is out of the layer's range"),
    ("99 0\n", "line 1: the layer is not an MoE layer of this model"),
    ("0 1 2\n", "line 1: a line holds more than a layer and an expert"),
])
def test_a_malformed_hot_file_is_refused(fixtures, tmp_path, hot, why):
    p = _run(_gen(fixtures / "m.moe4.gguf", ["--gpu", "off", "--cpu-moe"]), hot, tmp_path)
    assert p.returncode != 0
    assert "RUNNER_MOE_HOT: " + why in p.stderr.decode(errors="replace")


def test_the_knob_needs_cpu_moe(fixtures, tmp_path):
    p = _run(_gen(fixtures / "m.moe4.gguf", ["--gpu", "off"]), "0 1\n", tmp_path)
    assert p.returncode != 0
    assert "RUNNER_MOE_HOT: it places a --cpu-moe host layer's experts" in p.stderr.decode(errors="replace")


def test_a_model_outside_v1_is_refused(fixtures, tmp_path):
    # gemma-4's dual-branch expert FFN is outside v1's plain top-k SiLU shape
    p = _run(_gen(fixtures / "m.gemma4-moe.gguf", ["--gpu", "off", "--cpu-moe"]), "0 1\n", tmp_path)
    assert p.returncode != 0
    assert "RUNNER_MOE_HOT: v1 covers the plain top-k SiLU expert FFN only" in p.stderr.decode(errors="replace")


def test_comments_and_repeats_parse_and_the_cpu_ignores_the_bank(fixtures, tmp_path):
    model = fixtures / "m.moe4.gguf"
    plain = _run(_gen(model, ["--gpu", "off", "--cpu-moe"]))
    hot = _run(_gen(model, ["--gpu", "off", "--cpu-moe"]), "# a comment\n0 1  # trailing\n0 1\n\n1 3\n",
               tmp_path)
    assert plain.returncode == 0 and hot.returncode == 0, hot.stderr.decode(errors="replace")
    assert hot.stdout == plain.stdout


def _cuda():
    caps = json.loads(subprocess.run([str(EXE), "--caps"], cwd=ROOT, check=True,
                                     stdout=subprocess.PIPE).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("the hot bank needs a CUDA device")


def _score(model, cpu_moe, hot=None, tmp=None):
    """The score record's bytes and the route line's (experts on host, MoE layers, hot slots)."""
    p = _run(["-m", str(model), "--gpu", "auto", "-c", "1024", "--score", "-p", TEXT,
              "--cpu-moe", cpu_moe], hot, tmp,
             env={"RUNNER_SCORE_CHUNKED": "1", "RUNNER_SPEC_BATCH": "64", "RUNNER_DEBUG_ROUTE": "1"})
    err = p.stderr.decode(errors="replace")
    assert p.returncode == 0, err
    m = re.search(r"route: backend=cuda .*?experts_on_host=(\d+)/(\d+) moe_hot=(\d+)", err)
    assert m, err
    return p.stdout, tuple(int(x) for x in m.groups())


@pytest.mark.parametrize("hot,listed", [
    ("0 0\n0 1\n1 2\n", 3),
    ("".join(f"{l} {e}\n" for l in range(N_LAYER) for e in range(4)), 8),
])
def test_the_bank_gives_back_its_listed_experts_host_copies(fixtures, tmp_path, hot, listed):
    # The count is the hot file's own, by hand. It says what the runner asked the OS to drop, not
    # what the OS dropped: a fixture's slices are smaller than a page, so no resident-memory number
    # can be read here, and that claim's evidence is the R3.35 report's measurement on a real file.
    _cuda()
    p = _run(["-m", str(fixtures / "m.moe4.gguf"), "--gpu", "auto", "--cpu-moe", "99", "-p", "hello",
              "-n", "2", "--temp", "0"], hot, tmp_path)
    err = p.stderr.decode(errors="replace")
    assert p.returncode == 0, err
    assert f"RUNNER_MOE_HOT gave back the host copies of {listed} experts" in err


@pytest.mark.parametrize("name", ["m.moe4.gguf", "m-q8.gguf"])
def test_every_expert_in_the_bank_is_the_device_placements_bytes(fixtures, tmp_path, name):
    _cuda()
    model = fixtures / name
    every = "".join(f"{l} {e}\n" for l in range(N_LAYER) for e in range(4))
    hot, route = _score(model, "99", every, tmp_path)
    dev, dev_route = _score(model, "0")
    assert route == (N_LAYER, N_LAYER, 4 * N_LAYER), route
    assert dev_route == (0, N_LAYER, 0), dev_route
    assert body_bytes(hot) == body_bytes(dev)


@pytest.mark.parametrize("name", ["m.moe4.gguf", "m-q8.gguf"])
def test_a_two_expert_bank_is_stable_and_within_tolerance(fixtures, tmp_path, name):
    _cuda()
    model = fixtures / name
    two = "".join(f"{l} {e}\n" for l in range(N_LAYER) for e in (0, 3))
    a, route = _score(model, "99", two, tmp_path)
    b, _ = _score(model, "99", two, tmp_path)
    host = json.loads(_score(model, "99")[0])
    rec = json.loads(a)
    assert route == (N_LAYER, N_LAYER, 2 * N_LAYER), route
    assert body_bytes(a) == body_bytes(b)
    assert abs(rec["ppl"] - host["ppl"]) <= 1e-3 * host["ppl"], (rec["ppl"], host["ppl"])
