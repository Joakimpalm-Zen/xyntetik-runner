"""Metal's fast decode matvec is the default for Q4_K tensors, and only them.

The owner promoted `k_mvf_q4_K` (2026-10-23, suite R3.21.1) on its D1 rows:
mean KL 0.9993x on Llama-3.2-3B and 0.9998x on Qwen2.5-1.5B against the
publisher's float32 model (Wilcoxon p 0.43 / 0.75), decode +10.8% / +12.5%
on the Q4_K_M Llamas. It is a tolerance route: its fidelity anchor is those
rows and `./test-mv-tol MODEL.gguf` (0/64 teacher-forced flips), not this
file. This file pins the routing every other gate relies on:

- unset: Q4_K tensors take the fast kernel, every other type the identity
  kernel (Q8_0 and Q4_0 measured neutral and stay opt-in);
- RUNNER_METAL_MV=0: the identity kernel everywhere, the switch every
  CPU-against-GPU byte comparison sets;
- RUNNER_METAL_MV=1: the fast kernel for every type that has one, as before.

Each case is told apart by the bytes of a token-by-token `--score` (which
the fast kernel's reassociation changes on these fixtures) and by the route
line RUNNER_DEBUG_ROUTE prints.

The default covers dense models only: its D1 rows are dense, and D1 is held
as written for MoE families (owner, 2026-10-23), so a MoE model keeps the
identity kernel unless RUNNER_METAL_MV=1 (gpu_t.mv_q4k_default, set at load
from n_expert). That half has no fixture here: no MoE fixture reaches Metal
with Q4_K weights (their widths are not whole superblocks, so the quantizer
keeps Q5_0, and they fall back to the CPU), and the real Q4_K MoE files do
not fit the 8 GB machines this suite runs on.
"""
import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEXT = "the quick brown fox jumps over the lazy dog"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    caps = json.loads(subprocess.run([str(exe), "--caps"], capture_output=True, text=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "metal":
        pytest.skip("no Metal device")
    return exe


@pytest.fixture(scope="module")
def fixtures(runner_bin, tmp_path_factory):
    d = tmp_path_factory.mktemp("mv")
    base = d / "wide.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--wide", str(base)],
                   check=True, cwd=ROOT, capture_output=True)
    out = {}
    for q in ("q4_k", "q8_0"):
        f = d / f"wide-{q}.gguf"
        subprocess.run([str(runner_bin), "-m", str(base), "--quantize", str(f), "--quant", q],
                       check=True, cwd=ROOT, capture_output=True)
        out[q] = f
    return out


def score(runner_bin, model, mv):
    env = {k: v for k, v in __import__("os").environ.items() if k != "RUNNER_METAL_MV"}
    env["RUNNER_DEBUG_ROUTE"] = "1"
    if mv is not None:
        env["RUNNER_METAL_MV"] = mv
    r = subprocess.run([str(runner_bin), "-m", str(model), "--gpu", "auto", "--score",
                        "-p", TEXT, "--no-tray"], env=env, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip(), "empty score output"
    route = re.search(r"\bmv=(\S+)", r.stderr)
    assert route, r.stderr
    # the score file's own route record (score_route) names the matvec route,
    # which is asserted above from stderr; the arms compare on their scores
    scores = json.loads(r.stdout)
    assert f"mv={route.group(1)} " in scores.pop("score_route")["backend_detail"]
    return json.dumps(scores, sort_keys=True), route.group(1)


def test_q4_k_takes_the_fast_kernel_by_default(runner_bin, fixtures):
    default, route = score(runner_bin, fixtures["q4_k"], None)
    identity, id_route = score(runner_bin, fixtures["q4_k"], "0")
    fast, fast_route = score(runner_bin, fixtures["q4_k"], "1")
    assert (route, id_route, fast_route) == ("fast-q4_K", "identity", "fast")
    assert default == fast
    assert default != identity     # the fast kernel engaged: its bits differ


def test_q8_0_keeps_the_identity_kernel_by_default(runner_bin, fixtures):
    default, route = score(runner_bin, fixtures["q8_0"], None)
    identity, _ = score(runner_bin, fixtures["q8_0"], "0")
    fast, _ = score(runner_bin, fixtures["q8_0"], "1")
    assert route == "fast-q4_K"
    assert default == identity
    assert fast != identity        # the opt-in still reaches the Q8_0 kernel
