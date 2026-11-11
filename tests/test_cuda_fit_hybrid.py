"""`--fit` charges a hybrid's device instance exactly as the CUDA plan does (R3.46).

The fit predicts the layer split a CUDA load makes from the header alone. Its fixed charge (the headroom plus one
instance's activation scratch at the narrowest window) left out a hybrid's recurrent state and mixer scratch. That is
qwen35's Gated DeltaNet fold, and Mamba-2's conv window and SSM state, each held twice: live, and for the speculative
rollback. It also took 4 x n_embd as the FFN scratch width where the loader takes the expert width or a per-layer
array's widest entry. Qwen3.6-35B-A3B at ctx 131072 read 40/40 on the fit against the loader's 39/40. The plan, for
its part, charged qwen35's fold a second time: 4x what it allocates.

Both now charge model_gpu_recurrent_cost, once. RUNNER_DEBUG_VRAM prints the two figures in bytes, `fit: (bytes:
fixed F, headroom H)` and the loader's `gpu-split: (bytes: fixed F, headroom H)`, and they must agree on every hybrid
family the CUDA backend runs. The headroom is compared apart, because it follows the free memory each process saw.

The checks need a CUDA device; elsewhere the module skips.
"""
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIT = re.compile(r"fit: \(bytes: fixed (\d+), headroom (\d+)\)")
PLAN = re.compile(r"gpu-split: \(bytes: fixed (\d+), headroom (\d+)\)")

# (id, generator, its arguments, its environment)
FAMILIES = [
    ("qwen35", "make-test-ornith.py", [], {}),
    ("qwen35moe", "make-test-ornith.py", [], {"ORNITH_TEST_MOE": "1"}),
    ("clef", "make-test-clef.py", [], {}),
    ("qwen4exp", "make-test-qwen4exp.py", [], {}),
    ("granitehybrid", "make-test-hybrid.py", ["--dense"], {}),
    ("nemotron_h", "make-test-nemotron.py", [], {}),
    ("dense-control", "make-test-model.py", [], {}),
    # deepseek2's MLA (R3.48): the absorbed form the loader defaults to, and the naive one
    ("deepseek2", "make-test-model.py", ["--arch", "deepseek2"], {}),
    ("deepseek2-v2-lite-attention", "make-test-model.py", ["--arch", "deepseek2", "--mla-wide"], {}),
]
# the environment the fit and the load run under, per case
RUNTIME = {"deepseek2-naive": ("deepseek2", {"RUNNER_MLA_NAIVE": "1"})}


def _cuda(runner_bin):
    caps = subprocess.run([str(runner_bin), "--caps"], cwd=ROOT, stdout=subprocess.PIPE, timeout=30)
    return (json.loads(caps.stdout.decode()).get("gpu") or {}).get("backend") == "cuda"


@pytest.fixture(scope="module")
def models(runner_bin, tmp_path_factory):
    if not _cuda(runner_bin):
        pytest.skip("needs a CUDA device")
    d = tmp_path_factory.mktemp("fithybrid")
    out = {}
    for name, gen, args, env in FAMILIES:
        stem = d / name
        # make-test-hybrid.py takes a prefix and appends .gguf; the others take the file
        target = stem if gen in ("make-test-hybrid.py",) else stem.with_suffix(".gguf")
        subprocess.run([sys.executable, ROOT / "scripts" / gen, str(target), *args], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL, env={**os.environ, **env})
        path = stem.with_suffix(".gguf")
        if not path.exists():   # make-test-nemotron.py writes <arg>.gguf
            path = pathlib.Path(str(target) + ".gguf")
        out[name] = path
    return out


def _env(extra=None):
    env = {k: v for k, v in os.environ.items() if k != "RUNNER_MLA_NAIVE"}
    return {**env, "RUNNER_DEBUG_VRAM": "1", **(extra or {})}


@pytest.mark.parametrize("name", [f[0] for f in FAMILIES] + sorted(RUNTIME))
def test_the_fit_charges_the_instance_the_loader_plans(runner_bin, models, name):
    base, extra = RUNTIME.get(name, (name, {}))
    model = models[base]
    fit = subprocess.run([str(runner_bin), "--fit", str(model), "-c", "4096"], cwd=ROOT, capture_output=True,
                         text=True, errors="replace", timeout=120, env=_env(extra))
    assert fit.returncode == 0, fit.stderr[-1500:]
    load = subprocess.run([str(runner_bin), "-m", str(model), "-p", "hi", "-n", "1", "-c", "4096", "--gpu", "99",
                           "--no-tray", "-t", "2"], cwd=ROOT, capture_output=True, text=True, errors="replace",
                          timeout=300, env=_env(extra))
    assert load.returncode == 0, load.stderr[-1500:]
    f = FIT.search(fit.stderr)
    p = PLAN.search(load.stderr)
    assert f and p, (fit.stderr[-1500:], load.stderr[-1500:])
    fit_fixed, fit_head = int(f.group(1)), int(f.group(2))
    plan_fixed, plan_head = int(p.group(1)), int(p.group(2))
    assert fit_fixed - fit_head == plan_fixed - plan_head, (name, fit_fixed, fit_head, plan_fixed, plan_head)


# R3.58: a load that keeps every expert bank on the host (--cpu-moe, or
# nemotron_h_moe's CUDA default, R3.43a.2) uploads only the mixer tensors and
# dense FFNs, each rounded as a bound upload rounds it, with no expert-bias
# tables and no expert scale table. --fit charged every layer's whole file
# bytes and those tables, and it charged no Mamba-2 state for nemotron_h_moe.
# The fit now charges that load as the load does: the fixed charge, the split's
# weight bytes and its layer count agree byte for byte. (The bytes a split uses
# in all also count each layer's KV and the window scratch, where fit and load
# differ by the same few kilobytes on every case, R3.46's included.)
FIT_W = re.compile(r"fit: \(bytes: used \d+, weights (\d+), layers (\d+)\)")
PLAN_W = re.compile(r"gpu-split: \(bytes: used \d+, weights (\d+), layers (\d+)\)")
HOST_EXPERTS = [
    ("nemotron_h_moe-default", "make-test-hybrid.py", ["--arch", "nemotron_h_moe"], ".gguf", []),
    ("qwen3moe-cpu-moe", "make-test-moe.py", [], ".moe4.gguf", ["--cpu-moe"]),
]


@pytest.mark.parametrize("name,gen,args,suffix,flags", HOST_EXPERTS, ids=[h[0] for h in HOST_EXPERTS])
def test_the_fit_charges_a_load_whose_experts_stay_on_the_host(runner_bin, tmp_path, name, gen, args,
                                                                 suffix, flags):
    if not _cuda(runner_bin):
        pytest.skip("needs a CUDA device")
    stem = tmp_path / "m"
    subprocess.run([sys.executable, ROOT / "scripts" / gen, str(stem), *args], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    model = pathlib.Path(str(stem) + suffix)
    fit = subprocess.run([str(runner_bin), "--fit", str(model), "-c", "4096", *flags], cwd=ROOT,
                         capture_output=True, text=True, errors="replace", timeout=120, env=_env())
    assert fit.returncode == 0, fit.stderr[-1500:]
    assert "experts       on the host" in fit.stdout, fit.stdout
    load = subprocess.run([str(runner_bin), "-m", str(model), "-p", "hi", "-n", "1", "-c", "4096", "--no-tray",
                           "-t", "2", *flags], cwd=ROOT, capture_output=True, text=True, errors="replace",
                          timeout=300, env={**_env(), "RUNNER_GPU_FALLBACK": "fail"})
    assert load.returncode == 0, load.stderr[-1500:]
    assert re.search(r"experts 0/\d+ layers on GPU", load.stderr), load.stderr[-1500:]
    f, p = FIT.search(fit.stderr), PLAN.search(load.stderr)
    fw, pw = FIT_W.search(fit.stderr), PLAN_W.search(load.stderr)
    assert f and p and fw and pw, (fit.stderr[-1500:], load.stderr[-1500:])
    assert (int(f.group(1)), int(f.group(2))) == (int(p.group(1)), int(p.group(2))), name
    assert (int(fw.group(1)), int(fw.group(2))) == (int(pw.group(1)), int(pw.group(2))), name
