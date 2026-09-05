"""--lora: adapter loading at inference (adaptation D2).

Observed through --score (D1), which makes every gate a one-command
comparison of exact numerics:

  * a zero adapter (lora_b == 0) scores BYTE-identical to the bare base —
    the hook contributes exactly nothing when the delta is zero;
  * --lora-scale 0 likewise;
  * a real adapter changes the score, deterministically;
  * base + adapter approximates the MERGED reference model (W += (alpha/r)BA
    baked into the weights) within float tolerance — the mathematical
    definition of what applying a LoRA means;
  * hostile adapters (shape mismatch, half a pair) are refused with the
    offending tensor named — never silently skipped.
"""
import json
import pathlib
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer

PROMPT = "the quick brown fox jumps over the lazy dog and keeps on running"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp("lora")
    base = d / "base.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(base)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, ROOT / "scripts/make-test-lora.py",
                    str(base), str(d / "fx")], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return {"base": base, "adapter": d / "fx.adapter.gguf",
            "zero": d / "fx.zero.gguf", "merged": d / "fx.merged.gguf",
            "badshape": d / "fx.badshape.gguf",
            "extradim": d / "fx.extradim.gguf",
            "halfpair": d / "fx.halfpair.gguf",
            "f16": d / "fx.f16.gguf"}


def _score(runner_bin, model, lora=None, scale=None):
    cmd = [runner_bin, "-m", str(model), "--score", "-p", PROMPT,
           "-t", "2", "--gpu", "off"]
    if lora:
        cmd += ["--lora", str(lora)]
    if scale is not None:
        cmd += ["--lora-scale", str(scale)]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p.stdout


def test_zero_adapter_is_byte_identical_to_base(runner_bin, fx):
    assert (_score(runner_bin, fx["base"])
            == _score(runner_bin, fx["base"], lora=fx["zero"]))


def test_scale_zero_is_byte_identical_to_base(runner_bin, fx):
    assert (_score(runner_bin, fx["base"])
            == _score(runner_bin, fx["base"], lora=fx["adapter"], scale=0))


def test_real_adapter_changes_the_score_deterministically(runner_bin, fx):
    base = _score(runner_bin, fx["base"])
    a1 = _score(runner_bin, fx["base"], lora=fx["adapter"])
    a2 = _score(runner_bin, fx["base"], lora=fx["adapter"])
    assert a1 != base
    assert a1 == a2


def _completion(server):
    req = urllib.request.Request(
        server.base_url + "/v1/completions",
        data=json.dumps({"prompt": "hello there", "max_tokens": 8,
                         "temperature": 0, "seed": 1, "logprobs": 3,
                         "cache_prompt": False, "prefix_cache": False}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["choices"][0]


@pytest.mark.parametrize("mode", ["reload", "parallel", "registry"])
def test_served_adapter_survives_every_model_load(runner_bin, fx, mode):
    """Relative lifecycle gate: every load must preserve the initially
    adapted answer, and it must differ from the bare base. The adapter's
    numerical anchor is test_adapter_matches_the_merged_reference below.
    """
    args = ["--gpu", "off", "-t", "2"]
    with RunnerServer(runner_bin, fx["base"], extra_args=args) as srv:
        bare = _completion(srv)
    adapted_args = [*args, "--lora", str(fx["adapter"]), "--lora-scale", "100"]
    with RunnerServer(runner_bin, fx["base"], extra_args=adapted_args) as srv:
        expected = _completion(srv)
        assert expected != bare, "the adapter must affect the answer"
        if mode == "reload":
            req = urllib.request.Request(srv.base_url + "/unload", data=b"")
            with urllib.request.urlopen(req, timeout=30) as r:
                assert json.load(r)["status"] == "ok"
            assert _completion(srv) == expected
            return
    model = f"named={fx['base']}" if mode == "registry" else fx["base"]
    with RunnerServer(runner_bin, model, parallel=2 if mode == "parallel" else 1,
                      extra_args=adapted_args) as srv:
        for _ in range(4):
            assert _completion(srv) == expected


def test_adapter_matches_the_merged_reference(runner_bin, fx):
    adapted = json.loads(_score(runner_bin, fx["base"], lora=fx["adapter"]))
    merged = json.loads(_score(runner_bin, fx["merged"]))
    deltas = [abs(a - b) for a, b in
              zip(adapted["logprobs"], merged["logprobs"])]
    # Wx + B(Ax) vs (W+BA)x differ only by float summation order; a hook that
    # scaled wrong, indexed the wrong row, or hit the wrong projection lands
    # orders of magnitude past this
    assert max(deltas) <= 5e-4, max(deltas)


def test_f16_adapter_loads_and_matches_f32(runner_bin, fx):
    """The format llama.cpp's convert_lora_to_gguf emits (measured: the
    first community adapter tried was F16 and the loader refused it). The
    F16 twin holds the same values rounded to half precision, so it must
    load and score within rounding distance of the F32 adapter, and it must
    actually be the adapter (differ from the bare base)."""
    f32 = json.loads(_score(runner_bin, fx["base"], lora=fx["adapter"]))
    f16 = json.loads(_score(runner_bin, fx["base"], lora=fx["f16"]))
    base = json.loads(_score(runner_bin, fx["base"]))
    assert abs(f16["nll_mean"] - f32["nll_mean"]) < 5e-2, \
        (f16["nll_mean"], f32["nll_mean"])
    assert f16["nll_mean"] != base["nll_mean"]


@pytest.mark.parametrize("bad,needle", [
    ("badshape", b"attn_q"),
    ("extradim", b"attn_q"),
    ("halfpair", b"lora_a/lora_b pair"),
])
def test_hostile_adapters_are_refused_by_name(runner_bin, fx, bad, needle):
    p = subprocess.run(
        [runner_bin, "-m", str(fx["base"]), "--score", "-p", PROMPT,
         "-t", "2", "--gpu", "off", "--lora", str(fx[bad])],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert p.returncode != 0
    assert needle in p.stderr
