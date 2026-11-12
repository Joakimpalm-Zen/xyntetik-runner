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

The same three numerical gates run again on the device arm (R8.7.2), where
the adapter is applied by the CUDA kernels rather than the host hook. They
self-skip on a machine whose backend has no adapter path, and they are the
reason a GPU-resident model may load an adapter at all: without them the
failure mode is a model that quietly ignored it.
"""
import json
import os
import pathlib
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
sys.path.insert(0, str(ROOT / "scripts"))
from harness import RunnerServer
import score_record  # noqa: E402

PROMPT = "the quick brown fox jumps over the lazy dog and keeps on running"


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
            "inf": d / "fx.inf.gguf", "nan": d / "fx.nan.gguf",
            "f16": d / "fx.f16.gguf"}


def _score(runner_bin, model, lora=None, scale=None, gpu=False):
    cmd = [runner_bin, "-m", str(model), "--score", "-p", PROMPT,
           "-t", "2", "--gpu", "auto" if gpu else "off"]
    if lora:
        cmd += ["--lora", str(lora)]
    if scale is not None:
        cmd += ["--lora-scale", str(scale)]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    # the scores without score_route, which names the model file and the
    # adapter's route
    return score_record.body_bytes(p.stdout)


def _gpu_or_skip(runner_bin, fx):
    """Skip unless a backend actually applies the adapter on the device.

    Two ways to be absent, both a skip and not a pass: no GPU at all (the
    load line never mentions a backend), and a backend without adapter
    kernels (model_lora_load refuses, which is the point of the refusal).
    """
    cmd = [runner_bin, "-m", str(fx["base"]), "--score", "-p", "hi",
           "-t", "2", "--gpu", "auto", "--lora", str(fx["zero"])]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    err = p.stderr.decode(errors="replace")
    if p.returncode != 0:
        pytest.skip("no device adapter path: " + err.strip().splitlines()[-1])
    if "adapter on device" not in err:
        pytest.skip("no GPU backend engaged for this model")


def test_zero_adapter_is_byte_identical_to_base_on_device(runner_bin, fx):
    """The exactness anchor of the device path: a zero B contributes
    fmaf(scale, 0, y), which is y, so the adapted and unadapted GPU runs
    must agree BIT for bit. A misplaced hook, a wrong stride or a scratch
    read from the previous site all break this before they break tolerance.
    """
    _gpu_or_skip(runner_bin, fx)
    assert (_score(runner_bin, fx["base"], gpu=True)
            == _score(runner_bin, fx["base"], lora=fx["zero"], gpu=True))


def test_scale_zero_is_byte_identical_to_base_on_device(runner_bin, fx):
    _gpu_or_skip(runner_bin, fx)
    assert (_score(runner_bin, fx["base"], gpu=True)
            == _score(runner_bin, fx["base"], lora=fx["adapter"], scale=0,
                      gpu=True))


def test_device_adapter_matches_the_merged_reference(runner_bin, fx):
    """The absolute anchor: the merged model is W += (alpha/r)BA baked into
    the weights by a separate script, so it knows nothing about the kernel
    under test. Agreeing with it is agreeing with the definition of applying
    a LoRA, not with our own other implementation.
    """
    _gpu_or_skip(runner_bin, fx)
    adapted = json.loads(_score(runner_bin, fx["base"], lora=fx["adapter"],
                                gpu=True))
    merged = json.loads(_score(runner_bin, fx["merged"], gpu=True))
    deltas = [abs(a - b) for a, b in
              zip(adapted["logprobs"], merged["logprobs"])]
    assert max(deltas) <= 5e-4, max(deltas)


@pytest.fixture(scope="module")
def fx_all(tmp_path_factory):
    """The same fixtures with the adapter on all seven projections of block 0
    (attention q/k/v/output, ffn gate/up/down)."""
    d = tmp_path_factory.mktemp("lora-all")
    base = d / "base.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(base)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, ROOT / "scripts/make-test-lora.py",
                    str(base), str(d / "fx")], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL,
                   env=dict(os.environ, LORA_TARGETS="all"))
    return {"base": base, "adapter": d / "fx.adapter.gguf",
            "zero": d / "fx.zero.gguf", "merged": d / "fx.merged.gguf"}


def test_every_projection_site_matches_the_merged_reference(runner_bin, fx_all):
    """Seven hooks, seven ways to forget one. With the adapter on every
    projection of a block, the CPU path and the device path are each held to
    the merged model, so a missing or misplaced hook at any site is a miss
    here and not a quietly weaker adapter."""
    merged_cpu = json.loads(_score(runner_bin, fx_all["merged"]))
    cpu = json.loads(_score(runner_bin, fx_all["base"], lora=fx_all["adapter"]))
    d_cpu = max(abs(a - b) for a, b in zip(cpu["logprobs"], merged_cpu["logprobs"]))
    assert d_cpu <= 5e-4, d_cpu
    base = json.loads(_score(runner_bin, fx_all["base"]))
    assert max(abs(a - b) for a, b in
               zip(base["logprobs"], merged_cpu["logprobs"])) > 1e-3, \
        "the all-site adapter does not move the score, so the check is vacuous"
    _gpu_or_skip(runner_bin, fx_all)
    dev = json.loads(_score(runner_bin, fx_all["base"], lora=fx_all["adapter"],
                            gpu=True))
    merged_dev = json.loads(_score(runner_bin, fx_all["merged"], gpu=True))
    d_dev = max(abs(a - b) for a, b in zip(dev["logprobs"], merged_dev["logprobs"]))
    assert d_dev <= 5e-4, d_dev


def test_device_adapter_generates_what_the_cpu_generates(runner_bin, fx_all):
    """The score path is one token at a time. Generation prefills the prompt
    as a batch, which is a different dispatch shape through every hook."""
    _gpu_or_skip(runner_bin, fx_all)
    prompt = "the quick brown fox jumps over the lazy dog and keeps running"
    outs = []
    for gpu in ("off", "auto"):
        p = subprocess.run([runner_bin, "-m", str(fx_all["base"]), "--lora",
                            str(fx_all["adapter"]), "-p", prompt, "-n", "12",
                            "-b", "8", "--temp", "0", "-t", "2", "--gpu", gpu,
                            "--no-tray"], cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=120)
        assert p.returncode == 0, p.stderr.decode(errors="replace")
        outs.append(p.stdout)
    assert outs[0] == outs[1]


def _cuda_or_skip(runner_bin):
    """The decode graph is CUDA's: Metal applies the adapter on the device too
    but has no graph to replay."""
    p = subprocess.run([runner_bin, "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=60)
    caps = json.loads(p.stdout) if p.returncode == 0 else {}
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("no CUDA backend: the decode graph is CUDA's")


def test_device_adapter_decode_replays_its_graph(runner_bin, fx_all):
    """R6.68.25b: binding an adapter turned the CUDA decode graph off for the
    model's lifetime ("an adapter bound on the device after the decode graph's
    plan"), although the adapter's launches capture like any other and the
    bind comes before the first forward. Under RUNNER_REQUIRE_GRAPH_CAPTURE a
    capture that fails ends the run and an eager-by-design decode says why,
    so the adapted decode must print no reason and its route line must say
    graph=replay; its tokens must be the eager decode's (RUNNER_CUDA_GRAPH_OFF)
    and the CPU's. The anchors are the merged-reference tests above: their
    device score runs one token at a time through the same decode."""
    _gpu_or_skip(runner_bin, fx_all)
    _cuda_or_skip(runner_bin)
    prompt = "the quick brown fox jumps over the lazy dog and keeps running"

    def gen(gpu="auto", **env_extra):
        env = dict(os.environ, RUNNER_REQUIRE_GRAPH_CAPTURE="1",
                   RUNNER_DEBUG_ROUTE="1", **env_extra)
        p = subprocess.run([runner_bin, "-m", str(fx_all["base"]), "--lora",
                            str(fx_all["adapter"]), "-p", prompt, "-n", "24",
                            "-b", "8", "--temp", "0", "-t", "2", "--gpu", gpu,
                            "--no-tray"], cwd=ROOT, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=120)
        err = p.stderr.decode(errors="replace")
        assert p.returncode == 0, err[-2000:]
        return p.stdout, err

    out, err = gen()
    assert "adapter on device" in err, err[-2000:]
    notes = [l for l in err.splitlines() if "off by design" in l]
    assert not notes, notes
    decode = [l for l in err.splitlines() if l.startswith("route:") and " rows=1 " in l]
    assert decode and "graph=replay" in decode[-1], decode[-1:] or err[-2000:]
    assert out == gen(RUNNER_CUDA_GRAPH_OFF="1")[0]
    assert out == gen(gpu="off")[0]


def test_device_adapter_changes_the_score_deterministically(runner_bin, fx):
    _gpu_or_skip(runner_bin, fx)
    base = _score(runner_bin, fx["base"], gpu=True)
    a1 = _score(runner_bin, fx["base"], lora=fx["adapter"], gpu=True)
    a2 = _score(runner_bin, fx["base"], lora=fx["adapter"], gpu=True)
    assert a1 != base
    assert a1 == a2


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


# R6.68.116: an adapter that holds a NaN or an infinity is refused where it
# is read, by name, for every consumer of the loader. A NaN or an infinity in
# A or B is a NaN or infinite delta on every weight its row touches: served,
# it made every logit NaN (and --score exit 1, R6.68.115); merged, it wrote a
# model whose weights are not numbers, and the erased-delta floor let it pass.
NONFINITE = [("inf", b"lora_b' is not finite", b"infinity"),
             ("nan", b"lora_a' is not finite", b"NaN")]


def _refused(p, needle, what):
    err = p.stderr
    assert p.returncode != 0, err.decode(errors="replace")
    assert needle in err and what in err, err.decode(errors="replace")[-600:]


@pytest.mark.parametrize("bad,needle,what", NONFINITE)
def test_an_adapter_that_is_not_finite_is_refused_by_name(runner_bin, fx, tmp_path,
                                                          bad, needle, what):
    base, adapter = str(fx["base"]), str(fx[bad])
    cpu = ["-t", "2", "--gpu", "off"]
    run = lambda *a: subprocess.run([runner_bin, "-m", base, *a, *cpu], cwd=ROOT,  # noqa: E731
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=120)
    # served (--lora)
    _refused(run("--score", "-p", PROMPT, "--lora", adapter), needle, what)
    # a per-request adapter, at the server's startup
    _refused(run("--serve", "--port", "1", "--no-tray", "--adapter", f"t={adapter}"),
             needle, what)
    # a training run that continues from it, and one that resumes from it
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"prompt": "the runner", "completion": " trains"}) + "\n")
    _refused(run("--train", str(rows), "--train-steps", "1", "--lora", adapter,
                 "--train-out", str(tmp_path / "t.gguf")), needle, what)
    _refused(run("--train", str(rows), "--train-steps", "1", "--train-resume", adapter,
                 "--train-out", str(tmp_path / "r.gguf")), needle, what)
    # merged: refused, and neither the file nor its record is written
    merged = tmp_path / "merged.gguf"
    _refused(run("--lora", adapter, "--merge-lora", str(merged)), needle, what)
    assert not merged.exists() and not (tmp_path / "merged.gguf.merge.json").exists()
