"""Named contexts on a recurrent model on the device, CUDA or Metal (R4.48.22, R4.48.22.2).

A context pin (POST /v1/runner/contexts) and its import from a snapshot refused a recurrent model on a device
(qwen35 and qwen4exp, granite hybrid and nemotron-H on CUDA) since 2026-10-02, when the shared prefix tier could not
fork a device-resident fold. Since R4.26.7 the automatic tier forks one: model_recurrent_blob_save pulls the device
layers' fold through the host buffers and model_recurrent_blob_load pushes it back, so a pin refused what the tier
already served. No test held the automatic tier's device fork either; this one does.

The anchors, per family, every server on the device (its log says so, and RUNNER_GPU_FALLBACK=fail makes a fall to the
CPU an error):
- the automatic tier: a prompt P, then P + a question, forks P (prompt_reuse prefix_cache, P's tokens cached);
- a pin of P and a request naming it gives the automatic fork's tokens and logprobs bit for bit (both load the same
  snapshot, KV rows and fold, and feed the same question), and the text of a cold prefill of the same prompt (its
  logprobs to 1e-5: the cold prompt runs the question at another batch width);
- the pin exported to a snapshot file and imported into a new process gives the pin's bits again, every family
  (qwen4exp too since R4.48.22.1: a fold load rebuilds its PLE n-gram token history from the forked tokens).
The pin prefills its prompt in the shape a completion's prefill builds every recurrent fold in, the prompt but its last
token, then the last token alone (engine_feed_prompt, R4.48.22.2), so a pinned context is the automatic tier's state
on every backend and build by construction. Before, the pin fed one batch: equal on the CPU and a default CUDA build,
but not under a T3 build on CUDA (qwen4exp 4.8e-7 in its logprobs) nor on the Metal backend, where prefill chunks of 4
or more take the tiled GEMM, a tolerance class against the decode matvec by design (src/metal.m enc_mv_n), and the pin
answered differently from the automatic fork (nemotron-H 4.7e-7 in its logprobs, lfm2 more).
The cold prefill runs the whole prompt as one batch, so on Metal it is held to that backend's measured class (COLD_TOL).
Where a family runs is the backend's: lfm2's short convolution runs on the CPU under CUDA (arch.h cpu_only_nonmetal)
but on the device under Metal, as its --wide fixture quantized to Q4_K (the shape the Apple GPU path takes, R3.42), so
lfm2 is a case on Metal only; qwen4exp is not on the Metal backend yet (its load says so and runs it on the CPU), so
on Metal the test holds that load line and skips it.
The module skips without a CUDA or Metal device.
"""
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

SYS = ("You are a careful assistant for a warehouse inventory team. Answer "
       "with the item code first, then one short sentence. ")
QUESTION = "Where is item 42?"

# (id, generator, its arguments, its environment), as tests/test_cuda_rec_slots.py makes them
FAMILIES = [
    ("qwen35", "make-test-ornith.py", [], {}),
    ("qwen4exp", "make-test-qwen4exp.py", [], {}),
    ("granitehybrid", "make-test-hybrid.py", ["--dense"], {}),
    ("nemotron_h", "make-test-nemotron.py", [], {}),
    ("lfm2", "make-test-lfm2.py", ["--wide"], {}),
]
# a family the backend runs on the host is no device fold there. On Metal its load line is held and the case skips
# (when Metal takes the family the line is gone and the case fails: move it into the device cases); under CUDA lfm2's
# short convolution is the CPU's (arch.h cpu_only_nonmetal)
HOST_LINE = {("qwen4exp", "metal"): "(qwen4exp) is not on the metal backend yet"}
HOST_SKIP = {("lfm2", "cuda"): "lfm2's short convolution runs on the CPU under CUDA (arch.h cpu_only_nonmetal)"}
# a cold prefill's logprobs against the pinned fork's. On CUDA 1e-5 (the question at another batch width). On the
# Metal backend the cold prompt's one batch takes the tiled prefill GEMM where the fork's last rows take the matvec, a
# tolerance class by design (src/metal.m enc_mv_n). The house holds that class over teacher-forced logits
# (tests/test_tc_tol.c:58-60: top-1 agreement with a near-tie escape, mean |dlogit| <= 0.005 of the mean logit range,
# which reaches Metal's GEMM through gpu_tc_force), a quantity this test does not observe. So here its first gate is
# held as the text (top-1 at every generated position, equal), and the logprobs by a stated rule: 2x the worst gap
# measured over 5 repeated runs on an arm64 Mac, per weight type (every run gave the same values): F32 fixtures 6.67e-6
# (nemotron-H; qwen35 1.9e-6, granite hybrid 0), lfm2's Q4_K 3.49045e-3 (its activation tile staged at half).
METAL_COLD_WORST = {"f32": 6.67e-6, "q4_k": 3.49045e-3}
COLD_TOL = {"cuda": 1e-5, "metal": 2 * METAL_COLD_WORST["f32"]}
COLD_TOL_FAMILY = {("lfm2", "metal"): 2 * METAL_COLD_WORST["q4_k"]}


def _backend(runner_bin):
    caps = subprocess.run([str(runner_bin), "--caps"], cwd=ROOT, stdout=subprocess.PIPE, timeout=30)
    return (json.loads(caps.stdout.decode()).get("gpu") or {}).get("backend")


@pytest.fixture(scope="module")
def backend(runner_bin):
    b = _backend(runner_bin)
    if b not in ("cuda", "metal"):
        pytest.skip("needs a CUDA or Metal device")
    return b


@pytest.fixture(scope="module")
def models(runner_bin, backend, tmp_path_factory):
    d = tmp_path_factory.mktemp("recctx")
    out = {}
    for name, gen, args, env in FAMILIES:
        stem = d / name
        target = stem if gen == "make-test-hybrid.py" else stem.with_suffix(".gguf")
        subprocess.run([sys.executable, ROOT / "scripts" / gen, str(target), *args], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL, env={**os.environ, **env})
        path = stem.with_suffix(".gguf")
        if not path.exists():   # make-test-nemotron.py writes <arg>.gguf
            path = pathlib.Path(str(target) + ".gguf")
        if name == "lfm2":   # the wide fixture as Q4_K: the shape the Apple GPU path takes (R3.42)
            q = path.with_name("lfm2-q4k.gguf")
            subprocess.run([str(runner_bin), "-m", str(path), "--quantize", str(q), "--quant", "q4_k"], cwd=ROOT,
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            path = q
        out[name] = path
    return out


def _req(srv, method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(srv.base_url + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as r:
        return r.code, json.load(r)


def _serve(runner_bin, model, log, *extra):
    env = dict(os.environ, RUNNER_GPU_FALLBACK="fail")
    return RunnerServer(runner_bin, model, ctx=512, env=env, log_path=str(log),
                        extra_args=["--gpu", "99", "-t", "2", *extra])


def _complete(srv, prompt, **extra):
    st, b = _req(srv, "POST", "/v1/completions", {
        "prompt": prompt, "max_tokens": 6, "temperature": 0, "logprobs": 2, **extra})
    assert st == 200, b
    return b


def _ans(b):
    c = b["choices"][0]
    return c["text"], c["logprobs"]["token_logprobs"]


def _on_device(log, backend):
    text = pathlib.Path(log).read_text(errors="replace")
    if backend == "metal":
        assert "Metal backend" in text, text[-1500:]
    else:
        assert '"backend":"cuda"' in text or "CUDA backend" in text, text[-1500:]
    for bad in ("continuing on the CPU", "forward failed at runtime"):
        assert bad not in text, text[-1500:]


@pytest.mark.parametrize("name", [f[0] for f in FAMILIES])
def test_a_pinned_context_on_the_device_forks_as_the_automatic_tier_does(runner_bin, backend, models, name, tmp_path):
    model, prompt, logs = models[name], SYS + QUESTION, []
    if (name, backend) in HOST_SKIP:
        pytest.skip(HOST_SKIP[(name, backend)] + ": not a device fold there")
    if (name, backend) in HOST_LINE:
        log = tmp_path / "host.log"
        with _serve(runner_bin, model, log):
            pass
        assert HOST_LINE[(name, backend)] in log.read_text(errors="replace"), log.read_text(errors="replace")[-1500:]
        pytest.skip(f"{name} runs on the CPU under {backend} (its load says so): not a device fold there")
    # the automatic tier: P is published whole after its prefill, then P + the question is an exact full-prefix hit
    logs.append(tmp_path / "auto.log")
    with _serve(runner_bin, model, logs[-1]) as srv:
        _complete(srv, SYS)
        auto = _complete(srv, prompt)
    tel = auto["runner_telemetry"]
    assert tel["prompt_reuse"] == "prefix_cache", (name, tel)
    n_auto = tel["prompt_cached_tokens"]
    assert n_auto > 16, (name, tel)
    # cold: no tier at all
    logs.append(tmp_path / "cold.log")
    with _serve(runner_bin, model, logs[-1]) as srv:
        cold = _complete(srv, prompt, cache_prompt=False)
    assert cold["runner_telemetry"]["prompt_cached_tokens"] == 0, (name, cold["runner_telemetry"])
    # the pin, a request naming it, and its snapshot
    snaps = tmp_path / "snaps"
    logs.append(tmp_path / "pin.log")
    with _serve(runner_bin, model, logs[-1], "--kv-snapshots", str(snaps)) as srv:
        st, pin = _req(srv, "POST", "/v1/runner/contexts", {"id": "inv-sys", "prompt": SYS})
        assert st == 200, (name, st, pin)
        n = pin["tokens"]
        assert n == n_auto, (name, n, n_auto)
        # overwrite the slot's own rows and fold, so only the pinned snapshot can serve
        _complete(srv, "unrelated text that shares nothing")
        warm = _complete(srv, prompt, context_id="inv-sys")
        st, snap = _req(srv, "POST", "/v1/runner/contexts/inv-sys/snapshot", {})
        assert st == 200, (name, st, snap)
    tel = warm["runner_telemetry"]
    assert tel["context"] == {"id": "inv-sys", "tokens": n}, (name, tel)
    assert tel["prompt_cached_tokens"] == n and tel["prompt_reuse"] == "prefix_cache", (name, tel)
    assert _ans(warm) == _ans(auto), (name, _ans(warm), _ans(auto))
    assert _ans(warm)[0] == _ans(cold)[0], (name, _ans(warm), _ans(cold))
    tol = COLD_TOL_FAMILY.get((name, backend), COLD_TOL[backend])
    assert _ans(warm)[1] == pytest.approx(_ans(cold)[1], abs=tol), (name, tol, _ans(warm), _ans(cold))
    # the snapshot imported into a new process, on the device
    logs.append(tmp_path / "import.log")
    with _serve(runner_bin, model, logs[-1], "--kv-snapshots", str(snaps)) as srv:
        st, loaded = _req(srv, "POST", "/v1/runner/contexts", {"id": "inv-sys2", "snapshot": "inv-sys"})
        assert st == 200 and loaded["tokens"] == n, (name, st, loaded)
        imp = _complete(srv, prompt, context_id="inv-sys2")
    tel = imp["runner_telemetry"]
    assert tel["prompt_cached_tokens"] == n and tel["prompt_reuse"] == "prefix_cache", (name, tel)
    assert _ans(imp) == _ans(warm), (name, _ans(imp), _ans(warm))
    for log in logs:
        _on_device(log, backend)
