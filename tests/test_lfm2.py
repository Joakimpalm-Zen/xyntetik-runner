"""lfm2 / lfm2moe (Liquid LFM2, LFM2.5): short-convolution layers interleaved
with GQA attention, typed per layer off the attention.head_count_kv array.

The arithmetic is checked against transformers' float32 reference on the
publisher's files (docs/compat-reports/lfm2.5-reference-2026-10-30/README.md);
these fixtures pin the binding: the dense and MoE files load and decode
deterministically, the convolution takes part and its kernel length is read,
a convolution bias and an all-attention file are refused, the GPU is not
used for it yet, and LoRA and the backward refuse it by name. The state
gates (a restored, rewound or prefix-forked ring decoding bit for bit as a
cold one) run in tests/test_recurrent_rewind.c on the same fixtures.
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _make(tmp_path, name, *extra):
    path = tmp_path / f"{name}.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-lfm2.py", str(path), *extra],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


def _run(runner_bin, model, *extra):
    return subprocess.run([runner_bin, "-m", model, "-p", "hello world, again and again",
                           "-n", "24", "--temp", "0", "-c", "256", *extra],
                          cwd=ROOT, capture_output=True, timeout=60)


def _decode(runner_bin, model):
    p = _run(runner_bin, model, "--gpu", "off")
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p.stdout


def test_dense_and_moe_load_and_decode_deterministically(runner_bin, tmp_path):
    for name, extra in (("lfm2", ()), ("lfm2moe", ("--moe",))):
        m = _make(tmp_path, name, *extra)
        a = _decode(runner_bin, m)
        assert a and a == _decode(runner_bin, m), name


def test_the_kernel_length_is_read(runner_bin, tmp_path):
    three = _decode(runner_bin, _make(tmp_path, "l3"))
    four = _decode(runner_bin, _make(tmp_path, "l4", "--l-cache", "4"))
    assert three != four, "a four-tap kernel decoded the same text: l_cache is ignored"


def test_a_convolution_bias_and_an_all_attention_file_are_refused(runner_bin, tmp_path):
    p = _run(runner_bin, _make(tmp_path, "bias", "--conv-bias"), "--gpu", "off")
    assert p.returncode != 0
    assert b"only conv_bias=false files are supported" in p.stderr
    p = _run(runner_bin, _make(tmp_path, "attn", "--all-attention"), "--gpu", "off")
    assert p.returncode != 0
    assert b"needs both attention and short-convolution layers" in p.stderr


def _backend(runner_bin):
    caps = json.loads(subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, timeout=60).stdout)
    return (caps.get("gpu") or {}).get("backend")


def _route(p):
    """The route line's fields (RUNNER_DEBUG_ROUTE=1), as a dict of the first line."""
    for line in p.stderr.decode(errors="replace").splitlines():
        if line.startswith("route: "):
            return dict(f.split("=", 1) for f in line[7:].split(" ") if "=" in f)
    return {}


def test_cuda_does_not_take_it_yet(runner_bin, tmp_path):
    # Only a host with a usable device reaches the routing that keeps lfm2 on
    # the CPU by name; a CUDA build without a driver stops at the probe with
    # its own line, as tests/test_remove_sublayer.py's GPU checks skip there.
    if _backend(runner_bin) != "cuda":
        pytest.skip("no CUDA device in this build or host")
    p = _run(runner_bin, _make(tmp_path, "auto"))
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert b"lfm2 is validated on the CPU and the Apple GPU only" in p.stderr


def test_the_apple_gpu_takes_it_where_the_layer_weights_are_q4_k(runner_bin, tmp_path):
    """R3.42: lfm2's Apple GPU path is the default only where its layer weights are mostly Q4_K
    by bytes, the shape measured to pass (LFM2.5-350M and 1.2B Q4_K_M); an F32 file, like the
    measured Q8_0 ones, stays on the CPU and the route line says why."""
    if _backend(runner_bin) != "metal":
        pytest.skip("no Metal device in this build or host")
    f32 = _make(tmp_path, "wide-f32", "--wide")
    q4k = tmp_path / "wide-q4_k.gguf"
    subprocess.run([runner_bin, "-m", f32, "--quantize", str(q4k), "--quant", "q4_k"], cwd=ROOT,
                   check=True, capture_output=True, timeout=120)
    env = {"RUNNER_DEBUG_ROUTE": "1"}
    import os
    for model, backend, said in ((f32, "cpu", True), (q4k, "metal", False)):
        p = subprocess.run([runner_bin, "-m", str(model), "-p", "hello world, again and again", "-n", "4",
                            "--temp", "0", "--no-tray"], cwd=ROOT, capture_output=True, timeout=60,
                           env=dict(os.environ, **env))
        assert p.returncode == 0, p.stderr.decode(errors="replace")
        why = b"lfm2 takes the Apple GPU only where its layer weights are mostly Q4_K, measured"
        assert (why in p.stderr) == said, (model, p.stderr.decode(errors="replace")[-600:])
        assert _route(p).get("backend") == backend, (model, _route(p))
    assert b"this file's are F32" in _run(runner_bin, f32).stderr


def test_lora_and_training_are_refused_by_name(runner_bin, tmp_path):
    m = _make(tmp_path, "lfm2")
    p = subprocess.run([runner_bin, "-m", m, "--adapt-info", "--no-tray"],
                       cwd=ROOT, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    info = json.loads(p.stdout)
    assert info["training"]["supported"] is False
    assert info["training"]["reason"] == "recurrent architecture (short convolution)"
    assert "short-convolution hybrid" in json.dumps(info)
