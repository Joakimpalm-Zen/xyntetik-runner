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


def test_the_gpu_is_not_used_for_it_yet(runner_bin, tmp_path):
    # Only a host with a usable device reaches the routing that refuses lfm2
    # by name; a CUDA build without a driver stops at the probe with its own
    # line, as tests/test_remove_sublayer.py's GPU checks skip there.
    caps = json.loads(subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, timeout=60).stdout)
    if (caps.get("gpu") or {}).get("backend") not in ("cuda", "metal"):
        pytest.skip("no CUDA or Metal GPU device in this build or host")
    p = _run(runner_bin, _make(tmp_path, "auto"))
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert b"lfm2 is validated on the CPU only" in p.stderr


def test_lora_and_training_are_refused_by_name(runner_bin, tmp_path):
    m = _make(tmp_path, "lfm2")
    p = subprocess.run([runner_bin, "-m", m, "--adapt-info", "--no-tray"],
                       cwd=ROOT, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    info = json.loads(p.stdout)
    assert info["training"]["supported"] is False
    assert info["training"]["reason"] == "recurrent architecture (short convolution)"
    assert "short-convolution hybrid" in json.dumps(info)
