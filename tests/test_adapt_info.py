"""`runner --adapt-info`: adapter eligibility of a model file as JSON (R8.9.1).

Inference, serving an adapter and training one are three admission lists.
The command reads the last two from the checks `--lora` and `--train` run, so
the tests hold three shapes: a dense model that takes both, a routed-expert
model that serves an adapter and cannot train one, and the gemma-4 dual-branch
model that refuses both, each with its reason.
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import find_runner  # noqa: E402


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    d = tmp_path_factory.mktemp("adapt")
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(d / "dense.gguf")],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(d / "moe")],
                   check=True, stdout=subprocess.DEVNULL)
    return exe, d


def _info(exe, model):
    p = subprocess.run([exe, "-m", str(model), "--adapt-info", "--no-tray"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(p.stdout)


def test_a_dense_model_serves_and_trains(fx):
    exe, d = fx
    info = _info(exe, d / "dense.gguf")
    assert info["architecture"] == "llama"
    assert info["adapter_serving"]["supported"] is True
    assert info["adapter_serving"]["reason"] is None
    assert info["training"] == {"supported": True, "reason": None, "host": "cpu"}


def test_a_routed_expert_model_serves_but_does_not_train(fx):
    exe, d = fx
    info = _info(exe, d / "moe.afmoe-plain.gguf")
    assert info["adapter_serving"]["supported"] is True
    assert info["training"]["supported"] is False
    assert info["training"]["reason"] == "MoE FFN"


def test_the_refusal_is_the_loaders_own_sentence(fx):
    exe, d = fx
    model = d / "moe.gemma4-moe-hetero.gguf"
    info = _info(exe, model)
    assert info["adapter_serving"]["supported"] is False
    assert info["adapter_serving"]["device_kernels"] is None
    reason = info["adapter_serving"]["reason"]
    assert "gemma-4" in reason
    # and --lora says the same thing when it is actually asked
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(d / "b.gguf")],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, ROOT / "scripts/make-test-lora.py",
                    str(d / "b.gguf"), str(d / "fx")], check=True, stdout=subprocess.DEVNULL)
    p = subprocess.run([exe, "-m", str(model), "--lora", str(d / "fx.adapter.gguf"),
                        "-p", "hi", "-n", "1", "--gpu", "off", "--no-tray"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode != 0
    assert reason in p.stderr.decode(errors="replace")
