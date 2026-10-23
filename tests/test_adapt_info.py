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


def test_a_routed_expert_model_serves_and_trains_on_its_attention(fx):
    # R17.3 (2026-10-08): softmax-routed MoE trains an attention-only adapter
    # (the strict FD gate pins this fixture's backward, router term included)
    exe, d = fx
    info = _info(exe, d / "moe.afmoe-plain.gguf")
    assert info["adapter_serving"]["supported"] is True
    assert info["training"]["supported"] is True


def test_a_moe_router_outside_the_backward_does_not_train(fx):
    exe, d = fx
    info = _info(exe, d / "moe.ggroup.gguf")
    assert info["training"]["supported"] is False
    assert info["training"]["reason"] == "group-limited MoE routing"


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


def _banner(exe, model):
    p = subprocess.run([exe, "-m", str(model), "--adapt-info", "--no-tray"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    return p.stderr.decode(errors="replace")


def test_the_gemma4_banner_names_what_the_file_was_verified_on(fx):
    """The MoE variant printed the dense variant's line, a llama.cpp
    token-identity claim it never earned (lab, 2026-10-17)."""
    exe, d = fx
    err = _banner(exe, d / "moe.gemma4-moe.gguf")
    assert "gemma4: MoE variant" in err and "dense variant verified" not in err, err
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--arch", "gemma4",
                    str(d / "g4.gguf")], check=True, stdout=subprocess.DEVNULL)
    err = _banner(exe, d / "g4.gguf")
    assert "gemma4: dense variant verified" in err and "MoE variant" not in err, err


# R8.16: partial-dimension rope trains (the backward's rope transpose rotates
# the same rope_dim dims the forward does, gated in make test on a llama and a
# neox fixture). That refusal was what had kept stablelm out, whose LayerNorm
# the backward has no adjoint for, so LayerNorm is refused by name.
@pytest.mark.parametrize("arch", ["llama", "qwen3"])
def test_partial_dimension_rope_trains(fx, arch):
    exe, d = fx
    f = d / f"{arch}-rope8.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--arch", arch,
                    "--rope-dim", "8", str(f)], check=True, stdout=subprocess.DEVNULL)
    assert _info(exe, f)["training"] == {"supported": True, "reason": None, "host": "cpu"}


def test_layernorm_refuses_training_by_name(fx):
    exe, d = fx
    f = d / "stablelm.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--arch",
                    "stablelm", str(f)], check=True, stdout=subprocess.DEVNULL)
    info = _info(exe, f)
    assert info["adapter_serving"]["supported"] is True
    assert info["training"] == {"supported": False, "reason": "LayerNorm (stablelm)",
                                "host": "cpu"}
