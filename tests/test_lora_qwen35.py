"""--lora on qwen35 (Qwen3.5/3.6/3.8 hybrids) and on clef, its backbone.

A qwen35 model interleaves Gated DeltaNet layers with full-attention ones.
An adapter can sit on a DeltaNet layer's five projections (attn_qkv,
attn_gate, ssm_beta, ssm_alpha, ssm_out: PEFT's in_proj_qkv / in_proj_z /
in_proj_b / in_proj_a / out_proj as llama.cpp's converter names them), on a
full-attention layer's q/k/v/output (q spans the query AND its output gate,
which share one projection), and on every FFN. The gates are the dense ones
(tests/test_lora.py), with the same absolute anchor:

  * a zero adapter scores BYTE-identical to the bare base;
  * base + adapter matches the MERGED reference, W += (alpha/r) B A computed
    by scripts/make-test-lora.py in Python, at every hooked site at once.
    Alpha 32 makes each site's share large: a build missing any one of the
    six new hooks misses the reference by 4e-4 to 3e-2 at alpha 8 and by
    about 1e-2 at alpha 32, against under 1e-6 with all of them;
  * a qwen35 adapter (what the converter writes from the backbone) serves on
    a clef file as the clef-declared one does;
  * the merge folds the same projections, and refuses them where serving
    does (afmoe's own attn_gate is not a DeltaNet projection);
  * the Mamba-2 hybrids still refuse an adapter.

Metal runs qwen35 on the CPU, so the device arm is CUDA's; it self-skips
elsewhere and is in CUDA_GATED_TESTS so a CUDA host cannot skip it.
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = "the quick brown fox jumps over the lazy dog"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def _gen(script, out):
    subprocess.run([sys.executable, ROOT / "scripts" / script, str(out)], check=True,
                   cwd=ROOT, stdout=subprocess.DEVNULL)


def _lora(base, prefix, **env):
    import os
    subprocess.run([sys.executable, ROOT / "scripts/make-test-lora.py", str(base),
                    str(prefix)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, LORA_TARGETS="qwen35", LORA_ALPHA="32", **env))


@pytest.fixture(scope="module", params=["qwen35", "clef"])
def fx(request, tmp_path_factory):
    d = tmp_path_factory.mktemp("lora-q35")
    base = d / "base.gguf"
    _gen("make-test-ornith.py" if request.param == "qwen35" else "make-test-clef.py", base)
    _lora(base, d / "fx")
    return {"arch": request.param, "dir": d, "base": base,
            "adapter": d / "fx.adapter.gguf", "zero": d / "fx.zero.gguf",
            "merged": d / "fx.merged.gguf"}


def _score(runner_bin, model, lora=None, gpu=False):
    cmd = [runner_bin, "-m", str(model), "--score", "-p", PROMPT, "-t", "2",
           "--gpu", "auto" if gpu else "off"]
    if lora:
        cmd += ["--lora", str(lora)]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    if gpu:
        assert b"CUDA backend" in p.stderr, p.stderr.decode(errors="replace")
    return json.loads(p.stdout)["logprobs"]


def _gap(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def test_zero_adapter_is_byte_identical_to_the_base(runner_bin, fx):
    assert _score(runner_bin, fx["base"], fx["zero"]) == _score(runner_bin, fx["base"])


def test_adapter_matches_the_merged_reference_at_every_site(runner_bin, fx):
    adapted = _score(runner_bin, fx["base"], fx["adapter"])
    assert _gap(adapted, _score(runner_bin, fx["merged"])) <= 1e-5
    assert _gap(adapted, _score(runner_bin, fx["base"])) > 1e-3   # it does something


def test_a_qwen35_adapter_serves_on_clef(runner_bin, fx, tmp_path):
    """llama.cpp's converter names the architecture from the backbone, so an
    adapter trained on Clef's Qwen3.5/3.8 backbone says qwen35."""
    if fx["arch"] != "clef":
        pytest.skip("the clef fixture carries this case")
    _lora(fx["base"], tmp_path / "q", LORA_ARCH="qwen35")
    assert (_score(runner_bin, fx["base"], tmp_path / "q.adapter.gguf")
            == _score(runner_bin, fx["base"], fx["adapter"]))


def test_an_adapter_reaches_the_clef_decision(runner_bin, fx):
    """The decision head reads the backbone's hidden rows, so an adapter on the
    backbone moves the joint head's scores (the head itself takes none)."""
    if fx["arch"] != "clef":
        pytest.skip("the clef fixture carries this case")
    text = (ROOT / "tests" / "fixtures" / "clef-records.jsonl").read_text(encoding="utf-8")
    rec = next(json.loads(x) for x in text.split("\n")
               if x.strip() and json.loads(x)["id"] == "many-questions")
    rec["runner_logits"] = True

    def decide(*extra):
        p = subprocess.run([runner_bin, "-m", str(fx["base"]), "--gpu", "off", "-c", "4096",
                            *extra, "--systemone", "-"], input=(json.dumps(rec) + "\n").encode(),
                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        assert p.returncode == 0, p.stderr.decode(errors="replace")
        return json.loads(p.stdout)["runner_logits"]

    bare = decide()
    assert decide("--lora", str(fx["zero"])) == bare
    assert decide("--lora", str(fx["adapter"])) != bare


def test_the_merge_folds_the_deltanet_projections(runner_bin, fx, tmp_path):
    out = tmp_path / "merged.gguf"
    p = subprocess.run([runner_bin, "-m", str(fx["base"]), "--lora", str(fx["adapter"]),
                        "--merge-lora", str(out), "--quant", "keep"], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert _gap(_score(runner_bin, out), _score(runner_bin, fx["merged"])) <= 1e-5


def test_attn_gate_is_hooked_only_on_qwen35(runner_bin, tmp_path):
    """afmoe carries an attn_gate of its own (an attention output gate) that
    no hook covers: serving and the merge both refuse an adapter on it."""
    import os
    _gen("make-test-moe.py", tmp_path / "m")
    base = tmp_path / "m.afmoe-plain.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-lora.py", str(base),
                    str(tmp_path / "g")], check=True, cwd=ROOT, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, LORA_ONLY="attn_gate"))
    ad = tmp_path / "g.adapter.gguf"
    for extra in (["-p", "hi", "-n", "1"],
                  ["--merge-lora", str(tmp_path / "x.gguf"), "--quant", "keep"]):
        p = subprocess.run([runner_bin, "-m", str(base), "--gpu", "off", "--lora", str(ad),
                            *extra], cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=120)
        assert p.returncode != 0
        assert "blk.0.attn_gate.weight.lora_a" in p.stderr.decode(errors="replace")


def test_mamba2_hybrids_still_refuse_an_adapter(runner_bin, tmp_path):
    _gen("make-test-hybrid.py", tmp_path / "hy")
    p = subprocess.run([runner_bin, "-m", str(tmp_path / "hy.gguf"), "--adapt-info"],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    info = json.loads(p.stdout)["adapter_serving"]
    assert info["supported"] is False and "Mamba-2" in info["reason"]


# ------------------------------------------------------------- the device arm
@pytest.fixture(scope="module")
def cuda(runner_bin):
    caps = json.loads(subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, timeout=60).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("no CUDA device: Metal runs qwen35 on the CPU, where the "
                    "gates above already hold")
    return True


def test_zero_adapter_is_byte_identical_on_cuda(runner_bin, cuda, fx):
    assert (_score(runner_bin, fx["base"], fx["zero"], gpu=True)
            == _score(runner_bin, fx["base"], gpu=True))


def test_cuda_adapter_matches_the_merged_reference(runner_bin, cuda, fx):
    adapted = _score(runner_bin, fx["base"], fx["adapter"], gpu=True)
    assert _gap(adapted, _score(runner_bin, fx["merged"], gpu=True)) <= 5e-4
    assert _gap(adapted, _score(runner_bin, fx["base"], gpu=True)) > 1e-3
