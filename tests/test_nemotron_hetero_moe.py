"""Per-layer expert geometry on nemotron_h_moe (Nemotron 3.5 Bolt, R4.27.4.4 S2).

Bolt's MoE blocks differ in expert count (the router's shape gives it) and in
expert width: expert_feed_forward_length and expert_used_count are per-layer
arrays, 0 on the blocks without experts. Each block runs its experts at its own
width; the widest sizes the scratch.

The anchor is a zero-padded twin, not Runner: scripts/make-test-hybrid.py
--hetero writes two MoE blocks (4 experts 16 wide, 6 experts 24 wide) and
<out>.padded.gguf, the same draws with every expert padded with zeros to 24 and
the two keys as scalars. relu(0)^2 is 0 and a zero column of down adds nothing,
so the two compute the same function; the twin runs the uniform-width path.
Their per-position logprobs agree to f32 rounding on the one-token pass and on
a batched pass, with the experts at the latent width (Bolt's layout) and at
n_embd. A block run at another block's width shows as a disagreement, or as a
shape refusal. Then the loader's refusals of a malformed per-layer layout.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = "The capital of France is Paris, and of Italy"
# f32 rounding of a regrouped sum, in logprob units, with the margin
# tests/test_nemotron_latent_moe.py keeps for other compilers' vectorization
TOL = 1e-4


def make(d, name, latent, brk=None):
    env = dict(os.environ, HYBRID_TEST_AMP="0.6")
    if brk:
        env["HYBRID_TEST_HETERO_BREAK"] = brk
    pfx = d / name
    subprocess.run([sys.executable, ROOT / "scripts/make-test-hybrid.py", str(pfx),
                    "--arch", "nemotron_h_moe", "--hetero",
                    *(["--latent", str(latent)] if latent else [])],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL, env=env)
    return pfx.with_suffix(".gguf"), d / (name + ".padded.gguf")


@pytest.fixture(scope="module", params=[16, 0], ids=["latent", "plain"])
def pair(request, tmp_path_factory):
    return request.param, make(tmp_path_factory.mktemp("hetero"), "het", request.param)


def score(runner_bin, model, chunked=False):
    env = dict(os.environ, RUNNER_SCORE_CHUNKED="1" if chunked else "0")
    p = subprocess.run([str(runner_bin), "-m", str(model), "-p", PROMPT, "--score", "8",
                        "--gpu", "off", "--no-tray"], env=env,
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(p.stdout), p.stderr.decode(errors="replace")


def max_diff(a, b):
    assert len(a["logprobs"]) == len(b["logprobs"]) > 30
    return max(abs(x - y) for x, y in zip(a["logprobs"], b["logprobs"]))


@pytest.mark.parametrize("chunked", [False, True], ids=["token-pass", "batched-pass"])
def test_the_hetero_model_equals_its_padded_twin(runner_bin, pair, chunked):
    latent, (het, pad) = pair
    a, err = score(runner_bin, het, chunked)
    b, perr = score(runner_bin, pad, chunked)
    if latent:
        assert "run at width 16 (n_embd 32), 16 to 24 wide by layer" in err, err
    else:
        assert "the routed experts run 16 to 24 wide by layer" in err, err
    assert "wide by layer" not in perr, perr
    assert max_diff(a, b) <= TOL
    assert a["argmax"] == b["argmax"]


def test_the_batched_pass_equals_the_token_pass(runner_bin, pair):
    _, (het, _) = pair
    assert max_diff(score(runner_bin, het)[0], score(runner_bin, het, True)[0]) <= TOL


def test_the_twin_is_a_test_the_narrow_block_moves_the_logits(runner_bin, pair, tmp_path):
    """The comparison above means something for the narrower block only if its
    experts reach the logits: the same model with that block's experts zeroed
    must read far outside the tolerance."""
    latent, (het, _) = pair
    zero, _ = make(tmp_path, "zero", latent, "zero-narrow")
    assert max_diff(score(runner_bin, het)[0], score(runner_bin, zero)[0]) > 1000 * TOL


@pytest.mark.parametrize("brk,needle", [
    ("topk", "expert_used_count differs between layers (2 to 3): a per-layer top-k has no path here"),
    ("nonzero", "blk.0 has no experts, but expert_used_count 1 and expert_feed_forward_length 0"),
    ("short-array", "expert_used_count has 4 entries for 5 layers"),
    ("width", "tensor ffn_up_exps in blk.4 has shape"),
])
def test_a_broken_per_layer_layout_is_refused_by_name(runner_bin, tmp_path, brk, needle):
    het, _ = make(tmp_path, "brk", 16, brk)
    p = subprocess.run([str(runner_bin), "-m", str(het), "-p", "hi", "-n", "1",
                        "--gpu", "off", "--no-tray"],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    err = p.stderr.decode(errors="replace")
    assert p.returncode != 0, err
    assert needle in err, err


def test_varying_widths_on_cuda_take_no_split_and_name_the_refusal(runner_bin, tmp_path):
    """Every CUDA expert table takes the model's one width, so a file whose
    widths vary runs on the CPU and says so, and takes no default split (whose
    line would say what never runs). The plain hetero file reaches this rule;
    a latent one is refused earlier, by the LatentMoE rule."""
    caps = json.loads(subprocess.run([str(runner_bin), "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, check=True, text=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("the refusal is CUDA-side")
    het, _ = make(tmp_path, "cuda", 0)
    p = subprocess.run([str(runner_bin), "-m", str(het), "-p", "hi", "-n", "1", "--no-tray"],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    err = p.stderr.decode(errors="replace")
    assert p.returncode == 0, err
    assert "per-layer expert widths (blk.2 is 16 wide, the widest 24)" in err, err
    assert "its experts and their routing stay on the host" not in err, err


@pytest.mark.parametrize("latent,why", [
    (16, "LatentMoE (the routed experts at a latent width between ffn_latent_down and "
         "ffn_latent_up) has no device path"),
    (0, "per-layer expert widths have no device path"),
])
def test_the_fit_says_such_a_file_loads_on_the_cpu(runner_bin, tmp_path, latent, why):
    """--info fit sizes what the load does: a LatentMoE file, or one whose
    widths vary, loads on the CPU (cuda_admit refuses it by name), so the fit
    says so and sizes no split. Before R4.27.4.4 S2 it reported every layer
    on the GPU."""
    caps = json.loads(subprocess.run([str(runner_bin), "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, check=True, text=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("the fit's GPU lines need a device")
    het, _ = make(tmp_path, "fit", latent)
    p = subprocess.run([str(runner_bin), "--info", "fit", "-m", str(het)], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, text=True)
    assert p.returncode == 0, p.stderr
    assert "present, but the load runs on the CPU: " + why in p.stdout, p.stdout
    assert "layers on the GPU" not in p.stdout, p.stdout
