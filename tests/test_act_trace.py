"""RUNNER_ACT_TRACE (R4.11, RO-7): an activation trace that localizes a
difference the final output does not.

The mismatch fixture is the shape that started RO-7: an edit to a non-FFN
tensor that an export silently dropped. make-test-model.py writes the plain
toy model and a copy whose layer 1 attention output projection is scaled by
1.1, every other byte equal. Both answer the prompt with the same greedy text,
and their logits differ without saying where. scripts/act-trace-diff.py must
name layer 1's mixer branch, with the embedding and all of layer 0 matching
bit for bit, and the same with valued traces under a tolerance. Scaling layer
1's FFN down projection instead must be named as layer 1's FFN branch (its
mixer point matches). A trace of a file against itself matches. The trace is
bounded: it needs an explicit position list of at most 64 positions, and
without the variable nothing is written or said. A MoE layer's selection is
traced at the listed positions.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = "hello world foo bar"
DIFF = ROOT / "scripts/act-trace-diff.py"


def make(tmp, name, *extra):
    out = tmp / f"{name}.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(out), *extra],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return out


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("act")
    return {"plain": make(tmp, "plain"),
            "attn": make(tmp, "attn", "--scale-tensor", "blk.1.attn_output.weight:1.1"),
            "ffn": make(tmp, "ffn", "--scale-tensor", "blk.1.ffn_down.weight:1.1")}


def run(runner_bin, model, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("RUNNER_ACT_TRACE")}
    e.update({k: str(v) for k, v in env.items()})
    p = subprocess.run([str(runner_bin), "-m", str(model), "-p", PROMPT, "-n", "4",
                        "--temp", "0", "--gpu", "off", "-t", "2"],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=120, env=e)
    return p.returncode, p.stdout, p.stderr.decode(errors="replace")


def trace(runner_bin, model, path, pos="0,3,28,29", values=False):
    env = {"RUNNER_ACT_TRACE": path, "RUNNER_ACT_TRACE_POS": pos}
    if values:
        env["RUNNER_ACT_TRACE_VALUES"] = "1"
    rc, out, err = run(runner_bin, model, **env)
    assert rc == 0, err[-400:]
    return out, [json.loads(l) for l in pathlib.Path(path).read_text().splitlines()]


def diff(a, b, *extra):
    p = subprocess.run([sys.executable, DIFF, str(a), str(b), *extra],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    return p.returncode, p.stdout


def test_a_dropped_attention_edit_is_named_where_the_output_is_silent(runner_bin, models, tmp_path):
    out_p, plain = trace(runner_bin, models["plain"], tmp_path / "p.jsonl", values=True)
    out_a, attn = trace(runner_bin, models["attn"], tmp_path / "a.jsonl", values=True)
    # the output does not show it: same greedy text, the same top-1 everywhere
    # the head ran, logits that differ but say nothing about where
    assert out_p == out_a
    lp = [r for r in plain if r["point"] == "logits"]
    la = [r for r in attn if r["point"] == "logits"]
    assert lp and len(lp) == len(la)
    for x, y in zip(lp, la):
        assert x["hash"] != y["hash"]
        assert max(range(x["n"]), key=x["values"].__getitem__) == \
               max(range(y["n"]), key=y["values"].__getitem__)
    # the trace does: everything before layer 1's mixer is bit-identical
    key = lambda r: (r["pos"], r["layer"], r["point"])
    hp, ha = {key(r): r["hash"] for r in plain}, {key(r): r["hash"] for r in attn}
    assert set(hp) == set(ha)
    for k in hp:
        if k[1] < 1:
            assert hp[k] == ha[k], k
    for vals in (False, True):
        a = tmp_path / f"a{vals}.jsonl"
        p = tmp_path / f"p{vals}.jsonl"
        trace(runner_bin, models["attn"], a, values=vals)
        trace(runner_bin, models["plain"], p, values=vals)
        rc, text = diff(a, p)
        assert rc == 1, text
        assert "first difference: pos 0 layer 1 mixer" in text, text
        assert "layer 1's mixer branch" in text, text


def test_an_ffn_edit_is_named_as_the_ffn_branch(runner_bin, models, tmp_path):
    trace(runner_bin, models["ffn"], tmp_path / "f.jsonl")
    trace(runner_bin, models["plain"], tmp_path / "p.jsonl")
    rc, text = diff(tmp_path / "f.jsonl", tmp_path / "p.jsonl")
    assert rc == 1 and "first difference: pos 0 layer 1 resid" in text, text
    assert "layer 1's FFN branch" in text, text


def test_a_file_matches_itself_and_values_tolerate_nothing_else(runner_bin, models, tmp_path):
    trace(runner_bin, models["plain"], tmp_path / "x.jsonl", values=True)
    trace(runner_bin, models["plain"], tmp_path / "y.jsonl", values=True)
    rc, text = diff(tmp_path / "x.jsonl", tmp_path / "y.jsonl", "--rtol", "0")
    assert rc == 0 and "every common point matches" in text, text
    # a hashed side and a valued side are not comparable: refused, not guessed
    trace(runner_bin, models["plain"], tmp_path / "h.jsonl")
    rc, text = diff(tmp_path / "h.jsonl", tmp_path / "x.jsonl")
    assert rc == 2, text


def test_the_trace_is_bounded_and_off_by_default(runner_bin, models, tmp_path):
    rc, plain, err = run(runner_bin, models["plain"])
    assert rc == 0 and "act-trace" not in err
    # positions are required
    rc, _, err = run(runner_bin, models["plain"], RUNNER_ACT_TRACE=tmp_path / "t.jsonl")
    assert rc == 1 and "RUNNER_ACT_TRACE_POS" in err
    # at most 64 of them
    many = ",".join(str(i) for i in range(65))
    rc, _, err = run(runner_bin, models["plain"], RUNNER_ACT_TRACE=tmp_path / "u.jsonl",
                     RUNNER_ACT_TRACE_POS=many)
    assert rc == 1 and "1 to 64" in err
    # the trace changes nothing it watches
    out, recs = trace(runner_bin, models["plain"], tmp_path / "v.jsonl", pos="2")
    assert out == plain
    assert {r["pos"] for r in recs} == {2}
    assert all(r["pass"] >= 1 for r in recs)
    n_layer = max(r["layer"] for r in recs if r["point"] == "resid") + 1
    pts = [(r["layer"], r["point"]) for r in recs]
    assert pts[0] == (-1, "embd")
    for l in range(n_layer):
        assert (l, "mixer") in pts and (l, "resid") in pts


def test_a_moe_layer_traces_its_selection(runner_bin, tmp_path):
    base = tmp_path / "m"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(base)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    moe = pathlib.Path(str(base) + ".moe1.gguf")
    _, recs = trace(runner_bin, moe, tmp_path / "m.jsonl", pos="1")
    routes = [r for r in recs if r["point"] == "route"]
    assert routes and all(r["pos"] == 1 and r["experts"] == [0] for r in routes)
    _, again = trace(runner_bin, moe, tmp_path / "n.jsonl", pos="1")
    rc, text = diff(tmp_path / "m.jsonl", tmp_path / "n.jsonl")
    assert rc == 0, text


# ---------------------------------------------------------------- the reference side's mixer point (R6.38)
#
# scripts/act-trace-hf.py hooks the norm in front of a block's FFN: its input is
# the residual after the mixer branch was added, which is Runner's mixer point.
# Which norm that is depends on the block's layout. On gemma-family blocks
# (sandwich norms) post_attention_layernorm normalizes the attention OUTPUT
# before the residual add, so its input is the branch; the script took it
# anyway, being the first of its names it found, until 2026-11-10.

def _act_trace_hf():
    import importlib.util
    spec = importlib.util.spec_from_file_location("act_trace_hf", ROOT / "scripts/act-trace-hf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Block:
    def __init__(self, *names):
        for n in names:
            setattr(self, n, n)


@pytest.mark.parametrize("names,want", [
    # Gemma 2, 3 and 4 (transformers' Gemma*DecoderLayer): sandwich norms
    (("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
      "post_feedforward_layernorm"), "pre_feedforward_layernorm"),
    # Llama, Qwen, Granite: pre-norm, post_attention_layernorm is the FFN's norm
    (("input_layernorm", "post_attention_layernorm"), "post_attention_layernorm"),
    (("attention_norm", "ffn_norm"), "ffn_norm"),
    # OLMo 2: post-norm; no norm takes the residual after the add as its input
    (("post_attention_layernorm", "post_feedforward_layernorm"), None),
    (("input_layernorm",), None),
])
def test_the_mixer_point_follows_the_block_norm_layout(names, want):
    assert _act_trace_hf().mixer_norm(_Block(*names)) == want


def test_the_mixer_point_on_a_real_gemma_block_is_the_residual_after_attention():
    """The absolute anchor is transformers' own Gemma 3 forward: the residual after
    the attention add is the block input plus the normalized attention output, read
    from the forward's own modules, and the hooked norm's input must equal it."""
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    cfg = tr.Gemma3TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                              head_dim=16, max_position_embeddings=32, sliding_window=16)
    torch.manual_seed(0)
    model = tr.Gemma3ForCausalLM(cfg).eval()
    blk = model.model.layers[0]
    seen = {}
    blk.register_forward_pre_hook(lambda m, a, k: seen.__setitem__(
        "x", (a[0] if a else k["hidden_states"]).detach().clone()), with_kwargs=True)
    blk.post_attention_layernorm.register_forward_hook(
        lambda m, a, out: seen.__setitem__("branch_in", a[0].detach().clone()) or
        seen.__setitem__("branch_out", out.detach().clone()))
    _act_trace_hf().mixer_norm(blk).register_forward_pre_hook(
        lambda m, a: seen.__setitem__("mixer", a[0].detach().clone()))
    with torch.no_grad():
        model(torch.tensor([[1, 5, 9, 2]]))
    want = seen["x"] + seen["branch_out"]
    assert torch.allclose(seen["mixer"], want, atol=0, rtol=0)
    assert not torch.allclose(seen["mixer"], seen["branch_in"])  # the point it used to take
