"""deepseek2's CPU path against an independent reference of the publisher's math (R4.4.5).

A two-layer multi-head latent attention fixture (make-test-model.py --arch deepseek2: four heads of 8 nope + 8 rope
dims, an 8-wide value, a 16-wide latent, YaRN at factor 40) is scored by the runner on the CPU, and the same weights
run through a plain-Python forward written the way modeling_deepseek.py states it: Q per head [q_nope | q_pe], one
latent [kv_lora] and one rope key per token from attn_kv_a_mqa, the latent RMS-normed, kv_b decompressing it per head
into [k_nope | v], rope on the pe halves (consecutive pairs, the publisher's permute-then-rotate_half), the softmax
scale mscale^2 / sqrt(key_length) with mscale = 1 + 0.1 * mscale_all_dim * ln(factor). The runner's layout tricks
(heads as [pe | nope], V padded with zeros, the value width packed before attn_output) do not appear here, so a
mistake in them shows as a difference.
"""
import json
import math
import os
import pathlib
import re
import struct
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gguf_fixture import read_header  # noqa: E402

TEXT = "the quick brown fox jumps over the lazy dog"
# 176 tokens: past a 64-row chunk, so a forward carries the scratch's full batch
LONG = ("the quick brown fox jumps over the lazy dog and then the dog sleeps under the old oak tree by the "
        "river while the fox runs")


def _score(exe, model, out, naive):
    env = dict(os.environ, RUNNER_MLA_NAIVE="1" if naive else "0")
    p = subprocess.run([str(exe), "-m", str(model), "--gpu", "off", "-t", "2", "-c", "256", "--score", "-p", TEXT,
                        "--score-out", str(out), "--no-tray"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    d = tmp_path_factory.mktemp("ds2")
    model = d / "ds2.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make-test-model.py"), "--arch", "deepseek2", str(model)],
                   check=True, stdout=subprocess.DEVNULL)
    # the absorbed form (the default) and the naive one (RUNNER_MLA_NAIVE=1)
    return model, {False: _score(exe, model, d / "absorbed.json", False), True: _score(exe, model, d / "naive.json", True)}


def load(model):
    with open(model, "rb") as f:
        kv, tensors, end = read_header(f)
        align = kv.get("general.alignment", 32)
        base = (end + align - 1) // align * align
        w = {}
        for name, dims, gtype, off in tensors:
            assert gtype == 0, name                      # F32 throughout
            n = 1
            for x in dims:
                n *= x
            f.seek(base + off)
            w[name] = (list(dims), struct.unpack("<%df" % n, f.read(4 * n)))
    return kv, w


def matvec(t, x):
    (n_in, n_out), data = t[0][:2], t[1]
    return [sum(data[o * n_in + i] * x[i] for i in range(n_in)) for o in range(n_out)]


def rmsnorm(x, wt, eps):
    r = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v * r * g for v, g in zip(x, wt)]


def yarn_inv_freq(dim, base, factor, orig, beta_fast=32.0, beta_slow=1.0):
    # the publisher's yarn_find_correction_range (floor and ceil) and linear ramp
    def corr(rot):
        return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))
    lo, hi = max(math.floor(corr(beta_fast)), 0), min(math.ceil(corr(beta_slow)), dim - 1)
    out = []
    for j in range(dim // 2):
        extra = base ** (-2.0 * j / dim)
        ramp = min(1.0, max(0.0, (j - lo) / max(hi - lo, 0.001)))
        keep = 1.0 - ramp
        out.append(extra * keep + extra / factor * (1.0 - keep))
    return out


def rope(v, pos, inv):
    v = list(v)
    for j, fr in enumerate(inv):
        c, s = math.cos(pos * fr), math.sin(pos * fr)
        a, b = v[2 * j], v[2 * j + 1]
        v[2 * j], v[2 * j + 1] = a * c - b * s, a * s + b * c
    return v


def reference_logprobs(kv, w, tokens):
    A = "deepseek2"
    n_head = kv[A + ".attention.head_count"]
    hd = kv[A + ".attention.key_length"]
    vd = kv[A + ".attention.value_length"]
    r = kv[A + ".attention.kv_lora_rank"]
    rd = kv[A + ".rope.dimension_count"]
    nope = hd - rd
    eps = kv[A + ".attention.layer_norm_rms_epsilon"]
    factor = kv[A + ".rope.scaling.factor"]
    log_mul = kv[A + ".rope.scaling.yarn_log_multiplier"]
    inv = yarn_inv_freq(rd, kv[A + ".rope.freq_base"], factor, kv[A + ".rope.scaling.original_context_length"])
    mscale = 1.0 + 0.1 * (log_mul / 0.1) * math.log(factor)      # yarn_get_mscale(factor, mscale_all_dim)
    scale = mscale * mscale / math.sqrt(hd)
    n_layer = kv[A + ".block_count"]
    n_embd = kv[A + ".embedding_length"]
    emb = w["token_embd.weight"][1]
    xs = [list(emb[t * n_embd:(t + 1) * n_embd]) for t in tokens]
    for l in range(n_layer):
        g = lambda name: w["blk.%d.%s.weight" % (l, name)]     # noqa: E731
        qs, ks, vs = [], [], []
        for pos, x in enumerate(xs):
            xn = rmsnorm(x, g("attn_norm")[1], eps)
            q = matvec(g("attn_q"), xn)
            kva = matvec(g("attn_kv_a_mqa"), xn)
            lat = rmsnorm(kva[:r], g("attn_kv_a_norm")[1], eps)
            kpe = rope(kva[r:], pos, inv)
            kvo = matvec(g("attn_kv_b"), lat)
            qs.append([(q[h * hd:h * hd + nope], rope(q[h * hd + nope:(h + 1) * hd], pos, inv)) for h in range(n_head)])
            ks.append([(kvo[h * (nope + vd):h * (nope + vd) + nope], kpe) for h in range(n_head)])
            vs.append([kvo[h * (nope + vd) + nope:(h + 1) * (nope + vd)] for h in range(n_head)])
        new = []
        for pos, x in enumerate(xs):
            att = []
            for h in range(n_head):
                qn, qp = qs[pos][h]
                sc = [(sum(a * b for a, b in zip(qn, ks[t][h][0])) + sum(a * b for a, b in zip(qp, ks[t][h][1]))) * scale
                      for t in range(pos + 1)]
                m = max(sc)
                e = [math.exp(s - m) for s in sc]
                z = sum(e)
                att += [sum(e[t] / z * vs[t][h][d] for t in range(pos + 1)) for d in range(vd)]
            o = matvec(g("attn_output"), att)
            x = [a + b for a, b in zip(x, o)]
            xn = rmsnorm(x, g("ffn_norm")[1], eps)
            gt, up = matvec(g("ffn_gate"), xn), matvec(g("ffn_up"), xn)
            h_ = [gv / (1.0 + math.exp(-gv)) * uv for gv, uv in zip(gt, up)]
            x = [a + b for a, b in zip(x, matvec(g("ffn_down"), h_))]
            new.append(x)
        xs = new
    out = []
    head = w.get("output.weight", w["token_embd.weight"])
    for pos in range(len(tokens) - 1):
        xn = rmsnorm(xs[pos], w["output_norm.weight"][1], eps)
        lg = matvec(head, xn)
        m = max(lg)
        lse = m + math.log(sum(math.exp(v - m) for v in lg))
        out.append(lg[tokens[pos + 1]] - lse)
    return out


@pytest.mark.parametrize("naive", [False, True], ids=["absorbed", "naive"])
def test_mla_matches_the_publishers_math(fixture, naive):
    model, scores = fixture
    score = scores[naive]
    kv, w = load(model)
    tokens = score["tokens"]
    ref = reference_logprobs(kv, w, tokens)
    got = score["logprobs"]
    assert len(got) == len(ref)
    worst = max(abs(a - b) for a, b in zip(got, ref))
    # measured 4.0e-5 on 62 positions; the same reference with transformers' built-in scale (no mscale^2) reads
    # 2.1e-3, so the bound separates the two by 4x on each side
    assert worst < 5e-4, "max |runner - reference| logprob %.3g over %d positions" % (worst, len(ref))


def test_absorbed_equals_naive(fixture):
    # the same function re-associated: q.k = q_pe.k_pe + (k_b^T q_nope).latent, and
    # v_b applied after the weighted sum of latents rather than before
    _, scores = fixture
    a, b = scores[False], scores[True]
    assert a["tokens"] == b["tokens"]
    worst = max(abs(x - y) for x, y in zip(a["logprobs"], b["logprobs"]))
    assert worst < 1e-4, "absorbed against naive: max |dlogprob| %.3g" % worst


def test_a_full_batch_prefill_in_both_forms(fixture):
    # a prompt prefill feeds forwards of the scratch's full batch (n_batch
    # rows): the absorbed form once computed kv_b's width from its 576-wide
    # head, put qa and qr past the MLA scratch whenever a forward carried more
    # than about half the batch, and aborted on glibc's heap check here
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    model, _ = fixture
    out = {}
    for naive in (False, True):
        env = dict(os.environ, RUNNER_MLA_NAIVE="1" if naive else "0")
        p = subprocess.run([str(exe), "-m", str(model), "--gpu", "off", "-t", "2", "-c", "256", "-p", LONG,
                            "-n", "8", "--temp", "0", "--no-tray"], env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        err = p.stderr.decode(errors="replace")
        assert p.returncode == 0, err[-2000:]
        assert re.search(r"prompt: (\d+) tok", err) and int(re.search(r"prompt: (\d+) tok", err).group(1)) > 128, err
        out[naive] = p.stdout
    assert out[False] == out[True]


def test_the_family_template_and_refusals(fixture):
    model, _ = fixture
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    info = subprocess.run([str(exe), "--info", "adapter", "-m", str(model)], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    rec = json.loads(info.stdout)
    assert rec["architecture"] == "deepseek2"
    assert rec["adapter_serving"]["supported"] is False and "latent" in rec["adapter_serving"]["reason"]
    assert rec["training"]["supported"] is False
