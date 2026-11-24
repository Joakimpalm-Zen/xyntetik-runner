"""The publisher's equations for an olmo2 GGUF, in float64 numpy, from the file's own weights (R4.50.1).

The absolute anchor of the olmo2 CPU gate (AGENTS.md section 3): the expected logits come from these equations, not
from another build of the runner. They are transcribed from transformers' modeling_olmo3 (v4.57.0, the version
allenai/Olmo-3-7B-Think was published with, and main, which agree) and modeling_rope_utils' _compute_yarn_parameters:

- the block normalises only AFTER attention and after the MLP: x + norm(attn(x)), then x + norm(mlp(x));
- the QK norm is one RMSNorm over the FULL projection width (q over n_head * head_dim, k over n_kv * head_dim),
  before the rotation;
- the rotation is NeoX's (rotate_half);
- sliding layers (attention.sliding_window_pattern true, or the default 3-in-4 when the array is absent) attend to
  the last W keys (key j for query i when i - W < j <= i) and rotate at the base theta with no scaling;
- full layers rotate with YaRN when the file declares it (beta_fast 32, beta_slow 1, the correction range floored and
  ceiled, then clamped to [0, dim - 1], the ramp over dim / 2), with cos and sin times the attention factor;
- SwiGLU, no biases, the final RMSNorm, an untied head.

The same script runs a plain llama file (pre-norms, no QK norm, interleaved rotation, no window, no scaling) so its
bound can be calibrated on an admitted family through the same arithmetic.

`mutants` names a wrong implementation the gate must tell apart: "qk_per_head" (each head normed on its own, with
its slice of the weight), "swa_yarn" (the sliding layers on the full layers' YaRN), "pre_norm" (an unweighted RMS
norm before attention and before the MLP, which is what a NULL pre-norm weight computes) and "window_off".
"""
import math
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gguf_fixture  # noqa: E402


def load(path, dtype=np.float64):
    with open(path, "rb") as f:
        kv, tensors, end = gguf_fixture.read_header_typed(f)
        align = kv.get("general.alignment", (4, 32))[1]
        base = (end + align - 1) // align * align
        f.seek(0)
        raw = f.read()
    w = {}
    for name, dims, ttype, off in tensors:
        if ttype != gguf_fixture.T_F32:
            raise ValueError(f"{name}: the reference reads F32 tensors only")
        n = int(np.prod(dims))
        a = np.frombuffer(raw, dtype="<f4", count=n, offset=base + off).astype(dtype)
        w[name] = a.reshape(list(reversed(dims))) if len(dims) > 1 else a
    return {k: v for k, (_t, v) in kv.items()}, w


def rms(x, w, eps):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if w is None else y * w


def yarn_inv_freq(dim, base, factor, orig, beta_fast=32.0, beta_slow=1.0, dtype=np.float64):
    def corr(rot):
        return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))
    low, high = max(math.floor(corr(beta_fast)), 0), min(math.ceil(corr(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    pos = dtype(base) ** (np.arange(0, dim, 2, dtype=dtype) / dtype(dim))
    ramp = np.clip((np.arange(dim // 2, dtype=dtype) - dtype(low)) / dtype(high - low), 0.0, 1.0)
    extra = 1.0 - ramp
    return (1.0 / (factor * pos)) * (1.0 - extra) + (1.0 / pos) * extra


def rotate(x, pos, inv_freq, scale, neox):
    """x [n, heads, hd] at positions pos [n]."""
    ang = pos[:, None] * inv_freq[None, :]
    c, s = np.cos(ang)[:, None, :] * scale, np.sin(ang)[:, None, :] * scale
    if neox:
        h = x.shape[-1] // 2
        a, b = x[..., :h], x[..., h:]
        return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2], out[..., 1::2] = a * c - b * s, b * c + a * s
    return out


def logprobs(path, tokens, mutants=frozenset(), dtype=np.float64):
    """Every position's log-softmax over the vocabulary, [n_tokens, n_vocab]. dtype np.float32 evaluates the same
    equations in float32: its distance from the float64 result is how far float32 rounding alone moves this file's
    logits, the scale the gate's bound is set against."""
    kv, w = load(path, dtype)
    arch = kv["general.architecture"]
    olmo = arch == "olmo2"
    if not olmo and arch != "llama":
        raise ValueError(f"reference covers olmo2 and llama, not {arch}")
    E, H, KVH, L = (kv[f"{arch}.{k}"] for k in ("embedding_length", "attention.head_count",
                                                  "attention.head_count_kv", "block_count"))
    hd = E // H
    eps = dtype(kv[f"{arch}.attention.layer_norm_rms_epsilon"])
    theta = kv.get(f"{arch}.rope.freq_base", 10000.0)
    window = kv.get(f"{arch}.attention.sliding_window", 0) if olmo else 0
    pattern = kv.get(f"{arch}.attention.sliding_window_pattern")
    sliding = [bool(window) and (pattern[i] if pattern is not None else (i + 1) % 4 != 0) for i in range(L)]
    plain = (dtype(1.0) / dtype(theta) ** (np.arange(0, hd, 2, dtype=dtype) / dtype(hd)), dtype(1.0))
    full = plain
    if kv.get(f"{arch}.rope.scaling.type") == "yarn":
        factor = kv[f"{arch}.rope.scaling.factor"]
        att = kv.get(f"{arch}.rope.scaling.attn_factor", 1.0 + 0.1 * math.log(factor))
        full = (yarn_inv_freq(hd, theta, factor, kv[f"{arch}.rope.scaling.original_context_length"], dtype=dtype),
                dtype(att))
    tok = np.asarray(tokens)
    n = len(tok)
    pos = np.arange(n, dtype=dtype)
    x = w["token_embd.weight"][tok]
    for i in range(L):
        p = f"blk.{i}."
        pre = (lambda v, nw: rms(v, w[p + nw + ".weight"], eps)) if not olmo else \
              ((lambda v, nw: rms(v, None, eps)) if "pre_norm" in mutants else (lambda v, nw: v))
        h = pre(x, "attn_norm")
        q, k, v = h @ w[p + "attn_q.weight"].T, h @ w[p + "attn_k.weight"].T, h @ w[p + "attn_v.weight"].T
        if olmo:
            qn, kn = w[p + "attn_q_norm.weight"], w[p + "attn_k_norm.weight"]
            if "qk_per_head" in mutants:
                q = rms(q.reshape(n, H, hd), qn.reshape(H, hd), eps).reshape(n, H * hd)
                k = rms(k.reshape(n, KVH, hd), kn.reshape(KVH, hd), eps).reshape(n, KVH * hd)
            else:
                q, k = rms(q, qn, eps), rms(k, kn, eps)
        inv, sc = full if (not sliding[i] or "swa_yarn" in mutants) else plain
        q = rotate(q.reshape(n, H, hd), pos, inv, sc, olmo)
        k = rotate(k.reshape(n, KVH, hd), pos, inv, sc, olmo)
        v = v.reshape(n, KVH, hd)
        rep = H // KVH
        o = np.empty((n, H, hd), dtype=dtype)
        for t in range(n):
            lo = t - window + 1 if (sliding[i] and "window_off" not in mutants) else 0
            lo = max(lo, 0)
            for hh in range(H):
                kk = k[lo:t + 1, hh // rep]
                s = kk @ q[t, hh] / dtype(math.sqrt(hd))
                s = np.exp(s - s.max())
                o[t, hh] = (s / s.sum()) @ v[lo:t + 1, hh // rep]
        a = o.reshape(n, H * hd) @ w[p + "attn_output.weight"].T
        x = x + (rms(a, w[p + "post_attention_norm.weight"], eps) if olmo else a)
        h = pre(x, "ffn_norm")
        g, u = h @ w[p + "ffn_gate.weight"].T, h @ w[p + "ffn_up.weight"].T
        f = (g / (dtype(1.0) + np.exp(-g)) * u) @ w[p + "ffn_down.weight"].T
        x = x + (rms(f, w[p + "post_ffw_norm.weight"], eps) if olmo else f)
    logits = rms(x, w["output_norm.weight"], eps) @ w["output.weight"].T
    m = logits.max(axis=-1, keepdims=True)
    out = logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))
    assert out.dtype == dtype, out.dtype
    return out


def worst_delta(score, ref):
    """The largest |logprob difference| over every (row, id) a runner --score --score-top record lists, row e being
    the distribution after tokens[0..e]."""
    worst = 0.0
    for e, row in enumerate(score["top"]):
        for tid, lp in row:
            worst = max(worst, abs(lp - ref[e][tid]))
    return worst


if __name__ == "__main__":
    import json
    rec = json.load(open(sys.argv[2]))
    print(worst_delta(rec, logprobs(sys.argv[1], rec["tokens"], frozenset(sys.argv[3:]))))
