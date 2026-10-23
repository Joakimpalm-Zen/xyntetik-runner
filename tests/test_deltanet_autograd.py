"""The qwen35 backward against PyTorch autograd, element by element (R8.16).

tests/test_lora_grad.c holds the backward to central differences of the
engine's own loss; this holds it to an independent derivative. The fixture
(scripts/make-test-ornith.py: three Gated DeltaNet layers and one
full-attention layer, every weight F32) is read here and its forward written
again in float64 from the model's definition, with the full-coverage adapter
on every projection: per DeltaNet layer the five DeltaNet projections and the
FFN, on the attention layer the fused Q/gate, K, V, output and the FFN. Torch
differentiates it; the runner's gradient buffers are dumped by test-lora-grad
(RUNNER_GRAD_DUMP) for the same 14 tokens, and every buffer must match.

What the reference copies from the engine is the representation, not the
arithmetic: K and V are rounded to f16 as the cache stores them, with the
rounding passed through by the gradient (the backward reads the cached rows
and treats the rounding as the identity). The tolerance is set by that
rounding and by float32 against float64: a dropped or wrong term moves a
buffer by percents.

Needs torch, so it runs where torch is installed (the lab hosts); CI skips it.
"""
import os
import pathlib
import struct
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

ROOT = pathlib.Path(__file__).resolve().parents[1]
T = 14
SLOT_NAMES = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up",
              "ffn_down", "attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha", "ssm_out"]


def read_gguf(path):
    """Metadata and F32 tensors of a GGUF file (version 3, no quantized types)."""
    b = pathlib.Path(path).read_bytes()
    assert b[:4] == b"GGUF"
    off = 4
    def u(fmt):
        nonlocal off
        v = struct.unpack_from("<" + fmt, b, off)[0]
        off += struct.calcsize("<" + fmt)
        return v
    def s():
        nonlocal off
        n = u("Q"); v = b[off:off + n].decode(); off += n
        return v
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
              10: "Q", 11: "q", 12: "d"}
    def val(t):
        if t == 8:
            return s()
        if t == 9:
            et = u("I"); n = u("Q")
            return [val(et) for _ in range(n)]
        return u(scalar[t])
    version = u("I"); nt = u("Q"); nkv = u("Q")
    assert version >= 2
    meta = {}
    for _ in range(nkv):
        k = s(); meta[k] = val(u("I"))
    infos = []
    for _ in range(nt):
        name = s(); nd = u("I")
        dims = [u("Q") for _ in range(nd)]
        typ = u("I"); toff = u("Q")
        infos.append((name, dims, typ, toff))
    align = meta.get("general.alignment", 32)
    base = (off + align - 1) // align * align
    tens = {}
    for name, dims, typ, toff in infos:
        assert typ == 0, f"{name}: type {typ} (F32 only)"
        n = 1
        for d in dims:
            n *= d
        arr = struct.unpack_from(f"<{n}f", b, base + toff)
        # GGUF dims run fastest first: [ne0, ne1] is ne1 rows of ne0
        shape = list(reversed(dims))
        tens[name] = torch.tensor(arr, dtype=torch.float64).reshape(shape)
    return meta, tens


def rms(x, w, eps):
    return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * w


def f16_st(x):
    """f16 rounding of the value, identity for the gradient."""
    return x + (x.to(torch.float16).to(torch.float64) - x).detach()


def reference(base, adapter):
    meta, W = read_gguf(base)
    ameta, A = read_gguf(adapter)
    arch = meta["general.architecture"]
    P = lambda k: meta[f"{arch}.{k}"]
    E, H, KV = P("embedding_length"), P("attention.head_count"), P("attention.head_count_kv")
    eps = P("attention.layer_norm_rms_epsilon")
    sk, ng, nh, inner, K = (P("ssm.state_size"), P("ssm.group_count"),
                            P("ssm.time_step_rank"), P("ssm.inner_size"), P("ssm.conv_kernel"))
    hv, keydim = inner // nh, sk * ng
    interval = P("full_attention_interval")
    hd = P("attention.key_length")
    rope_dim, base_f = P("rope.dimension_count"), P("rope.freq_base")
    alpha = ameta["adapter.lora.alpha"]
    emb = W["token_embd.weight"]                 # [V][E]
    V = emb.shape[0]
    n_layer = 1 + max(int(n.split(".")[1]) for n in W if n.startswith("blk.") and "nextn" not in n)
    params = {}
    for name, t in A.items():
        if name.endswith(".lora_a") or name.endswith(".lora_b"):
            params[name] = t.clone().requires_grad_(True)

    def lin(x, l, tname):
        y = x @ W[f"blk.{l}.{tname}.weight"].T
        a = params.get(f"blk.{l}.{tname}.weight.lora_a")
        if a is not None:
            bb = params[f"blk.{l}.{tname}.weight.lora_b"]
            y = y + (alpha / a.shape[0]) * ((x @ a.T) @ bb.T)
        return y

    toks = [3 + (i * 7) % (V - 4) for i in range(T)]
    x = emb[toks]                                # [T][E]
    half = rope_dim // 2
    freqs = torch.tensor([base_f ** (-2.0 * j / rope_dim) for j in range(half)],
                         dtype=torch.float64)
    ang = torch.arange(T, dtype=torch.float64)[:, None] * freqs[None, :]
    cos, sin = torch.cos(ang), torch.sin(ang)

    def rope(v):                                 # v [T][heads][hd], neox halves
        v0, v1 = v[..., :half], v[..., half:rope_dim]
        r0 = v0 * cos[:, None, :] - v1 * sin[:, None, :]
        r1 = v0 * sin[:, None, :] + v1 * cos[:, None, :]
        return torch.cat([r0, r1, v[..., rope_dim:]], -1)

    for l in range(n_layer):
        xn = rms(x, W[f"blk.{l}.attn_norm.weight"], eps)
        recurrent = (l + 1) % interval != 0
        if recurrent:
            mix = lin(xn, l, "attn_qkv")         # [T][convdim]
            z = lin(xn, l, "attn_gate")
            bpre = lin(xn, l, "ssm_beta")
            apre = lin(xn, l, "ssm_alpha")
            cw = W[f"blk.{l}.ssm_conv1d.weight"]  # [convdim][K]
            pad = torch.cat([torch.zeros(K - 1, mix.shape[1], dtype=torch.float64), mix], 0)
            cpre = sum(cw[:, j][None, :] * pad[j:j + T] for j in range(K))
            c = cpre * torch.sigmoid(cpre)
            q = c[:, :keydim].reshape(T, ng, sk)
            k = c[:, keydim:2 * keydim].reshape(T, ng, sk)
            v = c[:, 2 * keydim:].reshape(T, nh, hv)
            q = q / torch.sqrt(eps + (q * q).sum(-1, keepdim=True))
            k = k / torch.sqrt(eps + (k * k).sum(-1, keepdim=True))
            dtb = W.get(f"blk.{l}.ssm_dt.bias", W.get(f"blk.{l}.ssm_dt")).reshape(-1)
            Aa = W[f"blk.{l}.ssm_a"].reshape(-1)
            beta = torch.sigmoid(bpre)
            a2 = apre + dtb
            sp = torch.where(a2 > 20, a2, torch.log1p(torch.exp(torch.clamp(a2, max=20))))
            decay = torch.exp(Aa * sp)
            ys = []
            S = [torch.zeros(hv, hv, dtype=torch.float64) for _ in range(nh)]
            for t in range(T):
                row = []
                for h in range(nh):
                    g = h % ng
                    St = decay[t, h] * S[h]
                    pred = St @ k[t, g]
                    delta = beta[t, h] * (v[t, h] - pred)
                    St = St + torch.outer(delta, k[t, g])
                    S[h] = St
                    row.append(St @ q[t, g] / (sk ** 0.5))
                ys.append(torch.stack(row))
            y = torch.stack(ys)                  # [T][nh][hv]
            o = rms(y, W[f"blk.{l}.ssm_norm.weight"], eps)
            zz = z.reshape(T, nh, hv)
            o = (o * zz * torch.sigmoid(zz)).reshape(T, inner)
            out = lin(o, l, "ssm_out")
        else:
            qg = lin(xn, l, "attn_q").reshape(T, H, 2 * hd)
            qh, gate = qg[..., :hd], qg[..., hd:]
            kk = lin(xn, l, "attn_k").reshape(T, KV, hd)
            vv = lin(xn, l, "attn_v").reshape(T, KV, hd)
            qh = rms(qh, W[f"blk.{l}.attn_q_norm.weight"], eps)
            kk = rms(kk, W[f"blk.{l}.attn_k_norm.weight"], eps)
            qh, kk = rope(qh), rope(kk)
            kk, vv = f16_st(kk), f16_st(vv)
            mul = H // KV
            heads = []
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
            for h in range(H):
                sc = (qh[:, h] @ kk[:, h // mul].T) / (hd ** 0.5)
                sc = sc.masked_fill(mask, float("-inf"))
                heads.append(torch.softmax(sc, -1) @ vv[:, h // mul])
            att = torch.stack(heads, 1) * torch.sigmoid(gate)
            out = lin(att.reshape(T, H * hd), l, "attn_output")
        x = x + out
        xn2 = rms(x, W[f"blk.{l}.post_attention_norm.weight"], eps)
        gg = lin(xn2, l, "ffn_gate")
        x = x + lin(gg * torch.sigmoid(gg) * lin(xn2, l, "ffn_up"), l, "ffn_down")
    logits = rms(x, W["output_norm.weight"], eps) @ emb.T
    lp = torch.log_softmax(logits, -1)
    loss = -sum(lp[t, toks[t + 1]] for t in range(T - 1))
    loss.backward()
    grads = {}
    for name, p in params.items():
        layer = int(name.split(".")[1])
        slot = SLOT_NAMES.index(name.split(".")[2])
        side = 0 if name.endswith("lora_a") else 1
        grads[(layer, slot, side)] = p.grad.reshape(-1)
    return float(loss), grads


def runner_grads(base, adapter, tmp):
    exe = ROOT / ("test-lora-grad.exe" if sys.platform == "win32" else "test-lora-grad")
    if not exe.exists():
        pytest.skip("test-lora-grad not built")
    dump = tmp / "grads.bin"
    p = subprocess.run([str(exe), str(base), str(adapter)], cwd=ROOT,
                       env=dict(os.environ, RUNNER_GRAD_DUMP=str(dump)),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    b = dump.read_bytes()
    loss = struct.unpack_from("<d", b, 0)[0]
    off, out = 8, {}
    while off < len(b):
        layer, slot, side, n = struct.unpack_from("<4i", b, off)
        off += 16
        out[(layer, slot, side)] = torch.tensor(struct.unpack_from(f"<{n}f", b, off),
                                                dtype=torch.float64)
        off += 4 * n
    return loss, out


@pytest.mark.parametrize("gen", ["make-test-ornith.py", "make-test-clef.py"])
def test_qwen35_backward_matches_autograd(tmp_path, gen):
    base = tmp_path / "base.gguf"
    subprocess.run([sys.executable, ROOT / "scripts" / gen, str(base)], check=True,
                   cwd=ROOT, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, ROOT / "scripts/make-test-lora.py", str(base),
                    str(tmp_path / "fx")], check=True, cwd=ROOT, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, LORA_TARGETS="qwen35"))
    adapter = tmp_path / "fx.full.gguf"
    rloss, rg = runner_grads(base, adapter, tmp_path)
    tloss, tg = reference(base, adapter)
    assert abs(rloss - tloss) <= 1e-4 * abs(tloss), (rloss, tloss)
    assert set(rg) == set(tg), sorted(set(rg) ^ set(tg))
    worst = 0.0
    for key in sorted(tg):
        a, b = rg[key], tg[key]
        rel = float((a - b).norm() / b.norm().clamp_min(1e-30))
        worst = max(worst, rel)
        assert rel <= 1e-2, f"layer {key[0]} {SLOT_NAMES[key[1]]} {'AB'[key[2]]}: rel {rel:.3g}"
    print(f"{gen}: loss {rloss:.6f} vs {tloss:.6f}; {len(tg)} buffers, worst rel L2 {worst:.3g}")
