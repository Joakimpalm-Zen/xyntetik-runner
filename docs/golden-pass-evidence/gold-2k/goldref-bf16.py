#!/usr/bin/env python3
"""Run scripts/gold-logits.py with its float32 reference held in half the memory.

The publisher's checkpoint is bfloat16, so the reference's float32 weights are
bfloat16 values widened exactly. Here the large matrices (every nn.Linear and
nn.Embedding weight) stay in bfloat16 storage and are widened to float32 at
the moment of use; every other parameter and buffer is float32, rotary
frequencies are recomputed in float32 as a float32 load computes them. The
arithmetic is the float32 reference's: the same float32 values reach the same
float32 operations. gold-logits.py itself is run unchanged (runpy), so the
scorer, its arguments and its report are the pinned ones.

Proven bitwise on apertus-8b (2-layer truncation). NOT valid for gemma-3 / 4:
their scaled embeddings multiply by a buffer built at load in the default
dtype (full gemma-4-E2B: max |diff| 34); GOLDREF_MODE=release keeps the
float32 load and only releases its weights after the forward.

usage: goldref-bf16.py GOLD_LOGITS_PY [gold-logits arguments...]
       GOLDREF_MODE=gguf GOLDREF_GGUF=FILE GGUF_PY=<llama.cpp>/gguf-py goldref-bf16.py ...
                                   (the reference is the file's own model in float32)
       goldref-bf16.py --selftest HF_DIR LAYERS   (bitwise check vs float32)
"""
import os, runpy, sys
import torch
import torch.nn.functional as F
from torch import nn
import transformers


def _linear_fwd(self, x):
    return F.linear(x, self.weight.float(), None if self.bias is None else self.bias.float())


def _embed_fwd(self, ids):
    return F.embedding(ids, self.weight, self.padding_idx, self.max_norm, self.norm_type,
                       self.scale_grad_by_freq, self.sparse).float()


def widen_at_use(model):
    big = set()
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.forward = _linear_fwd.__get__(m, type(m)); big.add(id(m.weight))
        elif isinstance(m, nn.Embedding):
            m.forward = _embed_fwd.__get__(m, type(m)); big.add(id(m.weight))
    for m in model.modules():
        for name, p in list(m.named_parameters(recurse=False)):
            if id(p) not in big and p.dtype != torch.float32:
                setattr(m, name, nn.Parameter(p.data.float(), requires_grad=False))
        for name, b in list(m.named_buffers(recurse=False)):
            if b is not None and b.is_floating_point() and b.dtype != torch.float32:
                m._buffers[name] = b.float()
        # rotary frequencies as a float32 load computes them
        if hasattr(m, "inv_freq") and hasattr(m, "rope_init_fn") and hasattr(m, "config"):
            inv, scale = m.rope_init_fn(m.config, m.inv_freq.device)
            m._buffers["inv_freq"] = inv.float()
            if "original_inv_freq" in m._buffers or hasattr(m, "original_inv_freq"):
                m.original_inv_freq = m._buffers["inv_freq"]
            if hasattr(m, "attention_scaling"):
                m.attention_scaling = scale
    model.config.dtype = torch.float32
    return model


def _release_after_forward(model):
    # gold-logits.py runs the reference once, over the whole span, and keeps
    # only the logits; the weights are released after that forward so the
    # serving endpoints' pages are not competing with them while it scores
    def hook(mod, inp, out):
        for p in mod.parameters():
            p.data = torch.empty(0, dtype=p.dtype)
    model.register_forward_hook(hook)
    return model


def _patch_release_only(cls):
    # GOLDREF_MODE=release: the float32 load exactly as gold-logits.py makes
    # it, only its weights released after the one forward (no storage change)
    orig = cls.from_pretrained.__func__
    def from_pretrained(klass, path, *a, **kw):
        return _release_after_forward(orig(klass, path, *a, **kw))
    cls.from_pretrained = classmethod(from_pretrained)


def _from_gguf(model, path):
    # GOLDREF_MODE=gguf: the float32 load with every quantized matrix replaced
    # by the GGUF's own values, dequantized exactly (gguf-py): the reference is
    # then the file's model in exact float32 arithmetic, so an engine's
    # distance to it measures its faithfulness to the file it was given
    import numpy as np
    sys.path.insert(0, os.environ["GGUF_PY"])
    import gguf
    rd = gguf.GGUFReader(path)
    arch = None
    for f in rd.fields.values():
        if f.name == "general.architecture":
            arch = bytes(f.parts[f.data[0]]).decode()
    nb = next(int(f.parts[f.data[0]][0]) for f in rd.fields.values() if f.name == arch + ".block_count")
    tmap = gguf.get_tensor_name_map(next(k for k, v in gguf.MODEL_ARCH_NAMES.items() if v == arch), nb)
    params = dict(model.named_parameters())
    by_gguf = {}
    for hf in params:
        cands = [hf]
        for pre in ("model.language_model.", "language_model.model.", "model."):
            if hf.startswith(pre):
                cands.append(hf[len(pre):]); cands.append("model." + hf[len(pre):])
        for c in cands:
            g = tmap.get_name(c, try_suffixes=(".weight", ".bias"))
            if g:
                by_gguf.setdefault(g, hf)
                break
    done, missing, replaced = 0, [], set()
    for t in rd.tensors:
        if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
            continue
        hf = by_gguf.get(t.name)
        if hf is None:
            missing.append(t.name); continue
        shape = tuple(reversed([int(x) for x in t.shape]))
        p = params[hf]
        if tuple(p.shape) != shape:
            missing.append("%s shape %s vs %s" % (t.name, tuple(p.shape), shape)); continue
        # row chunks: dequantizing the 262,144 x 8,960 per-layer embedding
        # table whole needs ~10 GB of numpy temporaries beside the float32
        # model, past a 32 GiB container (OOM-killed twice 2026-10-22);
        # each row's blocks are independent, so the values are the same
        rows = t.data.reshape(shape[0], -1)
        step = max(1, (64 << 20) // max(1, rows.shape[1]))
        with torch.no_grad():
            for r in range(0, shape[0], step):
                w = gguf.quants.dequantize(np.asarray(rows[r:r + step]), t.tensor_type)
                p[r:r + step].copy_(torch.from_numpy(np.asarray(w, dtype=np.float32).reshape(-1, shape[1])))
        done += 1; replaced.add(hf)
    # a GGUF tensor the HF model does not have (gemma-4's KV-sharing layers
    # keep attn_k / attn_v the forward never reads) is skipped; every 2-D
    # weight of the text model must have been replaced, or the reference
    # would silently be part publisher, part file
    text = [n for n, p in params.items() if p.dim() == 2 and not any(k in n for k in ("vision", "audio", "embed_vision", "embed_audio"))]
    left = [n for n in text if n not in replaced]
    print("goldref gguf: %d quantized matrices replaced from %s; GGUF tensors without an HF parameter: %d (%s); text-model 2-D weights not replaced: %d %s" % (
        done, path, len(missing), missing[:2], len(left), left[:6]), file=sys.stderr)
    if left or any("shape" in m for m in missing):
        raise SystemExit("goldref gguf: refusing a partial replacement")
    return model


def _patch_gguf(cls, path):
    orig = cls.from_pretrained.__func__
    def from_pretrained(klass, hf, *a, **kw):
        return _release_after_forward(_from_gguf(orig(klass, hf, *a, **kw), path))
    cls.from_pretrained = classmethod(from_pretrained)


def _patch(cls, release=False):
    orig = cls.from_pretrained.__func__
    def from_pretrained(klass, path, *a, **kw):
        if kw.get("dtype") == torch.float32 or kw.get("torch_dtype") == torch.float32:
            kw.pop("torch_dtype", None)
            kw["dtype"] = torch.bfloat16
            m = widen_at_use(orig(klass, path, *a, **kw))
            return _release_after_forward(m) if release else m
        return orig(klass, path, *a, **kw)
    cls.from_pretrained = classmethod(from_pretrained)


def _load(hf, cfg):
    # as gold-logits.py: the causal-LM class, else the image-text class
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(hf, config=cfg, dtype=torch.float32).eval()
    except (ValueError, KeyError, OSError):
        return transformers.AutoModelForImageTextToText.from_pretrained(hf, config=cfg, dtype=torch.float32).eval()


def selftest(hf, layers):
    cfg = transformers.AutoConfig.from_pretrained(hf)
    tcfg = getattr(cfg, "text_config", cfg)
    if layers == "all":
        layers = tcfg.num_hidden_layers
    tcfg.num_hidden_layers = int(layers)
    if hasattr(tcfg, "layer_types") and tcfg.layer_types:
        tcfg.layer_types = tcfg.layer_types[:int(layers)]
    if getattr(tcfg, "num_kv_shared_layers", 0) and int(layers) < len(getattr(tcfg, "layer_types", []) or []) + 1 and str(layers) != str(tcfg.num_hidden_layers):
        tcfg.num_kv_shared_layers = 0   # truncated: no layer left to share from
    torch.manual_seed(0)
    ids = torch.randint(0, min(tcfg.vocab_size, 100000), (1, 192))
    ref = _load(hf, cfg)
    with torch.no_grad():
        a = ref(ids).logits.float().clone()
    del ref
    _patch(transformers.AutoModelForCausalLM)
    _patch(transformers.AutoModelForImageTextToText)
    m = _load(hf, cfg)
    with torch.no_grad():
        b = m(ids).logits.float()
    same = torch.equal(a, b)
    print("selftest %s layers=%s: logits %s (max |diff| %.3g, %d positions x %d)" % (
        hf, layers, "BITWISE IDENTICAL" if same else "DIFFERENT", (a - b).abs().max().item(), a.shape[1], a.shape[2]))
    return 0 if same else 1


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        sys.exit(selftest(sys.argv[2], sys.argv[3]))
    if os.environ.get("GOLDREF_MODE") == "gguf":
        _patch_gguf(transformers.AutoModelForCausalLM, os.environ["GOLDREF_GGUF"])
        _patch_gguf(transformers.AutoModelForImageTextToText, os.environ["GOLDREF_GGUF"])
    elif os.environ.get("GOLDREF_MODE") == "release":
        _patch_release_only(transformers.AutoModelForCausalLM)
        _patch_release_only(transformers.AutoModelForImageTextToText)
    else:
        _patch(transformers.AutoModelForCausalLM, release=True)
        _patch(transformers.AutoModelForImageTextToText, release=True)
    script = sys.argv[1]
    sys.argv = [script] + sys.argv[2:]
    runpy.run_path(script, run_name="__main__")
