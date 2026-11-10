#!/usr/bin/env python3
"""An activation trace of a transformers model, in the format RUNNER_ACT_TRACE
writes, so scripts/act-trace-diff.py can compare Runner with the model's own
reference implementation point by point (R4.11, RO-7).

    scripts/act-trace-hf.py --hf DIR --prompt TEXT --pos 0,7 --out ref.jsonl
    RUNNER_ACT_TRACE=run.jsonl RUNNER_ACT_TRACE_POS=0,7 RUNNER_ACT_TRACE_VALUES=1 \\
        runner -m FILE.gguf -p TEXT -n 1 --temp 0 --gpu off
    scripts/act-trace-diff.py run.jsonl ref.jsonl --rtol 1e-3

One forward in float32 (--upcast-at-use keeps stored BF16 weights and casts
each decoder block at use, as gold-logits.py does) over the prompt's tokens,
which must be the ones Runner's -p feeds (the script prints them; a template
or a BOS difference shows up as an embd mismatch at position 0). Recorded at
each listed position, with values:

  embd    the first decoder block's input;
  mixer   the residual after the mixer branch was added, which is what
          Runner's mixer point is: the input of the norm in front of the
          FFN, whose name depends on the block's layout (mixer_norm);
  resid   each block's output;
  logits  the head's output.

MoE selections (Runner's route point) are model-specific in transformers and
are not written; act-trace-diff compares the points both sides have. A block
whose layout has no such norm gets no mixer point.
"""
import argparse
import importlib.util
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def mixer_norm(block):
    """The module whose INPUT is Runner's mixer point (the residual after the
    mixer branch was added), or None. By the block's norm layout, in
    transformers' names:

      sandwich norms (Gemma 2, 3 and 4): post_attention_layernorm normalizes
        the attention OUTPUT before the residual add (modeling_gemma4.py), so
        its input is the branch; the FFN's norm is pre_feedforward_layernorm;
      pre-norm (Llama, Qwen, Granite and most others): post_attention_layernorm
        is the FFN's norm;
      ffn_norm: the FFN's norm under that name;
      post-norm (OLMo 2: post_feedforward_layernorm and no pre-FFN norm): the
        FFN reads the residual directly and no norm has it as input, so there
        is no point to hook (R6.38).
    """
    def has(name):
        return getattr(block, name, None) is not None
    if has("pre_feedforward_layernorm"):
        return block.pre_feedforward_layernorm
    if has("post_feedforward_layernorm"):
        return None
    for name in ("post_attention_layernorm", "ffn_norm"):
        if has(name):
            return getattr(block, name)
    return None



def gold():
    spec = importlib.util.spec_from_file_location("gold_logits", os.path.join(HERE, "gold-logits.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def record(f, pos, layer, point, row):
    vals = [float(v) for v in row]
    h = 0xcbf29ce484222325
    for b in struct.pack(f"<{len(vals)}f", *vals):
        h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
    l2 = sum(v * v for v in vals) ** 0.5
    f.write(json.dumps({"v": 1, "pass": 1, "pos": pos, "layer": layer, "point": point,
                        "n": len(vals), "hash": f"{h:016x}", "l2": l2, "values": vals}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", required=True, help="HF model directory")
    ap.add_argument("--prompt", required=True, help="the text Runner's -p is given")
    ap.add_argument("--pos", required=True, help="comma-separated token positions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-special-tokens", action="store_true",
                    help="tokenize without the tokenizer's BOS/EOS additions")
    ap.add_argument("--upcast-at-use", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.threads)
    g = gold()
    want = sorted({int(p) for p in args.pos.split(",") if p.strip()})
    tok = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=args.trust_remote_code)
    ids = tok(args.prompt, add_special_tokens=not args.no_special_tokens).input_ids
    print(f"{len(ids)} tokens: {ids}", file=sys.stderr)
    if want[-1] >= len(ids):
        raise SystemExit(f"position {want[-1]} is past the prompt's {len(ids)} tokens")
    kw = {"trust_remote_code": args.trust_remote_code,
          "dtype": "auto" if args.upcast_at_use else torch.float32}
    try:
        model = AutoModelForCausalLM.from_pretrained(args.hf, **kw).eval()
    except ValueError:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.hf, **kw).eval()
    if args.upcast_at_use:
        g.upcast_at_use(model)
    blocks = g.decoder_blocks(model)
    if not len(blocks):
        raise SystemExit("no decoder stack found")

    rows = {}   # (layer, point) -> [seq, dim] float32 tensor

    def keep(layer, point, x):
        rows[(layer, point)] = x[0].detach().float()

    def first(t):
        return t[0] if isinstance(t, (tuple, list)) else t

    def pre_block(mod, a, k):
        keep(-1, "embd", a[0] if a else k["hidden_states"])

    blocks[0].register_forward_pre_hook(pre_block, with_kwargs=True)
    for l, blk in enumerate(blocks):
        blk.register_forward_hook(lambda mod, a, out, l=l: keep(l, "resid", first(out)))
        norm = mixer_norm(blk)
        if norm is not None:
            norm.register_forward_pre_hook(lambda mod, a, l=l: keep(l, "mixer", a[0]))
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].float()
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for p in want:
            record(f, p, -1, "embd", rows[(-1, "embd")][p])
            for l in range(len(blocks)):
                if (l, "mixer") in rows:
                    record(f, p, l, "mixer", rows[(l, "mixer")][p])
                record(f, p, l, "resid", rows[(l, "resid")][p])
            record(f, p, len(blocks), "logits", logits[p])
    print(f"wrote {args.out}: {len(want)} positions, {len(blocks)} blocks", file=sys.stderr)


if __name__ == "__main__":
    main()
