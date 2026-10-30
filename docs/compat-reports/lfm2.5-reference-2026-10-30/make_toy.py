"""A small random LFM2 MoE (lfm2moe) in transformers' own code, saved in
float32 with the publisher's tokenizer, plus the reference log-probabilities
of a text and how far the reference moves when one feature is taken away."""
import copy, json, shutil, sys, torch
from transformers import Lfm2MoeConfig, Lfm2MoeForCausalLM, AutoTokenizer
out, src, tok_src, text_path = sys.argv[1:5]
torch.manual_seed(0)
cfg = json.load(open(f"{src}/config.json"))
layers = ["conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "full_attention"]
cfg.update(dict(hidden_size=128, intermediate_size=192, num_attention_heads=4,
                num_key_value_heads=2, num_hidden_layers=len(layers), layer_types=layers,
                num_dense_layers=2, moe_intermediate_size=32, num_experts=8,
                num_experts_per_tok=3, vocab_size=65536, bos_token_id=1, eos_token_id=7,
                pad_token_id=0, max_position_embeddings=4096, dtype="float32",
                initializer_range=0.25))
for k in ("transformers_version", "architectures", "torch_dtype"):
    cfg.pop(k, None)
config = Lfm2MoeConfig(**cfg)
model = Lfm2MoeForCausalLM(config).float().eval()
with torch.no_grad():   # the selection bias and the norms away from their trivial init
    for n, b in model.named_buffers():
        if n.endswith("expert_bias"): b.copy_(torch.randn_like(b) * 0.3)
    for n, p in model.named_parameters():
        if "norm" in n: p.copy_(1 + 0.3 * torch.randn_like(p))
model.save_pretrained(out, safe_serialization=True)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    shutil.copy(f"{tok_src}/{f}", f"{out}/{f}")
tok = AutoTokenizer.from_pretrained(out)
ids = tok(open(text_path).read()[:1500])["input_ids"][:200]   # BOS included, as Runner encodes
def logp(m):
    with torch.no_grad():
        return torch.log_softmax(m(torch.tensor([ids])).logits[0].float(), -1)
lp = logp(model)
def kl(a, b):
    top2 = a.topk(2, dim=-1).values
    q = (top2[:, 0] - top2[:, 1]) >= 0.5
    same = a.argmax(-1) == b.argmax(-1)
    return {"mean_kl": float((a.exp() * (a - b)).sum(-1).mean()),
            "top1_pct": 100.0 * float(same.float().mean()),
            "margin_qualified_top1_pct": 100.0 * float(same[q].float().mean()) if q.any() else None,
            "margin_qualified_positions": int(q.sum())}
abl = {}
m = copy.deepcopy(model)
with torch.no_grad():
    for n, b in m.named_buffers():
        if n.endswith("expert_bias"): b.zero_()
abl["no selection bias"] = kl(lp, logp(m))
c2 = dict(cfg); c2["norm_topk_prob"] = False
m = Lfm2MoeForCausalLM(Lfm2MoeConfig(**c2)).float().eval(); m.load_state_dict(model.state_dict())
abl["routing weights not renormalised"] = kl(lp, logp(m))
m = copy.deepcopy(model)
with torch.no_grad():   # the convolution sees the current position only
    for layer in m.model.layers:
        if hasattr(layer, "conv"): layer.conv.conv.weight[..., :-1] = 0
abl["convolution without history"] = kl(lp, logp(m))
json.dump(abl, open(f"{out}/ablations.json", "w"), indent=1)
print("ablations:", json.dumps(abl))
top = torch.topk(lp, 20, dim=-1)
json.dump({"ids": ids, "top": [[[int(i), float(v)] for i, v in zip(top.indices[r], top.values[r])] for r in range(len(ids))]},
          open(f"{out}/reference.json", "w"))
print("toy", sum(p.numel() for p in model.parameters()), "params;", len(ids), "tokens; layers", layers)
