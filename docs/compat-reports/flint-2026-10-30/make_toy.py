"""A small random flint in transformers' own code, saved in float32 with the
publisher's tokenizer, plus the reference log-probabilities of a text."""
import json, shutil, sys, torch
from transformers import FlintConfig, FlintForCausalLM, AutoTokenizer
out, src = sys.argv[1], sys.argv[2]
torch.manual_seed(0)
cfg = json.load(open(f"{src}/config.json"))
layers = ["full_attention", "full_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "full_attention", "sliding_attention", "full_attention"]
cfg.update(dict(hidden_size=128, intermediate_size=192, num_attention_heads=4,
                num_key_value_heads=2, head_dim=32, num_hidden_layers=len(layers),
                layer_types=layers, mlp_only_layers=[0, 1], moe_intermediate_size=32,
                shared_expert_intermediate_size=64, num_experts=8, num_experts_per_tok=3,
                no_rope_layer_indices=[i for i, k in enumerate(layers) if k == "full_attention"],
                sliding_window=7, max_position_embeddings=4096, torch_dtype="float32",
                initializer_range=0.25))
cfg.pop("transformers_version", None); cfg.pop("architectures", None)
config = FlintConfig(**cfg)
model = FlintForCausalLM(config).float().eval()
with torch.no_grad():   # sinks and norms away from their trivial init, so each matters
    for n, p in model.named_parameters():
        if n.endswith("sinks"): p.copy_(torch.randn_like(p) * 2)
        elif "norm" in n: p.copy_(1 + 0.3 * torch.randn_like(p))
model.save_pretrained(out, safe_serialization=True)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    shutil.copy(f"{src}/{f}", f"{out}/{f}")
tok = AutoTokenizer.from_pretrained(out)
text = open(sys.argv[3]).read()[:1500]
ids = tok(text, add_special_tokens=False)["input_ids"][:200]
def logp(m):
    with torch.no_grad():
        return torch.log_softmax(m(torch.tensor([ids])).logits[0].float(), -1)
lp = logp(model)
# sensitivity: how far the reference itself moves when one feature is changed
import copy
def kl(a, b):
    # mean KL, top-1 agreement, and top-1 agreement where the reference's own
    # top-1 leads its second by at least 0.5 nats (the margin-qualified set)
    top2 = a.topk(2, dim=-1).values
    q = (top2[:, 0] - top2[:, 1]) >= 0.5
    same = a.argmax(-1) == b.argmax(-1)
    return {"mean_kl": float((a.exp() * (a - b)).sum(-1).mean()),
            "top1_pct": 100.0 * float(same.float().mean()),
            "margin_qualified_top1_pct": 100.0 * float(same[q].float().mean()) if q.any() else None,
            "margin_qualified_positions": int(q.sum())}
abl = {}
m2 = copy.deepcopy(model)
with torch.no_grad():
    for n, p in m2.named_parameters():
        if n.endswith("sinks"): p.fill_(-1e4)
abl["sinks off"] = kl(lp, logp(m2))
# transformers ropes a layer when its attention has a sliding window
# (modeling_flint: "Only sliding-window layers are rotary"); a window wider
# than the text makes a full layer rope without changing what it sees
m3 = copy.deepcopy(model)
for layer in m3.model.layers:
    if layer.self_attn.sliding_window is None:
        layer.self_attn.sliding_window = 1 << 30
abl["rope on every layer"] = kl(lp, logp(m3))
cfg3 = dict(cfg); cfg3["sliding_window"] = 4096
m4 = FlintForCausalLM(FlintConfig(**cfg3)).float().eval(); m4.load_state_dict(model.state_dict())
abl["no sliding window"] = kl(lp, logp(m4))
cfg4 = dict(cfg); cfg4["norm_topk_prob"] = False
m5 = FlintForCausalLM(FlintConfig(**cfg4)).float().eval(); m5.load_state_dict(model.state_dict())
abl["routing weights not renormalised"] = kl(lp, logp(m5))
json.dump(abl, open(f"{out}/ablations.json", "w"), indent=1)
print("ablations:", json.dumps(abl))
top = torch.topk(lp, 20, dim=-1)
json.dump({"ids": ids, "top": [[[int(i), float(v)] for i, v in zip(top.indices[r], top.values[r])] for r in range(len(ids))]},
          open(f"{out}/reference.json", "w"))
print("toy", sum(p.numel() for p in model.parameters()), "params;", len(ids), "tokens; layers", layers)
