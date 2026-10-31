"""How far the qwen3_5_moe toy's reference moves when one feature is taken
away: the agreement Runner reaches on the toy says something only if each
of these features changes the output by far more than that agreement."""
import copy, json, sys, torch
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
toy, ref_path, out = sys.argv[1:4]
ref = json.load(open(ref_path)); ids = ref["ids"]
model = Qwen3_5MoeForCausalLM.from_pretrained(toy, dtype=torch.float32).eval()
def logp(m):
    with torch.no_grad():
        return torch.log_softmax(m(torch.tensor([ids])).logits[0].float(), -1)
def kl(a, b):
    top2 = a.topk(2, dim=-1).values
    q = (top2[:, 0] - top2[:, 1]) >= 0.5
    same = a.argmax(-1) == b.argmax(-1)
    return {"mean_kl": float((a.exp() * (a - b)).sum(-1).mean()),
            "top1_pct": 100.0 * float(same.float().mean()),
            "margin_qualified_top1_pct": 100.0 * float(same[q].float().mean()) if q.any() else None,
            "margin_qualified_positions": int(q.sum())}
lp = logp(model)
# this transformers against the reference the toy was saved with
r_top = torch.tensor([[v for _, v in row] for row in ref["top"]])
r_idx = torch.tensor([[i for i, _ in row] for row in ref["top"]])
base_dev = float((lp.gather(1, r_idx) - r_top).abs().max())
abl = {"_this transformers against the saved reference, max |dlogp| over the top 20": base_dev}
m = copy.deepcopy(model)
with torch.no_grad():
    for l in m.model.layers: l.mlp.shared_expert.down_proj.weight.zero_()
abl["no shared expert"] = kl(lp, logp(m))
m = copy.deepcopy(model)
with torch.no_grad():
    for l in m.model.layers: l.mlp.shared_expert_gate.weight.zero_()
abl["shared expert gate fixed at sigmoid(0) = 0.5"] = kl(lp, logp(m))
m = copy.deepcopy(model)
for l in m.model.layers:
    r = l.mlp.gate
    def fwd(h, r=r):
        h = h.reshape(-1, r.hidden_dim)
        logits = torch.nn.functional.linear(h, r.weight)
        p = torch.softmax(logits, dtype=torch.float, dim=-1)
        v, i = torch.topk(p, r.top_k, dim=-1)
        return logits, v.to(logits.dtype), i
    r.forward = fwd
abl["routing weights not renormalised"] = kl(lp, logp(m))
m = copy.deepcopy(model)
n = 0
for mod in m.modules():
    if type(mod).__name__.endswith("RMSNormGated"): mod.activation = "sigmoid"; n += 1
abl[f"gated norm with sigmoid(z) instead of silu(z) ({n} norms)"] = kl(lp, logp(m))
c = Qwen3_5MoeTextConfig.from_pretrained(toy); rp = dict(c.rope_parameters or {}); rp["partial_rotary_factor"] = 1.0
c.rope_parameters = rp; c.partial_rotary_factor = 1.0
m = Qwen3_5MoeForCausalLM(c).float().eval(); m.load_state_dict(model.state_dict())
abl["rope over all head dims"] = kl(lp, logp(m))
m = copy.deepcopy(model)
with torch.no_grad():
    for l in m.model.layers:
        if hasattr(l, "linear_attn"): l.linear_attn.conv1d.weight[..., :-1] = 0
abl["convolution without history"] = kl(lp, logp(m))
json.dump(abl, open(out, "w"), indent=1)
print(json.dumps(abl, indent=1))
