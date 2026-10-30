import json, sys, torch
import transformers.models.flint.modeling_flint as mf
hf = sys.argv[1]; r = json.load(open(f"{hf}/reference.json")); s = json.load(open(sys.argv[2]))
orig = mf.apply_rotary_pos_emb
def rope_then_round(q, k, cos, sin, unsqueeze_dim=1):
    q2, k2 = orig(q, k, cos, sin, unsqueeze_dim)
    return q2, k2.half().float()
mf.apply_rotary_pos_emb = rope_then_round       # K of the sliding layers, after rope
orig_fwd = mf.FlintAttention.forward
def fwd(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
    ok, ov = self.k_proj.forward, self.v_proj.forward
    # V of every layer, and K of the NoPE layers (no rope, so it is cached as projected and normed)
    self.v_proj.forward = lambda x: ov(x).half().float()
    if self.sliding_window is None:
        kn = self.k_norm.forward
        self.k_norm.forward = lambda x: kn(x).half().float()
    try:
        return orig_fwd(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kw)
    finally:
        self.v_proj.forward = ov
        if self.sliding_window is None: self.k_norm.forward = kn
mf.FlintAttention.forward = fwd
m = mf.FlintForCausalLM.from_pretrained(hf, torch_dtype=torch.float32).eval()
with torch.no_grad():
    lp = torch.log_softmax(m(torch.tensor([r["ids"]])).logits[0].float(), -1)
md = 0; ag = 0; worst = None
for p in range(len(s["top"])):
    b = {int(i): float(v) for i, v in s["top"][p]}
    d = max(abs(float(lp[p, k]) - v) for k, v in b.items())
    if d > md: md, worst = d, p
    ag += int(lp[p].argmax()) == max(b, key=b.get)
print("HF with f16 K and V vs Runner: max |dlogp| %.2e at position %d, top-1 %.1f%%" % (md, worst, 100 * ag / len(s["top"])))
