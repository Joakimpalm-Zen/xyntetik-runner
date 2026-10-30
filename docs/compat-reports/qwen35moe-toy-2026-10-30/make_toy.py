"""A small random Qwen 3.5 MoE (qwen3_5_moe) in transformers' own code, saved
in float32 with the publisher's tokenizer, plus the reference
log-probabilities of a text: the shape of Qwen3.6-35B-A3B scaled down, with
Gated DeltaNet and gated-attention blocks, routed experts and the
sigmoid-gated shared expert on every block, partial rope."""
import json, shutil, sys, torch
from transformers import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM, AutoTokenizer
out, src, text_path = sys.argv[1:4]
torch.manual_seed(0)
cfg = json.load(open(f"{src}/config.json"))["text_config"]
layers = ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"]
cfg.update(dict(hidden_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                num_hidden_layers=len(layers), layer_types=layers, full_attention_interval=4,
                linear_num_key_heads=2, linear_key_head_dim=32, linear_num_value_heads=4,
                linear_value_head_dim=32, linear_conv_kernel_dim=4,
                num_experts=8, num_experts_per_tok=3, moe_intermediate_size=32,
                shared_expert_intermediate_size=48, mtp_num_hidden_layers=0,
                max_position_embeddings=4096, dtype="float32", initializer_range=0.25))
for k in ("transformers_version", "architectures", "torch_dtype", "model_type"):
    cfg.pop(k, None)
config = Qwen3_5MoeTextConfig(**cfg)
config.architectures = ["Qwen3_5MoeForCausalLM"]
model = Qwen3_5MoeForCausalLM(config).float().eval()
with torch.no_grad():   # norms, gates and the decay away from their trivial init
    for n, p in model.named_parameters():
        if "norm" in n and p.dim() == 1: p.copy_(0.3 * torch.randn_like(p))
        if n.endswith("dt_bias") or n.endswith("A_log"): p.copy_(0.5 * torch.randn_like(p))
model.save_pretrained(out, safe_serialization=True)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    shutil.copy(f"{src}/{f}", f"{out}/{f}")
tok = AutoTokenizer.from_pretrained(out)
ids = tok(open(text_path).read()[:1500])["input_ids"][:200]
with torch.no_grad():
    lp = torch.log_softmax(model(torch.tensor([ids])).logits[0].float(), -1)
top = []
for p in range(len(ids)):
    v, i = lp[p].topk(20)
    top.append([[int(a), float(b)] for a, b in zip(i, v)])
json.dump({"ids": ids, "top": top}, open(f"{out}/reference.json", "w"))
open(f"{out}/text.txt", "w").write(tok.decode(ids))
print("saved", out, len(ids), "tokens", sum(p.numel() for p in model.parameters()), "params")
