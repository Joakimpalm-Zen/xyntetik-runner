"""The simulator's control: uniform random routing in Mellum's shape (28 layers,
top-8 of 64), six prompts of 100 prompt and 300 decode tokens, seed 1, in the
files sim.py reads. Usage: synth.py OUTDIR"""
import json
import os
import random
import sys

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
random.seed(1)
with open(os.path.join(OUT, "runs.jsonl"), "w") as runs:
    for name in ["code", "copy", "edit", "long", "prose", "tool"]:
        with open(os.path.join(OUT, f"trace-{name}.jsonl"), "w") as f:
            for pos in range(100 + 300):
                for layer in range(28):
                    f.write(json.dumps({"pos": pos, "layer": layer, "experts": random.sample(range(64), 8),
                                        "gates": [], "norms": []}) + "\n")
        runs.write(json.dumps({"prompt": name, "usage": {"prompt_tokens": 100}}) + "\n")
