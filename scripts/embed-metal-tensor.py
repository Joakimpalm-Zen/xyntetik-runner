#!/usr/bin/env python3
"""Embed the separately admitted Metal 4 tensor source."""
import hashlib
import os

root = os.path.join(os.path.dirname(__file__), "..")
src = os.path.join(root, "src", "kernels_tensor.metal")
dst = os.environ.get("EMBED_OUT") or os.path.join(root, "src", "kernels_tensor_metal.h")
with open(src, encoding="utf-8") as f:
    text = f.read()
sha = hashlib.sha256(text.encode()).hexdigest()
out = ["// Generated from kernels_tensor.metal — do not edit.",
       "static const char *k_metal_tensor_src ="]
for line in text.split("\n"):
    line = line.replace("\\", "\\\\").replace('"', '\\"')
    out.append(f'    "{line}\\n"')
out += [";", f'static const char *k_metal_tensor_sha = "{sha}";']
with open(dst, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(out) + "\n")
print(f"wrote {dst}")
