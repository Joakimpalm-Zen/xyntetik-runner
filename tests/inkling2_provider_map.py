# R4.48: the name map for the test-side Inkling-2-Mini weight provider (tests/inkling2_provider.c).
#
# The map is the reference loader's OWN: the model class of transformers pull request 51050 at 6173a2f6
# (Inkling2LoopForCausalLM on the meta device, the checkpoint's text config), its conversion list
# (get_model_conversion_mapping: mapping["inkling2_loop_text"] and what it inherits), and the loader's own
# rename_source_key, applied to every key of the checkpoint's header in the loader's order (dot_natural_key). A
# target collects its sources by the converter's source pattern, as convert_and_load_state_dict_in_model does.
# Nothing here decodes a weight; the C provider does, from this table.
#
# Output, one line per reference parameter (693 for Inkling-2-Mini):
#   H <header_len> <file_size>
#   T <target> <ops> <ndim> <dims...> <nsrc> then per source pattern of the converter, in its order:
#       <checkpoint key> <dtype> <ndim> <dims...> <begin> <end>     (offsets relative to the data start, 8 + header_len)
#     or "-" for a pattern the checkpoint does not have (the MXFP8 layers have no amax / denominator).
#   <ops> is the converter's operation chain, e.g. "Inkling2LoopDequantizeExperts,Interleave(dim=1)", or "rename".
#   U <checkpoint key>  for every header key that maps to no parameter (the vision tower, the MTP head if any).
# usage: python inkling2_provider_map.py CKPT_DIR SAFETENSORS OUT_MAP   (PYTHONPATH = the PR's src; not python -I)
import sys, json, struct, os
import torch
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopConfig
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import (rename_source_key, WeightRenaming, WeightConverter, dot_natural_key,
                                             Interleave, Chunk, Concatenate)

ckpt_dir, st_path, out_path = sys.argv[1:4]
torch.set_num_threads(4)

with open(st_path, "rb") as f:
    hn = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hn))
fsize = os.path.getsize(st_path)
header.pop("__metadata__", None)

cfg = Inkling2LoopConfig.from_pretrained(ckpt_dir)
tcfg = cfg.get_text_config()
with torch.device("meta"):
    model = Inkling2LoopForCausalLM(tcfg)
model = model.float()
meta = model.state_dict()

weight_mapping = get_model_conversion_mapping(model)
renamings = [e for e in weight_mapping if isinstance(e, WeightRenaming)]
converters = [e for e in weight_mapping if isinstance(e, WeightConverter)]
pattern_to_converter = {k: c for c in converters for k in c.source_patterns}


def op_name(op):
    n = type(op).__name__
    if isinstance(op, (Interleave, Chunk, Concatenate)):
        n += "(dim=%d)" % op.dim
        if getattr(op, "inverse", False):
            n += "(inverse)"
    return n


# the loader's loop (convert_and_load_state_dict_in_model, steps 1 and 2), without the tensors
targets = {}      # target -> (converter or None, {pattern: [keys]})
unmatched = []
for key in sorted(header, key=dot_natural_key):
    renamed, pat = rename_source_key(key, renamings, converters, model.base_model_prefix, meta)
    if renamed not in meta and key in meta:
        renamed, pat = rename_source_key(key, [], [], base_model_prefix=model.base_model_prefix, meta_state_dict=meta)
    if renamed not in meta:
        unmatched.append(key)
        continue
    conv = pattern_to_converter[pat] if pat is not None else None
    ent = targets.setdefault(renamed, (conv, {}))
    if ent[0] is not conv:
        sys.exit("target %s collects from two transforms" % renamed)
    ent[1].setdefault(pat if pat is not None else key, []).append(key)

missing = [k for k in meta if k not in targets]
lines = ["H %d %d" % (hn, fsize)]
for tgt in meta:   # the model's own parameter order
    if tgt not in targets:
        continue
    conv, by_pat = targets[tgt]
    pats = conv.source_patterns if conv is not None else list(by_pat)
    ops = ",".join(op_name(o) for o in conv.operations) if conv is not None else "rename"
    shape = list(meta[tgt].shape)
    row = ["T", tgt, ops, str(len(shape))] + [str(d) for d in shape] + [str(len(pats))]
    for p in pats:
        keys = by_pat.get(p, [])
        if len(keys) > 1:
            sys.exit("target %s: pattern %s collects %d keys" % (tgt, p, len(keys)))
        if not keys:
            row.append("-")
            continue
        k = keys[0]; h = header[k]
        row += [k, h["dtype"], str(len(h["shape"]))] + [str(d) for d in h["shape"]] + [str(h["data_offsets"][0]), str(h["data_offsets"][1])]
    lines.append(" ".join(row))
for k in unmatched:
    lines.append("U " + k)
with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")
n_vis = sum(1 for k in unmatched if k.startswith("model.visual."))
print("parameters %d, mapped %d, missing %d; checkpoint keys %d, unmatched %d (vision %d, other %d)" %
      (len(meta), len(meta) - len(missing), len(missing), len(header), len(unmatched), n_vis, len(unmatched) - n_vis))
for k in missing:
    print("MISSING", k)
for k in unmatched:
    if not k.startswith("model.visual."):
        print("UNMATCHED", k)
print("renamings %d, converters %d" % (len(renamings), len(converters)))
sys.exit(1 if missing else 0)
