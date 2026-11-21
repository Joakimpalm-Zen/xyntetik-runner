# R4.48: the reference side of the provider identity test (tests/test_inkling2_provider.c). For each item of a
# manifest, the REFERENCE LOADER's float32: the source tensors read from the checkpoint with safetensors' safe_open,
# cast to float32 as the loader's _materialize_copy does for a float32 model, then the loader's own transform (the
# WeightConverter of transformers pull request 51050 at 6173a2f6, with its operation classes: Inkling2UnpadVocab,
# Inkling2LoopMergeMessageQueries, Inkling2LoopDequantizeExperts, Interleave; or a plain WeightRenaming), called
# through WeightTransform.add_tensor / convert as convert_and_load_state_dict_in_model calls it, with the model (on
# the meta device, float32) and the config passed. One tensor at a time.
#
# Manifest lines:
#   W <name>                   the whole parameter
#   X <name> <e,e,...>         an expert stack, those experts only: the sources' expert rows gathered first (the
#                              converter's ops act per expert: the dequantization in chunks of experts, the
#                              interleave along dim 1), so this is the converter on those experts
#   S <name>                   an expert stack whole, the converter on all its experts at once (the loader's call)
#   C <ctrl> <name> <e,...>    as X (the reference for a control)
# Output on stdout, per tensor or expert: "D <name> <e> <nbytes>\n" (e = -1 for a whole tensor), then the bytes.
# usage: python inkling2_provider_ref.py CKPT_DIR SAFETENSORS MANIFEST   (PYTHONPATH = the PR's src; not python -I)
import sys, copy, resource
import torch
from safetensors import safe_open
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopConfig
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import rename_source_key, WeightRenaming, WeightConverter

ckpt_dir, st_path, manifest = sys.argv[1:4]
torch.set_num_threads(4)
out = sys.stdout.buffer

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

f = safe_open(st_path, framework="pt")
sources = {}   # target -> (converter or None, [(pattern, key)])
for key in f.keys():
    renamed, pat = rename_source_key(key, renamings, converters, model.base_model_prefix, meta)
    if renamed not in meta:
        continue
    conv = pattern_to_converter[pat] if pat is not None else None
    sources.setdefault(renamed, (conv, []))[1].append((pat if pat is not None else key, key))


def load(key, experts=None):
    # the loader's materialize: slice, then .to(dtype=float32) (the param dtype)
    sl = f.get_slice(key)
    if experts is None:
        t = sl[...]
    else:
        t = torch.cat([sl[e:e + 1] for e in experts]) if len(sl.get_shape()) > 1 else sl[...][experts]
    return t.to(device="cpu", dtype=torch.float32)


def reference(target, experts=None):
    conv, srcs = sources[target]
    if conv is None:
        (pat, key), = srcs
        tr = WeightRenaming(key, target)
    else:
        tr = copy.deepcopy(conv)
    for pat, key in srcs:
        # an expert subset: the expert rows of the weight, the scale and amax; the denominator [1] is per tensor
        ex = experts if (experts is not None and "outer_scale_denominator" not in key) else None
        t = load(key, ex)
        tr.add_tensor(target, key, pat, (lambda t=t: t))
    res = tr.convert(target, model=model, config=tcfg)
    (name, val), = res.items()
    val = val[0] if isinstance(val, list) else val
    assert name == target, (name, target)
    assert val.dtype == torch.float32, val.dtype
    return val.contiguous()


def send(name, e, t):
    b = t.contiguous().numpy()
    out.write(b"D %s %d %d\n" % (name.encode(), e, b.nbytes))
    out.write(memoryview(b).cast("B"))


peak = 0
for line in open(manifest):
    w = line.split()
    if not w:
        continue
    if w[0] == "W":
        t = reference(w[1])
        send(w[1], -1, t)
    elif w[0] in ("X", "C"):
        name, ex = (w[1], w[2]) if w[0] == "X" else (w[2], w[3])
        experts = [int(x) for x in ex.split(",")]
        t = reference(name, experts)
        assert t.shape[0] == len(experts)
        for i, e in enumerate(experts):
            send(name, e, t[i])
    elif w[0] == "S":
        t = reference(w[1])
        for e in range(t.shape[0]):
            send(w[1], e, t[e])
    else:
        sys.exit("bad manifest line: " + line)
    del t
    out.flush()
print("ref: peak RSS %.2f GB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6), file=sys.stderr)
