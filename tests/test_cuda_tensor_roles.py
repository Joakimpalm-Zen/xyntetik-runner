"""Every weight tensor a layer binds has one CUDA role.

src/cuda.c reads a layer's tensors through two lists: layer_mixer_tensors()
(attention or the recurrent mixer, the hyper-connection mixers, the E-series
per-layer embedding projections) and layer_ffn_tensors_each() (the dense FFN
or the expert bank). The split plan's bytes, the prefix upload's reach, the
per-tensor bindings and admission's type check all walk them. A tensor in
neither is never budgeted or bound: that is how ssm_in went missing, and a
Mamba-2 hybrid under --cpu-moe failed on its first layer and ran on the CPU.
This test holds every gguf_tensor field of layer_t (src/model.h) to one of
the lists or to a named host-side role, so a new field needs a decision.
Source only: no device, no build.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

# layer_t tensor fields the device never reads through these lists, and why
HOST_SIDE = {
    "attn_norm": "uploaded as the f32 attn_norm_w table (shared_upload_layer_tables)",
    "ffn_norm": "uploaded as the f32 ffn_norm_w table (shared_upload_layer_tables)",
    "ple4_key": "the n-gram PLE block runs on the host (fwd_tile, ple4)",
    "ple4_value": "the n-gram PLE block runs on the host (fwd_tile, ple4)",
    "ple4_conv1d": "the n-gram PLE block runs on the host (fwd_tile, ple4)",
}


def layer_t_tensor_fields():
    src = (ROOT / "src/model.h").read_text()
    end = src.index("} layer_t;")
    start = src.rindex("typedef struct", 0, end)
    names = set()
    for line in src[start:end].splitlines():
        code = line.split("//")[0]
        if "gguf_tensor" not in code:
            continue
        decl = code.split("gguf_tensor", 1)[1]
        names.update(re.findall(r"\*+\s*(\w+)", decl))
    return names


def function_body(name):
    src = (ROOT / "src/cuda.c").read_text()
    m = re.search(r"\n(?:static )?[^\n;]*\b" + name + r"\([^;{]*\)\s*\{", src)
    assert m, f"{name} not found in src/cuda.c"
    j = src.index("\n}\n", m.end())
    return src[m.end():j]


def listed(name):
    return set(re.findall(r"ly->(\w+)", function_body(name)))


def test_layer_t_parse_sees_the_known_fields():
    fields = layer_t_tensor_fields()
    for f in ("wq", "ssm_in", "ffn_gate_exps", "moe_g", "hc_ffn_inject", "ple_proj"):
        assert f in fields, f


def test_every_layer_tensor_has_a_device_role():
    fields = layer_t_tensor_fields()
    roles = listed("layer_mixer_tensors") | listed("layer_ffn_tensors_each")
    missing = sorted(fields - roles - set(HOST_SIDE))
    assert not missing, (
        "layer_t tensors with no CUDA role: add each to layer_mixer_tensors() or "
        f"layer_ffn_tensors_each() in src/cuda.c, or to HOST_SIDE here: {missing}")


def test_roles_name_real_fields():
    fields = layer_t_tensor_fields()
    stale = sorted((listed("layer_mixer_tensors") | listed("layer_ffn_tensors_each")
                    | set(HOST_SIDE)) - fields
                   - {"is_moe", "moe_split", "moe_gemma"})
    assert not stale, f"roles naming no layer_t tensor field: {stale}"
