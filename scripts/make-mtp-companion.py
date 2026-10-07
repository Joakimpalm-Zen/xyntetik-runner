#!/usr/bin/env python3
"""Build the Qwen3.8-Flash-Next MTP companion GGUF from the publisher's BF16
checkpoint: the 31 ``mtp.*`` tensors plus the embedding and LM head, fetched by
HTTP byte range (``fetch_mtp.py`` next to the raw directory writes ``meta.json``
with dtype, shape, source shard, byte offsets and sha256 per tensor).

The release trunk GGUF (ISTA-DASLab GSQ-RCO IQ3_S) carries blocks 0-47 and no
MTP head. The companion follows llama.cpp's own MTP-only layout for this
architecture (conversion/qwen4exp.py + _QwenMtpMixin, src/models/qwen4exp.cpp),
so one file serves both engines:

  * the header is the trunk's (hyperparameters and tokenizer copied byte for
    byte), with ``block_count`` 49 (llama.cpp counts the MTP block in it and
    takes the trunk depth as block_count - nextn), ``nextn_predict_layers = 1``,
    ``attention.compress_ratios`` extended by one entry for the MTP block, and
    no PLE keys (the head never reads the n-gram table);
  * the MTP block sits at index 48: its own tensors as ``blk.48.*`` and the
    head's as ``blk.48.nextn.*``; ``token_embd`` and ``output`` are included so
    the file loads on its own as a draft model;
  * every RMS-norm gamma is stored +1 (the converter's zero-centred-gamma rule
    for this family, which the trunk follows and the engines expect);
  * ``eh_proj = [fc_embedding | fc_hidden]`` along the input axis
    (graph_mtp: eh_proj . concat(enorm(e), hnorm_s(h_s)) per hc stream);
  * ``gate_up_proj`` splits gate-first into ``ffn_gate_exps`` / ``ffn_up_exps``;
    ``index_qk_proj`` splits into the 4-head q projection and the k projection.

Storage: norms F32; hyper-connection, router and indexer tensors BF16 as the
trunk keeps them; attention, expert, shared-expert, embedding and LM-head
matrices Q8_0 (``--bf16`` keeps those BF16 too). Provenance (source repo,
revision, per-tensor sha256, this script) goes into ``xyntetik.mtp.*`` keys.

    python3 scripts/make-mtp-companion.py --raw ~/xs-work/mtp-companion/raw \
        --meta ~/xs-work/mtp-companion/meta.json \
        --trunk ~/xs-work/mtp-companion/trunk-part1-head.bin \
        --out Qwen3.8-Flash-Next-MTP-Q8_0.gguf
"""
import argparse, json, os, struct
import numpy as np

GGUF_MAGIC = 0x46554747
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
SCALAR_FMT = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I", T_I32: "<i",
              T_F32: "<f", T_BOOL: "<?", T_U64: "<Q", T_I64: "<q", T_F64: "<d"}
TYPE_F32, TYPE_F16, TYPE_Q8_0, TYPE_BF16 = 0, 1, 8, 30
FTYPE_MOSTLY_Q8_0, FTYPE_MOSTLY_BF16 = 7, 32
MTP_BLOCK = 48


def gguf_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_bytes(key, typ, val):
    out = gguf_str(key) + struct.pack("<I", typ)
    if typ == T_STR:
        return out + gguf_str(val)
    if typ == T_ARR:
        et, items = val
        out += struct.pack("<IQ", et, len(items))
        for it in items:
            out += gguf_str(it) if et == T_STR else struct.pack(SCALAR_FMT[et], it)
        return out
    return out + struct.pack(SCALAR_FMT[typ], val)


class TrunkHeader:
    """The trunk's KV section, each entry kept as raw bytes and as a parsed value."""

    def __init__(self, path):
        self.f = open(path, "rb")
        f = self.f
        assert struct.unpack("<I", f.read(4))[0] == GGUF_MAGIC, "not a GGUF file"
        self.version = struct.unpack("<I", f.read(4))[0]
        self.n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        self.entries = []  # (key, type, value, raw bytes)
        for _ in range(n_kv):
            start = f.tell()
            key = self._str()
            typ = struct.unpack("<I", f.read(4))[0]
            val = self._value(typ)
            end = f.tell()
            f.seek(start)
            self.entries.append((key, typ, val, f.read(end - start)))

    def _str(self):
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8", "replace")

    def _value(self, typ):
        if typ == T_STR:
            return self._str()
        if typ == T_ARR:
            et, n = struct.unpack("<IQ", self.f.read(12))
            return (et, [self._value(et) for _ in range(n)])
        fmt = SCALAR_FMT[typ]
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]


def bf16_to_f32(a16):
    return (a16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(a32):
    """Round to nearest even, like the publisher's own BF16 storage."""
    u = np.ascontiguousarray(a32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16)


def quantize_q8_0(a16):
    """ggml Q8_0: blocks of 32 along the input axis, f16 scale + 32 int8. Takes the BF16
    matrix as (rows, cols) uint16 and converts a slab of rows at a time."""
    rows, cols = a16.shape
    assert cols % 32 == 0, cols
    out = np.empty((rows * (cols // 32), 34), dtype=np.uint8)
    step = max(1, (64 << 20) // (cols * 4))
    for r0 in range(0, rows, step):
        blocks = bf16_to_f32(np.ascontiguousarray(a16[r0:r0 + step])).reshape(-1, 32)
        amax = np.abs(blocks).max(axis=1)
        d = (amax / 127.0).astype(np.float32)
        inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0).astype(np.float32)
        q = np.rint(blocks * inv[:, None]).clip(-128, 127).astype(np.int8)
        o = out[r0 * (cols // 32):(r0 + step) * (cols // 32)]
        o[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
        o[:, 2:] = q.view(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", required=True, help="directory of <tensor>.bin files (raw BF16 bytes)")
    ap.add_argument("--meta", required=True, help="meta.json written by the fetch (dtype, shape, sha256)")
    ap.add_argument("--trunk", required=True, help="the trunk GGUF (or its first 12 MB): header source")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bf16", action="store_true", help="keep the large matrices BF16 instead of Q8_0")
    a = ap.parse_args()

    meta = json.load(open(a.meta))
    tensors = meta["tensors"]

    def load(name):
        m = tensors[name]
        assert m["dtype"] == "BF16", (name, m["dtype"])
        arr = np.memmap(os.path.join(a.raw, name + ".bin"), dtype=np.uint16, mode="r")
        assert arr.size == int(np.prod(m["shape"])), name
        return arr.reshape(m["shape"])  # PyTorch layout: (out, in) or (expert, out, in)

    # each entry is (gguf name, ggml type, logical shape in PyTorch order, producer); the producer
    # runs once, at write time, so only one tensor is in memory at a time
    out = []

    def shape_of(name):
        return tuple(tensors[name]["shape"])

    def norm(gname, name):
        # zero-centred gamma stored +1, F32
        out.append((gname, TYPE_F32, shape_of(name), lambda: bf16_to_f32(load(name)) + 1.0))

    def bf16(gname, name, fn=None, shape=None):
        out.append((gname, TYPE_BF16, shape or shape_of(name), fn or (lambda: load(name))))

    def big(gname, name, fn=None, shape=None):
        shape = shape or shape_of(name)
        fn = fn or (lambda: load(name))
        if a.bf16:
            out.append((gname, TYPE_BF16, shape, fn))
            return

        def produce():
            arr16 = fn()
            return quantize_q8_0(arr16.reshape(-1, arr16.shape[-1]))
        out.append((gname, TYPE_Q8_0, shape, produce))

    P = "mtp."
    B = f"blk.{MTP_BLOCK}."
    L0 = P + "layers.0."
    n_embd = shape_of(P + "fc_embedding.weight")[0]

    # the head: projections of [embedding ; trunk residual per stream] and the block's own mixer
    norm(B + "nextn.enorm.weight", P + "pre_fc_norm_embedding.weight")
    norm(B + "nextn.hnorm.weight", P + "pre_fc_norm_hidden.weight")
    bf16(B + "nextn.eh_proj.weight", None, shape=(n_embd, 2 * n_embd),
         fn=lambda: np.concatenate([load(P + "fc_embedding.weight"), load(P + "fc_hidden.weight")], axis=1))
    norm(B + "nextn.hc_head_norm.weight", P + "hyper_connection_mixer.hc_norm.weight")
    bf16(B + "nextn.hc_head_down.weight", P + "hyper_connection_mixer.input_mix_weight_down.weight")
    bf16(B + "nextn.hc_head_up.weight", P + "hyper_connection_mixer.input_mix_weight_up.weight")

    # the block: two hyper-connection modules, full attention with the indexer, MoE with the shared expert
    for hf, gg in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
        norm(B + gg + "_norm.weight", L0 + hf + ".hc_norm.weight")
        bf16(B + gg + "_down.weight", L0 + hf + ".input_mix_weight_down.weight")
        bf16(B + gg + "_up.weight", L0 + hf + ".input_mix_weight_up.weight")
        bf16(B + gg + "_inject.weight", L0 + hf + ".block_inject_weight.weight")
    big(B + "attn_q.weight", L0 + "self_attn.q_proj.weight")  # [q | gate] per head, as the checkpoint stores it
    big(B + "attn_k.weight", L0 + "self_attn.k_proj.weight")
    big(B + "attn_v.weight", L0 + "self_attn.v_proj.weight")
    big(B + "attn_output.weight", L0 + "self_attn.o_proj.weight")
    norm(B + "attn_q_norm.weight", L0 + "self_attn.q_norm.weight")
    norm(B + "attn_k_norm.weight", L0 + "self_attn.k_norm.weight")
    qk_name = L0 + "self_attn.indexer.index_qk_proj.weight"  # (4*128 + 128, n_embd): q heads first
    n_q = 4 * 128
    bf16(B + "indexer.q_proj.weight", None, shape=(n_q, n_embd), fn=lambda: load(qk_name)[:n_q])
    bf16(B + "indexer.k_proj.weight", None, shape=(shape_of(qk_name)[0] - n_q, n_embd),
         fn=lambda: load(qk_name)[n_q:])
    norm(B + "indexer.q_norm.weight", L0 + "self_attn.indexer.q_layernorm.weight")
    norm(B + "indexer.k_norm.weight", L0 + "self_attn.indexer.k_layernorm.weight")
    bf16(B + "ffn_gate_inp.weight", L0 + "mlp.gate.weight")
    gu_name = L0 + "mlp.experts.gate_up_proj"  # (n_expert, 2*n_ff, n_embd), gate rows first
    n_exp, n_ff2, _ = shape_of(gu_name)
    n_ff = n_ff2 // 2
    big(B + "ffn_gate_exps.weight", None, shape=(n_exp, n_ff, n_embd), fn=lambda: load(gu_name)[:, :n_ff, :])
    big(B + "ffn_up_exps.weight", None, shape=(n_exp, n_ff, n_embd), fn=lambda: load(gu_name)[:, n_ff:, :])
    big(B + "ffn_down_exps.weight", L0 + "mlp.experts.down_proj")  # (n_expert, n_embd, n_ff)
    big(B + "ffn_gate_shexp.weight", L0 + "mlp.shared_expert.gate_proj.weight")
    big(B + "ffn_up_shexp.weight", L0 + "mlp.shared_expert.up_proj.weight")
    big(B + "ffn_down_shexp.weight", L0 + "mlp.shared_expert.down_proj.weight")
    bf16(B + "ffn_gate_inp_shexp.weight", None, shape=(n_embd,),
         fn=lambda: load(L0 + "mlp.shared_expert_gate.weight").reshape(-1))

    # embedding and LM head, so the file loads alone as a draft model
    big("token_embd.weight", "model.language_model.embed_tokens.weight")
    big("output.weight", "lm_head.weight")

    # header: the trunk's entries, adjusted for an MTP-only file
    trunk = TrunkHeader(a.trunk)
    drop = {"general.name", "general.file_type", "general.size_label", "general.quantized_by",
            "general.repo_url", "split.no", "split.count", "split.tensors.count",
            "general.quantization_version"}
    kvs = []
    n_layer = None
    for key, typ, val, raw in trunk.entries:
        if key in drop or key.startswith("quantize.") or key.startswith("qwen4exp.ple.") \
                or key == "qwen4exp.embedding_length_per_layer_input":
            continue
        if key == "qwen4exp.block_count":
            # llama.cpp's convention: block_count counts the MTP block too (n_layer = block_count - nextn)
            n_layer = val
            assert val == MTP_BLOCK, val
            kvs.append(kv_bytes(key, T_U32, MTP_BLOCK + 1))
            continue
        if key == "qwen4exp.attention.compress_ratios":
            et, ratios = val
            ratio = max(ratios)  # the MTP block is a full-attention QSA layer: same ratio
            kvs.append(kv_bytes(key, T_ARR, (et, list(ratios) + [ratio])))
            continue
        kvs.append(raw)  # tokenizer and hyperparameters byte for byte
    assert n_layer == MTP_BLOCK
    kvs.append(kv_bytes("qwen4exp.nextn_predict_layers", T_U32, 1))
    kvs.append(kv_bytes("general.name", T_STR, "Qwen3.8-Flash-Next MTP head"))
    kvs.append(kv_bytes("general.file_type", T_U32, FTYPE_MOSTLY_BF16 if a.bf16 else FTYPE_MOSTLY_Q8_0))
    kvs.append(kv_bytes("general.quantization_version", T_U32, 2))
    kvs.append(kv_bytes("xyntetik.mtp.source_repo", T_STR, meta["repo"]))
    kvs.append(kv_bytes("xyntetik.mtp.source_revision", T_STR, meta.get("revision_sha", meta["revision"])))
    names = sorted(tensors)
    kvs.append(kv_bytes("xyntetik.mtp.source_tensors", T_ARR, (T_STR, names)))
    kvs.append(kv_bytes("xyntetik.mtp.source_sha256", T_ARR, (T_STR, [tensors[n]["sha256"] for n in names])))
    kvs.append(kv_bytes("xyntetik.mtp.builder", T_STR,
                        "xyntetik-runner scripts/make-mtp-companion.py: publisher BF16 weights, norms +1 in F32, "
                        + ("large matrices BF16" if a.bf16 else "large matrices Q8_0") + ", no calibration"))
    kvs.append(kv_bytes("xyntetik.mtp.trunk", T_STR, "companion to ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF IQ3_S"))
    kvs.append(kv_bytes("xyntetik.mtp.license", T_STR, "Qwen Community License 1.0, as the source checkpoint"))

    def nbytes(typ, shape):
        n = int(np.prod(shape))
        return {TYPE_F32: 4 * n, TYPE_BF16: 2 * n, TYPE_Q8_0: n // 32 * 34}[typ]

    infos = b""
    off = 0
    for name, typ, shape, _ in out:
        nb = name.encode("utf-8")
        dims = list(reversed(shape))
        infos += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims))
        infos += b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", typ, off)
        off = (off + nbytes(typ, shape) + 31) & ~31
    head = struct.pack("<IIQQ", GGUF_MAGIC, 3, len(out), len(kvs)) + b"".join(kvs) + infos
    with open(a.out, "wb") as f:
        f.write(head)
        f.write(b"\0" * ((-len(head)) % 32))
        for name, typ, shape, produce in out:
            arr = np.ascontiguousarray(produce())
            assert arr.nbytes == nbytes(typ, shape), (name, arr.nbytes, shape)
            f.write(arr.tobytes())
            f.write(b"\0" * ((-arr.nbytes) % 32))
            print(f"  {name} type={typ} ne={list(reversed(shape))} {arr.nbytes / 1e6:.1f} MB", flush=True)
            del arr
    print(f"wrote {a.out}: {len(out)} tensors, {os.path.getsize(a.out) / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
