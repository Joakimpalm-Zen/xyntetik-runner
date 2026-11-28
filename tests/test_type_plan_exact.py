"""R4.55: --type-plan's substring trap, the "name" rule and --type-plan-show.

A "match" rule is a substring of a tensor's name, so {"match": "output.weight"} also takes every
blk.N.attn_output.weight. That made a hand-written stock map wrong on 32 tensors of OLMo 3. Now:
- {"name": ...} takes the one tensor whose whole name it is; the first rule that applies wins, either kind;
- a "match" rule that is a tensor's whole role yet takes another role warns, and --type-plan-strict refuses it;
- a "name" rule that names no tensor warns, and strict refuses it; a "match" rule that takes nothing only warns;
- --type-plan-show lists, per tensor, the rule that applied and the type the rewrite would write, and writes nothing.

The files are generated here with OLMo 3's own tensor names (32 layers, 11 tensors each, plus the embedding, the
output norm and the head: 355, of which 226 are matrices) at a small width.
"""
import json
import os
import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
H, V, FF, LAYERS = 64, 64, 128, 32
T_F32, T_Q4_0, T_Q8_0 = 0, 2, 8
# the method scout's stock Q4_K_M map as it was written (research repo, S-DEF1): rule 0 is the trap
STOCK = {"default": "q4_k", "rules": [{"match": "output.weight", "type": "q6_k"}] + [
    {"match": f"blk.{b}.{r}.weight", "type": "q6_k"}
    for b in (0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28, 29, 30, 31) for r in ("attn_v", "ffn_down")]}


def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _write(path, tensors, arch="olmo2"):
    """A GGUF v3 file: general.architecture `arch`, alignment 32, and the tensors (name, dims, type, bytes)."""
    kv = _s("general.architecture") + struct.pack("<I", 8) + _s(arch)
    kv += _s("general.alignment") + struct.pack("<II", 4, 32)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 2) + kv
    off, infos, blobs = 0, b"", []
    for name, dims, typ, data in tensors:
        infos += _s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", typ, off)
        pad = (-len(data)) % 32
        blobs.append(data + b"\0" * pad)
        off += len(data) + pad
    body = head + infos
    body += b"\0" * ((-len(body)) % 32)
    path.write_bytes(body + b"".join(blobs))
    return path


def _f32(n, seed):
    import random
    r = random.Random(seed)
    return struct.pack(f"<{n}f", *[(r.random() - 0.5) * 0.2 for _ in range(n)])


def _q4_0(n_el):
    """Q4_0 blocks: an fp16 scale of 0.01 and 16 bytes of nibbles."""
    return (struct.pack("<e", 0.01) + bytes(range(16))) * (n_el // 32)


def olmo_tensors(q4_0=()):
    t, seed = [], 0

    def add(name, dims):
        nonlocal seed
        seed += 1
        n = 1
        for d in dims:
            n *= d
        t.append((name, dims, T_Q4_0, _q4_0(n)) if name in q4_0 else (name, dims, T_F32, _f32(n, seed)))

    add("token_embd.weight", [H, V])
    add("output_norm.weight", [H])
    add("output.weight", [H, V])
    for b in range(LAYERS):
        p = f"blk.{b}."
        for r, dims in (("attn_q", [H, H]), ("attn_k", [H, H]), ("attn_v", [H, H]), ("attn_output", [H, H]),
                        ("attn_q_norm", [H]), ("attn_k_norm", [H]), ("post_attention_norm", [H]),
                        ("ffn_gate", [H, FF]), ("ffn_up", [H, FF]), ("ffn_down", [FF, H]), ("post_ffw_norm", [H])):
            add(p + r + ".weight", dims)
    return t


@pytest.fixture(scope="module")
def olmo(tmp_path_factory):
    d = tmp_path_factory.mktemp("tpx")
    return d, _write(d / "olmo.gguf", olmo_tensors())


def _run(runner_bin, *args):
    env = {k: v for k, v in os.environ.items() if k != "RUNNER_FORCE_REQUANT"}
    return subprocess.run([str(runner_bin), *map(str, args)], capture_output=True, text=True, env=env, timeout=300)


def _plan(d, name, plan):
    p = d / f"{name}.json"
    p.write_text(json.dumps(plan))
    return p


def _show(runner_bin, model, plan, *extra):
    r = _run(runner_bin, "-m", model, "--type-plan", plan, "--type-plan-show", *extra)
    lines = [json.loads(x) for x in r.stdout.splitlines() if x.strip()]
    return r, {x["tensor"]: x for x in lines if x["kind"] == "tensor"}, [x for x in lines if x["kind"] == "rule"]


def test_the_stock_map_shows_rule_0_taking_33_tensors_and_warns(runner_bin, olmo):
    d, model = olmo
    r, tensors, rules = _show(runner_bin, model, _plan(d, "stock", STOCK))
    assert r.returncode == 0, r.stderr
    assert len(tensors) == 3 + 11 * LAYERS
    took = sorted(n for n, x in tensors.items() if x["rule"] == 0)
    assert took == sorted(["output.weight"] + [f"blk.{b}.attn_output.weight" for b in range(LAYERS)])
    assert rules[0]["caught"] == 33 and "use {\"name\"" in rules[0]["warning"], rules[0]
    assert "also takes 32 tensor(s) of another role" in r.stderr, r.stderr
    assert all(x["warning"] is None for x in rules[1:]), [x for x in rules[1:] if x["warning"]]


def test_a_name_rule_takes_its_one_tensor(runner_bin, olmo):
    d, model = olmo
    plan = dict(STOCK, rules=[{"name": "output.weight", "type": "q6_k"}] + STOCK["rules"][1:])
    r, tensors, rules = _show(runner_bin, model, _plan(d, "named", plan))
    assert r.returncode == 0, r.stderr
    assert [n for n, x in tensors.items() if x["rule"] == 0] == ["output.weight"]
    assert rules[0]["by"] == "name" and rules[0]["caught"] == 1 and rules[0]["warning"] is None
    assert tensors["blk.5.attn_output.weight"]["by"] == "default"
    assert "warning" not in r.stderr, r.stderr


def test_strict_refuses_the_traps_and_keeps_a_generic_match(runner_bin, olmo):
    d, model = olmo
    # the substring trap: strict refuses the real run and the listing alike
    out = d / "strict.gguf"
    r = _run(runner_bin, "-m", model, "--rewrite", out, "--type-plan", _plan(d, "stock2", STOCK), "--type-plan-strict")
    assert r.returncode != 0 and not out.exists() and "use {\"name\"" in r.stderr, r.stderr
    r, _, _ = _show(runner_bin, model, d / "stock2.json", "--type-plan-strict")
    assert r.returncode != 0
    # a "name" that names no tensor: warned, and refused under strict
    typo = _plan(d, "typo", {"default": "keep", "rules": [{"name": "outptu.weight", "type": "q8_0"}]})
    r, _, rules = _show(runner_bin, model, typo)
    assert r.returncode == 0 and rules[0]["caught"] == 0 and "names no tensor" in rules[0]["warning"]
    r, _, _ = _show(runner_bin, model, typo, "--type-plan-strict")
    assert r.returncode != 0
    # a "match" that takes nothing: warned, but strict keeps working (a generic plan on a smaller file)
    gen = _plan(d, "generic", {"default": "keep", "rules": [{"match": "_exps.weight", "type": "q4_0"},
                                                             {"match": "attn_v.weight", "type": "q8_0"}]})
    r, tensors, rules = _show(runner_bin, model, gen, "--type-plan-strict")
    assert r.returncode == 0, r.stderr
    assert rules[0]["caught"] == 0 and "takes no tensor" in rules[0]["warning"]
    assert rules[1]["caught"] == LAYERS and rules[1]["warning"] is None   # one role, every layer: no warning


@pytest.mark.parametrize("rule,why", [({"match": "a", "name": "b", "type": "q8_0"}, "exactly one of"),
                                      ({"type": "q8_0"}, "exactly one of")])
def test_a_rule_needs_exactly_one_of_match_and_name(runner_bin, olmo, rule, why):
    d, model = olmo
    r, _, _ = _show(runner_bin, model, _plan(d, "bad", {"default": "keep", "rules": [rule]}))
    assert r.returncode != 0 and why in r.stderr, r.stderr


def test_an_unknown_rule_key_is_named_once_and_ignored(runner_bin, olmo):
    d, model = olmo
    plan = {"default": "keep", "rules": [{"match": "attn_v.weight", "type": "q8_0", "exact": True},
                                         {"match": "attn_k.weight", "type": "q8_0", "exact": True}]}
    r, tensors, _ = _show(runner_bin, model, _plan(d, "unknown", plan))
    assert r.returncode == 0, r.stderr
    assert r.stderr.count("has the key \"exact\"") == 1, r.stderr
    assert tensors["blk.3.attn_v.weight"]["rule"] == 0   # still a substring match: the key changes nothing


def _gguf_types(path):
    b = Path(path).read_bytes()
    o = 4 + 4

    def take(fmt):
        nonlocal o
        v = struct.unpack_from("<" + fmt, b, o)
        o += struct.calcsize("<" + fmt)
        return v[0]

    def text():
        nonlocal o
        n = take("Q")
        o += n
        return b[o - n:o].decode()

    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(t):
        if t == 8:
            return text()
        if t == 9:
            et, n = take("I"), take("Q")
            return [value(et) for _ in range(n)]
        return take(scalar[t])

    nt, nkv = take("Q"), take("Q")
    for _ in range(nkv):
        text()
        value(take("I"))
    types = {}
    for _ in range(nt):
        name = text()
        [take("Q") for _ in range(take("I"))]
        types[name] = take("I")
        take("Q")
    return types


NAMES = {0: "f32", 1: "f16", 2: "q4_0", 6: "q5_0", 7: "q5_1", 8: "q8_0", 11: "q3_k", 12: "q4_k", 13: "q5_k", 14: "q6_k", 30: "bf16"}


def test_the_listing_is_what_the_rewrite_writes(runner_bin, tmp_path):
    """The gate: "written" in --type-plan-show equals the type in the file a real run writes, for every tensor. The
    file holds a declined rule (q8_0 asked of a Q4_0 tensor: not smaller) and width fallbacks (q4_k and q6_k on
    64-wide rows, which their 256-value blocks do not divide)."""
    model = _write(tmp_path / "mixed.gguf", olmo_tensors(q4_0={f"blk.{b}.ffn_up.weight" for b in range(LAYERS)}))
    plan = _plan(tmp_path, "mixed", {"default": "q4_k", "rules": [
        {"name": "output.weight", "type": "q6_k"}, {"match": "ffn_up.weight", "type": "q8_0"},
        {"match": "attn_v.weight", "type": "q8_0"}, {"match": "norm", "type": "keep"}]})
    r, tensors, _ = _show(runner_bin, model, plan)
    assert r.returncode == 0, r.stderr
    whys = {x["why"] for x in tensors.values()}
    assert any("fallback" in w for w in whys) and any("not smaller" in w for w in whys), whys
    out = tmp_path / "mixed-out.gguf"
    w = _run(runner_bin, "-m", model, "--rewrite", out, "--type-plan", plan)
    assert w.returncode == 0, w.stderr
    got = _gguf_types(out)
    assert set(got) == set(tensors)
    wrong = {n: (tensors[n]["written"], NAMES.get(t, t)) for n, t in got.items() if NAMES.get(t, str(t)) != tensors[n]["written"]}
    assert not wrong, list(wrong.items())[:5]


def test_inkling2_dense_nvfp4_reports_expert_stacks_reason_not_width(runner_bin, tmp_path):
    """On inkling2_loop, NVFP4 is offered on 3-D *_exps.weight expert stacks only. Asking nvfp4
    for a 2-D dense tensor whose row width divides by 16 must report that architecture policy,
    not 'no type on offer divides the row', while a genuine width mismatch keeps the width reason.
    Likewise, an existing non-scalar .scale companion reports the .scale refusal, not width."""
    tensors_in = [
        ("blk.0.attn_q.weight", [64, 64], T_F32, _f32(64 * 64, 1)),
        ("blk.0.attn_k.weight", [64, 64], T_F32, _f32(64 * 64, 2)),
        ("blk.0.ffn_gate_exps.weight", [64, 64, 2], T_F32, _f32(64 * 64 * 2, 3)),
    ]
    model = _write(tmp_path / "ink2.gguf", tensors_in, arch="inkling2_loop")

    # 1. --type-plan-show separates the policy refusal (dense tensor, row 64) from a real width mismatch (q4_k on row 64)
    plan_both = _plan(tmp_path, "ink2-both", {"default": "keep", "rules": [
        {"name": "blk.0.attn_q.weight", "type": "nvfp4"},
        {"name": "blk.0.attn_k.weight", "type": "q4_k"},
        {"name": "blk.0.ffn_gate_exps.weight", "type": "nvfp4"},
    ]})
    r, tensors, _ = _show(runner_bin, model, plan_both)
    assert r.returncode == 0, r.stderr
    assert tensors["blk.0.ffn_gate_exps.weight"]["written"] == "nvfp4"
    assert tensors["blk.0.ffn_gate_exps.weight"]["why"] == "as asked"
    assert tensors["blk.0.attn_q.weight"]["written"] == "f32"
    assert tensors["blk.0.attn_q.weight"]["why"] == "kept: NVFP4 is offered only on this architecture's expert stacks", tensors["blk.0.attn_q.weight"]
    assert tensors["blk.0.attn_k.weight"]["written"] == "q5_0"
    assert tensors["blk.0.attn_k.weight"]["why"] == "width fallback: the block does not divide the row", tensors["blk.0.attn_k.weight"]

    # 2. --type-plan-strict refuses a dense-tensor nvfp4 rule with the policy reason, not the width reason
    plan_dense = _plan(tmp_path, "ink2-dense", {"default": "keep", "rules": [
        {"name": "blk.0.attn_q.weight", "type": "nvfp4"},
    ]})
    rs = _run(runner_bin, "-m", model, "--rewrite", tmp_path / "out-strict.gguf",
              "--type-plan", plan_dense, "--type-plan-strict")
    assert rs.returncode != 0
    assert "expert stacks" in rs.stderr and "does not divide their row width" not in rs.stderr, rs.stderr
    assert not (tmp_path / "out-strict.gguf").exists()

    # 3. --type-plan-strict still refuses a genuine width mismatch with the width reason
    plan_width = _plan(tmp_path, "ink2-width", {"default": "keep", "rules": [
        {"name": "blk.0.attn_k.weight", "type": "q4_k"},
    ]})
    rw = _run(runner_bin, "-m", model, "--rewrite", tmp_path / "out-width.gguf",
              "--type-plan", plan_width, "--type-plan-strict")
    assert rw.returncode != 0
    assert "does not divide their row width" in rw.stderr, rw.stderr
    assert not (tmp_path / "out-width.gguf").exists()

    # 4. Non-inkling2_loop tensor with a non-scalar .scale companion reports the .scale refusal, not width
    gemma_model = _write(tmp_path / "gemma4.gguf", [
        ("blk.0.ffn_down_exps.weight", [64, 64, 2], T_F32, _f32(64 * 64 * 2, 4)),
        ("blk.0.ffn_down_exps.scale", [2], T_F32, _f32(2, 5)),
    ], arch="gemma4")
    plan_scale = _plan(tmp_path, "gemma4-scale", {"default": "keep", "rules": [
        {"name": "blk.0.ffn_down_exps.weight", "type": "nvfp4"},
    ]})
    rg, gtensors, _ = _show(runner_bin, gemma_model, plan_scale)
    assert rg.returncode == 0, rg.stderr
    assert gtensors["blk.0.ffn_down_exps.weight"]["written"] == "f32"
    assert gtensors["blk.0.ffn_down_exps.weight"]["why"] == "kept: NVFP4 would overwrite a tensor's existing .scale companion", gtensors["blk.0.ffn_down_exps.weight"]
    rgs = _run(runner_bin, "-m", gemma_model, "--rewrite", tmp_path / "out-scale.gguf",
               "--type-plan", plan_scale, "--type-plan-strict")
    assert rgs.returncode != 0
    assert ".scale companion" in rgs.stderr and "does not divide their row width" not in rgs.stderr, rgs.stderr
    assert not (tmp_path / "out-scale.gguf").exists()
