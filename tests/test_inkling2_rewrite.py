"""R4.48.8.1: rewriting an inkling2_loop file (what derive writes and costs), on the tiny file (tests/inkling2_tiny.py).

Inkling 2 Mini's expert stacks are two-level NVFP4: E2M1 codes times a UE4M3 scale per 16 values, and per expert a
factor f[e] (blk.N.ffn_*_exps.scale) that the loader applies after the decode, for every stack type. So:
- a rewrite that keeps a stack keeps it and its factors bit for bit;
- one that re-quantizes a stack folds f[e] in, in the loader's order ((code * block_scale) * f[e]), and drops the
  .scale, which the loader would otherwise apply a second time;
- --quant-error costs those same folded rows, and a block the loop runs loop.count times costs loop.count times one
  application;
- --prune-experts slices the stacks, their factors, the router and the selection bias per physical layer, and refuses
  a plan that leaves the layers at different counts, since the Inkling loader reads one expert_count.

The factors here are powers of two. Scaling a row by one scales its Q8_0 block's f16 scale and leaves every code as
it was, so a fold shows exactly in the written bytes.
"""
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inkling2_tiny import E, FF, H, LOOP_END, LOOP_START, N_BLOCK, PASSES, T_F32, T_NVFP4, T_Q8_0  # noqa: E402
from inkling2_tiny import _kvs, _tensors, _write  # noqa: E402

T_MXFP4 = 39
FACTORS = [0.25, 0.5, 2.0, 1.0]
STACKS = ("ffn_gate_up_exps", "ffn_down_exps")
ROWS = {"ffn_gate_up_exps": 2 * FF, "ffn_down_exps": H}   # rows per expert
_SIZE = {T_F32: (4, 1), T_Q8_0: (34, 32), T_MXFP4: (17, 32), T_NVFP4: (36, 64), 30: (2, 1), 1: (2, 1)}


def _gguf(path):
    """(kv, {name: (dims, type, data bytes)}) of a GGUF v3 file."""
    b = Path(path).read_bytes()
    o = [4]

    def take(fmt):
        v = struct.unpack_from("<" + fmt, b, o[0])
        o[0] += struct.calcsize("<" + fmt)
        return v[0]

    def text():
        n = take("Q")
        o[0] += n
        return b[o[0] - n:o[0]].decode()

    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(t):
        if t == 8:
            return text()
        if t == 9:
            et, n = take("I"), take("Q")
            return [value(et) for _ in range(n)]
        return take(scalar[t])

    assert b[:4] == b"GGUF"
    take("I")
    nt, nkv = take("Q"), take("Q")
    kv = {}
    for _ in range(nkv):
        k = text()
        kv[k] = value(take("I"))
    infos = []
    for _ in range(nt):
        name = text()
        dims = [take("Q") for _ in range(take("I"))]
        infos.append((name, dims, take("I"), take("Q")))
    al = kv.get("general.alignment", 32)
    start = (o[0] + al - 1) // al * al
    out = {}
    for name, dims, typ, off in infos:
        n = 1
        for d in dims:
            n *= d
        ts, bs = _SIZE.get(typ, (0, 1))
        out[name] = (dims, typ, b[start + off:start + off + n // bs * ts] if ts else None)
    return kv, out


def _run(runner_bin, *args, force=False):
    env = dict(os.environ)
    env.pop("RUNNER_FORCE_REQUANT", None)
    env.pop("RUNNER_REQUANT_ONLY", None)
    if force:
        env["RUNNER_FORCE_REQUANT"] = "1"   # Q8_0 is wider than NVFP4: a rewrite declines it without this
    # the tiny vocabulary is bytes: a generation's output need not be UTF-8
    return subprocess.run([str(runner_bin), *map(str, args)], capture_output=True, encoding="utf-8", errors="replace",
                          env=env, timeout=300)


def _plan(path, rules):
    path.write_text(json.dumps({"default": "keep", "rules": [{"match": m, "type": t} for m, t in rules]}))
    return path


def _fit(runner_bin, model):
    return _run(runner_bin, "-m", model, "--info", "fit", "-c", "64", "--gpu", "off")


def _q8_blocks(data):
    """A Q8_0 tensor's blocks as (f16 scale, the 32 codes' bytes)."""
    return [(struct.unpack_from("<e", data, k)[0], data[k + 2:k + 34]) for k in range(0, len(data), 34)]


def _report(path):
    return {r["tensor"]: r for r in (json.loads(x) for x in Path(path).read_text().splitlines() if x.strip())}


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """The tiny file with FACTORS as every stack's per-expert factors, and the same stacks with every factor 1."""
    d = tmp_path_factory.mktemp("ink2rw")
    return (d, _write(d / "fac.gguf", _kvs(), _tensors(factors=FACTORS)),
            _write(d / "one.gguf", _kvs(), _tensors(factors=[1.0] * E)))


def test_keep_carries_the_stacks_and_their_factors_bit_for_bit(runner_bin, files):
    # a guard, not a red: main already copies a kept stack and its 1-D .scale as they are
    d, fac, _ = files
    out = d / "keep.gguf"
    r = _run(runner_bin, "-m", fac, "--quantize", out, "--type-plan",
             _plan(d / "keep.json", [("blk.0.attn_q.weight", "q8_0")]))
    assert r.returncode == 0, r.stderr
    _, src = _gguf(fac)
    _, got = _gguf(out)
    for b in range(N_BLOCK):
        for s in STACKS:
            for part in ("weight", "scale"):
                n = f"blk.{b}.{s}.{part}"
                assert got[n] == src[n], n
    assert got["blk.0.attn_q.weight"][1] == T_Q8_0
    assert _fit(runner_bin, out).returncode == 0


def test_a_requantized_stack_has_its_factors_folded_in_and_its_scale_dropped(runner_bin, files):
    d, fac, one = files
    plan = _plan(d / "q8.json", [("_exps.weight", "q8_0")])
    outs = {}
    for name, src in (("fac", fac), ("one", one)):
        outs[name] = d / f"q8-{name}.gguf"
        r = _run(runner_bin, "-m", src, "--quantize", outs[name], "--type-plan", plan, force=True)
        assert r.returncode == 0, r.stderr
    _, got = _gguf(outs["fac"])
    _, ref = _gguf(outs["one"])
    for b in range(N_BLOCK):
        for s in STACKS:
            assert f"blk.{b}.{s}.scale" not in got.keys(), f"blk.{b}.{s}.scale is still in the file: applied twice"
            n = f"blk.{b}.{s}.weight"
            assert got[n][1] == T_Q8_0
            per = ROWS[s] * (got[n][0][0] // 32)   # Q8_0 blocks per expert
            for k, ((dg, qg), (dr, qr)) in enumerate(zip(_q8_blocks(got[n][2]), _q8_blocks(ref[n][2]))):
                f = FACTORS[k // per]
                assert qg == qr and dg == dr * f, f"{n} block {k} (expert {k // per}): {dg} against {dr} x {f}"
    assert _fit(runner_bin, outs["fac"]).returncode == 0


def test_a_narrower_stack_drops_its_scale_and_the_file_loads(runner_bin, files):
    # MXFP4 (4.25 bits) is narrower than NVFP4 (4.5), so the rewrite takes it without RUNNER_FORCE_REQUANT, as a
    # ladder demotes; the tiny file's 64-wide rows hold no q3_k block, whose 32-block fallback is 4.5 bits and declined
    d, fac, _ = files
    out = d / "mxfp4.gguf"
    r = _run(runner_bin, "-m", fac, "--quantize", out, "--type-plan",
             _plan(d / "mxfp4.json", [("blk.1.ffn_down_exps.weight", "mxfp4")]))
    assert r.returncode == 0, r.stderr
    _, got = _gguf(out)
    assert got["blk.1.ffn_down_exps.weight"][1] == T_MXFP4
    assert "blk.1.ffn_down_exps.scale" not in got.keys(), "blk.1.ffn_down_exps.scale is still in the file"
    assert got["blk.1.ffn_gate_up_exps.scale"][1] == T_F32 and got["blk.1.ffn_gate_up_exps.weight"][1] == T_NVFP4
    f = _fit(runner_bin, out)
    assert f.returncode == 0, f.stderr


def test_quant_error_costs_the_folded_rows(runner_bin, files):
    # one factor of 1/4 on every expert: each round-trip difference is a quarter of the unit file's, its square 1/16
    d, _, one = files
    quarter = _write(d / "quarter.gguf", _kvs(), _tensors(factors=[0.25] * E))
    reps = {}
    for name, src in (("quarter", quarter), ("one", one)):
        reps[name] = d / f"qe-{name}.jsonl"
        r = _run(runner_bin, "-m", src, "--quant-error", reps[name], "--quant", "q8_0")
        assert r.returncode == 0, r.stderr
    q, u = _report(reps["quarter"]), _report(reps["one"])
    for b in range(N_BLOCK):
        for s in STACKS:
            n = f"blk.{b}.{s}.weight"
            eq, eu = q[n]["types"]["q8_0"]["error"], u[n]["types"]["q8_0"]["error"]
            assert eu > 0 and eq == pytest.approx(eu / 16, rel=1e-7), f"{n}: {eq} against {eu} / 16"
        n = f"blk.{b}.attn_k.weight"   # no factor: the same cost in both files
        assert q[n]["types"]["q8_0"]["error"] == u[n]["types"]["q8_0"]["error"]


def test_quant_error_counts_each_application_of_a_looped_block(runner_bin, files):
    d, fac, _ = files
    once = _write(d / "once.gguf", [(k, t, 1 if k == "inkling2_loop.loop.count" else v) for k, t, v in _kvs()],
                  _tensors(factors=FACTORS))
    reps = {}
    for name, src in (("loop", fac), ("once", once)):
        reps[name] = d / f"qe-{name}.jsonl"
        r = _run(runner_bin, "-m", src, "--quant-error", reps[name], "--quant", "q8_0,q3_k")
        assert r.returncode == 0, r.stderr
    lp, on = _report(reps["loop"]), _report(reps["once"])
    assert lp.keys() == on.keys()
    looped = 0
    for n in lp:
        b = int(n.split(".")[1]) if n.startswith("blk.") else -1
        times = PASSES if LOOP_START <= b < LOOP_END else 1
        looped += times > 1
        for t in ("q8_0", "q3_k"):
            a, c = lp[n]["types"][t], on[n]["types"][t]
            if a is None:
                assert c is None
                continue
            assert a["bytes"] == c["bytes"], n   # the bytes are the file's: one copy of the block
            assert a["error"] == pytest.approx(times * c["error"], rel=1e-7), f"{n} {t}: {a['error']} against {times} x"
    assert looped > 0


def _prune_plan(path, kept):
    path.write_text(json.dumps({f"layer_{b}": ids for b, ids in kept.items()}))
    return path


UNIFORM = {0: [0, 2], 1: [3, 1], 2: [1, 2], 3: [2, 0]}


def test_a_uniform_prune_slices_every_expert_tensor_and_loads(runner_bin, files):
    d, fac, _ = files
    out = d / "pruned.gguf"
    r = _run(runner_bin, "-m", fac, "--quantize", out, "--prune-experts", _prune_plan(d / "u.json", UNIFORM))
    assert r.returncode == 0, r.stderr
    kv, got = _gguf(out)
    _, src = _gguf(fac)
    assert kv["inkling2_loop.expert_count"] == 2
    for b, ids in UNIFORM.items():
        for s in STACKS:
            dims, typ, data = got[f"blk.{b}.{s}.weight"]
            assert dims[2] == 2 and typ == T_NVFP4
            sd = src[f"blk.{b}.{s}.weight"][2]
            per = len(sd) // E
            assert data == b"".join(sd[e * per:(e + 1) * per] for e in ids)
            sc = got[f"blk.{b}.{s}.scale"]
            assert sc[0] == [2], f"blk.{b}.{s}.scale is {sc[0]}, not sliced to the kept experts"
            assert list(struct.unpack("<2f", sc[2])) == [FACTORS[e] for e in ids]
        assert got[f"blk.{b}.ffn_gate_inp.weight"][0] == [H, 2]
        assert got[f"blk.{b}.exp_probs_b.bias"][0] == [2]
    f = _fit(runner_bin, out)
    assert f.returncode == 0, f.stderr
    g = _run(runner_bin, "-m", out, "--gpu", "off", "-t", "2", "-c", "64", "-p", "hello", "-n", "3", "--temp", "0")
    assert g.returncode == 0, g.stderr


def test_a_pruned_and_requantized_stack_folds_each_kept_experts_own_factor(runner_bin, files):
    d, fac, one = files
    plan = _plan(d / "pq8.json", [("_exps.weight", "q8_0")])
    pp = _prune_plan(d / "u2.json", UNIFORM)
    outs = {}
    for name, src in (("fac", fac), ("one", one)):
        outs[name] = d / f"pq8-{name}.gguf"
        r = _run(runner_bin, "-m", src, "--quantize", outs[name], "--prune-experts", pp, "--type-plan", plan,
                 force=True)
        assert r.returncode == 0, r.stderr
    _, got = _gguf(outs["fac"])
    _, ref = _gguf(outs["one"])
    for b, ids in UNIFORM.items():
        for s in STACKS:
            assert f"blk.{b}.{s}.scale" not in got.keys(), f"blk.{b}.{s}.scale is still in the file"
            n = f"blk.{b}.{s}.weight"
            per = ROWS[s] * (got[n][0][0] // 32)
            for k, ((dg, qg), (dr, qr)) in enumerate(zip(_q8_blocks(got[n][2]), _q8_blocks(ref[n][2]))):
                f = FACTORS[ids[k // per]]
                assert qg == qr and dg == dr * f, f"{n} block {k}: {dg} against {dr} x {f}"
    assert _fit(runner_bin, outs["fac"]).returncode == 0


@pytest.mark.parametrize("kept", [
    {0: [0, 1], 1: [0, 1, 2], 2: [0, 1], 3: [0, 1]},   # one layer keeps three
    {0: [0, 1], 1: [0, 1]},                            # two layers pruned, two left whole
], ids=["counts-differ", "some-layers-whole"])
def test_a_non_uniform_prune_is_refused_with_its_reason(runner_bin, files, kept):
    d, fac, _ = files
    out = d / "nonuniform.gguf"
    r = _run(runner_bin, "-m", fac, "--quantize", out, "--prune-experts", _prune_plan(d / "n.json", kept))
    assert r.returncode != 0, "a non-uniform prune of an inkling2_loop file was written"
    assert "inkling2_loop" in r.stderr and "one expert_count" in r.stderr, r.stderr
    assert not out.exists()
