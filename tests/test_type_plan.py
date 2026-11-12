"""--type-plan: per-tensor precision, and the tensors it must not touch.

The lesson this test exists for is the depth-slice one: a rewriting tool is
only trustworthy if you can show that what it did NOT target came through byte
for byte. So the integrity assertion here is not "the file still loads", it is
"every tensor outside the plan's rules has identical bytes to the source".

It also pins the format limit that killed the hot/cold expert experiment:
experts are stored stacked, one 3-D tensor per layer carrying every expert with
a single type, so a rule can move a whole expert bank and nothing finer.
"""
import json
import pathlib
import struct
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def moe_model(tmp_path_factory):
    base = tmp_path_factory.mktemp("typeplan") / "m"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(base)],
                   check=True, cwd=ROOT)
    return pathlib.Path(f"{base}.moe1.gguf")


def read_tensors(path):
    """{name: (type, ne, raw_bytes)} straight from the GGUF."""
    f = open(path, "rb")
    _magic, _ver, ntens, nkv = struct.unpack("<IIQQ", f.read(24))

    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    def skip(t):
        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t == 8:
            rstr(); return
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                skip(et)
            return
        f.read(sz[t])

    align = 32
    kv = {}
    for _ in range(nkv):
        k = rstr()
        t = struct.unpack("<I", f.read(4))[0]
        if k == "general.alignment" and t == 4:
            align = struct.unpack("<I", f.read(4))[0]
        else:
            skip(t)
    metas = []
    for _ in range(ntens):
        name = rstr()
        nd = struct.unpack("<I", f.read(4))[0]
        ne = struct.unpack("<%dQ" % nd, f.read(8 * nd))
        ty = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        metas.append((name, ty, ne, off))
    pos = f.tell()
    data_start = (pos + align - 1) // align * align
    out = {}
    order = sorted(metas, key=lambda m: m[3])
    for i, (name, ty, ne, off) in enumerate(order):
        end = order[i + 1][3] if i + 1 < len(order) else None
        f.seek(data_start + off)
        raw = f.read((end - off) if end is not None else -1)
        out[name] = (ty, ne, raw)
    f.close()
    return out


def run_plan(runner_bin, src, out, plan, tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    return subprocess.run(
        [runner_bin, "-m", str(src), "--quantize", str(out), "--type-plan", str(p)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)


def test_untargeted_tensors_are_byte_identical(runner_bin, moe_model, tmp_path):
    out = tmp_path / "out"
    proc = run_plan(runner_bin, moe_model, out,
                    {"default": "keep",
                     "rules": [{"match": "_exps.weight", "type": "q4_0"}]},
                    tmp_path)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    src = read_tensors(moe_model)
    dst = read_tensors(pathlib.Path(out))
    assert set(src) == set(dst)
    touched = untouched = 0
    for name, (ty, _ne, raw) in src.items():
        dty, _dne, draw = dst[name]
        if "_exps.weight" in name and ty != dty:
            touched += 1
            continue
        assert dty == ty, f"{name}: type changed without a rule ({ty} -> {dty})"
        assert draw == raw, f"{name}: bytes changed without a rule"
        untouched += 1
    assert touched > 0, "the plan matched nothing; the test proves nothing"
    assert untouched > 0


def test_unknown_type_is_rejected_not_silently_ignored(runner_bin, moe_model, tmp_path):
    out = tmp_path / "bad"
    proc = run_plan(runner_bin, moe_model, out,
                    {"default": "keep", "rules": [{"match": "x", "type": "q2_k"}]},
                    tmp_path)
    assert proc.returncode != 0
    assert b"q2_k" in proc.stderr
    assert not pathlib.Path(out).exists()


def test_a_plan_never_grows_a_tensor(runner_bin, moe_model, tmp_path):
    # asking a q4_0-class tensor to become q8_0 must leave it alone rather than
    # inflate the file; the whole-file target path has the same guard
    out = tmp_path / "grow"
    first = tmp_path / "small"
    assert run_plan(runner_bin, moe_model, first,
                    {"default": "q4_0"}, tmp_path).returncode == 0
    small = pathlib.Path(first)
    proc = run_plan(runner_bin, small, out, {"default": "q8_0"}, tmp_path)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    a, b = read_tensors(small), read_tensors(pathlib.Path(out))
    for name, (ty, _ne, _raw) in a.items():
        assert b[name][0] == ty, f"{name} grew from {ty} to {b[name][0]}"


def test_experts_are_stacked_so_per_expert_precision_is_not_expressible(moe_model):
    # the format fact that killed the hot/cold experiment, pinned so nobody
    # re-scopes it without meeting the same wall
    t = read_tensors(moe_model)
    exps = [(n, ty, ne) for n, (ty, ne, _r) in t.items() if "_exps.weight" in n]
    assert exps, "fixture has no stacked expert tensors"
    for name, _ty, ne in exps:
        assert len(ne) == 3, f"{name} is {len(ne)}-D; expected a stacked 3-D bank"


def test_a_plan_that_selects_nothing_is_rejected(runner_bin, moe_model, tmp_path):
    """The container was never validated as strictly as the rules inside it.

    Every one of these loaded as "zero rules, keep everything": the quantize
    exited 0, printed `0 tensors converted`, and wrote a byte-identical copy
    of the input while the caller believed they had built a selective-
    precision artifact. `"rule"` for `"rules"` -- one character -- was enough,
    and the type plan is the mechanism behind the sub-16GB 30B result, so a
    plan that quietly does nothing is the worst kind of silence here.
    """
    bad_plans = [
        {"default": "keep", "rule": [{"match": "_exps.weight", "type": "q4_0"}]},
        {"rules": {"match": "_exps.weight", "type": "q4_0"}},
        {"default": ["keep"]},
        {"rules": []},
        {},
    ]
    for i, plan in enumerate(bad_plans):
        out = tmp_path / f"nothing{i}"
        proc = run_plan(runner_bin, moe_model, out, plan, tmp_path)
        assert proc.returncode != 0, f"{plan} was accepted"
        assert not pathlib.Path(out).exists()


def test_quant_and_type_plan_together_are_refused(runner_bin, moe_model, tmp_path):
    """--quant is never read on the plan path, so it was silently dropped.

    Worse than dropped: the summary line named the discarded whole-file type,
    so `--quant q8_0 --type-plan keep.json` reported
    `out.gguf (Q8_0): 0 tensors converted` over a byte-identical copy.
    """
    plan = tmp_path / "keep.json"
    plan.write_text(json.dumps({"default": "keep"}))
    out = tmp_path / "both.gguf"
    proc = subprocess.run(
        [runner_bin, "-m", str(moe_model), "--quantize", str(out),
         "--quant", "q8_0", "--type-plan", str(plan)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    assert proc.returncode != 0
    assert b"--quant and --type-plan" in proc.stderr
    assert not out.exists()


def test_a_plan_run_does_not_report_a_whole_file_type(runner_bin, moe_model,
                                                      tmp_path):
    """`target` decides nothing under a plan, so naming it described the
    output as a type no tensor was written in."""
    out = tmp_path / "labelled"
    proc = run_plan(runner_bin, moe_model, out,
                    {"default": "keep",
                     "rules": [{"match": "_exps.weight", "type": "q4_0"}]},
                    tmp_path)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    err = proc.stderr.decode(errors="replace")
    assert "per-tensor plan" in err, err
    assert "(Q4_0):" not in err, err


def test_a_same_size_decline_is_reported(runner_bin, tmp_path):
    """q4_0 and q4_K are both 144 bytes per 256 values.

    The never-grow rule therefore declines `--quant q4_k` over a Q4_0 file and
    hands back a byte-identical copy -- a real quality request (per-group
    scales and mins instead of one scale per 32) dropped without a word. The
    decline stands; it must not be silent, the way the block-width decline
    beside it never was.
    """
    # a 256-divisible row width, so the block-width guard is not what declines
    wide = tmp_path / "wide.gguf"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-model.py", "--wide",
         str(wide)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    small = tmp_path / "q40.gguf"
    assert subprocess.run(
        [runner_bin, "-m", str(wide), "--quantize", str(small),
         "--quant", "q4_0"], cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=180).returncode == 0
    out = tmp_path / "q4k.gguf"
    proc = subprocess.run(
        [runner_bin, "-m", str(small), "--quantize", str(out),
         "--quant", "q4_k"], cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=180)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert small.read_bytes() == out.read_bytes(), \
        "the fixture no longer exercises the same-size decline"
    assert b"not smaller than what they already carry" in proc.stderr, \
        proc.stderr.decode(errors="replace")


def test_type_plan_strict_refuses_a_plan_it_would_not_apply(runner_bin, tmp_path):
    """RI-3 (owner 2026-10-02): the default predicts, reports and builds; the
    opt-in --type-plan-strict fails before writing anything when a rule would
    be declined or fall back. The fixture's rows are narrower than q4_K's
    256-value block, so a q4_k rule cannot be applied as written."""
    src = tmp_path / "narrow.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(src)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"default": "keep",
                                "rules": [{"match": "weight", "type": "q4_k"}]}))
    lax = tmp_path / "lax.gguf"
    proc = subprocess.run([runner_bin, "-m", str(src), "--quantize", str(lax),
                           "--type-plan", str(plan)], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    assert proc.returncode == 0 and lax.exists(), proc.stderr.decode(errors="replace")
    reported = proc.stderr.decode(errors="replace")
    assert "fallback" in reported or "kept their own type" in reported, reported
    strict = tmp_path / "strict.gguf"
    proc = subprocess.run([runner_bin, "-m", str(src), "--quantize", str(strict),
                           "--type-plan", str(plan), "--type-plan-strict"], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    err = proc.stderr.decode(errors="replace")
    assert proc.returncode == 1 and "--type-plan-strict" in err, err
    assert not strict.exists() and not list(tmp_path.glob("strict.gguf*")), err
    # a plan that applies exactly is unaffected by the flag
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"default": "keep",
                              "rules": [{"match": "ffn", "type": "q8_0"}]}))
    exact = tmp_path / "exact.gguf"
    proc = subprocess.run([runner_bin, "-m", str(src), "--quantize", str(exact),
                           "--type-plan", str(ok), "--type-plan-strict"], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    assert proc.returncode == 0 and exact.exists(), proc.stderr.decode(errors="replace")
    proc = subprocess.run([runner_bin, "-m", str(src), "--quantize", str(exact) + "2",
                           "--quant", "q8_0", "--type-plan-strict"], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    assert proc.returncode == 1 and b"needs --type-plan" in proc.stderr


def test_a_q4_k_fallback_row_is_a_real_q5_0(runner_bin, tmp_path):
    """A row that divides by 32 but not by 256 cannot take Q4_K; it falls back
    to Q5_0, the 32-block type of the nearest bit budget. From v0.5.7 to
    1.12.0 there was no Q5_0 writer: the float row was copied under a Q5_0
    header, and the file gave NaN logits (SmolLM2-135M, 576 wide, the
    pre-1.12.0 sweep). The test model's rows are 64 wide, so every quantized
    tensor takes the fallback; the file must score, and close to its source.
    """
    base = tmp_path / "base.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(base)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    out = tmp_path / "q4k.gguf"
    proc = subprocess.run([runner_bin, "-m", str(base), "--quantize", str(out),
                           "--quant", "q4_k"], cwd=ROOT, capture_output=True,
                          timeout=180)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert b"Q5_0:" in proc.stderr, proc.stderr.decode(errors="replace")

    def nll(path):
        p = subprocess.run([runner_bin, "-m", str(path), "--gpu", "off", "--score",
                            "-p", "the runner trains the model on the text it reads"],
                           cwd=ROOT, capture_output=True, timeout=180)
        assert p.returncode == 0, p.stderr.decode(errors="replace")
        assert b"nan" not in p.stdout, "the fallback file scores NaN"
        return json.loads(p.stdout)["nll_mean"]

    src, q = nll(base), nll(out)
    assert abs(q - src) < 0.1 * max(1.0, abs(src)), (src, q)


def test_a_fallback_never_grows_a_tensor_or_rewrites_it_as_itself(runner_bin, tmp_path):
    """R6.68.84: a row a 256-wide block cannot describe (352 here) is written
    in the 32-block cousin, q5_0 for q4_k. The whole-file rule decided "is
    the tensor already small enough" against the type ASKED for, not the one
    it would write, and reported the fallback before it decided:
    - a Q4_0 file asked for q4_k had those rows requantized into Q5_0, a
      larger type (198 bytes a row became 242), from already-quantized values;
    - a file that already carried the fallback was "written" in it again;
    - the log named a type "wrote" for a tensor that kept its own.
    The plan's copy of the rule had the first two right. One rule now."""
    wide = tmp_path / "wide.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(wide),
                    "--wide", "--n-ff", "352"], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    Q4_0, Q5_0, Q4_K = 2, 6, 12

    def requant(src, out, *how):
        p = subprocess.run([str(runner_bin), "-m", str(src), "--quantize", str(out), *how],
                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        assert p.returncode == 0, p.stderr.decode(errors="replace")[-400:]
        return p.stderr.decode(errors="replace")

    down = "blk.0.ffn_down.weight"
    q40 = tmp_path / "q40.gguf"
    requant(wide, q40, "--quant", "q4_0")
    src = read_tensors(q40)
    assert src[down][0] == Q4_0 and src[down][1][0] == 352
    # a whole-file target, and the same request as a plan: neither may grow it
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"default": "q4_k"}))
    for how in (("--quant", "q4_k"), ("--type-plan", str(plan))):
        out = tmp_path / ("out-%s.gguf" % how[0].strip("-"))
        err = requant(q40, out, *how)
        got = read_tensors(out)
        assert got[down][0] == Q4_0 and got[down][2] == src[down][2], (how, got[down][0])
        assert "wrote Q5_0" not in err and "32-block fallback type" not in err, (how, err[-600:])
        assert f"{down} wanted Q4_K" in err or "more" in err, (how, err[-600:])
    # the fallback already in place is kept, not written again
    q4k = tmp_path / "q4k.gguf"
    err = requant(wide, q4k, "--quant", "q4_k")
    assert f"{down} wanted Q4_K, wrote Q5_0" in err
    again = tmp_path / "again.gguf"
    err = requant(q4k, again, "--quant", "q4_k")
    assert read_tensors(again)[down][0] == Q5_0
    assert "32-block fallback type" not in err and ", wrote " not in err, err[-600:]
    assert read_tensors(again)[down][2] == read_tensors(q4k)[down][2]


def test_the_error_report_is_json_whatever_a_tensor_is_called(runner_bin, tmp_path):
    """R6.68.85: --quant-error wrote each tensor's name between quotes as the
    file spells it. A GGUF name is any string without a NUL, so one holding a
    quote made its line of the report unreadable to the planner."""
    wide = tmp_path / "wide.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--wide", str(wide)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    raw, name, odd_name = wide.read_bytes(), b"blk.0.attn_q.weight", b'blk.0.attn"q.weight'
    assert raw.count(name) == 1
    odd = tmp_path / "odd.gguf"
    odd.write_bytes(raw.replace(name, odd_name))
    report = tmp_path / "qe.jsonl"
    p = subprocess.run([str(runner_bin), "-m", str(odd), "--quant-error", str(report),
                        "--quant", "q4_0"], cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")[-300:]
    rows = [json.loads(l) for l in report.read_text().splitlines() if l.strip()]
    assert odd_name.decode() in [r["tensor"] for r in rows]


def test_a_type_name_it_cannot_write_lists_the_ones_it_can(runner_bin, tmp_path):
    """`--quant Q4_K_M` (llama.cpp's mix name) or `q4_K` read only "is not a
    writable type" (the pre-1.12.0 sweep). The refusal now lists the names,
    and every listed name is one --quant takes (src/quants.h QUANT_TYPE_NAMES
    against type_from_name)."""
    base = tmp_path / "base.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(base)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    p = subprocess.run([runner_bin, "-m", str(base), "--quantize", str(tmp_path / "x.gguf"),
                        "--quant", "Q4_K_M"], cwd=ROOT, capture_output=True, text=True,
                       timeout=60)
    assert p.returncode != 0
    line = next(l for l in p.stderr.splitlines() if "not a writable type" in l)
    names = line.split("(one of ")[1].rstrip(")").split("|")
    assert "q4_k" in names and "keep" in names, line
    header = (ROOT / "src/quants.h").read_text()
    assert f'#define QUANT_TYPE_NAMES "{"|".join(names)}"' in header
    for name in names:
        q = subprocess.run([runner_bin, "-m", str(base), "--quantize",
                            str(tmp_path / f"{name}.gguf"), "--quant", name],
                           cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert q.returncode == 0, (name, q.stderr)


def test_q5_k_q5_1_and_iq4_nl_are_written_as_asked_and_score(runner_bin, tmp_path):
    """R4.33: q5_k, q5_1 and iq4_nl are writable from --quant and --type-plan.
    Each writes its own type (ggml ids 13, 7 and 20) on a 256-divisible
    fixture, and the file scores close to its source. A q5_k rule on rows
    that are not a multiple of 256 writes Q5_1, its 32-block cousin, as
    Mellum 2.1's 896-wide expert down projections need. The writers' bytes
    are ggml's (tests/test_quantize.c, the quant-golden twins)."""
    wide = tmp_path / "wide.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--wide",
                    str(wide)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    narrow = tmp_path / "narrow.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(narrow)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)

    def nll(path):
        p = subprocess.run([runner_bin, "-m", str(path), "--gpu", "off", "--score",
                            "-p", "the runner trains the model on the text it reads"],
                           cwd=ROOT, capture_output=True, timeout=180)
        assert p.returncode == 0, p.stderr.decode(errors="replace")
        assert b"nan" not in p.stdout, (path, p.stdout)
        return json.loads(p.stdout)["nll_mean"]

    src = nll(wide)
    src_t = read_tensors(wide)
    quantized = [n for n, (_ty, ne, _raw) in src_t.items() if len(ne) == 2 and "embd" not in n]
    assert quantized
    for name, gid in (("q5_k", 13), ("q5_1", 7), ("iq4_nl", 20)):
        out = tmp_path / f"{name}.gguf"
        proc = subprocess.run([runner_bin, "-m", str(wide), "--quantize", str(out),
                               "--quant", name], cwd=ROOT, capture_output=True, timeout=180)
        assert proc.returncode == 0, (name, proc.stderr.decode(errors="replace"))
        dst = read_tensors(out)
        assert any(dst[n][0] == gid for n in quantized), (name, {n: dst[n][0] for n in quantized})
        q = nll(out)
        assert abs(q - src) < 0.1 * max(1.0, abs(src)), (name, src, q)

    plan = {"default": "keep", "rules": [{"match": "ffn_down", "type": "q5_1"},
                                         {"match": "attn", "type": "iq4_nl"},
                                         {"match": "ffn", "type": "q5_k"}]}
    mixed = tmp_path / "mixed.gguf"
    proc = run_plan(runner_bin, wide, mixed, plan, tmp_path)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    dst = read_tensors(mixed)
    seen = {dst[n][0] for n in quantized}
    for n in quantized:
        want = 7 if "ffn_down" in n else 20 if "attn" in n else 13 if "ffn" in n else src_t[n][0]
        assert dst[n][0] == want, (n, dst[n][0], want)
    assert {7, 13, 20} <= seen, seen
    nll(mixed)

    fb = tmp_path / "narrow-q5k.gguf"
    proc = subprocess.run([runner_bin, "-m", str(narrow), "--quantize", str(fb),
                           "--quant", "q5_k"], cwd=ROOT, capture_output=True, timeout=180)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert b"Q5_1:" in proc.stderr, proc.stderr.decode(errors="replace")
    narrow_t = read_tensors(narrow)
    assert any(t == 7 for n, (t, ne, _r) in read_tensors(fb).items()
               if len(ne) == 2 and narrow_t[n][0] != t), "no tensor took the Q5_1 fallback"
    nll(fb)
