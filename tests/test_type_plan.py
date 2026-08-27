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
import os
import pathlib
import struct
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if os.name == "nt" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


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
