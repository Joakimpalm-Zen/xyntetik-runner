"""`runner derive` (R17.6.1): make a file from a parent, measure it, write its card.

The toy parent is make-test-model.py's F32 model; the recipe rewrites it to
q4_0. The default reference is the parent file in Runner's own arithmetic:
both files are scored with `--score --score-top 20` and compared as
scripts/gold-logits.py compares a reference with a score file (KL over the
reference's top 20, renormalised, the other side floored where it lists
nothing; margin-qualified = the reference's top-1 to top-2 gap above 0.5
nats). The test recomputes that from the two score files on its own, so the
record's numbers are checked against an independent reading, not against
themselves.

The publisher reference (--reference hf:DIR) runs gold-logits.py; the test
points RUNNER_GOLD_LOGITS at a stand-in that writes a canned result, so the
card's pass and fail wording is checked without torch.
"""
import json
import math
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = (ROOT / "tests/fixtures/gold-corpus-2k.txt").read_text(encoding="utf-8")[:900]
WINDOW = "8:4:40"


def make_parent(tmp, name="parent", license_id="apache-2.0"):
    out = tmp / f"{name}.gguf"
    extra = ["--license", license_id] if license_id else []
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(out), *extra],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return out


def derive(runner_bin, parent, out, *extra, env=None):
    corpus = out.parent / "corpus.txt"
    corpus.write_text(CORPUS, encoding="utf-8")
    p = subprocess.run([str(runner_bin), "derive", "-m", str(parent), "-o", str(out),
                        "--corpus", str(corpus), "--window", WINDOW, "-c", "256", "-t", "2",
                        "--gpu", "off", *extra],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
                       env={**os.environ, **(env or {})})
    return p.returncode, p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace")


def gold_kld(gold, other):
    """scripts/gold-logits.py's kld(): KL(gold || other) over gold's tokens."""
    floor = min(other.values()) - 2.0 if other else -20.0
    keys = list(gold)
    g = [gold[k] for k in keys]
    o = [other.get(k, floor) for k in keys]
    gz = math.log(sum(math.exp(x) for x in g))
    oz = math.log(sum(math.exp(x) for x in o))
    return sum(math.exp(x - gz) * ((x - gz) - (y - oz)) for x, y in zip(g, o))


def tops(score):
    return [{i: lp for i, lp in row} for row in score["top"]]


@pytest.fixture(scope="module")
def made(runner_bin, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("derive")
    parent = make_parent(tmp)
    out = tmp / "child-q4_0.gguf"
    rc, so, se = derive(runner_bin, parent, out, "--quant", "q4_0")
    assert rc == 0, se[-1500:]
    return tmp, parent, out, so, se


def test_derive_writes_the_file_its_record_and_its_card(made):
    tmp, parent, out, so, se = made
    for suffix in ("", ".quant.json", ".derive.json", ".README.md"):
        assert pathlib.Path(str(out) + suffix).exists(), suffix
    rec = json.loads(pathlib.Path(str(out) + ".derive.json").read_text())
    assert rec["schema_version"] == "xyntetik.runner.derive.v1"
    assert rec["reference_kind"] == "parent-file"
    assert rec["recipe"]["quant"] == "q4_0"
    assert rec["parent"]["sha256"] and rec["file"]["sha256"]
    assert rec["parent"]["sha256"] != rec["file"]["sha256"]
    assert rec["file"]["bytes"] < rec["parent"]["bytes"]
    m = rec["measurement"]
    assert m["positions"] == 40 and m["window"] == {"start": 8, "stride": 4, "count": 40}
    assert m["mean_kld"] >= 0 and 0 <= m["top1_agreement_pct"] <= 100
    assert "bar" not in rec["measurement"] or rec["measurement"]["bar"] is None


def test_the_record_matches_an_independent_reading_of_the_two_score_files(made):
    tmp, parent, out, so, se = made
    rec = json.loads(pathlib.Path(str(out) + ".derive.json").read_text())
    ref = json.loads(pathlib.Path(rec["measurement"]["reference_score"]).read_text())
    run = json.loads(pathlib.Path(rec["measurement"]["file_score"]).read_text())
    assert ref["tokens"] == run["tokens"]
    k = ref.get("n_special", 0)
    rt, ft = tops(ref), tops(run)
    kl, agree, qual, qual_hit = [], 0, 0, 0
    for n in range(40):
        e = k + 8 + n * 4 - 1
        g, o = rt[e], ft[e]
        kl.append(gold_kld(g, o))
        gs = sorted(g.values(), reverse=True)
        a = max(g, key=g.get) == max(o, key=o.get)
        agree += a
        if gs[0] - gs[1] > 0.5:
            qual += 1
            qual_hit += a
    m = rec["measurement"]
    assert m["mean_kld"] == pytest.approx(sum(kl) / len(kl), rel=1e-6, abs=1e-9)
    assert m["max_kld"] == pytest.approx(max(kl), rel=1e-6, abs=1e-9)
    assert m["top1_agreement_pct"] == pytest.approx(100.0 * agree / 40)
    assert m["margin_qualified_positions"] == qual
    if qual:
        assert m["margin_qualified_top1_pct"] == pytest.approx(100.0 * qual_hit / qual)


def test_a_parent_file_card_names_its_reference_and_claims_no_bar(made):
    tmp, parent, out, so, se = made
    card = pathlib.Path(str(out) + ".README.md").read_text(encoding="utf-8")
    assert card.startswith("---\n")
    front, body = card.split("\n---\n", 1)
    assert "license: apache-2.0" in front
    head = "\n".join(l for l in body.strip().splitlines()[:6])
    assert "measured against the parent file in Runner's arithmetic, not the publisher's float32 model" in head
    low = card.lower()
    assert "passes the bar" not in low and "faithful" not in low
    assert "—" not in card
    assert str(tmp) not in card            # no host paths
    assert "runner derive" in card         # the recipe, reproducible


def test_derive_refuses_what_it_cannot_stand_behind(runner_bin, tmp_path):
    parent = make_parent(tmp_path)
    # no change from the parent
    rc, _, se = derive(runner_bin, parent, tmp_path / "same.gguf")
    assert rc != 0 and "recipe" in se
    # a parent without a licence: nothing to copy onto the card
    unlicensed = make_parent(tmp_path, "nolicense", license_id=None)
    rc, _, se = derive(runner_bin, unlicensed, tmp_path / "x.gguf", "--quant", "q4_0")
    assert rc != 0 and "license" in se
    assert not (tmp_path / "x.gguf").exists()
    # a window past the corpus
    rc, _, se = derive(runner_bin, parent, tmp_path / "y.gguf", "--quant", "q4_0", "--window", "8:4:4000")
    assert rc != 0 and "window" in se


def _fake_gold(tmp, mean, mq):
    script = tmp / "fake_gold.py"
    script.write_text(f'''import json, sys
a = sys.argv
out = a[a.index("--out") + 1]
json.dump({{"summary": {{"positions": 40, "sides": {{"a": {{
    "mean_kld_vs_gold": {mean}, "max_kld_vs_gold": {mean * 4}, "top1_agreement_pct": 96.0,
    "margin_qualified_top1_pct": {mq}}}}}, "reference_class": "FakeForCausalLM"}}, "rows": []}}, open(out, "w"))
''')
    return script


@pytest.mark.parametrize("mean,mq,passes", [(0.012, 99.0, True), (0.21, 90.0, False)])
def test_a_publisher_reference_card_says_pass_or_fail_first(runner_bin, tmp_path, mean, mq, passes):
    parent = make_parent(tmp_path)
    hf = tmp_path / "hf"
    hf.mkdir()
    fake = _fake_gold(tmp_path, mean, mq)
    out = tmp_path / "pub.gguf"
    rc, _, se = derive(runner_bin, parent, out, "--quant", "q4_0", "--reference", f"hf:{hf}",
                       env={"RUNNER_GOLD_LOGITS": str(fake)})
    assert rc == 0, se[-1500:]
    rec = json.loads(pathlib.Path(str(out) + ".derive.json").read_text())
    assert rec["reference_kind"] == "publisher-float32"
    assert rec["measurement"]["bar"]["passes"] is passes
    body = pathlib.Path(str(out) + ".README.md").read_text(encoding="utf-8").split("\n---\n", 1)[1]
    first = body.strip().splitlines()[0]
    assert ("passes the bar" in first) if passes else ("fails the bar" in first)


def test_the_publisher_reference_is_refused_without_its_python(runner_bin, tmp_path):
    parent = make_parent(tmp_path)
    hf = tmp_path / "hf"
    hf.mkdir()
    rc, _, se = derive(runner_bin, parent, tmp_path / "p.gguf", "--quant", "q4_0",
                       "--reference", f"hf:{hf}", env={"RUNNER_GOLD_PYTHON": "/nonexistent/python3"})
    assert rc != 0 and "torch" in se and "transformers" in se


def test_target_size_plans_as_type_plan_by_sensitivity_does(runner_bin, tmp_path):
    """--target-size BYTES --high T --low T: derive costs every tensor with
    --quant-error, demotes the cheapest groups (added error per saved byte)
    until the file fits, and rewrites with that plan. The plan must equal
    scripts/type-plan-by-sensitivity.py's on the same report and the same
    byte target (the reference implementation of the ranking), and the file
    must come in at or under the target."""
    parent = make_parent(tmp_path)
    report = tmp_path / "pre.jsonl"
    subprocess.run([str(runner_bin), "-m", str(parent), "--quant-error", str(report),
                    "--quant", "q8_0,q4_0"], check=True, capture_output=True)
    rows = [json.loads(l) for l in report.read_text().splitlines() if l.strip()]
    hi = sum(r["types"]["q8_0"]["bytes"] for r in rows)
    lo = sum(r["types"]["q4_0"]["bytes"] for r in rows)
    # halfway between all-high and all-low: some groups demoted, not all
    target = parent.stat().st_size - (hi - lo) // 2 - sum(
        r["rows"] * r["cols"] * 4 for r in rows) + hi
    out = tmp_path / "sized.gguf"
    rc, _, se = derive(runner_bin, parent, out, "--target-size", str(target),
                       "--high", "q8_0", "--low", "q4_0")
    assert rc == 0, se[-1500:]
    rec = json.loads(pathlib.Path(str(out) + ".derive.json").read_text())
    r = rec["recipe"]
    assert r["target_size"] == target and r["high"] == "q8_0" and r["low"] == "q4_0"
    assert 0 < len(r["demoted"]) < len({x["tensor"].split(".")[1] if x["tensor"].startswith("blk.")
                                        else x["tensor"] for x in rows})
    ours = json.loads(pathlib.Path(r["type_plan"]).read_text())
    p = subprocess.run([sys.executable, ROOT / "scripts/type-plan-by-sensitivity.py",
                        r["quant_error"], "--high", "q8_0", "--low", "q4_0",
                        "--save-bytes", str(r["save_bytes"])],
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    assert ours == json.loads(p.stdout)
    assert rec["file"]["bytes"] <= target
    # the target-size recipe writes its own plan and its own quant types
    rc, _, se = derive(runner_bin, parent, tmp_path / "z.gguf", "--target-size", str(target),
                       "--high", "q8_0", "--low", "q4_0", "--quant", "q4_0")
    assert rc != 0 and "--target-size" in se
    rc, _, se = derive(runner_bin, parent, tmp_path / "z.gguf", "--target-size", str(target))
    assert rc != 0 and "--high" in se


def test_files_beside_the_binary_are_found_from_its_real_directory(runner_bin, tmp_path):
    """derive finds scripts/ and the default corpus beside the binary itself:
    the directory of the resolved executable (readlink of /proc/self/exe on
    Linux, not the directory of that literal path). A copy in another
    directory looks beside the copy, and says where it looked."""
    import shutil
    other = tmp_path / "elsewhere"
    other.mkdir()
    copy = other / runner_bin.name
    shutil.copy2(runner_bin, copy)
    parent = make_parent(tmp_path)
    hf = tmp_path / "hf"
    hf.mkdir()
    rc, _, se = derive(copy, parent, tmp_path / "c.gguf", "--quant", "q4_0", "--reference", f"hf:{hf}")
    assert rc != 0
    # the path derive printed, in one separator style (Windows prints
    # backslashes), against where the copy's own scripts/ would be
    m = re.search(r"\((.*gold-logits\.py) is not readable\)", se)
    assert m, se[-600:]
    printed = m.group(1)
    assert ("/" in printed) != ("\\" in printed), printed   # not mixed
    want = os.path.realpath(other / "scripts" / "gold-logits.py")
    assert os.path.normcase(os.path.normpath(printed)) == os.path.normcase(os.path.normpath(want))
    assert "/proc/self" not in se
