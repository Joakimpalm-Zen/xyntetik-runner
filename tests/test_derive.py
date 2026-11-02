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
import shutil
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
    # the score file covers the span gold-logits.py reads: its special prefix,
    # then ids[:min_prefix + max_positions * stride + 1] (gold-logits.py's
    # `body`); one token short and the real script refuses the file after its
    # whole float32 forward (found on Qwen3-30B-A3B, 2026-11-01)
    sc = json.loads(pathlib.Path(rec["measurement"]["file_score"]).read_text())
    assert len(sc["tokens"]) >= sc.get("n_special", 0) + 8 + 40 * 4 + 1
    body = pathlib.Path(str(out) + ".README.md").read_text(encoding="utf-8").split("\n---\n", 1)[1]
    paras = [p for p in body.strip().split("\n\n")]
    # what the file is and is not for comes first, then the weakest number,
    # then the bar line (the card rules: limits first, the unflattering
    # metric before the flattering one)
    assert "not the publisher's model" in paras[0] and "measured on one text only" in paras[0]
    assert paras[1].startswith("Weakest number first:") and "top-1 agreement" in paras[1]
    assert ("passes the bar" in paras[2]) if passes else ("fails the bar" in paras[2])


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


def test_reference_args_reach_the_reference_script(runner_bin, tmp_path):
    """--reference-arg ARG (repeatable) passes ARG to gold-logits.py as given:
    how a reference too large for float32 in memory is run (--upcast-at-use,
    or --offload-dir DIR --max-memory GIB, #528). It belongs to --reference
    and is refused without it, rather than ignored."""
    parent = make_parent(tmp_path)
    hf = tmp_path / "hf"
    hf.mkdir()
    seen = tmp_path / "argv.json"
    script = tmp_path / "fake_gold_argv.py"
    script.write_text(f'''import json, sys
a = sys.argv
json.dump(a, open({str(seen)!r}, "w"))
json.dump({{"summary": {{"positions": 40, "sides": {{"a": {{"mean_kld_vs_gold": 0.01,
    "max_kld_vs_gold": 0.1, "top1_agreement_pct": 99.0, "margin_qualified_top1_pct": 100.0}}}},
    "reference_class": "FakeForCausalLM", "reference_dtype": "float32 arithmetic on weights stored at torch.bfloat16, cast at use per decoder block (2 blocks)"}}, "rows": []}}, open(a[a.index("--out") + 1], "w"))
''')
    rc, _, se = derive(runner_bin, parent, tmp_path / "r.gguf", "--quant", "q4_0",
                       "--reference", f"hf:{hf}", "--reference-arg", "--upcast-at-use",
                       "--reference-arg", "--trust-remote-code",
                       env={"RUNNER_GOLD_LOGITS": str(script)})
    assert rc == 0, se[-1500:]
    argv = json.loads(seen.read_text())
    assert "--upcast-at-use" in argv and "--trust-remote-code" in argv
    rec = json.loads(pathlib.Path(str(tmp_path / "r.gguf") + ".derive.json").read_text())
    assert rec["reference"]["args"] == ["--upcast-at-use", "--trust-remote-code"]
    # the card names the reference's arithmetic as the instrument reported it
    assert rec["reference"]["dtype"].startswith("float32 arithmetic on weights stored at torch.bfloat16")
    card = pathlib.Path(str(tmp_path / "r.gguf") + ".README.md").read_text()
    assert "float32 arithmetic, BF16 weights upcast at use" in card
    rc, _, se = derive(runner_bin, parent, tmp_path / "s.gguf", "--quant", "q4_0",
                       "--reference-arg", "--upcast-at-use")
    assert rc != 0 and "--reference-arg" in se


def test_license_flag_fills_a_missing_license_and_never_overrides(runner_bin, tmp_path):
    """A parent that declares no general.license (Qwen's own Qwen3-30B-A3B
    GGUFs, whose repository states Apache-2.0) can be derived with
    --license ID; the card says the licence was given, not copied, and the
    record names its source. A parent that declares one keeps it: a
    different --license is refused, so derive never relicenses a file."""
    bare = make_parent(tmp_path, "bare", license_id=None)
    out = tmp_path / "given.gguf"
    rc, _, se = derive(runner_bin, bare, out, "--quant", "q4_0", "--license", "apache-2.0")
    assert rc == 0, se[-1500:]
    card = pathlib.Path(str(out) + ".README.md").read_text()
    assert "license: apache-2.0" in card.split("\n---\n", 1)[0]
    assert "given with `--license`" in card
    rec = json.loads(pathlib.Path(str(out) + ".derive.json").read_text())
    assert rec["parent"]["license"] == "apache-2.0" and rec["parent"]["license_source"] == "--license"
    licensed = make_parent(tmp_path, "mit", license_id="mit")
    rc, _, se = derive(runner_bin, licensed, tmp_path / "x.gguf", "--quant", "q4_0",
                       "--license", "apache-2.0")
    assert rc != 0 and "license" in se and not (tmp_path / "x.gguf").exists()
    rc, _, se = derive(runner_bin, licensed, tmp_path / "y.gguf", "--quant", "q4_0", "--license", "mit")
    assert rc == 0, se[-800:]


def test_card_from_record_rewrites_the_card_without_measuring(made, runner_bin, tmp_path):
    """derive writes its card from the record it just wrote, so the record is
    the card's one source; `runner derive --card-from REC` rewrites the card
    from the record alone (a newer runner's wording, no new measurement) and
    gives the same bytes derive wrote."""
    _, _, out, _, _ = made
    rec = pathlib.Path(str(out) + ".derive.json")
    card = pathlib.Path(str(out) + ".README.md")
    want = card.read_bytes()
    redo = tmp_path / "again.README.md"
    p = subprocess.run([str(runner_bin), "derive", "--card-from", str(rec), "-o", str(redo)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert redo.read_bytes() == want
    # without -o: the card beside the record, under the file's own name
    card.unlink()
    p = subprocess.run([str(runner_bin), "derive", "--card-from", str(rec)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert card.read_bytes() == want
    bad = tmp_path / "not-a-record.json"
    bad.write_text('{"schema_version": "something.else"}')
    p = subprocess.run([str(runner_bin), "derive", "--card-from", str(bad), "-o", str(tmp_path / "x.md")],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "derive record" in p.stderr


def test_card_from_an_older_record_reads_the_arithmetic_from_its_gold_run(runner_bin, tmp_path):
    """A record from before reference.dtype existed: the card takes the
    reference's arithmetic from the gold run's own summary instead."""
    parent = make_parent(tmp_path)
    hf = tmp_path / "hf"
    hf.mkdir()
    script = tmp_path / "fake_gold_dtype.py"
    script.write_text('''import json, sys
a = sys.argv
json.dump({"summary": {"positions": 40, "sides": {"a": {"mean_kld_vs_gold": 0.01,
    "max_kld_vs_gold": 0.1, "top1_agreement_pct": 99.0, "margin_qualified_top1_pct": 100.0}},
    "reference_class": "FakeForCausalLM", "reference_dtype": "float32 arithmetic on weights stored at torch.bfloat16, cast at use per decoder block (2 blocks)"}, "rows": []}, open(a[a.index("--out") + 1], "w"))
''')
    out = tmp_path / "old.gguf"
    rc, _, se = derive(runner_bin, parent, out, "--quant", "q4_0", "--reference", f"hf:{hf}",
                       env={"RUNNER_GOLD_LOGITS": str(script)})
    assert rc == 0, se[-1200:]
    rec = pathlib.Path(str(out) + ".derive.json")
    d = json.loads(rec.read_text())
    del d["reference"]["dtype"]
    rec.write_text(json.dumps(d))
    p = subprocess.run([str(runner_bin), "derive", "--card-from", str(rec)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "float32 arithmetic, BF16 weights upcast at use" in pathlib.Path(str(out) + ".README.md").read_text()


def test_base_model_and_credits_name_what_the_files_cannot(runner_bin, tmp_path):
    """A parent GGUF without a repo URL and a reference held in a local folder
    leave the card unable to name the publisher's repository; --base-model
    REPO[@REV] names it (front matter base_model, and the reference by name and
    revision), and --credit TEXT (repeatable) credits inputs such as someone
    else's importance matrix. Both go into the record, and --card-from can
    give them for a record that lacks them, without measuring again."""
    parent = make_parent(tmp_path)
    hf = tmp_path / "hf"
    hf.mkdir()
    out = tmp_path / "named.gguf"
    rc, _, se = derive(runner_bin, parent, out, "--quant", "q4_0", "--reference", f"hf:{hf}",
                       "--base-model", "Org/Model-7B@abc1234", "--credit", "imatrix by someone (Apache-2.0)",
                       env={"RUNNER_GOLD_LOGITS": str(_fake_gold(tmp_path, 0.01, 99.0))})
    assert rc == 0, se[-1200:]
    card = pathlib.Path(str(out) + ".README.md").read_text()
    front, body = card.split("\n---\n", 1)
    assert "base_model: Org/Model-7B" in front and "abc1234" not in front
    assert "`Org/Model-7B` at `abc1234`" in body
    assert "## Credits" in body and "- imatrix by someone (Apache-2.0)" in body
    rec = json.loads(pathlib.Path(str(out) + ".derive.json").read_text())
    assert rec["parent"]["base_model_repo"] == "Org/Model-7B@abc1234"
    assert rec["credits"] == ["imatrix by someone (Apache-2.0)"]
    # an older record without them: --card-from fills them in
    del rec["credits"]
    rec["parent"]["base_model_repo"] = None
    old = tmp_path / "old.derive.json"
    old.write_text(json.dumps(rec))
    card2 = tmp_path / "old.md"
    p = subprocess.run([str(runner_bin), "derive", "--card-from", str(old), "-o", str(card2),
                        "--base-model", "Org/Model-7B@abc1234", "--credit", "imatrix by someone (Apache-2.0)"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert card2.read_text() == card


def _fake_gold_rows(tmp, name, rows_a, rows_b=None, mean=0.01, mq=99.0):
    """A stand-in gold run with per-position rows (pos, gold_margin, a/b kld and agree)."""
    rows = []
    for i, (k, ag) in enumerate(rows_a):
        r = {"pos": 8 + 4 * i, "gold_margin": 1.0 if i % 3 else 0.2, "a": {"kld": k, "agree": ag}}
        if rows_b:
            r["b"] = {"kld": rows_b[i][0], "agree": rows_b[i][1]}
        rows.append(r)
    out = tmp / name
    out.write_text(json.dumps({"summary": {"positions": len(rows), "reference_class": "FakeForCausalLM",
        "sides": {"a": {"mean_kld_vs_gold": sum(k for k, _ in rows_a) / len(rows_a)}}}, "rows": rows}))
    return out


def test_compare_renders_other_files_in_the_same_window_unflattering_first(runner_bin, tmp_path):
    """--compare LABEL=GOLD.json[:SIDE][:BYTES] (repeatable) puts other files
    measured in the same window on the card, with the paired tests: Wilcoxon on
    per-position KLD and McNemar on margin-qualified top-1, the numbers
    scripts/gold-logits.py computes (its functions are the anchor here). A file
    closer than this one is named before a file this one beats."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gold_logits", ROOT / "scripts" / "gold-logits.py")
    gl = importlib.util.module_from_spec(spec); spec.loader.exec_module(gl)
    n = 40
    ours = [(0.010 + 0.0005 * (i % 7), True) for i in range(n)]
    better = [(0.007 + 0.0004 * (i % 5), True) for i in range(n)]
    worse = [(0.090 + 0.001 * (i % 9), i % 4 != 0) for i in range(n)]
    parent = make_parent(tmp_path)
    hf = tmp_path / "hf"
    hf.mkdir()
    script = tmp_path / "fake_gold_rows.py"
    script.write_text(f'''import json, shutil, sys
a = sys.argv
shutil.copy({str(tmp_path / "ours.json")!r}, a[a.index("--out") + 1])
''')
    ours_g = _fake_gold_rows(tmp_path, "ours.json", ours)
    d = json.loads(ours_g.read_text())
    d["summary"]["sides"]["a"].update({"max_kld_vs_gold": 0.02, "top1_agreement_pct": 96.0,
                                       "margin_qualified_top1_pct": 99.0})
    ours_g.write_text(json.dumps(d))
    other = _fake_gold_rows(tmp_path, "other.json", worse, better)
    out = tmp_path / "cmp.gguf"
    rc, _, se = derive(runner_bin, parent, out, "--quant", "q4_0", "--reference", f"hf:{hf}",
                       "--compare", f"worse file={other}:a:300000",
                       "--compare", f"better file={other}:b:400000",
                       env={"RUNNER_GOLD_LOGITS": str(script)})
    assert rc == 0, se[-1500:]
    rec = json.loads(pathlib.Path(str(out) + ".derive.json").read_text())
    by = {c["label"]: c for c in rec["comparisons"]}
    for label, rows in (("worse file", worse), ("better file", better)):
        z, pw, _ = gl.wilcoxon_signed_rank([o[0] - r[0] for o, r in zip(ours, rows)])
        assert by[label]["wilcoxon_p"] == pytest.approx(pw, rel=1e-9)
        mq = [i for i in range(n) if i % 3]
        b = sum(1 for i in mq if ours[i][1] and not rows[i][1])
        c = sum(1 for i in mq if rows[i][1] and not ours[i][1])
        assert by[label]["mcnemar_p"] == pytest.approx(gl.mcnemar_exact(b, c), rel=1e-9)
    card = pathlib.Path(str(out) + ".README.md").read_text()
    sec = card.split("## Against other files in the same window", 1)[1]
    assert sec.index("better file") < sec.index("worse file")   # the unflattering comparison first
    assert "closer to the float32 model than this file" in sec
    # a record moved off the machine that measured it: --card-from finds the
    # gold run by name beside the record, computes the same comparisons, and
    # the recipe names the reference by repository, not the local folder
    moved = tmp_path / "moved"
    moved.mkdir()
    r2 = dict(rec)
    r2.pop("comparisons")
    r2["base_model_repo"] = None
    r2["measurement"] = dict(rec["measurement"], reference_score="/nowhere/" + pathlib.Path(
        rec["measurement"]["reference_score"]).name)
    (moved / "m.derive.json").write_text(json.dumps(r2))
    shutil.copy(rec["measurement"]["reference_score"], moved)
    card2 = moved / "m.md"
    p = subprocess.run([str(runner_bin), "derive", "--card-from", str(moved / "m.derive.json"), "-o",
                        str(card2), "--base-model", "org/model@abc123",
                        "--compare", f"worse file={other}:a:300000",
                        "--compare", f"better file={other}:b:400000"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-1500:]
    t2 = card2.read_text()
    assert t2.split("## Against other files in the same window", 1)[1].split("## ")[0] == \
        sec.split("## ")[0]
    assert "--reference hf:org/model@abc123" in t2 and f"--reference hf:{hf.name} " not in t2
    # a gold run from another window is refused, not paired over the overlap
    short = _fake_gold_rows(tmp_path, "short.json", worse[: n // 2])
    p = subprocess.run([str(runner_bin), "derive", "--card-from", str(moved / "m.derive.json"), "-o",
                        str(moved / "x.md"), "--compare", f"half={short}"], capture_output=True, text=True)
    assert p.returncode != 0 and "window" in p.stderr
    # a label is Markdown as written, so a reader can follow a link to the file
    link = "[org/other-GGUF Q4_K_M (official)](https://huggingface.co/org/other-GGUF)"
    p = subprocess.run([str(runner_bin), "derive", "--card-from", str(moved / "m.derive.json"), "-o",
                        str(moved / "l.md"), "--compare", f"{link}={other}:a:300000"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-1500:]
    t3 = (moved / "l.md").read_text()
    assert f"than {link} (0.00 GB" in t3 and f"`{link}`" not in t3
