"""The fidelity index (R7.21): every measured side of every gold-logits report, pass or fail.

scripts/fidelity-index.py reads the gold-logits.v4 reports under docs/compat-reports and
renders docs/fidelity-index.md and the site's /fidelity/ page. Nothing on them is typed
but the join from a report side to its file, and only for reports written before
gold-logits.py recorded the file itself. A side is identified by its own "file" record,
else by the join, else by a .gguf name that appears once in the roster; otherwise it is
listed as "file not identified", never dropped. The verdict is computed from the
numbers against the bar (mean KLD at most 0.05 and margin-qualified top-1 at least 97%,
docs/fidelity-method.md). Failing rows come first.
"""
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fidelity-index.py"


def _side(model, kld, mq, **extra):
    return {"endpoint": "x.score.json", "model": model, "mean_kld_vs_gold": kld,
            "max_kld_vs_gold": kld * 10, "top1_agreement_pct": mq, "margin_qualified_top1_pct": mq,
            **extra}


def _report(path, hf, sides, positions=300):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": "xyntetik.runner.gold-logits.v4",
                                "summary": {"positions": positions, "hf": hf,
                                            "corpus": "tests/fixtures/gold-corpus-2k.txt",
                                            "sides": sides}, "rows": []}), encoding="utf-8")


def _fixture(tmp_path):
    reports = tmp_path / "reports"
    own = {"repo": "pub/own-GGUF", "name": "Own-Q4_K_M.gguf", "sha256": "aa" * 32}
    _report(reports / "one" / "gold.json", "Pub/Model@1234abcd", {
        "a": _side("Own-Q4_K_M.gguf", 0.01, 99.0, file=own),     # its own record
        "b": _side("score file", 0.08, 99.0)})                   # unidentified, fails on KLD
    _report(reports / "two" / "gold.json", "Pub/Model@1234abcd", {
        "a": _side("Joined-Q8_0.gguf", 0.02, 95.0),              # the join; fails on top-1
        "b": _side("Roster-Q6_K.gguf", 0.05, 97.0)})             # the roster; on the bar: passes
    _report(reports / "three" / "other.json", "Pub/Model@1234abcd", {"a": _side("x", 0.0, 100.0)})
    (reports / "three" / "other.json").write_text(json.dumps({"schema_version": "something.else"}))
    join = tmp_path / "join.json"
    join.write_text(json.dumps({"schema_version": "xyntetik.runner.fidelity-join.v1", "sides": {
        "two/gold.json#a": {"repo": "third/joined-GGUF", "name": "Joined-Q8_0.gguf"}}}))
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({"rows": [{"id": "r", "files": [
        {"name": "Roster-Q6_K.gguf", "repo": "rost/er-GGUF", "sha256": "bb" * 32}]}]}))
    return reports, join, roster


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)


def _render(tmp_path):
    reports, join, roster = _fixture(tmp_path)
    out = tmp_path / "out"
    r = _run("--write", "--reports", reports, "--join", join, "--roster", roster, "--out", out)
    assert r.returncode == 0, r.stderr
    return out, (reports, join, roster)


def test_every_side_is_a_row_failures_first_and_the_unidentified_counted(tmp_path):
    out, _ = _render(tmp_path)
    md = (out / "docs" / "fidelity-index.md").read_text()
    assert "4 measured sides in 2 reports: 2 pass, 2 fail; the file of 1 is not identified" in md, md[:600]
    rows = [l for l in md.splitlines() if l.startswith("| ") and ("pass" in l or "fail" in l)]
    assert len(rows) == 4, rows
    verdicts = [("fail" if "| fail |" in l else "pass") for l in rows]
    assert verdicts == ["fail", "fail", "pass", "pass"], rows
    assert any("file not identified" in l and "score file" in l for l in rows), rows
    assert any("third/joined-GGUF" in l and "Joined-Q8_0.gguf" in l for l in rows), rows
    assert any("rost/er-GGUF" in l and "Roster-Q6_K.gguf" in l and "| pass |" in l for l in rows), rows
    assert any("pub/own-GGUF" in l and "| pass |" in l for l in rows), rows
    assert all("Pub/Model@1234abcd" in l for l in rows), rows


def test_the_site_page_carries_the_same_rows(tmp_path):
    out, _ = _render(tmp_path)
    page = (out / "site" / "pages" / "fidelity.html").read_text()
    assert page.startswith("<!--meta") and "path: /fidelity/" in page
    assert page.count("<tr class=\"fail\">") == 2 and page.count("<tr class=\"pass\">") == 2
    assert "file not identified" in page


def test_check_catches_a_page_edited_away_from_its_render(tmp_path):
    out, (reports, join, roster) = _render(tmp_path)
    md = out / "docs" / "fidelity-index.md"
    md.write_text(md.read_text().replace("| fail |", "| pass |", 1))
    r = _run("--check", "--reports", reports, "--join", join, "--roster", roster, "--out", out)
    assert r.returncode == 1 and "fidelity-index.md" in r.stderr


def test_a_join_entry_for_a_side_that_does_not_exist_is_refused(tmp_path):
    reports, join, roster = _fixture(tmp_path)
    j = json.loads(join.read_text())
    j["sides"]["gone/gold.json#a"] = {"repo": "x/y", "name": "Y.gguf"}
    join.write_text(json.dumps(j))
    r = _run("--write", "--reports", reports, "--join", join, "--roster", roster, "--out", tmp_path / "o")
    assert r.returncode != 0 and "gone/gold.json#a" in r.stderr


def test_the_committed_index_is_its_render():
    r = _run("--check")
    assert r.returncode == 0, r.stderr


def test_gold_logits_records_the_file_it_measured(tmp_path):
    spec = importlib.util.spec_from_file_location("gold_logits", ROOT / "scripts" / "gold-logits.py")
    gl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gl)
    f = tmp_path / "Some-Q4_K_M.gguf"
    f.write_bytes(b"GGUF" + bytes(range(256)) * 9)
    rec = gl.side_file(str(f), "owner/Some-GGUF")
    assert rec == {"repo": "owner/Some-GGUF", "name": "Some-Q4_K_M.gguf",
                   "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
