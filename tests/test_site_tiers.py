"""xyntetik.com renders the support roster at build time: the Runner page's
"Actively validated models" table and the landing page's model cards are
built from docs/model-tiers.json, never copied into the page sources.

The anchors are marks written out by hand for fixture cells whose version
and date are known, and the committed roster's own README render: the site
table holds the same rows in the same order as the README table.
"""

import html as htmllib
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROSTER = ROOT / "docs" / "model-tiers.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mt = load("model_tiers", ROOT / "scripts" / "model-tiers.py")
site = load("site_build", ROOT / "site" / "build.py")

KVIST_GGUF = "Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF"


def gate(day, version):
    return {"kind": "gate", "runner_version": version, "date": day,
            "device_class": "linux-x86_64-cuda", "result": "pass",
            "commit": "b" * 40, "caps": {"version": version},
            "output_sha256": "c" * 64, "ran": "tier-smoke",
            "evidence": "docs/model-evidence.md"}


def fixture(day="2026-10-23", version="1.5.1"):
    """The committed roster with one known cell: Kvist-14B Q8_0 on CUDA."""
    doc = json.loads(ROSTER.read_text(encoding="utf-8"))
    for r in doc["rows"]:
        if r["id"] == "xyntetik-kvist-14b-q8_0":
            r["backends"]["cuda"] = gate(day, version)
    return doc


def text_of(html):
    """The text a reader sees: inline tags vanish, cell and row tags part words."""
    html = re.sub(r"</?(?:code|strong|a|span)\b[^>]*>", "", html)
    return htmllib.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip())


def test_the_table_has_the_readmes_rows_in_the_readmes_order():
    doc = fixture()
    html = site.tiers_table_html(doc)
    # every README sub-row, cell for cell, as the text of a site row
    readme = [l for l in mt.render_readme(doc).splitlines() if l.startswith("| ")][1:]
    site_rows = [text_of(r) for r in re.findall(r"<tr[^>]*>.*?</tr>", html, re.S)][1:]
    assert len(site_rows) == len(readme)
    for md, got in zip(readme, site_rows):
        cells = [c.strip() for c in md.strip("|").split("|")]
        want = " ".join(re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", c).replace("`", "")
                        .replace("**", "").replace("\\|", "|") for c in cells if c)
        assert got == want


def test_a_recorded_cell_shows_its_version_and_date_and_links_its_row():
    html = site.tiers_table_html(fixture())
    row = [r for r in re.findall(r"<tr>.*?</tr>", html, re.S)
           if "Xyntetik-Kvist-14B-Q8_0.gguf" in r][0]
    cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
    assert [text_of(c) for c in cells[2:6]] == [
        "☐ not run", "✅ v1.5.1 2026-10-23", "☐ not run", "☐ not run"]
    assert ('href="https://github.com/Joakimpalm-Zen/xyntetik-runner/blob/main/'
            'docs/model-support.md#xyntetik-kvist-14b-q8_0"') in cells[6]


def test_a_declared_limit_links_the_manuals_support_matrix():
    html = site.tiers_table_html(fixture())
    row = [r for r in re.findall(r"<tr>.*?</tr>", html, re.S)
           if "Trinity-Nano-Preview-Q8_0.gguf" in r][0]
    assert ('<a href="https://github.com/Joakimpalm-Zen/xyntetik-runner/blob/main/'
            'MANUAL.md#support-matrix">n/a</a>') in row


def test_upcoming_models_follow_the_table_as_names_in_roster_order():
    doc = fixture()
    html = site.tiers_table_html(doc)
    names = ", ".join(mt.upcoming_names(doc))
    assert names.startswith("Qwen3.8-35B-A3B, Ornith-1.5-35B-A3B (text path only)")
    assert f"not counted as validated: {names}." in text_of(html)


def test_a_card_summary_names_the_marks_of_its_repos_files():
    doc = fixture()
    one = site.validated_html(doc, ["Joakimpalm-Zen/gemma-4-E2B-it-Q4_0-GGUF"])
    assert text_of(one) == ("Last validated: CPU ✅ v1.5.1 2026-10-23 · CUDA ☐ not run"
                            " · Metal ✅ v1.5.1 2026-10-23 · Train ✅ v1.5.1 2026-10-23")
    # files whose marks differ are named; files whose marks agree share a line
    kvist = text_of(site.validated_html(doc, [KVIST_GGUF]))
    assert kvist == (
        "Last validated: Xyntetik-Kvist-14B-BF16.gguf, Xyntetik-Kvist-14B-Q5_0-mix.gguf, "
        "Xyntetik-Kvist-14B-IQ4_NL-mix.gguf: CPU ☐ not run · CUDA ☐ not run · "
        "Metal ☐ not run · Train ☐ not run; Xyntetik-Kvist-14B-Q8_0.gguf: CPU ☐ not run"
        " · CUDA ✅ v1.5.1 2026-10-23 · Metal ☐ not run · Train ☐ not run")
    # a card is a link, so its summary carries none of its own
    assert "<a " not in site.validated_html(doc, ["Joakimpalm-Zen/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered-GGUF"])


def test_every_file_of_a_card_with_one_set_of_marks_is_counted():
    doc = json.loads(ROSTER.read_text(encoding="utf-8"))
    kvist = text_of(site.validated_html(doc, [KVIST_GGUF]))
    assert kvist == ("Last validated, all 4 files: CPU ☐ not run · CUDA ☐ not run · "
                     "Metal ☐ not run · Train ☐ not run")


def test_a_card_that_covers_no_roster_file_has_no_line(tmp_path):
    """The GENESIS card's repository is exempt (no GGUF, not a Runner
    model): the site prints no line for it, and never an exempt reason."""
    doc = json.loads(ROSTER.read_text(encoding="utf-8"))
    genesis = "Joakimpalm-Zen/Xyntetik-Blad-Genesis-0.7M"
    assert genesis in {e["repo"] for e in doc["exempt"]}
    try:
        site.validated_html(doc, [genesis])
        raise AssertionError("an exempt repository rendered a line")
    except KeyError:
        pass
    _, index = build(tmp_path, doc)
    card = index.split("GENESIS: learning with no human text", 1)[1].split("</a>", 1)[0]
    assert "Last validated" not in card and "support roster" not in card


def build(tmp_path, doc):
    tmp_path.mkdir(parents=True, exist_ok=True)
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "site"
    r = subprocess.run([sys.executable, str(ROOT / "site" / "build.py"),
                        "--roster", str(roster), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return ((out / "runner" / "index.html").read_text(encoding="utf-8"),
            (out / "index.html").read_text(encoding="utf-8"))


def test_the_built_pages_carry_the_render_of_the_roster_they_were_built_from(tmp_path):
    doc = fixture()
    runner, index = build(tmp_path / "a", doc)
    assert site.tiers_table_html(doc) in runner
    assert site.validated_html(doc, [KVIST_GGUF]) in index
    # a roster change changes the built pages: no stale copy anywhere
    later = fixture(day="2026-11-02", version="1.5.2")
    runner2, index2 = build(tmp_path / "b", later)

    def kvist_q8(page):
        return [r for r in re.findall(r"<tr>.*?</tr>", page, re.S)
                if "Xyntetik-Kvist-14B-Q8_0.gguf" in r][0]
    assert "✅ v1.5.1 2026-10-23" in kvist_q8(runner)
    assert "✅ v1.5.2 2026-11-02" in kvist_q8(runner2)
    assert "v1.5.1" not in kvist_q8(runner2)
    assert "✅ v1.5.2 2026-11-02" in index2 and "v1.5.2" not in index


def test_a_card_naming_a_repo_the_roster_does_not_know_fails_the_build(tmp_path):
    doc = fixture()
    doc["rows"] = [r for r in doc["rows"]
                   if not any(f.get("repo") == KVIST_GGUF for f in r.get("files") or [])]
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps(doc), encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "site" / "build.py"),
                        "--roster", str(roster), "--out", str(tmp_path / "site")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and KVIST_GGUF in r.stderr


def test_no_page_source_types_a_dated_mark():
    """The marks live in the roster; a page source that types one is a copy
    that goes stale on the next gate run. (The prose may name the marks.)"""
    for page in (ROOT / "site" / "pages").glob("*.html"):
        text = page.read_text(encoding="utf-8")
        assert not re.search(r"(✅|failed) v\d+\.\d+\.\d+ \d{4}-", text), page.name
