"""xyntetik.com renders the support roster at build time: the Runner page's
"Actively validated models" table and the landing page's model cards are
built from docs/model-tiers.json, never copied into the page sources.

The anchors are marks written out by hand for fixture cells whose version
and date are known, and the committed roster's own README render: the site
table holds the same rows in the same order as the README's family blocks.
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


def lora(day, version):
    return {"training": {"supported": True, "reason": None},
            "adapter_serving": {"supported": True, "reason": None},
            "runner_version": version, "date": day}


E2B_ROW = "gemma-4-e2b-it-q4_k_m-q4_0-mix"


def cleared():
    """The committed roster's shape (rows, files, order, declared limits) with
    every recorded gate cell and Train answer removed. The tests below pin
    marks they set themselves, so a record landing on main changes nothing
    here; before this, every --record moved the expected text."""
    doc = json.loads(ROSTER.read_text(encoding="utf-8"))
    for r in doc["rows"]:
        for b, cell in list((r.get("backends") or {}).items()):
            if cell and cell.get("kind") == "gate":
                r["backends"][b] = None
        if "lora" in r:
            r["lora"] = None
    return doc


def fixture(day="2026-10-23", version="1.5.1"):
    """The cleared roster with three known cells: Kvist-14B Q8_0 on CUDA,
    and the E2B mix on CPU and Metal with a Train answer."""
    doc = cleared()
    for r in doc["rows"]:
        if r["id"] == "xyntetik-kvist-14b-q8_0":
            r["backends"]["cuda"] = gate(day, version)
        if r["id"] == E2B_ROW:
            r["backends"]["cpu"] = gate(day, version)
            r["backends"]["metal"] = gate(day, version)
            r["lora"] = lora(day, version)
    return doc


def text_of(html):
    """The text a reader sees: inline tags vanish, cell and row tags part words."""
    html = re.sub(r"</?(?:code|strong|a|span)\b[^>]*>", "", html)
    return htmllib.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip())


def readme_models(doc):
    """(model, file rows) per model of the README's blocks, family by
    family and model by model within each, a file row as the contents of
    its cells."""
    out = []
    for block in re.findall(r"<details>\n(.*?)\n</details>", mt.render_readme(doc), re.S):
        for line, table in re.findall(r"<p>(.*?)</p>\n(<table>.*?</table>)", block, re.S):
            model = htmllib.unescape(re.match(r"<b>(.*?)</b>", line).group(1))
            rows = [re.findall(r"<td[^>]*>(.*?)</td>", tr)
                    for tr in re.findall(r"<tr>(.*?)</tr>", table) if "<td" in tr]
            out.append((model, rows))
    return out


def family_marks(doc):
    """Family name to its summary line's marks (label to mark), in the
    README's order."""
    out = {}
    for s in re.findall(r"<summary>(.*?)</summary>", mt.render_readme(doc)):
        m = re.match(r"<b>(.*?)</b>, \d+ models?: (.*)$", s)
        out[htmllib.unescape(m.group(1))] = dict(x.split(" ", 1) for x in m.group(2).split(" · "))
    return out


def site_words(cell):
    """A README cell as the site's table words it. The README writes the
    same marks compactly: not run as a blank cell, the version without its
    "v" and over its date. These forms are written out by hand."""
    cell = re.sub(r"^(✅|failed) (\S+)<br>(\S+)$", r"\1 v\2 \3", cell)
    return text_of(cell) if cell else "☐ not run"


def test_the_runner_page_links_to_the_readmes_list_and_the_roster():
    """Since 2026-10-27 the site carries no copy of the marks: the Runner
    page links to the README's model list and to docs/model-support.md."""
    text = (ROOT / "site" / "pages" / "runner.html").read_text(encoding="utf-8")
    assert "{{tiers:table}}" not in text
    assert "{{repo}}#validated-models" in text
    assert "docs/model-support.md" in text


def test_no_page_source_carries_a_validated_placeholder():
    for page in (ROOT / "site" / "pages").glob("*.html"):
        text = page.read_text(encoding="utf-8")
        assert "{{validated:" not in text, page.name


def test_no_page_source_types_a_dated_mark():
    """The marks live in the roster; a page source that types one is a copy
    that goes stale on the next gate run. (The prose may name the marks.)"""
    for page in (ROOT / "site" / "pages").glob("*.html"):
        text = page.read_text(encoding="utf-8")
        assert not re.search(r"(✅|failed) v\d+\.\d+\.\d+ \d{4}-", text), page.name
