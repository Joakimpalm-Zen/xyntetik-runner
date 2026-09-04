#!/usr/bin/env python3
"""Generate the Runner pages of xyntetik.com from repository content.

Standard library only. Reads README.md sections and selected docs/ pages,
rewrites relative links to GitHub URLs (or to site pages where one exists),
and writes Jekyll sources under site/runner/. The site cannot say anything
the repository does not: every page is a copy made at build time.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
OUT = SITE / "runner"
REPO = "https://github.com/Joakimpalm-Zen/xyntetik-runner"
BLOB = REPO + "/blob/main/"
RAW = "https://raw.githubusercontent.com/Joakimpalm-Zen/xyntetik-runner/main/"

# docs pages that become site pages: docs file -> site slug
DOC_PAGES = {
    "train-lora-on-quantized-gguf.md": "train-lora-on-quantized-gguf",
    "reproducible-lora-training-receipts.md": "reproducible-lora-training-receipts",
    "truncation-safe-tool-calling.md": "truncation-safe-tool-calling",
    "benchmarks.md": "benchmarks",
}
# README sections that become site pages: heading -> (slug, title)
README_PAGES = {
    "What Runner adds": ("what-it-adds", "What Runner adds"),
    "Models and conversion": ("models", "Models and published artifacts"),
    "Support matrix": ("support-matrix", "Support matrix"),
}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def readme_sections(text: str) -> dict[str, str]:
    """Top-level '## ' sections of the README, body without the heading."""
    out: dict[str, str] = {}
    parts = re.split(r"(?m)^## ", text)
    for part in parts[1:]:
        title, _, body = part.partition("\n")
        out[title.strip()] = body
    return out


def readme_lead(text: str) -> str:
    """Everything before the first '## ' heading, minus the H1."""
    head = text.split("\n## ", 1)[0]
    return re.sub(r"(?m)^# .*\n", "", head, count=1)


def rewrite_links(md: str, *, base: str) -> str:
    """Point relative links at site pages when they exist, else at GitHub.

    base is 'docs' for a docs page and 'root' for README content.
    """
    def fix(m: re.Match[str]) -> str:
        pre, target = m.group(1), m.group(2)
        if re.match(r"^(https?:|mailto:|#)", target) and not target.startswith("#"):
            return m.group(0)
        if target.startswith("#"):
            # anchors inside README content point back at the README on GitHub
            return f"{pre}{BLOB}README.md{target})" if base == "root" else m.group(0)
        path, _, anchor = target.partition("#")
        anchor = f"#{anchor}" if anchor else ""
        if base == "docs":
            rel = path[3:] if path.startswith("../") else path
            if not path.startswith("../"):
                rel = "docs/" + path
        else:
            rel = path
        rel = rel.lstrip("./")
        name = os.path.basename(rel)
        if rel.startswith("docs/") and name in DOC_PAGES and not rel.startswith("docs/assets"):
            return f"{pre}/runner/{DOC_PAGES[name]}/{anchor})"
        if rel.startswith("docs/assets/") or re.search(r"\.(png|gif|jpg|jpeg|svg)$", rel):
            return f"{pre}{RAW}{rel})"
        return f"{pre}{BLOB}{rel}{anchor})"
    return re.sub(r"(\]\()([^)\s]+)\)", fix, md)


def front(title: str, slug: str, description: str = "") -> str:
    desc = description.replace('"', "'")
    lines = ["---", "layout: default", f'title: "{title}"',
             f"permalink: /runner/{slug}/" if slug else "permalink: /runner/"]
    if desc:
        lines.append(f'description: "{desc}"')
    lines.append("---\n")
    return "\n".join(lines)


def first_paragraph(md: str) -> str:
    for block in md.split("\n\n"):
        b = block.strip()
        if b and not b.startswith(("#", "!", "|", "```", "<", "-", "*")):
            return re.sub(r"\s+", " ", re.sub(r"[*`\[\]]", "", b))[:200]
    return ""


def write(slug: str, title: str, body: str, description: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{slug or 'index'}.md").write_text(front(title, slug, description) + body, encoding="utf-8")


def build_revision() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if not sha:
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                 text=True, check=True).stdout.strip()
        except Exception:
            sha = "unknown"
    return sha


def main() -> int:
    readme = read(ROOT / "README.md")
    sections = readme_sections(readme)
    lead = rewrite_links(readme_lead(readme), base="root")
    # Runner landing: lead + sixty seconds + why
    parts = ["# Xyntetik Runner\n", lead]
    for h in ("Sixty seconds to a served model", "Why this and not llama.cpp?"):
        parts.append(f"\n## {h}\n" + rewrite_links(sections[h], base="root"))
    parts.append("\nThe rest of the README, from the command-line reference to the compatibility "
                 f"evidence, is [on GitHub]({BLOB}README.md).\n")
    write("", "Xyntetik Runner", "".join(parts),
          "A single-binary GGUF inference and LoRA training engine in C for CPU, CUDA and Metal.")
    for heading, (slug, title) in README_PAGES.items():
        body = rewrite_links(sections[heading], base="root")
        write(slug, title, f"# {title}\n" + body +
              f"\n\nThis page is the README section of the same name, copied at build time; "
              f"the [README on GitHub]({BLOB}README.md) is the source.\n",
              first_paragraph(body))
    for fname, slug in DOC_PAGES.items():
        text = read(ROOT / "docs" / fname)
        m = re.match(r"(?m)^# (.+)$", text)
        title = m.group(1).strip() if m else slug
        body = rewrite_links(text, base="docs")
        body += (f"\n\nSource: [docs/{fname}]({BLOB}docs/{fname}) in the repository, "
                 "copied at build time.\n")
        write(slug, title, body, first_paragraph(text))
    # docs index
    rows = []
    for p in sorted((ROOT / "docs").glob("*.md")):
        t = read(p)
        m = re.match(r"(?m)^# (.+)$", t)
        title = (m.group(1).strip() if m else p.stem).replace("|", "\\|")
        target = f"/runner/{DOC_PAGES[p.name]}/" if p.name in DOC_PAGES else f"{BLOB}docs/{p.name}"
        rows.append(f"| [{p.name}]({target}) | {title} |")
    write("docs", "Runner documentation",
          "# Runner documentation\n\nEvery page under `docs/` in the repository, by file name. "
          "Pages that exist on this site link here; the rest link to GitHub.\n\n"
          "| file | title |\n|---|---|\n" + "\n".join(rows) + "\n",
          "Index of every documentation page in the Runner repository.")
    (SITE / "_includes").mkdir(exist_ok=True)
    sha = build_revision()
    (SITE / "_includes" / "build.html").write_text(
        f'Generated from <a href="{REPO}/commit/{sha}">{sha[:12]}</a>.\n', encoding="utf-8")
    # the templates this script owns carry no em dashes (public-prose rule)
    for p in (SITE / "index.md", SITE / "_layouts" / "default.html", Path(__file__)):
        if "\u2014" in read(p):
            print(f"em dash in {p}", file=sys.stderr)
            return 1
    print(f"wrote {len(list(OUT.glob('*.md')))} pages under {OUT.relative_to(ROOT)} at {sha[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
