#!/usr/bin/env python3
"""Build xyntetik.com into site/_site from site/pages and site/assets.

Standard library only. Every page under site/pages is an HTML fragment with
a leading `<!--meta ... -->` block (title, description, path, optional nav
key). The shell (head, header, footer) is applied here, chart placeholders
(`{{chart:name}}`) are replaced with inline SVG generated from the data
tables below, and the output is checked before it is written: no em dashes
(public-prose rule), no unresolved placeholders, no internal link without a
page behind it.

The numbers in CHART_DATA are copied from the repository's own documents;
each chart names its source file and measurement date in its caption. The
site may not claim more than the README and docs do on the day it is built.
"""
from __future__ import annotations

import datetime as dt
import html
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
OUT = SITE / "_site"
REPO = "https://github.com/Joakimpalm-Zen/xyntetik-runner"
HF = "https://huggingface.co/Joakimpalm-Zen"
ORIGIN = "https://xyntetik.com"

NAV = [
    ("runner", "Runner", "/runner/"),
    ("evidence", "Evidence", "/evidence/"),
    ("about", "About", "/about/"),
    ("support", "Support", "/support/"),
]

REDIRECTS = {
    # v1 site paths, kept alive
    "/runner/benchmarks/": "/evidence/#benchmarks",
    "/runner/models/": "/evidence/#artifacts",
    "/runner/support-matrix/": "/runner/#support",
    "/runner/what-it-adds/": "/runner/#contracts",
    "/runner/docs/": REPO + "/tree/main/docs",
    "/donate/": "/support/",
}

# ---------------------------------------------------------------- data ---

CHART_DATA = {
    # docs/truncation-benchmark.md, measured 2026-08-19, granite-4.1-3b
    "truncation": {
        "budgets": [1, 2, 3, 5, 8, 16, 64],
        "engines": [
            ("Runner", ["ok"] * 7),
            ("vLLM 0.27.1", ["none"] * 6 + ["ok"]),
            ("llama.cpp b10488", ["none"] * 5 + ["bad", "ok"]),
            ("Ollama 0.32.14", ["none"] * 5 + ["err", "ok"]),
            ("TensorRT-LLM 1.2.1 †", ["none"] * 6 + ["ok"]),
            ("SGLang 0.5.17 †", ["none"] * 6 + ["ok"]),
        ],
    },
    # docs/benchmarks.md, 2026-09-02, MIG 1g.24gb, runner 6d705e9 vs llama.cpp b10353
    "speed": [
        ("Llama-3.2-3B Q4_K_M", 87.9, 101.8, 525.7, 6387.6),
        ("Phi-4-mini Q8_0", 53.8, 57.7, 355.6, 5960.6),
        ("granite-3.3-8b Q4_K_M", 39.7, 45.4, 220.5, 2288.9),
        ("granite-4.1-8b Q4_0", 41.7, 46.5, 151.1, 2359.2),
        ("Phi-3.5-mini Q4_K_M", 72.0, 83.2, 284.9, 5120.3),
        ("gemma-4-12B Q4_K_M", 23.2, 28.8, 123.7, 1439.2),
        ("Qwen3-30B-A3B Q4_K_M (MoE)", 62.2, 86.6, 128.5, 2326.3),
        ("gemma-4-26B-A4B Q4_0 (MoE)", 23.3, 56.2, 34.3, 2429.7),
        ("Qwen2.5-32B Q3_K_S", 1.8, 14.3, 15.3, 469.0),
    ],
    # docs/train-lora-on-quantized-gguf.md and README merge study, 2026-08-22
    "lora": [
        ("Base Qwen3-4B Q4_K_M", 0.69, "base"),
        ("Base + adapter (--lora)", 1.00, "ok"),
        ("Merged into F16", 1.00, "ok"),
        ("Merged into Q8_0", 1.00, "ok"),
        ("Merged into Q4_0", 0.69, "erased"),
    ],
    # docs/quant-fidelity.md: granite-4.1-3b 2026-08-14, Hermes-4-14B 2026-08-15
    "fidelity": {
        "rungs": ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0"],
        "granite": [100.0, 64.3, 57.1, 57.1, 50.0],
        "hermes": [100.0, 78.6, 78.6, 64.3, 50.0],
    },
}

# ---------------------------------------------------------------- charts --

def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def chart_truncation() -> str:
    d = CHART_DATA["truncation"]
    budgets, engines = d["budgets"], d["engines"]
    cell, gap, left, top = 44, 4, 176, 34
    w = left + len(budgets) * (cell + gap) + 8
    h = top + len(engines) * (cell + gap) + 8
    labels = {"ok": "executable tool call, arguments parse",
              "none": "no usable call",
              "bad": "tool_calls present, arguments do not parse",
              "err": "HTTP 500"}
    out = [f'<svg class="chart chart-grid" viewBox="0 0 {w} {h}" role="img" '
           f'aria-labelledby="trunc-title" xmlns="http://www.w3.org/2000/svg">',
           '<title id="trunc-title">Per token budget, does the client receive an executable tool call</title>']
    for j, b in enumerate(budgets):
        x = left + j * (cell + gap) + cell / 2
        lab = f"{b}" + (" ctrl" if b == 64 else "")
        out.append(f'<text class="ax" x="{x}" y="{top - 12}" text-anchor="middle">{lab}</text>')
    for i, (name, row) in enumerate(engines):
        y = top + i * (cell + gap)
        cls = "lab strong" if name == "Runner" else "lab"
        out.append(f'<text class="{cls}" x="{left - 12}" y="{y + cell / 2 + 4}" text-anchor="end">{esc(name)}</text>')
        for j, v in enumerate(row):
            x = left + j * (cell + gap)
            out.append(f'<g class="cell cell-{v}"><title>{esc(name)}, max_tokens {budgets[j]}: {labels[v]}</title>'
                       f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="6"/>')
            cx, cy = x + cell / 2, y + cell / 2
            if v == "ok":
                out.append(f'<path class="glyph" d="M{cx - 9} {cy + 1} l6 6 l12 -13" fill="none" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>')
            elif v == "none":
                out.append(f'<path class="glyph" d="M{cx - 7} {cy - 7} l14 14 M{cx + 7} {cy - 7} l-14 14" fill="none" stroke-width="2.5" stroke-linecap="round"/>')
            elif v == "bad":
                out.append(f'<text class="glyph-text" x="{cx}" y="{cy + 5}" text-anchor="middle">{{ "</text>')
            else:
                out.append(f'<text class="glyph-text" x="{cx}" y="{cy + 5}" text-anchor="middle">500</text>')
            out.append("</g>")
    out.append("</svg>")
    return "".join(out)


def chart_speed() -> str:
    rows = CHART_DATA["speed"]
    left, top, bw, gap, right = 236, 30, 18, 12, 64
    n = len(rows)
    plot_w = 300
    h = top + n * (bw + gap) + 12
    panel_w = left + plot_w + right

    def panel(title, idx_r, idx_l, x0):
        out = [f'<text class="ptitle" x="{x0 + left}" y="16">{title}</text>']
        for t in (0, 25, 50, 75, 100):
            x = x0 + left + plot_w * t / 100
            out.append(f'<line class="grid" x1="{x}" y1="{top - 4}" x2="{x}" y2="{h - 10}"/>')
            out.append(f'<text class="ax" x="{x}" y="{h - 0}" text-anchor="middle">{t}%</text>')
        for i, r in enumerate(rows):
            name, = r[:1]
            rv, lv = r[idx_r], r[idx_l]
            pct = 100.0 * rv / lv
            y = top + i * (bw + gap)
            wpx = plot_w * pct / 100
            out.append(f'<text class="lab" x="{x0 + left - 10}" y="{y + bw - 4}" text-anchor="end">{esc(name)}</text>')
            out.append(f'<g class="bar"><title>{esc(name)}: runner {rv} vs llama.cpp {lv} tok/s ({pct:.0f}%)</title>'
                       f'<rect x="{x0 + left}" y="{y}" width="{max(wpx, 2):.1f}" height="{bw}" rx="0" ry="0"/></g>')
            out.append(f'<text class="val" x="{x0 + left + wpx + 8:.1f}" y="{y + bw - 4}">{pct:.0f}%</text>')
        return "".join(out)

    w = panel_w * 2 + 24
    svg = [f'<svg class="chart chart-bars chart-speed" viewBox="0 0 {w} {h + 6}" role="img" aria-labelledby="speed-title" '
           'xmlns="http://www.w3.org/2000/svg">',
           '<title id="speed-title">Runner throughput as a percentage of llama.cpp, decode and prefill</title>',
           panel("Decode tokens/s, runner as % of llama.cpp", 1, 2, 0),
           panel("Prefill tokens/s, runner as % of llama.cpp", 3, 4, panel_w + 24),
           "</svg>"]
    return "".join(svg)


def chart_lora() -> str:
    rows = CHART_DATA["lora"]
    left, top, bw, gap, plot_w, right = 200, 22, 20, 14, 360, 70
    h = top + len(rows) * (bw + gap) + 24
    w = left + plot_w + right
    out = [f'<svg class="chart chart-bars" viewBox="0 0 {w} {h}" role="img" aria-labelledby="lora-title" '
           'xmlns="http://www.w3.org/2000/svg">',
           '<title id="lora-title">Held-out exact tool call score before and after training, and after merging</title>']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + plot_w * t
        out.append(f'<line class="grid" x1="{x}" y1="{top - 6}" x2="{x}" y2="{h - 18}"/>')
        out.append(f'<text class="ax" x="{x}" y="{h - 4}" text-anchor="middle">{t:.2f}</text>')
    for i, (name, v, kind) in enumerate(rows):
        y = top + i * (bw + gap)
        wpx = plot_w * v
        out.append(f'<text class="lab" x="{left - 10}" y="{y + bw - 5}" text-anchor="end">{esc(name)}</text>')
        out.append(f'<g class="bar bar-{kind}"><title>{esc(name)}: exact-call score {v:.2f}</title>'
                   f'<rect x="{left}" y="{y}" width="{wpx:.1f}" height="{bw}"/></g>')
        tag = {"ok": "", "base": "", "erased": "  erased"}[kind]
        out.append(f'<text class="val" x="{left + wpx + 8:.1f}" y="{y + bw - 5}">{v:.2f}{tag}</text>')
    out.append("</svg>")
    return "".join(out)


def chart_fidelity() -> str:
    d = CHART_DATA["fidelity"]
    rungs = d["rungs"]
    left, top, plot_w, plot_h, right, bottom = 56, 26, 440, 200, 160, 34
    w, h = left + plot_w + right, top + plot_h + bottom
    xs = [left + plot_w * i / (len(rungs) - 1) for i in range(len(rungs))]

    def y(v):
        return top + plot_h * (100 - v) / 60  # axis 40..100

    out = [f'<svg class="chart chart-lines" viewBox="0 0 {w} {h}" role="img" aria-labelledby="fid-title" '
           'xmlns="http://www.w3.org/2000/svg">',
           '<title id="fid-title">Tool-call fidelity down the quantization ladder: shape holds at 100 percent, argument agreement decays</title>']
    for t in (40, 60, 80, 100):
        out.append(f'<line class="grid" x1="{left}" y1="{y(t)}" x2="{left + plot_w}" y2="{y(t)}"/>')
        out.append(f'<text class="ax" x="{left - 8}" y="{y(t) + 4}" text-anchor="end">{t}%</text>')
    for x, r in zip(xs, rungs):
        out.append(f'<text class="ax" x="{x}" y="{h - 12}" text-anchor="middle">{r}</text>')
    # shared flat line: schema conformance and tool selection, both families
    out.append(f'<line class="ref" x1="{left}" y1="{y(100)}" x2="{left + plot_w}" y2="{y(100)}"/>')
    out.append(f'<text class="lab" x="{left + 8}" y="{y(100) - 9}">schema conformance and tool selection: 100% on every rung, both families</text>')
    series = [("s1", "granite", "granite-4.1-3b", 18), ("s2", "hermes", "Hermes-4-14B", -8)]
    for cls, key, label, dy in series:
        pts = list(zip(xs, [y(v) for v in d[key]]))
        path = "M" + " L".join(f"{px:.1f} {py:.1f}" for px, py in pts)
        out.append(f'<path class="line {cls}" d="{path}" fill="none" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for (px, py), v, r in zip(pts, d[key], rungs):
            out.append(f'<circle class="dot {cls}" cx="{px:.1f}" cy="{py:.1f}" r="4.5"><title>{label} {r}: argument agreement {v}%</title></circle>')
        ex, ey = pts[-1]
        out.append(f'<text class="lab" x="{ex + 10}" y="{ey + dy:.1f}">{label}</text>')
        out.append(f'<text class="val" x="{ex + 10}" y="{ey + dy + 13:.1f}">{d[key][-1]:.0f}% at Q4_0</text>')
    out.append("</svg>")
    return "".join(out)


CHARTS = {
    "truncation": chart_truncation,
    "speed": chart_speed,
    "lora": chart_lora,
    "fidelity": chart_fidelity,
}

# ----------------------------------------------------------------- shell --

MARKS = SITE / "assets" / "marks"


def mark(name: str, alt: str = "", size: int = 34) -> str:
    """The light and dark variants of one mark; CSS shows the one that matches."""
    return (f'<img class="mark-light" src="/assets/marks/{name}-light.svg" alt="{esc(alt)}" width="{size}" height="{size}">'
            f'<img class="mark-dark" src="/assets/marks/{name}-dark.svg" alt="" width="{size}" height="{size}" aria-hidden="true">')


def inline_mark(name: str) -> str:
    """Both variants inlined once (the hero), with classes the draw-on animation hooks."""
    out = []
    for variant in ("light", "dark"):
        svg = (MARKS / f"{name}-{variant}.svg").read_text(encoding="utf-8").strip()
        svg = re.sub(r'\s(width|height)="512"', "", svg, count=2)
        # both variants are inlined on one page and the hidden one would win
        # the id lookup, so every id (gradients, filters) is suffixed
        svg = re.sub(r'id="([a-z0-9]+)"', lambda m: f'id="{m.group(1)}-{variant}"', svg)
        svg = re.sub(r'url\(#([a-z0-9]+)\)', lambda m: f'url(#{m.group(1)}-{variant})', svg)
        # the main ring is the stroke-width 9 path; the glow (16, blurred) stays as is
        svg = svg.replace('stroke-width="9" stroke-linecap="round"/>', 'stroke-width="9" stroke-linecap="round" class="ring"/>', 1)
        svg = re.sub(r"<circle ", '<circle class="dot" ', svg)
        svg = svg.replace("<svg ", f'<svg class="hero-mark mark-{variant}" aria-hidden="true" ', 1)
        out.append(svg)
    return "".join(out)


LOGO = mark("xyntetik", "Xyntetik")


def meta_block(text: str) -> tuple[dict, str]:
    m = re.match(r"\s*<!--meta\n(.*?)-->\n?", text, re.S)
    if not m:
        raise SystemExit("page without meta block")
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, text[m.end():]


def shell(meta: dict, body: str, sha: str) -> str:
    title = meta["title"]
    full = title if meta.get("bare") else f"{title} · Xyntetik"
    desc = meta.get("description", "")
    path = meta["path"]
    active = meta.get("nav", "")
    nav = "".join(
        f'<a href="{href}"{" aria-current=\"page\"" if key == active else ""}>{label}</a>'
        for key, label, href in NAV)
    og_image = ORIGIN + "/assets/og.png"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(full)}</title>
<meta name="description" content="{esc(desc)}">
<link rel="canonical" href="{ORIGIN}{path}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Xyntetik">
<meta property="og:title" content="{esc(full)}">
<meta property="og:description" content="{esc(desc)}">
<meta property="og:url" content="{ORIGIN}{path}">
<meta property="og:image" content="{og_image}">
<meta name="twitter:card" content="summary_large_image">
<meta name="theme-color" content="#FAFAF8" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0E1516" media="(prefers-color-scheme: dark)">
<link rel="icon" href="/assets/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<link rel="stylesheet" href="/assets/site.css?v={sha[:8]}">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="top">
  <div class="wrap top-inner">
    <a class="brand" href="/" aria-label="Xyntetik home">
      <span class="brand-mark">{LOGO}</span>
      <span class="brand-word"><span class="x">x</span>yntetik</span>
    </a>
    <button class="nav-toggle" aria-expanded="false" aria-controls="nav" aria-label="Menu">
      <span></span><span></span>
    </button>
    <nav id="nav" class="nav">
      {nav}
      <span class="nav-sep" aria-hidden="true"></span>
      <a class="ext" href="{REPO}" rel="noopener">GitHub</a>
      <a class="ext" href="{HF}" rel="noopener">Hugging Face</a>
    </nav>
  </div>
</header>
<main id="main">
{body}
</main>
<footer class="foot">
  <div class="wrap foot-grid">
    <div class="foot-brand">
      <span class="brand-mark small">{LOGO}</span>
      <p><strong>Xyntetik</strong> is a Zenova AB program. Runner is free forever under Apache 2.0. Built in Sweden, runs on hardware you own.</p>
      <p class="muted">Every number on this site names the document it comes from and the date it was measured. Site source: <a href="{REPO}/tree/main/site">site/</a> in the runner repository, built from <a href="{REPO}/commit/{sha}">{sha[:10]}</a>.</p>
    </div>
    <div>
      <h4>Runner</h4>
      <a href="/runner/">What it is</a>
      <a href="/runner/#sixty-seconds">Sixty seconds to a served model</a>
      <a href="/runner/#compare">Where it wins, where it does not</a>
      <a href="{REPO}/releases">Releases</a>
      <a href="{REPO}/blob/main/README.md">README and CLI reference</a>
    </div>
    <div>
      <h4>Evidence</h4>
      <a href="/evidence/">All evidence</a>
      <a href="/runner/reproducible-lora-training-receipts/">Receipts and reproducibility</a>
      <a href="/runner/train-lora-on-quantized-gguf/">LoRA on quantized GGUF</a>
      <a href="/runner/truncation-safe-tool-calling/">Truncation-safe tool calls</a>
      <a href="{HF}">Artifacts on Hugging Face</a>
    </div>
    <div>
      <h4>Xyntetik</h4>
      <a href="/about/">About and Zenova AB</a>
      <a href="/suite/">Suite</a>
      <a href="/shade/">Shade</a>
      <a href="/support/">Support the work</a>
      <a href="mailto:hello@xyntetik.com">hello@xyntetik.com</a>
    </div>
  </div>
  <div class="wrap foot-legal">
    <span>&copy; {dt.date.today().year} Zenova AB. Runner source code Apache-2.0.</span>
    <span>Security reports: <a href="{REPO}/blob/main/SECURITY.md">SECURITY.md</a></span>
  </div>
</footer>
<script src="/assets/site.js?v={sha[:8]}" defer></script>
</body>
</html>
"""


def redirect_page(target: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={target}">
<link rel="canonical" href="{target if target.startswith('http') else ORIGIN + target}">
<title>Redirecting</title></head>
<body><p>This page moved to <a href="{target}">{target}</a>.</p></body></html>
"""


def latest_release() -> tuple[str, str]:
    """Version and date of the newest released entry in CHANGELOG.md."""
    for line in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"## (v\S+)\s*[-\u2014]\s*(\d{4}-\d{2}-\d{2})", line)
        if m:
            return m.group(1), m.group(2)
    raise SystemExit("no released version found in CHANGELOG.md")


def build_revision() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if not sha:
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                 text=True, check=True).stdout.strip()
        except Exception:
            sha = "0000000000000000000000000000000000000000"
    return sha


# ---------------------------------------------------------------- checks --

PLACEHOLDER_WORDS = ("lorem ipsum", "TODO", "TBD", "coming soon", "placeholder", "{{")


def check_page(path: str, out: str, known_paths: set[str]) -> list[str]:
    problems = []
    if "\u2014" in out:
        problems.append("em dash")
    for w in PLACEHOLDER_WORDS:
        if w in out:
            problems.append(f"placeholder token {w!r}")
    for href in re.findall(r'href="([^"]+)"', out):
        if href.startswith(("http", "mailto:", "#")):
            continue
        base = href.split("#")[0].split("?")[0]
        if base.startswith("/assets/"):
            if not (SITE / "assets" / base[len("/assets/"):]).exists():
                problems.append(f"missing asset {href}")
            continue
        if base not in known_paths:
            problems.append(f"internal link without a page: {href}")
    return [f"{path}: {p}" for p in problems]


# ----------------------------------------------------------------- main ---

def main() -> int:
    sha = build_revision()
    version, release_date = latest_release()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(SITE / "assets", OUT / "assets")

    pages = []
    for p in sorted((SITE / "pages").glob("*.html")):
        meta, body = meta_block(p.read_text(encoding="utf-8"))
        pages.append((meta, body))
    known = {m["path"] for m, _ in pages} | set(REDIRECTS)
    known.add("/404.html")

    problems: list[str] = []
    for meta, body in pages:
        def sub(m: re.Match[str]) -> str:
            kind, name = m.group(1), m.group(2)
            if kind == "chart":
                return CHARTS[name]()
            raise KeyError(kind)
        body = re.sub(r"\{\{(chart):([a-z_]+)\}\}", sub, body)
        body = re.sub(r"\{\{mark:([a-z]+):(\d+)\}\}", lambda m: mark(m.group(1), "", int(m.group(2))), body)
        body = re.sub(r"\{\{heromark:([a-z]+)\}\}", lambda m: inline_mark(m.group(1)), body)
        body = (body.replace("{{repo}}", REPO).replace("{{hf}}", HF)
                .replace("{{version}}", version).replace("{{release_date}}", release_date))
        out = shell(meta, body, sha)
        problems += check_page(meta["path"], out, known)
        target = OUT / meta["path"].lstrip("/") / "index.html"
        if meta["path"] == "/404.html":
            target = OUT / "404.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(out, encoding="utf-8")
    for src, dst in REDIRECTS.items():
        target = OUT / src.lstrip("/") / "index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(redirect_page(dst), encoding="utf-8")

    urls = sorted(m["path"] for m, _ in pages if m["path"] != "/404.html")
    today = dt.date.today().isoformat()
    (OUT / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(f"  <url><loc>{ORIGIN}{u}</loc><lastmod>{today}</lastmod></url>\n" for u in urls)
        + "</urlset>\n", encoding="utf-8")
    (OUT / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {ORIGIN}/sitemap.xml\n", encoding="utf-8")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")

    for p in (SITE / "assets" / "site.css", SITE / "assets" / "site.js", Path(__file__)):
        if "\u2014" in p.read_text(encoding="utf-8"):
            problems.append(f"{p.name}: em dash")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"wrote {len(pages)} pages and {len(REDIRECTS)} redirects to {OUT.relative_to(ROOT)} at {sha[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
