#!/usr/bin/env python3
"""The two `runner derive` graphics for the README and the site, light and dark.

    scripts/readme-svg-derive.py process
    scripts/readme-svg-derive.py result RECORD.json

`process` draws the usual way (several tools by hand) against derive (one
command: the file, its measurement against the publisher's float32 model, its
card). `result` draws size against distance from float32 for the files in
RECORD.json, the bar drawn in, every number read from the record:

    {"title": "...", "window": "...", "bar_mean_kld": 0.05,
     "files": [{"label": "...", "bytes": N, "mean_kld": x, "mq_top1_pct": y,
                "ours": true|false}, ...]}

Each writes site/assets/readme-derive-<name>-{light,dark}.svg, in the palette
and type of the other README graphics (readme-train, readme-idle).
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
PALETTE = {
    "light": {"bg": "#F6F8FA", "line": "#D0D7DE", "box": "#FFFFFF", "muted": "#5F6C72",
              "ink": "#16232A", "accent": "#0E7C6E", "warn": "#B35C00"},
    "dark": {"bg": "#161B22", "line": "#30363D", "box": "#21262D", "muted": "#8C9A98",
             "ink": "#E8ECEA", "accent": "#3ED2BD", "warn": "#E5A33B"},
}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, w, h, title, theme):
        self.c = PALETTE[theme]
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
                      f'height="{h}" role="img" aria-labelledby="t">',
                      f'<title id="t">{esc(title)}</title>',
                      f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="9" '
                      f'fill="{self.c["bg"]}" stroke="{self.c["line"]}"/>']

    def text(self, x, y, s, size=11.5, color="ink", anchor="start", weight=None):
        w = f' font-weight="{weight}"' if weight else ""
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" '
                          f'font-size="{size}"{w} fill="{self.c[color]}">{esc(s)}</text>')

    def box(self, x, y, w, h, lines, stroke="line", sw=1, colors=None):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
                          f'fill="{self.c["box"]}" stroke="{self.c[stroke]}" stroke-width="{sw}"/>')
        top = y + h / 2 - 7 * (len(lines) - 1) + 4
        for i, s in enumerate(lines):
            self.text(x + w / 2, top + 14 * i, s, anchor="middle",
                      color=(colors[i] if colors else "ink"))

    def arrow(self, x1, x2, y, label=None, color="muted"):
        self.parts.append(f'<line x1="{x1}" y1="{y}" x2="{x2 - 6}" y2="{y}" '
                          f'stroke="{self.c[color]}" stroke-width="1.2"/>')
        self.parts.append(f'<path d="M{x2 - 6} {y - 4} L{x2} {y} L{x2 - 6} {y + 4} Z" fill="{self.c[color]}"/>')
        if label:
            for i, s in enumerate(label if isinstance(label, list) else [label]):
                self.text((x1 + x2) / 2, y - 40 + 12 * i, s, size=10.5, color="muted", anchor="middle")

    def raw(self, s):
        self.parts.append(s.format(**self.c))

    def write(self, path):
        path.write_text("\n".join(self.parts + ["</svg>"]) + "\n", encoding="utf-8")


PROCESS_TITLE = ("The usual way makes a smaller model file with several tools by hand and rarely measures it; "
                 "runner derive is one command that makes the file, measures it against the publisher's "
                 "float32 model and writes its card")


def process(theme):
    s = Svg(640, 300, PROCESS_TITLE, theme)
    s.text(18, 28, "The usual way", size=12.5, color="muted", weight="600")
    s.text(122, 28, "several tools, by hand", size=10.5, color="muted")
    s.box(18, 70, 108, 44, ["parent file"])
    s.arrow(126, 176, 92, ["the same mix for", "every model"])
    s.box(176, 70, 108, 44, ["smaller file"])
    s.arrow(284, 334, 92, ["score it elsewhere,", "if at all"])
    s.box(334, 70, 118, 44, ["a number,", "against what?"])
    s.arrow(452, 502, 92, ["write it", "by hand"])
    s.box(502, 70, 120, 44, ["a model card"])
    s.text(622, 128, "nothing ties the card's claims to the file", size=10.5, color="warn", anchor="end")
    s.raw('<line x1="18" y1="142" x2="622" y2="142" stroke="{line}" stroke-dasharray="3 4"/>')
    s.text(18, 166, "runner derive", size=12.5, color="accent", weight="600")
    s.text(118, 166, "one command", size=10.5, color="muted")
    s.box(18, 196, 124, 62, ["parent file", "+ a recipe or", "a target size"], stroke="accent", sw=1.5)
    s.arrow(142, 178, 227, None, color="accent")
    s.box(178, 196, 144, 62, ["the file,", "precision placed by", "measured sensitivity"], stroke="accent",
          sw=1.5)
    s.box(328, 196, 144, 62, ["its measurement", "against the publisher's", "float32 model"], stroke="accent",
          sw=1.5, colors=["ink", "ink", "accent"])
    s.box(478, 196, 144, 62, ["its card and record:", "sha256, recipe, window", "and the numbers"],
          stroke="accent", sw=1.5, colors=["ink", "ink", "accent"])
    s.text(320, 284, "a file that fails the bar says so in the first line of its own card", size=10.5,
           color="muted", anchor="middle")
    return s


def result(rec, theme):
    files = rec["files"]
    bar = rec.get("bar_mean_kld", 0.05)
    w, h = 640, 300
    x0, x1, y0, y1 = 70, 600, 250, 60   # plot area: x is size, y is mean KLD (lower is better, at the bottom)
    sizes = [f["bytes"] / 1e9 for f in files]
    klds = [f["mean_kld"] for f in files]
    smin, smax = min(sizes) - 0.5, max(sizes) + 0.5
    kmax = max(max(klds), bar) * 1.25

    def X(gb): return x0 + (gb - smin) / (smax - smin) * (x1 - x0)
    def Y(k): return y0 - k / kmax * (y0 - y1)
    best = min(files, key=lambda f: f["mean_kld"])
    title = rec["title"]
    s = Svg(w, h, title, theme)
    s.text(18, 28, rec["heading"], size=12.5, color="muted", weight="600")
    s.text(18, 44, rec["window"], size=10.5, color="muted")
    s.raw(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{{line}}"/>')
    s.raw(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{{line}}"/>')
    yb = Y(bar)
    s.raw(f'<line x1="{x0}" y1="{yb:.1f}" x2="{x1}" y2="{yb:.1f}" stroke="{{warn}}" stroke-dasharray="4 4"/>')
    s.text(x1, yb - 6, f"bar: mean KLD {bar:g} (and 97% margin-qualified top-1)", size=10.5, color="warn",
           anchor="end")
    for k in (0, bar / 2, bar, bar * 1.5):
        if k <= kmax:
            s.text(x0 - 6, Y(k) + 4, f"{k:.3f}", size=10, color="muted", anchor="end")
    s.text(x0 - 46, (y0 + y1) / 2, "mean KLD", size=10.5, color="muted", anchor="middle")
    for gb in sorted({round(g, 1) for g in sizes}):
        s.text(X(gb), y0 + 16, f"{gb:.1f} GB", size=10, color="muted", anchor="middle")
    for f in files:
        x, y = X(f["bytes"] / 1e9), Y(f["mean_kld"])
        color = "accent" if f.get("ours") else "muted"
        s.raw(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{6 if f is best else 5}" fill="{{{color}}}"/>')
        anchor = "end" if x > (x0 + x1) / 2 else "start"
        dx = -10 if anchor == "end" else 10
        s.text(x + dx, y - 4, f["label"], size=11.5, color="ink", anchor=anchor,
               weight="600" if f.get("ours") else None)
        s.text(x + dx, y + 10, f"{f['mean_kld']:.4f} mean KLD, {f['mq_top1_pct']:.2f}% MQ top-1", size=10.5,
               color=color, anchor=anchor)
    s.text(320, 284, rec["footer"], size=10.5, color="muted", anchor="middle")
    return s


def main():
    out = ROOT / "site/assets"
    if sys.argv[1] == "process":
        for t in PALETTE:
            process(t).write(out / f"readme-derive-process-{t}.svg")
    elif sys.argv[1] == "result":
        rec = json.loads(pathlib.Path(sys.argv[2]).read_text())
        for t in PALETTE:
            result(rec, t).write(out / f"readme-derive-result-{t}.svg")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
