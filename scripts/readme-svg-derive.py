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
    """One horizontal bar per file, its length the mean KLD on a stated scale,
    the bar's 0.05 drawn across; size and margin-qualified top-1 beside the
    name. The files are drawn in the record's order."""
    files = rec["files"]
    bar = rec.get("bar_mean_kld", 0.05)
    kmax = 0.10 if max(f["mean_kld"] for f in files) <= 0.10 else max(f["mean_kld"] for f in files) * 1.1
    w, h = 640, 300
    x0, x1 = 262, 560
    def X(k): return x0 + k / kmax * (x1 - x0)
    s = Svg(w, h, rec["title"], theme)
    s.text(18, 28, rec["heading"], size=12.5, color="muted", weight="600")
    s.text(18, 44, rec["window"], size=10.5, color="muted")
    top, step = 72, 56
    xb = X(bar)
    s.raw(f'<line x1="{xb:.1f}" y1="{top - 10}" x2="{xb:.1f}" y2="{top + step * len(files) - 14}" '
          'stroke="{warn}" stroke-dasharray="4 4"/>')
    s.text(xb, top - 14, f"bar: mean KLD {bar:g}", size=10.5, color="warn", anchor="middle")
    for i, f in enumerate(files):
        y = top + step * i
        ours = f.get("ours")
        s.text(18, y + 14, f["label"], size=11.5, color="ink", weight="600" if ours else None)
        s.text(18, y + 29, f"{f['bytes'] / 1e9:.2f} GB, {f['mq_top1_pct']:.2f}% MQ top-1", size=10.5,
               color="muted")
        color = "accent" if ours else "muted"
        s.raw(f'<rect x="{x0}" y="{y + 6}" width="{X(f["mean_kld"]) - x0:.1f}" height="22" rx="3" '
              f'fill="{{{color}}}"/>')
        fails = f["mean_kld"] > bar or f["mq_top1_pct"] < 97.0
        s.text(X(f["mean_kld"]) + 6, y + 21, f"{f['mean_kld']:.4f}" + ("  fails" if fails else ""),
               size=11.5, color="warn" if fails else "ink", weight="600" if ours else None)
    ys = top + step * len(files) - 2
    s.raw(f'<line x1="{x0}" y1="{ys}" x2="{x1}" y2="{ys}" stroke="{{line}}"/>')
    for k in (0, kmax / 2, kmax):
        s.text(X(k), ys + 14, f"{k:.2f}", size=10, color="muted", anchor="middle")
    s.text((x0 + x1) / 2, ys + 28, "mean KLD against the float32 model (shorter is closer)", size=10.5,
           color="muted", anchor="middle")
    s.text(320, 288, rec["footer"], size=10.5, color="muted", anchor="middle")
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
