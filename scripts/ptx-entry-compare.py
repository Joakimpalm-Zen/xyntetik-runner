#!/usr/bin/env python3
"""Compare the PTX entries two embedded headers carry (src/kernels_ptx.h and src/kernels_ptx80.h, as embed-ptx.py
writes them), entry by entry and by name.

What counts as a change is the advisor's PTX ruling (R3.84 and R4.48.33):
- An entry is "changed" if its text differs after replacing nvcc's function index f in exactly two tokens,
  $L__BB<f>_<b> (f only; the block number b stays) and __local_depot<f>, and nothing wider. Adding or removing a
  kernel renumbers every later function's f, which is not a change to them.
- Otherwise the text is compared as check-ptx-source.py compares it: comments and spacing are not instructions;
  register names and operands are.
- Every label inside one entry must carry the same f. A mixed f fails loudly (exit 2): it means the split or the
  renumbering is not what this script assumes.

Out: the entries added, removed and changed (by name), whether the surviving entries keep their order, and whether
the text outside the entries changed. Exit 1 on any change, a reordering or changed module text, unless
--allow-changed is given; exit 2 on a mixed f or a parse error; 0 otherwise. Added and removed entries are reported
and do not by themselves fail.

usage: ptx-entry-compare.py OLD.h NEW.h [--allow-changed]
"""
import argparse
import ast
import re
import sys

ENTRY = re.compile(r"(?m)^[ \t]*(?:\.visible[ \t]+|\.weak[ \t]+)?\.entry[ \t]+([A-Za-z_$][\w$]*)")
LABEL = re.compile(r"\$L__BB(\d+)_(\d+)")
DEPOT = re.compile(r"__local_depot(\d+)")


class MixedIndex(Exception):
    """Labels inside one entry carry more than one function index."""


def embedded_text(path):
    """The PTX a byte-embedded header carries: its string literals, joined (as check-ptx-source.py reads it)."""
    with open(path, encoding="utf-8") as f:
        header = f.read()
    return "".join(ast.literal_eval(line.strip()) for line in header.splitlines() if line.strip().startswith('"'))


def normalize(text):
    """check-ptx-source.py's normalization: comments and spacing are not instructions."""
    text = re.sub(r"(?m)^\.version\s+\S+", ".version 7.8", text)
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    return " ".join(text.split())


def split_entries(text):
    """[(name, body)] in file order, and the module-level text outside every entry. An entry runs from its .entry
    line to the brace that closes its body."""
    entries, outside, pos = [], [], 0
    for m in ENTRY.finditer(text):
        if m.start() < pos:
            raise ValueError(f"entry {m.group(1)} starts inside the previous entry's body")
        brace = text.find("{", m.end())
        if brace < 0:
            raise ValueError(f"entry {m.group(1)} has no body")
        depth, end = 0, None
        for b in re.finditer(r"[{}]", text[brace:]):
            depth += 1 if b.group() == "{" else -1
            if depth == 0:
                end = brace + b.end()
                break
        if end is None:
            raise ValueError(f"entry {m.group(1)}'s body never closes")
        outside.append(text[pos:m.start()])
        entries.append((m.group(1), text[m.start():end]))
        pos = end
    outside.append(text[pos:])
    names = [n for n, _ in entries]
    if len(set(names)) != len(names):
        raise ValueError("an entry name appears twice")
    return entries, "".join(outside)


def normalize_entry(name, body):
    """The entry's text with its function index replaced in the two tokens, then normalized. Raises MixedIndex if
    its labels carry more than one f."""
    fs = {int(f) for f, _ in LABEL.findall(body)} | {int(f) for f in DEPOT.findall(body)}
    if len(fs) > 1:
        raise MixedIndex(f"entry {name}: its labels carry {len(fs)} function indices {sorted(fs)[:6]}")
    body = LABEL.sub(lambda m: f"$L__BB<f>_{m.group(2)}", body)
    body = DEPOT.sub("__local_depot<f>", body)
    return normalize(body)


def compare_texts(old, new):
    """The comparison of two PTX texts, as a dict."""
    a_entries, a_out = split_entries(old)
    b_entries, b_out = split_entries(new)
    a = {n: normalize_entry(n, t) for n, t in a_entries}
    b = {n: normalize_entry(n, t) for n, t in b_entries}
    a_order, b_order = [n for n, _ in a_entries], [n for n, _ in b_entries]
    common = set(a) & set(b)
    return {
        "entries_old": len(a), "entries_new": len(b),
        "added": [n for n in b_order if n not in a],
        "removed": [n for n in a_order if n not in b],
        "changed": [n for n in a_order if n in common and a[n] != b[n]],
        "order_identical": [n for n in a_order if n in common] == [n for n in b_order if n in common],
        "module_text_identical": normalize(a_out) == normalize(b_out),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--allow-changed", action="store_true",
                    help="report changed entries, a reordering or changed module text without failing")
    args = ap.parse_args(argv)
    try:
        r = compare_texts(embedded_text(args.old), embedded_text(args.new))
    except MixedIndex as e:
        print(f"ptx-entry-compare: FAIL: mixed function index: {e}", file=sys.stderr)
        return 2
    except (ValueError, SyntaxError, OSError) as e:
        print(f"ptx-entry-compare: FAIL: cannot read the PTX: {e}", file=sys.stderr)
        return 2
    print(f"entries: {r['entries_old']} -> {r['entries_new']}")
    for key in ("added", "removed", "changed"):
        names = r[key]
        print(f"{key}: {len(names)}" + (": " + ", ".join(names) if names else ""))
    print(f"survivors' order identical: {'yes' if r['order_identical'] else 'NO'}")
    print(f"module text identical: {'yes' if r['module_text_identical'] else 'NO'}")
    bad = r["changed"] or not r["order_identical"] or not r["module_text_identical"]
    return 1 if bad and not args.allow_changed else 0


if __name__ == "__main__":
    sys.exit(main())
