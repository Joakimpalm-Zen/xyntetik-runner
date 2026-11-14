#!/usr/bin/env python3
"""Generate tests/jsre_cases.h: what ECMA-262 says about hand-written patterns,
for tests/test_jsre.c (R2.18).

    scripts/gen-jsre-cases.py > tests/jsre_cases.h

The oracle is ECMA-262 itself: node's RegExp with the u flag, the meaning a
JSON Schema `pattern` has. A pattern the u flag rejects but legacy mode
accepts (an identity escape such as \\-, a lone brace, a lone bracket) is read
as legacy mode reads it, with code-point meanings: on BMP strings legacy mode
is the oracle, on others the pattern rewritten into its Unicode-mode spelling.
Python's re is asked too, only to count the cases where it parts from ECMA;
those are the point of most of the list.

Two kinds of case. Hand cases: patterns chosen for where engines and ECMA's
own modes part (\\d \\w \\s beyond ASCII, the dot and line terminators, astral
code points, legacy literals, [] and [^], the anchors). Generated divergences:
shorthand classes, their complements and the dot, inside and outside a class,
against code points drawn from the Unicode database (a fixed seed) where
Python's Unicode meanings and ECMA's ASCII ones disagree; only the pairs where
Python's re and ECMA really disagree are kept.

Each string's `want` is what the automaton must answer: ECMA's verdict, and a
refusal for any string JSON cannot spell without an escape (a control
character, a quote, a backslash), since a pattern-constrained string admits no
escape. Patterns jsre must refuse carry the reason it must give.
"""

import json
import random
import re
import subprocess
import sys
import unicodedata
import warnings

warnings.simplefilter("ignore")

D = ["123", "9", "09", "٣", "０", "߀", "१२", "𝟘", "1٣", "", "a"]
WS = [" ", " ", "﻿", " ", " ", " ", " ", " ", " ", " ",
      "　", "\u0085", "᠎", "​", "x", "\t", "\x1c"]

# (pattern, strings) or (pattern, None, the reason jsre must give)
HAND = [
    (r"^\d+$", D), (r"^\D$", ["٣", "0", "a", "０"]), (r"\d", ["a٣", "a3"]),
    (r"^\w+$", ["é", "ß", "ſ", "K", "_", "a1", "ǅ", "ﬁ", "ab"]), (r"^\W$", ["é", "a", "_", " ", "ß"]),
    (r"^\s$", WS), (r"^\S$", WS), (r"^[\s]$", WS[:6]), (r"^[^\s]$", ["\u0085", "᠎", " ", "a"]),
    (r"^.$", ["\n", "\r", " ", " ", "\u0085", "😀", "é", "a", ""]),
    (r"^..$", ["😀", "ab", "😀😀", "a😀"]), (r"^[^a]$", ["😀", "b", "a", "\U0010ffff"]),
    (r"^[😀-😂]$", ["😁", "😃", "😀"]), (r"^[\s\S]$", ["😀", " ", "a", "ab"]),
    (r"^a$", ["a", "a\n", "ab", "ba"]), (r"a$", ["ba", "ab"]), (r"^a", ["ba", "ab"]), (r"^$", ["", "a"]),
    (r"a{,2}", ["a{,2}", "aa", "a", ""]), (r"a{", ["a{", "a"]), (r"a}", ["a}", "a"]), (r"^{a}$", ["{a}", "a"]),
    (r"^x{1}}$", ["x}", "x"]), (r"^{$", ["{", ""]), (r"^\{\d+\}$", ["{12}", "12"]),
    (r"^\-\d$", ["-1", "1"]), (r"^\_$", ["_"]), (r"^a\@b$", ["a@b", "ab"]), (r"^a\ b$", ["a b", "ab"]),
    (r"^\:$", [":"]), (r"^\#$", ["#"]), (r"^\%\&\,$", ["%&,"]), (r"^\<\>\=$", ["<>="]), (r"^\!\~$", ["!~"]),
    (r"^\'$", ["'"]), (r"^\/$", ["/", "\\/"]), (r"^\.$", [".", "a"]),
    (r"^\x41$", ["A", "x41"]), (r"^A$", ["A"]), (r"^\u{41}$", ["A", "u{41}"]), (r"^\u{1F600}$", ["😀"]),
    (r"^😀$", ["😀", "a"]), (r"^[^]$", ["a", "😀", ""]), (r"a|[]", ["a", "b"]),
    (r"^[]a]$", ["]", "a", "a]", "]a"]), (r"^[[:alpha:]]$", ["a]", "a", ":]", "p]"]),
    (r"^[[a-z]*[-]?[a-z]*]*$", ["ab", "a-b", "[a]", "a]]", "A"]), (r"^\[\d\]$", ["[1]", "1"]),
    (r"^[a-]$", ["a", "-", "b"]), (r"^[-a]$", ["-", "a"]), (r"^[a\-z]$", ["-", "b", "a"]),
    (r"^[\w-]+$", ["a-b", "é"]), (r"^\x7f$", ["\x7f"]), (r"^a{2,}$", ["a", "aa", "aaaa"]),
    (r"^a{0}b$", ["b", "ab"]), (r"^(ab){2}$", ["abab", "ab"]), (r"^x*?y+?$", ["xy", "y", "xx"]),
    (r"^(|a)$", ["", "a", "aa"]), (r"a|", ["", "zzz"]), (r"^(a*)*$", ["", "aaa", "b"]),
    (r"^é$", ["é", "é"]), (r"^abc$", ["abc", "ABC"]), (r"^a{1000}$", ["a" * 1000, "a" * 999]),
    (r"^(?:a|b)+$", ["abba", "abc"]), (r"^[^\d\s]+$", ["ab", "a b", "a٣"]), (r"^[\D]$", ["٣", "3"]),
    (r"^[^\W]$", ["é", "a"]), (r"^\S+@\S+$", ["a@b", "a @b", "é@ "]), (r"^[\d-]+$", ["1-2", "-", "a"]),
    (r"^\d{3}-\d{4}$", ["555-1234", "55-1234", "٣٣٣-1234"]), (r"[a-z]+\.[a-z]{2,}", ["x.io", "x.i", "-x.com-"]),
    # refused, by name
    (r"^[\d-z]$", None, "a class range from a class escape"),
    (r"^\a$", None, "an unknown letter escape"), (r"^\e$", None, "an unknown letter escape"),
    (r"^a\z$", None, "an unknown letter escape"), (r"^a\Z$", None, "an unknown letter escape"),
    (r"\Aa", None, "an unknown letter escape"),
    (r"a**", None, "a quantifier on a quantifier"), (r"a{2}{3}", None, "a quantifier on a quantifier"),
    (r"a++", None, "a quantifier on a quantifier"), (r"(?>a)", None, "an inline flag"),
    (r"(?i)abc", None, "an inline flag"), (r"(?<n>a)", None, "a named group"),
    (r"(a)\1", None, "a backreference"), (r"\1", None, "a backreference"), (r"a(?=b)", None, "a lookahead"),
    (r"(?<!a)b", None, "a lookbehind"), (r"\bfoo\b", None, "a word boundary"),
    (r"\p{L}", None, "a Unicode property class"), (r"\k<a>", None, "a named backreference"),
    (r"x{2,1}", None, "a count range out of order"), (r"^a{5000}$", None, "automaton states"),
    (r"^\u{110000}$", None, "a malformed \\u{...} escape"), (r"^[", None, "an unterminated character class"),
    (r"[\b]", None, "no string JSON can spell"), (r"\cA", None, "no string JSON can spell"),
    (r"\0", None, "no string JSON can spell"), (r"\uD83D", None, "no string JSON can spell"),
    (r"^[\uD800-\uDFFF]$", None, "no string JSON can spell"), (r"^[]$", None, "no string JSON can spell"),
    (r'^"$', None, "no string JSON can spell"), (r"^\\$", None, "no string JSON can spell"),
]

SYNTAX = set("^$\\.*+?()[]{}|/")


def rewrite_u(p):
    """A legacy-mode pattern as Unicode mode spells it: an identity escape of a
    non-syntax character becomes the character, a brace or a ] that opens or
    closes nothing becomes escaped."""
    out, i, cls = [], 0, False
    while i < len(p):
        c = p[i]
        if c == "\\" and i + 1 < len(p):
            e = p[i + 1]
            out.append(c + e if e in SYNTAX or e.isalnum() or (e == "-" and cls) else e)
            i += 2
            continue
        if not cls and c == "{":
            m = re.match(r"\{\d+(,\d*)?\}", p[i:])
            out.append(m.group(0) if m else "\\{")
            i += len(m.group(0)) if m else 1
            continue
        if not cls and c in "}]":
            out.append("\\" + c)
            i += 1
            continue
        if c == "[" and not cls:
            cls = True
        elif c == "]" and cls:
            cls = False
        out.append(c)
        i += 1
    return "".join(out)


def generated():
    rng = random.Random(1114)
    cps = [c for c in range(0x80, 0x30000) if not 0xD800 <= c <= 0xDFFF]
    cat = {c: unicodedata.category(chr(c)) for c in cps}
    pool = (rng.sample([c for c in cps if cat[c] == "Nd"], 40) +
            rng.sample([c for c in cps if cat[c].startswith("L")], 40) +
            rng.sample([c for c in cps if cat[c] in ("Nl", "No")], 15) +
            rng.sample([c for c in cps if cat[c].startswith("M")], 15) +
            [0x85, 0xFEFF, 0x180E, 0x200B, 0x2028, 0x2029, 0xA0, 0x1680, 0x3000, 0x1C, 0x1D, 0x0D, 0x0A])
    chars = [chr(c) for c in pool] + ["5", "a", "_", " "]
    pats = [r"^\d$", r"^\D$", r"^\w$", r"^\W$", r"^\s$", r"^\S$", r"^.$", r"^[\d]$", r"^[^\d]$", r"^[\w]$",
            r"^[^\w]$", r"^[\s]$", r"^[^\s]$", r"^[\w.-]+$", r"^\d+$", r"^a\w*z$", r"^\S+$", r"^[\D\s]$",
            r"x.y", r"^[^\W\d]$"]
    out = []
    for p in pats:
        if p.startswith("^a"):
            ss = ["a" + c + "z" for c in chars]
        elif p == "x.y":
            ss = ["x" + c + "y" for c in chars]
        else:
            ss = chars
        out.append((p, ss))
    return out


def ask_node(cases):
    js = r"""
const cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
console.log(JSON.stringify({ version: process.version, r: cases.map(([p, rw, ss]) => {
  let u = null, b = null, w = null;
  try { const re = new RegExp(p, "u"); u = ss.map(s => re.test(s)); } catch {}
  try { const re = new RegExp(p); b = ss.map(s => re.test(s)); } catch {}
  if (u === null && rw !== null) try { const re = new RegExp(rw, "u"); w = ss.map(s => re.test(s)); } catch {}
  return { u, b, w };
}) }));"""
    r = subprocess.run(["node", "-e", js], input=json.dumps(cases), capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def producible(s):
    return all(not (ord(ch) < 0x20 or ch in '"\\') for ch in s)


def cstr(b):
    return '"' + "".join(chr(x) if 0x20 <= x < 0x7F and x not in b'"\\?' else "\\%03o" % x for x in b) + '"'


def main():
    compiled = [(p, c[1]) for c in HAND for p in [c[0]] if c[1] is not None]
    gen = generated()
    node = ask_node([[p, rewrite_u(p), ss] for p, ss in compiled + gen])
    rows, n_div, n_acc, n_rej, n_unsp, n_legacy = [], 0, 0, 0, 0, 0
    for k, ((p, ss), o) in enumerate(zip(compiled + gen, node["r"])):
        is_gen = k >= len(compiled)
        try:
            py = re.compile(p)
        except re.error:
            py = None
        for j, s in enumerate(ss):
            if o["u"] is not None:
                e = o["u"][j]
            elif o["b"] is not None and o["w"] is not None:
                n_legacy += 1
                e = o["b"][j] if all(ord(c) < 0x10000 for c in s) else o["w"][j]
            else:
                sys.exit("pattern %r: rejected by node in both modes but listed as compiled" % p)
            div = py is not None and bool(py.search(s)) != e
            if is_gen and not div:
                continue
            n_div += div
            want = e and producible(s)
            n_unsp += not producible(s)
            n_acc += want
            n_rej += not want
            rows.append("    { %s, %s, %d, %d, %d }," % (cstr(p.encode()), cstr(s.encode()), len(s.encode()),
                                                         int(want), int(e)))
    refused = ["    { %s, %s }," % (cstr(p.encode()), cstr(why.encode())) for p, _, why in (c for c in HAND if c[1] is None)]
    print("// Generated by scripts/gen-jsre-cases.py: do not edit; regenerate with")
    print("//     scripts/gen-jsre-cases.py > tests/jsre_cases.h")
    print("// Oracle: node %s RegExp (ECMA-262, u flag; legacy-only patterns as legacy" % node["version"])
    print("// mode reads them on BMP strings). Python %s re counted where it parts" % sys.version.split()[0])
    print("// from ECMA; Unicode database %s for the generated divergences." % unicodedata.unidata_version)
    print("// %d strings: %d to accept, %d to refuse (%d JSON cannot spell literally);" % (len(rows), n_acc, n_rej, n_unsp))
    print("// %d where Python's re and ECMA disagree; %d under a legacy-only pattern." % (n_div, n_legacy))
    print("#ifndef RUNNER_JSRE_CASES_H")
    print("#define RUNNER_JSRE_CASES_H")
    print("typedef struct { const char *pat, *s; int n; int want, ecma; } jsre_case;")
    print("static const jsre_case JSRE_CASES[] = {")
    print("\n".join(rows))
    print("};")
    print("typedef struct { const char *pat, *why; } jsre_refusal;")
    print("static const jsre_refusal JSRE_REFUSALS[] = {")
    print("\n".join(refused))
    print("};")
    print("#endif")


if __name__ == "__main__":
    main()
