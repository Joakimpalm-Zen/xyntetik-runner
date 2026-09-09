"""The function unit as anchored edits, not a rewrite (R15.2.3).

Measured on the public bank with a 7B at 4-bit: asked for the whole new
function, the model fixes the target test and breaks a neighbour, zero
verified in 52 samples. Asked for edits, it verifies some, but 22 of 52
answers quoted an anchor that is not in the function. Anchoring every
edit on one of the function's own lines, enforced by the runner's
JSON-schema decoding (the anchor is an ``enum`` of those lines, so an
invented anchor cannot be decoded at all), took the rejected answers to
zero without touching the model.

This module is that protocol in the product: the system prompt, the schema
built from the function, the application of a reply, and the inverse, a
human commit expressed as the same edits so a training completion has the
shape the evaluation asks for. An edit is one object,
``{line, until, mode, text}``: ``line`` and ``until`` are existing lines of
the function (matched with surrounding whitespace stripped; ``until`` is
``line`` for a one-line edit), ``mode`` is ``replace`` (the text takes the
place of the lines from ``line`` through ``until``, may span any number of
lines, empty deletes them) or ``insert_after`` (the text goes after
``line``). The runner's decoder accepts ``enum`` but refuses ``anyOf``
alternatives that share a first byte, so there is one object shape and the
mode is a field.

Indentation: text whose lines carry their own indentation is taken as
written; text with none on its first line (a model that dedented) is
re-indented to the anchor's depth, or to the block's depth after a line
that opens one.
"""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from typing import Any

SYSTEM = ("You are a software engineer. You receive a change request, the tests, and the current "
          "source of one function. Make the smallest change that satisfies the request: answer "
          "with a JSON object {\"edits\": [{\"line\": <an existing line of the function, verbatim>, "
          "\"until\": <the last existing line the edit covers, verbatim; the same line for a one-line "
          "edit>, \"mode\": \"replace\" or \"insert_after\", \"text\": <the new text: for replace it "
          "takes the place of the lines from line through until and may be empty to delete them; for "
          "insert_after it is added after line>}]}. Change as few lines as possible.")
MAX_EDITS = 12
ASK = "Answer with the edits object."
# The runner's schema decoder tracks the live candidates of an enum in a
# 64-bit mask and refuses more than 60 values. A function with more distinct
# lines than that cannot have its anchors enumerated; its schema keeps the
# shape and lets `line` be any string, and apply() refuses an anchor that
# is not in the function, so an invented anchor is measured as a rejection
# rather than decoded away. Measured 2026-09-09 on the wide bank: the long
# functions are where the enum mattered most, so this is a limit to lift in
# the decoder (a wider candidate set), not a design.
ENUM_CAP = 60


def anchor_lines(fn_text: str) -> list[str]:
    """The function's distinct non-blank lines, stripped: the only anchors an
    edit may name. A line that occurs twice is not an anchor (it would be
    ambiguous), so an edit there has to reach through a neighbour."""
    seen: dict[str, int] = {}
    for ln in fn_text.split("\n"):
        k = ln.strip()
        if k:
            seen[k] = seen.get(k, 0) + 1
    return [k for k, n in seen.items() if n == 1]


def schema(fn_text: str, *, max_edits: int = MAX_EDITS) -> dict[str, Any]:
    """The JSON schema the runner decodes the reply under."""
    anchors = anchor_lines(fn_text)
    line = {"type": "string", "enum": anchors} if len(anchors) <= ENUM_CAP else {"type": "string"}
    edit = {"type": "object",
            "properties": {"line": line,
                           "until": dict(line),
                           "mode": {"type": "string", "enum": ["replace", "insert_after"]},
                           "text": {"type": "string"}},
            "required": ["line", "until", "mode", "text"]}
    return {"type": "object",
            "properties": {"edits": {"type": "array", "minItems": 1, "maxItems": max_edits, "items": edit}},
            "required": ["edits"]}


def _indent(line: str) -> str:
    return line[:len(line) - len(line.lstrip())]


def _lines_for(body: str, indent: str) -> list[str]:
    """The lines a text contributes: verbatim when it carries indentation of
    its own, re-indented to ``indent`` when its first line has none."""
    if not body.strip():
        return []
    rl = body.strip("\n").split("\n")
    first = next(x for x in rl if x.strip())
    if _indent(first):
        return [x if x.strip() else "" for x in rl]
    common = min((len(x) - len(x.lstrip()) for x in rl if x.strip()), default=0)
    return [(indent + x[common:]) if x.strip() else "" for x in rl]


def apply(fn_text: str, reply: str) -> tuple[str, str]:
    """Apply a reply's edits to the function; (new text, reason), the reason
    empty on success. Every anchor must name exactly one line, a range may
    be named in either order, and edits may not overlap."""
    try:
        edits = json.loads(reply)["edits"]
        if not isinstance(edits, list):
            raise TypeError
    except (ValueError, KeyError, TypeError):
        return fn_text, "reply is not the edits object"
    if not edits:
        return fn_text, "no edits"
    lines = fn_text.split("\n")
    stripped = [x.strip() for x in lines]

    def locate(key: Any) -> int | str:
        k = str(key or "").strip()
        idx = [i for i, x in enumerate(stripped) if x == k]
        return idx[0] if len(idx) == 1 else f"line occurs {len(idx)} times: {k[:40]!r}"

    ops: list[tuple[int, int, dict[str, Any]]] = []
    for e in edits:
        if not isinstance(e, dict):
            return fn_text, "an edit is not an object"
        i = locate(e.get("line"))
        if isinstance(i, str):
            return fn_text, i
        j = i
        if e.get("mode", "replace") == "replace" and e.get("until") not in (None, ""):
            j = locate(e.get("until"))
            if isinstance(j, str):
                return fn_text, j
            if j < i:  # the range named backwards is the same range
                i, j = j, i
        ops.append((i, j, e))
    ops.sort(key=lambda t: t[0])
    for (_, e1, _), (i2, _, _) in zip(ops, ops[1:]):
        if i2 <= e1:
            return fn_text, "edits overlap"
    out = list(lines)
    for i, j, e in sorted(ops, key=lambda t: -t[0]):  # from the bottom, so indices stay valid
        body = str(e.get("text", ""))
        if e.get("mode", "replace") == "replace":
            out[i:j + 1] = _lines_for(body, _indent(lines[i]))
        else:
            # inserting after a line that opens a block takes the block's
            # indentation (the next non-blank line's), not the opener's
            indent = _indent(lines[i])
            nxt = next((x for x in lines[i + 1:] if x.strip()), "")
            if lines[i].rstrip().endswith(":") and nxt:
                indent = _indent(nxt)
            out[i + 1:i + 1] = _lines_for(body, indent)
    return "\n".join(out), ""


@dataclass(frozen=True)
class Derived:
    edits: list[dict[str, str]]
    text: str      # the function the edits produce, to be judged like any answer
    reason: str    # why the change could not be expressed, empty on success


def derive(base_fn: str, sol_fn: str, *, max_edits: int = MAX_EDITS) -> Derived:
    """The human's change as anchored edits. Each changed region becomes one
    ``replace`` from the nearest anchor at or above it through the nearest
    anchor at or below it (lines in between are carried in the text, since
    a blank or repeated line cannot be named); regions that come to share an
    anchor merge. The result is applied and returned for judging: a
    training completion has to earn its place under the same verifier as
    any answer, so a difference the protocol cannot express (a blank line
    it drops) is decided by the tests, never assumed harmless."""
    b, s = base_fn.split("\n"), sol_fn.split("\n")
    anchors = set(anchor_lines(base_fn))
    is_anchor = [x.strip() in anchors for x in b]
    ops = difflib.SequenceMatcher(None, b, s, autojunk=False).get_opcodes()
    if all(tag == "equal" for tag, *_ in ops):
        return Derived([], base_fn, "no change")
    m: dict[int, int] = {}  # base line -> solution line, for the lines the change keeps
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            for i in range(i1, i2):
                m[i] = j1 + (i - i1)
    # a block: base lines h..e inclusive, solution lines sj..ej exclusive, and
    # whether it inserts after h or replaces h..e
    blocks: list[list[Any]] = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        if tag == "insert" and i1 > 0 and is_anchor[i1 - 1]:
            blocks.append([i1 - 1, i1 - 1, j1, j2, "insert_after"])
            continue
        h = i1 - 1 if tag == "insert" else i1
        while h >= 0 and not is_anchor[h]:
            h -= 1
        if h < 0:
            return Derived([], base_fn, "no anchor at or above the change")
        e = i2 - 1 if tag != "insert" else i1  # an insertion spans the point it lands on
        while e < len(b) and not is_anchor[e]:
            e += 1
        if e >= len(b):
            return Derived([], base_fn, "no anchor at or below the change")
        sj = m[h] if h in m else j1
        ej = m[e] + 1 if e in m else j2
        blocks.append([h, e, sj, ej, "replace"])
    blocks.sort(key=lambda x: (x[0], x[1]))
    merged: list[list[Any]] = []
    for blk in blocks:
        if merged and blk[0] <= merged[-1][1]:
            last = merged[-1]
            last[1] = max(last[1], blk[1])
            last[2] = min(last[2], blk[2])
            last[3] = max(last[3], blk[3])
            last[4] = "replace"
        else:
            merged.append(list(blk))
    if len(merged) > max_edits:
        return Derived([], base_fn, f"needs {len(merged)} edits, the schema allows {max_edits}")
    edits = [{"line": b[h].strip(), "until": b[e].strip(), "mode": mode, "text": "\n".join(s[sj:ej])}
             for h, e, sj, ej, mode in merged]
    got, why = apply(base_fn, json.dumps({"edits": edits}))
    if why:
        return Derived([], base_fn, why)
    return Derived(edits, got, "")


def render(edits: list[dict[str, str]]) -> str:
    """The completion text a training example carries: the same compact JSON
    the decoder produces."""
    return json.dumps({"edits": edits}, separators=(",", ":"), ensure_ascii=False)
