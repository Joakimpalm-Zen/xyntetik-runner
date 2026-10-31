"""A runner --score record, and what a comparison of two of them may conclude.

`score_route` (the last field, since 2026-10-31) names the model file (its
name and bytes) and the path the record's forwards took: `solo` (one row a
forward, the decode arithmetic, the default) or `keep` (the batched rows of a
chunked score, RUNNER_SCORE_CHUNKED=1), with the widest rows, and the
backend's detail line. Two records are the same
scoring when everything but `score_route` is equal: the route record differs
whenever a route switch is on, whether or not it changed a logit.
"""
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def body(rec):
    """The record without its route field: what two routes' bytes are compared on."""
    return {k: v for k, v in rec.items() if k != "score_route"}


def body_bytes(out):
    """A --score stdout's bytes without its route field (the last one), still
    a JSON record: what a byte comparison across files or route switches
    compares."""
    head, sep, _ = out.partition(b',"score_route":')
    return head + b"}\n" if sep else out


def same_scores(a, b):
    return body(a) == body(b)


def path_of(rec):
    """'solo', 'keep', 'mixed', or None for a record that predates score_route."""
    r = rec.get("score_route")
    if not isinstance(r, dict):
        return None
    paths = set((r.get("paths") or {}).keys())
    if paths == {"solo"}:
        return "solo"
    if "keep" in paths and "solo" not in paths:
        return "keep"
    return "mixed" if paths else None


def model_of(rec):
    """(file name, bytes) of the scored model, or None for an older record."""
    m = (rec.get("score_route") or {}).get("model")
    if not isinstance(m, dict):
        return None
    return (m.get("file"), m.get("bytes"))


def rows_max(rec):
    r = rec.get("score_route") or {}
    return max((p.get("rows_max", 0) for p in (r.get("paths") or {}).values()), default=0)


def main(argv):
    """score_record.py A B: exit 0 when two --score files are the same bytes
    once their route field is left out (a byte gate across a route switch)."""
    if len(argv) != 2:
        sys.exit("usage: score_record.py A.json B.json")
    with open(argv[0], "rb") as fa, open(argv[1], "rb") as fb:
        return 0 if body_bytes(fa.read()) == body_bytes(fb.read()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
