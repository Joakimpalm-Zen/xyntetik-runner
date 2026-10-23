#!/usr/bin/env python3
"""A distillation teacher for `runner --train --train-teacher`, from a parent GGUF.

Scores each document with `runner --score --score-top K` on the parent (the
higher-precision file a quantized student should come back toward) and
writes the three files a distillation run reads:

    OUT/train.jsonl     one id-form line per document: the first token as
                        "prompt_tokens", the rest as "completion_tokens"
    OUT/topk_idx.npy    int32 [documents, positions, K]: the parent's top-K
                        token ids for each completion token
    OUT/topk_logp.npy   float32, the same shape: their natural-log
                        probabilities over the full vocabulary

Positions past a shorter document's end are padding (ids 0, log-probability
0) and are never read. The token ids are the parent's tokenization of the
text, so the student must share the parent's vocabulary, which every
quantization of one model does.

    kd-teacher.py --runner ./runner --parent model-Q8_0.gguf \\
        --docs docs.jsonl --topk 64 --out teacher/
    runner -m model-Q4_0.gguf --gpu off --train teacher/train.jsonl \\
        --train-teacher teacher/ --train-steps 100 --train-out adapter.gguf

`--docs` is a .jsonl file of {"text": ...} lines. A teacher from a pipeline
of its own (the publisher's float32 reference, say) is used the same way, as
long as it writes the same three files.
"""
import argparse
import json
import pathlib
import struct
import subprocess
import sys
import tempfile


def write_npy(path, descr, shape, payload):
    dims = ", ".join(str(s) for s in shape) + ("," if len(shape) == 1 else "")
    hdr = "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }" % (descr, dims)
    hdr += " " * ((64 - (10 + len(hdr) + 1) % 64) % 64) + "\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00")
        f.write(struct.pack("<H", len(hdr)))
        f.write(hdr.encode("ascii"))
        f.write(payload)


def score(runner, parent, text, k, extra):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        f.write(text)
        path = f.name
    try:
        p = subprocess.run([runner, "-m", parent, "--score", "--score-top", str(k),
                            "-f", path, *extra], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    finally:
        pathlib.Path(path).unlink()
    if p.returncode != 0:
        sys.exit("runner --score failed: " + p.stderr.decode(errors="replace")[-400:])
    return json.loads(p.stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runner", required=True)
    ap.add_argument("--parent", required=True, help="the teacher GGUF")
    ap.add_argument("--docs", required=True, help='.jsonl of {"text": ...} lines')
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("runner_args", nargs="*",
                    help="after --: extra runner arguments (e.g. --gpu off -c 2048)")
    a = ap.parse_args()
    if not 1 <= a.topk <= 256:
        sys.exit("--topk must be 1..256 (the range of --score-top)")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = [json.loads(l)["text"] for l in
            pathlib.Path(a.docs).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not docs:
        sys.exit("no documents in --docs")
    scored = []
    for i, text in enumerate(docs):
        s = score(a.runner, a.parent, text, a.topk, a.runner_args)
        toks, top = s["tokens"], s["top"]
        if len(toks) < 2 or len(top) != len(toks) - 1:
            sys.exit(f"document {i + 1}: {len(toks)} tokens, {len(top)} scored positions")
        if any(len(row) != a.topk for row in top):
            sys.exit(f"document {i + 1}: a position lists fewer than {a.topk} tokens")
        scored.append((toks, top))
    pos = max(len(t) - 1 for t, _ in scored)
    ids, lps = [], []
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for toks, top in scored:
            f.write(json.dumps({"prompt_tokens": toks[:1],
                                "completion_tokens": toks[1:]}) + "\n")
            for p in range(pos):
                row = top[p] if p < len(top) else [[0, 0.0]] * a.topk
                ids += [int(t) for t, _ in row]
                lps += [float(l) for _, l in row]
    shape = (len(scored), pos, a.topk)
    write_npy(out / "topk_idx.npy", "<i4", shape, struct.pack("<%di" % len(ids), *ids))
    write_npy(out / "topk_logp.npy", "<f4", shape, struct.pack("<%df" % len(lps), *lps))
    mass = [sum(2.718281828459045 ** l for _, l in row) for _, top in scored for row in top]
    print(json.dumps({"documents": len(scored), "positions": pos, "topk": a.topk,
                      "mean_topk_mass": round(sum(mass) / len(mass), 6),
                      "out": str(out)}))


if __name__ == "__main__":
    main()
