import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_capture_cli_accepts_an_immutable_reference_revision():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "difftok.py"), "--help"],
        text=True, capture_output=True, check=True)
    assert "--ref-revision" in proc.stdout


import hashlib  # noqa: E402
import json  # noqa: E402

import pytest  # noqa: E402

REFS = ROOT / "tests" / "compatibility" / "tokenizer-references"
CORPUS = ROOT / "tests" / "fixtures" / "tokenizer-corpus.txt"


def _corpus():
    return [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l]


@pytest.mark.parametrize("name", ["mistral-7b-instruct-v0.3", "phi-3.5-mini-instruct",
                                  "salamandra-7b-instruct"])
def test_sentencepiece_references_are_what_they_say(name):
    """R6.2.2 (owner 2026-10-03): these three rows gate on the publisher's
    SentencePiece model; the tokenizer.json capture is kept beside it as the
    informational second reference. The marker rows are exactly the strings
    that spell one of the capture's own listed markers."""
    cap = json.loads((REFS / f"{name}.json").read_text())
    assert cap["source"] == "sentencepiece"
    assert len(cap["model_sha256"]) == 64 and len(cap["ref_revision"]) == 40
    corpus = _corpus()
    h = hashlib.sha256()
    for s in corpus:
        h.update(s.encode("utf-8"))
        h.update(b"\0")
    assert cap["corpus_sha256"] == h.hexdigest()
    assert len(cap["ids"]) == len(corpus)
    want = [i for i, s in enumerate(corpus) if any(m in s for m in cap["special_markers"])]
    assert cap["special_marker_rows"] == want
    assert "<s>" in cap["special_markers"]
    info = json.loads((REFS / f"{name}.tokenizer-json.json").read_text())
    assert info["corpus_sha256"] == cap["corpus_sha256"] and "source" not in info


def _difftok(capture, tmp_path, *extra):
    gguf = ROOT / "test.gguf"
    if not gguf.exists() or not (ROOT / "difftok").exists():
        pytest.skip("test.gguf or difftok not built")
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "difftok.py"),
                           "--gguf", str(gguf), "--ref-ids", str(capture),
                           "--no-build", "--show", "0", *extra],
                          text=True, capture_output=True)


def test_marker_rows_are_reported_and_not_gated(tmp_path):
    out = subprocess.run([str(ROOT / "difftok"), str(ROOT / "test.gguf"), str(CORPUS)],
                         text=True, capture_output=True)
    if out.returncode != 0:
        pytest.skip("difftok cannot read test.gguf")
    ids = [[int(x) for x in l.split()] for l in out.stdout.splitlines()]
    corpus = _corpus()
    h = hashlib.sha256()
    for s in corpus:
        h.update(s.encode("utf-8"))
        h.update(b"\0")
    def write(rows, changed):
        theirs = [list(r) for r in ids]
        for i in changed:
            theirs[i] = theirs[i] + [0]
        p = tmp_path / "cap.json"
        p.write_text(json.dumps({"corpus_sha256": h.hexdigest(), "ids": theirs,
                                 "special_marker_rows": rows}))
        return p
    ok = _difftok(write([3], [3]), tmp_path)
    assert ok.returncode == 0 and "1 more differ on a spelled special marker" in ok.stdout, ok.stdout
    bad = _difftok(write([3], [4]), tmp_path)
    assert bad.returncode == 1 and "1/" in bad.stdout, bad.stdout
    info = _difftok(write([], [4, 5]), tmp_path, "--report-only")
    assert info.returncode == 0 and "2/" in info.stdout, info.stdout
