"""R4.48.8.3: RAM-bounded pre-computed reference logits scoring in `runner derive`
and `runner --score --score-ref`.

Uses the generated tiny inkling2_loop F32 fixture (tests/inkling2_tiny.py) and
its real forward logits (captured via RUNNER_INKLING2_ROWS) to verify:
  1. Exact full-vocab float64 KL parity (< 1e-12) against an independent Python
     implementation of compare_full.py's float64 log_softmax KL formula.
  2. Identity self-check (F32 fixture vs its own pinned reference logits:
     mean_kld < 1e-12, top1 = 100%, mq_top1 = 100%, bar.verdict = "pass").
  3. Bit-identical results across "npy" dense, "raw-f32" dense, and "npy" sparse
     (row_indices) manifests.
  4. Fail-closed refusals (sha256 mismatch, corpus sha256 mismatch, vocab/size
     mismatch, window out of range, token/n_special mismatch, planted NaN/Inf,
     path traversal in logits.path, --reference-arg paired with logits:).
  5. Downstream derive integration (--compare, --card-from, derive.json pinning
     fields including producer_mode).
  6. scripts/make-ref-logits.py wrap + --ram-cap-gib mode dispatch & control
     gate (patched-stream is never auto-selected silently without explicit opt-in).
"""
import hashlib
import json
import math
import os
import pathlib
import struct
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TINY_INK2 = ROOT / "tests/inkling2_tiny.py"
MAKE_REF = ROOT / "scripts/make-ref-logits.py"

# Deterministic byte corpus (> 180 bytes so window 8:4:30 fits within -c 192).
CORPUS_TEXT = (
    "The inkling2 looped architecture applies its middle layers three times "
    "across each sequence while keeping per-layer sliding-window and global "
    "attention states distinct. Reference scoring streams one float32 logit "
    "row at a time in float64 to match compare_full.py without loading the "
    "publisher model into RAM.\n"
)
WINDOW = "8:4:30"
CTX = 384


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(65536)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_npy_f32_2d(path: pathlib.Path, rows: int, cols: int, raw_f32: bytes) -> None:
    """Write a standard NumPy .npy v1.0 little-endian float32 C-order 2D file."""
    assert len(raw_f32) == rows * cols * 4
    d = f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}"
    # Pad header so (10 + len(hdr)) is a multiple of 64, ending with '\n'.
    hdr_b = d.encode("ascii")
    pad = (64 - ((10 + len(hdr_b) + 1) % 64)) % 64
    hdr = hdr_b + (b" " * pad) + b"\n"
    with path.open("wb") as f:
        f.write(b"\x93NUMPY\x01\x00")
        f.write(struct.pack("<H", len(hdr)))
        f.write(hdr)
        f.write(raw_f32)


def unpack_rows_f32(raw_f32: bytes, n_rows: int, n_vocab: int):
    assert len(raw_f32) == n_rows * n_vocab * 4
    out = []
    stride = n_vocab * 4
    fmt = f"<{n_vocab}f"
    for r in range(n_rows):
        out.append(struct.unpack_from(fmt, raw_f32, r * stride))
    return out


def compare_full_kl_row(ref_row, cand_row):
    """Exact compare_full.py float64 full-vocab KL(ref || cand) + margin + top1."""
    # Exact float64 log_softmax on ref and cand:
    mr = max(ref_row)
    mo = max(cand_row)
    sr = sum(math.exp(float(x) - mr) for x in ref_row)
    so = sum(math.exp(float(x) - mo) for x in cand_row)
    lse_r = mr + math.log(sr)
    lse_o = mo + math.log(so)
    kld = 0.0
    max_abs = 0.0
    for xr, xo in zip(ref_row, cand_row):
        fr = float(xr)
        fo = float(xo)
        lr = fr - lse_r
        lo = fo - lse_o
        kld += math.exp(lr) * (lr - lo)
        d = abs(fr - fo)
        if d > max_abs:
            max_abs = d
    # Top-1 and top-2 of ref, top-1 of cand (first argmax on tie, matching C >):
    r_best1_i = 0
    r_best1 = float(ref_row[0])
    r_best2 = -math.inf
    for i in range(1, len(ref_row)):
        v = float(ref_row[i])
        if v > r_best1:
            r_best2 = r_best1
            r_best1 = v
            r_best1_i = i
        elif v > r_best2:
            r_best2 = v
    c_best1_i = 0
    c_best1 = float(cand_row[0])
    for i in range(1, len(cand_row)):
        v = float(cand_row[i])
        if v > c_best1:
            c_best1 = v
            c_best1_i = i
    return {
        "kld": kld,
        "gold_margin": r_best1 - r_best2,
        "gold_top1": r_best1_i,
        "cand_top1": c_best1_i,
        "agree": r_best1_i == c_best1_i,
        "max_abs": max_abs,
    }


def make_manifest(
    dir_path: pathlib.Path,
    logits_name: str,
    fmt: str,
    n_rows: int,
    n_vocab: int,
    tokens,
    n_special: int,
    corpus_path: pathlib.Path | None,
    row_indices=None,
    producer_mode="patched",
    logits_sha=None,
    manifest_name="ref.ref.json",
) -> pathlib.Path:
    lpath = dir_path / logits_name
    m = {
        "schema_version": "xyntetik.runner.ref-logits.v1",
        "logits": {
            "path": logits_name,
            "format": fmt,
            "dtype": "float32",
            "shape": [n_rows, n_vocab],
            "sha256": logits_sha if logits_sha is not None else sha256_file(lpath),
            "row_indices": row_indices,
        },
        "tokens": list(tokens),
        "n_special": n_special,
        "n_vocab": n_vocab,
        "reference": {
            "repo": "test/inkling2-tiny",
            "revision": "0000000000000000000000000000000000000000",
            "class": "Inkling2LoopForCausalLM",
            "dtype": "float32",
            "device": "cpu",
            "producer": "test_derive_ref_logits.py",
            "producer_mode": producer_mode,
            "weights_sha256": "a" * 64,
        },
    }
    if corpus_path is not None:
        m["corpus"] = {
            "path": corpus_path.name,
            "sha256": sha256_file(corpus_path),
            "bytes": corpus_path.stat().st_size,
        }
    mp = dir_path / manifest_name
    mp.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    return mp


@pytest.fixture(scope="module")
def inkling2_fixture(runner_bin, tmp_path_factory):
    """Build tiny inkling2 F32 parent + capture its exact float32 forward logits."""
    work = tmp_path_factory.mktemp("ink2_ref")
    parent = work / "ink2_f32.gguf"
    subprocess.run(
        [sys.executable, str(TINY_INK2), str(parent), "--dense", "f32"],
        check=True,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
    )
    corpus = work / "corpus.txt"
    corpus.write_text(CORPUS_TEXT, encoding="utf-8")

    # Capture real forward rows [n_scored, n_vocab] from the F32 fixture.
    rows_raw = work / "ref.logits.f32"
    env = {**os.environ, "RUNNER_INKLING2_ROWS": str(rows_raw)}
    p = subprocess.run(
        [
            str(runner_bin),
            "-m",
            str(parent),
            "--score",
            "-f",
            str(corpus),
            "-c",
            str(CTX),
            "-t",
            "2",
            "--gpu",
            "off",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=True,
        timeout=120,
    )
    score_doc = json.loads(p.stdout.decode("utf-8"))
    tokens = score_doc["tokens"]
    n_special = score_doc.get("n_special", 0)
    n_scored = score_doc["n_scored"]
    raw_bytes = rows_raw.read_bytes()
    # inkling2_tiny.py has VUSED = 260
    n_vocab = 260
    assert len(raw_bytes) == n_scored * n_vocab * 4

    # Also write .npy dense and .npy sparse (exact window positions) variants.
    npy_dense = work / "ref.logits.npy"
    write_npy_f32_2d(npy_dense, n_scored, n_vocab, raw_bytes)

    start, stride, count = (int(x) for x in WINDOW.split(":"))
    win_indices = [n_special + start + i * stride - 1 for i in range(count)]
    sparse_raw = b"".join(
        raw_bytes[e * n_vocab * 4 : (e + 1) * n_vocab * 4] for e in win_indices
    )
    npy_sparse = work / "ref-sparse.logits.npy"
    write_npy_f32_2d(npy_sparse, len(win_indices), n_vocab, sparse_raw)

    man_npy = make_manifest(
        work,
        npy_dense.name,
        "npy",
        n_scored,
        n_vocab,
        tokens,
        n_special,
        corpus,
        manifest_name="ref-npy.ref.json",
    )
    man_raw = make_manifest(
        work,
        rows_raw.name,
        "raw-f32",
        n_scored,
        n_vocab,
        tokens,
        n_special,
        corpus,
        producer_mode="wrap-raw-f32",
        manifest_name="ref-raw.ref.json",
    )
    man_sparse = make_manifest(
        work,
        npy_sparse.name,
        "npy",
        len(win_indices),
        n_vocab,
        tokens,
        n_special,
        corpus,
        row_indices=win_indices,
        manifest_name="ref-sparse.ref.json",
    )
    return {
        "work": work,
        "parent": parent,
        "corpus": corpus,
        "tokens": tokens,
        "n_special": n_special,
        "n_scored": n_scored,
        "n_vocab": n_vocab,
        "ref_raw": raw_bytes,
        "ref_rows": unpack_rows_f32(raw_bytes, n_scored, n_vocab),
        "win_indices": win_indices,
        "man_npy": man_npy,
        "man_raw": man_raw,
        "man_sparse": man_sparse,
    }


def run_derive(runner_bin, parent, out, corpus, ref_spec, *extra):
    p = subprocess.run(
        [
            str(runner_bin),
            "derive",
            "-m",
            str(parent),
            "-o",
            str(out),
            "--corpus",
            str(corpus),
            "--window",
            WINDOW,
            "-c",
            str(CTX),
            "-t",
            "2",
            "--gpu",
            "off",
            "--quant",
            "q8_0",
            "--license",
            "apache-2.0",
            "--reference",
            ref_spec,
            *extra,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
    )
    return p.returncode, p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace")


def test_derive_logits_exact_kl_parity_and_formats(runner_bin, inkling2_fixture, tmp_path):
    """Exact full-vocab float64 KL parity (< 1e-12) against compare_full.py formula,
    plus bit-identical results across npy dense, raw-f32 dense, and npy sparse."""
    fx = inkling2_fixture
    out_npy = tmp_path / "child-npy.gguf"
    rc, so, se = run_derive(
        runner_bin, fx["parent"], out_npy, fx["corpus"], f"logits:{fx['man_npy']}"
    )
    assert rc == 0, f"derive failed:\nSTDOUT:\n{so}\nSTDERR:\n{se}"

    # Capture child's actual forward logits independently to recompute exact KL.
    child_rows_f32 = tmp_path / "child.logits.f32"
    subprocess.run(
        [
            str(runner_bin),
            "-m",
            str(out_npy),
            "--score",
            "-f",
            str(fx["corpus"]),
            "-c",
            str(CTX),
            "-t",
            "2",
            "--gpu",
            "off",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env={**os.environ, "RUNNER_INKLING2_ROWS": str(child_rows_f32)},
        check=True,
        timeout=120,
    )
    cand_rows = unpack_rows_f32(child_rows_f32.read_bytes(), fx["n_scored"], fx["n_vocab"])

    gold_doc = json.loads((tmp_path / "child-npy.gguf.gold.json").read_text(encoding="utf-8"))
    rec_doc = json.loads((tmp_path / "child-npy.gguf.derive.json").read_text(encoding="utf-8"))

    assert gold_doc["schema_version"] == "xyntetik.runner.gold-logits.v4"
    assert gold_doc["summary"]["kl_method"] == "full-vocab-f64"
    assert len(gold_doc["rows"]) == len(fx["win_indices"])

    expected_klds = []
    expected_agree = 0
    expected_qual = 0
    expected_qual_hit = 0
    for row_out, e in zip(gold_doc["rows"], fx["win_indices"]):
        exp = compare_full_kl_row(fx["ref_rows"][e], cand_rows[e])
        got_a = row_out["a"]
        assert abs(got_a["kld"] - exp["kld"]) < 1e-12, (e, got_a["kld"], exp["kld"])
        assert abs(row_out["gold_margin"] - exp["gold_margin"]) < 1e-6
        assert got_a["agree"] == exp["agree"]
        expected_klds.append(exp["kld"])
        expected_agree += int(exp["agree"])
        if exp["gold_margin"] > 0.5:
            expected_qual += 1
            expected_qual_hit += int(exp["agree"])

    meas = rec_doc["measurement"]
    assert abs(meas["mean_kld"] - (sum(expected_klds) / len(expected_klds))) < 1e-12
    assert abs(meas["max_kld"] - max(expected_klds)) < 1e-12
    assert abs(meas["top1_agreement_pct"] - 100.0 * expected_agree / len(expected_klds)) < 1e-9
    if expected_qual > 0:
        assert abs(
            meas["margin_qualified_top1_pct"] - 100.0 * expected_qual_hit / expected_qual
        ) < 1e-9

    # Check pinned provenance fields in derive.json.
    assert rec_doc["reference_kind"] == "publisher-float32"
    ref_rec = rec_doc["reference"]
    assert ref_rec["manifest_sha256"] == sha256_file(fx["man_npy"])
    assert ref_rec["logits_sha256"] == sha256_file(fx["work"] / "ref.logits.npy")
    assert ref_rec["logits_format"] == "npy"
    assert ref_rec["kl_method"] == "full-vocab-f64"
    assert ref_rec["producer_mode"] == "patched"
    assert rec_doc["measurement"]["bar"] is not None
    assert rec_doc["measurement"]["bar"]["verdict"] in ("pass", "fail", "not_decided")

    # Verify raw-f32 dense and npy sparse produce matching measurements.
    for label, man, exp_fmt, exp_mode in [
        ("raw", fx["man_raw"], "raw-f32", "wrap-raw-f32"),
        ("sparse", fx["man_sparse"], "npy", "patched"),
    ]:
        out_x = tmp_path / f"child-{label}.gguf"
        rc2, so2, se2 = run_derive(
            runner_bin, fx["parent"], out_x, fx["corpus"], f"logits:{man}"
        )
        assert rc2 == 0, f"derive ({label}) failed:\n{so2}\n{se2}"
        g2 = json.loads((tmp_path / f"child-{label}.gguf.gold.json").read_text(encoding="utf-8"))
        r2 = json.loads((tmp_path / f"child-{label}.gguf.derive.json").read_text(encoding="utf-8"))
        assert len(g2["rows"]) == len(gold_doc["rows"])
        for a, b in zip(gold_doc["rows"], g2["rows"]):
            assert abs(a["a"]["kld"] - b["a"]["kld"]) < 1e-15
            assert a["a"]["agree"] == b["a"]["agree"]
        assert r2["reference"]["logits_format"] == exp_fmt
        assert r2["reference"]["producer_mode"] == exp_mode

    # Verify --compare and --card-from work with the generated gold/derive JSON.
    out_cmp = tmp_path / "child-cmp.gguf"
    rc3, so3, se3 = run_derive(
        runner_bin,
        fx["parent"],
        out_cmp,
        fx["corpus"],
        f"logits:{fx['man_npy']}",
        "--compare",
        f"Baseline={tmp_path / 'child-npy.gguf.gold.json'}",
    )
    assert rc3 == 0, f"derive --compare failed:\n{so3}\n{se3}"
    card_out = tmp_path / "rebuilt.README.md"
    p_card = subprocess.run(
        [
            str(runner_bin),
            "derive",
            "--card-from",
            str(tmp_path / "child-npy.gguf.derive.json"),
            "-o",
            str(card_out),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert p_card.returncode == 0, p_card.stderr.decode(errors="replace")
    card_txt = card_out.read_text(encoding="utf-8")
    assert "--reference logits:" in card_txt


def test_score_ref_identity_self_check(runner_bin, inkling2_fixture):
    """Scoring the unquantized F32 fixture against its own pinned logits yields 0 KL."""
    fx = inkling2_fixture
    p = subprocess.run(
        [
            str(runner_bin),
            "-m",
            str(fx["parent"]),
            "--score",
            "--score-ref",
            str(fx["man_npy"]),
            "-f",
            str(fx["corpus"]),
            "-c",
            str(CTX),
            "-t",
            "2",
            "--gpu",
            "off",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    doc = json.loads(p.stdout.decode("utf-8"))
    assert doc["ref_kl_method"] == "full-vocab-f64"
    assert len(doc["ref_rows"]) == fx["n_scored"]
    for row in doc["ref_rows"]:
        assert abs(row["kld"]) < 1e-12
        assert row["agree"] is True
        assert row["max_abs"] == 0.0


def test_refusals_fail_closed(runner_bin, inkling2_fixture, tmp_path):
    """Fail-closed refusals: sha256, corpus, vocab, window, token/n_special, NaN/Inf, path traversal."""
    fx = inkling2_fixture

    # 1. Single flipped byte in logits -> sha256 mismatch refused BEFORE rewrite.
    d1 = tmp_path / "bad_sha"
    d1.mkdir()
    bad_raw = bytearray(fx["ref_raw"])
    bad_raw[17] ^= 0x01
    (d1 / "ref.logits.f32").write_bytes(bytes(bad_raw))
    m1 = make_manifest(
        d1,
        "ref.logits.f32",
        "raw-f32",
        fx["n_scored"],
        fx["n_vocab"],
        fx["tokens"],
        fx["n_special"],
        fx["corpus"],
        logits_sha=sha256_file(fx["work"] / "ref.logits.f32"),
    )
    out1 = d1 / "out.gguf"
    rc, _, se = run_derive(runner_bin, fx["parent"], out1, fx["corpus"], f"logits:{m1}")
    assert rc != 0
    assert "sha256" in se.lower()
    assert not out1.exists(), "must refuse before rewriting out.gguf"

    # 2. Corpus mismatch -> refused BEFORE rewrite.
    d2 = tmp_path / "bad_corpus"
    d2.mkdir()
    other_corpus = d2 / "other.txt"
    other_corpus.write_text(CORPUS_TEXT + " extra\n", encoding="utf-8")
    out2 = d2 / "out.gguf"
    rc, _, se = run_derive(
        runner_bin, fx["parent"], out2, other_corpus, f"logits:{fx['man_npy']}"
    )
    assert rc != 0
    assert "corpus" in se.lower() or "sha256" in se.lower()
    assert not out2.exists()

    # 3. Path traversal in logits.path -> refused BEFORE rewrite.
    d3 = tmp_path / "bad_path"
    d3.mkdir()
    m3_doc = json.loads(fx["man_npy"].read_text(encoding="utf-8"))
    m3_doc["logits"]["path"] = "../ink2_ref/ref.logits.npy"
    m3 = d3 / "ref.ref.json"
    m3.write_text(json.dumps(m3_doc), encoding="utf-8")
    out3 = d3 / "out.gguf"
    rc, _, se = run_derive(runner_bin, fx["parent"], out3, fx["corpus"], f"logits:{m3}")
    assert rc != 0
    assert not out3.exists()

    # 4. Window exceeding manifest rows -> refused BEFORE rewrite.
    d4 = tmp_path / "bad_win"
    d4.mkdir()
    short_rows = 10
    short_raw = fx["ref_raw"][: short_rows * fx["n_vocab"] * 4]
    (d4 / "short.f32").write_bytes(short_raw)
    m4 = make_manifest(
        d4,
        "short.f32",
        "raw-f32",
        short_rows,
        fx["n_vocab"],
        fx["tokens"],
        fx["n_special"],
        fx["corpus"],
    )
    out4 = d4 / "out.gguf"
    rc, _, se = run_derive(runner_bin, fx["parent"], out4, fx["corpus"], f"logits:{m4}")
    assert rc != 0
    assert not out4.exists()

    # 5. Wrong n_vocab -> refused in --score-ref.
    d5 = tmp_path / "bad_vocab"
    d5.mkdir()
    bad_v = fx["n_vocab"] - 1
    bad_v_raw = b"\x00" * (fx["n_scored"] * bad_v * 4)
    (d5 / "v.f32").write_bytes(bad_v_raw)
    m5 = make_manifest(
        d5, "v.f32", "raw-f32", fx["n_scored"], bad_v, fx["tokens"], fx["n_special"], None
    )
    p5 = subprocess.run(
        [
            str(runner_bin),
            "-m",
            str(fx["parent"]),
            "--score",
            "--score-ref",
            str(m5),
            "-f",
            str(fx["corpus"]),
            "-c",
            str(CTX),
            "-t",
            "2",
            "--gpu",
            "off",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert p5.returncode != 0
    assert "vocab" in p5.stderr.decode(errors="replace").lower()

    # 6. Token ID mismatch -> refused in --score-ref.
    d6 = tmp_path / "bad_tok"
    d6.mkdir()
    (d6 / "ref.logits.f32").write_bytes(fx["ref_raw"])
    bad_toks = list(fx["tokens"])
    bad_toks[5] = (bad_toks[5] + 1) % 256
    m6 = make_manifest(
        d6, "ref.logits.f32", "raw-f32", fx["n_scored"], fx["n_vocab"], bad_toks, fx["n_special"], None
    )
    p6 = subprocess.run(
        [
            str(runner_bin),
            "-m",
            str(fx["parent"]),
            "--score",
            "--score-ref",
            str(m6),
            "-f",
            str(fx["corpus"]),
            "-c",
            str(CTX),
            "-t",
            "2",
            "--gpu",
            "off",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert p6.returncode != 0
    assert "token" in p6.stderr.decode(errors="replace").lower()

    # 7. Planted NaN in a scored reference logit row -> refused in --score-ref.
    d7 = tmp_path / "nan_row"
    d7.mkdir()
    nan_raw = bytearray(fx["ref_raw"])
    pos0 = fx["win_indices"][0]
    struct.pack_into("<f", nan_raw, (pos0 * fx["n_vocab"] + 3) * 4, float("nan"))
    (d7 / "nan.f32").write_bytes(bytes(nan_raw))
    m7 = make_manifest(
        d7, "nan.f32", "raw-f32", fx["n_scored"], fx["n_vocab"], fx["tokens"], fx["n_special"], None
    )
    p7 = subprocess.run(
        [
            str(runner_bin),
            "-m",
            str(fx["parent"]),
            "--score",
            "--score-ref",
            str(m7),
            "-f",
            str(fx["corpus"]),
            "-c",
            str(CTX),
            "-t",
            "2",
            "--gpu",
            "off",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert p7.returncode != 0
    assert "finite" in p7.stderr.decode(errors="replace").lower() or "nan" in p7.stderr.decode(errors="replace").lower()

    # 8. --reference-arg paired with logits: -> refused before rewrite.
    out8 = tmp_path / "bad_refarg.gguf"
    rc, _, se = run_derive(
        runner_bin,
        fx["parent"],
        out8,
        fx["corpus"],
        f"logits:{fx['man_npy']}",
        "--reference-arg",
        "--foo",
    )
    assert rc != 0
    assert not out8.exists()

    # 9. --score-ref without --score -> refused.
    p9 = subprocess.run(
        [str(runner_bin), "-m", str(fx["parent"]), "--score-ref", str(fx["man_npy"])],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert p9.returncode != 0

    # 10. Planted +Inf in a scored reference logit row -> refused in --score-ref.
    d10 = tmp_path / "inf_row"
    d10.mkdir()
    inf_raw = bytearray(fx["ref_raw"])
    pos0 = fx["win_indices"][0]
    struct.pack_into("<f", inf_raw, (pos0 * fx["n_vocab"] + 5) * 4, float("inf"))
    (d10 / "inf.f32").write_bytes(bytes(inf_raw))
    m10 = make_manifest(
        d10, "inf.f32", "raw-f32", fx["n_scored"], fx["n_vocab"], fx["tokens"], fx["n_special"], None
    )
    p10 = subprocess.run(
        [
            str(runner_bin),
            "-m",
            str(fx["parent"]),
            "--score",
            "--score-ref",
            str(m10),
            "-f",
            str(fx["corpus"]),
            "-c",
            str(CTX),
            "-t",
            "2",
            "--gpu",
            "off",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert p10.returncode != 0
    assert "finite" in p10.stderr.decode(errors="replace").lower()

    # 11. --ram-cap below file size -> refused before scoring.
    out11 = tmp_path / "ram_cap_fail.gguf"
    rc11, _, se11 = run_derive(
        runner_bin,
        fx["parent"],
        out11,
        fx["corpus"],
        f"logits:{fx['man_npy']}",
        "--ram-cap",
        "1024",
    )
    assert rc11 != 0
    assert "ram-cap" in se11.lower()

    # 12. Symlink leaf in logits.path -> refused before rewrite on POSIX.
    if sys.platform != "win32":
        d12 = tmp_path / "sym_leaf"
        d12.mkdir()
        link = d12 / "link.logits.f32"
        link.symlink_to(fx["work"] / "ref.logits.f32")
        m12 = make_manifest(
            d12,
            "link.logits.f32",
            "raw-f32",
            fx["n_scored"],
            fx["n_vocab"],
            fx["tokens"],
            fx["n_special"],
            fx["corpus"],
            logits_sha=sha256_file(fx["work"] / "ref.logits.f32"),
        )
        out12 = d12 / "out.gguf"
        rc12, _, se12 = run_derive(
            runner_bin, fx["parent"], out12, fx["corpus"], f"logits:{m12}"
        )
        assert rc12 != 0
        assert not out12.exists()


def test_make_ref_logits_wrap_and_ram_cap_policy(inkling2_fixture, tmp_path):
    """scripts/make-ref-logits.py: wrap round-trip + explicit --ram-cap-gib policy
    (patched-stream requires explicit --mode patched-stream, never silently chosen)."""
    assert MAKE_REF.exists(), "scripts/make-ref-logits.py not present"
    fx = inkling2_fixture

    # Wrap raw-f32 + tokens JSON -> valid manifest that verifies cleanly.
    tok_json = tmp_path / "tokens.json"
    tok_json.write_text(
        json.dumps({"tokens": fx["tokens"], "n_special": fx["n_special"], "n_vocab": fx["n_vocab"]}),
        encoding="utf-8",
    )
    raw_copy = tmp_path / "lab.logits.f32"
    raw_copy.write_bytes(fx["ref_raw"])
    out_man = tmp_path / "lab.ref.json"
    p = subprocess.run(
        [
            sys.executable,
            str(MAKE_REF),
            "wrap",
            "--logits",
            str(raw_copy),
            "--tokens-json",
            str(tok_json),
            "--corpus",
            str(fx["corpus"]),
            "--repo",
            "test/inkling2-tiny",
            "--out",
            str(out_man),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    m = json.loads(out_man.read_text(encoding="utf-8"))
    assert m["schema_version"] == "xyntetik.runner.ref-logits.v1"
    assert m["logits"]["format"] == "raw-f32"
    assert m["logits"]["sha256"] == sha256_file(raw_copy)
    assert m["reference"]["producer_mode"] == "wrap-raw-f32"

    # Self-test mode dispatch under --ram-cap-gib:
    # patched fits under 80 GiB; under 32 GiB default mode refuses (does NOT silently
    # fall back to unvalidated patched-stream); explicit --mode patched-stream selects
    # patched-stream under 32 GiB; under 4 GiB even patched-stream refuses.
    for args, expect_rc, expect_word in [
        (["plan", "--mem-patched-gib", "58", "--mem-stream-gib", "9", "--ram-cap-gib", "80"], 0, "patched"),
        (["plan", "--mem-patched-gib", "58", "--mem-stream-gib", "9", "--ram-cap-gib", "32"], 2, "patched-stream"),
        (["plan", "--mem-patched-gib", "58", "--mem-stream-gib", "9", "--ram-cap-gib", "32", "--mode", "patched-stream"], 0, "patched-stream"),
        (["plan", "--mem-patched-gib", "58", "--mem-stream-gib", "9", "--ram-cap-gib", "4", "--mode", "patched-stream"], 2, "exceeds"),
    ]:
        pr = subprocess.run(
            [sys.executable, str(MAKE_REF), *args],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        assert pr.returncode == expect_rc, (args, pr.returncode, pr.stdout, pr.stderr)
        combined = (pr.stdout + pr.stderr).decode(errors="replace").lower()
        assert expect_word in combined, (args, combined)

    # Synthetic safetensors header estimation via `plan --hf`:
    fake_hf = tmp_path / "fake_hf"
    fake_hf.mkdir()
    (fake_hf / "config.json").write_text(json.dumps({"model_type": "inkling2_loop"}), encoding="utf-8")
    st_hdr = json.dumps(
        {
            "model.embed_tokens.weight": {"dtype": "BF16", "shape": [264, 64], "data_offsets": [0, 33792]},
            "model.layers.0.mlp.experts.gate_up_proj.weight": {"dtype": "U8", "shape": [4, 128, 32], "data_offsets": [33792, 50176]},
        }
    ).encode("utf-8")
    (fake_hf / "model.safetensors").write_bytes(struct.pack("<Q", len(st_hdr)) + st_hdr)
    p_est = subprocess.run(
        [sys.executable, str(MAKE_REF), "plan", "--hf", str(fake_hf), "--ram-cap-gib", "16"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert p_est.returncode == 0, p_est.stderr.decode(errors="replace")
    assert "mode=patched fits" in p_est.stdout.decode(errors="replace")
