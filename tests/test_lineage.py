"""--lineage (R17.1): the chain of provenance records from a file back to its origin.

A rewrite (--quantize), a training run (--train) and a merge (--merge-lora)
each write a record beside their output; every record names its inputs by
sha256 and links the record that produced each input; --sign-key signs the
record in place. The walker re-checks every link from the records alone:
a child's input hash must be its parent's output hash, the parent record
must be byte-for-byte the one the child recorded, a file still on disk must
match its record, and a signature must verify. Exit 0 all verified and
signed, 1 consistent but not all signed, 2 broken or no record.
"""
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    if not (ROOT / "test.gguf").exists():
        pytest.skip("run `make test.gguf` first")
    return exe


def _run(exe, args, cwd):
    return subprocess.run([str(exe), *map(str, args)], cwd=cwd, capture_output=True,
                          text=True, timeout=300)


def _chain(exe, d, sign=True):
    """base -> train an adapter -> merge it -> quantize the merged file."""
    shutil.copy(ROOT / "test.gguf", d / "base.gguf")
    (d / "data.txt").write_text("the cat sat on the mat.\nthe dog sat on the log.\n")
    key = []
    if sign:
        assert _run(exe, ["--keygen", "key.json"], d).returncode == 0
        key = ["--sign-key", "key.json"]
    steps = [
        ["-m", "base.gguf", "--train", "data.txt", "--train-steps", "1",
         "--train-out", "a.gguf", "--gpu", "off", *key],
        ["-m", "base.gguf", "--lora", "a.gguf", "--merge-lora", "merged.gguf", *key],
        ["-m", "merged.gguf", "--quantize", "final.gguf", "--quant", "q8_0", *key],
    ]
    for s in steps:
        p = _run(exe, s, d)
        assert p.returncode == 0, (s, p.stderr)
    for rec in ("a.gguf.train.json", "merged.gguf.merge.json", "final.gguf.quant.json"):
        assert (d / rec).exists(), rec


def host_paths(obj, roots):
    """Every string in a record that names a place on this machine: an
    absolute path, or anything under one of the test's directories."""
    found = []

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str) and (v.startswith("/") or v[1:3] in (":\\", ":/")
                                     or any(str(r) in v for r in roots)):
            found.append(v)
    walk(obj)
    return found


def test_records_name_files_by_base_name_and_walk_from_anywhere(runner_bin, tmp_path):
    """A record is published beside its file, so it names files by base name
    and sha256, never by where they sat on the machine that made them; the
    walker finds a named file beside the record that names it, so the chain
    still verifies, files included, after the folder moves."""
    work, elsewhere = tmp_path / "work", tmp_path / "elsewhere"
    work.mkdir(); elsewhere.mkdir()
    shutil.copy(ROOT / "test.gguf", work / "base.gguf")
    (work / "data.txt").write_text("the cat sat on the mat.\nthe dog sat on the log.\n")
    w = lambda n: str(work / n)
    assert _run(runner_bin, ["--keygen", w("key.json")], elsewhere).returncode == 0
    key = ["--sign-key", w("key.json")]
    for st in (["-m", w("base.gguf"), "--train", w("data.txt"), "--train-steps", "1",
                "--train-out", w("a.gguf"), "--gpu", "off", *key],
               ["-m", w("base.gguf"), "--lora", w("a.gguf"), "--merge-lora", w("merged.gguf"), *key],
               ["-m", w("merged.gguf"), "--quantize", w("final.gguf"), "--quant", "q8_0", *key]):
        p = _run(runner_bin, st, elsewhere)
        assert p.returncode == 0, (st, p.stderr)
    for rec in ("a.gguf.train.json", "merged.gguf.merge.json", "final.gguf.quant.json"):
        assert host_paths(json.loads((work / rec).read_text()), [tmp_path]) == [], rec
    moved = tmp_path / "moved"
    shutil.move(str(work), str(moved))
    p = _run(runner_bin, ["--lineage", str(moved / "final.gguf")], elsewhere)
    assert p.returncode == 0, p.stdout
    assert p.stdout.count("VERIFIED (signed") == 3, p.stdout
    assert "output file not present" not in p.stdout, p.stdout   # found beside its record
    b = bytearray((moved / "merged.gguf").read_bytes())
    b[-1] ^= 1
    (moved / "merged.gguf").write_bytes(bytes(b))
    p = _run(runner_bin, ["--lineage", str(moved / "final.gguf")], elsewhere)
    assert p.returncode == 2 and "no longer matches its record" in p.stdout, p.stdout


def test_a_signed_chain_verifies_back_to_its_origins(runner_bin, tmp_path):
    _chain(runner_bin, tmp_path)
    p = _run(runner_bin, ["--lineage", "final.gguf"], tmp_path)
    assert p.returncode == 0, p.stdout
    out = p.stdout
    assert out.count("VERIFIED (signed") == 3, out
    for step in ("quantize ->", "merge ->", "train ->"):
        assert step in out, out
    # the base model and the training data are where the chain starts
    assert out.count("ORIGIN") == 3, out          # base twice (merge, train), data once
    assert "RESULT: VERIFIED" in out
    # the quantize record names its input, output and building binary
    q = json.loads((tmp_path / "final.gguf.quant.json").read_text())
    assert q["schema_version"] == "xyntetik.runner.quant.v1"
    assert q["target"] == "q8_0" and len(q["build"]["binary_sha256"]) == 64
    assert q["base"]["record"]["path"].endswith("merged.gguf.merge.json")


def test_an_unsigned_chain_is_consistent_but_says_so(runner_bin, tmp_path):
    _chain(runner_bin, tmp_path, sign=False)
    p = _run(runner_bin, ["--lineage", "final.gguf"], tmp_path)
    assert p.returncode == 1, p.stdout
    assert p.stdout.count("UNSIGNED") == 3 and "BROKEN" not in p.stdout


def test_a_record_edited_after_the_fact_breaks_the_chain(runner_bin, tmp_path):
    _chain(runner_bin, tmp_path)
    rec = tmp_path / "merged.gguf.merge.json"
    rec.write_text(rec.read_text().replace('"lora_scale":1', '"lora_scale":2'))
    p = _run(runner_bin, ["--lineage", "final.gguf"], tmp_path)
    assert p.returncode == 2, p.stdout
    assert "changed after the next step recorded it" in p.stdout


def test_a_swapped_file_breaks_the_chain(runner_bin, tmp_path):
    _chain(runner_bin, tmp_path)
    final = tmp_path / "final.gguf"
    b = bytearray(final.read_bytes())
    b[-1] ^= 1
    final.write_bytes(bytes(b))
    p = _run(runner_bin, ["--lineage", "final.gguf"], tmp_path)
    assert p.returncode == 2, p.stdout
    assert "BROKEN" in p.stdout


def test_another_trusted_key_breaks_the_chain(runner_bin, tmp_path):
    _chain(runner_bin, tmp_path)
    other = _run(runner_bin, ["--keygen", "other.json"], tmp_path)
    assert other.returncode == 0
    pub = json.loads((tmp_path / "other.json").read_text())["public_key"]
    p = _run(runner_bin, ["--lineage", "final.gguf", "--trust-key", pub], tmp_path)
    assert p.returncode == 2, p.stdout
    assert "untrusted signature" in p.stdout


def test_a_file_with_no_record_is_refused(runner_bin, tmp_path):
    shutil.copy(ROOT / "test.gguf", tmp_path / "plain.gguf")
    p = _run(runner_bin, ["--lineage", "plain.gguf"], tmp_path)
    assert p.returncode == 2 and "NO RECORD" in p.stdout, p.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="no endless device file to cite")
@pytest.mark.parametrize("key", ["base", "output"])
def test_a_record_that_cites_a_device_file_does_not_hang_the_walk(runner_bin, tmp_path, key):
    """A record is someone else's file, and the walker hashes what it names.
    One that cited /dev/zero (an older record's full path is tried as it is
    written) was hashed with no end: the verifier never answered. The walker
    reads regular files only, so the citation is a file that is not there."""
    rec = {"schema_version": "xyntetik.runner.quant.v1", "runner": "x",
           "output": {"path": "out.gguf", "sha256": "0" * 64},
           "base": {"path": "base.gguf", "sha256": "1" * 64}}
    rec[key]["path"] = "/dev/zero"
    path = tmp_path / "out.gguf.record.json"
    path.write_text(json.dumps(rec))
    p = subprocess.run([str(runner_bin), "--lineage", str(path)], cwd=tmp_path,
                       capture_output=True, text=True, timeout=30)
    assert "RESULT:" in p.stdout, p.stdout + p.stderr
    assert p.returncode in (1, 2), p.stdout + p.stderr


def test_the_records_alone_verify_after_the_files_are_gone(runner_bin, tmp_path):
    work, audit = tmp_path / "work", tmp_path / "audit"
    work.mkdir(); audit.mkdir()
    _chain(runner_bin, work)
    for rec in work.glob("*.json"):
        shutil.copy(rec, audit / rec.name)
    shutil.rmtree(work)            # the models and the original paths are gone
    p = _run(runner_bin, ["--lineage", "final.gguf.quant.json"], audit)
    assert p.returncode == 0, p.stdout
    assert p.stdout.count("VERIFIED (signed") == 3
    assert "output file not present" in p.stdout


def test_an_answer_walks_back_from_its_receipt(runner_bin, tmp_path):
    # a served or one-shot answer's receipt names the model and links the
    # record of the step that made it: the chain runs from the answer back to
    # the base model and the training data
    _chain(runner_bin, tmp_path)
    p = _run(runner_bin, ["-m", "final.gguf", "-p", "hello", "-n", "4", "--temp", "0",
                          "--gpu", "off", "--transcript", "run.json",
                          "--sign-key", "key.json"], tmp_path)
    assert p.returncode == 0, p.stderr
    rec = json.loads((tmp_path / "run.json").read_text())
    assert rec["model"]["record"]["name"] == "final.gguf.quant.json"
    v = _run(runner_bin, ["-m", "final.gguf", "--verify", "run.json"], tmp_path)
    assert v.returncode == 0, v.stdout + v.stderr      # the link does not disturb replay
    w = _run(runner_bin, ["--lineage", "run.json"], tmp_path)
    assert w.returncode == 0, w.stdout
    assert "lineage of receipt" in w.stdout and w.stdout.count("VERIFIED (signed") == 3


def test_a_receipt_for_a_downloaded_model_ends_at_its_origin(runner_bin, tmp_path):
    shutil.copy(ROOT / "test.gguf", tmp_path / "plain.gguf")
    assert _run(runner_bin, ["--keygen", "key.json"], tmp_path).returncode == 0
    p = _run(runner_bin, ["-m", "plain.gguf", "-p", "hi", "-n", "2", "--temp", "0",
                          "--gpu", "off", "--transcript", "run.json",
                          "--sign-key", "key.json"], tmp_path)
    assert p.returncode == 0, p.stderr
    assert "record" not in json.loads((tmp_path / "run.json").read_text())["model"]
    w = _run(runner_bin, ["--lineage", "run.json"], tmp_path)
    assert w.returncode == 0 and "ORIGIN" in w.stdout, w.stdout
    # an edited receipt is caught by its own signature
    r = tmp_path / "run.json"
    assert '"threads":' in r.read_text()
    r.write_text(r.read_text().replace('"threads":', '"threads": ', 1))
    w2 = _run(runner_bin, ["--lineage", "run.json"], tmp_path)
    assert w2.returncode == 2 and "BROKEN" in w2.stdout, w2.stdout


def _free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_a_servers_receipts_show_versions_timeline_and_lineage(runner_bin, tmp_path):
    # a swap set served A, A, B, A: the receipts folder tells which version
    # answered what and when (B, then back to A: a rollback), the receipt
    # chain is continuous, and each version's lineage is walked once
    import time
    import urllib.request
    _chain(runner_bin, tmp_path)
    port = _free_port()
    srv = subprocess.Popen(
        [str(runner_bin), "--serve", "--port", str(port), "--gpu", "off", "--no-tray",
         "-m", f"a={tmp_path / 'final.gguf'},b={tmp_path / 'merged.gguf'}",
         "--receipts", "rc", "--sign-key", "key.json"],
        cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
                break
            except OSError:
                time.sleep(0.2)
        for m in "aaba":
            body = json.dumps({"model": m, "messages": [{"role": "user", "content": "hi"}],
                               "max_tokens": 2, "temperature": 0}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                         body, {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=120).read()
    finally:
        srv.terminate()
        srv.wait(timeout=30)
    w = _run(runner_bin, ["--lineage", "rc"], tmp_path)
    assert w.returncode == 0, w.stdout
    out = w.stdout
    assert "4 answers" in out and "4 signed" in out and "chain continuous" in out, out
    assert "timeline: v1 x2 v2 x1 v1 x1" in out, out
    assert "v1: model sha256" in out and ": 3 answers" in out and ": 1 answers" in out
    # a receipt removed from the middle breaks the chain
    receipts = sorted((tmp_path / "rc").glob("receipt-*.json"))
    receipts[1].unlink()
    w2 = _run(runner_bin, ["--lineage", "rc"], tmp_path)
    assert w2.returncode == 2 and "chain BROKEN" in w2.stdout, w2.stdout


# A quantize record states what the rewrite did, not only what it was asked:
# a RUNNER_REQUANT_ONLY run read "target q8_0" with 2 of 20 tensors in its
# scope, and a conversion the size check declined read the same as one that
# converted everything. The record carries the tensors converted and kept,
# and the switches that scoped or forced the conversion when they were set.

GGML_Q4_0, GGML_Q8_0, GGML_Q4_K = 2, 8, 12


def _types(path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from gguf_fixture import read_header
    with open(path, "rb") as f:
        _kv, tensors, _end = read_header(f)
    return {name: t for name, _dims, t, _off in tensors}


def _quantize(exe, src, out, quant, cwd, **env):
    p = subprocess.run([str(exe), "-m", str(src), "--quantize", str(out), "--quant", quant],
                       cwd=cwd, env={**{k: v for k, v in __import__("os").environ.items()
                                       if k not in ("RUNNER_REQUANT_ONLY", "RUNNER_FORCE_REQUANT")},
                                    **env},
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    assert p.returncode == 0, p.stderr.decode(errors="replace")[-800:]
    return json.loads(pathlib.Path(str(out) + ".quant.json").read_text())


def _changed(src, out):
    a, b = _types(src), _types(out)
    return sum(1 for n in b if b[n] != a[n]), sum(1 for n in b if b[n] == a[n])


def test_a_quant_record_counts_the_tensors_it_converted(runner_bin, tmp_path):
    out = tmp_path / "q8.gguf"
    rec = _quantize(runner_bin, ROOT / "test.gguf", out, "q8_0", tmp_path)
    converted, kept = _changed(ROOT / "test.gguf", out)
    assert converted > 0 and kept > 0
    assert rec["tensors"] == {"converted": converted, "kept": kept}, rec
    assert "requant_only" not in rec and "force_requant" not in rec, rec


def test_a_scoped_requant_says_its_scope(runner_bin, tmp_path):
    out = tmp_path / "scoped.gguf"
    rec = _quantize(runner_bin, ROOT / "test.gguf", out, "q8_0", tmp_path,
                    RUNNER_REQUANT_ONLY="attn_q")
    converted, kept = _changed(ROOT / "test.gguf", out)
    assert rec["requant_only"] == "attn_q", rec
    assert rec["tensors"] == {"converted": converted, "kept": kept}, rec
    assert 0 < converted < kept, (converted, kept)


def test_a_declined_conversion_reads_as_none_and_a_forced_one_says_so(runner_bin, tmp_path):
    src = tmp_path / "wide-src.gguf"
    subprocess.run([sys.executable, ROOT / "scripts" / "make-test-model.py", "--wide", "--n-ff", "2048",
                    str(src)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    q4k = tmp_path / "q4k.gguf"
    _quantize(runner_bin, src, q4k, "q4_k", tmp_path)
    declined = _quantize(runner_bin, q4k, tmp_path / "kept.gguf", "q4_0", tmp_path)
    assert declined["tensors"]["converted"] == 0, declined
    assert "force_requant" not in declined, declined
    forced = _quantize(runner_bin, q4k, tmp_path / "forced.gguf", "q4_0", tmp_path,
                       RUNNER_FORCE_REQUANT="1")
    n_q4k = sum(1 for t in _types(q4k).values() if t == GGML_Q4_K)
    assert forced["force_requant"] is True, forced
    assert forced["tensors"]["converted"] == n_q4k > 0, (forced, n_q4k)


def test_an_in_place_rewrite_records_the_file_it_replaced(runner_bin, tmp_path):
    """`-m m.gguf --quantize m.gguf`: the rewrite installs over its input, so the
    input is hashed before anything is installed. Hashed after, the record named
    the output as its own parent and the original left the lineage."""
    import hashlib
    m = tmp_path / "m.gguf"
    shutil.copy(ROOT / "test.gguf", m)
    before = hashlib.sha256(m.read_bytes()).hexdigest()
    rec = _quantize(runner_bin, m, m, "q8_0", tmp_path)
    after = hashlib.sha256(m.read_bytes()).hexdigest()
    assert after != before and rec["tensors"]["converted"] > 0
    assert rec["base"]["sha256"] == before, rec["base"]
    assert rec["output"]["sha256"] == after, rec["output"]


def test_a_rewrite_never_overwrites_the_record_it_links(runner_bin, tmp_path):
    """A rewrite's record links the record its input came with. A second
    in-place rewrite of m.gguf would write m.gguf.quant.json over that very
    record, leaving the link pointing at itself; it is refused before anything
    is written. A first in-place rewrite links nothing and is allowed."""
    import hashlib
    m = tmp_path / "m.gguf"
    shutil.copy(ROOT / "test.gguf", m)
    _quantize(runner_bin, m, m, "q8_0", tmp_path)
    rec = tmp_path / "m.gguf.quant.json"
    keep = (hashlib.sha256(m.read_bytes()).hexdigest(), rec.read_bytes())
    p = subprocess.run([str(runner_bin), "-m", str(m), "--quantize", str(m), "--quant", "q4_0"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert p.returncode != 0, p.stderr[-600:]
    assert "m.gguf.quant.json" in p.stderr and "new file" in p.stderr, p.stderr[-600:]
    assert (hashlib.sha256(m.read_bytes()).hexdigest(), rec.read_bytes()) == keep
    # to a new file the same input rewrites fine, and links the record intact
    out = tmp_path / "m4.gguf"
    rec4 = _quantize(runner_bin, m, out, "q4_0", tmp_path)
    assert rec4["base"]["record"]["path"] == "m.gguf.quant.json"
    assert rec4["base"]["record"]["sha256"] == hashlib.sha256(rec.read_bytes()).hexdigest()
