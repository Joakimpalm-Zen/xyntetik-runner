"""Signed, chained inference receipts (--keygen, --keygen-algo, --sign-key,
--transcript-prev, --require-signed, --trust-key).

The receipt is the transcript record plus two things a third party can check
with the file bytes and a public key alone: an Ed25519 signature over every
byte before the `,"signature"` key (the chain hash included), and a `prev`
link to the previous receipt's chain hash. Gates here: the signature verifies
with the embedded key and with the standard library's Ed25519 (the external
anchor, so the runner is not grading its own homework); a flipped signature
byte, a tampered record, a foreign trust key, a broken link and a missing
signature under --require-signed are each UNVERIFIABLE (exit 3) BEFORE any
replay; a good chain verifies end to end with one exit code.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("receipts")
    b = d / "base.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(b)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return b


@pytest.fixture(scope="module")
def key(runner_bin, tmp_path_factory):
    k = tmp_path_factory.mktemp("key") / "key.json"
    p = subprocess.run([runner_bin, "--keygen", str(k)], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    out = json.loads(p.stdout)
    assert out["schema_version"] == "xyntetik.runner.signkey.v1"
    doc = json.loads(k.read_bytes())
    assert doc["public_key"] == out["public_key"] and len(doc["seed"]) == 64
    return k, out["public_key"]


def _record(runner_bin, base, out, *extra):
    p = subprocess.run(
        [runner_bin, "-m", str(base), "-p", "the runner trains the",
         "-n", "8", "--temp", "0", "--gpu", "off", "-t", "2",
         "--transcript", str(out), *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return out.read_bytes()


def _verify(runner_bin, base, rec, *extra):
    return subprocess.run(
        [runner_bin, "-m", str(base), "--verify", str(rec), "--gpu", "off",
         "-t", "2", *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)


def test_keygen_needs_no_model_and_keys_differ(runner_bin, tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    pa = json.loads(subprocess.run([runner_bin, "--keygen", str(a)], cwd=ROOT,
                                   stdout=subprocess.PIPE, check=True).stdout)
    pb = json.loads(subprocess.run([runner_bin, "--keygen", str(b)], cwd=ROOT,
                                   stdout=subprocess.PIPE, check=True).stdout)
    assert pa["public_key"] != pb["public_key"], "keys must come from OS randomness"


def test_signature_verifies_with_an_independent_ed25519(runner_bin, base, key, tmp_path):
    kpath, pub = key
    rec = tmp_path / "r.json"
    raw = _record(runner_bin, base, rec, "--sign-key", str(kpath))
    doc = json.loads(raw)
    assert doc["signature"]["algo"] == "ed25519"
    assert doc["signature"]["public_key"] == pub
    signed = raw[:raw.rindex(b',"signature"')]
    # chain hash still covers the body before the chain, as before
    body = raw[:raw.rindex(b',"chain"')]
    assert doc["chain"]["hash"] == hashlib.sha256(body).hexdigest()
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        pytest.skip("cryptography not installed: no independent Ed25519")
    Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub)).verify(
        bytes.fromhex(doc["signature"]["sig"]), signed)


def test_chain_links_and_verifies_end_to_end(runner_bin, base, key, tmp_path):
    kpath, pub = key
    r1 = tmp_path / "r1.json"
    r2 = tmp_path / "r2.json"
    d1 = json.loads(_record(runner_bin, base, r1, "--sign-key", str(kpath)))
    d2 = json.loads(_record(runner_bin, base, r2, "--sign-key", str(kpath),
                            "--transcript-prev", str(r1)))
    assert d1["chain"]["prev"] == "0" * 64
    assert d2["chain"]["prev"] == d1["chain"]["hash"]
    p = _verify(runner_bin, base, r2, "--transcript-prev", str(r1),
                "--require-signed", "--trust-key", pub)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    v = json.loads(p.stdout)
    assert v["verdict"] == "VERIFIED" and v["signed"] is True
    assert v["public_key"] == pub and v["prev"] == d1["chain"]["hash"]
    # a broken link: r1 does not follow r2
    p = _verify(runner_bin, base, r1, "--transcript-prev", str(r2))
    assert p.returncode == 3 and b"chain link broken" in p.stderr


def test_forgeries_are_unverifiable_before_replay(runner_bin, base, key, tmp_path):
    kpath, pub = key
    rec = tmp_path / "r.json"
    raw = _record(runner_bin, base, rec, "--sign-key", str(kpath))
    # a flipped signature byte
    doc = json.loads(raw)
    sig = doc["signature"]["sig"]
    bad = ("1" if sig[0] == "0" else "0") + sig[1:]
    forged = tmp_path / "forged.json"
    forged.write_bytes(raw.replace(sig.encode(), bad.encode()))
    p = _verify(runner_bin, base, forged)
    assert p.returncode == 3 and b"signature does not verify" in p.stderr
    # a body edit with the chain hash recomputed still breaks the signature
    body = raw[:raw.rindex(b',"chain"')]
    tail = raw[raw.rindex(b',"chain"'):]
    body2 = body.replace(b'"finish":"length"', b'"finish":"stop"', 1)
    assert body2 != body
    import re
    tail2 = re.sub(rb'"hash":"[0-9a-f]{64}"',
                   b'"hash":"' + hashlib.sha256(body2).hexdigest().encode() + b'"',
                   tail, count=1)
    edited = tmp_path / "edited.json"
    edited.write_bytes(body2 + tail2)
    p = _verify(runner_bin, base, edited)
    assert p.returncode == 3 and b"signature does not verify" in p.stderr
    # a foreign trust key
    p = _verify(runner_bin, base, rec, "--trust-key", "00" * 32)
    assert p.returncode == 3 and b"signed by a different key" in p.stderr


def test_unsigned_record_under_require_signed(runner_bin, base, tmp_path):
    rec = tmp_path / "plain.json"
    raw = _record(runner_bin, base, rec)
    assert b'"signature"' not in raw
    p = _verify(runner_bin, base, rec)
    assert p.returncode == 0
    v = json.loads(p.stdout)
    assert v["signed"] is False and v["public_key"] == ""
    p = _verify(runner_bin, base, rec, "--require-signed")
    assert p.returncode == 3 and b"no signature" in p.stderr
    p = _verify(runner_bin, base, rec, "--trust-key", "00" * 32)
    assert p.returncode == 3 and b"unsigned" in p.stderr


def test_sign_key_file_is_validated(runner_bin, base, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "xyntetik.runner.signkey.v1",
                               "algo": "ed25519", "seed": "00" * 32,
                               "public_key": "ff" * 32}))
    rec = tmp_path / "r.json"
    p = subprocess.run(
        [runner_bin, "-m", str(base), "-p", "hi", "-n", "2", "--temp", "0",
         "--gpu", "off", "-t", "2", "--transcript", str(rec),
         "--sign-key", str(bad)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode != 0
    assert b"cannot load signing key" in p.stderr
    assert not rec.exists(), "a partial receipt must never be installed"


# ---- ML-DSA-44 (FIPS 204) keys: the post-quantum option -------------------
# The primitive's known answers are in tests/test_mldsa.c (NIST ACVP); here the
# receipt plumbing: keygen writes a 1312-byte key, the record's signature
# object says "ml-dsa-44" with a 2420-byte signature, verify accepts it under
# --require-signed with the key given in full or as sha256:<digest>, and the
# same forgeries are UNVERIFIABLE before replay.

@pytest.fixture(scope="module")
def pqkey(runner_bin, tmp_path_factory):
    k = tmp_path_factory.mktemp("pqkey") / "key.json"
    p = subprocess.run([runner_bin, "--keygen", str(k), "--keygen-algo", "ml-dsa-44"],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    out = json.loads(p.stdout)
    assert out["algo"] == "ml-dsa-44" and len(out["public_key"]) == 1312 * 2
    doc = json.loads(k.read_bytes())
    assert doc["algo"] == "ml-dsa-44" and doc["public_key"] == out["public_key"]
    assert len(doc["seed"]) == 64
    return k, out["public_key"]


def test_keygen_rejects_unknown_algo(runner_bin, tmp_path):
    p = subprocess.run([runner_bin, "--keygen", str(tmp_path / "k.json"),
                        "--keygen-algo", "rsa"], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert p.returncode != 0 and b"unknown algorithm" in p.stderr
    assert not (tmp_path / "k.json").exists()


def test_mldsa_receipt_signs_chains_and_verifies(runner_bin, base, pqkey, tmp_path):
    kpath, pub = pqkey
    r1 = tmp_path / "r1.json"
    r2 = tmp_path / "r2.json"
    raw1 = _record(runner_bin, base, r1, "--sign-key", str(kpath))
    d1 = json.loads(raw1)
    assert d1["signature"]["algo"] == "ml-dsa-44"
    assert d1["signature"]["public_key"] == pub
    assert len(d1["signature"]["sig"]) == 2420 * 2
    body = raw1[:raw1.rindex(b',"chain"')]
    assert d1["chain"]["hash"] == hashlib.sha256(body).hexdigest()
    d2 = json.loads(_record(runner_bin, base, r2, "--sign-key", str(kpath),
                            "--transcript-prev", str(r1)))
    assert d2["chain"]["prev"] == d1["chain"]["hash"]
    # trust by the full key
    p = _verify(runner_bin, base, r2, "--transcript-prev", str(r1),
                "--require-signed", "--trust-key", pub)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    v = json.loads(p.stdout)
    assert v["verdict"] == "VERIFIED" and v["signed"] is True and v["public_key"] == pub
    # trust by the key's digest (what fits in a policy file)
    digest = hashlib.sha256(bytes.fromhex(pub)).hexdigest()
    p = _verify(runner_bin, base, r2, "--transcript-prev", str(r1),
                "--require-signed", "--trust-key", "sha256:" + digest.upper())
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    # a foreign digest
    p = _verify(runner_bin, base, r2, "--trust-key", "sha256:" + "00" * 32)
    assert p.returncode == 3 and b"signed by a different key" in p.stderr


def test_mldsa_signing_is_deterministic(runner_bin, base, pqkey, tmp_path):
    # FIPS 204's hedged signing is disabled: the same key over the same bytes
    # gives the same signature, so a re-signed receipt is byte-identical
    kpath, _ = pqkey
    sigs = set()
    for name in ("a", "b"):
        rec = tmp_path / f"{name}.json"
        raw = _record(runner_bin, base, rec, "--sign-key", str(kpath))
        doc = json.loads(raw)
        signed = raw[:raw.rindex(b',"signature"')]
        sigs.add((hashlib.sha256(signed).hexdigest(), doc["signature"]["sig"]))
    # the two records differ (timestamps), so compare per signed span: each
    # distinct span has exactly one signature, and equal spans equal signatures
    spans = {}
    for span, sig in sigs:
        assert spans.setdefault(span, sig) == sig


def test_mldsa_forgeries_are_unverifiable_before_replay(runner_bin, base, pqkey, tmp_path):
    kpath, pub = pqkey
    rec = tmp_path / "r.json"
    raw = _record(runner_bin, base, rec, "--sign-key", str(kpath))
    doc = json.loads(raw)
    sig = doc["signature"]["sig"]
    bad = ("1" if sig[0] == "0" else "0") + sig[1:]
    forged = tmp_path / "forged.json"
    forged.write_bytes(raw.replace(sig.encode(), bad.encode()))
    p = _verify(runner_bin, base, forged)
    assert p.returncode == 3 and b"signature does not verify" in p.stderr
    body = raw[:raw.rindex(b',"chain"')]
    tail = raw[raw.rindex(b',"chain"'):]
    body2 = body.replace(b'"finish":"length"', b'"finish":"stop"', 1)
    assert body2 != body
    import re
    tail2 = re.sub(rb'"hash":"[0-9a-f]{64}"',
                   b'"hash":"' + hashlib.sha256(body2).hexdigest().encode() + b'"',
                   tail, count=1)
    edited = tmp_path / "edited.json"
    edited.write_bytes(body2 + tail2)
    p = _verify(runner_bin, base, edited)
    assert p.returncode == 3 and b"signature does not verify" in p.stderr
    # an Ed25519 trust key against an ML-DSA record is a different key
    p = _verify(runner_bin, base, rec, "--trust-key", "00" * 32)
    assert p.returncode == 3 and b"signed by a different key" in p.stderr


def test_recording_over_an_existing_receipt_replaces_it(runner_bin, base, tmp_path):
    """The receipt is written to a .partial and renamed into place. On
    Windows a plain rename refuses an existing destination, so the second
    recording to the same path failed with "failed writing" and left the
    old file; the writer now uses the platform replace (MoveFileEx with
    REPLACE_EXISTING there), like --quantize does."""
    rec = tmp_path / "same.json"
    first = _record(runner_bin, base, rec, "-n", "3")
    second = _record(runner_bin, base, rec, "-n", "5")
    assert first != second
    assert rec.read_bytes() == second
    assert not (tmp_path / "same.json.partial").exists()
