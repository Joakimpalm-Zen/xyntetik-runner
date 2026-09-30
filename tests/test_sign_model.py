"""`--sign-model`: write an OpenSSF Model Signing bundle (R1.2.5).

The runner signs with the key method (DSSE over an in-toto Statement v1,
predicate model_signing/signature/v1.0) using deterministic ECDSA (RFC 6979,
held to the RFC's own vectors in tests/test_ecdsa.c). Anchors outside the
runner: openssl makes every key and verifies every signature over the PAE
with the key it derives itself; hashlib recomputes each manifest digest and
the key hint; the reference verifier (`model_signing verify key`, sigstore
model-transparency) accepts the bundle when it is installed. The runner's own
load-time check is the last reader, not the first.
"""
import base64
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
OPENSSL = shutil.which("openssl")
MODEL_SIGNING = shutil.which("model_signing")

CURVES = {  # openssl curve name -> (dgst flag, receipt curve, receipt hash)
    "prime256v1": ("-sha256", "P-256", "sha256"),
    "secp384r1": ("-sha384", "P-384", "sha384"),
    "secp521r1": ("-sha512", "P-521", "sha512"),
}


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    if not OPENSSL:
        pytest.skip("openssl not available")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    d = tmp_path_factory.mktemp("sign")
    m = d / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return m


def _key(d, curve, form):
    """An openssl key: SEC1 ("EC PRIVATE KEY", ecparam) or PKCS#8
    ("PRIVATE KEY", genpkey), and the PEM public key openssl derives."""
    priv, pub = d / f"{curve}-{form}.pem", d / f"{curve}-{form}.pub.pem"
    if form == "sec1":
        cmd = [OPENSSL, "ecparam", "-name", curve, "-genkey", "-noout", "-out", priv]
    else:
        cmd = [OPENSSL, "genpkey", "-algorithm", "EC", "-pkeyopt",
               f"ec_paramgen_curve:{curve}", "-out", priv]
    subprocess.run(list(map(str, cmd)), check=True, stderr=subprocess.DEVNULL)
    subprocess.run([OPENSSL, "pkey", "-in", str(priv), "-pubout", "-out", str(pub)],
                   check=True, stderr=subprocess.DEVNULL)
    head = priv.read_text().splitlines()[0]
    assert head == ("-----BEGIN EC PRIVATE KEY-----" if form == "sec1"
                    else "-----BEGIN PRIVATE KEY-----"), head
    return priv, pub


def _run(runner_bin, *args):
    return subprocess.run([str(runner_bin), *map(str, args)], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=120)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _openssl_verifies(bundle, pub, dgst, tmp):
    env = bundle["dsseEnvelope"]
    payload = base64.b64decode(env["payload"])
    ptype = env["payloadType"].encode()
    pae = b"DSSEv1 %d %s %d %s" % (len(ptype), ptype, len(payload), payload)
    (tmp / "pae").write_bytes(pae)
    (tmp / "sig.der").write_bytes(base64.b64decode(env["signatures"][0]["sig"]))
    v = subprocess.run([OPENSSL, "dgst", dgst, "-verify", str(pub), "-signature",
                        str(tmp / "sig.der"), str(tmp / "pae")],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return v.returncode == 0 and b"Verified OK" in v.stdout


@pytest.mark.parametrize("curve,form", [("prime256v1", "sec1"),
                                        ("secp384r1", "pkcs8"),
                                        ("secp521r1", "sec1")])
def test_a_signed_model_verifies_everywhere(runner_bin, model, tmp_path, curve, form):
    dgst, cname, hname = CURVES[curve]
    priv, pub = _key(tmp_path, curve, form)
    m = tmp_path / "m.gguf"
    shutil.copyfile(model, m)
    p = _run(runner_bin, "--sign-model", m, "--model-key", priv)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    sig = tmp_path / "m.gguf.sig"   # beside the model, where a load finds it
    b = json.loads(sig.read_text())
    assert b["mediaType"] == "application/vnd.dev.sigstore.bundle.v0.3+json"
    # the key hint is the reference's: sha256 of the PEM public key
    assert b["verificationMaterial"]["publicKey"]["hint"] == _sha(pub.read_bytes())
    assert b["dsseEnvelope"]["payloadType"] == "application/vnd.in-toto+json"
    st = json.loads(base64.b64decode(b["dsseEnvelope"]["payload"]))
    assert st["_type"] == "https://in-toto.io/Statement/v1"
    assert st["predicateType"] == "https://model_signing/signature/v1.0"
    digest = _sha(m.read_bytes())
    assert st["predicate"]["resources"] == [
        {"algorithm": "sha256", "digest": digest, "name": "."}]
    ser = st["predicate"]["serialization"]
    assert (ser["method"], ser["hash_type"], ser["allow_symlinks"]) == ("files", "sha256", False)
    (subj,) = st["subject"]
    assert subj == {"name": "m.gguf",
                    "digest": {"sha256": _sha(bytes.fromhex(digest))}}
    # openssl, with the public key it derived, over the PAE it rebuilt
    assert _openssl_verifies(b, pub, dgst, tmp_path)
    # the runner's load-time gate
    rec = tmp_path / "r.json"
    v = _run(runner_bin, "-m", m, "-p", "hi", "-n", "1", "--gpu", "off", "-t", "2",
             "--model-pubkey", pub, "--require-signed-model", "--transcript", rec)
    assert v.returncode == 0 and b"model signature: verified" in v.stderr, v.stderr[-400:]
    ms = json.loads(rec.read_text())["model_signature"]
    assert (ms["curve"], ms["hash"]) == (cname, hname)
    # deterministic: the same key and model sign to the same bytes
    again = tmp_path / "again.sig"
    p = _run(runner_bin, "--sign-model", m, "--model-key", priv, "--model-sig", again)
    assert p.returncode == 0, p.stderr
    assert again.read_bytes() == sig.read_bytes()
    if MODEL_SIGNING:
        r = subprocess.run([MODEL_SIGNING, "verify", "key", "--public_key", str(pub),
                            "--signature", str(sig), str(m)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        assert r.returncode == 0, r.stdout.decode(errors="replace")[-400:]


def test_reference_verifier_accepts_the_bundle(runner_bin, model, tmp_path):
    """The cross-implementation anchor on its own line, so a CI without the
    package reports the skip instead of passing quietly."""
    if not MODEL_SIGNING:
        pytest.skip("model_signing (sigstore model-transparency) not installed")
    priv, pub = _key(tmp_path, "prime256v1", "sec1")
    m = tmp_path / "ref.gguf"
    shutil.copyfile(model, m)
    assert _run(runner_bin, "--sign-model", m, "--model-key", priv).returncode == 0
    r = subprocess.run([MODEL_SIGNING, "verify", "key", "--public_key", str(pub),
                        "--signature", str(tmp_path / "ref.gguf.sig"), str(m)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    assert r.returncode == 0, r.stdout.decode(errors="replace")[-400:]
    # and refuses it once the model changes
    raw = bytearray(m.read_bytes())
    raw[-3] ^= 1
    m.write_bytes(raw)
    r = subprocess.run([MODEL_SIGNING, "verify", "key", "--public_key", str(pub),
                        "--signature", str(tmp_path / "ref.gguf.sig"), str(m)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    assert r.returncode != 0


def test_a_split_model_is_signed_part_by_part(runner_bin, model, tmp_path):
    priv, pub = _key(tmp_path, "prime256v1", "pkcs8")
    d = tmp_path / "split"
    d.mkdir()
    subprocess.run([sys.executable, ROOT / "scripts/gguf-split.py", str(model),
                    str(d / "m"), "2"], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    p1, p2 = d / "m-00001-of-00002.gguf", d / "m-00002-of-00002.gguf"
    sig = tmp_path / "split.sig"
    p = _run(runner_bin, "--sign-model", p1, "--model-key", priv, "--model-sig", sig)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    b = json.loads(sig.read_text())
    st = json.loads(base64.b64decode(b["dsseEnvelope"]["payload"]))
    res = st["predicate"]["resources"]
    assert res == [{"algorithm": "sha256", "digest": _sha(f.read_bytes()), "name": f.name}
                   for f in (p1, p2)]
    root = _sha(b"".join(bytes.fromhex(r["digest"]) for r in res))
    assert st["subject"] == [{"name": "split", "digest": {"sha256": root}}]
    assert _openssl_verifies(b, pub, "-sha256", tmp_path)
    v = _run(runner_bin, "-m", p1, "-p", "hi", "-n", "1", "--gpu", "off", "-t", "2",
             "--model-sig", sig, "--model-pubkey", pub, "--require-signed-model")
    assert v.returncode == 0 and b"all 2 split part" in v.stderr, v.stderr[-400:]
    if MODEL_SIGNING:
        r = subprocess.run([MODEL_SIGNING, "verify", "key", "--public_key", str(pub),
                            "--signature", str(sig), str(d)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        assert r.returncode == 0, r.stdout.decode(errors="replace")[-400:]
    # a missing part is not signed around
    p2.rename(d / "moved.bin")
    p = _run(runner_bin, "--sign-model", p1, "--model-key", priv, "--model-sig",
             tmp_path / "partial.sig")
    assert p.returncode != 0 and b"part 2 of 2" in p.stderr, p.stderr
    assert not (tmp_path / "partial.sig").exists()


def test_refusals(runner_bin, model, tmp_path):
    priv, pub = _key(tmp_path, "prime256v1", "sec1")
    m = tmp_path / "m.gguf"
    shutil.copyfile(model, m)
    # no key named
    p = _run(runner_bin, "--sign-model", m)
    assert p.returncode != 0 and b"--model-key" in p.stderr
    # an existing bundle is not overwritten
    (tmp_path / "m.gguf.sig").write_text("keep me")
    p = _run(runner_bin, "--sign-model", m, "--model-key", priv)
    assert p.returncode != 0 and b"exists" in p.stderr
    assert (tmp_path / "m.gguf.sig").read_text() == "keep me"
    out = tmp_path / "o.sig"
    # an encrypted key: named as such, not read
    enc = tmp_path / "enc.pem"
    subprocess.run([OPENSSL, "pkey", "-in", str(priv), "-aes256", "-passout",
                    "pass:secret", "-out", str(enc)], check=True, stderr=subprocess.DEVNULL)
    p = _run(runner_bin, "--sign-model", m, "--model-key", enc, "--model-sig", out)
    assert p.returncode != 0 and b"encrypted" in p.stderr, p.stderr
    # a public key, an RSA key and a curve outside P-256/384/521
    rsa = tmp_path / "rsa.pem"
    subprocess.run([OPENSSL, "genpkey", "-algorithm", "RSA", "-pkeyopt",
                    "rsa_keygen_bits:2048", "-out", str(rsa)], check=True,
                   stderr=subprocess.DEVNULL)
    k1 = tmp_path / "k1.pem"
    subprocess.run([OPENSSL, "ecparam", "-name", "secp256k1", "-genkey", "-noout",
                    "-out", str(k1)], check=True, stderr=subprocess.DEVNULL)
    for bad in (pub, rsa, k1):
        p = _run(runner_bin, "--sign-model", m, "--model-key", bad, "--model-sig", out)
        assert p.returncode != 0 and b"P-256" in p.stderr, (bad, p.stderr)
    assert not out.exists()
