"""OpenSSF Model Signing (OMS) verification at load (--model-sig,
--model-pubkey, --require-signed-model).

Two bundle sources: the reference implementation (`model_signing`, the
sigstore model-transparency package) when it is importable, which is the
cross-implementation anchor, and an openssl-built bundle that follows the
spec's key method (DSSE PAE over an in-toto Statement v1, ECDSA P-256 with
SHA-256, DER signature) so CI without the package still runs the gates. Each
bundle must verify; a tampered model, a foreign key, a tampered payload and a
missing bundle under --require-signed-model must refuse the load; the
receipt records the verdict.
"""
import base64
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer

OPENSSL = shutil.which("openssl")


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    d = tmp_path_factory.mktemp("oms")
    m = d / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(m)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return m


@pytest.fixture(scope="module")
def p256(tmp_path_factory):
    if not OPENSSL:
        pytest.skip("openssl not available")
    d = tmp_path_factory.mktemp("key")
    priv, pub = d / "p256.pem", d / "p256.pub.pem"
    subprocess.run([OPENSSL, "ecparam", "-name", "prime256v1", "-genkey",
                    "-noout", "-out", str(priv)], check=True,
                   stderr=subprocess.DEVNULL)
    subprocess.run([OPENSSL, "pkey", "-in", str(priv), "-pubout", "-out",
                    str(pub)], check=True, stderr=subprocess.DEVNULL)
    return priv, pub


def _openssl_bundle(priv, model, out, resource_name=".", digest=None):
    """A key-method OMS bundle per spec v1.0 sections 5-6, signed with openssl."""
    d = digest or hashlib.sha256(model.read_bytes()).hexdigest()
    root = hashlib.sha256(bytes.fromhex(d)).hexdigest()
    stmt = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": model.name, "digest": {"sha256": root}}],
        "predicateType": "https://model_signing/signature/v1.0",
        "predicate": {
            "resources": [{"name": resource_name, "digest": d, "algorithm": "sha256"}],
            "serialization": {"method": "files", "hash_type": "sha256",
                              "allow_symlinks": False},
        },
    }
    payload = json.dumps(stmt, indent=2).encode()
    ptype = b"application/vnd.in-toto+json"
    pae = b"DSSEv1 %d %s %d %s" % (len(ptype), ptype, len(payload), payload)
    pae_file = out.with_suffix(".pae")
    pae_file.write_bytes(pae)
    sig = subprocess.run([OPENSSL, "dgst", "-sha256", "-sign", str(priv),
                          str(pae_file)], check=True, stdout=subprocess.PIPE).stdout
    bundle = {
        "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
        "verificationMaterial": {"publicKey": {"hint": "openssl-test"}, "tlogEntries": []},
        "dsseEnvelope": {
            "payload": base64.b64encode(payload).decode(),
            "payloadType": ptype.decode(),
            "signatures": [{"sig": base64.b64encode(sig).decode(), "keyid": ""}],
        },
    }
    out.write_text(json.dumps(bundle))
    return out


def _load(runner_bin, model, *extra, transcript=None):
    cmd = [runner_bin, "-m", str(model), "-p", "hi", "-n", "2", "--temp", "0",
           "--gpu", "off", "-t", "2", *extra]
    if transcript:
        cmd += ["--transcript", str(transcript)]
    return subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=120)


def _request(server, path, payload=None):
    data = b"" if path == "/unload" else json.dumps(payload).encode()
    req = urllib.request.Request(server.base_url + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as r:
        return r.code, json.load(r)


def test_named_model_requires_a_signature(runner_bin, model):
    # Absolute policy anchor: an unsigned file cannot satisfy "required".
    with RunnerServer(runner_bin, f"named={model}", extra_args=[
            "--gpu", "off", "-t", "2", "--require-signed-model"]) as srv:
        status, body = _request(srv, "/v1/completions",
                                {"prompt": "hi", "max_tokens": 2})
        assert status == 409, body
        assert body["error"]["code"] == "model_signature_refused"
        srv.assert_alive()


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_signature_policy_rechecked_on_reload(runner_bin, model, p256, tmp_path,
                                            named, explicit):
    # The independent OpenSSL signature must cover the file being loaded,
    # and a missing signature or changed model must never pass on reload.
    m = tmp_path / "served.gguf"
    shutil.copyfile(model, m)
    sig = _openssl_bundle(p256[0], m, tmp_path / "served.gguf.sig")
    signed = sig.read_bytes()
    flags = ["--gpu", "off", "-t", "2", "--require-signed-model",
             "--model-pubkey", str(p256[1])]
    if explicit:
        flags += ["--model-sig", str(sig)]
    with RunnerServer(runner_bin, f"named={m}" if named else m,
                      extra_args=flags) as srv:
        payload = {"prompt": "hi", "max_tokens": 2, "temperature": 0}
        assert _request(srv, "/v1/completions", payload)[0] == 200
        assert _request(srv, "/unload")[0] == 200
        sig.unlink()
        assert _request(srv, "/v1/completions", payload)[0] == 409
        sig.write_bytes(signed)
        assert _request(srv, "/v1/completions", payload)[0] == 200
        assert _request(srv, "/unload")[0] == 200
        raw = bytearray(m.read_bytes())
        raw[-1] ^= 1
        m.write_bytes(raw)
        assert _request(srv, "/v1/completions", payload)[0] == 409
        srv.assert_alive()


def test_openssl_bundle_verifies_and_gates(runner_bin, model, p256, tmp_path):
    priv, pub = p256
    sig = _openssl_bundle(priv, model, tmp_path / "model.sig")
    rec = tmp_path / "r.json"
    p = _load(runner_bin, model, "--model-sig", str(sig), "--model-pubkey",
              str(pub), "--require-signed-model", transcript=rec)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert b"model signature: verified" in p.stderr
    ms = json.loads(rec.read_bytes())["model_signature"]
    assert ms["status"] == "verified" and ms["curve"] == "P-256"
    assert ms["hash"] == "sha256" and ms["resource"] == "."
    assert ms["subject_digest"] == hashlib.sha256(
        hashlib.sha256(model.read_bytes()).digest()).hexdigest()
    # a foreign key: refused
    other = tmp_path / "other.pem"
    subprocess.run([OPENSSL, "ecparam", "-name", "prime256v1", "-genkey",
                    "-noout", "-out", str(other)], check=True,
                   stderr=subprocess.DEVNULL)
    other_pub = tmp_path / "other.pub.pem"
    subprocess.run([OPENSSL, "pkey", "-in", str(other), "-pubout", "-out",
                    str(other_pub)], check=True, stderr=subprocess.DEVNULL)
    p = _load(runner_bin, model, "--model-sig", str(sig), "--model-pubkey",
              str(other_pub))
    assert p.returncode != 0 and b"does not verify" in p.stderr
    # a tampered model (one byte): refused, the signature itself still good
    bad = tmp_path / "bad.gguf"
    raw = bytearray(model.read_bytes())
    raw[len(raw) // 2] ^= 1
    bad.write_bytes(raw)
    p = _load(runner_bin, bad, "--model-sig", str(sig), "--model-pubkey", str(pub))
    assert p.returncode != 0 and b"digest differs" in p.stderr
    # a tampered payload (manifest digest edited, signature stale): refused
    b = json.loads(sig.read_text())
    payload = json.loads(base64.b64decode(b["dsseEnvelope"]["payload"]))
    payload["predicate"]["resources"][0]["digest"] = hashlib.sha256(raw).hexdigest()
    b["dsseEnvelope"]["payload"] = base64.b64encode(
        json.dumps(payload, indent=2).encode()).decode()
    forged = tmp_path / "forged.sig"
    forged.write_text(json.dumps(b))
    p = _load(runner_bin, bad, "--model-sig", str(forged), "--model-pubkey", str(pub))
    assert p.returncode != 0 and b"does not verify" in p.stderr


def test_auto_detected_bundle_and_missing_key(runner_bin, model, p256, tmp_path):
    priv, pub = p256
    m = tmp_path / "auto.gguf"
    shutil.copy(model, m)
    _openssl_bundle(priv, m, tmp_path / "auto.gguf.sig")
    # no key: reported, not refused (nothing was asked for)
    p = _load(runner_bin, m)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert b"model signature: unverified (no trusted public key" in p.stderr
    # the key alone is enough to turn the auto-detected bundle into a gate
    p = _load(runner_bin, m, "--model-pubkey", str(pub))
    assert p.returncode == 0 and b"model signature: verified" in p.stderr
    # --require-signed-model without any bundle
    p = _load(runner_bin, model, "--require-signed-model")
    assert p.returncode != 0 and b"no signature bundle" in p.stderr


def test_unsupported_methods_never_verify(runner_bin, model, p256, tmp_path):
    priv, pub = p256
    sig = _openssl_bundle(priv, model, tmp_path / "model.sig")
    b = json.loads(sig.read_text())
    b["verificationMaterial"] = {"certificate": {"rawBytes": "AAAA"}}
    cert = tmp_path / "cert.sig"
    cert.write_text(json.dumps(b))
    p = _load(runner_bin, model, "--model-sig", str(cert), "--model-pubkey", str(pub))
    assert p.returncode != 0 and b"key method only" in p.stderr


def test_reference_tool_bundle(runner_bin, model, p256, tmp_path):
    """The cross-implementation anchor: a bundle written by the reference
    signer verifies here, byte for byte as it was written."""
    ms = shutil.which("model_signing") or str(pathlib.Path(sys.executable).with_name("model_signing"))
    if not pathlib.Path(ms).exists():
        pytest.skip("model_signing (sigstore model-transparency) not installed")
    priv, pub = p256
    m = tmp_path / "ref.gguf"
    shutil.copy(model, m)
    sig = tmp_path / "ref.gguf.sig"
    p = subprocess.run([ms, "sign", "key", "--private_key", str(priv),
                        "--signature", str(sig), str(m)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    if p.returncode != 0:
        pytest.skip("reference signer failed: " + p.stderr.decode(errors="replace")[-200:])
    p = _load(runner_bin, m, "--model-pubkey", str(pub), "--require-signed-model")
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert b"model signature: verified" in p.stderr
