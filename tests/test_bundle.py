"""Receipt bundles (R1.2.1): `--export-bundle` puts a receipt, its model
signature and key, and the receipt key's fingerprint in one directory;
`--check-bundle` verifies it offline.

Anchors outside the runner: hashlib for every file digest and for the receipt
key's fingerprint (sha256 of the key bytes), openssl for the model key's DER
fingerprint, and the bundle's own receipt replaying through `--verify` -- the
thing a verifier takes the bundle for.
"""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_oms import OPENSSL, _openssl_bundle  # noqa: E402


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def _run(runner_bin, *args):
    return subprocess.run([str(runner_bin), *map(str, args)], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=120)


@pytest.fixture(scope="module")
def signed(runner_bin, tmp_path_factory):
    if not OPENSSL:
        pytest.skip("openssl not available")
    d = tmp_path_factory.mktemp("bundle")
    model = d / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(model)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    priv, pub = d / "p.pem", d / "p.pub.pem"
    subprocess.run([OPENSSL, "ecparam", "-name", "prime256v1", "-genkey",
                    "-noout", "-out", str(priv)], check=True,
                   stderr=subprocess.DEVNULL)
    subprocess.run([OPENSSL, "pkey", "-in", str(priv), "-pubout", "-out",
                    str(pub)], check=True, stderr=subprocess.DEVNULL)
    sig = _openssl_bundle(priv, model, d / "model.gguf.sig")
    key = d / "receipt.key"
    kg = _run(runner_bin, "--keygen", key)
    assert kg.returncode == 0, kg.stderr
    pk = json.loads(key.read_text())["public_key"]
    rec = d / "run.json"
    p = _run(runner_bin, "-m", model, "-p", "hello", "-n", "4", "--temp", "0",
             "--gpu", "off", "-t", "2", "--model-sig", sig, "--model-pubkey",
             pub, "--transcript", rec, "--sign-key", key)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return {"dir": d, "model": model, "sig": sig, "pub": pub, "key": key,
            "pk": pk, "rec": rec}


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def test_the_bundle_is_what_it_says(runner_bin, signed):
    out = signed["dir"] / "b1"
    p = _run(runner_bin, "--export-bundle", signed["rec"], "--bundle-out", out,
             "--model-sig", signed["sig"], "--model-pubkey", signed["pub"])
    assert p.returncode == 0, p.stderr
    man = json.loads((out / "bundle.json").read_text())
    assert man["schema_version"] == "xyntetik.runner.bundle.v1"
    files = {f["name"]: f["sha256"] for f in man["files"]}
    assert files == {"receipt.json": _sha(signed["rec"]),
                     "model.sig": _sha(signed["sig"]),
                     "model-pubkey.pem": _sha(signed["pub"])}
    assert (out / "receipt.json").read_bytes() == signed["rec"].read_bytes()
    sigobj = man["receipt"]["signed_by"]
    assert sigobj["public_key"] == signed["pk"]
    assert sigobj["fingerprint"] == hashlib.sha256(
        bytes.fromhex(signed["pk"])).hexdigest()
    der = subprocess.run([OPENSSL, "pkey", "-pubin", "-in", str(signed["pub"]),
                          "-outform", "DER"], check=True,
                         stdout=subprocess.PIPE).stdout
    assert man["model"]["key_fingerprint"] == hashlib.sha256(der).hexdigest()
    rec = json.loads(signed["rec"].read_text())
    assert man["model"]["sha256"] == rec["model"]["sha256"] == _sha(signed["model"])
    assert man["binary"]["sha256"] == rec["build"]["binary_sha256"]
    assert man["model"]["oms"]["status"] == "verified"
    c = _run(runner_bin, "--check-bundle", out)
    assert c.returncode == 0 and c.stdout.startswith(b"OK:"), c.stdout
    # the verifier's use of it: the bundled receipt replays against the model
    v = _run(runner_bin, "-m", signed["model"], "--verify", out / "receipt.json",
             "--trust-key", signed["pk"], "--gpu", "off", "-t", "2")
    assert v.returncode == 0 and b"VERIFIED" in v.stdout + v.stderr, v.stdout


@pytest.mark.parametrize("victim", ["receipt.json", "model.sig",
                                    "model-pubkey.pem"])
def test_a_changed_or_missing_file_fails_the_check(runner_bin, signed, victim,
                                                   tmp_path):
    out = tmp_path / "b"
    assert _run(runner_bin, "--export-bundle", signed["rec"], "--bundle-out", out,
                "--model-sig", signed["sig"], "--model-pubkey",
                signed["pub"]).returncode == 0
    raw = bytearray((out / victim).read_bytes())
    raw[len(raw) // 2] ^= 1
    (out / victim).write_bytes(raw)
    c = _run(runner_bin, "--check-bundle", out)
    assert c.returncode == 2 and victim.encode() in c.stdout, c.stdout
    (out / victim).unlink()
    c = _run(runner_bin, "--check-bundle", out)
    assert c.returncode == 2 and b"missing" in c.stdout, c.stdout


def test_a_signed_manifest_pins_its_signer(runner_bin, signed, tmp_path):
    out = tmp_path / "b"
    p = _run(runner_bin, "--export-bundle", signed["rec"], "--bundle-out", out,
             "--sign-key", signed["key"])
    assert p.returncode == 0, p.stderr
    ok = _run(runner_bin, "--check-bundle", out, "--trust-key", signed["pk"])
    assert ok.returncode == 0 and b"manifest signed" in ok.stdout, ok.stdout
    other = tmp_path / "other.key"
    _run(runner_bin, "--keygen", other)
    opk = json.loads(other.read_text())["public_key"]
    assert _run(runner_bin, "--check-bundle", out,
                "--trust-key", opk).returncode == 2
    # an edited manifest no longer verifies
    man = (out / "bundle.json").read_text()
    (out / "bundle.json").write_text(man.replace('"os":"linux"', '"os":"lunix"')
                                     if '"os":"linux"' in man
                                     else man.replace('"arch":"', '"arch":"x'))
    assert _run(runner_bin, "--check-bundle", out).returncode == 2


def test_refusals(runner_bin, signed, tmp_path):
    # a receipt whose chain does not recompute is not packaged
    broken = tmp_path / "broken.json"
    broken.write_text(signed["rec"].read_text().replace('"hello"', '"hellp"', 1))
    p = _run(runner_bin, "--export-bundle", broken, "--bundle-out", tmp_path / "x")
    assert p.returncode != 0 and b"chain" in p.stderr
    assert not (tmp_path / "x" / "bundle.json").exists()
    # an existing bundle is not overwritten
    out = tmp_path / "b"
    assert _run(runner_bin, "--export-bundle", signed["rec"],
                "--bundle-out", out).returncode == 0
    assert _run(runner_bin, "--export-bundle", signed["rec"],
                "--bundle-out", out).returncode != 0
    # no manifest: unverifiable, not bad
    assert _run(runner_bin, "--check-bundle", tmp_path).returncode == 3
    # a manifest naming a path outside the directory is refused
    man = json.loads((out / "bundle.json").read_text())
    man["files"][0]["name"] = "../broken.json"
    (out / "bundle.json").write_text(json.dumps(man))
    c = _run(runner_bin, "--check-bundle", out)
    assert c.returncode == 2 and b"plain name" in c.stdout


def test_an_export_never_overwrites_or_removes_what_was_there(runner_bin, signed,
                                                              tmp_path):
    """Exporting into the directory a receipt lives in rewrote receipt.json
    onto itself, and a manifest that could not be signed then removed it (and
    any model.sig beside it): the only copy, gone. A directory that already
    holds one of the bundle's files is refused before anything is written."""
    d = tmp_path / "run"
    d.mkdir()
    rec = d / "receipt.json"
    shutil.copy(signed["rec"], rec)
    sig = d / "model.sig"
    sig.write_bytes(b"not this bundle's")
    before = rec.read_bytes()
    p = _run(runner_bin, "--export-bundle", rec, "--bundle-out", d, "--sign-key",
             d / "no-such.key")
    assert p.returncode != 0
    assert b"already exists" in p.stderr, p.stderr
    assert rec.read_bytes() == before
    assert sig.read_bytes() == b"not this bundle's"
    assert not (d / "bundle.json").exists()
