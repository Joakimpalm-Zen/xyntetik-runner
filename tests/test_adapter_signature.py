"""OMS verification for adapters (R1.2.3; split GGUF parts are
test_oms.py::test_every_split_part_is_verified).

An adapter changes the model that serves, so it answers to the operator's
trusted key (--model-pubkey) like the model does, from its own bundle
(<adapter>.sig, or --lora-sig), and --require-signed-model requires it. The
bundles are made with openssl, outside the runner; a flipped adapter byte
must refuse the load with the signature itself intact.
"""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402
from test_oms import OPENSSL, _openssl_bundle  # noqa: E402


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    if not OPENSSL:
        pytest.skip("openssl not available")
    d = tmp_path_factory.mktemp("asig")
    base = d / "base.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(base)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, ROOT / "scripts/make-test-lora.py",
                    str(base), str(d / "fx")], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    priv, pub = d / "k.pem", d / "k.pub.pem"
    subprocess.run([OPENSSL, "ecparam", "-name", "prime256v1", "-genkey",
                    "-noout", "-out", str(priv)], check=True,
                   stderr=subprocess.DEVNULL)
    subprocess.run([OPENSSL, "pkey", "-in", str(priv), "-pubout", "-out",
                    str(pub)], check=True, stderr=subprocess.DEVNULL)
    adapter = d / "fx.adapter.gguf"
    _openssl_bundle(priv, adapter, d / "fx.adapter.gguf.sig")
    return {"d": d, "base": base, "adapter": adapter, "priv": priv, "pub": pub}


def _run(runner_bin, *args):
    return subprocess.run([str(runner_bin), *map(str, args)], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=120)


def _gen(runner_bin, fx, adapter, *extra, transcript=None):
    args = ["-m", fx["base"], "-p", "hi", "-n", "2", "--temp", "0", "--gpu",
            "off", "-t", "2", "--lora", adapter, "--model-pubkey", fx["pub"],
            *extra]
    if transcript:
        args += ["--transcript", transcript]
    return _run(runner_bin, *args)


def test_a_signed_adapter_verifies_and_the_receipt_says_so(runner_bin, fx,
                                                           tmp_path):
    rec = tmp_path / "r.json"
    p = _gen(runner_bin, fx, fx["adapter"], transcript=rec)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert b"adapter signature: verified" in p.stderr
    r = json.loads(rec.read_text())
    a = r["adapter_signature"]
    assert a["status"] == "verified" and a["curve"] == "P-256"
    assert a["subject_digest"] == hashlib.sha256(hashlib.sha256(
        fx["adapter"].read_bytes()).digest()).hexdigest()


def test_a_changed_adapter_is_refused(runner_bin, fx, tmp_path):
    bad = tmp_path / "fx.adapter.gguf"
    raw = bytearray(fx["adapter"].read_bytes())
    raw[-8] ^= 1
    bad.write_bytes(raw)
    shutil.copyfile(fx["d"] / "fx.adapter.gguf.sig", tmp_path / "fx.adapter.gguf.sig")
    p = _gen(runner_bin, fx, bad)
    assert p.returncode != 0 and b"digest differs" in p.stderr, p.stderr[-400:]
    assert b"adapter signature" in p.stderr


def test_required_signatures_cover_the_adapter(runner_bin, fx, tmp_path):
    # the model signed, the adapter not: refused under --require-signed-model
    base = tmp_path / "base.gguf"
    shutil.copyfile(fx["base"], base)
    _openssl_bundle(fx["priv"], base, tmp_path / "base.gguf.sig")
    plain = tmp_path / "plain.adapter.gguf"
    shutil.copyfile(fx["adapter"], plain)
    p = _run(runner_bin, "-m", base, "-p", "hi", "-n", "1", "--gpu", "off",
             "--lora", plain, "--model-pubkey", fx["pub"],
             "--require-signed-model")
    assert p.returncode != 0 and b"adapter signature" in p.stderr, p.stderr[-400:]
    # an explicit --lora-sig names the bundle wherever it lives
    p = _run(runner_bin, "-m", base, "-p", "hi", "-n", "1", "--gpu", "off",
             "--lora", plain, "--lora-sig", fx["d"] / "fx.adapter.gguf.sig",
             "--model-pubkey", fx["pub"], "--require-signed-model")
    assert p.returncode == 0, p.stderr.decode(errors="replace")


def test_served_adapters_are_verified(runner_bin, fx, tmp_path):
    with RunnerServer(runner_bin, fx["base"], extra_args=[
            "--gpu", "off", "-t", "2", "--model-pubkey", str(fx["pub"]),
            "--adapter", f"tuned={fx['adapter']}"]) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/runner/provenance",
                                    timeout=30) as r:
            (a,) = json.load(r)["config"]["adapters"]
        assert a["signature"]["status"] == "verified"
    with RunnerServer(runner_bin, fx["base"], extra_args=[
            "--gpu", "off", "-t", "2", "--model-pubkey", str(fx["pub"]),
            "--lora", str(fx["adapter"])]) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/runner/provenance",
                                    timeout=30) as r:
            assert json.load(r)["adapter"]["signature"]["status"] == "verified"
    # a served adapter whose bytes changed refuses the start
    bad = tmp_path / "x.adapter.gguf"
    raw = bytearray(fx["adapter"].read_bytes())
    raw[-8] ^= 1
    bad.write_bytes(raw)
    shutil.copyfile(fx["d"] / "fx.adapter.gguf.sig", tmp_path / "x.adapter.gguf.sig")
    p = _run(runner_bin, "-m", fx["base"], "--serve", "--port", "1",
             "--no-tray", "--gpu", "off", "--model-pubkey", fx["pub"],
             "--adapter", f"bad={bad}")
    assert p.returncode != 0 and b"digest differs" in p.stderr, p.stderr[-400:]
