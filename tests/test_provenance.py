"""GET /v1/runner/provenance: what an operator can check about a running
server without an enclave (R10.2.1).

The anchors are computed OUTSIDE the runner: the binary's and the model's
sha256 come from hashlib over the files on disk, the OMS signature is made
with openssl, and the envelope sidecar is written by the test. The route must
report the same digests, the load-time signature and envelope verdicts, and
the effective configuration -- and it must say so honestly when the model file
on disk no longer is the file that was loaded.
"""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
sys.path.insert(0, str(ROOT / "tests"))
from harness import RunnerServer  # noqa: E402
from test_oms import OPENSSL, _openssl_bundle  # noqa: E402


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("prov") / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return m


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _get(srv, path="/v1/runner/provenance"):
    try:
        with urllib.request.urlopen(srv.base_url + path, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as r:
        return r.code, json.load(r)


def _post(srv, path, payload):
    req = urllib.request.Request(srv.base_url + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as r:
        return r.code, json.load(r)


def _settled(srv, timeout=30.0):
    """Poll until the model digest is no longer being computed."""
    deadline = time.monotonic() + timeout
    while True:
        status, body = _get(srv)
        assert status == 200, body
        m = body.get("model")
        if m is None or m["sha256_state"] != "hashing":
            return body
        assert time.monotonic() < deadline, body
        time.sleep(0.05)


def test_identity_matches_the_files_on_disk(runner_bin, model):
    with RunnerServer(runner_bin, model, ctx=512, parallel=2,
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        body = _settled(srv)
        assert body["object"] == "runner.provenance"
        # the anchors: digests an outside tool computed from the same bytes
        assert body["build"]["binary_sha256"] == _sha(runner_bin)
        m = body["model"]
        assert m["sha256_state"] == "done", m
        assert m["sha256"] == _sha(model)
        assert m["size"] == model.stat().st_size
        assert m["path"] == str(model)
        # no bundle was requested or discovered: no verdict is invented
        assert m["signature"] is None
        # no sidecar: the model sits outside the certification pipeline
        assert m["envelope"]["state"] == "unclassified"
        assert body["adapter"] is None
        prof = body["profile"]
        assert prof["device"] == "cpu" and prof["gpu"] is False
        assert prof["ctx"] == 512 and prof["slots"] == 2
        assert prof["kv"] == "f16"
        # the version is the one every other route reports
        _, caps = _get(srv, "/v1/capabilities")
        assert body["version"] == caps["version"]
        # an operator reads what the record proves and what it does not
        assert "not an attestation" in body["statement"]
        srv.assert_alive()


def test_signature_and_envelope_verdicts_are_the_load_time_ones(
        runner_bin, model, tmp_path):
    if not OPENSSL:
        pytest.skip("openssl not available")
    priv, pub = tmp_path / "k.pem", tmp_path / "k.pub.pem"
    subprocess.run([OPENSSL, "ecparam", "-name", "prime256v1", "-genkey",
                    "-noout", "-out", str(priv)], check=True,
                   stderr=subprocess.DEVNULL)
    subprocess.run([OPENSSL, "pkey", "-in", str(priv), "-pubout", "-out",
                    str(pub)], check=True, stderr=subprocess.DEVNULL)
    m = tmp_path / "signed.gguf"
    shutil.copyfile(model, m)
    sig = _openssl_bundle(priv, m, tmp_path / "signed.gguf.sig")
    caps = json.loads(subprocess.run([runner_bin, "--caps"], check=True,
                                     stdout=subprocess.PIPE).stdout)
    (tmp_path / "signed.gguf.envelope.json").write_text(json.dumps({
        "schema_version": "xyntetik.runner.envelope.v1",
        "runtime": {"version": caps["version"],
                    "kernel_set": {"backend": "cpu"}},
        "verdict": "experimental",
    }))
    with RunnerServer(runner_bin, m, ctx=256, extra_args=[
            "--gpu", "off", "-t", "2", "--model-sig", str(sig),
            "--model-pubkey", str(pub), "--require-signed-model"]) as srv:
        body = _settled(srv)
        s = body["model"]["signature"]
        assert s["status"] == "verified" and s["curve"] == "P-256"
        assert s["subject_digest"] == hashlib.sha256(
            hashlib.sha256(m.read_bytes()).digest()).hexdigest()
        env = body["model"]["envelope"]
        assert env["state"] == "experimental"
        assert "experimental" in env["detail"]
        # the sidecar changing AFTER the load does not rewrite the verdict the
        # load ran under
        (tmp_path / "signed.gguf.envelope.json").unlink()
        assert _get(srv)[1]["model"]["envelope"]["state"] == "experimental"


@pytest.mark.skipif(sys.platform == "win32",
                    reason="Windows refuses to rewrite a file the server has "
                           "mapped (EINVAL), so the change this detects "
                           "cannot be made there")
def test_a_model_file_changed_after_load_is_not_vouched_for(
        runner_bin, model, tmp_path):
    m = tmp_path / "served.gguf"
    shutil.copyfile(model, m)
    with RunnerServer(runner_bin, m, ctx=256,
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        before = _settled(srv)["model"]
        assert before["sha256"] == _sha(m)
        raw = bytearray(m.read_bytes())
        raw[len(raw) // 2] ^= 1
        time.sleep(1.1)   # a later whole second, so the mtime moves
        m.write_bytes(raw)
        after = _get(srv)[1]["model"]
        assert after["sha256_state"] == "changed_since_load", after
        # the digest of bytes the server did NOT load is never published
        assert after["sha256"] is None


def test_swap_mode_reports_nothing_until_a_model_is_resident(
        runner_bin, model):
    with RunnerServer(runner_bin, f"a={model}", ctx=256,
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        body = _get(srv)[1]
        assert body["model"] is None and body["profile"] is None
        assert body["build"]["binary_sha256"] == _sha(runner_bin)
        status, _ = _post(srv, "/v1/completions",
                          {"model": "a", "prompt": "hi", "max_tokens": 1})
        assert status == 200
        body = _settled(srv)
        assert body["model"]["id"] == "a"
        assert body["model"]["sha256"] == _sha(model)
        req = urllib.request.Request(srv.base_url + "/unload", data=b"",
                                     method="POST")
        urllib.request.urlopen(req, timeout=30).read()
        assert _get(srv)[1]["model"] is None


def test_the_route_takes_no_body(runner_bin, model):
    with RunnerServer(runner_bin, model, ctx=256,
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        req = urllib.request.Request(srv.base_url + "/v1/runner/provenance",
                                     data=b"{}", method="GET",
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=30)
        assert e.value.code == 400


def test_an_adapter_is_part_of_the_identity(runner_bin, model, tmp_path):
    subprocess.run([sys.executable, ROOT / "scripts/make-test-lora.py",
                    str(model), str(tmp_path / "fx")], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    adapter = tmp_path / "fx.adapter.gguf"
    with RunnerServer(runner_bin, model, ctx=256, extra_args=[
            "--gpu", "off", "-t", "2", "--lora", str(adapter),
            "--lora-scale", "0.5"]) as srv:
        a = _settled(srv)["adapter"]
        assert a["path"] == str(adapter)
        assert a["sha256"] == _sha(adapter)
        assert a["scale"] == 0.5
