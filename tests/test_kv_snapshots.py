"""Signed KV snapshots as memory with provenance (R1.12.1, R1.12.2).

A named context (R10.4) is a pinned KV prefix. `--kv-snapshots DIR` lets a
server write one to disk -- the KV in the prefix cache's own runner.prefix.v1
format, plus a manifest naming its sha256, the model and binary digests, the
KV type, the token count and digest, and the receipt that produced it,
chained and signed with --sign-key -- and lets a later server load it back
as a context, refusing a snapshot whose model, KV type or bytes differ. A
request built on a loaded snapshot says so in its receipt.

Anchors outside the runner: hashlib for every digest in the manifest and for
the chain, the CLI's `--check-record` for the manifest signature, and the
memory itself: a request forked from the loaded snapshot answers exactly what
the same prompt answers from a cold prefill, and its receipt replays VERIFIED.
"""
import hashlib
import os
import json
import pathlib
import struct
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

MEMORY = "you are a careful assistant. facts: the cat is grey; the dog is " \
         "brown; the owl is white; the key is under the mat."
QUESTION = MEMORY + " question: where is the key? answer:"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def fx(runner_bin, tmp_path_factory):
    d = tmp_path_factory.mktemp("kvsnap")
    model, other, m32 = d / "model.gguf", d / "other.gguf", d / "m32.gguf"
    # m32: a head width q8 KV accepts (the default fixture's keeps f16)
    for path, extra in ((model, []), (other, ["--qk-norm"]), (m32, ["--head-dim", "32"])):
        subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                        str(path), *extra], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
    key = d / "k.json"
    subprocess.run([runner_bin, "--keygen", key], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"d": d, "model": model, "other": other, "m32": m32, "key": key,
            "pk": json.loads(key.read_text())["public_key"]}


def _srv(runner_bin, model, snaps, *extra):
    return RunnerServer(runner_bin, model, ctx=512, extra_args=[
        "--gpu", "off", "-t", "2", "--kv-snapshots", str(snaps), *extra])


def _post(srv, path, payload):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as r:
        return r.code, json.load(r)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _answer(srv, **extra):
    st, body = _post(srv, "/v1/completions", {
        "prompt": QUESTION, "max_tokens": 8, "temperature": 0,
        "logprobs": 1, **extra})
    assert st == 200, body
    return body


@pytest.fixture(scope="module")
def saved(runner_bin, fx):
    """Server A: pin the memory, use it once with receipts on, snapshot it
    naming that receipt."""
    snaps, rcpt = fx["d"] / "snaps", fx["d"] / "receipts"
    with _srv(runner_bin, fx["model"], snaps, "--sign-key", str(fx["key"]),
              "--receipts", str(rcpt)) as srv:
        st, ctx = _post(srv, "/v1/runner/contexts", {"id": "mem", "prompt": MEMORY})
        assert st == 200, ctx
        used = _answer(srv, context_id="mem")
        produced_by = used["runner_telemetry"]["receipt"]["file"]
        st, snap = _post(srv, "/v1/runner/contexts/mem/snapshot",
                         {"receipt": produced_by})
        assert st == 200, snap
        # a second snapshot under the same name is refused, not overwritten
        st, again = _post(srv, "/v1/runner/contexts/mem/snapshot", {})
        assert st == 409, again
        cold = _answer(srv, cache_prompt=False)
    return {"snaps": snaps, "rcpt": rcpt, "ctx": ctx, "snap": snap,
            "produced_by": produced_by, "used": used, "cold": cold}


def test_the_manifest_is_what_it_says(fx, saved):
    snaps = saved["snaps"]
    man_raw = (snaps / "mem.kv.json").read_bytes()
    man = json.loads(man_raw)
    kv = (snaps / "mem.kv").read_bytes()
    assert man["schema_version"] == "xyntetik.runner.kv_snapshot.v1"
    assert man["name"] == "mem" and man["kv_file"] == "mem.kv"
    assert man["kv_sha256"] == _sha(kv) and man["kv_bytes"] == len(kv)
    assert kv[:16] == b"runner.prefix.v1"
    assert man["tokens"] == saved["ctx"]["tokens"]
    assert man["model"]["sha256"] == _sha(fx["model"].read_bytes())
    assert man["kv_type"] == "f16"
    # the context's token ids, digested as int32 little-endian
    toks = struct.unpack_from("<%di" % man["tokens"], kv, 16 + 4 + 8 + 4 + 8)
    assert man["token_sha256"] == _sha(struct.pack("<%di" % len(toks), *toks))
    rec = json.loads((saved["rcpt"] / saved["produced_by"]).read_text())
    assert man["producer"]["receipt"] == {"file": saved["produced_by"],
                                          "chain_hash": rec["chain"]["hash"]}
    assert list(toks) == rec["prompt"]["tokens"][:len(toks)]
    # chained and signed like any record: hashlib and the CLI verifier
    body = man_raw[:man_raw.rindex(b',"chain":')]
    assert man["chain"]["hash"] == _sha(body)
    assert man["signature"]["public_key"] == fx["pk"]
    assert saved["snap"]["kv_sha256"] == man["kv_sha256"]


def test_check_record_verifies_the_manifest(runner_bin, fx, saved):
    ok = subprocess.run([runner_bin, "--check-record", saved["snaps"] / "mem.kv.json",
                         "--trust-key", fx["pk"]], stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
    assert ok.returncode == 0, ok.stdout + ok.stderr


def test_a_loaded_snapshot_is_the_memory(runner_bin, fx, saved, tmp_path):
    rcpt = tmp_path / "r"
    with _srv(runner_bin, fx["model"], saved["snaps"], "--receipts", str(rcpt)) as srv:
        st, ctx = _post(srv, "/v1/runner/contexts", {"id": "mem2", "snapshot": "mem"})
        assert st == 200, ctx
        assert ctx["tokens"] == saved["ctx"]["tokens"]
        assert ctx["snapshot"]["kv_sha256"] == saved["snap"]["kv_sha256"]
        assert ctx["snapshot"]["signed_by"] == fx["pk"]
        warm = _answer(srv, context_id="mem2")
    # the loaded KV answers what a cold prefill answers: the same bytes, the
    # numbers to within the last digit (see test_named_contexts: a fork and a
    # cold prompt feed at different batch widths)
    assert warm["choices"][0]["text"] == saved["cold"]["choices"][0]["text"]
    assert warm["choices"][0]["logprobs"]["token_logprobs"] == pytest.approx(
        saved["cold"]["choices"][0]["logprobs"]["token_logprobs"], abs=1e-5)
    tel = warm["runner_telemetry"]
    assert tel["prompt_cached_tokens"] == saved["ctx"]["tokens"]
    # the receipt names the snapshot it started from, and replays
    rec_path = rcpt / tel["receipt"]["file"]
    rec = json.loads(rec_path.read_text())
    assert rec["serve"]["kv_snapshot"] == {
        "name": "mem", "kv_sha256": saved["snap"]["kv_sha256"],
        "manifest_chain_hash": json.loads(
            (saved["snaps"] / "mem.kv.json").read_text())["chain"]["hash"]}
    v = subprocess.run([runner_bin, "-m", fx["model"], "--verify", rec_path, "--gpu",
                        "off", "-t", "2", "-c", "512"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    assert v.returncode == 0 and b"VERIFIED" in v.stderr, v.stderr[-400:]


def _refused(srv, name):
    st, body = _post(srv, "/v1/runner/contexts", {"id": "x", "snapshot": name})
    return st, body.get("error", {})


def test_refusals(runner_bin, fx, saved, tmp_path):
    snaps = saved["snaps"]
    # another model: refused by the model digest
    with _srv(runner_bin, fx["other"], snaps) as srv:
        st, err = _refused(srv, "mem")
        assert st == 409 and err["code"] == "snapshot_model_mismatch", err
    # another KV type of the same model: refused
    with _srv(runner_bin, fx["m32"], snaps) as srv:
        assert _post(srv, "/v1/runner/contexts", {"id": "w", "prompt": MEMORY})[0] == 200
        st, snap = _post(srv, "/v1/runner/contexts/w/snapshot", {"name": "m32"})
        assert st == 200 and snap["signed"] is False, snap
    assert json.loads((snaps / "m32.kv.json").read_text())["kv_type"] == "f16"
    with _srv(runner_bin, fx["m32"], snaps, "--kv", "q8") as srv:
        st, err = _refused(srv, "m32")
        assert st == 409 and err["code"] == "snapshot_kv_type_mismatch", err
    # edited bytes: the KV file, then the manifest
    bad = tmp_path / "bad"
    bad.mkdir()
    kv = bytearray((snaps / "mem.kv").read_bytes())
    kv[len(kv) // 2] ^= 1
    (bad / "mem.kv").write_bytes(kv)
    (bad / "mem.kv.json").write_bytes((snaps / "mem.kv.json").read_bytes())
    (bad / "t.kv").write_bytes((snaps / "mem.kv").read_bytes())
    (bad / "t.kv.json").write_text((snaps / "mem.kv.json").read_text()
                                   .replace('"name":"mem"', '"name":"t"'))
    with _srv(runner_bin, fx["model"], bad) as srv:
        st, err = _refused(srv, "mem")
        assert st == 409 and err["code"] == "snapshot_digest_mismatch", err
        st, err = _refused(srv, "t")
        assert st == 409 and err["code"] == "snapshot_record_invalid", err
        st, err = _refused(srv, "absent")
        assert st == 404, err
        st, err = _refused(srv, "../mem")
        assert st == 400, err
    # snapshots are opt-in
    with RunnerServer(runner_bin, fx["model"], ctx=512,
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        st, err = _refused(srv, "mem")
        assert st == 400 and "--kv-snapshots" in err["message"], err


def test_the_snapshot_route_and_a_failed_write_say_what_happened(runner_bin, fx,
                                                                 tmp_path):
    """A snapshot that could not be written answered 404 context_not_found
    (the export's I/O error shared a value with "unknown context"), and any
    path that merely contained /snapshot was routed as one."""
    snaps = tmp_path / "ro"
    snaps.mkdir()
    with _srv(runner_bin, fx["m32"], snaps) as srv:
        assert _post(srv, "/v1/runner/contexts", {"id": "w", "prompt": MEMORY})[0] == 200
        st, body = _post(srv, "/v1/runner/contexts/w/snapshotx", {"name": "a"})
        assert st == 400, body
        st, body = _post(srv, "/v1/runner/contexts/w/snapshot/x", {"name": "a"})
        assert st == 400, body
        if sys.platform != "win32" and os.geteuid() != 0:
            snaps.chmod(0o500)
            try:
                st, body = _post(srv, "/v1/runner/contexts/w/snapshot", {"name": "a"})
            finally:
                snaps.chmod(0o700)
            assert st == 500 and body["error"]["code"] == "snapshot_write_failed", body
        st, body = _post(srv, "/v1/runner/contexts/w/snapshot", {"name": "a"})
        assert st == 200, body


def test_a_snapshot_is_loaded_only_from_a_trusted_key(runner_bin, fx, saved, tmp_path):
    """R1.12.3. A manifest was verified against the key it names, so an
    unsigned one, or one signed by anyone, loaded as well, even on a server
    started with --sign-key. A server that signs trusts its own key by
    default; --trust-key names another (the key, or sha256: of its bytes);
    with neither there is no anchor and the response says who signed."""
    import hashlib as h
    snaps = saved["snaps"]
    other = tmp_path / "other.json"
    subprocess.run([runner_bin, "--keygen", other], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    other_pk = json.loads(other.read_text())["public_key"]
    fp = "sha256:" + h.sha256(bytes.fromhex(fx["pk"])).hexdigest()

    def load(*extra, name="mem"):
        with _srv(runner_bin, fx["model"], snaps, *extra) as srv:
            st, body = _post(srv, "/v1/runner/contexts", {"id": "t", "snapshot": name})
            return st, body

    # an unsigned snapshot of the same model, made by a server with no key
    with _srv(runner_bin, fx["model"], snaps) as srv:
        assert _post(srv, "/v1/runner/contexts", {"id": "u", "prompt": MEMORY})[0] == 200
        st, snap = _post(srv, "/v1/runner/contexts/u/snapshot", {"name": "plain"})
        assert st == 200 and snap["signed"] is False, snap

    # its own key, the key by hex, the key by digest: loaded
    for extra in (["--sign-key", str(fx["key"])], ["--trust-key", fx["pk"]],
                  ["--trust-key", fp], ["--trust-key", fp.upper().replace("SHA256", "sha256")]):
        st, body = load(*extra)
        assert st == 200 and body["snapshot"]["signed_by"] == fx["pk"], (extra, body)
    # another signing key, another trusted key: refused, by name
    for extra in (["--sign-key", str(other)], ["--trust-key", other_pk],
                  ["--trust-key", "sha256:" + "0" * 64]):
        st, body = load(*extra)
        assert st == 409 and body["error"]["code"] == "snapshot_untrusted", (extra, body)
        assert "another key" in body["error"]["message"]
    # an unsigned snapshot under an anchor: refused; --trust-key outranks the
    # server's own signing key
    st, body = load("--sign-key", str(fx["key"]), name="plain")
    assert st == 409 and body["error"]["code"] == "snapshot_untrusted", body
    assert "unsigned" in body["error"]["message"]
    st, body = load("--sign-key", str(other), "--trust-key", fx["pk"])
    assert st == 200, body
    # no anchor: both load, and the response says who signed
    assert load()[0] == 200
    st, body = load(name="plain")
    assert st == 200 and body["snapshot"]["signed_by"] is None, body


def _delete(srv, path):
    req = urllib.request.Request(srv.base_url + path, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as r:
        return r.code, json.load(r)


def test_context_ids_with_colons_delete_encoded_or_bare(runner_bin, fx, tmp_path):
    """The id grammar allows ':' and the TypeScript client sends
    encodeURIComponent(id), so a DELETE arrived as %3A and the raw comparison
    found no such context (found 2026-10-05). Both spellings now work, and
    decoding does not widen the grammar: an encoded '/' is still refused."""
    with _srv(runner_bin, fx["model"], tmp_path / "snaps") as srv:
        st, _ = _post(srv, "/v1/runner/contexts", {"id": "user:42", "prompt": MEMORY})
        assert st == 200
        st, body = _delete(srv, "/v1/runner/contexts/user%3A42")
        assert st == 200 and body["id"] == "user:42" and body["deleted"]
        st, _ = _post(srv, "/v1/runner/contexts", {"id": "user:43", "prompt": MEMORY})
        assert st == 200
        st, body = _delete(srv, "/v1/runner/contexts/user:43")
        assert st == 200 and body["id"] == "user:43"
        # the snapshot route reads the same segment
        st, _ = _post(srv, "/v1/runner/contexts", {"id": "a:b", "prompt": MEMORY})
        assert st == 200
        # (a snapshot's default file name is the id, and file names exclude
        # ':', so a colon id names its snapshot explicitly)
        st, snap = _post(srv, "/v1/runner/contexts/a%3Ab/snapshot", {"name": "ab"})
        assert st == 200, snap
        for bad in ("user%2F42", "user%3", "user%zz", "%00"):
            st, err = _delete(srv, f"/v1/runner/contexts/{bad}")
            assert st == 400, (bad, err)
            assert err["error"]["code"] == "invalid_value"
