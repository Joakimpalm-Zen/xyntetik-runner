"""Evidence packs (R1.6.1): `--export-pack RECEIPTS_DIR` puts every served
receipt of one model and one build, the envelope manifest they name, and
opaque attachments (oversight records, a tool cassette) under one manifest,
one inference per receipt; `--check-pack` verifies it offline.

Anchors outside the runner: hashlib for every file digest, for the envelope
sidecar the receipts name and for the receipt key's fingerprint; the chain
read straight from the receipts' own `chain` objects; and a packed receipt
replaying VERIFIED through `--verify`, which is what a reviewer takes the
pack for.
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
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402


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


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _post(srv, path, payload):
    req = urllib.request.Request(srv.base_url + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


@pytest.fixture(scope="module")
def served(runner_bin, tmp_path_factory):
    """Three served turns, signed, under an envelope sidecar."""
    d = tmp_path_factory.mktemp("pack")
    model = d / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(model)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    caps = json.loads(_run(runner_bin, "--caps").stdout)
    side = d / "model.gguf.envelope.json"
    side.write_text(json.dumps({
        "schema_version": "xyntetik.runner.envelope.v1",
        "runtime": {"version": caps["version"], "kernel_set": {"backend": "cpu"}},
        "verdict": "experimental"}))
    key = d / "k.json"
    assert _run(runner_bin, "--keygen", key).returncode == 0
    pk = json.loads(key.read_text())["public_key"]
    receipts = d / "receipts"
    with RunnerServer(runner_bin, model, ctx=256, extra_args=[
            "--gpu", "off", "-t", "2", "--receipts", str(receipts),
            "--sign-key", str(key)]) as srv:
        for prompt in ("the quick brown fox", "a b c", "once upon a time"):
            _post(srv, "/v1/completions", {"prompt": prompt, "max_tokens": 5,
                                           "temperature": 0, "cache_prompt": False})
    oversight = d / "oversight-2026-10-02.json"
    oversight.write_text(json.dumps({"approved_by": "reviewer-1",
                                     "decision": "approve"}))
    files = sorted(receipts.glob("receipt-*.json"))
    assert len(files) == 3
    return {"dir": d, "model": model, "side": side, "key": key, "pk": pk,
            "receipts": receipts, "files": files, "oversight": oversight}


def _export(runner_bin, served, out, *extra):
    return _run(runner_bin, "--export-pack", served["receipts"], "--pack-out", out,
                "--pack-envelope", served["side"], "--pack-attach",
                served["oversight"], *extra)


def test_the_pack_is_what_it_says(runner_bin, served, tmp_path):
    out = tmp_path / "p"
    p = _export(runner_bin, served, out)
    assert p.returncode == 0, p.stderr
    man = json.loads((out / "pack.json").read_text())
    assert man["schema_version"] == "xyntetik.runner.evidence-pack.v1"
    files = {f["name"]: (f["role"], f["sha256"]) for f in man["files"]}
    want = {f.name: ("receipt", _sha(f)) for f in served["files"]}
    want["envelope.json"] = ("envelope", _sha(served["side"]))
    want[served["oversight"].name] = ("attachment", _sha(served["oversight"]))
    assert files == want
    recs = [json.loads(f.read_text()) for f in served["files"]]
    # one inference per receipt, its id the receipt's chain hash, in order
    assert [i["id"] for i in man["inferences"]] == [r["chain"]["hash"] for r in recs]
    assert [i["request_id"] for i in man["inferences"]] == \
        [r["serve"]["request_id"] for r in recs]
    fp = hashlib.sha256(bytes.fromhex(served["pk"])).hexdigest()
    assert {i["signer_fingerprint"] for i in man["inferences"]} == {fp}
    # the chain, read from the receipts themselves
    assert all(recs[k]["chain"]["prev"] == recs[k - 1]["chain"]["hash"]
               for k in range(1, 3))
    assert man["segment"] == {"receipts": 3, "contiguous": True, "breaks": 0,
                            "first_prev": recs[0]["chain"]["prev"],
                            "last_hash": recs[2]["chain"]["hash"]}
    # the envelope the receipts name is the one in the pack
    assert {r["envelope"]["manifest_sha256"] for r in recs} == {_sha(served["side"])}
    assert man["envelope"] == {"file": "envelope.json",
                               "sha256": _sha(served["side"]),
                               "verdict": "experimental"}
    assert man["model"]["sha256"] == _sha(served["model"])
    c = _run(runner_bin, "--check-pack", out)
    assert c.returncode == 0 and c.stdout.startswith(b"OK:"), c.stdout
    assert b"3 inferences, chain contiguous" in c.stdout, c.stdout
    # the reviewer's use of it: a packed receipt replays against the model
    v = _run(runner_bin, "-m", served["model"], "--verify",
             out / served["files"][1].name, "--trust-key", served["pk"],
             "--gpu", "off", "-t", "2", "-c", "256")
    assert v.returncode == 0 and b"VERIFIED" in v.stdout + v.stderr, v.stdout


@pytest.mark.parametrize("victim", ["receipt", "envelope.json", "attachment"])
def test_a_changed_file_fails_the_check(runner_bin, served, tmp_path, victim):
    out = tmp_path / "p"
    assert _export(runner_bin, served, out).returncode == 0
    name = {"receipt": served["files"][0].name,
            "attachment": served["oversight"].name}.get(victim, victim)
    raw = bytearray((out / name).read_bytes())
    raw[len(raw) // 2] ^= 1
    (out / name).write_bytes(raw)
    c = _run(runner_bin, "--check-pack", out)
    assert c.returncode == 2 and name.encode() in c.stdout, c.stdout


def test_a_receipt_added_to_the_directory_is_caught(runner_bin, served, tmp_path):
    out = tmp_path / "p"
    assert _export(runner_bin, served, out).returncode == 0
    shutil.copy(served["files"][0], out / "receipt-9999999999.json")
    c = _run(runner_bin, "--check-pack", out)
    assert c.returncode == 2 and b"not in the manifest" in c.stdout, c.stdout


def test_a_gap_in_the_chain_is_stated_and_checked(runner_bin, served, tmp_path):
    """A pack made from a directory a receipt was taken out of says so, and a
    manifest edited to claim the chain is whole no longer checks."""
    src = tmp_path / "gappy"
    src.mkdir()
    for f in (served["files"][0], served["files"][2]):
        shutil.copy(f, src / f.name)
    out = tmp_path / "p"
    p = _run(runner_bin, "--export-pack", src, "--pack-out", out)
    assert p.returncode == 0, p.stderr
    man = json.loads((out / "pack.json").read_text())
    assert man["segment"]["contiguous"] is False and man["segment"]["breaks"] == 1
    c = _run(runner_bin, "--check-pack", out)
    assert c.returncode == 0 and b"with breaks" in c.stdout, c.stdout
    text = (out / "pack.json").read_text()
    (out / "pack.json").write_text(text.replace('"contiguous":false,"breaks":1',
                                                '"contiguous":true,"breaks":0'))
    c = _run(runner_bin, "--check-pack", out)
    assert c.returncode == 2 and b"chain" in c.stdout, c.stdout


def test_a_signed_manifest_pins_its_signer(runner_bin, served, tmp_path):
    out = tmp_path / "p"
    assert _export(runner_bin, served, out, "--sign-key", served["key"]).returncode == 0
    ok = _run(runner_bin, "--check-pack", out, "--trust-key", served["pk"])
    assert ok.returncode == 0 and b"manifest signed" in ok.stdout, ok.stdout
    other = tmp_path / "o.json"
    _run(runner_bin, "--keygen", other)
    opk = json.loads(other.read_text())["public_key"]
    assert _run(runner_bin, "--check-pack", out, "--trust-key", opk).returncode == 2


def test_refusals(runner_bin, served, tmp_path):
    # an envelope the receipts did not run under
    wrong = tmp_path / "wrong.envelope.json"
    wrong.write_text(served["side"].read_text().replace("experimental", "certified"))
    p = _run(runner_bin, "--export-pack", served["receipts"], "--pack-out",
             tmp_path / "a", "--pack-envelope", wrong)
    assert p.returncode == 1 and b"not the envelope manifest" in p.stderr, p.stderr
    # an attachment that would shadow a member
    clash = tmp_path / "envelope.json"
    clash.write_text("{}")
    p = _run(runner_bin, "--export-pack", served["receipts"], "--pack-out",
             tmp_path / "b", "--pack-attach", clash)
    assert p.returncode == 1 and b"reserved" in p.stderr, p.stderr
    # a receipt whose chain no longer recomputes
    bad = tmp_path / "bad"
    bad.mkdir()
    text = served["files"][0].read_text()
    assert '"prompt_reuse":"none"' in text
    (bad / served["files"][0].name).write_text(
        text.replace('"prompt_reuse":"none"', '"prompt_reuse":"nonf"'))
    p = _run(runner_bin, "--export-pack", bad, "--pack-out", tmp_path / "c")
    assert p.returncode == 1 and b"does not recompute" in p.stderr, p.stderr
    # never into a directory that already holds a pack
    out = tmp_path / "d"
    assert _export(runner_bin, served, out).returncode == 0
    p = _export(runner_bin, served, out)
    assert p.returncode == 1 and b"already holds" in p.stderr, p.stderr
    # an empty directory is not a pack
    (tmp_path / "e").mkdir()
    c = _run(runner_bin, "--check-pack", tmp_path / "e")
    assert c.returncode == 3, c.stdout
