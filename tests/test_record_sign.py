"""`--sign-record` and `--check-record`: any JSON object file carries the
transcript's chain and signature, and is checked by the same code path a
notarized run is (R14.6). A tampered byte fails, a second signing is refused,
and the chain links by the previous record's hash."""
import json
import os
import pathlib
import stat
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")


def run(*args):
    return subprocess.run([str(RUNNER), *args], capture_output=True, text=True, cwd=ROOT, timeout=60)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
@pytest.mark.parametrize("algo", ["ed25519", "ml-dsa-44"])
def test_a_signing_key_is_readable_by_its_owner_alone(tmp_path, algo):
    """The key file holds the seed every signature is made from. It was
    written through fopen, so under the usual umask anyone on the machine
    could read it (0644), while the tool said "keep it private". It is
    created as a new file with mode 0600."""
    key = tmp_path / "key.json"
    old = os.umask(0o022)
    try:
        assert run("--keygen", str(key), "--keygen-algo", algo).returncode == 0
        assert stat.S_IMODE(key.stat().st_mode) == 0o600, oct(key.stat().st_mode)
        assert [p.name for p in tmp_path.iterdir()] == ["key.json"]
    finally:
        os.umask(old)


@pytest.mark.parametrize("algo", ["ed25519", "ml-dsa-44"])
def test_keygen_never_replaces_a_key(tmp_path, algo):
    """A second keygen on a path that holds a key replaced it without a word
    (exit 0). The old identity could then sign nothing more, and nothing said
    so. It is refused, the message names the file, and the file is untouched:
    its bytes and its mode (R6.68.59.1, the owner's pick)."""
    key = tmp_path / "key.json"
    assert run("--keygen", str(key), "--keygen-algo", algo).returncode == 0
    before = key.read_bytes()
    key.chmod(0o640)
    again = run("--keygen", str(key), "--keygen-algo", algo)
    assert again.returncode == 1, again.stderr
    assert str(key) in again.stderr and "never replaced" in again.stderr, again.stderr
    assert again.stdout == "", again.stdout
    assert key.read_bytes() == before
    if os.name != "nt":
        assert stat.S_IMODE(key.stat().st_mode) == 0o640
    assert [p.name for p in tmp_path.iterdir()] == ["key.json"]
    # another file that is not a key is not replaced either
    other = tmp_path / "notes.txt"
    other.write_text("mine")
    assert run("--keygen", str(other), "--keygen-algo", algo).returncode == 1
    assert other.read_text() == "mine"


def test_sign_check_tamper_and_chain(tmp_path):
    key = tmp_path / "key.json"
    assert run("--keygen", str(key)).returncode == 0
    pub = json.loads(key.read_text(encoding="utf-8"))["public_key"]
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"_type": "https://in-toto.io/Statement/v1", "subject": [{"name": "x", "digest": {"sha256": "00" * 32}}],
                             "predicateType": "https://xyntetik.com/shadow/delegation/v1", "predicate": {"verdict": "passed"}}) + "\n",
                 encoding="utf-8")
    assert run("--check-record", str(a)).stdout.startswith("UNSIGNED")
    assert run("--sign-record", str(a)).returncode == 1, "needs --sign-key"
    assert run("--sign-record", str(a), "--sign-key", str(key)).returncode == 0
    rec = json.loads(a.read_text(encoding="utf-8"))
    assert rec["chain"]["prev"] == "" and len(rec["chain"]["hash"]) == 64
    assert rec["signature"]["algo"] == "ed25519" and rec["signature"]["public_key"] == pub
    assert rec["predicate"] == {"verdict": "passed"}, "the body is untouched"
    ok = run("--check-record", str(a))
    assert ok.returncode == 0 and ok.stdout.startswith("OK:") and pub in ok.stdout and rec["chain"]["hash"] in ok.stdout
    assert run("--check-record", str(a), "--trust-key", pub).returncode == 0
    assert run("--check-record", str(a), "--trust-key", "ab" * 32).returncode == 2
    # a second signing is refused
    assert run("--sign-record", str(a), "--sign-key", str(key)).returncode == 1
    # the chain: b links to a's hash
    b = tmp_path / "b.json"
    b.write_text(json.dumps({"predicate": {"verdict": "failed"}}), encoding="utf-8")
    assert run("--sign-record", str(b), "--sign-key", str(key), "--record-prev", str(a)).returncode == 0
    assert json.loads(b.read_text(encoding="utf-8"))["chain"]["prev"] == rec["chain"]["hash"]
    # a tampered byte in the body fails the check
    text = a.read_text(encoding="utf-8").replace('"passed"', '"failed"')
    a.write_text(text, encoding="utf-8")
    bad = run("--check-record", str(a))
    assert bad.returncode == 2 and bad.stdout.startswith("BAD SIGNATURE")
    # a record that is not an object is refused
    c = tmp_path / "c.json"
    c.write_text("[1,2]\n", encoding="utf-8")
    assert run("--sign-record", str(c), "--sign-key", str(key)).returncode == 1
    # an empty object is refused (it would sign to invalid JSON)
    e = tmp_path / "e.json"
    e.write_text("{}\n", encoding="utf-8")
    assert run("--sign-record", str(e), "--sign-key", str(key)).returncode == 1
    # bytes appended after the signature are covered by nothing: not OK
    good = b.read_text(encoding="utf-8").rstrip()
    assert good.endswith("}}")
    b.write_text(good[:-1] + ',"note":"forged"}\n', encoding="utf-8")
    forged = run("--check-record", str(b))
    assert forged.returncode == 2 and "after its signature" in forged.stdout
    b.write_text(good + "\n", encoding="utf-8")
    assert run("--check-record", str(b)).returncode == 0


def test_delegation_receipt_round_trip_through_the_binary(tmp_path):
    """The client's statement body, signed and checked by the runner."""
    sys.path.insert(0, str(ROOT / "python" / "src"))
    from xyntetik_runner.shadow import receipt
    home = tmp_path / "home"
    key = receipt.keygen(home, str(RUNNER))
    assert key.is_file() and receipt.keygen(home, str(RUNNER)) == key, "one key per installation"
    body = receipt.statement(repo="/r/proj", head="a" * 40, request="fix it", patch_path="/p/x.patch",
                             patch_sha256="11" * 32, changed_paths=("a.py",), verdict="tests passed on the scratch copy",
                             tests_exit=0, task_class="function", model="/m/coder.gguf", model_sha256="22" * 32,
                             adapter_sha256="", runner_build="0.5.1", backend="cpu", budget_turns=6,
                             budget_wall_s=300.0, turns=2, tool_calls=2, wall_s=4.2, stop_reason="finish",
                             observed_at="2026-09-08T00:00:00Z")
    first = receipt.write_receipt(home, body, runner=str(RUNNER), key=key)
    second = receipt.write_receipt(home, dict(body, predicate=dict(body["predicate"], verdict="tests failed")),
                                   runner=str(RUNNER), key=key)
    assert receipt.check_with_runner(str(RUNNER), first.path)[0] == 0
    rc, text = receipt.check_with_runner(str(RUNNER), second.path, trust_key=first.public_key)
    assert rc == 0 and first.chain_hash in json.loads(second.path.read_text(encoding="utf-8"))["chain"]["prev"]
    listing = receipt.render_receipts(home, runner=str(RUNNER), check=True)
    assert listing.count("| OK |") == 2
    first.path.write_text(first.path.read_text(encoding="utf-8").replace('"tests passed', '"tests PASSED'), encoding="utf-8")
    assert "BAD SIGNATURE" in receipt.render_receipts(home, runner=str(RUNNER), check=True)
