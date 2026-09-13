"""`--sign-record` and `--check-record`: any JSON object file carries the
transcript's chain and signature, and is checked by the same code path a
notarized run is (R14.6). A tampered byte fails, a second signing is refused,
and the chain links by the previous record's hash."""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")


def run(*args):
    return subprocess.run([str(RUNNER), *args], capture_output=True, text=True, cwd=ROOT, timeout=60)


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
