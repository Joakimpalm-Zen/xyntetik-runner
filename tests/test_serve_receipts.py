"""Per-request receipts in serve mode (R1.2.2): `--receipts DIR`.

Every finished generation writes the CLI transcript record into DIR, chained
in write order and signed with --sign-key. The anchor is the CLI's own
`--verify`: a receipt a server wrote must replay VERIFIED against the model,
greedy and sampled, raw and chat -- the server's record of what it did is
checked by re-doing it on another code path. hashlib anchors the model and
binary digests.
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

ZEROS = "0" * 64


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def fx(runner_bin, tmp_path_factory):
    d = tmp_path_factory.mktemp("rcpt")
    model = d / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(model)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    key = d / "k.json"
    subprocess.run([runner_bin, "--keygen", key], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"model": model, "key": key,
            "pk": json.loads(key.read_text())["public_key"]}


def _serve(runner_bin, fx, dir_, *extra):
    return RunnerServer(runner_bin, fx["model"], ctx=256, extra_args=[
        "--gpu", "off", "-t", "2", "--receipts", str(dir_),
        "--sign-key", str(fx["key"]), *extra])


def _post(srv, path, payload):
    req = urllib.request.Request(srv.base_url + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _verify(runner_bin, fx, rec):
    return subprocess.run([runner_bin, "-m", fx["model"], "--verify", rec,
                           "--trust-key", fx["pk"], "--gpu", "off", "-t", "2",
                           "-c", "256"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=120)


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def test_served_receipts_replay_and_chain(runner_bin, fx, tmp_path):
    d = tmp_path / "r"
    # chatml, so the chat receipt names a template other than raw
    with _serve(runner_bin, fx, d, "--chat-template", "chatml") as srv:
        a = _post(srv, "/v1/completions", {
            "prompt": "the quick brown fox", "max_tokens": 6,
            "temperature": 0, "cache_prompt": False})
        b = _post(srv, "/v1/completions", {
            "prompt": "once upon a time", "max_tokens": 6,
            "temperature": 0.9, "seed": 11, "cache_prompt": False})
        c = _post(srv, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 6, "temperature": 0.7, "seed": 5,
            "cache_prompt": False})
    names = sorted(p.name for p in d.iterdir())
    assert names == ["receipt-0000000001.json", "receipt-0000000002.json",
                     "receipt-0000000003.json"]
    recs = [json.loads((d / n).read_text()) for n in names]
    for body, rec, name in zip((a, b, c), recs, names):
        t = body["runner_telemetry"]["receipt"]
        assert t == {"file": name, "chain_hash": rec["chain"]["hash"]}
        assert rec["model"]["sha256"] == _sha(fx["model"])
        assert rec["build"]["binary_sha256"] == _sha(runner_bin)
        assert rec["signature"]["public_key"] == fx["pk"]
        assert rec["serve"]["request_id"] == body["id"]
    assert recs[0]["chain"]["prev"] == ZEROS
    assert recs[1]["chain"]["prev"] == recs[0]["chain"]["hash"]
    assert recs[2]["chain"]["prev"] == recs[1]["chain"]["hash"]
    assert recs[2]["serve"]["api"] == "chat.completions"
    assert recs[0]["config"]["template"] == "raw"
    assert recs[2]["config"]["template"] == "chatml"
    # the anchor: each one re-done by the CLI replay
    for n in names:
        v = _verify(runner_bin, fx, d / n)
        out = (v.stdout + v.stderr).decode(errors="replace")
        assert v.returncode == 0 and "VERIFIED" in out, (n, out[-600:])


def test_retention_and_restart_continue_the_chain(runner_bin, fx, tmp_path):
    d = tmp_path / "r"
    with _serve(runner_bin, fx, d, "--receipts-keep", "2") as srv:
        for i in range(4):
            _post(srv, "/v1/completions", {"prompt": f"item {i}",
                                           "max_tokens": 2, "temperature": 0})
    names = sorted(p.name for p in d.iterdir())
    assert names == ["receipt-0000000003.json", "receipt-0000000004.json"]
    last = json.loads((d / names[-1]).read_text())["chain"]["hash"]
    with _serve(runner_bin, fx, d, "--receipts-keep", "2") as srv:
        _post(srv, "/v1/completions", {"prompt": "again", "max_tokens": 2,
                                       "temperature": 0})
    rec = json.loads((d / "receipt-0000000005.json").read_text())
    assert rec["chain"]["prev"] == last


def test_what_shaped_the_output_is_recorded(runner_bin, fx, tmp_path):
    d = tmp_path / "r"
    with _serve(runner_bin, fx, d) as srv:
        _post(srv, "/v1/completions", {
            "prompt": "x", "max_tokens": 8, "stop": ["e"],
            "response_format": {"type": "json_object"}})
    rec = json.loads(next(d.iterdir()).read_text())
    assert set(rec["serve"]["shaped_by"]) >= {"json_mode", "stop"}


def test_a_broken_newest_record_refuses_the_start(runner_bin, fx, tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "receipt-0000000007.json").write_text("{not json")
    p = subprocess.run([runner_bin, "-m", fx["model"], "--serve", "--port", "1",
                        "--no-tray", "--gpu", "off", "--receipts", d],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=60)
    assert p.returncode != 0 and b"second chain" in p.stderr, p.stderr[-300:]
    p = subprocess.run([runner_bin, "-m", fx["model"], "-p", "x",
                        "--receipts", d],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=60)
    assert p.returncode != 0 and b"--serve" in p.stderr


def test_a_served_schema_or_json_mode_turn_replays_under_its_constraint(
        runner_bin, fx, tmp_path):
    """D4b (R1.1.3): a served turn shaped by a JSON schema replays when the
    verifier names that schema (matched by digest), and one shaped by JSON
    mode replays under --json. A tool turn is still refused by name: its
    grammar is built from the request, which the record does not carry."""
    d = tmp_path / "r"
    schema = {"type": "object", "properties": {"n": {"type": "integer"}},
              "required": ["n"]}
    sfile = tmp_path / "schema.json"
    sfile.write_text(json.dumps(schema, indent=2))
    tools = [{"type": "function", "function": {
        "name": "f", "parameters": {"type": "object",
                                    "properties": {"a": {"type": "string"}},
                                    "required": ["a"]}}}]
    # a tool declaration's prompt does not fit the fixture's 256 tokens
    with RunnerServer(runner_bin, fx["model"], ctx=1024, extra_args=[
            "--gpu", "off", "-t", "2", "--receipts", str(d),
            "--sign-key", str(fx["key"]), "--chat-template", "chatml"]) as srv:
        _post(srv, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "a number"}],
            "max_tokens": 12, "temperature": 0.8, "seed": 3,
            "cache_prompt": False,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "s", "schema": schema}}})
        _post(srv, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "json please"}],
            "max_tokens": 12, "temperature": 0, "cache_prompt": False,
            "response_format": {"type": "json_object"}})
        _post(srv, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "call f"}],
            "max_tokens": 12, "temperature": 0, "cache_prompt": False,
            "tools": tools, "tool_choice": "required"})
    names = sorted(p.name for p in d.iterdir())
    schema_rec, json_rec, tool_rec = (d / n for n in names)

    def verify(rec, *extra):
        v = subprocess.run([runner_bin, "-m", fx["model"], "--verify", rec,
                            "--trust-key", fx["pk"], "--gpu", "off", "-t", "2",
                            "-c", "1024", *extra],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=120)
        return v.returncode, (v.stdout + v.stderr).decode(errors="replace")

    rc, out = verify(schema_rec, "--json-schema", str(sfile))
    assert rc == 0 and "VERIFIED" in out, out[-600:]
    rc, out = verify(schema_rec)                     # the schema is needed
    assert rc == 3 and "--json-schema" in out, out[-600:]
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"type": "object"}))
    rc, out = verify(schema_rec, "--json-schema", str(other))
    assert rc == 3, out[-600:]                       # and it must be this one
    rc, out = verify(json_rec, "--json")
    assert rc == 0 and "VERIFIED" in out, out[-600:]
    rc, out = verify(tool_rec)
    assert rc == 3 and "tools" in out, out[-600:]
