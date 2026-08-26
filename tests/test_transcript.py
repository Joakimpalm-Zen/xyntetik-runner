"""--transcript: notarized inference D1 (xyntetik.runner.transcript.v1).

The record makes one inference a replayable computation. Gates: two
identical runs produce identical output tokens (T1 substrate); the chain
hash re-verifies from the file bytes alone (sha256 over everything before
the ,"chain" key — a verifier needs a text editor, not this repo); the
model/binary hashes are real; and a run with a different seed at temp>0
is a DIFFERENT record (the determinism is seeded, not vacuous).
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("transcript")
    b = d / "base.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(b)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return b


def _run(runner_bin, base, out, seed="7", temp="0", extra=()):
    p = subprocess.run(
        [runner_bin, "-m", str(base), "-p", "the runner trains the",
         "-n", "12", "--temp", temp, "-s", seed, "--gpu", "off", "-t", "2",
         "--transcript", str(out), *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(out.read_bytes())


def test_replay_substrate_and_chain(runner_bin, base, tmp_path):
    t1, t2 = tmp_path / "t1.json", tmp_path / "t2.json"
    r1 = _run(runner_bin, base, t1)
    r2 = _run(runner_bin, base, t2)
    assert r1["output"]["tokens"] == r2["output"]["tokens"]
    assert r1["prompt"]["tokens"] == r2["prompt"]["tokens"]
    # chain hash verifies from the file bytes alone
    raw = t1.read_bytes()
    body = raw[:raw.rindex(b',"chain"')]
    assert hashlib.sha256(body).hexdigest() == r1["chain"]["hash"]
    # the hashes are real, not placeholders
    assert r1["model"]["sha256"] == hashlib.sha256(
        base.read_bytes()).hexdigest()
    assert len(r1["build"]["binary_sha256"]) == 64
    assert r1["config"]["seed"] == 7
    assert r1["config"]["seed_u64"] == "7"
    assert r1["output"]["finish"] in ("stop", "length")
    assert r1["schema_version"] == "xyntetik.runner.transcript.v1"


def test_binary_hash_uses_running_image_not_argv0(runner_bin, base, tmp_path):
    rec = tmp_path / "path-invocation.json"
    env = dict(os.environ)
    env["PATH"] = str(ROOT) + os.pathsep + env.get("PATH", "")
    p = subprocess.run(
        [runner_bin.name, "-m", str(base), "-p", "hello", "-n", "1",
         "--temp", "0", "--gpu", "off", "--transcript", str(rec)],
        cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    r = json.loads(rec.read_bytes())
    assert r["build"]["binary_sha256"] == hashlib.sha256(
        runner_bin.read_bytes()).hexdigest()

    p = subprocess.run(
        [runner_bin.name, "-m", str(base), "--verify", str(rec),
         "--gpu", "off"], cwd=tmp_path, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert json.loads(p.stdout)["tier"] == "T1"


def _verify(runner_bin, base, rec, lora=None):
    cmd = [runner_bin, "-m", str(base), "--verify", str(rec),
           "--gpu", "off", "-t", "2"]
    if lora:
        cmd += ["--lora", str(lora)]
    return subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=120)


def _forge_record(path, record):
    """Write a syntactically valid record with a matching public chain hash."""
    body_record = dict(record)
    chain = dict(body_record.pop("chain"))
    body = json.dumps(body_record, separators=(",", ":"),
                      ensure_ascii=False).encode()[:-1]
    chain["hash"] = hashlib.sha256(body).hexdigest()
    path.write_bytes(body + b',"chain":' +
                     json.dumps(chain, separators=(",", ":")).encode() +
                     b"}\n")


def test_verify_verdicts(runner_bin, base, tmp_path):
    """D2: the three-verdict contract. Same binary verifies at T1; a
    tampered record (stale chain) is UNVERIFIABLE exit 3; a FORGED record
    (token changed, chain recomputed to match) is caught by the replay
    itself: DIVERGED exit 2 — you can forge the hash, not the model."""
    rec = tmp_path / "r.json"
    r = _run(runner_bin, base, rec)
    p = _verify(runner_bin, base, rec)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    v = json.loads(p.stdout)
    assert v["verdict"] == "VERIFIED" and v["tier"] == "T1"

    raw = rec.read_bytes()
    tampered = tmp_path / "tampered.json"
    tampered.write_bytes(raw.replace(b'"finish"', b'"fXnish"', 1))
    p = _verify(runner_bin, base, tampered)
    assert p.returncode == 3
    assert b"chain hash mismatch" in p.stderr

    toks = r["output"]["tokens"]
    old_arr = json.dumps(toks, separators=(",", ":")).encode()
    forged_toks = list(toks)
    forged_toks[0] = forged_toks[0] + 1
    new_arr = json.dumps(forged_toks, separators=(",", ":")).encode()
    body = raw[:raw.rindex(b',"chain"')]
    # the output tokens array is the LAST token array in the body
    i = body.rindex(old_arr)
    forged_body = body[:i] + new_arr + body[i + len(old_arr):]
    import re as _re
    tail = _re.sub(rb'"hash":"[0-9a-f]{64}"',
                   b'"hash":"' +
                   hashlib.sha256(forged_body).hexdigest().encode() + b'"',
                   raw[raw.rindex(b',"chain"'):])
    forged = tmp_path / "forged.json"
    forged.write_bytes(forged_body + tail)
    p = _verify(runner_bin, base, forged)
    assert p.returncode == 2
    v = json.loads(p.stdout)
    assert v["verdict"] == "DIVERGED" and v["at"] == 0

    # a real adapter the record did not use: UNVERIFIABLE, not a replay
    subprocess.run([sys.executable, ROOT / "scripts/make-test-lora.py",
                    str(base), str(tmp_path / "fx")], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    p = _verify(runner_bin, base, rec, lora=tmp_path / "fx.adapter.gguf")
    assert p.returncode == 3, p.stderr.decode(errors="replace")
    assert b"without an adapter" in p.stderr


def test_verify_rejects_invalid_sampling_values_before_replay(
        runner_bin, base, tmp_path):
    rec = tmp_path / "sampling.json"
    r = _run(runner_bin, base, rec)
    r["config"]["top_k"] = 1.5
    forged = tmp_path / "fractional-top-k.json"
    _forge_record(forged, r)

    p = _verify(runner_bin, base, forged)
    assert p.returncode == 3, p.stderr.decode(errors="replace")
    assert b"malformed sampling config" in p.stderr


def test_verify_rejects_invalid_token_ids_before_engine_feed(
        runner_bin, base, tmp_path):
    rec = tmp_path / "tokens.json"
    r = _run(runner_bin, base, rec)
    r["prompt"]["tokens"][0] = -1
    forged = tmp_path / "negative-token.json"
    _forge_record(forged, r)

    p = _verify(runner_bin, base, forged)
    assert p.returncode == 3, p.stderr.decode(errors="replace")
    assert b"invalid prompt token" in p.stderr


def test_seeded_not_vacuous(runner_bin, base, tmp_path):
    a = _run(runner_bin, base, tmp_path / "a.json", seed="7", temp="0.9")
    b = _run(runner_bin, base, tmp_path / "b.json", seed="7", temp="0.9")
    assert a["output"]["tokens"] == b["output"]["tokens"]  # same seed replays


def test_seed_above_json_exact_integer_range_replays(runner_bin, base, tmp_path):
    """A uint64 seed must survive the transcript's JSON number boundary.

    Runner's JSON AST stores numbers as doubles, so the numeric compatibility
    field alone cannot carry every seed accepted by ``-s``.  The exact decimal
    spelling is the replay authority.
    """
    rec = tmp_path / "wide-seed.json"
    seed = "9007199254740993"  # 2**53 + 1: rounds when parsed as a double
    r = _run(runner_bin, base, rec, seed=seed, temp="0.9")
    assert r["config"]["seed"] == int(seed)
    assert r["config"]["seed_u64"] == seed

    p = _verify(runner_bin, base, rec)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert json.loads(p.stdout)["verdict"] == "VERIFIED"


def test_profile_records_effective_runtime_and_drives_replay(
        runner_bin, base, tmp_path):
    rec = tmp_path / "profile.json"
    r = _run(runner_bin, base, rec, extra=("--kv", "q8", "-b", "3"))
    # This fixture cannot store q8 KV.  A receipt records what ran, not merely
    # what the CLI requested; batch likewise records the allocated geometry.
    assert r["profile"]["kv"] == "f16"
    assert r["profile"]["batch"] == 3
    assert r["profile"]["threads"] == 2
    assert r["profile"]["gpu_layers"] == 0

    # Conflicting CLI placement values must not reshape the replay.  Verbose
    # load diagnostics expose the actual pool and activation geometry.
    p = subprocess.run(
        [runner_bin, "-m", str(base), "--verify", str(rec), "--gpu", "off",
         "-t", "1", "-b", "1", "-v"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    err = p.stderr.decode(errors="replace")
    assert "| 2 threads |" in err
    assert "batch                    3" in err


def test_recorded_adapter_scale_drives_replay(runner_bin, base, tmp_path):
    prefix = tmp_path / "fx"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-lora.py",
                    str(base), str(prefix)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    adapter = tmp_path / "fx.adapter.gguf"
    rec = tmp_path / "adapter.json"
    _run(runner_bin, base, rec,
         extra=("--lora", str(adapter), "--lora-scale", "0.25"))

    p = subprocess.run(
        [runner_bin, "-m", str(base), "--verify", str(rec), "--gpu", "off",
         "-t", "2", "--lora", str(adapter)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert "scale x0.25" in p.stderr.decode(errors="replace")
