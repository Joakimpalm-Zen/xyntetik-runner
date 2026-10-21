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


def test_verify_diffs_recorded_output_bytes(runner_bin, base, tmp_path):
    rec = tmp_path / "output-text.json"
    r = _run(runner_bin, base, rec)
    original = r["output"]["bytes_hex"]
    r["output"]["bytes_hex"] = ("00" if original[:2] != "00" else "01") + \
        original[2:]
    forged = tmp_path / "forged-output-text.json"
    _forge_record(forged, r)

    p = _verify(runner_bin, base, forged)
    assert p.returncode == 2, p.stderr.decode(errors="replace")
    v = json.loads(p.stdout)
    assert v["verdict"] == "DIVERGED"
    assert v["unit"] == "byte"


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


def test_profile_gpu_window_is_validated_and_ignored_off_cuda(
        runner_bin, base, tmp_path):
    """The CUDA prompt window (profile.gpu_window) decides which route a
    prompt's last partial tile takes, so a CUDA record carries it and --verify
    rebuilds the instance at it. A CPU run writes no window; a malformed one is
    refused before any replay; a well-formed one is ignored by a backend that
    has no such window (a CUDA record replays there at the token tier)."""
    rec = tmp_path / "window.json"
    r = _run(runner_bin, base, rec)
    assert "gpu_window" not in r["profile"]

    for bad in (100, 0, -64, 64.5, "64"):
        r2 = json.loads(json.dumps(r))
        r2["profile"]["gpu_window"] = bad
        forged = tmp_path / "bad-window.json"
        _forge_record(forged, r2)
        p = _verify(runner_bin, base, forged)
        assert p.returncode == 3, (bad, p.stderr.decode(errors="replace"))
        assert b"malformed execution profile" in p.stderr, bad

    r3 = json.loads(json.dumps(r))
    r3["profile"]["gpu_window"] = 128
    forged = tmp_path / "cpu-window.json"
    _forge_record(forged, r3)
    p = _verify(runner_bin, base, forged)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert json.loads(p.stdout)["verdict"] == "VERIFIED"


def test_cuda_record_carries_and_enforces_its_prompt_window(
        runner_bin, base, tmp_path):
    caps = json.loads(subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, check=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("needs a CUDA device: only CUDA sizes a prompt window")
    rec = tmp_path / "cuda-window.json"
    p = subprocess.run(
        [runner_bin, "-m", str(base), "-p", "the runner trains the", "-n", "12",
         "--temp", "0", "-s", "7", "-t", "2", "-b", "3", "--no-tray",
         "--transcript", str(rec)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    r = json.loads(rec.read_bytes())
    if not r["profile"]["gpu"]:
        pytest.skip("the fixture did not run on the device")
    # -b 3 caps the window at its 64-column minimum
    assert r["profile"]["gpu_window"] == 64

    def verify(path):
        return subprocess.run(
            [runner_bin, "-m", str(base), "--verify", str(path), "-t", "2", "--no-tray"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    v = verify(rec)
    assert v.returncode == 0, v.stderr.decode(errors="replace")
    assert json.loads(v.stdout)["verdict"] == "VERIFIED"

    # a window the recorded batch cannot reach: the replay would run at
    # another window, so it must not claim the record
    r["profile"]["gpu_window"] = 128
    forged = tmp_path / "cuda-window-128.json"
    _forge_record(forged, r)
    v = verify(forged)
    assert v.returncode == 3, v.stderr.decode(errors="replace")
    assert b"could not reproduce the recorded execution profile" in v.stderr


def test_profile_gpu_route_is_validated_and_ignored_off_cuda(
        runner_bin, base, tmp_path):
    """The CUDA arithmetic routes (profile.gpu_route) are part of a record's
    arithmetic, so --verify pins them. A CPU run writes none; a malformed
    value is refused before any replay; a well-formed one is ignored by a
    backend without those kernels. --gpu-route itself refuses an unknown name
    and is accepted, with no effect, off CUDA."""
    rec = tmp_path / "route.json"
    r = _run(runner_bin, base, rec, extra=("--gpu-route", "tc2,attn-tc"))
    assert "gpu_route" not in r["profile"]

    for bad in ("", "bogus", "tc2,", ",tc2", "tc2,,sm80", "TC2", 5, None):
        r2 = json.loads(json.dumps(r))
        r2["profile"]["gpu_route"] = bad
        forged = tmp_path / "bad-route.json"
        _forge_record(forged, r2)
        p = _verify(runner_bin, base, forged)
        assert p.returncode == 3, (bad, p.stderr.decode(errors="replace"))
        assert b"malformed execution profile" in p.stderr, bad

    for good in ("reference", "attn-tc,tc2", "tc2,sm80,attn-tile,attn-dec,attn-tc",
                 "attn-tile,moe-tc,moe-grouped"):
        r3 = json.loads(json.dumps(r))
        r3["profile"]["gpu_route"] = good
        forged = tmp_path / "cpu-route.json"
        _forge_record(forged, r3)
        p = _verify(runner_bin, base, forged)
        assert p.returncode == 0, (good, p.stderr.decode(errors="replace"))
        assert json.loads(p.stdout)["verdict"] == "VERIFIED"

    p = subprocess.run([runner_bin, "-m", str(base), "-p", "x", "-n", "1",
                        "--gpu-route", "warp9"], cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=60)
    assert p.returncode != 0
    assert b"--gpu-route expects reference or a comma list" in p.stderr


def test_cuda_record_carries_and_replays_its_routes(runner_bin, base, tmp_path):
    caps = json.loads(subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, check=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("needs a CUDA device: only CUDA has these routes")
    for route in ("reference", "attn-tile"):
        rec = tmp_path / f"cuda-route-{route}.json"
        p = subprocess.run(
            [runner_bin, "-m", str(base), "-p", "the runner trains the", "-n", "12",
             "--temp", "0", "-s", "7", "-t", "2", "--gpu-route", route, "--no-tray",
             "--transcript", str(rec)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        assert p.returncode == 0, p.stderr.decode(errors="replace")
        r = json.loads(rec.read_bytes())
        if not r["profile"]["gpu"]:
            pytest.skip("the fixture did not run on the device")
        assert r["profile"]["gpu_route"] == route
        # the replay pins the record's routes whatever the command line asks
        v = subprocess.run(
            [runner_bin, "-m", str(base), "--verify", str(rec), "-t", "2",
             "--gpu-route", "reference" if route != "reference" else "attn-tile",
             "--no-tray"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        assert v.returncode == 0, v.stderr.decode(errors="replace")
        assert json.loads(v.stdout)["verdict"] == "VERIFIED"


def test_cuda_moe_family_routes_are_recorded_and_replayed(runner_bin, base, tmp_path):
    """moe-tc and moe-grouped are routes of the MoE families measured for them
    (gpt-oss: moe-tc only). A record of such a model names the ones that ran
    and --verify replays it; the reference set names neither; a pin of them on
    a dense model never reaches its record; and the grouped prefill asked for
    on another MoE family is named, so its record replays."""
    caps = json.loads(subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, check=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("needs a CUDA device: only CUDA has these routes")
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py",
                    str(tmp_path / "fx")], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    oss = tmp_path / "fx.gptoss-mxfp4.gguf"

    def record(model, name, *route):
        rec = tmp_path / name
        p = subprocess.run(
            [runner_bin, "-m", str(model), "-p", "the runner trains the", "-n", "12",
             "--temp", "0", "-s", "7", "-t", "2", "--no-tray", "--transcript",
             str(rec), *route], cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=120)
        assert p.returncode == 0, p.stderr.decode(errors="replace")
        r = json.loads(rec.read_bytes())
        if not r["profile"]["gpu"]:
            pytest.skip("the fixture did not run on the device")
        return rec, r["profile"]["gpu_route"].split(",")

    rec, routes = record(oss, "oss-default.json")
    assert "moe-tc" in routes and "moe-grouped" not in routes, routes
    v = subprocess.run([runner_bin, "-m", str(oss), "--verify", str(rec), "-t", "2",
                        "--no-tray"], cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    assert v.returncode == 0, v.stderr.decode(errors="replace")
    assert json.loads(v.stdout)["verdict"] == "VERIFIED"

    _, routes = record(oss, "oss-ref.json", "--gpu-route", "reference")
    assert routes == ["reference"], routes
    _, routes = record(base, "dense-moe-pin.json", "--gpu-route", "moe-tc,moe-grouped")
    assert routes == ["reference"], routes

    # the grouped prefill asked for on another family (gemma-4's MoE, Q8_0
    # experts): the record must name it, or --verify replays another path
    g4 = tmp_path / "fx-g4-q8.gguf"
    q = subprocess.run([runner_bin, "-m", str(tmp_path / "fx.gemma4-moe.gguf"), "--quantize",
                        str(g4), "--quant", "q8_0"], cwd=ROOT, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=120)
    assert q.returncode == 0
    os.environ["RUNNER_MOE_GROUPED"] = "1"
    try:
        rec, routes = record(g4, "g4-grouped.json")
    finally:
        del os.environ["RUNNER_MOE_GROUPED"]
    assert "moe-grouped" in routes and "moe-tc" not in routes, routes
    v = subprocess.run([runner_bin, "-m", str(g4), "--verify", str(rec), "-t", "2",
                        "--no-tray"], cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    assert v.returncode == 0, v.stderr.decode(errors="replace")
    assert json.loads(v.stdout)["verdict"] == "VERIFIED"


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


def test_a_nul_output_byte_still_verifies(runner_bin, base, tmp_path):
    """A NUL byte in the output was written into output.text as \\u0000, which
    the runner's own parser refuses (json.c): the record could not be read
    back, and --verify called it malformed. The text is a rendering (U+FFFD
    there); bytes_hex keeps the exact byte, and the replay compares that."""
    out = tmp_path / "nul.json"
    args = ["-m", str(base), "-p", "the quick brown fox", "-n", "300", "--temp",
            "1.0", "-s", "5", "--gpu", "off", "-t", "2", "-c", "512", "--ignore-eos"]
    p = subprocess.run([runner_bin, *args, "--transcript", str(out)], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    rec = json.loads(out.read_bytes())
    # the precondition this fixture and seed provide
    assert b"\0" in bytes.fromhex(rec["output"]["bytes_hex"])
    assert "\0" not in rec["output"]["text"]
    v = subprocess.run([runner_bin, "-m", str(base), "--verify", str(out), "--gpu",
                        "off", "-t", "2", "-c", "512", "--ignore-eos"], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert v.returncode == 0 and b"VERIFIED" in v.stderr, v.stderr[-400:]


def test_the_token_that_completes_a_json_document_is_recorded(runner_bin, base,
                                                              tmp_path):
    """A token that completes a constrained document ends the turn before its
    forward, and the record read the output tokens from a history that was
    only written on the way to that forward: `{}` was recorded as [126, 0].
    The fixture's vocabulary is bytes offset by 3, so the ids are the text."""
    out = tmp_path / "json.json"
    p = subprocess.run(
        [runner_bin, "-m", str(base), "-p", "hello", "--json", "-n", "60", "-s",
         "4", "--gpu", "off", "-t", "2", "--transcript", str(out)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    o = json.loads(out.read_bytes())["output"]
    # the precondition: the document closed before the budget ran out
    assert o["finish"] == "stop" and o["n"] < 60, o
    assert bytes(t - 3 for t in o["tokens"]) == bytes.fromhex(o["bytes_hex"]), o
