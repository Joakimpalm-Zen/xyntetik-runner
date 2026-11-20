"""Planted-symlink and concurrent-writer checks for R6.101 partial outputs.

Every atomic output opens a new temporary file via plat_fopen_partial()
(O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW on POSIX; CREATE_NEW on Windows) with an
unpredictable 16-hex suffix (fixed only by RUNNER_TEST_PARTIAL_SUFFIX in tests),
and unlinks only the path it created if an error follows.
"""
import concurrent.futures
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

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


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("RUNNER_") and not k.startswith("XR_")}
    env.update(extra)
    return env


def _plant_symlink(link_path, victim_path, text="untouched"):
    victim_path.write_text(text)
    try:
        os.symlink(victim_path, link_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted here")


@pytest.fixture(scope="module")
def fx(runner_bin, tmp_path_factory):
    d = tmp_path_factory.mktemp("partial_fx")
    model = d / "base.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"),
                    str(model)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    key = d / "sign.json"
    subprocess.run([str(runner_bin), "--keygen", str(key)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    data = d / "train.jsonl"
    rows = [{"prompt": f"item {i}", "completion": f" ans {i}"} for i in range(4)]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rec = d / "unsigned.json"
    subprocess.run([str(runner_bin), "-m", str(model), "-p", "hello", "-n", "2",
                    "--temp", "0", "--gpu", "off", "-t", "2",
                    "--transcript", str(rec)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env=_clean_env())
    return {"d": d, "model": model, "key": key, "data": data, "rec": rec}


def test_planted_symlink_refused_for_cli_outputs(runner_bin, fx, tmp_path):
    """Planted symlinks at <target>.partial-fixed / <target>.tmp-fixed are
    never followed or unlinked, and destinations remain untouched."""
    env = _clean_env(RUNNER_TEST_PARTIAL_SUFFIX="fixed")

    # 1. --transcript (envelope.c:transcript)
    d1 = tmp_path / "transcript"
    d1.mkdir()
    out1 = d1 / "t.json"
    v1 = d1 / "victim.txt"
    _plant_symlink(d1 / "t.json.partial-fixed", v1)
    r = subprocess.run([str(runner_bin), "-m", str(fx["model"]), "-p", "hi",
                        "-n", "2", "--temp", "0", "--gpu", "off", "-t", "2",
                        "--transcript", str(out1)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode != 0, r.stderr
    assert v1.read_text() == "untouched"
    assert not out1.exists()

    # 2. --sign-record (envelope.c:record_sign)
    d2 = tmp_path / "sign_record"
    d2.mkdir()
    rec2 = d2 / "rec.json"
    orig_bytes = b'{"schema_version":"xyntetik.runner.custom.v1","ok":true}\n'
    rec2.write_bytes(orig_bytes)
    v2 = d2 / "victim.txt"
    _plant_symlink(d2 / "rec.json.partial-fixed", v2)
    r = subprocess.run([str(runner_bin), "--sign-record", str(rec2),
                        "--sign-key", str(fx["key"])],
                       cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode != 0 and "cannot write" in r.stderr, r.stderr
    assert v2.read_text() == "untouched"
    assert rec2.read_bytes() == orig_bytes

    # 3. --quantize (quantize.c)
    d3 = tmp_path / "quantize"
    d3.mkdir()
    qout = d3 / "q.gguf"
    qout.write_bytes(b"SENTINEL")
    v3 = d3 / "victim.txt"
    _plant_symlink(d3 / "q.gguf.partial-fixed", v3)
    r = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                        "--quantize", str(qout), "--quant", "q8_0"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode != 0, r.stderr
    assert v3.read_text() == "untouched"
    assert qout.read_bytes() == b"SENTINEL"

    # 4. --train adapter (model.c:model_lora_save)
    d4 = tmp_path / "lora"
    d4.mkdir()
    aout = d4 / "adapter.gguf"
    v4 = d4 / "victim.txt"
    _plant_symlink(d4 / "adapter.gguf.partial-fixed", v4)
    r = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                        "--train", str(fx["data"]), "--train-steps", "1",
                        "--lr", "3e-3", "--train-batch", "2", "--gpu", "off",
                        "-t", "2", "--train-out", str(aout)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode != 0, r.stderr
    assert v4.read_text() == "untouched"
    assert not aout.exists()

    # 5. --train optimizer sidecar (main.c:train_optim_save)
    d5 = tmp_path / "optim"
    d5.mkdir()
    aout5 = d5 / "adapter.gguf"
    v5 = d5 / "victim.txt"
    _plant_symlink(d5 / "adapter.gguf.optim.partial-fixed", v5)
    r = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                        "--train", str(fx["data"]), "--train-steps", "1",
                        "--save-every", "2", "--lr", "3e-3", "--train-batch", "2",
                        "--gpu", "off", "-t", "2", "--train-out", str(aout5)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode != 0, r.stderr
    assert v5.read_text() == "untouched"
    assert not (d5 / "adapter.gguf.optim").exists()

    # 6. --train --train-full (model.c:model_full_write, uses .tmp-)
    d6 = tmp_path / "full"
    d6.mkdir()
    fout = d6 / "full.gguf"
    v6 = d6 / "victim.txt"
    _plant_symlink(d6 / "full.gguf.tmp-fixed", v6)
    r = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                        "--train", str(fx["data"]), "--train-steps", "1",
                        "--lr", "3e-3", "--train-batch", "2", "--gpu", "off",
                        "-t", "2", "--train-full", "--train-out", str(fout)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode != 0, r.stderr
    assert v6.read_text() == "untouched"
    assert not fout.exists()


def _http_post(srv, path, payload):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        srv.base_url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as err:
        return err.code, json.load(err)


def test_planted_symlink_refused_for_kv_snapshot(runner_bin, fx, tmp_path):
    """Both pfx_write_one (<name>.kv) and kvsnap manifest (<name>.kv.json,
    unsigned and signed) refuse a symlink planted at their .partial-fixed path."""
    snaps_u = tmp_path / "snaps_unsigned"
    snaps_u.mkdir()
    vk = tmp_path / "vk.txt"
    vm = tmp_path / "vm.txt"
    _plant_symlink(snaps_u / "s_kv.kv.partial-fixed", vk)
    _plant_symlink(snaps_u / "s_man.kv.json.partial-fixed", vm)

    env = _clean_env(RUNNER_TEST_PARTIAL_SUFFIX="fixed")
    with RunnerServer(runner_bin, str(fx["model"]), ctx=256, env=env,
                      extra_args=["--gpu", "off", "-t", "2",
                                  "--kv-snapshots", str(snaps_u)]) as srv:
        st, body = _http_post(srv, "/v1/runner/contexts",
                              {"id": "c", "prompt": "hello world one two three"})
        assert st == 200, body

        st1, b1 = _http_post(srv, "/v1/runner/contexts/c/snapshot", {"name": "s_kv"})
        assert st1 == 500, b1
        assert vk.read_text() == "untouched"
        assert not (snaps_u / "s_kv.kv").exists()
        assert not (snaps_u / "s_kv.kv.json").exists()

        st2, b2 = _http_post(srv, "/v1/runner/contexts/c/snapshot", {"name": "s_man"})
        assert st2 == 500, b2
        assert vm.read_text() == "untouched"
        assert not (snaps_u / "s_man.kv").exists()
        assert not (snaps_u / "s_man.kv.json").exists()

    # Signed manifest path (kvsnap.c signed branch + nested record_sign)
    snaps_s = tmp_path / "snaps_signed"
    snaps_s.mkdir()
    vs = tmp_path / "vs.txt"
    _plant_symlink(snaps_s / "s_signed.kv.json.partial-fixed", vs)
    with RunnerServer(runner_bin, str(fx["model"]), ctx=256, env=env,
                      extra_args=["--gpu", "off", "-t", "2",
                                  "--kv-snapshots", str(snaps_s),
                                  "--sign-key", str(fx["key"])]) as srv:
        st, body = _http_post(srv, "/v1/runner/contexts",
                              {"id": "cs", "prompt": "hello world one two three"})
        assert st == 200, body
        st3, b3 = _http_post(srv, "/v1/runner/contexts/cs/snapshot", {"name": "s_signed"})
        assert st3 == 500, b3
        assert vs.read_text() == "untouched"
        assert not (snaps_s / "s_signed.kv").exists()
        assert not (snaps_s / "s_signed.kv.json").exists()


def test_legacy_partial_links_never_followed_or_removed(runner_bin, fx, tmp_path):
    """Faithful main-branch regression (no RUNNER_TEST_PARTIAL_SUFFIX):

    Plant symlinks at the legacy predictable names main used (.partial / .tmp)
    pointing to a 9-byte 'untouched' canary. Without RUNNER_TEST_PARTIAL_SUFFIX,
    main overwrote the canary (e.g. quantize: 9 bytes -> GGUF size); with
    R6.101 random partial suffixes, every site succeeds cleanly, leaves the
    canary untouched (9 bytes), and leaves each legacy symlink intact.
    """
    env = _clean_env()

    # 1. --transcript (<out>.partial)
    d1 = tmp_path / "leg_transcript"
    d1.mkdir()
    out1 = d1 / "t.json"
    v1 = d1 / "victim.txt"
    l1 = d1 / "t.json.partial"
    _plant_symlink(l1, v1)
    r = subprocess.run([str(runner_bin), "-m", str(fx["model"]), "-p", "hi",
                        "-n", "2", "--temp", "0", "--gpu", "off", "-t", "2",
                        "--transcript", str(out1)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stderr
    assert out1.is_file()
    assert v1.read_text() == "untouched"
    assert l1.is_symlink()

    # 2. --sign-record (<rec>.partial)
    d2 = tmp_path / "leg_sign"
    d2.mkdir()
    rec2 = d2 / "rec.json"
    rec2.write_text('{"schema_version":"xyntetik.runner.custom.v1","ok":true}\n')
    v2 = d2 / "victim.txt"
    l2 = d2 / "rec.json.partial"
    _plant_symlink(l2, v2)
    r = subprocess.run([str(runner_bin), "--sign-record", str(rec2),
                        "--sign-key", str(fx["key"])],
                       cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stderr
    assert "signature" in json.loads(rec2.read_text())
    assert v2.read_text() == "untouched"
    assert l2.is_symlink()

    # 3. --quantize (<out>.partial)
    d3 = tmp_path / "leg_quant"
    d3.mkdir()
    qout = d3 / "q.gguf"
    v3 = d3 / "victim.txt"
    l3 = d3 / "q.gguf.partial"
    _plant_symlink(l3, v3)
    r = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                        "--quantize", str(qout), "--quant", "q8_0"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stderr
    assert qout.is_file() and qout.stat().st_size > 0
    assert v3.read_text() == "untouched"
    assert l3.is_symlink()

    # 4. --train adapter (<out>.partial)
    d4 = tmp_path / "leg_lora"
    d4.mkdir()
    aout = d4 / "adapter.gguf"
    v4 = d4 / "victim.txt"
    l4 = d4 / "adapter.gguf.partial"
    _plant_symlink(l4, v4)
    r = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                        "--train", str(fx["data"]), "--train-steps", "1",
                        "--lr", "3e-3", "--train-batch", "2", "--gpu", "off",
                        "-t", "2", "--train-out", str(aout)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stderr
    assert aout.is_file()
    assert v4.read_text() == "untouched"
    assert l4.is_symlink()

    # 5. --train optimizer sidecar (<adapter>.optim.partial)
    d5 = tmp_path / "leg_optim"
    d5.mkdir()
    aout5 = d5 / "adapter.gguf"
    v5 = d5 / "victim.txt"
    l5 = d5 / "adapter.gguf.optim.partial"
    _plant_symlink(l5, v5)
    r = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                        "--train", str(fx["data"]), "--train-steps", "1",
                        "--save-every", "2", "--lr", "3e-3", "--train-batch", "2",
                        "--gpu", "off", "-t", "2", "--train-out", str(aout5)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stderr
    assert (d5 / "adapter.gguf.optim").is_file()
    assert v5.read_text() == "untouched"
    assert l5.is_symlink()

    # 6. --train --train-full (<out>.tmp)
    d6 = tmp_path / "leg_full"
    d6.mkdir()
    fout = d6 / "full.gguf"
    v6 = d6 / "victim.txt"
    l6 = d6 / "full.gguf.tmp"
    _plant_symlink(l6, v6)
    r = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                        "--train", str(fx["data"]), "--train-steps", "1",
                        "--lr", "3e-3", "--train-batch", "2", "--gpu", "off",
                        "-t", "2", "--train-full", "--train-out", str(fout)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stderr
    assert fout.is_file()
    assert v6.read_text() == "untouched"
    assert l6.is_symlink()

    # 7. KV snapshots: <name>.kv.partial + <name>.kv.json.partial (unsigned & signed)
    snaps_u = tmp_path / "leg_snaps_u"
    snaps_u.mkdir()
    vk = tmp_path / "leg_vk.txt"
    vm = tmp_path / "leg_vm.txt"
    lk = snaps_u / "leg_s.kv.partial"
    lm = snaps_u / "leg_s.kv.json.partial"
    _plant_symlink(lk, vk)
    _plant_symlink(lm, vm)
    with RunnerServer(runner_bin, str(fx["model"]), ctx=256, env=env,
                      extra_args=["--gpu", "off", "-t", "2",
                                  "--kv-snapshots", str(snaps_u)]) as srv:
        st, body = _http_post(srv, "/v1/runner/contexts",
                              {"id": "c", "prompt": "hello world one two three"})
        assert st == 200, body
        st1, b1 = _http_post(srv, "/v1/runner/contexts/c/snapshot", {"name": "leg_s"})
        assert st1 == 200, b1
        assert (snaps_u / "leg_s.kv").is_file()
        assert (snaps_u / "leg_s.kv.json").is_file()
        assert vk.read_text() == "untouched" and lk.is_symlink()
        assert vm.read_text() == "untouched" and lm.is_symlink()

    snaps_s = tmp_path / "leg_snaps_s"
    snaps_s.mkdir()
    vs = tmp_path / "leg_vs.txt"
    ls = snaps_s / "leg_ss.kv.json.partial"
    _plant_symlink(ls, vs)
    with RunnerServer(runner_bin, str(fx["model"]), ctx=256, env=env,
                      extra_args=["--gpu", "off", "-t", "2",
                                  "--kv-snapshots", str(snaps_s),
                                  "--sign-key", str(fx["key"])]) as srv:
        st, body = _http_post(srv, "/v1/runner/contexts",
                              {"id": "cs", "prompt": "hello world one two three"})
        assert st == 200, body
        st2, b2 = _http_post(srv, "/v1/runner/contexts/cs/snapshot", {"name": "leg_ss"})
        assert st2 == 200, b2
        assert (snaps_s / "leg_ss.kv.json").is_file()
        assert vs.read_text() == "untouched" and ls.is_symlink()


def test_two_concurrent_writers_do_not_collide(runner_bin, fx, tmp_path):
    """Smoke race: two processes writing to the same destination concurrently
    serialize cleanly, match one writer's complete artifact + sidecar, and
    leave no partial or lock files."""
    env = _clean_env()

    expected_q8 = tmp_path / "expected_q8.gguf"
    expected_q4 = tmp_path / "expected_q4.gguf"
    for target_name, out_p in (("q8_0", expected_q8), ("q4_0", expected_q4)):
        ref = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                              "--quantize", str(out_p), "--quant", target_name],
                             cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
        assert ref.returncode == 0, ref.stderr
    want_bytes = {
        "q8_0": expected_q8.read_bytes(),
        "q4_0": expected_q4.read_bytes(),
    }
    want_sha = {
        k: hashlib.sha256(v).hexdigest() for k, v in want_bytes.items()
    }

    for trial in range(4):
        qout = tmp_path / f"shared_{trial}.gguf"
        cmds = [
            [str(runner_bin), "-m", str(fx["model"]), "--quantize", str(qout), "--quant", "q8_0"],
            [str(runner_bin), "-m", str(fx["model"]), "--quantize", str(qout), "--quant", "q4_0"],
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(subprocess.run, c, cwd=ROOT, capture_output=True,
                                text=True, timeout=120, env=env)
                    for c in cmds]
            results = [f.result() for f in futs]
        assert all(r.returncode == 0 for r in results), [r.stderr for r in results]
        rec = json.loads(pathlib.Path(str(qout) + ".quant.json").read_text())
        tgt = rec["target"]
        assert tgt in want_bytes, rec
        got_bytes = qout.read_bytes()
        assert got_bytes == want_bytes[tgt]
        assert rec["output"]["sha256"] == want_sha[tgt] == hashlib.sha256(got_bytes).hexdigest()
        assert not list(tmp_path.glob(f"shared_{trial}.gguf.partial*"))
        assert not list(tmp_path.glob(f"shared_{trial}.gguf.quant.json.partial*"))
        assert not list(tmp_path.glob(f"shared_{trial}.gguf.quant.json.bak*"))
        assert not pathlib.Path(str(qout) + ".publish.lock").exists()

    for trial in range(4):
        tout = tmp_path / f"shared_{trial}.json"
        cmd = [str(runner_bin), "-m", str(fx["model"]), "-p", "hello", "-n", "2",
               "--temp", "0", "--gpu", "off", "-t", "2", "--transcript", str(tout)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(subprocess.run, cmd, cwd=ROOT, capture_output=True,
                                text=True, timeout=120, env=env)
                    for _ in range(2)]
            results = [f.result() for f in futs]
        if sys.platform != "win32":
            assert all(r.returncode == 0 for r in results), [r.stderr for r in results]
        else:
            assert any(r.returncode == 0 for r in results), [r.stderr for r in results]
            for r in results:
                if r.returncode != 0:
                    assert "error:" in r.stderr.lower(), r.stderr
        parsed = json.loads(tout.read_text())
        assert parsed["schema_version"].startswith("xyntetik.")
        assert not list(tmp_path.glob(f"shared_{trial}.json.partial*"))


def _wait_file(path, timeout=30.0):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {path}")
        time.sleep(0.02)


def test_failed_pair_install_preserves_old_pair_and_keeps_temp(runner_bin, fx, tmp_path):
    """R6.112: when replacing the provenance sidecar or output fails during
    publish, the existing (output, record) pair stays intact, no mismatched pair
    is ever left, and the new output stays under its temporary path named in stderr."""
    env = _clean_env()

    ref_q4 = tmp_path / "ref_q4.gguf"
    ref = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                          "--quantize", str(ref_q4), "--quant", "q4_0"],
                         cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert ref.returncode == 0, ref.stderr
    expected_q4 = ref_q4.read_bytes()

    # Case 1 (portable): sidecar destination blocked by non-empty dir on fresh target ->
    # output must NOT be installed; kept partial matches expected_q4 and is named.
    d1 = tmp_path / "blocked_fresh"
    d1.mkdir()
    q1 = d1 / "q.gguf"
    blocker = d1 / "q.gguf.quant.json"
    blocker.mkdir()
    (blocker / "keep").write_text("x")
    r1 = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                         "--quantize", str(q1), "--quant", "q4_0"],
                        cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert r1.returncode != 0, r1.stderr
    assert not q1.exists(), "output must not be installed when sidecar fails"
    kept1 = list(d1.glob("q.gguf.partial-*"))
    assert len(kept1) == 1, kept1
    assert kept1[0].name in r1.stderr, r1.stderr
    assert kept1[0].read_bytes() == expected_q4
    assert not list(d1.glob("q.gguf.quant.json.partial*"))
    assert not list(d1.glob("q.gguf.quant.json.bak*"))
    assert not (d1 / "q.gguf.publish.lock").exists()

    # Case 2 (macOS uchg probe from second read): immutable existing .quant.json
    # keeps OLD pair intact and preserves new partial on disk.
    if sys.platform == "darwin" and os.geteuid() != 0:
        d2 = tmp_path / "uchg_sidecar"
        d2.mkdir()
        q2 = d2 / "q.gguf"
        rec2 = d2 / "q.gguf.quant.json"
        r_init = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                                 "--quantize", str(q2), "--quant", "q8_0"],
                                cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
        assert r_init.returncode == 0, r_init.stderr
        old_out = q2.read_bytes()
        old_rec = rec2.read_bytes()
        subprocess.run(["chflags", "uchg", str(rec2)], check=True)
        try:
            r2 = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                                 "--quantize", str(q2), "--quant", "q4_0"],
                                cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
            assert r2.returncode != 0, r2.stderr
            assert q2.read_bytes() == old_out, "old output must remain intact when sidecar fails"
            assert rec2.read_bytes() == old_rec
            kept2 = list(d2.glob("q.gguf.partial-*"))
            assert len(kept2) == 1 and kept2[0].name in r2.stderr, (kept2, r2.stderr)
            assert kept2[0].read_bytes() == expected_q4
            assert not list(d2.glob("*.bak*"))
            assert not (d2 / "q.gguf.publish.lock").exists()
        finally:
            subprocess.run(["chflags", "nouchg", str(rec2)], check=False)

        # Case 3: immutable existing output q.gguf (Step A succeeds, Step B fails) ->
        # atomic restore of staged backup keeps OLD pair intact + keeps new partial.
        d3 = tmp_path / "uchg_output"
        d3.mkdir()
        q3 = d3 / "q.gguf"
        rec3 = d3 / "q.gguf.quant.json"
        r_init3 = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                                  "--quantize", str(q3), "--quant", "q8_0"],
                                 cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
        assert r_init3.returncode == 0, r_init3.stderr
        old_out3 = q3.read_bytes()
        old_rec3 = rec3.read_bytes()
        subprocess.run(["chflags", "uchg", str(q3)], check=True)
        try:
            r3 = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                                 "--quantize", str(q3), "--quant", "q4_0"],
                                cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
            assert r3.returncode != 0, r3.stderr
            assert q3.read_bytes() == old_out3
            assert rec3.read_bytes() == old_rec3, "old record must be restored cleanly"
            kept3 = list(d3.glob("q.gguf.partial-*"))
            assert len(kept3) == 1 and kept3[0].name in r3.stderr, (kept3, r3.stderr)
            assert kept3[0].read_bytes() == expected_q4
            assert not list(d3.glob("*.bak*"))
            assert not (d3 / "q.gguf.publish.lock").exists()
        finally:
            subprocess.run(["chflags", "nouchg", str(q3)], check=False)


def test_forced_lock_serialization_and_aliases(runner_bin, fx, tmp_path):
    """R6.112: pause seams force lock contention deterministically across
    path aliases (including unreadable 0300 dirs on POSIX and case/rel aliases)."""
    env = _clean_env()

    expected_q8 = tmp_path / "exp_q8.gguf"
    expected_q4 = tmp_path / "exp_q4.gguf"
    for qt, ep in (("q8_0", expected_q8), ("q4_0", expected_q4)):
        r = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                            "--quantize", str(ep), "--quant", qt],
                           cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
        assert r.returncode == 0, r.stderr
    b_q8 = expected_q8.read_bytes()
    b_q4 = expected_q4.read_bytes()

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    target = real_dir / "model.gguf"
    rec = real_dir / "model.gguf.quant.json"
    alias_target = target
    if sys.platform == "win32":
        alias_target = real_dir / "MODEL.GGUF"
    else:
        alias_dir = tmp_path / "alias"
        try:
            os.symlink(real_dir, alias_dir)
            alias_target = alias_dir / "model.gguf"
        except (OSError, NotImplementedError):
            alias_target = real_dir / "." / "model.gguf"

    ctl = tmp_path / "ctl"
    ctl.mkdir()
    w1_ready, w1_go = ctl / "w1.ready", ctl / "w1.go"
    w2_ready, w2_go = ctl / "w2.ready", ctl / "w2.go"

    p1 = subprocess.Popen(
        [str(runner_bin), "-m", str(fx["model"]), "--quantize", str(target), "--quant", "q8_0"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=_clean_env(RUNNER_TEST_PUBLISH_PAUSE_BEFORE=f"{w1_ready}|{w1_go}"),
    )
    p2 = None
    try:
        _wait_file(w1_ready)
        p2 = subprocess.Popen(
            [str(runner_bin), "-m", str(fx["model"]), "--quantize", str(alias_target), "--quant", "q4_0"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=_clean_env(RUNNER_TEST_PUBLISH_PAUSE_BEFORE=f"{w2_ready}|{w2_go}"),
        )
        time.sleep(0.35)
        assert not w2_ready.exists(), "writer 2 bypassed publish lock on alias"
        assert not target.exists() and not rec.exists()

        w1_go.write_text("go")
        _wait_file(w2_ready)
        assert target.read_bytes() == b_q8
        assert json.loads(rec.read_text())["target"] == "q8_0"

        w2_go.write_text("go")
        _, e1 = p1.communicate(timeout=60)
        _, e2 = p2.communicate(timeout=60)
        assert p1.returncode == 0, e1
        assert p2.returncode == 0, e2
        assert target.read_bytes() == b_q4
        parsed = json.loads(rec.read_text())
        assert parsed["target"] == "q4_0"
        assert parsed["output"]["sha256"] == hashlib.sha256(b_q4).hexdigest()
        assert not (real_dir / "model.gguf.publish.lock").exists()
    finally:
        w1_go.write_text("go")
        w2_go.write_text("go")
        for p in (p1, p2):
            if p and p.poll() is None:
                p.kill()
                p.wait()

    # POSIX unreadable directory (mode 0300): lock still serializes via lock file
    if sys.platform != "win32" and os.geteuid() != 0:
        d300 = tmp_path / "mode300"
        d300.mkdir()
        t300 = d300 / "out.gguf"
        ctl300 = tmp_path / "ctl300"
        ctl300.mkdir()
        a_ready, a_go = ctl300 / "a.ready", ctl300 / "a.go"
        b_ready, b_go = ctl300 / "b.ready", ctl300 / "b.go"
        d300.chmod(0o300)
        pa = pb = None
        try:
            pa = subprocess.Popen(
                [str(runner_bin), "-m", str(fx["model"]), "--quantize", str(t300), "--quant", "q8_0"],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=_clean_env(RUNNER_TEST_PUBLISH_PAUSE_BEFORE=f"{a_ready}|{a_go}"),
            )
            _wait_file(a_ready)
            pb = subprocess.Popen(
                [str(runner_bin), "-m", str(fx["model"]), "--quantize", str(t300), "--quant", "q4_0"],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=_clean_env(RUNNER_TEST_PUBLISH_PAUSE_BEFORE=f"{b_ready}|{b_go}"),
            )
            time.sleep(0.35)
            assert not b_ready.exists(), "mode 0300 directory must not bypass publish lock"
            a_go.write_text("go")
            _wait_file(b_ready)
            b_go.write_text("go")
            _, ea = pa.communicate(timeout=60)
            _, eb = pb.communicate(timeout=60)
            assert pa.returncode == 0 and pb.returncode == 0, (ea, eb)
        finally:
            a_go.write_text("go")
            b_go.write_text("go")
            for p in (pa, pb):
                if p and p.poll() is None:
                    p.kill()
                    p.wait()
            d300.chmod(0o755)
        assert t300.read_bytes() == b_q4
        assert json.loads((d300 / "out.gguf.quant.json").read_text())["target"] == "q4_0"


def test_crash_between_record_and_output_is_detected_and_recoverable(runner_bin, fx, tmp_path):
    """R6.112: a hard kill in the narrow crash window between Step A (record)
    and Step B (output) leaves NEW record beside OLD output; --lineage on
    either detects BROKEN instead of trusting it, and a subsequent writer
    reuses any leftover unlocked .publish.lock cleanly."""
    env = _clean_env()
    qout = tmp_path / "crash.gguf"
    rec = tmp_path / "crash.gguf.quant.json"
    r0 = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                         "--quantize", str(qout), "--quant", "q8_0"],
                        cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert r0.returncode == 0, r0.stderr
    q8_bytes = qout.read_bytes()

    ctl = tmp_path / "ctl"
    ctl.mkdir()
    mid_ready, mid_go = ctl / "mid.ready", ctl / "mid.go"
    p = subprocess.Popen(
        [str(runner_bin), "-m", str(fx["model"]), "--quantize", str(qout), "--quant", "q4_0"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=_clean_env(RUNNER_TEST_PUBLISH_PAUSE_BETWEEN=f"{mid_ready}|{mid_go}"),
    )
    try:
        _wait_file(mid_ready)
        p.kill()
        p.wait(timeout=15)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()

    # Old output still on disk, new record installed -> hash mismatch detected
    assert qout.read_bytes() == q8_bytes
    assert json.loads(rec.read_text())["target"] == "q4_0"
    for arg in (str(qout), str(rec)):
        lin = subprocess.run([str(runner_bin), "--lineage", arg],
                             cwd=ROOT, capture_output=True, text=True, timeout=60, env=env)
        assert lin.returncode == 2, (arg, lin.returncode, lin.stdout, lin.stderr)
        assert "BROKEN" in lin.stdout, lin.stdout

    # Next writer reuses the crashed lock file cleanly and restores matching pair
    r_next = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                             "--quantize", str(qout), "--quant", "q8_0"],
                            cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert r_next.returncode == 0, r_next.stderr
    assert qout.read_bytes() == q8_bytes
    assert json.loads(rec.read_text())["target"] == "q8_0"
    assert not (tmp_path / "crash.gguf.publish.lock").exists()
    lin_ok = subprocess.run([str(runner_bin), "--lineage", str(qout)],
                            cwd=ROOT, capture_output=True, text=True, timeout=60, env=env)
    assert lin_ok.returncode in (0, 1), (lin_ok.returncode, lin_ok.stdout, lin_ok.stderr)
    assert "BROKEN" not in lin_ok.stdout, lin_ok.stdout


def test_merge_lora_and_context_surgery_blocked_sidecar_keeps_temp(runner_bin, fx, tmp_path):
    """R6.112: --merge-lora (.merge.json) and --context-surgery (.context.json)
    share plat_publish_pair; a blocked sidecar preserves destination and keeps
    the staged .partial-* output named in stderr."""
    env = _clean_env()

    # 1. --merge-lora blocked sidecar
    dm = tmp_path / "merge"
    dm.mkdir()
    adapter = dm / "adapter.gguf"
    r_tr = subprocess.run(
        [str(runner_bin), "-m", str(fx["model"]), "--train", str(fx["data"]),
         "--train-steps", "1", "--lr", "3e-3", "--train-batch", "2",
         "--gpu", "off", "-t", "2", "--train-out", str(adapter)],
        cwd=ROOT, capture_output=True, text=True, timeout=120, env=env,
    )
    assert r_tr.returncode == 0, r_tr.stderr
    mout = dm / "merged.gguf"
    mblock = dm / "merged.gguf.merge.json"
    mblock.mkdir()
    (mblock / "keep").write_text("x")
    r_m = subprocess.run(
        [str(runner_bin), "-m", str(fx["model"]), "--lora", str(adapter),
         "--merge-lora", str(mout), "--quant", "keep", "--gpu", "off", "-t", "2"],
        cwd=ROOT, capture_output=True, text=True, timeout=120, env=env,
    )
    assert r_m.returncode != 0, r_m.stderr
    assert not mout.exists()
    kept_m = list(dm.glob("merged.gguf.partial-*"))
    assert len(kept_m) == 1 and kept_m[0].name in r_m.stderr, (kept_m, r_m.stderr)
    assert not list(dm.glob("merged.gguf.merge.json.partial*"))
    assert not (dm / "merged.gguf.publish.lock").exists()

    # 2. --context-surgery blocked sidecar
    dc = tmp_path / "ctx"
    dc.mkdir()
    ymodel = dc / "yarn.gguf"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/make-test-model.py"),
         "--yarn", "32,8", str(ymodel)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL,
    )
    cout = dc / "ctx.gguf"
    cblock = dc / "ctx.gguf.context.json"
    cblock.mkdir()
    (cblock / "keep").write_text("x")
    r_c = subprocess.run(
        [str(runner_bin), "-m", str(ymodel), "--context-surgery", str(cout),
         "-c", "512", "--yarn-factor", "64", "--gpu", "off"],
        cwd=ROOT, capture_output=True, text=True, timeout=120, env=env,
    )
    assert r_c.returncode != 0, r_c.stderr
    assert not cout.exists()
    kept_c = list(dc.glob("ctx.gguf.partial-*"))
    assert len(kept_c) == 1 and kept_c[0].name in r_c.stderr, (kept_c, r_c.stderr)
    assert not list(dc.glob("ctx.gguf.context.json.partial*"))
    assert not (dc / "ctx.gguf.publish.lock").exists()


def test_concurrent_kv_snapshot_same_name_serializes_pair(runner_bin, fx, tmp_path):
    """R6.112: concurrent KV snapshot saves across processes under the same name
    serialize on the publish lock; one succeeds (200), one sees destination present
    (409 snapshot_exists), and the surviving (.kv, .kv.json) pair matches & cleans up."""
    import threading
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    env = _clean_env()
    extra = ["--gpu", "off", "-t", "2", "--kv-snapshots", str(snaps),
             "--sign-key", str(fx["key"])]
    barrier = threading.Barrier(2)
    with RunnerServer(runner_bin, str(fx["model"]), ctx=256, env=env, extra_args=extra) as s1, \
         RunnerServer(runner_bin, str(fx["model"]), ctx=256, env=env, extra_args=extra) as s2:
        for s in (s1, s2):
            st, body = _http_post(s, "/v1/runner/contexts",
                                  {"id": "c", "prompt": "hello world one two three"})
            assert st == 200, body

        def _snap(srv):
            barrier.wait(timeout=15)
            return _http_post(srv, "/v1/runner/contexts/c/snapshot", {"name": "race"})

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(_snap, s1), pool.submit(_snap, s2)]
            results = [f.result() for f in futs]
    statuses = sorted(r[0] for r in results)
    assert statuses == [200, 409], results
    err_body = next(b for s, b in results if s == 409)
    assert "exists" in json.dumps(err_body).lower(), err_body
    kv_path = snaps / "race.kv"
    man_path = snaps / "race.kv.json"
    assert kv_path.exists() and man_path.exists()
    man = json.loads(man_path.read_text())
    assert man["kv_sha256"] == hashlib.sha256(kv_path.read_bytes()).hexdigest()
    pk = json.loads(fx["key"].read_text())["public_key"]
    chk = subprocess.run([str(runner_bin), "--check-record", str(man_path),
                          "--trust-key", pk],
                         cwd=ROOT, capture_output=True, text=True, timeout=60, env=env)
    assert chk.returncode == 0, (chk.stdout, chk.stderr)
    assert not list(snaps.glob("*.partial*"))
    assert not list(snaps.glob("*.bak*"))
    assert not (snaps / "race.kv.publish.lock").exists()
