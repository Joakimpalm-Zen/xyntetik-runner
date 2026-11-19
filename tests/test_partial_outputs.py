"""Planted-symlink and concurrent-writer checks for R6.101 partial outputs.

Every atomic output opens a new temporary file via plat_fopen_partial()
(O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW on POSIX; CREATE_NEW on Windows) with an
unpredictable 16-hex suffix (fixed only by RUNNER_TEST_PARTIAL_SUFFIX in tests),
and unlinks only the path it created if an error follows.
"""
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys

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
    """Two processes writing to the same destination concurrently pick distinct
    random partial names, never truncate each other, leave zero partial files,
    and produce one writer's complete artifact (on Windows a writer that loses
    an install/read race on the shared destination may fail cleanly)."""
    env = _clean_env()

    expected_gguf = tmp_path / "expected.gguf"
    ref = subprocess.run([str(runner_bin), "-m", str(fx["model"]),
                          "--quantize", str(expected_gguf), "--quant", "q8_0"],
                         cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert ref.returncode == 0, ref.stderr
    expected_bytes = expected_gguf.read_bytes()

    for trial in range(4):
        qout = tmp_path / f"shared_{trial}.gguf"
        cmd = [str(runner_bin), "-m", str(fx["model"]),
               "--quantize", str(qout), "--quant", "q8_0"]
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
        assert qout.read_bytes() == expected_bytes
        assert not list(tmp_path.glob(f"shared_{trial}.gguf.partial*"))

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
