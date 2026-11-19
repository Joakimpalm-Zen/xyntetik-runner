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
    orig_bytes = fx["rec"].read_bytes()
    rec2.write_bytes(orig_bytes)
    v2 = d2 / "victim.txt"
    _plant_symlink(d2 / "rec.json.partial-fixed", v2)
    r = subprocess.run([str(runner_bin), "--sign-record", str(rec2),
                        "--sign-key", str(fx["key"])],
                       cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode != 0, r.stderr
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


def test_two_concurrent_writers_do_not_collide(runner_bin, fx, tmp_path):
    """Two processes writing to the same destination concurrently pick distinct
    random partial names and stage/install output and sidecar together:

    - at least one writer exits 0 (on POSIX both do);
    - any writer that does not win exits non-zero with a clear error message;
    - zero .partial* files remain on disk;
    - the output file is one writer's complete valid artifact, and its
      provenance sidecar matches those exact bytes (output.sha256 == sha256(out)
      and target matches the winning writer even across competing targets).
    """
    env = _clean_env()

    # Include mixed-target races (q8_0 vs q4_0) so a cross-writer swap between
    # the GGUF and .quant.json sidecar would fail sha256 and target parity.
    quant_pairs = [("q8_0", "q8_0"), ("q4_0", "q8_0"), ("q8_0", "q4_0"), ("q8_0", "q8_0")]
    for trial, (q0, q1) in enumerate(quant_pairs):
        qout = tmp_path / f"shared_{trial}.gguf"
        cmds = [[str(runner_bin), "-m", str(fx["model"]),
                 "--quantize", str(qout), "--quant", q] for q in (q0, q1)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(subprocess.run, cmd, cwd=ROOT, capture_output=True,
                                text=True, timeout=120, env=env)
                    for cmd in cmds]
            results = [f.result() for f in futs]
        assert any(r.returncode == 0 for r in results), [r.stderr for r in results]
        for r in results:
            if r.returncode != 0:
                assert "error:" in r.stderr.lower(), r.stderr
        assert qout.is_file() and qout.stat().st_size > 0
        assert not list(tmp_path.glob(f"shared_{trial}.gguf.partial*"))
        assert not list(tmp_path.glob(f"shared_{trial}.gguf.quant.json.partial*"))
        qrec = pathlib.Path(f"{qout}.quant.json")
        assert qrec.is_file(), "winning quantize writer must leave matching .quant.json"
        rec = json.loads(qrec.read_text())
        assert rec["schema_version"] == "xyntetik.runner.quant.v1"
        actual_sha = hashlib.sha256(qout.read_bytes()).hexdigest()
        assert rec["output"]["sha256"] == actual_sha, (rec, actual_sha)
        assert rec["target"] in (q0, q1)
        chk = subprocess.run([str(runner_bin), "-m", str(qout), "-p", "hi",
                              "-n", "1", "--temp", "0", "--gpu", "off", "-t", "2"],
                             cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
        assert chk.returncode == 0, chk.stderr

    for trial in range(4):
        tout = tmp_path / f"shared_{trial}.json"
        cmd = [str(runner_bin), "-m", str(fx["model"]), "-p", "hello", "-n", "2",
               "--temp", "0", "--gpu", "off", "-t", "2", "--transcript", str(tout)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(subprocess.run, cmd, cwd=ROOT, capture_output=True,
                                text=True, timeout=120, env=env)
                    for _ in range(2)]
            results = [f.result() for f in futs]
        assert any(r.returncode == 0 for r in results), [r.stderr for r in results]
        for r in results:
            if r.returncode != 0:
                assert "error:" in r.stderr.lower(), r.stderr
        parsed = json.loads(tout.read_text())
        assert parsed["schema_version"].startswith("xyntetik.")
        assert not list(tmp_path.glob(f"shared_{trial}.json.partial*"))
