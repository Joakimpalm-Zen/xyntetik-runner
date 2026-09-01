"""Public-interface gates for scripts/kv-quality.py."""

import importlib.util
import pathlib
import socket
import subprocess
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "kv-quality.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("runner_kv_quality", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("kv-quality") / "test.gguf"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-model.py", path],
        check=True,
        cwd=ROOT,
    )
    return path


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_server_uses_the_model_id_advertised_by_runner(runner_bin, model):
    """The harness must not assume a placeholder model id is accepted."""
    kv_quality = _load_script()
    registered = "kv-quality-anchor=%s" % model

    with kv_quality.Server(
        runner_bin, registered, ctx=64, kv="f16", port=_free_port(), gpu="off"
    ) as server:
        count = server.count_tokens([{"role": "user", "content": "hello"}])

    # Independent anchor: a non-empty request has at least one token.  The
    # explicit registry alias also proves the id came from GET /v1/models,
    # rather than being guessed from the path basename.
    assert count >= 1
