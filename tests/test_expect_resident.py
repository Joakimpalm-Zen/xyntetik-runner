"""`expect_resident`: a request that says what it expects to find loaded is
served by that load or refused, and a refusal leaves the server as it was.

A client of an externally managed server can read `/health` and then send a
request, but an unload, an idle expiry or a swap in between turns that
request into one that loads a model. `/health` and the provenance record
carry `load_generation`, which moves on every load; a request can require
it, or the model file's sha256, and the check runs under the lock every
load, unload and swap takes.
"""
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import RunnerServer, find_runner  # noqa: E402


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    out = tmp_path_factory.mktemp("expect") / "m.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(out)],
                   check=True, stdout=subprocess.DEVNULL)
    return out


def _get(srv, path):
    with urllib.request.urlopen(srv.base_url + path, timeout=30) as r:
        return json.load(r)


def _post(srv, path, body):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def _ask(srv, expect=None, path="/v1/completions"):
    body = {"prompt": "hi", "max_tokens": 2, "temperature": 0}
    if path == "/v1/chat/completions":
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2,
                "temperature": 0}
    if expect is not None:
        body["expect_resident"] = expect
    return _post(srv, path, body)


def _server(model, parallel=1):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    return RunnerServer(exe, model, ctx=256, parallel=parallel,
                        extra_args=["--gpu", "off", "-t", "2"])


def test_a_refusal_never_loads_and_the_generation_names_the_load(model):
    with _server(model) as srv:
        assert _get(srv, "/health")["load_generation"] == 1
        assert _get(srv, "/v1/runner/provenance")["model"]["load_generation"] == 1
        assert _ask(srv, {"load_generation": 1})[0] == 200
        code, err = _ask(srv, {"load_generation": 2})
        assert code == 409 and err["error"]["code"] == "resident_mismatch"

        # unloaded: an expecting request is refused and nothing comes back
        req = urllib.request.Request(srv.base_url + "/unload", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            assert r.status == 200
        h = _get(srv, "/health")
        assert h["resident"] is None and h["load_generation"] == 1
        code, err = _ask(srv, {"load_generation": 1}, "/v1/chat/completions")
        assert code == 409 and err["error"]["code"] == "resident_mismatch"
        h = _get(srv, "/health")
        assert h["resident"] is None and h["load_generation"] == 1

        # an ordinary request reloads, and that is a new generation
        assert _ask(srv)[0] == 200
        assert _get(srv, "/health")["load_generation"] == 2
        assert _ask(srv, {"load_generation": 1})[0] == 409
        assert _ask(srv, {"load_generation": 2})[0] == 200


def test_the_digest_is_checked_once_it_is_known(model):
    with _server(model) as srv:
        sha = None
        for _ in range(100):
            rec = _get(srv, "/v1/runner/provenance")["model"]
            if rec["sha256_state"] == "done":
                sha = rec["sha256"]
                break
            time.sleep(0.05)
        assert sha, "the model digest never became known"
        assert _ask(srv, {"model_sha256": sha})[0] == 200
        assert _ask(srv, {"model_sha256": sha, "load_generation": 1})[0] == 200
        code, err = _ask(srv, {"model_sha256": "0" * 64})
        assert code == 409 and err["error"]["code"] == "resident_mismatch"


@pytest.mark.parametrize("bad", [{}, [], "x", {"load_generation": 0},
                                 {"load_generation": 1.5},
                                 {"load_generation": "1"},
                                 {"model_sha256": "abc"},
                                 {"model_sha256": "A" * 64}])
def test_a_malformed_expectation_is_a_400(model, bad):
    with _server(model) as srv:
        code, err = _ask(srv, bad)
        assert code == 400 and err["error"]["param"] == "expect_resident"


def test_a_multi_slot_server_checks_it_too(model):
    """Its slots hold the model directly: one load for the process's life."""
    with _server(model, parallel=2) as srv:
        assert _get(srv, "/health")["load_generation"] == 1
        assert _ask(srv, {"load_generation": 1})[0] == 200
        assert _ask(srv, {"load_generation": 2})[0] == 409
