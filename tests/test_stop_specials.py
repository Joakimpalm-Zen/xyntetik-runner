"""A stop that names a special token ends the turn there.

Control tokens decode to no bytes, so a string stop such as `<|eom|>` could
never match what the client received, and a Muse to=self turn ran straight
through `<|eom|><|start|>assistant to=<tool>` with the specials silently
removed (the lab, 2026-09-16). Now a stop string that spells a control token
exactly stops on that token, and `stop_token_ids` names ids directly, the
vLLM spelling. Driven over HTTP with the scripted-reply hook on the CI
fixture, whose `<s>` (id 1) is a control token that is not an engine stop.
"""
import json
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import RunnerServer, find_runner  # noqa: E402


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("stops") / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


@pytest.fixture(scope="module")
def server(model):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    with RunnerServer(exe, model, ctx=512, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2"],
                      env={"RUNNER_TEST_SCRIPTED_REPLY": "1"}) as srv:
        yield srv


def _post(server, payload):
    req = urllib.request.Request(server.base_url + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


BASE = {"prompt": "hi", "max_tokens": 16, "temperature": 0,
        "runner_test_reply": "Hello<s>tail"}


def test_a_control_token_is_invisible_in_the_text(server):
    status, d = _post(server, dict(BASE))
    assert status == 200
    assert d["choices"][0]["text"] == "Hellotail"       # `<s>` decodes to nothing
    assert d["choices"][0]["finish_reason"] == "stop"


def test_a_stop_string_that_spells_a_control_token_stops_there(server):
    status, d = _post(server, dict(BASE, stop=["<s>"]))
    assert status == 200, d
    assert d["choices"][0]["text"] == "Hello"
    assert d["choices"][0]["finish_reason"] == "stop"


def test_stop_token_ids_stop_there(server):
    status, d = _post(server, dict(BASE, stop_token_ids=[1]))
    assert status == 200, d
    assert d["choices"][0]["text"] == "Hello"
    assert d["choices"][0]["finish_reason"] == "stop"


def test_stop_token_ids_are_validated(server):
    for bad in ([-1], [10 ** 9], ["1"], "1", [1] * 9):
        status, d = _post(server, dict(BASE, stop_token_ids=bad))
        assert status == 400, (bad, d)
        assert "stop_token_ids" in json.dumps(d)


def test_an_ordinary_stop_string_still_matches_text(server):
    status, d = _post(server, dict(BASE, stop=["ta"]))
    assert status == 200, d
    assert d["choices"][0]["text"] == "Hello"
