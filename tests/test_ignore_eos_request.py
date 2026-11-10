"""ignore_eos on a completion request (R10.31), as vLLM's and llama.cpp's servers take it.

The fixture's model emits its end-of-text token first and always, so a turn
without ignore_eos stops at once and one with it runs to max_tokens: the
fixed-length rollout a trainer asks for. The server's --ignore-eos is the
default, a request may turn it either way, and a request's choice does not
outlive the request: each completion installs its own value, so a request or
a session before it cannot change the next one. A served ignore_eos turn records the constraint, and
--verify --ignore-eos replays it.
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
from harness import RunnerServer  # noqa: E402


@pytest.fixture(scope="module")
def eos_model(tmp_path_factory):
    m = tmp_path_factory.mktemp("ieos") / "eos.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--suppress-all-but-eos", str(m)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return m


def _post(srv, path, payload):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def _text(srv, **extra):
    return _post(srv, "/v1/completions", dict({"prompt": "the fox", "max_tokens": 6, "temperature": 0,
                                               "cache_prompt": False}, **extra))


def _chat(srv, **extra):
    return _post(srv, "/v1/chat/completions", dict({"messages": [{"role": "user", "content": "hi"}],
                                                    "max_tokens": 6, "temperature": 0,
                                                    "cache_prompt": False}, **extra))


def _ran(body):
    return body["usage"]["completion_tokens"], body["choices"][0]["finish_reason"]


def test_a_request_asks_for_a_fixed_length_turn(runner_bin, eos_model, tmp_path):
    d = tmp_path / "r"
    with RunnerServer(runner_bin, eos_model, ctx=256, extra_args=["--gpu", "off", "-t", "2",
                                                                  "--receipts", str(d)]) as srv:
        for call in (_text, _chat):
            code, body = call(srv)
            assert code == 200 and _ran(body) == (0, "stop"), (call.__name__, body)
            code, body = call(srv, ignore_eos=True)
            assert code == 200 and _ran(body) == (6, "length"), (call.__name__, body)
            # the request's choice is its own: the next request stops again
            code, body = call(srv)
            assert code == 200 and _ran(body) == (0, "stop"), (call.__name__, body)
        # null reads as absent; anything else is refused by name
        code, body = _text(srv, ignore_eos=None)
        assert code == 200 and _ran(body) == (0, "stop"), body
        code, body = _text(srv, ignore_eos="yes")
        assert code == 400 and body["error"]["param"] == "ignore_eos", body
        code, body = _text(srv, ignore_eos=True)
        assert code == 200 and _ran(body) == (6, "length"), body
    # the served turn names ignore_eos, and replays under --ignore-eos only
    rec_path = sorted(d.iterdir())[-1]
    rec = json.loads(rec_path.read_text())
    assert {"kind": "ignore_eos"} in rec["constraints"], rec.get("constraints")

    def verify(*extra):
        v = subprocess.run([runner_bin, "-m", eos_model, "--verify", rec_path, "--gpu", "off", "-t", "2",
                            "-c", "256", *extra], capture_output=True, timeout=120)
        return v.returncode, v.stderr.decode(errors="replace")
    rc, err = verify("--ignore-eos")
    assert rc == 0 and "VERIFIED" in err, err[-600:]
    rc, err = verify()
    assert rc == 3 and "ignore_eos" in err, err[-600:]


def test_the_servers_default_and_a_request_that_turns_it_off(runner_bin, eos_model, tmp_path):
    with RunnerServer(runner_bin, eos_model, ctx=256, extra_args=[
            "--gpu", "off", "-t", "2", "--ignore-eos", "--sessions", str(tmp_path / "s")]) as srv:
        code, body = _text(srv)
        assert code == 200 and _ran(body) == (6, "length"), body
        code, body = _chat(srv, ignore_eos=False)
        assert code == 200 and _ran(body) == (0, "stop"), body
        code, body = _text(srv)
        assert code == 200 and _ran(body) == (6, "length"), body
        # a session that turned it off does not change the next completion: each installs its own
        code, body = _post(srv, "/v1/runner/sessions", {"prompt": "the fox", "max_tokens": 4,
                                                        "temperature": 0, "ignore_eos": False})
        assert code == 200, body
        code, body = _text(srv)
        assert code == 200 and _ran(body) == (6, "length"), body
