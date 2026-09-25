"""Reasoning-channel sampling: the reasoning turn and the answer are different
jobs (R4.12.17).

A reasoning turn that loops needs randomness to escape it, and looping is
worst under greedy decoding (arXiv 2512.12895); a tool call's arguments must
stay deterministic, and a flat penalty across both garbles the call (R4.12.7,
0 of 8 sampled calls). So `reasoning_temperature` and its three companions
replace the sampler's knobs ONLY while the turn is a reasoning turn, and only
when a caller asks for them.

The load-bearing property is what happens when nobody asks: every existing
request, greedy included, must be byte-identical. The gate harnesses that
measure this model send temperature 0 explicitly, and the KLD tooling depends
on greedy meaning argmax.
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
def muse_model(tmp_path_factory):
    m = tmp_path_factory.mktemp("rsamp") / "muse.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--muse-glimmer",
                    "--control", "<|start|>,<|message|>,<|eom|>,<|eot|>", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


@pytest.fixture(scope="module")
def plain_model(tmp_path_factory):
    m = tmp_path_factory.mktemp("rsamp") / "plain.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


def _serve(model, extra=()):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    return RunnerServer(exe, model, ctx=2048, parallel=1,
                        extra_args=["--gpu", "off", "-t", "2", *extra])


@pytest.fixture(scope="module")
def muse(muse_model):
    with _serve(muse_model) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            srv.model_id = json.load(r)["data"][0]["id"]
        yield srv


@pytest.fixture(scope="module")
def muse_default_temp(muse_model):
    with _serve(muse_model, ["--reasoning-temp", "1.5"]) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            srv.model_id = json.load(r)["data"][0]["id"]
        yield srv


@pytest.fixture(scope="module")
def plain(plain_model):
    with _serve(plain_model) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            srv.model_id = json.load(r)["data"][0]["id"]
        yield srv


def _chat(server, **extra):
    payload = {"model": server.model_id, "max_tokens": 40, "temperature": 0,
               "enable_thinking": True,
               "messages": [{"role": "user", "content": "which is warmer"}]}
    payload.update(extra)
    req = urllib.request.Request(server.base_url + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _msg(body):
    return body["choices"][0]["message"]


def test_a_greedy_request_is_untouched_and_reproducible(muse):
    """The property every gate arm and the KLD tooling depends on."""
    a = _chat(muse)
    b = _chat(muse)
    assert a[0] == 200 and b[0] == 200
    assert _msg(a[1]) == _msg(b[1])


def test_the_reasoning_channel_takes_its_own_temperature(muse):
    base = _msg(_chat(muse)[1]).get("reasoning_content") or ""
    code, body = _chat(muse, reasoning_temperature=1.6, seed=11)
    assert code == 200, body
    hot = _msg(body).get("reasoning_content") or ""
    assert hot != base, "reasoning_temperature did not reach the reasoning turn"
    # a second seed differs again: it is sampling, not one fixed perturbation
    code, other = _chat(muse, reasoning_temperature=1.6, seed=29)
    assert code == 200
    assert (_msg(other).get("reasoning_content") or "") != hot


def test_a_zero_reasoning_temperature_is_greedy_again(muse):
    """Asking explicitly for 0 is a determinism request like any other, so it
    must reproduce the untouched run rather than merely be 'low'."""
    base = _msg(_chat(muse)[1])
    code, body = _chat(muse, reasoning_temperature=0)
    assert code == 200, body
    assert _msg(body) == base


def test_the_server_default_applies_and_a_request_can_override_it(muse_default_temp):
    hot = _msg(_chat(muse_default_temp, seed=3)[1]).get("reasoning_content") or ""
    cold = _msg(_chat(muse_default_temp, reasoning_temperature=0)[1]).get("reasoning_content") or ""
    assert hot != cold
    # and the override to 0 is reproducible, i.e. it really is greedy
    again = _msg(_chat(muse_default_temp, reasoning_temperature=0)[1]).get("reasoning_content") or ""
    assert again == cold


@pytest.mark.parametrize("bad", [
    {"reasoning_temperature": 3},
    {"reasoning_temperature": -1},
    {"reasoning_temperature": "hot"},
    {"reasoning_temperature": 0.5, "reasoning_top_k": -1},
    {"reasoning_temperature": 0.5, "reasoning_top_p": 2},
    {"reasoning_temperature": 0.5, "reasoning_top_k": 1.5},
])
def test_out_of_range_values_are_refused(muse, bad):
    code, body = _chat(muse, **bad)
    assert code == 400, (bad, body)
    assert "reasoning_" in json.dumps(body)


def test_a_model_without_a_reasoning_channel_refuses_the_knob(plain):
    """Refused rather than ignored: a knob that silently does nothing has the
    caller reading an unchanged trace as the setting's effect."""
    req = urllib.request.Request(
        plain.base_url + "/v1/completions",
        data=json.dumps({"model": plain.model_id, "prompt": "hi", "max_tokens": 4,
                         "temperature": 0, "reasoning_temperature": 1.0}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raise AssertionError(f"accepted with {r.status}")
    except urllib.error.HTTPError as e:
        assert e.code == 400
        assert "reasoning channel" in json.loads(e.read())["error"]["message"]
