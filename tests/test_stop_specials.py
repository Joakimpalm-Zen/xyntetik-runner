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


# ---- the two 2026-09-22 lab papercuts: render specials on request, and say
# ---- which terminator ended the turn

def _stream(server, payload):
    req = urllib.request.Request(server.base_url + "/v1/completions",
                                 data=json.dumps(dict(payload, stream=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read().decode()
    chunks = []
    for line in body.splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            chunks.append(json.loads(line[6:]))
    return chunks


def test_special_tokens_renders_the_control_token_in_text_and_logprobs(server):
    status, d = _post(server, dict(BASE, special_tokens=True, logprobs=1))
    assert status == 200, d
    c = d["choices"][0]
    assert c["text"] == "Hello<s>tail"
    lp = c["logprobs"]
    i = lp["token_ids"].index(1)
    assert lp["tokens"][i] == "<s>"
    # text_offset follows the rendered text: the token after `<s>` starts
    # past its spelling, not at the same byte
    assert lp["text_offset"][i] == len("Hello")
    assert lp["text_offset"][i + 1] == len("Hello<s>")


def test_without_the_flag_the_control_token_renders_empty(server):
    status, d = _post(server, dict(BASE, logprobs=1))
    assert status == 200, d
    c = d["choices"][0]
    assert c["text"] == "Hellotail"
    lp = c["logprobs"]
    i = lp["token_ids"].index(1)
    assert lp["tokens"][i] == ""
    assert lp["text_offset"][i + 1] == len("Hello")


def test_special_tokens_is_a_completions_extension(server):
    req = urllib.request.Request(
        server.base_url + "/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": "hi"}],
                         "max_tokens": 4, "special_tokens": True,
                         "runner_test_reply": "ok"}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            status, d = r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        status, d = e.code, json.loads(e.read())
    assert status == 400
    assert "special_tokens" in json.dumps(d)
    status, d = _post(server, dict(BASE, special_tokens="yes"))
    assert status == 400 and "special_tokens" in json.dumps(d)


def test_special_tokens_refuses_a_constraint(server):
    status, d = _post(server, dict(BASE, special_tokens=True,
                                   response_format={"type": "json_object"}))
    assert status == 400
    assert "special_tokens" in json.dumps(d)


def test_the_stop_token_is_reported_beside_finish_reason(server):
    status, d = _post(server, dict(BASE, stop_token_ids=[1]))
    assert status == 200, d
    c = d["choices"][0]
    assert c["finish_reason"] == "stop"
    assert c["stop_token_id"] == 1
    assert c["stop_token"] == "<s>"


def test_a_stop_string_reports_no_stop_token(server):
    status, d = _post(server, dict(BASE, stop=["ta"]))
    assert status == 200, d
    c = d["choices"][0]
    assert c["finish_reason"] == "stop"
    assert "stop_token_id" not in c and "stop_token" not in c
    # a scripted reply that simply ran out ended on no terminator either
    status, d = _post(server, dict(BASE))
    assert "stop_token_id" not in d["choices"][0]


def test_the_final_chunk_carries_the_stop_token(server):
    chunks = _stream(server, dict(BASE, stop_token_ids=[1]))
    finals = [ch["choices"][0] for ch in chunks
              if ch["choices"] and ch["choices"][0].get("finish_reason")]
    assert len(finals) == 1, chunks
    assert finals[0]["finish_reason"] == "stop"
    assert finals[0]["stop_token_id"] == 1
    assert finals[0]["stop_token"] == "<s>"
    text = "".join(ch["choices"][0].get("text", "") for ch in chunks if ch["choices"])
    assert text == "Hello"


def test_a_stop_as_the_first_token_still_reports_its_distribution(server):
    """A stop token is a decision, and on a fidelity comparison it is the
    interesting one: a quantisation that flips "stop here" against "keep
    going" changes where generation ends. The logprobs arrays align with the
    emitted TEXT and a stop decodes to no bytes, so the stop's own decision
    rides beside stop_token instead. Before this it was reported nowhere, and
    kld-compare-raw dropped the position as failed (lab, 2026-09-25)."""
    code, body = _post(server, {**BASE, "runner_test_reply": "<s>tail",
                                "logprobs": 3, "stop_token_ids": [1]})
    assert code == 200, body
    ch = body["choices"][0]
    assert ch["finish_reason"] == "stop" and ch["stop_token_id"] == 1
    sl = ch.get("stop_logprobs")
    assert sl is not None, ch
    assert isinstance(sl["logprob"], float)
    assert sl["top_logprobs"] and len(sl["top_token_ids"]) == len(sl["top_logprobs"])
    # the aligned arrays are untouched: no entry for a token with no bytes
    assert not (ch.get("logprobs") or {}).get("tokens")


def test_the_stop_distribution_is_absent_when_logprobs_were_not_asked_for(server):
    code, body = _post(server, {**BASE, "runner_test_reply": "<s>tail",
                                "stop_token_ids": [1]})
    assert code == 200, body
    ch = body["choices"][0]
    assert ch["stop_token_id"] == 1 and "stop_logprobs" not in ch


def test_text_before_a_stop_keeps_its_own_logprobs_aligned(server):
    code, body = _post(server, {**BASE, "runner_test_reply": "Hello<s>tail",
                                "logprobs": 3, "stop_token_ids": [1]})
    assert code == 200, body
    ch = body["choices"][0]
    lp = ch["logprobs"]
    # one entry per emitted text token, and the stop is not among them
    assert len(lp["tokens"]) == len(lp["token_logprobs"]) == len(lp["top_logprobs"])
    assert "".join(lp["tokens"]) == ch["text"]
    assert ch["stop_logprobs"]["logprob"] <= 0.0
