"""Family sampling defaults as a client sees them, and the diagnostics that
say what a request was served with.

The 2026-09-14 Windows report found Gemma 4 served under Gemma 3's preset,
which carried a repeat penalty of 1.10 that no Gemma generation_config has;
at the family's own temperature 1.0 that penalty turned every native tool
call into special-token soup, and the only way to see it was an A/B against
explicit request fields. Greedy gates never exercised it: at temperature 0
the penalty is bypassed by contract. So these tests run the family default
(positive temperature) with a fixed seed, and read the effective values back
from `runner_telemetry.sampling`, which is where a client can now see them.

The model is the CI fixture (random weights); the served preset is chosen by
the forced `--chat-template`, exactly as it is for a real checkpoint whose
template detects as that family. Nothing here judges output quality: the
claims are about which knobs applied and whether they were reported.
"""
import json
import pathlib
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import RunnerServer, find_runner  # noqa: E402

# what the publishers' generation_config.json files say (docs/cross-family-remedy-2026-09-14.md)
PUBLISHED = {
    "gemma4":    {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0, "repeat_penalty": 1.0},
    "qwen38":    {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0},
    "granite42": {"temperature": 1.0, "top_p": 0.95, "top_k": 0,  "min_p": 0.0, "repeat_penalty": 1.0},
    "qwen3-coder": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.05},
}
TEMPLATE_FOR = {"gemma4": "gemma4-mainline", "qwen38": "qwen38",
                "granite42": "granite42", "qwen3-coder": "qwen3-coder"}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("sampling") / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


def _server(model, template, extra=()):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    return RunnerServer(exe, model, ctx=2048, parallel=1,
                        extra_args=["--gpu", "off", "-t", "2",
                                    "--chat-template", template, *extra])


def _get(server, path):
    with urllib.request.urlopen(server.base_url + path, timeout=30) as r:
        return json.load(r)


def _post(server, path, payload, stream=False):
    req = urllib.request.Request(server.base_url + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
    if not stream:
        return json.loads(body)
    events = []
    for line in body.decode().splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            events.append(json.loads(line[6:]))
    return events


def _chat(server, model_id, **fields):
    body = {"model": model_id, "max_tokens": 48,
            "messages": [{"role": "user", "content": "Write a limerick."}]}
    body.update(fields)
    return _post(server, "/v1/chat/completions", body)


@pytest.mark.parametrize("preset", sorted(PUBLISHED))
def test_family_preset_matches_the_publisher(model, preset):
    """/v1/capabilities reports the served preset and its five values; they
    are the publisher's, repeat penalty included."""
    with _server(model, TEMPLATE_FOR[preset]) as srv:
        caps = _get(srv, "/v1/capabilities")
        s = caps["sampling"]
        assert s["preset"] == preset, s
        for k, v in PUBLISHED[preset].items():
            assert s[k] == pytest.approx(v, abs=1e-6), (preset, k, s)
        assert caps["template"] == TEMPLATE_FOR[preset], caps
        assert caps["tool_protocol"]["native"] is True, caps


def test_gemma4_default_request_is_served_at_the_published_values(model):
    """A request with no sampling fields, at the family's positive
    temperature: the diagnostics name the preset, every value comes from it,
    and the repeat penalty is 1.0."""
    with _server(model, "gemma4-mainline") as srv:
        mid = _get(srv, "/v1/models")["data"][0]["id"]
        d = _chat(srv, mid)
        t = d["runner_telemetry"]["sampling"]
        assert t["preset"] == "gemma4", t
        for k, v in PUBLISHED["gemma4"].items():
            assert t[k] == pytest.approx(v, abs=1e-6), (k, t)
        assert set(t["source"].values()) == {"preset"}, t
        tp = d["runner_telemetry"]["tool_protocol"]
        assert tp == {"template": "gemma4-mainline", "family": "gemma4",
                      "tools": False, "constrained": False,
                      "parse_only": False}, tp


def test_penalty_is_off_by_default_and_acts_when_asked(model):
    """Fixed seed, positive temperature: the default run equals an explicit
    repeat_penalty 1.0 run token for token, and differs from an explicit
    1.1 run, so the default really is off and the knob really acts. The
    greedy control shows why greedy gates could not see this: at
    temperature 0 the penalty is bypassed, and 1.0 and 1.1 agree."""
    with _server(model, "gemma4-mainline") as srv:
        mid = _get(srv, "/v1/models")["data"][0]["id"]
        default = _chat(srv, mid, seed=7)
        explicit_off = _chat(srv, mid, seed=7, repeat_penalty=1.0)
        explicit_on = _chat(srv, mid, seed=7, repeat_penalty=1.1)
        text = lambda d: d["choices"][0]["message"]["content"]  # noqa: E731
        assert text(default) == text(explicit_off), (text(default), text(explicit_off))
        assert text(default) != text(explicit_on), text(default)
        assert default["runner_telemetry"]["sampling"]["source"]["repeat_penalty"] == "preset"
        assert explicit_on["runner_telemetry"]["sampling"]["source"]["repeat_penalty"] == "request"
        assert explicit_on["runner_telemetry"]["sampling"]["repeat_penalty"] == pytest.approx(1.1)
        assert explicit_on["runner_telemetry"]["sampling"]["seed"] == 7
        greedy_off = _chat(srv, mid, temperature=0, repeat_penalty=1.0)
        greedy_on = _chat(srv, mid, temperature=0, repeat_penalty=1.1)
        assert text(greedy_off) == text(greedy_on)


def test_request_overrides_do_not_leak_into_the_next_request(model):
    with _server(model, "gemma4-mainline") as srv:
        mid = _get(srv, "/v1/models")["data"][0]["id"]
        _chat(srv, mid, temperature=0.2, repeat_penalty=1.5, top_k=3)
        d = _chat(srv, mid)
        t = d["runner_telemetry"]["sampling"]
        for k, v in PUBLISHED["gemma4"].items():
            assert t[k] == pytest.approx(v, abs=1e-6), (k, t)
        assert set(t["source"].values()) == {"preset"}, t


def test_cli_override_is_named_as_the_source(model):
    """--repeat-penalty on the command line beats the preset and is reported
    as the cli source; a request field still beats both."""
    with _server(model, "gemma4-mainline", extra=["--repeat-penalty", "1.3"]) as srv:
        mid = _get(srv, "/v1/models")["data"][0]["id"]
        caps = _get(srv, "/v1/capabilities")
        assert caps["sampling"]["repeat_penalty"] == pytest.approx(1.3)
        d = _chat(srv, mid)
        t = d["runner_telemetry"]["sampling"]
        assert t["repeat_penalty"] == pytest.approx(1.3) and t["source"]["repeat_penalty"] == "cli", t
        assert t["source"]["temperature"] == "preset", t
        d = _chat(srv, mid, repeat_penalty=1.0)
        t = d["runner_telemetry"]["sampling"]
        assert t["repeat_penalty"] == pytest.approx(1.0) and t["source"]["repeat_penalty"] == "request", t


def test_stream_usage_chunk_carries_the_diagnostics(model):
    with _server(model, "gemma4-mainline") as srv:
        mid = _get(srv, "/v1/models")["data"][0]["id"]
        events = _post(srv, "/v1/chat/completions", {
            "model": mid, "max_tokens": 8, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hi"}]}, stream=True)
        usage = [e for e in events if e.get("usage")]
        assert usage, events
        t = usage[-1]["runner_telemetry"]["sampling"]
        assert t["preset"] == "gemma4" and t["repeat_penalty"] == pytest.approx(1.0), t
        assert usage[-1]["runner_telemetry"]["tool_protocol"]["template"] == "gemma4-mainline"


def test_native_tool_turn_reports_its_contract(model):
    """With tools declared the diagnostics say which contract applied: the
    XML families parse without a grammar, Qwen3-Coder is constrained."""
    with _server(model, "qwen38") as srv:
        mid = _get(srv, "/v1/models")["data"][0]["id"]
        d = _chat(srv, mid, tools=[{"type": "function", "function": {
            "name": "bash", "parameters": {"type": "object", "properties": {
                "command": {"type": "string"}}, "required": ["command"]}}}])
        tp = d["runner_telemetry"]["tool_protocol"]
        assert tp == {"template": "qwen38", "family": "qwen3_xml", "tools": True,
                      "constrained": False, "parse_only": True}, tp
        assert d["runner_telemetry"]["sampling"]["preset"] == "qwen38"
    with _server(model, "qwen3-coder") as srv:
        mid = _get(srv, "/v1/models")["data"][0]["id"]
        bash = [{"type": "function", "function": {
            "name": "bash", "parameters": {"type": "object", "properties": {
                "command": {"type": "string"}}, "required": ["command"]}}}]
        # auto: the model's free turn, parsed; required: the grammar
        d = _chat(srv, mid, tools=bash)
        tp = d["runner_telemetry"]["tool_protocol"]
        assert tp["constrained"] is False and tp["parse_only"] is True, tp
        assert d["runner_telemetry"]["sampling"]["preset"] == "qwen3-coder"
        d = _chat(srv, mid, tools=bash, tool_choice="required")
        tp = d["runner_telemetry"]["tool_protocol"]
        assert tp["constrained"] is True and tp["parse_only"] is False, tp
