"""R2.2 closure provenance (owner 2026-10-02): when the budget ends inside a
constrained document, the engine's closer writes the tail that makes it
valid, and runner_telemetry.closure says which values it wrote, as JSON
pointers into the document, never inside tool_calls.

Anchor: the delivered document itself. Every path named resolves in it, the
byte counts add up to it, and a required enum the model never reached holds
the schema's first member while being reported as synthesized.
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

SCHEMA = {"type": "object",
          "properties": {"city": {"type": "string"},
                         "units": {"enum": ["celsius", "fahrenheit"]}},
          "required": ["city", "units"], "additionalProperties": False}


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    model = tmp_path_factory.mktemp("cp") / "m.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(model)],
                   check=True, stdout=subprocess.DEVNULL)
    with RunnerServer(exe, model, ctx=1024, extra_args=[
            "--gpu", "off", "-t", "2", "--chat-template", "chatml"]) as s:
        yield s


def _chat(srv, **extra):
    body = {"messages": [{"role": "user", "content": "weather in Paris, fahrenheit"}],
            "temperature": 0,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "w", "schema": SCHEMA}}}
    body.update(extra)
    req = urllib.request.Request(srv.base_url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _resolve(doc, pointer):
    node = doc
    for part in pointer.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def test_a_cut_document_names_what_the_closer_wrote(srv):
    d = _chat(srv, max_tokens=3)
    assert d["choices"][0]["finish_reason"] == "length", d
    text = d["choices"][0]["message"]["content"]
    doc = json.loads(text)                       # the closer made it valid
    cl = d["runner_telemetry"]["closure"]
    assert cl["closed"] is True
    assert cl["model_bytes"] + cl["closer_bytes"] == len(text.encode()), (cl, text)
    f = cl["fields"]
    assert "/units" in f["synthesized"], (f, text)
    assert doc["units"] == "celsius"             # the first member, never the model's
    named = f["synthesized"] + f["completed"]
    assert "/city" in named or text.encode()[:cl["model_bytes"]].count(b'"') >= 4, (f, text)
    for p in named:
        _resolve(doc, p)
    # strict clients see nothing new where they parse
    assert "closure" not in json.dumps(d["choices"])


def test_a_finished_document_reports_no_closure(srv):
    d = _chat(srv, max_tokens=512)
    if d["choices"][0]["finish_reason"] == "length":
        pytest.skip("the fixture did not finish the document within the budget")
    assert "closure" not in d["runner_telemetry"], d["runner_telemetry"]


def test_an_unconstrained_turn_has_no_closure(srv):
    req = urllib.request.Request(srv.base_url + "/v1/chat/completions", data=json.dumps({
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    assert "closure" not in d["runner_telemetry"]


@pytest.fixture(scope="module")
def scripted(tmp_path_factory):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    model = tmp_path_factory.mktemp("cps") / "m.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(model)],
                   check=True, stdout=subprocess.DEVNULL)
    with RunnerServer(exe, model, ctx=1024, extra_args=[
            "--gpu", "off", "-t", "2", "--chat-template", "chatml"],
            env={"RUNNER_TEST_SCRIPTED_REPLY": "1"}) as s:
        yield s


def test_a_string_cut_mid_value_is_completed_not_synthesized(scripted):
    """The model is scripted to write a long string and the budget cuts it:
    the closer only closes the string, so the value is "completed", while
    `units`, never reached, is "synthesized"."""
    full = '{"city":"Pariiiiiiiiiiiiiiiiiiiiiiiis","units":"fahrenheit"}'
    d = _chat(scripted, max_tokens=16, runner_test_reply=full)
    text = d["choices"][0]["message"]["content"]
    doc = json.loads(text)
    cl = d["runner_telemetry"]["closure"]
    assert doc["city"] and full.startswith(text[:cl["model_bytes"]]), (cl, text)
    assert cl["fields"]["completed"] == ["/city"], (cl, text)
    assert cl["fields"]["synthesized"] == ["/units"], (cl, text)
    assert doc["units"] == "celsius"


@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "weather"}],
                              "tool_choice": "required", "max_tokens": 6,
                              "tools": [{"type": "function", "function": {
                                  "name": "get_weather", "parameters": SCHEMA}}]}),
    ("/v1/responses", {"input": "weather", "tool_choice": "required",
                       "max_output_tokens": 6,
                       "tools": [{"type": "function", "name": "get_weather",
                                  "parameters": SCHEMA}]}),
    ("/v1/messages", {"max_tokens": 6, "messages": [{"role": "user", "content": "weather"}],
                      "tool_choice": {"type": "any"},
                      "tools": [{"name": "get_weather", "input_schema": SCHEMA}]}),
])
def test_a_cut_tool_call_says_the_closer_chose_it(srv, path, body):
    """The truncation-recovery call parses and executes; its telemetry says
    which of its parts the model never wrote, the tool name included."""
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(
        dict(body, temperature=0)).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    cl = d["runner_telemetry"]["closure"]
    assert cl["closed"] is True and cl["closer_bytes"] > 0, cl
    named = set(cl["fields"]["synthesized"] + cl["fields"]["completed"])
    assert "/arguments/units" in named, cl
    assert named <= {"/name", "/arguments", "/arguments/city", "/arguments/units"}, cl
