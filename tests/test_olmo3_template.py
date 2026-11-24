"""OLMo 3's chat template on the server (R4.50.2): its own function fields served, the OpenAI tool protocol refused.

allenai/Olmo-3-7B-Think's template reads messages[].functions and messages[].function_calls as STRINGS and an
`environment` role for results; it renders any other role, a `tool` turn included, as nothing. The renderer
(template.c render_olmo3) writes the native fields exactly: that is anchored by scripts/template-conformance.py's
olmo3 families (the publisher's template, every case byte for byte and token for token) and pinned offline by
tests/test_template.c's test_olmo3_shape. What this file holds is the surface around it, on the tiny olmo2 file served
with --chat-template olmo3:
- the OpenAI tool protocol is refused by name with a 400 on every chat surface, never rendered as nothing: tools[],
  a tool turn and tool_calls on /v1/chat/completions; tools and function_call items on /v1/responses; tools and
  tool_use / tool_result blocks on /v1/messages;
- a non-string functions / function_calls is refused (the template concatenates them as text);
- the native fields and the environment role are served;
- the environment role stays refused on any other template.
"""
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

NATIVE = "olmo3 chat template has no"
TOOL_TURN_END = "as strings and an environment role for results"
TOOLS_END = "results as an environment turn"


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("olmo3") / "olmo2.gguf"
    subprocess.run([sys.executable, ROOT / "scripts" / "make-test-olmo2.py", str(path)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return path


@pytest.fixture(scope="module")
def olmo3(runner_bin, model):
    # 1024: the byte-level fixture vocabulary spells the default system turn in some 300 tokens
    with RunnerServer(runner_bin, model, ctx=1024,
                      extra_args=["--gpu", "off", "-t", "2", "--chat-template", "olmo3"]) as srv:
        yield srv


def _post(srv, path, body):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _message(doc):
    err = doc.get("error")
    return err.get("message", "") if isinstance(err, dict) else str(err)


FN = '[{"name":"weather","parameters":{"type":"object"}}]'
TOOLS = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]


@pytest.mark.parametrize("case,body", [
    ("tools[]", {"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS}),
    ("a tool turn", {"messages": [{"role": "user", "content": "hi"},
                                  {"role": "tool", "content": "20", "tool_call_id": "c1"}]}),
    ("tool_calls", {"messages": [{"role": "user", "content": "hi"},
                                 {"role": "assistant", "content": None, "tool_calls": [
                                     {"id": "c1", "type": "function",
                                      "function": {"name": "weather", "arguments": "{}"}}]}]}),
])
def test_the_openai_tool_protocol_is_refused_by_name_on_chat(olmo3, case, body):
    status, doc = _post(olmo3, "/v1/chat/completions", {**body, "max_tokens": 1})
    assert status == 400 and NATIVE in _message(doc) and "messages[].functions" in _message(doc), (case, doc)
    # the whole sentence, never cut: its end arrives too (a 192-byte buffer once cut it on clang and failed gcc's build)
    assert _message(doc).endswith(TOOL_TURN_END if case != "tools[]" else TOOLS_END), (case, _message(doc))


@pytest.mark.parametrize("path,body", [
    ("/v1/responses", {"input": "hi", "tools": [{"type": "function", "name": "weather",
                                                 "parameters": {"type": "object"}}]}),
    ("/v1/responses", {"input": [{"type": "message", "role": "user", "content": "hi"},
                                 {"type": "function_call", "call_id": "c1", "name": "weather", "arguments": "{}"},
                                 {"type": "function_call_output", "call_id": "c1", "output": "20"}]}),
    ("/v1/messages", {"messages": [{"role": "user", "content": "hi"}],
                      "tools": [{"name": "weather", "input_schema": {"type": "object"}}]}),
    ("/v1/messages", {"messages": [{"role": "user", "content": "hi"},
                                   {"role": "assistant", "content": [{"type": "tool_use", "id": "c1",
                                                                      "name": "weather", "input": {}}]},
                                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1",
                                                                 "content": "20"}]}]}),
])
def test_the_tool_protocol_is_refused_by_name_on_the_other_surfaces(olmo3, path, body):
    status, doc = _post(olmo3, path, {**body, "max_tokens": 1})
    assert status == 400 and NATIVE in _message(doc), (path, doc)


@pytest.mark.parametrize("field,role", [("functions", "user"), ("function_calls", "assistant")])
def test_a_native_field_must_be_a_string(olmo3, field, role):
    msgs = [{"role": "user", "content": "hi"}]
    if role == "assistant":
        msgs.append({"role": "assistant", "content": "", field: [{"name": "weather"}]})
    else:
        msgs[0][field] = [{"name": "weather"}]
    status, doc = _post(olmo3, "/v1/chat/completions", {"messages": msgs, "max_tokens": 1})
    assert status == 400 and f"{field} must be a string" in _message(doc), doc


def test_the_native_fields_and_the_environment_role_are_served(olmo3):
    msgs = [{"role": "system", "content": "S", "functions": FN},
            {"role": "user", "content": "weather in Paris?", "functions": FN},
            {"role": "assistant", "content": None, "function_calls": '[{"name":"weather","arguments":{}}]'},
            {"role": "environment", "content": '{"temperature":20}'}]
    status, doc = _post(olmo3, "/v1/chat/completions", {"messages": msgs, "max_tokens": 1, "temperature": 0})
    assert status == 200, doc


def test_the_environment_role_stays_refused_elsewhere(runner_bin, model):
    with RunnerServer(runner_bin, model, ctx=256, extra_args=["--gpu", "off", "-t", "2",
                                                              "--chat-template", "chatml"]) as srv:
        status, doc = _post(srv, "/v1/chat/completions",
                            {"messages": [{"role": "environment", "content": "x"}], "max_tokens": 1})
    assert status == 400 and "role must be system, developer, user, assistant or tool" in _message(doc), doc


def test_info_tools_names_no_tool_protocol(runner_bin, model):
    p = subprocess.run([str(runner_bin), "-m", str(model), "--info", "tools", "--chat-template", "olmo3", "--gpu", "off"],
                       cwd=ROOT, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    info = json.loads(p.stdout)
    assert (info["template"], info["tool_family"], info["native_tool_protocol"]) == ("olmo3", "none", False), info
