"""OLMo 2 Instruct's chat template on the server (R4.50.10): the OpenAI tool protocol refused, never rendered as nothing.

allenai/OLMo-2-0425-1B-Instruct's template (revision 48d788ec) writes system, user and assistant turns, and renders any
other role, a `tool` turn included, as nothing; it reads no tools. The renderer (template.c TMPL_OLMO2) writes it
exactly: scripts/template-conformance.py's olmo2 family holds that against the publisher's template, every case byte
for byte and token for token, and tests/test_template.c's test_olmo2_shape pins its branches offline. What this file
holds is the surface around it, on the tiny olmo2 file served with --chat-template olmo2:
- the OpenAI tool protocol is refused by name with a 400 on every chat surface: tools[], a tool turn and tool_calls on
  /v1/chat/completions; tools and function_call items on /v1/responses; tools and tool_use / tool_result blocks on
  /v1/messages;
- a plain chat is served;
- --info tools names no tool protocol.
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

NATIVE = "olmo2 chat template has no tool protocol"


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("olmo2tmpl") / "olmo2.gguf"
    subprocess.run([sys.executable, ROOT / "scripts" / "make-test-olmo2.py", str(path)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return path


@pytest.fixture(scope="module")
def olmo2(runner_bin, model):
    with RunnerServer(runner_bin, model, ctx=512,
                      extra_args=["--gpu", "off", "-t", "2", "--chat-template", "olmo2"]) as srv:
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
def test_the_openai_tool_protocol_is_refused_by_name_on_chat(olmo2, case, body):
    status, doc = _post(olmo2, "/v1/chat/completions", {**body, "max_tokens": 1})
    msg = _message(doc)
    assert status == 400 and NATIVE in msg and msg.rstrip().endswith("as nothing"), (case, doc)


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
def test_the_tool_protocol_is_refused_by_name_on_the_other_surfaces(olmo2, path, body):
    status, doc = _post(olmo2, path, {**body, "max_tokens": 1})
    msg = _message(doc)
    assert status == 400 and NATIVE in msg and msg.rstrip().endswith("as nothing"), (path, doc)


def test_a_plain_chat_is_served(olmo2):
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": "Sunny."}, {"role": "user", "content": "and tomorrow?"}]
    status, doc = _post(olmo2, "/v1/chat/completions", {"messages": msgs, "max_tokens": 1, "temperature": 0})
    assert status == 200, doc


def test_info_tools_names_no_tool_protocol(runner_bin, model):
    p = subprocess.run([str(runner_bin), "-m", str(model), "--info", "tools", "--chat-template", "olmo2", "--gpu", "off"],
                       cwd=ROOT, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    info = json.loads(p.stdout)
    assert (info["template"], info["tool_family"], info["native_tool_protocol"]) == ("olmo2", "none", False), info
