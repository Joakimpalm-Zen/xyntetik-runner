"""A structured-output close is sized by the schema, not by a fixed buffer.

The closer used a 4,096-byte buffer. An accepted schema whose one enum
member is 5,000 characters, sent with max_tokens 2, came back through the
normal API as 4,097 characters with no closing quote: not JSON (external
review, 2026-09-27). The close now grows to fit, so what a caller receives
under a truncation still parses and still conforms.
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

MEMBER = "a" * 5000


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    m = tmp_path_factory.mktemp("close") / "plain.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    with RunnerServer(exe, m, ctx=8192, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            srv.model_id = json.load(r)["data"][0]["id"]
        yield srv


def _chat(server, schema, max_tokens):
    payload = {"model": server.model_id, "max_tokens": max_tokens, "temperature": 0,
               "messages": [{"role": "user", "content": "pick"}],
               "response_format": {"type": "json_schema",
                                   "json_schema": {"name": "pick", "schema": schema}}}
    req = urllib.request.Request(server.base_url + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def test_a_long_enum_member_closes_to_valid_json(server):
    schema = {"type": "object",
              "properties": {"k": {"type": "string", "enum": [MEMBER]}},
              "required": ["k"], "additionalProperties": False}
    body = _chat(server, schema, max_tokens=2)
    text = body["choices"][0]["message"]["content"]
    assert body["choices"][0]["finish_reason"] == "length"
    doc = json.loads(text)          # parses: the closing quote and brace are there
    assert doc == {"k": MEMBER}


def test_a_large_min_length_closes_to_a_conforming_string(server):
    schema = {"type": "object",
              "properties": {"k": {"type": "string", "minLength": 6000}},
              "required": ["k"], "additionalProperties": False}
    body = _chat(server, schema, max_tokens=2)
    doc = json.loads(body["choices"][0]["message"]["content"])
    assert len(doc["k"]) >= 6000
