"""Local $ref and $defs in request schemas (R2.9.2).

Pydantic, MCP servers and older generators send schemas that reach nested
definitions through $ref. Until 2026-11-02 every such schema was a 400
("unsupported schema keyword '$defs'"); R2.9.1 found eight of eight client
cases refused. The compiler now inlines local, non-recursive references at
compile time and refuses the rest by name. This file holds the resolver's
edges through the server; tests/conformance holds the client-shaped cases.
"""
import json
import pathlib
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import RunnerServer, find_runner, validate_against_schema  # noqa: E402

MCP = {"type": "object",
       "properties": {"filter": {"$ref": "#/$defs/Filter"}},
       "required": ["filter"],
       "$defs": {"Filter": {"type": "object",
                            "properties": {"op": {"type": "string", "enum": ["eq"]}},
                            "required": ["op"], "additionalProperties": False}}}
TREE = {"$defs": {"Node": {"type": "object", "properties": {
    "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}}}},
        "type": "object", "properties": {"root": {"$ref": "#/$defs/Node"}}}


def _model(tmp_path_factory):
    m = tmp_path_factory.mktemp("refs") / "plain.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


def _post(srv, payload):
    req = urllib.request.Request(srv.base_url + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def _schema(srv, schema, max_tokens=96):
    return _post(srv, {"max_tokens": max_tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": "go"}],
                       "response_format": {"type": "json_schema",
                                           "json_schema": {"name": "x", "schema": schema}}})


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    with RunnerServer(exe, _model(tmp_path_factory), ctx=4096, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2"]) as s:
        yield s


def _constrained(srv, schema):
    st, body = _schema(srv, schema)
    assert st == 200, body
    ch = body["choices"][0]
    if ch["finish_reason"] != "stop":
        pytest.skip("the toy model ran out of tokens before the document closed")
    doc = json.loads(ch["message"]["content"])
    validate_against_schema(doc, schema)
    return doc


def test_a_pointer_with_escapes_resolves(srv):
    # "~1" is "/" and "~0" is "~" in a JSON pointer
    schema = {"$defs": {"a/b~c": {"type": "string", "enum": ["only"]}},
              "type": "object", "properties": {"k": {"$ref": "#/$defs/a~1b~0c"}},
              "required": ["k"]}
    assert _constrained(srv, schema) == {"k": "only"}


def test_a_percent_encoded_fragment_resolves(srv):
    # a URI fragment: %20 is decoded first, and %7E1 becomes ~1, then "/"
    schema = {"$defs": {"My Type": {"type": "string", "enum": ["t"]},
                        "a/b": {"type": "string", "enum": ["s"]}},
              "type": "object", "properties": {"k": {"$ref": "#/$defs/My%20Type"},
                                               "j": {"$ref": "#/$defs/a%7E1b"}},
              "required": ["k", "j"]}
    assert _constrained(srv, schema) == {"k": "t", "j": "s"}


def test_a_reference_into_properties_resolves(srv):
    schema = {"type": "object",
              "properties": {"a": {"type": "string", "enum": ["x"]},
                             "b": {"$ref": "#/properties/a"}},
              "required": ["a", "b"]}
    assert _constrained(srv, schema) == {"a": "x", "b": "x"}


def test_a_ref_inside_a_value_is_data(srv):
    # const, enum, default and examples carry values; a "$ref" key there is
    # not followed (this one resolves nowhere, so following it would refuse)
    schema = {"type": "object",
              "properties": {"k": {"type": "string", "enum": ["v"],
                                   "default": {"$ref": "#/nowhere"},
                                   "examples": [{"$ref": "#/nowhere"}]}},
              "required": ["k"]}
    assert _constrained(srv, schema) == {"k": "v"}


def test_a_property_named_definitions_is_kept(srv):
    schema = {"type": "object",
              "properties": {"definitions": {"type": "string", "enum": ["d"]}},
              "required": ["definitions"]}
    assert _constrained(srv, schema) == {"definitions": "d"}


@pytest.mark.parametrize("schema,why", [
    (TREE, "recursive $ref '#/$defs/Node'"),
    ({"type": "object", "properties": {"self": {"$ref": "#"}}}, "recursive $ref '#'"),
    ({"$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}},
      "$ref": "#/$defs/A"}, "recursive $ref '#/$defs/A'"),
    ({"type": "object", "properties": {"a": {"$dynamicRef": "#node"}}},
     "$dynamicRef and $recursiveRef are not supported"),
    ({"type": "object", "properties": {"a": {"$ref": 7}}}, "$ref must be a string"),
])
def test_what_cannot_be_inlined_is_refused_by_name(srv, schema, why):
    st, body = _schema(srv, schema)
    assert st == 400, body
    assert why in body["error"]["message"], body


def test_an_anyof_refusal_names_anyof(srv):
    # R2.9.1: a nested anyOf was refused as "oneOf alternative 1 ..."
    schema = {"type": "object", "required": ["v"], "properties": {"v": {"anyOf": [
        {"anyOf": [{"type": "string"}, {"type": "integer"}]}, {"type": "boolean"}]}}}
    st, body = _schema(srv, schema)
    assert st == 400, body
    assert body["error"]["message"].endswith(
        "anyOf alternative 0 cannot be dispatched by its first byte"), body


def test_tools_and_a_referencing_response_format_together(srv):
    # the auto turn's final branch embeds the response_format schema in the
    # envelope, where its $refs would resolve against the wrong document
    st, body = _post(srv, {"max_tokens": 64, "temperature": 0,
                           "messages": [{"role": "user", "content": "go"}],
                           "tools": [{"type": "function",
                                      "function": {"name": "f", "parameters": MCP}}],
                           "tool_choice": "auto",
                           "response_format": {"type": "json_schema",
                                               "json_schema": {"schema": MCP}}})
    assert st == 200, body


# Every tool grammar family resolves its parameters before compiling them: the
# referencing schema is admitted (it was a 400 on all of them), and a recursive
# one is refused by name wherever the family compiles the schema (a family
# that only parses its native call syntax admits it).
@pytest.fixture(scope="module", params=["chatml", "qwen38", "gemma4", "granite",
                                        "qwen3-coder", "nemotron", "harmony", "mistral-v1"])
def fam(request, tmp_path_factory):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    with RunnerServer(exe, _model(tmp_path_factory), ctx=4096, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2",
                                  "--chat-template", request.param]) as s:
        s.family = request.param
        yield s


def _tool_req(params, choice):
    return {"max_tokens": 48, "temperature": 0,
            "messages": [{"role": "user", "content": "go"}], "tool_choice": choice,
            "tools": [{"type": "function", "function": {"name": "f", "parameters": params}}]}


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "f"}}],
                         ids=["required", "named"])
def test_every_family_admits_referencing_parameters(fam, choice):
    st, body = _post(fam, _tool_req(MCP, choice))
    assert st == 200, (fam.family, body)
    calls = body["choices"][0]["message"].get("tool_calls") or []
    if calls and body["choices"][0]["finish_reason"] == "tool_calls":
        assert json.loads(calls[0]["function"]["arguments"])["filter"]["op"] == "eq", \
            (fam.family, calls)


def test_every_family_refuses_or_parses_a_recursive_schema(fam):
    st, body = _post(fam, _tool_req(TREE, "required"))
    if st == 400:
        assert "recursive $ref '#/$defs/Node'" in body["error"]["message"], (fam.family, body)
    else:
        assert st == 200, (fam.family, body)
