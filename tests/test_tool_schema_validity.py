"""A tool whose `parameters` is not a JSON Schema is refused on every family.

A family whose tool calls are constrained compiled the schema and refused
`{"type": "nonsense"}`; a family whose calls are parsed from its native
syntax never compiled it, answered 200 and generated against a declaration
that means nothing (found by the family sweep on Qwen3.5, 2026-10-02). The
check is structural, not "does this engine support it": a native-protocol
turn still takes schemas outside the constrained subset.
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

BAD = [
    ({"type": "nonsense"}, "not a JSON Schema type"),
    ({"type": "object", "properties": {"a": {"type": ["string", "wat"]}}},
     "not a JSON Schema type"),
    ({"type": "object", "properties": []}, "properties"),
    ({"type": "object", "required": "city"}, "required"),
    ({"type": "object", "properties": {"a": {"items": 5}}}, "must be an object"),
    ("a string", "must be an object"),
]
# valid JSON Schema the constrained subset does not cover: a parsed family
# must keep taking it
EXOTIC = {"type": "object", "properties": {
    "when": {"type": "string", "format": "date-time"},
    "n": {"type": "integer", "multipleOf": 3},
    "x": {"if": {"type": "string"}, "then": {"minLength": 2}}},
    "patternProperties": {"^x-": {"type": "string"}}}


@pytest.fixture(scope="module", params=["granite", "qwen38", "gemma4"])
def srv(request, tmp_path_factory):
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    m = tmp_path_factory.mktemp("ts") / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    with RunnerServer(exe, m, ctx=2048, extra_args=[
            "--gpu", "off", "-t", "2", "--chat-template", request.param]) as s:
        s.family = request.param
        yield s


def _post(srv, payload):
    req = urllib.request.Request(srv.base_url + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def _req(params, choice="auto"):
    return {"messages": [{"role": "user", "content": "x"}], "max_tokens": 4,
            "tool_choice": choice, "tools": [{"type": "function", "function": {
                "name": "f", "parameters": params}}]}


@pytest.mark.parametrize("choice", ["auto", "required"])
@pytest.mark.parametrize("case", range(len(BAD)))
def test_a_malformed_schema_is_a_400_on_every_family(srv, case, choice):
    params, needle = BAD[case]
    st, body = _post(srv, _req(params, choice))
    assert st == 400, (srv.family, body)
    assert needle in body["error"]["message"], body


def test_a_parsed_family_still_takes_a_schema_outside_the_subset(srv):
    st, body = _post(srv, _req(EXOTIC, "auto"))
    if srv.family == "qwen38":
        assert st == 200, (srv.family, body)   # an auto turn is parsed here
    else:
        # these families compile the schema and may refuse what the
        # constrained subset does not cover
        assert st in (200, 400), body
