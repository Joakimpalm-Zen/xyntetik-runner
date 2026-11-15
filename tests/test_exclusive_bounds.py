"""An exclusive number bound at 0 stays exclusive in the runner itself (R2.20).

The bound was held as one ULP inside it, the smallest subnormal at 0. gcc's
-ffast-math links crtfastmath, which sets flush-to-zero and
denormals-are-zero at startup on x86, so in the Linux and Windows builds of
1.19.0 that subnormal compared as 0: under `exclusiveMinimum: 0` the
validator admitted every byte of {"x":0}, and a forced close with no digit
written closed to 0. tests/test_json_schema.c checks the validator in both
float modes by setting the flushed one by hand; this checks the server
process as built, in whatever mode its toolchain gives it.

A scripted reply (RUNNER_TEST_SCRIPTED_REPLY) shows which bytes the
constraint admitted: when it refuses one, the reply ends there and the closer
finishes the document, which runner_telemetry.closure reports.
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

SERVER_ARGS = ["--gpu", "off", "-t", "2", "--chat-template", "chatml"]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    model = tmp_path_factory.mktemp("bounds") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(model)],
                   check=True, stdout=subprocess.DEVNULL)
    with RunnerServer(exe, model, ctx=512, parallel=1, extra_args=SERVER_ARGS,
                      env={"RUNNER_TEST_SCRIPTED_REPLY": "1"}) as srv:
        yield srv


def _schema(bound):
    return {"type": "object", "properties": {"x": {"type": "number", bound: 0}},
            "required": ["x"], "additionalProperties": False}


def _reply(srv, bound, script):
    body = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 32, "temperature": 0,
            "runner_test_reply": script,
            "response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": _schema(bound)}}}
    req = urllib.request.Request(srv.base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"], out.get("runner_telemetry", {}).get("closure")


@pytest.mark.parametrize("bound,script", [
    ("exclusiveMinimum", '{"x":0}'), ("exclusiveMinimum", '{"x":0.0}'),
    ("exclusiveMaximum", '{"x":0}'), ("exclusiveMaximum", '{"x":-0}'),
])
def test_zero_is_refused_at_the_byte_that_ends_it(server, bound, script):
    content, closure = _reply(server, bound, script)
    # the closing brace after the zero was refused: the reply was cut there and
    # the closer wrote that brace (the number's own close is R2.19's)
    assert closure is not None, (bound, script, content)
    assert closure["model_bytes"] == len(script) - 1 and closure["closer_bytes"] >= 1, (bound, script, closure)


@pytest.mark.parametrize("script", ['{"x":-0.5}', '{"x":-7}'])
def test_a_close_with_no_digit_lands_inside(server, script):
    # a minus under a bound of 0 from below is refused where it is written, so
    # the closer writes the whole number: inside the bound, as a spelling a
    # parser reads back (on x86 it wrote 0, the subnormal edge read as zero)
    content, closure = _reply(server, "exclusiveMinimum", script)
    assert closure is not None and closure["model_bytes"] == len('{"x":'), (content, closure)
    assert json.loads(content)["x"] > 0, content


@pytest.mark.parametrize("bound,script", [("exclusiveMinimum", '{"x":0.5}'), ("exclusiveMaximum", '{"x":-0.5}')])
def test_a_value_inside_passes_untouched(server, bound, script):
    # the control: nothing refused, nothing closed
    content, closure = _reply(server, bound, script)
    assert closure is None and content == script, (content, closure)
