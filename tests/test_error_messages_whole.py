"""Every served refusal arrives whole, or says where it was cut (R4.48.25.1).

Two silent cuts, both fixed at the root:
- send_error_detail (src/http.c), the sink every refusal goes through, escaped the message into 384 bytes, so a longer
  one lost its end without a word. tests/test_send_error.c holds the sink itself.
- Four refusals echoed a request's own type string whole into a fixed buffer, so a long type cut the sentence
  mid-way: chat's content parts, the Anthropic tool_result's content, /v1/responses' input items and their content
  parts. They now echo it through err_echo (src/http.h): whole up to 64 bytes, else its first 64 bytes (on a UTF-8
  boundary) and "... (N bytes)".

Each case sends a 500-byte type (and one of 100 two-byte characters) and checks that the refusal ends with its own
last words and marks the cut. A short type is still echoed whole.
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


@pytest.fixture(scope="module")
def srv(runner_bin, tmp_path_factory):
    path = tmp_path_factory.mktemp("errwhole") / "olmo2.gguf"
    subprocess.run([sys.executable, ROOT / "scripts" / "make-test-olmo2.py", str(path)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    with RunnerServer(runner_bin, path, ctx=256, extra_args=["--gpu", "off", "-t", "2"]) as s:
        yield s


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


LONG = "x" * 500
WIDE = "é" * 100   # 200 bytes of UTF-8: the cut must not split one


def _cases(t):
    return [
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": [{"type": t, "text": "hi"}]}]},
         "only text is accepted"),
        ("/v1/responses", {"input": [{"type": t}]},
         "function_call_output and reasoning items"),
        ("/v1/responses", {"input": [{"type": "message", "role": "user", "content": [{"type": t, "text": "hi"}]}]},
         "Runner accepts text input only"),
        ("/v1/messages", {"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "f", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1",
                                          "content": [{"type": t, "text": "x"}]}]}]},
         "Runner accepts text tool results only"),
    ]


@pytest.mark.parametrize("value,nbytes", [(LONG, 500), (WIDE, 200)], ids=["ascii", "utf8"])
@pytest.mark.parametrize("case", range(4))
def test_a_long_echoed_type_is_marked_as_cut_and_the_refusal_ends_whole(srv, value, nbytes, case):
    path, body, ending = _cases(value)[case]
    status, doc = _post(srv, path, {**body, "max_tokens": 1})
    msg = _message(doc)
    assert status == 400, (path, doc)
    assert msg.endswith(ending), (path, msg)
    assert f"... ({nbytes} bytes)" in msg and value not in msg, (path, msg)
    assert "�" not in msg, (path, msg)   # the cut ends on a UTF-8 boundary


@pytest.mark.parametrize("case", range(4))
def test_a_short_type_is_echoed_whole(srv, case):
    path, body, ending = _cases("image")[case]
    status, doc = _post(srv, path, {**body, "max_tokens": 1})
    msg = _message(doc)
    assert status == 400 and msg.endswith(ending) and "image" in msg and "bytes)" not in msg, (path, msg)
