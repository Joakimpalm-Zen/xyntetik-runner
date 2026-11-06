"""No blocking client write happens while the device turn is held (R3.36).

A client that stops reading must not stop other slots. runner-turn-guard makes
that a build property: the turn runs on a CPU server too, and send_all ends the
process with exit 86 when it is called under the turn. So every request here is
a statement that its whole path writes nothing under the turn, and a path added
later is caught the first time a test drives it.

The anchor is the self test: with RUNNER_TURN_GUARD_SELFTEST a write is made
under the turn on purpose and must end the process. A green run on a guard that
cannot fire would prove nothing.
"""
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from server_probe import free_port  # noqa: E402

GUARD = ROOT / ("runner-turn-guard.exe" if sys.platform == "win32" else "runner-turn-guard")
MODEL = ROOT / "test.gguf"
PROMPT = "one two three four five six seven eight nine ten " * 6


class Server:
    def __init__(self, tmp, env=None, extra=()):
        self.port = free_port()
        self.log = tmp / "server.log"
        self.f = open(self.log, "wb")
        self.p = subprocess.Popen(
            [str(GUARD), "-m", str(MODEL), "--serve", "--port", str(self.port), "--gpu", "off", "--parallel", "2",
             "--draft", "lookup", "--sessions", str(tmp), "-c", "1024", "--no-tray", *extra],
            cwd=ROOT, stdout=self.f, stderr=self.f, env=dict(os.environ, **(env or {})))
        for _ in range(300):
            if self.p.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1).read()
                return
            except OSError:
                time.sleep(0.1)
        raise AssertionError(self.text())

    def text(self):
        self.f.flush()
        return self.log.read_text(errors="replace")

    def post(self, path, body, timeout=60):
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(r, timeout=timeout) as f:
                return f.status, f.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def close(self):
        self.p.kill()
        self.p.wait(timeout=30)
        self.f.close()


@pytest.fixture
def server(tmp_path):
    if not GUARD.exists() or not MODEL.exists():
        if os.environ.get("RUNNER_REQUIRE_TURN_GUARD"):
            pytest.fail("make runner-turn-guard test.gguf first")
        pytest.skip("runner-turn-guard or test.gguf not built")
    s = Server(tmp_path)
    yield s
    s.close()


def assert_clean(s):
    log = s.text()
    assert s.p.poll() is None and "turn-guard:" not in log, log[-800:]


def test_the_guard_fires_on_a_write_under_the_turn(tmp_path):
    if not GUARD.exists() or not MODEL.exists():
        if os.environ.get("RUNNER_REQUIRE_TURN_GUARD"):
            pytest.fail("make runner-turn-guard test.gguf first")
        pytest.skip("runner-turn-guard or test.gguf not built")
    s = Server(tmp_path, env={"RUNNER_TURN_GUARD_SELFTEST": "1"})
    try:
        with pytest.raises(OSError):
            s.post("/v1/completions", {"prompt": "hello", "max_tokens": 2}, timeout=20)
        assert s.p.wait(timeout=20) == 86
        assert "turn-guard: blocking client write under the device turn" in s.text()
    finally:
        s.close()


@pytest.mark.parametrize("path,body", [
    ("/v1/completions", {"prompt": PROMPT, "max_tokens": 48, "temperature": 0, "stream": True}),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 48, "temperature": 0,
                              "stream": True}),
    ("/v1/responses", {"input": PROMPT, "max_output_tokens": 48, "temperature": 0, "stream": True}),
    ("/v1/messages", {"model": "test", "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 48,
                      "temperature": 0, "stream": True}),
])
def test_a_speculative_stream_writes_nothing_under_the_turn(server, path, body):
    code, out = server.post(path, body)
    assert code == 200, out[:300]
    assert out, "the stream carried nothing"
    assert_clean(server)
    # the path under test really ran: a request that missed it proves nothing
    assert "[spec]" in server.text(), server.text()[-800:]


def test_a_buffered_speculative_request_writes_nothing_under_the_turn(server):
    code, out = server.post("/v1/completions", {"prompt": PROMPT, "max_tokens": 32, "temperature": 0})
    assert code == 200, out[:300]
    assert json.loads(out)["runner_telemetry"]["speculative"] is True
    assert_clean(server)


def test_a_session_answers_after_the_turn(server):
    code, out = server.post("/v1/runner/sessions", {"prompt": PROMPT, "max_tokens": 24, "temperature": 0})
    assert code == 200, out[:300]
    assert json.loads(out)["generated"] > 0
    assert_clean(server)


def test_a_suspended_session_resumes_after_the_turn(server):
    code, out = server.post("/v1/runner/sessions", {"prompt": PROMPT, "max_tokens": 24, "suspend_after": 8,
                                                    "temperature": 0})
    assert code == 200, out[:300]
    d = json.loads(out)
    assert d["finish_reason"] == "suspended" and d["id"], d
    assert_clean(server)
    code, out = server.post(f"/v1/runner/sessions/{d['id']}/resume", {})
    assert code == 200, out[:300]
    assert json.loads(out)["generated"] == 24
    assert_clean(server)


@pytest.mark.parametrize("path,body,terminal", [
    ("/v1/completions", {"prompt": PROMPT, "max_tokens": 200, "temperature": 0, "stream": True, "ignore_eos": True},
     b'"finish_reason":"error"'),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 200, "temperature": 0,
                              "stream": True, "ignore_eos": True}, b'"finish_reason":"error"'),
    ("/v1/responses", {"input": PROMPT, "max_output_tokens": 200, "temperature": 0, "stream": True,
                       "ignore_eos": True}, b'"reason":"client_too_slow"'),
    ("/v1/messages", {"model": "test", "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 200,
                      "temperature": 0, "stream": True, "ignore_eos": True}, b"event: error"),
])
def test_a_client_past_the_cap_is_cut_off_and_told_so(tmp_path, path, body, terminal):
    if not GUARD.exists() or not MODEL.exists():
        if os.environ.get("RUNNER_REQUIRE_TURN_GUARD"):
            pytest.fail("make runner-turn-guard test.gguf first")
        pytest.skip("runner-turn-guard or test.gguf not built")
    s = Server(tmp_path, env={"RUNNER_TEST_STREAM_OUT_CAP": "2000"})
    try:
        code, out = s.post(path, body)
        assert code == 200, out[:300]
        assert terminal in out, out[-500:]
        assert b"client_too_slow" in out, out[-500:]
        # the bytes before the cut arrived; a real terminator, not a bare close
        assert len(out) > 1000
        if path != "/v1/responses" and path != "/v1/messages":
            assert out.rstrip().endswith(b"data: [DONE]"), out[-200:]
        log = s.text()
        assert "client cut off for being too slow" in log, log[-600:]
        assert "[client too slow]" in log and "[client gone]" not in log, log[-600:]
        assert_clean(s)
    finally:
        s.close()


def test_other_slots_still_serve_after_all_of_it(server):
    for _ in range(2):
        assert server.post("/v1/completions", {"prompt": PROMPT, "max_tokens": 8, "temperature": 0,
                                               "stream": True})[0] == 200
        assert server.post("/v1/completions", {"prompt": "hello", "max_tokens": 4, "logprobs": 1})[0] == 200
    assert_clean(server)
