"""Generation that runs into the context says so.

A one-shot run whose -n does not fit in -c stopped at the context with exit 0
and no word about why, so a reply of 25 tokens for -n 200 read as the model's
choice (the pre-1.12.0 sweep, 2026-11-02). The CLI now prints one line naming
the limit and the -c that would hold the request; a run that ends on its own
or on -n prints none. The server already reports the same stop as a length
finish (OpenAI) and max_tokens (Anthropic); those are pinned here too.
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


@pytest.fixture(scope="module")
def exe():
    e = find_runner(ROOT)
    if not pathlib.Path(e).exists():
        pytest.skip("runner not built")
    return e


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("ctx") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


def oneshot(exe, model, *args):
    p = subprocess.run([str(exe), "-m", str(model), "-p", "the runner trains the",
                        "--temp", "0", "--gpu", "off", "-t", "2", "--no-tray", *args],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    return p.stderr


def test_a_run_that_fills_the_context_says_so(exe, model):
    err = oneshot(exe, model, "-c", "48", "-n", "200", "--ignore-eos")
    line = [ln for ln in err.splitlines() if ln.startswith("[context full")]
    assert len(line) == 1, err
    assert "-c 48" in line[0] and "of the 200 tokens" in line[0] and "-c 1024" in line[0]


def test_a_run_without_a_budget_that_fills_the_context_says_so(exe, model):
    err = oneshot(exe, model, "-c", "48", "-n", "-1", "--ignore-eos")
    assert "[context full: -c 48" in err and "larger -c" in err, err


def test_a_run_that_ends_on_its_budget_says_nothing_of_the_context(exe, model):
    err = oneshot(exe, model, "-c", "512", "-n", "4", "--ignore-eos")
    assert "context full" not in err, err


def post(srv, path, body, headers=()):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **dict(headers)})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def test_the_server_reports_a_full_context_as_a_length_finish(exe, model):
    with RunnerServer(exe, model, ctx=96, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        msgs = [{"role": "user", "content": "the runner trains the"}]
        chat = post(srv, "/v1/chat/completions",
                    {"messages": msgs, "max_tokens": 500, "temperature": 0})
        assert chat["choices"][0]["finish_reason"] == "length", chat
        assert chat["usage"]["completion_tokens"] < 500
        comp = post(srv, "/v1/completions",
                    {"prompt": "the runner trains the", "max_tokens": 500, "temperature": 0})
        assert comp["choices"][0]["finish_reason"] == "length", comp
        msg = post(srv, "/v1/messages",
                   {"messages": msgs, "max_tokens": 500, "temperature": 0},
                   headers={"anthropic-version": "2023-06-01"})
        assert msg["stop_reason"] == "max_tokens", msg
