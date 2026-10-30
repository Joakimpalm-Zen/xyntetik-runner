"""Qwen 3.6's preserve_thinking, as the publisher's template reads it.

Qwen/Qwen3.6-35B-A3B chat_template.jinja:100 keeps an assistant turn's
thought when `preserve_thinking` is true or the turn follows the last user
query; undefined (the default) keeps only the latter, as Qwen 3.5 does. The
server reads it at the top level or inside chat_template_kwargs, as it reads
enable_thinking, and the replayed thought is what changes the prompt: its
length is the observable here (usage.prompt_tokens on the fixture model).
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

from harness import RunnerServer, find_runner  # noqa: E402

HISTORY = [
    {"role": "user", "content": "First question?"},
    {"role": "assistant", "content": "First answer.",
     "reasoning_content": "a thought about the first question"},
    {"role": "user", "content": "Second question?"},
]


@pytest.fixture(scope="module")
def exe():
    e = find_runner(ROOT)
    if not pathlib.Path(e).exists():
        pytest.skip("runner not built")
    return e


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("q36") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


def prompt_tokens(srv, **extra):
    body = {"messages": HISTORY, "max_tokens": 1, "temperature": 0, **extra}
    req = urllib.request.Request(srv.base_url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())["usage"]["prompt_tokens"]
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def serve(exe, model, family):
    return RunnerServer(exe, model, ctx=1024, parallel=1,
                        extra_args=["--gpu", "off", "-t", "2", "--chat-template", family])


def test_preserve_thinking_replays_every_thought(exe, model):
    with serve(exe, model, "qwen36") as srv:
        status, default = prompt_tokens(srv)
        assert status == 200, default
        assert prompt_tokens(srv, chat_template_kwargs={"preserve_thinking": False}) == (200, default)
        assert prompt_tokens(srv, preserve_thinking=None) == (200, default)
        status, kept = prompt_tokens(srv, chat_template_kwargs={"preserve_thinking": True})
        assert status == 200 and kept > default, (default, kept)
        # the top-level spelling, as enable_thinking is taken
        assert prompt_tokens(srv, preserve_thinking=True) == (200, kept)


def test_preserve_thinking_must_be_a_boolean(exe, model):
    with serve(exe, model, "qwen36") as srv:
        status, doc = prompt_tokens(srv, chat_template_kwargs={"preserve_thinking": "yes"})
        assert status == 400 and "preserve_thinking" in json.dumps(doc), doc


def test_qwen35_has_no_such_switch(exe, model):
    """Qwen 3.5's template never reads preserve_thinking."""
    with serve(exe, model, "qwen35") as srv:
        status, default = prompt_tokens(srv)
        assert status == 200, default
        assert prompt_tokens(srv, chat_template_kwargs={"preserve_thinking": True}) == (200, default)
