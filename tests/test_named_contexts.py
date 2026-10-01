"""Named context handles (R10.4): POST /v1/runner/contexts pins a prefix,
`context_id` on a request builds on it.

The prefix cache serves repeated prefixes blindly and only while traffic keeps
them warm. A named context is prefilled once, pinned (no TTL, never evicted by
traffic) and forked by every request whose prompt starts with it; a request
that names a context its prompt does not start with is refused rather than
served cold under a name that promised a warm prefix.

Anchors: a forked context produces the same tokens and logprobs as the same
prompt prefilled cold (the fork installs the KV the prefill wrote, bit for
bit), and the cached-token count is the context's own length, which the test
reads back from the pin, not from the request it checks.
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
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

SYS = ("You are a careful assistant for a warehouse inventory team. Answer "
       "with the item code first, then one short sentence. ")


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("ctx") / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return m


def _req(srv, method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(srv.base_url + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as r:
        return r.code, json.load(r)


def _serve(runner_bin, model, env=None, template=None):
    e = dict(os.environ)
    e.update(env or {})
    extra = ["--gpu", "off", "-t", "2"]
    if template:
        extra += ["--chat-template", template]
    return RunnerServer(runner_bin, model, ctx=256, env=e, extra_args=extra)


def _complete(srv, prompt, **extra):
    return _req(srv, "POST", "/v1/completions", {
        "prompt": prompt, "max_tokens": 6, "temperature": 0, "logprobs": 2,
        **extra})


def test_a_forked_context_answers_exactly_as_a_cold_prompt(runner_bin, model):
    prompt = SYS + "Where is item 42?"
    with _serve(runner_bin, model) as srv:
        st, cold = _complete(srv, prompt, cache_prompt=False)
        assert st == 200, cold
        st, pin = _req(srv, "POST", "/v1/runner/contexts",
                       {"id": "inv-sys", "prompt": SYS})
        assert st == 200, pin
        assert pin["object"] == "runner.context" and pin["id"] == "inv-sys"
        n = pin["tokens"]
        assert n > 16
        # overwrite the slot's own KV, so only the pinned snapshot can serve
        assert _complete(srv, "unrelated text that shares nothing")[0] == 200
        st, warm = _complete(srv, prompt, context_id="inv-sys")
        assert st == 200, warm
        tel = warm["runner_telemetry"]
        assert tel["context"] == {"id": "inv-sys", "tokens": n}
        assert tel["prompt_cached_tokens"] == n
        assert tel["prompt_reuse"] == "prefix_cache"
        # the same bytes as the cold prefill, and the same numbers to within
        # the last digit: the fork feeds the suffix at another batch width
        # than the cold prompt, and CPU prefill is not batch-invariant on
        # every host (an AVX-512 native build read -4.33079 against -4.330789)
        assert warm["choices"][0]["text"] == cold["choices"][0]["text"]
        assert warm["choices"][0]["logprobs"]["token_logprobs"] == pytest.approx(
            cold["choices"][0]["logprobs"]["token_logprobs"], abs=1e-5)
        st, lst = _req(srv, "GET", "/v1/runner/contexts")
        assert st == 200
        (entry,) = lst["data"]
        assert entry["id"] == "inv-sys" and entry["tokens"] == n
        assert entry["hits"] >= 1 and entry["bytes"] > 0


def test_a_pinned_context_outlives_the_cache_ttl(runner_bin, model):
    with _serve(runner_bin, model, {"RUNNER_PREFIX_CACHE_TTL": "0.3"}) as srv:
        assert _req(srv, "POST", "/v1/runner/contexts",
                    {"id": "keep", "prompt": SYS})[0] == 200
        time.sleep(0.8)
        # any cache activity runs the expiry; an unpinned snapshot would go
        assert _complete(srv, "unrelated text that shares nothing")[0] == 200
        st, warm = _complete(srv, SYS + "What is in bay 3?", context_id="keep")
        assert st == 200, warm
        assert warm["runner_telemetry"]["prompt_reuse"] == "prefix_cache"


def test_refusals_name_the_problem(runner_bin, model):
    with _serve(runner_bin, model) as srv:
        assert _req(srv, "POST", "/v1/runner/contexts",
                    {"id": "inv-sys", "prompt": SYS})[0] == 200
        st, b = _complete(srv, "A different start. " + SYS, context_id="inv-sys")
        assert st == 409 and b["error"]["code"] == "context_mismatch", b
        assert "differs at token" in b["error"]["message"]
        st, b = _complete(srv, SYS + "x", context_id="nope")
        assert st == 404 and b["error"]["code"] == "context_not_found", b
        st, b = _complete(srv, SYS + "x", context_id="bad id!")
        assert st == 400, b
        st, b = _complete(srv, SYS + "x", context_id="inv-sys",
                          cache_prompt=False)
        assert st == 400 and "prefix cache" in b["error"]["message"], b
        for bad in ({"prompt": SYS}, {"id": "a", "prompt": SYS,
                                      "messages": [{"role": "user",
                                                    "content": "x"}]},
                    {"id": "a"}, {"id": "a", "prompt": 3}):
            st, b = _req(srv, "POST", "/v1/runner/contexts", bad)
            assert st == 400, (bad, b)


def test_delete_and_repin(runner_bin, model):
    with _serve(runner_bin, model) as srv:
        _, a = _req(srv, "POST", "/v1/runner/contexts",
                    {"id": "c", "prompt": SYS})
        _, b = _req(srv, "POST", "/v1/runner/contexts",
                    {"id": "c", "prompt": SYS + "Also mention the aisle. "})
        assert b["tokens"] > a["tokens"]
        _, lst = _req(srv, "GET", "/v1/runner/contexts")
        assert [e["tokens"] for e in lst["data"]] == [b["tokens"]]
        st, d = _req(srv, "DELETE", "/v1/runner/contexts/c")
        assert st == 200 and d["deleted"] is True
        assert _req(srv, "GET", "/v1/runner/contexts")[1]["data"] == []
        assert _req(srv, "DELETE", "/v1/runner/contexts/c")[0] == 404
        assert _complete(srv, SYS + "x", context_id="c")[0] == 404


def test_the_budget_refuses_a_context_it_cannot_hold(runner_bin, model):
    with _serve(runner_bin, model, {"RUNNER_PREFIX_CACHE_MB": "0"}) as srv:
        st, b = _req(srv, "POST", "/v1/runner/contexts",
                     {"id": "big", "prompt": SYS})
        assert st == 507 and b["error"]["code"] == "context_budget", b


def test_a_chat_context_is_the_prefix_of_the_chat_render(runner_bin, model):
    system = {"role": "system", "content": SYS}
    # chatml renders a system turn on its own; the fixture's own raw template
    # has nothing to render for one, and says so
    with _serve(runner_bin, model) as srv:
        st, b = _req(srv, "POST", "/v1/runner/contexts",
                     {"id": "chat-sys", "messages": [system]})
        assert st == 400 and "render" in b["error"]["message"], b
    with _serve(runner_bin, model, template="chatml") as srv:
        st, pin = _req(srv, "POST", "/v1/runner/contexts",
                       {"id": "chat-sys", "messages": [system]})
        assert st == 200, pin
        st, d = _req(srv, "POST", "/v1/chat/completions", {
            "messages": [system, {"role": "user", "content": "Item 42?"}],
            "max_tokens": 4, "temperature": 0, "context_id": "chat-sys"})
        assert st == 200, d
        tel = d["runner_telemetry"]
        assert tel["context"] == {"id": "chat-sys", "tokens": pin["tokens"]}
        assert tel["prompt_cached_tokens"] >= pin["tokens"]
