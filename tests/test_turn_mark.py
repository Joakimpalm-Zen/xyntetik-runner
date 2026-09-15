"""A recurrent slot resumes the next agent turn at the prompt boundary.

The 2026-09-14 OpenCode loop on Qwen 3.8 (a qwen35 hybrid: recurrent
DeltaNet layers between full-attention ones) re-folded its whole 7.5K-token
prompt on every turn, "0 cached" each time. An agent client's next request
replays the previous prompt, then the reply re-rendered by the template
(reasoning stripped, tool calls re-serialised), then the tool result or a
new user turn. The kept run of the slot's history therefore ends past the
prompt but short of the position the fold sits at, and a recurrent fold
cannot be sliced to an arbitrary position: it is restorable only where a
checkpoint was taken. The turn mark is that checkpoint, taken at the end
of every prefill; engine_rewind resumes from it whenever the kept run
reaches it.

Gate, on the qwen35 CI fixture served on the CPU: request 2 replays request
1's prompt with a different assistant reply and a new user turn. Its
telemetry (`prompt_cached_tokens`, and `prompt_reuse` saying how) must say
the slot resumed from the turn mark with request 1's whole prompt cached, and its greedy answer must equal the answer the same
request gets on a fresh slot (cache_prompt:false resets the slot). The
bit-for-bit logits identity is pinned one level down in
tests/test_recurrent_rewind.c; this is the served contract.
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
def ornith(tmp_path_factory):
    p = tmp_path_factory.mktemp("turnmark") / "ornith.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", str(p)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return p


def _post(server, body):
    req = urllib.request.Request(server.base_url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _chat(server, messages, **fields):
    with urllib.request.urlopen(server.base_url + "/v1/models", timeout=30) as r:
        mid = json.load(r)["data"][0]["id"]
    # prefix_cache:false keeps the SHARED tier out of the way: on the CPU an
    # exact full-prefix hit there forks KV + fold and would serve request 2
    # first (see the last request); the mark is the slot's OWN tier, the one
    # an offloaded recurrent model has, and it is what this gate is about.
    body = {"model": mid, "messages": messages, "max_tokens": 6,
            "temperature": 0, "seed": 1, "prefix_cache": False}
    body.update(fields)
    return _post(server, body)


def test_second_turn_resumes_from_the_turn_mark(ornith):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    first = [{"role": "user", "content": "List the files in this directory."}]
    second = first + [
        {"role": "assistant", "content": "I will run ls now."},
        {"role": "user", "content": "Now show me the biggest one."}]
    with RunnerServer(exe, ornith, ctx=512, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        def reuse(d):
            t = d["runner_telemetry"]
            return t["prompt_reuse"], t["prompt_cached_tokens"], t["timing"]["prefill_tokens"]

        d1 = _chat(srv, first)
        assert reuse(d1)[:2] == ("none", 0), reuse(d1)
        n1 = d1["usage"]["prompt_tokens"]
        assert reuse(d1)[2] == n1, (reuse(d1), n1)

        # the mark sits one token short of the prompt end (that token is
        # always fed, it is what produces the logits), so everything before
        # it is behind the mark and nothing before it is fed again
        d2 = _chat(srv, second)
        n2 = d2["usage"]["prompt_tokens"]
        assert reuse(d2) == ("turn_mark", n1 - 1, n2 - (n1 - 1)), (reuse(d2), n1, n2)

        # an identical replay resumes at the new mark and feeds one token
        d4 = _chat(srv, second)
        assert reuse(d4) == ("turn_mark", n2 - 1, 1), reuse(d4)
        assert d4["choices"][0]["message"]["content"] == d2["choices"][0]["message"]["content"]

        # the same request on a fresh slot gives the same greedy answer;
        # cache_prompt:false resets the slot and marks nothing
        d3 = _chat(srv, second, cache_prompt=False)
        assert reuse(d3)[:2] == ("none", 0), reuse(d3)
        assert d3["choices"][0]["message"]["content"] == d2["choices"][0]["message"]["content"]
        d5 = _chat(srv, second)
        assert reuse(d5)[:2] == ("recurrent_reset", 0), reuse(d5)
        assert d5["choices"][0]["message"]["content"] == d2["choices"][0]["message"]["content"]

        # with the shared tier on, a CPU slot forks the exact published
        # prefix (KV + fold blob, tracer 5) and the telemetry names that tier
        _chat(srv, first, prefix_cache=True)
        d7 = _chat(srv, second, prefix_cache=True)
        assert reuse(d7)[:2] == ("prefix_cache", n1), reuse(d7)
        assert d7["choices"][0]["message"]["content"] == d2["choices"][0]["message"]["content"]
