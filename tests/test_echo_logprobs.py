"""/v1/completions echo and prompt_logprobs: teacher-forced scoring of the
prompt over HTTP (R4.8).

`echo: true` returns the prompt in front of the completion, and with
`logprobs: N` the logprob arrays cover the prompt's tokens first (the first
token has nothing before it, so its entry is null), which is how an
OpenAI-compatible evaluation harness scores a continuation. `prompt_logprobs:
K` returns vLLM's per-position shape instead: the actual token's logprob and
rank plus the top K alternatives.

Anchors:
  * the numbers are `--score`'s, token for token: the same solo forward per
    position and the same log-softmax arithmetic (tests/test_score.py pins
    --score itself);
  * on the zero-branch fixture a position's logits depend on the previous
    token alone, by construction, so two positions that score the same token
    after the same token must carry the same number exactly.
"""
import json
import pathlib
import struct
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

PROMPT = "the quick brown fox jumps over the lazy dog and keeps on running"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("echo")
    out = {}
    for name, script, flags in [
            ("dense", "make-test-model.py", []),
            ("blind", "make-test-model.py", ["--zero-branches"]),
            ("recurrent", "make-test-ornith.py", [])]:
        out[name] = d / f"{name}.gguf"
        subprocess.run([sys.executable, ROOT / "scripts" / script, *flags,
                        str(out[name])], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
    return out


def _post(srv, payload, path="/v1/completions"):
    req = urllib.request.Request(srv.base_url + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as r:
        return r.code, json.load(r)


def _serve(runner_bin, model):
    # the fixtures' training context: above it YaRN rescales rope, and the
    # --score reference below runs at the model's own context
    return RunnerServer(runner_bin, model, ctx=256,
                        extra_args=["--gpu", "off", "-t", "2"])


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def _score(runner_bin, model, prompt):
    p = subprocess.run([runner_bin, "-m", str(model), "--score", "-p", prompt,
                        "-c", "256", "-t", "2", "--gpu", "off"], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(p.stdout)


@pytest.mark.parametrize("kind", ["dense", "recurrent"])
def test_echo_logprobs_are_the_score_numbers(runner_bin, models, kind):
    ref = _score(runner_bin, models[kind], PROMPT)
    n = ref["n_tokens"]
    with _serve(runner_bin, models[kind]) as srv:
        st, body = _post(srv, {"prompt": PROMPT, "max_tokens": 2,
                               "temperature": 0, "echo": True, "logprobs": 3})
        assert st == 200, body
        ch = body["choices"][0]
        assert ch["text"].startswith(PROMPT)
        lp = ch["logprobs"]
        assert lp["token_ids"][:n] == ref["tokens"]
        assert lp["token_logprobs"][0] is None
        assert lp["top_logprobs"][0] is None
        # exact: the wire's %.6f of the very float --score prints at %.9g
        for got, want in zip(lp["token_logprobs"][1:n], ref["logprobs"]):
            assert got == float(f"{_f32(want):.6f}")
        # the prompt entries are followed by the generated ones, one per token
        n_gen = body["usage"]["completion_tokens"]
        assert len(lp["token_ids"]) == n + n_gen
        assert len(lp["token_logprobs"]) == n + n_gen
        assert len(lp["tokens"]) == len(lp["top_logprobs"]) == n + n_gen
        # every scored prompt position lists 3 alternatives, best first (by
        # id: two ids can spell the same piece and share one map key), and
        # the actual token's logprob is never above the best one
        assert lp["top_token_ids"][0] is None
        for i in range(1, n):
            assert len(lp["top_token_ids"][i]) == 3
            alts = list(lp["top_logprobs"][i].values())
            if len(alts) == 3:   # no two alternatives spelled alike
                assert alts == sorted(alts, reverse=True)
            assert lp["token_logprobs"][i] <= max(alts) + 1e-6
        # the same request without echo generates the same tokens
        st, plain = _post(srv, {"prompt": PROMPT, "max_tokens": 2,
                                "temperature": 0, "logprobs": 3})
        assert st == 200, plain
        assert plain["choices"][0]["logprobs"]["token_ids"] == \
            lp["token_ids"][n:]
        assert ch["text"] == PROMPT + plain["choices"][0]["text"]


def test_blind_fixture_scores_depend_on_the_previous_token_only(runner_bin,
                                                                models):
    prompt = "the cat the cat the cat the cat"
    with _serve(runner_bin, models["blind"]) as srv:
        st, body = _post(srv, {"prompt": prompt, "max_tokens": 0,
                               "echo": True, "logprobs": 1})
        assert st == 200, body
        lp = body["choices"][0]["logprobs"]
        ids, lps = lp["token_ids"], lp["token_logprobs"]
        seen, repeats = {}, 0
        for i in range(1, len(ids)):
            key = (ids[i - 1], ids[i])
            if key in seen:
                assert lps[i] == seen[key], (i, key)
                repeats += 1
            seen[key] = lps[i]
        assert repeats >= 4   # the anchor must actually be exercised


def test_max_tokens_zero_scores_without_generating(runner_bin, models):
    with _serve(runner_bin, models["dense"]) as srv:
        st, body = _post(srv, {"prompt": PROMPT, "max_tokens": 0,
                               "echo": True, "logprobs": 1})
        assert st == 200, body
        ch = body["choices"][0]
        assert ch["text"] == PROMPT
        assert body["usage"]["completion_tokens"] == 0
        n = body["usage"]["prompt_tokens"]
        assert len(ch["logprobs"]["token_logprobs"]) == n


def test_echo_without_logprobs_returns_only_the_text(runner_bin, models):
    with _serve(runner_bin, models["dense"]) as srv:
        st, body = _post(srv, {"prompt": PROMPT, "max_tokens": 2,
                               "temperature": 0, "echo": True})
        assert st == 200, body
        ch = body["choices"][0]
        assert ch["text"].startswith(PROMPT) and len(ch["text"]) > len(PROMPT)
        assert "logprobs" not in ch


def test_prompt_logprobs_vllm_shape(runner_bin, models):
    ref_cli = _score(runner_bin, models["dense"], PROMPT)
    with _serve(runner_bin, models["dense"]) as srv:
        st, echo = _post(srv, {"prompt": PROMPT, "max_tokens": 0,
                               "echo": True, "logprobs": 1})
        assert st == 200, echo
        ref = echo["choices"][0]["logprobs"]
        st, body = _post(srv, {"prompt": PROMPT, "max_tokens": 1,
                               "temperature": 0, "prompt_logprobs": 2})
        assert st == 200, body
        pl = body["choices"][0]["prompt_logprobs"]
        assert len(pl) == len(ref["token_ids"])
        assert pl[0] is None
        for i in range(1, len(pl)):
            entry = pl[i]
            actual = str(ref["token_ids"][i])
            assert actual in entry
            # %.9g on both sides: the same float, bit for bit
            assert entry[actual]["logprob"] == ref_cli["logprobs"][i - 1]
            assert entry[actual]["logprob"] == pytest.approx(
                ref["token_logprobs"][i], abs=1e-6)
            ranks = sorted(v["rank"] for v in entry.values())
            assert ranks[:2] == [1, 2]
            assert 2 <= len(entry) <= 3
            best = [v for v in entry.values() if v["rank"] == 1][0]
            assert all(best["logprob"] >= v["logprob"] for v in entry.values())
            assert all(isinstance(v["decoded_token"], str)
                       for v in entry.values())
        # without echo the text is the completion alone
        assert not body["choices"][0]["text"].startswith(PROMPT)


@pytest.mark.parametrize("payload, needle", [
    ({"echo": True, "stream": True}, "stream"),
    ({"prompt_logprobs": 2, "stream": True}, "stream"),
    ({"prompt_logprobs": 21}, "prompt_logprobs"),
    ({"prompt_logprobs": -1}, "prompt_logprobs"),
    ({"prompt_logprobs": 1.5}, "prompt_logprobs"),
    ({"prompt_logprobs": "2"}, "prompt_logprobs"),
    ({"echo": "yes"}, "echo"),
])
def test_bad_requests_are_refused_by_name(runner_bin, models, payload,
                                          needle):
    with _serve(runner_bin, models["dense"]) as srv:
        st, body = _post(srv, {"prompt": PROMPT, "max_tokens": 1, **payload})
        assert st == 400, body
        assert needle in body["error"]["message"]


def test_chat_still_refuses_echo(runner_bin, models):
    with _serve(runner_bin, models["dense"]) as srv:
        st, body = _post(srv, {"messages": [{"role": "user",
                                             "content": "hi"}],
                               "echo": True}, "/v1/chat/completions")
        assert st == 400, body
        assert "echo" in body["error"]["message"]
