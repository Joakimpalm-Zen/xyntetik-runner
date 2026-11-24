"""A recurrent model's prefix fork picks the longest entry it can fork whole (R4.48.22.4).

The shared prefix tier picked the entry with the longest common prefix over all entries. A recurrent model forks
only an exact full-prefix hit (its fold is the one at the entry's end, and a fold is not sliceable), so when the
longest common prefix belonged to an entry the prompt did not hold whole, that hit was declined and no other entry was
tried. Every request publishes its whole prompt, so after a context's first use the request's own entry (the context
and that question) shared more with the next request naming the context than the pin did:
- pin P, a request naming it, an unrelated prompt, the same request again: 409 context_unavailable, the pin still in
  the tier;
- the automatic tier: P, P + Q, an unrelated prompt, P + Q again: prefilled cold (no cached tokens), with P's entry
  there to fork.
For a recurrent model the candidates are now the entries the prompt holds whole, and the longest of them is forked.
A non-recurrent model's pick is unchanged (it forks a partial hit, KV rows are sliceable): the dense case holds its
cached counts as they were, each the longest share there is.

The anchors: the request's answer and token counts. On the CPU each fork's tokens and logprobs equal the cold
prefill of the same prompt in a fresh process bit for bit; on CUDA the second use equals the first bit for bit and the
cold prefill's text, every server's log showing the device (RUNNER_GPU_FALLBACK=fail).
"""
import json
import os
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

SYS = ("You are a careful assistant for a warehouse inventory team. Answer "
       "with the item code first, then one short sentence. ")
Q1, Q2 = "Where is item 42?", "Where is item 7?"
UNRELATED = "unrelated text that shares nothing"
RECURRENT = {"qwen4exp": "make-test-qwen4exp.py", "qwen35": "make-test-ornith.py"}


def _cuda(runner_bin):
    caps = subprocess.run([str(runner_bin), "--caps"], cwd=ROOT, stdout=subprocess.PIPE, timeout=30)
    return (json.loads(caps.stdout.decode()).get("gpu") or {}).get("backend") == "cuda"


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("pick")
    out = {}
    for name, gen in {**RECURRENT, "dense": "make-test-model.py"}.items():
        out[name] = d / f"{name}.gguf"
        subprocess.run([sys.executable, ROOT / "scripts" / gen, str(out[name])], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
    return out


def _req(srv, payload):
    req = urllib.request.Request(srv.base_url + payload.pop("_path", "/v1/completions"),
                                 data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as r:
        return r.code, json.load(r)


def _complete(srv, prompt, **extra):
    st, b = _req(srv, {"prompt": prompt, "max_tokens": 6, "temperature": 0, "logprobs": 2, **extra})
    assert st == 200, (prompt[-20:], extra, b)
    c = b["choices"][0]
    return (c["text"], c["logprobs"]["token_logprobs"]), b["runner_telemetry"], b["usage"]["prompt_tokens"]


class _Run:
    def __init__(self, runner_bin, model, tmp_path, gpu):
        self.runner_bin, self.model, self.tmp_path, self.gpu, self.logs = runner_bin, model, tmp_path, gpu, []

    def serve(self):
        self.logs.append(self.tmp_path / ("server-%d.log" % len(self.logs)))
        return RunnerServer(self.runner_bin, self.model, ctx=512, log_path=str(self.logs[-1]),
                            env=dict(os.environ, RUNNER_GPU_FALLBACK="fail"),
                            extra_args=["--gpu", self.gpu, "-t", "2"])

    def cold(self, prompt):
        with self.serve() as srv:
            ans, tel, _ = _complete(srv, prompt, cache_prompt=False)
        assert tel["prompt_cached_tokens"] == 0, tel
        return ans

    def named_twice(self, q_second):
        # pin P, a request naming it, an unrelated prompt, a request naming it again
        with self.serve() as srv:
            st, pin = _req(srv, {"_path": "/v1/runner/contexts", "id": "sys", "prompt": SYS})
            assert st == 200, pin
            first, tel1, _ = _complete(srv, SYS + Q1, context_id="sys")
            _complete(srv, UNRELATED)
            st, b = _req(srv, {"prompt": SYS + q_second, "max_tokens": 6, "temperature": 0, "logprobs": 2,
                               "context_id": "sys"})
        assert st == 200, ("the second request naming the context", st, b)
        c = b["choices"][0]
        return pin["tokens"], first, tel1, (c["text"], c["logprobs"]["token_logprobs"]), b["runner_telemetry"], \
            b["usage"]["prompt_tokens"]

    def auto_twice(self):
        # the automatic tier: P, P + Q, an unrelated prompt, P + Q again
        with self.serve() as srv:
            _complete(srv, SYS)
            first, tel1, n = _complete(srv, SYS + Q1)
            _complete(srv, UNRELATED)
            second, tel2, _ = _complete(srv, SYS + Q1)
        return first, tel1, second, tel2, n

    def on_device(self):
        for log in self.logs:
            text = pathlib.Path(log).read_text(errors="replace")
            assert '"backend":"cuda"' in text or "CUDA backend" in text, text[-1500:]
            assert "continuing on the CPU" not in text, text[-1500:]


ARMS = [pytest.param(name, gpu, id=f"{name}-{'cpu' if gpu == 'off' else 'cuda'}")
        for gpu in ("off", "99") for name in RECURRENT]


def _arm(runner_bin, models, name, gpu, tmp_path):
    if gpu != "off" and not _cuda(runner_bin):
        pytest.skip("needs a CUDA device")
    return _Run(runner_bin, models[name], tmp_path, gpu)


@pytest.mark.parametrize("name,gpu", ARMS)
def test_a_named_context_serves_again_after_its_first_use(runner_bin, models, name, gpu, tmp_path):
    run = _arm(runner_bin, models, name, gpu, tmp_path)
    cold1, cold2 = run.cold(SYS + Q1), run.cold(SYS + Q2)
    # the same question again, and a question sharing words with the first
    for q, cold in ((Q1, cold1), (Q2, cold2)):
        n, first, tel1, second, tel2, _ = run.named_twice(q)
        assert tel1["prompt_cached_tokens"] == n and tel2["prompt_cached_tokens"] == n, (q, n, tel1, tel2)
        assert tel2["prompt_reuse"] == "prefix_cache" and tel2["context"] == {"id": "sys", "tokens": n}, tel2
        if gpu == "off":
            assert first == cold1 and second == cold, (q, first, second, cold1, cold)
        else:
            assert first[0] == cold1[0] and second[0] == cold[0], (q, first, second, cold1, cold)
            if q == Q1:
                assert second == first, (second, first)
    if gpu != "off":
        run.on_device()


@pytest.mark.parametrize("name,gpu", ARMS)
def test_the_automatic_tier_forks_a_prompt_again_after_an_unrelated_one(runner_bin, models, name, gpu, tmp_path):
    run = _arm(runner_bin, models, name, gpu, tmp_path)
    cold = run.cold(SYS + Q1)
    with run.serve() as srv:
        st, pin = _req(srv, {"_path": "/v1/runner/contexts", "id": "sys", "prompt": SYS})
        assert st == 200, pin
    n = pin["tokens"]
    first, tel1, second, tel2, _ = run.auto_twice()
    assert tel1["prompt_cached_tokens"] == n, tel1
    assert tel2["prompt_reuse"] == "prefix_cache" and tel2["prompt_cached_tokens"] == n, (
        "the automatic tier's second P + Q: P's entry, forked whole", tel2)
    assert second == first, (second, first)
    if gpu == "off":
        assert second == cold, (second, cold)
    else:
        assert second[0] == cold[0], (second, cold)
        run.on_device()


def test_a_recurrent_continuation_still_forks_its_longer_whole_entry(runner_bin, models, tmp_path):
    # P + Q1 published whole, then a prompt that extends it: that entry (longer than P's) is the one forked
    run = _Run(runner_bin, models["qwen4exp"], tmp_path, "off")
    with run.serve() as srv:
        _complete(srv, SYS)
        _, _, n_q1 = _complete(srv, SYS + Q1)
        _complete(srv, UNRELATED)
        _, tel, _ = _complete(srv, SYS + Q1 + " And item 7?")
    assert tel["prompt_reuse"] == "prefix_cache" and tel["prompt_cached_tokens"] == n_q1, (n_q1, tel)


def _stats(srv):
    with urllib.request.urlopen(srv.base_url + "/v1/runner/prefix-cache", timeout=30) as r:
        return json.load(r)


def _conversation(runner_bin, model, tmp_path, turns, mb):
    """A pinned context, then a conversation whose every turn extends the previous prompt, under a prefix budget of
    `mb` MB. Returns the server's numbers after every turn, and the requests made after the conversation."""
    log = tmp_path / f"conversation-{model.stem}.log"
    turn = "The next bin to check holds item %d, then the shelf beside it, and the one after that. "
    out = {"turns": []}
    with RunnerServer(runner_bin, model, ctx=2048, log_path=str(log), env=dict(os.environ, RUNNER_PREFIX_CACHE_MB=mb),
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        st, pin = _req(srv, {"_path": "/v1/runner/contexts", "id": "sys", "prompt": SYS})
        assert st == 200, pin
        out["pin"] = pin["tokens"]
        prompt, sizes = SYS, []
        for k in range(turns):
            prompt += turn.replace("%d", str(k))
            _, tel, n = _complete(srv, prompt)
            sizes.append(n)
            out["turns"].append((n, tel["prompt_cached_tokens"], _stats(srv)))
        # the newest turn's entry: a prompt extending it forks it whole
        _, out["newest"], _ = _complete(srv, prompt + "Where is item 42?")
        # the oldest turn's entry: a prompt extending it finds the pin at most
        first = SYS + turn.replace("%d", "0")
        _, out["oldest"], _ = _complete(srv, first + "Where is item 42?")
        _complete(srv, UNRELATED)
        _, out["named"], _ = _complete(srv, SYS + Q1, context_id="sys")
        out["sizes"], out["final"] = sizes, _stats(srv)
    return out


def test_a_recurrent_conversation_stays_within_the_prefix_budget(runner_bin, models, tmp_path):
    # Each turn's entry sits beside the last (a recurrent entry forks only at its own end), so a long conversation
    # under a small budget fills it: the total stays within the budget, the least recently used turns are evicted as
    # before, and the newest turn's entry and the pinned context survive.
    # 14 turns of the fixture's byte-level tokens: the last prompt about 1,870 tokens (ctx 2048), the turns' entries
    # together about 1.8 MB, past the 1 MB budget
    out = _conversation(runner_bin, models["qwen4exp"], tmp_path, turns=14, mb="1")
    budget = out["final"]["budget_bytes"]
    assert budget == 1 << 20, out["final"]
    for k, (n, cached, s) in enumerate(out["turns"]):
        assert s["bytes"] <= budget, (k, s)
        assert cached == (out["sizes"][k - 1] if k else out["pin"]), ("each turn forks the previous turn whole", k, n,
                                                                        cached)
    assert out["final"]["evictions"] > 0 and out["final"]["entries"] < len(out["turns"]) + 1, out["final"]
    assert out["newest"]["prompt_cached_tokens"] == out["sizes"][-1], out["newest"]
    assert out["oldest"]["prompt_cached_tokens"] == out["pin"], ("the oldest turn's entry was the one evicted",
                                                                 out["oldest"])
    assert out["named"]["prompt_cached_tokens"] == out["pin"], out["named"]


def test_a_dense_conversation_still_replaces_each_turns_entry(runner_bin, models, tmp_path):
    # a non-recurrent model's longer entry serves every shorter prefix, so each turn still replaces the last
    out = _conversation(runner_bin, models["dense"], tmp_path, turns=4, mb="1")
    assert [s["entries"] for _, _, s in out["turns"]] == [2, 2, 2, 2], out["turns"]
    assert out["final"]["evictions"] == 0, out["final"]


def test_a_dense_model_picks_as_before(runner_bin, models, tmp_path):
    # a non-recurrent model forks the longest common prefix, a partial hit included: each count is the longest share
    # there is (the request's own earlier prompt, bar the one token every request feeds)
    run = _Run(runner_bin, models["dense"], tmp_path, "off")
    n, _, tel1, _, tel2, n_prompt = run.named_twice(Q1)
    assert tel1["prompt_cached_tokens"] == n and tel2["prompt_cached_tokens"] == n_prompt - 1, (n, n_prompt, tel2)
    # a second question sharing words with the first: the first request's entry, forked partly, past the pin
    n, _, _, _, tel_q2, n_prompt2 = run.named_twice(Q2)
    assert n < tel_q2["prompt_cached_tokens"] < n_prompt2 - 1, (n, tel_q2, n_prompt2)
    _, tel1, _, tel2, n_prompt = run.auto_twice()
    assert tel1["prompt_cached_tokens"] == n and tel2["prompt_cached_tokens"] == n_prompt - 1, (tel1, tel2, n_prompt)


@pytest.mark.parametrize("name,gpu", ARMS)
def test_a_shorter_recurrent_prefix_published_after_a_longer_one_is_kept(runner_bin, models, name, gpu, tmp_path):
    # Publication order: P+Q1, then P, then P+Q2. Clear the live slot before
    # the last request so only the shared tier can supply P's fold.
    run = _arm(runner_bin, models, name, gpu, tmp_path)
    cold = run.cold(SYS + Q2)
    with run.serve() as srv:
        _complete(srv, SYS + Q1)
        _, _, n_p = _complete(srv, SYS)
        _complete(srv, SYS)  # an equal-length publication must refresh, not duplicate
        after_shorter = _stats(srv)
        _complete(srv, UNRELATED, cache_prompt=False)
        answer, tel, _ = _complete(srv, SYS + Q2)
    assert tel["prompt_reuse"] == "prefix_cache" and tel["prompt_cached_tokens"] == n_p, (
        "the later shorter entry must retain its own fold", n_p, tel, after_shorter)
    assert after_shorter["entries"] == 2, after_shorter
    if gpu == "off":
        assert answer == cold, (answer, cold)
    else:
        assert answer[0] == cold[0], (answer, cold)
        run.on_device()


def test_a_dense_shorter_prefix_still_uses_the_covering_entry(runner_bin, models, tmp_path):
    run = _Run(runner_bin, models["dense"], tmp_path, "off")
    with run.serve() as srv:
        _complete(srv, SYS + Q1)
        _, shorter, n_p = _complete(srv, SYS)
        assert _stats(srv)["entries"] == 1
        _complete(srv, UNRELATED, cache_prompt=False)
        _, later, n_later = _complete(srv, SYS + Q2)
    assert shorter["prompt_cached_tokens"] == n_p - 1, shorter
    assert n_p < later["prompt_cached_tokens"] < n_later - 1, later


def test_shorter_recurrent_entries_keep_the_existing_budget_and_pin(runner_bin, models, tmp_path):
    # Descending prefixes exercise new entries from the longer-first path.
    # Like the ascending conversation above, they fill a 1 MiB budget.
    turn = "The next bin holds item %d, then the shelf beside it, and the one after that. "
    prompts = [SYS]
    for k in range(16):
        prompts.append(prompts[-1] + turn.replace("%d", str(k)))
    with RunnerServer(runner_bin, models["qwen4exp"], ctx=2048,
                      log_path=str(tmp_path / "shorter-budget.log"),
                      env=dict(os.environ, RUNNER_PREFIX_CACHE_MB="1"),
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        status, pin = _req(srv, {"_path": "/v1/runner/contexts", "id": "sys", "prompt": SYS})
        assert status == 200, pin
        for prompt in reversed(prompts[1:]):
            _complete(srv, prompt)
            stats = _stats(srv)
            assert stats["budget_bytes"] == 1 << 20 and stats["bytes"] <= stats["budget_bytes"], stats
        assert stats["evictions"] > 0 and 2 < stats["entries"] < len(prompts), stats
        _complete(srv, UNRELATED, cache_prompt=False)
        _, tel, _ = _complete(srv, SYS + Q2, context_id="sys")
        assert tel["prompt_cached_tokens"] == pin["tokens"], tel
