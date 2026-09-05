"""`--draft-lookup`: prompt n-gram drafts through the verify walk.

The fourth draft source needs no weights and no draft forward: each round the
last n context tokens (prompt plus everything generated so far) are looked up
in the context itself, and the tokens that followed their most recent earlier
occurrence are proposed, up to --draft-k. The target verifies them in the
same batched walk the draft model, the MTP head and grammar fast-forward use,
so output is token-identical to plain decoding whatever the drafts.

Gates here (the fixture half; the real-model half is docs/context-drafts.md):
  - identity: greedy bytes AND seeded-sampling token ids (from --transcript)
    match the undrafted run on an input-grounded prompt, on a prompt with no
    repeats, and on a multi-turn chat that echoes a tool result;
  - the lookup is actually driven (rounds and drafts counted) on a prompt
    with repeats, and proposes NOTHING on a prompt with none;
  - the source is exclusive with --draft and --mtp and says so at startup;
  - accounting: the transcript carries the source and its counters.

Absolute anchor: the search itself is pinned against hand-computed proposals
in tests/test_lookup_draft.c; this file proves the wiring, not the search.
The real-model anchor (accepted > 0 on an echo prompt, which a search that
matched the wrong tokens could not produce) runs on SmolLM2-135M when the
file is present and is skipped, not passed, otherwise.
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

SMOL = ROOT / "models" / "SmolLM2-135M-Instruct-Q8_0.gguf"

# The prompt itself repeats, so the lookup has a match from the first round
# whatever the (random-weight) fixture generates.
REPEAT = ("the cat sat on the mat and the dog sat on the mat and the cat sat "
          "on the mat and the dog sat on the mat and the cat sat on the")
# No token bigram recurs here under any tokenizer: every byte is distinct, so
# no pair of adjacent tokens can occur twice however the bytes are grouped.
NOREPEAT = "abcdefghijklmnopqrstuvwxyz0123456789"
# A chat where the tool result is echoed back verbatim: the shape prompt
# lookup exists for.
TOOL_ECHO = ("<|user|>\nRun ls.\n<|tool|>\nREADME.md src tests Makefile "
             "LICENSE docs\n<|assistant|>\nThe directory contains: README.md "
             "src tests Makefile LICENSE docs\n<|user|>\nRepeat that.\n"
             "<|assistant|>\nThe directory contains: README.md src tests")


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model():
    m = ROOT / "test.gguf"
    if not m.exists():
        pytest.skip("test.gguf fixture not built")
    return m


@pytest.fixture(scope="module")
def mtp_model(tmp_path_factory):
    out = tmp_path_factory.mktemp("lookup") / "mtp.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    "--mtp-layers", "1", str(out)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return out


def _run(runner_bin, model, *extra, prompt=REPEAT, n=48, temp="0"):
    return subprocess.run(
        [runner_bin, "-m", str(model), "-p", prompt, "-n", str(n), "-c", "1024",
         "--temp", temp, "--gpu", "off", "-t", "2", *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)


def _stats(stderr):
    """Parse `spec: R rounds, D drafted, A accepted (...), grammar g/G, lookup l/L`."""
    for line in stderr.decode(errors="replace").splitlines():
        if line.startswith("spec:"):
            parts = line.replace(",", " ").replace("(", " ").split()
            rounds, drafted, accepted = int(parts[1]), int(parts[3]), int(parts[5])
            lk = parts[parts.index("lookup") + 1].split("/")
            return rounds, drafted, accepted, int(lk[1]), int(lk[0])
    return None


@pytest.mark.parametrize("prompt", [REPEAT, NOREPEAT, TOOL_ECHO],
                         ids=["repeat", "norepeat", "tool_echo"])
@pytest.mark.parametrize("k", ["1", "4"])
def test_lookup_never_changes_greedy_output(runner_bin, model, prompt, k):
    plain = _run(runner_bin, model, prompt=prompt)
    assert plain.returncode == 0, plain.stderr.decode(errors="replace")
    got = _run(runner_bin, model, "--draft-lookup", "--draft-k", k, prompt=prompt)
    assert got.returncode == 0, got.stderr.decode(errors="replace")
    assert got.stdout == plain.stdout, "the lookup must only propose, never decide"
    err = got.stderr.decode(errors="replace")
    assert "lookup: drafting from the context" in err
    stats = _stats(got.stderr)
    assert stats is not None, err
    rounds, drafted, accepted, lk_drafted, lk_accepted = stats
    assert rounds > 0
    assert drafted == lk_drafted, "every draft this run came from the lookup"
    assert accepted == lk_accepted
    assert accepted <= drafted and drafted <= rounds * int(k)


def test_lookup_is_driven_on_a_repeating_prompt(runner_bin, model):
    got = _run(runner_bin, model, "--draft-lookup", "--draft-k", "4")
    assert got.returncode == 0, got.stderr.decode(errors="replace")
    rounds, drafted, _, lk_drafted, _ = _stats(got.stderr)
    assert lk_drafted > 0, "the prompt repeats, so the first rounds must draft"


def test_lookup_proposes_nothing_without_a_match(runner_bin, model):
    """A round with no n-gram match must cost nothing, not guess. Two
    tokens: the first is sampled before any round, the one round after it
    searches for (last prompt token, first output token), which cannot recur
    because the last prompt token occurs once."""
    got = _run(runner_bin, model, "--draft-lookup", prompt=NOREPEAT, n=2)
    assert got.returncode == 0, got.stderr.decode(errors="replace")
    rounds, drafted, _, lk_drafted, _ = _stats(got.stderr)
    assert rounds > 0
    assert drafted == 0 and lk_drafted == 0, got.stderr.decode(errors="replace")


def _tokens(runner_bin, model, path, *extra, prompt=REPEAT):
    # --ignore-eos: the random fixture can sample end-of-text at once, and a
    # zero-token run would make the identity below vacuous
    p = _run(runner_bin, model, "--transcript", str(path), "-s", "11",
             "--ignore-eos", *extra, prompt=prompt, temp="0.9", n=40)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(path.read_bytes())


@pytest.mark.parametrize("prompt", [REPEAT, TOOL_ECHO], ids=["repeat", "tool_echo"])
def test_seeded_sampling_ids_are_identical(runner_bin, model, tmp_path, prompt):
    plain = _tokens(runner_bin, model, tmp_path / "plain.json", prompt=prompt)
    got = _tokens(runner_bin, model, tmp_path / "lookup.json", "--draft-lookup",
                  prompt=prompt)
    assert plain["output"]["tokens"] == got["output"]["tokens"]
    assert len(got["output"]["tokens"]) == 40
    # the transcript attributes the run to its source, with the counters
    assert "speculation" not in plain
    sp = got["speculation"]
    assert sp["source"] == "lookup"
    assert sp["rounds"] > 0 and sp["drafted"] >= sp["accepted"] >= 0
    assert sp["lookup_drafted"] == sp["drafted"]
    assert sp["lookup_accepted"] == sp["accepted"]


def test_lookup_is_exclusive_with_the_other_sources(runner_bin, model, mtp_model):
    both = _run(runner_bin, model, "--draft-lookup", "--draft", str(model), n=1)
    assert both.returncode != 0
    assert "one draft source per run" in both.stderr.decode(errors="replace")
    both = _run(runner_bin, mtp_model, "--draft-lookup", "--mtp", n=1)
    assert both.returncode != 0
    assert "one draft source per run" in both.stderr.decode(errors="replace")


@pytest.mark.parametrize("source", ["lookup", "mtp", "model"])
@pytest.mark.parametrize("limit", ["100", "-1"])
def test_speculation_stops_at_context_boundary(runner_bin, mtp_model, tmp_path,
                                             source, limit):
    """A 32-position context has exactly 32 minus prompt-length output seats.
    The final verify row must consume its pending token without emitting a
    bonus outside that allocation; transcript writing reads those same seats.
    """
    flags = {"lookup": ["--draft-lookup"], "mtp": ["--mtp"],
             "model": ["--draft", str(mtp_model)]}[source]
    records = []
    for name, extra in (("plain", []), (source, flags)):
        path = tmp_path / f"{name}.json"
        p = subprocess.run(
            [runner_bin, "-m", str(mtp_model), "-p", "hello", "-c", "32",
             "-n", limit, "--temp", "0", "--gpu", "off", "-t", "2",
             "--ignore-eos", "--transcript", str(path), *extra],
            cwd=ROOT, capture_output=True, timeout=60)
        assert p.returncode == 0, p.stderr.decode(errors="replace")
        rec = json.loads(path.read_bytes())
        assert rec["output"]["n"] == 32 - len(rec["prompt"]["tokens"])
        assert len(rec["output"]["tokens"]) == rec["output"]["n"]
        records.append(rec)
    assert records[0]["output"] == records[1]["output"]


@pytest.mark.skipif(not SMOL.exists(), reason="SmolLM2-135M not present")
def test_real_model_accepts_echoed_context(runner_bin):
    """The real-model anchor: a trained model continues a sentence it has
    already seen three times, so a correct search gets its drafts accepted.
    A search proposing the wrong tokens would still pass the identity gate
    (the walk rejects them) but could not pass this one. Measured 2026-09-04:
    35 of 40 drafts accepted, 3.69 tokens per round."""
    sentence = ("The quick brown fox jumps over the lazy dog near the river "
                "bank. ")
    prompt = sentence * 3 + "The quick brown fox jumps over the"
    plain = _run(runner_bin, SMOL, prompt=prompt, n=48)
    got = _run(runner_bin, SMOL, "--draft-lookup", prompt=prompt, n=48)
    assert plain.returncode == 0 and got.returncode == 0
    assert got.stdout == plain.stdout
    rounds, drafted, accepted, _, lk_accepted = _stats(got.stderr)
    assert drafted > 0 and lk_accepted == accepted
    assert accepted >= drafted // 2, "a trained model continues its own echo"
    assert rounds < 48 // 2, "and the walk emits more than two tokens a round"


def _get(server, path):
    with urllib.request.urlopen(server.base_url + path, timeout=30) as r:
        return r.read()


def _post(server, path, body):
    req = urllib.request.Request(
        server.base_url + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _metric(text, name):
    for line in text.splitlines():
        if line.startswith(name + " "):
            return int(float(line.split()[1]))
    raise AssertionError(f"{name} missing from /metrics")


def test_serve_reports_the_source_and_accounts_per_request(model):
    """Accounting gate, server side: /v1/capabilities names the source,
    runner_telemetry carries the per-request counters attributed to it, and
    the three runner_speculation_* counters on /metrics keep counting every
    source (the names PR #19 pinned)."""
    with RunnerServer(find_runner(ROOT), model, ctx=1024, parallel=1,
                      extra_args=["--gpu", "off", "--draft-lookup"]) as srv:
        caps = json.loads(_get(srv, "/v1/capabilities"))
        assert caps["draft"] == {"requested": True, "active": True,
                                 "source": "lookup"}
        before = _get(srv, "/metrics").decode()
        out = _post(srv, "/v1/completions",
                    {"prompt": REPEAT, "max_tokens": 24, "temperature": 0,
                     "cache_prompt": False})
        t = out["runner_telemetry"]
        assert t["speculative"] is True
        sp = t["speculation"]
        assert sp["source"] == "lookup"
        assert sp["rounds"] > 0 and sp["drafted"] > 0
        assert sp["lookup_drafted"] == sp["drafted"]
        assert sp["lookup_accepted"] == sp["accepted"] <= sp["drafted"]
        after = _get(srv, "/metrics").decode()
        for name, key in (("runner_speculation_rounds_total", "rounds"),
                          ("runner_speculation_drafted_tokens_total", "drafted"),
                          ("runner_speculation_accepted_tokens_total", "accepted")):
            assert _metric(after, name) - _metric(before, name) == sp[key], name
        # plain decoding keeps its telemetry byte-shape: no speculation object
        plain = _post(srv, "/v1/completions",
                      {"prompt": REPEAT, "max_tokens": 4, "temperature": 0,
                       "logprobs": 1, "cache_prompt": False})
        assert plain["runner_telemetry"]["speculative"] is False
        assert "speculation" not in plain["runner_telemetry"]
