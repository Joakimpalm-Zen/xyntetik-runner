"""phi3: the sliding window covers every layer and never changes the rotation.

phi3 exports publish a sliding window with no pattern key (Phi-3.5: 262144
over a 131072 context). The loader once read that as gemma2's alternation,
so every other layer roped from the raw local table, without the LongRoPE
short factors or the attention factor, and Phi-3.5 sat at mean KL 0.275
from its float32 reference where another engine on the same file reads
0.049. The fixture carries non-identity short factors, so a layer that ropes
without them moves the logits: a window at least the trained context must
give the same bytes as no window at all, and a window shorter than the
prompt must still mask (the window path is live, not switched off).

LongRoPE files carry two factor sets, and the publisher's code picks one per
sequence: the short set while the sequence is no longer than the model's
original context, the long set for a longer one. The runner picked from the
requested context, so with a context over the original one every sequence
took the long set from its first token: Phi-3.5-mini read 0.205 mean KLD on
a 4,096-token text against 0.004 at the original context. The set now
follows the sequence (R4.40): a short text at a larger context scores as at
the original one, a sequence that grows past it is computed again under the
long set, and a shorter prompt after it goes back. The bit-exact gates are
tests/test_longrope_sequence.c; this file holds what a user sees.
"""
import json
import urllib.request

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import score_record  # noqa: E402
PROMPT = "the runner reads a prompt in windows and every weight matrix is multiplied"


def _make(tmp_path, name, *extra):
    path = tmp_path / f"{name}.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    "--arch", "phi3", *extra, str(path)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


def _score(runner_bin, model):
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "--score", "-p", PROMPT, "--gpu", "off",
         "--no-tray", "-c", "256"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert b"phi3 LongRoPE short factors" in proc.stderr
    assert b'"n_scored"' in proc.stdout
    # the scores without score_route, which names the model file
    return score_record.body_bytes(proc.stdout)


def test_phi3_window_past_the_context_changes_nothing(runner_bin, tmp_path):
    plain = _score(runner_bin, _make(tmp_path, "plain"))
    wide = _score(runner_bin, _make(tmp_path, "wide", "--swa-window", "512"))
    assert plain == wide


def test_phi3_window_shorter_than_the_prompt_still_masks(runner_bin, tmp_path):
    plain = _score(runner_bin, _make(tmp_path, "plain"))
    short = _score(runner_bin, _make(tmp_path, "short", "--swa-window", "16"))
    assert plain != short


def _load_lines(runner_bin, model, ctx):
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "--score", "-p", PROMPT, "--gpu", "off",
         "--no-tray", "-c", str(ctx)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return proc.stderr.decode(errors="replace")


def test_phi3_context_over_the_original_takes_both_sets_and_warns_of_nothing(runner_bin, tmp_path):
    model = _make(tmp_path, "orig128", "--longrope-orig", "128")
    err = _load_lines(runner_bin, model, 256)
    assert "short factors for a sequence of up to 128 tokens and long factors for a longer one" in err
    assert "warning: -c" not in err, err


def test_phi3_context_within_the_original_says_nothing(runner_bin, tmp_path):
    model = _make(tmp_path, "orig128", "--longrope-orig", "128")
    err = _load_lines(runner_bin, model, 128)
    assert "phi3 LongRoPE short factors" in err
    assert "warning: -c" not in err


def _logprobs(runner_bin, model, ctx, prompt):
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "--score", "-p", prompt, "--gpu", "off",
         "--no-tray", "-c", str(ctx)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    d = json.loads(proc.stdout)
    return d["n_tokens"], d["logprobs"], proc.stderr.decode(errors="replace")


def test_phi3_short_text_at_a_larger_context_scores_as_at_the_original(runner_bin, tmp_path):
    """The defect: at -c 256 a 102-token text was rotated with the long
    factors. Its scores are now the ones -c 128 gives, to the bit."""
    model = _make(tmp_path, "orig128", "--longrope-orig", "128")
    n, at_orig, _ = _logprobs(runner_bin, model, 128, PROMPT)
    n2, wide, err = _logprobs(runner_bin, model, 256, PROMPT)
    assert n == n2 and n <= 128, (n, n2)
    assert wide == at_orig
    assert "computed again" not in err


def test_phi3_long_text_takes_the_long_set_from_its_first_token(runner_bin, tmp_path):
    """A text longer than the original context is one sequence on the long
    set: nothing is computed twice, and its first 102 tokens do not score as
    the same 102 tokens alone do."""
    model = _make(tmp_path, "orig128", "--longrope-orig", "128")
    n, short, _ = _logprobs(runner_bin, model, 256, PROMPT)
    n2, long, err = _logprobs(runner_bin, model, 256, PROMPT + " and " + PROMPT)
    assert n <= 128 < n2 <= 256, (n, n2)
    assert "computed again" not in err
    assert long[:len(short)] != short


def _complete(srv, prompt, n, **fields):
    body = {"prompt": prompt, "max_tokens": n, "temperature": 0, "seed": 1,
            "prefix_cache": False}
    body.update(fields)
    req = urllib.request.Request(srv.base_url + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


@pytest.mark.parametrize("gpu", ["off", "auto"])
def test_phi3_served_sequence_changes_set_where_it_passes_the_original_context(runner_bin, tmp_path, gpu):
    """Served, on the CPU and on the Apple GPU (which keeps a rope table per
    slot): a completion that generates past the original context is computed
    again once, the next shorter prompt keeps nothing of it and says why, and
    each answer is the one a slot with no history gives."""
    sys.path.insert(0, str(ROOT / "tests" / "conformance"))
    from harness import RunnerServer
    if gpu == "auto":
        caps = json.loads(subprocess.run([runner_bin, "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
                                         check=True).stdout)
        if (caps.get("gpu") or {}).get("backend") != "metal":
            pytest.skip("the Metal backend's per-slot rope table")
    model = _make(tmp_path, "orig128", "--longrope-orig", "128")
    n_short, _, _ = _logprobs(runner_bin, model, 256, PROMPT)
    grow = 128 - n_short + 12          # generation ends 12 tokens past the original context
    log_path = tmp_path / "server.log"
    with RunnerServer(str(runner_bin), model, ctx=256, parallel=1, log_path=str(log_path),
                      extra_args=["--gpu", gpu, "-t", "2"]) as srv:
        def reuse(d):
            return d["runner_telemetry"]["prompt_reuse"]
        fresh_short = _complete(srv, PROMPT, 8, cache_prompt=False)
        grown = _complete(srv, PROMPT, grow, cache_prompt=False, ignore_eos=True)
        assert grown["usage"]["completion_tokens"] == grow, grown["usage"]
        # the same prompt again: the slot holds a sequence on the long set
        after = _complete(srv, PROMPT, 8)
        assert reuse(after) == "rope_set", reuse(after)
        assert after["choices"][0]["text"] == fresh_short["choices"][0]["text"]
        # the grown text as a prompt of its own is long from its first token
        whole = _complete(srv, PROMPT + grown["choices"][0]["text"][:-4], 4,
                          cache_prompt=False, ignore_eos=True)
        assert whole["usage"]["prompt_tokens"] == n_short + grow - 4, whole["usage"]
        assert whole["choices"][0]["text"] == grown["choices"][0]["text"][-4:]
    log = log_path.read_text(encoding="utf-8", errors="replace")
    assert ("gpu: Metal backend" in log) == (gpu == "auto"), log[:2000]
    assert log.count("positions are computed again with the long factors") == 1, log[-2000:]
