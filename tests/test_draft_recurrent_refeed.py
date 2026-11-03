"""R3.3.8: a recurrent draft model no longer refeeds its whole context on
every rejection.

A draft whose layers fold state (LFM2's short convolutions, a DeltaNet) cannot
slice its fold back to the accepted prefix, so until 2026-11-03 every
rejection sent it to position 0 and the next catch-up refeeded the whole
context (the SS-000 study: an LFM2.5-350M draft for LFM2.5-1.2B on the CPU,
113 rounds at about 526 ms of draft work each, 0.13x on prose). Each round now
snapshots the draft's fold after its catch-up and a rejection restores it, the
target's own rollback pattern, so a round feeds the draft at most draft_k + 1
tokens.

The gate is the two properties that define the fix: greedy output identical
to plain decoding (the draft only proposes), and the tokens fed to the draft
bounded by the prompt plus draft_k + 1 a round, counted by the runner itself
(the `draft catch-up` field of the spec line: the tokens the draft is fed
to catch up, apart from the draft_k it drafts each round). The draft is the
same fixture's MoE twin, which shares the vocabulary and disagrees often
enough that rejections happen; a run without one would prove nothing and
fails. (The convolution-bias twin is refused at load as a draft, and a
refused draft runs no speculation at all: the test reads the spec line.)
"""
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = "the quick brown fox jumps over the lazy dog and the cat sat on the mat " * 6


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("rdraft")
    target, draft = d / "target.gguf", d / "draft.gguf"
    for out, extra in ((target, []), (draft, ["--moe"])):
        subprocess.run([sys.executable, ROOT / "scripts/make-test-lfm2.py", str(out), *extra],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return target, draft


def _run(runner_bin, target, *extra):
    p = subprocess.run([runner_bin, "-m", str(target), "-p", PROMPT, "-n", "48", "--temp", "0",
                        "--gpu", "off", "-t", "2", "--no-tray", *extra],
                       cwd=ROOT, capture_output=True, timeout=300)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p


@pytest.mark.parametrize("k", ["1", "4"])
def test_a_rejected_round_feeds_the_draft_at_most_k_plus_one_tokens(runner_bin, pair, k):
    target, draft = pair
    plain = _run(runner_bin, target)
    spec = _run(runner_bin, target, "--draft", str(draft), "--draft-k", k)
    assert spec.stdout == plain.stdout          # the draft only proposes
    err = spec.stderr.decode(errors="replace")
    m = re.search(r"spec: (\d+) rounds, (\d+) drafted, (\d+) accepted .* draft catch-up (\d+)", err)
    assert m, err
    rounds, drafted, accepted, fed = (int(g) for g in m.groups())
    assert 0 < accepted < drafted, (accepted, drafted)   # rejections happened
    n_prompt = int(re.search(r"prompt: (\d+) tok", err).group(1))
    # the catch-up of the prompt (and its pending token) once, then at most
    # draft_k + 1 tokens a round; a refeed from 0 on a rejection breaks this
    bound = n_prompt + 1 + rounds * (int(k) + 1)
    assert fed <= bound, (fed, bound, rounds, n_prompt)
