"""RUNNER_MOE_REPLAY (R3.21.2): routing recorded by RUNNER_MOE_TRACE on one
run, forced on another, on the CPU routers.

The fixture is make-test-moe.py's moe1: two experts, one used, a zero router
(uniform logits, so top-1 is expert 0 by index), expert 0 equal to the dense
FFN and expert 1 all zeros. Replaying a run's own trace must leave its output
byte-identical and count no differing choice; a trace that names expert 1
must change the output and count the choices it overrode; a trace with
another expert count, or one that is not a trace, is refused; and without the
variable nothing is said or changed.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = "The quick brown fox"


@pytest.fixture(scope="module")
def moe1(tmp_path_factory):
    base = tmp_path_factory.mktemp("replay") / "m"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(base)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return pathlib.Path(str(base) + ".moe1.gguf")


def gen(runner_bin, model, **env):
    e = {k: v for k, v in os.environ.items()
         if k not in ("RUNNER_MOE_TRACE", "RUNNER_MOE_REPLAY")}
    e.update({k: str(v) for k, v in env.items()})
    p = subprocess.run([str(runner_bin), "-m", str(model), "-p", PROMPT, "-n", "8",
                        "--temp", "0", "--gpu", "off", "-t", "2"],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=120, env=e)
    return p.returncode, p.stdout, p.stderr.decode(errors="replace")


def counts(err):
    line = next(l for l in err.splitlines() if l.startswith("moe-replay: ") and "forced" in l)
    nums = [int(w) for w in line.replace(",", " ").split() if w.isdigit()]
    return nums[0], nums[1]                   # forced, differed


def test_replay_pins_routing_and_counts_what_it_overrode(runner_bin, moe1, tmp_path):
    rc, plain, err = gen(runner_bin, moe1)
    assert rc == 0, err[-400:]
    assert "moe-replay" not in err            # off by default, silent
    trace = tmp_path / "trace.jsonl"
    rc, traced, err = gen(runner_bin, moe1, RUNNER_MOE_TRACE=trace)
    assert rc == 0 and traced == plain
    recs = [json.loads(l) for l in trace.read_text().splitlines()]
    assert recs and all(r["experts"] == [0] for r in recs)

    # its own routing again: the same bytes, nothing overridden
    rc, same, err = gen(runner_bin, moe1, RUNNER_MOE_REPLAY=trace)
    assert rc == 0, err[-400:]
    assert same == plain
    forced, differed = counts(err)
    assert forced == len(recs) and differed == 0

    # expert 1 (all zeros) at every recorded routing: the FFN contributes
    # nothing, so the text changes, and every choice was an override
    flipped = tmp_path / "flipped.jsonl"
    flipped.write_text("".join(json.dumps({**r, "experts": [1]}) + "\n" for r in recs))
    rc, other, err = gen(runner_bin, moe1, RUNNER_MOE_REPLAY=flipped)
    assert rc == 0, err[-400:]
    assert other != plain
    forced, differed = counts(err)
    assert forced == differed == len(recs)


def test_replay_refuses_a_trace_of_another_model_or_no_trace(runner_bin, moe1, tmp_path):
    two = tmp_path / "two.jsonl"
    two.write_text(json.dumps({"pos": 0, "layer": 0, "experts": [0, 1]}) + "\n")
    rc, _, err = gen(runner_bin, moe1, RUNNER_MOE_REPLAY=two)
    assert rc != 0 and "this model routes 1" in err
    junk = tmp_path / "junk.jsonl"
    junk.write_text("not a trace\n")
    rc, _, err = gen(runner_bin, moe1, RUNNER_MOE_REPLAY=junk)
    assert rc != 0 and "not RUNNER_MOE_TRACE records" in err
    rc, _, err = gen(runner_bin, moe1, RUNNER_MOE_REPLAY=tmp_path / "absent.jsonl")
    assert rc != 0 and "cannot be opened" in err
