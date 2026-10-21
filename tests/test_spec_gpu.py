"""Speculative decoding on a fully GPU-offloaded target (Metal and CUDA).

Historically refused ("needs the CPU verify path"); the forward now emits
every row's head for the walk (model_forward_verify / m->spec_want_all), and
each row is the solo step's bits on every backend (tests/test_spec_identity.c
holds that byte for byte). The contract tested here is speculation's own: the
draft NEVER changes the output — greedy with a draft must be byte-identical
to greedy without, and the run must actually speculate (anti-vacuity:
"spec:" telemetry with accepted tokens, no "ignoring --draft"). Self-draft
(the model drafting for itself) makes the acceptance near-total and the
fixture cheap; prompt lookup drafts from the context.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    caps = subprocess.run([exe, "--caps"], cwd=ROOT,
                          stdout=subprocess.PIPE, timeout=30)
    gpu = json.loads(caps.stdout.decode()).get("gpu")
    if not gpu or gpu.get("backend") not in ("metal", "cuda"):
        pytest.skip("needs a Metal or CUDA device for the full-offload verify")
    return exe


def run(runner_bin, *extra):
    proc = subprocess.run(
        [runner_bin, "-m", "test.gguf", "-p", "The quick brown fox",
         "-n", "24", "--temp", "0", "--no-tray", *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return proc.stdout, proc.stderr.decode(errors="replace")


def test_draft_engages_and_output_is_target_exact(runner_bin):
    plain, _ = run(runner_bin)
    drafted, err = run(runner_bin, "--draft", "test.gguf")
    assert "ignoring --draft" not in err, err
    assert "spec:" in err and " accepted" in err, err
    assert drafted == plain, "speculation changed the output — never allowed"


def test_prompt_lookup_is_target_exact(runner_bin):
    plain, _ = run(runner_bin, "-n", "64")
    drafted, err = run(runner_bin, "-n", "64", "--draft-lookup", "--draft-k", "12")
    assert "spec:" in err, err
    assert drafted == plain, "speculation changed the output — never allowed"


def test_cpu_only_path_still_speculates(runner_bin):
    # the pre-existing regime: --gpu off was always allowed
    plain, _ = run(runner_bin, "--gpu", "off")
    drafted, err = run(runner_bin, "--gpu", "off", "--draft", "test.gguf")
    assert "ignoring --draft" not in err
    assert drafted == plain
