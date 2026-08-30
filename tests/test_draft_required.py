"""`--draft-required`: turn a silently-dropped draft into a failed run.

A draft is refused at load on a vocabulary mismatch, a fully GPU-offloaded
target, a CUDA-resident recurrent state, an unreadable or unsupported file, or
out of memory, and the run then continues WITHOUT it. That default is
deliberate and documented, and it stays.

The gap it leaves is machine-readability, and only in one-shot mode. In serve
mode `GET /v1/capabilities` reports whether the draft is actually `active`, so
a harness can tell speculative decoding from its fallback. One-shot mode has no
such channel: the drop is a stderr line beside a successful exit, so a harness
that collects stdout and checks the return code records the unaccelerated
baseline and labels it speculative decoding. Measured externally 2026-08-30 on
a draft Runner cannot run at all.

`--draft-required` is the opt-in that closes it: the run fails rather than
silently measuring something else. The default is untouched, which is why the
first test here is a regression guard on it rather than an afterthought.
"""
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = "hello"


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
def unusable_draft(tmp_path_factory):
    """A file that is definitely not a loadable draft."""
    p = tmp_path_factory.mktemp("draft") / "not-a-model.gguf"
    p.write_bytes(b"GGUF this is not a model at all, only a header tease")
    return p


def _run(runner_bin, model, extra):
    return subprocess.run(
        [runner_bin, "-m", str(model), "-p", PROMPT, "-n", "1",
         "-t", "2", "--gpu", "off", *extra],
        cwd=ROOT, env=dict(os.environ),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)


def test_unusable_draft_is_dropped_and_the_run_still_succeeds(
        runner_bin, model, unusable_draft):
    """The documented default, pinned so the strict flag cannot change it."""
    p = _run(runner_bin, model, ["--draft", str(unusable_draft)])
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert p.stdout.strip(), "a dropped draft must still produce a completion"


def test_draft_required_fails_closed_on_an_unusable_draft(
        runner_bin, model, unusable_draft):
    p = _run(runner_bin, model,
             ["--draft", str(unusable_draft), "--draft-required"])
    assert p.returncode != 0, "an unusable draft must not exit 0 under --draft-required"
    err = p.stderr.decode(errors="replace")
    assert "--draft-required" in err or "draft" in err


def test_draft_required_is_transparent_when_the_draft_loads(
        runner_bin, model):
    """Same model as its own draft: loads, so the flag must change nothing."""
    plain = _run(runner_bin, model, ["--draft", str(model)])
    strict = _run(runner_bin, model,
                  ["--draft", str(model), "--draft-required"])
    assert plain.returncode == 0, plain.stderr.decode(errors="replace")
    assert strict.returncode == 0, strict.stderr.decode(errors="replace")
    assert strict.stdout == plain.stdout


def test_draft_required_without_a_draft_is_rejected(runner_bin, model):
    """The flag is meaningless alone, and a silent no-op would defeat it."""
    p = _run(runner_bin, model, ["--draft-required"])
    assert p.returncode != 0
    assert b"--draft" in p.stderr


def test_draft_required_is_refused_in_serve_mode(runner_bin, model):
    """Refused, not ignored: serve mode already answers over the wire.

    A guard against silent no-ops that was itself a silent no-op would be the
    exact failure it exists to prevent, so serve mode names the channel that
    does answer the question instead of accepting a flag it does not honour.
    """
    p = subprocess.run(
        [runner_bin, "-m", str(model), "--serve", "--draft", str(model),
         "--draft-required"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert p.returncode != 0
    assert b"/v1/capabilities" in p.stderr
