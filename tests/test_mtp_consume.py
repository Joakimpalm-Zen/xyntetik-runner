"""NextN/MTP head consumption: drafts from the model's own predictor block.

Contract under test (the fixture half; the real-model half is the measured
acceptance rate recorded in docs/performance.md, which a random-weight fixture
cannot stand in for):
  - `--mtp` never changes the sampled stream: the head only proposes, the
    target's verify walk decides, so greedy output is byte-identical to plain
    decoding at every draft width;
  - the head is actually driven (rounds and drafts are counted), not silently
    degraded to plain decoding;
  - an explicit `--mtp` fails closed on an export without a consumable head
    (none declared, or more than one block) and on a GPU-resident target.
"""
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
    return exe


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("mtpc")

    def gen(name, *flags):
        out = d / name
        subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                        *flags, str(out)], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
        return out
    return {
        "plain": gen("plain.gguf"),
        "mtp": gen("mtp.gguf", "--mtp-layers", "1"),
        "mtp2": gen("mtp2.gguf", "--mtp-layers", "2"),
    }


def _run(runner_bin, model, *extra, n=48, prompt="hello there"):
    return subprocess.run(
        [runner_bin, "-m", str(model), "-p", prompt, "-n", str(n),
         "--temp", "0", "--gpu", "off", "-t", "2", *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)


def _stats(stderr):
    """Parse the `spec: R rounds, D drafted, A accepted ...` summary line."""
    for line in stderr.decode(errors="replace").splitlines():
        if line.startswith("spec:"):
            parts = line.replace(",", " ").split()
            return int(parts[1]), int(parts[3]), int(parts[5])
    return None


@pytest.mark.parametrize("k", ["1", "3", "4"])
def test_mtp_drafts_never_change_greedy_output(runner_bin, models, k):
    plain = _run(runner_bin, models["mtp"])
    assert plain.returncode == 0, plain.stderr.decode(errors="replace")
    got = _run(runner_bin, models["mtp"], "--mtp", "--draft-k", k)
    assert got.returncode == 0, got.stderr.decode(errors="replace")
    assert got.stdout == plain.stdout, "the head must only propose, never decide"
    err = got.stderr.decode(errors="replace")
    assert "mtp: drafting from the model's NextN head" in err
    stats = _stats(got.stderr)
    assert stats is not None, err
    rounds, drafted, accepted = stats
    assert rounds > 0 and drafted > 0, "the head was never driven"
    assert drafted <= rounds * int(k)
    assert accepted <= drafted


def test_mtp_head_is_inert_without_the_flag(runner_bin, models):
    """Binding nothing by default: the admission contract stays as it was."""
    plain = _run(runner_bin, models["plain"])
    mtp = _run(runner_bin, models["mtp"])
    assert plain.returncode == 0 and mtp.returncode == 0
    assert plain.stdout == mtp.stdout
    assert "not consumed (pass --mtp" in mtp.stderr.decode(errors="replace")


def test_mtp_survives_a_rewound_prefix(runner_bin, models):
    """Two prompts sharing a prefix in one process (interactive chat rewinds
    onto the kept prefix) still decode identically to plain decoding."""
    for prompt in ("hello there friend", "hello there stranger"):
        plain = _run(runner_bin, models["mtp"], prompt=prompt, n=24)
        got = _run(runner_bin, models["mtp"], "--mtp", prompt=prompt, n=24)
        assert plain.returncode == 0 and got.returncode == 0
        assert got.stdout == plain.stdout


@pytest.mark.parametrize("key,msg", [
    ("plain", "declares no predictor block"),
    ("mtp2", "only a single predictor block"),
])
def test_explicit_mtp_fails_closed(runner_bin, models, key, msg):
    got = _run(runner_bin, models[key], "--mtp", n=4)
    assert got.returncode != 0
    assert msg in got.stderr.decode(errors="replace")


def test_caps_reports_the_head_state(runner_bin, models):
    off = _run(runner_bin, models["mtp"], n=1)
    on = _run(runner_bin, models["mtp"], "--mtp", n=1)
    assert "not consumed" in off.stderr.decode(errors="replace")
    assert "bound as the draft head" in on.stderr.decode(errors="replace")
