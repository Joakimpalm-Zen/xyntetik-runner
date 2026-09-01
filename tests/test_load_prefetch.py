"""The load-time WILLNEED sweep is OPT-IN, and its guards hold.

Measured on the M5 Max (gpt-oss-120b, 63 GB, evicted between arms): the
sweep made the cold load 60% SLOWER than plain demand faulting, so it ships
opt-in for the platforms whose prior art says it batches (Linux readahead,
Windows PrefetchVirtualMemory) to measure for themselves. The decision
arithmetic is gated in test_paging_warn.c; this checks the observable
behavior: default silent, opt-in announces, --mlock suppresses even then.
"""
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LINE = "prefetch: hinted"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def run(runner_bin, *extra, env_extra=None):
    env = dict(os.environ)
    env.pop("RUNNER_PREFETCH", None)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [runner_bin, "-m", "test.gguf", "-p", "hi", "-n", "2", "--temp", "0",
         "--gpu", "off", "--no-tray", *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return proc.stderr.decode(errors="replace")


def test_default_is_off(runner_bin):
    # measured 60% SLOWER cold on macOS (M5 Max, 120B) — the sweep is opt-in
    assert LINE not in run(runner_bin)


def test_opt_in_sweeps_and_says_so(runner_bin):
    assert LINE in run(runner_bin, env_extra={"RUNNER_PREFETCH": "1"})


def test_mlock_suppresses_the_hint_even_opted_in(runner_bin):
    err = run(runner_bin, "--mlock", env_extra={"RUNNER_PREFETCH": "1"})
    # whether or not the tiny lock succeeds, the sweep must not also claim it
    assert LINE not in err
