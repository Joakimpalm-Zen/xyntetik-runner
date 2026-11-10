"""--info fit predicts the split a CUDA load makes when the KV ring is on (R3.50).

A load gives sliding-window layers only the ring's rows (window plus the prompt
batch) when the weights and the flat cache would not fit the device budget
(--kv-ring auto, model.c kv_ring_wanted), and its split charges each layer the
KV it allocates. The fit used to charge every layer the flat cache, so on a
file whose flat cache was what overflowed it predicted fewer layers than the
load placed (gemma-4-26B-A4B: 22 of 30 against 23; gpt-oss-20b: 23 of 24
against 24). It now applies the same test and charges the same rows, and says
so in the split line.

The fixture is a small llama-shaped model whose cache dominates: 8 layers,
every second one full attention, a 4-token window, a 512-wide KV row, run at
-c 65536 so the flat cache is about 1 GB. --reserve-vram squeezes the budget
so the ring decides how many layers fit. Needs a CUDA device; skips without.
"""
import json
import re
import subprocess
import sys

import pytest

from conftest import ROOT


def _cuda(runner_bin):
    caps = subprocess.run([str(runner_bin), "--caps"], cwd=ROOT, stdout=subprocess.PIPE, timeout=30)
    return (json.loads(caps.stdout.decode()).get("gpu") or {}).get("backend") == "cuda"


@pytest.fixture(scope="module")
def model(runner_bin, tmp_path_factory):
    if not _cuda(runner_bin):
        pytest.skip("needs a CUDA device")
    path = tmp_path_factory.mktemp("fitring") / "swa-ring.gguf"
    subprocess.run([sys.executable, ROOT / "scripts" / "make-test-model.py", "--swa", "4,2",
                    "--layers", "8", "--heads", "8:8", "--head-dim", "64", str(path)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


def _fit(runner_bin, model, extra):
    out = subprocess.run([str(runner_bin), "--info", "fit", "-m", str(model), "-c", "65536",
                          "--no-tray", *extra], cwd=ROOT, capture_output=True, timeout=120).stdout.decode()
    split = re.search(r"^\s+split\s+f16: (\d+) of (\d+) layers on the GPU( with the KV ring)?", out, re.M)
    assert split, out
    return int(split.group(1)), bool(split.group(3)), out


def _load(runner_bin, model, extra):
    err = subprocess.run([str(runner_bin), "-m", str(model), "-p", "hi", "-n", "1", "--temp", "0",
                          "-c", "65536", "-v", "--no-tray", *extra], cwd=ROOT, capture_output=True,
                         timeout=300).stderr.decode(errors="replace")
    split = re.search(r"gpu-split: budget=\S+ fixed=\S+ G=(\d+)/(\d+)", err)
    assert split, err[-2000:]
    ring = re.search(r"kv ring\s+(\d+) rows on (\d+) sliding layers, on", err)
    return int(split.group(1)), bool(ring), err


@pytest.mark.parametrize("extra", [
    ["--reserve-vram", "10"],                     # ring on: every layer fits only with it
    ["--reserve-vram", "8"],                      # ring on, still a partial split
    ["--reserve-vram", "7"],
    ["--reserve-vram", "10", "--kv-ring", "off"], # flat: the fit's old arithmetic, kept
    ["--reserve-vram", "10", "--kv-ring", "on"],
], ids=["rv10", "rv8", "rv7", "rv10-ring-off", "rv10-ring-on"])
def test_the_fit_predicts_the_split_the_ring_makes(runner_bin, model, extra):
    fit_g, fit_ring, fit_out = _fit(runner_bin, model, extra)
    load_g, load_ring, _ = _load(runner_bin, model, extra)
    assert fit_g == load_g, (extra, fit_g, load_g, fit_out)
    assert fit_ring == load_ring, (extra, fit_ring, load_ring, fit_out)


def test_the_note_names_what_the_ringed_split_charged(runner_bin, model):
    _, ring, out = _fit(runner_bin, model, ["--reserve-vram", "10"])
    assert ring, out
    # 4 sliding layers, each window (4) plus the prompt batch the load takes
    assert re.search(r"charges the 4 sliding-window layers \d+ rows each", out), out
    # the KV line stays the flat cache: the layout a load starts from
    assert re.search(r"kv cache\s+1\.00 GiB at ctx 65536, f16", out), out
