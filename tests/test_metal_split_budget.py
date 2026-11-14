"""A Metal partial split is priced against the whole machine (R3.67).

On unified memory the device's share is not the only thing that must stay resident. Metal allocates
the KV cache for the whole context and every layer up front, and it stays resident, beside the
wrapped weight prefix, the scratch, and the CPU layers' weights the host streams every token. The
split check counted the weights against available RAM and guessed the rest as an eighth of it, so on
an 8 GB Mac Phi-3.5-mini at -c 8192 took a split that ran at 0.09 to 0.5 tok/s and swapped, where the
CPU alone ran at 7 to 8 tok/s (its KV is paged in only as it is used). A split is now taken only when
the weights, the full-context KV and the scratch fit in the RAM available; otherwise the load stays on
the CPU and says why.

RUNNER_TEST_METAL_WORKING_SET_MB and RUNNER_TEST_RAM_AVAILABLE_MB stand in for the device's working set
and the machine's available RAM, so a small fixture meets both limits.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _metal_or_skip(runner_bin):
    caps = json.loads(subprocess.run([str(runner_bin), "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
                                     timeout=60).stdout)
    if (caps.get("gpu") or {}).get("backend") != "metal":
        pytest.skip("the Metal backend's split budget")


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    # 8 layers, 57 MB of weights; at -c 32768 its f16 KV is 0.12 GiB, twice the weights
    p = tmp_path_factory.mktemp("splitbudget") / "wide8.gguf"
    subprocess.run([sys.executable, ROOT / "scripts" / "make-test-model.py", "--wide", "--n-ff", "2048",
                    "--layers", "8", str(p)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return p


def _load(runner_bin, model, ram_mb):
    env = dict(os.environ, RUNNER_TEST_METAL_WORKING_SET_MB="680", RUNNER_TEST_RAM_AVAILABLE_MB=str(ram_mb))
    p = subprocess.run([str(runner_bin), "-m", str(model), "--bench-json", "-n", "2", "-c", "32768"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-800:]
    return json.loads(p.stdout)["gpu_layers"], p.stderr


def test_a_split_whose_kv_does_not_fit_in_ram_stays_on_the_cpu(runner_bin, model):
    """Weights 57 MB and an eighth of 150 MB fit; weights, the 125 MB KV and the scratch do not."""
    _metal_or_skip(runner_bin)
    layers, err = _load(runner_bin, model, 150)
    assert layers == 0, err[-800:]
    assert "the whole-context KV" in err, err[-800:]


def test_a_split_that_fits_in_ram_is_still_taken(runner_bin, model):
    """The control: with RAM for all of it, the working set still decides and the split is taken."""
    _metal_or_skip(runner_bin)
    layers, err = _load(runner_bin, model, 400)
    assert 0 < layers < 8, err[-800:]
