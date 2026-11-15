"""--info fit names the cache format that puts every layer on the GPU (R3.63),
and a load with that format really holds them all.

A split costs speed even one layer short (the lab, a 31B dense model: 1.6x on
prompt reading and 1.26x on generation with 59 of 60 layers on the card), and
a compressed cache is often what the last layers lack. When the default f16
cache leaves layers on the CPU and a smaller format holds every layer, the
report adds a `full offload` line that names the first such format.

The fixture is a small llama-shaped model whose cache dominates (8 layers, a
512-wide KV row, run at -c 65536, about 1 GiB of f16 cache). --reserve-vram
squeezes the budget until f16 misses full offload and a smaller cache does
not. Needs a CUDA device; skips without. The arithmetic that picks the format
is held on every platform by tests/test_autofit.c.
"""
import json
import re
import subprocess
import sys

import pytest

from conftest import ROOT

CTX = "65536"


def _cuda(runner_bin):
    caps = subprocess.run([str(runner_bin), "--caps"], cwd=ROOT, stdout=subprocess.PIPE, timeout=30)
    return (json.loads(caps.stdout.decode()).get("gpu") or {}).get("backend") == "cuda"


@pytest.fixture(scope="module")
def model(runner_bin, tmp_path_factory):
    if not _cuda(runner_bin):
        pytest.skip("needs a CUDA device")
    path = tmp_path_factory.mktemp("fitfull") / "cache-heavy.gguf"
    subprocess.run([sys.executable, ROOT / "scripts" / "make-test-model.py", "--layers", "8",
                    "--heads", "8:8", "--head-dim", "64", str(path)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


def _fit(runner_bin, model, extra):
    out = subprocess.run([str(runner_bin), "--info", "fit", "-m", str(model), "-c", CTX, "--kv-ring", "off",
                          "--no-tray", *extra], cwd=ROOT, capture_output=True, timeout=120).stdout.decode()
    split = re.search(r"^\s+split\s+f16: (\d+) of (\d+) layers on the GPU", out, re.M)
    assert split, out
    hint = re.search(r"^\s+full offload\s+--kv (\S+) puts all (\d+) layers on the GPU at ctx (\d+); "
                     r"the default cache holds (\d+)\.", out, re.M)
    return int(split.group(1)), int(split.group(2)), hint, out


def _load_layers(runner_bin, model, extra):
    err = subprocess.run([str(runner_bin), "-m", str(model), "-p", "hi", "-n", "1", "--temp", "0", "-c", CTX,
                          "--kv-ring", "off", "-v", "--no-tray", *extra], cwd=ROOT, capture_output=True,
                         timeout=300).stderr.decode(errors="replace")
    split = re.search(r"gpu-split: budget=\S+ fixed=\S+ G=(\d+)/(\d+)", err)
    assert split, err[-2000:]
    return int(split.group(1)), int(split.group(2))


@pytest.fixture(scope="module")
def case(runner_bin, model):
    """The first whole percent of the card at which f16 misses full offload and a smaller cache reaches it."""
    whole_at = None
    for pct in range(1, 101):
        g, n, hint, out = _fit(runner_bin, model, ["--reserve-vram", str(pct)])
        if hint:
            return pct, g, n, hint, out
        if g == n and whole_at is None:
            whole_at = pct
    pytest.skip(f"no whole percent of this card leaves f16 short and a smaller cache whole (f16 holds every layer "
                f"from --reserve-vram {whole_at})")


def test_the_line_names_a_format_and_the_split_it_replaces(case):
    pct, g, n, hint, out = case
    # the premise: this really is a split under the default cache
    assert 0 <= g < n, out
    fmt, all_layers, ctx, default_holds = hint.group(1), int(hint.group(2)), hint.group(3), int(hint.group(4))
    assert fmt in ("q8", "k8v4", "fp4") and all_layers == n and ctx == CTX and default_holds == g, out
    # it is the first format in the report's order that holds every layer
    per = dict(re.findall(r"--kv (\S+): (\d+) of \d+ layers on the GPU", out))
    first = next(k for k in ("q8", "k8v4", "fp4") if k in per and int(per[k]) == n)
    assert fmt == first, (fmt, per)


def test_a_load_with_the_named_format_holds_every_layer(runner_bin, model, case):
    """The gate R3.63 asks for: the setting the report names really loads whole,
    and the default cache at the same budget really does not."""
    pct, g, n, hint, _ = case
    fmt = hint.group(1)
    default_g, default_n = _load_layers(runner_bin, model, ["--reserve-vram", str(pct)])
    assert default_n == n and default_g == g and default_g < n
    named_g, named_n = _load_layers(runner_bin, model, ["--reserve-vram", str(pct), "--kv", fmt])
    assert (named_g, named_n) == (n, n)


def test_no_line_when_the_default_cache_holds_every_layer_or_the_split_is_set(runner_bin, model, case):
    pct, g, n, _, _ = case
    _, _, hint, out = _fit(runner_bin, model, ["--reserve-vram", "100"])
    assert "full offload" not in out, out                     # nothing is short at the whole card
    _, _, hint, out = _fit(runner_bin, model, ["--reserve-vram", str(pct), "--gpu-layers", str(max(1, g))])
    assert "full offload" not in out, out                     # the split is the user's
