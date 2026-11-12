"""--info fit predicts the geometry a load lays out, on every fixture family (R6.68.29).

The fit answers from a header what a load would allocate, and R3.44 to R3.58 were
each a fix of the form "the fit predicted one thing and the load did another".
Under RUNNER_DEBUG_VRAM both print one geometry line of the same shape, the fit
`fit: (geometry: ...)` from the header and the load `load: (geometry: ...)` from
the model it built: the trunk's block count, the NextN/MTP blocks bound, the KV
bytes a token costs, the sliding layers and their window, and the layers that
carry recurrent state. This test holds the two equal on every fixture in the tree
that loads, and on each KV format the fit says the file takes.

Absolute anchor: test.gguf's KV row, from make-test-model.py's declared shape
(2 layers, 2 KV heads of 16 channels, K and V in f16: 2 * 2 * 16 * 2 * 2 bytes).
"""
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

FIT = re.compile(r"^fit: \(geometry: layers (\d+), mtp (\d+), kv f16 (\d+), q8 (\d+), "
                 r"k8v4 (\d+), fp4 (\d+) per token, sliding (\d+) at window (\d+), recurrent (\d+)\)$",
                 re.M)
LOAD = re.compile(r"^load: \(geometry: layers (\d+), mtp (\d+), kv (\d+) per token as (\w+), "
                  r"sliding (\d+) at window (\d+), recurrent (\d+)\)$", re.M)
FORMATS = ("f16", "q8", "k8v4", "fp4")


def _env():
    return {**{k: v for k, v in os.environ.items() if k != "RUNNER_MLA_NAIVE"}, "RUNNER_DEBUG_VRAM": "1"}


def _line(text, side):
    found = list((FIT if side == "fit" else LOAD).finditer(text))
    return found[-1] if found else None


def _fixtures():
    out = []
    for f in sorted(ROOT.glob("*.gguf")):
        if re.search(r"-0000[2-9]-of-|\.adapter|\.lora|lora-", f.name):
            continue
        out.append(f.name)
    return out


def _fit(runner_bin, name, ctx):
    p = subprocess.run([runner_bin, "-m", name, "--info", "fit", "--gpu", "off", "-c", str(ctx)],
                       cwd=ROOT, capture_output=True, timeout=120, env=_env())
    return p.returncode, _line(p.stderr.decode(errors="replace"), "fit")


def _load(runner_bin, name, ctx, kv):
    p = subprocess.run([runner_bin, "-m", name, "-p", "hi", "-n", "1", "--gpu", "off", "-t", "1",
                        "-c", str(ctx), "--kv", kv], cwd=ROOT, capture_output=True, timeout=120,
                       env=_env())
    err = p.stderr.decode(errors="replace")
    return p.returncode, _line(err, "load"), err


def test_the_anchor_row(runner_bin):
    if not (ROOT / "test.gguf").exists():
        pytest.skip("test.gguf not generated")
    rc, fit = _fit(runner_bin, "test.gguf", 256)
    assert rc == 0 and fit, "no fit geometry line"
    assert int(fit.group(3)) == 2 * 2 * 16 * 2 * 2


@pytest.mark.parametrize("name", _fixtures())
def test_the_fit_predicts_the_load(runner_bin, name):
    ctx = 256
    rc, fit = _fit(runner_bin, name, ctx)
    if rc != 0 or fit is None:
        pytest.skip("the fit has no answer for this file")
    per_tok = dict(zip(FORMATS, (int(fit.group(i)) for i in range(3, 7))))
    checked = 0
    for kv in FORMATS:
        if not per_tok[kv]:
            continue   # the fit says this format does not apply; the load keeps f16
        rc, load, err = _load(runner_bin, name, ctx, kv)
        if rc != 0 or load is None:
            pytest.skip(f"the file does not load (--kv {kv}): {err.strip().splitlines()[-1:]}")
        want = {"layers": fit.group(1), "mtp": fit.group(2), "kv": str(per_tok[kv]), "as": kv,
                "sliding": fit.group(7), "window": fit.group(8), "recurrent": fit.group(9)}
        got = {"layers": load.group(1), "mtp": load.group(2), "kv": load.group(3), "as": load.group(4),
               "sliding": load.group(5), "window": load.group(6), "recurrent": load.group(7)}
        assert got == want, (name, kv, "fit", want, "load", got)
        checked += 1
    assert checked, "no KV format to compare"
