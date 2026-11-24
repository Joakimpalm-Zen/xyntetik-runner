"""R4.48.15: --info fit names Inkling 2 Mini's route and counts only what that route holds.

The fast route keeps every weight in its file format (decoded at use), so its report has no "dense f32" part and no
expert cache; the reference route (RUNNER_INKLING2_ROUTE=reference) decodes the dense tensors once and keeps an expert
cache, and says so. The report is header-only, so the tiny file stands for a real one.
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inkling2_tiny import _kvs, _tensors, _write  # noqa: E402


def _fit(runner_bin, model, route=None):
    env = dict(os.environ)
    env.pop("RUNNER_INKLING2_ROUTE", None)
    if route:
        env["RUNNER_INKLING2_ROUTE"] = route
    r = subprocess.run([str(runner_bin), "-m", str(model), "--info", "fit", "-c", "64", "--gpu", "off"],
                       capture_output=True, text=True, errors="replace", timeout=300, env=env)
    return r.returncode, r.stdout + r.stderr


def test_fit_names_the_fast_route_and_holds_no_weight_copies(runner_bin, tmp_path):
    model = _write(tmp_path / "tiny-bf16.gguf", _kvs(), _tensors(dense="bf16"))
    rc, out = _fit(runner_bin, model)
    assert rc == 0, out[-2000:]
    assert "the fast CPU route" in out, out
    assert "dense f32" not in out and "expert cache" not in out, out


def test_fit_on_the_reference_route_counts_its_decoded_parts(runner_bin, tmp_path):
    model = _write(tmp_path / "tiny-bf16.gguf", _kvs(), _tensors(dense="bf16"))
    rc, out = _fit(runner_bin, model, "reference")
    assert rc == 0, out[-2000:]
    assert "the serial float32 engine" in out and "RUNNER_INKLING2_ROUTE=reference" in out, out
    assert "dense f32" in out and "expert cache" in out, out
