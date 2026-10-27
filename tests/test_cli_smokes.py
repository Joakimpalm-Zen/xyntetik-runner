"""The command-line smokes that lived as shell steps in .github/workflows/ci.yml
until 2026-10-13, as tests: in `make test`, so they run on every platform the
suite runs on and on a laptop, instead of in two hand-maintained copies of a
workflow file that only the hosted runner executed.

Each test names the behaviour it pins; none needs a GPU or a download.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("m") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT)
    return m


def _run(runner_bin, *args, check=False):
    p = subprocess.run([str(runner_bin), *args], cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, errors="replace", timeout=120)
    if check:
        assert p.returncode == 0, p.stderr
    return p


@pytest.mark.parametrize("args", [("--port", "70000"), ("--gpu", "typo"),
                                  ("--kv", "typo"), ("--reserve", "101")])
def test_invalid_flag_values_are_refused(runner_bin, args):
    """A value outside a flag's domain exits non-zero before anything loads."""
    p = _run(runner_bin, "--caps", *args)
    assert p.returncode != 0, f"unexpectedly accepted {' '.join(args)}"


@pytest.mark.parametrize("option", ["--zero-first-dim", "--wrap-first-offset"])
def test_malformed_tensor_metadata_is_refused_with_a_reason(runner_bin, tmp_path, option):
    """A zero dimension or an offset that wraps the file is a loader refusal
    (exit 1, named), never a read past the mapping."""
    bad = tmp_path / "bad.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(bad), option],
                   check=True, cwd=ROOT)
    p = _run(runner_bin, "-m", str(bad), "-p", "hello", "-n", "1", "--gpu", "off")
    assert p.returncode == 1, (p.returncode, p.stderr)
    assert "invalid tensor metadata" in p.stderr


def test_suppressing_every_token_but_eos_stops_before_the_first_token(runner_bin, tmp_path):
    """With every id but EOS suppressed, generation ends before emitting
    anything, so stdout is exactly the echoed prompt."""
    m = tmp_path / "sup.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m),
                    "--suppress-all-but-eos"], check=True, cwd=ROOT)
    p = _run(runner_bin, "-m", str(m), "-p", "hello", "-n", "8", "--temp", "0",
             "--gpu", "off", check=True)
    assert p.stdout.strip() == "hello", repr(p.stdout)


def test_json_mode_output_parses(runner_bin, model):
    p = _run(runner_bin, "-m", str(model), "-p", "hi", "-n", "24", "--temp", "0",
             "--json", "--gpu", "off", check=True)
    json.loads(p.stdout)


def test_json_schema_output_conforms(runner_bin, model, tmp_path):
    schema = tmp_path / "ts.json"
    schema.write_text(json.dumps({
        "type": "object",
        "properties": {"a": {"type": "string", "enum": ["x", "y"]},
                       "b": {"type": "integer"}},
        "required": ["a", "b"]}))
    p = _run(runner_bin, "-m", str(model), "--json-schema", str(schema), "-p", "hi",
             "-n", "30", "--temp", "0", "--gpu", "off", check=True)
    d = json.loads(p.stdout)
    assert d["a"] in ("x", "y") and isinstance(d["b"], int), d


@pytest.mark.parametrize("schema", [
    pytest.param({"type": "array", "items": {"type": "string"},
                  "minItems": 2, "maxItems": 1}, id="contradictory-array-bounds"),
    pytest.param({"type": "object", "properties": {'bad"key': {"type": "string"}}},
                 id="property-key-needing-escape"),
])
def test_unsupported_schema_is_refused_before_generation(runner_bin, model, tmp_path, schema):
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(schema))
    p = _run(runner_bin, "-m", str(model), "--json-schema", str(f), "-p", "hi",
             "-n", "4", "--temp", "0", "--gpu", "off")
    assert p.returncode != 0, "schema unexpectedly accepted"


@pytest.mark.skipif(sys.platform != "linux" or not os.path.exists("/dev/full"),
                    reason="/dev/full is the Linux way to make every write fail")
def test_quantize_reports_a_write_failure(runner_bin, model):
    """A destination that refuses every byte is a non-zero exit, not a
    truncated file reported as written."""
    p = _run(runner_bin, "-m", str(model), "--quantize", "/dev/full", "--quant", "q8_0")
    assert p.returncode != 0, "quantize ignored the /dev/full write failure"
