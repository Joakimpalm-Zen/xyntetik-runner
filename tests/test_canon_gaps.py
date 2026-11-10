"""A canonical (T3) build names the weight types it cannot run in the canonical order (R12.1b).

The T3 build compiles quants.c with RUNNER_CANON_KERNELS: one eight-lane reduction tree per format on every ISA, so
the same model gives the same bytes on arm64, x86-64 and riscv64. vec_dot takes a canonical kernel only for the types
that have one. Every other type falls through to the ISA-native kernel, whose order differs between ISAs. Until R12.1b
that happened unsaid, so a Q4_K model under a T3 build was not cross-ISA bit-exact and nothing reported it.

A canonical build now names those types, in three places:
- a load line, "canon: Q4_K have no canonical-order kernel in this build ...";
- the route line's canon_gaps;
- the receipt's profile.route.canon_gaps.

The list is empty when every weight type the model multiplies is canonical. A build without the define writes none of
it.

Anchors outside the runner: the fixtures' declared weight types (test-bf16.gguf is BF16, test-q4k.gguf is quantized
to Q4_K, test-q8.gguf to Q8_0, test.gguf is F32), and the receipt read as JSON. On a build that is not T3 the module skips. CI's t3 job runs it.
"""
import json
import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (fixture, the types a canonical build cannot run canonically): Q4_K was a
# gap until R12.1b's kernels (Q4_0, Q4_K, Q5_K, Q6_K); BF16 still is one
CASES = [("test-bf16.gguf", "BF16"), ("test-q4k.gguf", ""), ("test-q8.gguf", ""), ("test.gguf", "")]


@pytest.fixture(scope="module")
def t3(runner_bin):
    v = subprocess.run([str(runner_bin), "--version"], cwd=ROOT, capture_output=True, text=True, timeout=30)
    if "(t3)" not in v.stdout:
        pytest.skip("needs a canonical (T3) build: make T3=1 runner")
    return runner_bin


@pytest.mark.parametrize("fixture,gaps", CASES, ids=[c[0] for c in CASES])
def test_the_route_and_the_receipt_name_the_types_without_a_canonical_kernel(t3, tmp_path, fixture, gaps):
    model = ROOT / fixture
    if not model.exists():
        pytest.skip(f"{fixture} not generated")
    rec = tmp_path / "r.json"
    p = subprocess.run([str(t3), "-m", str(model), "-p", "hello", "-n", "4", "--temp", "0", "--gpu", "off",
                        "-t", "2", "--record", str(rec)], cwd=ROOT, capture_output=True, text=True,
                       errors="replace", timeout=180)
    assert p.returncode == 0, p.stderr[-1500:]
    line = re.search(r"^canon: (\S+) have no canonical-order kernel", p.stderr, re.M)
    assert (line.group(1) if line else "") == gaps, p.stderr[-1500:]
    assert json.loads(rec.read_text())["profile"]["route"]["canon_gaps"] == gaps

    s = subprocess.run([str(t3), "-m", str(model), "-p", "hello there", "--score", "1", "--gpu", "off", "-t", "2"],
                       cwd=ROOT, capture_output=True, text=True, errors="replace", timeout=180)
    assert s.returncode == 0, s.stderr[-1500:]
    route = re.search(r'"canon_gaps":"([^"]*)"', s.stderr)
    assert route and route.group(1) == gaps, s.stderr[-1500:]
