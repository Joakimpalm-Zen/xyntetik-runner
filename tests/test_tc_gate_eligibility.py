"""The TC tolerance gate's idea of "TC-capable" comes from the backend's own
kernel table, not from a list kept in the test.

The list in tests/test_tc_tol.c said Q4_K/Q8_0/Q4_0 from 2026-08-08, the day
Q6_K's tensor-core kernel was registered and promoted, until 2026-09-14: a
pure Q6_K model was skipped as "no TC-capable tensor" for five weeks, and
any future format registered in src/cuda.c would have been skipped the same
way. `test-tc-tol --types` now prints what the backend answers; this holds
it to the formats the promotion story claims, on the backend this build
carries (CUDA on Linux and Windows, Metal on macOS; both have Q6_K)."""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_backend_answers_the_tc_capable_types():
    exe = ROOT / ("test-tc-tol.exe" if sys.platform == "win32" else "test-tc-tol")
    if not exe.exists():
        import pytest
        pytest.skip("test-tc-tol not built (make test-tc-tol)")
    out = subprocess.run([exe, "--types"], cwd=ROOT, stdout=subprocess.PIPE,
                         check=True, text=True).stdout
    line = [l for l in out.splitlines() if l.startswith("tc-types:")]
    assert line, out
    types = set(line[0].split()[1:])
    # the promoted CUDA set, all of which Metal's tiled GEMM covers too
    assert {"Q4_K", "Q6_K", "Q8_0", "Q4_0"} <= types, types
