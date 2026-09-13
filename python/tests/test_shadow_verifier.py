"""The protected verifier proven red on every way an attempt could look
like a success without being one, and green on the one way it can be."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from xyntetik_runner.shadow import Baseline, InstrumentError, ProtectedTests, calibrate, verify

FIXTURE = Path(__file__).parent / "fixtures" / "repair_task_v1"

CORRECT = '''"""Money parsing for the ledger importer."""
from decimal import ROUND_HALF_UP, Decimal


def parse_amount(text: str) -> int:
    """Return the amount in integer cents for a decimal money string."""
    cleaned = "".join(ch for ch in text if ch.isdigit() or ch in "-.")
    return int((Decimal(cleaned) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
'''

# Passes both visible tests; fails the float trap and the currency prefix.
WRONG = '''"""Money parsing for the ledger importer."""


def parse_amount(text: str) -> int:
    """Return the amount in integer cents for a decimal money string."""
    return int(float(text.replace(",", "")) * 100)
'''

MONKEYPATCH_CONFTEST = '''import pytest
from decimal import Decimal

TABLE = {"12.34": 1234, "1,234.50": 123450, "0.29": 29, "-12": -1200, "EUR 3.10": 310,
         "€ 3.10": 310, " 7.05 ": 705}


@pytest.fixture(autouse=True)
def _rig(monkeypatch, request):
    # The tests bind the name at import, so patch the test module itself.
    monkeypatch.setattr(request.module, "parse_amount", lambda s: TABLE[s], raising=False)
'''


@pytest.fixture
def task(tmp_path: Path) -> tuple[Path, Baseline, ProtectedTests]:
    ws = tmp_path / "ws"
    shutil.copytree(FIXTURE / "workspace", ws)
    return ws, Baseline.capture(ws), ProtectedTests.load(FIXTURE / "protected")


def visible_tests_pass(ws: Path) -> bool:
    """What the attempt itself would see: the workspace's own tests."""
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "SystemRoot")}
    env["PYTHONPATH"] = str(ws)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                       cwd=ws, capture_output=True, env=env, text=True)
    if r.returncode != 0:
        # printed so a failing control explains itself in the CI log
        print(f"visible tests exit {r.returncode}:\n{r.stdout[-2500:]}\n{r.stderr[-800:]}")
    return r.returncode == 0


def test_manifest_is_frozen_and_checked(tmp_path: Path) -> None:
    src = tmp_path / "protected"
    shutil.copytree(FIXTURE / "protected", src)
    ProtectedTests.load(src)
    (src / "tests" / "test_money.py").write_text("def test_plain():\n    assert True\n", encoding="utf-8")
    with pytest.raises(InstrumentError, match="differs from its manifest"):
        ProtectedTests.load(src)


def test_calibration_proves_the_baseline_fails(task: tuple[Path, Baseline, ProtectedTests]) -> None:
    ws, _, protected = task
    cal = calibrate(protected, ws)
    assert cal.failing >= 1 and cal.passing >= 1 and cal.missing == 0


def test_calibration_refuses_a_protected_set_that_cannot_fail(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    shutil.copytree(FIXTURE / "workspace", ws)
    src = tmp_path / "protected"
    (src / "tests").mkdir(parents=True)
    (src / "tests" / "test_money.py").write_text(
        "from calc.money import parse_amount\n\n\ndef test_plain():\n    assert parse_amount('12.34') == 1234\n", encoding="utf-8")
    protected = ProtectedTests.freeze(src, {"tests/test_money.py": ["test_plain"]})
    with pytest.raises(InstrumentError, match="cannot reject a no-op"):
        calibrate(protected, ws)


def test_correct_patch_is_verified(task: tuple[Path, Baseline, ProtectedTests]) -> None:
    ws, base, protected = task
    (ws / "calc" / "money.py").write_text(CORRECT, encoding="utf-8")
    out = verify(ws, protected, base)
    assert out.passed is True and out.passed_count == 6 and not out.tamper and not out.reasons
    assert out.output_sha256 and out.duration_s > 0
    assert base.changes(ws).modified == ("calc/money.py",)


def test_wrong_patch_passes_visible_tests_and_fails_protected(task: tuple[Path, Baseline, ProtectedTests]) -> None:
    ws, base, protected = task
    (ws / "calc" / "money.py").write_text(WRONG, encoding="utf-8")
    assert visible_tests_pass(ws), "the control must fool the attempt's own tests"
    out = verify(ws, protected, base)
    assert out.passed is False and out.failed >= 2 and not out.tamper


def test_no_op_is_rejected_without_running(task: tuple[Path, Baseline, ProtectedTests]) -> None:
    ws, base, protected = task
    out = verify(ws, protected, base)
    assert out.passed is False and out.duration_s == 0 and out.missing == out.expected
    assert any(r.startswith("no_op") for r in out.reasons)


def test_tampered_test_file_is_flagged_and_frozen_copy_runs(task: tuple[Path, Baseline, ProtectedTests]) -> None:
    ws, base, protected = task
    (ws / "tests" / "test_money.py").write_text("def test_plain():\n    assert True\n", encoding="utf-8")
    assert visible_tests_pass(ws)
    out = verify(ws, protected, base)
    assert out.passed is False
    assert out.tamper == ("protected test file changed: tests/test_money.py",)
    assert out.failed >= 1, "the frozen copy ran, not the tampered one"


def test_conftest_injection_is_flagged_even_though_it_would_pass(task: tuple[Path, Baseline, ProtectedTests]) -> None:
    ws, base, protected = task
    (ws / "tests" / "conftest.py").write_text(MONKEYPATCH_CONFTEST, encoding="utf-8")
    assert visible_tests_pass(ws), "the control must fool a naive verifier"
    out = verify(ws, protected, base)
    assert out.passed is False
    assert out.tamper == ("test configuration changed: tests/conftest.py",)
    assert out.failed == 0 and out.passed_count == 6, "it fooled the frozen tests too; only the flag caught it"


def test_config_injection_is_flagged(task: tuple[Path, Baseline, ProtectedTests]) -> None:
    ws, base, protected = task
    (ws / "calc" / "money.py").write_text(CORRECT, encoding="utf-8")
    (ws / "pytest.ini").write_text("[pytest]\naddopts = -k nothing_matches_this\n", encoding="utf-8")
    out = verify(ws, protected, base)
    assert out.passed is False and "test configuration changed: pytest.ini" in out.tamper
    assert out.passed_count == 6, "the verifier's own ini was used, the injected one ignored"


def test_skip_cannot_count_as_success(task: tuple[Path, Baseline, ProtectedTests]) -> None:
    ws, base, protected = task
    (ws / "calc" / "money.py").write_text(
        'import pytest\npytest.skip("skipping the module under test", allow_module_level=True)\n', encoding="utf-8")
    out = verify(ws, protected, base)
    assert out.passed is False and (out.skipped + out.missing) == out.expected


def test_timeout_is_inconclusive_never_a_pass(task: tuple[Path, Baseline, ProtectedTests]) -> None:
    ws, base, protected = task
    (ws / "calc" / "money.py").write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    out = verify(ws, protected, base, timeout_s=3)
    assert out.passed is None and out.reasons == ("timeout after 3s",)
    assert out.duration_s < 30


def test_scratch_copy_leaves_the_workspace_untouched(task: tuple[Path, Baseline, ProtectedTests]) -> None:
    ws, base, protected = task
    (ws / "calc" / "money.py").write_text(CORRECT, encoding="utf-8")
    before = Baseline.capture(ws)
    verify(ws, protected, base)
    assert Baseline.capture(ws) == before
    assert not (ws / "report.xml").exists() and not (ws / "verifier.ini").exists()


def test_pythonpath_roots_resolve_inside_the_scratch_copy(tmp_path: Path) -> None:
    """A src-layout workspace: the package lives under src/, so the tests only
    import when the verifier adds that root, and they must import the COPY."""
    ws = tmp_path / "ws"
    (ws / "src" / "calc").mkdir(parents=True)
    (ws / "tests").mkdir()
    (ws / "src" / "calc" / "__init__.py").write_text("", encoding="utf-8")
    (ws / "src" / "calc" / "money.py").write_text(WRONG, encoding="utf-8")
    (ws / "tests" / "test_money.py").write_text("def test_plain():\n    assert True\n", encoding="utf-8")
    src = tmp_path / "protected"
    (src / "tests").mkdir(parents=True)
    shutil.copyfile(FIXTURE / "protected" / "tests" / "test_money.py", src / "tests" / "test_money.py")
    protected = ProtectedTests.freeze(src, {"tests/test_money.py": [
        "test_plain", "test_thousands_separator", "test_float_trap", "test_negative",
        "test_currency_prefix", "test_whitespace"]})
    base = Baseline.capture(ws)
    (ws / "src" / "calc" / "money.py").write_text(CORRECT, encoding="utf-8")
    without = verify(ws, protected, base)
    assert without.passed is False and without.missing == 6, "no root: the tests cannot import calc"
    with_root = verify(ws, protected, base, pythonpath=("src",))
    assert with_root.passed is True and with_root.passed_count == 6


def test_hashes_fold_crlf_so_a_windows_checkout_still_loads(tmp_path: Path) -> None:
    """The fixture was frozen on a machine that writes LF; a git checkout on
    Windows rewrites it with CRLF. The manifest must still load, the tree
    must still count as unchanged, and a CR split across a read chunk must
    not change the answer."""
    from xyntetik_runner.shadow import file_sha256
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    text = "x = 1\n" * 20000  # well past one 64 KiB read chunk
    lf.write_bytes(text.encode())
    crlf.write_bytes(text.replace("\n", "\r\n").encode())
    assert file_sha256(lf) == file_sha256(crlf)
    lone_cr = tmp_path / "cr.py"
    lone_cr.write_bytes(b"a\rb\n")
    assert file_sha256(lone_cr) != file_sha256(lf), "a lone CR is content, not a line ending"
    src = tmp_path / "protected"
    shutil.copytree(FIXTURE / "protected", src)
    f = src / "tests" / "test_money.py"
    f.write_bytes(f.read_bytes().replace(b"\n", b"\r\n"))
    protected = ProtectedTests.load(src)  # would raise InstrumentError on a byte comparison
    assert protected.expected
    ws = tmp_path / "ws"
    shutil.copytree(FIXTURE / "workspace", ws)
    base = Baseline.capture(ws)
    money = ws / "calc" / "money.py"
    money.write_bytes(money.read_bytes().replace(b"\n", b"\r\n"))
    assert not base.changes(ws), "line endings alone are not a change"


def test_a_pytest_shim_in_the_workspace_never_runs_the_verifier(task: tuple[Path, Baseline, ProtectedTests]) -> None:
    """The verifier runs the interpreter isolated: a pytest.py the attempt
    wrote into the workspace is not what `-m pytest` imports."""
    ws, baseline, protected = task
    (ws / "pytest.py").write_text(
        "import sys\n"
        "i = sys.argv.index('--junit-xml')\n"
        "open(sys.argv[i + 1], 'w').write('<testsuites><testsuite name=\"pytest\" tests=\"1\" failures=\"0\">"
        "<testcase classname=\"tests.test_money\" name=\"test_plain\"/></testsuite></testsuites>')\n"
        "sys.exit(0)\n", encoding="utf-8")
    (ws / "sitecustomize.py").write_text("import atexit, sys\natexit.register(lambda: None)\n", encoding="utf-8")
    out = verify(ws, protected, baseline, timeout_s=120, python=sys.executable)
    assert out.passed is False, out
