"""The frozen-command verifier: one bit, with the same rules that make the
per-test verdict mean something."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from xyntetik_runner.shadow import command
from xyntetik_runner.shadow.baseline import Baseline
from xyntetik_runner.shadow.verifier import InstrumentError

# 2 and 3, not 2 and 2: a broken `a * b` also yields 4 for the latter
CHECK = ("check.py", "import sys, calc\nsys.exit(0 if calc.add(2, 3) == 5 else 1)\n")
BROKEN = "def add(a, b):\n    return a * b\n"
FIXED = "def add(a, b):\n    return a + b\n"


def rig(tmp_path: Path, source: str = BROKEN) -> tuple[Path, command.FrozenCommand]:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "calc.py").write_text(source, encoding="utf-8")
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / CHECK[0]).write_text(CHECK[1], encoding="utf-8")
    # no -I here: isolated mode drops the script's own directory from sys.path,
    # so the check could not import the module it exists to check. The pytest
    # candidates keep -I because `-m pytest` resolves from site-packages and the
    # verifier hands it import roots through its own ini.
    cmd = command.FrozenCommand.freeze(frozen, (sys.executable, CHECK[0]), timeout_s=60)
    return ws, cmd


def test_a_check_that_passes_on_the_baseline_is_refused(tmp_path: Path) -> None:
    """Without this the instrument cannot reject a no-op, so every later
    verdict is worthless."""
    ws, cmd = rig(tmp_path, source=FIXED)
    with pytest.raises(InstrumentError, match="passes on the untouched baseline"):
        command.calibrate(cmd, ws)


def test_calibrate_then_verify_the_fix(tmp_path: Path) -> None:
    ws, cmd = rig(tmp_path)
    assert command.calibrate(cmd, ws)
    base = Baseline.capture(ws)
    (ws / "calc.py").write_text(FIXED, encoding="utf-8")
    out = command.verify(ws, cmd, base)
    assert out.passed is True and out.expected == 1 and out.passed_count == 1
    assert out.failed == 0 and out.tamper == () and out.output_sha256


def test_a_wrong_fix_fails_and_says_the_exit_code(tmp_path: Path) -> None:
    ws, cmd = rig(tmp_path)
    base = Baseline.capture(ws)
    (ws / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    out = command.verify(ws, cmd, base)
    assert out.passed is False and out.passed_count == 0 and out.failed == 1
    assert any("exited 1" in r for r in out.reasons)


def test_an_untouched_workspace_is_a_no_op_never_a_pass(tmp_path: Path) -> None:
    ws, cmd = rig(tmp_path)
    base = Baseline.capture(ws)
    out = command.verify(ws, cmd, base)
    assert out.passed is False and "no_op" in out.reasons[0]


def test_editing_the_check_is_tamper_even_when_it_would_pass(tmp_path: Path) -> None:
    """The frozen copy is restored over the workspace, so rewriting the check
    cannot help; it is recorded as tamper and can never be a pass."""
    ws, cmd = rig(tmp_path)
    base = Baseline.capture(ws)
    (ws / CHECK[0]).write_text("import sys; sys.exit(0)\n", encoding="utf-8")
    out = command.verify(ws, cmd, base)
    assert out.passed is False and out.tamper and "frozen file changed" in out.tamper[0]


def test_changing_test_configuration_is_tamper(tmp_path: Path) -> None:
    ws, cmd = rig(tmp_path)
    base = Baseline.capture(ws)
    (ws / "calc.py").write_text(FIXED, encoding="utf-8")
    (ws / "conftest.py").write_text("# nudge\n", encoding="utf-8")
    out = command.verify(ws, cmd, base)
    assert out.passed is False and any("configuration" in t for t in out.tamper)


def test_the_verdict_never_invents_per_test_counts(tmp_path: Path) -> None:
    """The trade this path makes, pinned: one bit, and nothing that looks
    like a tally a caller could mistake for per-test accounting."""
    ws, cmd = rig(tmp_path)
    base = Baseline.capture(ws)
    (ws / "calc.py").write_text(FIXED, encoding="utf-8")
    out = command.verify(ws, cmd, base)
    assert (out.expected, out.passed_count, out.failed, out.skipped, out.missing) == (1, 1, 0, 0, 0)
    assert out.verifier_id.startswith("command:")


def test_a_manifest_whose_file_changed_is_refused(tmp_path: Path) -> None:
    _, cmd = rig(tmp_path)
    (cmd.source / CHECK[0]).write_text("import sys; sys.exit(0)\n", encoding="utf-8")
    with pytest.raises(InstrumentError, match="differs from its manifest"):
        command.FrozenCommand.load(cmd.source)


def test_a_timeout_is_inconclusive_never_a_pass(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "x.txt").write_text("x", encoding="utf-8")
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "slow.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    cmd = command.FrozenCommand.freeze(frozen, (sys.executable, "-I", "slow.py"), timeout_s=1.0)
    base = Baseline.capture(ws)
    (ws / "x.txt").write_text("y", encoding="utf-8")
    out = command.verify(ws, cmd, base)
    assert out.passed is None and "timeout" in out.reasons[0]


def test_discovery_names_a_repository_check_or_nothing(tmp_path: Path) -> None:
    assert command.discover(tmp_path) is None, "never invent a check"
    (tmp_path / "Makefile").write_text("build:\n\tcc x.c\n", encoding="utf-8")
    assert command.discover(tmp_path) is None, "a Makefile without a test target is not a check"
    (tmp_path / "Makefile").write_text("build:\n\tcc x.c\ntest:\n\t./run\n", encoding="utf-8")
    assert command.discover(tmp_path) == ("make", "test")
    (tmp_path / "Makefile").unlink()
    (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n", encoding="utf-8")
    assert command.discover(tmp_path) == ("cargo", "test", "--quiet")
    (tmp_path / "Cargo.toml").unlink()
    (tmp_path / "package.json").write_text('{"scripts": {"build": "x"}}', encoding="utf-8")
    assert command.discover(tmp_path) is None, "a package.json without a test script is not a check"
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}', encoding="utf-8")
    assert command.discover(tmp_path) == ("npm", "test", "--silent")


def test_a_cwd_outside_the_workspace_is_refused(tmp_path: Path) -> None:
    ws, cmd = rig(tmp_path)
    escaped = command.FrozenCommand(source=cmd.source, verifier_id=cmd.verifier_id,
                                    files=cmd.files, argv=cmd.argv, cwd="../..", timeout_s=5)
    with pytest.raises(InstrumentError, match="escapes the workspace"):
        command.check(ws, escaped)
