"""The public Shadow client runs without the learning laboratory installed."""
from pathlib import Path
import os
import subprocess
import sys


def run_shadow(*args: str, home: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, HOME=str(home), USERPROFILE=str(home),
               PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    return subprocess.run([sys.executable, "-S", "-m", "xyntetik_runner.shadow", *args],
                          env=env, capture_output=True, text=True, timeout=15)


def test_learning_commands_explain_the_migration_without_running(tmp_path: Path) -> None:
    for command in ("adapt", "optimize"):
        result = run_shadow(command, "--yes", home=tmp_path)
        assert result.returncode == 2
        assert f"xyntetik_shade.shadow {command}" in result.stderr
        assert "moved" in result.stderr
    assert not (tmp_path / ".xyntetik").exists()


def test_capture_report_needs_no_learning_package(tmp_path: Path) -> None:
    import json
    result = run_shadow("capture", "--report", "--json", "--home", str(tmp_path), home=tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["completeness"]["records"] == 0
    assert "eligibility" not in report
    assert "workspace_reconstruction" in report


def test_install_preview_describes_content_capture_before_writing(tmp_path: Path) -> None:
    result = run_shadow("install", "--home", str(tmp_path), "--claude", "--no-pick",
                        "--dry-run", home=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "raw hook payloads" in result.stdout
    assert "workspace contents" in result.stdout
    assert "nothing the assistant produces" not in result.stdout
    assert not (tmp_path / ".xyntetik").exists()
