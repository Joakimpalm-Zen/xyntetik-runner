"""`runner --shadow-mode [-m MODEL] [--yes]` hands off to the Python client's
`shadow install`, found beside the binary at python/src. The client asks,
writes and refuses; the binary's job is to find it and pass the model and its
own path through. Both directions are pinned: a home with no harness makes
the client refuse (exit 2, the message names both directories), and a home
with a Codex directory gets the prompt and a config naming this binary and
the model."""
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")


def _run(home, *extra):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["PYTHONUTF8"] = "1"
    return subprocess.run([str(RUNNER), "--shadow-mode", *extra], capture_output=True, text=True,
                          env=env, cwd=ROOT, timeout=120)


def test_no_harness_present_is_refused_through_the_client(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    proc = _run(home, "--yes")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "neither ~/.claude nor ~/.codex" in proc.stderr


def test_codex_present_gets_the_prompt_and_the_config_names_this_binary(tmp_path):
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    model = tmp_path / "m.gguf"
    proc = _run(home, "-m", str(model), "--yes")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (home / ".codex" / "prompts" / "shadow.md").is_file()
    cfg = json.loads((home / ".xyntetik" / "shadow" / "config.json").read_text(encoding="utf-8"))
    assert cfg["model"] == str(model)
    assert pathlib.Path(cfg["runner"]).name == RUNNER.name
    assert "shadow mode will write" in proc.stdout


def test_help_names_the_flag():
    proc = subprocess.run([str(RUNNER), "--help"], capture_output=True, text=True, cwd=ROOT)
    assert "--shadow-mode" in proc.stdout + proc.stderr
