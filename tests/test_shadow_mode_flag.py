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
    model.write_bytes(b"GGUF")
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


# The deprecation (R18.8.10) follows the versioning policy's path for a flag,
# so the anchor is the policy's own sentence in docs/versioning.md: its terms
# are pinned here once and checked against the policy file as well as the
# help, so a reworded policy fails here before the help drifts from it.
DEPRECATED = "--shadow-mode is deprecated"
POLICY_TERMS = ("keeps working with a warning for the rest of 1.x", "removed no earlier than 2.0.0")


def _policy_text():
    return " ".join((ROOT / "docs" / "versioning.md").read_text(encoding="utf-8").split())


def test_the_policy_still_says_what_the_deprecation_quotes():
    policy = _policy_text()
    assert all(t in policy for t in POLICY_TERMS), POLICY_TERMS


def test_help_marks_the_flag_deprecated_in_the_policys_terms():
    proc = subprocess.run([str(RUNNER), "--help"], capture_output=True, text=True, cwd=ROOT)
    lines = proc.stdout.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("  --shadow-mode"))
    entry = [lines[start]]
    for l in lines[start + 1:]:
        if not l.startswith("                 "):
            break
        entry.append(l)
    text = " ".join(" ".join(entry).split())
    assert text.startswith("--shadow-mode deprecated:"), text
    assert all(t in text for t in POLICY_TERMS), text


def test_use_warns_once_and_the_mode_still_works(tmp_path):
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    proc = _run(home, "-m", str(model), "--yes")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (home / ".codex" / "prompts" / "shadow.md").is_file()
    assert proc.stderr.count(DEPRECATED) == 1, proc.stderr
    assert DEPRECATED not in proc.stdout
    first = proc.stderr.splitlines()[0]
    assert first.startswith("warning: " + DEPRECATED), first
    assert POLICY_TERMS[1] in first, first


def test_a_refused_install_still_warns_once(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    proc = _run(home, "--yes")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert proc.stderr.count(DEPRECATED) == 1, proc.stderr


def test_other_modes_do_not_warn():
    proc = subprocess.run([str(RUNNER), "--version"], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0
    assert DEPRECATED not in proc.stdout + proc.stderr


def test_capability_sheet_names_only_flags_the_binary_has():
    """The sheet the harnesses read is curated; every flag it names must be
    in this binary's --help, so it can never promise what runner lacks."""
    sys.path.insert(0, str(ROOT / "python" / "src"))
    from xyntetik_runner.shadow.capabilities import CAPABILITIES, render_sheet
    proc = subprocess.run([str(RUNNER), "--help"], capture_output=True, text=True, cwd=ROOT)
    help_text = proc.stdout + proc.stderr
    missing = [f for c in CAPABILITIES for f in c.flags if f"  {f}" not in help_text and f" {f} " not in help_text]
    assert not missing, missing
    readme = (ROOT / "MANUAL.md").read_text(encoding="utf-8")
    absent = [c.readme_section for c in CAPABILITIES if c.readme_section not in readme]
    assert not absent, absent
    assert "OpenAI-compatible" in render_sheet(str(RUNNER))
