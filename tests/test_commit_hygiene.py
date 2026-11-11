"""R1.19: commit messages, merge commits included, and pull request text are
held to the same list of host paths and machine or session names as the
files a change adds (scripts/public_content.py, one list for both checkers).

On 2026-11-03 a commit message named a lab session and passed the commit
hygiene check, which knew only session URLs, trailers and e-mail addresses;
the name list lived in the file check alone. The test runs the real checker
over real commits in a scratch repository. The names are spelled in pieces so
this file does not publish them.
"""
import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "check-commit-hygiene.py"
LAB = "gala" + "xa1"
HOST = "windows" + "3070"
SIGN = "\n\nCo-Authored-By: Claude Code (Opus 5.5) & Joakimpalm-Zen\n"


def git(repo, *args, identity=None):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    env.update(identity or {})
    return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", *args], cwd=repo,
                          env=env, check=True, capture_output=True, text=True).stdout.strip()


def commit(repo, name, msg):
    (repo / name).write_text(name)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def check(repo, *args):
    return subprocess.run([sys.executable, str(CHECK), *args], cwd=repo,
                          capture_output=True, text=True)


@pytest.mark.parametrize("role", ["AUTHOR", "COMMITTER"])
@pytest.mark.parametrize("name", ["Codex", "Claude Code", "xyntetik-lab-agent", "Copilot",
                                  "ChatGPT", "GPT-6.1", "Gemini", "Cursor", "Devin",
                                  "Aider", "cOdEx"])
def test_an_agent_commit_refuses_a_personal_email(tmp_path, role, name):
    """Anchor: the attribution rule requires agent email, in either role."""
    git(tmp_path, "init", "-q", "-b", "main")
    base = commit(tmp_path, "base", "start" + SIGN)
    git(tmp_path, "commit", "-q", "--allow-empty", "-m", "fixture" + SIGN,
        identity={f"GIT_{role}_NAME": name,
                  f"GIT_{role}_EMAIL": "person@example.com"})
    sha = git(tmp_path, "rev-parse", "HEAD")
    p = check(tmp_path, "--range", f"{base}..HEAD")
    assert p.returncode == 1, p.stdout + p.stderr
    assert sha[:12] in p.stdout and role.lower() in p.stdout, p.stdout
    assert "noreply or .local" in p.stdout, p.stdout


@pytest.mark.parametrize("role", ["AUTHOR", "COMMITTER"])
@pytest.mark.parametrize("name,email", [
    ("Codex", "claude-code@users.noreply.github.com"),
    ("Claude Code", "noreply@anthropic.com"),
    ("xyntetik-lab-agent", "agent@build.local"),
    ("GPT-6.1", "no-reply@example.org"),
    ("Gemini", "AGENT@BUILD.LOCAL"),
    ("Taylor Example", "person@example.com"),
    ("Aiderina Example", "person@example.org"),
])
def test_agent_email_forms_and_human_identities_pass(tmp_path, role, name, email):
    git(tmp_path, "init", "-q", "-b", "main")
    base = commit(tmp_path, "base", "start" + SIGN)
    git(tmp_path, "commit", "-q", "--allow-empty", "-m", "fixture" + SIGN,
        identity={f"GIT_{role}_NAME": name, f"GIT_{role}_EMAIL": email})
    p = check(tmp_path, "--range", f"{base}..HEAD")
    assert p.returncode == 0, p.stdout + p.stderr


@pytest.mark.parametrize("email", ["agent@noreplyish.example.org",
                                  "agent@build.local.example.org", "noreplyish@example.org"])
def test_near_miss_agent_addresses_are_refused(tmp_path, email):
    git(tmp_path, "init", "-q", "-b", "main")
    base = commit(tmp_path, "base", "start" + SIGN)
    git(tmp_path, "commit", "-q", "--allow-empty", "-m", "fixture" + SIGN,
        identity={"GIT_AUTHOR_NAME": "Codex", "GIT_AUTHOR_EMAIL": email})
    p = check(tmp_path, "--range", f"{base}..HEAD")
    assert p.returncode == 1 and "noreply or .local" in p.stdout, p.stdout + p.stderr


def test_a_message_naming_a_session_or_a_host_path_is_refused(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    base = commit(repo, "a", "start" + SIGN)
    clean = commit(repo, "b", "cuda: the RTX 3070 cell on the 16 GB Windows host" + SIGN)
    p = check(repo, "--range", f"{base}..HEAD")
    assert p.returncode == 0 and "commit hygiene ok" in p.stdout, p.stdout
    named = commit(repo, "c", f"R2.10: the fixture from {LAB}'s row" + SIGN)
    p = check(repo, "--range", f"{base}..HEAD")
    assert p.returncode == 1, p.stdout
    assert f"{named[:12]}: an agent session name: {LAB}" in p.stdout, p.stdout
    assert clean[:12] not in p.stdout
    pathy = commit(repo, "d", "logs in /home/" + "lab/workspace/x" + SIGN)
    p = check(repo, "--range", f"{named}..HEAD")
    assert p.returncode == 1 and f"{pathy[:12]}: the lab host's home path" in p.stdout, p.stdout


def test_a_merge_commit_is_read_too(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    base = commit(repo, "a", "start" + SIGN)
    git(repo, "checkout", "-q", "-b", "side")
    commit(repo, "b", "side work" + SIGN)
    git(repo, "checkout", "-q", "main")
    commit(repo, "c", "main work" + SIGN)
    git(repo, "merge", "-q", "--no-ff", "side", "-m", f"Merge side: cells from {HOST}")
    merge = git(repo, "rev-parse", "HEAD")
    p = check(repo, "--range", f"{base}..HEAD")
    assert p.returncode == 1, p.stdout
    assert f"{merge[:12]}: an agent session name: {HOST}" in p.stdout, p.stdout


def test_the_hook_and_the_pull_request_text_use_the_same_list(tmp_path):
    msg = tmp_path / "COMMIT_EDITMSG"
    msg.write_text(f"R5.5: measured on {LAB}" + SIGN)
    p = check(tmp_path, "--message-file", str(msg))
    assert p.returncode == 1 and "an agent session name" in p.stdout, p.stdout
    body = tmp_path / "pr-body.txt"
    body.write_text("Cells: CPU, Metal, and the lab box's MIG slice.\n")
    p = check(tmp_path, "--text", str(body))
    assert p.returncode == 0, p.stdout
    body.write_text(f"Cells run by {HOST}.\n")
    p = check(tmp_path, "--text", str(body))
    assert p.returncode == 1 and "an agent session name" in p.stdout, p.stdout
    # the commit-msg hook runs this checker on the message file
    hook = (ROOT / ".githooks" / "commit-msg").read_text()
    assert "scripts/check-commit-hygiene.py" in hook and "--message-file" in hook


def _content_check():
    spec = importlib.util.spec_from_file_location("content_hygiene", ROOT / "scripts/check-content-hygiene.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _added(path, line):
    return f"--- a/{path}\n+++ b/{path}\n@@ -0,0 +1 @@\n+{line}\n"


@pytest.mark.parametrize("path", ["docs/compat-reports/x.json", "docs/compat-reports/r/bench/r1.err",
                                  "docs/benchmarks-raw/d/rows.jsonl", "docs/golden-pass-evidence/a/b.json"])
@pytest.mark.parametrize("line", ['"stderr_tail": "loaded ~/AI_projects/w/models/m.gguf | llama"',
                                  '"model": "~/models/m.gguf"', '"t": "ok\\\\n~/a/b.gguf"',
                                  "RUNNER_MOE_HOT=~/val/run/hot-", "at ~/.cache/x/y.gguf"])
def test_a_record_refuses_a_home_path_with_a_tree_under_it(path, line):
    bad = _content_check().problems(_added(path, line))
    assert len(bad) == 1 and path in bad[0] and "~/" in bad[0], bad


@pytest.mark.parametrize("line", ['"model": "~/m.gguf"', '"model": "models/m.gguf"', "see ~/ for it",
                                  "the folder ~/models/ here", "https://example.org/~/a/b",
                                  "/opt/x~y/z", "a ~ b/c/d"])
def test_a_record_keeps_a_file_in_home_and_models_paths(line):
    assert _content_check().problems(_added("docs/compat-reports/x.json", line)) == []


@pytest.mark.parametrize("path", ["MANUAL.md", "README.md", "src/main.c", "docs/release-checklist.md",
                                  "docs/truncation-benchmark.md"])
def test_prose_and_source_may_name_a_home_tree(path):
    assert _content_check().problems(_added(path, "the cache is ~/.cache/xyntetik-runner/hf")) == []


def test_the_commit_message_check_does_not_read_the_record_rule():
    spec = importlib.util.spec_from_file_location("public_content", ROOT / "scripts/public_content.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.found("the cache is at ~/.cache/xyntetik-runner/hf") == []
