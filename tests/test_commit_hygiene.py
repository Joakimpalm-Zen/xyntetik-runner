"""R1.19: commit messages, merge commits included, and pull request text are
held to the same list of host paths and machine or session names as the
files a change adds (scripts/public_content.py, one list for both checkers).

On 2026-11-03 a commit message named a lab session and passed the commit
hygiene check, which knew only session URLs, trailers and e-mail addresses;
the name list lived in the file check alone. The test runs the real checker
over real commits in a scratch repository. The names are spelled in pieces so
this file does not publish them.
"""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "check-commit-hygiene.py"
LAB = "gala" + "xa1"
HOST = "windows" + "3070"
SIGN = "\n\nCo-Authored-By: Claude Code (Opus 5.5) & Joakimpalm-Zen\n"


def git(repo, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
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
