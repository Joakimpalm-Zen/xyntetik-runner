"""Pre-commit hook guard for linked worktree git identity (R6.103)."""
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _init_repo(tmp_path):
    main = tmp_path / "main"
    main.mkdir()
    _git(main, "init")
    _git(main, "config", "user.name", "Owner")
    _git(main, "config", "user.email", "owner@example.com")
    _git(main, "config", "extensions.worktreeConfig", "true")
    _git(main, "config", "core.hooksPath", ".githooks")

    hooks = main / ".githooks"
    hooks.mkdir()
    hook = hooks / "pre-commit"
    shutil.copy2(ROOT / ".githooks" / "pre-commit", hook)
    hook.chmod(0o755)
    scripts = main / "scripts"
    scripts.mkdir()
    shutil.copy2(ROOT / "scripts" / "check-content-hygiene.py", scripts / "check-content-hygiene.py")
    shutil.copy2(ROOT / "scripts" / "public_content.py", scripts / "public_content.py")

    sub = main / "sub"
    sub.mkdir()
    (sub / "a.txt").write_text("init\n", encoding="utf-8")
    _git(main, "add", ".")
    _git(main, "commit", "--no-verify", "-m", "initial commit")

    wt = tmp_path / "wt-agent"
    _git(main, "worktree", "add", "-b", "agent-branch", str(wt))
    return main, wt


def test_linked_worktree_requires_worktree_email_and_main_is_unaffected(tmp_path):
    main, wt = _init_repo(tmp_path)

    # 1. Linked worktree without worktree-local user.email is refused with both commands printed.
    (wt / "note.txt").write_text("hello\n", encoding="utf-8")
    _git(wt, "add", "note.txt")
    bad = _git(wt, "commit", "-m", "should fail", check=False)
    assert bad.returncode != 0
    assert "git config --worktree user.name" in bad.stderr
    assert "git config --worktree user.email" in bad.stderr
    assert "AGENTS.md section 6" in bad.stderr
    assert _git(wt, "rev-list", "--count", "HEAD").stdout.strip() == "1"

    # 2. Setting --worktree user.name and user.email lets the commit proceed as that identity.
    _git(wt, "config", "--worktree", "user.name", "Agent")
    _git(wt, "config", "--worktree", "user.email", "agent@example.invalid")
    good = _git(wt, "commit", "-m", "worktree identity commit")
    assert good.returncode == 0, good.stderr
    assert _git(wt, "log", "-1", "--format=%an <%ae>").stdout.strip() == "Agent <agent@example.invalid>"

    # 3. Main checkout stays unaffected (from root and from a subdirectory).
    (main / "root.txt").write_text("main root\n", encoding="utf-8")
    _git(main, "add", "root.txt")
    assert _git(main, "commit", "-m", "main root commit").returncode == 0

    (main / "sub" / "nested.txt").write_text("main sub\n", encoding="utf-8")
    _git(main / "sub", "add", "nested.txt")
    assert _git(main / "sub", "commit", "-m", "main sub commit").returncode == 0
    assert _git(main, "log", "-1", "--format=%an <%ae>").stdout.strip() == "Owner <owner@example.com>"
