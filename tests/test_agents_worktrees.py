"""Temporary-repo tests for scripts/agents-worktrees.sh and AGENTS.md sparse recipe (R6.97)."""
import os
import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "agents-worktrees.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash is not on PATH")


def _git(cwd, *args):
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _setup_repo(tmp_path):
    """Create a bare origin, main clone, and linked worktrees covering each state."""
    env_git = [
        ("user.name", "Test User"),
        ("user.email", "test@example.invalid"),
    ]
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "--initial-branch=main", str(bare)],
                   check=True, capture_output=True, text=True, timeout=30)

    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(bare), str(seed)],
                   check=True, capture_output=True, text=True, timeout=30)
    for k, v in env_git:
        _git(seed, "config", k, v)
    (seed / ".gitignore").write_text(".build/\n*.o\nrunner\n", encoding="utf-8")
    (seed / "README.md").write_text("base\n", encoding="utf-8")
    (seed / "src").mkdir()
    (seed / "src" / "main.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "initial commit")
    _git(seed, "push", "-u", "origin", "main")

    repo = tmp_path / "main-repo"
    subprocess.run(["git", "clone", str(bare), str(repo)],
                   check=True, capture_output=True, text=True, timeout=30)
    for k, v in env_git:
        _git(repo, "config", k, v)

    # 1. Merged & strictly clean linked worktree
    wt_merged_clean = tmp_path / "wt-merged-clean"
    _git(repo, "worktree", "add", "-b", "feat/merged-clean", str(wt_merged_clean), "origin/main")

    # 2. Merged + only ignored and unignored build outputs (must be treated as clean)
    wt_merged_build = tmp_path / "wt-merged-build"
    _git(repo, "worktree", "add", "-b", "feat/merged-build", str(wt_merged_build), "origin/main")
    (wt_merged_build / ".build").mkdir()
    (wt_merged_build / ".build" / "main.o").write_bytes(b"\x00" * 4096)
    (wt_merged_build / "build").mkdir()
    (wt_merged_build / "build" / "artifact.o").write_bytes(b"\x00" * 1024)
    (wt_merged_build / "test-full-grad").write_bytes(b"\x00" * 128)
    (wt_merged_build / "sample.out").write_text("log\n", encoding="utf-8")

    # 3. Merged + one untracked source file (must NOT be removed)
    wt_merged_untracked = tmp_path / "wt-merged-untracked"
    _git(repo, "worktree", "add", "-b", "feat/merged-untracked", str(wt_merged_untracked), "origin/main")
    (wt_merged_untracked / "src" / "helper.c").write_text("int helper(void) { return 42; }\n", encoding="utf-8")

    # 4. Merged + uncommitted modification to tracked file (must NOT be removed)
    wt_merged_dirty = tmp_path / "wt-merged-dirty"
    _git(repo, "worktree", "add", "-b", "feat/merged-dirty", str(wt_merged_dirty), "origin/main")
    (wt_merged_dirty / "src" / "main.c").write_text("int main(void) { return 1; }\n", encoding="utf-8")

    # 5. Unmerged commit ahead of origin/main + clean (must NOT be removed)
    wt_unmerged = tmp_path / "wt-unmerged"
    _git(repo, "worktree", "add", "-b", "feat/unmerged", str(wt_unmerged), "origin/main")
    for k, v in env_git:
        _git(wt_unmerged, "config", k, v)
    (wt_unmerged / "feature.txt").write_text("wip\n", encoding="utf-8")
    _git(wt_unmerged, "add", "feature.txt")
    _git(wt_unmerged, "commit", "-m", "unmerged feature")

    # 6. Merged detached linked worktree (must be removed)
    wt_detached = tmp_path / "wt-detached"
    _git(repo, "worktree", "add", "--detach", str(wt_detached), "origin/main")

    # 7. Merged + clean + locked linked worktree (must NOT be removed)
    wt_locked = tmp_path / "wt-locked"
    _git(repo, "worktree", "add", "-b", "feat/locked", str(wt_locked), "origin/main")
    _git(repo, "worktree", "lock", "--reason", "held", str(wt_locked))

    # Advance a real branch merged into origin/main (merge commit) to verify
    # merge-base ancestry checks work after a non-fast-forward PR merge too.
    wt_pr_merged = tmp_path / "wt-pr-merged"
    _git(repo, "worktree", "add", "-b", "feat/pr-merged", str(wt_pr_merged), "origin/main")
    for k, v in env_git:
        _git(wt_pr_merged, "config", k, v)
    (wt_pr_merged / "pr.txt").write_text("pr change\n", encoding="utf-8")
    _git(wt_pr_merged, "add", "pr.txt")
    _git(wt_pr_merged, "commit", "-m", "pr commit")
    _git(seed, "fetch", str(repo), "feat/pr-merged")
    _git(seed, "merge", "--no-ff", "-m", "Merge PR", "FETCH_HEAD")
    _git(seed, "push", "origin", "main")
    _git(repo, "fetch", "origin")

    return {
        "repo": repo,
        "merged_clean": wt_merged_clean,
        "merged_build": wt_merged_build,
        "merged_untracked": wt_merged_untracked,
        "merged_dirty": wt_merged_dirty,
        "unmerged": wt_unmerged,
        "detached": wt_detached,
        "locked": wt_locked,
        "pr_merged": wt_pr_merged,
    }


def _row_for(output, path):
    target_name = pathlib.Path(path).name
    for line in output.splitlines():
        if line.rstrip().endswith("/" + target_name) or line.rstrip().endswith("\\" + target_name):
            return line
    raise AssertionError(f"missing row for {target_name} in:\n{output}")


def test_list_and_remove_merged_temporary_repo(tmp_path):
    w = _setup_repo(tmp_path)

    listed = subprocess.run(
        [BASH, str(SCRIPT), str(w["repo"])],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert listed.returncode == 0, listed.stderr
    out = listed.stdout

    # Check every worktree row reports size, in-main status, cleanliness, and branch.
    row_main = _row_for(out, w["repo"])
    assert re.search(r"\b\d+(?:\.\d+)?[KMGT]\b", row_main), row_main
    assert "yes" in row_main and "clean" in row_main and "main" in row_main, row_main

    row_build = _row_for(out, w["merged_build"])
    assert "yes" in row_build and "clean" in row_build and "feat/merged-build" in row_build, row_build

    row_untracked = _row_for(out, w["merged_untracked"])
    assert "yes" in row_untracked and "dirty" in row_untracked and "feat/merged-untracked" in row_untracked, row_untracked

    row_dirty = _row_for(out, w["merged_dirty"])
    assert "yes" in row_dirty and "dirty" in row_dirty and "feat/merged-dirty" in row_dirty, row_dirty

    row_unmerged = _row_for(out, w["unmerged"])
    assert re.search(r"\bno\b", row_unmerged) and "clean" in row_unmerged and "feat/unmerged" in row_unmerged, row_unmerged

    row_detached = _row_for(out, w["detached"])
    assert "yes" in row_detached and "clean" in row_detached and "detached" in row_detached, row_detached

    row_locked = _row_for(out, w["locked"])
    assert "yes" in row_locked and "locked" in row_locked and "feat/locked" in row_locked, row_locked

    row_pr = _row_for(out, w["pr_merged"])
    assert "yes" in row_pr and "clean" in row_pr and "feat/pr-merged" in row_pr, row_pr

    # --dry-run (alone and paired with --remove-merged) prints what would be removed and removes nothing.
    for args in (["--dry-run", str(w["repo"])], ["--remove-merged", "--dry-run", str(w["repo"])]):
        dry = subprocess.run(
            [BASH, str(SCRIPT), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert dry.returncode == 0, dry.stderr
        would_lines = [ln for ln in dry.stdout.splitlines() if ln.startswith("would remove:")]
        would_names = {pathlib.Path(ln.split("would remove:", 1)[1].strip()).name for ln in would_lines}
        assert would_names == {"wt-merged-clean", "wt-merged-build", "wt-detached", "wt-pr-merged"}, would_lines
        for wt_path in w.values():
            assert wt_path.exists(), f"--dry-run must not remove {wt_path}"

    # --remove-merged must remove only linked, unlocked worktrees that are merged and non-build clean.
    removed = subprocess.run(
        [BASH, str(SCRIPT), "--remove-merged", str(w["repo"])],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert removed.returncode == 0, removed.stderr

    assert w["repo"].exists(), "main worktree must never be removed"
    assert not w["merged_clean"].exists()
    assert not w["merged_build"].exists()
    assert not w["detached"].exists()
    assert not w["pr_merged"].exists()

    assert w["merged_untracked"].exists(), "untracked non-build file must block removal"
    assert w["merged_dirty"].exists(), "uncommitted change must block removal"
    assert w["unmerged"].exists(), "unmerged branch must block removal"
    assert w["locked"].exists(), "locked worktree must not be removed"


def test_agents_md_documents_sparse_recipe_and_worktrees_script(tmp_path):
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "scripts/agents-worktrees.sh" in text
    assert "git worktree lock" in text
    assert "sparse-checkout set --cone" in text

    # Exercise the five-line cone sparse-checkout recipe against a temporary repo.
    repo = tmp_path / "large-repo"
    subprocess.run(["git", "init", "--initial-branch=main", str(repo)],
                   check=True, capture_output=True, text=True, timeout=30)
    _git(repo, "config", "user.name", "Parent User")
    _git(repo, "config", "user.email", "parent@example.invalid")
    (repo / "README.md").write_text("root\n", encoding="utf-8")
    (repo / "research" / "heavy").mkdir(parents=True)
    (repo / "research" / "heavy" / "weights.bin").write_bytes(b"heavy")
    (repo / "research" / "keep").mkdir(parents=True)
    (repo / "research" / "keep" / "plan.md").write_text("keep\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "update-ref", "refs/remotes/origin/main", head)

    wt = tmp_path / "wt-sparse"
    _git(repo, "worktree", "add", "--no-checkout", "-b", "feat/sparse", "../wt-sparse", "origin/main")
    _git(wt, "sparse-checkout", "set", "--cone", "research/keep")
    _git(wt, "checkout")
    _git(wt, "config", "--worktree", "user.name", "Antigravity")
    _git(wt, "config", "--worktree", "user.email", "antigravity@users.noreply.github.com")

    assert (wt / "README.md").exists()
    assert (wt / "research" / "keep" / "plan.md").exists()
    assert not (wt / "research" / "heavy").exists()
    assert _git(wt, "config", "user.name").stdout.strip() == "Antigravity"
    assert _git(repo, "config", "user.name").stdout.strip() == "Parent User"

