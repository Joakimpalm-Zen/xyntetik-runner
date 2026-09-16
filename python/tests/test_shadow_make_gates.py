"""Shadow mode admits built ranges: a C change is verified by the make gates
of the C test files it touched, frozen with the post-state Makefile. A
synthetic repository with one gate stands in for the runner's Makefile
convention (``tests/test_x.c`` is ``make test-x`` then ``./test-x``)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from xyntetik_runner.shadow import Baseline, Disposition, ProtectedTests, verify
from xyntetik_runner.shadow.attempt import Budget, Workspace
from xyntetik_runner.shadow.importer import Episode
from xyntetik_runner.shadow.tasks import (
    RepairTask,
    Rejection,
    admit,
    c_function_spans,
    class_from_diff_c,
    classify_range,
    gate_name,
)
from xyntetik_runner.shadow.verifier import MANIFEST

HAVE_TOOLCHAIN = bool(shutil.which("make")) and bool(shutil.which(os.environ.get("CC") or "cc"))
needs_toolchain = pytest.mark.skipif(not HAVE_TOOLCHAIN, reason="make and a C compiler are needed")

HEADER = "int add(int a, int b);\n"
BUGGY = "#include \"add.h\"\n\nint add(int a, int b) {\n    return a - b;\n}\n"
CORRECT = "#include \"add.h\"\n\nint add(int a, int b) {\n    return a + b;\n}\n"
WEAK_TEST = ("#include <assert.h>\n#include <stdio.h>\n#include \"add.h\"\n\n"
             "int main(void) {\n    assert(add(2, 0) == 2);\n    puts(\"add ok\");\n    return 0;\n}\n")
# the strong gate also reads a fixture the Makefile generates, as the
# runner's gates read test.gguf
STRONG_TEST = ("#include <assert.h>\n#include <stdio.h>\n#include \"add.h\"\n\n"
               "int main(void) {\n    FILE *f = fopen(\"fixture.gguf\", \"rb\");\n"
               "    assert(f != NULL);\n    fclose(f);\n"
               "    assert(add(2, 3) == 5);\n    puts(\"add ok\");\n    return 0;\n}\n")
MAKEFILE = ("CC ?= cc\n\n"
            "test-add: tests/test_add.c src/add.c src/add.h\n"
            "\t$(CC) -I src tests/test_add.c src/add.c -o $@\n\n"
            "fixture.gguf:\n"
            "\tprintf 'GGUF' > $@\n")


def git(repo: Path, *args: str, date: str = "2026-09-01T12:00:00+00:00") -> str:
    env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date,
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@x"}
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          env=env, check=True).stdout.strip()


def _write(r: Path, src: str, test: str) -> None:
    (r / "src" / "add.h").write_text(HEADER, encoding="utf-8")
    (r / "src" / "add.c").write_text(src, encoding="utf-8")
    (r / "tests" / "test_add.c").write_text(test, encoding="utf-8")
    (r / "Makefile").write_text(MAKEFILE, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """The buggy C state with a weak gate at 11:00; the fix with the
    stronger gate at 12:00."""
    r = tmp_path / "cproj"
    (r / "src").mkdir(parents=True)
    (r / "tests").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    _write(r, BUGGY, WEAK_TEST)
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "buggy", date="2026-09-01T11:00:00+00:00")
    _write(r, CORRECT, STRONG_TEST)
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "fix", date="2026-09-01T12:00:00+00:00")
    return r


def episode(cwd: Path, **kw: Any) -> Episode:
    base = dict(source="claude_code", session_id="abcdef123456", turn=1, cwd=str(cwd),
                started_at="2026-09-01T11:30:00Z", ended_at="2026-09-01T12:10:00Z",
                request="add must add, and the gate must prove it",
                request_sha256="x" * 64)
    base.update(kw)
    return Episode(**base)  # type: ignore[arg-type]


def _base_worktree(repo: Path, tmp_path: Path) -> Path:
    base = git(repo, "rev-parse", "HEAD^")
    ws = tmp_path / "ws"
    git(repo, "worktree", "add", "--detach", "-q", str(ws), base)
    return ws


def test_fixture_targets_are_the_gguf_names_the_makefile_can_build(tmp_path: Path) -> None:
    from xyntetik_runner.shadow.verifier import fixture_targets
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.c").write_text(
        'const char *a = "test.gguf"; const char *b = "models/real.gguf"; // "test-qk.gguf"\n',
        encoding="utf-8")
    (tmp_path / "Makefile").write_text("test.gguf: scripts/make.py\n\tpython3 scripts/make.py $@\n"
                                       "test-qk.gguf:\n\ttouch $@\n\ttest-x: tests/test_x.c\n",
                                       encoding="utf-8")
    assert fixture_targets(tmp_path, "tests/test_x.c") == ("test-qk.gguf", "test.gguf")
    assert fixture_targets(tmp_path, "tests/missing.c") == ()


def test_gate_names_follow_the_makefile_convention() -> None:
    assert gate_name("tests/test_add.c") == "test-add"
    assert gate_name("tests/test_penalty_window.c") == "test-penalty-window"
    t = classify_range(["src/engine.c", "src/engine.h", "tests/test_penalty_window.c",
                        "tests/test_sampler.py", "Makefile", "README.md"])
    assert t.build == ("src/engine.c", "src/engine.h")
    assert t.c_tests == ("tests/test_penalty_window.c",)
    assert t.py_tests == ("tests/test_sampler.py",) and t.makefile


@needs_toolchain
def test_a_built_range_is_admitted_with_its_gate_as_the_verifier(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "tasks"
    task = admit(episode(repo), out_dir=out, timeout_s=300)
    assert isinstance(task, RepairTask), task
    assert task.verifier_kind == "make" and task.gates == ("test-add",)
    assert task.test_files == ("tests/test_add.c",) and task.visible_test_files == ("tests/test_add.c",)
    assert task.expected_tests == 1 and task.baseline_failing == 1
    assert task.failing_at_base == ("tests/test_add.c::test-add",)
    assert task.task_class == "function" and task.src_files == 1
    manifest = json.loads((Path(task.protected_dir) / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["kind"] == "make"
    assert set(manifest["files"]) == {"tests/test_add.c", "Makefile"}
    # the task round-trips with its new fields
    loaded = RepairTask.load(Path(task.protected_dir).parent / "task.json")
    assert loaded == task


@needs_toolchain
def test_the_gate_verdict_is_the_binary_and_the_makefile_is_protected(repo: Path, tmp_path: Path) -> None:
    task = admit(episode(repo), out_dir=tmp_path / "tasks", timeout_s=300)
    assert isinstance(task, RepairTask)
    protected = ProtectedTests.load(Path(task.protected_dir))
    assert protected.kind == "make"
    ws = _base_worktree(repo, tmp_path)
    baseline = Baseline.capture(ws)
    noop = verify(ws, protected, baseline, timeout_s=300)
    assert noop.passed is False and noop.reasons[0].startswith("no_op")
    # the wrong fix: the frozen strong test fails, the attempt's own weak test is not consulted
    (ws / "src" / "add.c").write_text(BUGGY.replace("a - b", "a * b"), encoding="utf-8")
    wrong = verify(ws, protected, baseline, timeout_s=300)
    assert wrong.passed is False and wrong.failed == 1 and not wrong.tamper
    # the right fix passes on the frozen gate
    (ws / "src" / "add.c").write_text(CORRECT, encoding="utf-8")
    right = verify(ws, protected, baseline, timeout_s=300)
    assert right.passed is True and right.passed_count == 1, right
    # a Makefile that turns the gate into a no-op is tamper, whatever it prints
    (ws / "Makefile").write_text("test-add:\n\ttrue\n", encoding="utf-8")
    tampered = verify(ws, protected, baseline, timeout_s=300)
    assert tampered.passed is False and any("Makefile" in t for t in tampered.tamper)
    # a workspace that also rewrote the frozen test file is tamper too
    (ws / "Makefile").write_text(MAKEFILE, encoding="utf-8")
    (ws / "tests" / "test_add.c").write_text(WEAK_TEST.replace("add ok", "weakened"), encoding="utf-8")
    assert any("tests/test_add.c" in t for t in verify(ws, protected, baseline, timeout_s=300).tamper)


@needs_toolchain
def test_run_tests_builds_the_gates_beside_the_workspace_not_in_it(repo: Path, tmp_path: Path) -> None:
    ws_dir = _base_worktree(repo, tmp_path)
    baseline = Baseline.capture(ws_dir)
    ws = Workspace(ws_dir, visible_tests=("tests/test_add.c",), pythonpath=(), python=sys.executable,
                   budget=Budget(test_runs=3, test_timeout_s=120), gates=("test-add",))
    try:
        # the weak test at base passes on the buggy code: the attempt sees its own tests, not the frozen ones
        first = ws.run_tests()
        assert first.startswith("exit code 0") and "make test-add" in first, first
        ws.edit_file("src/add.c", "a - b", "a - a")
        ws.write_file("tests/test_add.c", STRONG_TEST)
        second = ws.run_tests()
        assert second.startswith("exit code 1"), second
        ws.edit_file("src/add.c", "a - a", "a + b")
        third = ws.run_tests()
        assert third.startswith("exit code 0"), third
        assert ws.run_tests().startswith("error: test-run budget")
    finally:
        ws.close()
    # objects and binaries never landed in the tree the baseline is compared against
    assert set(baseline.changes(ws_dir).paths) == {"src/add.c", "tests/test_add.c"}


@needs_toolchain
def test_a_built_range_without_a_c_test_is_ineligible(repo: Path, tmp_path: Path) -> None:
    (repo / "src" / "add.c").write_text(CORRECT + "\nint twice(int a) { return add(a, a); }\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "no gate", date="2026-09-01T12:05:00+00:00")
    # a window over the last commit only: its range is source without a gate
    r = admit(episode(repo, started_at="2026-09-01T12:02:00Z", ended_at="2026-09-01T12:10:00Z"),
              out_dir=tmp_path / "tasks", timeout_s=300)
    assert isinstance(r, Rejection) and r.disposition is Disposition.INELIGIBLE
    assert "no C test file" in r.reason


@needs_toolchain
def test_a_gate_that_passes_at_base_is_refused_as_an_instrument(repo: Path, tmp_path: Path) -> None:
    # a range whose test file changed only cosmetically: the frozen gate passes on the buggy base
    (repo / "tests" / "test_add.c").write_text(WEAK_TEST.replace("add ok", "still ok"), encoding="utf-8")
    (repo / "src" / "add.c").write_text(CORRECT, encoding="utf-8")
    git(repo, "reset", "-q", "--soft", "HEAD^")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "weak", date="2026-09-01T12:00:00+00:00")
    r = admit(episode(repo), out_dir=tmp_path / "tasks", timeout_s=300)
    assert isinstance(r, Rejection) and r.disposition is Disposition.UNREPLAYABLE
    assert r.reason.startswith("instrument:")


def test_a_manifest_frozen_before_kinds_loads_as_pytest(tmp_path: Path) -> None:
    src = tmp_path / "protected"
    (src / "tests").mkdir(parents=True)
    (src / "tests" / "test_x.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
    ProtectedTests.freeze(src, {"tests/test_x.py": ["test_a"]})
    m = json.loads((src / MANIFEST).read_text(encoding="utf-8"))
    assert m["kind"] == "pytest"
    del m["kind"]
    (src / MANIFEST).write_text(json.dumps(m), encoding="utf-8")
    assert ProtectedTests.load(src).kind == "pytest"


C_FILE = '''#include <stdio.h>
#include "x.h"

static const int table[] = { 1, 2, 3 };

/* a comment with a brace { that must not count */
static int helper(int a)
{
    if (a > 0) {
        return a; // trailing comment }
    }
    return -a;
}

int add(int a, int b) {
    const char *s = "{ not a brace }";
    (void)s;
    return helper(a) + b;
}
'''


def test_c_function_spans_follow_braces_not_comments_or_strings() -> None:
    spans = c_function_spans(C_FILE)
    assert [(s[2], s[0], s[1]) for s in spans] == [("helper", 7, 13), ("add", 15, 19)]


def test_class_of_a_c_change_is_function_inside_one_definition() -> None:
    inside = "@@ -18,1 +18,1 @@\n-    return helper(a) - b;\n+    return helper(a) + b;\n"
    assert class_from_diff_c(inside, C_FILE) == ("function", 1)
    two = inside + "@@ -12,1 +12,1 @@\n-    return a;\n+    return -a;\n"
    assert class_from_diff_c(two, C_FILE) == ("file", 2)
    outside = "@@ -4,1 +4,1 @@\n-static const int table[] = { 1, 2 };\n+static const int table[] = { 1, 2, 3 };\n"
    assert class_from_diff_c(outside, C_FILE) == ("file", 1)
    include_only = "@@ -2,0 +2,1 @@\n+#include \"x.h\"\n" + inside
    assert class_from_diff_c(include_only, C_FILE) == ("function", 2)


def test_the_summary_reads_the_newest_record_of_an_unattempted_episode() -> None:
    """An episode rejected under the old rule and admitted under the widened
    one is eligible now, whatever its first record said."""
    from xyntetik_runner.shadow.evidence import EpisodeEvidence, Identity, summarize
    ident = Identity(project="", task_class="file", context_band="", tool_set=(), verifier_id="",
                     environment_id="", model_sha256="", quant="", template_sha256="",
                     runner_build="", backend="", harness_version="")
    rec = lambda d, why: EpisodeEvidence(  # noqa: E731
        episode_id="capture:abc:1", source="capture", observed_at="2026-09-16T00:00:00Z",
        disposition=d, identity=ident, baseline_sha256="", patch_sha256=None, changed_paths=(),
        verifier=None, wall_s=0.0, reasons=(why,))
    old = rec(Disposition.INELIGIBLE, "range touches 3 file(s) that need a build")
    new = rec(Disposition.NOT_ATTEMPTED_RESOURCE, "admitted as x; not attempted yet")
    assert summarize([old]).eligible == 0
    assert summarize([old, new]).eligible == 1 and summarize([old, new]).observed == 1


@needs_toolchain
def test_the_bank_admits_a_make_commit_with_its_message_as_the_request(repo: Path, tmp_path: Path) -> None:
    from xyntetik_runner.shadow.bank import build_bank
    entries = build_bank(repo, out_dir=tmp_path / "bank", limit=5, max_commits=10, kind="make",
                         max_gates=1, timeout_s=300)
    admitted = [e for e in entries if e.task]
    assert len(admitted) == 1 and admitted[0].task is not None
    task = admitted[0].task
    assert task.verifier_kind == "make" and task.gates == ("test-add",)
    assert task.request == "fix" and task.episode_id.startswith("bank:")
    reasons = {e.reason for e in entries if not e.task}
    assert "unreplayable: first commit in the range has no parent" in reasons
    # the pytest-only bank refuses the same commit, by name
    only_py = build_bank(repo, out_dir=tmp_path / "bank2", limit=5, max_commits=10, kind="pytest")
    assert not [e for e in only_py if e.task] and "built range, pytest bank" in {e.reason for e in only_py}


def test_edit_only_refuses_to_overwrite_an_existing_file(tmp_path: Path) -> None:
    (tmp_path / "a.c").write_text("int a(void) { return 1; }\n", encoding="utf-8")
    ws = Workspace(tmp_path, visible_tests=(), pythonpath=(), python=sys.executable,
                   budget=Budget(), edit_only=True)
    out = ws.write_file("a.c", "int a(void) { return 2; }\n")
    assert out.startswith("error: a.c exists (1 lines)") and "edit_file" in out
    assert (tmp_path / "a.c").read_text(encoding="utf-8") == "int a(void) { return 1; }\n"
    assert ws.write_file("b.c", "int b;\n").startswith("wrote")
    assert ws.edit_file("a.c", "return 1", "return 2").startswith("edited")


@needs_toolchain
def test_show_tests_stages_the_frozen_test_and_edits_to_it_are_tamper(repo: Path, tmp_path: Path) -> None:
    from xyntetik_runner.shadow.cli import stage_solution_tests
    task = admit(episode(repo), out_dir=tmp_path / "tasks", timeout_s=300)
    assert isinstance(task, RepairTask)
    ws = _base_worktree(repo, tmp_path)
    assert (ws / "tests" / "test_add.c").read_text(encoding="utf-8") == WEAK_TEST
    staged = stage_solution_tests(task, ws)
    assert staged == ("tests/test_add.c",)
    assert (ws / "tests" / "test_add.c").read_text(encoding="utf-8") == STRONG_TEST
    assert not (ws / "Makefile").read_text(encoding="utf-8").startswith("test-add:\ntrue")
    baseline = Baseline.capture(ws)
    protected = ProtectedTests.load(Path(task.protected_dir))
    # the staged test is the visible specification: the workspace's own gate now fails at base
    w = Workspace(ws, visible_tests=staged, pythonpath=(), python=sys.executable,
                  budget=Budget(test_runs=2, test_timeout_s=120), gates=("test-add",))
    try:
        assert w.run_tests().startswith("exit code 1")
    finally:
        w.close()
    # the right fix passes; weakening the staged test is tamper, the frozen copy still judges
    (ws / "src" / "add.c").write_text(CORRECT, encoding="utf-8")
    assert verify(ws, protected, baseline, timeout_s=300).passed is True
    (ws / "tests" / "test_add.c").write_text(WEAK_TEST, encoding="utf-8")
    out = verify(ws, protected, baseline, timeout_s=300)
    assert out.passed is False and any("tests/test_add.c" in t for t in out.tamper)


def test_a_cohort_with_another_tool_set_is_not_a_repeat() -> None:
    from dataclasses import replace
    from xyntetik_runner.shadow.cli import attempt_key
    from xyntetik_runner.shadow.evidence import Identity
    base = Identity(project="p", task_class="file", context_band="", verifier_id="v",
                    environment_id="e", model_sha256="m", quant="q", template_sha256="t",
                    runner_build="b", backend="cpu", harness_version="h",
                    tool_set=("list_files", "read_file", "write_file", "edit_file", "run_tests"))
    shown = replace(base, tool_set=(*base.tool_set, "visible:solution-tests"))
    assert attempt_key("bank:x:1", base) != attempt_key("bank:x:1", shown)
    assert attempt_key("bank:x:1", base) == attempt_key("bank:x:1", replace(base, project="other"))


def test_an_adapted_cohort_is_not_a_repeat_of_the_base() -> None:
    from dataclasses import replace
    from xyntetik_runner.shadow.cli import attempt_key
    from xyntetik_runner.shadow.evidence import Identity
    base = Identity(project="p", task_class="file", context_band="", tool_set=("run_tests",),
                    verifier_id="v", environment_id="e", model_sha256="m", quant="q",
                    template_sha256="t", runner_build="b", backend="cpu", harness_version="h")
    assert attempt_key("bank:x:1", base) != attempt_key("bank:x:1", replace(base, adapter_sha256="a1"))
