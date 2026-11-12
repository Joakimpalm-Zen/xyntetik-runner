"""scripts/check-portability.py: POSIX calls the MinGW-w64 build lacks.

The Windows suite runs on main only, so a bare setenv in a C test passed every
PR check and broke main (#553, fixed in #554). The scan runs in the PR
metadata check; these fixtures hold what it flags and what it lets through,
and today's tree must pass.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_portability", ROOT / "scripts/check-portability.py")
MOD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MOD)


def _hits(tmp_path, src):
    f = tmp_path / "t.c"
    f.write_text(src)
    return [fn for _, fn in MOD.scan(f)]


def test_a_bare_setenv_fails(tmp_path):
    # the #553 shape
    assert _hits(tmp_path, 'int main(void) { setenv("RUNNER_CPU_I8", "1", 1); return 0; }\n') == ["setenv"]


@pytest.mark.parametrize("src", [
    # the compat form: plat_setenv, or a macro naming the POSIX spelling
    'int main(void) { plat_setenv("RUNNER_CPU_I8", "1"); return 0; }\n',
    '#ifdef _WIN32\n#define setenv_compat(k, v) _putenv_s(k, v)\n#else\n'
    '#define setenv_compat(k, v) setenv(k, v, 1)\n#endif\n',
    # branches Windows never compiles
    '#ifndef _WIN32\nvoid f(void) { setenv("A", "1", 1); }\n#endif\n',
    '#if !defined(_WIN32)\nvoid f(void) { unsetenv("A"); }\n#endif\n',
    '#ifdef _WIN32\nvoid f(void) { _putenv_s("A", "1"); }\n#else\nvoid f(void) { setenv("A", "1", 1); }\n#endif\n',
    '#ifdef _WIN32\nint g;\n#elif defined(__APPLE__)\nint h;\n#else\nint n = readlink("/proc/self/exe", 0, 0);\n#endif\n',
    '#ifdef __linux__\nint n = readlink("/proc/self/exe", 0, 0);\n#endif\n',
    # not code: a comment, a string, a longer name
    '// setenv(name, value, 1) would not build on MinGW\n',
    'const char *s = "fork()";\nint my_fork(void);\nint x = my_fork();\n',
    # guarded some other way, and said so
    'void f(void) { fork(); }  // portability: ok (built only where fork exists)\n',
])
def test_what_passes(tmp_path, src):
    assert _hits(tmp_path, src) == []


def test_a_windows_only_branch_is_still_scanned(tmp_path):
    # #ifdef _WIN32 code is exactly what MinGW compiles
    assert _hits(tmp_path, '#ifdef _WIN32\nvoid f(void) { setenv("A", "1", 1); }\n#endif\n') == ["setenv"]


def test_the_cli_exits_nonzero_on_a_hit(tmp_path):
    f = tmp_path / "bad.c"
    f.write_text("void f(void) { unsetenv(\"A\"); }\n")
    r = subprocess.run([sys.executable, ROOT / "scripts/check-portability.py", str(f)],
                       capture_output=True, text=True)
    assert r.returncode == 1
    assert "unsetenv() is not in the MinGW-w64 CRT" in r.stdout


# The Makefile: cmp is a command the Windows CI shell (MSYS2 UCRT64) lacks, so
# a recipe may run it only inside ifeq ($(BUILD_OS),Darwin) (R6.68.37).
def _make_hits(tmp_path, text):
    f = tmp_path / "Makefile"
    f.write_text(text)
    return [(no, c) for no, c in MOD.scan_make(f)]


def test_a_cmp_outside_the_darwin_block_fails(tmp_path):
    assert _make_hits(tmp_path, "t: runner\n\t./runner > a.out; cmp -s a.out b.out || exit 1\n") == [(2, "cmp")]
    assert _make_hits(tmp_path, "t:\n\t@set -e; \\\n\tif cmp -s a b; then echo same; fi\n") == [(3, "cmp")]


@pytest.mark.parametrize("text", [
    # the shape of the 27: a test-metal target's body inside the Darwin branch
    "t:\nifeq ($(BUILD_OS),Darwin)\n\tcmp -s a.out b.out || { echo differ; exit 1; }\nelse\n\t@echo skip\nendif\n",
    # a conditional nested inside the Darwin branch is still Darwin only
    "t:\nifeq ($(BUILD_OS),Darwin)\nifeq ($(T3),1)\n\tcmp -s a b\nendif\nendif\n",
    # an else-if chain's Darwin branch, and ifneq's false branch
    "t:\nifeq ($(BUILD_OS),Windows)\n\t@echo win\nelse ifeq ($(BUILD_OS),Darwin)\n\tcmp -s a b\nendif\n",
    "t:\nifneq ($(BUILD_OS),Darwin)\n\t@echo not mac\nelse\n\tcmp -s a b\nendif\n",
    # not the command: a comment line, a longer name, Python's filecmp.cmp
    "t:\n\t@# bytes via python: the Windows shell has no cmp here\n\t./cmp_tool a b\n",
    "t:\n\t$(PYTHON) -c \"import filecmp; filecmp.cmp('a','b',shallow=False)\"\n",
])
def test_what_passes_in_a_makefile(tmp_path, text):
    assert _make_hits(tmp_path, text) == []


@pytest.mark.parametrize("text, line", [
    # the false branch of the Darwin test runs everywhere else
    ("t:\nifeq ($(BUILD_OS),Darwin)\n\t@true\nelse\n\tcmp -s a b\nendif\n", 5),
    # ifneq Darwin's true branch, and after the Darwin block closes
    ("t:\nifneq ($(BUILD_OS),Darwin)\n\tcmp -s a b\nendif\n", 3),
    ("t:\nifeq ($(BUILD_OS),Darwin)\n\t@true\nendif\n\tcmp -s a b\n", 5),
])
def test_what_fails_in_a_makefile(tmp_path, text, line):
    assert _make_hits(tmp_path, text) == [(line, "cmp")]


def test_todays_makefile_holds_its_cmps_in_darwin_blocks_and_a_planted_one_fails(tmp_path):
    # the anchor: today's Makefile, which has its byte comparisons by cmp in the
    # test-metal targets; all pass, and one planted in a target every host runs
    # is the only line refused
    make = (ROOT / "Makefile").read_text(encoding="utf-8")
    n = sum(1 for l in make.split("\n") if l.startswith("\t") and MOD.MAKE_CALL.search(l))
    assert n > 0, "no recipe runs cmp, so the Darwin rule is checked on nothing"  # 27 on 2026-11-12
    assert MOD.scan_make(ROOT / "Makefile") == []
    planted = make + "\ntest-planted-cmp: runner\n\t./runner -p hi -n 4 > p.out; cmp -s p.out p.ref\n"
    f = tmp_path / "Makefile"
    f.write_text(planted, encoding="utf-8")
    assert MOD.scan_make(f) == [(planted.count("\n"), "cmp")]
    r = subprocess.run([sys.executable, ROOT / "scripts/check-portability.py", str(f)],
                       capture_output=True, text=True)
    assert r.returncode == 1
    assert "cmp runs in a recipe outside ifeq ($(BUILD_OS),Darwin)" in r.stdout


def test_todays_tree_passes():
    r = subprocess.run([sys.executable, ROOT / "scripts/check-portability.py"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
