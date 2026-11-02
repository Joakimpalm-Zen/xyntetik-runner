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


def test_todays_tree_passes():
    r = subprocess.run([sys.executable, ROOT / "scripts/check-portability.py"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
