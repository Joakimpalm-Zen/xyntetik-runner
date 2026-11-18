"""R20: on Windows a dirty build's identity is the build id the linker writes (-Wl,--build-id), so that id must be
derived from the linked CONTENT: an RSDS record alone proves nothing. This links a tiny program with the runner's
own Windows link flags and has it print the id that src/pe_buildid.h's walker reads from its own loaded image, the
way the runner reads its own:

- twice from one object: the two ids must be equal (the control: the id is not a timestamp or a random value);
- once from an object with one changed constant: its id must differ (the id follows the content).

If the changed program's id equalled the first, R20's Windows identity would be false as built. Skipped, with the
reason, where the platform's linker does not write PE images; never skipped on Windows.
"""
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

PROGRAM = r'''
#include <stdio.h>
#include <windows.h>
#include <psapi.h>
#include "pe_buildid.h"
static const volatile int content = CONTENT;
int main(void) {
    HMODULE h = GetModuleHandleW(NULL);
    MODULEINFO mi;
    char out[96];
    if (!h || !GetModuleInformation(GetCurrentProcess(), h, &mi, sizeof mi) ||
        !pe_buildid_walk((const uint8_t *)mi.lpBaseOfDll, (size_t)mi.SizeOfImage, out, sizeof out))
        return 2;
    printf("%s %d\n", out, content);
    return 0;
}
'''
# the runner's Windows link flags (Makefile: LDFLAGS += -lws2_32 -lpsapi -static, then -Wl,--build-id)
LINK = ["-lws2_32", "-lpsapi", "-static", "-Wl,--build-id"]


@pytest.mark.skipif(sys.platform != "win32",
                    reason="the build-id derivation is a property of MinGW's PE linker; this host links Mach-O or ELF")
def test_the_pe_build_id_follows_the_linked_content(tmp_path):
    # CC is a command line, not a file name: CI's Windows job sets "cc -Werror -Wshadow". Split it (posix=False on
    # Windows, so a backslash in a path stays a backslash) and keep its flags.
    cc_line = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
    assert cc_line, "no C compiler on PATH on a Windows host"
    cc = shlex.split(cc_line, posix=os.name != "nt")
    cc[0] = shutil.which(cc[0]) or cc[0]   # the compiler as a file CreateProcess can start
    (tmp_path / "prog.c").write_text(PROGRAM)

    def step(cmd):
        r = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, "step failed (exit %d): %s\n%s" % (r.returncode, " ".join(cmd), r.stdout + r.stderr)

    def obj(name, content):
        o = tmp_path / (name + ".o")
        step([*cc, "-O2", "-I", str(ROOT / "src"), "-DCONTENT=%d" % content, "-c", str(tmp_path / "prog.c"), "-o",
              str(o)])
        return o

    def link_and_read(o, name):
        exe = tmp_path / (name + ".exe")
        step([*cc, str(o), "-o", str(exe), *LINK])
        r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, ("the walker found no build id in its own image", name, r.returncode, r.stderr)
        return r.stdout.split()[0]

    one = obj("one", 1)
    a1, a2 = link_and_read(one, "a1"), link_and_read(one, "a2")
    b = link_and_read(obj("two", 2), "b")
    assert a1.startswith("pe-buildid:") and a1 == a2, ("one object linked twice gave two ids", a1, a2)
    assert b != a1, ("one changed constant left the build id as it was: the id does not follow the content", a1, b)
