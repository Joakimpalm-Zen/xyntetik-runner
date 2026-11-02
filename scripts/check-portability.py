#!/usr/bin/env python3
"""PR-time portability check: POSIX calls the Windows (MinGW-w64) build lacks.

The Windows suite runs on main only, so a test or source line that calls a
POSIX function MinGW-w64 does not declare passed every PR check and broke
main (2026-11-02: a bare setenv in tests/test_canon_kernels.c, #553 -> #554).
This scan runs in the PR `check` job and fails on such a call in src/ or
tests/ C sources outside code the preprocessor keeps from Windows.

The list holds only functions verified absent from the mingw-w64 CRT headers
(mingw-w64-headers/crt: stdlib.h, string.h, stdio.h, unistd.h, time.h,
strings.h, io.h, sys/stat.h, dirent.h; checked 2026-11-02). Functions those
headers do declare (mkstemp, mkdtemp, strndup, strtok_r, strcasecmp, usleep)
are not listed, and neither are the ones winpthreads supplies to every MSYS2
build (clock_gettime, nanosleep, gmtime_r, localtime_r).

A call is not flagged when it sits under a branch the preprocessor excludes
on Windows: `#ifndef _WIN32` and `#if !defined(_WIN32)`; `#ifdef __linux__`,
`__APPLE__` and the other non-Windows platforms; any `#elif` or `#else` after
an `#ifdef _WIN32` branch (the same for _WIN64, __MINGW32__ and _MSC_VER). Nor
inside a `#define` (a compat macro naming the POSIX spelling), in a comment or
a string, or on a line carrying a `portability: ok` comment for code guarded
some other way.

    python3 scripts/check-portability.py [FILE ...]     # default: src/ and tests/
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# absent from mingw-w64's CRT headers, with what to use instead
MISSING = {
    "setenv": "plat_setenv (compat.h) or the test's setenv_compat",
    "unsetenv": "plat_setenv(name, \"\") (an empty value is off) or setenv_compat",
    "realpath": "a _WIN32 branch (_fullpath)",
    "getline": "fgets into a growing buffer",
    "getdelim": "fgets into a growing buffer",
    "strsep": "strtok_r or a manual split",
    "timegm": "a _WIN32 branch (_mkgmtime)",
    "posix_memalign": "a _WIN32 branch (_aligned_malloc)",
    "aligned_alloc": "a _WIN32 branch (_aligned_malloc)",
    "fmemopen": "a temporary file",
    "open_memstream": "an sbuf",
    "fork": "a _WIN32 branch (CreateProcess, plat_spawn_wait)",
    "pipe": "a _WIN32 branch (_pipe)",
    "symlink": "a _WIN32 branch",
    "readlink": "a _WIN32 branch",
    "sigaction": "a _WIN32 branch (signal / SetConsoleCtrlHandler)",
    "mmap": "a _WIN32 branch (CreateFileMapping)",
    "getpagesize": "a _WIN32 branch (GetSystemInfo)",
}
CALL = re.compile(r"(?<![A-Za-z0-9_.>])(" + "|".join(MISSING) + r")\s*\(")
WIN = r"(?:_WIN32|_WIN64|__MINGW32__|__MINGW64__|_MSC_VER)"
# platforms the Windows build is never compiled as
POSIX = r"(?:__linux__|__APPLE__|__unix__|__FreeBSD__|__OpenBSD__|__NetBSD__)"


def cond_kind(cond):
    """What a conditional's expression says about Windows: "win" (only
    Windows), "posix" (never Windows) or "other"."""
    c = cond.strip()
    if re.fullmatch(r"!\s*defined\s*\(?\s*" + WIN + r"\s*\)?", c):
        return "posix"
    if re.fullmatch(r"defined\s*\(?\s*" + WIN + r"\s*\)?", c):
        return "win"
    if re.fullmatch(r"defined\s*\(?\s*" + POSIX + r"\s*\)?", c):
        return "posix"
    return "other"


ESCAPE = "portability: ok"


def strip_comments(text):
    """The source with comments and string literals blanked, line count kept."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(c + " " * (min(j, n) - i - 1) + (c if j < n and text[j] == c else ""))
            i = j + 1 if j < n and text[j] == c else j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def scan(path):
    """(line number, function) for each unguarded call in one file."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    raw_lines = raw.split("\n")
    lines = strip_comments(raw).split("\n")
    # each open conditional: [kind of the current branch ("win", "posix",
    # "other"), whether an earlier branch was Windows-only, kind of the first]
    stack, hits, in_define = [], [], False
    for no, line in enumerate(lines, 1):
        s = line.strip()
        if in_define:
            in_define = s.endswith("\\")
            continue
        m = re.match(r"#\s*(ifdef|ifndef|if|elif|else|endif|define)\b(.*)", s)
        if m:
            d, rest = m.group(1), m.group(2)
            if d in ("ifdef", "ifndef"):
                name = rest.strip()
                k = cond_kind("defined(" + name + ")")
                if d == "ifndef":
                    k = {"win": "posix", "posix": "other"}.get(k, "other")
                stack.append([k, k == "win", k])
            elif d == "if":
                k = cond_kind(rest)
                stack.append([k, k == "win", k])
            elif d == "elif" and stack:
                f = stack[-1]
                f[0] = "posix" if f[1] else cond_kind(rest)
                f[1] = f[1] or f[0] == "win"
            elif d == "else" and stack:
                f = stack[-1]
                f[0] = "posix" if f[1] else ("win" if f[2] == "posix" else "other")
            elif d == "endif" and stack:
                stack.pop()
            elif d == "define":
                in_define = s.endswith("\\")
            continue
        if any(f[0] == "posix" for f in stack):
            continue
        if ESCAPE in raw_lines[no - 1]:
            continue
        for m in CALL.finditer(line):
            hits.append((no, m.group(1)))
    return hits


def main(argv):
    files = [Path(a) for a in argv] or sorted(
        p for d in ("src", "tests") for p in (ROOT / d).rglob("*")
        if p.suffix in (".c", ".h", ".m") and p.is_file())
    bad = 0
    for f in files:
        for no, fn in scan(f):
            rel = f.relative_to(ROOT) if f.is_absolute() and ROOT in f.parents else f
            print(f"{rel}:{no}: {fn}() is not in the MinGW-w64 CRT; use {MISSING[fn]}, "
                  f"guard it from _WIN32, or mark a guarded line `{ESCAPE}`")
            bad += 1
    if bad:
        print(f"check-portability: {bad} call(s) the Windows build lacks", file=sys.stderr)
        return 1
    print(f"check-portability: ok ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
