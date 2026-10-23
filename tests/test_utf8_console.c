// plat_utf8_startup and the console: the engine writes UTF-8, and a console
// draws the bytes it is given in its output code page (437 or 850 out of the
// box), so every non-ASCII character of a reply came out as two or three
// wrong ones. The startup switches a console on stdout or stderr to UTF-8,
// leaves the code page alone when both are redirected (a pipe or a file takes
// the bytes as they are), and puts it back at exit, because the console is
// the shell's too and outlives this process.
//
// Windows only (the POSIX suite skips it by name). The parent allocates a
// console of its own and starts this binary as a child attached to it, three
// times: stdout and stderr on the console, both on a file, and stderr alone
// on the console. Each child runs the startup and reports the code page it
// saw; the parent reads the console's code page after each child exits.
//
// Absolute anchor: 65001 is CP_UTF8 in Microsoft's code page identifiers.
#ifdef _WIN32
#include "compat.h"

#include <windows.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REPORT "test-utf8-console.out"

static int child(int argc, char **argv) {
    plat_utf8_startup(&argc, &argv);
    FILE *f = fopen(REPORT, "w");
    if (!f) return 2;
    fprintf(f, "%u\n", GetConsoleOutputCP());
    fclose(f);
    return 0;   // returning from main runs the atexit handler: the restore
}

static HANDLE inheritable(const wchar_t *name, DWORD access, DWORD disposition) {
    SECURITY_ATTRIBUTES sa = { sizeof sa, NULL, TRUE };
    return CreateFileW(name, access, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                       disposition, FILE_ATTRIBUTE_NORMAL, NULL);
}

// the code page the child reported, 0 when it did not run
static unsigned run_child(HANDLE out, HANDLE err) {
    DeleteFileA(REPORT);
    wchar_t self[MAX_PATH];
    if (!GetModuleFileNameW(NULL, self, MAX_PATH)) return 0;
    wchar_t cmd[MAX_PATH + 32] = L"\"";
    wcscat(cmd, self);
    wcscat(cmd, L"\" --child");
    HANDLE in = inheritable(L"CONIN$", GENERIC_READ | GENERIC_WRITE, OPEN_EXISTING);
    STARTUPINFOW si = { .cb = sizeof si, .dwFlags = STARTF_USESTDHANDLES,
                        .hStdInput = in, .hStdOutput = out, .hStdError = err };
    PROCESS_INFORMATION pi;
    unsigned cp = 0;
    if (CreateProcessW(NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        WaitForSingleObject(pi.hProcess, 30000);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        FILE *f = fopen(REPORT, "r");
        if (f) {
            if (fscanf(f, "%u", &cp) != 1) cp = 0;
            fclose(f);
        }
    }
    CloseHandle(in);
    DeleteFileA(REPORT);
    return cp;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--child") == 0) return child(argc, argv);

    // a console of our own, so the test neither depends on nor disturbs the
    // one the suite runs in
    FreeConsole();
    if (!AllocConsole()) {
        fprintf(stderr, "FAIL: AllocConsole: error %lu\n", GetLastError());
        return 1;
    }
    // start from a legacy code page even on a machine set to UTF-8 system-wide
    UINT base = GetOEMCP() == CP_UTF8 ? 437 : GetOEMCP();
    if (!SetConsoleOutputCP(base)) {
        fprintf(stderr, "FAIL: SetConsoleOutputCP(%u): error %lu\n", base, GetLastError());
        return 1;
    }
    HANDLE con = inheritable(L"CONOUT$", GENERIC_READ | GENERIC_WRITE, OPEN_EXISTING);
    HANDLE file = inheritable(L"test-utf8-console.redirect", GENERIC_WRITE, CREATE_ALWAYS);
    if (con == INVALID_HANDLE_VALUE || file == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "FAIL: cannot open CONOUT$ or the redirect file\n");
        return 1;
    }

    int bad = 0;
    struct { const char *what; HANDLE out, err; unsigned want; } cases[] = {
        { "stdout and stderr on the console", con, con, 65001 },
        { "stdout and stderr redirected to a file", file, file, base },
        { "stdout redirected, stderr on the console", file, con, 65001 },
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) {
        unsigned seen = run_child(cases[i].out, cases[i].err);
        UINT after = GetConsoleOutputCP();
        if (seen != cases[i].want) {
            fprintf(stderr, "FAIL: %s: the child saw code page %u, expected %u\n",
                    cases[i].what, seen, cases[i].want);
            bad = 1;
        }
        if (after != base) {
            fprintf(stderr, "FAIL: %s: the console was left at %u, expected %u restored\n",
                    cases[i].what, after, base);
            bad = 1;
            SetConsoleOutputCP(base);
        }
    }
    CloseHandle(con);
    CloseHandle(file);
    DeleteFileA("test-utf8-console.redirect");
    if (!bad) fprintf(stderr, "test-utf8-console: 3 cases ok (base code page %u)\n", base);
    return bad;
}
#else
#include <stdio.h>
int main(void) {
    puts("skip: test-utf8-console (Windows console code page)");
    return 0;
}
#endif
