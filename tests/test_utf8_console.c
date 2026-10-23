// plat_utf8_startup, plat_console_read_line and the console.
//
// Output: the engine writes UTF-8, and a console draws the bytes it is given
// in its output code page (437 or 850 out of the box), so every non-ASCII
// character of a reply came out as two or three wrong ones. The startup switches a console on stdout or stderr to UTF-8,
// leaves the code page alone when both are redirected (a pipe or a file takes
// the bytes as they are), and puts it back at exit, because the console is
// the shell's too and outlives this process.
//
// Input: the interactive chat read a typed line with fgets, which gives the
// console's input code page (437 or 850), so a typed accented letter reached
// the tokenizer as one byte that is not UTF-8; plat_console_read_line reads
// UTF-16 and returns UTF-8. The parent types a line into its own console
// (WriteConsoleInputW) and reads it back through that call.
//
// Windows only (the POSIX suite skips it by name). The parent allocates a
// console of its own and starts this binary as a child attached to it, three
// times: stdout and stderr on the console, both on a file, and stderr alone
// on the console. Each child runs the startup and reports the code page it
// saw; the parent reads the console's code page after each child exits.
//
// Absolute anchors: 65001 is CP_UTF8 in Microsoft's code page identifiers,
// and the typed line's bytes are its UTF-8 encoding, derived by hand from
// the Unicode code points (U+00E9 C3 A9, U+65E5 E6 97 A5, U+672C E6 9C AC,
// U+1F600 F0 9F 98 80, typed as the surrogate pair D83D DE00).
#ifdef _WIN32
#include "compat.h"

#include <windows.h>
#include <fcntl.h>
#include <io.h>
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

static HANDLE inheritable(const wchar_t *name, DWORD access, DWORD disposition);

// type `units` (UTF-16, one key press each) into the console input buffer
static bool type_keys(HANDLE in, const WCHAR *units, size_t n) {
    INPUT_RECORD r[64];
    if (n > 64) return false;
    for (size_t i = 0; i < n; i++) {
        memset(&r[i], 0, sizeof r[i]);
        r[i].EventType = KEY_EVENT;
        r[i].Event.KeyEvent.bKeyDown = TRUE;
        r[i].Event.KeyEvent.wRepeatCount = 1;
        r[i].Event.KeyEvent.wVirtualKeyCode = units[i] == '\r' ? VK_RETURN : 0;
        r[i].Event.KeyEvent.uChar.UnicodeChar = units[i];
    }
    DWORD written = 0;
    return WriteConsoleInputW(in, r, (DWORD)n, &written) && written == n;
}

// one typed line, read back through plat_console_read_line on fd 0
static int typed_line_case(void) {
    HANDLE in = inheritable(L"CONIN$", GENERIC_READ | GENERIC_WRITE, OPEN_EXISTING);
    int fd = in == INVALID_HANDLE_VALUE ? -1 : _open_osfhandle((intptr_t)in, _O_RDONLY);
    if (fd < 0 || _dup2(fd, 0) != 0) {
        fprintf(stderr, "FAIL: cannot make the console input stdin\n");
        return 1;
    }
    if (!plat_is_terminal(0)) {
        fprintf(stderr, "FAIL: stdin is not seen as a console\n");
        return 1;
    }
    // "cafe" with its acute accent, two ideographs, a face outside the BMP
    static const WCHAR typed[] = { 'c', 'a', 'f', 0x00E9, ' ', 0x65E5, 0x672C, ' ',
                                   0xD83D, 0xDE00, '\r' };
    static const unsigned char want[] = { 0x63, 0x61, 0x66, 0xC3, 0xA9, 0x20,
                                          0xE6, 0x97, 0xA5, 0xE6, 0x9C, 0xAC, 0x20,
                                          0xF0, 0x9F, 0x98, 0x80 };
    static const WCHAR ctrl_z[] = { 0x1A, '\r' };
    int bad = 0;
    bool oom = false;
    if (!type_keys((HANDLE)_get_osfhandle(0), typed, sizeof typed / sizeof *typed)) {
        fprintf(stderr, "FAIL: WriteConsoleInputW: error %lu\n", GetLastError());
        return 1;
    }
    char *line = plat_console_read_line(&oom);
    if (!line || strlen(line) != sizeof want || memcmp(line, want, sizeof want) != 0) {
        fprintf(stderr, "FAIL: the typed line came back as %zu bytes:",
                line ? strlen(line) : 0);
        for (size_t i = 0; line && line[i]; i++)
            fprintf(stderr, " %02X", (unsigned char)line[i]);
        fprintf(stderr, " (expected %zu bytes of UTF-8)\n", sizeof want);
        bad = 1;
    }
    free(line);
    type_keys((HANDLE)_get_osfhandle(0), ctrl_z, 2);
    line = plat_console_read_line(&oom);
    if (line || oom) {
        fprintf(stderr, "FAIL: Ctrl+Z on an empty line is not the end of input\n");
        bad = 1;
    }
    free(line);
    if (!bad) fprintf(stderr, "test-utf8-console: typed line ok\n");
    return bad;
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
    if (!bad) fprintf(stderr, "test-utf8-console: 3 code page cases ok (base %u)\n", base);
    bad |= typed_line_case();
    return bad;
}
#else
#include <stdio.h>
int main(void) {
    puts("skip: test-utf8-console (Windows console code page)");
    return 0;
}
#endif
