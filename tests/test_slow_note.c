// plat_slow_note (R10.29 / R6.130): a wait that would otherwise be silent says
// its line once, from a watcher thread, when it is still running after the
// delay; a quick one ends first and says nothing. Also guards R6.130: the CUDA
// module-load timer note must state only the measured wait, never claiming an
// unverified JIT compile or driver cache behavior.
#include "compat.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        printf("ok: %s\n", what);
}

static int said_lines(FILE *f, const char *line) {
    char buf[1024];
    int n = 0;
    rewind(f);
    while (fgets(buf, sizeof buf, f)) {
        buf[strcspn(buf, "\n")] = 0;
        if (strcmp(buf, line) == 0) n++;
    }
    return n;
}

static bool contains_ci(const char *hay, const char *needle) {
    size_t nlen = strlen(needle);
    if (!nlen) return true;
    for (const char *p = hay; *p; p++) {
        size_t i = 0;
        while (i < nlen && p[i] &&
               tolower((unsigned char)p[i]) == tolower((unsigned char)needle[i]))
            i++;
        if (i == nlen) return true;
    }
    return false;
}

static char *read_text_file(const char *path) {
    FILE *f = plat_fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz <= 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[got] = '\0';
    return buf;
}

// Extract concatenated C string literals inside a slice [s, end).
static void collect_c_strings(const char *s, const char *end, char *out, size_t cap) {
    size_t o = 0;
    for (const char *p = s; p < end && *p; p++) {
        if (*p != '"') continue;
        p++;
        while (p < end && *p && *p != '"') {
            if (*p == '\\' && p + 1 < end) p++;
            if (o + 1 < cap) out[o++] = *p;
            p++;
        }
    }
    if (cap) out[o] = '\0';
}

// Find the matching ')' of a function call starting at the opening '(',
// ignoring parentheses and semicolons inside C string literals.
static const char *find_call_end(const char *lparen) {
    int depth = 0;
    for (const char *p = lparen; *p; p++) {
        if (*p == '"') {
            p++;
            while (*p && *p != '"') {
                if (*p == '\\' && p[1]) p++;
                p++;
            }
            if (!*p) return NULL;
            continue;
        }
        if (*p == '(') depth++;
        else if (*p == ')') {
            depth--;
            if (depth == 0) return p;
        }
    }
    return NULL;
}

// R6.130 guard: inspect every plat_slow_note_begin(...) call in src/cuda.c
// and refuse any timer-triggered note that claims a compile or driver cache
// it never detected.
static void check_cuda_slow_note_wording(void) {
    char *cuda = read_text_file("src/cuda.c");
    ck(cuda != NULL, "read src/cuda.c for R6.130 wording guard");
    if (!cuda) return;

    const char *call = strstr(cuda, "plat_slow_note_begin(");
    ck(call != NULL, "src/cuda.c has plat_slow_note_begin call");
    int calls = 0;
    while (call) {
        calls++;
        const char *lparen = strchr(call, '(');
        const char *end = lparen ? find_call_end(lparen) : NULL;
        ck(end != NULL, "plat_slow_note_begin call terminates cleanly");
        if (!end) break;

        char text[1024] = "";
        collect_c_strings(lparen, end, text, sizeof text);
        ck(text[0] != '\0', "CUDA slow note carries non-empty message text");
        ck(!contains_ci(text, "compil"),
           "CUDA slow note does not claim an undetected compile");
        ck(!contains_ci(text, "cache"),
           "CUDA slow note does not claim unverified driver cache behavior");
        ck(contains_ci(text, "loading") && contains_ci(text, "2 s"),
           "CUDA slow note states the measured wait (> 2 s) while loading");

        call = strstr(end, "plat_slow_note_begin(");
    }
    ck(calls == 1, "src/cuda.c has exactly one plat_slow_note_begin caller");
    free(cuda);
}

int main(void) {
    check_cuda_slow_note_wording();

    const char *line = "gpu: loading the GPU kernels is taking longer than 2 s; still waiting on the CUDA driver";
    FILE *out = tmpfile();
    ck(out != NULL, "a file to say it into");
    if (!out) return 1;

    // still running after the delay: the line is said once, while it runs
    plat_slow_note *n = plat_slow_note_begin(out, line, 0.2);
    ck(n != NULL, "a slow note begins");
    plat_sleep_ms(700);
    ck(said_lines(out, line) == 1, "the line is said while the wait is still running");
    ck(plat_slow_note_end(n), "end says the line was said");
    ck(said_lines(out, line) == 1, "and only once");

    // over before the delay: nothing is said, before or after the end
    FILE *quiet = tmpfile();
    plat_slow_note *q = plat_slow_note_begin(quiet, line, 1.0);
    plat_sleep_ms(50);
    ck(!plat_slow_note_end(q), "a quick wait says nothing");
    plat_sleep_ms(1200);
    ck(said_lines(quiet, line) == 0, "not even after its delay has passed");

    ck(!plat_slow_note_end(NULL), "no note ends as unsaid");
    fclose(out);
    fclose(quiet);
    return g_fail;
}
