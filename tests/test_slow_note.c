// plat_slow_note (R10.29): a wait that would otherwise be silent says its line once, from a watcher
// thread, when it is still running after the delay; a quick one ends first and says nothing. The GPU
// kernels' first JIT uses it, and this host needs no GPU to check the note itself.
#include "compat.h"

#include <stdio.h>
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

int main(void) {
    const char *line = "gpu: compiling the GPU kernels for this card and driver";
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
