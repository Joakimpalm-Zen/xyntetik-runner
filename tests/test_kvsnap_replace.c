// Windows-visible manifest replace test and planted-symlink refusal for
// kvsnap_write_manifest_file() (R6.101).
#include "kvsnap.h"
#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif

static int g_fail = 0;
static void ck(bool cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        printf("ok: %s\n", what);
}

static bool has_partial(const char *prefix) {
    size_t pn = strlen(prefix);
    plat_dir *dp = plat_opendir(".");
    if (!dp) return false;
    bool found = false;
    for (const char *nm; (nm = plat_readdir(dp)) != NULL; ) {
        if (strncmp(nm, prefix, pn) == 0) { found = true; break; }
    }
    plat_closedir(dp);
    return found;
}

int main(void) {
    const char *man = "test-kvsnap-replace.kv.json";
    const char *pfx = "test-kvsnap-replace.kv.json.partial";
    remove(man);

    const char *body1 = "{\"name\":\"first\"";
    ck(kvsnap_write_manifest_file(man, body1, strlen(body1), NULL),
       "initial kvsnap manifest write succeeds");
    ck(!has_partial(pfx), "initial manifest write leaves no .partial-*");

    // On Windows a plain rename refuses an existing destination; plat_replace_file
    // overwrites the existing manifest cleanly.
    const char *body2 = "{\"name\":\"second\"";
    ck(kvsnap_write_manifest_file(man, body2, strlen(body2), NULL),
       "second kvsnap manifest write replaces existing destination");
    ck(!has_partial(pfx), "second manifest write leaves no .partial-*");

    size_t got_n = 0;
    char *got = plat_read_file(man, 4096, &got_n);
    ck(got && strstr(got, "\"second\"") != NULL && strstr(got, "\"chain\"") != NULL,
       "replaced manifest holds second payload and chain");
    free(got);

#ifndef _WIN32
    const char *lnk = "test-kvsnap-replace.kv.json.partial-fixed";
    const char *victim = "test-kvsnap-replace-victim.txt";
    remove(lnk);
    remove(victim);
    FILE *vf = fopen(victim, "wb");
    if (vf) { fputs("untouched", vf); fclose(vf); }
    if (symlink(victim, lnk) == 0) {
        const char *body3 = "{\"name\":\"third\"";
        setenv("RUNNER_TEST_PARTIAL_SUFFIX", "fixed", 1);
        bool wr = kvsnap_write_manifest_file(man, body3, strlen(body3), NULL);
        unsetenv("RUNNER_TEST_PARTIAL_SUFFIX");
        ck(!wr, "kvsnap manifest write refuses symlink planted at .partial-fixed");

        size_t vn = 0;
        char *vbuf = plat_read_file(victim, 256, &vn);
        ck(vbuf && vn == 9 && memcmp(vbuf, "untouched", 9) == 0,
           "kvsnap manifest symlink refusal leaves victim untouched");
        free(vbuf);

        size_t mn = 0;
        char *mbuf = plat_read_file(man, 4096, &mn);
        ck(mbuf && strstr(mbuf, "\"second\"") != NULL,
           "kvsnap manifest symlink refusal preserves existing manifest");
        free(mbuf);
        remove(lnk);
    }
    remove(victim);
#endif

    remove(man);
    return g_fail ? 1 : 0;
}
