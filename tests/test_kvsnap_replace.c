// Windows-visible manifest replace test and planted-symlink refusal for
// kvsnap_write_manifest_file() (R6.101).
#include "kvsnap.h"
#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

static int g_fail = 0;
static void ck(bool cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        printf("ok: %s\n", what);
}

static int kv_fail_step_b(const char *tmp, const char *out) {
    (void)tmp; (void)out;
    return -1;
}

static bool file_exists(const char *p) {
    FILE *f = fopen(p, "rb");
    if (f) { fclose(f); return true; }
    return false;
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

    // Legacy-name case (no RUNNER_TEST_PARTIAL_SUFFIX): succeeds without
    // touching `<man>.partial` or overwriting its canary.
    const char *leg_lnk = "test-kvsnap-replace.kv.json.partial";
    remove(leg_lnk);
    vf = fopen(victim, "wb");
    if (vf) { fputs("untouched", vf); fclose(vf); }
    if (symlink(victim, leg_lnk) == 0) {
        unsetenv("RUNNER_TEST_PARTIAL_SUFFIX");
        const char *body4 = "{\"name\":\"legacy_ok\"";
        ck(kvsnap_write_manifest_file(man, body4, strlen(body4), NULL),
           "kvsnap manifest write succeeds with legacy .partial symlink present");
        size_t vn = 0;
        char *vbuf = plat_read_file(victim, 256, &vn);
        ck(vbuf && vn == 9 && memcmp(vbuf, "untouched", 9) == 0,
           "kvsnap legacy .partial canary untouched");
        free(vbuf);
        struct stat lst;
        ck(lstat(leg_lnk, &lst) == 0 && S_ISLNK(lst.st_mode),
           "kvsnap legacy .partial remains a symlink");
        remove(leg_lnk);
    }
    remove(victim);
#endif

    remove(man);

    // Pair publish tests for KV snapshots (.kv + .kv.json via plat_publish_pair):
    // 1. Fresh pair succeeds, cleans temps and lock file.
    const char *kv_f = "test-kvsnap-pair.kv";
    const char *man_f = "test-kvsnap-pair.kv.json";
    const char *kv_t = "test-kvsnap-pair.kv.partial-a";
    const char *man_t = "test-kvsnap-pair.kv.json.partial-a";
    remove(kv_f); remove(man_f); remove(kv_t); remove(man_t);
    FILE *kf = fopen(kv_t, "wb"); if (kf) { fputs("KV_BYTES_1", kf); fclose(kf); }
    FILE *mf = fopen(man_t, "wb"); if (mf) { fputs("{\"n\":1}", mf); fclose(mf); }
    ck(plat_publish_pair(kv_t, kv_f, man_t, man_f, true, NULL, NULL) == 0,
       "kvsnap pair publish succeeds on fresh destination");
    // 2. no_overwrite under lock returns -2 and cleans partials without overwriting.
    kf = fopen(kv_t, "wb"); if (kf) { fputs("KV_BYTES_2", kf); fclose(kf); }
    mf = fopen(man_t, "wb"); if (mf) { fputs("{\"n\":2}", mf); fclose(mf); }
    ck(plat_publish_pair(kv_t, kv_f, man_t, man_f, true, NULL, NULL) == -2,
       "kvsnap pair publish refuses overwrite when destination exists");
    FILE *chk = fopen(kv_t, "rb");
    ck(chk == NULL, "kvsnap refused overwrite unlinks staged kv partial");
    if (chk) fclose(chk);
    // 3. Step B failure on overwrite restores old manifest and keeps staged kv partial.
    kf = fopen(kv_t, "wb"); if (kf) { fputs("KV_BYTES_2", kf); fclose(kf); }
    mf = fopen(man_t, "wb"); if (mf) { fputs("{\"n\":2}", mf); fclose(mf); }
    plat_publish_ops ops = { .install_out = kv_fail_step_b };
    bool kept_kv = false;
    ck(plat_publish_pair(kv_t, kv_f, man_t, man_f, false, &kept_kv, &ops) == -1,
       "kvsnap pair publish reports Step B failure");
    ck(kept_kv, "kvsnap Step B failure preserves staged kv partial");
    size_t kvn = 0, mn = 0;
    char *kvb = plat_read_file(kv_f, 256, &kvn);
    char *mb = plat_read_file(man_f, 256, &mn);
    ck(kvb && kvn == 10 && memcmp(kvb, "KV_BYTES_1", 10) == 0,
       "kvsnap Step B failure preserves old .kv bytes");
    ck(mb && mn == 7 && memcmp(mb, "{\"n\":1}", 7) == 0,
       "kvsnap Step B failure atomically restores old .kv.json bytes");
    free(kvb); free(mb);
    ck(!has_partial("test-kvsnap-pair.kv.json.bak"),
       "kvsnap Step B failure leaves no .bak temp");
    ck(file_exists(kv_t) && !file_exists(man_t),
       "kvsnap Step B failure keeps kv partial and consumes manifest partial");
    remove(kv_t);
    remove(kv_f); remove(man_f);
    remove("test-kvsnap-pair.kv.publish.lock");
    return g_fail ? 1 : 0;
}
