// `-hf owner/repo[:TAG]`: a GGUF from the Hugging Face Hub by repository id.
// See hfhub.h for the contract. The selection is pure and tested on its own;
// the transfer shells out to curl and is tested against a fake hub
// (tests/test_hf_fetch.py) through RUNNER_CURL.
#include "hfhub.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "envelope.h"
#include "json.h"

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <process.h>
#include <windows.h>
#define HF_SEP '\\'
#define hf_isatty _isatty
#else
#include <sys/wait.h>
#include <unistd.h>
#define HF_SEP '/'
#define hf_isatty isatty
#endif

// ------------------------------------------------------------ selection

static const char *base_name(const char *name) {
    const char *b = strrchr(name, '/');
    return b ? b + 1 : name;
}

static bool ends_with_ci(const char *s, const char *suffix) {
    size_t n = strlen(s), m = strlen(suffix);
    if (m > n) return false;
    for (size_t i = 0; i < m; i++)
        if (tolower((unsigned char)s[n - m + i]) != tolower((unsigned char)suffix[i])) return false;
    return true;
}

static bool starts_with_ci(const char *s, const char *prefix) {
    for (; *prefix; s++, prefix++)
        if (!*s || tolower((unsigned char)*s) != tolower((unsigned char)*prefix)) return false;
    return true;
}

// `<prefix>-NNNNN-of-MMMMM.gguf`: *part (1-based) and *count when it matches
static bool part_suffix(const char *base, unsigned *part, unsigned *count, size_t *prefix_len) {
    size_t n = strlen(base);
    const char *tail = "-00000-of-00000.gguf";
    size_t m = strlen(tail);
    if (n < m + 1) return false;
    const char *p = base + n - m;
    if (p[0] != '-' || strncmp(p + 6, "-of-", 4) != 0 || !ends_with_ci(p + 15, ".gguf")) return false;
    for (int i = 1; i <= 5; i++) if (!isdigit((unsigned char)p[i])) return false;
    for (int i = 10; i <= 14; i++) if (!isdigit((unsigned char)p[i])) return false;
    *part = (unsigned)atoi(p + 1);
    *count = (unsigned)atoi(p + 10);
    *prefix_len = (size_t)(p - base);
    return *count > 0 && *part >= 1 && *part <= *count;
}

bool hf_is_first_part(const char *name, unsigned *count, size_t *prefix_len) {
    const char *base = base_name(name);
    unsigned part = 0, cnt = 0; size_t plen = 0;
    if (!part_suffix(base, &part, &cnt, &plen) || part != 1) return false;
    if (count) *count = cnt;
    if (prefix_len) *prefix_len = plen + (size_t)(base - name);
    return true;
}

// a candidate: a .gguf that is not a vision projector, and for a multipart
// file only its first part
static bool is_candidate(const char *name) {
    const char *base = base_name(name);
    if (!ends_with_ci(base, ".gguf")) return false;
    if (starts_with_ci(base, "mmproj")) return false;
    unsigned part = 0, cnt = 0; size_t plen = 0;
    if (part_suffix(base, &part, &cnt, &plen)) return part == 1;
    return true;
}

// the base name without ".gguf" and without a part suffix, into out
static void stem_of(const char *name, char *out, size_t cap) {
    const char *base = base_name(name);
    size_t n = strlen(base);
    unsigned part = 0, cnt = 0; size_t plen = 0;
    if (part_suffix(base, &part, &cnt, &plen)) n = plen;
    else if (ends_with_ci(base, ".gguf")) n -= 5;
    if (n >= cap) n = cap - 1;
    memcpy(out, base, n);
    out[n] = 0;
}

static bool token_boundary(char c) {
    return c == 0 || c == '-' || c == '_' || c == '.';
}

// `tag` as a whole token of `stem`, case-insensitive
static bool tag_matches(const char *stem, const char *tag) {
    size_t m = strlen(tag);
    if (!m) return true;
    for (const char *p = stem; *p; p++) {
        if (p != stem && !token_boundary(p[-1])) continue;
        size_t i = 0;
        while (i < m && p[i] && tolower((unsigned char)p[i]) == tolower((unsigned char)tag[i])) i++;
        if (i == m && token_boundary(p[m])) return true;
    }
    return false;
}

static void list_candidates(const char *const *names, int n, char *err, size_t cap, size_t at) {
    int listed = 0;
    for (int i = 0; i < n && at < cap; i++) {
        if (!is_candidate(names[i])) continue;
        if (listed == 8) { at += (size_t)snprintf(err + at, cap - at, ", ..."); break; }
        at += (size_t)snprintf(err + at, cap - at, "%s%s", listed ? ", " : "", base_name(names[i]));
        listed++;
    }
}

int hf_select_file(const char *const *names, int n, const char *tag, char *err, size_t errcap) {
    int found = -1, count = 0, total = 0;
    char stem[512];
    for (int i = 0; i < n; i++) {
        if (!is_candidate(names[i])) continue;
        total++;
        if (tag && *tag) {
            stem_of(names[i], stem, sizeof stem);
            if (!tag_matches(stem, tag)) continue;
        }
        if (found < 0) found = i;
        count++;
    }
    if (total == 0) {
        snprintf(err, errcap, "no GGUF file in the repository");
        return -1;
    }
    if (count == 1) return found;
    if (count == 0) {
        size_t at = (size_t)snprintf(err, errcap, "no GGUF file matches tag %s; the repository has: ", tag);
        list_candidates(names, n, err, errcap, at < errcap ? at : errcap - 1);
        return -1;
    }
    size_t at = (size_t)snprintf(err, errcap, "%d GGUF files %s; choose one with -hf REPO:TAG, from: ",
                                 count, tag && *tag ? "match the tag" : "in the repository");
    list_candidates(names, n, err, errcap, at < errcap ? at : errcap - 1);
    return -1;
}

// -------------------------------------------------------------- transfer

static bool mkdir_one(const char *p) {
#ifdef _WIN32
    return _mkdir(p) == 0 || errno == EEXIST || GetLastError() == ERROR_ALREADY_EXISTS;
#else
    return mkdir(p, 0755) == 0 || errno == EEXIST;
#endif
}

// mkdir -p over `path`'s directories (the last component included)
static bool mkdir_p(const char *path) {
    char buf[2048];
    size_t n = strlen(path);
    if (n >= sizeof buf) return false;
    memcpy(buf, path, n + 1);
    for (size_t i = 1; i < n; i++) {
        if (buf[i] != '/' && buf[i] != '\\') continue;
        char c = buf[i]; buf[i] = 0;
#ifdef _WIN32
        bool drive_root = i == 2 && buf[1] == ':';
#else
        bool drive_root = false;
#endif
        if (!drive_root && !mkdir_one(buf)) return false;
        buf[i] = c;
    }
    return mkdir_one(buf);
}

static bool file_size(const char *path, long long *size) {
    struct stat st;
    if (stat(path, &st) != 0) return false;
    *size = (long long)st.st_size;
    return true;
}

// run argv (NULL-terminated) and wait; the child's exit code, -1 when it
// could not be started
static int run_wait(char *const argv[]) {
#ifdef _WIN32
    intptr_t rc = _spawnvp(_P_WAIT, argv[0], (const char *const *)argv);
    return rc == -1 ? -1 : (int)rc;
#else
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        execvp(argv[0], argv);
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status) == 127 ? -1 : WEXITSTATUS(status);
    return -1;
#endif
}

static const char *curl_program(void) {
    const char *c = getenv("RUNNER_CURL");
    return c && *c ? c : "curl";
}

// curl -L --fail [-sS] [-H auth] -o out url ; 0 on success
static int curl_get(const char *url, const char *out, bool quiet, char *err, size_t errcap) {
    char auth[4096];
    const char *token = getenv("HF_TOKEN");
    if (token && *token) snprintf(auth, sizeof auth, "Authorization: Bearer %s", token);
    const char *argv[16];
    int n = 0;
    argv[n++] = curl_program();
    argv[n++] = "-L"; argv[n++] = "--fail"; argv[n++] = "--retry"; argv[n++] = "3";
    if (quiet) argv[n++] = "-sS";
    if (token && *token) { argv[n++] = "-H"; argv[n++] = auth; }
    argv[n++] = "-o"; argv[n++] = out;
    argv[n++] = url;
    argv[n] = NULL;
    int rc = run_wait((char *const *)argv);
    if (rc == -1) snprintf(err, errcap, "cannot run %s (set RUNNER_CURL, or install curl)", curl_program());
    else if (rc != 0) snprintf(err, errcap, "%s exited %d fetching %s", curl_program(), rc, url);
    return rc;
}

static char *read_whole(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) { free(buf); return NULL; }
    buf[n] = 0;
    if (len) *len = (size_t)n;
    return buf;
}

static bool cache_root(char *out, size_t cap) {
    const char *c = getenv("RUNNER_HF_CACHE");
    if (c && *c) return snprintf(out, cap, "%s", c) < (int)cap;
#ifdef _WIN32
    const char *base = getenv("LOCALAPPDATA");
    if (!base || !*base) return false;
    return snprintf(out, cap, "%s\\xyntetik-runner\\hf", base) < (int)cap;
#else
    const char *base = getenv("HOME");
    if (!base || !*base) return false;
    return snprintf(out, cap, "%s/.cache/xyntetik-runner/hf", base) < (int)cap;
#endif
}

// a repo file name as a cache-relative path: '/' becomes the platform's separator
static void local_name(const char *rfilename, char *out, size_t cap) {
    size_t n = strlen(rfilename);
    if (n >= cap) n = cap - 1;
    for (size_t i = 0; i < n; i++) out[i] = rfilename[i] == '/' ? HF_SEP : rfilename[i];
    out[n] = 0;
}

// one file of the repository: exists in the cache and verifies, or is fetched
// into `<target>.part`, verified, and renamed. `quiet` silences curl's
// progress meter (stderr is not a terminal); the one-line hf: notes stay,
// they are the record of what was loaded from where
static bool ensure_file(const char *repo, const char *rfilename, const char *repodir,
                        long long size, const char *sha, bool quiet, char *err, size_t errcap) {
    char rel[1024], target[2048], part[2100], url[2048];
    local_name(rfilename, rel, sizeof rel);
    if (snprintf(target, sizeof target, "%s%c%s", repodir, HF_SEP, rel) >= (int)sizeof target) {
        snprintf(err, errcap, "cache path too long for %s", rfilename);
        return false;
    }
    // the parent directories of a nested file
    char parent[2048];
    memcpy(parent, target, strlen(target) + 1);
    char *slash = strrchr(parent, HF_SEP);
    if (slash) { *slash = 0; if (!mkdir_p(parent)) { snprintf(err, errcap, "cannot create %s", parent); return false; } }

    char hex[65];
    long long have = 0;
    if (file_size(target, &have)) {
        bool ok = sha && *sha ? (envelope_file_sha256(target, hex) && strcmp(hex, sha) == 0)
                              : (size < 0 || have == size);
        if (ok) {
            fprintf(stderr, "hf: %s/%s cached\n", repo, rfilename);
            return true;
        }
        fprintf(stderr, "hf: %s/%s in the cache does not verify, fetching again\n", repo, rfilename);
        remove(target);
    }
    snprintf(part, sizeof part, "%s.part", target);
    remove(part);
    snprintf(url, sizeof url, "https://huggingface.co/%s/resolve/main/%s", repo, rfilename);
    fprintf(stderr, "hf: fetching %s/%s\n", repo, rfilename);
    if (curl_get(url, part, quiet, err, errcap) != 0) { remove(part); return false; }
    if (sha && *sha) {
        if (!envelope_file_sha256(part, hex) || strcmp(hex, sha) != 0) {
            remove(part);
            snprintf(err, errcap, "%s/%s: sha256 of the download does not match the Hub's LFS record "
                                  "(%.16s... expected %.16s...)", repo, rfilename, hex, sha);
            return false;
        }
    } else if (size >= 0 && (!file_size(part, &have) || have != size)) {
        remove(part);
        snprintf(err, errcap, "%s/%s: downloaded %lld bytes, the Hub lists %lld", repo, rfilename, have, size);
        return false;
    }
    if (rename(part, target) != 0) {
        remove(part);
        snprintf(err, errcap, "cannot move %s into place: %s", part, strerror(errno));
        return false;
    }
    return true;
}

char *hf_fetch(const char *spec, char *err, size_t errcap) {
    err[0] = 0;
    // owner/repo[:TAG]; a tag is what follows the last ':' after the slash
    const char *slash = strchr(spec, '/');
    if (!slash || slash == spec || !slash[1] || strchr(slash + 1, '/')) {
        snprintf(err, errcap, "-hf wants owner/repo[:TAG], got %s", spec);
        return NULL;
    }
    const char *colon = strrchr(slash, ':');
    char repo[512], tag[128] = "";
    size_t rn = colon ? (size_t)(colon - spec) : strlen(spec);
    if (rn >= sizeof repo || (colon && strlen(colon + 1) >= sizeof tag)) {
        snprintf(err, errcap, "-hf spec too long");
        return NULL;
    }
    memcpy(repo, spec, rn); repo[rn] = 0;
    if (colon) memcpy(tag, colon + 1, strlen(colon + 1) + 1);
    if (colon && !tag[0]) { snprintf(err, errcap, "-hf %s: empty tag after ':'", spec); return NULL; }

    char root[1024], repodir[1600];
    if (!cache_root(root, sizeof root)) {
        snprintf(err, errcap, "no cache directory: set RUNNER_HF_CACHE");
        return NULL;
    }
    // owner--repo, one directory per repository
    char flat[512];
    memcpy(flat, repo, rn + 1);
    for (char *p = flat; *p; p++) if (*p == '/') { memmove(p + 1, p, strlen(p) + 1); p[0] = '-'; p[1] = '-'; p++; }
    snprintf(repodir, sizeof repodir, "%s%c%s", root, HF_SEP, flat);
    if (!mkdir_p(repodir)) {
        snprintf(err, errcap, "cannot create cache directory %s: %s", repodir, strerror(errno));
        return NULL;
    }

    bool quiet = !hf_isatty(2);
    // the file list, with blob sizes and LFS hashes, fetched fresh every time
    // so a tag added to the repository since the last run is seen
    char api_url[1024], api_path[1700];
    snprintf(api_url, sizeof api_url, "https://huggingface.co/api/models/%s?blobs=true", repo);
    snprintf(api_path, sizeof api_path, "%s%c.files.json", repodir, HF_SEP);
    if (curl_get(api_url, api_path, true, err, errcap) != 0) {
        size_t at = strlen(err);
        if (at < errcap)
            snprintf(err + at, errcap - at, " (does %.200s exist? a gated or private repository needs HF_TOKEN)", repo);
        return NULL;
    }
    size_t jn = 0;
    char *js = read_whole(api_path, &jn);
    jv *doc = js ? json_parse(js, jn) : NULL;
    free(js);
    jv *sib = doc ? jv_get(doc, "siblings") : NULL;
    if (!sib || sib->type != J_ARR) {
        jv_free(doc);
        snprintf(err, errcap, "unexpected file list from the Hub for %s", repo);
        return NULL;
    }
    const char **names = calloc((size_t)(sib->n ? sib->n : 1), sizeof *names);
    if (!names) { jv_free(doc); snprintf(err, errcap, "out of memory"); return NULL; }
    int n = 0;
    for (int i = 0; i < sib->n; i++) {
        const char *rf = jv_str(jv_get(sib->items[i], "rfilename"), NULL);
        if (rf) names[n++] = rf;
    }
    int pick = hf_select_file(names, n, tag, err, errcap);
    if (pick < 0) { free(names); jv_free(doc); return NULL; }

    // every part of the chosen file, first part first
    const char *first = names[pick];
    unsigned parts = 1; size_t plen = 0;
    bool multipart = hf_is_first_part(first, &parts, &plen);
    bool ok = true;
    for (unsigned k = 1; ok && k <= parts; k++) {
        char want[1024];
        if (multipart) snprintf(want, sizeof want, "%.*s-%05u-of-%05u.gguf", (int)plen, first, k, parts);
        else snprintf(want, sizeof want, "%s", first);
        jv *entry = NULL;
        for (int i = 0; i < sib->n; i++)
            if (!strcmp(jv_str(jv_get(sib->items[i], "rfilename"), ""), want)) { entry = sib->items[i]; break; }
        if (!entry) { snprintf(err, errcap, "%s lists no part %u of %s", repo, k, first); ok = false; break; }
        jv *lfs = jv_get(entry, "lfs");
        const char *sha = lfs ? jv_str(jv_get(lfs, "sha256"), NULL) : NULL;
        double sz = jv_num(jv_get(entry, "size"), -1);
        ok = ensure_file(repo, want, repodir, sz < 0 ? -1 : (long long)sz, sha, quiet, err, errcap);
    }
    char *result = NULL;
    if (ok) {
        char rel[1024], target[2048];
        local_name(first, rel, sizeof rel);
        if (snprintf(target, sizeof target, "%s%c%s", repodir, HF_SEP, rel) >= (int)sizeof target)
            snprintf(err, errcap, "cache path too long for %s", first);
        else if (!(result = strdup(target)))
            snprintf(err, errcap, "out of memory");
    }
    free(names);
    jv_free(doc);
    return result;
}
