// Per-request receipts in serve mode (R1.2.2). See receipts.h.
#include "receipts.h"

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define rc_mkdir(p) _mkdir(p)
#else
#include <dirent.h>
#include <sys/stat.h>
#define rc_mkdir(p) mkdir(p, 0700)
#endif

// receipt-<10-digit sequence>.json: the sequence orders records within and
// across runs, and a lexical sort of the names is that order
#define RC_PREFIX "receipt-"
#define RC_SUFFIX ".json"

static struct {
    bool            on;
    char            dir[1024];
    char            sign_key[1024];
    bool            signing;
    int             keep;
    unsigned long long next_seq;
    char            head[65];
    pthread_mutex_t mu;
} RC = { .mu = PTHREAD_MUTEX_INITIALIZER };

static bool seq_of(const char *name, unsigned long long *seq) {
    size_t pn = strlen(RC_PREFIX), sn = strlen(RC_SUFFIX), n = strlen(name);
    if (n != pn + 10 + sn || strncmp(name, RC_PREFIX, pn) ||
        strcmp(name + pn + 10, RC_SUFFIX))
        return false;
    unsigned long long v = 0;
    for (size_t i = pn; i < pn + 10; i++) {
        if (name[i] < '0' || name[i] > '9') return false;
        v = v * 10 + (unsigned long long)(name[i] - '0');
    }
    *seq = v;
    return true;
}

// Every record sequence in the directory, ascending. Returns the count;
// *out is the caller's to free.
static int list_seqs(unsigned long long **out) {
    int n = 0, cap = 0;
    unsigned long long *v = NULL;
#ifdef _WIN32
    char pat[1100];
    snprintf(pat, sizeof pat, "%s\\" RC_PREFIX "*" RC_SUFFIX, RC.dir);
    struct _finddata_t fd;
    intptr_t h = _findfirst(pat, &fd);
    if (h != -1) {
        do {
            unsigned long long s;
            if (!seq_of(fd.name, &s)) continue;
            if (n == cap) {
                cap = cap ? cap * 2 : 64;
                unsigned long long *g = realloc(v, sizeof *v * (size_t)cap);
                if (!g) break;
                v = g;
            }
            v[n++] = s;
        } while (_findnext(h, &fd) == 0);
        _findclose(h);
    }
#else
    DIR *d = opendir(RC.dir);
    if (d) {
        struct dirent *de;
        while ((de = readdir(d)) != NULL) {
            unsigned long long s;
            if (!seq_of(de->d_name, &s)) continue;
            if (n == cap) {
                cap = cap ? cap * 2 : 64;
                unsigned long long *g = realloc(v, sizeof *v * (size_t)cap);
                if (!g) break;
                v = g;
            }
            v[n++] = s;
        }
        closedir(d);
    }
#endif
    // insertion sort: the directory holds at most --receipts-keep + 1 once
    // retention runs, and ordering is all that is asked
    for (int i = 1; i < n; i++) {
        unsigned long long x = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > x) { v[j + 1] = v[j]; j--; }
        v[j + 1] = x;
    }
    *out = v;
    return n;
}

static void path_of(char *out, size_t cap, unsigned long long seq) {
    snprintf(out, cap, "%s/" RC_PREFIX "%010llu" RC_SUFFIX, RC.dir, seq);
}

bool receipts_configure(const receipts_cfg *c) {
    pthread_mutex_lock(&RC.mu);
    RC.on = false;
    bool ok = false;
    if (!c || !c->dir || !*c->dir || strlen(c->dir) >= sizeof RC.dir) {
        fprintf(stderr, "error: --receipts needs a directory\n");
        goto out;
    }
    snprintf(RC.dir, sizeof RC.dir, "%s", c->dir);
    size_t dn = strlen(RC.dir);
    while (dn > 1 && (RC.dir[dn - 1] == '/' || RC.dir[dn - 1] == '\\'))
        RC.dir[--dn] = 0;
    if (rc_mkdir(RC.dir) != 0 && errno != EEXIST) {
        fprintf(stderr, "error: --receipts: cannot create %s\n", RC.dir);
        goto out;
    }
    RC.keep = c->keep > 0 ? c->keep : 0;
    RC.signing = c->sign_key && *c->sign_key;
    if (RC.signing) {
        signkey k;
        if (!signkey_load(c->sign_key, &k)) {
            fprintf(stderr, "error: --receipts: cannot load signing key %s\n",
                    c->sign_key);
            goto out;
        }
        memset(&k, 0, sizeof k);
        snprintf(RC.sign_key, sizeof RC.sign_key, "%s", c->sign_key);
    }
    // continue the chain from the newest record already there
    unsigned long long *seqs = NULL;
    int n = list_seqs(&seqs);
    snprintf(RC.head, sizeof RC.head, "%064d", 0);
    RC.next_seq = 1;
    if (n > 0) {
        char p[1100];
        path_of(p, sizeof p, seqs[n - 1]);
        if (!record_chain_hash(p, RC.head)) {
            fprintf(stderr, "error: --receipts: the newest record %s has no "
                    "readable chain hash; continuing would start a second "
                    "chain beside it. Move it away or use another "
                    "directory.\n", p);
            free(seqs);
            goto out;
        }
        RC.next_seq = seqs[n - 1] + 1;
    }
    free(seqs);
    RC.on = true;
    ok = true;
    fprintf(stderr, "receipts: %s (%s, %s, chain %s)\n", RC.dir,
            RC.signing ? "signed" : "unsigned",
            RC.keep ? "bounded" : "keeping every record",
            n > 0 ? "continued" : "new");
out:
    pthread_mutex_unlock(&RC.mu);
    return ok;
}

bool receipts_enabled(void) {
    pthread_mutex_lock(&RC.mu);
    bool on = RC.on;
    pthread_mutex_unlock(&RC.mu);
    return on;
}

bool receipts_write(transcript_info *ti, char name_out[64], char chain_out[65]) {
    pthread_mutex_lock(&RC.mu);
    if (!RC.on) { pthread_mutex_unlock(&RC.mu); return false; }
    // one lock around the whole write: the chain order IS the write order
    char path[1100], chain[65];
    unsigned long long seq = RC.next_seq;
    path_of(path, sizeof path, seq);
    ti->out_path = path;
    ti->prev_hash = RC.head;
    ti->sign_key_path = RC.signing ? RC.sign_key : NULL;
    ti->chain_out = chain;
    bool ok = transcript_write(ti);
    if (ok) {
        memcpy(RC.head, chain, sizeof chain);
        memcpy(chain_out, chain, sizeof chain);
        RC.next_seq = seq + 1;
        snprintf(name_out, 64, RC_PREFIX "%010llu" RC_SUFFIX, seq);
        if (RC.keep > 0) {
            unsigned long long *seqs = NULL;
            int n = list_seqs(&seqs);
            for (int i = 0; i + RC.keep < n; i++) {
                char old[1100];
                path_of(old, sizeof old, seqs[i]);
                remove(old);
            }
            free(seqs);
        }
    }
    ti->out_path = NULL;
    ti->prev_hash = NULL;
    ti->chain_out = NULL;
    pthread_mutex_unlock(&RC.mu);
    return ok;
}

void receipts_reset(void) {
    pthread_mutex_lock(&RC.mu);
    RC.on = false;
    pthread_mutex_unlock(&RC.mu);
}
