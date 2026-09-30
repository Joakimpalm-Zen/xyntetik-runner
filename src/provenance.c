// Identity of the running server (R10.2.1). See provenance.h for what each
// field vouches for.
#include "provenance.h"
#include "build_arch.h"
#include "compat.h"
#include "envelope.h"
#include "model.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#define PROV_OS "windows"
#elif defined(__APPLE__)
#define PROV_OS "macos"
#else
#define PROV_OS "linux"
#endif

typedef struct {
    uint64_t size, ino;
    int64_t  mtime, ctime;
} file_id;

// Everything below is guarded by mu. `gen` counts loads, so a digest that
// finishes after its model was swapped out is dropped instead of being filed
// under the model that replaced it.
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  hashed = PTHREAD_COND_INITIALIZER;   // hstate left H_HASHING
static char     binary_sha[65];
static bool     binary_ok;
static bool     resident;
static uint64_t gen;
static char    *model_path, *adapter_path;
static float    adapter_scale;
static char     adapter_sha[65];
static bool     adapter_ok;
static char     sig_json[512];      // oms_result_json, "" = no bundle
static int      env_state;
static char     env_detail[256];
static char     loaded_utc[32];
static file_id  load_id;
static bool     load_id_ok;
static enum { H_HASHING, H_DONE, H_FAILED } hstate;
static char     model_sha[65];

static bool identify(const char *path, file_id *out) {
    return model_file_identity(path, NULL, &out->size, &out->ino,
                               &out->mtime, &out->ctime);
}

static bool same_file(const file_id *a, const file_id *b) {
    return a->size == b->size && a->ino == b->ino &&
           a->mtime == b->mtime && a->ctime == b->ctime;
}

void provenance_init(void) {
    char *exe = plat_executable_path();
    char sha[65];
    bool ok = exe && envelope_file_sha256(exe, sha);
    free(exe);
    pthread_mutex_lock(&mu);
    binary_ok = ok;
    if (ok) memcpy(binary_sha, sha, sizeof sha);
    pthread_mutex_unlock(&mu);
}

typedef struct {
    char    *path;
    uint64_t gen;
} hash_job;

// The digest is taken OUTSIDE the lock (it reads the whole file) and filed
// under it only if no other load has happened meanwhile. The file's identity
// is checked on both sides of the read: bytes that changed while being hashed
// produce a digest of no file that ever existed, so that digest is dropped.
static void *hash_model(void *arg) {
    hash_job *j = arg;
    file_id before, after;
    char sha[65];
    bool ok = identify(j->path, &before) &&
              envelope_file_sha256(j->path, sha) &&
              identify(j->path, &after) && same_file(&before, &after);
    pthread_mutex_lock(&mu);
    if (resident && gen == j->gen) {
        // a file that was already replaced before the read began is caught by
        // the render-time check; here it only matters that the bytes hashed
        // are the bytes identified at load
        ok = ok && load_id_ok && same_file(&before, &load_id);
        hstate = ok ? H_DONE : H_FAILED;
        if (ok) memcpy(model_sha, sha, sizeof sha);
    }
    pthread_cond_broadcast(&hashed);
    pthread_mutex_unlock(&mu);
    free(j->path);
    free(j);
    return NULL;
}

void provenance_note_load(const provenance_load *l) {
    char *mp = strdup(l->model_path);
    char *ap = l->adapter_path ? strdup(l->adapter_path) : NULL;
    // An adapter is small (megabytes); hash it here, so the record never
    // names an adapter it has no digest for.
    char asha[65] = "";
    bool aok = ap && envelope_file_sha256(ap, asha);
    file_id id;
    bool id_ok = identify(l->model_path, &id);
    char utc[32] = "";
    time_t now = time(NULL);
    struct tm g;
#ifdef _WIN32
    gmtime_s(&g, &now);
#else
    gmtime_r(&now, &g);
#endif
    strftime(utc, sizeof utc, "%Y-%m-%dT%H:%M:%SZ", &g);

    pthread_mutex_lock(&mu);
    free(model_path);
    free(adapter_path);
    model_path = mp;
    adapter_path = ap;
    adapter_scale = l->adapter_scale;
    adapter_ok = aok;
    memcpy(adapter_sha, asha, sizeof asha);
    sig_json[0] = 0;
    if (l->signature && l->signature->status[0])
        oms_result_json(l->signature, sig_json, sizeof sig_json);
    env_state = l->envelope_state;
    snprintf(env_detail, sizeof env_detail, "%s",
             l->envelope_detail ? l->envelope_detail : "");
    memcpy(loaded_utc, utc, sizeof utc);
    load_id = id;
    load_id_ok = id_ok;
    model_sha[0] = 0;
    hstate = H_HASHING;
    resident = mp != NULL;
    uint64_t my_gen = ++gen;
    pthread_mutex_unlock(&mu);

    hash_job *j = malloc(sizeof *j);
    char *jp = mp ? strdup(mp) : NULL;
    pthread_t th;
    pthread_attr_t at;
    bool started = false;
    if (j && jp && pthread_attr_init(&at) == 0) {
        j->path = jp;
        j->gen = my_gen;
        pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
        started = pthread_create(&th, &at, hash_model, j) == 0;
        pthread_attr_destroy(&at);
    }
    if (!started) {
        free(j);
        free(jp);
        pthread_mutex_lock(&mu);
        if (gen == my_gen) hstate = H_FAILED;
        pthread_cond_broadcast(&hashed);
        pthread_mutex_unlock(&mu);
    }
}

bool provenance_digests(char model[65], char binary[65]) {
    pthread_mutex_lock(&mu);
    uint64_t my_gen = gen;
    while (resident && gen == my_gen && hstate == H_HASHING)
        pthread_cond_wait(&hashed, &mu);
    file_id now;
    bool ok = resident && gen == my_gen && hstate == H_DONE && binary_ok &&
              load_id_ok && identify(model_path, &now) && same_file(&now, &load_id);
    if (ok) {
        memcpy(model, model_sha, 65);
        memcpy(binary, binary_sha, 65);
    }
    pthread_mutex_unlock(&mu);
    return ok;
}

void provenance_note_unload(void) {
    pthread_mutex_lock(&mu);
    resident = false;
    pthread_cond_broadcast(&hashed);
    gen++;   // an in-flight digest belongs to a model that is gone
    free(model_path);
    free(adapter_path);
    model_path = adapter_path = NULL;
    pthread_mutex_unlock(&mu);
}

static const char *envelope_state_name(int s) {
    switch (s) {
    case ENV_CERTIFIED:     return "certified";
    case ENV_OUTSIDE:       return "outside";
    case ENV_EXPERIMENTAL:  return "experimental";
    case ENV_INDETERMINATE: return "indeterminate";
    default:                return "unclassified";
    }
}

static void put_str(sbuf *b, const char *s) {
    sb_lit(b, "\"");
    sb_esc(b, s, strlen(s));
    sb_lit(b, "\"");
}

void provenance_render(sbuf *b, const char *model_id) {
    pthread_mutex_lock(&mu);
    sb_lit(b, "\"build\":{\"binary_sha256\":");
    if (binary_ok) put_str(b, binary_sha);
    else           sb_lit(b, "null");
    sb_lit(b, ",\"compiler\":");
    put_str(b, __VERSION__);
    sb_lit(b, ",\"os\":\"" PROV_OS "\",\"arch\":\"" RUNNER_BUILD_ARCH "\"");
#ifdef RUNNER_T3_BUILD
    sb_lit(b, ",\"flavor\":\"t3\"");
#endif
    sb_lit(b, "},\"model\":");
    if (!resident) {
        sb_lit(b, "null,\"adapter\":null");
        pthread_mutex_unlock(&mu);
        return;
    }
    // Re-identify on every read: a file replaced after its digest was taken
    // must not keep that digest on display.
    file_id now;
    bool changed = !load_id_ok || !identify(model_path, &now) ||
                   !same_file(&now, &load_id);
    const char *st = changed           ? "changed_since_load"
                   : hstate == H_DONE  ? "done"
                   : hstate == H_HASHING ? "hashing" : "unreadable";
    sb_lit(b, "{\"id\":");
    put_str(b, model_id ? model_id : "");
    sb_lit(b, ",\"path\":");
    put_str(b, model_path);
    sb_lit(b, ",\"sha256\":");
    if (!changed && hstate == H_DONE) put_str(b, model_sha);
    else                              sb_lit(b, "null");
    sb_fmt(b, ",\"sha256_state\":\"%s\"", st);
    if (load_id_ok) sb_fmt(b, ",\"size\":%llu", (unsigned long long)load_id.size);
    else            sb_lit(b, ",\"size\":null");
    sb_fmt(b, ",\"loaded_utc\":\"%s\",\"signature\":", loaded_utc);
    if (sig_json[0]) sb_put(b, sig_json, strlen(sig_json));
    else             sb_lit(b, "null");
    sb_fmt(b, ",\"envelope\":{\"state\":\"%s\",\"detail\":",
           envelope_state_name(env_state));
    put_str(b, env_detail);
    sb_lit(b, "}},\"adapter\":");
    if (!adapter_path) {
        sb_lit(b, "null");
    } else {
        sb_lit(b, "{\"path\":");
        put_str(b, adapter_path);
        sb_lit(b, ",\"sha256\":");
        if (adapter_ok) put_str(b, adapter_sha);
        else            sb_lit(b, "null");
        sb_fmt(b, ",\"scale\":%g}", (double)adapter_scale);
    }
    pthread_mutex_unlock(&mu);
}
