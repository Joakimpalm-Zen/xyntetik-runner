// Instance discovery registry — see instances.h for the contract.
#include "instances.h"
#include "compat.h"
#include "json.h"
#include "runner.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#include <io.h>
#define getpid _getpid
#include <process.h>
#else
#include <dirent.h>
#include <limits.h>
#include <signal.h>
#include <unistd.h>
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/proc.h>
#endif
#endif

// ------------------------------------------------------------------ paths

static bool path_child(char *out, size_t cap, const char *base,
                       char separator, const char *name) {
    size_t nb = strlen(base), nn = strlen(name);
    if (nb >= cap || nn >= cap - nb - 1) return false;
    memcpy(out, base, nb);
    out[nb] = separator;
    memcpy(out + nb + 1, name, nn + 1);
    return true;
}

static bool mkdir_one(const char *p) {
#ifdef _WIN32
    return _mkdir(p) == 0 || errno == EEXIST || GetLastError() == ERROR_ALREADY_EXISTS;
#else
    return mkdir(p, 0755) == 0 || errno == EEXIST;
#endif
}

static bool path_exists(const char *p) {
#ifdef _WIN32
    return GetFileAttributesA(p) != INVALID_FILE_ATTRIBUTES;
#else
    struct stat st;
    return stat(p, &st) == 0;
#endif
}

// One-time migration of the pre-rename state dir (Gridcore -> Xyntetik):
// move the whole old tree when the new one does not exist yet, so config.json,
// managed.log and the instance registry survive the rename instead of
// stranding a live install (the tray-reports-missing-model failure of 1aed406).
static void migrate_old_tree(const char *base, char sep, const char *oldname,
                             const char *newbase) {
    char old[1024];
    if (!path_child(old, sizeof old, base, sep, oldname)) return;
    if (!path_exists(newbase) && path_exists(old)) rename(old, newbase);
}

const char *instances_dir(void) {
    static char dir[1024];
    static bool made = false;
    if (made) return dir[0] ? dir : NULL;
    made = true;
#ifdef _WIN32
    const char *base = getenv("APPDATA");
    if (!base || !*base) { dir[0] = 0; return NULL; }
    char a[1024], b[1024], c[1024];
    if (!path_child(a, sizeof a, base, '\\', "xyntetik") ||
        !path_child(b, sizeof b, a, '\\', "runner") ||
        !path_child(c, sizeof c, b, '\\', "instances")) {
        dir[0] = 0;
        return NULL;
    }
    migrate_old_tree(base, '\\', "gridcore", a);
    if (!mkdir_one(a) || !mkdir_one(b) || !mkdir_one(c)) { dir[0] = 0; return NULL; }
    memcpy(dir, c, strlen(c) + 1);
#else
    const char *base = getenv("HOME");
    if (!base || !*base) { dir[0] = 0; return NULL; }
    char a[1024], b[1024], c[1024];
    if (!path_child(a, sizeof a, base, '/', ".xyntetik") ||
        !path_child(b, sizeof b, a, '/', "runner") ||
        !path_child(c, sizeof c, b, '/', "instances")) {
        dir[0] = 0;
        return NULL;
    }
    migrate_old_tree(base, '/', ".gridcore", a);
    if (!mkdir_one(a) || !mkdir_one(b) || !mkdir_one(c)) { dir[0] = 0; return NULL; }
    memcpy(dir, c, strlen(c) + 1);
#endif
    return dir;
}

bool instance_pid_alive(long pid) {
    if (pid <= 0) return false;
#ifdef _WIN32
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
    if (!h) return false;
    DWORD code = 0;
    bool alive = GetExitCodeProcess(h, &code) && code == STILL_ACTIVE;
    CloseHandle(h);
    return alive;
#else
    if (kill((pid_t)pid, 0) != 0 && errno != EPERM) return false;
    // a zombie still answers kill(pid, 0) but is dead for every purpose a
    // reader has: it serves nothing and no signal can stop it further.
    // Counting it alive turns an unreaped child into a permanent ghost row
    // that Stop can never clear.
#ifdef __APPLE__
    struct kinfo_proc kp;
    size_t len = sizeof kp;
    int mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_PID, (int)pid };
    if (sysctl(mib, 4, &kp, &len, NULL, 0) == 0 && len >= sizeof kp &&
        kp.kp_proc.p_stat == SZOMB)
        return false;
#elif defined(__linux__)
    char sp[64], buf[512];
    snprintf(sp, sizeof sp, "/proc/%ld/stat", pid);
    FILE *f = fopen(sp, "rb");
    if (f) {
        size_t rn = fread(buf, 1, sizeof buf - 1, f);
        fclose(f);
        buf[rn] = 0;
        // state is the field after the parenthesized comm
        char *rp = strrchr(buf, ')');
        if (rp && rp[1] == ' ' && (rp[2] == 'Z' || rp[2] == 'X'))
            return false;
    }
#endif
    return true;
#endif
}

// ------------------------------------------------------------- own record

// Remembered so set_port / unregister can rewrite or remove without the
// caller re-passing everything.
static struct {
    bool  registered;
    char  mode[16];
    int   port;
    uint64_t procstart;
    sbuf  models_json;   // pre-serialized models array content
} g_self;

static void self_path(char *out, size_t cap) {
    const char *d = instances_dir();
    if (!d) { out[0] = 0; return; }
#ifdef _WIN32
    snprintf(out, cap, "%s\\%ld.json", d, (long)getpid());
#else
    snprintf(out, cap, "%s/%ld.json", d, (long)getpid());
#endif
}

static bool write_self(void) {
    char path[1200], tmp[1240];
    self_path(path, sizeof path);
    if (!path[0]) return false;
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return false;
    fprintf(f,
        "{\"pid\": %ld, \"procstart\": %llu, \"started\": %lld, "
        "\"mode\": \"%s\", \"port\": %d,\n"
        " \"version\": \"%s\",\n \"models\": [%s]}\n",
        (long)getpid(), (unsigned long long)g_self.procstart,
        (long long)time(NULL), g_self.mode, g_self.port, RUNNER_VERSION,
        g_self.models_json.s ? g_self.models_json.s : "");
    bool ok = fclose(f) == 0;
#ifdef _WIN32
    // rename() cannot replace an existing file on Windows
    ok = ok && MoveFileExA(tmp, path, MOVEFILE_REPLACE_EXISTING);
#else
    ok = ok && rename(tmp, path) == 0;
#endif
    if (!ok) remove(tmp);
    return ok;
}

// A crashed process leaves its record, and one killed between the write and
// the rename leaves `<pid>.json.tmp`. Only a reader swept them, and the only
// reader in this binary is the tray: a machine that never ran it kept every
// record of every dead process. Each registering process now sweeps first.
// The list itself removes dead records; the temporaries are removed here, and
// only when the pid in their name is not alive, so a live writer's is kept.
static void sweep_stale(void) {
    int n = 0;
    instances_list_free(instances_list(&n), n);
    const char *d = instances_dir();
    if (!d) return;
#ifdef _WIN32
    char pat[1100];
    snprintf(pat, sizeof pat, "%s\\*.json.tmp", d);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        const char *name = fd.cFileName;
        char path[1300];
        snprintf(path, sizeof path, "%s\\%s", d, name);
#else
    DIR *dp = opendir(d);
    if (!dp) return;
    struct dirent *de;
    while ((de = readdir(dp)) != NULL) {
        const char *name = de->d_name;
        size_t l = strlen(name);
        if (l < 10 || strcmp(name + l - 9, ".json.tmp") != 0) continue;
        char path[1300];
        snprintf(path, sizeof path, "%s/%s", d, name);
#endif
        char *end;
        long pid = strtol(name, &end, 10);
        if (end == name || strcmp(end, ".json.tmp") != 0) continue;
        if (pid > 0 && !instance_pid_alive(pid)) remove(path);
#ifdef _WIN32
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    }
    closedir(dp);
#endif
}

bool instances_register(const char *mode, int port,
                        const char *const *model_names,
                        const char *const *model_paths, int n_models) {
    sweep_stale();
    snprintf(g_self.mode, sizeof g_self.mode, "%s", mode ? mode : "cli");
    g_self.port = port;
    if (!plat_pid_start_time((long)getpid(), &g_self.procstart))
        g_self.procstart = 0;
    free(g_self.models_json.s);
    memset(&g_self.models_json, 0, sizeof g_self.models_json);
    for (int i = 0; i < n_models; i++) {
        if (i) sb_lit(&g_self.models_json, ", ");
        sb_lit(&g_self.models_json, "{\"name\": \"");
        sb_esc(&g_self.models_json, model_names[i], strlen(model_names[i]));
        sb_lit(&g_self.models_json, "\", \"path\": \"");
        // absolute path: a controller in another cwd must be able to
        // restart or inspect the file the instance was launched with.
        // realpath's output buffer MUST be PATH_MAX bytes: glibc's
        // fortified build aborts on a smaller one regardless of how short
        // the resolved path is (macOS PATH_MAX is 1024 so a short buffer
        // passes there — every Linux CI job catches it)
#ifdef _WIN32
        char abs[_MAX_PATH];
        if (!_fullpath(abs, model_paths[i], sizeof abs))
            snprintf(abs, sizeof abs, "%s", model_paths[i]);
#else
        char abs[PATH_MAX];
        if (!realpath(model_paths[i], abs))
            snprintf(abs, sizeof abs, "%s", model_paths[i]);
#endif
        sb_esc(&g_self.models_json, abs, strlen(abs));
        sb_lit(&g_self.models_json, "\"}");
    }
    if (g_self.models_json.failed) return false;
    g_self.registered = write_self();
    return g_self.registered;
}

bool instances_set_port(int port) {
    if (!g_self.registered) return false;
    g_self.port = port;
    return write_self();
}

void instances_unregister(void) {
    if (!g_self.registered) return;
    g_self.registered = false;
    char path[1200];
    self_path(path, sizeof path);
    if (path[0]) remove(path);
}

// ------------------------------------------------------------------- list

static void rec_free_members(instance_rec *r);

// Grow to fit one more record. Returns NULL only when the array could not
// grow, and then WITHOUT disturbing the caller's array: `arr = realloc(arr, n)`
// loses the original pointer, and the old code both leaked it and handed the
// caller a NULL that instances_list still reported a count for.
static instance_rec *push_rec(instance_rec *arr, int *n, int *cap) {
    if (*n == *cap) {
        int want = *cap ? *cap * 2 : 8;
        instance_rec *grown = realloc(arr, sizeof(instance_rec) * (size_t)want);
        if (!grown) return NULL;
        arr = grown;
        *cap = want;
    }
    memset(&arr[*n], 0, sizeof(instance_rec));
    return arr;
}

static bool parse_rec(const char *path, instance_rec *r) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    char buf[16384];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    jv *v = json_parse(buf, n);
    if (!v) return false;
    r->pid     = (long)jv_num(jv_get(v, "pid"), 0);
    r->started = (long long)jv_num(jv_get(v, "started"), 0);
    r->procstart = (uint64_t)jv_num(jv_get(v, "procstart"), 0);
    r->port    = (int)jv_num(jv_get(v, "port"), 0);
    snprintf(r->mode, sizeof r->mode, "%s", jv_str(jv_get(v, "mode"), "?"));
    snprintf(r->version, sizeof r->version, "%s", jv_str(jv_get(v, "version"), "?"));
    jv *ms = jv_get(v, "models");
    if (ms && ms->type == J_ARR && ms->n > 0) {
        r->model_names = calloc((size_t)ms->n, sizeof(char *));
        r->model_paths = calloc((size_t)ms->n, sizeof(char *));
        if (r->model_names && r->model_paths) {
            // All or nothing. Publishing n_models while an entry is NULL
            // hands every reader a string that is not there -- the arrays are
            // consumed as `%s` by the listing and tray surfaces, and the free
            // path below tolerates NULL but no reader does. On a failed
            // strdup the record reports no models rather than a broken one.
            bool ok = true;
            for (int i = 0; i < ms->n && ok; i++) {
                r->model_names[i] = strdup(jv_str(jv_get(ms->items[i], "name"), "?"));
                r->model_paths[i] = strdup(jv_str(jv_get(ms->items[i], "path"), ""));
                ok = r->model_names[i] && r->model_paths[i];
            }
            if (ok) {
                r->n_models = ms->n;
            } else {
                for (int i = 0; i < ms->n; i++) {
                    free(r->model_names[i]);
                    free(r->model_paths[i]);
                }
                free(r->model_names); free(r->model_paths);
                r->model_names = r->model_paths = NULL;
            }
        }
    }
    jv_free(v);
    return r->pid > 0;
}

static bool record_owner_alive(const instance_rec *r) {
    if (!instance_pid_alive(r->pid)) return false;
    if (r->procstart) {
        uint64_t live_start;
        if (plat_pid_start_time(r->pid, &live_start) &&
            live_start != r->procstart)
            return false;
    }
    return true;
}

instance_rec *instances_list(int *n_out) {
    *n_out = 0;
    const char *d = instances_dir();
    if (!d) return NULL;
    instance_rec *arr = NULL;
    int n = 0, cap = 0;
#ifdef _WIN32
    char pat[1100];
    snprintf(pat, sizeof pat, "%s\\*.json", d);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return NULL;
    do {
        char path[1300];
        snprintf(path, sizeof path, "%s\\%s", d, fd.cFileName);
#else
    DIR *dp = opendir(d);
    if (!dp) return NULL;
    struct dirent *de;
    while ((de = readdir(dp)) != NULL) {
        size_t l = strlen(de->d_name);
        if (l < 6 || strcmp(de->d_name + l - 5, ".json") != 0) continue;
        char path[1300];
        snprintf(path, sizeof path, "%s/%s", d, de->d_name);
#endif
        instance_rec r = {0};
        bool ok = parse_rec(path, &r);
        if (ok && record_owner_alive(&r)) {
            // Out of memory stops the enumeration; it does not discard what has
            // already been read. A short list is a true answer, and the record
            // files are left alone -- this is not a reason to sweep a live
            // process away.
            instance_rec *grown = push_rec(arr, &n, &cap);
            if (!grown) { rec_free_members(&r); break; }
            arr = grown;
            arr[n++] = r;
        } else {
            // stale (crash leftover) or unreadable: sweep it
            rec_free_members(&r);
            remove(path);
        }
#ifdef _WIN32
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    }
    closedir(dp);
#endif
    *n_out = n;
    return arr;
}

static void rec_free_members(instance_rec *r) {
    for (int m = 0; m < r->n_models; m++) {
        free(r->model_names[m]);
        free(r->model_paths[m]);
    }
    free(r->model_names);
    free(r->model_paths);
    r->model_names = r->model_paths = NULL;
    r->n_models = 0;
}

void instances_list_free(instance_rec *recs, int n) {
    for (int i = 0; i < n; i++) rec_free_members(&recs[i]);
    free(recs);
}

// ------------------------------------------------------------ startup lease
//
// The Runner package's StartupLease (python/src/xyntetik_runner/lease.py), in
// C for the tray (R4.12.24). Every rule below is that class's rule; the
// comments there carry the reasons. A claim is a directory holding owner.json,
// renamed onto the lease path in one step, so exactly one claimant wins; a
// held lease names a live owner whose start identity still matches.

#define LEASE_RECORD "owner.json"
#define LEASE_TTL_DEFAULT 900.0

typedef struct {
    long pid;                  // 0 when absent or not a positive integer
    bool has_start, has_token;
    char start[96];
    char token[96];
    char source[1200];         // the file read, whose mtime is the silence clock
} lease_rec;

static char lease_sep(void) {
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

static bool lease_is_dir(const char *p) {
    struct stat st;
    return stat(p, &st) == 0 && (st.st_mode & S_IFMT) == S_IFDIR;
}

// A claim or a moved-aside record holds owner.json and nothing else.
static void lease_remove_tree(const char *p) {
    if (!lease_is_dir(p)) {
        remove(p);
        return;
    }
    char f[1300];
    if (path_child(f, sizeof f, p, lease_sep(), LEASE_RECORD)) remove(f);
#ifdef _WIN32
    _rmdir(p);
#else
    rmdir(p);
#endif
}

// A sibling of the lease path: <dir>/.<name>.<tag>.<kind>
static bool lease_sibling(char *out, size_t cap, const char *path,
                          const char *tag, const char *kind) {
    const char *slash = strrchr(path, '/');
#ifdef _WIN32
    const char *bs = strrchr(path, '\\');
    if (!slash || (bs && bs > slash)) slash = bs;
#endif
    size_t dl = slash ? (size_t)(slash - path) + 1 : 0;
    const char *name = slash ? slash + 1 : path;
    int n = snprintf(out, cap, "%.*s.%s.%s.%s", (int)dl, path, name, tag, kind);
    return n > 0 && (size_t)n < cap;
}

static uint64_t lease_splitmix(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// A token only has to differ from every other claimant's, as the Rust port's
// keyed hash does: /dev/urandom where there is one, else the pid, a fine
// clock, a per-process sequence and an address, mixed. (Not rand_s:
// test_instances_oom.c compiles this file after <stdlib.h>, too late for the
// _CRT_RAND_S its declaration needs under MinGW.)
static void lease_hex_token(char out[33]) {
    unsigned char b[16];
    bool ok = false;
#ifndef _WIN32
    FILE *f = fopen("/dev/urandom", "rb");
    if (f) {
        ok = fread(b, 1, sizeof b, f) == sizeof b;
        fclose(f);
    }
#endif
    if (!ok) {
        static unsigned seq = 0;
        uint64_t clk;
#ifdef _WIN32
        LARGE_INTEGER q;
        QueryPerformanceCounter(&q);
        clk = (uint64_t)q.QuadPart;
#else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        clk = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
        uint64_t s = clk ^ ((uint64_t)plat_pid_self() << 32) ^
                     ((uint64_t)time(NULL) << 12) ^ (uint64_t)(uintptr_t)out ^
                     ((uint64_t)++seq << 48);
        uint64_t h0 = lease_splitmix(&s), h1 = lease_splitmix(&s);
        memcpy(b, &h0, 8);
        memcpy(b + 8, &h1, 8);
    }
    static const char hx[] = "0123456789abcdef";
    for (int i = 0; i < 16; i++) {
        out[2 * i] = hx[b[i] >> 4];
        out[2 * i + 1] = hx[b[i] & 15];
    }
    out[32] = 0;
}

bool runner_lease_start_identity(long pid, char *out, size_t cap) {
    if (pid <= 0 || cap == 0) return false;
#ifdef _WIN32
    // lease.py: str(creation.value), the creation FILETIME as one number
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
    if (!h) return false;
    FILETIME create, exit_t, kern, user;
    bool ok = GetProcessTimes(h, &create, &exit_t, &kern, &user);
    CloseHandle(h);
    if (!ok) return false;
    unsigned long long v =
        ((unsigned long long)create.dwHighDateTime << 32) | create.dwLowDateTime;
    int n = snprintf(out, cap, "%llu", v);
    return n > 0 && (size_t)n < cap;
#elif defined(__linux__)
    // lease.py: field 22 of /proc/<pid>/stat, verbatim (clock ticks since
    // boot); comm may hold spaces and parentheses, so count from the last ')'
    char sp[64], buf[2048];
    snprintf(sp, sizeof sp, "/proc/%ld/stat", pid);
    FILE *f = fopen(sp, "rb");
    if (!f) return false;
    size_t rn = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[rn] = 0;
    char *p = strrchr(buf, ')');
    if (!p || p[1] != ' ') return false;
    p += 2;
    for (int field = 0; field < 19; field++) {
        while (*p && *p != ' ') p++;
        while (*p == ' ') p++;
        if (!*p) return false;
    }
    size_t l = strcspn(p, " \n");
    if (l == 0 || l >= cap) return false;
    memcpy(out, p, l);
    out[l] = 0;
    return true;
#elif defined(__APPLE__)
    // lease.py and the Suite ask `ps -o lstart=` with LC_ALL=C and TZ=UTC,
    // which formats the kernel's start second with strftime("%c") in the C
    // locale: "%a %b %e %H:%M:%S %Y" in UTC. The same rendering from the same
    // second, without a subprocess (the tray never launches through a shell);
    // tests/test_lease_interop.py holds it equal to ps's own output.
    struct kinfo_proc kp;
    size_t len = sizeof kp;
    int mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_PID, (int)pid };
    if (sysctl(mib, 4, &kp, &len, NULL, 0) != 0 || len < sizeof kp) return false;
    time_t s = (time_t)kp.kp_proc.p_starttime.tv_sec;
    struct tm tm;
    if (!gmtime_r(&s, &tm)) return false;
    static const char *const day[] = { "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat" };
    static const char *const mon[] = { "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };
    int n = snprintf(out, cap, "%s %s %2d %02d:%02d:%02d %d", day[tm.tm_wday],
                     mon[tm.tm_mon], tm.tm_mday, tm.tm_hour, tm.tm_min,
                     tm.tm_sec, tm.tm_year + 1900);
    return n > 0 && (size_t)n < cap;
#else
    (void)out;
    return false;   // unverifiable: the record ages out after the TTL
#endif
}

static bool lease_read(const char *path, lease_rec *r) {
    memset(r, 0, sizeof *r);
    if (lease_is_dir(path)) {
        if (!path_child(r->source, sizeof r->source, path, lease_sep(), LEASE_RECORD))
            return false;
    } else {
        snprintf(r->source, sizeof r->source, "%s", path);
    }
    FILE *f = fopen(r->source, "rb");
    if (!f) return false;
    char buf[8192];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    jv *v = json_parse(buf, n);
    if (!v) return false;
    if (v->type != J_OBJ) { jv_free(v); return false; }
    // owner_pid, or clu_pid from the former Clu-local lease; an integer or a
    // string of one, as lease.py's int() accepts
    jv *pv = jv_get(v, "owner_pid");
    if (!pv || pv->type == J_NULL) pv = jv_get(v, "clu_pid");
    if (pv && pv->type == J_NUM && pv->num == floor(pv->num) && pv->num > 0 &&
        pv->num < 9.0e15) {
        r->pid = (long)pv->num;
    } else if (pv && pv->type == J_STR) {
        long long q;
        if (parse_i64(jv_str(pv, ""), 1, LONG_MAX, &q)) r->pid = (long)q;
    }
    jv *sv = jv_get(v, "owner_start");
    if (sv && sv->type == J_STR) {
        snprintf(r->start, sizeof r->start, "%s", jv_str(sv, ""));
        r->has_start = true;
    } else if (sv && sv->type == J_NUM && sv->num == floor(sv->num)) {
        snprintf(r->start, sizeof r->start, "%.0f", sv->num);
        r->has_start = true;
    }
    jv *tv = jv_get(v, "token");
    if (tv && tv->type == J_STR) {
        snprintf(r->token, sizeof r->token, "%s", jv_str(tv, ""));
        r->has_token = true;
    }
    jv_free(v);
    return true;
}

static bool lease_same_token(const lease_rec *a, const lease_rec *b) {
    if (a->has_token != b->has_token) return false;
    return !a->has_token || strcmp(a->token, b->token) == 0;
}

// Ownership that can be neither proven nor disproven is honoured while the
// record is younger than the window: a crashed owner ages out, a live one
// is not stolen from before then.
static bool lease_unverified_fresh(const lease_rec *r) {
    struct stat st;
    if (!r->source[0] || stat(r->source, &st) != 0) return false;
    double ttl = env_f64("RUNNER_UNVERIFIED_LEASE_TTL", 0, 1e12, LEASE_TTL_DEFAULT);
    return difftime(time(NULL), st.st_mtime) < ttl;
}

static bool lease_still_owned(const lease_rec *r) {
    if (r->pid <= 0 || !plat_pid_alive(r->pid)) return false;
    if (!r->has_start) return lease_unverified_fresh(r);
    char live[96];
    if (!runner_lease_start_identity(r->pid, live, sizeof live))
        return lease_unverified_fresh(r);
    return strcmp(live, r->start) == 0;   // differ: the pid was reused
}

static bool lease_write_claim(const char *claim, const char *token) {
    char f[1300];
    if (!path_child(f, sizeof f, claim, lease_sep(), LEASE_RECORD)) return false;
    char start[96], esc[96 * 6 + 2];
    long pid = plat_pid_self();
    bool has_start = runner_lease_start_identity(pid, start, sizeof start);
    if (has_start) json_escape(start, strlen(start), esc, sizeof esc);
    FILE *o = fopen(f, "wb");
    if (!o) return false;
    // `created` is informational (no reader judges by it); whole seconds do
    int w = fprintf(o, "{\"owner_pid\": %ld, \"owner_start\": %s%s%s, "
                       "\"token\": \"%s\", \"created\": %lld.0}",
                    pid, has_start ? "\"" : "", has_start ? esc : "null",
                    has_start ? "\"" : "", token, (long long)time(NULL));
    return fclose(o) == 0 && w > 0;
}

bool runner_lease_path(int port, char *out, size_t cap) {
    const char *inst = instances_dir();
    if (!inst || port <= 0) return false;
    // the state root is the registry's parent: <root>/instances
    char root[1024];
    snprintf(root, sizeof root, "%s", inst);
    char *cut = strrchr(root, lease_sep());
    if (!cut) return false;
    *cut = 0;
    char dir[1100], name[32];
    if (!path_child(dir, sizeof dir, root, lease_sep(), "leases") || !mkdir_one(dir))
        return false;
    snprintf(name, sizeof name, "runner-%d.pid", port);
    return path_child(out, cap, dir, lease_sep(), name);
}

bool runner_lease_acquire(runner_lease *l, const char *path) {
    memset(l, 0, sizeof *l);
    int n = snprintf(l->path, sizeof l->path, "%s", path);
    if (n <= 0 || (size_t)n >= sizeof l->path) return false;
    lease_hex_token(l->token);
    char claim[1300];
    if (!lease_sibling(claim, sizeof claim, path, l->token, "claim")) return false;
    for (int attempt = 0; attempt < 8; attempt++) {
        lease_remove_tree(claim);
        if (!mkdir_one(claim) || !lease_write_claim(claim, l->token)) {
            lease_remove_tree(claim);
            return false;
        }
        if (rename(claim, path) == 0) {
            l->held = true;
            return true;
        }
        if (!path_exists(path)) {
            lease_remove_tree(claim);
            continue;
        }
        lease_rec rec;
        lease_read(path, &rec);
        if (lease_still_owned(&rec)) {
            lease_remove_tree(claim);
            return false;
        }
        char tag[33], stale[1300];
        lease_hex_token(tag);
        if (lease_sibling(stale, sizeof stale, path, tag, "stale") &&
            rename(path, stale) == 0) {
            // between judging the record stale and moving it, a rival may
            // have reclaimed: delete only the exact record judged
            lease_rec moved;
            lease_read(stale, &moved);
            if (!lease_same_token(&moved, &rec)) {
                rename(stale, path);   // theirs; a third claim may have landed
                lease_remove_tree(claim);
                continue;
            }
            lease_remove_tree(stale);
        }
        lease_remove_tree(claim);
    }
    return false;
}

void runner_lease_release(runner_lease *l) {
    if (!l->held) return;
    l->held = false;
    lease_rec rec;
    if (!lease_read(l->path, &rec) || !rec.has_token || strcmp(rec.token, l->token) != 0)
        return;
    char tag[33], stale[1300];
    lease_hex_token(tag);
    if (!lease_sibling(stale, sizeof stale, l->path, tag, "stale") ||
        rename(l->path, stale) != 0)
        return;
    lease_rec moved;
    lease_read(stale, &moved);
    if (!moved.has_token || strcmp(moved.token, l->token) != 0) {
        // the path changed after the read above: the record moved belongs to
        // another owner, so put it back unless a claim already took its place
        rename(stale, l->path);
        return;
    }
    lease_remove_tree(stale);
}

long runner_lease_holder(const char *path) {
    lease_rec r;
    return lease_read(path, &r) ? r.pid : 0;
}
