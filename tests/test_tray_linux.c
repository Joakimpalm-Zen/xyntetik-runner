// R18.15: Linux's tray on a private session bus, against a fake tray host.
//
//   ./test-tray-linux ./runner [TINY_MODEL.gguf]
//
// The test starts its own dbus-daemon (so the desktop's bus is never
// touched), owns a FAKE org.kde.StatusNotifierWatcher on it with the same
// bus code the tray uses (src/tray_dbus.c), and runs `runner --tray` as a
// child with a temporary HOME and XDG_CONFIG_HOME. It checks, through the
// bus only:
//   1. the item registers with the watcher;
//   2. IconPixmap: five sizes, each w*h*4 bytes of ARGB32 in network byte
//      order, byte-equal to the core's glyph converted here by hand; a
//      registry record changes the state, NewIcon arrives, the bytes change;
//   3. GetLayout(0, -1, []) is tray_menu_build()'s items under the spec's
//      mapping (labels, separators, submenu nesting, toggle state), and a
//      second record brings LayoutUpdated with a higher revision;
//   4. the live control: clicking the autostart row writes the .desktop file
//      with a quoted Exec line, clicking again removes it; a click on an id
//      the layout does not have does nothing and the tray stays up;
//   5. Quit ends the tray with 0;
//   6. no watcher on the bus: --tray exits 1 with the no-host line;
//   7. no bus at all: exit 1 with the no-bus line;
//   8. the bus goes away under a tray whose managed --serve child is up
//      (with TINY_MODEL): the tray exits and the child with it, no process
//      left.
// Built with the real <dbus/dbus.h> when pkg-config has it
// (TRAY_TEST_REAL_HEADER): our own struct declarations are then asserted
// against the real ones' sizes and field offsets.
// Exit 77 (skip) without dbus-daemon or libdbus.
#if !defined(__linux__)
#include <stdio.h>
int main(void) { printf("SKIP: Linux only\n"); return 77; }
#else
#include "tray.h"
#include "tray_dbus.h"
#include "compat.h"
#include "instances.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>

#ifdef TRAY_TEST_REAL_HEADER
#include <dbus/dbus.h>
// bar 2: our declarations match the real ABI
_Static_assert(sizeof(tdb_error) == sizeof(DBusError), "tdb_error size");
_Static_assert(offsetof(tdb_error, name) == offsetof(DBusError, name), "tdb_error.name");
_Static_assert(offsetof(tdb_error, message) == offsetof(DBusError, message), "tdb_error.message");
_Static_assert(offsetof(tdb_error, reserved) == offsetof(DBusError, padding1), "tdb_error.padding1");
_Static_assert(sizeof(tdb_iter) == sizeof(DBusMessageIter), "tdb_iter size");
_Static_assert(offsetof(tdb_iter, a) == offsetof(DBusMessageIter, dummy1), "tdb_iter.dummy1");
_Static_assert(offsetof(tdb_iter, b) == offsetof(DBusMessageIter, dummy2), "tdb_iter.dummy2");
_Static_assert(offsetof(tdb_iter, c) == offsetof(DBusMessageIter, dummy3), "tdb_iter.dummy3");
_Static_assert(offsetof(tdb_iter, d) == offsetof(DBusMessageIter, dummy4), "tdb_iter.dummy4");
_Static_assert(offsetof(tdb_iter, d) + 8 * sizeof(int) == offsetof(DBusMessageIter, pad1), "tdb_iter.pad1");
_Static_assert(offsetof(tdb_iter, e) == offsetof(DBusMessageIter, pad2), "tdb_iter.pad2");
_Static_assert(offsetof(tdb_iter, f) == offsetof(DBusMessageIter, pad3), "tdb_iter.pad3");
_Static_assert(sizeof(tdb_bool) == sizeof(dbus_bool_t), "dbus_bool_t");
_Static_assert(TDB_MSG_CALL == DBUS_MESSAGE_TYPE_METHOD_CALL && TDB_MSG_SIGNAL == DBUS_MESSAGE_TYPE_SIGNAL &&
               (int)TDB_HANDLED == (int)DBUS_HANDLER_RESULT_HANDLED && (int)TDB_BUS_SESSION == (int)DBUS_BUS_SESSION &&
               TDB_NAME_DO_NOT_QUEUE == DBUS_NAME_FLAG_DO_NOT_QUEUE &&
               TDB_NAME_PRIMARY_OWNER == DBUS_REQUEST_NAME_REPLY_PRIMARY_OWNER &&
               TDB_STRUCT == DBUS_TYPE_STRUCT && TDB_DICT_ENTRY == DBUS_TYPE_DICT_ENTRY, "constants");
#endif

extern char **environ;

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } \
                           else { printf("ok: " __VA_ARGS__); printf("\n"); } fflush(stdout); } while (0)

static const tdb_api *D;
static tdb_conn *g_c;
static char g_registered[128];   // the item's service name, from RegisterStatusNotifierItem
static char g_item_owner[128];   // the sender of that call (the tray's unique name)
static int  g_new_icon, g_layout_updated;
static uint32_t g_layout_rev;
static char g_tmp[512];

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int filter(tdb_conn *c, tdb_msg *m, void *ud) {
    (void)c; (void)ud;
    const char *iface = D->msg_get_interface(m), *member = D->msg_get_member(m);
    if (!iface || !member) return TDB_NOT_YET_HANDLED;
    int type = D->msg_get_type(m);
    if (type == TDB_MSG_CALL && !strcmp(iface, "org.kde.StatusNotifierWatcher") &&
        !strcmp(member, "RegisterStatusNotifierItem")) {
        tdb_iter it;
        const char *s = NULL;
        if (D->iter_init(m, &it) && D->iter_get_arg_type(&it) == TDB_STRING) D->iter_get_basic(&it, &s);
        snprintf(g_registered, sizeof g_registered, "%s", s ? s : "");
        snprintf(g_item_owner, sizeof g_item_owner, "%s", D->msg_get_sender(m) ? D->msg_get_sender(m) : "");
        tdb_msg *r = D->msg_new_method_return(m);
        if (r) { D->conn_send(g_c, r, NULL); D->msg_unref(r); }
        return TDB_HANDLED;
    }
    if (type == TDB_MSG_SIGNAL && !strcmp(iface, "org.kde.StatusNotifierItem") && !strcmp(member, "NewIcon")) {
        g_new_icon++;
        return TDB_HANDLED;
    }
    if (type == TDB_MSG_SIGNAL && !strcmp(iface, "com.canonical.dbusmenu") && !strcmp(member, "LayoutUpdated")) {
        tdb_iter it;
        if (D->iter_init(m, &it) && D->iter_get_arg_type(&it) == TDB_UINT32) D->iter_get_basic(&it, &g_layout_rev);
        g_layout_updated++;
        return TDB_HANDLED;
    }
    return TDB_NOT_YET_HANDLED;
}

static void pump(double secs) {
    double end = now_s() + secs;
    while (now_s() < end) D->conn_read_write_dispatch(g_c, 50);
}
static bool pump_until(int *counter, int above, double secs) {
    double end = now_s() + secs;
    while (now_s() < end && *counter <= above) D->conn_read_write_dispatch(g_c, 50);
    return *counter > above;
}

// ------------------------------------------------------- processes

static pid_t spawn_env(char *const argv[], char *const envp[], int out_fd) {
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    if (out_fd >= 0) posix_spawn_file_actions_adddup2(&fa, out_fd, 1);
    pid_t pid = -1;
    if (posix_spawnp(&pid, argv[0], &fa, NULL, argv, envp ? envp : environ) != 0) pid = -1;
    posix_spawn_file_actions_destroy(&fa);
    return pid;
}

// a private session bus; its address goes to DBUS_SESSION_BUS_ADDRESS
static pid_t bus_start(char *addr, size_t cap) {
    int fd[2];
    if (pipe(fd)) return -1;
    char pfd[48], where[700];
    snprintf(pfd, sizeof pfd, "--print-address=%d", fd[1]);
    // the socket in the test's own directory, not the system's temporary one
    snprintf(where, sizeof where, "--address=unix:dir=%s", g_tmp);
    char *argv[] = { "dbus-daemon", "--session", "--nofork", where, pfd, NULL };
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addclose(&fa, fd[0]);
    pid_t pid = -1;
    if (posix_spawnp(&pid, "dbus-daemon", &fa, NULL, argv, environ) != 0) pid = -1;
    posix_spawn_file_actions_destroy(&fa);
    close(fd[1]);
    size_t n = 0;
    ssize_t r;
    while (pid > 0 && n + 1 < cap && (r = read(fd[0], addr + n, cap - 1 - n)) > 0) {
        n += (size_t)r;
        if (memchr(addr, '\n', n)) break;
    }
    close(fd[0]);
    addr[n] = 0;
    char *nl = strchr(addr, '\n');
    if (nl) *nl = 0;
    return addr[0] ? pid : -1;
}

static int wait_exit(pid_t pid, double secs, bool *exited) {
    double end = now_s() + secs;
    int st = 0;
    *exited = false;
    while (now_s() < end) {
        pid_t w = waitpid(pid, &st, WNOHANG);
        if (w == pid) { *exited = true; return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st); }
        if (D && g_c) D->conn_read_write_dispatch(g_c, 50);
        else usleep(50000);
    }
    return -1;
}

static bool pid_alive(long pid) { return pid > 0 && kill((pid_t)pid, 0) == 0; }

// ------------------------------------------------------- bus calls

static tdb_msg *call(const char *dest, const char *path, const char *iface, const char *method,
                     bool (*args)(tdb_iter *, void *), void *ud) {
    tdb_msg *m = D->msg_new_method_call(dest, path, iface, method);
    if (!m) return NULL;
    if (args) {
        tdb_iter it;
        D->iter_init_append(m, &it);
        if (!args(&it, ud)) { D->msg_unref(m); return NULL; }
    }
    tdb_error e;
    D->error_init(&e);
    tdb_msg *r = D->conn_send_with_reply_and_block(g_c, m, 5000, &e);
    D->msg_unref(m);
    if (D->error_is_set(&e)) { printf("  (call %s: %s)\n", method, e.message ? e.message : "?"); D->error_free(&e); }
    return r;
}

static bool args_get(tdb_iter *it, void *ud) {
    const char **two = ud;
    return tdb_append_str(D, it, TDB_STRING, two[0]) && tdb_append_str(D, it, TDB_STRING, two[1]);
}

// IconPixmap: up to 8 (w, h, bytes) triples, the bytes copied out
typedef struct { int n, w[8], h[8]; unsigned char *px[8]; size_t len[8]; } pixmaps;

static bool read_pixmaps(pixmaps *out) {
    memset(out, 0, sizeof *out);
    const char *two[2] = { "org.kde.StatusNotifierItem", "IconPixmap" };
    tdb_msg *r = call(g_item_owner, "/StatusNotifierItem", "org.freedesktop.DBus.Properties", "Get", args_get, two);
    if (!r) return false;
    tdb_iter it, var, arr, st, ay;
    bool ok = D->iter_init(r, &it) && D->iter_get_arg_type(&it) == TDB_VARIANT;
    if (ok) {
        D->iter_recurse(&it, &var);
        ok = D->iter_get_arg_type(&var) == TDB_ARRAY;
    }
    if (ok)
        for (D->iter_recurse(&var, &arr); D->iter_get_arg_type(&arr) == TDB_STRUCT && out->n < 8; D->iter_next(&arr)) {
            int32_t w = 0, h = 0;
            D->iter_recurse(&arr, &st);
            if (D->iter_get_arg_type(&st) != TDB_INT32) { ok = false; break; }
            D->iter_get_basic(&st, &w);
            D->iter_next(&st);
            D->iter_get_basic(&st, &h);
            D->iter_next(&st);
            if (D->iter_get_arg_type(&st) != TDB_ARRAY) { ok = false; break; }
            D->iter_recurse(&st, &ay);
            const unsigned char *p = NULL;
            int n = 0;
            D->iter_get_fixed_array(&ay, &p, &n);
            out->w[out->n] = w;
            out->h[out->n] = h;
            out->len[out->n] = (size_t)n;
            out->px[out->n] = malloc(n > 0 ? (size_t)n : 1);
            if (n > 0 && p) memcpy(out->px[out->n], p, (size_t)n);
            out->n++;
        }
    D->msg_unref(r);
    return ok;
}
static void free_pixmaps(pixmaps *p) { for (int i = 0; i < p->n; i++) free(p->px[i]); p->n = 0; }

// the spec's conversion, written out here: premultiplied BGRA to straight
// ARGB32, bytes A, R, G, B
static void expect_argb(tray_icon_state st, int px, unsigned char *out) {
    unsigned char *bgra = malloc((size_t)px * px * 4);
    tray_glyph_render(st, px, 0xF0F0F0u, bgra);
    for (int i = 0; i < px * px; i++) {
        unsigned a = bgra[4 * i + 3], r = bgra[4 * i + 2], g = bgra[4 * i + 1], b = bgra[4 * i];
        if (a) { r = (r * 255 + a / 2) / a; g = (g * 255 + a / 2) / a; b = (b * 255 + a / 2) / a; }
        out[4 * i] = (unsigned char)a;
        out[4 * i + 1] = (unsigned char)(r > 255 ? 255 : r);
        out[4 * i + 2] = (unsigned char)(g > 255 ? 255 : g);
        out[4 * i + 3] = (unsigned char)(b > 255 ? 255 : b);
    }
    free(bgra);
}

static bool pixmaps_are(const pixmaps *p, tray_icon_state st, char *why, size_t cap) {
    static const int sizes[5] = { 16, 22, 24, 32, 48 };
    if (p->n != 5) { snprintf(why, cap, "%d sizes, want 5", p->n); return false; }
    for (int i = 0; i < 5; i++) {
        int s = sizes[i];
        if (p->w[i] != s || p->h[i] != s || p->len[i] != (size_t)s * s * 4) {
            snprintf(why, cap, "entry %d is %dx%d with %zu bytes, want %dx%d with %d", i, p->w[i], p->h[i],
                     p->len[i], s, s, s * s * 4);
            return false;
        }
        unsigned char *want = malloc((size_t)s * s * 4);
        expect_argb(st, s, want);
        bool same = !memcmp(want, p->px[i], (size_t)s * s * 4);
        bool opaque = false;
        for (int k = 0; k < s * s; k++) if (p->px[i][4 * k] == 255) opaque = true;
        free(want);
        if (!same) { snprintf(why, cap, "the %d px bytes differ from the core's glyph in ARGB32", s); return false; }
        if (p->px[i][0] != 0 || !opaque) {
            snprintf(why, cap, "the %d px pixmap needs a transparent corner and an opaque pixel", s);
            return false;
        }
    }
    snprintf(why, cap, "5 sizes, ARGB32 bytes equal to the core's glyph");
    return true;
}

// ------------------------------------------------------- the layout

typedef struct node { int id; char label[1100]; char type[32]; int toggle; bool is_toggle; int nkids; struct node *kids[64]; } node;

static void free_node(node *n) { if (!n) return; for (int i = 0; i < n->nkids; i++) free_node(n->kids[i]); free(n); }

static node *parse_node(tdb_iter *st_outer) {
    tdb_iter st, dict, ent, var, kids, kv;
    node *n = calloc(1, sizeof *n);
    if (!n) return NULL;
    n->toggle = -1;
    D->iter_recurse(st_outer, &st);
    if (D->iter_get_arg_type(&st) != TDB_INT32) { free(n); return NULL; }
    D->iter_get_basic(&st, &n->id);
    D->iter_next(&st);
    for (D->iter_recurse(&st, &dict); D->iter_get_arg_type(&dict) == TDB_DICT_ENTRY; D->iter_next(&dict)) {
        const char *k = NULL;
        D->iter_recurse(&dict, &ent);
        D->iter_get_basic(&ent, &k);
        D->iter_next(&ent);
        D->iter_recurse(&ent, &var);
        int t = D->iter_get_arg_type(&var);
        if (!strcmp(k, "label") && t == TDB_STRING) { const char *s; D->iter_get_basic(&var, &s); snprintf(n->label, sizeof n->label, "%s", s); }
        if (!strcmp(k, "type") && t == TDB_STRING) { const char *s; D->iter_get_basic(&var, &s); snprintf(n->type, sizeof n->type, "%s", s); }
        if (!strcmp(k, "toggle-type") && t == TDB_STRING) n->is_toggle = true;
        if (!strcmp(k, "toggle-state") && t == TDB_INT32) { int32_t v; D->iter_get_basic(&var, &v); n->toggle = v; }
    }
    D->iter_next(&st);
    for (D->iter_recurse(&st, &kids); D->iter_get_arg_type(&kids) == TDB_VARIANT && n->nkids < 64; D->iter_next(&kids)) {
        D->iter_recurse(&kids, &kv);
        node *c = parse_node(&kv);
        if (c) n->kids[n->nkids++] = c;
    }
    return n;
}

static bool args_layout(tdb_iter *it, void *ud) {
    (void)ud;
    tdb_iter a;
    return tdb_append_i32(D, it, 0) && tdb_append_i32(D, it, -1) && D->iter_open_container(it, TDB_ARRAY, "s", &a) &&
           D->iter_close_container(it, &a);
}

static node *get_layout(uint32_t *rev) {
    tdb_msg *r = call(g_item_owner, "/MenuBar", "com.canonical.dbusmenu", "GetLayout", args_layout, NULL);
    if (!r) return NULL;
    tdb_iter it;
    node *root = NULL;
    if (D->iter_init(r, &it) && D->iter_get_arg_type(&it) == TDB_UINT32) {
        D->iter_get_basic(&it, rev);
        D->iter_next(&it);
        if (D->iter_get_arg_type(&it) == TDB_STRUCT) root = parse_node(&it);
    }
    D->msg_unref(r);
    return root;
}

// the spec's table, applied to the core's own items: ids 1 + index; a
// SUB_BEGIN .. SUB_END run nests under the row before SUB_BEGIN
static bool layout_matches(const node *root, const tray_item *it, int n, char *why, size_t cap) {
    int parent[256], stack[16], depth = 0, last = 0;
    for (int i = 0; i < n; i++) {
        parent[i] = depth ? stack[depth - 1] : 0;
        if (it[i].kind == TRAY_K_SUB_BEGIN) { if (depth < 16) stack[depth++] = last; }
        else if (it[i].kind == TRAY_K_SUB_END) { if (depth) depth--; }
        else last = i + 1;
    }
    // walk: each node's children, in order, are the rows whose parent is it
    const node *q[256];
    int qn = 0;
    q[qn++] = root;
    for (int h = 0; h < qn; h++) {
        const node *nd = q[h];
        int k = 0;
        for (int i = 0; i < n; i++) {
            if (parent[i] != nd->id || it[i].kind == TRAY_K_SUB_BEGIN || it[i].kind == TRAY_K_SUB_END) continue;
            if (k >= nd->nkids) { snprintf(why, cap, "node %d lacks child %d", nd->id, i + 1); return false; }
            const node *c = nd->kids[k++];
            if (c->id != i + 1) { snprintf(why, cap, "node %d's child is %d, want %d", nd->id, c->id, i + 1); return false; }
            if (it[i].kind == TRAY_K_SEP) {
                if (strcmp(c->type, "separator")) { snprintf(why, cap, "item %d is not a separator", c->id); return false; }
            } else {
                char want[1100];
                size_t o = 0;
                for (const char *p = it[i].label; *p && o + 2 < sizeof want; p++) {
                    if (*p == '_') want[o++] = '_';
                    want[o++] = *p;
                }
                want[o] = 0;
                if (strcmp(c->label, want)) { snprintf(why, cap, "item %d is labelled \"%.300s\", want \"%.300s\"", c->id, c->label, want); return false; }
            }
            if (it[i].kind == TRAY_K_CHECK && (!c->is_toggle || c->toggle != (it[i].checked ? 1 : 0))) {
                snprintf(why, cap, "item %d's toggle is %d, want %d", c->id, c->toggle, it[i].checked);
                return false;
            }
            if (qn < 256) q[qn++] = c;
        }
        if (k != nd->nkids) { snprintf(why, cap, "node %d has %d children, want %d", nd->id, nd->nkids, k); return false; }
    }
    snprintf(why, cap, "%d items, by the spec's mapping", n);
    return true;
}

static bool args_event(tdb_iter *it, void *ud) {
    int id = *(int *)ud;
    tdb_iter v;
    return tdb_append_i32(D, it, id) && tdb_append_str(D, it, TDB_STRING, "clicked") &&
           D->iter_open_container(it, TDB_VARIANT, "i", &v) && tdb_append_i32(D, &v, 0) &&
           D->iter_close_container(it, &v) && tdb_append_u32(D, it, 0);
}
static bool click(int id) {
    tdb_msg *r = call(g_item_owner, "/MenuBar", "com.canonical.dbusmenu", "Event", args_event, &id);
    if (r) D->msg_unref(r);
    return r != NULL;
}

static int find_action(const tray_item *it, int n, int action) {
    for (int i = 0; i < n; i++) if (it[i].action == action && it[i].kind != TRAY_K_LABEL) return i + 1;
    return 0;
}

static bool connect_bus(bool watcher) {
    tdb_error e;
    D->error_init(&e);
    g_c = D->bus_get_private(TDB_BUS_SESSION, &e);
    if (!g_c) { if (D->error_is_set(&e)) D->error_free(&e); return false; }
    D->conn_set_exit_on_disconnect(g_c, 0);
    if (watcher) {
        int rq = D->bus_request_name(g_c, "org.kde.StatusNotifierWatcher", TDB_NAME_DO_NOT_QUEUE, &e);
        if (D->error_is_set(&e) || rq != TDB_NAME_PRIMARY_OWNER) { if (D->error_is_set(&e)) D->error_free(&e); return false; }
    }
    D->bus_add_match(g_c, "type='signal',interface='org.kde.StatusNotifierItem'", &e);
    if (D->error_is_set(&e)) D->error_free(&e);
    D->bus_add_match(g_c, "type='signal',interface='com.canonical.dbusmenu'", &e);
    if (D->error_is_set(&e)) D->error_free(&e);
    D->conn_add_filter(g_c, filter, NULL, NULL);
    return true;
}
static void disconnect_bus(void) {
    if (!g_c) return;
    D->conn_close(g_c);
    D->conn_unref(g_c);
    g_c = NULL;
}

static void write_record(long pid, const char *mode) {
    char p[700];
    snprintf(p, sizeof p, "%s/home/.xyntetik/runner/instances/%ld.json", g_tmp, pid);
    FILE *f = fopen(p, "wb");
    if (!f) return;
    fprintf(f, "{\"pid\": %ld, \"started\": 1, \"mode\": \"%s\", \"port\": 0, \"version\": \"t\", \"models\": []}\n",
            pid, mode);
    fclose(f);
}

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char *b = calloc(1, 8192);
    if (b) { size_t n = fread(b, 1, 8191, f); b[n] = 0; }
    fclose(f);
    return b;
}

static pid_t start_tray(const char *runner, int out_fd) {
    char *argv[] = { (char *)runner, "--tray", NULL };
    return spawn_env(argv, NULL, out_fd);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc == 2 && !strcmp(argv[1], "--hold-watcher")) {
        // a helper for tests/test_tray_not_raised_without_console.py: the fake
        // tray host on $DBUS_SESSION_BUS_ADDRESS, answering registrations,
        // until stdin closes; "ready" on stdout once it owns the name
        const char *w = NULL;
        D = tdb_load(&w);
        if (!D || !connect_bus(true)) { fprintf(stderr, "hold-watcher: %s\n", w ? w : "no bus"); return 1; }
        printf("ready\n");
        fcntl(0, F_SETFL, fcntl(0, F_GETFL) | O_NONBLOCK);
        for (;;) {
            D->conn_read_write_dispatch(g_c, 100);
            char b;
            ssize_t r = read(0, &b, 1);
            if (r == 0 || (r < 0 && errno != EAGAIN)) break;
        }
        disconnect_bus();
        return 0;
    }
    if (argc < 2) { fprintf(stderr, "usage: %s RUNNER [TINY_MODEL.gguf] | --hold-watcher\n", argv[0]); return 2; }
    char runner[PATH_MAX];
    if (!realpath(argv[1], runner)) { fprintf(stderr, "no runner at %s\n", argv[1]); return 2; }
    const char *model = argc > 2 ? argv[2] : NULL;
    const char *why = NULL;
    D = tdb_load(&why);
    if (!D) { printf("SKIP: %s\n", why); return 77; }
    char *probe[] = { "dbus-daemon", "--version", NULL };
    int devnull = open("/dev/null", O_WRONLY);
    pid_t pp = spawn_env(probe, NULL, devnull);
    int pst = 0;
    if (pp <= 0 || waitpid(pp, &pst, 0) != pp || !WIFEXITED(pst) || WEXITSTATUS(pst) != 0) {
        printf("SKIP: no dbus-daemon on PATH\n");
        return 77;
    }
#ifdef TRAY_TEST_REAL_HEADER
    printf("ok: tdb_error and tdb_iter match <dbus/dbus.h>'s DBusError and DBusMessageIter (static asserts)\n");
#else
    printf("note: built without <dbus/dbus.h>: the layout asserts did not run here\n");
#endif

    const char *base = getenv("TMPDIR");
    snprintf(g_tmp, sizeof g_tmp, "%s/tray-linux-XXXXXX", base && *base ? base : "/tmp");
    if (!mkdtemp(g_tmp)) { perror("mkdtemp"); return 2; }
    char home[700], cfg[700], rt[700], idir[800];
    snprintf(home, sizeof home, "%s/home", g_tmp);
    snprintf(cfg, sizeof cfg, "%s/config", g_tmp);
    snprintf(rt, sizeof rt, "%s/run", g_tmp);
    mkdir(home, 0700);
    mkdir(cfg, 0700);
    mkdir(rt, 0700);
    setenv("HOME", home, 1);
    setenv("XDG_CONFIG_HOME", cfg, 1);
    setenv("XDG_RUNTIME_DIR", rt, 1);
    unsetenv("XYNTETIK_TRAY_DUMP");
    unsetenv("XYNTETIK_TRAY_ICON_DUMP");
    snprintf(idir, sizeof idir, "%s/.xyntetik/runner/instances", home);
    char d1[760], d2[780];
    snprintf(d1, sizeof d1, "%s/.xyntetik", home);
    snprintf(d2, sizeof d2, "%s/.xyntetik/runner", home);
    mkdir(d1, 0700);
    mkdir(d2, 0700);
    mkdir(idir, 0700);

    // ---- 7. no bus at all
    {
        unsetenv("DBUS_SESSION_BUS_ADDRESS");
        int fd[2];
        if (pipe(fd)) return 2;
        char *a[] = { runner, "--tray", NULL };
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_adddup2(&fa, fd[1], 2);
        pid_t p = -1;
        posix_spawn(&p, runner, &fa, NULL, a, environ);
        posix_spawn_file_actions_destroy(&fa);
        close(fd[1]);
        char buf[1024] = "";
        ssize_t n = read(fd[0], buf, sizeof buf - 1);
        if (n > 0) buf[n] = 0;
        close(fd[0]);
        bool ex;
        int rc = wait_exit(p, 20, &ex);
        CHECK(ex && rc == 1 && strstr(buf, "no D-Bus session bus"),
              "7. no session bus: --tray exits %d (want 1) saying \"%.*s\"", rc, (int)strcspn(buf, "\n"), buf);
    }

    // ---- the bus and the fake host
    char addr[512];
    pid_t bus = bus_start(addr, sizeof addr);
    CHECK(bus > 0, "a private session bus (%s)", addr);
    if (bus <= 0) return 1;
    setenv("DBUS_SESSION_BUS_ADDRESS", addr, 1);

    // ---- 6. a bus with no tray host
    {
        CHECK(connect_bus(false), "the test connects without the watcher");
        int fd[2];
        if (pipe(fd)) return 2;
        char *a[] = { runner, "--tray", NULL };
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_adddup2(&fa, fd[1], 2);
        pid_t p = -1;
        posix_spawn(&p, runner, &fa, NULL, a, environ);
        posix_spawn_file_actions_destroy(&fa);
        close(fd[1]);
        char buf[1024] = "";
        ssize_t n = read(fd[0], buf, sizeof buf - 1);
        if (n > 0) buf[n] = 0;
        close(fd[0]);
        bool ex;
        int rc = wait_exit(p, 20, &ex);
        CHECK(ex && rc == 1 && strstr(buf, "no tray host on the session bus"),
              "6. no watcher: --tray exits %d (want 1) saying \"%.*s\"", rc, (int)strcspn(buf, "\n"), buf);
        disconnect_bus();
    }

    CHECK(connect_bus(true), "the fake org.kde.StatusNotifierWatcher owns its name");
    int log_fd = open("test-tray-linux.tray.log", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    pid_t tray = start_tray(runner, log_fd);
    double t0 = now_s();
    while (!g_registered[0] && now_s() - t0 < 15) D->conn_read_write_dispatch(g_c, 50);
    // 1. registration
    char want_name[128];
    snprintf(want_name, sizeof want_name, "org.kde.StatusNotifierItem-%ld-1", (long)tray);
    CHECK(g_registered[0] && !strcmp(g_registered, want_name),
          "1. the item registers as %s (got \"%s\" from %s)", want_name, g_registered, g_item_owner);

    // 2. the pixmaps, idle
    pixmaps px;
    char why2[256];
    CHECK(read_pixmaps(&px) && pixmaps_are(&px, TRAY_ICON_IDLE, why2, sizeof why2), "2. IconPixmap idle: %s", why2);
    // a record makes the state LOADED: NewIcon, and the bytes change
    char *sl[] = { "sleep", "120", NULL };
    pid_t helper = spawn_env(sl, NULL, -1);
    int icons_before = g_new_icon, layouts_before = g_layout_updated;
    write_record(helper, "serve");
    bool got_icon = pump_until(&g_new_icon, icons_before, 8);
    pixmaps px2;
    bool read2 = read_pixmaps(&px2);
    bool differ = read2 && px2.n == px.n && px.n > 0 && memcmp(px.px[0], px2.px[0], px.len[0]) != 0;
    CHECK(got_icon && read2 && differ && pixmaps_are(&px2, TRAY_ICON_LOADED, why2, sizeof why2),
          "2. a record: NewIcon arrives (%d), the bytes change (%d), and are the loaded glyph: %s", got_icon, differ,
          why2);
    free_pixmaps(&px);
    free_pixmaps(&px2);
    pump_until(&g_layout_updated, layouts_before, 4);

    // 3. the layout against the core's own items (built here, from the same HOME)
    static tray_item items[128];
    uint32_t rev = 0;
    node *root = get_layout(&rev);
    int n = tray_menu_build(items, 128);
    char why3[1400];
    CHECK(root && root->id == 0 && layout_matches(root, items, n, why3, sizeof why3), "3. GetLayout(0, -1): %s",
          root ? why3 : "no answer");
    free_node(root);
    // a second record: LayoutUpdated, with a higher revision
    pid_t helper2 = spawn_env(sl, NULL, -1);
    layouts_before = g_layout_updated;
    write_record(helper2, "serve");
    bool upd = pump_until(&g_layout_updated, layouts_before, 8);
    CHECK(upd && g_layout_rev > rev, "3. a second record: LayoutUpdated arrives with revision %u > %u", g_layout_rev, rev);

    // 4. the live control: the autostart toggle, by id
    n = tray_menu_build(items, 128);
    int auto_id = find_action(items, n, TRAY_ACT_TOGGLE_AUTOSTART);
    char desk[800];
    snprintf(desk, sizeof desk, "%s/autostart/xyntetik-runner-tray.desktop", cfg);
    struct stat sb;
    CHECK(auto_id > 0 && stat(desk, &sb) != 0, "4. the autostart row is id %d and no autostart file exists yet", auto_id);
    bool clicked = click(auto_id);
    pump(0.5);
    char *dtext = slurp(desk);
    char want_exec[PATH_MAX + 32];
    snprintf(want_exec, sizeof want_exec, "Exec=\"%s\" --tray\n", runner);
    CHECK(clicked && dtext && strstr(dtext, want_exec) && strstr(dtext, "[Desktop Entry]"),
          "4. clicking it writes %s with %.*s (clicked %d; the file: %s)", desk, (int)strcspn(want_exec, "\n"),
          want_exec, clicked, dtext ? dtext : "none");
    free(dtext);
    clicked = click(auto_id);
    pump(0.5);
    CHECK(clicked && stat(desk, &sb) != 0, "4. clicking it again removes the file");
    // an id the layout does not have: nothing acts, the tray stays up
    clicked = click(9999);
    pump(0.5);
    uint32_t rev2 = 0;
    node *again = get_layout(&rev2);
    CHECK(clicked && again && stat(desk, &sb) != 0 && waitpid(tray, NULL, WNOHANG) == 0,
          "4. a click on id 9999 acts on nothing and the tray answers on");
    free_node(again);

    // 5. Quit
    int quit_id = find_action(items, n, TRAY_ACT_QUIT);
    click(quit_id);
    bool ex;
    int rc = wait_exit(tray, 15, &ex);
    CHECK(quit_id > 0 && ex && rc == 0, "5. Quit (id %d) ends the tray with %d (want 0)", quit_id, rc);
    if (!ex) { kill(tray, SIGKILL); waitpid(tray, NULL, 0); }

    // ---- 8. the bus goes away under a tray with a managed --serve child
    if (model) {
        char cfgjson[800];
        snprintf(cfgjson, sizeof cfgjson, "%s/.xyntetik/runner/config.json", home);
        FILE *f = fopen(cfgjson, "wb");
        char mreal[PATH_MAX];
        if (f && realpath(model, mreal)) {
            fprintf(f, "{\"last_model\": \"%s\", \"last_args\": \"--gpu off -c 128 -t 1 --no-tray\", \"port\": %d}\n",
                    mreal, 18000 + (int)(getpid() % 2000));
            fclose(f);
        } else if (f) fclose(f);
        g_registered[0] = 0;
        tray = start_tray(runner, log_fd);
        t0 = now_s();
        while (!g_registered[0] && now_s() - t0 < 15) D->conn_read_write_dispatch(g_c, 50);
        n = tray_menu_build(items, 128);
        int start_id = find_action(items, n, TRAY_ACT_START_MANAGED);
        click(start_id);
        // the managed child appears in the registry as a serve that is not ours
        long child = 0;
        for (int k = 0; k < 200 && !child; k++) {
            int ni = 0;
            instance_rec *r = instances_list(&ni);
            for (int i = 0; i < ni; i++)
                if (!strcmp(r[i].mode, "serve") && r[i].pid != helper && r[i].pid != helper2) child = r[i].pid;
            instances_list_free(r, ni);
            if (!child) pump(0.1);
        }
        CHECK(g_registered[0] && start_id > 0 && child > 0 && pid_alive(child),
              "8. a tray started a managed --serve (pid %ld) from its menu (row %d)", child, start_id);
        disconnect_bus();
        kill(bus, SIGKILL);
        waitpid(bus, NULL, 0);
        bus = -1;
        rc = wait_exit(tray, 20, &ex);
        bool child_gone = false;
        for (int k = 0; k < 100 && !(child_gone = !pid_alive(child)); k++) usleep(100000);
        CHECK(ex && rc == 1 && child_gone,
              "8. the bus killed under it: the tray exits %d (want 1) and its managed child is gone (%d)", rc, child_gone);
        if (!ex) { kill(tray, SIGKILL); waitpid(tray, NULL, 0); }
        if (child && pid_alive(child)) kill((pid_t)child, SIGKILL);
    } else {
        printf("note: no TINY_MODEL given: test 8 (the bus killed under a managed child) did not run\n");
    }

    disconnect_bus();
    if (bus > 0) { kill(bus, SIGTERM); waitpid(bus, NULL, 0); }
    kill(helper, SIGKILL);
    kill(helper2, SIGKILL);
    waitpid(helper, NULL, 0);
    waitpid(helper2, NULL, 0);
    printf(fails ? "FAIL %d\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
#endif
