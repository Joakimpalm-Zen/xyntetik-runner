// Linux tray backend (R18.15): a StatusNotifierItem with a dbusmenu menu on
// the session bus, the protocol KDE Plasma, Ubuntu's GNOME (its AppIndicator
// extension), XFCE and Cinnamon's tray plugins host.
//
// The bus library is libdbus-1, dlopen'd at --tray time (tray_dbus.h): no
// build dependency, no DT_NEEDED entry, and a machine without it loses only
// the tray. Everything the menu says and does is the portable core's
// (tray_menu_build / tray_menu_act); this file renders it onto two D-Bus
// objects and forwards clicks back:
//   /StatusNotifierItem  org.kde.StatusNotifierItem: the icon, its tooltip
//   /MenuBar             com.canonical.dbusmenu (version 3): the menu tree
// One process, one thread: the loop dispatches the bus with a 200 ms timeout
// and refreshes the icon and the menu every 2 s.
#if defined(__linux__)
#include "tray.h"
#include "tray_dbus.h"
#include "compat.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

#define SNI_PATH   "/StatusNotifierItem"
#define MENU_PATH  "/MenuBar"
#define SNI_IFACE  "org.kde.StatusNotifierItem"
#define MENU_IFACE "com.canonical.dbusmenu"
#define PROPS_IFACE "org.freedesktop.DBus.Properties"
#define WATCHER    "org.kde.StatusNotifierWatcher"
#define PORTAL     "org.freedesktop.portal.Desktop"
#define TRAY_MAX_ITEMS 128

static const tdb_api *D;
static tdb_conn *g_bus;
static char g_name[96];             // org.kde.StatusNotifierItem-<pid>-1
static volatile sig_atomic_t g_signalled;

// the menu the host last saw. Ids are the rows' OWN (review finding 2): a row keeps its id while it is in the
// menu, and an id is never given to another row, so a click the host sends against a layout the tray has since
// changed (a server's record gone, the rows below it moved up) finds the row it meant, or none, and never the row
// that moved into its place. g_id[i] is item i's id, g_parent[i] its parent's id (0: the root).
static tray_item g_pub[TRAY_MAX_ITEMS];
static int       g_npub;
static int       g_id[TRAY_MAX_ITEMS];
static int       g_parent[TRAY_MAX_ITEMS];
#define TRAY_MAX_KEYS 4096
static struct { char key[600]; int id; } *g_keys;   // every row identity given an id so far
static int       g_nkeys, g_next_id = 1;

// the tray host's absence (review finding 1, the advisor's refinement): the time the watcher's name lost its
// owner, 0 while it has one. Past the grace window the tray ends through the core's Quit, which stops the server
// the tray started (managed) and only that one, so a server someone else started (follow) keeps running.
static double g_host_gone_at;
static double g_host_grace_s = 30.0;

// the zenity / kdialog fallback, run without blocking the loop (review finding 3): its pid and its stdout
static pid_t  g_dlg_pid = -1;
static int    g_dlg_fd = -1;
static char   g_dlg_buf[1100];
static size_t g_dlg_n;
static uint32_t  g_rev = 1;
static tray_icon_state g_icon = TRAY_ICON_IDLE;

// the file chooser: a portal request in flight (its handle path) or none
static char g_pick_handle[256];
static bool g_picker_none;          // neither the portal nor zenity / kdialog

static double now_mono(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

static const int ICON_SIZES[] = { 16, 22, 24, 32, 48 };
#define N_ICON_SIZES ((int)(sizeof ICON_SIZES / sizeof ICON_SIZES[0]))

// ------------------------------------------------------------------ pixmaps

// The core renders premultiplied BGRA (a Windows DIB row). StatusNotifierItem
// takes ARGB32 in network byte order, not premultiplied: each pixel's bytes
// are A, R, G, B. Light (0xF0F0F0): GNOME's top bar is dark, and a pixmap
// cannot follow the theme.
#define ICON_RGB 0xF0F0F0u
static bool icon_argb(tray_icon_state st, int px, unsigned char *out) {
    unsigned char *bgra = malloc((size_t)px * px * 4);
    if (!bgra) return false;
    bool ok = tray_glyph_render(st, px, ICON_RGB, bgra);
    // The glyph is one colour, so a pixel's straight colour IS that colour wherever it has coverage: only alpha
    // varies. Un-premultiplying the core's rounded BGRA instead lost the colour on low-alpha edge pixels
    // (bar 6's anchor: every coloured pixel 0xF0 +-1).
    for (int i = 0; ok && i < px * px; i++) {
        unsigned a = bgra[4 * i + 3];
        out[4 * i] = (unsigned char)a;
        out[4 * i + 1] = a ? (unsigned char)(ICON_RGB >> 16) : 0;
        out[4 * i + 2] = a ? (unsigned char)(ICON_RGB >> 8) : 0;
        out[4 * i + 3] = a ? (unsigned char)ICON_RGB : 0;
    }
    free(bgra);
    return ok;
}

static bool put_pixmaps(tdb_iter *it) {
    tdb_iter arr;
    if (!D->iter_open_container(it, TDB_ARRAY, "(iiay)", &arr)) return false;
    bool ok = true;
    for (int k = 0; ok && k < N_ICON_SIZES; k++) {
        int px = ICON_SIZES[k];
        unsigned char *buf = malloc((size_t)px * px * 4);
        tdb_iter st, ay;
        ok = buf && icon_argb(g_icon, px, buf) &&
             D->iter_open_container(&arr, TDB_STRUCT, NULL, &st) &&
             tdb_append_i32(D, &st, px) && tdb_append_i32(D, &st, px) &&
             D->iter_open_container(&st, TDB_ARRAY, "y", &ay);
        if (ok) {
            const unsigned char *p = buf;
            ok = D->iter_append_fixed_array(&ay, TDB_BYTE, &p, px * px * 4) &&
                 D->iter_close_container(&st, &ay) && D->iter_close_container(&arr, &st);
        }
        free(buf);
    }
    return D->iter_close_container(it, &arr) && ok;
}

static const char *icon_word(tray_icon_state st) {
    return st == TRAY_ICON_RUNNING ? "working" : st == TRAY_ICON_LOADED ? "model loaded" : "idle";
}

// ------------------------------------------------------- variant helpers

static bool var_str(tdb_iter *it, int type, const char *s) {
    char sig[2] = { (char)type, 0 };
    tdb_iter v;
    return D->iter_open_container(it, TDB_VARIANT, sig, &v) && tdb_append_str(D, &v, type, s) &&
           D->iter_close_container(it, &v);
}
static bool var_bool(tdb_iter *it, bool b) {
    tdb_iter v;
    return D->iter_open_container(it, TDB_VARIANT, "b", &v) && tdb_append_bool(D, &v, b) &&
           D->iter_close_container(it, &v);
}
static bool var_i32(tdb_iter *it, int32_t i) {
    tdb_iter v;
    return D->iter_open_container(it, TDB_VARIANT, "i", &v) && tdb_append_i32(D, &v, i) &&
           D->iter_close_container(it, &v);
}
static bool var_u32(tdb_iter *it, uint32_t u) {
    tdb_iter v;
    return D->iter_open_container(it, TDB_VARIANT, "u", &v) && tdb_append_u32(D, &v, u) &&
           D->iter_close_container(it, &v);
}

// ------------------------------------------------- StatusNotifierItem props

static const char *const SNI_PROPS[] = {
    "Category", "Id", "Title", "Status", "WindowId", "IconName", "IconPixmap", "OverlayIconName",
    "AttentionIconName", "ToolTip", "ItemIsMenu", "Menu", NULL,
};

static bool put_tooltip(tdb_iter *it) {
    char title[96];
    snprintf(title, sizeof title, "Runner: %s", icon_word(g_icon));
    tdb_iter v, st, px;
    return D->iter_open_container(it, TDB_VARIANT, "(sa(iiay)ss)", &v) &&
           D->iter_open_container(&v, TDB_STRUCT, NULL, &st) && tdb_append_str(D, &st, TDB_STRING, "") &&
           D->iter_open_container(&st, TDB_ARRAY, "(iiay)", &px) && D->iter_close_container(&st, &px) &&
           tdb_append_str(D, &st, TDB_STRING, title) && tdb_append_str(D, &st, TDB_STRING, "") &&
           D->iter_close_container(&v, &st) && D->iter_close_container(it, &v);
}

// write property `p` of the item as a variant into `it`; false: no such property
static bool sni_prop(tdb_iter *it, const char *p, bool *known) {
    *known = true;
    if (!strcmp(p, "Category")) return var_str(it, TDB_STRING, "ApplicationStatus");
    if (!strcmp(p, "Id"))       return var_str(it, TDB_STRING, "xyntetik-runner");
    if (!strcmp(p, "Title"))    return var_str(it, TDB_STRING, "Runner");
    if (!strcmp(p, "Status"))   return var_str(it, TDB_STRING, "Active");
    if (!strcmp(p, "WindowId")) return var_i32(it, 0);
    if (!strcmp(p, "IconName") || !strcmp(p, "OverlayIconName") || !strcmp(p, "AttentionIconName"))
        return var_str(it, TDB_STRING, "");
    if (!strcmp(p, "IconPixmap")) {
        tdb_iter v;
        return D->iter_open_container(it, TDB_VARIANT, "a(iiay)", &v) && put_pixmaps(&v) &&
               D->iter_close_container(it, &v);
    }
    if (!strcmp(p, "ToolTip"))    return put_tooltip(it);
    if (!strcmp(p, "ItemIsMenu")) return var_bool(it, true);
    if (!strcmp(p, "Menu"))       return var_str(it, TDB_PATH, MENU_PATH);
    *known = false;
    return false;
}

static const char *const MENU_PROPS[] = { "Version", "TextDirection", "Status", "IconThemePath", NULL };

static bool menu_prop(tdb_iter *it, const char *p, bool *known) {
    *known = true;
    if (!strcmp(p, "Version"))       return var_u32(it, 3);
    if (!strcmp(p, "TextDirection")) return var_str(it, TDB_STRING, "ltr");
    if (!strcmp(p, "Status"))        return var_str(it, TDB_STRING, "normal");
    if (!strcmp(p, "IconThemePath")) {
        tdb_iter v, a;
        return D->iter_open_container(it, TDB_VARIANT, "as", &v) &&
               D->iter_open_container(&v, TDB_ARRAY, "s", &a) && D->iter_close_container(&v, &a) &&
               D->iter_close_container(it, &v);
    }
    *known = false;
    return false;
}

// ------------------------------------------------------------- the menu

// the dbusmenu label: a lone '_' is a mnemonic marker, so each one is doubled
static void menu_label(char *out, size_t cap, const char *in) {
    size_t o = 0;
    for (const char *p = in; *p && o + 2 < cap; p++) {
        if (*p == '_') out[o++] = '_';
        out[o++] = *p;
    }
    out[o] = 0;
}

static bool row_shown(const tray_item *t) { return t->kind != TRAY_K_SUB_BEGIN && t->kind != TRAY_K_SUB_END; }

// a row's identity: what it is and does (kind, action, arg, label), and which occurrence of that it is in this
// build (two separators are the same row otherwise). Its id: the one that identity was given before, or a new one.
static int id_for(const tray_item *t, int occ) {
    char key[600];
    snprintf(key, sizeof key, "%d|%d|%ld|%d|%.500s", t->kind, t->action, t->arg, occ, t->label);
    for (int k = 0; k < g_nkeys; k++) if (!strcmp(g_keys[k].key, key)) return g_keys[k].id;
    if (!g_keys) g_keys = calloc(TRAY_MAX_KEYS, sizeof *g_keys);
    int id = g_next_id++;
    if (g_keys && g_nkeys < TRAY_MAX_KEYS) {   // a full table still never reuses an id: the row just gets a new one
        snprintf(g_keys[g_nkeys].key, sizeof g_keys[g_nkeys].key, "%s", key);
        g_keys[g_nkeys++].id = id;
    }
    return id;
}

// ids by identity, then parents by the core's nesting: the items between SUB_BEGIN and SUB_END are the children
// of the row before SUB_BEGIN
static void link_ids(const tray_item *it, int n, int *ids, int *parent) {
    for (int i = 0; i < n; i++) {
        int occ = 0;
        for (int j = 0; j < i; j++)
            occ += it[j].kind == it[i].kind && it[j].action == it[i].action && it[j].arg == it[i].arg &&
                   !strcmp(it[j].label, it[i].label);
        ids[i] = id_for(&it[i], occ);
    }
    int stack[16], depth = 0, last_row = 0;
    for (int i = 0; i < n; i++) {
        parent[i] = depth ? stack[depth - 1] : 0;
        if (it[i].kind == TRAY_K_SUB_BEGIN) { if (depth < 16) stack[depth++] = last_row; }
        else if (it[i].kind == TRAY_K_SUB_END) { if (depth) depth--; }
        else last_row = ids[i];
    }
}

// the published item with this id, -1 for none (an id the layout does not hold, or a SUB marker's)
static int idx_of(int id) {
    for (int i = 0; id > 0 && i < g_npub; i++) if (g_id[i] == id && row_shown(&g_pub[i])) return i;
    return -1;
}

static bool has_children(int id) {
    for (int i = 0; i < g_npub; i++)
        if (g_parent[i] == id && row_shown(&g_pub[i])) return true;
    return false;
}

static bool dict_entry_begin(tdb_iter *dict, tdb_iter *e, const char *key) {
    return D->iter_open_container(dict, TDB_DICT_ENTRY, NULL, e) && tdb_append_str(D, e, TDB_STRING, key);
}

static bool item_props(tdb_iter *dict, int id) {
    tdb_iter e;
    if (id == 0) {
        return dict_entry_begin(dict, &e, "children-display") && var_str(&e, TDB_STRING, "submenu") &&
               D->iter_close_container(dict, &e);
    }
    const int ix = idx_of(id);
    if (ix < 0) return false;
    const tray_item *t = &g_pub[ix];
    bool ok = true;
    if (t->kind == TRAY_K_SEP) {
        return dict_entry_begin(dict, &e, "type") && var_str(&e, TDB_STRING, "separator") &&
               D->iter_close_container(dict, &e);
    }
    char label[1100];
    menu_label(label, sizeof label, t->label);
    ok = ok && dict_entry_begin(dict, &e, "label") && var_str(&e, TDB_STRING, label) &&
         D->iter_close_container(dict, &e);
    bool enabled = t->kind != TRAY_K_LABEL && !(t->action == TRAY_ACT_PICK_MODEL && g_picker_none);
    ok = ok && dict_entry_begin(dict, &e, "enabled") && var_bool(&e, enabled) && D->iter_close_container(dict, &e);
    if (t->kind == TRAY_K_CHECK) {
        ok = ok && dict_entry_begin(dict, &e, "toggle-type") && var_str(&e, TDB_STRING, "checkmark") &&
             D->iter_close_container(dict, &e) &&
             dict_entry_begin(dict, &e, "toggle-state") && var_i32(&e, t->checked ? 1 : 0) &&
             D->iter_close_container(dict, &e);
    }
    if (has_children(id)) {
        ok = ok && dict_entry_begin(dict, &e, "children-display") && var_str(&e, TDB_STRING, "submenu") &&
             D->iter_close_container(dict, &e);
    }
    return ok;
}

// (ia{sv}av): one node and, to `depth` (-1: all), its children
static bool put_node(tdb_iter *it, int id, int depth) {
    tdb_iter st, dict, kids;
    bool ok = D->iter_open_container(it, TDB_STRUCT, NULL, &st) && tdb_append_i32(D, &st, id) &&
              D->iter_open_container(&st, TDB_ARRAY, "{sv}", &dict) && item_props(&dict, id) &&
              D->iter_close_container(&st, &dict) && D->iter_open_container(&st, TDB_ARRAY, "v", &kids);
    if (ok && depth != 0) {
        for (int i = 0; ok && i < g_npub; i++) {
            if (g_parent[i] != id || !row_shown(&g_pub[i])) continue;
            tdb_iter v;
            ok = D->iter_open_container(&kids, TDB_VARIANT, "(ia{sv}av)", &v) &&
                 put_node(&v, g_id[i], depth < 0 ? -1 : depth - 1) && D->iter_close_container(&kids, &v);
        }
    }
    return ok && D->iter_close_container(&st, &kids) && D->iter_close_container(it, &st);
}

static bool items_equal(const tray_item *a, int na, const tray_item *b, int nb) {
    if (na != nb) return false;
    for (int i = 0; i < na; i++)
        if (a[i].kind != b[i].kind || a[i].action != b[i].action || a[i].arg != b[i].arg ||
            a[i].checked != b[i].checked || strcmp(a[i].label, b[i].label))
            return false;
    return true;
}

static void emit(tdb_msg *m) {
    if (!m) return;
    D->conn_send(g_bus, m, NULL);
    D->msg_unref(m);
}

// rebuild from the core; publish (bump the revision, LayoutUpdated) only on a change
static bool menu_refresh(void) {
    static tray_item fresh[TRAY_MAX_ITEMS];
    int n = tray_menu_build(fresh, TRAY_MAX_ITEMS);
    if (n > TRAY_MAX_ITEMS) n = TRAY_MAX_ITEMS;
    if (items_equal(fresh, n, g_pub, g_npub)) return false;
    memcpy(g_pub, fresh, sizeof(tray_item) * (size_t)n);
    g_npub = n;
    link_ids(g_pub, g_npub, g_id, g_parent);
    g_rev++;
    tdb_msg *m = D->msg_new_signal(MENU_PATH, MENU_IFACE, "LayoutUpdated");
    if (m) {
        tdb_iter it;
        D->iter_init_append(m, &it);
        if (!tdb_append_u32(D, &it, g_rev) || !tdb_append_i32(D, &it, 0)) { D->msg_unref(m); m = NULL; }
    }
    emit(m);
    return true;
}

static void icon_refresh(void) {
    tray_icon_state now = tray_icon();
    if (now == g_icon) return;
    g_icon = now;
    emit(D->msg_new_signal(SNI_PATH, SNI_IFACE, "NewIcon"));
    emit(D->msg_new_signal(SNI_PATH, SNI_IFACE, "NewToolTip"));
}

// ---------------------------------------------------------- file chooser

static bool on_path(const char *prog) {
    const char *path = getenv("PATH");
    if (!path) return false;
    char dir[1024];
    for (const char *p = path; *p;) {
        const char *c = strchr(p, ':');
        size_t l = c ? (size_t)(c - p) : strlen(p);
        if (l && l < sizeof dir - strlen(prog) - 2) {
            memcpy(dir, p, l);
            snprintf(dir + l, sizeof dir - l, "/%s", prog);
            if (access(dir, X_OK) == 0) return true;
        }
        if (!c) break;
        p = c + 1;
    }
    return false;
}

// org.freedesktop.DBus.NameHasOwner as a method call with a timeout of our own (review finding 4):
// dbus_bus_name_has_owner waits libdbus's default 25 s on a wedged bus, and this runs inside a serve's start
static bool has_owner(const tdb_api *d, tdb_conn *c, const char *name, int timeout_ms) {
    tdb_msg *m = d->msg_new_method_call("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus",
                                        "NameHasOwner");
    if (!m) return false;
    tdb_iter it;
    d->iter_init_append(m, &it);
    tdb_error e;
    d->error_init(&e);
    tdb_msg *r = tdb_append_str(d, &it, TDB_STRING, name) ? d->conn_send_with_reply_and_block(c, m, timeout_ms, &e)
                                                          : NULL;
    d->msg_unref(m);
    if (d->error_is_set(&e)) d->error_free(&e);
    tdb_bool has = 0;
    tdb_iter ri;
    if (r && d->iter_init(r, &ri) && d->iter_get_arg_type(&ri) == TDB_BOOL) d->iter_get_basic(&ri, &has);
    if (r) d->msg_unref(r);
    return has != 0;
}

static bool portal_present(void) { return has_owner(D, g_bus, PORTAL, 1000); }

// percent-decode a file:// URI into a path; false for any other scheme
static bool uri_path(const char *uri, char *out, size_t cap) {
    if (strncmp(uri, "file://", 7)) return false;
    const char *p = uri + 7;
    if (*p != '/') return false;   // file://host/... is not a local path
    size_t o = 0;
    for (; *p && o + 1 < cap; p++) {
        if (*p == '%' && p[1] && p[2]) {
            char h[3] = { p[1], p[2], 0 };
            char *end;
            long v = strtol(h, &end, 16);
            if (*end || v <= 0) return false;
            out[o++] = (char)v;
            p += 2;
        } else {
            out[o++] = *p;
        }
    }
    out[o] = 0;
    return *p == 0;
}

// zenity or kdialog, run from an argument vector (never a shell). The loop
// reads its answer as it comes (dialog_poll), so the bus keeps being served
// while the dialog is open.
static void pick_with_dialog(void) {
    char *const zen[] = { "zenity", "--file-selection", "--title=Choose a model", "--file-filter=*.gguf", NULL };
    char *const kde[] = { "kdialog", "--getopenfilename", ".", "*.gguf", NULL };
    char *const *argv = on_path("zenity") ? zen : on_path("kdialog") ? kde : NULL;
    if (!argv) { tray_menu_act(TRAY_ACT_PICK_MODEL, 0, NULL); return; }
    int fd[2];
    if (pipe(fd) != 0) { tray_menu_act(TRAY_ACT_PICK_MODEL, 0, NULL); return; }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fd[1], 1);
    posix_spawn_file_actions_addclose(&fa, fd[0]);
    pid_t pid;
    int rc = posix_spawnp(&pid, argv[0], &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(fd[1]);
    if (rc != 0) { close(fd[0]); tray_menu_act(TRAY_ACT_PICK_MODEL, 0, NULL); return; }
    fcntl(fd[0], F_SETFL, fcntl(fd[0], F_GETFL) | O_NONBLOCK);
    g_dlg_pid = pid;
    g_dlg_fd = fd[0];
    g_dlg_n = 0;
}

// the open dialog, from the loop: take what it wrote; when it has exited, act on its answer
static void dialog_poll(void) {
    if (g_dlg_pid <= 0) return;
    ssize_t r;
    while (g_dlg_n + 1 < sizeof g_dlg_buf &&
           (r = read(g_dlg_fd, g_dlg_buf + g_dlg_n, sizeof g_dlg_buf - 1 - g_dlg_n)) > 0)
        g_dlg_n += (size_t)r;
    int st = 0;
    if (waitpid(g_dlg_pid, &st, WNOHANG) != g_dlg_pid) return;
    while (g_dlg_n + 1 < sizeof g_dlg_buf &&
           (r = read(g_dlg_fd, g_dlg_buf + g_dlg_n, sizeof g_dlg_buf - 1 - g_dlg_n)) > 0)
        g_dlg_n += (size_t)r;
    close(g_dlg_fd);
    g_dlg_fd = -1;
    g_dlg_pid = -1;
    size_t n = WIFEXITED(st) && WEXITSTATUS(st) == 0 ? g_dlg_n : 0;
    g_dlg_buf[n] = 0;
    while (n && (g_dlg_buf[n - 1] == '\n' || g_dlg_buf[n - 1] == '\r')) g_dlg_buf[--n] = 0;
    tray_menu_act(TRAY_ACT_PICK_MODEL, 0, n ? g_dlg_buf : NULL);
    menu_refresh();
    icon_refresh();
}

// org.freedesktop.portal.FileChooser.OpenFile: asynchronous; the answer is a
// Response signal on the request path, which the filter takes
static void pick_with_portal(void) {
    // the request path is predictable from our unique name and a token, so
    // the match is in place before the call can answer
    char token[32], sender[96], handle[256];
    snprintf(token, sizeof token, "runner%ld", (long)time(NULL) % 100000);
    snprintf(sender, sizeof sender, "%s", D->bus_get_unique_name(g_bus) + 1);   // past ':'
    for (char *c = sender; *c; c++) if (*c == '.') *c = '_';
    snprintf(handle, sizeof handle, "/org/freedesktop/portal/desktop/request/%s/%s", sender, token);
    char rule[400];
    snprintf(rule, sizeof rule, "type='signal',interface='org.freedesktop.portal.Request',member='Response',path='%s'",
             handle);
    tdb_error e;
    D->error_init(&e);
    D->bus_add_match(g_bus, rule, &e);
    if (D->error_is_set(&e)) { D->error_free(&e); pick_with_dialog(); return; }
    tdb_msg *m = D->msg_new_method_call(PORTAL, "/org/freedesktop/portal/desktop",
                                        "org.freedesktop.portal.FileChooser", "OpenFile");
    if (!m) { pick_with_dialog(); return; }
    tdb_iter it, opts, ent, v, filters, f, pats, pat;
    D->iter_init_append(m, &it);
    bool ok = tdb_append_str(D, &it, TDB_STRING, "") && tdb_append_str(D, &it, TDB_STRING, "Choose a model") &&
              D->iter_open_container(&it, TDB_ARRAY, "{sv}", &opts) &&
              dict_entry_begin(&opts, &ent, "handle_token") && var_str(&ent, TDB_STRING, token) &&
              D->iter_close_container(&opts, &ent) &&
              dict_entry_begin(&opts, &ent, "filters") &&
              D->iter_open_container(&ent, TDB_VARIANT, "a(sa(us))", &v) &&
              D->iter_open_container(&v, TDB_ARRAY, "(sa(us))", &filters) &&
              D->iter_open_container(&filters, TDB_STRUCT, NULL, &f) &&
              tdb_append_str(D, &f, TDB_STRING, "GGUF models") &&
              D->iter_open_container(&f, TDB_ARRAY, "(us)", &pats) &&
              D->iter_open_container(&pats, TDB_STRUCT, NULL, &pat) && tdb_append_u32(D, &pat, 0) &&
              tdb_append_str(D, &pat, TDB_STRING, "*.gguf") && D->iter_close_container(&pats, &pat) &&
              D->iter_close_container(&f, &pats) && D->iter_close_container(&filters, &f) &&
              D->iter_close_container(&v, &filters) && D->iter_close_container(&ent, &v) &&
              D->iter_close_container(&opts, &ent) && D->iter_close_container(&it, &opts);
    tdb_msg *r = ok ? D->conn_send_with_reply_and_block(g_bus, m, 5000, &e) : NULL;
    D->msg_unref(m);
    if (!r) { if (D->error_is_set(&e)) D->error_free(&e); pick_with_dialog(); return; }
    // the portal may name a different path than predicted (an older portal): follow its answer
    tdb_iter ri;
    const char *path = NULL;
    if (D->iter_init(r, &ri) && D->iter_get_arg_type(&ri) == TDB_PATH) D->iter_get_basic(&ri, &path);
    snprintf(g_pick_handle, sizeof g_pick_handle, "%s", path ? path : handle);
    if (path && strcmp(path, handle)) {
        snprintf(rule, sizeof rule,
                 "type='signal',interface='org.freedesktop.portal.Request',member='Response',path='%s'", path);
        D->bus_add_match(g_bus, rule, &e);
        if (D->error_is_set(&e)) D->error_free(&e);
    }
    D->msg_unref(r);
}

static void portal_response(tdb_msg *m) {
    g_pick_handle[0] = 0;
    tdb_iter it, arr, ent, var, uris;
    uint32_t response = 2;
    char picked[1100] = "";
    if (D->iter_init(m, &it) && D->iter_get_arg_type(&it) == TDB_UINT32) {
        D->iter_get_basic(&it, &response);
        if (response == 0 && D->iter_next(&it) && D->iter_get_arg_type(&it) == TDB_ARRAY) {
            for (D->iter_recurse(&it, &arr); D->iter_get_arg_type(&arr) == TDB_DICT_ENTRY; D->iter_next(&arr)) {
                const char *key = NULL;
                D->iter_recurse(&arr, &ent);
                if (D->iter_get_arg_type(&ent) != TDB_STRING) continue;
                D->iter_get_basic(&ent, &key);
                if (!key || strcmp(key, "uris") || !D->iter_next(&ent)) continue;
                D->iter_recurse(&ent, &var);
                if (D->iter_get_arg_type(&var) != TDB_ARRAY) continue;
                D->iter_recurse(&var, &uris);
                const char *u = NULL;
                if (D->iter_get_arg_type(&uris) == TDB_STRING) D->iter_get_basic(&uris, &u);
                if (u && !uri_path(u, picked, sizeof picked)) picked[0] = 0;
            }
        }
    }
    tray_menu_act(TRAY_ACT_PICK_MODEL, 0, picked[0] ? picked : NULL);
}

// --------------------------------------------------------- the dispatcher

static void reply_error(tdb_msg *m, const char *name, const char *text) {
    emit(D->msg_new_error(m, name, text));
}

static void act_on(int id) {
    const int ix = idx_of(id);
    if (ix < 0) return;                              // a row no longer in the menu (or never): nothing acts
    const tray_item *t = &g_pub[ix];
    if (t->kind != TRAY_K_ACTION && t->kind != TRAY_K_CHECK) return;
    if (t->action == TRAY_ACT_PICK_MODEL) {
        if (g_picker_none) return;
        if (g_pick_handle[0] || g_dlg_pid > 0) return;   // one dialog at a time
        if (portal_present()) pick_with_portal();
        else pick_with_dialog();
        return;
    }
    tray_menu_act(t->action, t->arg, NULL);
}

static const char INTROSPECT_SNI[] =
    "<!DOCTYPE node PUBLIC \"-//freedesktop//DTD D-BUS Object Introspection 1.0//EN\" "
    "\"http://www.freedesktop.org/standards/dbus/1.0/introspect.dtd\">"
    "<node><interface name=\"org.kde.StatusNotifierItem\">"
    "<property name=\"Category\" type=\"s\" access=\"read\"/><property name=\"Id\" type=\"s\" access=\"read\"/>"
    "<property name=\"Title\" type=\"s\" access=\"read\"/><property name=\"Status\" type=\"s\" access=\"read\"/>"
    "<property name=\"WindowId\" type=\"i\" access=\"read\"/><property name=\"IconName\" type=\"s\" access=\"read\"/>"
    "<property name=\"IconPixmap\" type=\"a(iiay)\" access=\"read\"/>"
    "<property name=\"OverlayIconName\" type=\"s\" access=\"read\"/>"
    "<property name=\"AttentionIconName\" type=\"s\" access=\"read\"/>"
    "<property name=\"ToolTip\" type=\"(sa(iiay)ss)\" access=\"read\"/>"
    "<property name=\"ItemIsMenu\" type=\"b\" access=\"read\"/><property name=\"Menu\" type=\"o\" access=\"read\"/>"
    "<method name=\"ContextMenu\"><arg name=\"x\" type=\"i\" direction=\"in\"/><arg name=\"y\" type=\"i\" direction=\"in\"/></method>"
    "<method name=\"Activate\"><arg name=\"x\" type=\"i\" direction=\"in\"/><arg name=\"y\" type=\"i\" direction=\"in\"/></method>"
    "<method name=\"SecondaryActivate\"><arg name=\"x\" type=\"i\" direction=\"in\"/><arg name=\"y\" type=\"i\" direction=\"in\"/></method>"
    "<method name=\"Scroll\"><arg name=\"delta\" type=\"i\" direction=\"in\"/><arg name=\"orientation\" type=\"s\" direction=\"in\"/></method>"
    "<signal name=\"NewIcon\"/><signal name=\"NewToolTip\"/><signal name=\"NewStatus\"><arg type=\"s\"/></signal>"
    "</interface></node>";

static const char INTROSPECT_MENU[] =
    "<!DOCTYPE node PUBLIC \"-//freedesktop//DTD D-BUS Object Introspection 1.0//EN\" "
    "\"http://www.freedesktop.org/standards/dbus/1.0/introspect.dtd\">"
    "<node><interface name=\"com.canonical.dbusmenu\">"
    "<property name=\"Version\" type=\"u\" access=\"read\"/><property name=\"TextDirection\" type=\"s\" access=\"read\"/>"
    "<property name=\"Status\" type=\"s\" access=\"read\"/><property name=\"IconThemePath\" type=\"as\" access=\"read\"/>"
    "<method name=\"GetLayout\"><arg type=\"i\" direction=\"in\"/><arg type=\"i\" direction=\"in\"/>"
    "<arg type=\"as\" direction=\"in\"/><arg type=\"u\" direction=\"out\"/><arg type=\"(ia{sv}av)\" direction=\"out\"/></method>"
    "<method name=\"GetGroupProperties\"><arg type=\"ai\" direction=\"in\"/><arg type=\"as\" direction=\"in\"/>"
    "<arg type=\"a(ia{sv})\" direction=\"out\"/></method>"
    "<method name=\"GetProperty\"><arg type=\"i\" direction=\"in\"/><arg type=\"s\" direction=\"in\"/>"
    "<arg type=\"v\" direction=\"out\"/></method>"
    "<method name=\"Event\"><arg type=\"i\" direction=\"in\"/><arg type=\"s\" direction=\"in\"/>"
    "<arg type=\"v\" direction=\"in\"/><arg type=\"u\" direction=\"in\"/></method>"
    "<method name=\"EventGroup\"><arg type=\"a(isvu)\" direction=\"in\"/><arg type=\"ai\" direction=\"out\"/></method>"
    "<method name=\"AboutToShow\"><arg type=\"i\" direction=\"in\"/><arg type=\"b\" direction=\"out\"/></method>"
    "<method name=\"AboutToShowGroup\"><arg type=\"ai\" direction=\"in\"/><arg type=\"ai\" direction=\"out\"/>"
    "<arg type=\"ai\" direction=\"out\"/></method>"
    "<signal name=\"ItemsPropertiesUpdated\"><arg type=\"a(ia{sv})\"/><arg type=\"a(ias)\"/></signal>"
    "<signal name=\"LayoutUpdated\"><arg type=\"u\"/><arg type=\"i\"/></signal>"
    "</interface></node>";

static int arg_i32(tdb_iter *it, int32_t *out) {
    if (D->iter_get_arg_type(it) != TDB_INT32) return 0;
    D->iter_get_basic(it, out);
    D->iter_next(it);
    return 1;
}
static const char *arg_str(tdb_iter *it) {
    const char *s = NULL;
    if (D->iter_get_arg_type(it) != TDB_STRING) return NULL;
    D->iter_get_basic(it, &s);
    D->iter_next(it);
    return s;
}

static void handle_properties(tdb_msg *m, const char *path, const char *member) {
    bool sni = !strcmp(path, SNI_PATH);
    tdb_iter in;
    D->iter_init(m, &in);
    const char *iface = arg_str(&in);
    const char *want = sni ? SNI_IFACE : MENU_IFACE;
    tdb_msg *r = NULL;
    tdb_iter out;
    bool known = false, ok = false;
    if (!strcmp(member, "Get")) {
        const char *p = arg_str(&in);
        if (!iface || !p || strcmp(iface, want)) { reply_error(m, "org.freedesktop.DBus.Error.UnknownInterface", "no such interface"); return; }
        r = D->msg_new_method_return(m);
        if (!r) return;
        D->iter_init_append(r, &out);
        ok = sni ? sni_prop(&out, p, &known) : menu_prop(&out, p, &known);
        if (!known) { D->msg_unref(r); reply_error(m, "org.freedesktop.DBus.Error.UnknownProperty", p); return; }
    } else if (!strcmp(member, "GetAll")) {
        r = D->msg_new_method_return(m);
        if (!r) return;
        D->iter_init_append(r, &out);
        tdb_iter dict, e;
        ok = D->iter_open_container(&out, TDB_ARRAY, "{sv}", &dict);
        if (iface && !strcmp(iface, want)) {
            const char *const *names = sni ? SNI_PROPS : MENU_PROPS;
            for (int i = 0; ok && names[i]; i++)
                ok = dict_entry_begin(&dict, &e, names[i]) &&
                     (sni ? sni_prop(&e, names[i], &known) : menu_prop(&e, names[i], &known)) &&
                     D->iter_close_container(&dict, &e);
        }
        ok = ok && D->iter_close_container(&out, &dict);
    } else {
        reply_error(m, "org.freedesktop.DBus.Error.PropertyReadOnly", "read-only properties");
        return;
    }
    if (ok) emit(r); else if (r) D->msg_unref(r);
}

static void handle_menu(tdb_msg *m, const char *member) {
    tdb_iter in, out;
    D->iter_init(m, &in);
    tdb_msg *r = NULL;
    if (!strcmp(member, "GetLayout")) {
        int32_t parent = 0, depth = -1;
        arg_i32(&in, &parent);
        arg_i32(&in, &depth);
        if (parent < 0 || (parent > 0 && idx_of(parent) < 0)) {
            reply_error(m, "org.freedesktop.DBus.Error.InvalidArgs", "no such item");
            return;
        }
        r = D->msg_new_method_return(m);
        if (!r) return;
        D->iter_init_append(r, &out);
        if (tdb_append_u32(D, &out, g_rev) && put_node(&out, parent, depth)) emit(r);
        else D->msg_unref(r);
        return;
    }
    if (!strcmp(member, "GetGroupProperties")) {
        tdb_iter ids;
        r = D->msg_new_method_return(m);
        if (!r) return;
        D->iter_init_append(r, &out);
        tdb_iter arr;
        bool ok = D->iter_open_container(&out, TDB_ARRAY, "(ia{sv})", &arr);
        if (D->iter_get_arg_type(&in) == TDB_ARRAY) {
            for (D->iter_recurse(&in, &ids); ok && D->iter_get_arg_type(&ids) == TDB_INT32; D->iter_next(&ids)) {
                int32_t id;
                D->iter_get_basic(&ids, &id);
                if (id < 0 || (id > 0 && idx_of(id) < 0)) continue;
                tdb_iter st, dict;
                ok = D->iter_open_container(&arr, TDB_STRUCT, NULL, &st) && tdb_append_i32(D, &st, id) &&
                     D->iter_open_container(&st, TDB_ARRAY, "{sv}", &dict) && item_props(&dict, id) &&
                     D->iter_close_container(&st, &dict) && D->iter_close_container(&arr, &st);
            }
        }
        if (ok && D->iter_close_container(&out, &arr)) emit(r); else D->msg_unref(r);
        return;
    }
    if (!strcmp(member, "GetProperty")) {
        int32_t id = -1;
        arg_i32(&in, &id);
        const char *name = arg_str(&in);
        const int ix = idx_of(id);
        if (ix < 0 || !name) { reply_error(m, "org.freedesktop.DBus.Error.InvalidArgs", "no such item"); return; }
        // one property, out of the item's full set
        r = D->msg_new_method_return(m);
        if (!r) return;
        D->iter_init_append(r, &out);
        bool ok = false;
        if (!strcmp(name, "label")) {
            char label[1100];
            menu_label(label, sizeof label, g_pub[ix].label);
            ok = var_str(&out, TDB_STRING, label);
        } else if (!strcmp(name, "enabled")) {
            ok = var_bool(&out, g_pub[ix].kind != TRAY_K_LABEL);
        } else if (!strcmp(name, "toggle-state")) {
            ok = var_i32(&out, g_pub[ix].checked ? 1 : 0);
        }
        if (ok) emit(r);
        else { D->msg_unref(r); reply_error(m, "org.freedesktop.DBus.Error.InvalidArgs", "no such property"); }
        return;
    }
    if (!strcmp(member, "Event") || !strcmp(member, "EventGroup")) {
        // answer first: a click's action may take seconds (a stop's wait)
        int click_ids[TRAY_MAX_ITEMS], nclick = 0;
        if (!strcmp(member, "Event")) {
            int32_t id = -1;
            arg_i32(&in, &id);
            const char *ev = arg_str(&in);
            if (ev && !strcmp(ev, "clicked")) click_ids[nclick++] = id;
            r = D->msg_new_method_return(m);
        } else {
            tdb_iter arr, st;
            if (D->iter_get_arg_type(&in) == TDB_ARRAY)
                for (D->iter_recurse(&in, &arr); D->iter_get_arg_type(&arr) == TDB_STRUCT && nclick < TRAY_MAX_ITEMS;
                     D->iter_next(&arr)) {
                    int32_t id = -1;
                    D->iter_recurse(&arr, &st);
                    arg_i32(&st, &id);
                    const char *ev = arg_str(&st);
                    if (ev && !strcmp(ev, "clicked")) click_ids[nclick++] = id;
                }
            r = D->msg_new_method_return(m);
            if (r) {
                tdb_iter o, errs;
                D->iter_init_append(r, &o);
                if (!D->iter_open_container(&o, TDB_ARRAY, "i", &errs) || !D->iter_close_container(&o, &errs)) {
                    D->msg_unref(r);
                    r = NULL;
                }
            }
        }
        emit(r);
        D->conn_flush(g_bus);
        for (int i = 0; i < nclick; i++) act_on(click_ids[i]);
        if (!tray_should_quit()) { menu_refresh(); icon_refresh(); }
        return;
    }
    if (!strcmp(member, "AboutToShow")) {
        bool changed = menu_refresh();
        r = D->msg_new_method_return(m);
        if (!r) return;
        D->iter_init_append(r, &out);
        if (tdb_append_bool(D, &out, changed)) emit(r); else D->msg_unref(r);
        return;
    }
    if (!strcmp(member, "AboutToShowGroup")) {
        bool changed = menu_refresh();
        r = D->msg_new_method_return(m);
        if (!r) return;
        D->iter_init_append(r, &out);
        tdb_iter a, b;
        bool ok = D->iter_open_container(&out, TDB_ARRAY, "i", &a) && (!changed || tdb_append_i32(D, &a, 0)) &&
                  D->iter_close_container(&out, &a) && D->iter_open_container(&out, TDB_ARRAY, "i", &b) &&
                  D->iter_close_container(&out, &b);
        if (ok) emit(r); else D->msg_unref(r);
        return;
    }
    reply_error(m, "org.freedesktop.DBus.Error.UnknownMethod", member);
}

static void register_item(void) {
    tdb_msg *m = D->msg_new_method_call(WATCHER, "/StatusNotifierWatcher", WATCHER, "RegisterStatusNotifierItem");
    if (!m) return;
    tdb_iter it;
    D->iter_init_append(m, &it);
    if (tdb_append_str(D, &it, TDB_STRING, g_name)) {
        tdb_error e;
        D->error_init(&e);
        tdb_msg *r = D->conn_send_with_reply_and_block(g_bus, m, 5000, &e);
        if (r) D->msg_unref(r);
        if (D->error_is_set(&e)) {
            fprintf(stderr, "tray: the tray host refused the item (%s)\n", e.message ? e.message : "no reason");
            D->error_free(&e);
        }
    }
    D->msg_unref(m);
}

static int filter(tdb_conn *c, tdb_msg *m, void *ud) {
    (void)c; (void)ud;
    int type = D->msg_get_type(m);
    const char *iface = D->msg_get_interface(m), *member = D->msg_get_member(m), *path = D->msg_get_path(m);
    if (type == TDB_MSG_SIGNAL && iface && member) {
        if (!strcmp(iface, "org.freedesktop.DBus") && !strcmp(member, "NameOwnerChanged")) {
            // a tray host that restarts (a shell restart, an extension reload) gets the item again
            tdb_iter it;
            const char *name = NULL, *old_owner = NULL, *new_owner = NULL;
            if (D->iter_init(m, &it)) {
                name = arg_str(&it);
                old_owner = arg_str(&it);
                new_owner = arg_str(&it);
            }
            (void)old_owner;
            if (name && !strcmp(name, WATCHER)) {
                if (new_owner && *new_owner) {
                    g_host_gone_at = 0;           // a host (back): the item again
                    register_item();
                } else if (!g_host_gone_at) {
                    g_host_gone_at = now_mono();  // gone: the loop ends the tray if it stays gone
                }
            }
            return TDB_HANDLED;
        }
        if (!strcmp(iface, "org.freedesktop.portal.Request") && !strcmp(member, "Response") && path &&
            g_pick_handle[0] && !strcmp(path, g_pick_handle)) {
            portal_response(m);
            menu_refresh();
            icon_refresh();
            return TDB_HANDLED;
        }
        return TDB_NOT_YET_HANDLED;
    }
    if (type != TDB_MSG_CALL || !member || !path) return TDB_NOT_YET_HANDLED;
    bool sni = !strcmp(path, SNI_PATH), menu = !strcmp(path, MENU_PATH);
    if (!sni && !menu) return TDB_NOT_YET_HANDLED;
    if (iface && !strcmp(iface, "org.freedesktop.DBus.Introspectable") && !strcmp(member, "Introspect")) {
        tdb_msg *r = D->msg_new_method_return(m);
        tdb_iter o;
        if (r) {
            D->iter_init_append(r, &o);
            if (tdb_append_str(D, &o, TDB_STRING, sni ? INTROSPECT_SNI : INTROSPECT_MENU)) emit(r);
            else D->msg_unref(r);
        }
        return TDB_HANDLED;
    }
    if (iface && !strcmp(iface, PROPS_IFACE)) { handle_properties(m, path, member); return TDB_HANDLED; }
    if (menu) { handle_menu(m, member); return TDB_HANDLED; }
    // the item's four methods: the menu is the item (ItemIsMenu), so all are no-ops
    if (!strcmp(member, "Activate") || !strcmp(member, "SecondaryActivate") || !strcmp(member, "ContextMenu") ||
        !strcmp(member, "Scroll")) {
        emit(D->msg_new_method_return(m));
        return TDB_HANDLED;
    }
    reply_error(m, "org.freedesktop.DBus.Error.UnknownMethod", member);
    return TDB_HANDLED;
}

// ------------------------------------------------------------- the loop

static void on_signal(int sig) { (void)sig; g_signalled = 1; }

static int tray_fail(const char *why) {
    fprintf(stderr, "error: --tray: %s\n", why);
    return 1;
}

int tray_platform_run(void) {
    if (!tdb_session_bus_export()) return tray_fail("no D-Bus session bus: the tray needs a desktop session");
    const char *why = NULL;
    D = tdb_load(&why);
    if (!D) return tray_fail(why);
    tdb_error e;
    D->error_init(&e);
    g_bus = D->bus_get_private(TDB_BUS_SESSION, &e);
    if (!g_bus) {
        char msg[400];
        snprintf(msg, sizeof msg, "no D-Bus session bus: the tray needs a desktop session (%s)",
                 D->error_is_set(&e) && e.message ? e.message : "the bus refused the connection");
        if (D->error_is_set(&e)) D->error_free(&e);
        return tray_fail(msg);
    }
    // a lost bus must come back here, to end the way Quit ends, not exit()
    D->conn_set_exit_on_disconnect(g_bus, 0);
    bool watcher = has_owner(D, g_bus, WATCHER, 2000);
    if (!watcher) {
        D->conn_close(g_bus);
        D->conn_unref(g_bus);
        return tray_fail("no tray host on the session bus (GNOME needs the AppIndicator extension)");
    }
    snprintf(g_name, sizeof g_name, "org.kde.StatusNotifierItem-%ld-1", (long)getpid());
    int rq = D->bus_request_name(g_bus, g_name, TDB_NAME_DO_NOT_QUEUE, &e);
    if (D->error_is_set(&e) || rq != TDB_NAME_PRIMARY_OWNER) {
        if (D->error_is_set(&e)) D->error_free(&e);
        D->conn_close(g_bus);
        D->conn_unref(g_bus);
        return tray_fail("the session bus refused the tray's name");
    }
    D->bus_add_match(g_bus, "type='signal',sender='org.freedesktop.DBus',interface='org.freedesktop.DBus',"
                            "member='NameOwnerChanged',arg0='" WATCHER "'", &e);
    if (D->error_is_set(&e)) D->error_free(&e);
    if (!D->conn_add_filter(g_bus, filter, NULL, NULL)) {
        D->conn_close(g_bus);
        D->conn_unref(g_bus);
        return tray_fail("out of memory setting up the bus");
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGHUP, &sa, NULL);

    g_picker_none = !portal_present() && !on_path("zenity") && !on_path("kdialog");
    // TEST-ONLY: RUNNER_TRAY_TEST_HOST_GRACE_MS shortens the window a tray host may be gone (tests/test_tray_linux.c)
    const char *gv = getenv("RUNNER_TRAY_TEST_HOST_GRACE_MS");
    if (gv && *gv && atol(gv) > 0) g_host_grace_s = atol(gv) / 1000.0;
    g_icon = tray_icon();
    menu_refresh();
    register_item();

    int rc = 0;
    double next = now_mono() + 2.0;
    while (!tray_should_quit()) {
        if (g_signalled) { tray_menu_act(TRAY_ACT_QUIT, 0, NULL); break; }
        if (!D->conn_read_write_dispatch(g_bus, 200)) {
            // the bus is gone (a logout): end the way Quit ends it, the managed instance with it
            tray_menu_act(TRAY_ACT_QUIT, 0, NULL);
            rc = 1;
            break;
        }
        dialog_poll();
        if (g_host_gone_at && now_mono() - g_host_gone_at >= g_host_grace_s) {
            // the host stayed gone: no icon will show, so the tray ends as Quit ends it. The core's Quit stops only
            // the server this tray started; one someone else started keeps running
            fprintf(stderr, "tray: tray host gone for %.0f s; ending as Quit does\n", g_host_grace_s);
            tray_menu_act(TRAY_ACT_QUIT, 0, NULL);
            rc = 1;
            break;
        }
        if (now_mono() >= next) {
            next = now_mono() + 2.0;
            icon_refresh();
            menu_refresh();
            D->conn_flush(g_bus);
        }
    }
    if (g_dlg_pid > 0) { kill(g_dlg_pid, SIGTERM); waitpid(g_dlg_pid, NULL, 0); close(g_dlg_fd); g_dlg_pid = -1; }
    if (D->conn_get_is_connected(g_bus)) {
        D->conn_flush(g_bus);
        D->conn_close(g_bus);
    }
    D->conn_unref(g_bus);
    g_bus = NULL;
    return rc;
}

// ------------------------------------------------------------- autostart

static bool autostart_path(char *out, size_t cap) {
    const char *xdg = getenv("XDG_CONFIG_HOME"), *home = getenv("HOME");
    int n;
    if (xdg && *xdg && xdg[0] == '/') n = snprintf(out, cap, "%s/autostart/xyntetik-runner-tray.desktop", xdg);
    else if (home && *home) n = snprintf(out, cap, "%s/.config/autostart/xyntetik-runner-tray.desktop", home);
    else return false;
    return n > 0 && (size_t)n < cap;
}

bool tray_platform_autostart_get(void) {
    char p[1100];
    if (!autostart_path(p, sizeof p)) return false;
    FILE *f = fopen(p, "r");
    if (!f) return false;
    char line[1300];
    bool hidden = false;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "Hidden=true", 11)) hidden = true;
    fclose(f);
    return !hidden;
}

// The Exec line's path, quoted by the Desktop Entry rules: inside double
// quotes, '"', '`', '$' and '\' take a backslash; then the string-value rule
// doubles every backslash; '%' is a field code, written "%%". A path with a
// newline or a control byte has no spelling there and is refused.
static bool exec_quote(const char *path, char *out, size_t cap) {
    size_t o = 0;
    if (cap < 3) return false;
    out[o++] = '"';
    for (const char *p = path; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (c < 0x20 || c == 0x7f) return false;
        const char *rep = c == '"' ? "\\\\\"" : c == '`' ? "\\\\`" : c == '$' ? "\\\\$" : c == '\\' ? "\\\\\\\\"
                        : c == '%' ? "%%" : NULL;
        size_t l = rep ? strlen(rep) : 1;
        if (o + l + 2 >= cap) return false;
        if (rep) { memcpy(out + o, rep, l); o += l; }
        else out[o++] = (char)c;
    }
    out[o++] = '"';
    out[o] = 0;
    return true;
}

static bool mkdirs_for(const char *file) {
    char d[1100];
    snprintf(d, sizeof d, "%s", file);
    for (char *p = d + 1; *p; p++) {
        if (*p != '/') continue;
        *p = 0;
        if (mkdir(d, 0700) != 0 && errno != EEXIST) return false;
        *p = '/';
    }
    return true;
}

bool tray_platform_autostart_set(bool on) {
    char p[1100];
    if (!autostart_path(p, sizeof p)) return false;
    if (!on) return unlink(p) == 0 || errno == ENOENT;
    // the binary's real path: plat_executable_path() says /proc/self/exe on
    // Linux, which names whatever process reads it, never this binary
    char *self = realpath("/proc/self/exe", NULL);
    if (!self || self[0] != '/') { free(self); return false; }
    char quoted[2400];
    bool q = exec_quote(self, quoted, sizeof quoted);
    free(self);
    if (!q || !mkdirs_for(p)) return false;
    char tmp[1200];
    snprintf(tmp, sizeof tmp, "%s.tmp-%ld", p, (long)getpid());
    FILE *f = fopen(tmp, "w");
    if (!f) return false;
    int w = fprintf(f, "[Desktop Entry]\nType=Application\nName=Runner\n"
                       "Comment=xyntetik-runner's tray: the local runners, a model to start\n"
                       "Exec=%s --tray\nTerminal=false\nX-GNOME-Autostart-enabled=true\n", quoted);
    bool ok = w > 0 && fflush(f) == 0;
    ok = fclose(f) == 0 && ok;
    if (!ok || rename(tmp, p) != 0) { unlink(tmp); return false; }
    return true;
}

// ------------------------------------------------------------ icon dump

// A PNG with stored (uncompressed) deflate blocks: enough to look at the
// glyph without a display, and what the auto-raise test reads.
static unsigned long png_crc(const unsigned char *b, size_t n, unsigned long c) {
    static unsigned long t[256];
    static bool made;
    if (!made) {
        for (unsigned long i = 0; i < 256; i++) {
            unsigned long v = i;
            for (int k = 0; k < 8; k++) v = v & 1 ? 0xedb88320UL ^ (v >> 1) : v >> 1;
            t[i] = v;
        }
        made = true;
    }
    c ^= 0xffffffffUL;
    for (size_t i = 0; i < n; i++) c = t[(c ^ b[i]) & 0xff] ^ (c >> 8);
    return c ^ 0xffffffffUL;
}
static void be32(unsigned char *p, unsigned long v) {
    p[0] = (unsigned char)(v >> 24); p[1] = (unsigned char)(v >> 16); p[2] = (unsigned char)(v >> 8); p[3] = (unsigned char)v;
}
static bool png_chunk(FILE *f, const char *type, const unsigned char *data, size_t n) {
    unsigned char h[8];
    be32(h, (unsigned long)n);
    memcpy(h + 4, type, 4);
    unsigned long c = png_crc(h + 4, 4, 0);
    if (n) c = png_crc(data, n, c);
    unsigned char t[4];
    be32(t, c);
    return fwrite(h, 1, 8, f) == 8 && (!n || fwrite(data, 1, n, f) == n) && fwrite(t, 1, 4, f) == 4;
}

static bool png_write(const char *path, const unsigned char *bgra, int px) {
    size_t row = 1 + (size_t)px * 4, raw_n = row * (size_t)px;
    unsigned char *raw = malloc(raw_n);
    if (!raw) return false;
    for (int y = 0; y < px; y++) {
        unsigned char *r = raw + row * (size_t)y;
        r[0] = 0;
        for (int x = 0; x < px; x++) {
            const unsigned char *s = bgra + ((size_t)y * px + x) * 4;
            unsigned a = s[3], R = s[2], G = s[1], B = s[0];
            if (a) { R = (R * 255 + a / 2) / a; G = (G * 255 + a / 2) / a; B = (B * 255 + a / 2) / a; }
            r[1 + x * 4] = (unsigned char)(R > 255 ? 255 : R);
            r[2 + x * 4] = (unsigned char)(G > 255 ? 255 : G);
            r[3 + x * 4] = (unsigned char)(B > 255 ? 255 : B);
            r[4 + x * 4] = (unsigned char)a;
        }
    }
    size_t blocks = (raw_n + 65534) / 65535, z_n = 2 + raw_n + blocks * 5 + 4;
    unsigned char *z = malloc(z_n);
    if (!z) { free(raw); return false; }
    size_t o = 0;
    z[o++] = 0x78; z[o++] = 0x01;
    unsigned long s1 = 1, s2 = 0;
    for (size_t i = 0; i < raw_n; i++) { s1 = (s1 + raw[i]) % 65521; s2 = (s2 + s1) % 65521; }
    for (size_t at = 0; at < raw_n; at += 65535) {
        size_t n = raw_n - at < 65535 ? raw_n - at : 65535;
        z[o++] = at + n >= raw_n ? 1 : 0;
        z[o++] = (unsigned char)n; z[o++] = (unsigned char)(n >> 8);
        z[o++] = (unsigned char)~n; z[o++] = (unsigned char)(~n >> 8);
        memcpy(z + o, raw + at, n);
        o += n;
    }
    be32(z + o, (s2 << 16) | s1);
    o += 4;
    unsigned char ihdr[13];
    be32(ihdr, (unsigned long)px);
    be32(ihdr + 4, (unsigned long)px);
    ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    FILE *f = fopen(path, "wb");
    bool ok = f && fwrite("\x89PNG\r\n\x1a\n", 1, 8, f) == 8 && png_chunk(f, "IHDR", ihdr, 13) &&
              png_chunk(f, "IDAT", z, o) && png_chunk(f, "IEND", NULL, 0);
    if (f) ok = fclose(f) == 0 && ok;
    free(z);
    free(raw);
    return ok;
}

bool tray_platform_icon_dump(const char *dir, int px) {
    static const char *const names[] = { "tray-idle.png", "tray-loaded.png", "tray-running.png" };
    unsigned char *bgra = malloc((size_t)px * px * 4);
    if (!bgra) return false;
    bool ok = true;
    for (int s = 0; ok && s < 3; s++) {
        char p[1100];
        ok = snprintf(p, sizeof p, "%s/%s", dir, names[s]) < (int)sizeof p &&
             tray_glyph_render((tray_icon_state)s, px, 0x202020u, bgra) && png_write(p, bgra, px);
    }
    free(bgra);
    return ok;
}

// ---------------------------------------- the auto-raise question (main.c)

bool tray_linux_session_ok(void) {
    if (!tdb_session_bus_named()) return false;
    const char *w = getenv("WAYLAND_DISPLAY"), *x = getenv("DISPLAY");
    if (!(w && *w) && !(x && *x)) return false;
    const char *why = NULL;
    const tdb_api *d = tdb_load(&why);
    if (!d) return false;   // no library, no tray: never spawn one that would only fail
    tdb_session_bus_export();
    tdb_error e;
    d->error_init(&e);
    tdb_conn *c = d->bus_get_private(TDB_BUS_SESSION, &e);
    if (!c) { if (d->error_is_set(&e)) d->error_free(&e); return false; }
    d->conn_set_exit_on_disconnect(c, 0);
    bool has = has_owner(d, c, WATCHER, 1000);
    d->conn_close(c);
    d->conn_unref(c);
    return has;
}
#endif // __linux__
