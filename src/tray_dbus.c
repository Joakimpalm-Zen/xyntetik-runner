// libdbus-1 at run time for the Linux tray (tray_dbus.h).
#if defined(__linux__)
#include "tray_dbus.h"

#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static tdb_api g_api;
static const tdb_api *g_loaded;
static char g_why[256];

const tdb_api *tdb_load(const char **why) {
    if (g_loaded) return g_loaded;
    if (g_why[0]) { if (why) *why = g_why; return NULL; }
    void *h = dlopen("libdbus-1.so.3", RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        snprintf(g_why, sizeof g_why, "the D-Bus library libdbus-1.so.3 could not be loaded (%s)", dlerror());
        if (why) *why = g_why;
        return NULL;
    }
    // every symbol resolved here, before any call; one missing disables the tray
    static const struct { const char *name; size_t off; } syms[] = {
#define S(field, sym) { sym, offsetof(tdb_api, field) }
        S(error_init, "dbus_error_init"),
        S(error_free, "dbus_error_free"),
        S(error_is_set, "dbus_error_is_set"),
        S(bus_get_private, "dbus_bus_get_private"),
        S(bus_get_unique_name, "dbus_bus_get_unique_name"),
        S(bus_request_name, "dbus_bus_request_name"),
        S(bus_name_has_owner, "dbus_bus_name_has_owner"),
        S(bus_add_match, "dbus_bus_add_match"),
        S(conn_set_exit_on_disconnect, "dbus_connection_set_exit_on_disconnect"),
        S(conn_add_filter, "dbus_connection_add_filter"),
        S(conn_read_write_dispatch, "dbus_connection_read_write_dispatch"),
        S(conn_send, "dbus_connection_send"),
        S(conn_flush, "dbus_connection_flush"),
        S(conn_send_with_reply_and_block, "dbus_connection_send_with_reply_and_block"),
        S(conn_close, "dbus_connection_close"),
        S(conn_unref, "dbus_connection_unref"),
        S(conn_get_is_connected, "dbus_connection_get_is_connected"),
        S(msg_new_method_call, "dbus_message_new_method_call"),
        S(msg_new_method_return, "dbus_message_new_method_return"),
        S(msg_new_signal, "dbus_message_new_signal"),
        S(msg_new_error, "dbus_message_new_error"),
        S(msg_unref, "dbus_message_unref"),
        S(msg_get_type, "dbus_message_get_type"),
        S(msg_get_interface, "dbus_message_get_interface"),
        S(msg_get_member, "dbus_message_get_member"),
        S(msg_get_path, "dbus_message_get_path"),
        S(msg_get_sender, "dbus_message_get_sender"),
        S(iter_init, "dbus_message_iter_init"),
        S(iter_init_append, "dbus_message_iter_init_append"),
        S(iter_get_arg_type, "dbus_message_iter_get_arg_type"),
        S(iter_get_basic, "dbus_message_iter_get_basic"),
        S(iter_next, "dbus_message_iter_next"),
        S(iter_recurse, "dbus_message_iter_recurse"),
        S(iter_get_fixed_array, "dbus_message_iter_get_fixed_array"),
        S(iter_append_basic, "dbus_message_iter_append_basic"),
        S(iter_append_fixed_array, "dbus_message_iter_append_fixed_array"),
        S(iter_open_container, "dbus_message_iter_open_container"),
        S(iter_close_container, "dbus_message_iter_close_container"),
#undef S
    };
    // the table's size is the count of entries above: a field added without
    // its symbol would stay NULL, which this refuses at build time
    _Static_assert(sizeof syms / sizeof syms[0] * sizeof(void (*)(void)) == sizeof(tdb_api),
                   "every tdb_api entry has its symbol");
    tdb_api api;
    memset(&api, 0, sizeof api);
    for (size_t i = 0; i < sizeof syms / sizeof syms[0]; i++) {
        void *p = dlsym(h, syms[i].name);
        if (!p) {
            snprintf(g_why, sizeof g_why, "the D-Bus library libdbus-1.so.3 has no %s", syms[i].name);
            dlclose(h);
            if (why) *why = g_why;
            return NULL;
        }
        memcpy((char *)&api + syms[i].off, &p, sizeof p);
    }
    g_api = api;
    g_loaded = &g_api;
    return g_loaded;
}

bool tdb_session_bus_named(void) {
    const char *a = getenv("DBUS_SESSION_BUS_ADDRESS");
    if (a && *a) return true;
    const char *rt = getenv("XDG_RUNTIME_DIR");
    if (!rt || !*rt) return false;
    char p[1024];
    if (snprintf(p, sizeof p, "%s/bus", rt) >= (int)sizeof p) return false;
    struct stat st;
    return stat(p, &st) == 0 && S_ISSOCK(st.st_mode);
}

bool tdb_session_bus_export(void) {
    if (!tdb_session_bus_named()) return false;
    const char *a = getenv("DBUS_SESSION_BUS_ADDRESS");
    if (a && *a) return true;
    char addr[1100];
    if (snprintf(addr, sizeof addr, "unix:path=%s/bus", getenv("XDG_RUNTIME_DIR")) >= (int)sizeof addr) return false;
    return setenv("DBUS_SESSION_BUS_ADDRESS", addr, 1) == 0;
}

bool tdb_append_str(const tdb_api *d, tdb_iter *it, int type, const char *s) {
    const char *v = s ? s : "";
    return d->iter_append_basic(it, type, &v);
}
bool tdb_append_i32(const tdb_api *d, tdb_iter *it, int32_t v) { return d->iter_append_basic(it, TDB_INT32, &v); }
bool tdb_append_u32(const tdb_api *d, tdb_iter *it, uint32_t v) { return d->iter_append_basic(it, TDB_UINT32, &v); }
bool tdb_append_bool(const tdb_api *d, tdb_iter *it, bool v) {
    tdb_bool b = v ? 1 : 0;
    return d->iter_append_basic(it, TDB_BOOL, &b);
}
#endif // __linux__
