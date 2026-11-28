// The session bus for Linux's tray (R18.15): libdbus-1, loaded at --tray time.
//
// No -dev package is needed to build Runner and no DT_NEEDED entry is added:
// libdbus-1.so.3 is dlopen'd when a tray starts, every function resolved
// into tdb_api below, and a machine without it loses only the tray (one line
// naming the library). The declarations are Runner's own, written from the
// libdbus API as documented, not copied header text. The few structs a
// caller must allocate (an error, a message iterator) are declared with the
// real ones' size and field offsets; tests/test_tray_linux.c builds against
// the real <dbus/dbus.h> and static-asserts both.
#ifndef RUNNER_TRAY_DBUS_H
#define RUNNER_TRAY_DBUS_H

#include <stdbool.h>
#include <stdint.h>

typedef struct tdb_conn tdb_conn;   // DBusConnection, opaque
typedef struct tdb_msg  tdb_msg;    // DBusMessage, opaque
typedef uint32_t        tdb_bool;   // dbus_bool_t

// DBusError: two public strings, five one-bit flags in one unsigned int, a
// pointer of padding
typedef struct {
    const char  *name;
    const char  *message;
    unsigned int flags;
    void        *reserved;
} tdb_error;

// DBusMessageIter on an LP64 target: two pointers, a uint32, nine ints,
// two pointers (a caller only allocates it; libdbus owns the meaning)
typedef struct {
    void    *a, *b;
    uint32_t c;
    int      d[9];
    void    *e, *f;
} tdb_iter;

// the type codes (the wire format's ASCII letters)
enum {
    TDB_BYTE = 'y', TDB_BOOL = 'b', TDB_INT32 = 'i', TDB_UINT32 = 'u',
    TDB_STRING = 's', TDB_PATH = 'o', TDB_VARIANT = 'v', TDB_ARRAY = 'a',
    TDB_STRUCT = 'r', TDB_DICT_ENTRY = 'e', TDB_INVALID = 0,
};
enum { TDB_MSG_CALL = 1, TDB_MSG_RETURN = 2, TDB_MSG_ERROR = 3, TDB_MSG_SIGNAL = 4 };
enum { TDB_HANDLED = 0, TDB_NOT_YET_HANDLED = 1 };
enum { TDB_BUS_SESSION = 0 };
enum { TDB_NAME_DO_NOT_QUEUE = 4, TDB_NAME_PRIMARY_OWNER = 1 };

typedef int  (*tdb_filter_fn)(tdb_conn *, tdb_msg *, void *);
typedef void (*tdb_free_fn)(void *);

typedef struct {
    void      (*error_init)(tdb_error *);
    void      (*error_free)(tdb_error *);
    tdb_bool  (*error_is_set)(const tdb_error *);
    tdb_conn *(*bus_get_private)(int, tdb_error *);
    const char *(*bus_get_unique_name)(tdb_conn *);
    int       (*bus_request_name)(tdb_conn *, const char *, unsigned, tdb_error *);
    tdb_bool  (*bus_name_has_owner)(tdb_conn *, const char *, tdb_error *);
    void      (*bus_add_match)(tdb_conn *, const char *, tdb_error *);
    void      (*conn_set_exit_on_disconnect)(tdb_conn *, tdb_bool);
    tdb_bool  (*conn_add_filter)(tdb_conn *, tdb_filter_fn, void *, tdb_free_fn);
    tdb_bool  (*conn_read_write_dispatch)(tdb_conn *, int);
    tdb_bool  (*conn_send)(tdb_conn *, tdb_msg *, uint32_t *);
    void      (*conn_flush)(tdb_conn *);
    tdb_msg  *(*conn_send_with_reply_and_block)(tdb_conn *, tdb_msg *, int, tdb_error *);
    void      (*conn_close)(tdb_conn *);
    void      (*conn_unref)(tdb_conn *);
    tdb_bool  (*conn_get_is_connected)(tdb_conn *);
    tdb_msg  *(*msg_new_method_call)(const char *, const char *, const char *, const char *);
    tdb_msg  *(*msg_new_method_return)(tdb_msg *);
    tdb_msg  *(*msg_new_signal)(const char *, const char *, const char *);
    tdb_msg  *(*msg_new_error)(tdb_msg *, const char *, const char *);
    void      (*msg_unref)(tdb_msg *);
    int       (*msg_get_type)(tdb_msg *);
    const char *(*msg_get_interface)(tdb_msg *);
    const char *(*msg_get_member)(tdb_msg *);
    const char *(*msg_get_path)(tdb_msg *);
    const char *(*msg_get_sender)(tdb_msg *);
    tdb_bool  (*iter_init)(tdb_msg *, tdb_iter *);
    void      (*iter_init_append)(tdb_msg *, tdb_iter *);
    int       (*iter_get_arg_type)(tdb_iter *);
    void      (*iter_get_basic)(tdb_iter *, void *);
    tdb_bool  (*iter_next)(tdb_iter *);
    void      (*iter_recurse)(tdb_iter *, tdb_iter *);
    void      (*iter_get_fixed_array)(tdb_iter *, void *, int *);
    tdb_bool  (*iter_append_basic)(tdb_iter *, int, const void *);
    tdb_bool  (*iter_append_fixed_array)(tdb_iter *, int, const void *, int);
    tdb_bool  (*iter_open_container)(tdb_iter *, int, const char *, tdb_iter *);
    tdb_bool  (*iter_close_container)(tdb_iter *, tdb_iter *);
} tdb_api;

// Load libdbus-1.so.3 and resolve every entry of tdb_api. On failure *why
// names the library or the missing symbol and NULL is returned; no entry is
// ever NULL in a returned table, so no call through a null pointer is
// reachable. Loaded once per process.
const tdb_api *tdb_load(const char **why);

// The session bus as the environment names it: DBUS_SESSION_BUS_ADDRESS, or
// $XDG_RUNTIME_DIR/bus when that socket exists. False when neither does
// (libdbus would otherwise try an X11 autolaunch, which a headless box has
// no use for).
bool tdb_session_bus_named(void);

// small helpers over a loaded table
bool tdb_append_str(const tdb_api *d, tdb_iter *it, int type, const char *s);
bool tdb_append_i32(const tdb_api *d, tdb_iter *it, int32_t v);
bool tdb_append_u32(const tdb_api *d, tdb_iter *it, uint32_t v);
bool tdb_append_bool(const tdb_api *d, tdb_iter *it, bool v);

#endif // RUNNER_TRAY_DBUS_H
