// Tray core gate: menu building against the registry, config-driven managed
// spawn, stop escalation, and quit-stops-managed-only semantics — all
// headless (links the stub backend on every platform). Re-invokes ITSELF as
// the managed child: spawn_managed launches self_exe with --serve args, and
// child mode writes a marker file with its pid+argv then idles until killed.
#include "instances.h"
#include "tray.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#include <process.h>
#include <sys/utime.h>
#define getpid _getpid
#define setenv_compat(k, v) _putenv_s(k, v)
static void msleep(int ms) { Sleep(ms); }
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <utime.h>
#include <unistd.h>
#define setenv_compat(k, v) setenv(k, v, 1)
static void msleep(int ms) {
    struct timespec ts = { ms / 1000, (ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}
#endif

bool tray_http_get_for_test(int port, const char *path, char *out, int cap);

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); fails++; } \
} while (0)

static char g_home[512];

static void marker_path(char *out, size_t cap) {
    snprintf(out, cap, "%s/spawned.txt", g_home);
}

static bool menu_has(const tray_item *it, int n, const char *needle) {
    for (int i = 0; i < n; i++)
        if (strstr(it[i].label, needle)) return true;
    return false;
}

static void pin_mtime(const char *path, time_t stamp) {
#ifdef _WIN32
    struct __utimbuf64 t = { (__time64_t)stamp, (__time64_t)stamp };
    CHECK(_utime64(path, &t) == 0, "test fixture pins config mtime");
#else
    struct utimbuf t = { stamp, stamp };
    CHECK(utime(path, &t) == 0, "test fixture pins config mtime");
#endif
}

#ifndef _WIN32
static void check_tray_http_short_write(void) {
    int listener = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(listener >= 0, "short-write fixture opens a listener");
    if (listener < 0) return;
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    CHECK(bind(listener, (struct sockaddr *)&addr, sizeof addr) == 0,
          "short-write fixture binds loopback");
    CHECK(listen(listener, 1) == 0, "short-write fixture listens");
    socklen_t alen = sizeof addr;
    CHECK(getsockname(listener, (struct sockaddr *)&addr, &alen) == 0,
          "short-write fixture resolves its port");

    pid_t child = fork();
    CHECK(child >= 0, "short-write fixture forks a peer");
    if (child == 0) {
        int client = accept(listener, NULL, NULL);
        char req[512] = {0};
        size_t n = 0;
        while (client >= 0 && n < sizeof(req) - 1 &&
               !strstr(req, "\r\n\r\n")) {
            ssize_t r = read(client, req + n, sizeof(req) - 1 - n);
            if (r <= 0) break;
            n += (size_t)r;
            req[n] = 0;
        }
        bool complete = strstr(req, "GET /health HTTP/1.1\r\n") != NULL &&
                        strstr(req, "\r\n\r\n") != NULL;
        static const char response[] =
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
            "Connection: close\r\n\r\n{}";
        if (complete) write(client, response, sizeof(response) - 1);
        if (client >= 0) close(client);
        close(listener);
        _exit(complete ? 0 : 3);
    }
    if (child < 0) { close(listener); return; }

    setenv_compat("RUNNER_TEST_TRAY_SEND_CHUNK", "7");
    char response[512];
    bool ok = tray_http_get_for_test((int)ntohs(addr.sin_port), "/health",
                                     response, sizeof response);
    unsetenv("RUNNER_TEST_TRAY_SEND_CHUNK");
    close(listener);
    int status = 0;
    waitpid(child, &status, 0);
    CHECK(ok, "tray loopback GET survives short stream writes");
    CHECK(ok && strstr(response, "\r\n\r\n{}") != NULL,
          "tray loopback GET receives the response after a short write");
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "tray sent the complete request after a short write");
}
#endif

int main(int argc, char **argv) {
    // ---- child mode: we ARE the "managed runner" the core spawned
    if (argc > 1 && strcmp(argv[1], "--serve") == 0) {
        const char *home = getenv("XYNTETIK_TEST_HOME");
        if (!home) return 2;
        char mp[600];
        snprintf(mp, sizeof mp, "%s/spawned.txt", home);
        FILE *f = fopen(mp, "wb");
        if (!f) return 2;
        fprintf(f, "%ld\n", (long)getpid());
        for (int i = 1; i < argc; i++) fprintf(f, "%s\n", argv[i]);
        fclose(f);
        msleep(30000);  // idle until the controller stops us
        return 0;
    }

#ifndef _WIN32
    check_tray_http_short_write();
#endif

    // ---- parent mode: private registry + config under a fake HOME
    snprintf(g_home, sizeof g_home,
#ifdef _WIN32
             "%s\\xyntetik-tray-test-%ld", getenv("TEMP"), (long)getpid());
    _mkdir(g_home);
    setenv_compat("APPDATA", g_home);
#else
             "/tmp/xyntetik-tray-test-%ld", (long)getpid());
    mkdir(g_home, 0755);
    setenv_compat("HOME", g_home);
#endif
    setenv_compat("XYNTETIK_TEST_HOME", g_home);

    // The core re-reads config.json whenever it has changed on disk, but the
    // file must still exist BEFORE the first menu build for the start row to
    // be offered. instances_dir() is <root>/instances; config.json is a sibling.
    char cfg[700];
    const char *idir = instances_dir();
    CHECK(idir != NULL, "instances dir resolves under fake HOME");
    snprintf(cfg, sizeof cfg, "%s/../config.json", idir);
    // raw backslashes are invalid JSON escapes, so the Windows TEMP path
    // must go into config.json with forward slashes (fopen accepts both)
    char home_json[512];
    snprintf(home_json, sizeof home_json, "%s", g_home);
    for (char *c = home_json; *c; c++) if (*c == '\\') *c = '/';
    FILE *f = fopen(cfg, "wb");
    fprintf(f, "{\"last_model\": \"%s/fake-model.gguf\","
               " \"last_args\": \"-c 512\", \"port\": 8127}\n", home_json);
    fclose(f);
    char mdl[600];
    snprintf(mdl, sizeof mdl, "%s/fake-model.gguf", home_json);
    f = fopen(mdl, "wb"); fputs("x", f); fclose(f);

    // 1. empty registry: setup rows present, no stop rows
    tray_item items[128];
    int n = tray_menu_build(items, 128);
    CHECK(n > 0, "menu builds");
    CHECK(menu_has(items, n, "no runners active"), "empty state row present");
    CHECK(!menu_has(items, n, "Stop"), "no stop rows with empty registry");

    // 1b. a menu that does not fit the caller's array must be a PREFIX of the
    // one that does, and must stay inside the array.
    //
    // Every row's text is written by the snprintf AFTER the row, and that
    // snprintf ran whether or not the row was placed. So a dropped row
    // re-labelled the last row that WAS placed: the backend rendered "Quit
    // controller" on a row whose action was Stop. At cap 0 nothing is ever
    // placed and the same writes land on it[0] and it[-1], outside the
    // caller's array in both directions.
    {
        tray_item full[128];
        int nfull = tray_menu_build(full, 128);
        CHECK(nfull > 3, "enough rows to truncate");
        for (int cap2 = 1; cap2 < nfull; cap2++) {
            tray_item guard[136];
            memset(guard, 0x5A, sizeof guard);
            tray_item *part = guard + 4;
            int k = tray_menu_build(part, cap2);
            CHECK(k == cap2, "a truncated build reports exactly cap rows");
            for (int i = 0; i < k && i < nfull; i++)
                if (strcmp(part[i].label, full[i].label) ||
                    part[i].kind != full[i].kind ||
                    part[i].action != full[i].action ||
                    part[i].arg != full[i].arg) {
                    CHECK(false, "truncated menu is a prefix of the full menu");
                    break;
                }
        }
        tray_item guard0[9];
        memset(guard0, 0x5A, sizeof guard0);
        int z = tray_menu_build(guard0 + 4, 0);
        CHECK(z == 0, "cap 0 builds no rows");
        bool clean = true;
        for (size_t b = 0; b < sizeof guard0; b++)
            if (((unsigned char *)guard0)[b] != 0x5A) clean = false;
        CHECK(clean, "cap 0 writes nothing, above or below the array");
    }

    // 2. configured model: start row is offered and START_MANAGED spawns
    // self --serve with the configured argv
    CHECK(menu_has(items, n, "Start default runner"), "configured start row");
    CHECK(menu_has(items, n, "fake-model.gguf"), "start row names the model");

    // 2a. a config edited behind the tray's back is picked up at the next menu
    // build. This used to need a tray restart (workmac issue 5), and the tray
    // is a long-lived process, so "restart it" is the whole cost of the bug.
    //
    // Deliberately written with no delay: st_mtime is whole seconds on some
    // filesystems, so an edit landing inside the same second as the read above
    // is invisible to the timestamp alone. The two model names differ in
    // length, which is what the size half of the check is there for.
    char alt[600];
    snprintf(alt, sizeof alt, "%s/edited-model.gguf", home_json);
    f = fopen(alt, "wb"); fputs("x", f); fclose(f);
    f = fopen(cfg, "wb");
    fprintf(f, "{\"last_model\": \"%s/edited-model.gguf\","
               " \"last_args\": \"-c 512\", \"port\": 8127}\n", home_json);
    fclose(f);
    n = tray_menu_build(items, 128);
    CHECK(menu_has(items, n, "edited-model.gguf"),
          "externally edited config is visible without a tray restart");
    CHECK(!menu_has(items, n, "fake-model.gguf"),
          "the stale model is gone, not merely joined by the new one");

    // put the original back; every step below asserts against it
    f = fopen(cfg, "wb");
    fprintf(f, "{\"last_model\": \"%s/fake-model.gguf\","
               " \"last_args\": \"-c 512\", \"port\": 8127}\n", home_json);
    fclose(f);
    n = tray_menu_build(items, 128);
    CHECK(menu_has(items, n, "fake-model.gguf"), "restored config re-read too");

    // Editors commonly publish an intermediate invalid file and then the
    // repaired bytes within one timestamp tick. The invalid observation must
    // not consume the (mtime,size) identity, or an equal-sized repair remains
    // invisible for the lifetime of the tray.
    char repaired_model[600];
    snprintf(repaired_model, sizeof repaired_model, "%s/repaired-model.gguf",
             home_json);
    f = fopen(repaired_model, "wb"); fputs("x", f); fclose(f);
    char repaired[1600];
    int repaired_n = snprintf(
        repaired, sizeof repaired,
        "{\"last_model\": \"%s\", \"last_args\": \"-c 512\", "
        "\"port\": 8127}\n", repaired_model);
    CHECK(repaired_n > 0 && (size_t)repaired_n < sizeof repaired,
          "repaired config fixture fits");
    char malformed[1600];
    memcpy(malformed, repaired, (size_t)repaired_n + 1);
    malformed[0] = '!';
    time_t pinned = time(NULL) + 120;
    f = fopen(cfg, "wb"); fwrite(malformed, 1, (size_t)repaired_n, f); fclose(f);
    pin_mtime(cfg, pinned);
    n = tray_menu_build(items, 128);
    CHECK(menu_has(items, n, "fake-model.gguf"),
          "invalid edit retains the last-known-good config");
    f = fopen(cfg, "wb"); fwrite(repaired, 1, (size_t)repaired_n, f); fclose(f);
    pin_mtime(cfg, pinned);
    n = tray_menu_build(items, 128);
    CHECK(menu_has(items, n, "repaired-model.gguf"),
          "equal-size equal-mtime repair is re-read after an invalid edit");

    // Restore the original again for the lifecycle assertions below.
    f = fopen(cfg, "wb");
    fprintf(f, "{\"last_model\": \"%s/fake-model.gguf\","
               " \"last_args\": \"-c 512\", \"port\": 8127}\n", home_json);
    fclose(f);
    n = tray_menu_build(items, 128);
    CHECK(menu_has(items, n, "fake-model.gguf"),
          "config after repair remains reloadable");
    remove(repaired_model);

    // A failed config install must keep both the durable last-known-good file
    // and the tray's in-memory selection. Otherwise the menu claims the new
    // model was saved until restart, while a partial config can erase the old
    // model permanently.
    remove(alt); // keep PICK_MODEL from spawning a child in the failure case
    setenv_compat("RUNNER_TEST_TRAY_CONFIG_INSTALL_FAIL", "1");
    tray_menu_act(TRAY_ACT_PICK_MODEL, 0, alt);
#ifdef _WIN32
    _putenv_s("RUNNER_TEST_TRAY_CONFIG_INSTALL_FAIL", "");
#else
    unsetenv("RUNNER_TEST_TRAY_CONFIG_INSTALL_FAIL");
#endif
    n = tray_menu_build(items, 128);
    CHECK(menu_has(items, n, "fake-model.gguf"),
          "failed config install restores the in-memory model");
    CHECK(!menu_has(items, n, "edited-model.gguf"),
          "failed config install does not expose an unsaved model");
    {
        char saved[1024] = {0};
        FILE *cf = fopen(cfg, "rb");
        size_t got = cf ? fread(saved, 1, sizeof saved - 1, cf) : 0;
        if (cf) fclose(cf);
        saved[got] = 0;
        CHECK(strstr(saved, "fake-model.gguf") != NULL,
              "failed config install preserves the durable config");
        CHECK(strstr(saved, "edited-model.gguf") == NULL,
              "failed config install never publishes partial new state");
        char partial[800];
        snprintf(partial, sizeof partial, "%s.partial-%ld", cfg, (long)getpid());
        cf = fopen(partial, "rb");
        CHECK(cf == NULL, "failed config install removes its partial file");
        if (cf) fclose(cf);
    }

    // 2a-bis. a configured model that no longer exists on disk must read as
    // MISSING, never as a crash. Publishing a checkpoint and reclaiming the
    // local copy is routine here, and before this the tray spawned a doomed
    // child, the child died at GGUF open, and the menu said "exited (failed
    // start or crash)" — blaming the engine for a stale pointer. Proven red by
    // reverting the MG_MISSING branch: the Start row comes back and the spawn
    // is attempted against a file that is not there.
    remove(mdl);
    n = tray_menu_build(items, 128);
    CHECK(menu_has(items, n, "model file missing"),
          "a deleted model is named as missing");
    CHECK(menu_has(items, n, "Choose another model"),
          "the menu offers the fix instead of a doomed start");
    CHECK(!menu_has(items, n, "Start default runner"),
          "no start row for a model that cannot load");
    CHECK(!menu_has(items, n, "no runners active"),
          "the missing-model row replaces the empty-state row");
    {
        char mp0[600];
        marker_path(mp0, sizeof mp0);
        remove(mp0);
        tray_menu_act(TRAY_ACT_START_MANAGED, 0, NULL);
        msleep(300);
        FILE *m0 = fopen(mp0, "rb");
        CHECK(m0 == NULL, "START_MANAGED refuses to spawn a missing model");
        if (m0) fclose(m0);
    }
    // restore it: every step below asserts against a model that exists, and
    // the state must heal on its own since MISSING is derived, not stored
    f = fopen(mdl, "wb"); fputs("x", f); fclose(f);
    n = tray_menu_build(items, 128);
    CHECK(menu_has(items, n, "Start default runner"),
          "restoring the file heals the menu with no tray restart");

    tray_menu_act(TRAY_ACT_START_MANAGED, 0, NULL);

    char mp[600];
    marker_path(mp, sizeof mp);
    long child = 0;
    char args[512] = "";
    for (int t = 0; t < 60; t++) {  // up to 6 s for the child to come up
        FILE *m = fopen(mp, "rb");
        if (m) {
            char line[512];
            if (fgets(line, sizeof line, m)) child = atol(line);
            while (fgets(line, sizeof line, m)) {
                line[strcspn(line, "\r\n")] = 0;
                strncat(args, line, sizeof args - strlen(args) - 2);
                strncat(args, " ", sizeof args - strlen(args) - 1);
            }
            fclose(m);
            if (child > 0) break;
        }
        msleep(100);
    }
    CHECK(child > 0, "managed child spawned and wrote its marker");
    CHECK(strstr(args, "--serve") != NULL, "child got --serve");
    CHECK(strstr(args, "fake-model.gguf") != NULL, "child got the configured model");
    CHECK(strstr(args, "--port 8127") != NULL, "child got the configured port");
    CHECK(strstr(args, "-c 512") != NULL, "child got last_args tail");
    CHECK(instance_pid_alive(child), "child is alive before quit");

    // 2b. the child never registers (it is not a real runner), so the core
    // must report STARTING: visible row, cancel option, no live Start row
    n = tray_menu_build(items, 128);
    CHECK(menu_has(items, n, "starting fake-model.gguf"), "starting row shown");
    CHECK(menu_has(items, n, "Cancel start"), "cancel available while starting");
    CHECK(menu_has(items, n, "Default runner: starting"), "start row disabled to starting");
    CHECK(!menu_has(items, n, "Start default runner"), "no live start row while starting");

    // 2c. kill the child behind the core's back: menu must flip to the
    // exited warning with a log row and a restart offer
#ifdef _WIN32
    {
        HANDLE h = OpenProcess(PROCESS_TERMINATE, FALSE, (DWORD)child);
        CHECK(h != NULL, "can open child to simulate a crash");
        if (h) { TerminateProcess(h, 1); CloseHandle(h); }
    }
#else
    kill((pid_t)child, SIGKILL);
#endif
    bool exited_row = false;
    for (int t = 0; t < 50 && !exited_row; t++) {
        msleep(100);
        n = tray_menu_build(items, 128);
        exited_row = menu_has(items, n, "exited");
    }
    CHECK(exited_row, "crash surfaces as the exited warning");
    CHECK(menu_has(items, n, "View log"), "log row offered after exit");
    CHECK(menu_has(items, n, "Restart default runner"), "restart offered after exit");

    // 2d. restart spawns a fresh child and clears the warning
    remove(mp);
    tray_menu_act(TRAY_ACT_START_MANAGED, 0, NULL);
    child = 0;
    for (int t = 0; t < 60 && child == 0; t++) {
        FILE *m = fopen(mp, "rb");
        if (m) {
            char line[512];
            if (fgets(line, sizeof line, m)) child = atol(line);
            fclose(m);
        }
        msleep(100);
    }
    CHECK(child > 0, "restart spawned a fresh child");
    n = tray_menu_build(items, 128);
    CHECK(!menu_has(items, n, "exited"), "warning cleared once restarted");

    // 3. a foreign record must survive quit; only the managed child stops
    char foreign[700];
    snprintf(foreign, sizeof foreign, "%s/%ld.json", idir, (long)getpid());
    f = fopen(foreign, "wb");
    fprintf(f, "{\"pid\": %ld, \"started\": 1, \"mode\": \"serve\","
               " \"port\": 9001, \"version\": \"t\", \"models\": []}\n",
            (long)getpid());  // our own pid: always alive, never our child
    fclose(f);

    tray_menu_act(TRAY_ACT_QUIT, 0, NULL);
    CHECK(tray_should_quit(), "quit flag set");

    bool child_dead = false;
    for (int t = 0; t < 50; t++) {
        if (!instance_pid_alive(child)) { child_dead = true; break; }
        msleep(100);
    }
    CHECK(child_dead, "managed child stopped by quit");

    int nl = 0;
    instance_rec *r = instances_list(&nl);
    bool foreign_alive = false;
    for (int i = 0; i < nl; i++)
        if (r[i].port == 9001) foreign_alive = true;
    instances_list_free(r, nl);
    CHECK(foreign_alive, "foreign instance record untouched by quit");

    // 4. after a user-initiated quit the menu is back to a clean Start —
    // no lingering exited warning for an instance the user stopped
    n = tray_menu_build(items, 128);
    CHECK(menu_has(items, n, "Start default runner"), "start row back after quit");
    CHECK(!menu_has(items, n, "exited"), "no exited warning after intentional stop");

    // 5. tray_ensure_running() is idempotent. A tray now follows every --serve
    // and -i session, so this runs constantly and MUST NOT stack processes;
    // the one-tray-per-machine guard is the child's backstop, not the caller's
    // licence to spawn.
    //
    // Only the already-registered branch is exercised, deliberately. The spawn
    // branch re-execs argv[0] with --tray, and argv[0] here is this test — it
    // would recurse into the suite rather than start a tray. The spawn path is
    // covered by the manual checks recorded in the commit.
    instances_register("tray", 0, NULL, NULL, 0);
    int before = 0;
    instances_list_free(instances_list(&before), before);
    tray_ensure_running();
    msleep(300);
    int after = 0;
    instances_list_free(instances_list(&after), after);
    CHECK(after == before, "no second tray spawned when one is already registered");
    instances_unregister();

    remove(foreign);
    remove(mp);
    remove(cfg);
    remove(mdl);

    if (fails) { fprintf(stderr, "test_tray_core: %d FAILURES\n", fails); return 1; }
    printf("tray core tests ok\n");
    return 0;
}
