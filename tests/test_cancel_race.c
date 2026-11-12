// A cancel answers for the request it named, even after its slot is reused
// (R6.68.8), and a cancel waiter outlives no server it borrowed from (R6.68.9).
// HTTP is the boundary. The test-only barrier (RUNNER_TEST_CANCEL_WAIT) holds
// the waiter while real requests finish or the server stops: scheduler timing
// must not decide whether this gate bites.
// Absolute anchors: a completed request must yield HTTP 200, not 504, and a
// one-token successor must not replace the cancelled request's (>1) count.
// The stop case's anchor is the sanitizer: a waiter that reads its slot after
// the server freed it is a heap-use-after-free under ASan.
// Usage: test-cancel-race-bin MODEL [PORT] [stop]
#include "runner.h"
#include "compat.h"
#include "http.h"
#include "server.h"
#include "json.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define env_set(k, v) _putenv_s(k, v)
#else
#define env_set(k, v) setenv(k, v, 1)
#endif

static const char *model = "test.gguf";
static int port = 18107, fail, server_rc;
static atomic_int hold_waiter, waiter_entered, server_returned;

void runner_test_cancel_wait(void) {
    atomic_store(&waiter_entered, 1);
    double until = plat_now() + 10;
    while (atomic_load(&hold_waiter)) {
        if (plat_now() > until) { fputs("FAIL: waiter barrier timed out\n", stderr); exit(1); }
        plat_sleep_ms(1);
    }
}

static void ck(bool ok, const char *what) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", what); fail = 1; }
}

static void *serve(void *ud) {
    (void)ud;
    char spec[1024];
    snprintf(spec, sizeof spec, "fixture=%s", model);
    model_params p = {0};
    p.gpu_mode = GPU_OFF;
    p.n_threads = 1; p.n_ctx = 512; p.n_batch = 8;
    sampler smp = {0};
    sampler_reset(&smp);
    sampler_override ov = {0};
    sampler_resolve(&smp, NULL, NULL, -1, &ov);
    server_rc = server_run(NULL, NULL, spec, &p, smp, &ov, port, 1, 1, 0,
                          NULL, 0, false, 0, NULL, -1, false, false, -1, false, NULL);
    atomic_store(&server_returned, 1);
    return NULL;
}

static char *request(const char *path, const char *body, const char *id) {
    sock_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == SOCK_INVALID) return NULL;
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&a, sizeof a) != 0) { sock_close(fd); return NULL; }
    sock_recv_timeout(fd, 35); sock_send_timeout(fd, 5);
    char head[2048];
    int n = snprintf(head, sizeof head,
                     "%s %s HTTP/1.1\r\nHost: localhost\r\n"
                     "Content-Length: %zu\r\n%s%s%s\r\n",
                     body ? "POST" : "GET", path, body ? strlen(body) : 0,
                     id ? "X-Request-Id: " : "", id ? id : "", id ? "\r\n" : "");
    if (n < 0 || (size_t)n >= sizeof head || !send_all(fd, head, (size_t)n) ||
        (body && !send_all(fd, body, strlen(body)))) { sock_close(fd); return NULL; }
    char *out = malloc(65536);
    if (!out) { sock_close(fd); return NULL; }
    size_t got = 0;
    while (got < 65535) {
        int r = sock_recv(fd, out + got, 65535 - got);
        if (r <= 0) break;
        got += (size_t)r;
    }
    out[got] = 0; sock_close(fd);
    return out;
}

static jv *body_json(const char *r) {
    const char *b = r ? strstr(r, "\r\n\r\n") : NULL;
    return b ? json_parse(b + 4, strlen(b + 4)) : NULL;
}

static int generated(const char *r) {
    jv *v = body_json(r);
    int n = (int)jv_num(jv_get(jv_get(v, "usage"), "completion_tokens"), -1);
    jv_free(v); return n;
}

static void *completion(void *ud) {
    char **out = ud;
    *out = request("/v1/completions", "{\"model\":\"fixture\",\"prompt\":\"hi\","
                   "\"max_tokens\":64,\"ignore_eos\":true,\"temperature\":0}", "cancel-me");
    return NULL;
}

static void *cancel(void *ud) {
    char **out = ud;
    *out = request("/v1/runner/requests/cancel-me/cancel", "", NULL);
    return NULL;
}

static bool generating(void) {
    char *r = request("/health", NULL, NULL);
    jv *v = body_json(r), *rows = jv_get(v, "requests");
    bool yes = false;
    for (int i = 0; rows && i < rows->n; i++) {
        jv *q = rows->items[i];
        if (!strcmp(jv_str(jv_get(q, "phase"), ""), "generate") &&
            jv_num(jv_get(q, "generated"), 0) >= 2) yes = true;
    }
    jv_free(v); free(r); return yes;
}

int main(int argc, char **argv) {
    if (argc > 1) model = argv[1];
    if (argc > 2) port = atoi(argv[2]);
    env_set("RUNNER_TEST_STEP_DELAY_MS", "5");
    f16_init(); sock_init();
    pthread_t st, at, ct;
    if (pthread_create(&st, NULL, serve, NULL)) return 1;
    double until = plat_now() + 10;
    bool up = false;
    while (!up && plat_now() < until) {
        char *r = request("/health", NULL, NULL);
        up = r && strstr(r, "200 OK"); free(r);
        if (!up) plat_sleep_ms(5);
    }
    if (!up) { fputs("FAIL: server did not start\n", stderr); return 1; }
    bool stop_case = argc > 3 && !strcmp(argv[3], "stop");
    char *a = NULL, *c = NULL;
    atomic_store(&hold_waiter, 1);
    if (pthread_create(&at, NULL, completion, &a)) return 1;
    until = plat_now() + 10;
    bool active = false;
    while (!(active = generating()) && plat_now() < until) plat_sleep_ms(1);
    ck(active, "the cancelled request generated at least two tokens");
    if (pthread_create(&ct, NULL, cancel, &c)) return 1;
    until = plat_now() + 10;
    while (!atomic_load(&waiter_entered) && plat_now() < until) plat_sleep_ms(1);
    ck(atomic_load(&waiter_entered), "cancel reached the waiter barrier");
    if (stop_case) {
        // The server stops while the waiter is held. A server that frees its
        // slots without waiting for cancel waiters returns first, and the
        // waiter released after it reads freed memory; one that waits is
        // still in its drain when the barrier opens.
        server_request_stop();
        until = plat_now() + 2;
        while (!atomic_load(&server_returned) && plat_now() < until) plat_sleep_ms(5);
        atomic_store(&hold_waiter, 0);
        pthread_join(ct, NULL);
        pthread_join(at, NULL);
        pthread_join(st, NULL);
        ck(c && (strstr(c, "200 OK") || strstr(c, " 503 ")), "the cancel is answered across the stop");
        ck(server_rc == 0, "server stopped cleanly");
        free(a); free(c);
        if (!fail) puts("cancel race (stop): passed");
        return fail;
    }
    pthread_join(at, NULL);
    int a_gen = generated(a);
    ck(a && strstr(a, "200 OK") && a_gen >= 2, "cancelled request has its own positive token count");
    char *b = request("/v1/completions", "{\"model\":\"fixture\",\"prompt\":\"hi\","
                      "\"max_tokens\":1,\"ignore_eos\":true,\"temperature\":0}", NULL);
    ck(b && strstr(b, "200 OK") && generated(b) == 1, "the successor completed exactly one token");
    atomic_store(&hold_waiter, 0);
    pthread_join(ct, NULL);
    ck(c && strstr(c, "200 OK"), "cancel answers 200 after a successor completes");
    jv *v = body_json(c);
    ck(jv_num(jv_get(v, "generated"), -1) == a_gen, "cancel reports its own request's count");
    jv_free(v); free(a); free(b); free(c);
    server_request_stop(); pthread_join(st, NULL);
    ck(server_rc == 0, "server stopped cleanly");
    if (!fail) puts("cancel race: passed");
    return fail;
}
