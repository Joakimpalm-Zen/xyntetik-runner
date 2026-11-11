// The non-blocking client writer the device turn uses (R3.36). A stream written
// while the turn is held must never block on the client, must lose no byte and
// reorder none, and must stop at a stated bound. Loopback TCP, not a socketpair,
// which Windows lacks. No GPU and no model.
//
// The anchor: the byte counts and the pattern are computed here by hand from
// the inputs, not read back from the writer.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "http.h"
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <unistd.h>
#endif

static int failures;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); failures++; } } while (0)

static double now(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

// a connected pair: *w is the server's end, *r the client's
static void pair(sock_t *w, sock_t *r) {
    sock_t l = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t al = sizeof a;
    if (bind(l, (struct sockaddr *)&a, sizeof a) || listen(l, 1) ||
        getsockname(l, (struct sockaddr *)&a, &al)) { puts("no loopback"); exit(1); }
    *r = socket(AF_INET, SOCK_STREAM, 0);
    int small = 2048;
    setsockopt(*r, SOL_SOCKET, SO_RCVBUF, (const char *)&small, sizeof small);
    if (connect(*r, (struct sockaddr *)&a, sizeof a)) { puts("no connect"); exit(1); }
    *w = accept(l, NULL, NULL);
    sock_close(l);
}

// byte i of the stream is i mod 251: a prime, so a chunk boundary shows
static void fill(char *b, size_t at, size_t n) {
    for (size_t i = 0; i < n; i++) b[i] = (char)((at + i) % 251);
}

// Writes CHUNK bytes at a time until the writer refuses, peer not reading.
// Returns the bytes it accepted; the last result is in *last.
#define CHUNK 158
static size_t fill_until_refused(stream_out *o, sock_t w, int *last, double *slowest) {
    char b[CHUNK];
    size_t at = 0;
    for (long i = 0; i < 2000000; i++) {
        fill(b, at, CHUNK);
        double t0 = now();
        *last = stream_out_write(o, w, b, CHUNK);
        double d = now() - t0;
        if (d > *slowest) *slowest = d;
        if (*last != STREAM_OUT_OK) return at;
        at += CHUNK;
    }
    return at;
}

static void test_a_stalled_peer_never_blocks_and_stops_at_the_cap(void) {
    sock_t w, r;
    pair(&w, &r);
    stream_out o;
    stream_out_init(&o, 64 * 1024);
    int last = 0;
    double slowest = 0;
    size_t took = fill_until_refused(&o, w, &last, &slowest);
    CHECK(last == STREAM_OUT_FULL);
    // no write waited on the client; the socket's own blocking write would
    // have sat in its send timeout here
    CHECK(slowest < 0.5);
    CHECK(stream_out_pending(&o) <= 64 * 1024);
    CHECK(took > 0);
    stream_out_free(&o);
    sock_close(w);
    sock_close(r);
}

// the bound is exact: the cap itself fits, one byte over does not
static void test_the_cap_is_exact(void) {
    sock_t w, r;
    pair(&w, &r);
    stream_out o;
    stream_out_init(&o, 1000);
    stream_out_block_for_test(&o, true);
    char b[1001];
    fill(b, 0, sizeof b);
    CHECK(stream_out_write(&o, w, b, 1000) == STREAM_OUT_OK);
    CHECK(stream_out_pending(&o) == 1000);
    CHECK(stream_out_write(&o, w, b, 1) == STREAM_OUT_FULL);
    CHECK(stream_out_pending(&o) == 1000);
    stream_out_free(&o);
    stream_out_init(&o, 1000);
    stream_out_block_for_test(&o, true);
    CHECK(stream_out_write(&o, w, b, 1001) == STREAM_OUT_FULL);
    CHECK(stream_out_pending(&o) == 0);
    CHECK(stream_out_write(&o, w, b, 0) == STREAM_OUT_OK);
    stream_out_free(&o);
    sock_close(w);
    sock_close(r);
}

typedef struct { sock_t r; size_t want; int ok; } reader_arg;

#ifdef _WIN32
#include <windows.h>
#define THREAD_RET DWORD WINAPI
#else
#include <pthread.h>
#define THREAD_RET void *
#endif

static THREAD_RET reader(void *p) {
    reader_arg *a = p;
    char *b = malloc(1 << 16);
    size_t at = 0;
    a->ok = 1;
    while (at < a->want) {
        int n = sock_recv(a->r, b, 1 << 16);
        if (n <= 0) { a->ok = 0; break; }
        for (int i = 0; i < n; i++)
            if ((unsigned char)b[i] != (at + (size_t)i) % 251) { a->ok = 0; break; }
        at += (size_t)n;
    }
    free(b);
    return 0;
}

// every byte, in order, whatever split between the socket and the queue
static void test_a_late_reader_gets_every_byte_in_order(void) {
    sock_t w, r;
    pair(&w, &r);
    stream_out o;
    stream_out_init(&o, 4u << 20);
    enum { TOTAL = 1500 * CHUNK * 10 };
    char b[CHUNK];
    for (size_t at = 0; at < TOTAL; at += CHUNK) {
        fill(b, at, CHUNK);
        CHECK(stream_out_write(&o, w, b, CHUNK) == STREAM_OUT_OK);
    }
    reader_arg a = { r, TOTAL, 0 };
#ifdef _WIN32
    HANDLE th = CreateThread(NULL, 0, reader, &a, 0, NULL);
#else
    pthread_t th;
    pthread_create(&th, NULL, reader, &a);
#endif
    CHECK(stream_out_drain(&o, w));
    CHECK(stream_out_pending(&o) == 0);
#ifdef _WIN32
    WaitForSingleObject(th, INFINITE);
#else
    pthread_join(th, NULL);
#endif
    CHECK(a.ok == 1);
    stream_out_free(&o);
    sock_close(w);
    sock_close(r);
}

// the queue goes first, so a later write cannot overtake it
static void test_a_queued_byte_is_not_overtaken(void) {
    sock_t w, r;
    pair(&w, &r);
    stream_out o;
    stream_out_init(&o, 1000);
    stream_out_block_for_test(&o, true);
    CHECK(stream_out_write(&o, w, "ab", 2) == STREAM_OUT_OK);
    stream_out_block_for_test(&o, false);
    CHECK(stream_out_write(&o, w, "cd", 2) == STREAM_OUT_OK);
    CHECK(stream_out_pending(&o) == 0);
    char got[8] = {0};
    size_t n = 0;
    while (n < 4) {
        int k = sock_recv(r, got + n, 4 - n);
        if (k <= 0) break;
        n += (size_t)k;
    }
    CHECK(n == 4 && !memcmp(got, "abcd", 4));
    stream_out_free(&o);
    sock_close(w);
    sock_close(r);
}

// A send can be accepted until the kernel reports the peer's reset. Expect
// a few writes (well under a millisecond locally), bounded here by time.
static void test_a_closed_peer_is_gone_soon(void) {
    sock_t w, r;
    pair(&w, &r);
    sock_close(r);
    stream_out o;
    stream_out_init(&o, 1 << 20);
    int res = STREAM_OUT_OK;
    double t0 = now();
    while (res == STREAM_OUT_OK && now() - t0 < 0.5)
        res = stream_out_write(&o, w, "hello", 5);
    CHECK(res == STREAM_OUT_GONE);
    CHECK(now() - t0 < 0.5);
    stream_out_free(&o);
    sock_close(w);
}

// the socket is back to blocking afterwards: the writes after the turn rely on it
static void test_the_socket_stays_a_blocking_socket(void) {
    sock_t w, r;
    pair(&w, &r);
    stream_out o;
    stream_out_init(&o, 1 << 20);
    CHECK(stream_out_write(&o, w, "x", 1) == STREAM_OUT_OK);
    stream_out_free(&o);
    sock_send_timeout(w, 0.3);
    char big[4096];
    memset(big, 0, sizeof big);
    double t0 = now();
    bool ok = true;
    for (int i = 0; i < 4000 && ok; i++) ok = send_all(w, big, sizeof big);
    CHECK(!ok);                 // the peer never read
    CHECK(now() - t0 >= 0.2);   // and it was the timeout that ended it
    sock_close(w);
    sock_close(r);
}

// A real server socket has a write timeout, and macOS answers send(MSG_DONTWAIT)
// on a full socket only after waiting that timeout out: 30 s a write, under the
// device turn, which is how a stalled client held the turn past 300 s on Metal
// (R3.36). So this fills a real TCP socket with the timeout set, and times every
// call. The anchor is the timeout itself: no call may come near it.
static void test_a_full_socket_does_not_wait_out_the_write_timeout(void) {
    sock_t w, r;
    pair(&w, &r);
    sock_send_timeout(w, 2.0);
    stream_out o;
    stream_out_init(&o, (size_t)8 << 20);
    char frame[158];
    memset(frame, 'x', sizeof frame);
    double worst = 0;
    int after_full = 0;
    for (int i = 0; i < 40000 && after_full < 200 && worst < 1.0; i++) {
        double t0 = now();
        int rc = stream_out_write(&o, w, frame, sizeof frame);
        double took = now() - t0;
        if (took > worst) worst = took;
        CHECK(rc == STREAM_OUT_OK);
        if (stream_out_pending(&o) > 0) after_full++;
    }
    CHECK(after_full == 200);   // the socket really did fill, and the queue took over
    CHECK(worst < 1.0);         // half the timeout: one wait for it is 2 s
    stream_out_free(&o);
    sock_close(w);
    sock_close(r);
}

int main(void) {
    sock_init();
    test_a_stalled_peer_never_blocks_and_stops_at_the_cap();
    test_the_cap_is_exact();
    test_a_late_reader_gets_every_byte_in_order();
    test_a_queued_byte_is_not_overtaken();
    test_a_closed_peer_is_gone_soon();
    test_the_socket_stays_a_blocking_socket();
    test_a_full_socket_does_not_wait_out_the_write_timeout();
    if (failures) { printf("stream_out: %d failed\n", failures); return 1; }
    puts("stream_out ok");
    return 0;
}
