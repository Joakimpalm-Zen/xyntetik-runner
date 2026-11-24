// R4.48.25.1: every served error carries its whole message. send_error_detail (src/http.c), the sink every refusal on
// every surface goes through, escaped the message into 384 bytes and the param and code into 128 each, so a longer
// reason lost its end without a word. A client then read half a sentence as the whole refusal.
//
// Over a socketpair, each error's body is parsed back and its fields compared with what was sent, byte for byte:
//   - a message over 1,500 bytes with quotes, backslashes, newlines, a tab, a control byte and UTF-8, at 400 (the
//     invalid_request_error type), with a param over 300 bytes and a code over 200;
//   - a short message at 500 (server_error), param and code null;
//   - the Content-Length header equals the body's length.
// POSIX only (socketpair); the sink is the same code on Windows.
#include "http.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
int main(void) {
    printf("skip: test_send_error uses socketpair (POSIX); the sink is the same code on Windows\n");
    return 0;
}
#else
#include <sys/socket.h>
#include <unistd.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } \
                           else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)

// send one error, read the whole response back; *body points into the returned buffer
static char *round_trip(int code, const char *msg, const char *param, const char *ecode, char **body, size_t *blen,
                        long *content_length) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) { printf("FAIL: socketpair\n"); exit(1); }
    // the response is written before it is read: a buffer larger than any body here
    int big = 1 << 20;
    setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &big, sizeof big);
    setsockopt(sv[1], SOL_SOCKET, SO_RCVBUF, &big, sizeof big);
    send_error_detail(sv[0], code, msg, param, ecode);
    close(sv[0]);
    size_t cap = 1 << 16, n = 0;
    char *buf = malloc(cap + 1);
    for (ssize_t r; buf && (r = read(sv[1], buf + n, cap - n)) > 0;) n += (size_t)r;
    close(sv[1]);
    if (!buf) { printf("FAIL: allocation\n"); exit(1); }
    buf[n] = 0;
    char *sep = strstr(buf, "\r\n\r\n");
    *body = sep ? sep + 4 : buf + n;
    *blen = (size_t)(buf + n - *body);
    const char *cl = strstr(buf, "Content-Length: ");
    *content_length = cl ? strtol(cl + 16, NULL, 10) : -1;
    return buf;
}

int main(void) {
    // whole copies of each pattern, so no UTF-8 sequence is split
    static const char mp[] = "the whole reason, \"quoted\" \\ \n\t\x01 caf\xc3\xa9; ";
    static const char pp[] = "messages[3].content \"x\" ";
    static const char cp[] = "code_\\_";
    enum { MREP = 1500 / (sizeof mp - 1) + 1, PREP = 300 / (sizeof pp - 1) + 1, CREP = 200 / (sizeof cp - 1) + 1 };
    char msg[MREP * (sizeof mp - 1) + 1], param[PREP * (sizeof pp - 1) + 1], ecode[CREP * (sizeof cp - 1) + 1];
    msg[0] = param[0] = ecode[0] = 0;
    for (int i = 0; i < MREP; i++) strcat(msg, mp);
    for (int i = 0; i < PREP; i++) strcat(param, pp);
    for (int i = 0; i < CREP; i++) strcat(ecode, cp);

    char *body; size_t blen; long clen;
    char *resp = round_trip(400, msg, param, ecode, &body, &blen, &clen);
    jv *doc = json_parse(body, blen);
    jv *err = doc ? jv_get(doc, "error") : NULL;
    const char *got = err ? jv_str(jv_get(err, "message"), NULL) : NULL;
    CHECK(got && !strcmp(got, msg), "a %zu-byte message arrives whole (got %zu bytes)", strlen(msg), got ? strlen(got) : 0);
    const char *gp = err ? jv_str(jv_get(err, "param"), NULL) : NULL;
    const char *gc = err ? jv_str(jv_get(err, "code"), NULL) : NULL;
    CHECK(gp && !strcmp(gp, param), "a %zu-byte param arrives whole (got %zu bytes)", strlen(param), gp ? strlen(gp) : 0);
    CHECK(gc && !strcmp(gc, ecode), "a %zu-byte code arrives whole (got %zu bytes)", strlen(ecode), gc ? strlen(gc) : 0);
    const char *ty = err ? jv_str(jv_get(err, "type"), NULL) : NULL;
    CHECK(ty && !strcmp(ty, "invalid_request_error"), "a 400 is an invalid_request_error (%s)", ty ? ty : "none");
    CHECK(clen == (long)blen, "Content-Length %ld equals the body's %zu bytes", clen, blen);
    jv_free(doc);
    free(resp);

    resp = round_trip(500, "out of memory", NULL, NULL, &body, &blen, &clen);
    doc = json_parse(body, blen);
    err = doc ? jv_get(doc, "error") : NULL;
    ty = err ? jv_str(jv_get(err, "type"), NULL) : NULL;
    got = err ? jv_str(jv_get(err, "message"), NULL) : NULL;
    jv *jp = err ? jv_get(err, "param") : NULL, *jc = err ? jv_get(err, "code") : NULL;
    CHECK(got && !strcmp(got, "out of memory") && ty && !strcmp(ty, "server_error") && jp && jp->type == J_NULL &&
          jc && jc->type == J_NULL, "a 500 is a server_error with param and code null");
    CHECK(clen == (long)blen, "Content-Length %ld equals the body's %zu bytes", clen, blen);
    jv_free(doc);
    free(resp);

    printf(fails ? "FAIL %d\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
#endif
