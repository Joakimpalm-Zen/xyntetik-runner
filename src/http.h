// HTTP transport and framing: sockets, request parsing, response writing.
//
// Everything here was the first 250 lines of server.c. It is the layer that
// knows about file descriptors and CRLF and nothing about inference: what a
// socket is on this platform, how to read one HTTP request off it safely, and
// how to write a response or an OpenAI-shaped error back.
//
// The platform typedefs live in this header rather than in http.c because
// server.c still calls socket/bind/listen/accept/select directly for the
// listener; centralising the winsock-vs-BSD divergence in one place is most of
// what this split is worth.
#ifndef RUNNER_HTTP_H
#define RUNNER_HTTP_H

#include <stddef.h>
#include <stdbool.h>

#include "json.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET sock_t;
#define SOCK_INVALID INVALID_SOCKET
#else
#include <signal.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <sys/socket.h>
typedef int sock_t;
#define SOCK_INVALID (-1)
#endif

void sock_init(void);
int  sock_recv(sock_t fd, char *buf, size_t n);
int  sock_send(sock_t fd, const char *buf, size_t n);
int  sock_peek(sock_t fd, char *buf, size_t n);
// Non-consuming, non-blocking proof of an orderly peer close. Readable bytes
// and every error (including would-block) mean "not proven closed".
bool sock_peer_closed(sock_t fd);
void sock_close(sock_t fd);
// Close after a reply on a connection whose request may be unread: half-close,
// drain (bounded), close, so the reply leaves as FIN, never RST (see http.c).
void sock_close_drained(sock_t fd);
// The reason a socket call failed, from the platform's own error channel.
const char *sock_errstr(void);
// Did the last failed receive end on its SO_RCVTIMEO rather than an error?
bool sock_timed_out(void);
void sock_recv_timeout(sock_t fd, double s);
void sock_send_timeout(sock_t fd, double s);
// One non-blocking send: the bytes taken, 0 when the socket would block, -1 when
// the connection is gone. The socket stays blocking for every other caller.
int  sock_send_nb(sock_t fd, const char *buf, size_t n);

// A client write that cannot block, for a stream written while the device turn
// is held (R3.36): one client's pace must not stop the other slots. What the
// socket will not take now waits here, in order, up to a bound; the caller
// writes the rest with stream_out_drain once the turn is back.
//
// A client this far behind is cut off rather than held for. The queue only ever
// grows by whole writes, so a refused write leaves the stream parseable.
#define STREAM_OUT_CAP ((size_t)4 << 20)
enum { STREAM_OUT_OK, STREAM_OUT_GONE, STREAM_OUT_FULL };
typedef struct {
    char  *buf;
    size_t head, tail, alloc, cap;
    bool   blocked;    // stream_out_block_for_test
} stream_out;
void   stream_out_init(stream_out *o, size_t cap);
int    stream_out_write(stream_out *o, sock_t fd, const char *s, size_t n);
bool   stream_out_drain(stream_out *o, sock_t fd);   // blocking: never under the turn
size_t stream_out_pending(const stream_out *o);
void   stream_out_free(stream_out *o);
// The socket reads as full whatever it would take, so a test reaches the cap
// without first filling the kernel's buffers.
void   stream_out_block_for_test(stream_out *o, bool blocked);

#ifdef RUNNER_TEST_TURN_GUARD
// Nonzero on a thread holding the device turn; send_all ends the process then.
// A test build only (make runner-turn-guard): see tests/test_turn_write_guard.py.
extern _Thread_local int sock_write_forbidden;
#endif

// Parse the two HTTP fields that decide request framing; false rejects the
// request outright. See http.c for why a duplicate Content-Length is refused.
bool parse_request_framing(char *first_header, char *header_end,
                           size_t *content_length);
// Split "METHOD path HTTP/1.x" and point first_header at the line after it.
bool parse_request_line(char *hdr, char method[8], char path[256],
                        char **first_header);
bool validate_request_authority(char *first_header, char *header_end);
// One header's value as a request id: 1 to 64 bytes of A-Z a-z 0-9 . _ : -.
// 0 when the header is absent, 1 with the value in out, -1 when it is sent
// twice or its value is not such an id.
int request_header_id(char *first_header, char *header_end, const char *name,
                      char out[65]);
bool request_id_valid(const char *s, size_t n);

bool send_all(sock_t fd, const char *s, size_t n);
const char *reason_phrase(int code);
void send_response(sock_t fd, int code, const char *ctype, const char *body,
                   size_t blen);
// OpenAI-shaped error bodies. send_built sends a built JSON body, or 500 when
// the builder ran out of memory: a short body that still parses as success is
// worse than an error, because the client cannot tell anything went wrong.
void send_error(sock_t fd, int code, const char *message);
void send_error_detail(sock_t fd, int code, const char *message,
                       const char *param, const char *error_code);
void send_built(sock_t fd, sbuf *b);

#endif // RUNNER_HTTP_H
