# Server notes: the incidents behind the rules

The comments in `src/server.c` state the rules; this page keeps what was
observed when each one was made.

## Shutdown drain

Until 2026-10-16 the drain had no bound: it waited for every active request to
run out its own `max_tokens` or deadline, so one SIGTERM to a busy server
looked ignored for minutes. On the lab, a 1,200-token generation finished
111.5 s after the signal, and a prefill bound by page faults needed SIGKILL.
GRACE (2 s) and HARD (10 s) came from that.

The hard bound was then fixed twice on the same day. It stopped watching after
the joins and printed with `fprintf`, so a full stderr pipe defeated it; the
first fix let it watch until the process ended, and a cleanly stopped server's
watchdog exited the process at its old deadline, during the next server's
life. Hence `drain_note` and `drain_owner`.

## Accept loop

- A long generation under single-slot serving used to block `/health` until
  the Xyntetik watchdog declared a live runner "unhealthy: timed out", which
  is why tiny requests are answered on the accept path.
- On 2026-10-05 a stored-response GET to a client that stopped reading parked
  the accept loop, and nothing new was served. Hence the two rules: only
  answers that fit a socket buffer are written there, and every write carries
  the 30 s send timeout.
- SIGTERM and `accept()`: with the accept loop on a non-main thread, `/proc`
  showed it in `inet_csk_accept` long after the handler had run and closed
  the descriptor, which is why the handler calls `shutdown()` first.

## Cancels

- A cancel used to wait on the slot's last-ended sequence and count, which the
  next request overwrote: a fast request ending on the slot within one poll
  left the waiter at a sequence that never came back (504 after 30 s, its
  request long stopped) and with the other request's count.
- A detached thread per cancel keeps its Windows HANDLE under winpthreads after
  it exits: a soak went from 265 to 456 handles over 400 cancels. One
  long-lived thread now answers them all.

## Unload and keep_alive

- `/unload` is POST-only since 0.1.5-alpha. As a GET it was reachable from any
  web page the user visited: `<img src="http://127.0.0.1:PORT/unload">` frees
  the model with no preflight, no CORS and no rebinding, because loopback
  binding does not stop a browser. A POST is not a CORS simple request unless
  its Content-Type says so, so requiring one restores the preflight. Host and
  Origin validation is the other half and is tracked separately.
- Without a registry, `keep_alive` was range-checked, accepted and dropped, so
  `keep_alive: 0` was answered as if the model had unloaded. It is refused with
  400, not the 409 `/unload` uses: `/unload` is a management route refused by
  the state of its target, while `keep_alive` is a field of a well-formed
  completion that cannot be satisfied here, the shape of the surface's other
  request-field refusals.
- `keep_alive: 0` used to call `unload_resident()` directly, which freed the
  model but left the draft and every KV prefix snapshot resident (up to
  `RUNNER_PREFIX_CACHE_MB`, 512 MB by default), and ran with the request still
  counted active. It now records the wish for the same safe point `/unload`
  uses.

## Server lifetime

- A second `server_run` in one process did not work: several state fields are
  only ever set, so its slot workers saw a shut-down queue and exited, and the
  server answered nothing but `/health` (served on the accept path, which is
  why a casual check looked fine); teardown then joined an already-joined
  thread. `tests/test_server_restart.c` now serves twice. Per-instance state
  is filed; it would mean threading a context through six translation units.
- Before `serve_release`, about twenty refusals before the serving loop each
  returned with the loaded model and tokenizer still allocated.
- A Continue user pinned the `/v1/models` id with `-m llama=path` and lost
  `--parallel 2` without asking for swapping; a one-entry registry now gives
  way to the slots.
- A message whose text could not be assembled after an allocation failure
  used to be skipped like an empty one, so a turn dropped out of the
  conversation and the request still answered 200.
