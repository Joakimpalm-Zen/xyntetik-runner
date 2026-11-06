# R3.36 on macOS

SmolLM2-135M Q8_0, `--parallel 2`, `--draft lookup`, Metal, Apple M1, macOS 27.0.1, a 6,000-token speculative stream,
one round each. Sessions were not read. `read.py` with the GPU sampler removed, `-c 8192`, one round and a 300 s probe
limit. The other slot's wait:

| Build | Reading | Stalled | Trickling |
|---|---|---|---|
| main, 7d51410a | 86.5 s | 60.0 s | 60.0 s |
| this branch before the fix, 20d90416 | 85 to 86 s | not served in 300 s | not served in 300 s |
| this branch | 86.0 s | 86.1 s | 86.1 s |

On main the client's write times out and ends its generation, so the wait is shorter than behind a reader. Trickling did
not run unbounded there at this size, unlike Windows.

**The cause.** On this macOS, `send(MSG_DONTWAIT)` on a blocking socket with a full buffer waits out the socket's whole
`SO_SNDTIMEO` before it gives EAGAIN: 30,001 ms at 30 s, 2.001 s at 2 s, 1.001 s at 1 s, on TCP and on a Unix socket pair,
and with no timeout set the stalled test did not return in over 200 s. `O_NONBLOCK` on the same socket gives EAGAIN in
0.001 ms. A `sample` of the server showed the slot thread in `stream_out_write` and `__sendto` for all 716 samples of 8 s,
under the device turn. The writer took each EAGAIN as a queued write and went on, so the next write waited again: the
generation ran at 9.3 tok/s against 70 and held the turn until the client closed.

`recv(MSG_PEEK | MSG_DONTWAIT)` does not have this trap: 0.003 to 0.026 ms on a quiet socket with `SO_RCVTIMEO` set, with
data and with a closed peer, so `sock_peer_closed` is unchanged.

**The fix.** `sock_send_nb` sets `O_NONBLOCK` around the send, as the Windows branch does with `FIONBIO`. With it all
three generations ran the full 6,000 tokens at 69 tok/s, and a client that read five seconds late got the same text.
With `MSG_DONTWAIT` put back, the new test in `tests/test_stream_out.c` fails.
