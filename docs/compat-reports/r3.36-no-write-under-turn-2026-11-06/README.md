# R3.36: no blocking client write under the device turn

**Result: a client can no longer extend the device turn.** Before, a client that stopped reading a speculative stream
froze every other slot for 30 s with the GPU idle, and one that trickled held them past 150 s. Now both read the same
as a client that reads: 21.1 to 21.2 s for a 6000-token generation, with the GPU at 97%.

**Limits, first:**
- **Linux is not measured.** All of this ran on Windows on an RTX 3070. `sock_send_nb` has a separate POSIX path, and
  Linux's `send` blocks differently: the session write that did not stall here may well have stalled there.
- **New behaviour: a client 4 MiB behind a speculative stream is cut off.** Today's alternative was stopping every
  other slot. The 4 MiB cut is tested end to end with a smaller cap on a CPU server; 6000 tokens is under 1 MiB, so the
  full cap was not reached on the card.
- **A reading client's generation still holds the turn for its length.** The other slot waits 21 s here. That is
  R3.5.2's question, and this item does not touch it. Plain streams are unchanged.
- **A trickling client still ties up its own slot** while its tail drains, as on the plain path.
- **One model, one card, one prompt per case.**

## The rule, and the audit

No blocking client write happens while the device turn is held, on any path. Every span between taking and giving the
turn was read in full. Three sites broke the rule:

| Site | Fix |
|---|---|
| The speculative walk's stream writes (`chunk_send`, `sse_send`) | `stream_out`: non-blocking, queued, drained after the turn |
| `session_answer`, start and resume | RNS-1's shape: built under the turn, written after it |
| The `context_unavailable` 409 | R3.37, separately |

The decode worker, embeddings, decide, systemone, rerank and context pinning already wrote after release.

## The change

**Streams.** While the turn is held, a stream goes through `stream_out` in `http.c`. Each write tries a non-blocking
send, sending anything queued first, and keeps what the socket will not take. After the turn, the slot thread writes the
rest with the ordinary blocking path. Nothing under the turn can block, so this does not depend on the write timeout.
Only the path that holds the turn uses it (`sched_generate_holds_turn`).

**The cut.** Past `STREAM_OUT_CAP` (4 MiB, about 26,000 tokens at the measured 158 bytes a token), the generation stops
and the client is sent what was queued, then its surface's terminator. Refused frames are dropped whole, so the stream
stays parseable. It reuses the path a generation fault takes:

| Surface | What the client sees last |
|---|---|
| OpenAI | `finish_reason: "error"`, `finish_detail: "client_too_slow"`, then `data: [DONE]` |
| Responses | `response.incomplete`, reason `client_too_slow` |
| Anthropic | an `error` event, `finish_detail: "client_too_slow"` |

The log reads `[slot N] ID: client cut off for being too slow`, and the request line ends `[client too slow]`.

**Memory:** at most 4 MiB a slot, allocated only for what a client has not taken. A reading client queues nothing.

**Abort:** a closed socket fails on the next write, within a token. A stalled open socket is found only at the cap or at
the drain, so its generation runs on meanwhile, bounded by `max_tokens`.

**Sessions.** `session_answer` builds a `session_reply`, and each caller writes it after `sched_prefill_end`. Windows did
not stall on that write, because Winsock takes one large send whole (`design-session-read.log`: 109.7 to 110.4 s for all
three clients on main). It broke the rule all the same.

## The reads

`read.py`, the 3070, Qwen3-1.7B Q8_0, `--parallel 2 --draft lookup`, the other slot's device wait. Each client asserts
its 200. A probe alone reads 0.0 s.

| | Reading (control) | Stalled | Trickling, 2 KB per 5 s |
|---|---:|---:|---:|
| Spec, before (main 230286e1) | 21.2 s, GPU 97% | 30.2-30.3 s, GPU 9% | over 150 s |
| **Spec, after** | 21.22-21.23 s, GPU 97% | **21.09-21.21 s, GPU 97%** | **21.11-21.16 s, GPU 97%** |
| **Session, after** | 21.54-21.58 s, GPU 97% | 21.58-21.60 s, GPU 97% | 21.58-21.59 s, GPU 97% |

**Identity:** a client reading at once and one reading 5 s late got the same 6,000 characters, both ending `[DONE]`.

## The gates

- **The guard (`tests/test_turn_write_guard.py`, 13 of 13).** `make runner-turn-guard` builds the rule in: `send_all`
  ends the process with exit 86 under the turn, and the turn runs on a CPU server. On main before this change it read 7
  failed: all four streaming surfaces and both session paths, plus a request made after them (`guard-before.txt`). Its
  anchor is a self test that writes under the turn on purpose and must die. Each stream test asserts the speculative
  path ran. Runs in `make test`, and fails rather than skips there if the binary is missing.
- **The writer (`tests/test_stream_out.c`).** Loopback sockets: a stalled peer never blocks a write, the cap is exact to
  the byte, a late reader gets every byte in order, a queued byte is not overtaken, a closed peer is found at once, and
  the socket stays blocking afterwards. No GPU.
- **The cut, end to end,** on all four surfaces, in the guard test, with a 2,000-byte cap. It caught a bug here: the
  flag first stopped the terminator as well, ending the stream with no `[DONE]`.
- **Identity:** `make test-cuda-spec-identity`, 17 configurations byte-identical to solo steps; `tests/test_spec_gpu.py`
  16 of 16.
- **Others:** `tests/conformance` 498 passed, 0 failed, 30 skipped. Fault framing, sessions, disconnect, named
  contexts, stop latency, snapshots and CI contracts pass; their skips are POSIX-only cases. `test-sched-turn` and
  `scripts/help-parity.py` pass. Full `make test` was not run on this host.
- **R3.37 fixed the third site on its own.** The guard does not drive that race, so it is not what proves that fix.

## Files

- `read.py`, `read.log`: the done test.
- `guard-before.txt`: the guard on main before the change.
- `design-session-read.log`, `stall-before.log`, `trickle-before.log`: the reads on main that this answers.
