# R10.22: `WinError 10054` under `pytest -n`, two servers on one port

**Result: not #651, not a crash and not unread data.** Two parallel tests' servers shared one port. When one test
stopped its server, the other test's connection was reset. The defect is old and Windows only: `SO_REUSEADDR` let a
second server bind a port in use. A second server now exits with `cannot bind`, and the harness takes another port.

**Limits, first:**
- **CI's own instance is matched by signature, not observed.** The harness deletes its server log on that path, so CI
  kept no stderr. One failure was caught here with everything kept.
- **The loop does not tell the commits apart.** 1 in 720 against 0 in 720 is one event. The evidence that this is old
  is the forced collision on both commits and the age of the code.
- **A clean loop after the fix is weak evidence** at about 1 in 2,400 servers. The tests below are the proof; the loop
  shows nothing else broke.
- **Measured on Windows.** Linux and macOS keep `SO_REUSEADDR`, and were not run.

## The evidence

**The loop** (`drive.py`), the failing test's own `_serve`, interleaved between 4e3bb663 and ca175b27:

| | 4e3bb663 (#651) | ca175b27 (before) |
|---|---:|---:|
| Solo | 0 of 160 | 0 of 160 |
| 8 parallel workers | 1 of 720 | 0 of 720 |

**The one failure:** ring phase, 10054, server alive. Port 27591 was held by two servers at once, overlapping 0.122 s,
the only overlap among 2,412 servers. The other test stopped its server 13 ms before the reset. The failing server
logged no request, though its client already had two answers: they came from the other test's server.

**Forced, on both commits, identical** (`collide.py`): a second server on a port in use binds and prints
`server listening`. All 4 requests go to the first, and the second logs none. Kill the first with a connection queued,
and the client's first send gets `ConnectionResetError` 10054, CI's signature (`client.py:1141` there is the headers
send). The second stays healthy.

**Age:** `SO_REUSEADDR` dates from 2026-07-10. `free_port` draws a random port from 20000-32767 and closes it before the
server binds. Its comment relied on the loser failing loudly, which is true on Linux and was false on Windows.

## The option

Microsoft documents that `SO_EXCLUSIVEADDRUSE` can delay a rebind, so it was measured against a plain bind:

| | Today | Plain bind | `SO_EXCLUSIVEADDRUSE` |
|---|---|---|---|
| Second server on a port in use | shared it | refused | refused |
| Restart at once after traffic, 7 in `TIME_WAIT` | ok | ok | ok |
| Restart after a hard kill, client connected | ok | ok | ok |

The hazard did not appear. Exclusive is used, since it also keeps another process from taking the port over.

## After the fix

`collide.py`: the second server exits 1 with the new error and the first keeps serving. Under load: 1,200 pairs, no
failure, no two servers on one port. `tests/test_port_in_use.py` holds the refusal, the error text, the restart, and
the harness taking another port.

## Files

- `drive.py`: the loop, keeping stderr, request numbers and port lifetimes.
- `collide.py`: the forced collision on one tree.
