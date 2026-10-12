# Security Policy

Runner follows the [versioning policy](docs/versioning.md). Only the
[latest 1.x release](https://github.com/Joakimpalm-Zen/xyntetik-runner/releases/latest)
is supported; there are no backports.

## Threat model

Runner is a local inference engine for a single trusted machine. Its trust
boundary is the operating-system user account, not the network:

- **The server binds `127.0.0.1`, hardcoded.** There is no flag to bind
  another interface. Nothing outside the machine can reach it directly;
  LAN access in the Xyntetik suite goes through the dashboard layer, never
  to runner itself.
- **No authentication, by design.** Any process that can open a loopback
  socket can use the API — the same set of processes that could already
  read your files or burn your GPU. Auth belongs to whatever layer you put
  in front of runner, not inside it.
- **Browser authority is validated.** Every request must carry exactly one
  `Host` naming `localhost`, `127.0.0.1`, or `[::1]` (with an optional valid
  port). When `Origin` is present it must name the same loopback set. This
  blocks a public web page or DNS-rebinding hostname from treating the local,
  unauthenticated API as its own origin.
- **Clients are assumed cooperative but not perfect.** The server defends
  against accidents (stalls, oversized requests, malformed JSON), not
  against a hostile local process — a hostile local process already has
  your account.

If you reverse-proxy or port-forward runner onto a network yourself, you
are outside this model: add authentication and TLS in the proxy, and
understand that runner has never claimed to be safe against hostile
network input.

## The HTTP surface

The built-in server is a deliberate subset of HTTP/1.1, kept in one source
layer (`src/http.c` plus the routing in `src/server.c`) so its behavior can be
audited as a unit. It is not a
general-purpose web server and does not try to be:

| Property | Behavior |
|---|---|
| Body framing | `Content-Length` only — no chunked transfer-encoding |
| Connection lifecycle | `Connection: close` on every response, no keep-alive |
| Request header | capped at 16 KB |
| Request authority | one loopback `Host`; optional `Origin` must be loopback |
| Request body | capped at 32 MB |
| Request read deadline | header + body must arrive within 10 s, else `408` and the inference slot is released |
| TLS | none — loopback traffic only |

API and schema features outside the documented subset fail closed (`400`).
The current HTTP framing limits are listed above; the loopback-only
listener is a required part of the threat model, not a substitute for a hardened
internet-facing parser.

## Untrusted input

The one input runner routinely takes from the internet is the **model
file**. GGUF metadata and tensor layouts are bounds-checked before use
(malformed dimension and offset metadata are rejected — this is
CI-tested), but a model file is still a large binary you chose to
download: prefer sources you trust, as you would with any executable.

Request bodies are parsed by runner's own strict JSON parser (no
dependencies, rejects malformed input outright), and JSON Schemas are
validated at compile time before they ever drive sampling.

**Chat message content is tokenized as text.** Runner marks the framing
written by its chat template and recognizes control tokens only within those
marks. A template control marker (e.g. `<|im_start|>system`) spelled in a
caller's message content remains text; it does not create a control token or
forge a turn boundary. This protects template framing, not the model's
interpretation of instructions in ordinary text.

**Raw completion prompts are caller-composed.** `/v1/completions` recognizes
control-token spellings in raw text prompts and accepts explicit token-ID
prompts exactly as sent. This is intentional under the trusted-local-caller
model above. Operators serving untrusted callers through a proxy are
responsible for authentication, authorization, and the inputs they forward.

## Hardening in place

- Zero third-party runtime dependencies — no bundled dependency tree or
  transitive package surface. Release binaries still use ordinary operating
  system libraries; Linux builds are dynamically linked to libc/libm and the
  platform loader rather than being fully static.
- All request reads are bounded in size and time; buffers are fixed-size
  and NUL-terminated before parsing.
- CI runs the engine under AddressSanitizer/UBSan on every push, plus
  smoke tests for the failure paths above (stalled clients, malformed
  models, oversized parameters).

## Reporting a vulnerability

Found something? Please use
[GitHub private vulnerability reporting](https://github.com/Joakimpalm-Zen/xyntetik-runner/security/advisories/new)
for anything sensitive, or an
[ordinary issue](https://github.com/Joakimpalm-Zen/xyntetik-runner/issues/new)
for hardening suggestions that don't need coordinated disclosure. Include
`runner --version` output and, if you can, a reproducer.
