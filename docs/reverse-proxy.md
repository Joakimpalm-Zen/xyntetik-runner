# TLS and an API key in front of Runner (R7.7)

This proof-of-concept recipe runs Caddy 2.11.9 in front of Runner on the same
Linux host. Caddy terminates TLS and requires one bearer API key for every
path, including health and model discovery. Runner keeps its hardcoded
`127.0.0.1` listener; no TLS, authentication, remote bind or engine code is
added. A local process with access to that loopback port can still call
Runner directly, as described in [the security model](../SECURITY.md).

Use Docker Engine on Linux with host networking and Docker Compose v2,
OpenSSL, a Runner binary and a model you already trust. The Compose container
runs as your supplied numeric user/group so it can read the private TLS key,
with a read-only filesystem and no added capabilities. Use an unprivileged
front port (8443 by default). This is a single-key PoC, not a multi-user
identity or authorization system.

## Start the engine on loopback

From the repository root, in one terminal:

```sh
./runner -m model.gguf --serve --port 8080 --no-tray
```

Leave Runner on loopback. Publish only the proxy's HTTPS port, never the
engine's HTTP port. The provided Compose file contains only the proxy.
Its `network_mode: host` reaches the host's loopback listener; a bridged
container's `127.0.0.1` would refer to that container instead. Host networking
uses the host's ports directly, so the file has no `ports:` mappings.
The Compose verification targets rootful Docker Engine on Linux.

## Prepare the TLS endpoint and key

In another terminal on that host:

```sh
cd examples/reverse-proxy
umask 077
mkdir -p certs
export RUNNER_PROXY_HOST=localhost
export RUNNER_PROXY_UID="$(id -u)"
export RUNNER_PROXY_GID="$(id -g)"
export RUNNER_PROXY_API_KEY="$(openssl rand -hex 32)"
openssl req -x509 -newkey rsa:2048 -nodes -days 7 \
  -subj "/CN=${RUNNER_PROXY_HOST}" \
  -addext "subjectAltName=DNS:${RUNNER_PROXY_HOST},IP:127.0.0.1" \
  -keyout certs/tls.key -out certs/tls.crt

docker compose config --quiet
docker compose up --detach
```

The startup guard rejects an absent, empty, non-hex or non-64-character key
before Caddy starts. The key is generated at runtime; no working credential
is included in the repository. Keep the exported key and `certs/tls.key`
private. `certs/` and `.env` are ignored by git. `config --quiet` validates
without printing the expanded secret.

This local PoC uses a self-signed certificate. Give clients `certs/tls.crt`
as their explicit trust anchor; do not disable certificate verification.
For a remote PoC, set `RUNNER_PROXY_HOST` to a DNS name clients can resolve
to this host and supply `tls.crt`/`tls.key` valid for that name. Use a
certificate from the client's trusted CA, or distribute this PoC certificate
as its trust anchor. Replace the files with your existing CA-issued
certificate/key when moving beyond the local demonstration.

`RUNNER_PORT` defaults to 8080 and `RUNNER_PROXY_HTTPS_PORT` to 8443.
`RUNNER_PROXY_CERT_DIR` can select a different host certificate directory;
its files remain read-only inside the container. Restart/recreate the proxy
after changing its key or certificate configuration.

## Check the boundary

Without a key, HTTPS returns 401:

```sh
curl --cacert certs/tls.crt -i https://localhost:8443/v1/models
```

With the key, the same endpoint returns Runner's model list:

```sh
curl --cacert certs/tls.crt https://localhost:8443/v1/models \
  -H "Authorization: Bearer ${RUNNER_PROXY_API_KEY}"
curl --cacert certs/tls.crt https://localhost:8443/v1/completions \
  -H "Authorization: Bearer ${RUNNER_PROXY_API_KEY}" \
  -H 'Content-Type: application/json' \
  --data '{"prompt":"x","max_tokens":4,"temperature":0}'
```

Caddy authenticates first, then rewrites upstream `Host` to the loopback
address and removes `Origin` and `Authorization`. Runner's Host/Origin
validation remains active; the key is consumed by the proxy rather than
forwarded to the engine. Only HTTPS is configured: automatic plaintext
redirects and the Caddy admin API are disabled. Responses, including SSE,
are flushed through the proxy. The engine's existing framing/body limits
still apply; the tested clients send JSON with `Content-Length`.

To stop this PoC proxy:

```sh
docker compose down
```

## Executable verification

The required Linux CI job runs the actual Compose file and a generated tiny
CPU fixture, not a mock upstream. The shared path classifier selects it only
for changes to `compose.yaml`, `Caddyfile`, `start.sh`, this recipe page, or
`scripts/test-reverse-proxy.py`. Unrelated prose and C-only diffs skip it:

```sh
make -j2 runner
python3 scripts/test-reverse-proxy.py --runner ./runner
```

The smoke test generates a temporary certificate and key, verifies TLS with
that certificate, checks missing/wrong keys return 401, checks an invalid
configured key cannot start, then performs authenticated model discovery,
fixture completion and SSE. It inspects the OS listening sockets before and
after: Runner must listen only on `127.0.0.1`. Direct loopback access stays
unauthenticated, and direct non-loopback Origin values still get 403.
Cleanup removes only the test's own Compose project and temporary files.

For a lightweight native Caddy check on macOS or Linux, the same config and
startup guard can be exercised without a Docker VM:

```sh
python3 scripts/test-reverse-proxy.py --runner ./runner --native-caddy CADDY_BINARY
```

The native check does not replace the Compose CI test. Native Caddy 2.11.9
was exercised locally; the Compose result is recorded in this change's PR.

References: [Caddy header matching](https://caddyserver.com/docs/caddyfile/matchers#header),
[upstream header controls](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy#headers),
[TLS certificate files](https://caddyserver.com/docs/caddyfile/directives/tls),
and [Docker host networking](https://docs.docker.com/engine/network/drivers/host/).
