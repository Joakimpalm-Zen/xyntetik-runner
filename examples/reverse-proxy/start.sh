#!/bin/sh
set -eu
# A fixed hex token cannot become a Caddy header wildcard or config token.
case "${RUNNER_PROXY_API_KEY:-}" in
    ''|*[!0123456789abcdefABCDEF]*)
        echo 'RUNNER_PROXY_API_KEY must be 64 hexadecimal characters' >&2
        exit 1
        ;;
esac
if [ "${#RUNNER_PROXY_API_KEY}" -ne 64 ]; then
    echo 'RUNNER_PROXY_API_KEY must be 64 hexadecimal characters' >&2
    exit 1
fi
exec "${CADDY:-caddy}" run --config "${RUNNER_PROXY_CONFIG:-/etc/caddy/Caddyfile}" --adapter caddyfile
