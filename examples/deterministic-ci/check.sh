#!/usr/bin/env bash
# A snapshot test for a language model's output: fetch a pinned Runner
# release and a pinned model file, check both against their sha256, and
# replay the committed record. Exit 0 means the model produced the recorded
# tokens again; 2 means it did not (DIVERGED); 3 means the record could not
# be replayed (a different model file, or a constraint it does not carry).
#
#   examples/deterministic-ci/check.sh            # verify snapshot.json
#   examples/deterministic-ci/check.sh --record   # write a new snapshot.json
#
# Runs on Linux x86_64 and macOS arm64; downloads go to $CACHE (default
# .cache/deterministic-ci) and are reused when their digests still match.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
CACHE=${CACHE:-.cache/deterministic-ci}

RUNNER_TAG=v1.9.0
MODEL_URL=https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-Q8_0.gguf
MODEL_SHA=5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83
MODEL=SmolLM2-135M-Instruct-Q8_0.gguf
PROMPT="The three primary colours are"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)  ASSET=runner-linux-x86_64
                 ASSET_SHA=9c3bff0b22528be2b5b1d57637b06e178eee0401fe5da5d937869aa9602cee5e ;;
  Darwin-arm64)  ASSET=runner-macos-arm64
                 ASSET_SHA=1ac715c020c1c2674996a0707efab7f3575b822c5065f3f927bda1a7d37ce21b ;;
  *) echo "no pinned Runner release for $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# fetch URL to FILE unless FILE already has digest SHA; refuse any other bytes
fetch() {
  local url=$1 file=$2 want=$3
  if [ ! -f "$file" ] || [ "$(sha256 "$file")" != "$want" ]; then
    curl -fsSL --retry 3 -o "$file.part" "$url"
    mv "$file.part" "$file"
  fi
  local got
  got=$(sha256 "$file")
  if [ "$got" != "$want" ]; then
    echo "$file: sha256 $got, expected $want" >&2
    exit 1
  fi
}

mkdir -p "$CACHE"
fetch "https://github.com/Joakimpalm-Zen/xyntetik-runner/releases/download/$RUNNER_TAG/$ASSET" \
      "$CACHE/$ASSET" "$ASSET_SHA"
chmod +x "$CACHE/$ASSET"
fetch "$MODEL_URL" "$CACHE/$MODEL" "$MODEL_SHA"

cd "$CACHE"
if [ "${1:-}" = "--record" ]; then
  # greedy, so the text is the model's argmax at every step; CPU only, so the
  # record does not depend on whether the machine has a GPU
  "./$ASSET" -m "$MODEL" --gpu off -t 2 -p "$PROMPT" -n 40 -s 42 --temp 0 \
    --record "$HERE/snapshot.json"
  exit 0
fi
"./$ASSET" -m "$MODEL" --gpu off -t 2 --verify "$HERE/snapshot.json"
