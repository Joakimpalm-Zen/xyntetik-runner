#!/bin/sh
# The audit demo: a 1,000-token run is recorded, signed, handed over and
# verified by replay, and a copy with one output token changed is caught.
#
#   scripts/audit-demo.sh [MODEL.gguf] [RUNNER]
#
# MODEL defaults to SmolLM2-135M-Instruct Q8_0 fetched from the Hugging Face
# Hub (bartowski's GGUF, 145 MB), RUNNER to ./runner. Everything is written to a fresh
# temporary directory, which is printed and kept. Exit 0 only when the
# untouched record verifies and every tampered copy is refused.
#
# What it proves is in docs/audit-demo.md: the output is the one this model,
# binary, settings and seed produce, and the record was signed by the key
# whose public half is printed. Not who ran it, or when.
set -u
RUNNER=${2:-./runner}
DIR=$(mktemp -d "${TMPDIR:-/tmp}/runner-audit.XXXXXX") || exit 2
if [ $# -ge 1 ] && [ -n "$1" ]; then
    MODEL_ARGS="-m $1"
else
    MODEL_ARGS="-hf bartowski/SmolLM2-135M-Instruct-GGUF:Q8_0"
fi
PROMPT="Write a long, detailed account of how a lighthouse keeper spends one winter week, day by day."
now() { python3 -c 'import time; print(time.time())'; }
since() { python3 -c "import sys; print('%.1f s' % (float(sys.argv[2]) - float(sys.argv[1])))" "$1" "$(now)"; }
step() { printf '\n== %s\n' "$*"; }

T0=$(now)
step "1. a signing key (Ed25519)"
"$RUNNER" --keygen "$DIR/key.json" > "$DIR/pubkey.txt" 2>&1 || { cat "$DIR/pubkey.txt"; exit 2; }
PUB=$(grep -o -E '[0-9a-f]{64}' "$DIR/pubkey.txt" | head -n 1)
[ -n "$PUB" ] || { echo "no public key in the keygen output"; cat "$DIR/pubkey.txt"; exit 2; }
echo "public key $PUB"

step "2. record 1,000 sampled tokens, signed"
t=$(now)
# shellcheck disable=SC2086
"$RUNNER" $MODEL_ARGS -p "$PROMPT" -n 1000 --ignore-eos --temp 0.8 -s 42 \
    --no-tray --transcript "$DIR/run.json" --sign-key "$DIR/key.json" \
    > "$DIR/output.txt" 2> "$DIR/record.log" || { tail -5 "$DIR/record.log"; exit 2; }
echo "recorded in $(since "$t"): $(wc -c < "$DIR/run.json" | tr -d ' ') bytes, output in $DIR/output.txt"

step "3. the verifier replays it, trusting only that key"
t=$(now)
# shellcheck disable=SC2086
"$RUNNER" $MODEL_ARGS --verify "$DIR/run.json" --ignore-eos --require-signed \
    --trust-key "$PUB" --no-tray > "$DIR/verify.log" 2>&1
rc=$?
grep -E "VERIFIED|DIVERGED|UNVERIFIABLE" "$DIR/verify.log" | tail -n 1
echo "exit $rc in $(since "$t")"
[ "$rc" -eq 0 ] || { echo "FAIL: the untouched record did not verify"; exit 1; }

step "4. tampered copies are refused"
# Edited as bytes, the way a forger would: the record's chain hash is the
# sha256 of every byte before its last ,"chain":, so re-serialising the JSON
# would break it for a reason that has nothing to do with the tampering.
python3 - "$DIR" <<'PY'
import hashlib, re, sys
d = sys.argv[1]
raw = open(d + "/run.json", "rb").read()
cut = raw.rindex(b',"chain":')
head, tail = raw[:cut], raw[cut:]
# output token 500, changed by one
start = head.index(b'"tokens":[', head.index(b'"output":')) + len(b'"tokens":[')
end = head.index(b"]", start)
ids = head[start:end].split(b",")
ids[500] = str(int(ids[500]) + 1).encode()
forged = head[:start] + b",".join(ids) + head[end:]
# (a) the token changed, nothing else: the chain no longer recomputes
open(d + "/tampered-token.json", "wb").write(forged + tail)
# (b) the token changed AND the chain hash recomputed, the signature dropped:
#     internally consistent, so only the replay can tell
chain = tail[:tail.index(b"}") + 1]
chain = re.sub(rb'"hash":"[0-9a-f]{64}"',
               b'"hash":"' + hashlib.sha256(forged).hexdigest().encode() + b'"',
               chain, count=1)
open(d + "/forged-consistent.json", "wb").write(forged + chain + b"}")
PY
fails=0
check() {  # record require-signed(0/1)
    if [ "$2" = 1 ]; then extra="--require-signed --trust-key $PUB"; else extra=""; fi
    # shellcheck disable=SC2086
    "$RUNNER" $MODEL_ARGS --verify "$DIR/$1.json" --ignore-eos $extra --no-tray \
        > "$DIR/$1.log" 2>&1
    rc=$?
    why=$(grep -E '^(VERIFIED|DIVERGED|UNVERIFIABLE)' "$DIR/$1.log" | tail -n 1)
    echo "$1${extra:+, signed record required}: exit $rc, $why"
    [ "$rc" -ne 0 ] || fails=$((fails + 1))
}
check tampered-token 1
check forged-consistent 1
check forged-consistent 0
[ "$fails" -eq 0 ] || { echo "FAIL: a tampered record verified"; exit 1; }

echo
echo "done in $(since "$T0"); files in $DIR"
