#!/bin/zsh
# usage: srv.sh RUNNER MODEL ARM(plain|lookup) PORT -> per prompt: gen tok/s and a content hash
R=$1; M=$2; ARM=$3; PORT=$4
X=""; [ "$ARM" = lookup ] && X="--draft-lookup"
$R -m $M --serve --port $PORT --gpu off -t 4 --no-tray $=X > srv-$ARM.log 2>&1 &
S=$!
for i in $(seq 100); do curl -s -o /dev/null http://127.0.0.1:$PORT/health && break; sleep 0.2; done
for p in edit prose; do
  python3 - "$PORT" "$p" "$ARM" <<'PY'
import json, sys, time, urllib.request, hashlib
port, p, arm = sys.argv[1:4]
msg = open(f"{p}.txt").read()
body = {"messages": [{"role": "user", "content": msg}], "max_tokens": 300, "temperature": 0}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=900)); t = time.time() - t0
c = d["choices"][0]["message"]["content"]; n = d["usage"]["completion_tokens"]
open(f"chat-{p}-{arm}.txt", "w").write(c)
print(f"{p:6s} {arm:6s} {n} tok in {t:.2f} s wall, content sha {hashlib.sha256(c.encode()).hexdigest()[:12]}")
PY
done
grep -o 'gen: [0-9]* tok, [0-9.]* tok/s' srv-$ARM.log | tail -2
kill $S; wait $S 2>/dev/null
