"""R4.48.12 client, VERSION 2 (claim_client.py, which produced the committed Claim 1 records, is unchanged).
One /v1/chat/completions request per prompt, PREREG 1.2's body (messages, tools with tool_choice "auto" when the
prompt has tools, max_tokens, temperature 0, reasoning_effort), one at a time, cold. Each response is saved, with its
receipt's prompt and output ids (the server runs with --record DIR).
THE CHANGE (Bocka R4.48.12.3, finding 1): Runner's receipt lists the generated ids WITHOUT the stop token, while the
response reports it (choices[0].stop_token_id). Version 1 stored the receipt's ids as gen_ids, so every answer that
stopped ended its gen_ids one token short of the stop (llama.cpp's client stores the stop token). Version 2 stores
gen_ids = the receipt's ids + [stop_token_id] when the response reports a stop token (an int) and the receipt's ids do
not already end with it; nothing else changes. Each record keeps the receipt's own ids as gen_ids_receipt and says
what was done in stop_token_appended (true / false) and stop_token_id (the response's, or null).
usage: claim_client_v2.py BASE_URL PROMPTS.jsonl EFFORT MAX_TOKENS RECEIPT_DIR OUT.jsonl [--split primary] [--ids a,b]"""
import argparse, json, os, time, urllib.request


def with_stop(gen, d):
    """gen_ids with the response's reported stop token appended once; (ids, appended, stop_token_id)."""
    ch = (d.get("choices") or [{}])[0]
    st = ch.get("stop_token_id")
    if gen is None or not isinstance(st, int) or isinstance(st, bool):
        return gen, False, st if isinstance(st, int) and not isinstance(st, bool) else None
    if gen and gen[-1] == st:
        return list(gen), False, st
    return list(gen) + [st], True, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("prompts"); ap.add_argument("effort"); ap.add_argument("max_tokens", type=int)
    ap.add_argument("receipts"); ap.add_argument("out"); ap.add_argument("--split"); ap.add_argument("--ids")
    a = ap.parse_args()
    want = set(a.ids.split(",")) if a.ids else None
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    with open(a.out, "a") as fo:
        for line in open(a.prompts):
            r = json.loads(line)
            if a.split and r.get("split") != a.split: continue
            if want and r["id"] not in want: continue
            if r["id"] in done: continue
            body = {"messages": r["messages"], "max_tokens": a.max_tokens, "temperature": 0, "reasoning_effort": a.effort,
                    "stream": False}
            if r.get("tools"):
                body["tools"] = r["tools"]
                body["tool_choice"] = "auto"
            t0 = time.time()
            req = urllib.request.Request(a.base + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=36000) as resp:
                d = json.load(resp)
            wall = time.time() - t0
            rec = (d.get("runner_telemetry") or {}).get("receipt") or {}
            rj = json.load(open(os.path.join(a.receipts, rec["file"]))) if rec.get("file") else {}
            g_rec = (rj.get("output") or {}).get("tokens")
            gen, appended, st = with_stop(g_rec, d)
            fo.write(json.dumps({"id": r["id"], "category": r.get("category") or r.get("subset"), "wall_s": round(wall, 3),
                                 "prompt_ids": (rj.get("prompt") or {}).get("tokens"),
                                 "gen_ids": gen, "gen_ids_receipt": g_rec, "stop_token_appended": appended,
                                 "stop_token_id": st, "client_version": 2,
                                 "finish": (rj.get("output") or {}).get("finish"), "receipt": rec, "response": d}) + "\n")
            fo.flush()
            print(r["id"], round(wall, 1), "s", (d.get("usage") or {}).get("completion_tokens"),
                  "stop appended" if appended else "", flush=True)


if __name__ == "__main__":
    main()
