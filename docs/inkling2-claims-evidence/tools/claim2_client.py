"""R4.48.12 Claim 2 client (PREREG 2.2 / 2.3): one server cell, one mode, the 20 primary targets.
Each request carries Claim 1's body (claim_client_v2.py / claim_client_llama.py: messages, tools with tool_choice
"auto" when the prompt has tools, max_tokens, temperature 0, reasoning_effort, stream false; the server's flags carry
the rest of PREREG 2.2's sampling), plus "cache_prompt": false (PREREG 2.2's cold prefix: both servers bypass prefix
reuse with it).
  alone:   per target, ONE request; the other slots idle.
  batched: per target, NINE requests at once, the target and the 8 companions (claim2-companions.jsonl, in its order),
           one thread each, released together by one threading.Barrier(9). T_batch_wall_s runs from the barrier's
           release to the 9th response's end; agg_tok_s = the 9 completion_tokens summed / T_batch_wall_s (PREREG 2.3(4)).
           For llama.cpp the prompt ids are read (apply-template, tokenize) BEFORE the barrier, so no extra request
           overlaps the round.
Ids:
  runner: the server runs with --record DIR; prompt and generated ids from each request's receipt, gen_ids with the
          response's stop token appended exactly as claim_client_v2.py's with_stop (copied verbatim below);
  llama:  __verbose.tokens (return_tokens / verbose), the prompt ids by /apply-template + /tokenize, raw content and
          /detokenize as claim_client_llama.py.
Telemetry per round (PREREG 2.3(3)): runner: GET /metrics and GET /v1/runner/prefix-cache before and after,
GET /v1/runner/provenance after, POST /v1/runner/prefix-cache/clear before (cold); llama: GET /slots and GET /metrics
before and after when the server serves them (else recorded as unavailable); the slot logs are the run script's server
stderr, aligned by each round's UTC times recorded here.
One JSON line per round; a rerun resumes after the last complete round (by target id).
usage: claim2_client.py ENGINE BASE_URL TARGETS.jsonl COMPANIONS.jsonl MODE OUT.jsonl --effort low --max-tokens 256
       [--receipts DIR] [--cell LABEL] [--ids a,b]"""
import argparse, datetime, json, os, threading, time, urllib.error, urllib.request


def with_stop(gen, d):
    """gen_ids with the response's reported stop token appended once; (ids, appended, stop_token_id)."""
    ch = (d.get("choices") or [{}])[0]
    st = ch.get("stop_token_id")
    if gen is None or not isinstance(st, int) or isinstance(st, bool):
        return gen, False, st if isinstance(st, int) and not isinstance(st, bool) else None
    if gen and gen[-1] == st:
        return list(gen), False, st
    return list(gen) + [st], True, st


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Server:
    def __init__(self, base, engine, receipts):
        self.base, self.engine, self.receipts = base.rstrip("/"), engine, receipts

    def post(self, path, body, timeout=36000):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)

    def get(self, path, timeout=60):
        """(status, text); a missing endpoint is recorded, not fatal"""
        try:
            with urllib.request.urlopen(self.base + path, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, None
        except Exception as e:   # noqa: BLE001
            return None, repr(e)

    def telemetry(self, when):
        t = {"when": when, "utc": utc()}
        paths = ["/metrics", "/v1/runner/prefix-cache"] if self.engine == "runner" else ["/slots", "/metrics"]
        if self.engine == "runner" and when == "after":
            paths.append("/v1/runner/provenance")
        for p in paths:
            st, txt = self.get(p)
            t[p] = {"status": st, "body": txt}
        return t

    def cold(self):
        # Runner's clear takes NO request body (MANUAL's endpoint table; a JSON body is refused with 400)
        if self.engine == "runner":
            req = urllib.request.Request(self.base + "/v1/runner/prefix-cache/clear", data=b"", method="POST")
            with urllib.request.urlopen(req, timeout=600) as resp:
                txt = resp.read().decode("utf-8", "replace")
            try:
                return json.loads(txt)
            except ValueError:
                return {"status": resp.status, "body": txt}
        return None

    def prompt_ids(self, body):
        if self.engine != "llama":
            return None, None
        prompt = self.post("/apply-template", body)["prompt"]
        return prompt, self.post("/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]

    def request(self, body):
        req = dict(body, cache_prompt=False)
        if self.engine == "llama":
            req.update(return_tokens=True, verbose=True)
        t0 = time.perf_counter()
        d = self.post("/v1/chat/completions", req)
        t1 = time.perf_counter()
        rec = {"t0": t0, "t1": t1, "wall_s": round(t1 - t0, 4),
               "completion_tokens": (d.get("usage") or {}).get("completion_tokens"),
               "finish": (d.get("choices") or [{}])[0].get("finish_reason")}
        if self.engine == "llama":
            v = d.pop("__verbose", None) or {}
            gen = v.get("tokens") or []
            rec.update(gen_ids=gen, raw_content=v.get("content"), stop_type=v.get("stop_type"),
                       stopping_word=v.get("stopping_word"), tokens_evaluated=v.get("tokens_evaluated"),
                       tokens_predicted=v.get("tokens_predicted"))
        else:
            r = (d.get("runner_telemetry") or {}).get("receipt") or {}
            rj = json.load(open(os.path.join(self.receipts, r["file"]))) if r.get("file") else {}
            g = (rj.get("output") or {}).get("tokens")
            gen, appended, st = with_stop(g, d)
            rec.update(gen_ids=gen, gen_ids_receipt=g, stop_token_appended=appended, stop_token_id=st,
                       prompt_ids=(rj.get("prompt") or {}).get("tokens"), receipt_finish=(rj.get("output") or {}).get("finish"),
                       receipt=r)
        rec["response"] = d
        return rec


def body_for(r, a):
    # identical to claim_client_v2.py's / claim_client_llama.py's body
    body = {"messages": r["messages"], "max_tokens": a.max_tokens, "temperature": 0, "reasoning_effort": a.effort,
            "stream": False}
    if r.get("tools"):
        body["tools"] = r["tools"]
        body["tool_choice"] = "auto"
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["runner", "llama"]); ap.add_argument("base"); ap.add_argument("targets")
    ap.add_argument("companions"); ap.add_argument("mode", choices=["alone", "batched"]); ap.add_argument("out")
    ap.add_argument("--effort", required=True); ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--receipts"); ap.add_argument("--cell", default=""); ap.add_argument("--ids")
    a = ap.parse_args()
    if a.engine == "runner" and not a.receipts:
        ap.error("runner needs --receipts (the server's --record DIR)")
    srv = Server(a.base, a.engine, a.receipts)
    want = set(a.ids.split(",")) if a.ids else None
    targets = [json.loads(l) for l in open(a.targets)]
    targets = [r for r in targets if r.get("split") == "primary" and (not want or r["id"] in want)]
    comps = [json.loads(l) for l in open(a.companions)]
    assert len(comps) == 8, len(comps)
    # resume: only a COMPLETE round counts as done (an incomplete one, from a client-side failure such as a timeout or a
    # dropped connection, runs again; the stats take the LAST complete line per id and report the incomplete rounds).
    # An ENGINE defect halts and voids (PREREG 4.2(4)): no resume then.
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                x = json.loads(line)
            except Exception:   # noqa: BLE001  (a torn last line: that round reruns)
                continue
            if x.get("complete") is True:
                done.add(x["id"])
    with open(a.out, "a") as fo:
        for r in targets:
            if r["id"] in done:
                continue
            group = [("target", r)] + ([("companion", c) for c in comps] if a.mode == "batched" else [])
            bodies = [body_for(x, a) for _, x in group]
            pids = [srv.prompt_ids(b) for b in bodies]   # before the round (llama only)
            cold = srv.cold()
            before = srv.telemetry("before")
            results, errors = [None] * len(group), [None] * len(group)
            rel = {}
            bar = threading.Barrier(len(group), action=lambda: rel.setdefault("t", time.perf_counter()))

            def run(k):
                try:
                    bar.wait()
                    results[k] = srv.request(bodies[k])
                except Exception as e:   # noqa: BLE001
                    errors[k] = repr(e)
            start_utc = utc()
            ths = [threading.Thread(target=run, args=(k,)) for k in range(len(group))]
            for th in ths: th.start()
            for th in ths: th.join()
            end_utc = utc()
            after = srv.telemetry("after")
            ok = all(x is not None for x in results)
            t_rel = rel.get("t")
            wall = max(x["t1"] for x in results) - t_rel if ok and t_rel else None
            toks = sum(x["completion_tokens"] or 0 for x in results) if ok else None
            reqs = []
            for k, (role, x) in enumerate(group):
                q = results[k] or {}
                q = dict(q, role=role, id=x["id"], error=errors[k],
                         start_offset_s=round(q["t0"] - t_rel, 4) if q and t_rel else None,
                         end_offset_s=round(q["t1"] - t_rel, 4) if q and t_rel else None)
                q.pop("t0", None); q.pop("t1", None)
                if pids[k][1] is not None:
                    q["prompt"], q["prompt_ids"] = pids[k]
                reqs.append(q)
            fo.write(json.dumps({"id": r["id"], "category": r.get("category"), "engine": a.engine, "mode": a.mode,
                                 "cell": a.cell, "complete": ok, "round_start_utc": start_utc, "round_end_utc": end_utc,
                                 "T_batch_wall_s": round(wall, 4) if wall is not None else None,
                                 "completion_tokens_sum": toks,
                                 "agg_tok_s": round(toks / wall, 4) if wall else None,
                                 "cold": cold, "telemetry": [before, after], "requests": reqs}) + "\n")
            fo.flush()
            print(r["id"], a.mode, a.cell, "complete" if ok else "INCOMPLETE",
                  round(wall, 1) if wall else None, "s", toks, "tokens", flush=True)
            if not ok:
                raise SystemExit("a request failed: %s" % [e for e in errors if e])


if __name__ == "__main__":
    main()
