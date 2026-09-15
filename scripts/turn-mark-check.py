#!/usr/bin/env python3
"""The prompt-reuse gate for one served model: does the second agent turn
re-prefill the whole prompt?

The 2026-09-14 OpenCode loop on Qwen 3.8 (a recurrent hybrid served on
CUDA) logged "0 cached" on every turn: each request re-folded its whole
7.5K-token prompt, 586 s per turn. This gate sends the shape an agent
client sends, greedy, and reads back what the runner says it kept
(`runner_telemetry.prompt_cached_tokens`, and `prompt_reuse` saying how)
and whether the answer is the one a fresh slot gives:

  1. request 1: a user turn (prompt P1) -> the slot generates a reply;
  2. request 2: P1 replayed, a DIFFERENT assistant reply (what a template
     re-rendering does to reasoning and tool calls), a tool-result or user
     turn -> must keep all of P1 but a few boundary tokens, and must answer
     exactly as the same request on a fresh slot (cache_prompt:false);
  3. request 2 again, verbatim -> must keep all but one token.

Every turn's prefill seconds and reason are recorded; the model file, the
binary and the served preset are hashed into the record so a PASS names
what passed. Exit status 0 only if every check passed.

Usage:
  turn-mark-check.py --base-url http://127.0.0.1:8080 --out evidence.json
                     [--model ID] [--model-file PATH] [--runner PATH]
                     [--prompt-file PATH] [--max-tokens N] [--label TEXT]
"""
import argparse
import hashlib
import json
import platform
import sys
import time
import urllib.error
import urllib.request

BOUNDARY_SLACK = 8   # tokens a template may re-tokenize at the reply boundary
FILLER = ("You are a coding agent working in a repository. The project has "
          "a Makefile, a src directory with C files, a tests directory with "
          "Python and C tests, a README and a CHANGELOG. ")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return json.load(r)


def chat(base, model, messages, max_tokens, timeout, **fields):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "seed": 1, "stream": False}
    body.update(fields)
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit("HTTP %d from the server: %s" % (e.code, e.read()[:300].decode("utf-8", "replace")))
    wall = time.monotonic() - t0
    ch = d["choices"][0]
    rt = d.get("runner_telemetry") or {}
    tm = rt.get("timing") or {}
    return {"wall_s": round(wall, 3), "content": ch["message"].get("content") or "",
            "reasoning": ch["message"].get("reasoning_content") or "",
            "finish_reason": ch.get("finish_reason"),
            "prompt_tokens": d["usage"]["prompt_tokens"],
            "cached_tokens": rt.get("prompt_cached_tokens"),
            "cache": rt.get("prompt_reuse"),
            "prefill_seconds": tm.get("prefill_seconds"),
            "prefill_tokens": tm.get("prefill_tokens")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model")
    ap.add_argument("--model-file")
    ap.add_argument("--runner")
    ap.add_argument("--prompt-file", help="system prompt text (default: a "
                    "~1K-token filler so the prefill is measurable)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    caps = get(args.base_url, "/v1/capabilities")
    model = args.model or get(args.base_url, "/v1/models")["data"][0]["id"]
    system = (open(args.prompt_file).read() if args.prompt_file
              else FILLER * 12)
    first = [{"role": "system", "content": system},
             {"role": "user", "content": "List the files in the current "
              "directory, then tell me which one you would read first."}]
    second = first + [
        {"role": "assistant", "content": "I will list the directory first."},
        {"role": "user", "content": "The listing shows Makefile, README.md, "
         "src and tests. Which file do you read first, and why?"}]
    record = {
        "gate": "turn-mark-check", "label": args.label,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(), "platform": platform.platform(),
        "base_url": args.base_url, "model": model,
        "model_file": args.model_file,
        "model_sha256": sha256(args.model_file) if args.model_file else None,
        "runner": args.runner,
        "runner_sha256": sha256(args.runner) if args.runner else None,
        "sampling_served": caps.get("sampling"), "template": caps.get("template"),
        "context": caps.get("context"), "max_tokens": args.max_tokens,
        "turns": {}, "checks": {}, "verdict": "NOT RUN",
    }

    def write():
        with open(args.out, "w") as f:
            json.dump(record, f, indent=1)

    write()
    turns = record["turns"]
    turns["first"] = t1 = chat(args.base_url, model, first, args.max_tokens, args.timeout)
    write()
    turns["second"] = t2 = chat(args.base_url, model, second, args.max_tokens, args.timeout)
    write()
    turns["second_replay"] = t3 = chat(args.base_url, model, second, args.max_tokens, args.timeout)
    write()
    turns["second_fresh"] = t4 = chat(args.base_url, model, second, args.max_tokens,
                                      args.timeout, cache_prompt=False)
    write()
    n1 = t1["prompt_tokens"]
    checks = record["checks"]
    # The second turn may only prefill what it added (the re-rendered reply
    # and the new turn) plus a few boundary tokens: a template can close
    # the first prompt's generation header differently once a reply follows
    # it, and those tokens re-tokenize. What must not happen is the whole
    # first prompt being fed again.
    checks["second_turn_keeps_the_first_prompt"] = {
        "pass": (t2["cached_tokens"] or 0) >= n1 - BOUNDARY_SLACK,
        "cached_tokens": t2["cached_tokens"], "first_prompt_tokens": n1,
        "cache": t2["cache"]}
    checks["verbatim_replay_keeps_all_but_one"] = {
        "pass": t3["cached_tokens"] == t2["prompt_tokens"] - 1,
        "cached_tokens": t3["cached_tokens"], "prompt_tokens": t2["prompt_tokens"],
        "cache": t3["cache"]}
    checks["kept_turn_answers_like_a_fresh_slot"] = {
        "pass": t2["content"] == t4["content"] and t3["content"] == t4["content"],
        "second": t2["content"][:200], "fresh": t4["content"][:200]}
    checks["prefill_is_the_tail_only"] = {
        "pass": t2["prefill_tokens"] is not None and
                t2["prefill_tokens"] == t2["prompt_tokens"] - (t2["cached_tokens"] or 0),
        "second_prefill_seconds": t2["prefill_seconds"],
        "fresh_prefill_seconds": t4["prefill_seconds"]}
    for k, v in checks.items():
        print("%s: %s" % (k, "PASS" if v["pass"] else "FAIL"), flush=True)
    print("first prefill %.2fs (%d tok); second %.2fs (%s cached, %s); replay "
          "%.2fs (%s cached, %s); fresh %.2fs" % (
              t1["prefill_seconds"] or 0, t1["prefill_tokens"] or 0,
              t2["prefill_seconds"] or 0, t2["cached_tokens"], t2["cache"],
              t3["prefill_seconds"] or 0, t3["cached_tokens"], t3["cache"],
              t4["prefill_seconds"] or 0))
    ok = all(v["pass"] for v in checks.values())
    record["verdict"] = "PASS" if ok else "FAIL"
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
