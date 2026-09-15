#!/usr/bin/env python3
"""The agent-protocol gate for one served model at its shipped defaults.

The 2026-09-14 Windows report reached the runner as a PowerShell script that
sent the same two tool requests to whatever model was resident and read the
parsed `tool_calls` back, streamed and buffered. This is that script as a
gate: the same cases, the same reading, run against a live server, with the
result bound to what was actually measured (the binary's hash, the model
file's hash, the served sampling preset, the template family, the request
bodies) and written incrementally, so a run that is cancelled keeps every
case it completed.

Sampling is left at the family preset unless overridden on the command line,
because that is what a real client sends. A case PASSES when both the
streamed and the buffered turn end with `finish_reason: tool_calls`, carry at
least one parsed call whose arguments are valid JSON for the declared
schema, and leave no tool framing in `content`. The prose case additionally
requires non-empty content beside the call. Model choices (which tool, which
pattern) are recorded, not judged: this gate is about the protocol reaching
the client, task quality is a different suite.

Usage:
  tool-protocol-check.py --base-url http://127.0.0.1:8080 --out evidence.json
                         [--model ID] [--model-file PATH] [--runner PATH]
                         [--cases A,B,C] [--max-tokens N] [--temperature T]
                         [--repeat-penalty P] [--label TEXT]
Exit status is 0 only if every case passed.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
import urllib.error
import urllib.request

BASH = {"type": "function", "function": {
    "name": "bash", "description": "Run a shell command",
    "parameters": {"type": "object",
                   "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}
GLOB = {"type": "function", "function": {
    "name": "glob", "description": "Find files matching a glob pattern",
    "parameters": {"type": "object",
                   "properties": {"pattern": {"type": "string"}},
                   "required": ["pattern"]}}}

CASES = {
    "A": {"title": "single call",
          "prompt": "List the files in the current directory.",
          "tools": [BASH], "prose": False},
    "B": {"title": "prose and parallel calls",
          "prompt": "Check whether boss.json, boss.toml and boss.yaml exist "
                    "anywhere in this project. Use the glob tool for each "
                    "pattern.",
          "tools": [GLOB, BASH], "prose": False},
    "C": {"title": "prose then call",
          "prompt": "Briefly say what you are about to do, then list the "
                    "files in the current directory.",
          "tools": [BASH], "prose": True},
}
FRAMING = ("<tool_call", "<|tool_call", "<function=", "<parameter=",
           "<atem:invoke", "to=functions.")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return json.load(r)


def post(base, body, stream, timeout):
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    wall = time.monotonic() - t0
    if not stream:
        d = json.loads(raw)
        ch = d["choices"][0]
        msg = ch["message"]
        calls = [{"name": c["function"]["name"],
                  "arguments": c["function"]["arguments"], "id": c.get("id")}
                 for c in (msg.get("tool_calls") or [])]
        return {"wall_s": round(wall, 3), "finish_reason": ch.get("finish_reason"),
                "content": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or "",
                "calls": calls, "usage": d.get("usage"),
                "telemetry": d.get("runner_telemetry")}
    content, reasoning, finish, calls, detail = "", "", None, {}, None
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.startswith("data: ") or line[6:] == "[DONE]":
            continue
        e = json.loads(line[6:])
        for ch in e.get("choices", []):
            delta = ch.get("delta", {})
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
            for tc in delta.get("tool_calls") or []:
                c = calls.setdefault(tc["index"], {"name": None, "arguments": "",
                                                   "id": None})
                fn = tc.get("function", {})
                if fn.get("name"):
                    c["name"] = fn["name"]
                if tc.get("id"):
                    c["id"] = tc["id"]
                c["arguments"] += fn.get("arguments") or ""
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
        if e.get("runner_telemetry"):
            detail = e["runner_telemetry"]
    return {"wall_s": round(wall, 3), "finish_reason": finish,
            "content": content, "reasoning": reasoning,
            "calls": [calls[i] for i in sorted(calls)], "telemetry": detail}


def schema_ok(call, tools):
    decl = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    if call["name"] not in decl:
        return False, "undeclared tool %r" % call["name"]
    try:
        args = json.loads(call["arguments"])
    except ValueError as e:
        return False, "arguments are not JSON: %s" % e
    if not isinstance(args, dict):
        return False, "arguments are not an object"
    params = decl[call["name"]]
    for req in params.get("required", []):
        if req not in args:
            return False, "missing required %r" % req
    types = {"string": str, "integer": int, "number": (int, float),
             "boolean": bool, "array": list, "object": dict}
    for k, v in args.items():
        spec = params.get("properties", {}).get(k)
        if spec is None:
            return False, "undeclared argument %r" % k
        t = spec.get("type")
        if t in types and not isinstance(v, types[t]):
            return False, "argument %r is not %s" % (k, t)
    return True, ""


def judge(case, turn):
    reasons = []
    if turn["finish_reason"] != "tool_calls":
        reasons.append("finish_reason %r" % turn["finish_reason"])
    if not turn["calls"]:
        reasons.append("no parsed call")
    for c in turn["calls"]:
        ok, why = schema_ok(c, case["tools"])
        if not ok:
            reasons.append(why)
    for m in FRAMING:
        if m in turn["content"]:
            reasons.append("framing %r in content" % m)
    if case["prose"] and not turn["content"].strip():
        reasons.append("no prose beside the call")
    return reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="model id to send (default: the served one)")
    ap.add_argument("--model-file", help="GGUF path, hashed into the record")
    ap.add_argument("--runner", help="runner binary, hashed into the record")
    ap.add_argument("--cases", default="A,B,C")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--repeat-penalty", type=float)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    caps = get(args.base_url, "/v1/capabilities")
    models = get(args.base_url, "/v1/models")
    model = args.model or models["data"][0]["id"]
    record = {
        "gate": "tool-protocol-check", "label": args.label,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(), "platform": platform.platform(),
        "base_url": args.base_url, "model": model,
        "model_file": args.model_file,
        "model_sha256": sha256(args.model_file) if args.model_file else None,
        "runner": args.runner,
        "runner_sha256": sha256(args.runner) if args.runner else None,
        "sampling_served": caps.get("sampling"),
        "context": caps.get("context"),
        "overrides": {k: v for k, v in
                      (("temperature", args.temperature),
                       ("repeat_penalty", args.repeat_penalty)) if v is not None},
        "max_tokens": args.max_tokens,
        "cases": {}, "verdict": "NOT RUN",
    }

    def write():
        with open(args.out, "w") as f:
            json.dump(record, f, indent=1)

    write()
    all_ok = True
    for key in args.cases.split(","):
        case = CASES[key]
        entry = {"title": case["title"], "prompt": case["prompt"],
                 "tools": [t["function"]["name"] for t in case["tools"]],
                 "modes": {}}
        record["cases"][key] = entry
        for stream in (False, True):
            body = {"model": model, "stream": stream,
                    "messages": [{"role": "user", "content": case["prompt"]}],
                    "tools": case["tools"], "max_tokens": args.max_tokens}
            body.update(record["overrides"])
            mode = "stream" if stream else "buffered"
            try:
                turn = post(args.base_url, body, stream, args.timeout)
                reasons = judge(case, turn)
                turn["result"] = "PASS" if not reasons else "FAIL"
                turn["reasons"] = reasons
            except (urllib.error.URLError, OSError, ValueError) as e:
                turn = {"result": "FAIL", "reasons": ["transport: %s" % e]}
            entry["modes"][mode] = turn
            all_ok = all_ok and turn["result"] == "PASS"
            print("%s %s %s: %s %s" % (key, case["title"], mode, turn["result"],
                                      "; ".join(turn.get("reasons", []))),
                  flush=True)
            write()
    record["verdict"] = "PASS" if all_ok else "FAIL"
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
