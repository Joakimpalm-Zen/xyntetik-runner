#!/usr/bin/env python3
"""How long does an agent wait at each turn of a growing conversation?

A decode rate says nothing about the wait an agent user feels. That wait is
the time to the first token of every turn, and after the first turn it is
almost entirely prefill: the client resends the whole history, and whatever
the server cannot reuse from its cache it computes again.

This drives one scripted agent loop against a served model, the way an
OpenAI-compatible coding client does: a system prompt, tools declared, a
streamed request per turn, the model's own reply and a long tool result
appended to the history each time. The conversation is fixed (the tool
results are canned, replies come from the model at temperature 0), so two
runs of the same binary and model are the same workload.

Per turn it records the prompt tokens the server reported, how many of them
it reused, the time to the first streamed token, the generation rate and the
turn's wall time. A turn that reuses less than the previous turn's whole
prompt recomputed history the cache already held; the report counts those
tokens, because they are the responsiveness problem a rate hides.

    agent-turns-bench.py --runner ./runner --model M.gguf [--turns 10]
        [--gpu auto|off] [--ctx 16384] [--result-tokens 600] [--out FILE]
    agent-turns-bench.py --endpoint http://127.0.0.1:8080 --model NAME
"""
import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SYSTEM = ("You are a coding assistant working in a repository. Use the tools "
          "to read files before you answer. Keep answers short.")
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the repository.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Repository-relative path."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search", "description": "Search the repository for a string.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
]
USER_TURNS = [
    "Where is the request timeout configured? Start by reading config/server.toml.",
    "Now check how src/http/client.py uses it.",
    "Is the timeout applied to retries as well? Look at src/http/retry.py.",
    "Search for other places that hard-code 30 seconds.",
    "Read tests/test_timeouts.py and tell me what it covers.",
    "What would break if the default went to 60 seconds?",
    "Check src/worker/queue.py for a dependency on the old value.",
    "Read docs/operations.md, the section on timeouts.",
    "Summarize the three places that need to change.",
    "Write the one-line change for config/server.toml.",
    "And the matching change for the retry module.",
    "Anything in the changelog that mentions this setting?",
]


def canned_result(turn, words):
    """A deterministic tool result of roughly `words` words: file-like text
    whose lines differ per turn, so no turn's result is a prefix of another."""
    lines = []
    n = 0
    i = 0
    while n < words:
        line = ("%04d  timeout_s = %d  # turn %d: the %s path reads this value "
                "and passes it to the socket layer unchanged"
                % (i, 30 + (i * 7 + turn) % 17, turn,
                   ("client", "retry", "worker", "config")[i % 4]))
        lines.append(line)
        n += len(line.split())
        i += 1
    return "\n".join(lines)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(base, proc, seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if proc is not None and proc.poll() is not None:
            raise SystemExit("the server exited during startup")
        try:
            urllib.request.urlopen(base + "/health", timeout=2).read()
            return time.time() - t0
        except Exception:
            time.sleep(0.25)
    raise SystemExit("the server did not become ready")


def stream_turn(base, model, messages, max_tokens, tool_choice):
    body = {"model": model, "messages": messages, "tools": TOOLS,
            "tool_choice": tool_choice, "temperature": 0,
            "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    content, calls, usage, telemetry, finish = [], {}, None, None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
                telemetry = ev.get("runner_telemetry")
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if first is None and (d.get("content") or d.get("tool_calls")
                                      or d.get("reasoning_content")):
                    first = time.time() - t0
                if d.get("content"):
                    content.append(d["content"])
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0),
                                         {"id": None, "name": "", "arguments": ""})
                    if tc.get("id"):
                        c["id"] = tc["id"]
                    f = tc.get("function") or {}
                    if f.get("name"):
                        c["name"] = f["name"]
                    if f.get("arguments"):
                        c["arguments"] += f["arguments"]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    wall = time.time() - t0
    return {"content": "".join(content),
            "calls": [calls[k] for k in sorted(calls)],
            "usage": usage or {}, "telemetry": telemetry or {},
            "finish": finish, "ttft_s": first, "wall_s": wall}


def run(base, model, turns, max_tokens, result_words, tool_choice):
    messages = [{"role": "system", "content": SYSTEM}]
    rows = []
    prev_prompt = 0
    for t in range(turns):
        messages.append({"role": "user", "content": USER_TURNS[t % len(USER_TURNS)]})
        r = stream_turn(base, model, messages, max_tokens, tool_choice)
        u = r["usage"]
        prompt = u.get("prompt_tokens", 0)
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is None:
            cached = r["telemetry"].get("prompt_cached_tokens", 0)
        gen = u.get("completion_tokens", 0)
        # what a perfect cache would have reused: everything the previous
        # turn's request sent (its reply and the new messages are new)
        missed = max(0, prev_prompt - cached)
        gen_s = r["wall_s"] - (r["ttft_s"] or r["wall_s"])
        rows.append({"turn": t + 1, "prompt_tokens": prompt, "cached_tokens": cached,
                     "recomputed_history_tokens": missed,
                     "new_tokens": prompt - cached, "completion_tokens": gen,
                     "ttft_s": r["ttft_s"], "wall_s": r["wall_s"],
                     "gen_tok_s": (gen / gen_s) if gen and gen_s > 0 else None,
                     "finish": r["finish"], "tool_calls": len(r["calls"])})
        prev_prompt = prompt
        # the history a client sends back: the reply as delivered, then the
        # result of each call (or nothing, when the model answered in text)
        if r["calls"]:
            messages.append({"role": "assistant", "content": r["content"] or None,
                             "tool_calls": [{"id": c["id"] or ("call_%d_%d" % (t, i)),
                                             "type": "function",
                                             "function": {"name": c["name"],
                                                          "arguments": c["arguments"]}}
                                            for i, c in enumerate(r["calls"])]})
            for i, c in enumerate(r["calls"]):
                messages.append({"role": "tool",
                                 "tool_call_id": c["id"] or ("call_%d_%d" % (t, i)),
                                 "content": canned_result(t, result_words)})
        else:
            messages.append({"role": "assistant", "content": r["content"]})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner", default=str(ROOT / "runner"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint", help="an already-running OpenAI-compatible server")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--result-tokens", type=int, default=600,
                    help="approximate size of each canned tool result, in words")
    ap.add_argument("--tool-choice", default="auto")
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--extra-arg", action="append", default=[])
    ap.add_argument("--repeat", type=int, default=2,
                    help="run the conversation this many times on one server; "
                         "the first is cold, the rest are warm")
    ap.add_argument("--server-log", help="keep the spawned server's stderr here")
    ap.add_argument("--out")
    a = ap.parse_args()

    proc = None
    load_s = None
    if a.endpoint:
        base = a.endpoint.rstrip("/")
        model = a.model
    else:
        port = free_port()
        cmd = [a.runner, "--serve", "--no-tray", "-m", a.model, "--port", str(port),
               "-c", str(a.ctx), "--gpu", a.gpu]
        if a.threads:
            cmd += ["-t", str(a.threads)]
        cmd += a.extra_arg
        logf = open(a.server_log, "w") if a.server_log else subprocess.DEVNULL
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=logf)
        base = "http://127.0.0.1:%d" % port
        load_s = wait_ready(base, proc, 900)
        model = Path(a.model).name
    try:
        runs = []
        for k in range(a.repeat):
            rows = run(base, model, a.turns, a.max_tokens, a.result_tokens, a.tool_choice)
            runs.append({"pass": "cold" if k == 0 else "warm", "turns": rows})
            tt = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
            print("%s: %d turns, wall %.1f s, ttft median %.2f s max %.2f s, "
                  "recomputed history %d of %d prompt tokens" % (
                      runs[-1]["pass"], len(rows), sum(r["wall_s"] for r in rows),
                      sorted(tt)[len(tt) // 2] if tt else float("nan"),
                      max(tt) if tt else float("nan"),
                      sum(r["recomputed_history_tokens"] for r in rows),
                      sum(r["prompt_tokens"] for r in rows)))
            for r in rows:
                print("  turn %2d  prompt %5d  cached %5d  recomputed %5d  ttft %6.2f s  "
                      "gen %3d tok  %s%s" % (
                          r["turn"], r["prompt_tokens"], r["cached_tokens"],
                          r["recomputed_history_tokens"], r["ttft_s"] or -1,
                          r["completion_tokens"], r["finish"],
                          "  (%d call)" % r["tool_calls"] if r["tool_calls"] else ""))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
    rec = {"schema": "xyntetik.runner.agent-turns.v1", "model": Path(a.model).name,
           "turns": a.turns, "max_tokens": a.max_tokens,
           "result_words": a.result_tokens, "tool_choice": a.tool_choice,
           "gpu": a.gpu, "ctx": a.ctx, "server_ready_s": load_s, "runs": runs}
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
