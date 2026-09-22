#!/usr/bin/env python3
"""Tool-use evaluation with shifted distribution and dual-leg scoring.

Evaluates a base model and optional adapter on a hand-written tool-use prompt
set with distribution shift from the original training templates. Scores both
raw greedy completions (raw leg) and native /v1/chat/completions with tools
and choice_logprobs (native leg).

Usage:
    eval-tooluse-shifted.py --runner ./runner --model M.gguf \\
        [--lora A.gguf [--lora-scale S]] [--threads N] [--freeze]
"""
import argparse
import hashlib
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def extract_json(text):
    """Extract first valid JSON object from text."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def load_catalog(path):
    """Load tool catalog from JSON."""
    with open(path) as f:
        catalog = json.load(f)
    # Build schema dict for validation
    schemas = {}
    for tool in catalog:
        schemas[tool["name"]] = set(tool.get("required", []))
    return catalog, schemas


def load_labels_and_verify():
    """Load evaluation set and verify labels haven't changed."""
    set_path = "evals/tooluse-shifted/set-v1.jsonl"
    labels_hash_path = "evals/tooluse-shifted/LABELS.sha256"

    # Load labels
    labels = []
    with open(set_path) as f:
        for line in f:
            row = json.loads(line)
            labels.append([row["id"], row["gold_tool"], row["gold_args"]])
    labels.sort(key=lambda x: x[0])

    # Verify hash
    canonical = "\n".join(json.dumps(label) for label in labels)
    current_hash = hashlib.sha256(canonical.encode()).hexdigest()

    with open(labels_hash_path) as f:
        stored_hash = f.read().strip()

    if current_hash != stored_hash:
        sys.exit(f"Labels have been modified! Expected {stored_hash}, got {current_hash}")

    return labels, current_hash


def sha256_file(path):
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runner_version(binary):
    """Get runner version string."""
    try:
        out = subprocess.run([binary, "--version"], capture_output=True,
                           text=True, timeout=30).stdout.strip()
        return out.splitlines()[0] if out else None
    except Exception:
        return None


def free_port():
    """Find a free TCP port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(args, adapter, port):
    """Launch the runner server."""
    cmd = [args.runner, "-m", args.model, "--serve", "--port", str(port),
           "--no-tray", "--gpu", "off"]
    if args.threads > 0:
        cmd += ["-t", str(args.threads)]
    if adapter:
        cmd += ["--lora", adapter]

    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)


def wait_ready(port, timeout=300):
    """Wait for server to be ready."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return True
        except Exception:
            time.sleep(1)
    return False


def stop(proc):
    """Stop the server process."""
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def schema_for_catalog(catalog):
    """Build JSON schema for tool choice."""
    tool_names = [t["name"] for t in catalog if t["name"] != "none"]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "tool_call",
            "schema": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": tool_names},
                    "args": {"type": "object"}
                },
                "required": ["tool", "args"],
                "additionalProperties": False
            }
        }
    }


def render_system_prompt(catalog):
    """Render system prompt with full catalog."""
    tool_sigs = []
    for tool in catalog:
        if tool["name"] == "none":
            continue
        args = ", ".join(tool.get("required", []))
        tool_sigs.append(f"{tool['name']}({args})")
    tools_str = ", ".join(tool_sigs)

    return (
        f"You are a tool-calling assistant. Available tools: {tools_str}. "
        "Reply with ONLY one JSON object of the form "
        "{\"tool\": \"<name>\", \"args\": {...}} and nothing else. "
        "If no available tool fits the request, reply exactly "
        "{\"tool\": \"none\", \"args\": {}}.\nRequest: %s\nJSON:"
    )


def score_raw_leg(rows, port, system_prompt, catalog, schemas):
    """Score raw /v1/completions leg."""
    results = []
    n = len(rows)
    parses = tool_ok = schema_ok = exact = 0

    for row in rows:
        prompt = system_prompt % row["prompt"]
        body = json.dumps({
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                text = json.loads(r.read())["choices"][0]["text"]
        except Exception as e:
            text = None

        obj = extract_json(text) if text else None
        gold = row["gold"]

        verdicts = {
            "id": row["id"],
            "category": row.get("category", "unknown"),
            "prompt": row["prompt"],
            "output": text or "",
            "parsed": obj,
            "gold_tool": gold["tool"],
            "gold_args": gold["args"],
            "json_parse": False,
            "right_tool": False,
            "schema_valid": False,
            "exact_match": False
        }

        if obj and isinstance(obj, dict):
            verdicts["json_parse"] = True
            parses += 1

            if obj.get("tool") == gold["tool"]:
                verdicts["right_tool"] = True
                tool_ok += 1

                a = obj.get("args")
                want = schemas.get(gold["tool"], set())
                valid_schema = (isinstance(a, dict) and
                               set(a.keys()) == want and
                               all(str(v).strip() for v in a.values()))

                if not valid_schema:
                    if gold["tool"] == "none" and a in ({}, None):
                        valid_schema = True

                if valid_schema:
                    verdicts["schema_valid"] = True
                    schema_ok += 1

                    if obj == gold:
                        verdicts["exact_match"] = True
                        exact += 1

        results.append(verdicts)

    return {
        "n": n,
        "json_parses": parses,
        "right_tool": tool_ok,
        "schema_valid": schema_ok,
        "exact_match": exact,
        "json_parses_rate": round(parses / n, 4) if n > 0 else 0,
        "right_tool_rate": round(tool_ok / n, 4) if n > 0 else 0,
        "schema_valid_rate": round(schema_ok / n, 4) if n > 0 else 0,
        "exact_match_rate": round(exact / n, 4) if n > 0 else 0,
        "rows": results
    }


def score_native_leg(rows, port, catalog):
    """Score /v1/chat/completions native tools leg."""
    results = []
    n = len(rows)
    tool_ok = args_ok = exact = 0

    tools = []
    for tool in catalog:
        if tool["name"] == "none":
            continue
        tool_def = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": tool.get("args", {}),
                    "required": tool.get("required", [])
                }
            }
        }
        tools.append(tool_def)

    for row in rows:
        body = json.dumps({
            "model": "ignored",
            "messages": [{"role": "user", "content": row["prompt"]}],
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 64,
            "choice_logprobs": True
        }).encode()

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"}
        )

        refusal = None
        calls = []
        choice_records = []

        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.loads(r.read())
                choice = resp["choices"][0]

                if choice.get("finish_reason") == "tool_calls":
                    calls = choice.get("tool_calls", [])

                # Extract choice_logprobs if available
                choice_logprobs = choice.get("choice_logprobs")
                if choice_logprobs:
                    for rec in choice_logprobs:
                        choice_records.append({
                            "index": rec.get("index"),
                            "n_legal": len(rec.get("alternatives", [])),
                            "coverage": rec.get("coverage", 0),
                            "alternatives": rec.get("alternatives", [])
                        })
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:200]
            refusal = f"HTTP {e.code}: {detail}"
        except Exception as e:
            refusal = str(e)[:200]

        gold = row["gold"]
        gold_tool = gold["tool"]
        gold_args = gold["args"]

        # Score
        tool_match = False
        args_match = False
        exact_match = False

        if calls and gold_tool != "none":
            # Check if the first tool call matches
            call = calls[0]
            tool_name = call.get("function", {}).get("name")
            tool_args = call.get("function", {}).get("arguments", {})

            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except Exception:
                    tool_args = {}

            if tool_name == gold_tool:
                tool_match = True
                tool_ok += 1

                if tool_args == gold_args:
                    args_match = True
                    args_ok += 1
                    exact_match = True
                    exact += 1
        elif not calls and gold_tool == "none":
            tool_match = True
            tool_ok += 1
            args_match = True
            args_ok += 1
            exact_match = True
            exact += 1

        verdicts = {
            "id": row["id"],
            "category": row.get("category", "unknown"),
            "prompt": row["prompt"],
            "gold_tool": gold_tool,
            "gold_args": gold_args,
            "emitted_calls": calls,
            "tool_match": tool_match,
            "args_match": args_match,
            "exact_match": exact_match,
            "refusal": refusal,
            "choice_records": choice_records
        }
        results.append(verdicts)

    return {
        "n": n,
        "tool_ok": tool_ok,
        "args_ok": args_ok,
        "exact_match": exact,
        "tool_ok_rate": round(tool_ok / n, 4) if n > 0 else 0,
        "args_ok_rate": round(args_ok / n, 4) if n > 0 else 0,
        "exact_match_rate": round(exact / n, 4) if n > 0 else 0,
        "rows": results
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runner", default="./runner")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora")
    ap.add_argument("--lora-scale", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--freeze", action="store_true",
                    help="Only freeze labels, do not run evaluation")
    args = ap.parse_args()

    # Load catalog
    catalog, schemas = load_catalog("evals/tooluse-shifted/catalog-v1.json")

    # Load labels and verify
    labels, labels_hash = load_labels_and_verify()

    if args.freeze:
        print(f"Labels frozen: sha256={labels_hash}")
        return

    # Load eval set
    rows = []
    with open("evals/tooluse-shifted/set-v1.jsonl") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)

    # Build gold from labels
    gold_by_id = {label[0]: {"tool": label[1], "args": label[2]}
                  for label in labels}
    for row in rows:
        row["gold"] = gold_by_id[row["id"]]

    # Start server
    port = free_port()
    print(f"Starting server on port {port}...", file=sys.stderr)
    srv = serve(args, args.lora, port)

    if not wait_ready(port):
        srv.kill()
        sys.exit("Server failed to start")

    time.sleep(1)  # Brief stabilization

    print("Server ready, scoring...", file=sys.stderr)

    # Score both legs
    system_prompt = render_system_prompt(catalog)

    try:
        print("Raw leg...", file=sys.stderr)
        raw_scores = score_raw_leg(rows, port, system_prompt, catalog, schemas)

        print("Native leg...", file=sys.stderr)
        native_scores = score_native_leg(rows, port, catalog)
    finally:
        stop(srv)

    # Prepare output record
    adapter_name = os.path.basename(args.lora) if args.lora else None
    host_info = platform.platform()

    record = {
        "schema_version": "xyntetik.runner.tooluse-shifted.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_file": os.path.basename(args.model),
        "model_sha256": sha256_file(args.model),
        "adapter_file": adapter_name,
        "adapter_sha256": sha256_file(args.lora) if args.lora else None,
        "adapter_scale": args.lora_scale if args.lora else None,
        "runner_version": runner_version(args.runner),
        "host": host_info,
        "catalog_sha256": sha256_file("evals/tooluse-shifted/catalog-v1.json"),
        "set_sha256": sha256_file("evals/tooluse-shifted/set-v1.jsonl"),
        "labels_sha256": labels_hash,
        "raw_leg": raw_scores,
        "native_leg": native_scores
    }

    # Save result
    out_dir = "evals/tooluse-shifted/results"
    os.makedirs(out_dir, exist_ok=True)

    model_base = os.path.basename(args.model).replace(".gguf", "")
    adapter_suffix = f"-{adapter_name.replace('.gguf', '')}" if adapter_name else "-base"
    host_suffix = platform.system().lower()

    out_file = f"{out_dir}/{model_base}{adapter_suffix}-{host_suffix}.json"

    with open(out_file, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\nResult written to {out_file}", file=sys.stderr)

    # Print summary
    print(f"\n{model_base}{adapter_suffix} on {host_suffix}:")
    print(f"  Raw leg (greedy /v1/completions):")
    print(f"    - json_parse:   {raw_scores['json_parses']:3d}/{raw_scores['n']} ({raw_scores['json_parses_rate']:.1%})")
    print(f"    - right_tool:   {raw_scores['right_tool']:3d}/{raw_scores['n']} ({raw_scores['right_tool_rate']:.1%})")
    print(f"    - schema_valid: {raw_scores['schema_valid']:3d}/{raw_scores['n']} ({raw_scores['schema_valid_rate']:.1%})")
    print(f"    - exact_match:  {raw_scores['exact_match']:3d}/{raw_scores['n']} ({raw_scores['exact_match_rate']:.1%})")
    print(f"  Native leg (/v1/chat/completions with tools):")
    print(f"    - tool_ok:      {native_scores['tool_ok']:3d}/{native_scores['n']} ({native_scores['tool_ok_rate']:.1%})")
    print(f"    - args_ok:      {native_scores['args_ok']:3d}/{native_scores['n']} ({native_scores['args_ok_rate']:.1%})")
    print(f"    - exact_match:  {native_scores['exact_match']:3d}/{native_scores['n']} ({native_scores['exact_match_rate']:.1%})")


if __name__ == "__main__":
    main()
