#!/usr/bin/env python3
"""Tool-use evaluation with shifted distribution and four scoring legs.

Evaluates a base model and optional adapter on a hand-written tool-use prompt
set with distribution shift from the original training templates. Four legs:

  raw     greedy /v1/completions on the training template with the full
          catalog; JSON parse, right tool, schema, field-level arguments.
  native  /v1/chat/completions with tools (the model's own protocol),
          choice_logprobs recorded at every constrained decision point.
  decide  POST /v1/decide (continuation-v1) on the training template: the
          tool choice as a distribution over every catalog name, scored with
          log loss, Brier and ECE, and a confidence certificate.
  next    the second call of a multi-intent request, teacher-forced: the
          gold first call and a canned result are replayed and the following
          call is scored (set v2 only; rows carrying gold_next).

Usage:
    eval-tooluse-shifted.py --runner ./runner --model M.gguf \\
        [--lora A.gguf [--lora-scale S]] [--threads N] [--gpu off|auto]
        [--set v1|v2] [--decisions-out FILE] [--freeze]
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

EVAL_DIR = "evals/tooluse-shifted"
SETS = {
    "v1": {"set": "set-v1.jsonl", "labels": "LABELS.sha256", "next": False},
    "v2": {"set": "set-v2.jsonl", "labels": "LABELS-v2.sha256", "next": True},
}
SCHEMA_VERSION = "xyntetik.runner.tooluse-shifted.v2"


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


def canonical_labels(rows, version):
    """The label rows whose sha256 is frozen. v1 froze [id, tool, args];
    v2 also freezes the synonym list and the second call."""
    labels = []
    for row in rows:
        if version == "v1":
            labels.append([row["id"], row["gold_tool"], row["gold_args"]])
        else:
            labels.append([row["id"], row["gold_tool"], row["gold_args"],
                           row.get("also_ok", []), row.get("gold_next")])
    labels.sort(key=lambda x: x[0])
    return labels


def labels_hash(rows, version):
    canonical = "\n".join(json.dumps(label, ensure_ascii=False)
                          for label in canonical_labels(rows, version))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_set(version):
    """Load the set and verify the labels against the frozen hash. Every row
    gets a `gold` dict: tool, args, also_ok (synonym siblings that count as
    the right tool) and next (the second call, or None)."""
    spec = SETS[version]
    rows = []
    with open(os.path.join(EVAL_DIR, spec["set"])) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    current = labels_hash(rows, version)
    with open(os.path.join(EVAL_DIR, spec["labels"])) as f:
        stored = f.read().strip()
    if current != stored:
        sys.exit(f"Labels have been modified! Expected {stored}, got {current}")
    for row in rows:
        row["gold"] = {"tool": row["gold_tool"], "args": row["gold_args"],
                       "also_ok": list(row.get("also_ok", [])),
                       "next": row.get("gold_next")}
    return rows, current


def limit_rows(rows, per_category):
    """The first `per_category` rows of each category, in set order."""
    seen = {}
    out = []
    for r in rows:
        c = r.get("category", "unknown")
        if seen.get(c, 0) < per_category:
            seen[c] = seen.get(c, 0) + 1
            out.append(r)
    return out


def load_labels_and_verify():
    """v1 entry point kept for callers of the first scorer."""
    rows, h = load_set("v1")
    return canonical_labels(rows, "v1"), h


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
           "--no-tray", "--gpu", getattr(args, "gpu", "off")]
    if args.threads > 0:
        cmd += ["-t", str(args.threads)]
    if adapter:
        cmd += ["--lora", adapter, "--lora-scale", str(args.lora_scale)]

    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def served_model_id(port):
    """The id the server serves (GET /v1/models), or None. The chat leg must
    name a served model: a placeholder is refused with 404 and a refused
    request is not a decision (the first ZEN run scored 150 refusals per
    arm as 52 none-matches, 2026-09-23)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=20) as r:
            body = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    ids = [m.get("id") for m in body.get("data", []) if isinstance(m, dict) and m.get("id")]
    return ids[0] if ids else None


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


def post_json(url, body, timeout=600):
    """POST a JSON body; returns (parsed_response, refusal_string)."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        return None, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return None, str(e)[:200]


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


# ------------------------------------------------------------ field scoring

def canon_path(p):
    if p in ("./", "."):
        return "."
    if p.startswith("./"):
        p = p[2:]
    if len(p) > 1 and p.endswith("/"):
        p = p[:-1]
    return p


def norm_value(got, gold):
    """Normalise a produced argument value against the gold value's shape:
    strings are stripped, a path's trailing slash and a leading './' are
    dropped, and a digit string is read as an integer when the gold is one.
    The label decides the shape; the model is not penalised for spelling
    '.' as './'."""
    if isinstance(gold, bool):
        return got
    if isinstance(gold, int):
        if isinstance(got, bool):
            return got
        if isinstance(got, int):
            return got
        if isinstance(got, float) and got.is_integer():
            return int(got)
        if isinstance(got, str) and got.strip().lstrip("-").isdigit():
            return int(got.strip())
        return got
    if isinstance(gold, str) and isinstance(got, str):
        s = got.strip()
        g = gold.strip()
        if s != g and ("/" in s or "/" in g or s in ("./", ".") or g == "."):
            s = canon_path(s)
        return s
    return got


def gold_value(v):
    if isinstance(v, str):
        s = v.strip()
        return canon_path(s) if ("/" in s or s in ("./", ".")) else s
    return v


def score_args(got, gold_args, required):
    """Field-level argument scoring. Returns keys_ok (the produced key set is
    exactly the required set), fields_total, fields_ok (required fields whose
    normalised value equals the gold), and exact (keys_ok and every field
    ok)."""
    if not isinstance(got, dict):
        got = {}
    keys_ok = set(got.keys()) == set(required)
    total = len(gold_args)
    ok = 0
    for k, v in gold_args.items():
        if k in got and norm_value(got[k], v) == gold_value(v):
            ok += 1
    return {"keys_ok": keys_ok, "fields_total": total, "fields_ok": ok,
            "exact": keys_ok and ok == total}


def tool_matches(name, gold):
    return name == gold["tool"] or name in gold.get("also_ok", [])


def fresh_counts():
    return {"tool_ok": 0, "keys_ok": 0, "fields_total": 0, "fields_ok": 0,
            "exact": 0, "empty_outputs": 0, "refusals": 0}


def tally(counts, verdict):
    counts["tool_ok"] += bool(verdict["tool_ok"])
    counts["keys_ok"] += bool(verdict["keys_ok"])
    counts["fields_total"] += verdict["fields_total"]
    counts["fields_ok"] += verdict["fields_ok"]
    counts["exact"] += bool(verdict["exact"])


def score_call(name, args, gold, schemas):
    """Score one produced call (name, args) against a gold call. name None
    means no call was produced, which is right only when the gold is none."""
    v = {"tool_ok": False, "keys_ok": False, "fields_total": len(gold["args"]),
         "fields_ok": 0, "exact": False}
    if name is None:
        if gold["tool"] == "none":
            v.update(tool_ok=True, keys_ok=True, exact=True)
        return v
    if not tool_matches(name, gold):
        return v
    v["tool_ok"] = True
    if gold["tool"] == "none":
        ok = args in ({}, None)
        v.update(keys_ok=ok, exact=ok)
        return v
    fs = score_args(args, gold["args"], schemas.get(gold["tool"], set()))
    v.update(keys_ok=fs["keys_ok"], fields_ok=fs["fields_ok"], exact=fs["exact"])
    return v


FAILED = {"tool": "__failed__", "args": {}, "also_ok": []}


def per_category(rows_out):
    out = {}
    for r in rows_out:
        c = out.setdefault(r["category"], {"n": 0, "tool_ok": 0, "exact": 0,
                                           "fields_total": 0, "fields_ok": 0})
        c["n"] += 1
        c["tool_ok"] += bool(r["tool_ok"])
        c["exact"] += bool(r["exact"])
        c["fields_total"] += r.get("fields_total", 0)
        c["fields_ok"] += r.get("fields_ok", 0)
    return out


# ------------------------------------------------------------------ raw leg

def score_raw_leg(rows, port, system_prompt, catalog, schemas):
    """Score raw /v1/completions leg."""
    results = []
    n = len(rows)
    counts = fresh_counts()
    parses = schema_ok = 0

    for row in rows:
        prompt = system_prompt % row["prompt"]
        resp, refusal = post_json(f"http://127.0.0.1:{port}/v1/completions",
                                  {"prompt": prompt, "max_tokens": 64, "temperature": 0})
        text = None
        if resp is not None:
            try:
                text = resp["choices"][0]["text"]
            except Exception:
                refusal = "malformed response"
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
            "exact_match": False,
            "refusal": refusal,
        }
        if refusal:
            counts["refusals"] += 1
        elif not text:
            counts["empty_outputs"] += 1

        if obj and isinstance(obj, dict):
            verdicts["json_parse"] = True
            parses += 1
            name = obj.get("tool")
            args = obj.get("args")
            if tool_matches(name, gold):
                verdicts["right_tool"] = True
                want = schemas.get(gold["tool"], set())
                valid_schema = (isinstance(args, dict) and set(args.keys()) == want and
                                all(str(v).strip() for v in args.values()))
                if not valid_schema and gold["tool"] == "none" and args in ({}, None):
                    valid_schema = True
                if valid_schema:
                    verdicts["schema_valid"] = True
                    schema_ok += 1
            sc = score_call(name, args, gold, schemas)
        else:
            # nothing parseable is a wrong answer even when the gold is none:
            # the template asks for a JSON object either way
            sc = score_call(None, None, {**FAILED, "args": gold["args"]}, schemas)
        verdicts.update(tool_ok=sc["tool_ok"], keys_ok=sc["keys_ok"],
                        fields_total=sc["fields_total"], fields_ok=sc["fields_ok"],
                        exact=sc["exact"], exact_match=sc["exact"])
        tally(counts, sc)
        results.append(verdicts)

    return {
        "n": n,
        "json_parses": parses,
        "right_tool": counts["tool_ok"],
        "schema_valid": schema_ok,
        "exact_match": counts["exact"],
        "keys_ok": counts["keys_ok"],
        "fields_total": counts["fields_total"],
        "fields_ok": counts["fields_ok"],
        "empty_outputs": counts["empty_outputs"],
        "refusals": counts["refusals"],
        "json_parses_rate": round(parses / n, 4) if n > 0 else 0,
        "right_tool_rate": round(counts["tool_ok"] / n, 4) if n > 0 else 0,
        "schema_valid_rate": round(schema_ok / n, 4) if n > 0 else 0,
        "exact_match_rate": round(counts["exact"] / n, 4) if n > 0 else 0,
        "by_category": per_category(results),
        "rows": results
    }


# --------------------------------------------------------------- native leg

def native_tools(catalog):
    tools = []
    for tool in catalog:
        if tool["name"] == "none":
            continue
        tools.append({
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
        })
    return tools


def chat_request(model_id, messages, tools):
    return {
        "model": model_id,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        # Qwen3 thinks by default and a 64-token budget was spent inside
        # the think block on every prompt (the first ZEN run: 0 calls,
        # every row finish_reason length). The leg measures the decision,
        # not the reasoning budget: thinking off, a budget a call fits in.
        "enable_thinking": False,
        "max_tokens": 256,
        "choice_logprobs": True
    }


def first_call(resp):
    """(finish_reason, content_head, calls, choice_records) from a chat response."""
    choice = resp["choices"][0]
    finish = choice.get("finish_reason")
    msg = choice.get("message") or {}
    content = (msg.get("content") or "")[:200] or None
    calls = []
    if finish == "tool_calls":
        calls = msg.get("tool_calls") or choice.get("tool_calls", [])
    records = []
    for rec in choice.get("choice_logprobs") or []:
        records.append({"index": rec.get("index"),
                        "n_legal": len(rec.get("alternatives", [])),
                        "coverage": rec.get("coverage", 0),
                        "alternatives": rec.get("alternatives", [])})
    return finish, content, calls, records


def call_name_args(call):
    fn = call.get("function", {}) if isinstance(call, dict) else {}
    name = fn.get("name")
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    return name, args


def score_native_leg(rows, port, catalog, model_id, schemas=None):
    """Score /v1/chat/completions native tools leg. A refused request is a
    failed row, never a match, and the refusal count is reported."""
    schemas = schemas or {t["name"]: set(t.get("required", [])) for t in catalog}
    results = []
    n = len(rows)
    counts = fresh_counts()
    tools = native_tools(catalog)

    for row in rows:
        body = chat_request(model_id, [{"role": "user", "content": row["prompt"]}], tools)
        resp, refusal = post_json(f"http://127.0.0.1:{port}/v1/chat/completions", body)
        finish = content = None
        calls, choice_records = [], []
        if resp is not None:
            try:
                finish, content, calls, choice_records = first_call(resp)
            except Exception:
                refusal = "malformed response"

        gold = row["gold"]
        if refusal is not None:
            counts["refusals"] += 1
            sc = score_call(None, None, {**FAILED, "args": gold["args"]}, schemas)
        else:
            if not calls and not content:
                counts["empty_outputs"] += 1
            name, args = call_name_args(calls[0]) if calls else (None, None)
            sc = score_call(name, args, gold, schemas)
            tally(counts, sc)

        results.append({
            "id": row["id"],
            "category": row.get("category", "unknown"),
            "prompt": row["prompt"],
            "gold_tool": gold["tool"],
            "gold_args": gold["args"],
            "emitted_calls": calls,
            "tool_match": sc["tool_ok"],
            "args_match": sc["exact"],
            "exact_match": sc["exact"],
            "tool_ok": sc["tool_ok"],
            "keys_ok": sc["keys_ok"],
            "fields_total": sc["fields_total"],
            "fields_ok": sc["fields_ok"],
            "exact": sc["exact"],
            "refusal": refusal,
            "finish_reason": finish,
            "content_head": content,
            "choice_records": choice_records
        })

    return {
        "n": n,
        "tool_ok": counts["tool_ok"],
        "args_ok": counts["exact"],
        "exact_match": counts["exact"],
        "keys_ok": counts["keys_ok"],
        "fields_total": counts["fields_total"],
        "fields_ok": counts["fields_ok"],
        "refusals": counts["refusals"],
        "empty_outputs": counts["empty_outputs"],
        "tool_ok_rate": round(counts["tool_ok"] / n, 4) if n > 0 else 0,
        "args_ok_rate": round(counts["exact"] / n, 4) if n > 0 else 0,
        "exact_match_rate": round(counts["exact"] / n, 4) if n > 0 else 0,
        "by_category": per_category(results),
        "rows": results
    }


# ----------------------------------------------------------------- next leg

CANNED_RESULTS = {
    "search_files": "src/a.py\nsrc/b.py\n",
    "find_files": "tests/test_api.py\ntests/test_cli.py\n",
    "grep_text": "src/a.py:12: TODO tidy\nsrc/b.py:40: TODO remove\n",
    "read_file": "line one\nline two\nline three\n",
    "read_lines": "line ten\nline eleven\n",
    "head_file": "line one\nline two\n",
    "cat_file": "line one\nline two\nline three\n",
    "write_file": "ok: written",
    "append_file": "ok: appended",
    "replace_in_file": "ok: 2 replacements",
    "truncate_file": "ok: truncated",
    "list_dir": "a.py\nb.py\nsub/\n",
    "tree": ".\n|-- a.py\n`-- sub\n    `-- c.py\n",
    "ls_recursive": "a.py\nsub/c.py\n",
    "run_command": "exit 0\n",
    "run_tests": "3 passed\n",
    "run_pytest": "1 failed, 4 passed\n",
    "git_status": "On branch main\nnothing to commit, working tree clean\n",
    "git_diff": "diff --git a/a.py b/a.py\n-old\n+new\n",
    "git_log": "abc123 first commit\n",
    "http_get": "200 OK\n{\"status\": \"ok\"}\n",
    "fetch_url": "200 OK\n{\"status\": \"ok\"}\n",
    "download_file": "ok: saved 1024 bytes",
}


def next_messages(row):
    """The teacher-forced transcript: the user turn, the GOLD first call as
    the assistant's tool call, and a canned result for it."""
    gold = row["gold"]
    call = {"id": "call_0", "type": "function",
            "function": {"name": gold["tool"], "arguments": json.dumps(gold["args"])}}
    return [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_0", "name": gold["tool"],
         "content": CANNED_RESULTS.get(gold["tool"], "ok")},
    ]


def score_next_leg(rows, port, catalog, model_id, schemas=None):
    """Score the second call of each multi-intent row against gold_next."""
    schemas = schemas or {t["name"]: set(t.get("required", [])) for t in catalog}
    tools = native_tools(catalog)
    rows = [r for r in rows if r["gold"].get("next")]
    results = []
    counts = fresh_counts()
    for row in rows:
        gold = {"tool": row["gold"]["next"]["tool"], "args": row["gold"]["next"]["args"],
                "also_ok": []}
        body = chat_request(model_id, next_messages(row), tools)
        body.pop("choice_logprobs", None)
        resp, refusal = post_json(f"http://127.0.0.1:{port}/v1/chat/completions", body)
        finish = content = None
        calls = []
        if resp is not None:
            try:
                finish, content, calls, _ = first_call(resp)
            except Exception:
                refusal = "malformed response"
        if refusal is not None:
            counts["refusals"] += 1
            sc = score_call(None, None, {**FAILED, "args": gold["args"]}, schemas)
        else:
            if not calls and not content:
                counts["empty_outputs"] += 1
            name, args = call_name_args(calls[0]) if calls else (None, None)
            sc = score_call(name, args, gold, schemas)
            tally(counts, sc)
        results.append({
            "id": row["id"], "category": row.get("category", "unknown"),
            "prompt": row["prompt"],
            "first_call": {"tool": row["gold"]["tool"], "args": row["gold"]["args"]},
            "gold_next": gold, "emitted_calls": calls,
            "tool_ok": sc["tool_ok"], "keys_ok": sc["keys_ok"],
            "fields_total": sc["fields_total"], "fields_ok": sc["fields_ok"],
            "exact": sc["exact"], "refusal": refusal, "finish_reason": finish,
            "content_head": content,
        })
    n = len(rows)
    return {
        "n": n,
        "tool_ok": counts["tool_ok"],
        "exact": counts["exact"],
        "keys_ok": counts["keys_ok"],
        "fields_total": counts["fields_total"],
        "fields_ok": counts["fields_ok"],
        "refusals": counts["refusals"],
        "empty_outputs": counts["empty_outputs"],
        "tool_ok_rate": round(counts["tool_ok"] / n, 4) if n else 0,
        "exact_rate": round(counts["exact"] / n, 4) if n else 0,
        "rows": results,
    }


# --------------------------------------------------------------- decide leg

def load_calibration_module():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cl-calibration.py")
    spec = importlib.util.spec_from_file_location("cl_calibration", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decide_state(system_prompt, prompt):
    """The training template up to the opening quote of the tool name, so
    the option is scored as the verbatim continuation."""
    return (system_prompt % prompt) + ' {"tool": "'


def score_decide_leg(rows, port, system_prompt, catalog, model_id, target=0.9, bins=10):
    """Tool choice as a distribution over every catalog name (none included)
    through POST /v1/decide, rendering continuation-v1. Per row: probs over
    the options, the mass on the acceptable set (gold plus synonyms), the
    argmax, log loss, Brier. Summary: top-1 accuracy, mean log loss, mean
    Brier, ECE over top-1 confidence, and the confidence certificate."""
    options = [t["name"] for t in catalog]
    idx = {name: i for i, name in enumerate(options)}
    results = []
    decisions = []   # (confidence, hit, brier) for cl-calibration
    refusals = 0
    for row in rows:
        gold = row["gold"]
        acceptable = {gold["tool"]} | set(gold.get("also_ok", []))
        body = {"model": model_id, "state": decide_state(system_prompt, row["prompt"]),
                "rendering": "continuation-v1",
                "questions": [{"id": row["id"], "question": "tool", "options": options}]}
        resp, refusal = post_json(f"http://127.0.0.1:{port}/v1/decide", body)
        rec = {"id": row["id"], "category": row.get("category", "unknown"),
               "prompt": row["prompt"], "gold_tool": gold["tool"],
               "acceptable": sorted(acceptable), "refusal": refusal}
        d = None
        if resp is not None:
            try:
                d = resp["decisions"][0]
                if list(d["options"]) != options:
                    raise ValueError("echoed options differ")
            except Exception:
                d = None
                rec["refusal"] = "malformed response"
        if d is None:
            refusals += 1
            rec.update(top=None, top_ok=False, p_acceptable=0.0, nll=None, brier=None, probs=None)
            results.append(rec)
            continue
        probs = [float(p) for p in d["probs"]]
        top = options[int(d["argmax"])]
        p_acc = sum(probs[idx[a]] for a in acceptable)
        p_acc = min(max(p_acc, 0.0), 1.0)
        nll = max(0.0, -math.log(max(p_acc, 1e-12)))
        brier = (p_acc - 1.0) ** 2 + sum(probs[i] ** 2 for i, o in enumerate(options) if o not in acceptable)
        conf = max(probs)
        hit = top in acceptable
        rec.update(top=top, top_ok=hit, confidence=conf, p_acceptable=p_acc, nll=nll,
                   brier=brier, probs=probs, n_tokens=list(d.get("n_tokens", [])))
        decisions.append((conf, hit, brier))
        results.append(rec)

    cal = load_calibration_module()
    summary = cal.summarize(decisions, bins=bins) if decisions else \
        {"n": 0, "accuracy": None, "brier": None, "ece": None, "bins": []}
    cert = cal.certificate(decisions, target) if decisions else None
    scored = [r for r in results if r["nll"] is not None]
    by_cat = {}
    for r in scored:
        c = by_cat.setdefault(r["category"], {"n": 0, "top_ok": 0, "nll_sum": 0.0})
        c["n"] += 1
        c["top_ok"] += bool(r["top_ok"])
        c["nll_sum"] += r["nll"]
    for c in by_cat.values():
        c["mean_nll"] = round(c["nll_sum"] / c["n"], 4) if c["n"] else None
        del c["nll_sum"]
    return {
        "n": len(rows),
        "options": options,
        "rendering": "continuation-v1",
        "refusals": refusals,
        "top1_ok": sum(1 for r in scored if r["top_ok"]),
        "mean_nll": round(sum(r["nll"] for r in scored) / len(scored), 4) if scored else None,
        "mean_brier": round(summary["brier"], 4) if summary.get("brier") is not None else None,
        "ece": round(summary["ece"], 4) if summary.get("ece") is not None else None,
        "reliability": summary.get("bins", []),
        "certificate": cert,
        "by_category": by_cat,
        "rows": results,
    }


def write_decisions(decide_leg, path):
    """The decide leg as labeled decisions for scripts/cl-calibration.py:
    alternatives carry the option index as their id."""
    options = decide_leg["options"]
    with open(path, "w") as f:
        for r in decide_leg["rows"]:
            if r.get("probs") is None:
                continue
            alts = sorted(({"id": i, "prob": p} for i, p in enumerate(r["probs"])),
                          key=lambda a: -a["prob"])
            f.write(json.dumps({"id": r["id"], "alternatives": alts,
                                "correct_id": options.index(r["gold_tool"]),
                                "acceptable_ids": [options.index(a) for a in r["acceptable"]]}) + "\n")


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner", default="./runner")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--gpu", default="off", help="runner --gpu value (default off: the CPU path is the record)")
    ap.add_argument("--set", default="v2", choices=sorted(SETS), help="prompt set (default v2)")
    ap.add_argument("--legs", default="raw,native,decide,next",
                    help="comma list of legs to run (default all)")
    ap.add_argument("--certificate-target", type=float, default=0.9,
                    help="accuracy the confidence certificate must reach (default 0.9)")
    ap.add_argument("--decisions-out", help="also write the decide leg as cl-calibration.py input")
    ap.add_argument("--out", help="record path (default results/<model>-<adapter>-<set>-<host>.json)")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke runs: score only the first N rows of each category (record marked partial)")
    ap.add_argument("--freeze", action="store_true",
                    help="Only verify and print the frozen labels hash, do not run evaluation")
    args = ap.parse_args()

    catalog, schemas = load_catalog(os.path.join(EVAL_DIR, "catalog-v1.json"))
    rows, labels_sha = load_set(args.set)
    if args.limit > 0:
        rows = limit_rows(rows, args.limit)

    if args.freeze:
        print(f"Labels frozen ({args.set}): sha256={labels_sha}")
        return

    legs = {s.strip() for s in args.legs.split(",") if s.strip()}
    if "next" in legs and not SETS[args.set]["next"]:
        legs.discard("next")

    port = free_port()
    print(f"Starting server on port {port}...", file=sys.stderr)
    srv = serve(args, args.lora, port)
    if not wait_ready(port):
        srv.kill()
        sys.exit("Server failed to start")
    time.sleep(1)
    print("Server ready, scoring...", file=sys.stderr)

    system_prompt = render_system_prompt(catalog)
    record_legs = {}
    try:
        model_id = served_model_id(port)
        if not model_id:
            print("error: the server lists no model (GET /v1/models); the native, decide and next legs cannot run", file=sys.stderr)
            sys.exit(2)
        if "raw" in legs:
            print("Raw leg...", file=sys.stderr)
            record_legs["raw_leg"] = score_raw_leg(rows, port, system_prompt, catalog, schemas)
        if "native" in legs:
            print("Native leg...", file=sys.stderr)
            record_legs["native_leg"] = score_native_leg(rows, port, catalog, model_id, schemas)
        if "decide" in legs:
            print("Decide leg...", file=sys.stderr)
            record_legs["decide_leg"] = score_decide_leg(rows, port, system_prompt, catalog,
                                                         model_id, target=args.certificate_target)
        if "next" in legs:
            print("Next-call leg...", file=sys.stderr)
            record_legs["next_leg"] = score_next_leg(rows, port, catalog, model_id, schemas)
    finally:
        stop(srv)

    adapter_name = os.path.basename(args.lora) if args.lora else None
    record = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "set_version": args.set,
        "set_file": SETS[args.set]["set"],
        "model_file": os.path.basename(args.model),
        "model_sha256": sha256_file(args.model),
        "adapter_file": adapter_name,
        "adapter_sha256": sha256_file(args.lora) if args.lora else None,
        "adapter_scale": args.lora_scale if args.lora else None,
        "runner_version": runner_version(args.runner),
        "host": platform.platform(),
        "threads": args.threads,
        "gpu": args.gpu,
        "catalog_sha256": sha256_file(os.path.join(EVAL_DIR, "catalog-v1.json")),
        "set_sha256": sha256_file(os.path.join(EVAL_DIR, SETS[args.set]["set"])),
        "labels_sha256": labels_sha,
        "model_id": model_id,
        "partial": args.limit if args.limit > 0 else None,
    }
    record.update(record_legs)

    out_dir = os.path.join(EVAL_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    model_base = os.path.basename(args.model).replace(".gguf", "")
    adapter_suffix = f"-{adapter_name.replace('.gguf', '')}" if adapter_name else "-base"
    host_suffix = platform.system().lower()
    set_suffix = "" if args.set == "v1" else f"-{args.set}"
    out_file = args.out or f"{out_dir}/{model_base}{adapter_suffix}{set_suffix}-{host_suffix}.json"
    with open(out_file, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nResult written to {out_file}", file=sys.stderr)
    if args.decisions_out and "decide_leg" in record_legs:
        write_decisions(record_legs["decide_leg"], args.decisions_out)

    fatal = False
    for leg in ("native_leg", "decide_leg", "next_leg"):
        lg = record_legs.get(leg)
        if lg and lg.get("n") and lg.get("refusals", 0) == lg["n"]:
            print(f"error: every {leg} request was refused; the record is written but is not a measurement", file=sys.stderr)
            fatal = True

    print(f"\n{model_base}{adapter_suffix} ({args.set}) on {host_suffix}:")
    rl = record_legs.get("raw_leg")
    if rl:
        print("  Raw leg (greedy /v1/completions):")
        print(f"    - json_parse:   {rl['json_parses']:3d}/{rl['n']}")
        print(f"    - right_tool:   {rl['right_tool']:3d}/{rl['n']}")
        print(f"    - exact_match:  {rl['exact_match']:3d}/{rl['n']}")
        print(f"    - fields_ok:    {rl['fields_ok']:3d}/{rl['fields_total']}   empty {rl['empty_outputs']}  refused {rl['refusals']}")
    nl = record_legs.get("native_leg")
    if nl:
        print("  Native leg (/v1/chat/completions with tools):")
        print(f"    - tool_ok:      {nl['tool_ok']:3d}/{nl['n']}")
        print(f"    - exact_match:  {nl['exact_match']:3d}/{nl['n']}")
        print(f"    - fields_ok:    {nl['fields_ok']:3d}/{nl['fields_total']}   empty {nl['empty_outputs']}  refused {nl['refusals']}")
    dl = record_legs.get("decide_leg")
    if dl:
        print("  Decide leg (/v1/decide continuation-v1 over the catalog):")
        print(f"    - top1_ok:      {dl['top1_ok']:3d}/{dl['n'] - dl['refusals']}   refused {dl['refusals']}")
        print(f"    - mean_nll:     {dl['mean_nll']}   brier {dl['mean_brier']}   ece {dl['ece']}")
        c = dl.get("certificate") or {}
        print(f"    - certificate:  threshold {c.get('threshold')} covers {c.get('coverage')} at accuracy {c.get('accuracy')} (target {c.get('target')})")
    xl = record_legs.get("next_leg")
    if xl:
        print("  Next-call leg (teacher-forced second call):")
        print(f"    - tool_ok:      {xl['tool_ok']:3d}/{xl['n']}")
        print(f"    - exact:        {xl['exact']:3d}/{xl['n']}   empty {xl['empty_outputs']}  refused {xl['refusals']}")
    if fatal:
        sys.exit(2)


if __name__ == "__main__":
    main()
