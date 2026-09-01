#!/usr/bin/env python3
"""Tool-choice decision-boundary lane over `choice_logprobs` (UNLABELED).

Runs a bank of deliberately ambiguous tool-choice prompts under one or more
serving conditions (a base model, the same base with different adapters)
and records, per prompt and condition, the grammar decision at which the
tool name is chosen: the legal alternatives the grammar exposed, the top
branch, the runner-up, their raw-logprob margin, the probed coverage, and
the full deterministic call. `summarize` then reports where conditions
DISAGREE and how tight the margins are.

What this lane is not: it carries no gold labels and computes no accuracy,
on purpose. A label would turn a decision-sensitivity probe into a quality
benchmark and quietly change the question. Its records therefore cannot be
fed to `cl-calibration.py` (which needs a `correct_id` per decision), and a
disagreement count is a property of the bank that was run, never a rate in
real traffic.

The default bank is the "Xyntetik Runner Tool-Choice Boundary Prompt Bank"
v1.0, originally contributed by John6666, 2026, dual-licensed MIT OR
Apache-2.0 (Apache-2.0 selected here), carried byte-identical to the
published file. Its five families name how the cases were constructed, not
which tool is right. See docs/tool-choice-boundary-lane.md.

Usage:

    tool-choice-boundary.py run --runner ./runner --model base.gguf \\
        --condition base --condition bf16=adapter-bf16.gguf \\
        --condition q4=adapter-q4.gguf --out records.jsonl
    tool-choice-boundary.py run --url http://127.0.0.1:8080 \\
        --condition NAME --out records.jsonl      # one already-served condition
    tool-choice-boundary.py summarize records.jsonl [--json]

No third-party imports.
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

RECORD_VERSION = "tool_choice_boundary_record_v1"

# The study's tool set and prompt contract, verbatim from make-tooluse-data.py
# (the train.jsonl the published adapters were trained against). The tool
# decision is the grammar branch whose legal alternatives are the first
# pieces of these names.
TOOLS = ["search_files", "read_file", "write_file", "list_dir", "none"]
TOOL_SIGNATURES = "search_files(pattern, path), read_file(path), " \
                  "write_file(path, content), list_dir(path)"
SYSTEM = ("You are a tool-calling assistant. Available tools: %s. "
          "Reply with ONLY one JSON object of the form "
          "{\"tool\": \"<name>\", \"args\": {...}} and nothing else. "
          "If no available tool fits the request, reply exactly "
          "{\"tool\": \"none\", \"args\": {}}.\nRequest: %%s\nJSON:"
          % TOOL_SIGNATURES)

DEFAULT_BANK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "tool-choice-boundary-bank.jsonl")


def schema_for(tools):
    return {"type": "json_schema", "json_schema": {"name": "tool_call",
            "schema": {"type": "object", "properties": {
                "tool": {"type": "string", "enum": list(tools)},
                "args": {"type": "object"}},
                "required": ["tool", "args"],
                "additionalProperties": False}}}


def tool_for_token(token, tools):
    """The tool a legal piece leads to, or None when it is ambiguous or not
    a tool piece at all. `no`, `non` and `n` are all the `none` branch."""
    s = token.strip().lstrip("\"'")
    if not s:
        return None
    hits = [t for t in tools if t.startswith(s)]
    return hits[0] if len(hits) == 1 else None


def tool_decision(recs, tools):
    """The first decision record whose legal alternatives lead to at least
    two distinct tools, read at the tool level: the best logprob per tool,
    the top branch, the runner-up (a DIFFERENT tool, never a shorter piece
    of the same name) and their raw-logprob margin."""
    for rec in recs:
        by_tool, piece = {}, {}
        for a in rec["alternatives"]:
            t = tool_for_token(a["token"], tools)
            if t is None:
                continue
            if t not in by_tool or a["logprob"] > by_tool[t]:
                by_tool[t] = a["logprob"]
                piece[t] = a["token"]
        if len(by_tool) < 2:
            continue
        ranked = sorted(by_tool.items(), key=lambda kv: -kv[1])
        (t1, lp1), (t2, lp2) = ranked[0], ranked[1]
        return {"index": rec["index"], "n_legal": rec["n_legal"],
                "coverage": rec["coverage"],
                "alternatives": rec["alternatives"],
                "by_tool": by_tool, "top1": t1, "top1_piece": piece[t1],
                "top2": t2, "margin_nat": lp1 - lp2}
    return None


def parse_call(text):
    """The emitted call as {tool, args}, or None if it is not one object."""
    try:
        obj = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(text[start:end + 1])
        except Exception:
            return None
    if not isinstance(obj, dict) or "tool" not in obj:
        return None
    return {"tool": obj.get("tool"), "args": obj.get("args")}


def load_bank(path):
    data = open(path, "rb").read()
    rows = [json.loads(l) for l in data.decode("utf-8").splitlines()
            if l.strip()]
    for r in rows:
        missing = {"id", "family", "prompt"} - set(r)
        if missing:
            sys.exit("bank row %r lacks %s" % (r, sorted(missing)))
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        sys.exit("bank has duplicate ids")
    return rows, hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- serving

def _get(url, path, timeout=2):
    with urllib.request.urlopen(url + path, timeout=timeout) as r:
        return json.load(r)


def _post(url, path, body, timeout=600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        sys.exit("%s %s -> HTTP %d: %s" % (path, body.get("prompt", "")[:40],
                                         e.code, detail))


def wait_ready(url, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            _get(url, "/health")
            return True
        except Exception:
            time.sleep(1)
    return False


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(args, adapter, port):
    cmd = [args.runner, "-m", args.model, "--serve", "--port", str(port),
           "--no-tray", "--gpu", args.gpu]
    if args.threads > 0:
        cmd += ["-t", str(args.threads)]
    if adapter:
        cmd += ["--lora", adapter]
    log = open(args.serve_log, "ab") if args.serve_log else subprocess.DEVNULL
    return subprocess.Popen(cmd, stdout=log, stderr=log), log


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def probe(url, prompt, tools, max_tokens, probe_width):
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "choice_logprobs": True, "response_format": schema_for(tools)}
    if probe_width:
        body["choice_logprobs_probe"] = probe_width
    return _post(url, "/v1/completions", body)


def parse_condition(spec):
    name, _, adapter = spec.partition("=")
    if not name:
        sys.exit("--condition needs a name (NAME or NAME=adapter.gguf)")
    return name, (adapter or None)


def runner_version(binary):
    try:
        out = subprocess.run([binary, "--version"], capture_output=True,
                             text=True, timeout=30).stdout.strip()
        return out.splitlines()[0] if out else None
    except Exception:
        return None


def cmd_run(args):
    bank, bank_sha = load_bank(args.bank)
    conds = [parse_condition(c) for c in args.condition]
    if args.url and (len(conds) != 1 or args.runner or args.model):
        sys.exit("--url serves exactly one --condition and takes no "
                 "--runner/--model")
    if not args.url and not (args.runner and args.model):
        sys.exit("run needs --url, or --runner and --model")
    tools = list(TOOLS)
    template = SYSTEM
    if args.template:
        template = open(args.template, encoding="utf-8").read()
        if template.count("%s") != 1:
            sys.exit("--template must contain exactly one %s for the request")
    template_sha = hashlib.sha256(template.encode()).hexdigest()
    version = runner_version(args.runner) if args.runner else None
    hashes = {}
    if args.hash_artifacts and args.model:
        hashes[args.model] = sha256_file(args.model)
        for _, adapter in conds:
            if adapter:
                hashes[adapter] = sha256_file(adapter)
    out = open(args.out, "a", encoding="utf-8")
    n_written = 0
    for name, adapter in conds:
        proc, log = None, None
        if args.url:
            url = args.url.rstrip("/")
        else:
            port = args.port or free_port()
            url = "http://127.0.0.1:%d" % port
            proc, log = serve(args, adapter, port)
            if not wait_ready(url, args.ready_timeout):
                stop(proc)
                sys.exit("condition %s: server never became ready" % name)
        try:
            for row in bank:
                t0 = time.time()
                resp = probe(url, template % row["prompt"], tools,
                             args.max_tokens, args.probe)
                choice = resp["choices"][0]
                recs = choice.get("choice_logprobs", [])
                dec = tool_decision(recs, tools)
                call = parse_call(choice["text"])
                record = {
                    "record": RECORD_VERSION,
                    "id": row["id"], "family": row["family"],
                    "prompt": row["prompt"], "condition": name,
                    "model": os.path.basename(args.model) if args.model
                             else None,
                    "adapter": os.path.basename(adapter) if adapter else None,
                    "model_sha256": hashes.get(args.model),
                    "adapter_sha256": hashes.get(adapter),
                    "runner_version": version,
                    "gpu": args.gpu if not args.url else None,
                    "host": platform.platform(),
                    "bank_sha256": bank_sha,
                    "template_sha256": template_sha,
                    "tools": tools,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime()),
                    "full_text": choice["text"],
                    "full_call": call,
                    "n_decisions": len(recs),
                    "all_decisions": recs,
                    "decision": dec,
                    "seconds": round(time.time() - t0, 3),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                n_written += 1
                if dec:
                    print("%-6s %-16s %-13s > %-13s margin %7.3f nat  "
                          "cov %.4f  call %s" % (
                              name, row["id"], dec["top1"], dec["top2"],
                              dec["margin_nat"], dec["coverage"],
                              call["tool"] if call else None),
                          file=sys.stderr, flush=True)
                else:
                    print("%-6s %-16s (no tool branch among the probed "
                          "alternatives)  call %s" % (
                              name, row["id"], call["tool"] if call else None),
                          file=sys.stderr, flush=True)
        finally:
            if proc is not None:
                stop(proc)
            if log not in (None, subprocess.DEVNULL):
                log.close()
    out.close()
    print("wrote %d records to %s" % (n_written, args.out), file=sys.stderr)


# -------------------------------------------------------------- summarize

def load_records(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    bad = [r for r in recs if r.get("record") != RECORD_VERSION]
    if bad:
        sys.exit("%d records are not %s" % (len(bad), RECORD_VERSION))
    return recs


def _quartiles(xs):
    if len(xs) < 2:
        return {"min": min(xs), "p25": xs[0], "median": xs[0],
                "max": max(xs)}
    q = statistics.quantiles(xs, n=4)
    return {"min": min(xs), "p25": q[0], "median": statistics.median(xs),
            "max": max(xs)}


def summarize(recs):
    """Cross-condition summary as a plain dict. Unlabeled by construction:
    there is no accuracy, gold or correctness field anywhere in it."""
    conds, by, fam, adapter_of = [], {}, {}, {}
    ids = []
    for r in recs:
        if r["condition"] not in conds:
            conds.append(r["condition"])
            adapter_of[r["condition"]] = r.get("adapter")
        if r["id"] not in fam:
            ids.append(r["id"])
            fam[r["id"]] = r["family"]
        by[(r["id"], r["condition"])] = r
    adapters = [c for c in conds if adapter_of[c]]
    bases = [c for c in conds if not adapter_of[c]]

    def top1(i, c):
        r = by.get((i, c))
        return r["decision"]["top1"] if r and r.get("decision") else None

    def margin(i, c):
        r = by.get((i, c))
        return r["decision"]["margin_nat"] if r and r.get("decision") else None

    def call(i, c):
        r = by.get((i, c))
        return r["full_call"]["tool"] if r and r.get("full_call") else None

    complete = [i for i in ids if all((i, c) in by for c in conds)]
    rows = []
    for i in ids:
        for c in conds:
            r = by.get((i, c))
            if not r:
                continue
            d = r.get("decision")
            rows.append({"id": i, "family": fam[i], "condition": c,
                         "top1": d["top1"] if d else None,
                         "top2": d["top2"] if d else None,
                         "margin_nat": d["margin_nat"] if d else None,
                         "coverage": d["coverage"] if d else None,
                         "call": call(i, c)})

    disagreements = []
    if len(adapters) >= 2:
        for i in complete:
            tops = {c: top1(i, c) for c in adapters}
            if len(set(tops.values())) > 1:
                disagreements.append({
                    "id": i, "family": fam[i], "prompt": by[(i, conds[0])]["prompt"],
                    "top1": tops,
                    "margin_nat": {c: margin(i, c) for c in adapters},
                    "call": {c: call(i, c) for c in adapters},
                    "call_disagrees": len({call(i, c) for c in adapters}) > 1})
    moved = []
    if bases and adapters:
        b = bases[0]
        for i in complete:
            if any(top1(i, c) != top1(i, b) for c in adapters):
                moved.append({"id": i, "family": fam[i],
                              "adapters_agree": len({top1(i, c)
                                                     for c in adapters}) == 1})
    margins = {}
    for c in conds:
        ms = [margin(i, c) for i in ids if margin(i, c) is not None]
        if ms:
            margins[c] = dict(_quartiles(ms), n=len(ms),
                              n_le_3=sum(m <= 3 for m in ms),
                              n_le_1=sum(m <= 1 for m in ms))
    tightest = Counter()
    wider_than_all = []
    if len(adapters) >= 2:
        for i in complete:
            ms = {c: margin(i, c) for c in adapters}
            if any(v is None for v in ms.values()):
                continue
            tightest[min(ms, key=ms.get)] += 1
    covs = [r["decision"]["coverage"] for r in recs if r.get("decision")]
    top1_vs_call = [(r["id"], r["condition"]) for r in recs
                    if r.get("decision") and r.get("full_call")
                    and r["decision"]["top1"] != r["full_call"]["tool"]]
    return {
        "record": RECORD_VERSION, "unlabeled": True,
        "bank_sha256": sorted({r.get("bank_sha256") for r in recs}),
        "conditions": conds, "adapters": adapters, "bases": bases,
        "prompts": len(ids), "complete_prompts": len(complete),
        "records": len(recs),
        "decision_missing": sum(1 for r in recs if not r.get("decision")),
        "coverage": ({"min": min(covs), "median": statistics.median(covs)}
                     if covs else None),
        "top1_differs_from_call": top1_vs_call,
        "rows": rows,
        "adapter_disagreements": disagreements,
        "adapter_disagreement_families": dict(Counter(
            d["family"] for d in disagreements)),
        "base_vs_adapter_moved": moved,
        "margins": margins,
        "tightest_adapter": dict(tightest),
    }


def _f(x, w=7):
    return ("%" + str(w) + ".3f") % x if x is not None else "-"


def render(s):
    out = []
    out.append("# Tool-choice boundary summary (UNLABELED: no accuracy "
               "anywhere in this report)\n")
    out.append("- conditions: %s (adapters: %s; base: %s)" % (
        ", ".join(s["conditions"]), ", ".join(s["adapters"]) or "none",
        ", ".join(s["bases"]) or "none"))
    out.append("- prompts: %d (%d complete across all conditions); records: "
               "%d; decision missing: %d" % (
                   s["prompts"], s["complete_prompts"], s["records"],
                   s["decision_missing"]))
    if s["coverage"]:
        out.append("- coverage at the tool decision: min %.4f, median %.4f"
                   % (s["coverage"]["min"], s["coverage"]["median"]))
    out.append("- rows where the decision's top branch differs from the "
               "emitted call: %d" % len(s["top1_differs_from_call"]))
    out.append("- bank sha256: %s\n" % ", ".join(
        str(b) for b in s["bank_sha256"]))
    if s["adapters"]:
        out.append("## Adapter branch disagreements: %d of %d prompts" % (
            len(s["adapter_disagreements"]), s["complete_prompts"]))
        out.append("A property of this bank, not a rate in real traffic. "
                   "Families: %s\n" % (
                       s["adapter_disagreement_families"] or "{}"))
        for d in s["adapter_disagreements"]:
            out.append("- **%s** (%s): %s; margins %s; calls %s%s :: %r" % (
                d["id"], d["family"], d["top1"],
                {c: round(m, 3) if m is not None else None
                 for c, m in d["margin_nat"].items()},
                d["call"], " (calls DISAGREE)" if d["call_disagrees"] else "",
                d["prompt"]))
        out.append("")
    if s["base_vs_adapter_moved"]:
        agree = sum(1 for m in s["base_vs_adapter_moved"] if m["adapters_agree"])
        out.append("## Branch moved by the adaptation (vs %s): %d prompts, "
                   "adapters unanimous on %d\n" % (
                       s["bases"][0], len(s["base_vs_adapter_moved"]), agree))
    out.append("## Margins (top1 - top2 raw logprob, nat)\n")
    out.append("| condition | n | min | p25 | median | max | n <= 3 | n <= 1 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for c, m in s["margins"].items():
        out.append("| %s | %d | %.3f | %.3f | %.3f | %.3f | %d | %d |" % (
            c, m["n"], m["min"], m["p25"], m["median"], m["max"],
            m["n_le_3"], m["n_le_1"]))
    if s["tightest_adapter"]:
        out.append("\nTightest adapter per prompt: %s" % s["tightest_adapter"])
    out.append("\n## All decisions\n")
    out.append("| id | family | condition | top1 | top2 | margin | coverage | call |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in s["rows"]:
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["id"], r["family"], r["condition"], r["top1"], r["top2"],
            _f(r["margin_nat"], 0), _f(r["coverage"], 0), r["call"]))
    return "\n".join(out) + "\n"


def cmd_summarize(args):
    s = summarize(load_records(args.records))
    if args.json:
        json.dump(s, sys.stdout, indent=1)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render(s))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="probe every bank prompt per condition")
    r.add_argument("--runner", help="runner binary to launch per condition")
    r.add_argument("--model", help="base GGUF")
    r.add_argument("--url", help="already-running server (one condition)")
    r.add_argument("--condition", action="append", required=True,
                   help="NAME (base) or NAME=adapter.gguf; repeatable")
    r.add_argument("--bank", default=DEFAULT_BANK)
    r.add_argument("--template",
                   help="prompt template file with one %%s for the request "
                        "(default: the study's own contract, verbatim); its "
                        "sha256 is recorded so records never mix silently")
    r.add_argument("--out", required=True, help="records JSONL (appended)")
    r.add_argument("--max-tokens", type=int, default=64)
    r.add_argument("--probe", type=int, default=0,
                   help="choice_logprobs_probe width (server default if 0)")
    r.add_argument("--threads", type=int, default=0)
    r.add_argument("--gpu", default="off",
                   help="--gpu mode for launched servers (default off: one "
                        "backend, comparable across hosts)")
    r.add_argument("--port", type=int, default=0)
    r.add_argument("--ready-timeout", type=int, default=300)
    r.add_argument("--serve-log", help="append launched servers' output here")
    r.add_argument("--hash-artifacts", action="store_true",
                   help="record sha256 of the model and adapters")
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("summarize", help="unlabeled cross-condition report")
    s.add_argument("records")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_summarize)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
