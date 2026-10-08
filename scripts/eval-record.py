#!/usr/bin/env python3
"""Evaluation as a recorded step (R17.2): the result travels with the file.

An evaluation writes `<model>.eval.<kind>.json` beside the model it judged:
the model and every other input by sha256 (linking the lineage record that
made each one, as the rewrite, merge and training records do), the method
and its settings, the measured numbers, the thresholds the caller set and
whether they passed. `--sign-key` signs it in place (`runner --sign-record`).
`runner --lineage` shows a file's evaluations beside its chain, and
`runner --require-eval KIND` refuses to load a model without a passing one.

Two kinds:

  fidelity  the model against a reference over a text corpus, through
            scripts/kld-compare-raw.py: mean KL, top-1 agreement (plain and
            margin-qualified), top-8 overlap, positions.

      eval-record.py fidelity --model q.gguf --reference ref.gguf \\
          --corpus text.txt [--max-positions 200] [--max-mean-kld 0.05] \\
          [--min-top1 95] [--sign-key key.json] [--runner ./runner]

  agent     an agent-bank run (xyntetik_runner.shadow replay evidence) on the
            model: tasks attempted and verified, engine-error aborts, walls,
            tool calls, and turns that carried more than one call when the
            attempt transcripts are given.

      eval-record.py agent --model q.gguf --evidence evidence.jsonl \\
          [--attempts DIR] [--min-verified 4] [--sign-key key.json]

Exit 0 when the record is written and every threshold set passed, 1 when a
threshold failed (the record is still written, with "pass": false), 2 on an
error (no record written).
"""
import argparse
import ast
import glob
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile

SCHEMA = "xyntetik.runner.eval.v1"
ROOT = pathlib.Path(__file__).resolve().parents[1]
SIDECARS = (".quant.json", ".merge.json", ".train.json", ".context.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar(path):
    """The lineage record a step wrote beside PATH, newest first (as the runner looks)."""
    found = [path + s for s in SIDECARS if os.path.exists(path + s)]
    return max(found, key=os.path.getmtime) if found else None


def file_entry(path):
    e = {"path": str(path), "sha256": sha256_file(path)}
    rec = sidecar(str(path))
    if rec:
        e["record"] = {"path": rec, "sha256": sha256_file(rec)}
    return e


def runner_version(runner):
    try:
        p = subprocess.run([runner, "--version"], capture_output=True, text=True, timeout=30)
        return p.stdout.strip().split()[-1] if p.returncode == 0 and p.stdout.strip() else None
    except OSError:
        return None


def judge(checks):
    """checks: list of (name, ok or None). pass is None when no threshold was set."""
    set_ = [(n, ok) for n, ok in checks if ok is not None]
    return (all(ok for _, ok in set_) if set_ else None), {n: ok for n, ok in set_}


def write_record(model, kind, rec, sign_key, runner):
    out = str(model) + f".eval.{kind}.json"
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    os.replace(tmp, out)
    if sign_key:
        p = subprocess.run([runner, "--sign-record", out, "--sign-key", sign_key],
                           capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            print(f"error: signing {out} failed: {p.stderr.strip()}", file=sys.stderr)
            return None
    return out


def fidelity(args):
    with tempfile.TemporaryDirectory() as td:
        res_path = os.path.join(td, "kld.json")
        cmd = [sys.executable, str(ROOT / "scripts" / "kld-compare-raw.py"),
               "--model-a", args.model, "--model-b", args.reference,
               "--runner", args.runner, "--corpus", args.corpus,
               "--max-positions", str(args.max_positions), "--out", res_path]
        if args.threads:
            cmd += ["--threads", str(args.threads)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(res_path):
            print(f"error: kld-compare-raw.py produced no result:\n{p.stderr[-2000:]}",
                  file=sys.stderr)
            return 2
        kld = json.load(open(res_path))
    if kld.get("mean_kld") is None:
        print("error: no positions scored", file=sys.stderr)
        return 2
    metrics = {k: kld.get(k) for k in (
        "positions_scored", "positions_failed", "mean_kld", "top1_agreement_pct",
        "top1_margin_qualified_pct", "mean_top8_overlap", "tie_band_nats")}
    ok, detail = judge([
        ("max_mean_kld", None if args.max_mean_kld is None
         else metrics["mean_kld"] <= args.max_mean_kld),
        ("min_top1_pct", None if args.min_top1 is None
         else metrics["top1_agreement_pct"] >= args.min_top1),
    ])
    rec = {
        "schema_version": SCHEMA, "kind": "fidelity",
        "runner": runner_version(args.runner),
        "subject": file_entry(args.model),
        "reference": file_entry(args.reference),
        "corpus": file_entry(args.corpus),
        "method": {"tool": "scripts/kld-compare-raw.py",
                   "tool_schema": kld.get("schema_version"),
                   "max_positions": args.max_positions},
        "metrics": metrics,
        "thresholds": {"max_mean_kld": args.max_mean_kld, "min_top1_pct": args.min_top1},
        "checks": detail, "pass": ok,
    }
    out = write_record(args.model, "fidelity", rec, args.sign_key, args.runner)
    if not out:
        return 2
    print(json.dumps({"record": out, "metrics": metrics, "pass": ok}, indent=1))
    return 1 if ok is False else 0


def count_multi_call_turns(attempts_dir):
    turns = multi = 0
    for f in glob.glob(os.path.join(attempts_dir, "*", "*.transcript.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("role") != "assistant":
                continue
            tc = r.get("tool_calls") or []
            if isinstance(tc, str):
                try:
                    tc = ast.literal_eval(tc)
                except (ValueError, SyntaxError):
                    tc = []
            turns += 1
            multi += len(tc) > 1
    return turns, multi


def agent(args):
    try:
        ev = [json.loads(l) for l in open(args.evidence, encoding="utf-8") if l.strip()]
    except (OSError, ValueError) as e:
        print(f"error: cannot read {args.evidence}: {e}", file=sys.stderr)
        return 2
    if not ev:
        print("error: the evidence file is empty", file=sys.stderr)
        return 2
    walls = [r.get("wall_s", 0) for r in ev]
    res = [r.get("resources", {}) for r in ev]
    metrics = {
        "tasks_attempted": len(ev),
        "tasks_verified": sum(1 for r in ev if str(r.get("disposition", "")).startswith("verified")),
        "engine_error_aborts": sum(1 for r in ev if "model error" in ((r.get("reasons") or [""])[0])),
        "wall_s_total": round(sum(walls), 1),
        "wall_s_mean": round(sum(walls) / len(walls), 1),
        "turns": int(sum(x.get("turns", 0) for x in res)),
        "tool_calls": int(sum(x.get("tool_calls", 0) for x in res)),
    }
    if args.attempts:
        t, m = count_multi_call_turns(args.attempts)
        metrics["multi_call_turns"] = m
        metrics["transcript_turns"] = t
    ok, detail = judge([
        ("min_verified", None if args.min_verified is None
         else metrics["tasks_verified"] >= args.min_verified),
    ])
    rec = {
        "schema_version": SCHEMA, "kind": "agent",
        "runner": runner_version(args.runner),
        "subject": file_entry(args.model),
        "evidence": file_entry(args.evidence),
        "method": {"tool": "xyntetik_runner.shadow replay",
                   "harness": (ev[0].get("identity") or {}).get("harness_version")},
        "metrics": metrics,
        "thresholds": {"min_verified": args.min_verified},
        "checks": detail, "pass": ok,
    }
    out = write_record(args.model, "agent", rec, args.sign_key, args.runner)
    if not out:
        return 2
    print(json.dumps({"record": out, "metrics": metrics, "pass": ok}, indent=1))
    return 1 if ok is False else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)
    exe = "runner.exe" if sys.platform == "win32" else "runner"
    for name in ("fidelity", "agent"):
        sp = sub.add_parser(name)
        sp.add_argument("--model", required=True, help="the file under evaluation")
        sp.add_argument("--runner", default=str(ROOT / exe))
        sp.add_argument("--sign-key", default=None, help="sign the record in place")
        if name == "fidelity":
            sp.add_argument("--reference", required=True)
            sp.add_argument("--corpus", required=True)
            sp.add_argument("--max-positions", type=int, default=200)
            sp.add_argument("--threads", type=int, default=0)
            sp.add_argument("--max-mean-kld", type=float, default=None)
            sp.add_argument("--min-top1", type=float, default=None,
                            help="minimum top-1 agreement, percent")
        else:
            sp.add_argument("--evidence", required=True, help="shadow replay evidence.jsonl")
            sp.add_argument("--attempts", default=None,
                            help="the replay's attempts/ directory (multi-call turns)")
            sp.add_argument("--min-verified", type=int, default=None)
    args = ap.parse_args(argv)
    return fidelity(args) if args.kind == "fidelity" else agent(args)


if __name__ == "__main__":
    sys.exit(main())
