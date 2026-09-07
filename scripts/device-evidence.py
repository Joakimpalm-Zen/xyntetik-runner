#!/usr/bin/env python3
"""The hardware this project claims, and when each piece was last verified on it.

CI proves what CI can reach: three platforms build, and the suite runs on
GPU-less hosted machines. It cannot prove anything about the devices the
README's platform table promises, because those devices are desks in a lab.
Nothing watches that gap, so it grows silently: between 2026-08-13 and
2026-09-07 the Windows suite accumulated eight failures and one error, and the
only reason anybody found out was that somebody needed the box for something
else.

The failure mode is not "a test broke". It is "nothing could see that nothing
had been checked". So this file gates on the thing that actually decays,
which is STALENESS PER DEVICE CLASS, and `make release-check` refuses a tag
when a class the README claims has gone past its policy.

Two rules keep it from becoming a spreadsheet nobody trusts:

  * a class only exists here if CI CANNOT verify it. Anything a hosted runner
    proves every push belongs in CI, not in a ledger;
  * a row is written by `--record`, from the runner's own `--caps` output, so
    "verified on an RTX 3070" is a machine's statement about a machine and not
    a line somebody typed.

    scripts/device-evidence.py --check
    scripts/device-evidence.py --record windows-x86_64-cuda \
        --ran "make OS=Windows_NT -j2 test" --result pass --caps caps.json
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "docs", "device-evidence.json")
SCHEMA = "xyntetik.runner.device-evidence.v1"


def load(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: schema is {doc.get('schema')!r}, expected {SCHEMA!r}")
    return doc


def today(override=None):
    if override:
        return datetime.date.fromisoformat(override)
    return datetime.datetime.now(datetime.timezone.utc).date()


def age_days(row, now):
    return (now - datetime.date.fromisoformat(row["date"])).days


def check(doc, now, out=None):
    """Report every class; return the list of stale ones.

    Prints all rows rather than only the failures, because the number that
    matters is how old the FRESHEST evidence is, and a check that only speaks
    up when it is angry trains people to read silence as coverage.
    """
    # resolved here, not in the signature: a default bound at import time
    # holds the ORIGINAL stdout, so anything that replaces it later (pytest's
    # capture, a caller redirecting) would find this function writing past it
    out = out or sys.stdout
    stale = []
    width = max(len(c["id"]) for c in doc["classes"])
    for cls in doc["classes"]:
        last = cls.get("last_verified")
        limit = cls.get("max_age_days", doc["policy"]["default_max_age_days"])
        if not last:
            state, detail = "NEVER", "no evidence recorded"
            stale.append(cls["id"])
        else:
            days = age_days(last, now)
            detail = "%s, %d day%s ago, %s" % (
                last["date"], days, "" if days == 1 else "s", last["result"])
            if last["result"] != "pass":
                state = "FAILING"
                stale.append(cls["id"])
            elif days > limit:
                state = "STALE"
                stale.append(cls["id"])
            else:
                state = "ok"
        print("%-6s %-*s  %s (policy %d days)" % (state, width, cls["id"],
                                                  detail, limit), file=out)
    return stale


def caps_fingerprint(caps):
    """The part of --caps that identifies the machine a run happened on.

    Deliberately not the whole blob: RAM figures and the sampling presets move
    for reasons that have nothing to do with which device this is, and a
    fingerprint that changes every run cannot be compared.
    """
    gpu = caps.get("gpu") or {}
    return {
        "version": caps.get("version"),
        "os": caps.get("os"),
        "arch": caps.get("arch"),
        "cpu_cores": caps.get("cpu_cores"),
        "gpu_backend": gpu.get("backend"),
        "gpu_name": gpu.get("name"),
    }


def record(doc, cls_id, ran, result, caps_path, evidence, now):
    for cls in doc["classes"]:
        if cls["id"] == cls_id:
            break
    else:
        raise SystemExit("unknown device class %r; known: %s" % (
            cls_id, ", ".join(c["id"] for c in doc["classes"])))
    with open(caps_path, encoding="utf-8") as f:
        caps = json.load(f)
    fp = caps_fingerprint(caps)
    want = cls.get("expect")
    if want:
        wrong = {k: (fp.get(k), v) for k, v in want.items() if fp.get(k) != v}
        if wrong:
            raise SystemExit(
                "refusing to record %s: --caps came from the wrong machine (%s)"
                % (cls_id, ", ".join("%s is %r, expected %r" % (k, got, exp)
                                     for k, (got, exp) in wrong.items())))
    # The commit is provenance, not the point: the row's claim is that THIS
    # machine ran THAT command and passed, which the caps fingerprint and the
    # date carry. A recording made where git is not on PATH (an msys2 shell
    # without it, a release tarball) records a null commit and says so, rather
    # than refusing to record anything at all.
    commit = None
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           text=True)
        if r.returncode == 0:
            commit = r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    cls["last_verified"] = {
        "date": now.isoformat(),
        "commit": commit,
        "ran": ran,
        "result": result,
        "caps": fp,
    }
    if evidence:
        cls["last_verified"]["evidence"] = evidence
    return cls


def backfill(doc, cls_id, report_path, now):
    """Seed a row from an already-committed machine-written report.

    The ledger starts today; the hardware did not. Rather than hand-type a
    first row, or start with a class marked NEVER that would block the next
    tag for a reason nobody chose, a class can be seeded from a report the
    box itself wrote. The row records WHICH file it came from, so a reader
    can tell a seeded row from one `--record` wrote and can go and look.
    """
    for cls in doc["classes"]:
        if cls["id"] == cls_id:
            break
    else:
        raise SystemExit("unknown device class %r" % cls_id)
    with open(os.path.join(ROOT, report_path), encoding="utf-8") as f:
        rep = json.load(f)
    if rep.get("schema_version") != "xyntetik.runner.model-compat-report.v1":
        raise SystemExit("%s is not a model-compat report" % report_path)
    host = rep.get("host") or {}
    generated = rep.get("generated_utc", "")
    cls["last_verified"] = {
        "date": generated.split("T")[0],
        "commit": None,
        "ran": "scripts/compat_matrix.py (see the report)",
        "result": "pass",
        "caps": {
            "version": (rep.get("runner") or {}).get("version"),
            "os": (host.get("os") or "").lower() or None,
            "arch": host.get("machine"),
            "gpu_backend": cls.get("expect", {}).get("gpu_backend"),
            "gpu_name": None,
        },
        "evidence": report_path,
        "source": "backfilled from a committed report, not written by --record",
    }
    return cls


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=LEDGER)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--record", metavar="ID")
    ap.add_argument("--backfill", metavar="ID")
    ap.add_argument("--from", dest="from_report", metavar="PATH")
    ap.add_argument("--ran")
    ap.add_argument("--result", choices=("pass", "fail"))
    ap.add_argument("--caps", help="a file holding `runner --caps` output")
    ap.add_argument("--evidence", help="repo-relative path to the committed report")
    ap.add_argument("--today", help="override the date (tests)")
    args = ap.parse_args()

    doc = load(args.json)
    now = today(args.today)

    if args.backfill:
        if not args.from_report:
            raise SystemExit("--backfill needs --from PATH")
        backfill(doc, args.backfill, args.from_report, now)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        print("backfilled %s from %s" % (args.backfill, args.from_report))
        return 0

    if args.record:
        if not (args.ran and args.result and args.caps):
            raise SystemExit("--record needs --ran, --result and --caps")
        record(doc, args.record, args.ran, args.result, args.caps,
               args.evidence, now)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        print("recorded %s at %s" % (args.record, now.isoformat()))
        return 0

    stale = check(doc, now)
    if stale:
        print("\n%d device class%s need%s a run: %s" % (
            len(stale), "" if len(stale) == 1 else "es",
            "s" if len(stale) == 1 else "", ", ".join(stale)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
