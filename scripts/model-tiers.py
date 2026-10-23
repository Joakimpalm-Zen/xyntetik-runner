#!/usr/bin/env python3
"""The model support roster: which files Runner actively validates, and when
each one last passed its gate on each backend.

A tier is a maintenance and validation commitment, not a compatibility list:
the support matrix, `--caps` and `--doctor` say what loads, and the evidence
documents say what was measured. This file says which of those
configurations are re-checked on every release, and it carries the result of
that check per backend so the README can show it without anybody typing it.

Rows are one per model FILE (a sharded set is one file with several parts).
A Tier 1 row's backend cell is one of exactly three things:

  * a recorded gate: written by `--record` from a smoke run's result file,
    naming the Runner version, the day, the device class, the commit, the
    machine's own `--caps` fingerprint and the digest of the output it
    compared;
  * a declared limit: the backend has no path for this file, with the
    support-matrix link that says so;
  * null: nobody ran the gate. Never a pass, and never inferred from a
    neighbouring file or backend.

The same run records the file's `--adapt-info` answer (owner decision 6):
whether the LoRA backward admits the file and whether an adapter can be
served on it, each with the refusal sentence when not.

    scripts/model-tiers.py --check
    scripts/model-tiers.py --stale [--today YYYY-MM-DD]
    scripts/model-tiers.py --record ROW --result smoke-result.json

`--stale` is the release gate's question, asked the way
scripts/device-evidence.py asks it of the device classes: every Tier 1 row's
smoke cell (the backend and device class the row names) must exist, must have
passed, and must be no older than the row's max age (the policy default
unless the row sets a tighter one). A recorded failure on any backend of the
row refuses too. scripts/check-release.py refuses a tag while any row is not
current.
"""

import argparse
import datetime
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROSTER = os.path.join(ROOT, "docs", "model-tiers.json")
DEVICE_EVIDENCE = os.path.join(ROOT, "docs", "device-evidence.json")
SCHEMA = "xyntetik.runner.model-tiers.v1"
RESULT_SCHEMA = "xyntetik.runner.tier-smoke-result.v1"

TIERS = ("tier1", "tier2", "tier3", "upcoming")
BACKENDS = ("cpu", "cuda", "metal")
# Section 4 of the workorder: a feature not listed is not claimed. A feature
# may name its protocol after a colon (tools:qwen38).
FEATURES = ("text", "tools", "structured_output", "reasoning", "vision",
            "adapters", "mtp", "draft", "embeddings", "decisions")
# How the row's numbers relate to the publisher's float32 implementation.
#   measured    a float32 comparison exists on this checkpoint or on the
#               family's smallest member (the anchor text says which);
#   owed        feasible and not run yet;
#   infeasible  no float32 reference fits the hardware this project has;
#   self        this project is the publisher and its BF16 file is the
#               reference.
FP32_REFERENCE = ("measured", "owed", "infeasible", "self")
ANOTHER_ENGINE = "anchor: agreement with another engine, not fidelity"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def load(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: schema is {doc.get('schema')!r}, expected {SCHEMA!r}")
    return doc


def save(doc, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


def device_classes(devices):
    """Device class id -> its `expect` block, from the device ledger."""
    return {c["id"]: c.get("expect") or {} for c in devices.get("classes", [])}


def _date(s):
    try:
        datetime.date.fromisoformat(s)
        return True
    except (TypeError, ValueError):
        return False


def _text(v):
    return isinstance(v, str) and v.strip() != ""


def first_word(text):
    """The leading run of letters: "Experimental: the..." starts with the
    word Experimental, which a whitespace split would read as
    "Experimental:"."""
    m = re.match(r"[A-Za-z]+", text or "")
    return m.group(0) if m else ""


def _repo_path_exists(p):
    if p.startswith(("http://", "https://")):
        return True
    return os.path.exists(os.path.join(ROOT, p.split("#", 1)[0]))


def _cell_errors(where, cell, classes):
    if cell is None:
        return []
    if not isinstance(cell, dict) or cell.get("kind") not in ("gate", "limit"):
        return [f"{where}: a cell is null, a gate or a limit, not {cell!r}"]
    if cell["kind"] == "limit":
        bad = [k for k in ("reason", "matrix") if not _text(cell.get(k))]
        return [f"{where}: a declared limit needs {', '.join(bad)}"] if bad else []
    errs = []
    if not VERSION_RE.match(str(cell.get("runner_version", ""))):
        errs.append(f"{where}: runner_version {cell.get('runner_version')!r}")
    if not _date(cell.get("date")):
        errs.append(f"{where}: date {cell.get('date')!r}")
    if cell.get("device_class") not in classes:
        errs.append(f"{where}: device_class {cell.get('device_class')!r} is not "
                    f"a class in docs/device-evidence.json")
    if cell.get("result") not in ("pass", "fail"):
        errs.append(f"{where}: result {cell.get('result')!r}")
    if cell.get("commit") is not None and not COMMIT_RE.match(str(cell["commit"])):
        errs.append(f"{where}: commit {cell.get('commit')!r}")
    if not SHA256_RE.match(str(cell.get("output_sha256", ""))):
        errs.append(f"{where}: output_sha256")
    if not isinstance(cell.get("caps"), dict):
        errs.append(f"{where}: caps fingerprint missing")
    if not _text(cell.get("ran")):
        errs.append(f"{where}: ran")
    if not _text(cell.get("evidence")) or not _repo_path_exists(cell["evidence"]):
        errs.append(f"{where}: evidence {cell.get('evidence')!r} does not exist")
    return errs


def _lora_errors(where, lora):
    if lora is None:
        return []
    if not isinstance(lora, dict):
        return [f"{where}: lora is null or a recorded --adapt-info answer"]
    errs = []
    for half in ("training", "adapter_serving"):
        h = lora.get(half)
        if not isinstance(h, dict) or not isinstance(h.get("supported"), bool):
            errs.append(f"{where}: lora.{half}.supported must be true or false")
            continue
        if h["supported"] and h.get("reason") is not None:
            errs.append(f"{where}: lora.{half} is supported and carries a reason")
        if not h["supported"] and not _text(h.get("reason")):
            errs.append(f"{where}: lora.{half} is refused without its reason")
    if not VERSION_RE.match(str(lora.get("runner_version", ""))):
        errs.append(f"{where}: lora.runner_version {lora.get('runner_version')!r}")
    if not _date(lora.get("date")):
        errs.append(f"{where}: lora.date {lora.get('date')!r}")
    return errs


def _file_errors(where, files):
    if not isinstance(files, list) or not files:
        return [f"{where}: a file row names its files"]
    errs = []
    for i, f in enumerate(files):
        if not _text(f.get("name")):
            errs.append(f"{where}: files[{i}].name")
        if not SHA256_RE.match(str(f.get("sha256", ""))):
            errs.append(f"{where}: files[{i}].sha256 must be 64 lowercase hex")
        if not isinstance(f.get("bytes"), int) or f["bytes"] <= 0:
            errs.append(f"{where}: files[{i}].bytes")
    return errs


def validate(doc, devices):
    """Every structural error in the roster, as a list of sentences."""
    classes = device_classes(devices)
    errs = []
    policy = doc.get("policy") or {}
    if not isinstance(policy.get("default_max_age_days"), int) or \
            policy["default_max_age_days"] <= 0:
        errs.append("policy.default_max_age_days must be a positive integer")
    if list(policy.get("backends") or []) != list(BACKENDS):
        errs.append(f"policy.backends must be {list(BACKENDS)}")
    for e in doc.get("exempt") or []:
        if not (_text(e.get("repo")) and _text(e.get("reason"))):
            errs.append(f"exempt entry {e!r} needs repo and reason")
    seen = set()
    for row in doc.get("rows") or []:
        rid = row.get("id")
        where = f"row {rid!r}"
        if not isinstance(rid, str) or not ID_RE.match(rid):
            errs.append(f"{where}: id must match {ID_RE.pattern}")
        if rid in seen:
            errs.append(f"{where}: duplicate id")
        seen.add(rid)
        tier = row.get("tier")
        if tier not in TIERS:
            errs.append(f"{where}: tier {tier!r} is not one of {TIERS}")
            continue
        for k in ("model", "claim", "architecture"):
            if not _text(row.get(k)):
                errs.append(f"{where}: {k}")
        claim = row.get("claim") or ""
        # owner decision 4 and section 3: the qualifier is the first thing
        # a reader sees, never a footnote
        if row.get("experimental") and first_word(claim) != "Experimental":
            errs.append(f"{where}: an experimental row's claim starts with 'Experimental'")
        if row.get("vision_upstream") and not claim.startswith("Text path only"):
            errs.append(f"{where}: a vision model's claim starts with 'Text path only'")
        t = row.get("transition") or {}
        if t.get("tier") != tier:
            errs.append(f"{where}: the latest transition leads to {t.get('tier')!r}, "
                        f"not to the row's tier {tier!r}")
        if not _date(t.get("effective")):
            errs.append(f"{where}: transition.effective {t.get('effective')!r}")
        if not _text(t.get("reason")):
            errs.append(f"{where}: transition.reason")
        lvr = t.get("last_validated_release")
        if lvr is not None and not VERSION_RE.match(str(lvr)):
            errs.append(f"{where}: transition.last_validated_release {lvr!r}")
        # owner decision 1: a published file is Tier 1 unless the exception
        # is decided and written down (gpt-oss-20b-keep30, 2026-10-22)
        if row.get("published") and tier != "tier1" and \
                not _text(row.get("published_exception")):
            errs.append(f"{where}: a published file below Tier 1 needs a "
                        f"published_exception")
        if "files" in row:
            errs += _file_errors(where, row["files"])
        errs += _lora_errors(where, row.get("lora"))
        backends = row.get("backends")
        if backends is not None:
            if not isinstance(backends, dict) or sorted(backends) != sorted(BACKENDS):
                errs.append(f"{where}: backends has exactly {list(BACKENDS)}")
                backends = {}
            for b in BACKENDS:
                errs += _cell_errors(f"{where} {b}", backends.get(b), classes)
            if tier == "upcoming" and any(
                    (c or {}).get("kind") == "gate" for c in backends.values()):
                errs.append(f"{where}: an Upcoming row carries no gate; promote it first")
        mad = row.get("max_age_days")
        if mad is not None and (not isinstance(mad, int) or mad <= 0):
            errs.append(f"{where}: max_age_days {mad!r}")
        if tier == "upcoming" and not _text(row.get("missing")):
            errs.append(f"{where}: an Upcoming row says what is missing")
        if tier != "tier1":
            continue
        # Tier 1: the envelope of section 4 and the smoke of section 5
        if "files" not in row:
            errs += _file_errors(where, None)
        if backends is None:
            errs.append(f"{where}: a Tier 1 row has backends cpu, cuda and metal")
            backends = {}
        if "lora" not in row:
            errs.append(f"{where}: a Tier 1 row has a lora field (null until recorded)")
        smoke = row.get("smoke") or {}
        if smoke.get("device_class") not in classes:
            errs.append(f"{where}: smoke.device_class {smoke.get('device_class')!r} "
                        f"is not a class in docs/device-evidence.json")
        if smoke.get("backend") not in BACKENDS:
            errs.append(f"{where}: smoke.backend {smoke.get('backend')!r}")
        elif (backends.get(smoke["backend"]) or {}).get("kind") == "limit":
            errs.append(f"{where}: the smoke backend {smoke['backend']} is a "
                        f"declared limit for this file")
        trunk = row.get("smoke_trunk")
        if trunk is not None and not any(r.get("id") == trunk and r.get("tier") == "tier1"
                                         for r in doc.get("rows") or []):
            errs.append(f"{where}: smoke_trunk {trunk!r} is not a Tier 1 row")
        env = row.get("envelope") or {}
        feats = env.get("features")
        if not isinstance(feats, list) or not feats:
            errs.append(f"{where}: envelope.features lists what is claimed")
        else:
            for ft in feats:
                if str(ft).split(":", 1)[0] not in FEATURES:
                    errs.append(f"{where}: feature {ft!r} is not one of {FEATURES}")
        ref = env.get("fp32_reference")
        if ref not in FP32_REFERENCE:
            errs.append(f"{where}: envelope.fp32_reference {ref!r}")
        if not _text(env.get("anchor")):
            errs.append(f"{where}: envelope.anchor")
        elif ref == "infeasible" and ANOTHER_ENGINE not in env["anchor"]:
            errs.append(f"{where}: no float32 reference is feasible, so the "
                        f"anchor says '{ANOTHER_ENGINE}'")
        ev = env.get("evidence")
        if not isinstance(ev, list) or not ev:
            errs.append(f"{where}: envelope.evidence links the evidence")
        else:
            for p in ev:
                if not _text(p) or not _repo_path_exists(p):
                    errs.append(f"{where}: evidence {p!r} does not exist")
    return errs


def today(override=None):
    if override:
        return datetime.date.fromisoformat(override)
    return datetime.datetime.now(datetime.timezone.utc).date()


def stale(doc, now, out=None):
    """Report every Tier 1 row; return the ids of those that are not current.

    Prints the fresh rows too, as the device ledger does: the number that
    matters is how old the freshest evidence is, and a check that only
    speaks when it is angry trains a reader to take silence for coverage.
    """
    # resolved here, not in the signature, so a replaced stdout (pytest's
    # capture) is the one written to
    out = out or sys.stdout
    default = doc["policy"]["default_max_age_days"]
    rows = [r for r in doc["rows"] if r["tier"] == "tier1"]
    width = max((len(r["id"]) for r in rows), default=0)
    bad = []
    for row in rows:
        limit = row.get("max_age_days", default)
        smoke = row["smoke"]
        where = f"{smoke['device_class']} {smoke['backend']}"
        cells = row.get("backends") or {}
        cell = cells.get(smoke["backend"])
        failing = sorted(b for b, c in cells.items()
                         if (c or {}).get("kind") == "gate" and c.get("result") != "pass")
        if failing:
            state, detail = "FAILING", "recorded failure on " + ", ".join(failing)
        elif (cell or {}).get("kind") != "gate":
            state, detail = "NEVER", f"no smoke recorded on {where}"
        elif cell["device_class"] != smoke["device_class"]:
            state, detail = "NEVER", (f"the {smoke['backend']} cell was recorded on "
                                      f"{cell['device_class']}; the row names {where}")
        else:
            days = (now - datetime.date.fromisoformat(cell["date"])).days
            detail = "%s, %s on %s, %d day%s ago" % (
                where, cell["runner_version"], cell["date"], days,
                "" if days == 1 else "s")
            state = "STALE" if days > limit else "ok"
        if state != "ok":
            bad.append(row["id"])
        print("%-7s %-*s  %s (policy %d days)" % (state, width, row["id"], detail, limit),
              file=out)
    return bad


def record(doc, row_id, res, devices):
    """Write one smoke run's result into its row: the backend cell and the
    file's --adapt-info answer. Refuses anything that would make the roster
    say something no machine said."""
    if res.get("schema") != RESULT_SCHEMA:
        raise SystemExit(f"result schema is {res.get('schema')!r}, expected {RESULT_SCHEMA!r}")
    if res.get("row") != row_id:
        raise SystemExit(f"the result is for row {res.get('row')!r}, not {row_id!r}")
    rows = {r["id"]: r for r in doc["rows"]}
    row = rows.get(row_id)
    if row is None:
        raise SystemExit(f"unknown row {row_id!r}")
    if row["tier"] != "tier1":
        raise SystemExit(f"{row_id} is {row['tier']}: only a Tier 1 row is gated")
    backend = res.get("backend")
    if backend not in BACKENDS:
        raise SystemExit(f"backend {backend!r} is not one of {BACKENDS}")
    if (row["backends"].get(backend) or {}).get("kind") == "limit":
        raise SystemExit(f"{row_id}: {backend} is a declared limit for this file, "
                         f"not a gate to record")
    if res.get("status") not in ("match", "mismatch"):
        # a pinning run wrote the pin and compared nothing
        raise SystemExit(f"status {res.get('status')!r}: only a run compared "
                         f"against an existing pin is a gate")
    want = [f["sha256"] for f in row["files"]]
    got = [f.get("sha256") for f in res.get("files") or []]
    if got != want:
        raise SystemExit(f"{row_id}: the run hashed {got}, the row names sha256 {want}")
    classes = device_classes(devices)
    dc = res.get("device_class")
    if dc not in classes:
        raise SystemExit(f"device class {dc!r} is not in docs/device-evidence.json")
    caps = res.get("caps") or {}
    expect = classes[dc]
    wrong = [f"{k} is {caps.get(k)!r}, expected {expect[k]!r}"
             for k in ("os", "arch") if k in expect and caps.get(k) != expect[k]]
    if backend != "cpu" and caps.get("gpu_backend") != backend:
        wrong.append(f"gpu_backend is {caps.get('gpu_backend')!r}, expected {backend!r}")
    if wrong:
        raise SystemExit(f"refusing to record {row_id} on {dc}: the caps came from "
                         f"the wrong machine ({', '.join(wrong)})")
    # the banner is the backend's own statement that it took the model; a
    # GPU build that fell back prints none, and its output is a CPU result
    banner = res.get("banner")
    if backend == "cpu" and banner:
        raise SystemExit(f"a cpu cell needs a run with no GPU backend banner, got {banner!r}")
    if backend != "cpu" and not banner:
        raise SystemExit(f"a {backend} cell needs the backend banner of a run that offloaded")
    if not VERSION_RE.match(str(caps.get("version", ""))):
        raise SystemExit(f"caps version {caps.get('version')!r}")
    day = str(res.get("generated_utc", ""))[:10]
    if not _date(day):
        raise SystemExit(f"generated_utc {res.get('generated_utc')!r}")
    found = bool((res.get("expect") or {}).get("found"))
    row["backends"][backend] = {
        "kind": "gate",
        "runner_version": caps["version"],
        "date": day,
        "device_class": dc,
        "result": "pass" if res["status"] == "match" and found else "fail",
        "commit": res.get("commit"),
        "caps": caps,
        "output_sha256": (res.get("output") or {}).get("sha256"),
        "ran": res.get("ran"),
        "evidence": (res.get("pin") or {}).get("path"),
    }
    info = res.get("adapt_info")
    if not isinstance(info, dict):
        raise SystemExit("the result carries no --adapt-info answer")
    row["lora"] = {
        "training": {"supported": info["training"]["supported"],
                     "reason": info["training"]["reason"]},
        "adapter_serving": {"supported": info["adapter_serving"]["supported"],
                            "reason": info["adapter_serving"]["reason"]},
        "runner_version": caps["version"],
        "date": day,
    }
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=ROSTER)
    ap.add_argument("--device-evidence", default=DEVICE_EVIDENCE)
    ap.add_argument("--check", action="store_true",
                    help="validate the roster's structure (no clock involved)")
    ap.add_argument("--stale", action="store_true",
                    help="the release gate: refuse while a Tier 1 row is not current")
    ap.add_argument("--today", help="the day ages are measured from (tests); "
                                    "default the UTC clock")
    ap.add_argument("--record", metavar="ROW",
                    help="write a smoke run's result into ROW")
    ap.add_argument("--result", help="the result file scripts/tier-smoke.py wrote")
    args = ap.parse_args()

    doc = load(args.json)
    with open(args.device_evidence, encoding="utf-8") as f:
        devices = json.load(f)
    if args.record:
        if not args.result:
            raise SystemExit("--record needs --result FILE")
        with open(args.result, encoding="utf-8") as f:
            res = json.load(f)
        row = record(doc, args.record, res, devices)
        errs = validate(doc, devices)
        if errs:
            raise SystemExit("refusing to write a roster that does not validate:\n"
                             + "\n".join(errs))
        save(doc, args.json)
        cell = row["backends"][res["backend"]]
        print(f"recorded {args.record} {res['backend']}: {cell['result']} at "
              f"{cell['runner_version']} on {cell['date']} ({cell['device_class']})")
        return 0
    errs = validate(doc, devices)
    for e in errs:
        print(e, file=sys.stderr)
    if errs:
        print(f"\n{args.json}: {len(errs)} error(s)", file=sys.stderr)
        return 1
    if args.stale:
        bad = stale(doc, today(args.today))
        if bad:
            print("\n%d Tier 1 file%s need%s a smoke run: %s" % (
                len(bad), "" if len(bad) == 1 else "s",
                "s" if len(bad) == 1 else "", ", ".join(bad)), file=sys.stderr)
            return 1
        return 0
    rows = doc["rows"]
    counts = {t: sum(1 for r in rows if r["tier"] == t) for t in TIERS}
    print("roster ok: " + ", ".join(f"{counts[t]} {t}" for t in TIERS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
