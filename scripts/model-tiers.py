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
served on it, each with the refusal sentence when not. A row's
`transitions` is its tier history, newest last; the transition log in
docs/model-support.md is rendered from it.

    scripts/model-tiers.py --check
    scripts/model-tiers.py --stale [--today YYYY-MM-DD]
    scripts/model-tiers.py --record ROW --result smoke-result.json
    scripts/model-tiers.py --write-docs     # re-render README.md's table
    scripts/model-tiers.py --docs-check     # fail when it differs from the render

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
# What the file is, and how a companion is loaded beside the row it runs
# with: an MTP head as `-m TRUNK --mtp-file HEAD`, an adapter as
# `-m BASE --lora ADAPTER`. A model file runs on its own.
FILE_KINDS = {"model": None, "mtp-head": "mtp-file", "adapter": "lora"}
ANOTHER_ENGINE = "anchor: agreement with another engine, not fidelity"
# where every declared limit points: the manual's support matrix, the one
# place a reader checks what a backend lacks; other evidence for a limit
# belongs in the row's envelope
MATRIX_LINK = "MANUAL.md#support-matrix"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

README = os.path.join(ROOT, "README.md")
README_BEGIN = ("<!-- model-tiers:begin (rendered from docs/model-tiers.json by "
                "scripts/model-tiers.py --write-docs; do not edit) -->")
README_END = "<!-- model-tiers:end -->"
SUPPORT_DOC = "docs/model-support.md"
DOC = os.path.join(ROOT, "docs", "model-support.md")
DOC_BEGIN = ("<!-- model-tiers:rows:begin (rendered from docs/model-tiers.json by "
             "scripts/model-tiers.py --write-docs; do not edit) -->")
DOC_END = "<!-- model-tiers:rows:end -->"
TIER_NAMES = {"tier1": "Tier 1, actively validated", "tier2": "Tier 2, maintained on report",
              "tier3": "Tier 3, unmaintained", "upcoming": "Upcoming, planned admission"}
FP32_WORDS = {
    "measured": "a publisher float32 comparison exists",
    "owed": "a publisher float32 comparison is feasible and owed",
    "infeasible": "no publisher float32 reference is feasible",
    "self": "this project is the publisher and its BF16 file is the reference",
}


def load(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: schema is {doc.get('schema')!r}, expected {SCHEMA!r}")
    return doc


def save(doc, path):
    # LF on every host: a text-mode write on Windows would make it CRLF
    with open(path, "w", encoding="utf-8", newline="\n") as f:
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
        if bad:
            return [f"{where}: a declared limit needs {', '.join(bad)}"]
        if cell["matrix"] != MATRIX_LINK:
            return [f"{where}: a declared limit's matrix is {MATRIX_LINK!r}, not "
                    f"{cell['matrix']!r}; other evidence goes in the envelope"]
        return []
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
    by_id = {r.get("id"): r for r in doc.get("rows") or []}
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
        # the row's history of tiers, newest last: the transition log is
        # rendered from it, so a move appends and the earlier entries stay
        ts = row.get("transitions")
        if not isinstance(ts, list) or not ts:
            errs.append(f"{where}: transitions lists the row's tier history, newest last")
            ts = []
        for i, t in enumerate(ts):
            if t.get("tier") not in TIERS:
                errs.append(f"{where}: transitions[{i}].tier {t.get('tier')!r}")
            if not _date(t.get("effective")):
                errs.append(f"{where}: transitions[{i}].effective {t.get('effective')!r}")
            elif i and _date(ts[i - 1].get("effective")) and \
                    t["effective"] < ts[i - 1]["effective"]:
                errs.append(f"{where}: transitions are in date order, newest last")
            if not _text(t.get("reason")):
                errs.append(f"{where}: transitions[{i}].reason")
            lvr = t.get("last_validated_release")
            if lvr is not None and not VERSION_RE.match(str(lvr)):
                errs.append(f"{where}: transitions[{i}].last_validated_release {lvr!r}")
        if ts and ts[-1].get("tier") != tier:
            errs.append(f"{where}: the latest transition leads to {ts[-1].get('tier')!r}, "
                        f"not to the row's tier {tier!r}")
        # owner decision 1: a published file is Tier 1 unless the exception
        # is decided and written down (gpt-oss-20b-keep30, 2026-10-22)
        if row.get("published") and tier != "tier1" and \
                not _text(row.get("published_exception")):
            errs.append(f"{where}: a published file below Tier 1 needs a "
                        f"published_exception")
        if "files" in row:
            errs += _file_errors(where, row["files"])
        kind = row.get("file_kind", "model")
        rw = row.get("runs_with")
        if kind not in FILE_KINDS:
            errs.append(f"{where}: file_kind {kind!r} is not one of {list(FILE_KINDS)}")
        elif kind == "model" and rw is not None:
            errs.append(f"{where}: a model file runs on its own and names no runs_with")
        elif kind != "model" and rw is None and tier == "tier1":
            errs.append(f"{where}: a {kind} file does not run on its own, and without "
                        f"runs_with the per-release smoke cannot run it")
        elif kind != "model" and rw is not None:
            target = by_id.get((rw or {}).get("row"))
            if (rw or {}).get("as") != FILE_KINDS[kind]:
                errs.append(f"{where}: runs_with.as is {FILE_KINDS[kind]!r} for a {kind}")
            if target is None or not target.get("files"):
                errs.append(f"{where}: runs_with names {(rw or {}).get('row')!r}, not a "
                            f"row with files")
            elif target.get("file_kind", "model") != "model":
                errs.append(f"{where}: runs_with names another companion; a companion "
                            f"runs with a model file")
            elif tier == "tier1":
                b = (row.get("smoke") or {}).get("backend")
                if ((target.get("backends") or {}).get(b) or {}).get("kind") == "limit":
                    errs.append(f"{where}: runs_with {target['id']} has a declared limit "
                                f"on {b}, the backend this row's smoke runs on")
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
    rw = row.get("runs_with")
    if rw:
        # a companion's run is the named row's file plus this one: both must
        # be the bytes their rows name
        base = rows.get(rw["row"]) or {}
        w = res.get("with") or {}
        want_w = [f["sha256"] for f in base.get("files") or []]
        got_w = [f.get("sha256") for f in w.get("files") or []]
        if w.get("row") != rw["row"] or got_w != want_w:
            raise SystemExit(f"{row_id} runs with {rw['row']}: the run used "
                             f"{w.get('row')!r} with sha256 {got_w}, the roster names "
                             f"{want_w}")
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


# --- the README table ---------------------------------------------------------

def _mark(cell):
    """A backend cell's mark: exactly one of a recorded gate with its version
    and date, a declared limit linked to its matrix row, or not run."""
    if cell is None:
        return "☐ not run"
    if cell["kind"] == "limit":
        return f"[n/a]({cell['matrix']})"
    if cell["result"] == "pass":
        return f"✅ v{cell['runner_version']} {cell['date']}"
    # a recorded failure refuses the next tag; until then it is shown as one
    return f"failed v{cell['runner_version']} {cell['date']}"


def _train(lora):
    if lora is None:
        return "☐ not run"
    t = lora["training"]
    if t["supported"]:
        return f"✅ v{lora['runner_version']} {lora['date']}"
    return "no: " + t["reason"].replace("|", "\\|")


def _qualifiers(rows):
    pre = ["Experimental"] if any(r.get("experimental") for r in rows) else []
    return pre + (["Text path only"] if any(r.get("vision_upstream") for r in rows) else [])


def _qualified(rows, name):
    """The qualifier a model leads with, then its name."""
    return ": ".join(_qualifiers(rows) + [name])


def _readme_label(rows, name):
    """The table's model cell: the qualifier as plain text, then the name in
    bold ("Text path only: **Gemma 4 12B**")."""
    return "".join(q + ": " for q in _qualifiers(rows)) + f"**{name}**"


def _model_groups(rows):
    """Rows grouped by model, groups sorted by model name, files within a
    group in roster order: the render does not depend on insertion order."""
    groups = {}
    for r in rows:
        groups.setdefault(r["model"], []).append(r)
    return sorted(groups.items(), key=lambda kv: (kv[0].casefold(), kv[0]))


def render_readme(doc):
    """The block between README_BEGIN and README_END.

    One row per Tier 1 model, its qualifier before its bold name, then one
    sub-row per roster file of that model, run or not: every roster file is
    one the release gate holds Runner to, and a quant the matrix admits
    outside the roster is the matrix's claim, not this table's. Upcoming
    models follow as names, in the roster's order (the order of admission).
    """
    groups = _model_groups([r for r in doc["rows"] if r["tier"] == "tier1"])
    # a blank line on each side keeps the markers' HTML comments apart from
    # the table and the paragraph in every markdown renderer
    lines = ["", "", "| Model | File | CPU | CUDA | Metal | Train | Evidence |",
             "|---|---|---|---|---|---|---|"]
    for model, rows in groups:
        lines.append(f"| {_readme_label(rows, model)} | | | | | | |")
        for r in rows:
            files = r["files"]
            name = f"`{files[0]['name']}`" + (f" ({len(files)} parts)"
                                             if len(files) > 1 else "")
            cells = [name] + [_mark(r["backends"].get(b)) for b in BACKENDS]
            cells += [_train(r.get("lora")), f"[envelope]({SUPPORT_DOC}#{r['id']})"]
            lines.append("| | " + " | ".join(cells) + " |")
    upcoming = []
    for r in doc["rows"]:
        if r["tier"] == "upcoming":
            n = r["model"] + (" (text path only)" if r.get("vision_upstream") else "")
            if n not in upcoming:
                upcoming.append(n)
    lines += ["", "**Upcoming**, not counted as validated: " + ", ".join(upcoming) + ".",
              "", ""]
    return "\n".join(lines)


# --- docs/model-support.md ------------------------------------------------------

def _from_docs(path):
    """A repository path as a link from docs/model-support.md."""
    if path.startswith(("http://", "https://")):
        return path
    return path[len("docs/"):] if path.startswith("docs/") else "../" + path


def _cell_detail(cell):
    if cell is None:
        return "☐ not run"
    if cell["kind"] == "limit":
        return f"[n/a]({_from_docs(cell['matrix'])}): {cell['reason']}"
    pin = cell["evidence"]
    return (f"{_mark(cell)} on {cell['device_class']} at commit "
            f"`{(cell.get('commit') or 'unknown')[:12]}`, output sha256 "
            f"`{cell['output_sha256'][:16]}`, [pin]({_from_docs(pin)})")


def _lora_detail(lora):
    if lora is None:
        return ["- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer)."]
    out = []
    for half, label in (("training", "training"), ("adapter_serving", "adapter serving")):
        h = lora[half]
        mark = "supported" if h["supported"] else "no: " + h["reason"]
        out.append(f"- **LoRA {label}:** {mark} (runner {lora['runner_version']}, "
                   f"{lora['date']}).")
    return out


def _files(row):
    out = []
    for f in row.get("files") or []:
        where = ""
        if f.get("repo"):
            where = f", from [{f['repo']}](https://huggingface.co/{f['repo']})"
            if f.get("revision"):
                where += f" at revision `{f['revision'][:12]}`"
        size = f", {f['bytes']:,} bytes" if f.get("bytes") else ""
        out.append(f"`{f['name']}` (sha256 `{f['sha256']}`{size}{where})")
    return out


def _row_doc(row, by_id):
    files = row.get("files") or []
    title = f"`{files[0]['name']}`" + (f" ({len(files)} parts)" if len(files) > 1 else "") \
        if files else row["model"]
    lines = [f'<a id="{row["id"]}"></a>', f"#### {title}", "", row["claim"], ""]
    if files:
        lines.append("- **File:** " + "; ".join(_files(row)) + ".")
    up = row.get("upstream")
    if up:
        rev = f" at revision `{up['revision'][:12]}`" if up.get("revision") else ""
        lines.append(f"- **Upstream checkpoint:** {up['checkpoint']}{rev}.")
    if row.get("quantization"):
        lines.append(f"- **Quantization:** {row['quantization']}.")
    rw = row.get("runs_with")
    if rw:
        flag = {"mtp-file": "--mtp-file", "lora": "--lora"}[rw["as"]]
        base = by_id.get(rw["row"], {})
        lines.append(f"- **Runs with:** [{base.get('model', rw['row'])}](#{rw['row']}), "
                     f"loaded beside it with `{flag}`; the smoke and its pin are of "
                     f"that composed run.")
    if row.get("published_exception"):
        lines.append(f"- **Published below Tier 1:** {row['published_exception']}")
    env = row.get("envelope")
    if env:
        feats = ", ".join(f.replace(":", " (") + (")" if ":" in f else "")
                          for f in env["features"])
        lines.append(f"- **Features claimed:** {feats}. A feature not listed is not claimed.")
        lines.append(f"- **Numerical anchor ({FP32_WORDS[env['fp32_reference']]}):** "
                     f"{env['anchor']}")
        lines.append(f"- **Context tested:** {env.get('context') or 'not recorded'}.")
        if env.get("limitations"):
            lines.append(f"- **Known limitations:** {env['limitations']}")
        lines.append(f"- **Still missing:** {env.get('missing') or 'nothing the audit named'}")
        ev = ", ".join(f"[{e}]({_from_docs(e)})" for e in env["evidence"])
        lines.append(f"- **Evidence:** {ev}.")
    elif row.get("anchor"):
        lines.append(f"- **Numerical anchor:** {row['anchor']}")
    if row.get("missing"):
        lines.append(f"- **Missing for Tier 1:** {row['missing']}")
    if row.get("smoke"):
        sm = row["smoke"]
        age = (f"an age limit of {row['max_age_days']} days" if row.get("max_age_days")
               else "the policy's age limit")
        lines.append(f"- **Per-release smoke:** on {sm['device_class']}, {sm['backend']} "
                     f"backend, {age}.")
    if row.get("backends"):
        for b in BACKENDS:
            lines.append(f"- **{b.upper() if b != 'metal' else 'Metal'}:** "
                         f"{_cell_detail(row['backends'].get(b))}")
    if row["tier"] == "tier1" or row.get("lora") is not None:
        lines += _lora_detail(row.get("lora"))
    hist = "; ".join(f"{TIER_NAMES[t['tier']]} from {t['effective']}" for t in row["transitions"])
    lines += [f"- **Tier history:** {hist} ([log](#transition-log)).", ""]
    return lines


def render_doc(doc):
    """The block between DOC_BEGIN and DOC_END: every row's envelope by
    tier, the exempt repositories, and the transition log."""
    by_id = {r["id"]: r for r in doc["rows"]}
    lines = ["", ""]
    for tier in TIERS:
        rows = [r for r in doc["rows"] if r["tier"] == tier]
        lines += [f'<a id="{tier}"></a>', f"## {TIER_NAMES[tier]}", ""]
        if not rows:
            lines += ["No members.", ""]
            continue
        if tier == "upcoming":
            # the roster's order is the order of admission, which is the point
            groups = {}
            for r in rows:
                groups.setdefault(r["model"], []).append(r)
            groups = list(groups.items())
        else:
            groups = _model_groups(rows)
        for model, rs in groups:
            lines += [f"### {_qualified(rs, model)}", ""]
            for r in rs:
                lines += _row_doc(r, by_id)
    lines += ['<a id="exempt"></a>', "## Published, not a Runner model file", ""]
    for e in doc.get("exempt") or []:
        path = f" (`{e['path']}`)" if e.get("path") else ""
        lines.append(f"- [{e['repo']}](https://huggingface.co/{e['repo']}){path}: {e['reason']}")
    lines += ["", '<a id="transition-log"></a>', "## Transition log", "",
              "Every tier a row has held, oldest first. Last validated release is "
              "the Runner version of the newest passing gate the roster had recorded on "
              "the row when it moved; none recorded means the roster held no gate for it "
              "then, which is every row at the 2026-10-22 migration.", "",
              "| Effective | Row | Tier | Reason | Last validated release |",
              "|---|---|---|---|---|"]
    entries = []
    for i, r in enumerate(doc["rows"]):
        name = (r.get("files") or [{"name": r["model"]}])[0]["name"]
        for j, t in enumerate(r["transitions"]):
            entries.append((t["effective"], i, j, f"[{r['model']}, `{name}`](#{r['id']})"
                            if r.get("files") else f"[{r['model']}](#{r['id']})", t))
    for eff, _, _, label, t in sorted(entries, key=lambda e: e[:3]):
        lvr = t.get("last_validated_release") or "none recorded"
        reason = t["reason"].replace("|", "\\|")
        lines.append(f"| {eff} | {label} | {TIER_NAMES[t['tier']]} | {reason} | {lvr} |")
    lines += ["", ""]
    return "\n".join(lines)


def _blocks(doc):
    """(path, begin, end, rendered text) for every rendered block."""
    return [(README, README_BEGIN, README_END, render_readme(doc)),
            (DOC, DOC_BEGIN, DOC_END, render_doc(doc))]


def _replace_block(text, begin, end, body, path):
    if text.count(begin) != 1 or text.count(end) != 1:
        raise SystemExit(f"{path}: expected one {begin!r} and one {end!r}")
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    return head + begin + body + end + tail


def docs_drift(doc):
    """The files whose rendered block differs from the roster's render."""
    bad = []
    for path, begin, end, body in _blocks(doc):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if _replace_block(text, begin, end, body, path) != text:
            bad.append(os.path.relpath(path, ROOT))
    return bad


def write_docs(doc):
    """Re-render every block from the roster (after a --record or --retire)."""
    for path, begin, end, body in _blocks(doc):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        new = _replace_block(text, begin, end, body, path)
        if new != text:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(new)


def is_committed_roster(path):
    """Only the repository's own roster drives the repository's docs."""
    return os.path.abspath(path) == os.path.abspath(ROSTER)


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
    ap.add_argument("--write-docs", action="store_true",
                    help="re-render the README table from the roster")
    ap.add_argument("--docs-check", action="store_true",
                    help="fail when the README table differs from the render")
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
        if is_committed_roster(args.json):
            write_docs(doc)
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
    if args.write_docs or args.docs_check:
        errs = validate(doc, devices)
        if errs:
            raise SystemExit("the roster does not validate:\n" + "\n".join(errs))
        if args.write_docs:
            write_docs(doc)
            return 0
        bad = docs_drift(doc)
        for b in bad:
            print(f"{b}: the rendered block differs from docs/model-tiers.json; "
                  f"run scripts/model-tiers.py --write-docs", file=sys.stderr)
        return 1 if bad else 0
    if args.stale:
        bad = stale(doc, today(args.today))
        if bad:
            # the rows went to stdout, which a pipe buffers while stderr is
            # written at once: flush, or a reader of both in one pipe gets
            # the summary before the rows it summarises
            sys.stdout.flush()
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
