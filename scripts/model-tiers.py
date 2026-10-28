#!/usr/bin/env python3
"""The model support roster: which files Runner validates, and when each one
last passed a validation run on each backend.

A tier is a maintenance and validation commitment, not a compatibility list:
the support matrix, `--info caps` and `--info doctor` say what loads, and the evidence
documents say what was measured. This file says which of those
configurations the project validates, and it carries the Runner version and
date of each file's last validation run per backend so docs/model-support.md
can show the paper trail without anybody typing it. A record is not re-run on
a schedule or by age (policy decision 2026-10-28): validation runs happen in
the bigger validation runs the owner starts and on immediate need, and a new
admission records the pass that admitted it.

Rows are one per model FILE (a sharded set is one file with several parts).
Every Tier 1 and Tier 2 row names its `family`, the display name
docs/model-support.md lists its model under beside its sibling models.
A Tier 1 row's backend cell is one of exactly three things:

  * a recorded gate: written by `--record` from a smoke run's result file,
    naming the Runner version, the day, the device class, the commit, the
    machine's own `--info caps` fingerprint and the digest of the output it
    compared;
  * a declared limit: the backend has no path for this file, with the
    support-matrix link that says so;
  * null: nobody ran the gate. Never a pass, and never inferred from a
    neighbouring file or backend.

The same run records the file's `--info adapter` answer (policy decision 6):
whether the LoRA backward admits the file and whether an adapter can be
served on it, each with the refusal sentence when not. A row's
`transitions` is its tier history, newest last; the transition log in
docs/model-support.md is rendered from it.

    scripts/model-tiers.py --check
    scripts/model-tiers.py --gate [--today YYYY-MM-DD]
    scripts/model-tiers.py --record ROW --result smoke-result.json
    scripts/model-tiers.py --write-docs     # re-render docs/model-support.md
    scripts/model-tiers.py --docs-check     # fail when it differs from the render

docs/model-support.md is the only render: it opens with the validation runs
at a glance (one block per family, Tier 1 then Tier 2, then the Upcoming
models) and continues with every row's envelope. The README, a published file's Hugging Face card and
the site link there (a row's id is the anchor) instead of carrying a copy of
its marks, so none can drift and nothing watches them (2026-10-27).

`--gate` is the release gate's question: every Tier 1 row must hold a
recorded pass on the backend its smoke names, and no backend of the row may
hold a recorded failure. The record's age is printed and never refuses.
scripts/check-release.py refuses a tag while any row fails that question.
"""

import argparse
import datetime
import html
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
# with: an MTP head as `-m TRUNK --draft mtp:HEAD`, an adapter as
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


def borrowed_classes(devices=None):
    """The ids of the device classes the project does not own (ownership
    "borrowed" in the device ledger: lent with no terms, usable while there,
    never depended on). Their records are dated evidence like any other;
    their age never refuses a tag, a recorded failure on them does. With no
    ledger given, the committed one."""
    if devices is None:
        with open(DEVICE_EVIDENCE, encoding="utf-8") as f:
            devices = json.load(f)
    return frozenset(c["id"] for c in devices.get("classes", [])
                     if c.get("ownership") == "borrowed")


def cell_borrowed(cell, borrowed):
    """Whether a backend cell is a gate recorded on borrowed hardware."""
    return bool(cell) and cell.get("kind") == "gate" and cell.get("device_class") in borrowed


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
        return [f"{where}: lora is null or a recorded --info adapter answer"]
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
        # docs/model-support.md lists a row's model under its
        # family; an Upcoming row is listed by its model alone
        if tier in ("tier1", "tier2") and not _text(row.get("family")):
            errs.append(f"{where}: a {tier} row names its family")
        elif "family" in row and not _text(row["family"]):
            errs.append(f"{where}: family is a display name, not {row['family']!r}")
        claim = row.get("claim") or ""
        # policy decision 4 and section 3: the qualifier is the first thing
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
        # policy decision 1: a published file is Tier 1 unless the exception
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
    # a model is listed once, so all of its rows sit under one family
    families = {}
    for row in doc.get("rows") or []:
        if _text(row.get("model")) and _text(row.get("family")):
            families.setdefault(row["model"], set()).add(row["family"])
    for model, fams in sorted(families.items()):
        if len(fams) > 1:
            errs.append(f"model {model!r}: its rows name the families {sorted(fams)}; "
                        f"a model is listed under one family")
    return errs


def today(override=None):
    if override:
        return datetime.date.fromisoformat(override)
    return datetime.datetime.now(datetime.timezone.utc).date()


def gate(doc, now, out=None, borrowed=None):
    """Report every Tier 1 row; return the ids of those the release gate
    refuses: no recorded pass on the backend the row's smoke names, or a
    recorded failure on any backend.

    Prints the passing rows too, with the age of their record, as the device
    ledger does: a check that only speaks when it is angry trains a reader
    to take silence for coverage. The age is information for the paper
    trail and never refuses (policy decision 2026-10-28: no re-testing on a
    schedule; a record stands until the next validation run or an
    immediate need re-runs it).

    A row whose smoke class is borrowed hardware is reported the same way,
    but a missing record on it is not in the returned list: the project may
    use that machine and must not depend on it, so its absence cannot hold
    a release. A recorded failure is returned on any class.
    """
    # resolved here, not in the signature, so a replaced stdout (pytest's
    # capture) is the one written to
    out = out or sys.stdout
    if borrowed is None:
        borrowed = borrowed_classes()
    rows = [r for r in doc["rows"] if r["tier"] == "tier1"]
    width = max((len(r["id"]) for r in rows), default=0)
    bad = []
    for row in rows:
        smoke = row["smoke"]
        where = f"{smoke['device_class']} {smoke['backend']}"
        cells = row.get("backends") or {}
        cell = cells.get(smoke["backend"])
        failing = sorted(b for b, c in cells.items()
                         if (c or {}).get("kind") == "gate" and c.get("result") != "pass")
        if failing:
            state, detail = "FAILING", "recorded failure on " + ", ".join(failing)
        elif (cell or {}).get("kind") != "gate":
            state, detail = "NEVER", f"no validation run recorded for {smoke['backend']} (the row names {where})"
        else:
            days = (now - datetime.date.fromisoformat(cell["date"])).days
            detail = "%s on %s, %s on %s, %d day%s ago" % (
                smoke["backend"], cell["device_class"], cell["runner_version"],
                cell["date"], days, "" if days == 1 else "s")
            state = "ok"
        lent = smoke["device_class"] in borrowed
        if lent:
            detail += ", borrowed hardware: dated evidence, its absence does not gate"
        if state == "FAILING" or (state != "ok" and not lent):
            bad.append(row["id"])
        print("%-7s %-*s  %s" % (state, width, row["id"], detail), file=out)
    return bad


def record(doc, row_id, res, devices):
    """Write one smoke run's result into its row: the backend cell and the
    file's --info adapter answer. Refuses anything that would make the roster
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
        raise SystemExit("the result carries no --info adapter answer")
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

def mark_parts(cell):
    """A backend cell's mark as (text, link): exactly one of a recorded gate
    with its version and date, a declared limit with the support-matrix path
    it links to, or not run. docs/model-support.md renders these in full
    in each row's envelope; its overview tables write the same states
    compactly (overview_mark)."""
    if cell is None:
        return "☐ not run", None
    if cell["kind"] == "limit":
        return "n/a", cell["matrix"]
    if cell["result"] == "pass":
        return f"✅ v{cell['runner_version']} {cell['date']}", None
    # a recorded failure refuses the next tag; until then it is shown as one
    return f"failed v{cell['runner_version']} {cell['date']}", None


def _mark(cell, base=""):
    """The mark in Markdown; `base` makes the limit's link absolute for a
    page that does not live in this repository."""
    text, link = mark_parts(cell)
    return f"[{text}]({base}{link})" if link else text



def _qualifiers(rows):
    pre = ["Experimental"] if any(r.get("experimental") for r in rows) else []
    return pre + (["Text path only"] if any(r.get("vision_upstream") for r in rows) else [])


def _qualified(rows, name):
    """The qualifier a model leads with, then its name."""
    return ": ".join(_qualifiers(rows) + [name])


def _model_groups(rows):
    """Rows grouped by model, groups sorted by model name, files within a
    group in roster order: the render does not depend on insertion order."""
    groups = {}
    for r in rows:
        groups.setdefault(r["model"], []).append(r)
    return sorted(groups.items(), key=lambda kv: (kv[0].casefold(), kv[0]))



def family_of(row):
    """The family a row is listed under: its `family`, or for a row that
    names none (an Upcoming row) its model."""
    return row.get("family") or row["model"]


def _family_groups(rows):
    """Rows grouped by family, the families in the order of each one's
    first row (the roster's order), each family's models as _model_groups
    sorts them."""
    fams = {}
    for r in rows:
        fams.setdefault(family_of(r), []).append(r)
    return [(fam, _model_groups(rs)) for fam, rs in fams.items()]


def families(doc, tier):
    """The families the overview lists for a tier: one (family, models) per
    family in roster order, its models as (qualifiers, model, rows) sorted
    by name, each with its roster files in roster order."""
    return [(fam, [(_qualifiers(rs), model, rs) for model, rs in groups])
            for fam, groups in _family_groups(
                [r for r in doc["rows"] if r["tier"] == tier])]


def tier1_families(doc):
    """The Tier 1 families, the ones every surface lists (the site)."""
    return families(doc, "tier1")


def tier1_groups(doc):
    """The models every surface lists (the overview, the site): one
    (qualifiers, model, rows) per Tier 1 model, family by family in the
    overview's order."""
    return [g for _, models in tier1_families(doc) for g in models]


def upcoming_names(doc):
    """Upcoming models as names, in the roster's order (the order of
    admission), a vision model marked text path only."""
    names = []
    for r in doc["rows"]:
        if r["tier"] == "upcoming":
            n = r["model"] + (" (text path only)" if r.get("vision_upstream") else "")
            if n not in names:
                names.append(n)
    return names


OVERVIEW_LABELS = ("CPU", "CUDA", "Metal")
NONE_RUN = "-"
PARTIAL = "🟨"
OVERVIEW_LEGEND = (
    "✅ *version date*: recorded gate · blank: nobody has run it yet · n/a: declared "
    "limit, linked to the manual · failed *version date*: recorded failure · Train: the "
    "file's own `--info adapter` answer, where ✅ means the LoRA backward admits the file, "
    "not that training it gives good results. A family's line sums up every file of its "
    "models, per backend and for Train: ✅ all recorded · "
    f"{PARTIAL} *k of n*: k of its n files recorded, a declared limit beside them "
    "counted as not run · a hyphen: none run · n/a: a declared limit on every file · "
    "failed: a recorded failure on any file. For Train, recorded means the LoRA "
    "backward admits the file, and a refusal counts as a declared limit. A mark "
    "with <sup>b</sup> was recorded on borrowed hardware: dated evidence the project "
    "may use and does not depend on. Each file name links to its envelope below.")


def _esc(text):
    return html.escape(text, quote=False)


def overview_mark(cell, borrowed=frozenset()):
    """A backend cell of the overview's file table, as its <td>: the states of
    mark_parts, compact. A recorded gate is its version over its date in a
    cell that does not wrap (GitHub's table layout otherwise splits a date
    at its hyphens when a neighbour claims the width), not run is a blank
    cell, a declared limit is n/a linked to the matrix. A gate recorded on
    borrowed hardware carries a superscript b after its version."""
    if cell is None:
        return "<td></td>"
    if cell["kind"] == "limit":
        return f'<td><a href="{html.escape(_from_docs(cell["matrix"]))}">n/a</a></td>'
    word = "✅" if cell["result"] == "pass" else "failed"
    lent = "<sup>b</sup>" if cell_borrowed(cell, borrowed) else ""
    return f"<td nowrap>{word} {cell['runner_version']}{lent}<br>{cell['date']}</td>"


def overview_train(lora):
    """The Train cell: the recorded --info adapter answer, blank when no gate
    has recorded one; a refusal is the engine's sentence and may wrap."""
    if lora is None:
        return "<td></td>"
    t = lora["training"]
    if t["supported"]:
        return f"<td nowrap>✅ {lora['runner_version']}<br>{lora['date']}</td>"
    return f"<td>no: {_esc(t['reason'])}</td>"


def _overview_file(row):
    """The File cell: the name linked to its envelope further down the page,
    a sharded set by its first part; a row without a file (a Tier 2 model
    admitted on report) by its model name."""
    files = row.get("files") or []
    if not files:
        return f'<td><a href="#{html.escape(row["id"])}">{_esc(row["model"])}</a></td>'
    parts = f" ({len(files)} parts)" if len(files) > 1 else ""
    return (f'<td><a href="#{html.escape(row["id"])}">'
            f"<code>{_esc(files[0]['name'])}</code></a>{parts}</td>")


def _cell_state(cell):
    """A backend cell as a family line counts it: "ok" for a recorded pass,
    "fail", "limit", or None when nobody ran it."""
    if cell is None:
        return None
    if cell["kind"] == "limit":
        return "limit"
    return "ok" if cell["result"] == "pass" else "fail"


def _train_state(lora):
    """A Train answer as a family line counts it: a file the LoRA backward
    admits is recorded, a refusal is the engine's declared limit."""
    if lora is None:
        return None
    return "ok" if lora["training"]["supported"] else "limit"


def family_mark(states):
    """One backend's (or Train's) mark over a family's files, from each
    file's state: n/a when every file has the limit, failed when any
    recorded a failure, ✅ when every file is recorded, NONE_RUN when none
    is, and PARTIAL with k of n otherwise. n counts every file, so a limit
    beside recorded files counts as not run."""
    if all(s == "limit" for s in states):
        return "n/a"
    if "fail" in states:
        return "failed"
    k, n = states.count("ok"), len(states)
    if k == n:
        return "✅"
    return f"{PARTIAL} {k} of {n}" if k else NONE_RUN


def _family_summary(family, models):
    """A family block's summary line, plain text: the bold family name, how
    many models it holds, and a mark per backend and for Train over every
    file of those models. It is HTML (GitHub renders no markdown inside
    <summary>), so the name is escaped and bold is <b>."""
    rows = [r for _, _, rs in models for r in rs]
    marks = [f"{label} {family_mark([_cell_state((r.get('backends') or {}).get(b)) for r in rows])}"
             for b, label in zip(BACKENDS, OVERVIEW_LABELS)]
    marks.append("Train " + family_mark([_train_state(r.get("lora")) for r in rows]))
    n = len(models)
    return (f"<summary><b>{_esc(family)}</b>, {n} model{'' if n == 1 else 's'}: "
            + " · ".join(marks) + "</summary>")


def _model_heading(quals, model):
    """A model's line inside its family's block: the bold name, then its
    label in plain words."""
    label = "".join(", " + q[0].lower() + q[1:] for q in quals)
    return f"<p><b>{_esc(model)}</b>{_esc(label)}</p>"


def render_overview(doc, borrowed=None):
    """The validation runs at a glance, the opening of docs/model-support.md's
    rendered block (render_doc). Until 2026-10-27 this was the README's
    block; the README now says what the check is and links here. Since
    2026-10-28 it lists Tier 2 the same way under Tier 1, so one table holds
    the version and date of every file's last validation run.

    The legend, then one collapsed <details> block per family, Tier 1
    families first and Tier 2 families under their own line, each in the
    roster's order: its summary line names the family, counts its models
    and gives a mark per backend and for Train over all their files;
    inside, each model's line (its name and its label in plain words) and a
    table with one row per roster file of that model, run or not, the file
    name linked to its envelope. A Tier 1 file is one the release gate
    holds Runner to; a Tier 2 file's marks are dated evidence. A quant the
    matrix admits outside the roster is the matrix's claim, not this list's.
    Upcoming models follow as one line of names, in the roster's order (the
    order of admission).

    A block is plain HTML, one line per file: GitHub's markdown passes an
    HTML block through until a blank line, so none may fall inside one, and
    only an HTML cell can be kept from wrapping.
    """
    if borrowed is None:
        borrowed = borrowed_classes()
    head = "".join(f"<th>{h}</th>" for h in ("File",) + OVERVIEW_LABELS + ("Train",))
    lines = [OVERVIEW_LEGEND, ""]
    for tier, intro in (("tier1", "**Tier 1, actively validated** (the release gate holds "
                                  "Runner to these):"),
                        ("tier2", "**Tier 2, maintained on report** (dated evidence, "
                                  "not gated):")):
        fams = families(doc, tier)
        if not fams:
            continue
        lines += [intro, ""]
        for family, models in fams:
            lines += ["<details>", _family_summary(family, models)]
            for quals, model, rows in models:
                lines += [_model_heading(quals, model), "<table>",
                          f"<thead><tr>{head}</tr></thead>", "<tbody>"]
                for r in rows:
                    cells = [_overview_file(r)] + [overview_mark((r.get("backends") or {}).get(b),
                                                                 borrowed) for b in BACKENDS]
                    cells.append(overview_train(r.get("lora")))
                    lines.append("<tr>" + "".join(cells) + "</tr>")
                lines += ["</tbody>", "</table>"]
            lines += ["</details>", ""]
    lines += ["**Upcoming** (planned, not supported yet): "
              + ", ".join(upcoming_names(doc)) + "."]
    return "\n".join(lines)


# --- docs/model-support.md ------------------------------------------------------

def _from_docs(path):
    """A repository path as a link from docs/model-support.md."""
    if path.startswith(("http://", "https://")):
        return path
    return path[len("docs/"):] if path.startswith("docs/") else "../" + path


def _cell_detail(cell, borrowed=frozenset()):
    if cell is None:
        return "☐ not run"
    if cell["kind"] == "limit":
        return f"[n/a]({_from_docs(cell['matrix'])}): {cell['reason']}"
    pin = cell["evidence"]
    lent = " (borrowed hardware)" if cell_borrowed(cell, borrowed) else ""
    return (f"{_mark(cell)} on {cell['device_class']}{lent} at commit "
            f"`{(cell.get('commit') or 'unknown')[:12]}`, output sha256 "
            f"`{cell['output_sha256'][:16]}`, [pin]({_from_docs(pin)})")


def _lora_detail(lora):
    if lora is None:
        return ["- **LoRA:** ☐ not run (no gate has recorded the file's `--info adapter` answer)."]
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


def _row_doc(row, by_id, level=4, borrowed=frozenset()):
    files = row.get("files") or []
    title = f"`{files[0]['name']}`" + (f" ({len(files)} parts)" if len(files) > 1 else "") \
        if files else row["model"]
    lines = [f'<a id="{row["id"]}"></a>', f"{'#' * level} {title}", "", row["claim"], ""]
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
        flag = {"mtp-file": "--draft mtp:FILE", "lora": "--lora"}[rw["as"]]
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
        lent = (" (borrowed hardware: dated evidence, its absence does not hold a release)"
                if sm["device_class"] in borrowed else "")
        cell = (row.get("backends") or {}).get(sm["backend"])
        last = (f"last run {cell['runner_version']} on {cell['date']}"
                if (cell or {}).get("kind") == "gate" else "no run recorded yet")
        lines.append(f"- **Validation run:** on {sm['device_class']}{lent}, "
                     f"{sm['backend']} backend; {last}.")
    if row.get("backends"):
        for b in BACKENDS:
            lines.append(f"- **{b.upper() if b != 'metal' else 'Metal'}:** "
                         f"{_cell_detail(row['backends'].get(b), borrowed)}")
    if row["tier"] == "tier1" or row.get("lora") is not None:
        lines += _lora_detail(row.get("lora"))
    hist = "; ".join(f"{TIER_NAMES[t['tier']]} from {t['effective']}" for t in row["transitions"])
    lines += [f"- **Tier history:** {hist} ([log](#transition-log)).", ""]
    return lines


def render_doc(doc, borrowed=None):
    """The block between DOC_BEGIN and DOC_END: the validation runs at a
    glance (render_overview), then every row's envelope by tier, family and model
    (an Upcoming model on its own, in the order of admission), the exempt
    repositories, and the transition log."""
    if borrowed is None:
        borrowed = borrowed_classes()
    by_id = {r["id"]: r for r in doc["rows"]}
    # a blank line on each side keeps the marker's HTML comment apart from
    # the legend; the <details> blocks are HTML, so no blank line falls
    # inside one
    lines = ["", "", '<a id="overview"></a>', "## Validation runs at a glance", "",
             render_overview(doc, borrowed), ""]
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
            for model, rs in groups.items():
                lines += [f"### {_qualified(rs, model)}", ""]
                for r in rs:
                    lines += _row_doc(r, by_id, borrowed=borrowed)
            continue
        # the overview's grouping: families in roster order, models by name
        for family, groups in _family_groups(rows):
            lines += [f"### {family}", ""]
            for model, rs in groups:
                lines += [f"#### {_qualified(rs, model)}", ""]
                for r in rs:
                    lines += _row_doc(r, by_id, level=5, borrowed=borrowed)
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


def _blocks(doc, borrowed=None):
    """(path, begin, end, rendered text) for every rendered block."""
    if borrowed is None:
        borrowed = borrowed_classes()
    return [(DOC, DOC_BEGIN, DOC_END, render_doc(doc, borrowed))]


def _replace_block(text, begin, end, body, path):
    if text.count(begin) != 1 or text.count(end) != 1:
        raise SystemExit(f"{path}: expected one {begin!r} and one {end!r}")
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    return head + begin + body + end + tail


def docs_drift(doc, borrowed=None):
    """The files whose rendered block differs from the roster's render."""
    bad = []
    for path, begin, end, body in _blocks(doc, borrowed):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if _replace_block(text, begin, end, body, path) != text:
            bad.append(os.path.relpath(path, ROOT))
    return bad


def write_docs(doc, borrowed=None):
    """Re-render every block from the roster (after a --record or --retire)."""
    for path, begin, end, body in _blocks(doc, borrowed):
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
    ap.add_argument("--gate", action="store_true",
                    help="the release gate: refuse while a Tier 1 row has no recorded "
                         "pass on its backend or a recorded failure on any")
    ap.add_argument("--today", help="the day the printed ages are measured from "
                                    "(tests); default the UTC clock")
    ap.add_argument("--record", metavar="ROW",
                    help="write a smoke run's result into ROW")
    ap.add_argument("--result", help="the result file scripts/tier-smoke.py wrote")
    ap.add_argument("--write-docs", action="store_true",
                    help="re-render docs/model-support.md from the roster")
    ap.add_argument("--docs-check", action="store_true",
                    help="fail when docs/model-support.md differs from the render")
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
            write_docs(doc, borrowed_classes(devices))
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
            write_docs(doc, borrowed_classes(devices))
            return 0
        bad = docs_drift(doc, borrowed_classes(devices))
        for b in bad:
            print(f"{b}: the rendered block differs from docs/model-tiers.json; "
                  f"run scripts/model-tiers.py --write-docs", file=sys.stderr)
        return 1 if bad else 0
    if args.gate:
        bad = gate(doc, today(args.today), borrowed=borrowed_classes(devices))
        if bad:
            # the rows went to stdout, which a pipe buffers while stderr is
            # written at once: flush, or a reader of both in one pipe gets
            # the summary before the rows it summarises
            sys.stdout.flush()
            print("\n%d Tier 1 file%s need%s a validation run: %s" % (
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
