#!/usr/bin/env python3
"""The per-release smoke over the model roster (docs/model-tiers.json).

For each Tier 1 file the roster gates on this machine's device class (or the
one row named with --row): hash the file and refuse it unless it is the
row's, ask the binary for its --caps and for the file's --adapt-info answer,
run one greedy prompt, and byte-compare stdout to the pin committed under
tests/compatibility/tier-smoke/. --record writes the result into the row's
cell for the backend it ran on, and the --adapt-info answer into the row's
lora field, through scripts/model-tiers.py's own record().

    scripts/tier-smoke.py --row ID --model FILE --backend metal --pin
    scripts/tier-smoke.py --row ID --model FILE --backend metal --record
    scripts/tier-smoke.py --models DIR --record      # every row for this class
    scripts/tier-smoke.py --retire --row ID --backend metal --device-class C

What the gate proves, and against what. Its absolute anchors are the file's
sha256 (the row's, read from the publisher's Hugging Face listing, so the run
is on the published bytes) and the fact the prompt asks for (the
continuation of "The capital of France is" must contain "Paris"). The byte
pin is not an anchor: it was written by an earlier run of this engine, so a
match proves this build reproduces that one, and a mismatch says something
changed. Generated tokens may change between releases on purpose
(docs/versioning.md); a mismatch is then read, the reason written in a
commit, the pin deleted and pinned again. This script never overwrites a pin.

A pin belongs to one device class and one backend. CPU and Metal kernels
differ in summation order, so neither inherits the other's pin; a backend
with no pin of its own has not been run.

A pin must be long enough to catch a change: its generated continuation (the
stdout after the echoed prompt, without the CLI's closing newline) is at
least the manifest's min_generated_bytes, or it is not written, and an
existing pin under the minimum is not compared against. --retire removes a
pin, its manifest entry and the roster cell that cites it, together.
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "tests", "compatibility", "tier-smoke.json")
PINS = os.path.join(ROOT, "tests", "compatibility", "tier-smoke")
MANIFEST_SCHEMA = "xyntetik.runner.tier-smoke.v1"

_spec = importlib.util.spec_from_file_location(
    "model_tiers", os.path.join(ROOT, "scripts", "model-tiers.py"))
mt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mt)

BANNER_RE = re.compile(r"^gpu: (Metal|CUDA) backend on .*$", re.M)
THREADS_RE = re.compile(r"^loaded .*\| (\d+) threads \|", re.M)


def runner_cmd(path):
    # a .py runner is the tests' stand-in process; the real one is a binary
    return [sys.executable, path] if path.endswith(".py") else [path]


def run(cmd, timeout):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    """Repo-relative when the path is inside this checkout."""
    p = os.path.abspath(path)
    r = os.path.relpath(p, ROOT)
    return p if r.startswith("..") else r.replace(os.sep, "/")


def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except OSError:
        return None


def caps_of(runner):
    p = run(runner + ["--caps"], 60)
    if p.returncode != 0:
        raise SystemExit("runner --caps failed: " + p.stderr.decode(errors="replace"))
    caps = json.loads(p.stdout)
    gpu = caps.get("gpu") or {}
    # the fields scripts/device-evidence.py fingerprints a machine with
    return {"version": caps.get("version"), "os": caps.get("os"),
            "arch": caps.get("arch"), "cpu_cores": caps.get("cpu_cores"),
            "gpu_backend": gpu.get("backend"), "gpu_name": gpu.get("name")}


def this_class(caps, devices, override=None):
    """The device class this machine is, by its own --caps."""
    classes = mt.device_classes(devices)
    if override:
        if override not in classes:
            raise SystemExit(f"unknown device class {override!r}")
        return override
    hits = [cid for cid, exp in classes.items()
            if all(caps.get(k) == v for k, v in exp.items())]
    if len(hits) != 1:
        raise SystemExit(f"this machine matches device classes {hits}; name one "
                         f"with --device-class")
    return hits[0]


def generated(out, prompt):
    """The model's continuation: stdout after the echoed prompt, without the
    newline the CLI prints when generation ends."""
    cont = out[len(prompt):] if out.startswith(prompt) else out
    return cont[:-1] if cont.endswith(b"\n") else cont


def retire(row_id, backend, dc, roster_path, manifest, manifest_path, pins_dir, devices):
    """Remove a pin, its manifest entry and the roster cell whose evidence it
    is. The cell goes because its evidence does; nothing else in the row
    (the other backends, the --adapt-info answer) cites the pin."""
    doc = mt.load(roster_path)
    key = f"{dc}/{backend}"
    pin_path = os.path.join(pins_dir, f"{row_id}.{dc}.{backend}.txt")
    entry = (manifest.get("rows", {}).get(row_id) or {}).get("pins", {})
    if key not in entry and not os.path.exists(pin_path):
        raise SystemExit(f"{row_id} {key} has no pin to retire")
    entry.pop(key, None)
    if os.path.exists(pin_path):
        os.remove(pin_path)
    retired = None
    for row in doc["rows"]:
        cell = (row.get("backends") or {}).get(backend)
        if row["id"] == row_id and (cell or {}).get("evidence") == rel(pin_path):
            row["backends"][backend] = None
            retired = cell
    errs = mt.validate(doc, devices)
    if errs:
        raise SystemExit("refusing to write a roster that does not validate:\n"
                         + "\n".join(errs))
    mt.save(doc, roster_path)
    if mt.is_committed_roster(roster_path):
        mt.write_docs(doc)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"retired {row_id} {key}" + (f" and its {backend} cell ({retired['runner_version']} "
                                       f"on {retired['date']})" if retired else ""))


def find_file(name, models_dir):
    for dirpath, _, files in os.walk(models_dir):
        if name in files:
            return os.path.join(dirpath, name)
    return None


def hash_files(row, paths):
    """The files' digests, refusing any file that is not the row's."""
    hashed = []
    for want, path in zip(row["files"], paths):
        got = sha256_file(path)
        if got != want["sha256"]:
            raise SystemExit(f"{row['id']}: {path} has sha256 {got}, the row names "
                             f"{want['sha256']}; refusing to run on another file")
        hashed.append({"sha256": got, "bytes": os.path.getsize(path)})
    return hashed


# how a companion file joins the run of the model it names (runs_with.as)
COMPANION_FLAG = {"mtp-file": "--mtp-file", "lora": "--lora"}


def smoke_one(row, backend, dc, model_paths, runner, caps, manifest, pins_dir,
              pin, timeout, with_row=None, with_paths=None):
    """One gate run; returns the result document (schema tier-smoke-result.v1).

    A companion row (runs_with) is run as the model it names plus this file:
    `-m TRUNK --mtp-file HEAD` or `-m BASE --lora ADAPTER`. Both files are
    hashed against their rows first, and the pin is the composed run's,
    under the companion's own row id.
    """
    # the files first: a run on other bytes is not this row's evidence
    hashed = hash_files(row, model_paths)
    companion = []
    with_hashed = None
    if with_row is not None:
        with_hashed = hash_files(with_row, with_paths)
        companion = [COMPANION_FLAG[row["runs_with"]["as"]], model_paths[0]]
        model = with_paths[0]
    else:
        model = model_paths[0]

    # --adapt-info answers for the model the configuration trains and serves
    # on: for a companion, the trunk or base it runs with
    p = run(runner + ["-m", model, "--adapt-info", "--no-tray"], timeout)
    if p.returncode != 0:
        raise SystemExit("--adapt-info failed: " + p.stderr.decode(errors="replace"))
    adapt = json.loads(p.stdout)

    spec = dict(manifest.get("defaults") or {})
    spec.update({k: v for k, v in (manifest.get("rows", {}).get(row["id"]) or {}).items()
                 if k != "pins"})
    args = ["-m", model] + companion + ["-p", spec["prompt"], "-n", str(spec["n_predict"])]
    args += list(spec.get("args") or [])
    if backend == "cpu":
        args += ["--gpu", "off"]
    t0 = time.monotonic()
    p = run(runner + args, timeout)
    seconds = round(time.monotonic() - t0, 2)
    stderr = p.stderr.decode("utf-8", errors="replace")
    if p.returncode != 0:
        raise SystemExit(f"{row['id']} {backend}: the run exited {p.returncode}:\n{stderr}")
    out = p.stdout

    # the backend's own statement that it took the model; a fallback line
    # means it did not, whatever printed before it
    m = BANNER_RE.search(stderr)
    banner = m.group(0) if m and "using CPU" not in stderr else None
    t = THREADS_RE.search(stderr)

    prompt = spec["prompt"].encode("utf-8")
    cont = generated(out, prompt)
    minimum = spec.get("min_generated_bytes")
    if not isinstance(minimum, int) or minimum <= 0:
        raise SystemExit(f"{row['id']}: the manifest names no min_generated_bytes")
    expect = spec.get("expect")
    found = bool(expect) and expect.encode("utf-8") in cont

    key = f"{dc}/{backend}"
    name = f"{row['id']}.{dc}.{backend}.txt"
    pin_path = os.path.join(pins_dir, name)
    entry = manifest.setdefault("rows", {}).setdefault(row["id"], {}).setdefault("pins", {})
    out_sha = hashlib.sha256(out).hexdigest()
    # what produced the bytes: a pin compared under other settings would
    # report the manifest's change as the engine's
    made_with = {"prompt": spec["prompt"], "n_predict": spec["n_predict"],
                 "args": list(spec.get("args") or [])}
    if pin:
        if os.path.exists(pin_path) or key in entry:
            raise SystemExit(f"{row['id']} {key} is already pinned ({rel(pin_path)}); "
                             f"retire it (--retire), with the reason in the commit, "
                             f"to re-pin")
        if len(cont) < minimum:
            raise SystemExit(f"{row['id']} {key}: the continuation is {len(cont)} bytes, "
                             f"under the manifest's minimum of {minimum}; a pin that "
                             f"short catches almost no change. Give the row a prompt "
                             f"or n_predict that keeps the model generating.")
        os.makedirs(pins_dir, exist_ok=True)
        with open(pin_path, "wb") as f:
            f.write(out)
        entry[key] = {"path": rel(pin_path), "sha256": out_sha, "bytes": len(out),
                      "spec": made_with,
                      "pinned": {"date": datetime.datetime.now(datetime.timezone.utc)
                                 .date().isoformat(),
                                 "runner_version": caps["version"],
                                 "commit": git_commit()}}
        status = "pinned"
    else:
        if not os.path.exists(pin_path) or key not in entry:
            raise SystemExit(f"{row['id']} {key}: no pin to compare against; run once "
                             f"with --pin and commit the pin")
        if entry[key].get("spec") != made_with:
            raise SystemExit(f"{rel(pin_path)} was pinned with different settings "
                             f"({entry[key].get('spec')}) than the manifest's now "
                             f"({made_with}): retire it (--retire) and pin again")
        with open(pin_path, "rb") as f:
            want = f.read()
        if hashlib.sha256(want).hexdigest() != entry[key]["sha256"]:
            raise SystemExit(f"{rel(pin_path)} does not match its manifest digest")
        short = len(generated(want, prompt))
        if short < minimum:
            raise SystemExit(f"{rel(pin_path)} holds {short} generated bytes, under "
                             f"the minimum of {minimum}: retire it (--retire) and pin "
                             f"again rather than compare against it")
        status = "match" if out == want else "mismatch"
    return {
        "schema": mt.RESULT_SCHEMA,
        "row": row["id"],
        "backend": backend,
        "device_class": dc,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": git_commit(),
        "ran": " ".join(shlex.quote(a) for a in
                        ["runner"] + [os.path.basename(a) if a in (model, *companion[1:])
                                      else a for a in args]),
        "caps": caps,
        "banner": banner,
        "threads": int(t.group(1)) if t else None,
        "seconds": seconds,
        "files": hashed,
        "with": ({"row": with_row["id"], "files": with_hashed}
                 if with_row is not None else None),
        "status": status,
        "output": {"sha256": out_sha, "bytes": len(out)},
        "pin": {"path": rel(pin_path), "sha256": entry[key]["sha256"]},
        "expect": {"substring": expect, "found": found},
        "adapt_info": adapt,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner", default=os.path.join(ROOT, "runner"))
    ap.add_argument("--roster", default=mt.ROSTER)
    ap.add_argument("--device-evidence", default=mt.DEVICE_EVIDENCE)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--pins", default=PINS)
    ap.add_argument("--row", help="one Tier 1 row (default: every row this class gates)")
    ap.add_argument("--model", action="append",
                    help="the row's file (repeat for each part of a sharded set)")
    ap.add_argument("--with-model", action="append",
                    help="the file of the row a companion runs with (runs_with), "
                         "repeat for each part")
    ap.add_argument("--models", help="a directory searched for the rows' file names")
    ap.add_argument("--backend", choices=mt.BACKENDS,
                    help="default: the backend the row's smoke names")
    ap.add_argument("--device-class", help="default: matched from this machine's --caps")
    ap.add_argument("--pin", action="store_true",
                    help="write the pin (a row and backend with none yet)")
    ap.add_argument("--record", action="store_true",
                    help="write the result into docs/model-tiers.json")
    ap.add_argument("--retire", action="store_true",
                    help="remove the pin of --row/--backend, its manifest entry and "
                         "the roster cell that cites it")
    ap.add_argument("--out", help="also write the result document here")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()
    if args.pin and args.record:
        raise SystemExit("--pin compares nothing, so it records nothing; pin, "
                         "commit the pin, then run with --record")

    doc = mt.load(args.roster)
    with open(args.device_evidence, encoding="utf-8") as f:
        devices = json.load(f)
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit(f"{args.manifest}: schema {manifest.get('schema')!r}")
    if args.retire:
        if not (args.row and args.backend and args.device_class):
            raise SystemExit("--retire needs --row, --backend and --device-class")
        this_class({}, devices, args.device_class)
        retire(args.row, args.backend, args.device_class, args.roster, manifest,
               args.manifest, args.pins, devices)
        return 0
    runner = runner_cmd(args.runner)
    caps = caps_of(runner)
    dc = this_class(caps, devices, args.device_class)

    tier1 = [r for r in doc["rows"] if r["tier"] == "tier1"]
    if args.row:
        todo = [r for r in tier1 if r["id"] == args.row]
        if not todo:
            raise SystemExit(f"{args.row!r} is not a Tier 1 row")
    else:
        todo = [r for r in tier1 if r["smoke"]["device_class"] == dc]
    failed = 0
    for row in todo:
        backend = args.backend or row["smoke"]["backend"]
        if args.model:
            paths = args.model
        elif args.models:
            paths = [find_file(f["name"], args.models) for f in row["files"]]
        else:
            raise SystemExit("name the file with --model or a directory with --models")
        with_row = with_paths = None
        if row.get("runs_with"):
            with_row = next(r for r in doc["rows"] if r["id"] == row["runs_with"]["row"])
            if args.with_model:
                with_paths = args.with_model
            elif args.models:
                with_paths = [find_file(f["name"], args.models) for f in with_row["files"]]
            else:
                raise SystemExit(f"{row['id']} runs with {with_row['id']}: name its file "
                                 f"with --with-model or a directory with --models")
            if len(with_paths) != len(with_row["files"]) or not all(with_paths):
                print(f"SKIP  {row['id']}: the file of {with_row['id']} not found",
                      file=sys.stderr)
                failed += 1
                continue
        if len(paths) != len(row["files"]) or not all(paths):
            print(f"SKIP  {row['id']}: file not found", file=sys.stderr)
            failed += 1
            continue
        res = smoke_one(row, backend, dc, paths, runner, caps, manifest, args.pins,
                        args.pin, args.timeout, with_row, with_paths)
        if args.pin:
            with open(args.manifest, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
                f.write("\n")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
                f.write("\n")
        line = (f"{res['status']:<8} {row['id']} {dc} {backend} "
                f"({res['seconds']} s, fact {'found' if res['expect']['found'] else 'MISSING'})")
        print(line)
        if args.record:
            mt.record(doc, row["id"], res, devices)
            errs = mt.validate(doc, devices)
            if errs:
                raise SystemExit("refusing to write a roster that does not validate:\n"
                                 + "\n".join(errs))
            mt.save(doc, args.roster)
            if mt.is_committed_roster(args.roster):
                mt.write_docs(doc)
            cell = next(r for r in doc["rows"] if r["id"] == row["id"])["backends"][backend]
            print(f"recorded {row['id']} {backend}: {cell['result']}")
        if res["status"] == "mismatch" or (res["status"] == "match" and not res["expect"]["found"]):
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
