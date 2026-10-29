#!/usr/bin/env python3
"""The per-release smoke over the model roster (docs/model-tiers.json).

For each Tier 1 file the roster gates on this machine's device class (or the
one row named with --row, which may be a Tier 2 row whose run is dated
evidence and gates nothing): hash the file and refuse it unless it is the
row's, ask the binary for its --caps and for the file's --adapt-info answer,
serve the file and ask it one user turn through its own chat template (POST
/v1/chat/completions, temperature 0, max_tokens n_predict, not streamed),
and byte-compare the reply's content to the pin committed under
tests/compatibility/tier-smoke/. --record writes the result into the row's
cell for the backend it ran on, and the --adapt-info answer into the row's
lora field, through scripts/model-tiers.py's own record().

    scripts/tier-smoke.py --row ID --model FILE --backend metal --pin
    scripts/tier-smoke.py --row ID --model FILE --backend metal --record
    scripts/tier-smoke.py --models DIR --record      # every row for this class
    scripts/tier-smoke.py --retire --row ID --backend metal --device-class C

The device class is matched from this machine's --caps against
docs/device-evidence.json, or named with --device-class. A CPU cell of a GPU
class runs with the GPU visible and --backend cpu: the smoke serves it with
--gpu off itself, and a hidden GPU (CUDA_VISIBLE_DEVICES empty) leaves --caps
with no GPU, which matches no GPU class. When no class matches, the refusal
lists every class with the caps fields it expected otherwise.

Why the chat path: `runner -p` is a raw one-shot prompt with no chat
template, and an instruct model given one continues the text instead of
answering it (the Windows CUDA host's first run pinned Granite 4.2
continuing the prompt with its own "Constraints:"). A user turn is what a
client sends, so it is what the gate measures. The request sends
enable_thinking false by default: a thinking family would otherwise spend
the whole budget on reasoning_content and leave no answer to pin.

What the gate proves, and against what. Its absolute anchors are the file's
sha256 (the row's, read from the publisher's Hugging Face listing, so the run
is on the published bytes) and the fact the prompt asks for (the reply to
"List the eight planets ... in order" must name "Mercury"). The byte pin is
not an anchor: it was written by an earlier run of this engine, so a match
proves this build reproduces that one, and a mismatch says something
changed. Generated tokens may change between releases on purpose
(docs/versioning.md); a mismatch is then read, the reason written in a
commit, the pin retired and pinned again. This script never overwrites a pin.

A pin belongs to one device class and one backend. CPU and Metal kernels
differ in summation order, so neither inherits the other's pin; a backend
with no pin of its own has not been run.

A pin holds the fact and is long enough to catch a change: a reply that does
not name the fact, or is shorter than the manifest's min_generated_bytes,
is not pinned (the reply is printed), and an existing pin under the minimum
is not compared against. A pin is the reply's UTF-8 bytes as sent, written
in binary, and every JSON file is written with LF on every host. --retire
removes a pin, its manifest entry and the roster cell that cites it,
together.
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
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "tests", "compatibility", "tier-smoke.json")
PINS = os.path.join(ROOT, "tests", "compatibility", "tier-smoke")
MANIFEST_SCHEMA = "xyntetik.runner.tier-smoke.v1"

_spec = importlib.util.spec_from_file_location(
    "model_tiers", os.path.join(ROOT, "scripts", "model-tiers.py"))
mt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mt)

# the tests' own server start: a free port, /health, SIGTERM and a log of
# the server's stdout and stderr, where the backend banner is read
sys.path.insert(0, os.path.join(ROOT, "tests", "conformance"))
from _process import RunnerServer  # noqa: E402

# the shared helpers live beside this script; a test that loads this file by
# path has not put the directory on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from file_digest import sha256_file  # noqa: E402

BANNER_RE = re.compile(r"^gpu: (Metal|CUDA) backend on .*$", re.M)
THREADS_RE = re.compile(r"^loaded .*\| (\d+) threads\b", re.M)


def runner_cmd(path):
    # a .py runner is the tests' stand-in process; the real one is a binary
    return [sys.executable, path] if path.endswith(".py") else [path]


def run(cmd, timeout):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout)


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


def _caps_field(key, value):
    """One caps field as the refusal reads it: a GPU that --caps does not
    report (src/main.c prints "gpu": null) is "no GPU", not "gpu_backend None"."""
    if value is None:
        return "no GPU" if key == "gpu_backend" else f"no {key}"
    return f"{key} {value}"


def no_class_message(caps, classes, where):
    """Why no class matched: every class the ledger has, the caps fields each
    one expected otherwise, and how a GPU class's CPU cell is run. A bare
    "matches device classes []" sent the operator of a CPU row on the Linux
    CUDA host, run with the GPU hidden, to read the script."""
    if not classes:
        return f"{where} names no device classes"
    keys = []
    for exp in classes.values():
        keys += [k for k in exp if k not in keys]
    lines = [f"no device class in {where} matches this machine's --caps "
             f"({', '.join(_caps_field(k, caps.get(k)) for k in keys)}):"]
    hidden_gpu = []
    for cid, exp in classes.items():
        diff = [k for k, v in exp.items() if caps.get(k) != v]
        lines.append(f"  - caps report {', '.join(_caps_field(k, caps.get(k)) for k in diff)}; "
                     f"class {cid} expects {', '.join(f'{k} {exp[k]}' for k in diff)}")
        if diff == ["gpu_backend"] and caps.get("gpu_backend") is None:
            hidden_gpu.append(cid)
    if caps.get("gpu_backend") is None:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            lines.append(f"CUDA_VISIBLE_DEVICES is {os.environ['CUDA_VISIBLE_DEVICES']!r} "
                         f"in this environment, and an empty value or an index with no "
                         f"device hides the GPU from the runner.")
        named = " or ".join(f"--device-class {c}" for c in hidden_gpu) or "--device-class ID"
        lines.append("A CPU cell of a GPU class is run on that class's machine with the "
                     "GPU visible and --backend cpu (the smoke serves with --gpu off "
                     f"itself), or with the class named: {named}.")
    else:
        lines.append("If this machine is one of these classes, name it with --device-class.")
    return "\n".join(lines)


def this_class(caps, devices, override=None, where="docs/device-evidence.json"):
    """The device class this machine is, by its own --caps."""
    classes = mt.device_classes(devices)
    if override:
        if override not in classes:
            raise SystemExit(f"unknown device class {override!r}; {where} names "
                             f"{', '.join(classes) or 'none'}")
        return override
    hits = [cid for cid, exp in classes.items()
            if all(caps.get(k) == v for k, v in exp.items())]
    if not hits:
        raise SystemExit(no_class_message(caps, classes, where))
    if len(hits) > 1:
        raise SystemExit(f"this machine matches device classes {hits}; name one "
                         f"with --device-class")
    return hits[0]


def write_json(obj, path):
    """JSON as the repository keeps it: UTF-8 and LF on every host. A text-mode
    write on Windows turns every newline into CRLF and the whole manifest into
    a diff."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def chat(port, body, timeout):
    """One buffered chat turn: the reply's content as UTF-8 bytes, how the turn
    ended, its completion tokens, and any reasoning_content beside it."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        doc = json.loads(r.read().decode("utf-8"))
    choice = doc["choices"][0]
    msg = choice.get("message") or {}
    return ((msg.get("content") or "").encode("utf-8"), choice.get("finish_reason"),
            (doc.get("usage") or {}).get("completion_tokens"),
            msg.get("reasoning_content") or "")


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
    write_json(manifest, manifest_path)
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
    for k in ("prompt", "n_predict", "ctx", "min_generated_bytes"):
        if k not in spec:
            raise SystemExit(f"{row['id']}: the manifest names no {k}")
    extra = companion + list(spec.get("server_args") or [])
    if backend == "cpu":
        extra += ["--gpu", "off"]
    # a row's arguments for one backend only, e.g. the layer split of a file
    # larger than the card: {"cuda": ["--gpu", "30"]}
    backend_args = list((spec.get("backend_args") or {}).get(backend) or [])
    extra += backend_args
    body = {"messages": [{"role": "user", "content": spec["prompt"]}],
            "temperature": 0, "max_tokens": spec["n_predict"], "stream": False}
    body.update(spec.get("request") or {})

    # the model's own chat template: the file is served and asked one user
    # turn, as a client would; the CLI's -p is a raw continuation, which an
    # instruct model answers by continuing the prompt rather than replying
    with tempfile.TemporaryDirectory() as td:
        log = os.path.join(td, "server.log")
        t0 = time.monotonic()
        try:
            with RunnerServer(runner, model, extra_args=extra, ctx=spec["ctx"],
                              start_timeout=timeout, log_path=log) as srv:
                out, finish, ctoks, reasoning = chat(srv.port, body, timeout)
        except SystemExit:
            raise
        except Exception as e:  # TransportError, HTTP errors: the gate did not run
            tail = open(log, "rb").read()[-2000:].decode("utf-8", "replace") \
                if os.path.exists(log) else ""
            raise SystemExit(f"{row['id']} {backend}: the server run failed: {e}\n{tail}")
        seconds = round(time.monotonic() - t0, 2)
        with open(log, "rb") as f:
            server_log = f.read().decode("utf-8", "replace")

    # the backend's own statement that it took the model; a fallback line
    # means it did not, whatever printed before it
    m = BANNER_RE.search(server_log)
    banner = m.group(0) if m and "using CPU" not in server_log else None
    t = THREADS_RE.search(server_log)

    minimum = spec["min_generated_bytes"]
    expect = spec.get("expect")
    found = bool(expect) and expect.encode("utf-8") in out

    key = f"{dc}/{backend}"
    name = f"{row['id']}.{dc}.{backend}.txt"
    pin_path = os.path.join(pins_dir, name)
    entry = manifest.setdefault("rows", {}).setdefault(row["id"], {}).setdefault("pins", {})
    out_sha = hashlib.sha256(out).hexdigest()
    # what produced the bytes: a pin compared under other settings would
    # report the manifest's change as the engine's
    made_with = {"prompt": spec["prompt"], "n_predict": spec["n_predict"],
                 "ctx": spec["ctx"], "server_args": list(spec.get("server_args") or []),
                 "request": dict(spec.get("request") or {})}
    if backend_args:
        made_with["backend_args"] = backend_args
    if pin:
        if os.path.exists(pin_path) or key in entry:
            raise SystemExit(f"{row['id']} {key} is already pinned ({rel(pin_path)}); "
                             f"retire it (--retire), with the reason in the commit, "
                             f"to re-pin")
        shown = out.decode("utf-8", "replace")
        if not found:
            raise SystemExit(f"{row['id']} {key}: the reply does not name {expect!r}, so "
                             f"it is not pinned (a pin always holds the fact). The "
                             f"reply, finish_reason {finish}, {ctoks} tokens:\n{shown}")
        if len(out) < minimum:
            why = (f"; reasoning_content held {len(reasoning)} bytes" if reasoning else "")
            raise SystemExit(f"{row['id']} {key}: the reply is {len(out)} bytes, under "
                             f"the manifest's minimum of {minimum}{why}; a pin that "
                             f"short catches almost no change. Give the row a prompt, "
                             f"n_predict or request that keeps the model answering.")
        os.makedirs(pins_dir, exist_ok=True)
        with open(pin_path, "wb") as f:
            f.write(out)
        entry[key] = {"path": rel(pin_path), "sha256": out_sha, "bytes": len(out),
                      "finish_reason": finish, "completion_tokens": ctoks,
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
        if len(want) < minimum:
            raise SystemExit(f"{rel(pin_path)} holds {len(want)} bytes, under the "
                             f"minimum of {minimum}: retire it (--retire) and pin "
                             f"again rather than compare against it")
        status = "match" if out == want else "mismatch"
    shown_args = ["-m", os.path.basename(model)] + \
        [os.path.basename(a) if a in companion[1:] else a for a in companion] + \
        ["--serve", "-c", str(spec["ctx"])] + extra[len(companion):]
    ran = (" ".join(shlex.quote(a) for a in ["runner"] + shown_args)
           + "; POST /v1/chat/completions "
           + " ".join(f"{k}={json.dumps(v)}" for k, v in body.items() if k != "messages"))
    return {
        "schema": mt.RESULT_SCHEMA,
        "row": row["id"],
        "backend": backend,
        "device_class": dc,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": git_commit(),
        "ran": ran,
        "caps": caps,
        "banner": banner,
        "threads": int(t.group(1)) if t else None,
        "seconds": seconds,
        "files": hashed,
        "with": ({"row": with_row["id"], "files": with_hashed}
                 if with_row is not None else None),
        "status": status,
        "output": {"sha256": out_sha, "bytes": len(out), "finish_reason": finish,
                   "completion_tokens": ctoks},
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
        this_class({}, devices, args.device_class, rel(args.device_evidence))
        retire(args.row, args.backend, args.device_class, args.roster, manifest,
               args.manifest, args.pins, devices)
        return 0
    runner = runner_cmd(args.runner)
    caps = caps_of(runner)
    dc = this_class(caps, devices, args.device_class, rel(args.device_evidence))

    tier1 = [r for r in doc["rows"] if r["tier"] == "tier1"]
    if args.row:
        # a named row may be Tier 2: its run is dated evidence that gates
        # nothing, and a Tier 2 admission records the pass that admitted it
        todo = [r for r in doc["rows"] if r["id"] == args.row
                and r["tier"] in ("tier1", "tier2") and r.get("smoke")]
        if not todo:
            raise SystemExit(f"{args.row!r} is not a Tier 1 or Tier 2 row with a smoke")
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
            write_json(manifest, args.manifest)
        if args.out:
            write_json(res, args.out)
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
