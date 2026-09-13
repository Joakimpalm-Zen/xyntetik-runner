"""Capture completeness, workspace reconstruction and verification binding.

Learning-method eligibility is owned by Shade, which consumes these records.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import capture as cap

FIELDS = ["raw_payload", "provenance_labelled", "linked_to_task",
          "attribution_established", "parent_worker_explicit", "pre_or_post_state",
          "state_contents_stored", "state_complete", "commit_head",
          "model_interaction", "verification"]

# What each candidate method needs from ONE captured task, stated as a
# requirement list rather than prose so the answer is computed, not argued.


def load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("schema") == cap.SCHEMA:
            rows.append(r)
    return rows


def threads(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group records by the task they belong to. A record with no task id is kept
    under the empty key rather than dropped, because an unlinked record is a
    finding."""
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        tid = str((r.get("provenance") or {}).get("task_id") or "")
        out.setdefault(tid, []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r.get("timestamp") or "")
    return out


def completeness_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agg = {f: 0 for f in FIELDS}
    for r in rows:
        c = r.get("completeness") or {}
        for f in FIELDS:
            if c.get(f):
                agg[f] += 1
    n = len(rows) or 1
    return {"records": len(rows),
            "present": agg,
            "share": {f: round(agg[f] / n, 3) for f in FIELDS}}


def replay(home: Path, rec: dict[str, Any]) -> dict[str, Any]:
    """Can this record's state be reconstructed from the blob store, and does it
    agree with the manifest the record claims?

    Reports three outcomes, never one: `reconstructed` (every stored blob came
    back and the manifest matches), `partial` (some content was never stored,
    which the record already said), and `broken` (the record claims content the
    store cannot produce, or the manifest disagrees with its own files)."""
    ws = (rec.get("state") or {}).get("workspace") or {}
    repos = ws.get("repos") or {}
    want = missing = recovered = not_stored = 0
    mismatched: list[str] = []
    for repo, snap in repos.items():
        for rel, e in (snap.get("files") or {}).items():
            if e.get("deleted"):
                continue
            want += 1
            if not e.get("blob"):
                not_stored += 1
                continue
            data = cap.get_blob(home, e["sha256"])
            if data is None:
                missing += 1
                continue
            import hashlib
            if hashlib.sha256(data).hexdigest() != e["sha256"]:
                mismatched.append(f"{repo}/{rel}")
                continue
            recovered += 1
    # the manifest the record claims must be the manifest its own files produce
    claimed = ws.get("combined_sha256")
    import hashlib as _h
    body = "\n".join(f"{k} {v.get('manifest_sha256','')}" for k, v in sorted(repos.items()))
    recomputed = _h.sha256(body.encode("utf-8")).hexdigest() if repos else None
    manifest_ok = (claimed == recomputed) if repos else None
    if missing or mismatched or manifest_ok is False:
        outcome = "broken"
    elif not_stored or ws.get("truncated"):
        outcome = "partial"
    else:
        outcome = "reconstructed"
    return {"outcome": outcome, "files_claimed": want, "recovered": recovered,
            "not_stored_by_design": not_stored, "blobs_missing": missing,
            "hash_mismatched": mismatched, "manifest_agrees": manifest_ok,
            "state_truncated": bool(ws.get("truncated"))}


def rebuild(home: Path, ws: dict[str, Any], dest: Path) -> dict[str, Any]:
    """Materialise a captured state: the repository at its recorded head, with the
    stored dirty contents written over it.

    This is what separates **workspace reconstruction** from a **reproduced
    verification**. Reconstruction is a property of the record: did every file the
    record claims come back. Reproduction is a property of the world: does running
    the recorded command against that reconstruction give the recorded verdict.
    They can disagree in both directions, which is why they are two functions and
    two columns.
    """
    import subprocess
    out: dict[str, Any] = {"repos": {}, "restored": 0, "unrestorable": [],
                           "dest": str(dest)}
    for index, (repo, snap) in enumerate((ws.get("repos") or {}).items()):
        head = snap.get("head")
        name = Path(repo).name
        tree = dest / f"{index}-{name}"
        if not head or not Path(repo).is_dir():
            out["repos"][repo] = {"ok": False, "why": "no head, or the repository is gone"}
            continue
        r = subprocess.run(["git", "-C", repo, "worktree", "add", "--detach", "-q",
                            str(tree), head], capture_output=True, text=True)
        if r.returncode != 0:
            out["repos"][repo] = {"ok": False, "why": f"worktree add failed: {r.stderr.strip()[:160]}"}
            continue
        wrote, missed = 0, []
        for rel, e in (snap.get("files") or {}).items():
            target = tree / rel
            if e.get("deleted"):
                try:
                    target.unlink()
                    wrote += 1
                except Exception:
                    missed.append(rel)
                continue
            if not e.get("blob"):
                missed.append(rel)          # never stored, by a recorded bound
                continue
            data = cap.get_blob(home, e["sha256"])
            if data is None:
                missed.append(rel)
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                wrote += 1
            except Exception:
                missed.append(rel)
        out["repos"][repo] = {"ok": True, "worktree": str(tree),
                              "restored": wrote, "unrestorable": missed}
        out["restored"] += wrote
        out["unrestorable"] += [f"{name}/{m}" for m in missed]
    return out


def rebuild_cleanup(rb: dict[str, Any]) -> None:
    import shutil
    import subprocess
    for repo, info in (rb.get("repos") or {}).items():
        wt = info.get("worktree")
        if wt:
            subprocess.run(["git", "-C", repo, "worktree", "remove", "--force", wt],
                           capture_output=True, text=True)
    try:
        shutil.rmtree(rb.get("dest") or "", ignore_errors=True)
    except Exception:
        pass
    for repo in (rb.get("repos") or {}):
        subprocess.run(["git", "-C", repo, "worktree", "prune"], capture_output=True, text=True)


def binding(thread: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """How each verification is bound to a state, and any before/after mismatch.

    The binding is the manifest hash the verify event recorded. The hook fires
    AFTER the tool call returns, so the snapshot is taken a moment after the tests
    ran and the binding can drift. This reports which recorded state it matches:

      `pre`  -- the verification ran against the state the task STARTED from,
               which usually means nothing had been edited yet;
      `post` -- the latest captured state;
      `intermediate` -- neither, i.e. the workspace moved between the
               verification and the next capture point. The verdict is still
               bound to a named state, and that state is not one we snapshotted
               independently, so it is reported as intermediate rather than
               silently matched to the nearest one.
    """
    states = [r for r in thread if ((r.get("state") or {}).get("workspace") or {}).get("repos")]
    def mani(r: dict[str, Any]) -> str:
        return str(((r.get("state") or {}).get("workspace") or {}).get("combined_sha256") or "")
    pre = mani(states[0]) if states else ""
    post = mani(states[-1]) if states else ""
    out = []
    for r in thread:
        v = r.get("verification")
        if not v:
            continue
        b = str(v.get("bound_to_manifest_sha256") or "")
        where = "pre" if b and b == pre else ("post" if b and b == post else
                                             ("intermediate" if b else "unbound"))
        out.append({"event_id": r.get("event_id"), "command": v.get("command"),
                    "recorded_verdict": v.get("verdict"), "exit_code": v.get("exit_code"),
                    "bound_to": b[:16], "bound_to_which_state": where,
                    "pre_manifest": pre[:16], "post_manifest": post[:16],
                    "before_after_mismatch": bool(b) and b != post,
                    "at": v.get("at")})
    return out


def reproduce_verification(home: Path, thread: list[dict[str, Any]],
                           *, timeout_s: float = 300.0) -> list[dict[str, Any]]:
    """Re-run each recorded verification against the reconstructed state.

    Separate from reconstruction on purpose. Outcomes:
      `reproduced`      -- the rebuild ran and gave the recorded verdict
      `not_reproduced`  -- the rebuild ran and gave a different verdict
      `not_attempted`   -- no verification, or the state could not be rebuilt
      `instrument_error`-- the command could not run at all (exit 2-5, timeout)
    """
    import subprocess
    import tempfile
    out = []
    states = [r for r in thread if ((r.get("state") or {}).get("workspace") or {}).get("repos")]
    for r in thread:
        v = r.get("verification")
        if not v:
            continue
        row = {"event_id": r.get("event_id"), "command": v.get("command"),
               "recorded_verdict": v.get("verdict")}
        bound = v.get("bound_to_manifest_sha256")
        state = next((s for s in states if bound and
                      s["state"]["workspace"].get("combined_sha256") == bound), None)
        ws = state["state"]["workspace"] if state else None
        if ws is None or replay(home, state)["outcome"] != "reconstructed":
            row.update(outcome="not_attempted", why="the bound state is absent, partial or broken")
            out.append(row)
            continue
        dest = Path(tempfile.mkdtemp(prefix="capreplay-"))
        rb = rebuild(home, ws, dest)
        row["reconstruction"] = {"restored": rb["restored"],
                                 "unrestorable": rb["unrestorable"][:8],
                                 "repos_ok": sum(1 for i in rb["repos"].values() if i.get("ok"))}
        try:
            if (row["reconstruction"]["repos_ok"] != len(ws.get("repos") or {})
                    or rb["unrestorable"]):
                row.update(outcome="not_attempted", why="the bound workspace could not be rebuilt completely")
            else:
                # Resolve the recorded cwd within its repository, not whichever
                # repository happened to be first in the snapshot.
                recorded_cwd = Path(str(v.get("cwd") or "")).resolve()
                candidates = []
                for repo, info in rb["repos"].items():
                    try:
                        rel = recorded_cwd.relative_to(Path(repo).resolve())
                    except ValueError:
                        continue
                    candidates.append((len(Path(repo).parts), Path(info["worktree"]) / rel))
                if not candidates or not v.get("command"):
                    row.update(outcome="not_attempted", why="the recorded command or directory is not reconstructable")
                    out.append(row)
                    continue
                tree = max(candidates, key=lambda item: item[0])[1]
                pr = subprocess.run(["bash", "-lc", str(v["command"])],
                                    cwd=str(tree), capture_output=True, text=True,
                                    timeout=timeout_s)
                got = {0: "passed", 1: "failed"}.get(pr.returncode)
                row["replay_exit_code"] = pr.returncode
                row["replay_verdict"] = got
                row["replay_tail"] = (pr.stdout + pr.stderr)[-500:]
                if got is None:
                    row["outcome"] = "instrument_error"
                elif got == v.get("verdict"):
                    row["outcome"] = "reproduced"
                else:
                    row["outcome"] = "not_reproduced"
        except subprocess.TimeoutExpired:
            row.update(outcome="instrument_error", why=f"timed out after {timeout_s}s")
        except Exception as e:  # noqa: BLE001
            row.update(outcome="instrument_error", why=f"{type(e).__name__}: {e}")
        finally:
            rebuild_cleanup(rb)
        out.append(row)
    return out


def costs(rows: list[dict[str, Any]], home: Path) -> dict[str, Any]:
    """Snapshot latency and storage, beside completeness, over ALL records.

    Partial records stay in the denominator: a capture that is fast because it
    gave up is not a fast capture, and excluding its records would make the
    latency look better the worse the completeness got.
    """
    walls = []
    files = []
    for r in rows:
        ws = (r.get("state") or {}).get("workspace") or {}
        if ws.get("wall_s") is not None:
            walls.append(float(ws["wall_s"]))
        files.append(int(ws.get("file_count") or 0))
    walls.sort()
    def pct(q: float) -> float:
        return round(walls[min(len(walls) - 1, int(q * len(walls)))], 3) if walls else 0.0
    nblob = nbytes = 0
    bd = home / cap.BLOBS_DIR
    if bd.is_dir():
        for f in bd.rglob("*"):
            if f.is_file():
                nblob += 1
                try:
                    nbytes += f.stat().st_size
                except Exception:
                    pass
    cap_path = home / cap.CAPTURE2_FILE
    return {"snapshot_latency_s": {"n": len(walls), "median": pct(0.5),
                                   "p95": pct(0.95), "max": round(walls[-1], 3) if walls else 0.0},
            "files_per_snapshot": {"median": (sorted(files)[len(files) // 2] if files else 0),
                                   "max": (max(files) if files else 0)},
            "blob_store": {"objects": nblob, "bytes": nbytes},
            "capture_file_bytes": cap_path.stat().st_size if cap_path.is_file() else 0,
            "denominator_note": "every record, partial ones included"}




def report(home: Path, path: Path | None = None) -> dict[str, Any]:
    p = path or (home / cap.CAPTURE2_FILE)
    rows = load(p)
    th = threads(rows)
    reps = [replay(home, r) for r in rows]
    by_outcome: dict[str, int] = {}
    for r in reps:
        by_outcome[r["outcome"]] = by_outcome.get(r["outcome"], 0) + 1
    binds = {tid: binding(recs) for tid, recs in th.items() if tid}
    binds = {k: v for k, v in binds.items() if v}
    uncertain = [r.get("event_id") for r in rows
                 if (r.get("provenance") or {}).get("ownership_uncertain")]
    return {"path": str(p),
            "completeness": completeness_table(rows),
            "costs": costs(rows, home),
            "workspace_reconstruction": {"by_outcome": by_outcome, "records": len(reps)},
            "verification_binding": binds,
            "attribution": {
                "established": sum(1 for r in rows
                                   if (r.get("provenance") or {}).get("attribution") == cap.ATTR_SELF),
                "association_heuristic": sum(1 for r in rows
                                             if (r.get("provenance") or {}).get("attribution") == cap.ATTR_HEURISTIC),
                "ownership_uncertain": uncertain},
            "threads": {"count": len(th), "unlinked_records": len(th.get("", []))}}



def render(rep: dict[str, Any]) -> str:
    L: list[str] = []
    c = rep["completeness"]
    L.append(f"# capture v2 report — {rep['path']}\n")
    L.append(f"## 1. Capture completeness ({c['records']} records)\n")
    L.append("| field | present | share |")
    L.append("|---|---:|---:|")
    for f in FIELDS:
        L.append(f"| {f} | {c['present'][f]} | {c['share'][f]:.0%} |")
    L.append("")
    co = rep["costs"]
    L.append("### latency and storage, over every record including partials\n")
    L.append(f"- snapshot wall: median **{co['snapshot_latency_s']['median']}s**, "
             f"p95 **{co['snapshot_latency_s']['p95']}s**, max "
             f"{co['snapshot_latency_s']['max']}s (n={co['snapshot_latency_s']['n']})")
    L.append(f"- files per snapshot: median {co['files_per_snapshot']['median']}, "
             f"max {co['files_per_snapshot']['max']}")
    L.append(f"- blob store: {co['blob_store']['objects']} objects, "
             f"{co['blob_store']['bytes']:,} bytes; capture file "
             f"{co['capture_file_bytes']:,} bytes\n")
    L.append(f"## 2a. Workspace reconstruction "
             f"({rep['workspace_reconstruction']['records']} records)\n")
    L.append("A property of the record plus the blob store: did every file the "
             "record claims come back.\n")
    L.append("| outcome | records |")
    L.append("|---|---:|")
    for k, v in sorted(rep["workspace_reconstruction"]["by_outcome"].items()):
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("## 2b. Verification binding, and any before/after mismatch\n")
    L.append("A verification's binding is the manifest hash recorded with it. The "
             "hook fires after the tool call returns, so the binding can drift "
             "from the next capture point; `intermediate` means it matched "
             "neither the pre- nor the post-state and is reported as such rather "
             "than matched to the nearest one.\n")
    if not rep["verification_binding"]:
        L.append("_no verification recorded yet_\n")
    else:
        L.append("| task | command | recorded verdict | bound to | which state | before/after mismatch |")
        L.append("|---|---|---|---|---|---|")
        for tid, binds in rep["verification_binding"].items():
            for b in binds:
                L.append(f"| `{tid[-8:]}` | `{str(b['command'])[:34]}` | "
                         f"{b['recorded_verdict']} | `{b['bound_to']}` | "
                         f"**{b['bound_to_which_state']}** | {b['before_after_mismatch']} |")
        L.append("")
    at = rep["attribution"]
    L.append("## 2c. Attribution\n")
    L.append(f"- established (`self`): **{at['established']}**")
    L.append(f"- association heuristic (inherited): **{at['association_heuristic']}**")
    L.append(f"- ownership **uncertain** (a named job was first seen under a "
             f"different thread): **{len(at['ownership_uncertain'])}**"
             + (f" — {', '.join(e[-12:] for e in at['ownership_uncertain'][:6])}"
                if at["ownership_uncertain"] else ""))
    L.append("")
    L.append("An inherited link is an association, not established ownership. "
             "Where a notification's job was first seen under another thread both "
             "candidates are retained and nothing is assigned.\n")
    L.append(f"threads: {rep['threads']['count']}, "
             f"unlinked records: {rep['threads']['unlinked_records']}\n")
    return "\n".join(L)
