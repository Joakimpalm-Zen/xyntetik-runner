#!/usr/bin/env python3
"""Read GitHub job/step timings through gh; do not change or dispatch any run."""
import argparse
import datetime as dt
import json
from pathlib import Path
import statistics
import subprocess


def api(path):
    return json.loads(subprocess.check_output(["gh", "api", path], text=True))

def seconds(a, b):
    return (dt.datetime.fromisoformat(b.replace("Z", "+00:00")) -
            dt.datetime.fromisoformat(a.replace("Z", "+00:00"))).total_seconds()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="Joakimpalm-Zen/xyntetik-runner")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default=".build/ci-history.json")
    args = ap.parse_args()
    if not 1 <= args.limit <= 100:
        ap.error("limit must be 1..100")
    root = f"repos/{args.repo}/actions"
    runs = api(f"{root}/workflows/ci.yml/runs?status=completed&per_page={args.limit}")["workflow_runs"]
    records = []
    buckets = {}
    for run in runs:
        page = 1
        while True:
            jobs = api(f"{root}/runs/{run['id']}/jobs?filter=all&per_page=100&page={page}")["jobs"]
            for job in jobs:
                if not job.get("started_at") or not job.get("completed_at"):
                    continue
                row = dict(run=run["id"], attempt=job.get("run_attempt"), event=run["event"],
                           sha=run["head_sha"], name=job["name"], conclusion=job["conclusion"],
                           labels=job.get("labels"), started_at=job["started_at"],
                           seconds=seconds(job["started_at"], job["completed_at"]), steps=job.get("steps", []))
                if run.get("run_started_at"):
                    row["run_queue_seconds"] = seconds(run["created_at"], run["run_started_at"])
                    row["job_start_delay_seconds"] = seconds(run["run_started_at"], job["started_at"])
                records.append(row)
                for step in row["steps"]:
                    if step.get("started_at") and step.get("completed_at"):
                        key = (run["event"], job["name"], step["name"])
                        buckets.setdefault(key, []).append(seconds(step["started_at"], step["completed_at"]))
            if len(jobs) < 100:
                break
            page += 1
    summary = []
    for key, values in buckets.items():
        values.sort()
        summary.append(dict(event=key[0], job=key[1], step=key[2], samples=len(values),
                            median=statistics.median(values), p95=values[max(0, (95 * len(values) + 99) // 100 - 1)]))
    summary.sort(key=lambda r: r["p95"], reverse=True)
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(jobs=records, summary=summary), indent=2) + "\n")
    for row in summary[:30]:
        print(f"{row['event']:15} {row['job']:24} {row['median']:7.1f}s p95 {row['p95']:7.1f}s {row['step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
