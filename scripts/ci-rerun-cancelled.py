#!/usr/bin/env python3
"""Rerun a ci workflow run once when job(s) ended cancelled with zero steps (R6.102).

Hosted macOS runners occasionally cancel a job after waiting 15 minutes for
capacity without running a single step. .github/workflows/scheduled-alert.yml's
`rerun` job runs this script on the default branch when a `ci` run completes:

- Only `ci` runs on `run_attempt == 1` whose run conclusion is not `cancelled`
  (a superseded PR push or manual cancel has `conclusion == "cancelled"`).
- At least one non-gate job must have ended `cancelled` with zero steps (`steps == []`).
- Any non-green job that ran any step (other than the `pr-gate` summary job)
  or failed independently blocks the rerun: a job that ran a step is never
  rerun by this mechanism.
- On a match, runs `gh run rerun <run_id> --repo <owner/name> --failed` (unless
  `--dry-run`) and writes `reran=true` to `GITHUB_OUTPUT`.
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

API = "https://api.github.com"
SKIP_OR_OK = ("success", "skipped")


class Api:
    def __init__(self, repo, token):
        self.repo, self.token = repo, token

    def call(self, method, path):
        url = path if path.startswith("http") else f"{API}/repos/{self.repo}{path}"
        req = urllib.request.Request(url, method=method)
        req.add_header("Accept", "application/vnd.github+json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"null")


def list_jobs(api, run_id):
    jobs, page = [], 1
    while True:
        d = api.call("GET", f"/actions/runs/{run_id}/jobs?per_page=100&page={page}")
        batch = d.get("jobs", [])
        jobs.extend(batch)
        if len(jobs) >= d.get("total_count", 0) or not batch:
            break
        page += 1
    return jobs


def decide(run, jobs, workflow="ci"):
    """Return (should_rerun, reason, cancelled_zero_job_names)."""
    if workflow and run.get("name") != workflow:
        return False, f"workflow {run.get('name')!r} is not {workflow!r}", []
    attempt = run.get("run_attempt") or 1
    if attempt != 1:
        return False, f"run_attempt is {attempt}, not 1: never rerun more than once", []
    if run.get("conclusion") == "cancelled":
        return False, "run itself was cancelled (superseded or cancelled manually)", []

    cancelled_zero = []
    blocked = []
    for j in jobs:
        conc = j.get("conclusion")
        name = j.get("name") or ""
        steps = j.get("steps") or []
        if conc in SKIP_OR_OK:
            continue
        if name == "pr-gate" and conc == "failure":
            continue
        if conc == "cancelled" and len(steps) == 0 and name != "pr-gate":
            cancelled_zero.append(name)
            continue
        blocked.append((name, conc, len(steps)))

    if not cancelled_zero:
        return False, "no job ended cancelled with zero steps executed", []
    if blocked:
        return False, f"other non-green jobs ran steps or failed: {blocked}", []
    return True, f"cancelled before first step: {', '.join(cancelled_zero)}", cancelled_zero


def write_output(path, key, value):
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def main(argv=None, api=None, runner=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workflow", default="ci")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT", ""))
    a = ap.parse_args(argv)

    api = api or Api(a.repo, os.environ.get("GH_TOKEN"))
    runner = runner or (lambda cmd: subprocess.run(cmd, check=True))
    run = api.call("GET", f"/actions/runs/{a.run_id}")
    jobs = list_jobs(api, a.run_id)
    rerun, reason, _ = decide(run, jobs, workflow=a.workflow)
    print(f"run {a.run_id} (event={run.get('event')}, attempt={run.get('run_attempt', 1)}): {reason}")
    if rerun:
        cmd = ["gh", "run", "rerun", str(a.run_id), "--repo", a.repo, "--failed"]
        if a.dry_run:
            print(f"dry run: would run {' '.join(cmd)}")
        else:
            runner(cmd)
        write_output(a.github_output, "reran", "true")
    else:
        write_output(a.github_output, "reran", "false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
