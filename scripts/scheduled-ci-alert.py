#!/usr/bin/env python3
"""Report a scheduled CI run that ended red as one labelled issue (R6.45.2).

A scheduled run sits outside pr-gate, so a failure in it reached nobody: the
daily macOS suite went red on 2026-11-10 at 12:51Z (test-mvcanon) and was
noticed four hours later. .github/workflows/scheduled-alert.yml runs this
script when a scheduled run of a watched workflow completes.

- **Red** (failure, cancelled, timed_out, startup_failure): the workflow's one
  open issue labelled LABEL is commented on, or opened when there is none. The
  comment names each failed job, its failed step and the step's first failing
  log line.
- **Green** (success): that issue, if open, is closed with a comment.
- **Anything else** (skipped, neutral, still running): nothing.

One issue per workflow, titled "Scheduled run red: <workflow>", so a green run
of one workflow never closes another's. A second red day comments on the same
issue instead of opening a new one: the public tracker carries one line of
news per red streak, not one per day (competitor-freshness.yml says why that
matters).

    scheduled-ci-alert.py --repo OWNER/NAME --run-id ID [--label L] [--dry-run]
                          [--any-event]

GH_TOKEN authorizes the API. It needs actions: read (the run, its jobs and their
logs) and issues: write. --dry-run reads everything and prints the writes it
would make. --any-event acts on a run that was not scheduled, for testing the
workflow by dispatch against a known run.
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"
LABEL = "scheduled-ci-red"
RED = ("failure", "cancelled", "timed_out", "startup_failure")
TITLE = "Scheduled run red: {workflow}"

# A GitHub log line: an ISO timestamp, a space, the text. Group markers and
# the runner's own closing error frame the failure; they are not it.
STAMP = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z ")
FRAME = re.compile(r"^##\[(group|endgroup|error|warning|notice|debug)\]|^Process completed")
# What a failure looks like in this repository's jobs, strongest first. A
# bare "error:" is not on the list: the gate targets print the refusals their
# negative cases expect ("error: GGUF metadata key contains an embedded NUL").
STRONG = re.compile(
    r"^FAILED |^ERROR |^FAIL\b|\bFAIL:|^E   |AssertionError|Segmentation fault|"
    r"^\S+:\d+(:\d+)?: (fatal )?error: |^=+ .*\b\d+ (failed|errors?)\b"
)
MAKE = re.compile(r"make(\[\d+\])?: \*\*\*")


def step_lines(log, started=None, completed=None):
    """The log's lines, timestamps removed, that fall inside one step: the API
    gives each step's started_at and completed_at to the second, and every log
    line carries its own timestamp. Without the times, every line."""
    out = []
    for raw in log.splitlines():
        m = STAMP.match(raw)
        text = raw[m.end():] if m else raw
        text = text.rstrip("\r\n")
        if started and m:
            t = m.group(1)
            if t < started.rstrip("Z") or (completed and t > completed.rstrip("Z")):
                continue
        out.append(text)
    return out


def first_failing_line(lines):
    """The first line that reads as a failure by STRONG, else make's first
    `***` line, else the last line of output before the runner's closing
    `##[error]`, else None."""
    body = [x for x in lines if x.strip() and not FRAME.match(x)]
    for pat in (STRONG, MAKE):
        for x in body:
            if pat.search(x):
                return x.strip()[:300]
    return body[-1].strip()[:300] if body else None


class Api:
    def __init__(self, repo, token, dry_run, out=sys.stdout):
        self.repo, self.token, self.dry_run, self.out = repo, token, dry_run, out

    def call(self, method, path, body=None, raw=False):
        url = path if path.startswith("http") else f"{API}/repos/{self.repo}{path}"
        req = urllib.request.Request(url, method=method,
                                     data=json.dumps(body).encode() if body is not None else None)
        req.add_header("Accept", "application/vnd.github+json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        if body is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        return data.decode("utf-8", "replace") if raw else json.loads(data or b"null")

    def write(self, method, path, body):
        if self.dry_run:
            print(f"dry run: {method} {path} {json.dumps(body)}", file=self.out)
            return {}
        return self.call(method, path, body)


def failed_jobs(api, run_id):
    jobs, page = [], 1
    while True:
        d = api.call("GET", f"/actions/runs/{run_id}/jobs?per_page=100&page={page}")
        jobs += d.get("jobs", [])
        if len(jobs) >= d.get("total_count", 0) or not d.get("jobs"):
            break
        page += 1
    out = []
    for j in jobs:
        if j.get("conclusion") not in RED:
            continue
        steps = [s for s in j.get("steps") or [] if s.get("conclusion") in RED]
        step = steps[0] if steps else None
        line = None
        if j.get("conclusion") == "failure":
            try:
                log = api.call("GET", f"/actions/jobs/{j['id']}/logs", raw=True)
                line = first_failing_line(step_lines(log, step and step.get("started_at"),
                                                     step and step.get("completed_at")))
            except (urllib.error.URLError, OSError) as e:
                line = f"(the log could not be read: {e})"
        out.append({"name": j["name"], "conclusion": j["conclusion"], "url": j.get("html_url"),
                    "step": step["name"] if step else None, "line": line})
    return out


def report(run, jobs):
    lines = [f"[{run['name']} run {run['id']}]({run['html_url']}) ended **{run['conclusion']}** "
             f"(scheduled, created {run['created_at']}, commit {run['head_sha'][:12]})."]
    if not jobs:
        lines.append("\nNo job ended red: the run itself was cancelled or failed to start.")
    for j in jobs:
        where = f" at step \"{j['step']}\"" if j["step"] else ""
        lines.append(f"\n- [{j['name']}]({j['url']}): {j['conclusion']}{where}")
        if j["line"]:
            lines.append(f"  - first failing line: `{j['line'].replace('`', chr(39))}`")
    return "\n".join(lines)


def find_issue(api, label, title):
    issues = api.call("GET", f"/issues?state=open&labels={label}&per_page=100")
    for i in issues:
        if i.get("title") == title and "pull_request" not in i:
            return i
    return None


def main(argv=None, api=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--any-event", action="store_true")
    a = ap.parse_args(argv)
    api = api or Api(a.repo, os.environ.get("GH_TOKEN"), a.dry_run)
    run = api.call("GET", f"/actions/runs/{a.run_id}")
    if run.get("event") != "schedule" and not a.any_event:
        print(f"run {a.run_id} was started by {run.get('event')!r}, not the schedule: nothing to do")
        return 0
    conclusion = run.get("conclusion")
    would = "dry run: would have " if a.dry_run else ""
    title = TITLE.format(workflow=run["name"])
    issue = find_issue(api, a.label, title)
    if conclusion in RED:
        body = report(run, failed_jobs(api, a.run_id))
        print(body)
        if issue:
            api.write("POST", f"/issues/{issue['number']}/comments", {"body": body})
            print(f"{would}commented on #{issue['number']}")
        else:
            try:
                api.call("GET", f"/labels/{a.label}")
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                api.write("POST", "/labels", {"name": a.label, "color": "b60205",
                                              "description": "A scheduled CI run ended red"})
            api.write("POST", "/issues", {"title": title, "body": body, "labels": [a.label]})
            print(f"{would}opened an issue")
    elif conclusion == "success":
        if issue:
            api.write("POST", f"/issues/{issue['number']}/comments",
                      {"body": f"[{run['name']} run {run['id']}]({run['html_url']}) is green again."})
            api.write("PATCH", f"/issues/{issue['number']}", {"state": "closed"})
            print(f"{would}closed #{issue['number']}")
        else:
            print("green, and no open issue")
    else:
        print(f"conclusion {conclusion!r}: nothing to do")
    return 0


if __name__ == "__main__":
    sys.exit(main())
