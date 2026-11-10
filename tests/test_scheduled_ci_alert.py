"""scripts/scheduled-ci-alert.py: a scheduled CI run that ends red reaches one labelled issue (R6.45.2).

The script is driven through main() with a fake GitHub API. The run and jobs
answers are recorded from the API for the scheduled ci run of 2026-11-10 that
went red at 12:51Z (tests/fixtures/scheduled-ci-alert/run-42818173044.json).
That job's log needs a token to download, so its log is synthetic, in GitHub's
line format (synthetic-gates-core.log). It carries the gate target's expected
refusals beside the real failure, which is what the first-failing-line rule
has to tell apart.
"""
import importlib.util
import io
import json
import pathlib
import urllib.error

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "scheduled-ci-alert"
spec = importlib.util.spec_from_file_location("alert", ROOT / "scripts" / "scheduled-ci-alert.py")
alert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alert)

RECORDED = json.loads((FIX / "run-42818173044.json").read_text(encoding="utf-8"))
LOG = (FIX / "synthetic-gates-core.log").read_text(encoding="utf-8")
RUN_ID = str(RECORDED["run"]["id"])


class FakeApi(alert.Api):
    """Answers from recorded responses; records every write."""

    def __init__(self, run, jobs, issues=(), label=True, log=LOG, dry_run=False):
        super().__init__("o/r", "token", dry_run, out=io.StringIO())
        self.run, self.jobs, self.issues, self.label, self.log = run, jobs, list(issues), label, log
        self.writes = []

    def call(self, method, path, body=None, raw=False):
        if method != "GET":
            self.writes.append((method, path, body))
            return {}
        if path == f"/actions/runs/{self.run['id']}":
            return self.run
        if path.startswith(f"/actions/runs/{self.run['id']}/jobs"):
            return self.jobs
        if path.startswith("/actions/jobs/") and path.endswith("/logs"):
            if self.log is None:
                raise urllib.error.HTTPError(path, 403, "Forbidden", {}, None)
            return self.log
        if path.startswith("/issues?"):
            return self.issues
        if path.startswith("/labels/"):
            if not self.label:
                raise urllib.error.HTTPError(path, 404, "Not Found", {}, None)
            return {"name": path.rsplit("/", 1)[1]}
        raise AssertionError(f"unexpected GET {path}")


def run_with(conclusion="failure", event="schedule", name="ci"):
    r = dict(RECORDED["run"], conclusion=conclusion, event=event, name=name)
    return r


def main(api, *extra):
    return alert.main(["--repo", "o/r", "--run-id", str(api.run["id"]), *extra], api=api)


def test_a_red_run_opens_one_labelled_issue_naming_the_job_step_and_line():
    api = FakeApi(run_with(), RECORDED["jobs"], label=False)
    assert main(api) == 0
    methods = [(m, p) for m, p, _ in api.writes]
    assert methods == [("POST", "/labels"), ("POST", "/issues")]
    issue = api.writes[1][2]
    assert issue["title"] == "Scheduled run red: ci"
    assert issue["labels"] == ["scheduled-ci-red"]
    body = issue["body"]
    assert "macos-suite-gates-core" in body and 'step "make test-gates-core"' in body
    # the step's failure, not the expected refusal before it or the build
    # step's line outside the step's time window
    assert "first failing line: `FAIL: test-mvcanon q4_0 row 3" in body
    assert "embedded NUL" not in body and "test_old.c" not in body
    assert RECORDED["run"]["html_url"] in body


def test_a_second_red_run_comments_on_the_open_issue():
    open_issue = {"number": 7, "title": "Scheduled run red: ci", "labels": [{"name": "scheduled-ci-red"}]}
    api = FakeApi(run_with(), RECORDED["jobs"], issues=[open_issue])
    assert main(api) == 0
    assert [(m, p) for m, p, _ in api.writes] == [("POST", "/issues/7/comments")]


def test_a_green_run_closes_the_issue_and_only_its_own_workflows():
    issues = [{"number": 7, "title": "Scheduled run red: ci"},
              {"number": 8, "title": "Scheduled run red: competitor result freshness"}]
    api = FakeApi(run_with("success"), RECORDED["jobs"], issues=issues)
    assert main(api) == 0
    assert [(m, p) for m, p, _ in api.writes] == [("POST", "/issues/7/comments"),
                                                  ("PATCH", "/issues/7")]
    assert api.writes[1][2] == {"state": "closed"}


def test_a_green_run_with_no_open_issue_writes_nothing():
    api = FakeApi(run_with("success"), RECORDED["jobs"])
    assert main(api) == 0 and api.writes == []


@pytest.mark.parametrize("conclusion", ["cancelled", "timed_out", "startup_failure"])
def test_every_red_conclusion_reports(conclusion):
    jobs = {"total_count": 0, "jobs": []}
    api = FakeApi(run_with(conclusion), jobs)
    assert main(api) == 0
    assert [(m, p) for m, p, _ in api.writes] == [("POST", "/issues")]
    assert "No job ended red" in api.writes[0][2]["body"]


def test_a_run_not_started_by_the_schedule_is_left_alone_unless_asked():
    api = FakeApi(run_with(event="push"), RECORDED["jobs"])
    assert main(api) == 0 and api.writes == []
    api = FakeApi(run_with(event="push"), RECORDED["jobs"])
    assert main(api, "--any-event") == 0 and api.writes


def test_a_dry_run_reads_everything_and_writes_nothing():
    api = FakeApi(run_with(), RECORDED["jobs"], dry_run=True)
    calls = []
    real = api.call
    api.call = lambda m, p, b=None, raw=False: calls.append(m) or real(m, p, b, raw)
    assert main(api, "--dry-run") == 0
    assert set(calls) == {"GET"} and api.writes == []
    assert "dry run: POST /issues" in api.out.getvalue()


def test_an_unreadable_log_is_named_not_guessed():
    api = FakeApi(run_with(), RECORDED["jobs"], log=None)
    assert main(api) == 0
    assert "the log could not be read" in api.writes[-1][2]["body"]


@pytest.mark.parametrize("lines,want", [
    (["error: an expected refusal", "FAILED tests/test_x.py::test_y - assert 1 == 2"],
     "FAILED tests/test_x.py::test_y - assert 1 == 2"),
    (["src/model.c:40:5: error: unknown type name 'foo'", "make: *** [all] Error 1"],
     "src/model.c:40:5: error: unknown type name 'foo'"),
    (["building", "make: *** [Makefile:1: x] Error 2", "##[error]Process completed with exit code 2."],
     "make: *** [Makefile:1: x] Error 2"),
    (["the last words before the exit", "##[error]Process completed with exit code 1."],
     "the last words before the exit"),
    (["##[error]Process completed with exit code 1."], None),
])
def test_the_first_failing_line(lines, want):
    assert alert.first_failing_line(lines) == want


def test_step_lines_keep_the_step_window_only():
    got = alert.step_lines(LOG, "2026-11-10T12:49:13Z", "2026-11-10T12:51:26Z")
    assert got[0] == "##[group]Run make test-gates-core"
    assert got[-1] == "##[error]Process completed with exit code 2."
    assert not any("test_old.c" in x for x in got)
