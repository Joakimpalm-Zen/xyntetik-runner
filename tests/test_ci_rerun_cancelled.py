"""scripts/ci-rerun-cancelled.py: rerun once when a ci job cancels with zero steps (R6.102)."""
import importlib.util
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "ci-rerun-cancelled"

spec = importlib.util.spec_from_file_location("rerun_cancelled", ROOT / "scripts" / "ci-rerun-cancelled.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


class _FakeApi:
    def __init__(self, data):
        self.run = data["run"]
        self.jobs = data["jobs"]

    def call(self, method, path):
        assert method == "GET"
        rid = self.run["id"]
        if path == f"/actions/runs/{rid}":
            return self.run
        if path.startswith(f"/actions/runs/{rid}/jobs"):
            return self.jobs
        raise AssertionError(f"unexpected GET {path}")


def test_scheduled_zero_step_cancel_reruns_once(tmp_path):
    data = _load("run-44310758378-schedule-cancelled.json")
    cmds = []
    out = tmp_path / "out.txt"
    rc = mod.main(
        ["--repo", "o/r", "--run-id", str(data["run"]["id"]), "--github-output", str(out)],
        api=_FakeApi(data),
        runner=cmds.append,
    )
    assert rc == 0
    assert cmds == [["gh", "run", "rerun", "44310758378", "--repo", "o/r", "--failed"]]
    assert out.read_text(encoding="utf-8").strip() == "reran=true"


def test_pr_zero_step_cancel_reruns_once(tmp_path):
    data = _load("run-44346880375-a1-pr-cancelled.json")
    cmds = []
    out = tmp_path / "out.txt"
    rc = mod.main(
        ["--repo", "o/r", "--run-id", str(data["run"]["id"]), "--github-output", str(out)],
        api=_FakeApi(data),
        runner=cmds.append,
    )
    assert rc == 0
    assert cmds == [["gh", "run", "rerun", "44346880375", "--repo", "o/r", "--failed"]]
    assert out.read_text(encoding="utf-8").strip() == "reran=true"


def test_real_step_failure_is_not_rerun(tmp_path):
    data = _load("run-44339049414-pr-real-failure.json")
    cmds = []
    out = tmp_path / "out.txt"
    rc = mod.main(
        ["--repo", "o/r", "--run-id", str(data["run"]["id"]), "--github-output", str(out)],
        api=_FakeApi(data),
        runner=cmds.append,
    )
    assert rc == 0
    assert cmds == []
    assert out.read_text(encoding="utf-8").strip() == "reran=false"


def test_second_attempt_is_never_rerun(tmp_path):
    data = _load("run-44310758378-schedule-cancelled.json")
    data["run"]["run_attempt"] = 2
    cmds = []
    out = tmp_path / "out.txt"
    rc = mod.main(
        ["--repo", "o/r", "--run-id", str(data["run"]["id"]), "--github-output", str(out)],
        api=_FakeApi(data),
        runner=cmds.append,
    )
    assert rc == 0
    assert cmds == []
    assert out.read_text(encoding="utf-8").strip() == "reran=false"


def test_superseded_cancelled_pr_run_is_not_rerun():
    data = _load("run-44361399885-pr-superseded.json")
    rerun, reason, jobs = mod.decide(data["run"], data["jobs"]["jobs"])
    assert not rerun and "cancelled" in reason and jobs == []


def test_mixed_zero_step_cancel_and_step_failure_is_not_rerun():
    data = _load("run-44310758378-schedule-cancelled.json")
    for j in data["jobs"]["jobs"]:
        if j["name"] == "make-test":
            j["conclusion"] = "failure"
            break
    rerun, _, _ = mod.decide(data["run"], data["jobs"]["jobs"])
    assert not rerun


def test_dry_run_does_not_invoke_gh_but_gates_alert(tmp_path, capsys):
    data = _load("run-44310758378-schedule-cancelled.json")
    cmds = []
    out = tmp_path / "out.txt"
    rc = mod.main(
        ["--repo", "o/r", "--run-id", str(data["run"]["id"]), "--dry-run", "--github-output", str(out)],
        api=_FakeApi(data),
        runner=cmds.append,
    )
    assert rc == 0 and cmds == []
    assert "dry run: would run gh run rerun 44310758378 --repo o/r --failed" in capsys.readouterr().out
    assert out.read_text(encoding="utf-8").strip() == "reran=true"


def test_workflow_permissions_and_default_branch_checkout():
    yml = (ROOT / ".github" / "workflows" / "scheduled-alert.yml").read_text(encoding="utf-8")
    rerun_block, alert_block = yml.split("\n  alert:", 1)
    assert "actions: write" in rerun_block
    assert "actions: write" not in alert_block
    assert "issues: write" not in rerun_block
    assert "needs.rerun.outputs.reran != 'true'" in alert_block
    # workflow_run checks out default branch only (no ref: override in checkout)
    assert not re.search(r"^\s*ref\s*:", yml, re.M)
