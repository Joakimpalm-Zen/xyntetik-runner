"""Keep every sanitizer report a test's child processes write, and fail on any.

Loaded with `-p tests.sanitizer_reports` against a sanitized `./runner` (the
`make debug` build copied over it; `make test-python-sanitized` does that).
Most tests read a child's stderr for their own purposes and never look for a
report in it, and a server's report lands in a log the harness may delete, so
a leak or a memory error can pass a green suite unseen (R6.68.20: the muse
g.doc leak). This plugin moves the reports out of stderr into files:

- every child that inherits the run's ASAN_OPTIONS / UBSAN_OPTIONS gets
  `log_path=<dir>/<test id>` appended, so the file names the test that
  started the process (a module-scoped server is named after the first test
  that used it);
- a child whose env sets its own value for one of them keeps it untouched:
  a test that configures a sanitizer (a negative control) owns its output;
- LSAN_OPTIONS is passed through as the run set it (the suppressions file
  for the driver's process-lifetime allocations, tests/lsan.supp).

At the end of the session every report file is listed with its first
SUMMARY line and the run fails, even when every test passed.

The directory is SANITIZER_REPORTS_DIR, default .build/ci/sanitizer-reports;
it is emptied at the start so a report always belongs to this run.
"""
import os
import pathlib
import re
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
_DIR = pathlib.Path(os.environ.get("SANITIZER_REPORTS_DIR",
                                   ROOT / ".build" / "ci" / "sanitizer-reports"))
_KEYS = ("ASAN_OPTIONS", "UBSAN_OPTIONS")
_base = {}
_orig_init = subprocess.Popen.__init__


def _without_log_path(opts):
    return ":".join(o for o in opts.split(":") if o and not o.startswith("log_path="))


def _test_name():
    t = os.environ.get("PYTEST_CURRENT_TEST", "collection").split(" (")[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", t)[:180]


def _init(self, *args, **kwargs):
    env = kwargs.get("env")
    env = dict(os.environ if env is None else env)
    stem = _DIR / _test_name()
    for key in _KEYS:
        inherited = os.environ.get(key, "")
        if env.get(key, inherited) != inherited:
            continue  # the test set its own; leave it alone
        tag = key.split("_")[0].lower()
        env[key] = ":".join(filter(None, (_base[key], f"log_path={stem}.{tag}")))
    kwargs["env"] = env
    _orig_init(self, *args, **kwargs)


def _is_worker(config):
    # under pytest-xdist the controller owns the directory and the verdict;
    # a worker clearing it would delete another worker's reports
    return hasattr(config, "workerinput")


def pytest_configure(config):
    if not _is_worker(config):
        shutil.rmtree(_DIR, ignore_errors=True)
    _DIR.mkdir(parents=True, exist_ok=True)
    for key in _KEYS:
        _base[key] = _without_log_path(os.environ.get(key, ""))
    subprocess.Popen.__init__ = _init


def _summary(path):
    try:
        text = path.read_text(errors="replace")
    except OSError as e:
        return f"(unreadable: {e})"
    for line in text.splitlines():
        if line.startswith("SUMMARY:"):
            return line
    for line in text.splitlines():
        if "runtime error:" in line or "ERROR:" in line:
            return line.strip()
    return "(no SUMMARY line)"


def pytest_sessionfinish(session, exitstatus):
    subprocess.Popen.__init__ = _orig_init
    if _is_worker(session.config):
        return
    reports = sorted(p for p in _DIR.iterdir() if p.is_file())
    if not reports:
        return
    tr = session.config.pluginmanager.get_plugin("terminalreporter")
    lines = [f"{len(reports)} sanitizer report file(s) in {_DIR}:"]
    lines += [f"  {p.name}: {_summary(p)}" for p in reports]
    if tr is not None:
        tr.ensure_newline()
        tr.section("sanitizer reports", sep="=", red=True)
        for line in lines:
            tr.write_line(line)
    else:
        print("\n".join(lines))
    session.exitstatus = 1
