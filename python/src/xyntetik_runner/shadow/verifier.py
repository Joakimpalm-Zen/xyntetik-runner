"""The protected verifier: frozen tests the attempt cannot reach.

The tests that decide an episode live in a directory OUTSIDE the workspace,
frozen by a manifest of hashes. Verification copies the workspace to a
scratch directory, writes the frozen tests over whatever the workspace has
at those paths, runs pytest with the verifier's own configuration file so
the workspace's cannot change collection, and reads a JUnit report.

What cannot count as success, each with its own reason string:

* a no-op (the workspace is byte-identical to its baseline);
* a changed protected test file or a changed pytest configuration file
  (``conftest.py``, ``pytest.ini``, ``pyproject.toml``, ``setup.cfg``,
  ``tox.ini``): the run still happens, the result is a tamper finding;
* a skipped or missing expected test;
* a timeout: the verdict is ``None`` (inconclusive), never a pass.

Calibration is the gate on the gate: the frozen tests must fail on the
untouched baseline. A protected set that passes on the baseline cannot
reject a no-op, and ``calibrate`` refuses it rather than trusting it.

Not a sandbox. The scratch copy contains only regular files (no symlinks),
the environment handed to pytest is minimal, and the process group is
killed on timeout, but the tests run as the calling user with the network
reachable. OS isolation is R14.3.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from xyntetik_runner.shadow.baseline import Baseline, file_sha256, tree_hashes, tree_sha256
from xyntetik_runner.shadow.evidence import VerifierOutcome

MANIFEST = "manifest.json"
MANIFEST_SCHEMA = "xyntetik.shadow.protected.v1"
CONFIG_BASENAMES = frozenset({"conftest.py", "pytest.ini", ".pytest.ini", "pyproject.toml",
                              "setup.cfg", "tox.ini"})
VERIFIER_INI = "[pytest]\naddopts =\n"
VERIFIER_INI_NAME = ".xyntetik-shadow-verifier.ini"
_ENV_PASSTHROUGH = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "SystemRoot", "TEMP", "TMP",
                    "PYTHONIOENCODING")


class InstrumentError(RuntimeError):
    """The verifier itself is wrong, and no verdict from it may be used."""


def _classname(rel: str) -> str:
    return rel[:-3].replace("/", ".") if rel.endswith(".py") else rel.replace("/", ".")


@dataclass(frozen=True)
class ProtectedTests:
    """Frozen test files at their workspace-relative paths, plus the tests
    each is expected to contain."""

    source: Path
    verifier_id: str
    files: Mapping[str, str]
    expected: tuple[tuple[str, str], ...]

    @classmethod
    def load(cls, source: Path) -> ProtectedTests:
        try:
            m = json.loads((source / MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise InstrumentError(f"protected manifest unreadable: {source / MANIFEST}: {e}") from e
        if m.get("schema_version") != MANIFEST_SCHEMA:
            raise InstrumentError(f"protected manifest schema {m.get('schema_version')!r} unknown")
        files: dict[str, str] = dict(m["files"])
        for rel, sha in files.items():
            p = source / rel
            if not p.is_file():
                raise InstrumentError(f"protected file missing: {rel}")
            if file_sha256(p) != sha:
                raise InstrumentError(f"protected file differs from its manifest: {rel}")
        expected = tuple((rel, name) for rel, names in m["expected"].items() for name in names)
        if not expected:
            raise InstrumentError("protected manifest lists no expected tests")
        for rel, _ in expected:
            if rel not in files:
                raise InstrumentError(f"expected tests in a file the manifest does not freeze: {rel}")
        vid = str(m.get("verifier_id") or f"protected:{tree_sha256(files)[:16]}")
        return cls(source=source, verifier_id=vid, files=files, expected=expected)

    @classmethod
    def freeze(cls, source: Path, expected: Mapping[str, Sequence[str]], *,
               verifier_id: str | None = None) -> ProtectedTests:
        """Write the manifest for every file under ``source`` and load it."""
        files = {p.relative_to(source).as_posix(): file_sha256(p)
                 for p in sorted(source.rglob("*")) if p.is_file() and p.name != MANIFEST}
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "verifier_id": verifier_id or f"protected:{tree_sha256(files)[:16]}",
            "files": files,
            "expected": {rel: list(names) for rel, names in expected.items()},
        }
        (source / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")
        return cls.load(source)

    @property
    def test_files(self) -> tuple[str, ...]:
        return tuple(sorted({rel for rel, _ in self.expected}))


@dataclass(frozen=True)
class _Run:
    outcomes: Mapping[tuple[str, str], str]
    report_sha256: str
    duration_s: float
    timed_out: bool
    returncode: int


def _copy_tree(src: Path, dst: Path) -> None:
    for rel in tree_hashes(src):
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, target)


def _kill(proc: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except OSError:
        pass


def _run_protected(protected: ProtectedTests, tree: Path, *, timeout_s: float,
                   python: str, pythonpath: Sequence[str] = ()) -> _Run:
    scratch = Path(tempfile.mkdtemp(prefix="xyntetik-shadow-"))
    try:
        ws = scratch / "ws"
        ws.mkdir()
        _copy_tree(tree, ws)
        for rel in protected.files:
            target = ws / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(protected.source / rel, target)
        # The verifier's own ini sits at the workspace root: pytest takes the
        # rootdir from the -c file's directory, and with the ini elsewhere the
        # JUnit classnames come out empty and every expected test reads as
        # missing. Being named on -c also makes it the only ini pytest reads,
        # so a pytest.ini or pyproject the attempt wrote cannot change
        # collection (the change is still flagged as tamper).
        ini = ws / VERIFIER_INI_NAME
        if ini.exists():
            raise InstrumentError(f"workspace already contains {VERIFIER_INI_NAME}")
        ini.write_text(VERIFIER_INI, encoding="utf-8")
        report = scratch / "report.xml"
        cmd = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-c", VERIFIER_INI_NAME,
               "-o", "junit_family=xunit2", "--junit-xml", str(report), *protected.test_files]
        env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
        # Import roots are workspace-relative and resolved inside the scratch
        # copy, so a test can never import the original tree by accident.
        roots = [str(ws)] + [str(ws / rel) for rel in pythonpath]
        env.update({"HOME": str(scratch), "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": os.pathsep.join(roots)})
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, cwd=ws, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                start_new_session=(os.name == "posix"))
        timed_out = False
        try:
            out, _ = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill(proc)
            out, _ = proc.communicate()
        duration = time.monotonic() - t0
        outcomes: dict[tuple[str, str], str] = {}
        digest = hashlib.sha256(out or b"").hexdigest()
        if report.is_file():
            data = report.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            for case in ET.fromstring(data).iter("testcase"):
                key = (case.get("classname", ""), case.get("name", ""))
                if case.find("failure") is not None or case.find("error") is not None:
                    outcomes[key] = "failed"
                elif case.find("skipped") is not None:
                    outcomes[key] = "skipped"
                else:
                    outcomes[key] = "passed"
        return _Run(outcomes=outcomes, report_sha256=digest, duration_s=duration,
                    timed_out=timed_out, returncode=proc.returncode)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _key(rel: str, name: str) -> tuple[str, str]:
    """JUnit keys a test by (classname, name): the module dotted path, plus
    the class chain for methods. An expected id ``Class::test`` therefore
    joins its class onto the module classname."""
    if "::" in name:
        cls, _, leaf = name.rpartition("::")
        return (_classname(rel) + "." + cls.replace("::", "."), leaf)
    return (_classname(rel), name)


def _judge(protected: ProtectedTests, run: _Run) -> tuple[int, int, int, int]:
    passed = failed = skipped = missing = 0
    for rel, name in protected.expected:
        state = run.outcomes.get(_key(rel, name))
        if state == "passed":
            passed += 1
        elif state == "failed":
            failed += 1
        elif state == "skipped":
            skipped += 1
        else:
            missing += 1
    return passed, failed, skipped, missing


@dataclass(frozen=True)
class Calibration:
    failing: int
    passing: int
    missing: int
    report_sha256: str


def calibrate(protected: ProtectedTests, baseline_root: Path, *, timeout_s: float = 600.0,
              python: str = sys.executable, pythonpath: Sequence[str] = ()) -> Calibration:
    """Prove the instrument can fail: the frozen tests must not pass on the
    untouched baseline. Raises ``InstrumentError`` otherwise. ``pythonpath``
    lists workspace-relative import roots (``python/src``, ``packages/x/src``)."""
    run = _run_protected(protected, baseline_root, timeout_s=timeout_s, python=python,
                         pythonpath=pythonpath)
    if run.timed_out:
        raise InstrumentError(f"protected tests timed out on the baseline after {timeout_s}s")
    passed, failed, skipped, missing = _judge(protected, run)
    if failed == 0 and missing == 0 and skipped == 0:
        raise InstrumentError("protected tests pass on the untouched baseline; the instrument "
                              "cannot reject a no-op and its verdicts are worthless")
    if skipped:
        raise InstrumentError(f"{skipped} protected test(s) skip on the baseline; a skip is not "
                              "a failure the fix can turn into a pass")
    if run.returncode == 0:
        raise InstrumentError("pytest exited 0 on the baseline while the report shows failures; "
                              "the report and the process disagree")
    return Calibration(failing=failed, passing=passed, missing=missing,
                       report_sha256=run.report_sha256)


def check(tree: Path, protected: ProtectedTests, *, timeout_s: float = 600.0,
          python: str = sys.executable, pythonpath: Sequence[str] = ()) -> VerifierOutcome:
    """The frozen tests on a tree with no baseline: no no-op or tamper logic.
    This is the admission check on a known solution, not a verdict on an
    attempt; ``verify`` is the verdict."""
    run = _run_protected(protected, tree, timeout_s=timeout_s, python=python,
                         pythonpath=pythonpath)
    passed, failed, skipped, missing = _judge(protected, run)
    if run.timed_out:
        return VerifierOutcome(verifier_id=protected.verifier_id, passed=None,
                               expected=len(protected.expected), passed_count=passed,
                               failed=failed, skipped=skipped, missing=missing,
                               reasons=(f"timeout after {timeout_s:g}s",),
                               output_sha256=run.report_sha256,
                               duration_s=round(run.duration_s, 3))
    ok = failed == 0 and skipped == 0 and missing == 0
    reasons = tuple(r for r, n in ((f"{failed} expected test(s) failed", failed),
                                   (f"{skipped} expected test(s) skipped", skipped),
                                   (f"{missing} expected test(s) not run", missing)) if n)
    return VerifierOutcome(verifier_id=protected.verifier_id, passed=ok,
                           expected=len(protected.expected), passed_count=passed, failed=failed,
                           skipped=skipped, missing=missing, reasons=reasons,
                           output_sha256=run.report_sha256, duration_s=round(run.duration_s, 3))


def verify(workspace: Path, protected: ProtectedTests, baseline: Baseline, *,
           timeout_s: float = 600.0, python: str = sys.executable,
           pythonpath: Sequence[str] = ()) -> VerifierOutcome:
    """Verdict on the workspace as it stands, against its baseline.
    ``pythonpath`` lists workspace-relative import roots for the tests."""
    changes = baseline.changes(workspace)
    n_expected = len(protected.expected)
    if not changes:
        return VerifierOutcome(verifier_id=protected.verifier_id, passed=False,
                               expected=n_expected, passed_count=0, failed=0, skipped=0,
                               missing=n_expected,
                               reasons=("no_op: workspace identical to its baseline",))
    tamper: list[str] = []
    for rel in changes.paths:
        if rel in protected.files:
            tamper.append(f"protected test file changed: {rel}")
        elif Path(rel).name in CONFIG_BASENAMES:
            tamper.append(f"test configuration changed: {rel}")
    run = _run_protected(protected, workspace, timeout_s=timeout_s, python=python,
                         pythonpath=pythonpath)
    passed, failed, skipped, missing = _judge(protected, run)
    reasons: list[str] = []
    verdict: bool | None
    if run.timed_out:
        verdict = None
        reasons.append(f"timeout after {timeout_s:g}s")
    else:
        verdict = failed == 0 and skipped == 0 and missing == 0 and not tamper
        if failed:
            reasons.append(f"{failed} expected test(s) failed")
        if skipped:
            reasons.append(f"{skipped} expected test(s) skipped")
        if missing:
            reasons.append(f"{missing} expected test(s) not run")
        if tamper:
            reasons.append("tamper")
    return VerifierOutcome(verifier_id=protected.verifier_id, passed=verdict,
                           expected=n_expected, passed_count=passed, failed=failed,
                           skipped=skipped, missing=missing, tamper=tuple(tamper),
                           reasons=tuple(reasons), output_sha256=run.report_sha256,
                           duration_s=round(run.duration_s, 3))
