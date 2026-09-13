"""Frozen repository commands as calibrated verifiers.

The command and protected files are fixed outside the editable workspace. A
baseline must fail before a changed workspace can earn a passing verdict.
No-ops and changes to protected tests or configuration cannot pass. A command
returns one overall result; this interface does not invent per-test credit.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from xyntetik_runner.shadow.baseline import Baseline, file_sha256, tree_sha256
from xyntetik_runner.shadow.evidence import VerifierOutcome
from xyntetik_runner.shadow.verifier import (CONFIG_BASENAMES, InstrumentError, _copy_tree, _kill)

MANIFEST = "command-manifest.json"
MANIFEST_SCHEMA = "xyntetik.shadow.command.v1"
_ENV_PASSTHROUGH = ("PATH", "HOME", "SYSTEMROOT", "SystemRoot", "COMSPEC", "TMPDIR", "TEMP",
                    "LANG", "LC_ALL", "USERPROFILE")


@dataclass(frozen=True)
class FrozenCommand:
    """A repeatable check and the files it reads, frozen together."""

    source: Path
    verifier_id: str
    files: Mapping[str, str]
    argv: tuple[str, ...]
    cwd: str = "."
    timeout_s: float = 900.0

    @classmethod
    def load(cls, source: Path) -> FrozenCommand:
        try:
            m = json.loads((source / MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise InstrumentError(f"command manifest unreadable: {source / MANIFEST}: {e}") from e
        if m.get("schema_version") != MANIFEST_SCHEMA:
            raise InstrumentError(f"command manifest schema {m.get('schema_version')!r} unknown")
        files: dict[str, str] = dict(m.get("files") or {})
        for rel, sha in files.items():
            p = source / rel
            if not p.is_file():
                raise InstrumentError(f"frozen file missing: {rel}")
            if file_sha256(p) != sha:
                raise InstrumentError(f"frozen file differs from its manifest: {rel}")
        argv = tuple(str(a) for a in (m.get("argv") or ()))
        if not argv:
            raise InstrumentError("command manifest names no command")
        return cls(source=source, verifier_id=str(m.get("verifier_id") or "command:unknown"),
                   files=files, argv=argv, cwd=str(m.get("cwd") or "."),
                   timeout_s=float(m.get("timeout_s") or 900.0))

    @classmethod
    def freeze(cls, source: Path, argv: Sequence[str], *, cwd: str = ".",
               timeout_s: float = 900.0, verifier_id: str | None = None) -> FrozenCommand:
        files = {p.relative_to(source).as_posix(): file_sha256(p)
                 for p in sorted(source.rglob("*")) if p.is_file() and p.name != MANIFEST}
        ident = verifier_id or ("command:" + hashlib.sha256(
            ("\x00".join(argv) + "\x00" + tree_sha256(files)).encode()).hexdigest()[:16])
        (source / MANIFEST).write_text(json.dumps({
            "schema_version": MANIFEST_SCHEMA, "verifier_id": ident, "files": files,
            "argv": list(argv), "cwd": cwd, "timeout_s": timeout_s,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return cls.load(source)


@dataclass(frozen=True)
class _Run:
    returncode: int
    timed_out: bool
    output: str
    output_sha256: str
    duration_s: float


def _run(cmd: FrozenCommand, tree: Path) -> _Run:
    """Run the frozen check over a scratch copy, with the frozen files
    restored over whatever the workspace holds."""
    import os
    scratch = Path(tempfile.mkdtemp(prefix="xyntetik-shadow-cmd-"))
    try:
        ws = scratch / "ws"
        ws.mkdir()
        _copy_tree(tree, ws)
        for rel in cmd.files:
            target = ws / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cmd.source / rel, target)
        env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
        env.update({"HOME": str(scratch), "PYTHONDONTWRITEBYTECODE": "1", "CI": "1"})
        work = (ws / cmd.cwd).resolve()
        if ws.resolve() not in work.parents and work != ws.resolve():
            raise InstrumentError(f"command cwd escapes the workspace: {cmd.cwd}")
        t0 = time.monotonic()
        proc = subprocess.Popen(list(cmd.argv), cwd=str(work), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                start_new_session=(sys.platform != "win32"))
        timed_out = False
        try:
            out = proc.communicate(timeout=cmd.timeout_s)[0] or b""
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill(proc)
            out = proc.communicate()[0] or b""
        text = out.decode("utf-8", "replace")
        return _Run(returncode=proc.returncode if proc.returncode is not None else -1,
                    timed_out=timed_out, output=text[-8000:],
                    output_sha256=hashlib.sha256(out).hexdigest(),
                    duration_s=time.monotonic() - t0)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _outcome(cmd: FrozenCommand, run: _Run, *, tamper: Sequence[str] = (),
             no_op: bool = False) -> VerifierOutcome:
    """One bit, said plainly. `expected` is 1 because the check is the unit;
    a caller reading `passed_count` gets 1 or 0 and never a fabricated
    per-test tally."""
    if no_op:
        return VerifierOutcome(verifier_id=cmd.verifier_id, passed=False, expected=1,
                               passed_count=0, failed=0, skipped=0, missing=1,
                               reasons=("no_op: workspace identical to its baseline",))
    if run.timed_out:
        return VerifierOutcome(verifier_id=cmd.verifier_id, passed=None, expected=1,
                               passed_count=0, failed=0, skipped=0, missing=1,
                               reasons=(f"timeout after {cmd.timeout_s:g}s",),
                               output_sha256=run.output_sha256,
                               duration_s=round(run.duration_s, 3))
    ok = run.returncode == 0 and not tamper
    reasons: list[str] = []
    if run.returncode != 0:
        reasons.append(f"the frozen check exited {run.returncode}")
    if tamper:
        reasons.append("tamper")
    return VerifierOutcome(verifier_id=cmd.verifier_id, passed=ok, expected=1,
                           passed_count=1 if ok else 0, failed=0 if ok else 1,
                           skipped=0, missing=0, tamper=tuple(tamper), reasons=tuple(reasons),
                           output_sha256=run.output_sha256, duration_s=round(run.duration_s, 3))


def calibrate(cmd: FrozenCommand, baseline_root: Path) -> str:
    """Prove the instrument can fail: the check must NOT pass on the
    untouched baseline. Returns the baseline output's hash."""
    run = _run(cmd, baseline_root)
    if run.timed_out:
        raise InstrumentError(f"the frozen check timed out on the baseline after {cmd.timeout_s:g}s")
    if run.returncode == 0:
        raise InstrumentError("the frozen check passes on the untouched baseline; the instrument "
                              "cannot reject a no-op and its verdicts are worthless")
    return run.output_sha256


def check(tree: Path, cmd: FrozenCommand) -> VerifierOutcome:
    """The frozen check on a tree with no baseline: the admission check on a
    known solution, not a verdict on an attempt."""
    return _outcome(cmd, _run(cmd, tree))


def verify(workspace: Path, cmd: FrozenCommand, baseline: Baseline) -> VerifierOutcome:
    """Verdict on the workspace as it stands, against its baseline."""
    changes = baseline.changes(workspace)
    if not changes:
        return _outcome(cmd, _Run(0, False, "", "", 0.0), no_op=True)
    tamper = [f"frozen file changed: {rel}" for rel in changes.paths if rel in cmd.files]
    tamper += [f"test configuration changed: {rel}" for rel in changes.paths
               if rel not in cmd.files and Path(rel).name in CONFIG_BASENAMES]
    return _outcome(cmd, _run(cmd, workspace), tamper=tamper)


# ------------------------------------------------------------------ discovery

#: Conservative and explicit. A repository gets a check only if one of these
#: is plainly present; guessing a build command is how a verifier starts
#: reporting failures that are about the environment rather than the change.
CANDIDATES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("Makefile", ("make", "test"), ("test:",)),
    ("makefile", ("make", "test"), ("test:",)),
    ("package.json", ("npm", "test", "--silent"), ('"test"',)),
    ("Cargo.toml", ("cargo", "test", "--quiet"), ()),
    ("go.mod", ("go", "test", "./..."), ()),
    ("pyproject.toml", (sys.executable, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider"), ()),
    ("setup.py", (sys.executable, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider"), ()),
    ("tox.ini", (sys.executable, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider"), ()),
)


def discover(tree: Path) -> tuple[str, ...] | None:
    """The repository's own check, or None. Never invented: the marker file
    must exist and, where a target can be absent, contain the target."""
    for marker, argv, needles in CANDIDATES:
        p = tree / marker
        if not p.is_file():
            continue
        if needles:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not any(n in text for n in needles):
                continue
        return argv
    return None
