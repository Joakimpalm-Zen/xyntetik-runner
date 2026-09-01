#!/usr/bin/env python3
"""Release artifact consistency gate.

This is intentionally narrow: it checks the files that are packaged or directly
drive release packaging. Historical reports may mention older versions; the
packaged README and workflow comments must not drift.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# What counts as a RELEASE STRING in these files.
#
# This required the `-alpha` suffix, which v0.2.0 retired -- so from that
# release on it matched nothing anywhere in the tree and the three checks built
# on it (README, workflow, current docs) were vacuous. Found with
# .github/workflows/release.yml carrying `git tag v0.2.0` against a 0.3.0 tree,
# in the exact file the workflow scan targets, green.
#
# A bare dotted triple cannot be the pattern: this tree cites competitor
# versions (vLLM 0.27.1, Ollama 0.32.14), tool versions (Docker 29.1.3) and
# loopback addresses, and none of those are drift. So a release string is one
# of three unambiguous spellings -- the retired -alpha form, a `v`-prefixed tag
# spelling, or a version this project names as its own.
RELEASE_STRING_RE = re.compile(
    r"(?<![\w.\-])(v?\d+\.\d+\.\d+-alpha)(?![\w.\-])"
    r"|(?<![\w.\-])(v\d+\.\d+\.\d+)(?![\w.\-])"
    r"|\b[Rr]unner:?[ \t]+`?(v?\d+\.\d+\.\d+(?:-alpha)?)(?![\w.\-])")


def fail(msg):
    print(f"release-check: {msg}", file=sys.stderr)
    return False


def binary_version(binary):
    proc = subprocess.run([str(Path(binary).resolve()), "--version"], text=True,
                          capture_output=True, timeout=20)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"{binary} --version failed")
    return proc.stdout.strip()


def stale_release_strings(text, version, tag):
    stale = set()
    for match in RELEASE_STRING_RE.finditer(text):
        s = match.group(1) or match.group(2) or match.group(3)
        if s not in {version, tag}:
            stale.add(s)
    return sorted(stale)


def read(path):
    return Path(path).read_text(encoding="utf-8")


def python_version_for_release(version):
    """Translate Runner's alpha spelling to its PEP 440 package spelling."""
    if version.endswith("-alpha"):
        return version[:-len("-alpha")] + "a0"
    return version


def pyproject_version(path):
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', read(path), re.M)
    if not match:
        raise ValueError(f"no project version found in {path}")
    return match.group(1)


# Strings that name PRIVATE repositories. This repo is public: an evidence
# doc that cites a private path or commit leaks internal topology (found in
# the wild by the 2026-08-24 review — docs published from a measurement
# session carried suite paths for a week). The class-level fix is this scan;
# the instance-level fixes were commit d9557b3.
PRIVATE_MARKERS = ("xyntetik-suite", "xyntetik-shade")


def private_reference_scan():
    try:
        proc = subprocess.run(
            ["git", "grep", "-l", "-i", "-E", "|".join(PRIVATE_MARKERS),
             "--", ":!scripts/check-release.py"],
            cwd=ROOT, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        # no git on PATH (the Windows msys CI python): the same tree is
        # scanned by the Linux/macOS jobs of the same release, so skipping
        # here loses nothing — but say so rather than pass silently
        print("release-check: note: git unavailable, private-reference "
              "scan skipped on this job (covered by sibling jobs)")
        return True
    except subprocess.TimeoutExpired:
        # A hung scan is not a skippable scan: unlike a missing git, a hang
        # can hit every sibling job identically, so nothing else covers it.
        return fail("private-reference scan timed out after 60s")
    except OSError as e:
        # PermissionError and friends are NOT "git unavailable": they can
        # afflict all jobs at once (repo permissions), so the sibling-job
        # argument does not apply and the gate fails closed.
        return fail(f"private-reference scan could not run: {e}")
    if proc.returncode not in (0, 1):
        detail = proc.stderr.strip() or f"git grep exited {proc.returncode}"
        return fail(f"private-reference scan failed: {detail}")
    hits = [l for l in proc.stdout.splitlines() if l.strip()]
    if hits:
        return fail("private-repo references in public tree: "
                    + ", ".join(hits))
    return True


def check(args):
    ok = True
    ok &= private_reference_scan()
    version = args.tag[1:] if args.tag.startswith("v") else args.tag
    expected_binary = f"runner {version}"

    got = binary_version(args.binary)
    if got != expected_binary:
        ok &= fail(f"binary version {got!r} does not match tag {args.tag!r}")

    readme = read(args.readme)
    # v0.2.0 retired the -alpha suffix: the README banner says "Pre-1.0"
    # with the exact version; older tags keep the alpha phrasing.
    if (f"Pre-1.0 (`{version}`)" not in readme and
            f"Public alpha (`{version}`)" not in readme):
        ok &= fail(f"README does not identify the {version} banner")
    if f"./runner --version   # -> runner {version}" not in readme:
        ok &= fail("README version-output example is not in sync")
    for stale in stale_release_strings(readme, version, args.tag):
        ok &= fail(f"README contains stale release string {stale!r}")

    changelog = read(args.changelog)
    if not re.search(rf"^## v?{re.escape(version)}\b", changelog, re.M):
        ok &= fail(f"CHANGELOG has no section for {version}")

    build_info = read(args.build_info)
    build_lines = build_info.splitlines()
    if build_lines[:1] != [expected_binary]:
        ok &= fail("BUILD-INFO first line does not match binary version")
    if f"tag:        {args.tag}" not in build_lines:
        ok &= fail("BUILD-INFO tag line is inconsistent")
    if f"commit:     {args.commit}" not in build_lines:
        ok &= fail("BUILD-INFO commit line is inconsistent")

    expected_python = python_version_for_release(version)
    try:
        got_python = pyproject_version(args.python_pyproject)
    except (OSError, ValueError) as exc:
        ok &= fail(f"cannot read Python package version: {exc}")
    else:
        if got_python != expected_python:
            ok &= fail(
                f"Python package version {got_python!r} does not match "
                f"release {version!r} ({expected_python!r})"
            )

    # Naming a workflow that is not there means it moved or was renamed, which
    # is the one answer this gate must never give quietly -- the same call the
    # compat-report block below makes, in the same words: skipping on a missing
    # file turns the check into no check.
    release_workflow = args.release_workflow
    if not release_workflow.exists():
        ok &= fail(f"release workflow {release_workflow} does not exist")
    else:
        workflow = release_workflow.read_text(encoding="utf-8")
        for stale in stale_release_strings(workflow, version, args.tag):
            ok &= fail(f"release workflow contains stale release string {stale!r}")

    for path in args.current_docs:
        for stale in stale_release_strings(read(path), version, args.tag):
            ok &= fail(f"current document {path} contains stale release string {stale!r}")

    # A compatibility ledger nobody publishes is a private spreadsheet. The
    # reports have existed since 2026-08-13; this makes producing one part of
    # cutting a release instead of part of remembering to. The gate only
    # asks that a dated report for THIS version exists — what it contains is
    # the manifest's business, and a report whose checks did not run says so
    # itself (RNR-007).
    # Not passing a directory at all means "do not ask for a report" and
    # releases fine. Naming one that is not there means the ledger moved or
    # was deleted, which is the one answer this gate must never give quietly:
    # skipping on a missing directory turns the check into no check.
    if args.compat_reports:
        stem = f"{version}-"
        reports = (sorted(args.compat_reports.iterdir())
                   if args.compat_reports.is_dir() else [])
        mine = [p for p in reports
                if p.name.startswith(stem) and p.suffix == ".json"]
        if not mine:
            ok &= fail(
                f"no compat report for {version} in {args.compat_reports} "
                f"(expected {stem}<date>.json — run scripts/compat_matrix.py "
                f"on a box that has the pinned models and commit the report)"
            )
        # v0.4.5 shipped with a report that existed and measured nothing:
        # 25 models, every check `not_executed`, because the release box had
        # 2 of the 25 files. Existence is not evidence. A release needs at
        # least one report for its version in which at least one check
        # actually ran; `not_executed` rows stay honest, they just cannot be
        # the whole ledger.
        elif executed_checks(mine) == 0:
            ok &= fail(
                f"compat report(s) for {version} executed no check at all "
                f"({', '.join(p.name for p in mine)}): every row is "
                f"not_executed or the file is not a matrix report — run "
                f"scripts/compat_matrix.py --execute-checks on a box that "
                f"has the pinned models"
            )

    return ok


def executed_checks(reports):
    """How many checks across these compat reports actually ran."""
    n = 0
    for path in reports:
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for model in (doc.get("models") or []) if isinstance(doc, dict) else []:
            for check in (model.get("checks") or {}).values():
                if isinstance(check, dict) and \
                        check.get("status") not in (None, "not_executed"):
                    n += 1
    return n


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--changelog", type=Path, default=ROOT / "CHANGELOG.md")
    parser.add_argument("--build-info", type=Path, required=True)
    parser.add_argument("--release-workflow", type=Path,
                        default=ROOT / ".github/workflows/release.yml")
    parser.add_argument("--python-pyproject", type=Path,
                        default=ROOT / "python/pyproject.toml")
    parser.add_argument("--compat-reports", type=Path,
                        default=ROOT / "docs" / "compat-reports",
                        help="directory of dated per-release compatibility "
                             "reports; the release must ship one for its own "
                             "version")
    parser.add_argument("--commit", required=True,
                        help="commit SHA expected in BUILD-INFO.txt")
    # No default here: argparse APPENDS to one, so a caller passing
    # --current-doc to NARROW the scan silently widened it instead -- and if
    # any of the three built-ins were ever deleted from the tree, the widened
    # scan raised FileNotFoundError from read() rather than reporting it.
    parser.add_argument(
        "--current-doc", dest="current_docs", type=Path, action="append",
        help="current (non-historical) document that may not contain stale "
             "versions; repeatable, and REPLACES the built-in set",
    )
    args = parser.parse_args(argv)
    if args.current_docs is None:
        args.current_docs = [ROOT / "SECURITY.md", ROOT / "CONTRIBUTING.md",
                             ROOT / "docs/moe-support.md"]
    return 0 if check(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
