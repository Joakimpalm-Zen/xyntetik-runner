import argparse
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _git_checkout(root):
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
                              capture_output=True, text=True).stdout.strip() == "true"
    except OSError:
        return False


# A passing check() runs the release check's tree scans, which are `git grep`
# over this checkout and fail closed when the tree is not one (a source
# archive): that refusal is right for a release, so a test that needs every
# check to pass needs a checkout. Without one it skips, saying why; with git
# absent altogether the scans are skipped by the script itself.
needs_git_checkout = pytest.mark.skipif(
    not _git_checkout(ROOT),
    reason="needs a git checkout: the release check's tree scans run git grep")
SPEC = importlib.util.spec_from_file_location(
    "check_release", ROOT / "scripts/check-release.py"
)
check_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_release)


def _parse(argv):
    """The real parser, stopped before check() runs."""
    holder = {}

    def capture(args):
        holder["args"] = args
        return True

    original = check_release.check
    check_release.check = capture
    try:
        check_release.main(argv)
    finally:
        check_release.check = original
    return holder["args"]


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def good_args(tmp_path):
    return argparse.Namespace(
        tag="v1.2.0",
        binary=tmp_path / "runner",
        readme=write(
            tmp_path / "README.md",
            "Version `1.2.0`\n"
            "./runner --version   # -> runner 1.2.0\n",
        ),
        changelog=write(tmp_path / "CHANGELOG.md", "## v1.2.0\n"),
        build_info=write(
            tmp_path / "BUILD-INFO.txt",
            "runner 1.2.0\n"
            "tag:        v1.2.0\n"
            "commit:     abc123\n",
        ),
        release_workflow=write(
            tmp_path / "release.yml", "# release v1.2.0\n"
        ),
        python_pyproject=write(
            tmp_path / "pyproject.toml",
            '[project]\nversion = "1.2.0"\n',
        ),
        site_pages=None,
        commit="abc123",
        current_docs=[],
        compat_reports=report_dir(tmp_path, "1.2.0-2026-01-01.json"),
        device_evidence=device_ledger(tmp_path),
        evidence_root=tmp_path,
        model_tiers=tier_roster(tmp_path),
        tiers_today=TIERS_TODAY,
    )


# The tier gate is anchored on a fixed day, not the clock, for the ages it
# prints: the release under test is "cut" on 2026-10-23 and the roster's one
# Tier 1 row was validated by Runner 1.5.0 the day before, or (since the
# 2026-10-28 decision that age never refuses) months before, or never.
TIERS_TODAY = "2026-10-23"


def tier_roster(tmp_path, gated_on="2026-10-22", version="1.5.0"):
    path = tmp_path / "model-tiers.json"
    cell = None if gated_on is None else {
        "kind": "gate", "runner_version": version, "date": gated_on,
        "device_class": "test-class", "result": "pass",
        "commit": "b" * 40, "caps": {"version": version},
        "output_sha256": "c" * 64, "ran": "scripts/tier-smoke.py",
        "evidence": "docs/model-evidence.md"}
    write(path, json.dumps({
        "schema": "xyntetik.runner.model-tiers.v1",
        "policy": {"backends": ["cpu", "cuda", "metal"]},
        "exempt": [],
        "rows": [{
            "id": "toy-q8_0", "model": "Toy", "family": "Toy", "tier": "tier1",
            "claim": "A toy Tier 1 row.", "architecture": "llama",
            "upstream": {"checkpoint": "toy/toy", "revision": None},
            "licence": {"id": "apache-2.0", "from": "toy/toy"},
            "files": [{"name": "toy.gguf", "sha256": "a" * 64, "bytes": 1}],
            "envelope": {"features": ["text"], "fp32_reference": "measured",
                         "anchor": "toy", "evidence": ["docs/model-evidence.md"]},
            "smoke": {"device_class": "test-class", "backend": "cpu"},
            "backends": {"cpu": cell, "cuda": None, "metal": None},
            "lora": None,
            "transitions": [{"tier": "tier1", "effective": "2026-10-22",
                           "reason": "test", "last_validated_release": None}],
        }],
    }) + "\n")
    return path


GATE_SUMMARY = "docs/device-runs-2026-01-01/test-cuda/cuda-gate-20260101T000000Z.json"


def device_ledger(tmp_path, days_ago=1, result="pass", ownership=None, cuda_version="1.2.0",
                  cuda_ran="scripts/cuda-gate.sh: test-cuda-gated", cuda_summary="pass"):
    """A ledger with one class, verified `days_ago` days ago, and a borrowed
    CUDA class whose full gate (scripts/cuda-gate.sh) passed for
    `cuda_version`, its summary written under tmp_path as `cuda_summary`
    says: "pass", "fail", "other-commit", or None for no file (R6.44).

    The real one lives in docs/ and describes real hardware; a test that read
    it would pass or fail on whether somebody had run the lab's Windows box
    this month, which is not what any of these tests are about.
    """
    import datetime
    when = (datetime.datetime.now(datetime.timezone.utc).date()
            - datetime.timedelta(days=days_ago))
    path = tmp_path / "device-evidence.json"
    write(path, json.dumps({
        "schema": "xyntetik.runner.device-evidence.v1",
        "policy": {"default_max_age_days": 45},
        "classes": [{
            "id": "test-class",
            "max_age_days": 45,
            **({"ownership": ownership} if ownership else {}),
            "last_verified": {"date": when.isoformat(), "commit": "abc123",
                              "ran": "make test", "result": result},
        }, {
            "id": "test-cuda",
            "max_age_days": 45,
            "ownership": "borrowed",
            "last_verified": {"date": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
                              "commit": "abc123", "ran": cuda_ran, "result": "pass",
                              "caps": {"version": cuda_version, "gpu_backend": "cuda"},
                              "evidence": GATE_SUMMARY},
        }],
    }) + "\n")
    summary = tmp_path / GATE_SUMMARY
    summary.parent.mkdir(parents=True, exist_ok=True)
    if summary.exists():
        summary.unlink()
    if cuda_summary is not None:
        write(summary, json.dumps({
            "schema": "xyntetik.runner.cuda-gate.v2", "stamp": "20260101T000000Z",
            "class": "test-cuda", "commit": "def456" if cuda_summary == "other-commit" else "abc123",
            "result": "fail" if cuda_summary == "fail" else "pass", "steps": []}) + "\n")
    return path


def report_dir(tmp_path, *names):
    """A compat-report directory holding EXACTLY the named dated reports.

    Cleared on entry: the helper is called twice per test (once by good_args,
    once by the test itself), and a leftover report from the first call would
    satisfy the gate the second call is trying to trip.
    """
    d = tmp_path / "compat-reports"
    d.mkdir(exist_ok=True)
    for stale in d.iterdir():
        stale.unlink()
    for name in names:
        write(d / name, EXECUTED_REPORT)
    return d


# The smallest matrix report in which one check actually ran. A placeholder
# `{}` used to satisfy the gate, which is how v0.4.5 shipped a ledger that
# measured nothing.
EXECUTED_REPORT = json.dumps({"models": [{"id": "x", "checks": {
    "load": {"status": "pass"}}}]}) + "\n"
NOTHING_RAN_REPORT = json.dumps({"models": [{"id": "x", "checks": {
    "load": {"status": "not_executed", "reason": "model_file_not_found"},
    "chat": {"status": "not_executed", "reason": "model_file_not_found"}}}],
    "complete": False}) + "\n"


@needs_git_checkout
def test_release_check_accepts_consistent_artifacts(monkeypatch, tmp_path):
    args = good_args(tmp_path)
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 1.2.0"
    )
    assert check_release.check(args)


@needs_git_checkout
def test_release_check_accepts_legacy_alpha_package_spelling(monkeypatch, tmp_path):
    args = good_args(tmp_path)
    args.tag = "v0.1.3-alpha"
    write(args.readme, "Public alpha (`0.1.3-alpha`)\n"
          "./runner --version   # -> runner 0.1.3-alpha\n")
    write(args.changelog, "## v0.1.3-alpha\n")
    write(args.build_info, "runner 0.1.3-alpha\ntag:        v0.1.3-alpha\n"
          "commit:     abc123\n")
    write(args.python_pyproject, '[project]\nversion = "0.1.3a0"\n')
    write(args.release_workflow, "# release workflow\n")
    args.compat_reports = report_dir(tmp_path, "0.1.3-alpha-2026-01-01.json")
    args.device_evidence = device_ledger(tmp_path, cuda_version="0.1.3-alpha")
    monkeypatch.setattr(check_release, "binary_version", lambda _: "runner 0.1.3-alpha")
    assert check_release.check(args)


def test_release_check_rejects_stale_readme(monkeypatch, tmp_path, capsys):
    args = good_args(tmp_path)
    args.readme.write_text(
        args.readme.read_text() + "old example v1.1.4\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 1.2.0"
    )
    assert not check_release.check(args)
    assert "stale release string 'v1.1.4'" in capsys.readouterr().err


def test_release_check_rejects_unfolded_changelog_entries(monkeypatch, tmp_path, capsys):
    """R6.21: an entry left in changes/ would be missing from the release notes."""
    args = good_args(tmp_path)
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 1.2.0"
    )
    changes = tmp_path / "changes"
    changes.mkdir()
    write(changes / "README.md", "# the format, not an entry\n")
    assert check_release.check(args)
    write(changes / "R6.21.md", "- **Left behind (R6.21).**\n")
    capsys.readouterr()
    assert not check_release.check(args)
    assert ("changes/ still holds 1 unfolded changelog entry (R6.21.md): "
            "run python3 scripts/changelog-entries.py fold --write") in capsys.readouterr().err


def test_release_check_rejects_build_commit_drift(monkeypatch, tmp_path, capsys):
    args = good_args(tmp_path)
    args.commit = "different"
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 1.2.0"
    )
    assert not check_release.check(args)
    assert "BUILD-INFO commit line is inconsistent" in capsys.readouterr().err


def test_release_check_rejects_python_version_drift(monkeypatch, tmp_path, capsys):
    args = good_args(tmp_path)
    args.python_pyproject.write_text(
        '[project]\nversion = "1.1.4"\n', encoding="utf-8"
    )
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 1.2.0"
    )
    assert not check_release.check(args)
    assert "Python package version" in capsys.readouterr().err


def test_release_check_rejects_stale_current_document(monkeypatch, tmp_path, capsys):
    args = good_args(tmp_path)
    args.current_docs = [write(tmp_path / "SECURITY.md", "public v1.1.4\n")]
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 1.2.0"
    )
    assert not check_release.check(args)
    assert "current document" in capsys.readouterr().err


# The ledger is only credible if every release ships one. The reports existed
# from 2026-08-13 but nothing made producing them part of cutting a release,
# so the habit depended on somebody remembering.

def test_release_check_requires_a_compat_report_for_this_release(
        monkeypatch, tmp_path, capsys):
    args = good_args(tmp_path)
    args.compat_reports = report_dir(tmp_path, "1.1.4-2026-01-01.json")
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 1.2.0"
    )
    assert not check_release.check(args)
    assert "compat report" in capsys.readouterr().err


@needs_git_checkout
def test_release_check_accepts_any_dated_report_for_this_release(
        monkeypatch, tmp_path):
    args = good_args(tmp_path)
    args.compat_reports = report_dir(
        tmp_path, "1.1.4-2026-01-01.json", "1.2.0-2026-02-09.json"
    )
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 1.2.0"
    )
    assert check_release.check(args)


@needs_git_checkout
def test_release_check_skips_the_report_gate_when_no_directory_is_given(
        monkeypatch, tmp_path):
    """An older checkout, or a caller that does not pass the flag, still
    releases: the gate binds the process, it does not break the tool."""
    args = good_args(tmp_path)
    args.compat_reports = None
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 1.2.0"
    )
    assert check_release.check(args)


def test_release_check_fails_when_the_named_report_directory_is_missing(
        monkeypatch, tmp_path, capsys):
    """Passing None means "do not ask for a report". Naming a directory that
    does not exist means the reports moved or were deleted — and a gate that
    answers "fine" to that has checked nothing at all."""
    args = good_args(tmp_path)
    args.compat_reports = tmp_path / "compat-reports-renamed"
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 1.2.0"
    )
    assert not check_release.check(args)
    assert "compat report" in capsys.readouterr().err


def test_private_reference_scan_rejects_a_fatal_git_error(monkeypatch, capsys):
    monkeypatch.setattr(
        check_release.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 128, stdout="", stderr="fatal: not a git repository\n"
        ),
    )

    assert not check_release.private_reference_scan()
    err = capsys.readouterr().err
    assert "private-reference scan failed" in err
    assert "not a git repository" in err


# The three checks that lean on RELEASE_STRING_RE were dead from v0.2.0, the
# release that retired the -alpha suffix: the pattern REQUIRED it, so it
# matched nothing anywhere in the tree. Live proof at the time this was found:
# .github/workflows/release.yml carried `git tag v0.2.0` while the tree was at
# 0.3.0, in the exact file the workflow scan targets, and the gate was green.

@needs_git_checkout
def test_release_check_accepts_the_1x_banner(monkeypatch, tmp_path, capsys):
    """From 1.0.0 the README banner reads "Version `X.Y.Z`" and points at the
    versioning policy; the gate accepts it and still demands the exact
    version, so a 1.0.0 banner does not satisfy a 1.0.1 release."""
    args = good_args(tmp_path)
    args.tag = "v1.0.0"
    write(args.readme,
          "> **Version `1.0.0`.** See docs/versioning.md.\n"
          "./runner --version   # -> runner 1.0.0\n")
    write(args.changelog, "## v1.0.0\n")
    write(args.build_info,
          "runner 1.0.0\ntag:        v1.0.0\ncommit:     abc123\n")
    write(args.python_pyproject, '[project]\nversion = "1.0.0"\n')
    write(args.release_workflow, "# release workflow\n")
    args.compat_reports = report_dir(tmp_path, "1.0.0-2026-01-01.json")
    args.device_evidence = device_ledger(tmp_path, cuda_version="1.0.0")
    monkeypatch.setattr(check_release, "binary_version", lambda _: "runner 1.0.0")
    assert check_release.check(args)

    args.tag = "v1.0.1"
    write(args.changelog, "## v1.0.1\n")
    write(args.build_info,
          "runner 1.0.1\ntag:        v1.0.1\ncommit:     abc123\n")
    write(args.python_pyproject, '[project]\nversion = "1.0.1"\n')
    args.compat_reports = report_dir(tmp_path, "1.0.1-2026-01-01.json")
    monkeypatch.setattr(check_release, "binary_version", lambda _: "runner 1.0.1")
    capsys.readouterr()
    assert not check_release.check(args)
    assert "does not identify the 1.0.1 banner" in capsys.readouterr().err


def test_stale_strings_are_found_in_the_current_spelling(monkeypatch, tmp_path,
                                                         capsys):
    """A release string without -alpha is still a release string."""
    args = good_args(tmp_path)
    args.tag = "v0.3.0"
    write(args.readme,
          "Pre-1.0 (`0.3.0`)\n"
          "./runner --version   # -> runner 0.3.0\n"
          "see the v0.2.0 notes\n")
    write(args.changelog, "## v0.3.0\n")
    write(args.build_info,
          "runner 0.3.0\ntag:        v0.3.0\ncommit:     abc123\n")
    write(args.python_pyproject, '[project]\nversion = "0.3.0"\n')
    args.compat_reports = report_dir(tmp_path, "0.3.0-2026-01-01.json")
    monkeypatch.setattr(check_release, "binary_version", lambda _: "runner 0.3.0")
    assert not check_release.check(args)
    assert "stale release string 'v0.2.0'" in capsys.readouterr().err


def test_a_dated_mark_in_the_readme_is_a_stale_string(monkeypatch, tmp_path, capsys):
    """Until 2026-10-27 the README carried the rendered tier table, whose
    dated cells could name an older release inside their age limit, and the
    scan read around it. The marks live in docs/model-support.md now, so the
    whole README is scanned: an older release named anywhere in it refuses."""
    _pinned(monkeypatch)
    args = good_args(tmp_path)
    base = args.readme.read_text(encoding="utf-8")
    write(args.readme, base + "old example v1.1.4\n")
    assert not check_release.check(args)
    assert "stale release string 'v1.1.4'" in capsys.readouterr().err


def test_a_stale_version_banner_in_the_manual_is_a_stale_string(monkeypatch, tmp_path,
                                                              capsys):
    """MANUAL.md carries the README's "Version `X.Y.Z`" banner. The scan read
    a version only after the word Runner, so that copy said 1.6.0 from 1.7.0
    to 1.13.0's cut; the banner form is a release string now."""
    _pinned(monkeypatch)
    args = good_args(tmp_path)
    write(tmp_path / "MANUAL.md", "> **Version `1.1.0`.** See docs/versioning.md.\n")
    assert not check_release.check(args)
    assert "MANUAL.md contains stale release string '1.1.0'" in capsys.readouterr().err


def test_competitor_versions_are_not_read_as_release_strings():
    """The README cites other runtimes by version, and the docs cite runner's
    own history. Neither is drift, and flagging either would fail every
    release for text that is correct."""
    text = ("| vLLM 0.27.1 | ... |\n| Ollama 0.32.14 | ... |\n"
            "Docker 29.1.3 + NVIDIA Container Toolkit 1.19.1\n"
            "base_url=\"http://127.0.0.1:8080/v1\"\n"
            "there is no --host/0.0.0.0 flag\n"
            "(see the 0.1.6/0.1.7 notes)\n")
    assert check_release.stale_release_strings(text, "0.3.0", "v0.3.0") == []


def test_a_missing_release_workflow_fails_rather_than_skipping(monkeypatch,
                                                              tmp_path, capsys):
    """Naming a file that is not there means it moved or was deleted, and a
    gate that answers "fine" to that has checked nothing -- the reasoning the
    compat-report gate in the same function already spells out."""
    args = good_args(tmp_path)
    args.release_workflow = tmp_path / "release-renamed.yml"
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 1.2.0")
    assert not check_release.check(args)
    assert "release workflow" in capsys.readouterr().err


def test_current_doc_replaces_the_default_set_rather_than_adding_to_it(tmp_path):
    """argparse APPENDS to a non-empty default, so a caller narrowing the scan
    silently widened it -- and picked up whichever of the three defaults had
    since been deleted from the tree as a FileNotFoundError."""
    only = tmp_path / "ONE.md"
    write(only, "x\n")
    parsed = _parse(["--tag", "v1.2.0", "--binary", "b",
                     "--build-info", "bi", "--commit", "c",
                     "--current-doc", str(only)])
    assert [str(p) for p in parsed.current_docs] == [str(only)]
    default = _parse(["--tag", "v1.2.0", "--binary", "b",
                      "--build-info", "bi", "--commit", "c"])
    assert len(default.current_docs) == 3


def test_scan_failure_modes_fail_closed_except_missing_git(monkeypatch, capsys):
    """Only a genuinely absent git may skip the private-reference scan: that
    is the one failure the sibling platform jobs provably cover. A hang or a
    PermissionError can afflict every job of the release identically, so
    treating them as skips would green a release nothing scanned. The old
    handler caught OSError wholesale and PermissionError is an OSError."""
    def raising(exc):
        def run(*a, **k):
            raise exc
        return run

    monkeypatch.setattr(check_release.subprocess, "run",
                        raising(FileNotFoundError("no git")))
    assert check_release.private_reference_scan()          # skip, says so
    assert "skipped" in capsys.readouterr().out

    monkeypatch.setattr(check_release.subprocess, "run",
                        raising(PermissionError(13, "denied")))
    assert not check_release.private_reference_scan()      # fail closed

    monkeypatch.setattr(check_release.subprocess, "run",
                        raising(subprocess.TimeoutExpired(cmd="git", timeout=60)))
    assert not check_release.private_reference_scan()      # fail closed


def test_release_check_refuses_a_report_in_which_nothing_ran(
        monkeypatch, tmp_path, capsys):
    """Existence is not evidence: a report whose every check is
    not_executed is a placeholder, and the gate names it as one."""
    args = good_args(tmp_path)
    args.compat_reports = report_dir(tmp_path, "1.2.0-2026-01-01.json")
    write(args.compat_reports / "1.2.0-2026-01-01.json",
          NOTHING_RAN_REPORT)
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 1.2.0"
    )
    assert not check_release.check(args)
    assert "executed no check" in capsys.readouterr().err


@needs_git_checkout
def test_release_check_counts_executed_checks_across_all_dated_reports(
        monkeypatch, tmp_path):
    """An empty report beside a real one for the same version is fine: the
    version has evidence somewhere in its ledger."""
    args = good_args(tmp_path)
    args.compat_reports = report_dir(
        tmp_path, "1.2.0-2026-01-01.json", "1.2.0-2026-01-02.json")
    write(args.compat_reports / "1.2.0-2026-01-01.json",
          NOTHING_RAN_REPORT)
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 1.2.0"
    )
    assert check_release.check(args)


@needs_git_checkout
def test_release_check_requires_readme_and_site_to_link_the_same_hf_repos(
        monkeypatch, tmp_path, capsys):
    """A card published with a README entry and no site entry, or the reverse,
    is the drift AGENTS.md rule 4 names; the release refuses until both agree."""
    args = good_args(tmp_path)
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 1.2.0"
    )
    args.readme = write(
        tmp_path / "README.md",
        "Version `1.2.0`\n"
        "./runner --version   # -> runner 1.2.0\n"
        "https://huggingface.co/Joakimpalm-Zen/Only-In-README\n"
        "https://huggingface.co/Joakimpalm-Zen/Both-GGUF\n",
    )
    pages = tmp_path / "pages"
    pages.mkdir()
    write(pages / "evidence.html",
          '<a href="{{hf}}/Both-GGUF">x</a> <a href="{{hf}}/Only-On-Site">y</a>')
    args.site_pages = pages
    assert check_release.check(args) is False
    cap = capsys.readouterr()
    out = cap.out + cap.err
    assert "Only-In-README" in out and "Only-On-Site" in out
    write(pages / "evidence.html",
          '<a href="{{hf}}/Both-GGUF">x</a> '
          '<a href="https://huggingface.co/Joakimpalm-Zen/Only-In-README">y</a>')
    assert check_release.check(args) is True


@needs_git_checkout
def test_release_check_parity_sees_dataset_links_too(monkeypatch, tmp_path, capsys):
    """Evidence repositories live under `datasets/`; a parity check that only
    read the model spelling let a migrated report drift out of the README or
    the site unseen (2026-09-15 restructure)."""
    args = good_args(tmp_path)
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 1.2.0"
    )
    args.readme = write(
        tmp_path / "README.md",
        "Version `1.2.0`\n"
        "./runner --version   # -> runner 1.2.0\n"
        "https://huggingface.co/datasets/Joakimpalm-Zen/Only-In-README-report\n"
        "https://huggingface.co/datasets/Joakimpalm-Zen/Both-report\n",
    )
    pages = tmp_path / "pages"
    pages.mkdir()
    write(pages / "evidence.html",
          '<a href="{{hfd}}/Both-report">x</a> <a href="{{hfd}}/Only-On-Site-report">y</a>')
    args.site_pages = pages
    assert check_release.check(args) is False
    out = capsys.readouterr()
    out = out.out + out.err
    assert "Only-In-README-report" in out and "Only-On-Site-report" in out
    write(pages / "evidence.html",
          '<a href="{{hfd}}/Both-report">x</a> '
          '<a href="https://huggingface.co/datasets/Joakimpalm-Zen/Only-In-README-report">y</a>')
    assert check_release.check(args) is True


def _pinned(monkeypatch):
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 1.2.0"
    )


def test_release_check_refuses_stale_device_evidence(monkeypatch, tmp_path):
    """The classes CI cannot reach are the ones that rot unseen.

    Nineteen days passed between Windows runs before anybody noticed, and
    nothing in the release path asked. It asks now.
    """
    _pinned(monkeypatch)
    args = good_args(tmp_path)
    args.device_evidence = device_ledger(tmp_path, days_ago=200)
    assert check_release.check(args) is False


def test_release_check_refuses_a_failing_device_class(monkeypatch, tmp_path):
    _pinned(monkeypatch)
    args = good_args(tmp_path)
    args.device_evidence = device_ledger(tmp_path, days_ago=1, result="fail")
    assert check_release.check(args) is False


def test_release_check_refuses_a_missing_device_ledger(monkeypatch, tmp_path):
    """Naming a ledger that is not there must not read as "nothing to check"."""
    _pinned(monkeypatch)
    args = good_args(tmp_path)
    args.device_evidence = tmp_path / "gone.json"
    assert check_release.check(args) is False


@needs_git_checkout
def test_release_check_accepts_stale_evidence_on_borrowed_hardware(monkeypatch, tmp_path):
    """Owner decision 2026-10-27: a lent machine may be used, never depended
    on, so its age cannot hold a release; a failure on it still does."""
    _pinned(monkeypatch)
    args = good_args(tmp_path)
    args.device_evidence = device_ledger(tmp_path, days_ago=200, ownership="borrowed")
    assert check_release.check(args) is True
    args.device_evidence = device_ledger(tmp_path, days_ago=1, result="fail",
                                         ownership="borrowed")
    assert check_release.check(args) is False


@needs_git_checkout
def test_release_check_accepts_a_never_validated_tier1_row_on_borrowed_hardware(monkeypatch, tmp_path):
    _pinned(monkeypatch)
    args = good_args(tmp_path)
    args.device_evidence = device_ledger(tmp_path, days_ago=1, ownership="borrowed")
    args.model_tiers = tier_roster(tmp_path, gated_on=None)
    assert check_release.check(args) is True


@needs_git_checkout
def test_release_check_accepts_fresh_device_evidence(monkeypatch, tmp_path):
    _pinned(monkeypatch)
    args = good_args(tmp_path)
    args.device_evidence = device_ledger(tmp_path, days_ago=3)
    assert check_release.check(args) is True


def test_release_check_does_not_read_a_rows_record(monkeypatch, tmp_path):
    """Validation runs when the project chooses, not with every release
    (owner decision 2026-11-07), so a row with no recorded pass never
    refuses a tag."""
    _pinned(monkeypatch)
    args = good_args(tmp_path)
    args.model_tiers = tier_roster(tmp_path, gated_on=None)
    assert check_release.check(args) is True


@needs_git_checkout
def test_release_check_accepts_an_old_tier1_record(monkeypatch, tmp_path):
    """Policy decision 2026-10-28: a record's age never refuses a tag; the
    roster keeps the version and date as the paper trail."""
    _pinned(monkeypatch)
    args = good_args(tmp_path)
    args.model_tiers = tier_roster(tmp_path, gated_on="2026-08-01")
    assert check_release.check(args) is True


@needs_git_checkout
def test_release_check_accepts_a_fresh_tier1_row(monkeypatch, tmp_path):
    _pinned(monkeypatch)
    args = good_args(tmp_path)
    args.model_tiers = tier_roster(tmp_path, gated_on="2026-10-22")
    assert check_release.check(args) is True


# The release workflow's Windows job runs its steps in the msys2 shell, so a
# PowerShell cmdlet on its own line is "command not found" and the build
# fails at packaging, after the binary was built and checked. The first tag
# after the python client joined the archive found `Remove-Item` there: the
# release job never ran, on a commit every CI job had passed, because CI never
# runs these steps. The steps are shell; PowerShell is called explicitly.
POWERSHELL_CMDLETS = ("Remove-Item", "Compress-Archive", "Copy-Item",
                      "Get-ChildItem", "New-Item", "Move-Item", "Expand-Archive")


def test_release_workflow_steps_are_shell_not_powershell():
    text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    offenders = []
    for n, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for cmdlet in POWERSHELL_CMDLETS:
            if stripped.startswith(cmdlet) or stripped.startswith(f"{cmdlet} "):
                offenders.append(f"{n}: {stripped[:70]}")
    assert not offenders, "PowerShell cmdlet on a shell step (wrap it in `powershell -Command`):\n" + "\n".join(offenders)


def test_private_markers_match_internals_not_only_names():
    """R6.8: a text that describes the private repositories without naming
    them is caught by their internal paths. The examples are assembled here
    so this file does not itself carry a marker."""
    import re
    rx = re.compile("|".join(map(re.escape, check_release.PRIVATE_MARKERS)),
                    re.I)
    hits = ["packages" + "/thane/src/app.py",
            "import xyntetik" + "_loadout",
            "crates" + "/suite-core/src/lib.rs",
            "research/archive" + "/suite/docs/plans/x.md",
            "see BLACKWELL" + "_INVENTORY.md"]
    for text in hits:
        assert rx.search(text), text
    misses = ["xyntetik_runner.shadow", "python/src/xyntetik_runner",
              "plan item R12.1", "scripts/runner-control.sh",
              "packages that ship", "the vault of receipts"]
    for text in misses:
        assert not rx.search(text), text


def test_home_directory_paths_are_recognised():
    """R6.2.x: a home directory in committed evidence names a machine. A
    scratch directory that happens to contain `home` is not one."""
    import re
    rx = re.compile(check_release.HOME_PATH_RE)
    for text in ["loaded /" + "home/lab/workspace/m.gguf",
                 '"path": "/' + "Users/someone/models/m.gguf",
                 "C:" + "\\Users\\someone\\m.gguf",
                 "C:" + "\\\\Users\\\\someone\\\\m.gguf",
                 "C:" + "/Users" + "/someone/m.gguf"]:
        assert rx.search(text), text
    for text in ["/tmp/agent-sweep-x/home/.continue/config.yaml",
                 "~/xs-work/models/m.gguf", "models/m.gguf"]:
        assert not rx.search(text), text


def test_kl_tables_need_a_margin_qualified_column():
    """R6.7.8: a table that reports mean KL without the margin-qualified
    top-1 beside it is refused; one that has it, or says it was not
    measured, passes; a table without KL is not a fidelity table."""
    md_bad = "| model | top-1 | mean KLD |\n|---|---|---|\n| a | 99% | 0.01 |\n"
    md_ok = ("| model | margin-qualified top-1 | mean KLD |\n|---|---|---|\n"
             "| a | not measured | 0.01 |\n")
    md_other = "| model | tok/s |\n|---|---|\n| a | 12 |\n"
    assert check_release.kl_tables_without_margin(md_bad)
    assert not check_release.kl_tables_without_margin(md_ok)
    assert not check_release.kl_tables_without_margin(md_other)
    html_bad = "<table><tr><th>engine</th><th>mean KL</th></tr></table>"
    html_ok = ("<table><tr><th>engine</th><th>margin-qualified top-1</th>"
               "<th>mean KL</th></tr></table>")
    assert check_release.kl_tables_without_margin(html_bad, html=True)
    assert not check_release.kl_tables_without_margin(html_ok, html=True)
    assert check_release.kl_table_scan()   # the tree itself is clean


def test_check_scans_the_tree_once_while_it_stands(monkeypatch, tmp_path):
    """The tree scans depend on the checkout, not on the release under test,
    so a process that calls check() repeatedly (this module does, twenty-eight
    times) scans once while the tree is unchanged. On the hosted Windows runner
    every repeat cost about sixteen seconds and this file alone was the root
    pytest's wall."""
    calls = {"kl": 0, "private": 0, "home": 0}

    def counting(key):
        def scan():
            calls[key] += 1
            return True
        return scan

    monkeypatch.setattr(check_release, "_tree_scan", None)
    monkeypatch.setattr(check_release, "kl_table_scan", counting("kl"))
    monkeypatch.setattr(check_release, "private_reference_scan", counting("private"))
    monkeypatch.setattr(check_release, "home_path_scan", counting("home"))
    monkeypatch.setattr(check_release, "binary_version", lambda _: "runner 1.2.0")
    monkeypatch.setattr(check_release, "tree_state", lambda root=None: "tree as it stands")
    assert check_release.check(good_args(tmp_path))
    assert check_release.check(good_args(tmp_path))
    assert calls == {"kl": 1, "private": 1, "home": 1}

    # The tree changes under the process: the next check() scans again, and
    # a tree git cannot describe is never cached.
    monkeypatch.setattr(check_release, "tree_state", lambda root=None: "tree after an edit")
    assert check_release.check(good_args(tmp_path))
    assert calls == {"kl": 2, "private": 2, "home": 2}
    monkeypatch.setattr(check_release, "tree_state", lambda root=None: None)
    assert check_release.check(good_args(tmp_path))
    assert check_release.check(good_args(tmp_path))
    assert calls == {"kl": 4, "private": 4, "home": 4}


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git on PATH")
def test_tree_state_sees_a_new_file_an_edit_and_a_commit(tmp_path):
    """The key behind the memo follows what the scans read: a new untracked
    file, a second edit of that file, a deletion and a commit are each a new
    state; the same tree is the same state."""
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path)] + list(args), check=True,
                       capture_output=True, text=True)
    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
        "--allow-empty", "-m", "root")
    state = check_release.tree_state(tmp_path)
    assert state is not None and state == check_release.tree_state(tmp_path)

    note = tmp_path / "docs-note.md"
    note.write_text("| KL | margin |\n", encoding="utf-8")
    with_file = check_release.tree_state(tmp_path)
    assert with_file != state

    note.write_text("| KL | margin |\n| 1 | 2 |\n", encoding="utf-8")
    os.utime(note, ns=(os.stat(note).st_atime_ns, os.stat(note).st_mtime_ns + 10**9))
    edited = check_release.tree_state(tmp_path)
    assert edited not in (state, with_file)

    git("add", "docs-note.md")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "note")
    committed = check_release.tree_state(tmp_path)
    assert committed not in (state, with_file, edited)
    assert committed == check_release.tree_state(tmp_path)

    note.unlink()
    assert check_release.tree_state(tmp_path) not in (state, with_file, edited, committed)
    assert check_release.tree_state(tmp_path / "not-a-checkout") is None


def test_a_failed_tree_scan_stays_failed_while_the_tree_stands(monkeypatch, tmp_path):
    """The memo keeps a refusal as firmly as a pass: a scan that failed once
    is not retried into a green by a later check() on the same tree."""
    monkeypatch.setattr(check_release, "_tree_scan", None)
    monkeypatch.setattr(check_release, "tree_state", lambda root=None: "tree as it stands")
    monkeypatch.setattr(check_release, "kl_table_scan", lambda: True)
    monkeypatch.setattr(check_release, "home_path_scan", lambda: True)
    monkeypatch.setattr(check_release, "private_reference_scan",
                        lambda: check_release.fail("private-repo references in public tree: x"))
    monkeypatch.setattr(check_release, "binary_version", lambda _: "runner 1.2.0")
    assert not check_release.check(good_args(tmp_path))
    monkeypatch.setattr(check_release, "private_reference_scan", lambda: True)
    assert not check_release.check(good_args(tmp_path))


# ---------------------------------------------------------------- changelog entries (R6.21)
# scripts/changelog-entries.py: one entry file per pull request in changes/, folded into
# CHANGELOG.md at release. Each test builds a small git repository of its own: a main branch with
# a CHANGELOG, and a branch made the way a pull request would make it. They live here, beside
# check-release's refusal of an unfolded entry, so the change adds no test file to the Makefile's
# root-pytest line (one long line, which every such addition makes another conflict on).
CE_TOOL = ROOT / "scripts" / "changelog-entries.py"

CE_CHANGELOG = """# Changelog

Intro text.

## Unreleased

- **An entry already on main (R1.1).**
  - detail

## v1.0.0 - 2026-01-01

- **Released (R0.1).**
"""


def ce_git(repo, *args, env=None):
    e = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
             GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid", **(env or {}))
    return subprocess.run(["git", *args], cwd=repo, env=e, check=True, capture_output=True, text=True).stdout


def ce_commit(repo, message, when=None):
    ce_git(repo, "add", "-A")
    env = None if when is None else {"GIT_AUTHOR_DATE": f"{when} +0000", "GIT_COMMITTER_DATE": f"{when} +0000"}
    ce_git(repo, "commit", "-q", "--allow-empty", "-m", message, env=env)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    ce_git(r, "init", "-q", "-b", "main")
    (r / "CHANGELOG.md").write_text(CE_CHANGELOG)
    (r / "changes").mkdir()
    (r / "changes" / "README.md").write_text("# Changelog entries\n")
    ce_commit(r, "base", when="2026-11-01T10:00:00")
    ce_git(r, "switch", "-q", "-c", "pr")
    return r


def ce_run(repo, *args):
    p = subprocess.run([sys.executable, str(CE_TOOL), *args], cwd=repo, capture_output=True, text=True,
                       env=dict(os.environ, GITHUB_ACTIONS=""))
    return p.returncode, p.stdout + p.stderr


def ce_check(repo, *extra):
    return ce_run(repo, "check", "--base", "main", *extra)


def ce_entry(repo, name, text):
    (repo / "changes" / name).write_text(text)


def test_one_entry_file_passes(repo):
    ce_entry(repo, "R2.1.md", "- **A fix (R2.1).**\n  - detail\n")
    ce_commit(repo, "R2.1")
    code, out = ce_check(repo)
    assert code == 0, out
    assert "ok: one entry, changes/R2.1.md" in out


def test_no_entry_and_no_reason_fails_and_says_what_to_do(repo):
    (repo / "src.c").write_text("int x;\n")
    ce_commit(repo, "code only")
    code, out = ce_check(repo)
    assert code == 1
    assert "no changelog entry: add changes/<plan ID>.md" in out
    assert "Changelog: none, <why>" in out


def test_a_reason_in_a_commit_message_or_the_description_passes(repo):
    (repo / "src.c").write_text("int x;\n")
    ce_commit(repo, "tidy\n\nChangelog: none, a comment fix with no effect")
    code, out = ce_check(repo)
    assert code == 0, out
    assert "ok: no entry, because: a comment fix with no effect" in out

    ce_git(repo, "commit", "-q", "--amend", "-m", "tidy")
    code, _ = ce_check(repo)
    assert code == 1
    body = repo.parent / "body.txt"
    body.write_text("## What\n\nchangelog: none - test-only change\n")
    code, out = ce_check(repo, "--text", str(body))
    assert code == 0, out
    assert "because: test-only change" in out


def test_a_pull_request_opened_before_the_switch_is_warned_not_failed(repo):
    # the fixture's base commit, which adds changes/README.md, is the switch: 2026-11-01 10:00 UTC
    (repo / "src.c").write_text("int x;\n")
    ce_commit(repo, "code only")
    code, out = ce_check(repo, "--opened", "2026-10-30T08:00:00Z")
    assert code == 0, out
    assert "warning: no changelog entry" in out
    assert "opened before changes/ existed" in out
    code, out = ce_check(repo, "--opened", "2026-11-01T10:00:01Z")
    assert code == 1
    assert "error: no changelog entry" in out


def test_two_entries_fail(repo):
    ce_entry(repo, "R2.1.md", "- **One (R2.1).**\n")
    ce_entry(repo, "R2.2.md", "- **Two (R2.2).**\n")
    ce_commit(repo, "two")
    code, out = ce_check(repo)
    assert code == 1
    assert "2 changelog entries (changes/R2.1.md, changes/R2.2.md): one per pull request" in out


def test_an_entry_written_into_changelog_counts_with_a_warning(repo):
    text = CE_CHANGELOG.replace("## Unreleased\n\n", "## Unreleased\n\n- **Legacy (R3.1).**\n  - detail\n\n")
    (repo / "CHANGELOG.md").write_text(text)
    ce_commit(repo, "a branch cut before the switch")
    code, out = ce_check(repo)
    assert code == 0, out
    assert "warning: the entry is written straight into CHANGELOG.md" in out

    ce_entry(repo, "R3.1.md", "- **Also a file (R3.1).**\n")
    ce_commit(repo, "and a file")
    code, out = ce_check(repo)
    assert code == 1
    assert "2 changelog entries (changes/R3.1.md, 1 in CHANGELOG.md)" in out


def test_rewording_an_entry_already_on_main_is_not_an_entry(repo):
    (repo / "CHANGELOG.md").write_text(CE_CHANGELOG.replace("  - detail\n", "  - better detail\n"))
    ce_commit(repo, "reword")
    code, out = ce_check(repo)
    assert code == 1
    assert "no changelog entry" in out


def test_a_branch_that_merged_main_in_counts_only_its_own_entry(repo):
    ce_entry(repo, "R2.1.md", "- **Mine (R2.1).**\n")
    ce_commit(repo, "mine", when="2026-11-02T10:00:00")
    ce_git(repo, "switch", "-q", "main")
    ce_entry(repo, "R9.9.md", "- **Theirs, merged first (R9.9).**\n")
    ce_commit(repo, "theirs", when="2026-11-02T11:00:00")
    ce_git(repo, "switch", "-q", "pr")
    ce_git(repo, "merge", "-q", "--no-edit", "main")
    code, out = ce_check(repo)
    assert code == 0, out
    assert "ok: one entry, changes/R2.1.md" in out


@pytest.mark.parametrize("name,text,problem", [
    ("R2.1.md", "A sentence, not a list item.\n", "starts with '- '"),
    ("R2.1.md", "- **Entry.**\n\n## v9.9.9\n", "no headings in an entry"),
    ("R2.1.md", "- **One.**\n- **Two.**\n", "2 top-level items"),
    ("R2.1.md", "\n\n", "empty"),
    ("bad name!.md", "- **Entry.**\n", "name it changes/<plan ID or short slug>.md"),
])
def test_a_malformed_entry_fails(repo, name, text, problem):
    ce_entry(repo, name, text)
    ce_commit(repo, "bad entry")
    code, out = ce_check(repo)
    assert code == 1
    assert problem in out


def test_the_release_fold_passes_the_check(repo):
    ce_git(repo, "switch", "-q", "main")
    ce_entry(repo, "R2.1.md", "- **Waiting (R2.1).**\n")
    ce_commit(repo, "R2.1 merged")
    ce_git(repo, "switch", "-q", "-c", "release/1.1.0")
    code, out = ce_run(repo, "fold", "--write", "--changelog", str(repo / "CHANGELOG.md"))
    assert code == 0, out
    ce_commit(repo, "fold")
    code, out = ce_check(repo)
    assert code == 0, out
    assert "ok: a release fold, 1 entry folded" in out


def test_fold_puts_entries_on_top_newest_first_and_deletes_them(repo):
    ce_git(repo, "switch", "-q", "main")
    ce_entry(repo, "R5.1.md", "- **Older (R5.1).**\n  - its detail\n")
    ce_commit(repo, "older", when="2026-11-03T10:00:00")
    ce_entry(repo, "A1.md", "- **Newer (A1).**\n")
    ce_commit(repo, "newer", when="2026-11-04T10:00:00")
    changelog = repo / "CHANGELOG.md"

    code, out = ce_run(repo, "fold", "--changelog", str(changelog))
    assert code == 0, out
    assert "would put 2 entries at the top of ## Unreleased, newest first (changes/A1.md, changes/R5.1.md)" in out
    assert changelog.read_text() == CE_CHANGELOG, "a dry run wrote the file"

    code, out = ce_run(repo, "fold", "--write", "--changelog", str(changelog))
    assert code == 0, out
    assert changelog.read_text() == CE_CHANGELOG.replace(
        "## Unreleased\n\n",
        "## Unreleased\n\n- **Newer (A1).**\n\n- **Older (R5.1).**\n  - its detail\n\n")
    assert sorted(p.name for p in (repo / "changes").iterdir()) == ["README.md"]

    code, out = ce_run(repo, "fold", "--write", "--changelog", str(changelog))
    assert (code, out.strip()) == (0, "nothing to fold: changes/ holds no entries")


def test_fold_into_an_empty_unreleased_section(repo):
    changelog = repo / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\n## Unreleased\n\n## v1.0.0 - 2026-01-01\n\n- **Released.**\n")
    ce_entry(repo, "R2.1.md", "- **Only (R2.1).**\n")
    code, out = ce_run(repo, "fold", "--write", "--changelog", str(changelog))
    assert code == 0, out
    assert changelog.read_text() == (
        "# Changelog\n\n## Unreleased\n\n- **Only (R2.1).**\n\n## v1.0.0 - 2026-01-01\n\n- **Released.**\n")


def test_fold_refuses_without_an_unreleased_heading_or_with_a_bad_entry(repo):
    changelog = repo / "CHANGELOG.md"
    ce_entry(repo, "R2.1.md", "not a list item\n")
    code, out = ce_run(repo, "fold", "--write", "--changelog", str(changelog))
    assert code == 1 and "starts with '- '" in out
    assert (repo / "changes" / "R2.1.md").exists()
    changelog.write_text("# Changelog\n\n## v1.0.0\n")
    code, out = ce_run(repo, "fold", "--write", "--changelog", str(changelog))
    assert code != 0 and "no '## Unreleased' heading" in out


def test_the_repository_changes_folder_follows_the_rule():
    """The format file is not an entry, and every waiting entry is well formed."""
    folder = ROOT / "changes"
    assert (folder / "README.md").is_file()
    spec = importlib.util.spec_from_file_location("changelog_entries", CE_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for p in sorted(folder.glob("*.md")):
        if p.name != "README.md":
            assert mod.entry_problems(f"changes/{p.name}", p.read_text(encoding="utf-8")) == []


# A merge train: one pull request that is main plus the merged heads of several reviewed pull
# requests, its text carrying "Changelog: train". Each first-parent merge is a member and passes
# the rule on its own.
CE_TRAIN_TEXT = "Merge train A: #1 and #2\n\nChangelog: train\n"


def ce_member(repo, name, files, message, when=None):
    ce_git(repo, "switch", "-q", "-c", name, "main")
    for path, text in files.items():
        (repo / path).write_text(text)
    ce_commit(repo, message, when=when)


def ce_train(repo, *members, messages=None):
    ce_git(repo, "switch", "-q", "-C", "train", "main")
    for m in members:
        msg = (messages or {}).get(m, f"Merge {m} into the train")
        ce_git(repo, "merge", "-q", "--no-ff", "-m", msg, m)
    return ce_git(repo, "rev-parse", "train").strip()


def ce_check_train(repo, text=CE_TRAIN_TEXT, head="train"):
    body = repo.parent / "train.txt"
    body.write_text(text)
    return ce_check(repo, "--head", head, "--text", str(body))


def test_a_two_member_train_passes_where_one_pull_request_would_not(repo):
    ce_member(repo, "m1", {"changes/R2.1.md": "- **One (R2.1).**\n"}, "R2.1")
    ce_member(repo, "m2", {"changes/R2.2.md": "- **Two (R2.2).**\n"}, "R2.2")
    ce_train(repo, "m1", "m2")
    code, out = ce_check_train(repo)
    assert code == 0, out
    assert "(Merge m1 into the train): ok: one entry, changes/R2.1.md" in out
    assert "(Merge m2 into the train): ok: one entry, changes/R2.2.md" in out
    assert "ok: a train of 2 members, each passing on its own" in out
    # the same branch without the line is one pull request with two entries, as before
    code, out = ce_check_train(repo, text="Merge train A\n")
    assert code == 1
    assert "2 changelog entries (changes/R2.1.md, changes/R2.2.md): one per pull request" in out


def test_a_train_member_with_two_entries_fails_naming_it(repo):
    ce_member(repo, "m1", {"changes/R2.1.md": "- **One (R2.1).**\n"}, "R2.1")
    ce_member(repo, "m2", {"changes/R2.2.md": "- **Two (R2.2).**\n",
                           "changes/R2.3.md": "- **Three (R2.3).**\n"}, "R2.2 and R2.3")
    head = ce_train(repo, "m1", "m2")
    code, out = ce_check_train(repo)
    assert code == 1, out
    assert (f"error: train member {head[:12]} (Merge m2 into the train): 2 changelog entries "
            "(changes/R2.2.md, changes/R2.3.md): one per pull request") in out
    assert f"1 of 2 train members fail the rule: {head[:12]}" in out


def test_a_train_member_entry_still_holds_one_item(repo):
    ce_member(repo, "m1", {"changes/R2.1.md": "- **One.**\n- **Two.**\n"}, "R2.1")
    head = ce_train(repo, "m1")
    code, out = ce_check_train(repo)
    assert code == 1, out
    assert f"train member {head[:12]} (Merge m1 into the train): changes/R2.1.md: 2 top-level items" in out


def test_a_train_with_a_plain_commit_on_top_fails(repo):
    ce_member(repo, "m1", {"changes/R2.1.md": "- **One (R2.1).**\n"}, "R2.1")
    ce_train(repo, "m1")
    (repo / "src.c").write_text("int x;\n")
    ce_commit(repo, "a fix-up on the train")
    top = ce_git(repo, "rev-parse", "train").strip()
    code, out = ce_check_train(repo)
    assert code == 1, out
    assert f"{top[:12]} (a fix-up on the train) on the first-parent line is not a merge commit" in out


def test_the_train_line_without_merge_commits_is_refused(repo):
    ce_entry(repo, "R2.1.md", "- **One (R2.1).**\n")
    ce_commit(repo, "R2.1")
    code, out = ce_check_train(repo, head="pr")
    assert code == 1, out
    assert "(R2.1) on the first-parent line is not a merge commit" in out
    ce_git(repo, "switch", "-q", "-c", "empty", "main")
    code, out = ce_check_train(repo, head="empty")
    assert code == 1, out
    assert "'Changelog: train', but the pull request has no commits" in out


def test_a_train_member_reason_is_its_own_not_the_trains(repo):
    ce_member(repo, "m1", {"changes/R2.1.md": "- **One (R2.1).**\n"}, "R2.1")
    ce_member(repo, "m2", {"src.c": "int x;\n"}, "code only")
    head = ce_train(repo, "m1", "m2")
    code, out = ce_check_train(repo, text=CE_TRAIN_TEXT + "Changelog: none, the members carry theirs\n")
    assert code == 1, out
    assert f"error: train member {head[:12]} (Merge m2 into the train): no changelog entry" in out
    # the member's merge commit is one of its own commits, so a reason there counts
    ce_train(repo, "m1", "m2", messages={"m2": "Merge m2 into the train\n\nChangelog: none, a test-only change"})
    code, out = ce_check_train(repo)
    assert code == 0, out
    assert "(Merge m2 into the train): ok: no entry, because: a test-only change" in out


def test_a_train_member_cut_before_the_switch_is_warned_not_failed(repo):
    # the fixture's base commit, which adds changes/README.md, is the switch: 2026-11-01 10:00 UTC
    ce_member(repo, "m1", {"src.c": "int x;\n"}, "cut before the switch", when="2026-10-30T08:00:00")
    ce_train(repo, "m1")
    code, out = ce_check_train(repo)
    assert code == 0, out
    assert "warning: train member" in out and "no changelog entry" in out
    assert "its branch was cut before changes/ existed" in out



@needs_git_checkout
def test_release_check_requires_the_full_cuda_gate_for_this_release(monkeypatch, tmp_path, capsys):
    """R6.44: release-cuda-gate.py's smoke and probe pass on a binary whose
    own device targets are red (R3.56's check was, from 2026-11-04 to
    2026-11-10). The release needs scripts/cuda-gate.sh's pass for itself."""
    _pinned(monkeypatch)
    args = good_args(tmp_path)
    assert check_release.check(args) is True
    args.device_evidence = device_ledger(tmp_path, cuda_version="1.1.4")
    assert check_release.check(args) is False
    assert "no full CUDA gate recorded for 1.2.0" in capsys.readouterr().err
    # a row for this version written by anything but the full gate does not count
    args.device_evidence = device_ledger(tmp_path, cuda_ran="scripts/release-cuda-gate.py")
    assert check_release.check(args) is False


def test_release_check_refuses_a_full_gate_row_without_its_summary(monkeypatch, tmp_path, capsys):
    _pinned(monkeypatch)
    args = good_args(tmp_path)
    args.device_evidence = device_ledger(tmp_path, cuda_summary=None)
    assert check_release.check(args) is False
    assert "missing or unreadable" in capsys.readouterr().err


@pytest.mark.parametrize("summary", ["fail", "other-commit"])
def test_release_check_refuses_a_summary_that_disagrees_with_its_row(monkeypatch, tmp_path, capsys,
                                                                     summary):
    _pinned(monkeypatch)
    args = good_args(tmp_path)
    args.device_evidence = device_ledger(tmp_path, cuda_summary=summary)
    assert check_release.check(args) is False
    assert "does not record the row's pass" in capsys.readouterr().err


@pytest.mark.skipif(shutil.which("make") is None or shutil.which("sh") is None,
                    reason="needs make and sh")
@pytest.mark.skipif(sys.platform == "win32",
                    reason="the wrapper is a POSIX sh script: Windows process creation resolves "
                           "`make` to make.exe and never runs it, so this test would call the real "
                           "make (main went red on the Windows suite, 2026-11-10); the wrapper is "
                           "for scripts/cuda-gate.sh on the Linux CUDA host")
def test_release_make_puts_the_recipe_cflags_on_makes_command_line(tmp_path):
    """tools/release-make/make, first on PATH, is how scripts/cuda-gate.sh runs
    on a release binary without a step relinking it (R6.44): its CFLAGS must be
    release.yml's own x86-64 line, on make's command line, and the wrapper must
    find the real make rather than itself."""
    import re
    line = next(l for l in (ROOT / ".github/workflows/release.yml").read_text().splitlines()
                if 'make CFLAGS="' in l)
    recipe = re.search(r'make CFLAGS="([^"]*)"', line).group(1)
    arch = re.search(r"arch_flags: (-march=x86-64-v3)",
                     (ROOT / ".github/workflows/release.yml").read_text()).group(1)
    want = recipe.replace("${{ matrix.arch_flags }}", arch)
    write(tmp_path / "Makefile", 'CFLAGS += -DAPPENDED\nall:\n\t@echo "$(CFLAGS)"\n')
    env = {**os.environ, "PATH": f"{ROOT / 'tools' / 'release-make'}{os.pathsep}{os.environ['PATH']}"}
    env.pop("CFLAGS", None)
    out = subprocess.run(["make", "-s", "-C", str(tmp_path)], env=env, capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stderr
    # a command-line CFLAGS is not extended by the Makefile's +=, so the objects are the recipe's
    assert out.stdout.strip() == want, (out.stdout, want)
