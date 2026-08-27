import argparse
import importlib.util
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
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
        tag="v0.1.3-alpha",
        binary=tmp_path / "runner",
        readme=write(
            tmp_path / "README.md",
            "Public alpha (`0.1.3-alpha`)\n"
            "./runner --version   # -> runner 0.1.3-alpha\n",
        ),
        changelog=write(tmp_path / "CHANGELOG.md", "## v0.1.3-alpha\n"),
        build_info=write(
            tmp_path / "BUILD-INFO.txt",
            "runner 0.1.3-alpha\n"
            "tag:        v0.1.3-alpha\n"
            "commit:     abc123\n",
        ),
        release_workflow=write(
            tmp_path / "release.yml", "# release v0.1.3-alpha\n"
        ),
        python_pyproject=write(
            tmp_path / "pyproject.toml",
            '[project]\nversion = "0.1.3a0"\n',
        ),
        commit="abc123",
        current_docs=[],
        compat_reports=report_dir(tmp_path, "0.1.3-alpha-2026-01-01.json"),
    )


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
        write(d / name, "{}\n")
    return d


def test_release_check_accepts_consistent_artifacts(monkeypatch, tmp_path):
    args = good_args(tmp_path)
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha"
    )
    assert check_release.check(args)


def test_release_check_rejects_stale_readme(monkeypatch, tmp_path, capsys):
    args = good_args(tmp_path)
    args.readme.write_text(
        args.readme.read_text() + "old example 0.1.2-alpha\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha"
    )
    assert not check_release.check(args)
    assert "stale release string '0.1.2-alpha'" in capsys.readouterr().err


def test_release_check_rejects_build_commit_drift(monkeypatch, tmp_path, capsys):
    args = good_args(tmp_path)
    args.commit = "different"
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha"
    )
    assert not check_release.check(args)
    assert "BUILD-INFO commit line is inconsistent" in capsys.readouterr().err


def test_release_check_rejects_python_version_drift(monkeypatch, tmp_path, capsys):
    args = good_args(tmp_path)
    args.python_pyproject.write_text(
        '[project]\nversion = "0.1.2a0"\n', encoding="utf-8"
    )
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha"
    )
    assert not check_release.check(args)
    assert "Python package version" in capsys.readouterr().err


def test_release_check_rejects_stale_current_document(monkeypatch, tmp_path, capsys):
    args = good_args(tmp_path)
    args.current_docs = [write(tmp_path / "SECURITY.md", "public 0.1.2-alpha\n")]
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha"
    )
    assert not check_release.check(args)
    assert "current document" in capsys.readouterr().err


# The ledger is only credible if every release ships one. The reports existed
# from 2026-08-13 but nothing made producing them part of cutting a release,
# so the habit depended on somebody remembering.

def test_release_check_requires_a_compat_report_for_this_release(
        monkeypatch, tmp_path, capsys):
    args = good_args(tmp_path)
    args.compat_reports = report_dir(tmp_path, "0.1.2-alpha-2026-01-01.json")
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha"
    )
    assert not check_release.check(args)
    assert "compat report" in capsys.readouterr().err


def test_release_check_accepts_any_dated_report_for_this_release(
        monkeypatch, tmp_path):
    args = good_args(tmp_path)
    args.compat_reports = report_dir(
        tmp_path, "0.1.2-alpha-2026-01-01.json", "0.1.3-alpha-2026-02-09.json"
    )
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha"
    )
    assert check_release.check(args)


def test_release_check_skips_the_report_gate_when_no_directory_is_given(
        monkeypatch, tmp_path):
    """An older checkout, or a caller that does not pass the flag, still
    releases: the gate binds the process, it does not break the tool."""
    args = good_args(tmp_path)
    args.compat_reports = None
    monkeypatch.setattr(
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha"
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
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha"
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
        check_release, "binary_version", lambda _: "runner 0.1.3-alpha")
    assert not check_release.check(args)
    assert "release workflow" in capsys.readouterr().err


def test_current_doc_replaces_the_default_set_rather_than_adding_to_it(tmp_path):
    """argparse APPENDS to a non-empty default, so a caller narrowing the scan
    silently widened it -- and picked up whichever of the three defaults had
    since been deleted from the tree as a FileNotFoundError."""
    only = tmp_path / "ONE.md"
    write(only, "x\n")
    parsed = _parse(["--tag", "v0.1.3-alpha", "--binary", "b",
                     "--build-info", "bi", "--commit", "c",
                     "--current-doc", str(only)])
    assert [str(p) for p in parsed.current_docs] == [str(only)]
    default = _parse(["--tag", "v0.1.3-alpha", "--binary", "b",
                      "--build-info", "bi", "--commit", "c"])
    assert len(default.current_docs) == 3
