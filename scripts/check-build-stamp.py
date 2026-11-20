#!/usr/bin/env python3
"""Verify that a built runner binary reports a clean git build stamp.

Starts the binary as a temporary server with --gpu off, reads
GET /v1/runner/provenance, and asserts that build.commit matches the
checked-out commit and build.dirty is False.

Also exposes --extract-release-build <os> so CI workflows can run the exact
compiler recipe declared in .github/workflows/release.yml without duplicating
flags by hand.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from _process import RunnerServer  # noqa: E402


def resolve_expected_commit(explicit):
    if explicit:
        return explicit.strip()
    env_sha = os.environ.get("GITHUB_SHA", "").strip()
    if env_sha:
        return env_sha
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise SystemExit(
            f"check-build-stamp: could not determine expected commit: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def extract_release_build(target_os, workflow_path=ROOT / ".github/workflows/release.yml"):
    """Return the exact `make ...` build command from release.yml for `target_os`."""
    text = Path(workflow_path).read_text(encoding="utf-8")
    matrix_re = re.compile(
        r"-\s+os:\s+(\S+)\s*\n\s+artifact:\s+\S+\s*\n\s+binary:\s+\S+\s*\n\s+arch_flags:\s+(\S+)"
    )
    arch_map = {m.group(1): m.group(2) for m in matrix_re.finditer(text)}
    if target_os not in arch_map:
        raise SystemExit(
            f"check-build-stamp: os {target_os!r} not found in {workflow_path} matrix ({sorted(arch_map)})"
        )
    build_m = re.search(
        r"-\s+name:\s+build\s*\n\s+run:\s+([^\n]+)",
        text,
    )
    if not build_m:
        raise SystemExit(f"check-build-stamp: '- name: build' recipe not found in {workflow_path}")
    cmd = build_m.group(1).strip()
    return cmd.replace("${{ matrix.arch_flags }}", arch_map[target_os])


def check_binary(binary_path, model_path, expected_commit):
    binary = Path(binary_path).resolve()
    if not binary.is_file():
        raise SystemExit(f"check-build-stamp: binary not found: {binary}")

    with tempfile.TemporaryDirectory(prefix="build-stamp-") as td:
        td_path = Path(td)
        if model_path is None:
            model = td_path / "test.gguf"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/make-test-model.py"), str(model)],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        else:
            model = Path(model_path)
            if not model.is_file():
                raise SystemExit(f"check-build-stamp: model not found: {model}")

        log_path = td_path / "server.log"
        with RunnerServer(
            str(binary),
            str(model),
            ctx=512,
            parallel=1,
            extra_args=["--gpu", "off", "-t", "1"],
            log_path=str(log_path),
        ) as srv:
            with urllib.request.urlopen(srv.base_url + "/v1/runner/provenance", timeout=30) as resp:
                if resp.status != 200:
                    raise SystemExit(f"check-build-stamp: /v1/runner/provenance returned HTTP {resp.status}")
                doc = json.load(resp)

    build = doc.get("build")
    if not isinstance(build, dict):
        raise SystemExit(f"check-build-stamp: missing 'build' object in provenance response: {doc!r}")

    print(f"build-stamp: expected={expected_commit} build={json.dumps(build, sort_keys=True)}")
    commit = build.get("commit")
    dirty = build.get("dirty")
    errors = []
    if commit != expected_commit:
        errors.append(f"build.commit={commit!r} (expected {expected_commit!r})")
    if dirty is not False:
        errors.append(f"build.dirty={dirty!r} (expected False)")
    if errors:
        raise SystemExit("check-build-stamp: FAIL: " + "; ".join(errors))
    print("build-stamp: ok")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", help="Path to runner binary")
    parser.add_argument("--model", help="Optional path to GGUF model (builds a tiny test.gguf if omitted)")
    parser.add_argument("--commit", help="Expected 40-hex commit SHA (defaults to GITHUB_SHA or git HEAD)")
    parser.add_argument(
        "--extract-release-build",
        metavar="OS",
        help="Print the release.yml build command for OS (e.g. ubuntu-24.04, macos-latest, windows-latest)",
    )
    args = parser.parse_args(argv)

    if args.extract_release_build:
        print(extract_release_build(args.extract_release_build))
        return 0

    if not args.binary:
        parser.error("--binary is required unless --extract-release-build is set")

    expected = resolve_expected_commit(args.commit)
    return check_binary(args.binary, args.model, expected)


if __name__ == "__main__":
    raise SystemExit(main())
