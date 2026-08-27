"""Opt-in update check: is a newer runner release published?

Doctrine, settled 2026-08-26 after the investigation this module closes:

* OPT-IN ONLY. Nothing in this package or the runner binary ever calls
  this implicitly. The runner is offline-first; a startup phone-home
  would break the no-network promise, and there is no throttled
  "only sometimes" variant because a promise with exceptions is not a
  promise. You run the check when you ask for it:

      python -m xyntetik_runner.update_check           # binary on PATH
      python -m xyntetik_runner.update_check 0.3.0     # explicit version

* NOTIFY ONLY. This module prints a line; it never downloads, installs,
  or modifies anything.

* WHAT IS SENT: one anonymous GET to the public GitHub Releases API for
  this repository. No identifiers are added beyond what any HTTP client
  sends. If that is still too chatty for your environment, do not run
  it; the Releases page in a browser answers the same question.

The C binary deliberately does not implement this: a single-binary C11
program with no dependencies has no TLS, and shelling out to curl from
the runtime would smuggle a network dependency into a binary whose
whole point is not having one.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional, cast

RELEASES_LATEST = ("https://api.github.com/repos/"
                   "Joakimpalm-Zen/xyntetik-runner/releases/latest")


@dataclass
class UpdateStatus:
    current: str
    latest: str
    newer_available: bool
    url: str


def _version_tuple(tag: str) -> tuple[int, ...]:
    """v0.1.20-alpha -> (0, 1, 20). Pre-release suffixes are ignored for
    ordering: this is a notify-only check, and 'a newer numbered tag
    exists' is the honest bar it answers."""
    m = re.match(r"v?(\d+)\.(\d+)\.(\d+)", tag.strip())
    if not m:
        raise ValueError(f"unparseable version: {tag!r}")
    return tuple(int(g) for g in m.groups())


def _fetch_latest() -> dict[str, Any]:
    with urllib.request.urlopen(RELEASES_LATEST, timeout=10) as r:
        return cast(dict[str, Any], json.load(r))


def check_latest(current: str,
                 fetch: Callable[[], dict[str, Any]] = _fetch_latest) -> UpdateStatus:
    """Compare `current` against the latest published release tag.

    `fetch` is injectable so tests never touch the network. Raises on
    network failure or unparseable versions rather than guessing: an
    update check that silently reports "up to date" when offline would
    be the quiet lie this project does not tell.
    """
    data = fetch()
    tag = data["tag_name"]
    return UpdateStatus(
        current=current,
        latest=tag,
        newer_available=_version_tuple(tag) > _version_tuple(current),
        url=data.get("html_url",
                     "https://github.com/Joakimpalm-Zen/xyntetik-runner/releases"),
    )


def _binary_version() -> Optional[str]:
    try:
        out = subprocess.run(["runner", "--version"], capture_output=True,
                             text=True, timeout=10).stdout
        m = re.search(r"(\d+\.\d+\.\d+(?:-[A-Za-z0-9.]+)?)", out)
        return m.group(1) if m else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def main(argv: list[str]) -> int:
    current = argv[1] if len(argv) > 1 else _binary_version()
    if not current:
        print("could not determine the installed version: pass it explicitly, "
              "e.g. `python -m xyntetik_runner.update_check 0.3.0`",
              file=sys.stderr)
        return 2
    try:
        st = check_latest(current)
    except Exception as e:  # offline, rate-limited, unparseable: say so
        print(f"update check failed (this is fine offline): {e}",
              file=sys.stderr)
        return 3
    if st.newer_available:
        print(f"newer release available: {st.latest} "
              f"(you have {st.current}) -> {st.url}")
        return 1
    print(f"up to date: {st.current} is the latest published release")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
