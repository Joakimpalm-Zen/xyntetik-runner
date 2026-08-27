#!/usr/bin/env python3
"""Check published torture-result runtime versions against upstream releases.

This performs metadata-only registry requests. It never starts a runtime,
loads a model, or runs inference. Registry failures are reported as skips so a
transient network outage cannot make the scheduled workflow red.

Significance policy (2026-08-25): published comparison rows are DATED
SNAPSHOTS, per the README's own framing - they claim what was measured on a
date, not what current upstream does. So drift only counts as STALE when it
could plausibly change a conclusion: a major/minor version jump for semver
runtimes, or a large build-number gap for llama.cpp. Patch releases land
weekly and rebenchmarking per patch is an unmeetable standard nobody's
claims depend on; patch-level drift is still REPORTED (in `drift`) so the
JSON stays honest, it just does not page a human.
"""

import argparse
import json
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "tests" / "torture" / "results"
REGISTRIES = {
    "llama.cpp": "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
    "ollama": "https://api.github.com/repos/ollama/ollama/releases/latest",
    "vllm": "https://pypi.org/pypi/vllm/json",
}


def normalized_version(runtime, value):
    parts = str(value).strip().split()
    if not parts:
        raise ValueError(f"empty {runtime} release {value!r}")
    text = parts[0].removeprefix("v")
    if runtime == "llama.cpp":
        match = re.fullmatch(r"b(\d+)", text)
        if not match:
            raise ValueError(f"invalid llama.cpp release {value!r}")
        return (int(match.group(1)),), f"b{int(match.group(1))}"
    match = re.fullmatch(r"(\d+(?:\.\d+)+)(?:[-+].*)?", text)
    if not match:
        raise ValueError(f"invalid {runtime} release {value!r}")
    canonical = match.group(1)
    return tuple(int(part) for part in canonical.split(".")), canonical


def read_published(results_dir=RESULTS):
    """Return the newest published row for each tracked competitor."""
    newest = {}
    for path in sorted(Path(results_dir).glob("**/report.json")):
        try:
            runtime = json.loads(path.read_text())["runtime"]
            name, raw = runtime["name"], runtime["version"]
            if name not in REGISTRIES:
                continue
            key, version = normalized_version(name, raw)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid published report {path}: {exc}") from exc
        previous = newest.get(name)
        if previous is None or key > previous["key"]:
            newest[name] = {"key": key, "version": version,
                            "reports": [str(path)]}
        elif key == previous["key"]:
            previous["reports"].append(str(path))
    return {name: {"version": row["version"], "reports": row["reports"]}
            for name, row in sorted(newest.items())}


def fetch_json(url):
    request = Request(url, headers={"Accept": "application/json",
                                    "User-Agent": "xyntetik-runner-freshness"})
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def upstream_version(runtime, payload):
    raw = payload["info"]["version"] if runtime == "vllm" else payload["tag_name"]
    return normalized_version(runtime, raw)


# llama.cpp ships several builds a day; a gap under this many builds is
# routine cadence, not a conclusion-threatening divergence
LLAMA_CPP_BUILD_GAP = 300


def significant_drift(runtime, published_key, upstream_key):
    """True when the gap could plausibly change a benchmark conclusion."""
    if published_key >= upstream_key:
        return False
    if runtime == "llama.cpp":
        return upstream_key[0] - published_key[0] >= LLAMA_CPP_BUILD_GAP
    # semver runtimes: major or minor jump counts, a patch bump does not
    return published_key[:2] < upstream_key[:2]


def check_freshness(published, fetch=fetch_json):
    result = {"stale": [], "drift": [], "current": [], "skipped": []}
    for runtime, row in sorted(published.items()):
        try:
            payload = fetch(REGISTRIES[runtime])
            upstream_key, upstream = upstream_version(runtime, payload)
            published_key, _ = normalized_version(runtime, row["version"])
        except Exception as exc:  # registry/network/schema failures are skips
            result["skipped"].append({"runtime": runtime, "reason": str(exc)})
            continue
        finding = {"runtime": runtime, "published": row["version"],
                   "upstream": upstream, "reports": row["reports"]}
        if significant_drift(runtime, published_key, upstream_key):
            result["stale"].append(finding)
        elif published_key < upstream_key:
            result["drift"].append(finding)
        else:
            result["current"].append(finding)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        published = read_published(args.results)
    except ValueError as exc:
        parser.error(str(exc))
    if not published:
        # Path.glob raises nothing on a directory that is missing or no longer
        # holds report.json, so a reorganisation of tests/torture/results would
        # leave this permanently green with zero rows. A ledger gate with
        # nothing to check is a configuration error -- exit 2, which the
        # scheduled workflow fails on and does not mistake for "stale".
        parser.error(f"no report.json under {args.results}: nothing to check")
    result = check_freshness(published)
    # Every row skipped means nothing was COMPARED, which is not the same
    # answer as "nothing is stale". check_freshness catches a renamed registry
    # field exactly as it catches a DNS failure, so an upstream schema change
    # or a run of rate-limited API calls lands every runtime in `skipped` --
    # and the workflow only reacts to exit 1, so the weekly job would stay
    # green forever. Exit 2 for the same reason the empty-results gate above
    # does: a ledger gate with nothing to check is a configuration error.
    if result["skipped"] and not (result["stale"] or result["drift"] or
                                  result["current"]):
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        reasons = "; ".join(f"{row['runtime']}: {row['reason']}"
                            for row in result["skipped"])
        parser.error(f"every registry lookup failed, so nothing was "
                     f"compared ({reasons})")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for status in ("stale", "current", "skipped"):
            for row in result[status]:
                if status == "skipped":
                    print(f"SKIP {row['runtime']}: {row['reason']}")
                else:
                    print(f"{status.upper()} {row['runtime']}: published "
                          f"{row['published']}, upstream {row['upstream']}")
    return 1 if result["stale"] else 0


if __name__ == "__main__":
    sys.exit(main())
