import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "competitor_freshness", ROOT / "scripts" / "competitor-freshness.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _report(root, directory, runtime, version):
    path = root / directory / "report.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "runtime": {"name": runtime, "version": version}}))
    return path


def test_uses_newest_published_row_and_ignores_runner(tmp_path):
    _report(tmp_path, "old", "vllm", "0.25.0")
    newest = _report(tmp_path, "new", "vllm", "0.26.0 (--tool-call-parser hermes)")
    _report(tmp_path, "own", "runner", "runner 0.1.5-alpha")
    rows = MOD.read_published(tmp_path)
    assert rows == {"vllm": {"version": "0.26.0", "reports": [str(newest)]}}


def test_reports_stale_versions_from_official_registry_payloads(tmp_path):
    _report(tmp_path, "llama", "llama.cpp", "b10076")
    _report(tmp_path, "ollama", "ollama", "0.32.1")
    _report(tmp_path, "vllm", "vllm", "0.26.0")
    payloads = {
        MOD.REGISTRIES["llama.cpp"]: {"tag_name": "b10077"},
        MOD.REGISTRIES["ollama"]: {"tag_name": "v0.32.1"},
        MOD.REGISTRIES["vllm"]: {"info": {"version": "0.27.0"}},
    }
    result = MOD.check_freshness(
        MOD.read_published(tmp_path), lambda url: payloads[url])
    # significance policy: a 1-build llama.cpp gap is routine cadence
    # (reported as drift, does not page a human); a vllm MINOR jump is stale
    assert [(row["runtime"], row["published"], row["upstream"])
            for row in result["stale"]] == [
                ("vllm", "0.26.0", "0.27.0")]
    assert [(row["runtime"], row["published"], row["upstream"])
            for row in result["drift"]] == [
                ("llama.cpp", "b10076", "b10077")]
    assert [row["runtime"] for row in result["current"]] == ["ollama"]
    assert result["skipped"] == []
    # the policy's own boundaries
    assert MOD.significant_drift("llama.cpp", (10076,), (10400,))
    assert not MOD.significant_drift("llama.cpp", (10076,), (10176,))
    assert not MOD.significant_drift("ollama", (0, 32, 14), (0, 32, 15))
    assert MOD.significant_drift("ollama", (0, 32, 14), (0, 33, 0))


def test_unreachable_registry_is_a_skip_not_stale(tmp_path):
    _report(tmp_path, "ollama", "ollama", "0.32.1")

    def unavailable(_url):
        raise OSError("temporary DNS failure")

    result = MOD.check_freshness(MOD.read_published(tmp_path), unavailable)
    assert result["stale"] == []
    assert result["current"] == []
    assert result["skipped"][0]["runtime"] == "ollama"
    assert "temporary DNS failure" in result["skipped"][0]["reason"]


def test_a_results_directory_with_no_reports_is_a_configuration_error(tmp_path):
    """Path.glob on a missing or reorganised directory yields nothing and
    raises nothing, so the scheduled workflow would have gone green forever the
    moment the reports moved. The workflow treats >1 as a hard failure and only
    opens the stale-results issue on exactly 1, so exit 2 is the right answer:
    nothing was checked, and nothing is stale either."""
    import pytest

    with pytest.raises(SystemExit) as caught:
        MOD.main(["--results", str(tmp_path / "moved-away"), "--json"])

    assert caught.value.code == 2


def test_a_populated_results_directory_still_runs(tmp_path):
    _report(tmp_path, "vllm", "vllm", "0.26.0")
    assert MOD.read_published(tmp_path)


def test_every_registry_failing_is_a_configuration_error_not_a_pass(tmp_path):
    """A run that compared nothing is not a run that found nothing stale.

    `except Exception` catches a renamed registry field exactly as it catches
    a DNS failure, so a schema change upstream or three rate-limited GitHub
    calls from an unauthenticated scheduled runner put every row in `skipped`
    -- and the workflow only reacts to exit 1, so the weekly job stays green
    forever with a JSON artifact nobody opens. This is the same reasoning the
    empty-results gate already applies one function over: a ledger gate with
    nothing to check is a configuration error.
    """
    import pytest

    _report(tmp_path, "ollama", "ollama", "0.32.1")
    _report(tmp_path, "vllm", "vllm", "0.26.0")

    def unavailable(_url):
        raise OSError("temporary DNS failure")

    original = MOD.check_freshness
    MOD.check_freshness = lambda published: original(published, unavailable)
    try:
        with pytest.raises(SystemExit) as caught:
            MOD.main(["--results", str(tmp_path), "--json"])
    finally:
        MOD.check_freshness = original
    assert caught.value.code == 2


def test_a_blank_published_version_is_a_report_error_not_a_stale_verdict(tmp_path):
    """`"".split()[0]` raises IndexError, which is not in read_published's
    except tuple and not in main's either, so it escaped as a traceback with
    exit 1 -- the code the workflow reserves for "a published row is stale".
    The workflow then parses an empty JSON file and fails for a reason
    unrelated to the actual cause."""
    import pytest

    _report(tmp_path, "vllm", "vllm", "   ")
    with pytest.raises(SystemExit) as caught:
        MOD.main(["--results", str(tmp_path), "--json"])
    assert caught.value.code == 2
