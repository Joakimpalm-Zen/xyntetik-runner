"""Speculative telemetry reports execution, not a configured draft."""

import json
import pathlib
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import RunnerServer, find_runner  # noqa: E402


def _post_file(server, name):
    payload = (ROOT / "tests" / "fixtures" / name).read_bytes()
    request = urllib.request.Request(
        server.base_url + "/v1/completions", data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def test_logprobs_reports_speculation_as_not_engaged():
    """Logprob capture selects the solo walk even when --draft is loaded."""
    model = ROOT / "test.gguf"
    with RunnerServer(
        find_runner(ROOT), model, ctx=256, parallel=1,
        extra_args=["--gpu", "off", "--draft", str(model)],
    ) as server:
        plain = _post_file(server, "completion_spec.json")
        assert plain["runner_telemetry"]["speculative"] is True

        logged = _post_file(server, "completion_spec_logprobs.json")
        assert logged["choices"][0]["logprobs"]["token_logprobs"]
        assert logged["runner_telemetry"]["speculative"] is False


def _http(server, path, payload=None):
    data = b"" if path == "/unload" else (
        json.dumps(payload).encode() if payload is not None else None)
    req = urllib.request.Request(server.base_url + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


@pytest.mark.parametrize("parallel", [1, 2])
@pytest.mark.parametrize("source", ["lookup", "mtp", "model", "refused"])
def test_capabilities_report_resident_drafting(tmp_path, parallel, source):
    """Loaded, configured heads are active; an absent or refused draft is
    not. Requests independently prove the advertised path is being driven.
    """
    model = tmp_path / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    "--mtp-layers", "1", str(model)], check=True,
                   stdout=subprocess.DEVNULL)
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"not a model")
    flags = {"lookup": ["--draft-lookup"], "mtp": ["--mtp"],
             "model": ["--draft", str(model)],
             "refused": ["--draft", str(bad)]}[source]
    with RunnerServer(find_runner(ROOT), model, ctx=256, parallel=parallel,
                      extra_args=["--gpu", "off", "-t", "2", *flags]) as srv:
        expected = {"requested": True, "active": source != "refused"}
        if source != "refused":
            expected["source"] = source
        caps = _http(srv, "/v1/capabilities")
        if source == "refused":
            assert caps["draft"].pop("reason", None), "refusal needs a reason"
        assert caps["draft"] == expected
        assert caps["mtp"]["consumed"] is (source == "mtp")
        payload = {"prompt": "hello there", "max_tokens": 24,
                   "temperature": 0, "seed": 1}
        for _ in range(2):
            out = _http(srv, "/v1/completions", payload)["runner_telemetry"]
            assert out["speculative"] is (source != "refused")
            if source != "refused":
                assert out["speculation"]["source"] == source
                assert out["speculation"]["rounds"] > 0
        if parallel == 1:
            assert _http(srv, "/unload")["status"] == "ok"
            idle = _http(srv, "/v1/capabilities")
            assert idle["draft"]["active"] is False
            assert "source" not in idle["draft"]
            assert idle["mtp"]["consumed"] is False
            _http(srv, "/v1/completions", payload)
            loaded = _http(srv, "/v1/capabilities")
            loaded["draft"].pop("reason", None)
            assert loaded["draft"] == expected
