"""needs_confirmation (R2.1.3): a constrained turn reports how sure the model
was at the choices the grammar shaped, on every chat surface.

`confirm_below: p` asks for runner_telemetry.decision: over the turn's
grammar-shaped decisions (a step where the grammar removed some probed
candidates, so a choice between branches, not free text), the lowest
posterior the chosen token had among the legal ones, its margin, where it
fell, and needs_confirmation = that posterior < p.
"""
import json
import pathlib
import string
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import RunnerServer, find_runner  # noqa: E402

# Twelve tools whose names differ in their first byte: the fixture's
# random weights put two legal bytes inside the probed top candidates only
# when there are many of them (tests/conformance/test_choice_logprobs.py).
TOOLS = [{"type": "function", "function": {
            "name": c + "_tool", "parameters": {"type": "object",
                                                "properties": {"x": {"type": "integer"}},
                                                "required": ["x"]}}}
         for c in string.ascii_lowercase[:12]]


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    model = tmp_path_factory.mktemp("cb") / "m.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(model)],
                   check=True, stdout=subprocess.DEVNULL)
    with RunnerServer(exe, model, ctx=8192, extra_args=[
            "--gpu", "off", "-t", "2", "--chat-template", "chatml"]) as s:
        yield s


def _post(srv, path, body):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def _chat(srv, **extra):
    body = {"messages": [{"role": "user", "content": "pick one"}],
            "tools": TOOLS, "tool_choice": "required", "temperature": 0.7,
            "seed": 3, "max_tokens": 40, "choice_logprobs_probe": 64}
    body.update(extra)
    return _post(srv, "/v1/chat/completions", body)


def test_the_answer_is_consistent_with_its_threshold(srv):
    code, lo = _chat(srv, confirm_below=1e-6)
    assert code == 200, lo
    d = lo["runner_telemetry"]["decision"]
    assert d["decisions"] >= 1           # at least the tool name was a choice
    assert 0 <= d["min_chosen_prob"] <= 1 and d["min_margin"] <= d["min_chosen_prob"]
    assert d["needs_confirmation"] == (d["min_chosen_prob"] < 1e-6)
    code, hi = _chat(srv, confirm_below=0.999999)
    h = hi["runner_telemetry"]["decision"]
    # the same seeded turn: the same decisions, judged against another line
    assert h["min_chosen_prob"] == d["min_chosen_prob"]
    assert h["needs_confirmation"] == (h["min_chosen_prob"] < 0.999999)
    # asking for the answer does not print the per-step records
    assert "choice_logprobs" not in hi["choices"][0]


def test_every_chat_surface_answers(srv):
    code, r = _post(srv, "/v1/responses", {
        "input": "pick one", "tool_choice": "required", "max_output_tokens": 40,
        "temperature": 0, "confirm_below": 0.5,
        "tools": [{"type": "function", "name": t["function"]["name"],
                   "parameters": t["function"]["parameters"]} for t in TOOLS]})
    assert code == 200 and "decision" in r["runner_telemetry"], r
    code, m = _post(srv, "/v1/messages", {
        "max_tokens": 40, "temperature": 0, "confirm_below": 0.5,
        "messages": [{"role": "user", "content": "pick one"}],
        "tool_choice": {"type": "any"},
        "tools": [{"name": t["function"]["name"],
                   "input_schema": t["function"]["parameters"]} for t in TOOLS]})
    assert code == 200 and "decision" in m["runner_telemetry"], m


@pytest.mark.parametrize("extra,why", [
    ({"confirm_below": 0}, "probability"),
    ({"confirm_below": 1}, "probability"),
    ({"confirm_below": "high"}, "probability"),
    ({"confirm_below": 0.5, "stream": True}, "buffered"),
])
def test_unusable_requests_are_refused(srv, extra, why):
    code, err = _chat(srv, **extra)
    assert code == 400 and why in err["error"]["message"], err


def test_an_unconstrained_turn_has_nothing_to_report(srv):
    code, err = _post(srv, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
        "confirm_below": 0.5})
    assert code == 400 and "constrained" in err["error"]["message"], err
