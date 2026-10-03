"""R4.12.3: the engine's reasoning reserve, held against the constant the
agent-torture harness excuses by.

Under a tool choice or schema the engine closes a reasoning prelude at half
of max_tokens and spends the rest on the call (prelude_max in src/engine.c),
so only a budget too small to split ends with no call. scripts/agent-torture.py
excuses exactly that budget (UNSPLITTABLE_BUDGET). The two are tied here by
behaviour, not by a comment: on a reasoning template with a required tool, the
largest budget that returns no call must equal the harness constant, and every
larger budget must return a call. If the engine's split rule changes, this
fails before the harness can excuse the wrong cases.
"""
import importlib.util
import json
import pathlib
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import RunnerServer, find_runner  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "agent_torture", ROOT / "scripts" / "agent-torture.py")
TORTURE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TORTURE)

TOOLS = [{"type": "function", "function": {
    "name": "dispatch_job",
    "parameters": {"type": "object", "properties": {"job": {"type": "string"}},
                   "required": ["job"]}}}]


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    model = tmp_path_factory.mktemp("rr") / "m.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(model)],
                   check=True, stdout=subprocess.DEVNULL)
    # granite42 opens a reasoning channel on the generation prompt
    with RunnerServer(exe, model, ctx=2048, extra_args=[
            "--gpu", "off", "-t", "2", "--chat-template", "granite42"]) as s:
        yield s


def _turn(srv, budget):
    body = {"messages": [{"role": "user", "content": "run the job"}],
            "tools": TOOLS, "tool_choice": "required", "temperature": 0,
            "max_tokens": budget}
    req = urllib.request.Request(srv.base_url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def test_only_the_unsplittable_budget_loses_the_call(srv):
    no_call = []
    for budget in (1, 2, 3, 5, 8):
        d = _turn(srv, budget)
        t = d["runner_telemetry"]
        assert t["tool_protocol"]["constrained"], t
        assert t.get("finish_detail") == "reasoning_limit", (budget, t)
        if not d["choices"][0]["message"].get("tool_calls"):
            no_call.append(budget)
    assert no_call == list(range(1, TORTURE.UNSPLITTABLE_BUDGET + 1)), no_call
