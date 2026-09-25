"""The loop guard: detect a repeating reasoning turn and close it (R4.12.18).

A distilled model that starts repeating inside a reasoning turn runs to the
token limit and the turn is lost. The lab measured 6 of 9 closed-loop failures
that way on a 14B student, with 18 of 19 capped training rollouts carrying a
repeated span, and the literature says looping is universal in small and
distilled models and worst under greedy decoding (arXiv 2512.12895).

This is the detect-and-force-close half. Detection looks at the GENERATED
suffix only and, on a hit inside a reasoning turn, closes the turn through the
same path the reasoning budget uses, so the model writes its own next header
and everything downstream sees ordinary generation. Rewind-and-resample is
deliberately not here.

Driven with the scripted-reply hook, which is the only way to make a fixture
loop on purpose: the script supplies the repetition, and what is measured is
where the runner stops accepting it.
"""
import json
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import RunnerServer, find_runner  # noqa: E402

# the shape the lab measured: a short arithmetic span repeated back to back
LOOP = "28 + 5 = 28 + 5 = " * 6
OPEN_REASONING = "<|start|>assistant to=self<|message|>"
OPEN_ANSWER = "<|start|>assistant to=user<|message|>"
GUARD = {"loop_guard": True, "loop_guard_span": 4, "loop_guard_repeats": 3}


@pytest.fixture(scope="module")
def muse_model(tmp_path_factory):
    m = tmp_path_factory.mktemp("loop") / "muse.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--muse-glimmer",
                    "--control", "<|start|>,<|message|>,<|eom|>,<|eot|>", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


@pytest.fixture(scope="module")
def plain_model(tmp_path_factory):
    m = tmp_path_factory.mktemp("loop") / "plain.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


def _serve(model, extra=()):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    return RunnerServer(exe, model, ctx=2048, parallel=1,
                        extra_args=["--gpu", "off", "-t", "2", *extra],
                        env={"RUNNER_TEST_SCRIPTED_REPLY": "1"})


@pytest.fixture(scope="module")
def muse(muse_model):
    with _serve(muse_model) as srv:
        yield srv


@pytest.fixture(scope="module")
def muse_guarded(muse_model):
    with _serve(muse_model, ["--loop-guard"]) as srv:
        yield srv


@pytest.fixture(scope="module")
def plain(plain_model):
    with _serve(plain_model) as srv:
        yield srv


def _post(server, **body):
    payload = {"prompt": OPEN_REASONING, "max_tokens": 200, "temperature": 0,
               "runner_test_reply": LOOP}
    payload.update(body)
    req = urllib.request.Request(server.base_url + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _guard(body):
    return body.get("runner_telemetry", {}).get("loop_guard")


def test_without_the_guard_the_loop_runs_to_the_budget(muse):
    code, body = _post(muse)
    assert code == 200, body
    assert _guard(body) is None
    assert body["choices"][0]["text"] == LOOP[:len(body["choices"][0]["text"])]
    assert len(body["choices"][0]["text"]) > 100


def test_the_guard_closes_a_repeating_reasoning_turn(muse):
    code, body = _post(muse, **GUARD)
    assert code == 200, body
    g = _guard(body)
    assert g["interventions"] == 1
    # it closed the turn rather than ending the request
    assert g["ended_turn"] is False
    # and it did so early: the unguarded run emitted the whole script
    assert len(body["choices"][0]["text"]) < 40


def test_the_guard_is_scoped_to_reasoning_by_default(muse):
    """Every measured runaway is a reasoning turn, and a repeated span in an
    ANSWER is often a legitimate table or list, so widening is a request."""
    code, body = _post(muse, prompt=OPEN_ANSWER, **GUARD)
    assert code == 200, body
    assert _guard(body)["interventions"] == 0
    assert len(body["choices"][0]["text"]) > 100


def test_widening_it_ends_the_turn_outside_reasoning(muse):
    code, body = _post(muse, prompt=OPEN_ANSWER, loop_guard_everywhere=True, **GUARD)
    assert code == 200, body
    g = _guard(body)
    assert g["interventions"] == 1 and g["ended_turn"] is True
    assert len(body["choices"][0]["text"]) < 40


def test_a_run_with_no_repetition_is_untouched(muse):
    """The guard must not fire on ordinary text, or it is worse than nothing."""
    plain_text = "the quick brown fox jumps over the lazy dog and keeps going"
    code, body = _post(muse, runner_test_reply=plain_text, **GUARD)
    assert code == 200, body
    assert _guard(body)["interventions"] == 0
    assert body["choices"][0]["text"].startswith("the quick brown fox")


def test_the_server_default_applies_and_a_request_can_turn_it_off(muse_guarded):
    code, body = _post(muse_guarded)
    assert code == 200, body
    assert _guard(body)["interventions"] == 1

    code, body = _post(muse_guarded, loop_guard=False)
    assert code == 200, body
    assert _guard(body) is None
    assert len(body["choices"][0]["text"]) > 100


@pytest.mark.parametrize("bad", [
    {"loop_guard_span": 1}, {"loop_guard_span": 500}, {"loop_guard_repeats": 1},
    {"loop_guard_window": 4}, {"loop_guard_span": 1.5},
    {"loop_guard_span": 100, "loop_guard_repeats": 4, "loop_guard_window": 64},
])
def test_out_of_range_settings_are_refused(muse, bad):
    code, body = _post(muse, loop_guard=True, **bad)
    assert code == 400, (bad, body)
    assert "loop_guard" in json.dumps(body)


def test_a_model_with_no_reasoning_channel_refuses_the_default_scope(plain):
    """Refused rather than ignored: with no channel to watch and no widening,
    the guard could never fire, and one that silently never runs is worse
    than none at all."""
    code, body = _post(plain, prompt="hi", loop_guard=True)
    assert code == 400, body
    assert "reasoning channel" in json.dumps(body)
    # widened, it works on that model
    code, body = _post(plain, prompt="hi", loop_guard=True,
                       loop_guard_everywhere=True, loop_guard_span=4,
                       loop_guard_repeats=3)
    assert code == 200, body
    assert _guard(body)["interventions"] == 1
