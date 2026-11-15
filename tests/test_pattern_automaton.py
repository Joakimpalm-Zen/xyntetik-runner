"""JSON Schema `pattern` through the automaton, end to end (R2.18).

A pattern the segment compiler cannot express (an alternation, a group) goes to
src/jsre.c. These drive the runner, not the library:

- generation under such a pattern produces strings the pattern accepts;
- the same constrained request gives the same bytes twice in one server, in a
  fresh one, replayed by --verify, and with the pattern cache cut to one byte
  (RUNNER_TEST_JSRE_POOL_BYTES=1, whose flush count is the check that the
  small cache was really used);
- two slots serving one schema at once each give their solo bytes: a
  compiled schema, and the pattern cache inside it, belongs to one request;
- the kept-state limit ends a request as a named fault, in the response body
  and the log (which pattern, how many states), never as a narrower document.
  RUNNER_TEST_JSRE_KEPT_MAX lowers the limit so a scripted reply reaches it.
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer, find_runner  # noqa: E402

# (ab|cd) and the group are not a segment shape; the bound forces a close
PATTERN = "^(ab|cd){1,4}-[0-9]{2}$"
SCHEMA = {"type": "object", "properties": {"id": {"type": "string", "pattern": PATTERN}},
          "required": ["id"], "additionalProperties": False}
# every position of a long run is a distinct automaton state: a few bytes
# reach a lowered kept-state limit
LIMIT_PATTERN = "^(a|b){0,30}c?$"
LIMIT_SCHEMA = {"type": "object", "properties": {"k": {"type": "string", "pattern": LIMIT_PATTERN}},
                "required": ["k"], "additionalProperties": False}
SERVER_ARGS = ["--gpu", "off", "-t", "2", "--chat-template", "chatml"]


@pytest.fixture(scope="module")
def exe():
    e = find_runner(ROOT)
    if not pathlib.Path(e).exists():
        pytest.skip("runner not built")
    return e


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("jsre") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


def _post(srv, path, body):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _chat(schema, seed, **extra):
    return {"messages": [{"role": "user", "content": "Give an id."}], "max_tokens": 64,
            "temperature": 0.9, "seed": seed,
            "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}, **extra}


def _content(srv, seed):
    status, body = _post(srv, "/v1/chat/completions", _chat(SCHEMA, seed))
    assert status == 200, body
    return body["choices"][0]["message"]["content"]


def _valid(text):
    doc = json.loads(text)
    assert set(doc) == {"id"}, text
    assert re.fullmatch(r"(ab|cd){1,4}-[0-9]{2}", doc["id"]), text   # ASCII only: Python and ECMA agree
    return doc


def test_output_under_a_pattern_the_segments_cannot_take(exe, model, tmp_path):
    with RunnerServer(exe, model, ctx=1024, parallel=1, extra_args=SERVER_ARGS) as srv:
        ids = {_valid(_content(srv, seed))["id"] for seed in range(1, 9)}
    # a constraint that enforced nothing would not land 8 random walks on this language;
    # one that forced a single string would give one id
    assert len(ids) > 1, ids


def test_the_same_request_gives_the_same_bytes(exe, model, tmp_path):
    seeds = [3, 5, 8]
    with RunnerServer(exe, model, ctx=1024, parallel=1, extra_args=SERVER_ARGS) as srv:
        first = [_content(srv, s) for s in seeds]
        again = [_content(srv, s) for s in seeds]
    with RunnerServer(exe, model, ctx=1024, parallel=1, extra_args=SERVER_ARGS) as srv:
        fresh = [_content(srv, s) for s in seeds]
    log = tmp_path / "small-cache.log"
    with RunnerServer(exe, model, ctx=1024, parallel=1, extra_args=SERVER_ARGS, log_path=str(log),
                      env={"RUNNER_TEST_JSRE_POOL_BYTES": "1"}) as srv:
        small = [_content(srv, s) for s in seeds]
    for t in first:
        _valid(t)
    assert first == again == fresh == small
    text = log.read_text(encoding="utf-8", errors="replace")
    assert "test seam: RUNNER_TEST_JSRE_POOL_BYTES=1 lowers" in text, text[-2000:]
    flushes = [int(n) for n in re.findall(r"pattern cache flushed (\d+) times", text)]
    # the live check: the one-byte cache really flushed while giving those bytes
    assert flushes and max(flushes) > 0, text[-2000:]


def test_two_slots_on_one_schema_each_give_their_solo_bytes(exe, model):
    seeds = [11, 12, 13, 14]
    with RunnerServer(exe, model, ctx=1024, parallel=2, extra_args=SERVER_ARGS) as srv:
        solo = {s: _content(srv, s) for s in seeds}
        together = {}

        def run(s):
            together[s] = _content(srv, s)

        for pair in (seeds[:2], seeds[2:]):
            ts = [threading.Thread(target=run, args=(s,)) for s in pair]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
    for s in seeds:
        _valid(solo[s])
    assert together == solo


def test_a_record_replays_on_a_cold_cache(exe, model, tmp_path):
    schema = tmp_path / "s.json"
    schema.write_text(json.dumps(SCHEMA), encoding="utf-8")
    rec = tmp_path / "r.json"
    run = subprocess.run([str(exe), "-m", str(model), "--json-schema", str(schema), "-p", "Give an id.",
                          "-n", "48", "--temp", "0.9", "-s", "21", "--gpu", "off", "--no-tray",
                          "--record", str(rec)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-1500:]
    for env in ({}, {"RUNNER_TEST_JSRE_POOL_BYTES": "1"}):
        v = subprocess.run([str(exe), "--verify", str(rec), "-m", str(model), "--json-schema", str(schema),
                            "--gpu", "off", "--no-tray"], cwd=ROOT, capture_output=True, text=True,
                           timeout=120, env={**os.environ, **env})
        out = v.stdout + v.stderr
        # the exit code alone is not the verdict: an UNVERIFIABLE record exits 0 too
        assert v.returncode == 0 and "VERIFIED (" in out and "replayed identically" in out, (env, out[-1500:])


@pytest.fixture(scope="module")
def limited(exe, model, tmp_path_factory):
    log = tmp_path_factory.mktemp("limit") / "server.log"
    with RunnerServer(exe, model, ctx=1024, parallel=1, extra_args=SERVER_ARGS, log_path=str(log),
                      env={"RUNNER_TEST_SCRIPTED_REPLY": "1", "RUNNER_TEST_JSRE_KEPT_MAX": "4"}) as srv:
        yield srv, log


SCRIPT = '{"k":"abababababab"}'


def test_the_kept_state_limit_is_a_named_fault(limited):
    srv, log = limited
    status, body = _post(srv, "/v1/chat/completions",
                         _chat(LIMIT_SCHEMA, 1, runner_test_reply=SCRIPT, temperature=0))
    assert status == 200, body
    choice = body["choices"][0]
    assert choice["finish_reason"] == "error", body
    tel = body["runner_telemetry"]
    assert tel["finish_detail"] == "pattern_limit", tel
    assert LIMIT_PATTERN in tel["fault"] and "more states than one request may keep" in tel["fault"], tel
    assert "4 states kept" in tel["fault"], tel
    text = log.read_text(encoding="utf-8", errors="replace")
    assert "test seam: RUNNER_TEST_JSRE_KEPT_MAX=4 lowers" in text
    assert "constrained generation stopped: " in text and LIMIT_PATTERN in text, text[-2000:]


def test_the_limit_on_the_messages_surface(limited):
    srv, _ = limited
    with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
        model_id = json.loads(r.read())["data"][0]["id"]
    status, body = _post(srv, "/v1/messages", {
        "model": model_id, "max_tokens": 64, "messages": [{"role": "user", "content": "Give an id."}],
        "runner_test_reply": SCRIPT,
        # one entry point for a request's schema on every surface (request_schema)
        "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": LIMIT_SCHEMA}}})
    assert body.get("type") == "error" and body["error"]["type"] == "api_error", body
    assert LIMIT_PATTERN in body["error"]["message"], body
    assert body["runner_telemetry"]["finish_detail"] == "pattern_limit", body


def test_the_limit_on_the_responses_surface(limited):
    srv, _ = limited
    status, body = _post(srv, "/v1/responses", {
        "input": "Give an id.", "max_output_tokens": 64, "runner_test_reply": SCRIPT,
        "text": {"format": {"type": "json_schema", "name": "x", "schema": LIMIT_SCHEMA}}})
    assert status == 200, body
    assert body["status"] == "incomplete", body
    assert body["incomplete_details"]["reason"] == "pattern_limit", body
    assert LIMIT_PATTERN in body["runner_telemetry"]["fault"], body


def test_below_the_limit_the_same_reply_completes(exe, model):
    with RunnerServer(exe, model, ctx=1024, parallel=1, extra_args=SERVER_ARGS,
                      env={"RUNNER_TEST_SCRIPTED_REPLY": "1"}) as srv:
        status, body = _post(srv, "/v1/chat/completions",
                             _chat(LIMIT_SCHEMA, 1, runner_test_reply=SCRIPT, temperature=0))
    assert status == 200, body
    assert body["choices"][0]["finish_reason"] == "stop", body
    assert json.loads(body["choices"][0]["message"]["content"]) == {"k": "abababababab"}
    assert "fault" not in body.get("runner_telemetry", {})
