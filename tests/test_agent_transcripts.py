"""Agent transcripts, D4a (R1.1.2): a record says which tool calls the turn
made and what constrained its output, and `--verify` refuses a constrained
record it cannot replay instead of calling the difference a divergence.

A grammar, a stop sequence or a scripted reply changes the tokens the sampler
would have chosen; a replay without it disagrees with the record by
construction, and "DIVERGED" would blame the model or the build for what the
record simply did not say. So:

- every record names its constraints (`constraints`, with the digest of a
  schema or tool list, computed here independently with hashlib over the
  compact JSON), and a served turn's calls (`tool_calls`, compared with the
  calls the response itself delivered, buffered and streamed, on every
  surface);
- a served constrained record is UNVERIFIABLE (exit 3), naming what shaped
  it; a CLI record made under --json, --json-schema or --ignore-eos replays
  when the verifier is given the same constraint (the schema matched by
  digest) and is refused otherwise -- and a verifier that adds a constraint
  the record did not have is refused too.
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

WEATHER_PARAMS = {"type": "object",
                  "properties": {"city": {"type": "string"}},
                  "required": ["city"]}
CHAT_TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": WEATHER_PARAMS}}]
RESP_TOOLS = [{"type": "function", "name": "get_weather",
               "description": "Current weather for a city",
               "parameters": WEATHER_PARAMS}]
ANTH_TOOLS = [{"name": "get_weather", "description": "Current weather for a city",
               "input_schema": WEATHER_PARAMS}]
NO_THINK = {"enable_thinking": False}


def _compact_sha(v):
    return hashlib.sha256(json.dumps(v, separators=(",", ":")).encode()).hexdigest()


def _call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("d4a") / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return m


def _run(runner_bin, *args):
    return subprocess.run([str(runner_bin), *map(str, args)], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=180)


def _verify(runner_bin, model, rec, *extra, ctx=4096):
    return _run(runner_bin, "-m", model, "--verify", rec, "--gpu", "off",
                "-t", "2", "-c", ctx, *extra)


def _post(srv, path, payload, stream=False):
    req = urllib.request.Request(srv.base_url + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
    if not stream:
        return json.loads(body)
    return [json.loads(line[6:]) for line in body.decode().splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"]


@pytest.fixture(scope="module")
def srv(runner_bin, model, tmp_path_factory):
    d = tmp_path_factory.mktemp("d4a-receipts")
    with RunnerServer(runner_bin, model, ctx=4096, parallel=1, extra_args=[
            "--gpu", "off", "-t", "2", "--chat-template", "qwen38",
            "--receipts", str(d)],
            env={"RUNNER_TEST_SCRIPTED_REPLY": "1"}) as s:
        s.receipts = d
        yield s


def _receipt(srv, body):
    name = body["runner_telemetry"]["receipt"]["file"]
    return srv.receipts / name, json.loads((srv.receipts / name).read_text())


def _newest(srv):
    p = sorted(srv.receipts.iterdir())[-1]
    return p, json.loads(p.read_text())


def _chat_calls(srv, reply, stream):
    payload = {"max_tokens": 400, "tools": CHAT_TOOLS, "stream": stream,
               "messages": [{"role": "user", "content": "Weather?"}],
               "chat_template_kwargs": NO_THINK, "runner_test_reply": reply}
    if not stream:
        d = _post(srv, "/v1/chat/completions", payload)
        calls = [(c["function"]["name"], c["function"]["arguments"])
                 for c in d["choices"][0]["message"]["tool_calls"]]
        return calls, _receipt(srv, d)
    chunks = _post(srv, "/v1/chat/completions", payload, stream=True)
    acc = {}
    for c in chunks:
        for ch in c.get("choices", []):
            for tc in ch.get("delta", {}).get("tool_calls", []) or []:
                e = acc.setdefault(tc["index"], ["", ""])
                f = tc.get("function", {})
                e[0] += f.get("name", "") or ""
                e[1] += f.get("arguments", "") or ""
    return [tuple(acc[i]) for i in sorted(acc)], _newest(srv)


def _responses_calls(srv, reply, stream):
    payload = {"input": "Weather?", "tools": RESP_TOOLS, "stream": stream,
               "max_output_tokens": 400, "chat_template_kwargs": NO_THINK,
               "runner_test_reply": reply}
    if stream:
        events = _post(srv, "/v1/responses", payload, stream=True)
        (done,) = [e for e in events if e.get("type") == "response.completed"]
        body = done["response"]
    else:
        body = _post(srv, "/v1/responses", payload)
    calls = [(o["name"], o["arguments"]) for o in body["output"]
             if o["type"] == "function_call"]
    return calls, (_receipt(srv, body) if not stream else _newest(srv))


def _messages_calls(srv, reply, stream):
    payload = {"max_tokens": 400, "tools": ANTH_TOOLS, "stream": stream,
               "messages": [{"role": "user", "content": "Weather?"}],
               "chat_template_kwargs": NO_THINK, "runner_test_reply": reply}
    if not stream:
        d = _post(srv, "/v1/messages", payload)
        calls = [(b["name"], b["input"]) for b in d["content"]
                 if b["type"] == "tool_use"]
        return calls, _receipt(srv, d)
    events = _post(srv, "/v1/messages", payload, stream=True)
    blocks = {}
    for e in events:
        if e.get("type") == "content_block_start" and \
                e["content_block"]["type"] == "tool_use":
            blocks[e["index"]] = [e["content_block"]["name"], ""]
        elif e.get("type") == "content_block_delta" and \
                e["delta"].get("type") == "input_json_delta":
            blocks[e["index"]][1] += e["delta"]["partial_json"]
    calls = [(blocks[i][0], json.loads(blocks[i][1])) for i in sorted(blocks)]
    return calls, _newest(srv)


SURFACES = {"chat": (_chat_calls, CHAT_TOOLS), "responses": (_responses_calls, RESP_TOOLS),
            "messages": (_messages_calls, ANTH_TOOLS)}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_served_tool_calls_are_recorded_and_refused_loudly(runner_bin, model, srv,
                                                           surface, stream):
    fn, tools = SURFACES[surface]
    reply = _call("get_weather", city="Paris") + "\n" + _call("get_weather", city="Oslo")
    calls, (path, rec) = fn(srv, reply, stream)
    assert len(calls) == 2, calls
    # the record's calls are the ones the response delivered, in order
    got = [(c["name"], json.loads(c["arguments"])) for c in rec["tool_calls"]]
    want = [(n, a if isinstance(a, dict) else json.loads(a)) for n, a in calls]
    assert got == want == [("get_weather", {"city": "Paris"}),
                           ("get_weather", {"city": "Oslo"})]
    if surface != "messages":   # the arguments string itself, byte for byte
        assert [c["arguments"] for c in rec["tool_calls"]] == [a for _, a in calls]
    # an auto turn in the family's native syntax is parsed, not constrained
    # (template.c, parse_only): the tokens are the sampler's, so the tools
    # are no constraint -- the scripted reply that dictated them is
    assert [c["kind"] for c in rec["constraints"]] == ["scripted"]
    v = _verify(runner_bin, model, path)
    err = v.stderr.decode(errors="replace")
    assert v.returncode == 3 and "UNVERIFIABLE" in err and "scripted" in err, err[-600:]
    assert "DIVERGED" not in err + v.stdout.decode(errors="replace")


def test_a_tool_grammar_is_a_recorded_constraint(runner_bin, model, srv):
    """A required choice compiles the XML grammar: the tools shaped the
    tokens, and the record names them by digest."""
    d = _post(srv, "/v1/chat/completions", {
        "max_tokens": 400, "tools": CHAT_TOOLS, "tool_choice": "required",
        "messages": [{"role": "user", "content": "Weather?"}],
        "chat_template_kwargs": NO_THINK,
        "runner_test_reply": _call("get_weather", city="Rome")})
    (call,) = d["choices"][0]["message"]["tool_calls"]
    path, rec = _receipt(srv, d)
    assert rec["tool_calls"] == [{"name": "get_weather",
                                  "arguments": call["function"]["arguments"]}]
    kinds = {c["kind"]: c for c in rec["constraints"]}
    assert kinds["tools"] == {"kind": "tools", "sha256": _compact_sha(CHAT_TOOLS),
                              "choice": "required"}
    assert rec["serve"]["shaped_by"] == [c["kind"] for c in rec["constraints"]]
    v = _verify(runner_bin, model, path)
    err = v.stderr.decode(errors="replace")
    assert v.returncode == 3 and "tools" in err, err[-600:]


def test_served_schema_and_stop_are_recorded(runner_bin, model, srv):
    schema = {"type": "object", "properties": {"a": {"type": "string"}},
              "required": ["a"]}
    d = _post(srv, "/v1/completions", {
        "prompt": "x", "max_tokens": 12, "stop": ["e"],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "t", "schema": schema}}})
    path, rec = _receipt(srv, d)
    kinds = {c["kind"]: c for c in rec["constraints"]}
    assert kinds["json_schema"]["sha256"] == _compact_sha(schema)
    assert kinds["stop"]["sequences"] == ["e"]
    assert "tool_calls" not in rec
    v = _verify(runner_bin, model, path)
    err = v.stderr.decode(errors="replace")
    assert v.returncode == 3 and "json_schema" in err, err[-600:]
    # an unconstrained turn carries no constraints and still replays
    d = _post(srv, "/v1/completions", {"prompt": "the fox", "max_tokens": 6,
                                       "temperature": 0, "cache_prompt": False})
    path, rec = _receipt(srv, d)
    assert "constraints" not in rec and "tool_calls" not in rec
    v = _verify(runner_bin, model, path)
    assert v.returncode == 0, v.stderr.decode(errors="replace")[-600:]


def _cli(runner_bin, model, rec, *extra):
    p = _run(runner_bin, "-m", model, "-p", "the quick brown fox", "-n", "12",
             "--temp", "0.8", "-s", "7", "--gpu", "off", "-t", "2", "-c", "256",
             "--transcript", rec, *extra)
    assert p.returncode == 0, p.stderr.decode(errors="replace")[-400:]
    return json.loads(rec.read_text())


def test_cli_json_mode_replays_only_under_json_mode(runner_bin, model, tmp_path):
    rec = _cli(runner_bin, model, tmp_path / "j.json", "--json")
    assert rec["constraints"] == [{"kind": "json_mode"}]
    v = _verify(runner_bin, model, tmp_path / "j.json", ctx=256)
    err = v.stderr.decode(errors="replace")
    assert v.returncode == 3 and "json_mode" in err and "--json" in err, err[-600:]
    v = _verify(runner_bin, model, tmp_path / "j.json", "--json", ctx=256)
    assert v.returncode == 0 and b"VERIFIED" in v.stderr, v.stderr[-600:]


def test_cli_schema_replays_only_under_the_same_schema(runner_bin, model, tmp_path):
    schema = {"type": "object", "properties": {"n": {"type": "integer"}},
              "required": ["n"]}
    sf = tmp_path / "s.json"
    sf.write_text(json.dumps(schema, indent=2))   # spelling is not identity
    rec = _cli(runner_bin, model, tmp_path / "s-rec.json", "--json-schema", sf)
    assert rec["constraints"] == [{"kind": "json_schema",
                                   "sha256": _compact_sha(schema)}]
    v = _verify(runner_bin, model, tmp_path / "s-rec.json", ctx=256)
    assert v.returncode == 3 and b"json_schema" in v.stderr, v.stderr[-600:]
    v = _verify(runner_bin, model, tmp_path / "s-rec.json", "--json-schema", sf, ctx=256)
    assert v.returncode == 0 and b"VERIFIED" in v.stderr, v.stderr[-600:]
    other = tmp_path / "o.json"
    other.write_text(json.dumps({"type": "object", "properties": {
        "n": {"type": "string"}}, "required": ["n"]}))
    v = _verify(runner_bin, model, tmp_path / "s-rec.json", "--json-schema", other, ctx=256)
    assert v.returncode == 3 and b"json_schema" in v.stderr, v.stderr[-600:]


def test_a_constraint_the_record_lacks_is_refused(runner_bin, model, tmp_path):
    rec = _cli(runner_bin, model, tmp_path / "p.json")
    assert "constraints" not in rec
    for extra in (["--json"], ["--ignore-eos"]):
        v = _verify(runner_bin, model, tmp_path / "p.json", *extra, ctx=256)
        assert v.returncode == 3 and b"UNVERIFIABLE" in v.stderr, (extra, v.stderr[-400:])
    v = _verify(runner_bin, model, tmp_path / "p.json", ctx=256)
    assert v.returncode == 0, v.stderr[-400:]
    rec = _cli(runner_bin, model, tmp_path / "e.json", "--ignore-eos")
    assert rec["constraints"] == [{"kind": "ignore_eos"}]
    assert _verify(runner_bin, model, tmp_path / "e.json", ctx=256).returncode == 3
    v = _verify(runner_bin, model, tmp_path / "e.json", "--ignore-eos", ctx=256)
    assert v.returncode == 0, v.stderr[-400:]


def test_a_record_that_only_lists_shaped_by_is_refused(runner_bin, model, srv, tmp_path):
    """Serve receipts written before D4a name what shaped them only in
    serve.shaped_by; that is enough to refuse them."""
    d = _post(srv, "/v1/completions", {"prompt": "x", "max_tokens": 6, "stop": ["e"],
                                       "temperature": 0})
    path, rec = _receipt(srv, d)
    assert rec["serve"]["shaped_by"] == ["stop"]
    raw = path.read_text()
    cut = raw.rindex(',"chain":')
    body = json.loads(raw[:cut] + "}")
    del body["constraints"]
    head = json.dumps(body, separators=(",", ":"))[:-1]
    old = tmp_path / "old.json"
    old.write_text(head + ',"chain":' + json.dumps(
        {"algo": "sha256", "prev": rec["chain"]["prev"],
         "hash": hashlib.sha256(head.encode()).hexdigest()}, separators=(",", ":")) + "}")
    v = _verify(runner_bin, model, old)
    err = v.stderr.decode(errors="replace")
    assert v.returncode == 3 and "stop" in err and "chain" not in err, err[-600:]
