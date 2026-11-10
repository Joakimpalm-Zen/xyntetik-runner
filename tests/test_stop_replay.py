"""A served turn cut by a stop sequence replays under --verify (R1.1.3.2).

The server matches its stop strings on the visible text only: the bytes pass
the thinking-tag splitter first, which starts where the prompt left the turn,
and a match aborts generation on the token that completes it. The record
carries the sequences (`constraints`) and the splitter's start
(`serve.reasoning_open`), and the replay re-does the cut from them rather than
taking the record's length on trust. So:

- a turn cut by a stop string, or by a stop token id, replays VERIFIED;
- a stop string that appears only in the reasoning text does not cut the
  turn, served or replayed;
- a record whose stop or start was changed after the fact replays DIVERGED,
  and one that does not say where a thinking model's turn began is
  UNVERIFIABLE by name.
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


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("stop-replay")
    plain, q3 = d / "m.gguf", d / "q3.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(plain)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    # the qwen3 architecture gives the fixture <think> tags (model.c)
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--arch", "qwen3",
                    "--qk-norm", str(q3)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return {"plain": plain, "q3": q3}


def _post(srv, payload, path="/v1/completions"):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _verify(runner_bin, model, rec, *extra, ctx=256):
    v = subprocess.run([runner_bin, "-m", model, "--verify", rec, "--gpu", "off",
                        "-t", "2", "-c", str(ctx), *extra],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    return v.returncode, v.stderr.decode(errors="replace")


def _newest(d):
    path = sorted(d.iterdir())[-1]
    return path, json.loads(path.read_text())


def _first_char_at(text, start=3):
    """An ASCII character whose first occurrence is at or after `start`, so a
    stop on it cuts the turn there and nowhere earlier."""
    return next((c for i, c in enumerate(text) if i >= start and c.isascii()
                 and c.isprintable() and text.find(c) == i), None)


def _rewrite(path, out, change):
    """The record with `change` applied and its chain hash recomputed, as an
    unsigned record from an older or tampered writer would read."""
    raw = path.read_text()
    cut = raw.rindex(',"chain":')
    body = json.loads(raw[:cut] + "}")
    prev = json.loads(raw)["chain"]["prev"]
    change(body)
    head = json.dumps(body, separators=(",", ":"))[:-1]
    out.write_text(head + ',"chain":' + json.dumps(
        {"algo": "sha256", "prev": prev,
         "hash": hashlib.sha256(head.encode()).hexdigest()}, separators=(",", ":")) + "}")
    return out


def _cut_turn(srv, base):
    """A sampled turn, then the same turn with a stop on one of its characters:
    the stop must really cut it for the test to mean anything."""
    for seed in range(1, 40):
        q = dict(base, seed=seed)
        full = _post(srv, q)
        text = full["choices"][0]["text"]
        c = _first_char_at(text)
        if c and full["choices"][0]["finish_reason"] == "length":
            cut = _post(srv, dict(q, stop=[c]))
            assert cut["choices"][0]["finish_reason"] == "stop", cut
            return c, cut
    pytest.fail("no sampled turn had a usable stop character")


BASE = {"prompt": "the quick brown fox", "max_tokens": 24, "temperature": 1.0,
        "cache_prompt": False}


def test_a_turn_cut_by_a_stop_string_replays(runner_bin, models, tmp_path):
    d = tmp_path / "r"
    with RunnerServer(runner_bin, models["plain"], ctx=256, extra_args=[
            "--gpu", "off", "-t", "2", "--receipts", str(d)]) as srv:
        stop, _ = _cut_turn(srv, BASE)
    path, rec = _newest(d)
    assert rec["constraints"] == [{"kind": "stop", "sequences": [stop]}]
    assert rec["serve"]["reasoning_open"] is False
    assert len(rec["output"]["tokens"]) < BASE["max_tokens"]
    rc, out = _verify(runner_bin, models["plain"], path)
    assert rc == 0 and "VERIFIED" in out, out[-600:]
    # the cut is re-done, not read off the record: a stop the output never
    # contains runs the replay to max_tokens, past the recorded end
    other = _rewrite(path, tmp_path / "other.json", lambda b: b["constraints"][0]
                     .__setitem__("sequences", ["☃☃"]))
    rc, out = _verify(runner_bin, models["plain"], other)
    assert rc == 2 and "DIVERGED" in out, out[-600:]


def test_a_turn_cut_by_a_stop_token_id_replays(runner_bin, models, tmp_path):
    d = tmp_path / "r"
    with RunnerServer(runner_bin, models["plain"], ctx=256, extra_args=[
            "--gpu", "off", "-t", "2", "--receipts", str(d)]) as srv:
        _post(srv, dict(BASE, seed=3))
        _, full = _newest(d)
        toks = full["output"]["tokens"]
        at = next(i for i in range(3, len(toks)) if toks.index(toks[i]) == i)
        cut = _post(srv, dict(BASE, seed=3, stop_token_ids=[toks[at]]))
    assert cut["choices"][0]["finish_reason"] == "stop", cut
    path, rec = _newest(d)
    assert rec["constraints"] == [{"kind": "stop", "sequences": [],
                                   "token_ids": [toks[at]]}]
    assert rec["output"]["tokens"] == toks[:at]
    rc, out = _verify(runner_bin, models["plain"], path)
    assert rc == 0 and "VERIFIED" in out, out[-600:]


@pytest.mark.parametrize("template,opened", [("qwen38", True), ("chatml", False)])
def test_the_stop_is_matched_where_the_turn_began(runner_bin, models, tmp_path,
                                                  template, opened):
    """qwen38's prompt opens the thought block, so the turn's first bytes are
    reasoning and a stop on them does not fire; chatml's does not, so the same
    kind of stop cuts the answer. Both replay VERIFIED."""
    d = tmp_path / "r"
    chat = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 20,
            "temperature": 1.0, "cache_prompt": False}
    with RunnerServer(runner_bin, models["q3"], ctx=2048, extra_args=[
            "--gpu", "off", "-t", "2", "--receipts", str(d),
            "--chat-template", template]) as srv:
        for seed in range(1, 40):
            full = _post(srv, dict(chat, seed=seed), "/v1/chat/completions")
            msg = full["choices"][0]["message"]
            text = msg.get("reasoning_content") if opened else msg.get("content")
            stop = _first_char_at(text or "")
            if stop and full["choices"][0]["finish_reason"] == "length":
                break
        else:
            pytest.fail("no sampled turn had a usable stop character")
        assert not opened or not msg.get("content"), msg  # all of it reasoning
        cut = _post(srv, dict(chat, seed=seed, stop=[stop]), "/v1/chat/completions")
    assert cut["choices"][0]["finish_reason"] == ("length" if opened else "stop"), cut
    path, rec = _newest(d)
    assert rec["serve"]["reasoning_open"] is opened
    assert rec["constraints"][0]["sequences"] == [stop]
    rc, out = _verify(runner_bin, models["q3"], path, ctx=2048)
    assert rc == 0 and "VERIFIED" in out, out[-600:]
    if opened:
        # read as a turn that began in the answer, the stop fires on what was
        # reasoning, so the replay ends early
        flipped = _rewrite(path, tmp_path / "flipped.json",
                           lambda b: b["serve"].__setitem__("reasoning_open", False))
        rc, out = _verify(runner_bin, models["q3"], flipped, ctx=2048)
        assert rc == 2 and "DIVERGED" in out, out[-600:]
        # a record from before the field cannot say, and is refused by name
        old = _rewrite(path, tmp_path / "old.json",
                       lambda b: b["serve"].pop("reasoning_open"))
        rc, out = _verify(runner_bin, models["q3"], old, ctx=2048)
        assert rc == 3 and "reasoning_open" in out, out[-600:]


def test_a_stop_beside_a_schema_replays_under_the_schema(runner_bin, models, tmp_path):
    """The validator is told what the stop dropped, as served, so the closing
    tail continues the same document in the replay."""
    d = tmp_path / "r"
    schema = {"type": "object", "properties": {"a": {"type": "string"}},
              "required": ["a"]}
    sfile = tmp_path / "s.json"
    sfile.write_text(json.dumps(schema))
    q = {"prompt": "x", "max_tokens": 16, "temperature": 1.0, "cache_prompt": False,
         "response_format": {"type": "json_schema",
                             "json_schema": {"name": "t", "schema": schema}}}
    with RunnerServer(runner_bin, models["plain"], ctx=256, extra_args=[
            "--gpu", "off", "-t", "2", "--receipts", str(d)]) as srv:
        for seed in range(1, 40):
            full = _post(srv, dict(q, seed=seed))
            text = full["choices"][0]["text"]
            # a stop inside the string value, past the document's opening
            start = text.find(':"') + 3 if ':"' in text else len(text)
            stop = _first_char_at(text, start)
            if stop and stop not in '"\\{}':
                break
        else:
            pytest.fail("no sampled document had a usable stop character")
        cut = _post(srv, dict(q, seed=seed, stop=[stop]))
    assert cut["choices"][0]["finish_reason"] == "stop", cut
    json.loads(cut["choices"][0]["text"])   # closed to a legal document
    path, rec = _newest(d)
    assert [c["kind"] for c in rec["constraints"]] == ["json_schema", "stop"]
    rc, out = _verify(runner_bin, models["plain"], path, "--json-schema", str(sfile))
    assert rc == 0 and "VERIFIED" in out, out[-600:]
    rc, out = _verify(runner_bin, models["plain"], path)
    assert rc == 3 and "json_schema" in out, out[-600:]


@pytest.mark.parametrize("template", ["qwen38", "chatml"])
def test_a_served_schema_turn_on_a_thinking_model_replays(runner_bin, models, tmp_path,
                                                          template):
    """Found beside R1.1.3.2: a served chat turn on a model with thinking tags
    forwards its reasoning bytes beside a constrained document, and one whose
    prompt opened the reasoning block holds the grammar until it closes. The
    replay did neither, so such a turn replayed DIVERGED at its first token or
    byte; it starts where the server started it now."""
    d = tmp_path / "r"
    schema = {"type": "object", "properties": {"a": {"type": "string"}},
              "required": ["a"]}
    sfile = tmp_path / "s.json"
    sfile.write_text(json.dumps(schema))
    with RunnerServer(runner_bin, models["q3"], ctx=2048, extra_args=[
            "--gpu", "off", "-t", "2", "--receipts", str(d),
            "--chat-template", template]) as srv:
        for seed in (1, 2):
            _post(srv, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 24,
                        "temperature": 1.0, "seed": seed, "cache_prompt": False,
                        "response_format": {"type": "json_schema",
                                            "json_schema": {"name": "t", "schema": schema}}},
                  "/v1/chat/completions")
    for path in sorted(d.iterdir()):
        rec = json.loads(path.read_text())
        assert rec["serve"]["reasoning_open"] is (template == "qwen38")
        rc, out = _verify(runner_bin, models["q3"], path, "--json-schema", str(sfile),
                          ctx=2048)
        assert rc == 0 and "VERIFIED" in out, (path.name, out[-600:])
