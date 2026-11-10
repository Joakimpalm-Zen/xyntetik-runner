"""A served turn under a reasoning budget or the loop guard replays under
--verify (R1.1.3.3).

Both end a reasoning turn by a forced close: the request's sentence as
tokens, then the close token, written into the turn without sampling; the
guard with `loop_guard_everywhere` ends a turn that has no channel to close.
The record carries what the replay needs to do the same: the budget with the
sentence's token ids (`message_tokens`), and the guard's settings. So:

- a turn whose budget forced a close, with the default sentence, with none,
  and on a raw completion that resumed an open reasoning turn, replays
  VERIFIED; so does a turn the guard closed, and one it ended;
- a record whose budget or guard settings were changed after the fact
  replays DIVERGED, because the replay closes where the settings say;
- a record from before the settings were written is UNVERIFIABLE by name.
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

CHAT = {"max_tokens": 60, "temperature": 0, "enable_thinking": True, "cache_prompt": False,
        "messages": [{"role": "user", "content": "Which is warmer, Sydney or Oslo?"}]}
OPEN_REASONING = "<|start|>assistant to=self<|message|>"


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("think-replay")
    muse, plain = d / "muse.gguf", d / "plain.gguf"
    # a real ` to=self` reasoning channel closed by one control token, which
    # the budget and the guard's close need (test_reasoning_budget.py)
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--muse-glimmer",
                    "--control", "<|start|>,<|message|>,<|eom|>,<|eot|>", str(muse)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(plain)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return {"muse": muse, "plain": plain}


def _post(srv, payload, path):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _verify(runner_bin, model, rec, ctx):
    v = subprocess.run([runner_bin, "-m", model, "--verify", rec, "--gpu", "off",
                        "-t", "2", "-c", str(ctx)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    return v.returncode, v.stderr.decode(errors="replace")


def _rewrite(path, out, change):
    """The record with `change` applied to its first constraint and the chain
    hash recomputed, as an unsigned record from an older or tampered writer
    would read."""
    raw = path.read_text()
    cut = raw.rindex(',"chain":')
    body = json.loads(raw[:cut] + "}")
    change(body["constraints"][0])
    head = json.dumps(body, separators=(",", ":"))[:-1]
    out.write_text(head + ',"chain":' + json.dumps(
        {"algo": "sha256", "prev": json.loads(raw)["chain"]["prev"],
         "hash": hashlib.sha256(head.encode()).hexdigest()}, separators=(",", ":")) + "}")
    return out


@pytest.fixture(scope="module")
def muse_records(runner_bin, models, tmp_path_factory):
    d = tmp_path_factory.mktemp("muse-receipts")
    turns = {
        "budget": ({"reasoning_max_tokens": 6}, "/v1/chat/completions"),
        "bare": ({"reasoning_max_tokens": 6, "reasoning_budget_message": ""},
                 "/v1/chat/completions"),
        "guard": ({"loop_guard": True, "loop_guard_span": 2, "loop_guard_repeats": 2},
                  "/v1/chat/completions"),
        # a raw completion that resumes the reasoning turn: the budget counts
        # from the prompt's open marker (engine_think_prime)
        "raw": ({"prompt": OPEN_REASONING, "max_tokens": 40, "temperature": 0,
                 "cache_prompt": False, "reasoning_max_tokens": 6}, "/v1/completions"),
    }
    tel = {}
    with RunnerServer(runner_bin, models["muse"], ctx=2048, extra_args=[
            "--gpu", "off", "-t", "2", "--receipts", str(d)]) as srv:
        for name, (extra, path) in turns.items():
            body = _post(srv, extra if path == "/v1/completions" else dict(CHAT, **extra), path)
            tel[name] = body.get("runner_telemetry", {})
    return dict(zip(turns, sorted(d.iterdir()))), tel


@pytest.mark.parametrize("name", ["budget", "bare", "guard", "raw"])
def test_a_forced_close_replays(runner_bin, models, muse_records, name):
    recs, tel = muse_records
    # each turn really was closed by the engine, or the test means nothing
    if name == "guard":
        assert tel[name]["loop_guard"]["interventions"] >= 1, tel[name]
    else:
        assert tel[name]["reasoning_budget"]["forced_close"] is True, tel[name]
    rec = json.loads(recs[name].read_text())
    kind = rec["constraints"][0]
    if name == "guard":
        assert kind == {"kind": "loop_guard", "span": 2, "repeats": 2, "window": 256,
                        "max_closes": 3, "everywhere": False}
    else:
        assert kind["kind"] == "reasoning_budget" and kind["tokens"] == 6
        assert (kind["message_tokens"] == []) == (name == "bare"), kind
    rc, out = _verify(runner_bin, models["muse"], recs[name], 2048)
    assert rc == 0 and "VERIFIED" in out, out[-600:]


@pytest.mark.parametrize("name,change", [
    ("budget", lambda c: c.__setitem__("tokens", 9)),
    ("budget", lambda c: c.__setitem__("message_tokens", [])),
    ("guard", lambda c: c.__setitem__("repeats", 5)),
])
def test_changed_settings_replay_diverged(runner_bin, models, muse_records, tmp_path,
                                         name, change):
    recs, _ = muse_records
    rec = _rewrite(recs[name], tmp_path / "changed.json", change)
    rc, out = _verify(runner_bin, models["muse"], rec, 2048)
    assert rc == 2 and "DIVERGED" in out, out[-600:]


@pytest.mark.parametrize("name,field,drop", [
    ("budget", "message_tokens", ["message_tokens"]),
    ("guard", "span", ["span", "repeats", "window", "max_closes", "everywhere"]),
])
def test_a_record_without_the_settings_is_refused(runner_bin, models, muse_records,
                                                 tmp_path, name, field, drop):
    recs, _ = muse_records
    rec = _rewrite(recs[name], tmp_path / "old.json",
                   lambda c: [c.pop(k) for k in drop])
    rc, out = _verify(runner_bin, models["muse"], rec, 2048)
    assert rc == 3 and "UNVERIFIABLE" in out and field in out, out[-600:]


def test_a_turn_the_guard_ended_replays(runner_bin, models, tmp_path):
    """With no reasoning channel, loop_guard_everywhere ends a repeating
    turn; the replay ends it at the same token, and only there."""
    d = tmp_path / "r"
    with RunnerServer(runner_bin, models["plain"], ctx=256, extra_args=[
            "--gpu", "off", "-t", "2", "--receipts", str(d)]) as srv:
        body = _post(srv, {"prompt": "the quick brown fox", "max_tokens": 40,
                           "temperature": 0, "cache_prompt": False, "loop_guard": True,
                           "loop_guard_everywhere": True, "loop_guard_span": 2,
                           "loop_guard_repeats": 3, "loop_guard_window": 16},
                     "/v1/completions")
    assert body["runner_telemetry"]["loop_guard"]["ended_turn"] is True, body
    path = next(d.iterdir())
    rec = json.loads(path.read_text())
    assert len(rec["output"]["tokens"]) < 40
    rc, out = _verify(runner_bin, models["plain"], path, 256)
    assert rc == 0 and "VERIFIED" in out, out[-600:]
    later = _rewrite(path, tmp_path / "later.json", lambda c: c.__setitem__("repeats", 4))
    rc, out = _verify(runner_bin, models["plain"], later, 256)
    assert rc == 2 and "DIVERGED" in out, out[-600:]
