"""Session images over HTTP (R1.3.5): a served generation suspended to an
image and resumed exactly, forked reproducibly, read and deleted.

The anchor is the CLI's: a generation suspended after k tokens and resumed
produces, byte for byte, the text of the same generation run straight
through, and so does one suspended twice. A fork under a seed is the same
text every time it is asked for and differs from the parent's.
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


@pytest.fixture(scope="module")
def exe():
    e = find_runner(ROOT)
    if not pathlib.Path(e).exists():
        pytest.skip("runner not built")
    return e


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    out = tmp_path_factory.mktemp("sess") / "m.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(out)],
                   check=True, stdout=subprocess.DEVNULL)
    return out


def _post(srv, path, body):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def _get(srv, path, method="GET"):
    req = urllib.request.Request(srv.base_url + path, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


START = {"prompt": "the quick brown fox", "max_tokens": 40, "temperature": 0.9,
         "seed": 7, "ignore_eos": True}


def test_suspend_resume_and_fork_are_exact(exe, model, tmp_path):
    with RunnerServer(exe, model, ctx=256, extra_args=[
            "--gpu", "off", "-t", "2", "--sessions", str(tmp_path / "s")]) as srv:
        code, straight = _post(srv, "/v1/runner/sessions", START)
        assert code == 200 and straight["id"] is None
        assert straight["finish_reason"] == "length" and straight["generated"] == 40

        code, a = _post(srv, "/v1/runner/sessions", dict(START, suspend_after=15))
        assert code == 200 and a["finish_reason"] == "suspended"
        assert a["generated"] == 15 and len(a["id"]) == 64
        assert (tmp_path / "s" / (a["id"] + ".session")).is_file()

        code, b = _post(srv, f"/v1/runner/sessions/{a['id']}/resume", {})
        assert code == 200 and b["parent"] == a["id"] and b["id"] is None
        assert a["text"] + b["text"] == straight["text"]

        # suspended twice: still the straight run
        code, c = _post(srv, f"/v1/runner/sessions/{a['id']}/resume",
                        {"suspend_after": 28})
        code, d = _post(srv, f"/v1/runner/sessions/{c['id']}/resume", {})
        assert a["text"] + c["text"] + d["text"] == straight["text"]

        # the same state is the same image: a second suspend at 15 is the id
        code, a2 = _post(srv, "/v1/runner/sessions", dict(START, suspend_after=15))
        assert a2["id"] == a["id"]

        # a fork under a seed is reproducible, and it is another continuation
        f1 = _post(srv, f"/v1/runner/sessions/{a['id']}/resume", {"fork_seed": 99})[1]
        f2 = _post(srv, f"/v1/runner/sessions/{a['id']}/resume", {"fork_seed": 99})[1]
        f3 = _post(srv, f"/v1/runner/sessions/{a['id']}/resume", {"fork_seed": 5})[1]
        assert f1["forked"] and f1["text"] == f2["text"]
        assert f1["text"] != b["text"] or f3["text"] != b["text"]
        # 0 is a seed too (mixed, so never the rng's fixed point)
        z1 = _post(srv, f"/v1/runner/sessions/{a['id']}/resume", {"fork_seed": 0})
        z2 = _post(srv, f"/v1/runner/sessions/{a['id']}/resume", {"fork_seed": 0})
        assert z1[0] == 200 and z1[1]["forked"] and z1[1]["text"] == z2[1]["text"]

        code, meta = _get(srv, f"/v1/runner/sessions/{a['id']}")
        assert code == 200 and meta["generated"] == 15 and meta["max_tokens"] == 40
        code, gone = _get(srv, f"/v1/runner/sessions/{a['id']}", "DELETE")
        assert code == 200 and gone["deleted"]
        code, err = _post(srv, f"/v1/runner/sessions/{a['id']}/resume", {})
        assert code == 404 and err["error"]["code"] == "session_not_found"

        # an ordinary request after a suspended session is served as usual
        code, comp = _post(srv, "/v1/completions", {"prompt": "hello",
                                                    "max_tokens": 4})
        assert code == 200


def test_requests_that_cannot_be_served_are_refused(exe, model, tmp_path):
    with RunnerServer(exe, model, ctx=256, extra_args=[
            "--gpu", "off", "-t", "2", "--sessions", str(tmp_path / "s")]) as srv:
        assert _post(srv, "/v1/runner/sessions", {"max_tokens": 4})[0] == 400
        assert _post(srv, "/v1/runner/sessions",
                     dict(START, suspend_after=40))[0] == 400
        assert _post(srv, "/v1/runner/sessions/zz/resume", {})[0] == 400
        code, a = _post(srv, "/v1/runner/sessions", dict(START, suspend_after=10))
        assert _post(srv, f"/v1/runner/sessions/{a['id']}/resume",
                     {"suspend_after": 5})[0] == 409   # before the image's point
    with RunnerServer(exe, model, ctx=256, extra_args=["--gpu", "off", "-t", "2"]) as srv:
        code, err = _post(srv, "/v1/runner/sessions", START)
        assert code == 404 and err["error"]["code"] == "sessions_off"


def test_the_flag_needs_a_server(exe, model, tmp_path):
    p = subprocess.run([exe, "-m", str(model), "-p", "x", "--sessions",
                        str(tmp_path)], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=60)
    assert p.returncode != 0 and b"--serve" in p.stderr


def test_suspended_session_reports_its_tokens(exe, model, tmp_path):
    """The response is written after the slot is reset for the next request;
    `tokens` read the reset position and said 0 while the image carried the
    real count (found 2026-10-05)."""
    with RunnerServer(exe, model, ctx=256, extra_args=[
            "--gpu", "off", "-t", "2", "--sessions", str(tmp_path / "s")]) as srv:
        code, a = _post(srv, "/v1/runner/sessions", dict(START, suspend_after=15))
        assert code == 200 and a["finish_reason"] == "suspended"
        code, meta = _get(srv, f"/v1/runner/sessions/{a['id']}")
        assert a["generated"] == 15 == meta["generated"]
        assert a["tokens"] == meta["tokens"] > 15


def test_a_session_does_not_inherit_the_previous_request(exe, model, tmp_path):
    """A chat turn sets JSON mode, a reasoning budget or the loop guard on the
    slot's engine at its start and never cleared them; a session on that slot
    then generated under the previous request's constraint (found 2026-10-05).
    The anchor: a session after a json_object turn is byte-identical to the
    same session on a fresh server."""
    args = ["--gpu", "off", "-t", "2", "--parallel", "1",
            "--sessions", str(tmp_path / "s")]
    with RunnerServer(exe, model, ctx=256, extra_args=args) as srv:
        code, clean = _post(srv, "/v1/runner/sessions", START)
        assert code == 200
    with RunnerServer(exe, model, ctx=256, extra_args=args) as srv:
        code, j = _post(srv, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "object please"}],
            "max_tokens": 12, "response_format": {"type": "json_object"},
            "temperature": 0})
        assert code == 200, j
        code, after = _post(srv, "/v1/runner/sessions", START)
        assert code == 200
        assert after["text"] == clean["text"]
        assert after["generated"] == clean["generated"] == 40


def test_a_resumed_image_is_not_decoded_against_the_previous_rows(exe, model, tmp_path):
    """A resume installs the image's KV into the host cache. CUDA keeps a
    device mirror of that cache, which re-read the host only when a forward's
    position was not the previous one plus 1, so a resume landing exactly
    there decoded against the rows the slot's previous request left on the
    device: two sessions with prompts of one length, suspended at one point,
    leave the slot one row short of each other's resume (found 2026-10-21).
    The anchor is the straight run; the device is the host's (the test runs
    on whatever backend the host has, and the CUDA gate runs it there)."""
    other = dict(START, prompt="the lazy yellow dog")   # one byte a token: same length
    assert len(other["prompt"]) == len(START["prompt"])
    with RunnerServer(exe, model, ctx=256, extra_args=[
            "-t", "2", "--parallel", "1", "--sessions", str(tmp_path / "s")]) as srv:
        code, straight = _post(srv, "/v1/runner/sessions", START)
        assert code == 200 and straight["generated"] == 40
        code, a = _post(srv, "/v1/runner/sessions", dict(START, suspend_after=15))
        assert code == 200 and a["finish_reason"] == "suspended"
        code, b = _post(srv, "/v1/runner/sessions", dict(other, suspend_after=15))
        assert code == 200 and b["tokens"] == a["tokens"]
        code, rest = _post(srv, f"/v1/runner/sessions/{a['id']}/resume", {})
        assert code == 200
        assert a["text"] + rest["text"] == straight["text"]


@pytest.mark.parametrize("field,value", [
    ("temperature", "hot"), ("temperature", -1), ("temperature", 11),
    ("top_p", 1.5), ("min_p", -0.1), ("repeat_penalty", "2"),
    ("seed", -1), ("seed", 1.5), ("seed", "7"), ("seed", 2.0 ** 60),
    ("ignore_eos", "yes"), ("max_tokens", "40"), ("top_k", -1),
])
def test_session_numbers_are_validated(exe, model, tmp_path, field, value):
    """Every sampler knob is type- and range-checked before it is used: a
    string temperature used to read as the default and a negative one as
    itself (found 2026-10-05)."""
    with RunnerServer(exe, model, ctx=256, extra_args=[
            "--gpu", "off", "-t", "2", "--sessions", str(tmp_path / "s")]) as srv:
        code, err = _post(srv, "/v1/runner/sessions", dict(START, **{field: value}))
        assert code == 400, (field, value, err)
        assert err["error"]["code"] == "invalid_value"
        assert err["error"].get("param") == field


def test_resume_validates_its_numbers(exe, model, tmp_path):
    with RunnerServer(exe, model, ctx=256, extra_args=[
            "--gpu", "off", "-t", "2", "--sessions", str(tmp_path / "s")]) as srv:
        code, a = _post(srv, "/v1/runner/sessions", dict(START, suspend_after=10))
        for body in ({"fork_seed": -1}, {"fork_seed": "9"}, {"fork_seed": 1.5},
                     {"suspend_after": "20"}, {"suspend_after": 1.5}):
            code, err = _post(srv, f"/v1/runner/sessions/{a['id']}/resume", body)
            assert code == 400, (body, err)
            assert err["error"]["code"] == "invalid_value"


@pytest.mark.parametrize("fork_seed", [2 ** 53, 2 ** 53 + 1, 1e30])
def test_fork_seed_is_held_to_the_session_seed_range(exe, model, tmp_path, fork_seed):
    """fork_seed is a seed, held to the rule a session's own seed is: a
    positive integer below 2^53, the range a double holds exactly. It was
    checked only for being a whole double, so 2**53 and 2**53 + 1 both ran as
    2**53 and wrote one session image (2026-10-16 review), and 1e30 reached
    a long long cast it does not fit. Each is refused by name; 2**53 - 1 runs."""
    with RunnerServer(exe, model, ctx=256, extra_args=[
            "--gpu", "off", "-t", "2", "--sessions", str(tmp_path / "s")]) as srv:
        code, a = _post(srv, "/v1/runner/sessions", dict(START, suspend_after=5))
        assert code == 200
        code, err = _post(srv, f"/v1/runner/sessions/{a['id']}/resume",
                          {"fork_seed": fork_seed})
        assert code == 400 and "fork_seed" in json.dumps(err), (code, err)
        code, ok = _post(srv, f"/v1/runner/sessions/{a['id']}/resume",
                         {"fork_seed": 2 ** 53 - 1})
        assert code == 200 and ok["forked"], (code, ok)
