"""`/v1/completions` with `prompt` as an array of token ids (OpenAI's API,
vLLM and llama.cpp accept it; Runner answered "missing prompt" until
2026-10-09). The ids are used exactly as sent, never re-tokenized: the anchor
is the receipt, whose `prompt.tokens` is what `--verify` replays, so a
request sent as text and the same request sent as the ids its own receipt
recorded must leave identical token lists, identical greedy output, and a
receipt that replays VERIFIED. Malformed forms are refused, never guessed.
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
from harness import RunnerServer  # noqa: E402

TEXT = "the quick brown fox"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("tid") / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return m


def _post(srv, payload):
    req = urllib.request.Request(srv.base_url + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _receipts(d):
    return sorted(pathlib.Path(d).glob("receipt-*.json"))


def test_token_ids_are_used_exactly_and_replay(runner_bin, model, tmp_path):
    d = tmp_path / "r"
    with RunnerServer(runner_bin, model, ctx=256, extra_args=[
            "--gpu", "off", "-t", "2", "--receipts", str(d)]) as srv:
        st, a = _post(srv, {"prompt": TEXT, "max_tokens": 6, "temperature": 0,
                            "cache_prompt": False})
        assert st == 200, a
        ids = json.loads(_receipts(d)[0].read_text())["prompt"]["tokens"]
        assert ids and all(isinstance(i, int) for i in ids), ids
        st, b = _post(srv, {"prompt": ids, "max_tokens": 6, "temperature": 0,
                            "cache_prompt": False})
        assert st == 200, b
    recs = _receipts(d)
    assert len(recs) == 2
    rb = json.loads(recs[1].read_text())
    assert rb["prompt"]["tokens"] == ids            # used as sent, not re-tokenized
    # the record also carries a text, decoded from the ids: readable, but not
    # always the caller's original spelling (a SentencePiece piece decodes
    # with its word-boundary space, a control token as its spelling), which
    # is why the ids, not the text, are what --verify replays
    assert isinstance(rb["prompt"]["text"], str) and rb["prompt"]["text"], rb
    assert b["choices"][0]["text"] == a["choices"][0]["text"]
    assert b["usage"]["prompt_tokens"] == len(ids)
    v = subprocess.run([runner_bin, "-m", model, "--verify", recs[1], "--gpu", "off",
                        "-t", "2", "-c", "256"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    assert v.returncode == 0 and b"VERIFIED" in v.stdout + v.stderr, v.stderr.decode()


@pytest.mark.parametrize("prompt,needle", [
    ([], "empty"),
    ([-1], "token id"),
    ([10**9], "token id"),
    ([1.5], "token id"),
    (["a", "b"], "array of token ids"),
    ([[1, 2]], "array of token ids"),
])
def test_malformed_token_prompts_are_refused(runner_bin, model, prompt, needle):
    with RunnerServer(runner_bin, model, ctx=256,
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        st, r = _post(srv, {"prompt": prompt, "max_tokens": 2})
    assert st == 400, r
    assert needle in r["error"]["message"], r
