"""A response is JSON whatever the forward computed.

JSON has no NaN and no infinity; printf writes them as `nan` and `inf`. When
a forward goes non-finite (an overflowed activation, a failed device kernel,
a NaN in the weights) every number derived from the logits is NaN, and each
endpoint that printed one with a bare `%g` answered 200 with a body no parser
takes: log-probabilities on completions and chat, rerank's scores, decide's
log-probabilities and probabilities, the decision telemetry. A number the
model computed is now written through one writer, which writes `null` for a
value that is not finite. (Embeddings refuse instead: an embedding with a
hole is no embedding. tests/test_embeddings_pooling.py holds that.)

The fixture is a small model with a NaN written into every row of its
embedding table, so every position's hidden state is NaN from the first
layer on.
"""
import json
import pathlib
import struct
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer, find_runner  # noqa: E402
from test_embeddings_pooling import _embedding_table  # noqa: E402


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    m = tmp_path_factory.mktemp("nonfinite") / "nan.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    data, at, width, rows = _embedding_table(m)
    data = bytearray(data)
    for r in range(rows):
        struct.pack_into("<f", data, at + r * width * 4, float("nan"))
    m.write_bytes(data)
    with RunnerServer(exe, m, ctx=256, parallel=1, extra_args=["--gpu", "off", "-t", "1"]) as srv:
        yield srv


def _post(srv, path, body):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


REQUESTS = {
    "completion logprobs": ("/v1/completions", {
        "prompt": "hello", "max_tokens": 4, "temperature": 0, "logprobs": 3}),
    "completion echo logprobs": ("/v1/completions", {
        "prompt": "hello there", "max_tokens": 2, "temperature": 0, "logprobs": 2, "echo": True}),
    "chat logprobs": ("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "temperature": 0,
        "logprobs": True, "top_logprobs": 3}),
    "rerank": ("/v1/rerank", {"query": "a fox", "documents": ["a dog", "a cat"]}),
    "decide": ("/v1/decide", {"state": "the quick brown fox", "questions": [
        {"id": "q1", "question": "what did the fox do", "options": ["jumps", "sleeps"]}]}),
}


@pytest.mark.parametrize("name", sorted(REQUESTS))
def test_a_non_finite_forward_still_answers_json(server, name):
    path, body = REQUESTS[name]
    status, raw = _post(server, path, body)
    # the body parses, whether the endpoint answered or refused
    parsed = json.loads(raw)
    assert status in (200, 400, 500), (status, raw[:200])
    if status == 200:
        # and where a number could not be given, it is null, not a made-up value
        assert b"null" in raw, raw[:300]
    else:
        assert "error" in parsed, raw[:300]


def _message(raw):
    err = json.loads(raw)["error"]
    return err["message"] if isinstance(err, dict) else err


@pytest.mark.parametrize("name", ["decide", "rerank"])
def test_decide_and_rerank_refuse_a_score_that_is_not_finite(server, name):
    """R6.68.119: decide's and rerank's scores are their answer. From a
    non-finite forward they answered 200: decide an argmax (its comparison
    skips NaN, so option 0 or a finite one won) with null probabilities, and
    rerank an order sorted by a comparator a NaN breaks. Both refuse now, as
    embeddings do, with an error naming the score."""
    path, body = REQUESTS[name]
    status, raw = _post(server, path, body)
    assert status == 500, (status, raw[:300])
    assert "not finite" in _message(raw), raw[:300]


def test_the_cli_decide_refuses_a_score_that_is_not_finite(server, tmp_path):
    """--decide runs the same handler over JSON lines: an error line naming
    the score, and exit 1."""
    lines = tmp_path / "q.jsonl"
    lines.write_text(json.dumps(REQUESTS["decide"][1]) + "\n")
    p = subprocess.run([find_runner(ROOT), "-m", server.model, "--decide", str(lines),
                        "--gpu", "off", "-t", "1"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    assert p.returncode != 0, p.stdout.decode(errors="replace")
    out = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
    assert out and "not finite" in out[0]["error"]["message"], out
