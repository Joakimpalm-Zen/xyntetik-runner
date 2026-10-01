"""POST /v1/rerank: relevance as a constrained yes/no choice (R10.3.1).

No reranker model is needed: each document is put to the served model as a
question with exactly two legal answers, and the score is the model's own
probability of "yes" renormalized over {yes, no}. The readout is /v1/decide's
exact in-context scorer, which tests/test_decide.py already pins against
`--score`, so the chain of anchors is rerank -> decide -> score.

The absolute anchor is the zero-branch fixture: its attention and FFN write
nothing into the residual stream, so the logits at a position depend on that
position's token alone. Every document's prompt ends in the same tokens, so
the fixture CANNOT tell documents apart, by construction and not by anything
the runner computes: every score must be identical, and a tie must keep the
input order.
"""
import json
import math
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

DOCS = ["the cat sat on the mat", "stock prices fell sharply today",
        "a kitten slept on a rug", "quantum chromodynamics"]
QUERY = "where did the cat sleep?"
INSTRUCTION = ("Judge whether the document answers the query. "
               "Answer yes or no.")


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def _fixture(tmp_path_factory, name, *flags):
    m = tmp_path_factory.mktemp("rerank") / f"{name}.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"),
                    *flags, str(m)], check=True, stdout=subprocess.DEVNULL)
    return m


@pytest.fixture(scope="module")
def dense(tmp_path_factory):
    return _fixture(tmp_path_factory, "dense")


@pytest.fixture(scope="module")
def blind(tmp_path_factory):
    return _fixture(tmp_path_factory, "blind", "--zero-branches")


def _post(srv, path, payload):
    req = urllib.request.Request(srv.base_url + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as r:
        return r.code, json.load(r)


def _serve(runner_bin, model):
    return RunnerServer(runner_bin, model, ctx=1024,
                        extra_args=["--gpu", "off", "-t", "2"])


def _raw_prompt(query, doc):
    # the documented raw-v1 rendering, spelled out independently here
    return (f"{INSTRUCTION}\n\nQuery: {query}\nDocument: {doc}\n"
            f"Relevant:")


def test_raw_scores_are_the_decide_readout_of_the_same_prompt(runner_bin,
                                                              dense):
    with _serve(runner_bin, dense) as srv:
        st, body = _post(srv, "/v1/rerank", {"query": QUERY, "documents": DOCS,
                                             "rendering": "raw-v1"})
        assert st == 200, body
        assert body["object"] == "rerank"
        res = body["results"]
        assert sorted(r["index"] for r in res) == list(range(len(DOCS)))
        by_index = {r["index"]: r for r in res}
        for i, doc in enumerate(DOCS):
            st, d = _post(srv, "/v1/decide", {
                "state": _raw_prompt(QUERY, doc),
                "rendering": "continuation-v1",
                "questions": [{"options": [" yes", " no"]}]})
            assert st == 200, d
            dec = d["decisions"][0]
            r = by_index[i]
            assert r["relevance_score"] == pytest.approx(dec["probs"][0],
                                                         abs=1e-6)
            assert r["logit"] == pytest.approx(
                dec["logprobs"][0] - dec["logprobs"][1], abs=2e-6)
        # the relevance score is the logistic of the logit, exactly
        for r in res:
            assert r["relevance_score"] == pytest.approx(
                1 / (1 + math.exp(-r["logit"])), rel=1e-6)
        # ranked best first, and each margin is the gap to the next rank
        logits = [r["logit"] for r in res]
        assert logits == sorted(logits, reverse=True)
        for a, b in zip(res, res[1:]):
            assert a["margin"] == pytest.approx(a["logit"] - b["logit"],
                                                abs=1e-6)
        assert res[-1]["margin"] is None
        assert body["usage"]["prompt_tokens"] > 0
        env = body["envelope"]
        assert env["rendering"] == "raw-v1" and env["options"] == ["yes", "no"]


def test_the_blind_fixture_ties_every_document(runner_bin, blind):
    with _serve(runner_bin, blind) as srv:
        for rendering in ("chat-v1", "raw-v1"):
            st, body = _post(srv, "/v1/rerank", {
                "query": QUERY, "documents": DOCS, "rendering": rendering})
            assert st == 200, body
            res = body["results"]
            assert len({r["relevance_score"] for r in res}) == 1, res
            assert [r["index"] for r in res] == list(range(len(DOCS)))
            assert all(r["margin"] in (0, None) for r in res)


def test_document_order_does_not_change_a_score(runner_bin, dense):
    with _serve(runner_bin, dense) as srv:
        _, a = _post(srv, "/v1/rerank", {"query": QUERY, "documents": DOCS})
        _, b = _post(srv, "/v1/rerank", {"query": QUERY,
                                         "documents": DOCS[::-1]})
        sa = {DOCS[r["index"]]: r["logit"] for r in a["results"]}
        sb = {DOCS[::-1][r["index"]]: r["logit"] for r in b["results"]}
        assert sa == sb


def test_top_n_documents_and_object_form(runner_bin, dense):
    with _serve(runner_bin, dense) as srv:
        docs = [{"text": d} for d in DOCS]
        st, full = _post(srv, "/v1/rerank", {"query": QUERY, "documents": docs,
                                             "return_documents": True})
        assert st == 200, full
        for r in full["results"]:
            assert r["document"]["text"] == DOCS[r["index"]]
        st, top = _post(srv, "/v1/rerank", {"query": QUERY, "documents": DOCS,
                                            "top_n": 2})
        assert st == 200, top
        assert len(top["results"]) == 2
        assert "document" not in top["results"][0]
        assert [r["index"] for r in top["results"]] == \
               [r["index"] for r in full["results"][:2]]


@pytest.mark.parametrize("payload, needle", [
    ({"documents": DOCS}, "query"),
    ({"query": "", "documents": DOCS}, "query"),
    ({"query": QUERY}, "documents"),
    ({"query": QUERY, "documents": []}, "documents"),
    ({"query": QUERY, "documents": ["ok", 3]}, "document"),
    ({"query": QUERY, "documents": [{"body": "x"}]}, "document"),
    ({"query": QUERY, "documents": DOCS, "top_n": 0}, "top_n"),
    ({"query": QUERY, "documents": DOCS, "top_n": 1.5}, "top_n"),
    ({"query": QUERY, "documents": DOCS, "rendering": "fancy"}, "rendering"),
    ({"query": QUERY, "documents": DOCS, "instruction": 7}, "instruction"),
])
def test_malformed_requests_name_the_problem(runner_bin, dense, payload,
                                             needle):
    with _serve(runner_bin, dense) as srv:
        st, body = _post(srv, "/v1/rerank", payload)
        assert st == 400, body
        assert needle in body["error"]["message"]
