"""/v1/completions `top_logprobs` is a JSON object: it must not repeat a key.

Two ids can share a key. A byte-level vocabulary splits a multi-byte character
across ids, and every lone partial byte escapes to the same "\\ufffd", so
several of a position's candidates can render identically (the Mellum2.1 parity
record, 2026-10-29, found five positions that repeated a key). The server merges
candidates that share an escaped spelling into one entry carrying the log of
their summed probability, and `top_token_ids` keeps every id.

The fixture's byte-fallback vocabulary has 128 lone high bytes, so a top-20
table on random weights collides at almost every position: the test checks
that it did, so it cannot pass vacuously.
"""
import collections
import json
import math
import pathlib
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

PROMPT = "the quick brown fox jumps over the lazy dog"


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("lpkeys") / "dense.gguf"
    subprocess.run([sys.executable, ROOT / "scripts" / "make-test-model.py",
                    str(path)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


def _post_raw(srv, payload):
    req = urllib.request.Request(srv.base_url + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        assert r.status == 200
        return r.read()


def _parse_strict(raw):
    """json.loads, recording every key an object repeats."""
    repeated = []

    def hook(pairs):
        counts = collections.Counter(k for k, _ in pairs)
        repeated.extend(k for k, n in counts.items() if n > 1)
        return dict(pairs)

    return json.loads(raw, object_pairs_hook=hook), repeated


def test_top_logprobs_never_repeats_a_key(runner_bin, model):
    with RunnerServer(runner_bin, model, ctx=256,
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        raw = _post_raw(srv, {"prompt": PROMPT, "max_tokens": 4, "temperature": 0,
                              "echo": True, "logprobs": 20})
        body, repeated = _parse_strict(raw)
        assert repeated == [], f"repeated keys: {repeated!r}"
        lp = body["choices"][0]["logprobs"]
        merged_somewhere = False
        for top, ids in zip(lp["top_logprobs"], lp["top_token_ids"]):
            if top is None:
                continue
            assert len(ids) == 20, "top_token_ids keeps every id"
            assert len(set(ids)) == 20
            assert 1 <= len(top) <= len(ids)
            merged_somewhere |= len(top) < len(ids)
            # the merged map is still a distribution's top slice
            assert sum(math.exp(v) for v in top.values()) <= 1.0 + 1e-6
        assert merged_somewhere, "the fixture produced no shared spelling"


def test_merged_entry_is_the_log_of_the_summed_probability(runner_bin, model):
    with RunnerServer(runner_bin, model, ctx=256,
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        echo, _ = _parse_strict(_post_raw(srv, {"prompt": PROMPT, "max_tokens": 0,
                                                "echo": True, "logprobs": 20}))
        per_id, _ = _parse_strict(_post_raw(srv, {"prompt": PROMPT, "max_tokens": 1,
                                                  "temperature": 0,
                                                  "prompt_logprobs": 20}))
    top = echo["choices"][0]["logprobs"]["top_logprobs"]
    pl = per_id["choices"][0]["prompt_logprobs"]
    assert len(top) == len(pl)
    checked = 0
    for i in range(1, len(pl)):
        # vLLM's shape is keyed by id: group its entries by spelling. It may
        # also carry the actual token beside the top 20; keep the ranked 20.
        ranked = sorted(pl[i].values(), key=lambda v: v["rank"])[:20]
        groups = collections.defaultdict(list)
        for v in ranked:
            groups[v["decoded_token"]].append(v["logprob"])
        expect = {k: math.log(sum(math.exp(x) for x in lps))
                  for k, lps in groups.items()}
        assert set(top[i]) == set(expect), i
        for k, v in expect.items():
            assert top[i][k] == pytest.approx(v, abs=1e-5), (i, k)
        checked += any(len(lps) > 1 for lps in groups.values())
    assert checked, "no prompt position merged two ids"
