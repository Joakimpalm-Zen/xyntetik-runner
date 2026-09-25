"""Margin-qualified top-1 agreement (quality bar v2).

Plain top-1 counts every argmax flip as a disagreement, including flips
between two tokens the REFERENCE model itself could barely separate. Measured
2026-08-13, that is what the 5-6 bit band of the quality ladder was failing
on: granite-4.1-3b Q6_K scored mean KLD 0.0155 — three times inside the 0.05
bound — while missing the 97% top-1 criterion by four points, and
Phi-4-mini Q8_0 scored KLD 0.0082 with 94.75% top-1. Distributions that close
are not damaged; their argmaxes are coin-flipping at near-ties.

v2 ADDS a column. Plain top-1 is still computed and still reported; these
tests pin the new one's semantics, especially the two that are easy to get
wrong: the band edge, and which side's margin is consulted.
"""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "kld_raw", ROOT / "scripts" / "kld-compare-raw.py")
kld_raw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kld_raw)

BAND = kld_raw.DEFAULT_TIE_BAND


def dist(**kw):
    return dict(kw)


def test_agreement_is_agreement_under_both():
    # same argmax: both columns count it, margin never consulted
    a = dist(x=-0.1, y=-3.0)
    b = dist(x=-0.2, y=-2.0)
    kld, agree, marg, overlap = kld_raw.score_pair(a, b)
    assert agree is True
    assert marg is True


def test_exact_tie_on_the_reference_is_forgiven():
    # The reference cannot separate x from y at all, so whichever the variant
    # picks is not evidence of damage. NB max() breaks the reference's tie
    # toward its first key, so the variant must prefer the OTHER one for the
    # argmaxes to differ at all — an exact tie where both happen to land on
    # the same token is plain agreement, not a forgiven flip.
    a = dist(x=-0.9, y=-0.5)     # variant picks y
    b = dist(x=-1.0, y=-1.0)     # reference is exactly tied, max() picks x
    _, agree, marg, _ = kld_raw.score_pair(a, b)
    assert agree is False
    assert marg is True


def test_just_inside_the_band_is_forgiven():
    a = dist(x=-0.1, y=-5.0)                 # variant picks x
    b = dist(x=-1.0 - BAND * 0.9, y=-1.0)    # reference picks y, narrowly
    _, agree, marg, _ = kld_raw.score_pair(a, b)
    assert agree is False
    assert marg is True


def test_just_outside_the_band_is_not_forgiven():
    a = dist(x=-0.1, y=-5.0)                 # variant picks x
    b = dist(x=-1.0 - BAND * 1.1, y=-1.0)    # reference prefers y decisively
    _, agree, marg, _ = kld_raw.score_pair(a, b)
    assert agree is False
    assert marg is False


def test_margin_is_read_from_the_reference_not_the_variant():
    # The asymmetry that matters: the VARIANT is unsure (tiny margin) while the
    # REFERENCE is certain. That is a real disagreement — the reference had a
    # clear opinion and the variant lost it — so it must NOT be forgiven.
    a = dist(x=-1.0, y=-1.0 - BAND * 0.1)    # variant barely picks x
    b = dist(x=-9.0, y=-0.001)               # reference is certain it is y
    _, agree, marg, _ = kld_raw.score_pair(a, b)
    assert agree is False
    assert marg is False, "a confident reference must not be forgiven"

    # and the mirror image: reference unsure, variant certain -> forgiven
    a2 = dist(x=-0.001, y=-9.0)
    b2 = dist(x=-1.0 - BAND * 0.1, y=-1.0)
    _, agree2, marg2, _ = kld_raw.score_pair(a2, b2)
    assert agree2 is False
    assert marg2 is True


def test_single_entry_reference_has_no_margin_and_is_not_forgiven():
    # A reference distribution with one usable entry gives no top-two gap to
    # measure. Refusing to forgive is the conservative reading.
    a = dist(x=-0.1, y=-4.0)
    b = dist(y=-0.2)
    _, agree, marg, _ = kld_raw.score_pair(a, b)
    assert agree is False
    assert marg is False


def test_a_pick_outside_the_band_is_not_forgiven_however_close_the_top_two_are():
    """The criterion must consult what the VARIANT picked, not just the gap.

    Until 2026-08-19 this asked only "were the reference's #1 and #2 close?".
    That question does not mention the variant at all, so a near-tie at the top
    of the reference forgave EVERY flip below it -- including a variant
    confidently emitting a token the reference rates at e^-12. The tie band is
    supposed to forgive a coin-flip between two candidates the reference cannot
    separate; it was forgiving anything at all as long as a coin-flip existed
    somewhere in the distribution.
    """
    b = dist(x=-1.0, y=-1.2, z=-12.0)   # #1/#2 within the band, z is nowhere
    assert kld_raw.top_two_margin(b) < BAND    # the old criterion's whole test
    a = dist(x=-9.0, y=-9.0, z=-0.01)   # the variant is certain it is z
    _, agree, marg, _ = kld_raw.score_pair(a, b)
    assert agree is False
    assert marg is False, "a pick the reference rates at e^-12 is not a near-tie"


def test_a_pick_inside_the_band_is_still_forgiven_in_a_wider_distribution():
    # the other half of the case above: same reference, and the variant picks
    # the runner-up the reference genuinely could not separate from its top.
    b = dist(x=-1.0, y=-1.2, z=-12.0)
    a = dist(x=-9.0, y=-0.01, z=-9.0)   # variant picks y, inside the band
    _, agree, marg, _ = kld_raw.score_pair(a, b)
    assert agree is False
    assert marg is True


def test_a_pick_the_reference_never_reported_is_not_forgiven():
    """Absent from the reference's top-N is not evidence of a near-tie.

    Both sides report a truncated top-N, so a variant's argmax can simply be
    missing from the reference's list. There is no gap to measure against it,
    and treating "unmeasured" as "tied" would forgive exactly the divergences
    the criterion exists to catch."""
    b = dist(x=-1.0, y=-1.05)           # reference's own top two are tied
    a = dist(q=-0.01, x=-9.0)           # variant picks a token b never reported
    _, agree, marg, _ = kld_raw.score_pair(a, b)
    assert agree is False
    assert marg is False


def test_band_is_documented_and_conservative():
    # tc-tol forgives a flip inside 0.02 of the logit range; on the models this
    # project measures that is 0.5-1.1 nats (see the derivation in the script).
    # The default must sit at the conservative end of that translation.
    assert 0.4 <= BAND <= 0.6


# ---------------------------------------------------------- stop positions
#
# A position whose greedy next token is a stop has no emitted token, so the
# logprobs arrays are empty and the position used to be counted as FAILED and
# dropped. Both-sides stops cost nothing; a ONE-SIDED stop is a top-1
# disagreement, and dropping it biased agreement upward (lab, 2026-09-25:
# 1 of 500 positions on a release pass).

class _Resp:
    def __init__(self, body):
        self._b = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        import json
        return json.dumps(self._b).encode()


def _serve(monkeypatch, body):
    monkeypatch.setattr(kld_raw.urllib.request, "urlopen",
                        lambda req, timeout=0: _Resp(body))


def test_a_stop_position_raises_stop_not_keyerror(monkeypatch):
    _serve(monkeypatch, {"choices": [{
        "finish_reason": "stop", "stop_token": "</s>", "stop_token_id": 2,
        "stop_logprobs": {"logprob": -0.5,
                          "top_logprobs": {"</s>": -0.5, "x": -1.5},
                          "top_token_ids": [2, 9]}}]})
    with pytest.raises(kld_raw.StopPosition) as e:
        kld_raw.query("http://x", "m", "prefix")
    assert e.value.token == "</s>"
    # the distribution rides along, so a both-sides stop can still be scored
    assert e.value.dist == {"</s>": -0.5, "x": -1.5}


def test_a_stop_without_a_distribution_is_still_a_stop(monkeypatch):
    """A server older than the stop_logprobs field, or one that reports none:
    the position is still known to be a stop and still counts, it just carries
    no distribution to put in the KLD mean."""
    _serve(monkeypatch, {"choices": [{"finish_reason": "stop",
                                      "stop_token": "<|eot|>"}]})
    with pytest.raises(kld_raw.StopPosition) as e:
        kld_raw.query("http://x", "m", "prefix")
    assert e.value.token == "<|eot|>" and e.value.dist is None


def test_an_empty_logprobs_block_without_a_stop_is_still_an_error(monkeypatch):
    """Only a STOP explains a missing distribution. Anything else is the bug
    the old except branch was hiding, and must not be read as a stop."""
    _serve(monkeypatch, {"choices": [{"finish_reason": "length",
                                      "logprobs": {"tokens": []}}]})
    with pytest.raises(KeyError):
        kld_raw.query("http://x", "m", "prefix")


def test_two_stops_on_the_same_token_agree_and_score(monkeypatch):
    """Both sides stopping on the same token is agreement, and when both
    carry a distribution the position also joins the KLD mean."""
    d = {"</s>": -0.5, "x": -1.5}
    kld, agree, marg, overlap = kld_raw.score_pair(d, d)
    assert agree is True and marg is True and kld == pytest.approx(0.0, abs=1e-9)


def test_every_position_is_counted_exactly_once(monkeypatch, tmp_path):
    """The rates divide by the number of scored positions, and a stop must
    not be counted twice. An earlier version appended in the stop branch and
    then again in the scorer, so a 500-position run reported rates over 504
    (lab, 2026-09-25)."""
    import json as _json
    stop_body = {"choices": [{"finish_reason": "stop", "stop_token": "</s>",
                              "stop_logprobs": {"logprob": -0.1,
                                                "top_logprobs": {"</s>": -0.1, "x": -3.0},
                                                "top_token_ids": [2, 9]}}]}
    text_body = {"choices": [{"finish_reason": "length", "logprobs": {
        "tokens": ["x"], "token_logprobs": [-0.2],
        "top_logprobs": [{"x": -0.2, "</s>": -2.5}]}}]}
    # side A stops on every third position, side B never stops
    calls = {"n": 0}

    def fake(req, timeout=0):
        calls["n"] += 1
        a_side = calls["n"] % 2 == 1
        pos = (calls["n"] - 1) // 2
        return _Resp(stop_body if (a_side and pos % 3 == 0) else text_body)

    monkeypatch.setattr(kld_raw.urllib.request, "urlopen", fake)
    monkeypatch.setattr(kld_raw, "served_model_ids", lambda ep: ["a", "b"])
    corpus = tmp_path / "c.txt"
    corpus.write_text(" ".join(f"w{i}" for i in range(40)))
    out = tmp_path / "r.json"
    rc = kld_raw.main(["--endpoint-a", "http://a", "--model-name-a", "a",
                       "--endpoint-b", "http://b", "--model-name-b", "b",
                       "--corpus", str(corpus), "--max-positions", "9",
                       "--stride", "1", "--out", str(out)])
    assert rc == 0
    r = _json.loads(out.read_text())
    assert r["positions_scored"] == 9
    assert r["rate_denominator"] == r["positions_scored"]
    assert len(r["positions"]) == r["positions_scored"]
    assert r["positions_stop"] == 3 and r["positions_stop_one_sided"] == 3
    # every one-sided stop is a disagreement, so the rate is bounded by it
    assert r["top1_agreement_pct"] <= 100.0 * (9 - 3) / 9 + 1e-9
