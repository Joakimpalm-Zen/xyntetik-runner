"""The evidence contract: a record cannot claim what its verifier did not
see, cannot carry frontier content, and the summary never headlines a
percentage before the floor."""
from __future__ import annotations

import pytest

from xyntetik_runner.shadow import (
    HEADLINE_MIN_EPISODES,
    Disposition,
    EpisodeEvidence,
    Identity,
    VerifierOutcome,
    render,
    summarize,
)

IDENT = Identity(project="p", task_class="repair", context_band="<8k", tool_set=("read", "write"),
                 verifier_id="repair_task_v1", environment_id="host", model_sha256="m" * 64,
                 quant="Q4_K_M", template_sha256="t" * 64, runner_build="0.5.1", backend="cpu",
                 harness_version="shadow-0.1")


def passing(**kw: object) -> VerifierOutcome:
    base = dict(verifier_id="repair_task_v1", passed=True, expected=6, passed_count=6,
                failed=0, skipped=0, missing=0)
    base.update(kw)
    return VerifierOutcome(**base)  # type: ignore[arg-type]


def record(disposition: Disposition, verifier: VerifierOutcome | None, **kw: object) -> EpisodeEvidence:
    fields = dict(episode_id="e1", source="manual", observed_at="2026-09-08T00:00:00Z",
                  disposition=disposition, identity=IDENT, baseline_sha256="b" * 64,
                  patch_sha256="p" * 64, changed_paths=("calc/money.py",), verifier=verifier,
                  wall_s=1.0)
    fields.update(kw)
    return EpisodeEvidence(**fields)  # type: ignore[arg-type]


def test_verified_needs_a_clean_pass() -> None:
    record(Disposition.VERIFIED_LOCAL_ATTEMPT, passing())
    for bad in (None, passing(passed=False), passing(tamper=("x",)), passing(skipped=1),
                passing(missing=1), passing(passed=None)):
        with pytest.raises(ValueError):
            record(Disposition.VERIFIED_LOCAL_ATTEMPT, bad)


def test_local_failed_needs_a_failure_or_tamper() -> None:
    record(Disposition.LOCAL_FAILED, passing(passed=False, failed=1))
    record(Disposition.LOCAL_FAILED, passing(passed=True, tamper=("conftest.py",)))
    for bad in (None, passing(), passing(passed=None)):
        with pytest.raises(ValueError):
            record(Disposition.LOCAL_FAILED, bad)


def test_inconclusive_carries_no_verdict_and_others_carry_no_verifier() -> None:
    record(Disposition.VERIFIER_INCONCLUSIVE, None)
    record(Disposition.VERIFIER_INCONCLUSIVE, passing(passed=None))
    with pytest.raises(ValueError):
        record(Disposition.VERIFIER_INCONCLUSIVE, passing())
    for d in (Disposition.INELIGIBLE, Disposition.UNREPLAYABLE, Disposition.AGREEMENT_ONLY,
              Disposition.NOT_ATTEMPTED_RESOURCE):
        record(d, None)
        with pytest.raises(ValueError):
            record(d, passing())


def test_frontier_content_is_refused_by_construction() -> None:
    with pytest.raises(ValueError, match="frontier content"):
        record(Disposition.INELIGIBLE, None, frontier_content_loaded=True)


def test_json_round_trip() -> None:
    r = record(Disposition.VERIFIED_LOCAL_ATTEMPT, passing(reasons=("ok",)),
               resources={"rss_mb": 12.5}, reasons=("fine",))
    assert EpisodeEvidence.from_json(r.to_json()) == r


def test_summary_is_per_episode_and_per_cohort_and_withholds_the_percentage() -> None:
    other = Identity(**{**IDENT.__dict__, "model_sha256": "o" * 64})
    recs = [record(Disposition.VERIFIED_LOCAL_ATTEMPT, passing(), episode_id=f"v{i}")
            for i in range(4)]
    recs += [record(Disposition.LOCAL_FAILED, passing(passed=False, failed=1), episode_id=f"f{i}")
             for i in range(2)]
    # the same episodes attempted by a second stack: still six episodes
    recs += [record(Disposition.LOCAL_FAILED, passing(passed=False, failed=1), episode_id=f"v{i}",
                    identity=other) for i in range(4)]
    recs += [record(Disposition.INELIGIBLE, None, episode_id="i0"),
             record(Disposition.UNREPLAYABLE, None, episode_id="u0"),
             record(Disposition.VERIFIER_INCONCLUSIVE, None, episode_id="q0"),
             record(Disposition.AGREEMENT_ONLY, None, episode_id="a0"),
             record(Disposition.NOT_ATTEMPTED_RESOURCE, None, episode_id="n0"),
             record(Disposition.NOT_ATTEMPTED_RESOURCE, None, episode_id="v0")]
    s = summarize(recs)
    assert (s.observed, s.eligible, s.unattempted, s.agreement_only) == (11, 9, 1, 1)
    assert s.by_disposition["verified_local_attempt"] == 4 and s.by_disposition["local_failed"] == 2
    rows = {c.model: c for c in s.cohorts}
    assert rows["m" * 64].verified == 4 and rows["m" * 64].failed == 2
    assert rows["o" * 64].verified == 0 and rows["o" * 64].attempted == 4
    text = render(s)
    assert "11 episodes observed" in text and "9 eligible for replay" in text
    assert "verified over eligible 4 of 9; verified over observed 4 of 11" in text
    assert "%" not in text.split("(no percentage")[0]
    assert f"no percentage before {HEADLINE_MIN_EPISODES}" in text


def test_percentage_appears_only_at_the_floor_and_only_over_eligible() -> None:
    recs = [record(Disposition.VERIFIED_LOCAL_ATTEMPT, passing(), episode_id=f"v{i}")
            for i in range(HEADLINE_MIN_EPISODES)]
    recs += [record(Disposition.INELIGIBLE, None, episode_id=f"i{i}") for i in range(10)]
    text = render(summarize(recs))
    assert f"verified over eligible {HEADLINE_MIN_EPISODES} of {HEADLINE_MIN_EPISODES} (100%)" in text
    assert f"verified over observed {HEADLINE_MIN_EPISODES} of {HEADLINE_MIN_EPISODES + 10}" in text
    assert "no percentage" not in text
