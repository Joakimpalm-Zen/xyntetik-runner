"""The evidence contract for shadow mode (R14.1).

One record per observed episode. The disposition is a closed list so that
every episode lands somewhere and the denominators are complete: an episode
the instrument could not evaluate is counted as such, never dropped. The
identity partition names the subject actually measured (model, quant,
adapter, runner build, backend, harness, verifier, environment), because a
new adapter or verifier opens a new cohort and old evidence becomes history.

Two invariants are enforced by construction rather than by review:

* a record cannot claim ``verified_local_attempt`` unless it carries a
  verifier outcome that passed with no tamper finding, and cannot claim
  ``local_failed`` unless the verifier failed or found tampering;
* a record cannot carry frontier content: the flag exists so that an
  importer has to state it, and stating it true refuses the record.

The summary shows counts first and both denominators (verified over
eligible, verified over everything observed). It refuses to print a
percentage before thirty independent eligible episodes, and it never
prints one over the observed total at all: "N of your tasks" is a count.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = "xyntetik.shadow.evidence.v1"
HEADLINE_MIN_EPISODES = 30


class Disposition(str, Enum):
    INELIGIBLE = "ineligible"
    NOT_ATTEMPTED_RESOURCE = "not_attempted_resource"
    UNREPLAYABLE = "unreplayable"
    VERIFIER_INCONCLUSIVE = "verifier_inconclusive"
    LOCAL_FAILED = "local_failed"
    VERIFIED_LOCAL_ATTEMPT = "verified_local_attempt"
    AGREEMENT_ONLY = "agreement_only"


ELIGIBLE = frozenset({
    Disposition.NOT_ATTEMPTED_RESOURCE,
    Disposition.VERIFIER_INCONCLUSIVE,
    Disposition.LOCAL_FAILED,
    Disposition.VERIFIED_LOCAL_ATTEMPT,
    Disposition.AGREEMENT_ONLY,
})
ATTEMPTED = ELIGIBLE - {Disposition.NOT_ATTEMPTED_RESOURCE}
CHECKED = frozenset({Disposition.LOCAL_FAILED, Disposition.VERIFIED_LOCAL_ATTEMPT})


@dataclass(frozen=True)
class Identity:
    """The subject measured. Not the model name: the whole stack."""

    project: str
    task_class: str
    context_band: str
    tool_set: tuple[str, ...]
    verifier_id: str
    environment_id: str
    model_sha256: str
    quant: str
    template_sha256: str
    runner_build: str
    backend: str
    harness_version: str
    adapter_sha256: str | None = None
    adapter_scale: float | None = None

    def cohort_key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class VerifierOutcome:
    """What the protected verifier saw. ``passed`` is ``None`` when the
    verifier could not reach a verdict (timeout, crash)."""

    verifier_id: str
    passed: bool | None
    expected: int
    passed_count: int
    failed: int
    skipped: int
    missing: int
    tamper: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    output_sha256: str = ""
    duration_s: float = 0.0


@dataclass(frozen=True)
class EpisodeEvidence:
    episode_id: str
    source: str
    observed_at: str
    disposition: Disposition
    identity: Identity
    baseline_sha256: str
    patch_sha256: str | None
    changed_paths: tuple[str, ...]
    verifier: VerifierOutcome | None
    wall_s: float
    resources: Mapping[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    frontier_content_loaded: bool = False
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.frontier_content_loaded:
            raise ValueError("a record that loaded frontier content is refused: "
                             "the importer reads boundary, request, directory and time only")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unknown schema_version {self.schema_version!r}")
        v = self.verifier
        d = self.disposition
        if d is Disposition.VERIFIED_LOCAL_ATTEMPT:
            if v is None or v.passed is not True or v.tamper or v.skipped or v.missing:
                raise ValueError("verified_local_attempt needs a verifier outcome that passed "
                                 "with no tamper, skipped or missing tests")
        elif d is Disposition.LOCAL_FAILED:
            if v is None or v.passed is None or (v.passed and not v.tamper):
                raise ValueError("local_failed needs a verifier outcome that failed or found "
                                 "tampering")
        elif d is Disposition.VERIFIER_INCONCLUSIVE:
            if v is not None and v.passed is not None:
                raise ValueError("verifier_inconclusive cannot carry a verdict")
        elif v is not None:
            raise ValueError(f"{d.value} carries no verifier outcome")

    def to_json(self) -> str:
        data = asdict(self)
        data["disposition"] = self.disposition.value
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> EpisodeEvidence:
        data: dict[str, Any] = json.loads(text)
        ident = data.pop("identity")
        ident["tool_set"] = tuple(ident["tool_set"])
        ver = data.pop("verifier")
        if ver is not None:
            ver["tamper"] = tuple(ver["tamper"])
            ver["reasons"] = tuple(ver["reasons"])
        return cls(
            identity=Identity(**ident),
            verifier=VerifierOutcome(**ver) if ver is not None else None,
            disposition=Disposition(data.pop("disposition")),
            changed_paths=tuple(data.pop("changed_paths")),
            reasons=tuple(data.pop("reasons")),
            **data,
        )


@dataclass(frozen=True)
class Summary:
    observed: int
    eligible: int
    attempted: int
    checked: int
    verified: int
    failed: int
    inconclusive: int
    agreement_only: int
    cohorts: int
    by_disposition: Mapping[str, int]

    @property
    def headline_allowed(self) -> bool:
        return self.eligible >= HEADLINE_MIN_EPISODES


def summarize(records: Iterable[EpisodeEvidence]) -> Summary:
    by: dict[str, int] = {d.value: 0 for d in Disposition}
    cohorts: set[str] = set()
    n = 0
    for r in records:
        n += 1
        by[r.disposition.value] += 1
        cohorts.add(r.identity.cohort_key())
    count = {d: by[d.value] for d in Disposition}
    return Summary(
        observed=n,
        eligible=sum(count[d] for d in ELIGIBLE),
        attempted=sum(count[d] for d in ATTEMPTED),
        checked=sum(count[d] for d in CHECKED),
        verified=count[Disposition.VERIFIED_LOCAL_ATTEMPT],
        failed=count[Disposition.LOCAL_FAILED],
        inconclusive=count[Disposition.VERIFIER_INCONCLUSIVE],
        agreement_only=count[Disposition.AGREEMENT_ONLY],
        cohorts=len(cohorts),
        by_disposition=by,
    )


def render(s: Summary) -> str:
    """Counts first, both denominators, no percentage before the floor."""
    lines = [
        f"{s.observed} episodes observed",
        f"{s.eligible} eligible for replay",
        f"{s.attempted} attempted",
        f"{s.verified} passed independent checks",
        f"{s.failed} failed checks",
        f"{s.inconclusive} could not be evaluated",
        f"{s.agreement_only} agreement only (not evidence)",
        f"{s.observed - s.eligible} outside this evaluation's scope",
        f"{s.cohorts} identity cohorts",
    ]
    if s.eligible:
        ratio = f"verified over eligible: {s.verified} of {s.eligible}"
        if s.headline_allowed:
            ratio += f" ({100 * s.verified / s.eligible:.0f}%)"
        lines.append(ratio)
    if s.observed:
        lines.append(f"verified over observed: {s.verified} of {s.observed}")
    if not s.headline_allowed:
        lines.append(f"(no percentage before {HEADLINE_MIN_EPISODES} independent eligible episodes)")
    return "\n".join(lines)
