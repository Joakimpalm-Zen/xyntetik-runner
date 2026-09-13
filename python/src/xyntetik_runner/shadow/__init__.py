"""Shadow execution and evidence contracts.

Capture, replay, verification, delegation and configured artifact serving use
only the standard library. Learning recipes are optional consumers in Shade.
Scratch worktrees are not an OS security sandbox; repository tests run on the
host with the operator's permissions.
"""

from xyntetik_runner.shadow.baseline import Baseline, Changes, file_sha256, tree_hashes, tree_sha256
from xyntetik_runner.shadow.evidence import (
    HEADLINE_MIN_EPISODES,
    SCHEMA_VERSION,
    CohortSummary,
    Disposition,
    EpisodeEvidence,
    Identity,
    Summary,
    VerifierOutcome,
    render,
    summarize,
)
from xyntetik_runner.shadow.scaffold import SCAFFOLD_SCHEMA, Scaffold
from xyntetik_runner.shadow.verifier import (
    CONFIG_BASENAMES,
    Calibration,
    InstrumentError,
    ProtectedTests,
    calibrate,
    check,
    fixed_ids,
    verify,
)

__all__ = [
    "CONFIG_BASENAMES",
    "HEADLINE_MIN_EPISODES",
    "SCAFFOLD_SCHEMA",
    "SCHEMA_VERSION",
    "Baseline",
    "Calibration",
    "Changes",
    "CohortSummary",
    "Disposition",
    "EpisodeEvidence",
    "Identity",
    "InstrumentError",
    "ProtectedTests",
    "Scaffold",
    "Summary",
    "VerifierOutcome",
    "calibrate",
    "check",
    "fixed_ids",
    "file_sha256",
    "render",
    "summarize",
    "tree_hashes",
    "tree_sha256",
    "verify",
]
