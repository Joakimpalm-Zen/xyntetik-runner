"""Shadow mode (plan epic R14): evidence about what a local model can do.

Two things live here and nothing else. The evidence contract
(`evidence`): one record per observed episode with a disposition from a
closed list, an identity partition, and a verifier outcome, plus a summary
that shows counts and both denominators and refuses a headline percentage
before thirty independent eligible episodes. The protected verifier
(`verifier`): frozen tests that live outside the workspace, run on a fresh
copy of it, and cannot be satisfied by a no-op, a tampered test file, an
injected pytest configuration or a skip.

Stdlib only. This package does not route requests, does not train, and
does not sandbox: the verifier runs the workspace's tests on the host in a
scratch copy, and OS isolation is a separate story (R14.3).
"""

from xyntetik_runner.shadow.baseline import Baseline, Changes, file_sha256, tree_hashes, tree_sha256
from xyntetik_runner.shadow.evidence import (
    HEADLINE_MIN_EPISODES,
    SCHEMA_VERSION,
    Disposition,
    EpisodeEvidence,
    Identity,
    Summary,
    VerifierOutcome,
    render,
    summarize,
)
from xyntetik_runner.shadow.verifier import (
    CONFIG_BASENAMES,
    Calibration,
    InstrumentError,
    ProtectedTests,
    calibrate,
    check,
    verify,
)

__all__ = [
    "CONFIG_BASENAMES",
    "HEADLINE_MIN_EPISODES",
    "SCHEMA_VERSION",
    "Baseline",
    "Calibration",
    "Changes",
    "Disposition",
    "EpisodeEvidence",
    "Identity",
    "InstrumentError",
    "ProtectedTests",
    "Summary",
    "VerifierOutcome",
    "calibrate",
    "check",
    "file_sha256",
    "render",
    "summarize",
    "tree_hashes",
    "tree_sha256",
    "verify",
]
