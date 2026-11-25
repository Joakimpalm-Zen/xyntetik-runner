**Limits first: DRAFT, not release-ready.** Claims 2 and 3 and the OLMo CPU
speed rows are unfinished. The actual 1.22.0 artifacts, Windows and Mac
release gates, compat/device reports and final release-check remain pending.
No tag, push, publication or site deployment is part of this preparation.

Prepares 1.22.0 from `83da2dc3`: a compatible minor release for Inkling 2
Mini and OLMo 2/3 support and new execution paths. Version pins are a separate
commit (`0ea15c6e`); README calls the candidate unpublished and keeps 1.21.0
identified as the last published release. The house fold tool assembled
all 41 pending entries plus the run-3 result entry, without hand-merging
bodies. The provisional date is to be confirmed at the actual cut.

The [notes draft](RELEASE-NOTES.md) leads with limits and includes the
Inkling fast CPU default's run-3 verdict/figures and Claim 1 result, the
scoped opt-in CUDA measurement, OLMo backend/CPU-training admission and the
7B bar, Q8_0 tensor-core prefill, and Windows gate test fixes. The
[site-copy draft](SITE-DRAFT.md) leads with the feature claim and keeps its
limits in the same passage. Neither draft supplies results for the three
explicit placeholders.

[Gate record](GATE-RECORD.md): `make -n release-check` prints a plan only;
the checker has no dedicated dry-validation mode. The real release gate
has not passed. Source/text consistency checks are preparation, not a
substitute for artifacts or device gates. The release checklist remains the
procedure for those runs and the final publication decision.

README/manual impact: synchronized candidate pins, clear unpublished
summary and links to the CPU result. Site impact: reviewable copy only,
no deployment. Model-card impact: no files/cards published by this PR;
existing community file identities and dated backend evidence are retained.
