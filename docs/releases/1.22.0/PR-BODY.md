**Limits first: DRAFT, not release-ready.** Claim 2 is the provisional sixteen-thread half, with one-thread cells
after the release. The measured OLMo 7B CPU rows are slower, with different
builds and tool workloads; the 1B CPU row is missing. The actual 1.22.0 artifacts, Windows and Mac
release gates, compat/device reports and final release-check remain pending.
No tag, push, publication or site deployment is part of this preparation.

Prepares 1.22.0 from `83da2dc3`: a compatible minor release for Inkling 2
Mini and OLMo 2/3 support and new execution paths. Version pins are a separate
commit (`0ea15c6e`); README calls the candidate unpublished and keeps 1.21.0
identified as the last published release. The house fold tool assembled
all 41 pending entries plus the run-3 and release Claims 1, 2 and 3 plus OLMo CPU-speed result entries, without hand-merging
bodies. The provisional date is to be confirmed at the actual cut.

The [notes draft](RELEASE-NOTES.md) leads with limits and includes the
Inkling fast CPU default's run-3 verdict/figures, separately from release
Claim 1's Runner-versus-llama.cpp comparison (pre-registered 40/100 versus
20/100 first, plus the additional Unsloth 16/100 column; erratum 3's
94/100 beside it, decided after the verdicts), the
scoped opt-in CUDA measurement, OLMo backend/CPU-training admission and the
7B bar, Q8_0 tensor-core prefill, and Windows gate test fixes. The
[site-copy draft](SITE-DRAFT.md) leads with the feature claim and keeps its
limits in the same passage. The notes include the final Claim 3 TIE, the provisional sixteen-thread
Claim 2 tables and additional Unsloth columns. The OLMo CPU-speed rows report all three slower ratios and their build,
prompt, batching and sampler differences; the 1B CPU row is absent. No
full-matrix Claim 2 result is claimed.

[Gate record](GATE-RECORD.md): `make -n release-check` prints a plan only;
the checker has no dedicated dry-validation mode. The real release gate
has not passed. Source/text consistency checks are preparation, not a
substitute for artifacts or device gates. The release checklist remains the
procedure for those runs and the final publication decision.

README/manual impact: synchronized candidate pins, clear unpublished
summary and links to the CPU result. Site impact: reviewable copy only,
no deployment. Model-card impact: no files/cards published by this PR;
existing community file identities and dated backend evidence are retained.
Unsloth columns use the published CPU prebuilt b12649-mix-c7059d6 as
published; the source build at the release tag commit could not load the model.
