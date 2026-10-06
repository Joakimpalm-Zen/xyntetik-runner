# Community results

Results from people who ran a published file or the engine on their own
hardware. Success and failure are both recorded. To add one, open a
[result report](https://github.com/Joakimpalm-Zen/xyntetik-runner/issues/new?template=result_report.yml);
to ask for a file, open a
[model request](https://github.com/Joakimpalm-Zen/xyntetik-runner/issues/new?template=model_request.yml).

What goes here: the file or model, the engine and version, the hardware, what
was run, what was measured and over how many runs, and what it was compared
against. A single run is recorded as a single run. Nothing on this page is a
project claim; the project's own measurements are in the README and the model
cards, each with its method.

## Results

| Date | Who | File or feature | Hardware | What was measured | Result |
|---|---|---|---|---|---|
| 2026-08-23 | independent reproduction, Hugging Face forum | LoRA training determinism, GPU-assisted backward | Tesla T4 (sm_75) | adapter sha256 and loss, CPU against GPU | Identical, byte for byte. The same report found that rebuilding from source under a different ISA profile changes the adapter bytes; that boundary is now part of the [determinism scope](determinism-scope.md). Written up in [adaptation-engine.md](adaptation-engine.md#reproducibility-scoped-by-an-external-reproduction). |

## Open requests

Model requests are tracked as issues with the
[`model-request`](https://github.com/Joakimpalm-Zen/xyntetik-runner/issues?q=is%3Aissue+label%3Amodel-request)
label. A request does not promise a file: a derivative is published only after
it is measured against its parent, and the result is published either way.
