# Published comparison inputs

The retained SmolLM2 refresh reports (`report.json` per runtime) name the
measured versions used by `scripts/competitor-freshness.py`; the raw
request/response transcripts behind them are archived byte for byte in
Shade under `research/runner-engine/archive/runner-docs-2026-10-13c/`. They are dated comparison inputs, not
fresh certifications of newer runtimes. New local matrix output can be
written here with `scripts/agent-torture.py --out`; select and publish only
evidence behind a current claim. The six-engine truncation evidence is
retained separately under `tests/torture/truncation/`.
