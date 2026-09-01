# gpt-oss-120b context-surgery report — 2026-09-01

## Decision

Two standalone, byte-preserving research GGUFs were produced: a 262,144-token
YaRN x64 candidate and a 524,288-token YaRN x128 candidate. Both are
mechanically valid, independently reopen, and contain byte-identical tensor
payloads relative to the 131,072-token parent. The 262K candidate preserves a
small 4K retrieval control exactly (9/9 parent, 9/9 candidate).

Neither artifact is qualified for long-context use. The first 32K CPU-only
screen took long enough that the nine-cell matrix was stopped after about 76
minutes of server time. One cell completed but its answer was not retained
because the interrupted harness writes JSON only at suite completion; it is
unscored. The 524K artifact was not inference-tested because the 262K artifact
did not clear its intermediate gate. Status is therefore **research artifact,
not recommended release**.

## Artifacts

| Artifact | Context metadata | YaRN factor | Bytes | SHA-256 | Status |
|---|---:|---:|---:|---|---|
| Parent `gpt-oss-120b-MXFP4.gguf` | 131,072 | 32 | 63,387,346,208 | `582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d` | control |
| `gpt-oss-120b-MXFP4-262k-yarn64.gguf` | 262,144 | 64 | 63,387,346,208 | `7e0d856c7b61d6e6fdf9be2af7d21fe739e4896048b76f798719e2a51bd4b2e8` | mechanical pass; 4K smoke pass; long gate incomplete |
| `gpt-oss-120b-MXFP4-524k-yarn128.gguf` | 524,288 | 128 | 63,387,346,208 | `331983a3fc8cc34fc7842c00356c8021bde80691014bccf2995fab0fa1c91299` | mechanical pass only |

The compiler changed only `gpt-oss.context_length` and
`gpt-oss.rope.scaling.factor`. It kept all 687 tensors in their original
types and independently reparsed and `memcmp`-verified names, types, shapes,
and payloads before success. The per-artifact `.context.json` files are the
machine-written source of truth.

## Frozen hypotheses and stopping rule

The preregistration was written before candidate inference. It expected x64 to
be brittle at long context and x128 to fail without training. It required
parent controls, three seeds at three depths, exact-match scoring, and a 262K
kill below 50% through 192K or total failure at two consecutive lengths. The
524K run was forbidden until the 262K intermediate gate passed.

The matrix stopped earlier for an operationally unwanted result: CPU-only 32K
turnaround made the planned ladder a multi-day job on this host. That is not a
model-quality kill under the preregistered thresholds, so no pass or fail is
assigned at 32K. This preserves the difference between “not demonstrated” and
“demonstrated not to work.”

## Results

| Gate | Parent x32 | 262K x64 | 524K x128 | Interpretation |
|---|---:|---:|---:|---|
| GGUF opens and advertises intended metadata | pass | pass | pass | Loader/mechanical only |
| Tensor payload identity | n/a | 687/687 byte-identical | 687/687 byte-identical | No weight surgery occurred |
| 4K retrieval, 3 seeds × 3 depths | 9/9 | 9/9 | not run | x64 preserved this short smoke |
| 32K retrieval | n/a | unscored/incomplete | not run | First cell completed; response not retained; matrix stopped |
| 64K–262K retrieval ladder | n/a | not run | not run | Not established |
| RULER / multi-needle / coding-tool acceptance | not run | not run | not run | Not qualified |

The first parent run produced empty `content` in 9/9 cases because GPT-OSS
spent the evaluator's 24-token ceiling in Harmony `reasoning_content`. That
record is retained as invalid evidence. Before any candidate output was seen,
the harness was corrected to discover Runner's `/v1/models` id and to send
`enable_thinking:false`; a public-interface regression test locks both
behaviors. The corrected parent then returned the three externally known codes
at every depth, establishing the absolute anchor.

## Resource estimates and observed host behavior

Runner's header-only estimates for this 59.03 GiB checkpoint are:

| Context | F16 KV | Q8 KV | Weights + F16 KV | Weights + Q8 KV |
|---:|---:|---:|---:|---:|
| 262,144 | 18.00 GiB | 9.56 GiB | 77.03 GiB | 68.59 GiB |
| 524,288 | 36.00 GiB | 19.12 GiB | 95.03 GiB | 78.15 GiB |

These totals exclude runtime scratch, allocator overhead, and the operating
system. They suggest both metadata targets can fit a 128 GB M5 Max, with 524K
F16 materially tighter. Fit is not quality and not acceptable latency.

The screening host was an AMD Ryzen Threadripper 9980X (64 physical cores,
128 logical), 250 GiB RAM, plus an RTX PRO 6000 Blackwell Max-Q MIG 1g.24gb
slice. The long screen intentionally used `--gpu off`, F16 KV, one server slot,
and Runner's 32-thread cap. At 32,743 prompt tokens, the first completed request
generated two answer tokens at 4.0 tok/s and incurred 22,972 major page-ins on
the cold standalone file. A second cell began with 3,286 cached prompt tokens
before the run was stopped.

The target M5 evidence already committed in Runner shows the 131K parent fully
resident with zero swap, roughly 75–80 prompt tok/s and 65.7 decode tok/s. The
next meaningful long-context screen should run there; extrapolation is not a
substitute for measurement.

## Runner changes

All Runner work is isolated on local branch `context-surgery`, based on current
`origin/main` commit `37fc11b`:

- `255a5f5` — native `--yarn-factor` runtime override.
- `59e512e` — byte-preserving `--context-surgery` compiler and provenance.
- `aca10ef` — preregistration ledger.
- `bb6e18c` — evaluator discovers the advertised model id.
- `a0dcd31` — evaluator reserves its output budget for task answers.

Local `main` remains exactly at `origin/main`. The branch was not pushed
because the configured GitHub credential is invalid; no remote state changed.

## Verification

Passed:

- `pytest -q tests/test_kv_quality.py`: 2 passed.
- `pytest -q tests/test_cli_files.py`: 27 passed.
- `make test-quantize`.
- `make test-help-interface`: CLI/README parity, 64 options.
- Targeted runtime YaRN tests: 3 passed.
- Both full-file compiler runs: 0 tensors converted, 687 kept, independent
  payload verification passed.

The broad `conda run -n ccbuild make test` ran for about nine minutes and then
stopped before completion because Make has no rule to produce the pre-existing
`test-swa.gguf` prerequisite for `test-metal-fuse`. This is an infrastructure
block, not a failed context-surgery assertion, and a full-suite pass is not
claimed.

## Reproduction

```bash
git switch context-surgery
conda run -n ccbuild make -j

BASE=/home/lab/workspace/models/gpt-oss-120b-MXFP4/gpt-oss-120b-MXFP4.gguf
OUT=/home/lab/workspace/models/gpt-oss-120b-context-surgery

./runner -m "$BASE" --context-surgery "$OUT/gpt-oss-120b-MXFP4-262k-yarn64.gguf" \
  -c 262144 --yarn-factor 64
./runner -m "$BASE" --context-surgery "$OUT/gpt-oss-120b-MXFP4-524k-yarn128.gguf" \
  -c 524288 --yarn-factor 128

conda run -n ccbuild python3 scripts/kv-quality.py \
  --binary ./runner \
  --models "$OUT/gpt-oss-120b-MXFP4-262k-yarn64.gguf" \
  --suites retrieval --lengths 4096 --depths 10,50,90 --pads 0 \
  --kv f16 --gpu off --out "$OUT/candidate-262k-retrieval-4k.json" -v
```

## Recommended next work

1. Run parent and x64 candidate on the 128 GB M5 Max with exact 32K, 64K,
   128K, 160K, 192K, 224K, and 250K cells from the frozen ledger. Keep the
   parent/control and candidate on the same Runner build and KV type.
2. Stop x64 on the existing kill criteria. Do not test x128 unless x64 clears
   the intermediate gate.
3. If inference-only scaling fails, compare at least two positional methods in
   a 25–100M-token pilot with experts and router frozen. Select on retrieval,
   short-context KLD, and coding/tool regressions, not on training loss alone.
4. Only a survivor runs full RULER, multi-needle ordering, paired coding/tool
   acceptance, and a pinned llama.cpp cross-runtime check.

No continued-training method is selected by this report. The goal is increased
usable context; YaRN metadata scaling was the cheapest falsifiable candidate,
not a commitment to the method.
