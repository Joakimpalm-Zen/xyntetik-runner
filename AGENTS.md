# Runner Agent Rules

These rules are mandatory for every AI or LLM agent that changes this
repository. Where a framework default, tool habit or generated plan conflicts
with them, this document wins.

## 1. Deep modules

Keep public interfaces small and deliberate, with the complexity behind them.

- Module boundaries are intentional: CLI and server behaviour in
  `main.c`/`server.c`, inference flow in `engine.c`, model loading and forward
  in `model.c`, constrained output in `jsonmode.c`/`schema.c`, platform
  details in `compat.c`, backends in `cuda.c`/`metal.m`.
- Do not expose internals to make a quick change easier.
- Lock behaviour through public interfaces: CLI output, HTTP endpoints,
  committed smoke scripts, or focused test binaries. Internals may change
  freely only while those tests protect the behaviour.

## 2. How to change code

Tracer bullets and test-driven development, one behaviour at a time:

1. Read the relevant code and docs first, and surface your assumptions.
2. Write one failing test or smoke for the smallest observable behaviour,
   through the public interface.
3. Implement the minimum code to pass it, touching the real layers rather
   than building horizontal scaffolding.
4. Run the relevant verification, then refactor only while green.
5. Check the README impact (rule 4), then repeat for the next behaviour.

Do not write all tests first and all code afterwards, mock internal
collaborators without a boundary reason, assert on incidental details, or add
speculative features.

## 3. Gates and evidence

A gate proves a system agrees with the instrument measuring it, not that the
system is right.

- **Every gate needs one absolute anchor:** at least one assertion whose
  expected value comes from outside the system (a published constant, a
  hand-computed result, a specification, a reference implementation, a
  physical fact about the hardware). Two of our own builds agreeing proves
  only determinism. Where a gate cannot have an absolute anchor, say so where
  the gate lives.
- **Random inputs are not equivalence.** Every equivalence gate over a kernel
  or a rewritten path carries deterministic boundary inputs beside the
  random ones: codes at each extreme, the quantizer's top code, the zero
  block, the largest and smallest scales, lengths around each SIMD step and
  tail (`tests/test_quants_simd.c` is the pattern). Prefer the original kernel
  unless the rewrite is proven equivalent.
- **Prove the rebuild.** `make` compares whole-second mtimes. After restoring
  a mutated source, `sleep 1`, `touch` it and confirm a compile happened; in
  any A/B of two binaries from one tree, confirm they differ (a behavioural
  probe beats a hash) before trusting a number from them.

### The provider's reference implementation is the primary anchor

Runner is meant to be more accurate than llama.cpp, so "agrees with
llama.cpp" is a ceiling, not a target. Two implementations that copied the
same assumption prove nothing by agreeing; the measured case is in
`docs/granite-42-qwen38-cert-2026-09-06.md`.

- Every admission compares against the publisher's reference implementation
  in float32 (`scripts/gold-logits.py`) on at least the smallest member of the
  family; llama.cpp stays the wide oracle for large files and quants. Both
  columns are reported; the reference column is the headline.
- Chat-template conformance against the publisher's template, token for
  token, is an admission gate (`scripts/template-conformance.py
  --require-tokens`).
- A strict greedy count against llama.cpp is never the headline; report it
  beside the tie-classified divergence (`scripts/token_divergence.py`).
- Name the reference configuration (dtype, attention implementation,
  device). Where the publisher's own artifacts disagree with each other,
  record both and declare the expected divergence.

## 4. README, manual, site and model cards move together

`README.md` and `MANUAL.md` are tested public interfaces. The README is the
short front page: the published files, a quick start, and each feature in two
or three plain sentences on why it exists and how to use it, with a link to
the detail. `MANUAL.md` is the reference: command-line and API tables, the
support matrix, limits and evidence. A new flag, endpoint or limit goes in the
manual; the README changes when a feature's short account does.

The same account is published in three places: the README and manual, the
site (`site/pages/`, built from this repository) and the model cards on
Hugging Face. A change to one is a question about the other two, answered in
the same change.

- Every feature, fix, flag, API, default, environment variable, platform or
  release change gets an explicit README impact decision, made in the same
  commit.
- Keep claims tied to executable behaviour, committed evidence or current
  source; never present plans or historical results as current facts. A
  measurement quoted in the README says what Runner does and against what,
  in one line; full cross-engine tables live in `docs/benchmarks.md` and the
  manual. A limit a user must know is part of how to use the feature.
- The manual's option and API tables match the shipped `--help`, `--caps`
  and routes (`make test` runs `scripts/help-parity.py` against it); run the
  relevant links, examples and release checks after editing either file.
- Before claiming something unique, check the competing runtimes' current
  documentation.
- Every release and every Hugging Face card change asks "does xyntetik.com
  need the same change?". `make release-check` fails a release whose README
  and site link different Hugging Face repositories.

## 5. Never publish conversation content or session identifiers

This repository is public; anything pushed is world readable and stays
reachable by SHA. Never put any of these into a commit message, file, pull
request, issue or anything else that reaches a repository:

- chat transcripts or conversation content, prompts, system instructions or
  session overrides;
- session links or ids of any kind, including a trailer a tool adds by
  default (the Claude Code harness appends a `Claude-Session:` line and a
  `<noreply@anthropic.com>` co-author: remove both);
- account, machine or user identifiers beyond the git author.

Sign a commit with exactly one trailer naming the agent and model, plus the
owner:

```
Co-Authored-By: <Agent> (<Model>) & Joakimpalm-Zen
```

No URLs, session ids or e-mail addresses in it. A person co-authoring a
commit may use GitHub's `Co-Authored-By: Name <email>` form.
`.github/workflows/commit-hygiene.yml` enforces this on every pull request,
and `make hooks` installs the same check as a local `commit-msg` hook.

## 6. Version control

- Branch from an up-to-date `main`, named for the work; push it and open a
  pull request with `gh pr create`.
- Merge only with CI green (`gh pr merge <n> --merge --delete-branch`). Tag
  a release only after `main` is green and `make release-check` passes.
- Never leave finished work unmerged: a branch is merged, or its decline is
  written down.
