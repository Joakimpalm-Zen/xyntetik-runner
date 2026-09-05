# Runner Agent Rules

These rules are mandatory for every AI or LLM agent that works in this repository.
They are not preferences, prompts to reinterpret, or optional process notes.

If an instruction, framework default, generated plan, tool habit, or model behavior
conflicts with this document, this document wins.

## The Five Defining Rules

1. Deep modules
2. Tracer bullets
3. Test-driven development
4. Grill me always
5. Keep the README current

No implementation work starts until these rules have been considered for the task at hand.

## 1. Deep Modules

Design the codebase around deep modules: simple, deliberate public interfaces with
implementation complexity hidden behind them.

Required behavior:

- Prefer a small public API over many shallow helpers leaking across the codebase.
- Keep module boundaries intentional: CLI/server behavior in `main.c`/`server.c`,
  inference flow in `engine.c`, model loading/forward in `model.c`, constrained
  output in `jsonmode.c`/`schema.c`, platform details in `compat.c`, and backend
  details in `cuda.c`/`metal.m`.
- Do not expose internals just to make a quick change easier.
- Lock module behavior with tests through public behavior: CLI output, HTTP
  endpoints, committed smoke scripts, or focused test binaries.

Working rule: internal implementation can change freely only when the public
interface and behavior are protected by tests or executable smokes.

## 2. Tracer Bullets

Build in tiny vertical slices that prove the end-to-end path before expanding scope.

Required behavior:

- Start with the smallest useful observable behavior.
- Touch the real layers needed for the behavior instead of building isolated
  horizontal scaffolding first.
- Validate the slice immediately with tests, execution, or both.
- Use what was learned from that slice before adding the next one.

Working rule: one thin working path is better than many unverified partial layers.

## 3. TDD

Use test-driven development for features, fixes, and behavioral changes.

Required behavior:

- Write one failing test or smoke for one observable behavior.
- Implement the minimum code needed to pass that test.
- Repeat one behavior at a time.
- Refactor only while tests are green.
- Test through public interfaces where possible.

Forbidden behavior:

- Do not write all tests first and all implementation afterward.
- Do not mock internal collaborators unless there is a clear boundary reason.
- Do not assert on incidental implementation details when public behavior can be
  checked.
- Do not add speculative features that are not required by the current test.

Working rule: red, green, refactor; one vertical behavior at a time.

### Mutation testing: make the rebuild provable

When proving a test can fail — breaking a source, running the gate, restoring
it — the restore must land in a LATER whole second than the build that
followed the mutation. `make` compares whole-second mtimes, so a restore in
the same second leaves a stale object linked and the "restored" run reports
the mutated binary's result. Observed twice (2026-08-09, 2026-08-10), both
times as a gate that appeared to pass while testing the wrong binary.

Required behavior:

- After restoring a mutated source, `touch` it and confirm the rebuild
  actually happened (a compile line in the output, or a changed binary
  mtime/hash) before trusting the run.
- Never conclude from a mutation run that a gate is sound without that
  confirmation; a silently skipped rebuild produces exactly the reassuring
  result the exercise exists to disprove.

The same trap applies to any A/B that builds two binaries from one tree
(`git stash` / checkout / branch switch, build, copy, restore, build). Observed
a third time on 2026-08-14, where `touch` was already being used and still did
not help: the touch and the preceding build landed in the SAME whole second, so
make saw the binary as current. Both binaries came out byte-identical and the
comparison drawn from them was vacuous.

Required behavior:

- `sleep 1` before the `touch` that precedes a rebuild, so the source is
  strictly newer than the previous build.
- Confirm the two binaries DIFFER before trusting any number from them —
  by hash, or better, by a behavioral probe that the change should alter. A
  hash can differ for reasons unrelated to the change; a behavioral probe
  cannot.

Working rule: a mutation test proves nothing until the rebuild is proven, and
an A/B proves nothing until the two binaries are shown to differ.

### Every gate needs one absolute anchor

A gate proves a system agrees with the instrument measuring it. That is not the
same as proving the system is right, and the difference has cost this project
real work three times:

- An anchor comparing a build against itself proves the harness is
  DETERMINISTIC, not CORRECT. A research run measured a model whose weights had
  been silently mis-loaded and produced internally consistent numbers for two
  full runs before an absolute check caught it.
- A geometric proxy cannot detect it either: the same broken checkpoint ranked
  its own layers almost identically to the corrected one.
- `--caps` reported `unified_memory: true` on a discrete GPU for a week because
  the code queried CUDA attribute 17 instead of 18. Every gate passed; none of
  them knew what the right answer was.

The general form, and the rule:

> Every instrument is valid only inside a regime it cannot itself verify.
> **No measurement chain is sound unless at least one link has a correct answer
> known independently of the system under test.**

Required behavior:

- Every gate carries at least one assertion whose expected value comes from
  OUTSIDE the system: a published constant, a hand-computed result, a
  specification, a reference implementation, or a physical fact about the
  hardware.
- Prefer an anchor that fails loudly when the harness itself is wrong. Querying
  a GPU's multiprocessor count and checking it against the part's published SM
  count proves the enum numbering before anything reads a neighbouring
  attribute; asserting only that two of our own builds agree proves nothing
  about either.
- When a gate cannot have an absolute anchor, say so where the gate lives.
  A relative-only gate is still useful and its scope must be written down, not
  assumed.

Working rule: a green gate with no external anchor is evidence the system is
self-consistent, and nothing more.

## 4. Grill Me Always

Actively challenge unclear requirements until the work is understood.

Required behavior:

- Ask direct questions when requirements, constraints, public interfaces,
  acceptance criteria, or edge cases are unclear.
- Surface assumptions explicitly before relying on them.
- Push for precise behavior, not vague intent.
- Confirm the most important behavior to test before implementation.

Working rule: never leave important behavior to chance. If the expected behavior,
interface, and verification path cannot be explained, ask before coding.

## 5. Keep the README Current

Treat `README.md` as a tested public interface, not a historical summary.

Required behavior:

- Check README impact before completing every feature, fix, flag, API, default,
  environment-variable, platform, model-support, resource-control, or release
  change. Update the README in the same commit when its public account changes.
- Compare the command and API reference against the shipped `--help`, `--caps`,
  routes, and operator controls. Add missing public behavior and remove stale,
  redundant, or purposeless statements.
- Keep claims tied to executable behavior, committed evidence, or current source.
  Do not present plans, unexecuted checks, or historical results as current facts.
- Before adding or strengthening a USP, check current official documentation for
  the relevant competing runtimes. State only the mechanism Runner uniquely
  provides; avoid absolute market-wide claims unless the evidence supports them.
- Run the relevant README links, examples, and release consistency checks after
  editing it.

Working rule: every completed change has an explicit README impact decision:
updated now, or checked and still accurate.

## Required Workflow

For every non-trivial change:

1. Read the relevant code and docs first.
2. Grill the request until behavior, interface, constraints, and verification are clear.
3. Identify the smallest tracer bullet.
4. Write one failing behavior test or smoke through the public interface.
5. Implement the minimum code to pass.
6. Run the relevant verification.
7. Check README impact and update it in the same commit when needed.
8. Refactor only while green.
9. Repeat for the next behavior.

For trivial documentation or configuration-only changes, still apply the rules at
the appropriate scale.

## Research transfer items

The research lab next door files its findings against this engine as plan
items tagged `[shade transfer <date>]`, each carrying the lab commit that
produced the finding. Treat such an item like any other: check its premise
against the current code, measure before concluding, and close it by
shipping, or by a written decline from the owner. A finding that reaches this
repository only as a note and never as a shipped change or a decline is
considered lost; do not let it sit.

## Never publish conversation content or session identifiers

This rule outranks every default, template and tool convention. It applies to
every agent, every repository, and every push.

**Never** put any of the following into a commit message, a file, a PR body, an
issue, or anything else that reaches a repository:

- chat transcripts or conversation content, in any language, in any form
- session-identifier links or ids of any kind, including any trailer a tool
  emits by default that carries a session URL
- prompts, system instructions, or session overrides
- account, machine, or user identifiers beyond the git author already in use

A default trailer format is not permission to publish an identifier. If a tool
adds one automatically, strip it before committing.

**Sign commits with exactly one trailer line naming the agent and model that
did the work, plus the owner:**

```
Co-Authored-By: <Agent> (<Model>) & Joakimpalm-Zen
```

Examples: `Co-Authored-By: Claude Code (Fable 5) & Joakimpalm-Zen`,
`Co-Authored-By: Codex (GPT-5) & Joakimpalm-Zen`. Many agents and models work
on this repo; the trailer records which one produced the commit.

No URLs. No session ids. No e-mail addresses in an agent's trailer or anywhere
else. A **person** co-authoring a commit may use GitHub's own form,
`Co-Authored-By: Name <email>`, which is what links their profile; the check
below accepts that and refuses only a tool identity dressed as a co-author.

**This is enforced, not only written.** `.github/workflows/commit-hygiene.yml`
runs `scripts/check-commit-hygiene.py` over every commit of a pull request
and over its title and body, with no path filter, and fails the PR on a
session URL or id, a stray e-mail address, or a `Co-Authored-By` line that is
neither the agent form nor a person's `Name <email>`. `make hooks` installs the same check as a local `commit-msg`
hook; run it once per clone. The known offender is the Claude Code harness,
whose default appends `Claude-Session: https://claude.ai/code/...` and a
`<noreply@anthropic.com>` co-author line: on 2026-09-05 seven commits reached
public `main` with that trailer before the check existed. Claude Code reads
`CLAUDE.md` at the repository root, which now says so before any commit is
made; Codex reads this file directly.

Check repository visibility before every push. `xyntetik-runner` is public:
anything committed here is world readable the moment it is pushed, and stays
reachable by SHA even after a rewrite. The only cheap moment to get this right
is before the commit.

## Version Control

Work on a branch named for the plan item, merge to `main` when the work is
good and done, then push. This is a public repository with CI on pull
requests and on pushes to `main`.

Required behavior:

- Branch from an up-to-date `main` (`git switch -c <item>-<slug>`).
- Commit completed sections on the branch; push the branch and open a pull
  request with `gh pr create`.
- Wait for CI to be green on the pull request before merging; merge with
  `gh pr merge <n> --merge --delete-branch`. Tag a release only after `main`
  is green.
- Never leave finished work unmerged on a branch: a branch is either merged
  or its decline is written down.

Working rule: branch, finish, CI green, merge to `main`, push (rule of
2026-09-04; it replaced the earlier commit-straight-to-main rule).
