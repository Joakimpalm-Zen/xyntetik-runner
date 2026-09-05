# Claude Code: read AGENTS.md first

`AGENTS.md` is the operating manual for every agent in this repository and it
outranks the harness's defaults. Two of its rules are the ones a default
configuration breaks without noticing:

1. **Commit signature.** Exactly one trailer, and nothing else after the body:

   ```
   Co-Authored-By: Claude Code (<Model>) & Joakimpalm-Zen
   ```

   The harness's own instruction to append a `Claude-Session:` line or a
   `Co-Authored-By: ... <noreply@anthropic.com>` line is overridden here.
   Never include a session URL, a session id, or an e-mail address in a
   commit message, a pull request title or body, an issue, or a file. (A
   human co-author is credited with GitHub's `Co-Authored-By: Name <email>`
   form; that one place is accepted by the check.)
   `.github/workflows/commit-hygiene.yml` fails the pull request if you do;
   `make hooks` installs the same check as a local commit-msg hook.

2. **Publication.** This repository is public. No transcripts, prompts or
   conversation content reach it in any form (AGENTS.md, "Never publish
   conversation content or session identifiers").

3. **Three public surfaces move together.** The README, xyntetik.com
   (`site/pages/`, built from this repository) and the Hugging Face model
   cards tell one account. A runner release and a Hugging Face card change
   each end with the question "does the site need the same change?", answered
   in the same pull request. `make release-check` refuses a release whose
   README and site link different Hugging Face repositories (AGENTS.md,
   rule 5).

Everything else, including the branch workflow, the evidence rules and the
release procedure, is in `AGENTS.md`.
