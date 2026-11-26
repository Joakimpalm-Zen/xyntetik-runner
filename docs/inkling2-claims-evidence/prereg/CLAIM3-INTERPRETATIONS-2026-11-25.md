# R4.48.12 Claim 3: how the metrics' definitions were applied to the records (2026-11-25, decided AFTER the five Runner and mainline arms existed)

This is disclosed as written after the outputs existed: Runner's three arms and mainline's two, 40 records each, sha256 in the lab's run log. It changes no threshold, arm, prompt, setting, interval method or claim wording in PREREG 3.x. It records how PREREG 3.3's definitions map onto the fields the records actually carry. Each item is the plain reading of the PREREG text and was not chosen for its effect on any figure.

1. **`finish_reason`** is the response's `choices[0].finish_reason` (Runner's record field `finish` is the client's own copy).
2. **R** is the generated ids strictly between each `<|content_thinking|>` (200008) and the next `<|end_message|>` (200010), concatenated in order. A span that never closes runs to the turn's end. This is the earlier addendum's span rule (`ADDENDUM-2026-11-24-claims23-clients.md`, item 2).
3. **`unclosed_thinking_exhaustion`** holds when all three are true:
   - the id count is 2048;
   - `finish_reason` is `length`;
   - the last `<|content_thinking|>` has no later `<|end_message|>`.
4. **The loop guard's fields.** PREREG 3.3 names `loop_guard.closes`, but the records carry `loop_guard.interventions` and `ended_turn`. So "closes > 0" is read as `interventions > 0`, and a loop-ended turn as `finish_reason == "loop"` or `ended_turn`.
5. **A plain answer.** It is a `<|content_text|>` ... `<|end_message|>` segment outside thinking, decoded from the ids with the GGUF vocabulary, that holds `def <entry_point>(` either in a python code block or as plain text. "Stubbed `pass` only" means the definition's body is only `pass` or `...`: by `ast` when it parses (docstrings dropped), else by its indented lines.
6. **The tool path.** The engine's parsed `choices[0].message.tool_calls` holds a `submit_patch` call whose arguments parse to a JSON object with:
   - a non-empty string `fixed_code` containing `def <entry_point>(`;
   - a non-empty string `root_cause`.

   The primary count takes only `finish_reason` in {`tool_calls`, `stop`}. A `length` turn with such a call counts in the `with_truncation_recovery` companion row, Runner arms only, as PREREG 3.3 states.
7. **Token counts, as the earlier addendum fixed them before any output:**
   - `reasoning_tokens = len(R)`;
   - `completion_tokens` is the generated id count, with `usage.completion_tokens` reported beside wherever the two differ;
   - `output_tokens = completion_tokens - reasoning_tokens`.
8. **`unit_test_pass` (the secondary diagnostic) is NOT computed for this report.** It would execute model-written code against humanevalpack's test snippets. The measurement host is shared and borrowed, and it offers no isolation for running untrusted code. This is a disclosed deviation from PREREG 3.3's secondary item, which has no influence on `k_usable` or on any claim. It is to be computed later in an isolated environment.
9. **The wording's placeholders.**
   - In the tie wording, F ("the loop guard fired on F/40") is the number of runner-lg-rb turns with `interventions > 0`.
   - The worse wording was not selected, but its F would count runner-lg-rb's unusable turns that ended `length` or `loop`, or had an intervention, or had a forced reasoning-budget close.
