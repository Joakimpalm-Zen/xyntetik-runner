# Model support scope

New model-support work prioritizes European and US open-weight families,
including the UK and Switzerland. This is a prioritization policy, not a
filter that removes already-admitted models.

Existing supported families remain maintained, including admitted Qwen
layouts and experimental `qwen4exp`. A family name or geographic origin is
not proof that every checkpoint, quantization, context layout, or backend
is supported. Check the [support matrix](../MANUAL.md#support-matrix),
`runner --info caps`, and `runner -m model.gguf --info doctor` for the actual case.

Current admitted families include Llama, Gemma, Phi, gpt-oss, dense and
hybrid Granite, Trinity/AFMoE, Nemotron, Mistral-derived layouts, SmolLM,
StableLM, Apertus, Muse Glimmer, and the admitted Qwen variants. Granite is
implemented; it is not a future admission. Backend and training limits are
properties of the architecture/layout and are listed in the manual.

Support outside the priority area is declined by default unless explicitly
admitted. Completing evidence or fixing an existing supported path is
maintenance. Existing API compatibility with OpenAI and Anthropic remains
independent of model-origin priorities.

Admission requires the same shape, tokenizer/template, publisher-reference,
backend, and failure-path evidence for every family. The policy chooses
where support effort goes, not how strictly a candidate is checked.
