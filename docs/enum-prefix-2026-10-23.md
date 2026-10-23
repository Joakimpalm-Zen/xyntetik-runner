# Enum members that share a token prefix, 2026-10-23

A grid-world agent on the Windows CUDA host almost never chose the move `"E"`
when the same JSON-schema enum also held `"EAT"`. This note records why. The
constrained decoder is faithful: it samples the model's own probability of
each legal string. The skew comes from the model and the tokenizer. Qwen3
spells `"EAT"` as the token `E` and then `AT`, so the two values share their
first token and the model chooses between them one step later, after it has
already written `"E`. What it predicts at that point decides the answer.

## The report

Runner 1.5.1, `Qwen_Qwen3-4B-Q4_K_M.gguf`, `--serve --parallel 12 -c 1024
--gpu auto` (CUDA). Requests were `POST /v1/chat/completions` with
`temperature` 0.7, `max_tokens` 16, a seed per request, a system prompt ending
`Reply with JSON only. /no_think`, a user turn with a 5x5 grid, and
`response_format` set to this schema:

```json
{"type":"object","properties":{"action":{"type":"string",
 "enum":["N","S","E","W","EAT","REST"]}},"required":["action"],
 "additionalProperties":false}
```

Over about 66,000 requests: EAT 74 to 76%, N 12 to 16%, S 6%, W 3%, REST 0.2
to 2.2%, E 0.2 to 0.4%. In an A/B with 24 seeds per cell and the same prompt
per cell, the host changed only the enum word, from EAT to FEED:

| food at | enum with EAT | enum with FEED |
|---|---|---|
| N | N 24 | N 24 |
| E | EAT 22, E 2 | E 24 |
| S | S 24 | S 24 |
| W | W 24 | W 24 |

Only `"E"` loses, and only when `"EAT"` is a member.

## The branch point

Qwen3's vocabulary splits `{"action": "EAT"}` into `{"`, `action`, `":`,
` "`, `E`, `AT`, `"}`, and `{"action": "E"}` into the same tokens without
`AT` (`--score` on the Qwen2.5 vocabulary it shares, and `E` then `AT` at
indices 8 and 9 of every Qwen3 record below).
After the opening quote, the legal first tokens under the EAT enum are `E`,
`N`, `S`, `W`, `REST` and partial spellings such as `R` and `EA`. Both `"E"`
and `"EAT"` begin with `E`, so the token `E` carries both. One step later the
legal set is `AT` against the closing quote in its spellings (`"`, `"\n`,
`"\n\n`, ...). Choosing between the two values happens there.

At every step the sampler drops the illegal candidates and renormalizes over
the legal ones (`tests/test_sampler.c` holds that a constrained pick equals an
unconstrained pick from a vocabulary whose illegal entries were never likely).
Renormalizing divides every legal continuation by the same number, so the
ratio between `AT` and the close is the model's. That ratio is also the ratio
of the two whole strings: `"EAT"` and `"E"` share every conditional except
that branch step, and the close after `"EAT"` is close to certain. Scoring the
options whole, as the sum of in-context conditionals that `/v1/decide`
computes, therefore gives the same split.

## The host's record at the branch

One request from the food-at-E cell of the EAT enum: the same binary with
`--gpu off`, seed 2000, temperature 0.7 and `choice_logprobs: true`. It chose
`{"action":"EAT"}`.

| index | candidate | posterior | raw logprob |
|---|---|---|---|
| 8 (first value token) | `E` | 0.999999 | 0 |
| 8 | `N` (next best) | | -14.8 |
| 9 (after `E`) | `AT` | 0.8409 | -10.02 |
| 9 | `"\n` | 0.1544 | -11.72 |
| 9 | `"` | 0.0046 | -15.22 |
| 9 | `"\n\n` | | -21.29 |

At index 9, `coverage` read 1.0 and `n_legal` 4. The legal continuations held
about 1e-4 of the model's probability. The model's own next token was outside
the enum, almost certainly `AST` on the way to `EAST`. The sampler
renormalized over what was left, and `AT` happened to be about five times
likelier than the close. In the FEED arm the close is the only legal
continuation after `E`, so `"E"` wins there by the same accident. On this
record the model meant EAST. Neither enum let it say so, and which member it
got depended on how the other members tokenize.

## Reproduction on the Metal laptop

The same file (`bartowski/Qwen_Qwen3-4B-GGUF`, Q4_K_M, 2,497,280,960 bytes),
Runner built from main at 89a9bb64 (1.5.1 plus unreleased changes, none in
sampling or the schema validator), Metal, `--serve -c 1024`. The field prompt
was not available, so the prompt is a grid task written to resemble it. The
agent is at the centre with the food one cell east, under the hint `Food one
step away: N 0, S 0, E 2, W 0.`, the rule `EAT eats food on your cell` and
the instruction `Choose one action: N, S, E, W, EAT or REST.` (FEED in the
other arm). Each arm sends 200 requests (100 for the food-at-N controls) with
seeds from 1000 up. The first sampled token of each turn is `<think>` at
probability 0.9999, the only draw a small seed spoils (see below), so the
counts are unaffected. The field-sampling arms use the request's temperature 0.7
and the Qwen3 preset's top_p 0.95 and top_k 20. The pure arms use temperature
1 with top_p 1, top_k 0 and min_p 0. The predicted column comes from the raw
logprobs below, with the sampler's own temperature and top-p steps applied to
the legal set.

| arm | enum | food | sampling | observed | predicted |
|---|---|---|---|---|---|
| 1 | N,S,E,W,EAT,REST | E | field | EAT 186, E 14 | E 13.6 |
| 2 | N,S,E,W,FEED,REST | E | field | E 158, FEED 42 | E 156.0, FEED 44.0 |
| 3 | EAT,E,N,S,W,REST | E | field | EAT 186, E 14 | arm 1, seed for seed |
| 4 | N,S,E,W,EAT,REST | E | pure | EAT 169, E 31 | E 28.1 |
| 5 | N,S,E,W,FEED,REST | E | pure | E 144, FEED 56 | E 141.4, FEED 58.2 |
| 6 | N,S,E,W,EAT,REST | N | field | EAT 100 | EAT 100.0 |
| 7 | N,S,E,W,FEED,REST | N | field | N 92, FEED 8 | N 90.8, FEED 9.2 |

Raw logprobs from `choice_logprobs` (`choice_logprobs_probe` 64, temperature
0). Index 8 is the first value token and index 9 the step after it:

| arm | index | candidate | raw logprob | posterior |
|---|---|---|---|---|
| EAT, food E | 8 | `E` | 0.000 | 1.0000 |
| EAT, food E | 8 | `N` | -13.503 | 0.0000 |
| EAT, food E | 9 | `AT` | -0.153 | 0.8595 |
| EAT, food E | 9 | `"\n` | -1.984 | 0.1377 |
| EAT, food E | 9 | `"` | -5.882 | 0.0028 |
| FEED, food E | 8 | `E` | -0.347 | 0.7069 |
| FEED, food E | 8 | `FE` | -1.234 | 0.2911 |
| FEED, food E | 8 | `N` | -6.228 | 0.0020 |
| FEED, food E | 9 | `"\n` | -0.000 | 1.0000 |
| EAT, food N | 8 | `E` | -0.067 | 0.9352 |
| EAT, food N | 8 | `N` | -2.739 | 0.0646 |
| EAT, food N | 9 | `AT` | 0.000 | 1.0000 |
| FEED, food N | 8 | `N` | -0.184 | 0.8320 |
| FEED, food N | 8 | `FE` | -1.783 | 0.1680 |

What the laptop establishes:

- **The branch point is where the tokenizer puts it.** Index 9 is `AT`
  against the close, and nothing is decided at index 8.
- **The sampler is faithful.** Every observed count lies within 0.6 standard
  deviations of the count predicted from the model's own conditionals (arm 1:
  14 against 13.6, sd 3.6; arm 2: 158 against 156.0, sd 5.9; arm 4: 31
  against 28.1, sd 4.9; arm 5: 144 against 141.4, sd 6.4). Teacher-forced
  scoring of the whole strings in the arm-1 context gives `"E"` 0.139 and
  `"EAT"` 0.861 over the six members, where the pure sampler's prediction is
  0.141 and 0.859; in the arm-2 context it gives `"E"` 0.712, `"FEED"` 0.286
  and `"N"` 0.002, against 0.72 and 0.28 observed in arm 5. Sampling with no
  constraint at all, continuing the same prefix through `"E` at temperature
  1, wrote `AT` 171 times and the close 29 times in 200 (predicted 171.9 and
  28.1). The same chat turn with no `response_format` answered EAT 85 times,
  E 9 times and something unparsable 6 times in 100. That continuation probe
  uses large, mixed seeds: with a small request seed the first sampled token
  was always the most likely one, a separate defect fixed on this branch.
- **The order of the enum does not matter.** Arm 3 reproduces arm 1 seed for
  seed, because the token mask depends only on which strings are legal.
- **This prompt does not isolate the prefix the way the host's A/B does.**
  With the food north, the EAT arm still answers EAT 100 times in 100 while
  the FEED arm answers FEED 8 times: on this prompt the model wants the word
  EAT whatever the direction. At index 9 the legal set held 0.999 of the raw
  mass, so here the model itself continues `"E` with `AT`. The host's record
  is the other case, where the model wanted neither member. Both cases come
  from the model and the tokenizer, and the sampler reproduces both.

## What to do

- Spell enum members the way the model would say them, and keep any member
  from being a token prefix of another: `NORTH`, `SOUTH`, `EAST`, `WEST`,
  `FEED`, `REST`. The model then chooses between words it would write,
  at their first token.
- When the options must be compared whole, `/v1/decide` scores each one
  verbatim and returns their distribution. It gives the same ratio as the
  sampler for a prefix pair, so it does not fix a bad spelling; it does report
  the probabilities instead of a single sample.
- Read `legal_mass` in `choice_logprobs` (added on this branch): the
  probability the legal candidates held before renormalizing. On the host's
  record it is about 1e-4 at the branch; on the laptop it is 0.999. A small
  `legal_mass` says the schema made the choice, not the model.

## Method and files

The counting harness posts the chat requests above and parses
`choices[0].message.content` as JSON. Whole-value scores come from
`/v1/completions` with `echo: true`, `logprobs: 1` and `max_tokens: 0` on the
model's chat template rendered by hand, followed by the generated prefix
`<think>\n\n</think>{\n\n"action": "` and each option with its close. That is
the same sum of in-context conditionals `/v1/decide` computes; `/v1/decide`
itself tokenizes its state as plain text, so it cannot carry the chat
template's control tokens. No code changed for the finding. `legal_mass` is
the change that came out of it.
