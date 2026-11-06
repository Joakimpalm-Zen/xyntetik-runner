# The qwen2-trained split under a qwen35 label: a header-keyed rule (R4.27.5, 2026-11-06)

**What is not flattering, first:**
- The rule is a list. It knows four sources by what their converters wrote into the header: Clef, MiMo-V2.6-Distill,
  openbmb's MiniCPM-V-4.7-35B-A3B and empero-ai's Qwen3.8-35B-A3B-Distill. A finetune that ships the same split
  under a qwen35 label, and is in none of these four, still reads with the qwen35 split.
- The two new keys are header fields the converter took from the publishers' own metadata. One is a checkpoint
  name, "35a3_hf_v47_0923". The other is the generic "Ours", held to the empero-ai tag and the base model. A
  requant that rewrites them falls outside the rule.
- Which split each finetune's training saw is inferred from what transformers 5.x does (below), not measured on
  the publishers' pipelines.
- One Clef row, cloudflare-clef-iq3_s-recovered, declares general.architecture qwen35 and reads with the qwen35
  split. That is unchanged here, and it is named as a finding, not fixed.

## Why a rule, and why keyed on the header

- **transformers 5.x reads the regex differently.** It loads a Qwen3.5/3.6 tokenizer.json as Qwen2Tokenizer with
  the qwen2 regex: `\p{L}+`, so a combining mark splits off its letter. Qwen's own file says `[\p{L}\p{M}]+`.
  - It does this in memory: transformers 5.16.1 on Qwen3.6-35B-A3B's own files gives the qwen2 ids on 3 of 3
    differing corpus strings and 5,503 of 5,503 differing vocabulary entries, and the `[\p{L}\p{M}]+` ids on none.
  - It does this on save: save_pretrained writes the qwen2 pre-tokenizer. 5.14.1 and 5.16.1 agree.
  - So a finetune trained under transformers 5.x saw the qwen2 split, and the tokenizer.json it ships says so.
    empero-ai's config was saved by 5.17.0 and openbmb's by 5.18.0.
- **No published label can carry the difference.** llama.cpp's converter labels a pre-tokenizer by sha256 over
  `AutoTokenizer.encode(chktxt)` (conversion/base.py, get_vocab_base_pre). chktxt holds Khmer vowel signs, so the
  two splits hash apart, but both hashes are registered as "qwen35":
  - `d30d75d9...`, from Qwen's own tokenizer.json (ref Qwen/Qwen3.5-9B-Instruct);
  - `1444df51...`, from the `\p{L}+` tokenizer.json (ref openbmb/MiniCPM-V-4_6). empero-ai's file hashes to it,
    and so do Qwen3.6-35B-A3B's own files under transformers 5.16.1.

  Every Qwen3.5/3.6-family model converted under transformers 5.x is labelled qwen35 with the qwen2 split.
- **llama.cpp at load:** QWEN35 is the `[\p{L}\p{M}]+` regex and QWEN2 exactly `\p{L}+`, with no per-file rule. A
  user can force the label with `--override-kv tokenizer.ggml.pre=str:qwen2`. The honest label for the split is
  "qwen2", and Runner reads a file labelled so with that split, needing no rule.
- **The precedent:** Runner already read two qwen35-labelled sources as qwen2.
  - Clef, by its architecture name.
  - XiaomiMiMo's Qwen3.5 distill, by its chat template's `<tool_call><function=` opener.

  Each was measured the same way: 3 of 721 under qwen35, 0 under qwen2. R4.27.5 moves both into one rule list and
  adds two entries.

## The entries

| Source | Header keys (all must match; the label must be qwen35) |
|---|---|
| Clef | general.architecture `clef` |
| XiaomiMiMo's Qwen3.5 distill | the chat template's `<tool_call><function=` opener |
| openbmb MiniCPM-V-4.7-35B-A3B (every file at cf943fbe, thinking ones too) | general.architecture `qwen35moe`, general.name `35a3_hf_v47_0923` |
| empero-ai Qwen3.8-35B-A3B-Distill (every file at b1f9d1dc) | general.architecture `qwen35moe`, general.name `Ours`, `empero-ai` in general.tags, general.base_model.0.repo_url `https://huggingface.co/Qwen/Qwen3.6-35B-A3B` |

A whole-file sha256 key was weighed. It would mean hashing 15 to 69 GB at load, or a 42-entry table that misses
requants.

A firing rule prints, for example:

```
tokenizer: openbmb's MiniCPM-V-4.7-35B-A3B (general.name 35a3_hf_v47_0923) was trained with the qwen2 split (\p{L}+, marks split from letters); the declared qwen35 is read as qwen2
```

## The checks, each declared before its read

- **K-a, the targets:** difftok against each publisher's tokenizer.json, on the files' real headers.
  - empero-ai's Q4_K_M against its JSON at bcc2dbe2: 0 of 721 corpus strings and 0 of 247,091 vocabulary
    entries. Before the change it was 3 and 5,503.
  - openbmb's Q4_K_M against its JSON at 98e79ba9: 0 and 0. Before the change it was 3 and 5,503.
  - The note printed on both.
- **K-b, every other row:** every file of every row with files (74 rows, 78 files), read as a header by HTTP range
  at its pinned revision.
  - The difftok harness built before the change (main 3e761116) and the new one gave byte-identical ids on the
    721 corpus strings and the 3 tier-smoke prompts, on all 74 rows. 73 were read on main 3e761116's roster; the
    Mellum row came with main fb7bbeb8 and was read after that merge.
  - Per row: kb-token-ids.json.
  - The note is new on the four Clef rows whose file declares architecture clef, and on MiMo's row. Nowhere else.
  - The three LoRA adapter rows have no vocabulary, and both harnesses refuse them alike.
- **K-c, the pins of the qwen35-family rows:** the 17 linux-x86_64-cuda pins of the qwen35 family, compared by tier-smoke
  without --pin or --record, each file held from the Hub at its pinned revision with its sha256 equal to the Hub's.
  Per run: kc-pins.json.
  - The 9 CPU pins, on the change before the 1.14.0 merge (249e291b on main 3e761116), the GPU hidden: all 9 match,
    output sha256 equal to the pin, the fact found. The base's CPU pin is among them.
  - The 8 CUDA pins, on the change merged with main 43f3a090 (v1.14.0), a CUDA MIG slice, 1g.24gb, sm_120: all 8
    match. The base's CUDA pin is among them. The release changed no source beyond the version string, so the CPU
    pins were not rerun.
  - ornith-1.5-35b-a3b-q8_0's CUDA pin matched on a rerun. Its first run ended in a harness-reported error (exit 1,
    no result) after the server had logged a complete reply (37 prompt and 48 generated tokens). The load average
    was 65.89 92.02 118.09 at its end, from other work on the host; the 1-minute load had reached 1,324 at
    03:49Z. The kept stderr ends with the shutdown's unload line. That fits the server not exiting within
    10 s of SIGTERM while unloading a 37.8 GB file, a stop-under-load pattern seen before in the test suite. It also
    fits a client-side error on reading the reply. The message's head was not kept, so neither is shown. The rerun,
    alone at load 19.67 28.01 58.18, exited 0 with the pin's sha256.
  - The 2 rows pinned only on Windows or Metal, the Windows pins of 2 more, and the 5 unpinned rows were not compared
    here; K-b covers their tokenizer.
- **The unit test:** tests/test_tokenizer.c, `test_qwen2_trained_rules`, on the qwen35 fixture with header keys put
  in front of its own.
  - Each rule's keys read qwen2. Without the base, without the tag, with another base, or with the MiniCPM name on
    another architecture, the file stays qwen35.
  - Each case checks the Devanagari mark split.
  - A mutant that drops the tag condition fails it.
