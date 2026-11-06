# The project's Clef IQ3_S backbone: its tokenizer split and its row's claim (R4.36, 2026-11-06)

**What is not flattering, first:**
- The row `cloudflare-clef-iq3_s-recovered` claimed "a decision model served at /v1/systemone" from its admission on
  2026-10-27. Its file has never had a decision head; the decision model is the `cloudflare-clef-iq3_s-native` row's
  file, added to the repository on 2026-10-28.
- Since then the backbone has tokenized unlike its publisher around combining marks, and its two recovery adapters
  were served on a split they were not trained on there.
- The fix is a rule keyed on header fields (architecture qwen35, general.name "Clef Backbone View"); a requant that
  renames the file falls outside it.

## The files

- `Joakimpalm-Zen/Cloudflare-clef-IQ3_S-recovered-GGUF`: the row pins `Clef-backbone-IQ3_S.gguf` at 9e9ce879 (sha256
  `dafa8752...`, 12,419,329,216 bytes). Its header: architecture qwen35, general.name "Clef Backbone View", 851
  tensors, no clef.* keys and no decision tensors.
- The repository's head (427a06ce) also carries `Clef-IQ3_S.clef.gguf` (sha256 `1e142e60...`), pinned by
  `cloudflare-clef-iq3_s-native`: architecture clef, the same 851 tensors byte for byte (each tensor's data hashed in
  both files, streamed from the Hub, 12,408,334,336 bytes of tensor data), plus the 160 tensors of the decision head.
- `Cloudflare/clef` at 0b331204: tokenizer.json `06b95093...`, the qwen2 split (the same blob transformers 5.x writes
  on re-saving a Qwen 3.5 tokenizer; R4.27.5).

## The checks, each declared before its read

- **K-a:** difftok on the backbone's real header against Cloudflare/clef's tokenizer.json: before, 3 of 721 corpus
  strings and 5,503 of 247,091 vocabulary entries differed; with the entry, 0 and 0, and the note names the rule.
- **K-b:** every roster file's header (75 rows) through the harness before and after: 74 byte-identical; the backbone
  alone different, on exactly the three combining-mark strings of the corpus (Devanagari and Thai).
- **K-c:** the house readings' corpus (`harness-prose-heldout.txt`, Wikitext-2 test, no combining marks among its
  20,000 characters) tokenizes byte-identically before and after on the backbone, as one text (4,883 tokens) and line
  by line. The readings (base 0.0879 / 94.0%, QR-001e 0.0736 / 95.6%, QR-001b 0.0757 / 95.6%) were taken by Runner on
  both sides (the IQ3_S backbone against its BF16 twin, both in the qwen35 view), so their tokens, and the numbers,
  carry over.
- **The adapters' training split:** every round was trained on token ids of Cloudflare/clef's own tokenizer (the
  qwen2 split), in torch for QR-001b and in Runner's trainer from the same ids for the later rounds; the backbone now
  serves that split.
- **The unit test:** `test_qwen2_trained_rules` reads the backbone's keys as qwen2 and the same name on another
  architecture as declared; a mutant of the name fails it.
