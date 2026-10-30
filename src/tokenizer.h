// SPM / BPE tokenizer built from GGUF vocabulary metadata.
#ifndef RUNNER_TOKENIZER_H
#define RUNNER_TOKENIZER_H

#include "gguf.h"

// The stop set's parts, so every array that holds it is sized from them and
// none can fill: eos, the file's declared eog ids, the turn ends Runner knows
// by spelling, and the longest per-architecture list (Harmony's two).
// tok_stop_ids asserts the two lists' lengths against these.
enum { TOK_EOG_MAX = 32, TOK_STOP_SPELLINGS = 12, TOK_STOP_ARCH_MAX = 2,
       TOK_STOP_MAX = 1 + TOK_EOG_MAX + TOK_STOP_SPELLINGS + TOK_STOP_ARCH_MAX };

enum { TOK_SPM, TOK_BPE, TOK_BPE_SPM }; // BPE_SPM: gemma4 (spaces to U+2581, raw-UTF-8 BPE)

// BPE pre-tokenizer split rules, from tokenizer.ggml.pre. GPT2 is the original
// regex (a leading space may join a run); the newer families let any single
// non-letter/non-digit character lead a letter run and cap digit runs, which
// changes where pre-token boundaries fall. An unrecognised value keeps GPT2
// and is reported (pre_unknown below): the split may not be the publisher's.
enum { TOK_PRE_GPT2, TOK_PRE_LLAMA3, TOK_PRE_QWEN2, TOK_PRE_SMOLLM, TOK_PRE_AFMOE,
       TOK_PRE_TEKKEN, TOK_PRE_LLAMA4, TOK_PRE_QWEN35,
       // openbmb MiniCPM5: the Llama 3 rule applied AFTER a split that
       // isolates every digit group, so a whitespace run that ends at a
       // digit stays one pre-token (see pre_split_next's digit_ends)
       TOK_PRE_MINICPM5,
       // XHToken Spark 2.5 (spark2_5): afmoe's main regex with digits split
       // one by one, a narrower CJK set, and newlines taken one at a time
       TOK_PRE_SPARK25,
       // IFM K2-Horizon (k2-horizon): the Llama 3 regex with combining marks
       // and ZWNJ/ZWJ inside letter runs; NFC
       TOK_PRE_K2H };

typedef struct { const char *key; uint32_t klen; int32_t val; } hmap_ent;
typedef struct { hmap_ent *e; size_t cap; } hmap;

typedef struct {
    int      model;         // TOK_SPM | TOK_BPE
    int      pre;           // TOK_PRE_* split rules (BPE only)
    int      n_vocab;
    gg_str  *tokens;        // borrowed from gguf kv
    float   *scores;        // SPM (may be NULL)
    int32_t *ttype;         // token type per id (may be NULL)
    int      bos_id, eos_id, unk_id;
    // tokenizer.ggml.eog_token_ids: the ids the file itself declares as
    // ending generation (a file that declares more than TOK_EOG_MAX is
    // refused). tok_stop_ids makes each a stop.
    int      eog_ids[TOK_EOG_MAX];
    int      n_eog;
    bool     add_bos, add_space_prefix;
    // tokenizer.ggml.add_eos_token: the checkpoint's own tokenizer appends
    // the end token (embedding models such as Qwen3-Embedding pool on it).
    // Honoured where a whole input is encoded as-is: /v1/embeddings.
    bool     add_eos;
    // the publisher's tokenizer composes its input to NFC before splitting
    bool     nfc;
    // A BPE file's tokenizer.ggml.pre value that names no split rule here, so
    // GPT2 is assumed; "" when the name is known or absent, or when the
    // vocabulary is not plain BPE (no other model reads the split). Characters outside
    // [A-Za-z0-9._-] are stored as '?', so it can go into a log line or a JSON
    // report as it is (--doctor reports it).
    char     pre_unknown[48];
    hmap     vocab;         // token string -> id
    hmap     merges;        // "left right" -> rank (BPE)
    char    *merges_buf;    // owned storage for merge keys
    // Special (control/user-defined) tokens, grouped by first byte: group b is
    // special_ids[sb_off[b] .. sb_off[b+1]), and within a group they stay in
    // length-descending order so the longest match still wins. tok_encode
    // probes at every byte offset, so the grouping is what keeps that scan off
    // the whole list (gemma-3 ships 6,414 of these).
    int     *special_ids;
    int      n_special;
    int      sb_off[257];
    int      b2u[256];      // BPE byte -> codepoint
    int      u2b[512];      // BPE codepoint -> byte (-1 = none)
    // Read-only after tokenizer_init: parallel slots share one tokenizer and
    // encode concurrently. Per-call state lives on the calling thread.
} tokenizer;

bool tokenizer_init(tokenizer *t, gguf_file *g);
void tokenizer_free(tokenizer *t);
// returns number of tokens written to out (capacity cap); add_bos per call
int  tok_encode(tokenizer *t, const char *text, int32_t *out, int cap,
                bool add_bos, bool parse_special);

// Prompt marks. A rendered chat prompt is template scaffolding around caller
// text, and the two are not the same kind of bytes: `<|im_end|>` typed into a
// message is text, `<|im_end|>` the template wrote is a control token. The
// renderer brackets every byte it owns in PROMPT_RAW_OPEN/CLOSE, two values
// valid UTF-8 never contains (json.c refuses anything above 0xF4), and
// tok_encode_prompt recognizes a special token only inside those brackets;
// everything else, whatever it spells, is text. The marks are never
// tokenized and tok_strip_marks removes them where a prompt is shown or
// compared as text.
#define PROMPT_RAW_OPEN  ((char)0xFD)
#define PROMPT_RAW_CLOSE ((char)0xFC)
int    tok_encode_prompt(tokenizer *t, const char *text, int32_t *out, int cap,
                         bool add_bos);
size_t tok_strip_marks(char *s);

// The encoders above write at most `cap` tokens and say nothing about the
// rest, and no byte count bounds a token count: a byte-fallback vocabulary
// spells the test fixture's space as three tokens, 81 for a 39-byte text.
// This one sizes the buffer to the text, growing until the encoding fits
// with `spare` slots after it. TOK_TEXT is caller text (no control tokens),
// TOK_RAW a caller-composed prompt (specials recognized anywhere), TOK_PROMPT
// a rendered prompt (control tokens inside the marks only). Returns the
// token count with the buffer in *out, the caller's to free, or -1 on
// allocation failure (*out NULL).
typedef enum { TOK_TEXT, TOK_RAW, TOK_PROMPT } tok_mode;
int    tok_encode_fit(tokenizer *t, const char *text, bool add_bos, tok_mode mode,
                      int spare, int32_t **out);
// raw-byte encode without BOS/specials/segment normalization (see tokenizer.c);
// -1 when a temporary allocation failed, never a silently shorter encoding
int  tok_encode_raw(tokenizer *t, const char *text, int n,
                    int32_t *out, int cap);
// decode one token into buf (returns bytes written, no NUL); control tokens -> 0
int  tok_decode(tokenizer *t, int id, char *buf, int cap);
// decoded text of token id even if control (for stop-token matching); NULL if oob
const char *tok_raw(tokenizer *t, int id);
bool tok_is_control(tokenizer *t, int id);
// control or user-defined: a token the tokenizer emits only as itself
bool tok_is_special(tokenizer *t, int id);
// The ids that end a generated turn for this vocabulary on architecture
// `arch`: eos, the file's eog_token_ids, then the turn terminators Runner
// knows by spelling. Writes at most cap ids to out and returns the count.
int tok_stop_ids(tokenizer *t, const char *arch, int *out, int cap);
int  tok_find(tokenizer *t, const char *s); // exact vocab lookup, -1 if absent
// Test hook for the merge gate: force the linear rescan (0) or the candidate
// queue (1) regardless of segment length; -1 returns to the length rule. The
// two must produce identical ids on every input -- see
// tests/test_tokenizer_merge.c.
void tok_merge_force(int on);

#endif // RUNNER_TOKENIZER_H
