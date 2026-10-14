#define _CRT_RAND_S
// runner — CLI: one-shot completion, interactive chat, and server launcher.
#include "decide.h"
#include "dpo.h"
#include "hfhub.h"
#include "runner.h"
#include "instances.h"
#include "tray.h"

#include "compat.h"
#include "json.h"
#include "envelope.h"
#include "lineage.h"
#include "oms.h"
#include "server.h"
#include "provenance.h"
#include "bundle.h"
#include "receipts.h"
#include "watermark.h"
#include "kvsnap.h"
#include "session.h"
#include "build_arch.h"
// for render_prompt_alloc: chat mode renders the same prompts the chat route
// does, so it uses the same measured-size renderer rather than a second one
#include "api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include "pmath.h"

// Bare-invocation tray launch needs to know whether a human is at the other
// end: a piped/redirected probe (CI, scripts, `runner | head`) must get usage
// text, never a GUI event loop.
#ifdef _WIN32
#include <io.h>
#define RUNNER_TTY(f) _isatty(_fileno(f))
#else
#include <unistd.h>
#define RUNNER_TTY(f) isatty(fileno(f))
#endif
#include <sys/stat.h>

// quantize_gguf is declared in runner.h

// ---------------------------------------------------------------- misc

// CLI flags share the one strict parser (compat.c) with the env overrides; the
// only difference is a hard exit on a bad flag versus a warn-and-default on a
// bad env var.
static long long int_arg(const char *opt, const char *s, long long min,
                         long long max) {
    long long v;
    if (!parse_i64(s, min, max, &v)) {
        fprintf(stderr, "error: %s expects an integer in [%lld, %lld]\n",
                opt, min, max);
        exit(1);
    }
    return v;
}

static uint64_t u64_arg(const char *opt, const char *s) {
    uint64_t v;
    if (!parse_u64(s, 0, UINT64_MAX, &v)) {
        fprintf(stderr, "error: %s expects an unsigned integer\n", opt);
        exit(1);
    }
    return v;
}

static double float_arg(const char *opt, const char *s, double min, double max) {
    double v;
    if (!parse_f64(s, min, max, &v)) {
        fprintf(stderr, "error: %s expects a finite number in [%g, %g]\n",
                opt, min, max);
        exit(1);
    }
    return v;
}

// Transcript fields cross an untrusted JSON boundary.  Keep their admission
// rules identical to the CLI's before narrowing a double to float/int: a
// matching public chain hash proves byte integrity, not that the author was
// Runner, and out-of-range floating-to-integer conversions are undefined C.
static bool record_num(jv *obj, const char *key, double min, double max,
                       double *out) {
    jv *v = jv_get(obj, key);
    if (!v || v->type != J_NUM || !isfinite(v->num) ||
        v->num < min || v->num > max)
        return false;
    *out = v->num;
    return true;
}

static bool record_int(jv *obj, const char *key, int min, int max, int *out) {
    double d;
    if (!record_num(obj, key, (double)min, (double)max, &d) ||
        d != trunc(d))
        return false;
    *out = (int)d;
    return true;
}

static bool record_token(jv *v, int n_vocab, int32_t *out) {
    if (!v || v->type != J_NUM || !isfinite(v->num) || v->num < 0 ||
        v->num >= n_vocab || v->num != trunc(v->num))
        return false;
    *out = (int32_t)v->num;
    return true;
}

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool record_hex(jv *v, const char **out, size_t *out_n) {
    if (!v || v->type != J_STR) return false;
    size_t n = strlen(v->str);
    if (n % 2 != 0) return false;
    for (size_t i = 0; i < n; i++)
        if (hex_nibble(v->str[i]) < 0) return false;
    *out = v->str;
    *out_n = n / 2;
    return true;
}

// Read a whole file into a NUL-terminated malloc'd buffer, or NULL on any
// error: open, seek, a negative or oversized length (ftell can return -1 and a
// hostile/special file can be enormous), allocation, or a read error. The size
// cap also keeps the later size arithmetic well clear of overflow. *out_len
// gets the byte count (excluding the terminator).
#define MAX_INPUT_FILE (256u * 1024u * 1024u)
// Seed bytes for --keygen: the OS generator, never the C library's.
// Build flavor: the portable bit-exact configuration (make T3=1) marks its
// receipts and its --version so a verifier can tell which binary wrote a
// record; the default build writes no flavor field at all.
#ifdef RUNNER_T3_BUILD
#define RUNNER_BUILD_FLAVOR "t3"
#define RUNNER_BUILD_FLAVOR_STR "t3"
#else
#define RUNNER_BUILD_FLAVOR NULL
#define RUNNER_BUILD_FLAVOR_STR ""
#endif

// Case-insensitive equality of two hex strings (a key pasted from another
// tool may be upper case; the runner writes lower case).
static bool hex_eq_nocase(const char *a, const char *b) {
    for (;; a++, b++) {
        int ca = (unsigned char)*a, cb = (unsigned char)*b;
        if (ca >= 'A' && ca <= 'F') ca += 'a' - 'A';
        if (cb >= 'A' && cb <= 'F') cb += 'a' - 'A';
        if (ca != cb) return false;
        if (!ca) return true;
    }
}

static bool os_random(uint8_t *out, size_t n) {
#ifdef _WIN32
    for (size_t i = 0; i < n; i += 4) {
        unsigned v;
        if (rand_s(&v) != 0) return false;
        for (size_t k = 0; k < 4 && i + k < n; k++)
            out[i + k] = (uint8_t)(v >> (8 * k));
    }
    return true;
#else
    FILE *f = fopen("/dev/urandom", "rb");
    if (!f) return false;
    size_t r = fread(out, 1, n, f);
    fclose(f);
    return r == n;
#endif
}

// The chain hash of a receipt on disk (its `chain.hash`), for linking the
// next receipt to it and for checking a link.
static bool receipt_chain_hash(const char *path, char out[65]);

static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0 || (unsigned long)sz > MAX_INPUT_FILE || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)sz, f);
    bool err = ferror(f) != 0;
    fclose(f);
    if (err) { free(buf); return NULL; }
    buf[got] = 0;
    if (out_len) *out_len = got;
    return buf;
}

// A JSON schema's identity in a record (R1.1.2): the sha256 of its compact
// JSON, key order kept -- what serve receipts record for a request's schema,
// so how the file is spelled (indentation, spacing) is not identity.
static bool schema_file_digest(const char *path, char hex[65]) {
    size_t n = 0;
    char *b = read_file(path, &n);
    if (!b) return false;
    jv *v = json_parse(b, n);
    free(b);
    if (!v) return false;
    sbuf d = {0};
    jv_dump(v, &d);
    jv_free(v);
    bool ok = !d.failed;
    if (ok) envelope_data_sha256(d.s, d.n, hex);
    free(d.s);
    return ok;
}

// R1.3: the CLI's image writer. The loop and the image are session.c's
// (shared with the server's /v1/runner/sessions); this adds the schema digest
// a --json-schema run carries and the line the CLI prints.
#define session_steps session_run
static bool session_image_out(const char *path, const engine *e,
                              const char *model_path, const float *logits,
                              int n_prompt, int max_new, bool json_mode,
                              bool ignore_eos, const char *schema_file) {
    char ssha[65] = "";
    if (schema_file && !schema_file_digest(schema_file, ssha)) {
        fprintf(stderr, "error: session: cannot read %s\n", schema_file);
        return false;
    }
    char sha[65];
    if (!session_image_write(path, e, model_path, NULL, NULL, logits, n_prompt,
                       max_new, json_mode, ignore_eos, ssha, sha))
        return false;
    fprintf(stderr, "session image -> %s (sha256 %s; %d tokens, %d of %d "
            "generated)\n", path, sha, e->pos, e->gen_count, max_new);
    return true;
}

// R1.8.2: score a token sequence against the key and print the verdict as
// one JSON object; exit 0 WATERMARKED, 2 NOT_DETECTED, 3 INSUFFICIENT.
static int wm_report(const wm_key *k, const int32_t *toks, int n, int start,
                     const char *source, const char *record_kid) {
    wm_score sc;
    wm_detect(k, toks, n, start, &sc);
    const char *v = wm_verdict(&sc);
    printf("{\"schema_version\":\"xyntetik.runner.watermark_detect.v1\","
           "\"key_id\":\"%s\",\"source\":\"%s\",", k->id, source);
    if (record_kid) {
        bool hex = strlen(record_kid) == 16;
        for (int i = 0; hex && i < 16; i++)
            hex = (record_kid[i] >= '0' && record_kid[i] <= '9') ||
                  (record_kid[i] >= 'a' && record_kid[i] <= 'f');
        printf("\"record_key_id\":\"%s\",", hex ? record_kid : "malformed");
    }
    printf("\"scheme\":\"%s\",\"layers\":%d,\"context\":%d,\"tokens\":%d,"
           "\"scored\":%d,\"g_ones\":%lld,\"g_n\":%lld,\"mean_g\":%.6f,"
           "\"z\":%.4f,\"p_value\":%.4g,\"verdict\":\"%s\"}\n",
           WM_SCHEME, WM_LAYERS, WM_CONTEXT, sc.tokens, sc.scored, sc.g_sum,
           sc.g_n, sc.mean, sc.z, sc.p_value, v);
    fprintf(stderr, "%s: z %.2f over %d scored of %d tokens (p %.3g)\n", v,
            sc.z, sc.scored, sc.tokens, sc.p_value);
    return !strcmp(v, "WATERMARKED") ? 0 : !strcmp(v, "NOT_DETECTED") ? 2 : 3;
}

// The token ids of a JSON array, each an int32 >= 0; NULL on anything else.
static int32_t *json_token_ids(jv *arr, int *n_out) {
    if (!arr || arr->type != J_ARR) return NULL;
    int32_t *t = malloc(sizeof *t * (size_t)(arr->n ? arr->n : 1));
    if (!t) return NULL;
    for (int i = 0; i < arr->n; i++) {
        double v = jv_num(arr->items[i], -1);
        if (!arr->items[i] || arr->items[i]->type != J_NUM || v < 0 ||
            v > INT32_MAX || v != (double)(int32_t)v) { free(t); return NULL; }
        t[i] = (int32_t)v;
    }
    *n_out = arr->n;
    return t;
}

// Agent transcripts, D4a (R1.1.2): may this record be replayed with the
// constraints this verifier was given? A replay runs the sampler; a grammar,
// a stop sequence or a scripted reply shaped the recorded tokens beyond it,
// and a replay without them could only disagree with the record -- which a
// DIVERGED verdict would blame on the model or the build. A CLI record made
// under --json, --json-schema or --ignore-eos replays when the verifier is
// given the same constraint (the schema matched by digest); anything a served
// turn was shaped by is refused (replaying it is D4b); and a constraint the
// verifier adds that the record does not name is refused too. False with the
// reason in `why`.
static bool replay_constraints_ok(jv *rec, bool json_mode,
                                  const char *schema_file, bool ignore_eos,
                                  char *why, size_t cap) {
    jv *cons = jv_get(rec, "constraints");
    jv *serve = jv_get(rec, "serve");
    if (cons && cons->type != J_ARR) {
        snprintf(why, cap, "malformed constraints");
        return false;
    }
    if (serve) {
        // D4b (R1.1.3): a served turn shaped by a JSON schema, JSON mode or
        // ignore_eos replays under the same constraint the CLI records do,
        // given the schema by --json-schema (matched by digest below): the
        // server compiles a response_format into the same engine constraint
        // the CLI flag does, over the prompt tokens the record carries. What
        // the replay cannot rebuild is refused by name: a tool envelope
        // (declarations, choice and the native or generic grammar a request
        // builds), stop sequences, a reasoning budget or the loop guard.
        // Records written before D4a name what shaped them only in shaped_by.
        jv *list = cons ? cons : jv_get(serve, "shaped_by");
        char names[256] = "";
        size_t k = 0;
        int n = 0, rebuilt = 0;
        for (int i = 0; list && list->type == J_ARR && i < list->n; i++) {
            jv *it = list->items[i];
            const char *kind = it && it->type == J_STR ? it->str
                             : jv_str(jv_get(it, "kind"), "?");
            if (cons && (!strcmp(kind, "json_schema") ||
                         !strcmp(kind, "json_mode") ||
                         !strcmp(kind, "ignore_eos")))
                rebuilt++;
            int w = snprintf(names + k, sizeof names - k, "%s%s", n ? ", " : "", kind);
            if (w > 0 && (size_t)w < sizeof names - k) k += (size_t)w;
            n++;
        }
        if (n > rebuilt) {
            snprintf(why, cap, "the served output was shaped beyond the sampler "
                     "by %s, which a replay does not rebuild (a served JSON "
                     "schema, JSON mode or ignore_eos does replay), so a replay "
                     "could only disagree with the record", names);
            return false;
        }
    }
    bool rj = false, re = false;
    const char *rs = NULL;
    for (int i = 0; cons && i < cons->n; i++) {
        const char *kind = jv_str(jv_get(cons->items[i], "kind"), NULL);
        if (!kind) {
            snprintf(why, cap, "malformed constraints");
            return false;
        }
        if (!strcmp(kind, "json_mode")) rj = true;
        else if (!strcmp(kind, "ignore_eos")) re = true;
        else if (!strcmp(kind, "json_schema")) {
            rs = jv_str(jv_get(cons->items[i], "sha256"), "");
            if (strlen(rs) != 64) {
                snprintf(why, cap, "malformed json_schema constraint");
                return false;
            }
        } else {
            snprintf(why, cap, "the output was shaped beyond the sampler by %s, "
                     "which this build does not replay", kind);
            return false;
        }
    }
    if (rj != json_mode) {
        snprintf(why, cap, rj ? "the output was generated under --json "
                 "(constraint json_mode); verify with --json to replay it"
                 : "--json given, but the record's output was not generated "
                 "under it (no json_mode constraint)");
        return false;
    }
    if (re != ignore_eos) {
        snprintf(why, cap, re ? "the output was generated under --ignore-eos "
                 "(constraint ignore_eos); verify with --ignore-eos to replay it"
                 : "--ignore-eos given, but the record's output was not "
                 "generated under it (no ignore_eos constraint)");
        return false;
    }
    if (rs && !schema_file) {
        snprintf(why, cap, "the output was generated under a JSON schema "
                 "(constraint json_schema, sha256 %s); verify with "
                 "--json-schema naming that schema to replay it", rs);
        return false;
    }
    if (!rs && schema_file) {
        snprintf(why, cap, "--json-schema given, but the record's output was "
                 "not generated under a schema (no json_schema constraint)");
        return false;
    }
    if (rs) {
        char have[65];
        if (!schema_file_digest(schema_file, have)) {
            snprintf(why, cap, "cannot read --json-schema %s as JSON", schema_file);
            return false;
        }
        if (!hex_eq_nocase(have, rs)) {
            snprintf(why, cap, "--json-schema %s is a different schema (sha256 "
                     "%s) from the record's json_schema constraint (sha256 %s)",
                     schema_file, have, rs);
            return false;
        }
    }
    return true;
}

static bool receipt_chain_hash(const char *path, char out[65]) {
    size_t n = 0;
    char *buf = read_file(path, &n);
    if (!buf) return false;
    jv *j = json_parse(buf, n);
    free(buf);
    if (!j) return false;
    const char *h = jv_str(jv_get(jv_get(j, "chain"), "hash"), NULL);
    bool ok = h && strlen(h) == 64;
    if (ok) memcpy(out, h, 65);
    jv_free(j);
    return ok;
}

// Write a machine-readable record beside an artifact. The document is built
// through sbuf/sb_esc by the caller, so paths remain valid JSON on both POSIX
// (where quotes are legal filename bytes) and Windows (where separators are
// backslashes). The sidecar name is length-derived instead of living behind a
// fixed PATH_MAX-shaped buffer: an artifact we successfully wrote must not get
// a silently truncated or missing provenance filename.
static bool write_provenance(const char *artifact, const char *suffix,
                             const char *kind, const sbuf *doc) {
    size_t an = strlen(artifact), sn = strlen(suffix);
    if (doc->failed || an > SIZE_MAX - sn - 1) {
        fprintf(stderr, "error: cannot build %s provenance record\n", kind);
        return false;
    }
    char *path = malloc(an + sn + 1);
    if (!path) {
        fprintf(stderr, "error: cannot allocate %s provenance path\n", kind);
        return false;
    }
    memcpy(path, artifact, an);
    memcpy(path + an, suffix, sn + 1);
    FILE *f = fopen(path, "wb");
    bool ok = f && fwrite(doc->s, 1, doc->n, f) == doc->n &&
              fflush(f) == 0 && !ferror(f);
    if (f && fclose(f) != 0) ok = false;
    if (ok) fprintf(stderr, "%s: provenance -> %s\n", kind, path);
    else    fprintf(stderr, "error: cannot write %s provenance to %s\n",
                    kind, path);
    free(path);
    return ok;
}

// Write a step's provenance record and, when --sign-key was given, sign it in
// place (R17.1): the lineage walker reads both.
static bool write_record(const char *artifact, const char *suffix,
                         const char *kind, const sbuf *doc, const char *sign_key) {
    if (!write_provenance(artifact, suffix, kind, doc)) return false;
    size_t an = strlen(artifact), sn = strlen(suffix);
    char *path = malloc(an + sn + 1);
    if (!path) return false;
    memcpy(path, artifact, an);
    memcpy(path + an, suffix, sn + 1);
    bool ok = lineage_sign(path, sign_key);
    if (!ok) fprintf(stderr, "error: cannot sign the %s record %s\n", kind, path);
    free(path);
    return ok;
}

typedef struct { int32_t *t; float *w; int n; } train_ex;

static void train_examples_free(train_ex *exs, int n_ex) {
    for (int i = 0; i < n_ex; i++) {
        free(exs[i].t);
        free(exs[i].w);
    }
    free(exs);
}

static bool train_line_blank(const char *s, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (s[i] != ' ' && s[i] != '\t' && s[i] != '\r') return false;
    return true;
}

static bool train_example_append(train_ex **exs, int *n_ex,
                                 int32_t *t, float *w, int n) {
    if (*n_ex == INT_MAX) return false;
    size_t count = (size_t)*n_ex + 1;
    if (count > SIZE_MAX / sizeof(**exs)) return false;
    train_ex *grown = realloc(*exs, sizeof(**exs) * count);
    if (!grown) return false;
    *exs = grown;
    grown[(*n_ex)++] = (train_ex){ t, w, n };
    return true;
}

// Parse the two documented training-data forms behind one ownership boundary.
// A JSONL example is tokenized in full before its length is compared with the
// requested training context: using the context as tok_encode's output cap
// would silently turn an over-long completion into a shorter training target.
// The token that ends an assistant turn in the model's own template, or
// the declared EOS when the template has no separate terminator (llama2,
// mistral, zephyr, granite, apertus, ornith, raw). -1 when even that is
// missing. Mirrors the strings the engine stops on (engine.c) so an adapter
// taught to end its turn ends it on the token serving stops at.
static int train_eot_id(tokenizer *tok, int tmpl) {
    const char *s = NULL;
    switch (tmpl) {
    case TMPL_CHATML: case TMPL_CHATML_THINK: case TMPL_HERMES4: case TMPL_QWEN38: case TMPL_QWEN3_CODER: s = "<|im_end|>"; break;
    case TMPL_LLAMA3: s = "<|eot_id|>"; break;
    case TMPL_GEMMA: s = "<end_of_turn>"; break;
    case TMPL_GEMMA4: case TMPL_GEMMA4_MAINLINE: s = "<turn|>"; break;
    case TMPL_MUSE: s = "<|eot|>"; break;
    case TMPL_PHI3: case TMPL_PHI4: s = "<|end|>"; break;
    case TMPL_HARMONY: s = "<|return|>"; break;
    default: break;
    }
    int id = s ? tok_find(tok, s) : -1;
    return id >= 0 ? id : tok->eos_id;
}

// One DPO pair: the same prompt with two completions, tokenized, each with a
// transition mask that is 1 over its own completion and 0 over the shared
// prompt. Two token arrays rather than one shared prefix, because
// model_lora_backward_w teacher-forces from position 0 and owns the KV rows.
typedef struct { int32_t *tw, *tl; float *mw, *ml; int nw, nl, np; } dpo_pair;

static void dpo_pairs_free(dpo_pair *ps, int n) {
    for (int i = 0; i < n; i++) {
        free(ps[i].tw); free(ps[i].tl); free(ps[i].mw); free(ps[i].ml);
    }
    free(ps);
}

// Tokenize prompt+completion into one buffer, growing until it fits. Same
// discipline as the JSONL branch of train_examples_load and for the same
// measured reason: tok_encode truncates silently at its capacity, and "one
// token per byte" is not a bound on a SentencePiece vocabulary with byte
// fallback.
static bool dpo_tokenize(tokenizer *tok, const char *prompt, const char *completion,
                         bool no_bos, int eot_id, bool eot,
                         int32_t **out, int *out_np, int *out_n) {
    size_t pn = strlen(prompt), cn = strlen(completion);
    if (pn > SIZE_MAX - cn - 8 || pn + cn + 8 > INT_MAX) return false;
    int cap = (int)(2 * (pn + cn) + 32);
    int32_t *t = NULL;
    int np = -1, nc = -1;
    for (int attempt = 0; attempt < 6; attempt++) {
        free(t);
        t = malloc(sizeof(*t) * (size_t)cap);
        if (!t) return false;
        np = tok_encode(tok, prompt, t, cap, !no_bos, true);
        nc = np >= 0 ? tok_encode(tok, completion, t + np, cap - np, false, false) : -1;
        if (np < 0 || nc < 0 || np + nc + 1 < cap) break;
        if (attempt == 5 || cap > INT_MAX / 2) break;
        cap *= 2;
    }
    if (!t) return false;
    if (np < 0 || nc < 0 || np + nc + 1 >= cap) { free(t); return false; }
    if (eot) t[np + nc++] = eot_id;
    *out = t; *out_np = np; *out_n = np + nc;
    return true;
}

static bool dpo_pairs_load(tokenizer *tok, const char *path, bool no_bos,
                           int wctx, int eot_id, bool eot_all,
                           dpo_pair **out, int *out_n) {
    size_t data_n = 0;
    char *data = read_file(path, &data_n);
    if (!data) {
        fprintf(stderr, "error: cannot read %s\n", path);
        return false;
    }
    dpo_pair *ps = NULL;
    int n = 0;
    size_t off = 0;
    int line_no = 0;
    while (off < data_n) {
        const char *line = data + off;
        const char *nl = memchr(line, '\n', data_n - off);
        size_t line_n = nl ? (size_t)(nl - line) : data_n - off;
        off += line_n + (nl != NULL);
        line_no++;
        if (train_line_blank(line, line_n)) continue;
        jv *v = json_parse(line, line_n);
        const char *prompt = v && v->type == J_OBJ ? jv_str(jv_get(v, "prompt"), NULL) : NULL;
        const char *chosen = v && v->type == J_OBJ ? jv_str(jv_get(v, "chosen"), NULL) : NULL;
        const char *rejected = v && v->type == J_OBJ ? jv_str(jv_get(v, "rejected"), NULL) : NULL;
        if (!prompt || !chosen || !rejected) {
            fprintf(stderr, "error: %s line %d needs a JSON object with string "
                    "{\"prompt\",\"chosen\",\"rejected\"}\n", path, line_no);
            jv_free(v); goto fail;
        }
        jv *ev = jv_get(v, "end_of_turn");
        bool eot = eot_all;
        if (ev) {
            if (ev->type != J_BOOL) {
                fprintf(stderr, "error: %s line %d end_of_turn must be true or false\n",
                        path, line_no);
                jv_free(v); goto fail;
            }
            eot = ev->b;
        }
        if (eot && eot_id < 0) {
            fprintf(stderr, "error: %s line %d asks for an end-of-turn token but this "
                    "model declares none\n", path, line_no);
            jv_free(v); goto fail;
        }
        dpo_pair q = {0};
        int npw = 0, npl = 0;
        if (!dpo_tokenize(tok, prompt, chosen, no_bos, eot_id, eot, &q.tw, &npw, &q.nw) ||
            !dpo_tokenize(tok, prompt, rejected, no_bos, eot_id, eot, &q.tl, &npl, &q.nl)) {
            fprintf(stderr, "error: %s line %d could not be tokenized without "
                    "truncation\n", path, line_no);
            free(q.tw); free(q.tl); jv_free(v); goto fail;
        }
        jv_free(v);
        // A pair whose two halves tokenize the shared prompt differently is
        // not a pair over the same input, and the margin would then measure
        // the prompt as well as the completion. Refuse it by name.
        if (npw != npl || memcmp(q.tw, q.tl, sizeof(int32_t) * (size_t)npw) != 0) {
            fprintf(stderr, "error: %s line %d: the two completions do not share an "
                    "identical prompt encoding (%d vs %d tokens)\n", path, line_no, npw, npl);
            free(q.tw); free(q.tl); goto fail;
        }
        q.np = npw;
        if (q.nw < 2 || q.nl < 2) {
            fprintf(stderr, "error: %s line %d: a completion is empty\n", path, line_no);
            free(q.tw); free(q.tl); goto fail;
        }
        if (q.nw > wctx || q.nl > wctx) {
            fprintf(stderr, "error: %s line %d needs %d/%d tokens, past the %d-token "
                    "training context; raise --train-ctx or shorten it\n",
                    path, line_no, q.nw, q.nl, wctx);
            free(q.tw); free(q.tl); goto fail;
        }
        // transition t scores toks[t+1], so the completion's first target is
        // transition np-1
        q.mw = malloc(sizeof(float) * (size_t)(q.nw - 1));
        q.ml = malloc(sizeof(float) * (size_t)(q.nl - 1));
        if (!q.mw || !q.ml) {
            free(q.tw); free(q.tl); free(q.mw); free(q.ml); goto fail;
        }
        for (int i = 0; i < q.nw - 1; i++) q.mw[i] = i >= q.np - 1 ? 1.0f : 0.0f;
        for (int i = 0; i < q.nl - 1; i++) q.ml[i] = i >= q.np - 1 ? 1.0f : 0.0f;
        if (n == INT_MAX) { free(q.tw); free(q.tl); free(q.mw); free(q.ml); goto fail; }
        dpo_pair *grown = realloc(ps, sizeof(*ps) * ((size_t)n + 1));
        if (!grown) { free(q.tw); free(q.tl); free(q.mw); free(q.ml); goto fail; }
        ps = grown;
        ps[n++] = q;
    }
    free(data);
    if (!n) {
        fprintf(stderr, "error: %s has no pairs\n", path);
        dpo_pairs_free(ps, n);
        return false;
    }
    *out = ps; *out_n = n;
    return true;
fail:
    free(data);
    dpo_pairs_free(ps, n);
    return false;
}

static bool train_examples_load(tokenizer *tok, const char *path, bool no_bos,
                                int wctx, int eot_id, bool eot_all,
                                train_ex **out, int *out_n) {
    size_t data_n = 0;
    char *data = read_file(path, &data_n);
    if (!data) {
        fprintf(stderr, "error: cannot read %s (regular files up to %u MiB "
                "are supported)\n", path, MAX_INPUT_FILE / (1024u * 1024u));
        return false;
    }
    train_ex *exs = NULL;
    int n_ex = 0;
    size_t path_n = strlen(path);
    bool jsonl = path_n >= 6 && strcmp(path + path_n - 6, ".jsonl") == 0;

    if (jsonl) {
        size_t off = 0;
        int line_no = 0;
        while (off < data_n) {
            const char *line = data + off;
            const char *nl = memchr(line, '\n', data_n - off);
            size_t line_n = nl ? (size_t)(nl - line) : data_n - off;
            off += line_n + (nl != NULL);
            line_no++;
            if (train_line_blank(line, line_n)) continue;

            jv *v = json_parse(line, line_n);
            const char *prompt = v && v->type == J_OBJ
                ? jv_str(jv_get(v, "prompt"), NULL) : NULL;
            const char *completion = v && v->type == J_OBJ
                ? jv_str(jv_get(v, "completion"), NULL) : NULL;
            if (!prompt || !completion) {
                fprintf(stderr, "error: %s line %d needs a JSON object with "
                        "string {\"prompt\",\"completion\"}\n", path, line_no);
                jv_free(v);
                goto fail;
            }
            // "end_of_turn": true appends the template's turn terminator to
            // the completion as one more target; --train-eot does it for
            // every line. Opt-in, since a completion that is a file or a
            // code fragment must not learn to end a turn after it.
            jv *ev = jv_get(v, "end_of_turn");
            bool eot = eot_all;
            if (ev) {
                if (ev->type != J_BOOL) {
                    fprintf(stderr, "error: %s line %d end_of_turn must be true or false\n",
                            path, line_no);
                    jv_free(v);
                    goto fail;
                }
                eot = ev->b;
            }
            if (eot && eot_id < 0) {
                fprintf(stderr, "error: %s line %d asks for an end-of-turn token but "
                        "this model declares none\n", path, line_no);
                jv_free(v);
                goto fail;
            }
            jv *wv = jv_get(v, "weight");
            double weight = 1.0;
            if (wv) {
                if (wv->type != J_NUM || !isfinite(wv->num) ||
                    wv->num < -(double)FLT_MAX || wv->num > (double)FLT_MAX) {
                    fprintf(stderr, "error: %s line %d weight must be a "
                            "finite number representable as f32\n", path,
                            line_no);
                    jv_free(v);
                    goto fail;
                }
                weight = wv->num;
            }

            size_t prompt_n = strlen(prompt), completion_n = strlen(completion);
            if (prompt_n > SIZE_MAX - completion_n - 8 ||
                prompt_n + completion_n + 8 > INT_MAX) {
                fprintf(stderr, "error: %s line %d is too large to tokenize\n",
                        path, line_no);
                jv_free(v);
                goto fail;
            }
            // tok_encode truncates silently at its capacity, and "one token
            // per byte" is not a bound: a SentencePiece vocabulary with byte
            // fallback and segment normalization emitted 68 tokens for 52
            // bytes on the test fixture, which the old fixed-size buffer cut
            // to 60 without a word. So the buffer grows until the encoding
            // fits with room to spare (the end-of-turn slot included), and a
            // line that still fills it is refused rather than truncated.
            size_t bytes = prompt_n + completion_n;
            int cap = (int)(2 * bytes + 32);
            int32_t *tokens = NULL;
            int np = -1, nc = -1;
            for (int attempt = 0; attempt < 6; attempt++) {
                free(tokens);
                tokens = malloc(sizeof(*tokens) * (size_t)cap);
                if (!tokens) break;
                np = tok_encode(tok, prompt, tokens, cap, !no_bos, true);
                nc = np >= 0 ? tok_encode(tok, completion, tokens + np, cap - np,
                                          false, false) : -1;
                if (np < 0 || nc < 0 || np + nc + 1 < cap) break;   // fits, or failed
                if (attempt == 5 || cap > INT_MAX / 2) break;      // cap stays the allocation
                cap *= 2;
            }
            if (!tokens) {
                fprintf(stderr, "error: out of memory loading %s line %d\n",
                        path, line_no);
                jv_free(v);
                goto fail;
            }
            if (np >= 0 && nc >= 0 && np + nc + 1 >= cap) {
                fprintf(stderr, "error: %s line %d tokenizes past %d tokens for %zu bytes; "
                        "refusing to truncate it\n", path, line_no, cap, bytes);
                free(tokens);
                jv_free(v);
                goto fail;
            }
            if (eot && np >= 0 && nc >= 0) tokens[np + nc++] = eot_id;
            int total = np >= 0 && nc >= 0 ? np + nc : -1;
            if (np < 1 || nc < 1 || total > wctx) {
                if (total > wctx)
                    fprintf(stderr, "error: %s line %d is %d tokens, above "
                            "--train-ctx %d; refusing to truncate it\n", path,
                            line_no, total, wctx);
                else
                    fprintf(stderr, "error: %s line %d does not tokenize into "
                            "a prompt and completion\n", path, line_no);
                free(tokens);
                jv_free(v);
                goto fail;
            }
            float *weights = malloc(sizeof(*weights) * (size_t)(total - 1));
            if (!weights) {
                fprintf(stderr, "error: out of memory loading %s line %d\n",
                        path, line_no);
                free(tokens);
                jv_free(v);
                goto fail;
            }
            // Transition t predicts token t+1: completion targets begin at
            // transition np-1. Prompt transitions are fully masked.
            for (int i = 0; i < total - 1; i++)
                weights[i] = i >= np - 1 ? (float)weight : 0.0f;
            if (!train_example_append(&exs, &n_ex, tokens, weights, total)) {
                fprintf(stderr, "error: out of memory loading training data\n");
                free(tokens);
                free(weights);
                jv_free(v);
                goto fail;
            }
            jv_free(v);
        }
    } else {
        if (memchr(data, '\0', data_n)) {
            fprintf(stderr, "error: training text %s contains a NUL byte\n",
                    path);
            goto fail;
        }
        // The same growth discipline as the JSONL branch, for the same
        // reason: tok_encode stops silently at its capacity, and a vocabulary
        // with byte fallback spells a space as three tokens, so "one token
        // per byte" cut a text of short words to half its length without a
        // word. Grow until the encoding fits with room to spare; refuse a
        // text that still fills the buffer rather than train on a prefix.
        if (data_n > (size_t)INT_MAX / 4) {
            fprintf(stderr, "error: training text %s is too large\n", path);
            goto fail;
        }
        int cap_n = (int)(2 * data_n + 32);
        int32_t *all = NULL;
        int ntok = -1;
        for (int attempt = 0; attempt < 6; attempt++) {
            free(all);
            all = malloc(sizeof(*all) * (size_t)cap_n);
            if (!all) break;
            ntok = tok_encode(tok, data, all, cap_n, !no_bos, true);
            if (ntok < 0 || ntok + 1 < cap_n) break;   // fits, or failed
            if (attempt == 5 || cap_n > INT_MAX / 2) break;
            cap_n *= 2;
        }
        if (!all) {
            fprintf(stderr, "error: out of memory loading training data\n");
            goto fail;
        }
        if (ntok >= 0 && ntok + 1 >= cap_n) {
            fprintf(stderr, "error: training text %s tokenizes past %d tokens "
                    "for %zu bytes; refusing to truncate it\n", path, cap_n,
                    data_n);
            free(all);
            goto fail;
        }
        if (ntok < 2) {
            fprintf(stderr, "error: training text tokenizes to %d tokens\n",
                    ntok);
            free(all);
            goto fail;
        }
        for (int st = 0; st + 2 <= ntok; st += wctx - 1) {
            int n = ntok - st < wctx ? ntok - st : wctx;
            if (n < 2) break;
            int32_t *tokens = malloc(sizeof(*tokens) * (size_t)n);
            if (!tokens) {
                fprintf(stderr, "error: out of memory loading training data\n");
                free(all);
                goto fail;
            }
            memcpy(tokens, all + st, sizeof(*tokens) * (size_t)n);
            if (!train_example_append(&exs, &n_ex, tokens, NULL, n)) {
                fprintf(stderr, "error: out of memory loading training data\n");
                free(tokens);
                free(all);
                goto fail;
            }
        }
        free(all);
    }
    free(data);
    if (n_ex < 1) {
        fprintf(stderr, "error: no training examples in %s\n", path);
        train_examples_free(exs, n_ex);
        return false;
    }
    *out = exs;
    *out_n = n_ex;
    return true;

fail:
    free(data);
    train_examples_free(exs, n_ex);
    return false;
}

// One-shot and interactive CLI state has a shorter lifetime than the process
// in embedding/tests. Keep its teardown explicit instead of relying on exit(2)
// to reclaim it; that also makes the documented sanitizer command a real leak
// gate. Server mode owns and frees the same objects inside server_run().
static void cli_cleanup(engine *e, int32_t *toks, tokenizer *tok, model_t *m) {
    if (e) {
        schema_free((snode *)e->schema);
        if (e->dm) { model_free(e->dm); free(e->dm); }
        free(e->hist);
    }
    free(toks);
    tokenizer_free(tok);
    model_free(m);
    prefix_cache_clear();
}

// --tool-info resolves, from the model's chat template alone, which native
// tool-call protocol family the runtime speaks for this model and whether that
// protocol is native (vs the generic marker/envelope fallback): the same
// answer runner_telemetry.tool_protocol gives per request, from the
// template layer's own selection (tool_protocol_name), never a static
// arch->family table that would drift from the renderer.
static const char *tool_family_for(int tmpl, bool *native) {
    return tool_protocol_name(tmpl, native);
}

// One line of chat input, newline stripped, in a buffer that grows to fit it.
// NULL at EOF with nothing read, or on allocation failure (*oom set then).
//
// A fixed buffer here is not a truncation, it is a SPLIT: fgets stops at the
// buffer and the remainder of the same paste arrives as the next line, so
// pasting a file into chat asked two unrelated questions, the second starting
// mid-word. Callers free the result.
static char *read_line(FILE *f, bool *oom) {
    size_t cap = 1024, n = 0;
    char *buf = malloc(cap);
    if (!buf) { *oom = true; return NULL; }
    for (;;) {
        if (!fgets(buf + n, (int)(cap - n), f)) {
            if (n == 0) { free(buf); return NULL; }
            return buf;                      // EOF after a newline-less tail
        }
        n += strlen(buf + n);
        if (n && buf[n - 1] == '\n') { buf[n - 1] = 0; return buf; }
        if (n + 1 < cap) return buf;         // short read: EOF, no newline
        // fgets filled the buffer, so the line continues past it. The cap
        // keeps cap - n inside the int fgets takes; a gigabyte-long single
        // chat line fails loudly rather than overflowing that conversion.
        if (cap > (size_t)INT_MAX / 2) { free(buf); *oom = true; return NULL; }
        char *bigger = realloc(buf, cap * 2);
        if (!bigger) { free(buf); *oom = true; return NULL; }
        buf = bigger;
        cap *= 2;
    }
}

static char *unescape(const char *s) {
    char *out = malloc(strlen(s) + 1);
    if (!out) { fprintf(stderr, "error: out of memory\n"); exit(1); }
    size_t m = 0;
    for (size_t i = 0; s[i]; i++) {
        if (s[i] == '\\' && s[i + 1]) {
            i++;
            switch (s[i]) {
                case 'n': out[m++] = '\n'; break;
                case 't': out[m++] = '\t'; break;
                case 'r': out[m++] = '\r'; break;
                case '\\': out[m++] = '\\'; break;
                default: out[m++] = '\\'; out[m++] = s[i]; break;
            }
        } else out[m++] = s[i];
    }
    out[m] = 0;
    return out;
}

// Requested help is an answer and goes to stdout; help printed BECAUSE
// something was wrong is a diagnostic and goes to stderr. `runner --help` is
// how a script or a package manifest discovers this interface, and a tool
// that answers that question on the error stream cannot be piped.
static void usage_to(FILE *f, const char *prog) {
    fprintf(f,
        "usage: %s -m model.gguf [options]\n\n"
        "options:\n"
        "  -m PATH        GGUF model file\n"
        "  -hf REPO[:TAG] fetch the GGUF from the Hugging Face Hub instead\n"
        "                 (owner/repo, TAG picks the quant when the repo has\n"
        "                 several; cached under ~/.cache/xyntetik-runner/hf,\n"
        "                 sha256-verified; HF_TOKEN for gated repos)\n"
        "  -p TEXT        prompt (one-shot completion; \\n etc. are unescaped)\n"
        "  -f FILE        read prompt from file (appended after -p text)\n"
        "  -i             interactive chat mode\n"
        "  --serve        HTTP server mode (OpenAI-compatible API)\n"
        "  --tray         BE the desktop tray/menu-bar controller (macOS,\n"
        "                 Windows) instead of running a model. Required when\n"
        "                 there is no terminal — launchd, Task Scheduler, a\n"
        "                 service wrapper — since the launches below all need\n"
        "                 one. Bare `runner` at a terminal, or a double-click,\n"
        "                 does the same thing without the flag\n"
        "  --no-tray      opt out of the tray entirely. The tray otherwise\n"
        "                 follows any session you sit with (--serve, -i) at a\n"
        "                 terminal and is LEFT RUNNING afterwards, so the next\n"
        "                 model can be loaded from it. One-shot -p runs and\n"
        "                 tooling modes never raise one\n"
        "  --port N       server port (default 8080)\n"
        "  --parallel N   parallel inference slots in server mode (default 1)\n"
        "                 -m \"name=path,name2=path2\" serves multiple models,\n"
        "                 loading the requested one on demand (swap mode)\n"
        "  --ttl N        swap mode: unload an idle model after N seconds\n"
        "                 (default 300, 0 = never)\n"
        "  --force-uncertified  load a model whose measured-envelope sidecar\n"
        "                 records an OUTSIDE-envelope verdict for this runtime\n"
        "                 (refused by default) — loads with a loud warning\n"
        "  --json         constrain output to a single valid JSON object\n"
        "  --json-schema F constrain output to the JSON Schema in file F\n"
        "  --quantize OUT rewrite the model to OUT.gguf (see --quant) and exit\n"
        "  --context-surgery OUT  compile -c/--yarn-factor into GGUF metadata;\n"
        "                 tensor payloads are independently verified unchanged\n"
        "  --quant T      quantize target: q8_0 | q4_0 | q3_k | q4_k | q6_k |\n"
        "                 f16 | bf16 | keep (default q4_0, or keep if\n"
        "                 --prune-experts or --merge-lora is the reason to\n"
        "                 rewrite)\n"
        "  --transcript F record the one-shot -p run as a replay-verifiable\n"
        "                 transcript (xyntetik.runner.transcript.v1): model and\n"
        "                 binary sha256, profile, config, seed, prompt and\n"
        "                 output token ids, chain hash. Same binary + same\n"
        "                 record replays bit-exact; cross-ISA replays\n"
        "                 token-exact (libm differs)\n"
        "  --verify F     replay a transcript (see --transcript) against the\n"
        "                 loaded -m model and byte-diff the result. Verdicts:\n"
        "                 VERIFIED (exit 0; tier T1 same-binary or T2\n"
        "                 token-replay), DIVERGED at token N (exit 2),\n"
        "                 UNVERIFIABLE (exit 3: bad chain hash, wrong model\n"
        "                 sha, wrong adapter, or output shaped by a constraint\n"
        "                 the replay would not reproduce; --json, --json-schema\n"
        "                 and --ignore-eos must match the record's). The\n"
        "                 record's config and seed override CLI sampling flags\n"
        "  --require-eval K  refuse to load -m unless it carries a passing\n"
        "                 evaluation of kind K (fidelity | agent) made for\n"
        "                 this file, signed by --trust-key when given\n"
        "                 (scripts/eval-record.py writes them)\n"
        "  --lineage F    walk F's provenance chain (a model or adapter file,\n"
        "                 or its record) back to its origins: every link\n"
        "                 re-hashed, every signature checked (--trust-key\n"
        "                 pins the signer); exit 0 verified and signed,\n"
        "                 1 consistent but not all signed, 2 broken or no record;\n"
        "                 on a --receipts folder: the model and adapter versions\n"
        "                 that answered, when and in what order, then each\n"
        "                 version's chain\n"
        "  --sign-record F  sign any JSON object file in place with --sign-key,\n"
        "                 the transcript's chain + signature (--record-prev P\n"
        "                 links it to record P); --check-record F verifies one\n"
        "  --export-bundle R  with --bundle-out DIR: put receipt R, its model\n"
        "                 signature (--model-sig) and key (--model-pubkey), the\n"
        "                 receipt key's fingerprint and a sha256 manifest in one\n"
        "                 directory a verifier can take; --sign-key signs the\n"
        "                 manifest. Needs no -m\n"
        "  --bundle-out DIR  where --export-bundle writes (created)\n"
        "  --check-bundle DIR  verify a bundle offline: files match the\n"
        "                 manifest, the receipt chain recomputes, signatures\n"
        "                 verify (--trust-key pins the manifest signer). Exit\n"
        "                 0 OK, 2 BAD, 3 UNVERIFIABLE. Does not replay: that is\n"
        "                 --verify with the model\n"
        "  --export-pack D  with --pack-out DIR: an evidence pack of every\n"
        "                 receipt-*.json in D (one model, one build), the model\n"
        "                 signature and key, --pack-envelope F (the envelope\n"
        "                 manifest the receipts name) and --pack-attach F files\n"
        "                 (oversight records, ...; repeatable), under a sha256\n"
        "                 manifest with one inference per receipt and the chain\n"
        "                 they form; --sign-key signs it. Needs no -m\n"
        "  --pack-out DIR  where --export-pack writes (created)\n"
        "  --pack-envelope F, --pack-attach F  see --export-pack\n"
        "  --check-pack DIR  verify a pack offline like --check-bundle, and\n"
        "                 that the receipts form the chain it states\n"
        "  --sessions DIR  with --serve: POST /v1/runner/sessions starts a\n"
        "                 raw-prompt generation that suspend_after N images\n"
        "                 into DIR; POST /v1/runner/sessions/{id}/resume\n"
        "                 continues it exactly (fork_seed S forks it); GET and\n"
        "                 DELETE /v1/runner/sessions/{id}\n"
        "  --kv-snapshots DIR  with --serve: POST /v1/runner/contexts/{id}/\n"
        "                 snapshot writes a named context to DIR (KV + manifest\n"
        "                 with its digests, model, KV type and producing receipt,\n"
        "                 signed with --sign-key); {\"id\",\"snapshot\"} on\n"
        "                 POST /v1/runner/contexts loads one back, refusing\n"
        "                 another model, KV type or changed bytes, and, with\n"
        "                 --trust-key or --sign-key, any manifest not signed\n"
        "                 by that key\n"
        "  --session-out F  with -p or --resume: write the generation's image\n"
        "                 to F (tokens, KV and recurrent state, sampler and rng,\n"
        "                 constraint, next-token logits; SHA-256 trailer; never\n"
        "                 overwritten). CPU, solo step loop, finite -n\n"
        "  --suspend-after N  stop after N generated tokens and write the\n"
        "                 image there (with --session-out)\n"
        "  --resume F     continue the generation imaged in F with the same -m:\n"
        "                 the continuation, and its image, are the ones the\n"
        "                 uninterrupted run makes, byte for byte\n"
        "  --fork-seed N  with --resume: continue under rng seed N instead of\n"
        "                 the image's own (a reproducible fork)\n"
        "  --watermark F  mark sampled output with the tournament watermark\n"
        "                 key F (-p runs and --serve; off by default; greedy is\n"
        "                 never changed); records carry its key id, and\n"
        "                 --verify replays a marked record only with F\n"
        "  --watermark-keygen F  write a new watermark key to F; needs no -m\n"
        "  --detect-watermark F  score F against --watermark's key: a\n"
        "                 transcript record (no -m) or a text (-m for its\n"
        "                 tokenizer). Exit 0 WATERMARKED, 2 NOT_DETECTED,\n"
        "                 3 INSUFFICIENT\n"
        "  --keygen F     write a receipt-signing key (xyntetik.runner.signkey\n"
        "                 .v1) to F and print its public key; needs no -m\n"
        "  --keygen-algo A  ed25519 (default; 32-byte key, 64-byte signature)\n"
        "                 or ml-dsa-44 (FIPS 204 post-quantum; 1312-byte key,\n"
        "                 2420-byte signature, deterministic)\n"
        "  --receipts DIR  with --serve: write a receipt (the --transcript\n"
        "                 record, replayable by --verify) for every finished\n"
        "                 generation into DIR, chained in write order across\n"
        "                 restarts and signed with --sign-key when given; each\n"
        "                 response's runner_telemetry.receipt names its file\n"
        "  --receipts-keep N  keep the newest N receipts (default 0: all)\n"
        "  --sign-key F   with --transcript or --receipts: sign the receipt\n"
        "                 with the key in F; with --quantize, --merge-lora,\n"
        "                 --train or --context-surgery: sign the step's\n"
        "                 provenance record\n"
        "                 (a signature over every byte before the ,\"signature\"\n"
        "                 key, chain hash included; the object names the algo)\n"
        "  --transcript-prev F  with --transcript: link the receipt to F (its\n"
        "                 chain hash becomes this record's chain.prev); with\n"
        "                 --verify: check that link (UNVERIFIABLE on a break)\n"
        "  --require-signed  with --verify: an unsigned record is UNVERIFIABLE\n"
        "  --trust-key HEX   with --verify: the record must be signed by this\n"
        "                 Ed25519 public key (a signature by any other key is\n"
        "                 UNVERIFIABLE)\n"
        "  --model-sig F  OpenSSF Model Signing bundle for -m (default: -m's\n"
        "                 path + .sig when it exists); verified against\n"
        "                 --model-pubkey at load, key method only (P-256/384/521)\n"
        "  --model-pubkey F  PEM public key the model signature must verify\n"
        "                 with; an explicit --model-sig that does not verify\n"
        "                 refuses the load\n"
        "  --require-signed-model  refuse to load -m unless its OMS signature\n"
        "                 verifies with --model-pubkey\n"
        "  --sign-model F  write an OMS bundle for model F (every part of a\n"
        "                 split GGUF) to F.sig, or to --model-sig; deterministic\n"
        "                 ECDSA (RFC 6979) with --model-key. Needs no -m\n"
        "  --model-key F  the PEM EC private key --sign-model signs with (SEC1\n"
        "                 or PKCS#8, unencrypted, P-256/384/521)\n"
        "  --merge-lora OUT  fold --lora into the base weights and write\n"
        "                 OUT.gguf + an OUT.gguf.merge.json provenance record:\n"
        "                 W' = W + (alpha/r)*B*A per adapted projection, each\n"
        "                 tensor requantized to its own type (or --quant T).\n"
        "                 The merged file runs in any GGUF runtime, but a\n"
        "                 quantized output type rounds the delta through its\n"
        "                 grid — measure the merged artifact before trusting\n"
        "                 it; base + --lora stays the exact form. Refused\n"
        "                 (destination untouched) when the output keeps under\n"
        "                 50%% of the delta; the kept share and changed\n"
        "                 bytes go in the record\n"
        "  --merge-allow-erased  with --merge-lora: write the merge even\n"
        "                 when the output grid rounded the delta away\n"
        "  --type-plan-strict  with --type-plan: fail before writing when\n"
        "                 any rule would be declined or fall back, instead of\n"
        "                 reporting it and writing the file\n"
        "  --type-plan PLAN.json  per-tensor precision while rewriting:\n"
        "                 {\"default\":\"keep\",\"rules\":[{\"match\":\"_exps.weight\",\n"
        "                 \"type\":\"q4_0\"}]}. First matching rule wins; types are\n"
        "                 keep|q8_0|q4_0|q3_k|q4_k|q6_k|f16|bf16. Per-tensor is\n"
        "                 the finest GGUF can\n"
        "                 express: experts are stored stacked, one tensor per\n"
        "                 layer, so per-EXPERT precision is not representable\n"
        "  --remove-sublayer attn:N[,mlp:M,...]  drop block N's attention\n"
        "                 (or block M's dense FFN) tensors while rewriting;\n"
        "                 the absence is declared by a 0 in the per-block\n"
        "                 attention.head_count / head_count_kv (or\n"
        "                 feed_forward_length) array, llama.cpp's own\n"
        "                 convention; the block's pre-norm stays. The runner\n"
        "                 omits the branch and reserves no KV rows for it\n"
        "                 (CPU path). A mechanism, not a quality claim:\n"
        "                 measure the artifact against its parent\n"
        "  --prune-experts LIST.json  drop MoE experts per layer while\n"
        "                 rewriting: {\"layer_N\":[kept expert ids...]};\n"
        "                 a layer absent from the file keeps every expert.\n"
        "                 Combine with --quant to also requantize survivors\n"
        "  -n N           max tokens to generate (default 256, -1 = until EOS)\n"
        "  -c N           context length (default: min(model max, 4096));\n"
        "                 beyond the training context, YaRN rope scaling is\n"
        "                 applied automatically\n"
        "  -b N           prompt batch size (default 64 with --gpu off;\n"
        "                 otherwise auto-sized from free RAM: 64/256/512)\n"
        "  -t N           threads (default: physical cores, capped at 32)\n"
        "  -s N           RNG seed (default: time)\n"
        "  --think        in -i chat, request the family's thinking prompt shape\n"
        "  --no-think     in -i chat, request its non-thinking shape; otherwise\n"
        "                 runner renders what the family's own reference\n"
        "                 template renders. Server requests use enable_thinking\n"
        "  --temp F       temperature (0 = greedy: returns the model's argmax\n"
        "                 with no repeat penalty applied)\n"
        "  --top-k N      top-k (0 = off)\n"
        "  --top-p F      top-p\n"
        "  --min-p F      min-p vs the top candidate (0 = off)\n"
        "  --repeat-penalty F  penalty on recently emitted tokens (1 = off)\n"
        "                 the five options above default to the model family's\n"
        "                 published settings; see --caps for the preset table\n"
        "  --rope-scale F force linear rope position scaling by F\n"
        "  --yarn-factor F override a model's native YaRN factor; preserves\n"
        "                 its original context and correction parameters\n"
        "  --rope-base F  override rope frequency base\n"
        "  --system TEXT  system prompt for interactive chat (-i) only\n"
        "  --chat-template chatml|chatml-think|llama2|llama3|mistral|mistral-v1|\n"
        "                 mistral-nemo|zephyr|phi3|phi4|gemma|gemma4|gemma4-mainline|\n"
        "                 apertus|ornith|qwen35|qwen35-nothink|nemotron|nemotron35|\n"
        "                 hermes4|granite|granite4|granite42|\n"
        "                 qwen38|qwen3-coder|muse|harmony|raw\n"
        "                 (default: auto). Applies to chat and --serve; not\n"
        "                 valid with a multi-model -m swap set\n"
        "  --no-bos       do not add BOS token\n"
        "  --ignore-eos   keep generating past end-of-text tokens\n"
        "  --gpu auto|off GPU offload if a backend is available (default auto)\n"
        "  --gpu-layers N force N leading layers on the GPU, rest on CPU\n"
        "                 (0 = no GPU, same as --gpu off; omit for auto-fit)\n"
        "  --cpu-moe [N|auto]  keep sparse MoE expert FFNs in RAM while CUDA\n"
        "                 runs attention and other dense tensors on the GPU.\n"
        "                 Bare = every expert layer on the host; N = only the\n"
        "                 deepest N expert layers (the rest stay on the GPU);\n"
        "                 auto = fill the VRAM budget with expert banks first\n"
        "  --wait-for-vram [S]  when another registered runner is holding the\n"
        "                 GPU, queue for up to S seconds (default 300) instead\n"
        "                 of refusing. Without it a runner that does not fit\n"
        "                 fails immediately, naming who holds what.\n"
        "  --vram-priority N  advisory small-integer priority (default 0,\n"
        "                 also RUNNER_VRAM_PRIORITY). Shown in the refusal\n"
        "                 listing; among several --wait-for-vram waiters on\n"
        "                 one GPU, a higher N is admitted first once space\n"
        "                 frees, but only among waiters that currently fit.\n"
        "                 No effect without --wait-for-vram.\n"
        "  --yield-on-request  in --serve, release the resident model at the\n"
        "                 next idle point between requests when another\n"
        "                 process has asked it to (vram_request_yield). A\n"
        "                 request, never preemption: nothing forces this.\n"
        "  --reserve P    cap this runner at P%% of TOTAL VRAM and RAM. With\n"
        "                 -c 0 the context is auto-fit to whatever the\n"
        "                 reservation leaves after the weights\n"
        "  --reserve-vram P / --reserve-ram P   as --reserve, one budget only\n"
        "  --reserve-cpu P  size the thread count as P%% of cores\n"
        "  --kv f16|q8|k8v4|fp4  KV cache storage (default f16). q8 roughly\n"
        "                 halves the cache; k8v4 keeps K at q8 and stores V as\n"
        "                 fp4 (about 41%% of f16); fp4 stores both as fp4 (about\n"
        "                 28%%; E2M1 values, one UE4M3 scale per 16 channels,\n"
        "                 quantised after RoPE). All three are lossy: output\n"
        "                 differs from an f16 cache, fp4 K costs the most\n"
        "  --mlock        wire the weights into RAM so the OS cannot evict them.\n"
        "                 Off by default and allowed to fail: on a machine with\n"
        "                 little headroom, locking the model can cause the very\n"
        "                 pressure it avoids. Without it a loaded machine can\n"
        "                 page the weights out and every token reads from disk\n"
        "  --moe-prefetch on|off|auto   hand routed experts to the OS as whole\n"
        "                 blocks before the FFN reads them (MoE models only).\n"
        "                 auto (default): on for Apple Silicon when the weights\n"
        "                 exceed available RAM — measured 1.26-1.43x decode\n"
        "                 there — and off elsewhere, where the same A/B\n"
        "                 measured nothing on fast storage\n"
        "  --draft PATH   small same-vocab GGUF for speculative decoding\n"
        "                 (one-shot, chat, and single-model --serve)\n"
        "  --draft-k N    draft tokens per round (default 4)\n"
        "  --mtp          draft from the model's own NextN/MTP head (CPU path)\n"
        "  --mtp-file PATH  the NextN/MTP head from a companion GGUF (llama.cpp's\n"
        "                 MTP-only export) when the model file carries none; implies --mtp\n"
        "  --draft-lookup draft from n-gram matches in the context itself\n"
        "                 (prompt lookup: no weights, no draft forward; one\n"
        "                 draft source per run, exclusive with --draft/--mtp)\n"
        "  --reasoning-budget N  close a reasoning turn once it has spent N\n"
        "                 tokens (budget forcing; 0 = off, the default). Per\n"
        "                 request: reasoning_max_tokens. Needs a model whose\n"
        "                 reasoning turn ends on a single token\n"
        "  --bare-recipients  Muse only: write a bare tool name verbatim in\n"
        "                 the system turn's valid-recipients line (\"read\")\n"
        "                 instead of the reference's namespace pattern\n"
        "                 (\"read.*\"); off by default (the reference\n"
        "                 rendering). Per request: bare_recipients\n"
        "  --loop-guard   close a reasoning turn that starts repeating (a span\n"
        "                 repeated back to back); off by default. Per request:\n"
        "                 loop_guard, loop_guard_span/repeats/window,\n"
        "                 loop_guard_everywhere\n"        "  --reasoning-temp F  temperature for REASONING turns only; calls and\n"
        "                 answers keep the request's own sampler (off by\n"
        "                 default). Per request: reasoning_temperature, with\n"
        "                 reasoning_top_p / reasoning_min_p / reasoning_top_k\n"        "  --reasoning-budget-message S  the sentence the forced close writes\n"
        "                 before the close token (default: a short neutral\n"
        "                 one; \"\" for a bare close). Per request:\n"
        "                 reasoning_budget_message\n"
        "  --draft-required  fail instead of decoding plain if --draft is\n"
        "                 refused (local CLI only; serve mode reports\n"
        "                 draft.active from /v1/capabilities)\n"
        "  --bench-json   run a small decode benchmark and print JSON metrics\n"
        "  --score        teacher-forced scoring: per-token log P(token|prefix)\n"
        "                 over the raw -p/-f text (no template, no sampling),\n"
        "                 printed as JSON with NLL and perplexity\n"
        "  --decide FILE  typed decisions: one /v1/decide request per line of\n"
        "                 FILE (state, questions[{question, options}]), one\n"
        "                 response per line on stdout; options scored verbatim\n"
        "                 in context, no sampling (- reads stdin)\n"
        "  --lora FILE    load a LoRA adapter GGUF beside the frozen base\n"
        "                 (dense projections, on the CPU and on an offloaded\n"
        "                 CUDA or Metal block; fails closed otherwise)\n"
        "  --lora-scale F multiply the adapter's trained alpha/r (default 1.0)\n"
        "  --lora-sig F   OMS bundle for the --lora adapter (default: its path\n"
        "                 + .sig when it exists); verified with --model-pubkey,\n"
        "                 and required by --require-signed-model like the model\n"
        "  --adapter NAME=PATH  with --serve: load a LoRA adapter once and serve\n"
        "                 it per request as \"model\": \"<model>:NAME\" (CPU\n"
        "                 hooks, --gpu off; repeatable, up to 16; exclusive\n"
        "                 with --lora; --lora-scale applies)\n"
        "  --train FILE   AdamW LoRA training (CPU): plain text, or .jsonl\n"
        "                 lines {\"prompt\",\"completion\",\"weight\"} with the\n"
        "                 prompt masked from the loss. Deterministic by\n"
        "                 default: same data + seed -> byte-identical adapter\n"
        "  --train-eot    append the template's end-of-turn token to every\n"
        "                 completion (a jsonl line can set \"end_of_turn\")\n"
        "  --train-dpo F  preference pairs .jsonl {\"prompt\",\"chosen\",\n"
        "                 \"rejected\"}: DPO (arXiv 2305.18290) over the\n"
        "                 adapter; the reference policy is these same weights\n"
        "                 with the adapter bypassed, so it costs no second copy\n"
        "  --dpo-beta F   DPO beta, the reference-deviation weight (0.1)\n"
        "  --train-steps N --lr F --train-ctx N --save-every N\n"
        "                 training-loop knobs (defaults 100, 1e-4, 128, 0)\n"
        "  --train-out F  adapter GGUF to write (default adapter-out.gguf)\n"
        "  --lora-rank R  fresh-adapter rank when no --lora is given (8)\n"
        "  --caps         print machine capabilities as JSON and exit\n"
        "  --tool-info    load -m MODEL and print its native tool-call protocol,\n"
        "                 chat template and sampling preset\n"
        "  --adapt-info   load -m MODEL and print, as JSON, whether an adapter can\n"
        "                 be served on it and trained on it, with the reason\n"
        "                 when it cannot\n"
        "  --doctor       load -m MODEL, run one short probe and print a diagnostic\n"
        "                 report as JSON: version, placement, template, sampler,\n"
        "                 memory, timings, findings and a next step for each.\n"
        "                 The report holds no prompt or reply text unless\n"
        "                 --doctor-include-text is given; read it before sharing\n"
        "  --shadow-mode  install shadow mode for Claude Code and Codex if present\n"
        "                 (asks first; --yes skips the question); -m MODEL names\n"
        "                 the model the /shadow offload will serve\n"
        "                 as one JSON line, then exit\n"
        "  --fit PATH     estimate whether a GGUF fits this machine and exit:\n"
        "                 RAM, and on a discrete GPU the layers the offload\n"
        "                 budget would hold; reads only the header, so a\n"
        "                 partial download works\n"
        "  --version      print the runner version and exit\n"
        "  --parent-pid N exit when process N dies, not when the thread that\n"
        "                 launched this one does (supervisor cleanup)\n"
        "  -v             verbose model info\n"
        "  -h, --help     print this help to stdout and exit\n",
        prog);
}

static void usage(const char *prog) { usage_to(stderr, prog); }

// --transcript capture: stream to stdout AND keep the exact bytes for the
// record (the transcript's output.text is what the terminal saw, verbatim)
typedef struct {
    char *buf;
    size_t n, cap;
    bool echo, failed;
} outcap_t;
static int outcap_cb(void *ud, const char *bytes, int n) {
    outcap_t *o = (outcap_t *)ud;
    if (n < 0 || (size_t)n > SIZE_MAX - o->n - 1) {
        o->failed = true;
    } else if (o->n + (size_t)n + 1 > o->cap) {
        size_t need = o->n + (size_t)n + 1;
        size_t cap = need <= (SIZE_MAX - 256) / 2
                   ? need * 2 + 256 : need;
        char *nb = realloc(o->buf, cap);
        if (nb) { o->buf = nb; o->cap = cap; }
        else o->failed = true;
    }
    if (!o->failed) {
        memcpy(o->buf + o->n, bytes, (size_t)n);
        o->n += (size_t)n;
        o->buf[o->n] = 0;
    }
    if (o->echo && n > 0) {
        fwrite(bytes, 1, (size_t)n, stdout);
        fflush(stdout);
    }
    return o->failed ? 1 : 0;
}

static int stdout_cb(void *ud, const char *bytes, int n) {
    (void)ud;
    fwrite(bytes, 1, n, stdout);
    fflush(stdout);
    return 0;
}

static int discard_cb(void *ud, const char *bytes, int n) {
    (void)ud; (void)bytes; (void)n;
    return 0;
}

// chat-mode display: thinking channels are shown between [thinking] markers
// instead of leaking the architecture's raw thinking tags
typedef struct {
    think_split ts;
    bool in_think;
} chat_out;

static int chat_emit(void *ud, int reasoning, const char *bytes, int n) {
    chat_out *c = ud;
    if (reasoning && !c->in_think)  { fputs("[thinking]\n", stdout);   c->in_think = true; }
    if (!reasoning && c->in_think)  { fputs("\n[/thinking]\n", stdout); c->in_think = false; }
    fwrite(bytes, 1, n, stdout);
    fflush(stdout);
    return 0;
}

static int chat_cb(void *ud, const char *bytes, int n) {
    chat_out *c = ud;
    return think_feed(&c->ts, bytes, n, chat_emit, c);
}

// --fit: what a GGUF would cost on this machine, from its HEADER alone.
//
// The point is to answer "should I download this" before spending the
// bandwidth, so it reads through gguf_open_header() and works on a file that
// is only a few megabytes of header. It does NOT fetch anything: the runner
// does not speak HTTP and a fit check is not a reason to teach it. The README
// documents the ranged-read one-liner that turns a URL into a header file.
static void fit_gib(char *buf, size_t n, uint64_t bytes) {
    snprintf(buf, n, "%.2f GiB", (double)bytes / (double)(1ull << 30));
}

static int run_fit_check(const char *path, int n_ctx_want,
                         const model_params *mp) {
    gguf_file g;
    if (!gguf_open_header(&g, path)) return 1;   // already reported why

    model_fit f;
    if (!model_fit_report(&g, n_ctx_want, &f)) {
        fprintf(stderr, "error: %s does not carry the geometry a fit check "
                        "needs (architecture, block_count, head counts)\n", path);
        gguf_close(&g);
        return 1;
    }
    // A discrete GPU holds the leading layers, so RAM only has to hold the
    // rest: answering from RAM alone told an RTX 3070 box that an 8B model
    // it serves with 35 of 40 layers on the card would page every token.
    char gname[128] = "";
    uint64_t gbudget = 0, gheadroom = 0;
    if (mp->gpu_mode != GPU_OFF && gpu_available(gname, sizeof gname) &&
        gpu_offload_budget(mp->reserve_vram_pct, &gbudget, &gheadroom))
        model_fit_gpu(&g, &f, gbudget, gheadroom, mp->gpu_layers_override);

    char w[32], h[32], kf[32], kq[32], av[32];
    fit_gib(w, sizeof w, f.weights);
    fit_gib(h, sizeof h, f.hot);
    fit_gib(kf, sizeof kf, f.kv_f16);
    fit_gib(kq, sizeof kq, f.kv_q8);
    char k4[32], k84[32];
    fit_gib(k4, sizeof k4, f.kv_fp4);
    fit_gib(k84, sizeof k84, f.kv_k8v4);
    fit_gib(av, sizeof av, f.available);

    uint32_t split = gguf_get_u32(&g, "split.count", 0);
    printf("fit: %s\n", path);
    printf("  model         %s, %d layers", f.arch, f.n_layer);
    if (f.sparse) printf(", MoE %d experts, %d used", f.n_expert, f.n_expert_used);
    printf("\n");
    if (split > 1)
        printf("  NOTE          this is part 1 of a %u-part split; the sizes "
               "below are THIS PART only\n", split);
    printf("  weights       %s\n", w);
    if (f.sparse)
        printf("  hot set       %s  (only the routed experts a token actually "
               "uses)\n", h);
    printf("  kv cache      %s at ctx %d, f16", kf, f.n_ctx);
    if (f.kv_q8) printf("   |  %s with --kv q8", kq);
    else         printf("   |  --kv q8 unavailable (head_dim is not a multiple of 32)");
    if (f.kv_k8v4) printf("   |  %s with --kv k8v4", k84);
    if (f.kv_fp4) printf("   |  %s with --kv fp4", k4);
    else          printf("   |  --kv fp4 unavailable (head_dim is not a multiple of 16)");
    printf("\n");
    if (f.available) printf("  available RAM %s right now\n", av);
    else             printf("  available RAM unknown on this platform\n");
    static const char *const kvname[4] = { "f16", "--kv q8", "--kv k8v4", "--kv fp4" };
    if (f.gpu) {
        char gb[32], fx[32];
        fit_gib(gb, sizeof gb, f.gpu_budget);
        fit_gib(fx, sizeof fx, f.gpu_fixed);
        printf("  gpu           %s, offload budget %s right now (%s of it for "
               "embeddings, scratch and headroom)\n", gname, gb, fx);
        printf("  split        ");
        for (int k = 0, first = 1; k < 4; k++) {
            if (f.gpu_layers[k] < 0) continue;
            char hn[32];
            fit_gib(hn, sizeof hn, f.host_need[k]);
            printf("%s %s: %d of %d layers on the GPU, %s in RAM",
                   first ? "" : "  |", kvname[k], f.gpu_layers[k], f.n_layer, hn);
            first = 0;
        }
        printf("\n");
    }

    const char *v = model_fit_verdict(&f);
    printf("  verdict       %s", v);
    if (f.gpu && f.available) {
        int k = !strcmp(v, "FITS") ? 0 : !strcmp(v, "FITS WITH --kv q8") ? 1
              : !strcmp(v, "FITS WITH --kv k8v4") ? 2 : !strcmp(v, "FITS WITH --kv fp4") ? 3 : -1;
        if (k >= 0) {
            char slack[32];
            fit_gib(slack, sizeof slack, f.available - f.host_need[k]);
            printf(" — %d of %d layers on the GPU, %s of RAM to spare at ctx %d\n",
                   f.gpu_layers[k], f.n_layer, slack, f.n_ctx);
        } else {
            uint64_t least = UINT64_MAX;
            for (int j = 0; j < 4; j++)
                if (f.gpu_layers[j] >= 0 && f.host_need[j] < least) least = f.host_need[j];
            char over[32];
            fit_gib(over, sizeof over, least - f.available);
            printf(" — %s more RAM than is available even with the smallest cache "
                   "and the GPU full. The CPU layers would page from disk.\n", over);
            printf("                fixes: a smaller quantization, a shorter -c, "
                   "or free GPU memory (another process may hold it)\n");
        }
    } else if (!f.available) {
        printf("\n                available RAM could not be read, so this is "
               "sizes only\n");
    } else if (!strcmp(v, "FITS")) {
        char slack[32];
        fit_gib(slack, sizeof slack, f.available - f.hot - f.kv_f16);
        printf(" — %s to spare at ctx %d\n", slack, f.n_ctx);
    } else if (!strcmp(v, "FITS WITH --kv q8")) {
        char over[32];
        fit_gib(over, sizeof over, f.hot + f.kv_f16 - f.available);
        printf(" — an f16 cache is %s over; q8 halves it\n", over);
    } else if (!strcmp(v, "FITS WITH --kv k8v4")) {
        char over[32];
        fit_gib(over, sizeof over, f.hot + f.kv_q8 - f.available);
        printf(" — a q8 cache is %s over; k8v4 (fp4 V) takes a quarter off it\n", over);
    } else if (!strcmp(v, "FITS WITH --kv fp4")) {
        char over[32];
        fit_gib(over, sizeof over, f.hot + (f.kv_q8 ? f.kv_q8 : f.kv_f16) - f.available);
        printf(" — a q8 cache is %s over; fp4 halves it again\n", over);
    } else {
        char over[32];
        fit_gib(over, sizeof over, f.hot + (f.kv_fp4 ? f.kv_fp4 : f.kv_q8 ? f.kv_q8 : f.kv_f16)
                                   - f.available);
        printf(" — %s over even with the smallest cache. Every token would "
               "page from disk.\n", over);
        printf("                fixes: a smaller quantization, a shorter -c, "
               "or --gpu-layers to split it\n");
    }
    if (f.sparse)
        printf("  note          the verdict uses the hot set; a sparse MoE runs "
               "usefully while the file exceeds RAM\n");
    printf("  note          KV is an upper bound: models with per-layer KV "
           "geometry (shared KV, MLA) use less\n");

    gguf_close(&g);
    return 0;
}


// --shadow-mode: exec the Python client's `shadow install`. The client is
// stdlib-only and lives at <exe dir>/python/src in the source tree and in the
// release archive, so PYTHONPATH gets that directory when it exists; an
// installed package is found by Python itself otherwise. Everything the user
// is asked, and everything written, is the Python side's, which has the tests.
#ifdef _WIN32
#include <process.h>
#endif
static int run_shadow_mode(const char *model, bool yes) {
    char *exe = plat_executable_path();
#ifndef _WIN32
    // Linux answers "/proc/self/exe": a link, whose directory is /proc/self.
    // The client lives beside the real binary, and the recorded runner path
    // must outlive this process, so resolve the link first.
    if (exe) {
        char *real = realpath(exe, NULL);
        if (real) { free(exe); exe = real; }
    }
#endif
    char client[4096] = "";
    if (exe) {
        const char *slash = strrchr(exe, '/');
#ifdef _WIN32
        const char *bslash = strrchr(exe, '\\');
        if (!slash || (bslash && bslash > slash)) slash = bslash;
#endif
        size_t dir_len = slash ? (size_t)(slash - exe) : 0;
        if (dir_len && dir_len < sizeof client - 32) {
            memcpy(client, exe, dir_len);
            client[dir_len] = '\0';
            strcat(client, "/python/src");
            char probe[4096];
            snprintf(probe, sizeof probe, "%s/xyntetik_runner/shadow/cli.py", client);
            struct stat st;
            if (stat(probe, &st) != 0) client[0] = '\0';
        }
    }
    if (client[0]) {
        const char *old = getenv("PYTHONPATH");
        char joined[8192];
#ifdef _WIN32
        const char *sep = ";";
#else
        const char *sep = ":";
#endif
        if (old && *old) snprintf(joined, sizeof joined, "%s%s%s", client, sep, old);
        else snprintf(joined, sizeof joined, "%s", client);
#ifdef _WIN32
        _putenv_s("PYTHONPATH", joined);
#else
        setenv("PYTHONPATH", joined, 1);
#endif
    }
    const char *argv[16];
    int n = 0;
    argv[n++] = "python3";
    argv[n++] = "-m";
    argv[n++] = "xyntetik_runner.shadow";
    argv[n++] = "install";
    argv[n++] = "--runner";
    argv[n++] = exe ? exe : "runner";
    if (model) { argv[n++] = "--model"; argv[n++] = model; }
    if (yes) argv[n++] = "--yes";
    argv[n] = NULL;
#ifdef _WIN32
    argv[0] = "python";
    intptr_t rc = _spawnvp(_P_WAIT, "python", argv);
    if (rc == -1) { argv[0] = "py"; rc = _spawnvp(_P_WAIT, "py", argv); }
    if (rc == -1) {
        fprintf(stderr, "error: --shadow-mode needs Python 3 on PATH (python or py); "
                        "the client is stdlib-only and ships beside this binary\n");
        return 2;
    }
    return (int)rc;
#else
    execvp("python3", (char *const *)argv);
    argv[0] = "python";
    execvp("python", (char *const *)argv);
    fprintf(stderr, "error: --shadow-mode needs python3 on PATH; the client is "
                    "stdlib-only and ships beside this binary (python/src)\n");
    return 2;
#endif
}

// --caps: what this build and machine can do, as one JSON document, with no
// model loaded. Every field comes from the build or a live query, never from
// a flag, which is why it takes no options.
static int run_caps(void) {
    char gname[128];
    bool has_gpu = gpu_available(gname, sizeof(gname));
    printf("{\"version\":\"%s\",\"os\":\"%s\",\"arch\":\"%s\",\"cpu_cores\":%d,"
           "\"ram_bytes\":%llu,\"ram_available_bytes\":%llu,\"gpu\":",
           RUNNER_VERSION,
#if defined(_WIN32)
           "windows",
#elif defined(__APPLE__)
           "macos",
#else
           "linux",
#endif
           RUNNER_BUILD_ARCH,
           plat_cpu_count(), (unsigned long long)plat_ram_bytes(),
           // Not total RAM: "model file size <= RAM" is the test that
           // passes right before a machine starts thrashing. A launcher
           // needs to know what is actually free plus reclaimable, so it
           // can refuse a model that will not stay resident.
           (unsigned long long)plat_ram_available_bytes());
    if (has_gpu) {
        size_t vfree = 0, vtotal = 0;
        bool vm = gpu_mem_info(&vfree, &vtotal);
#ifdef __APPLE__
        printf("{\"backend\":\"metal\",\"name\":\"%s\",\"unified_memory\":true",
               gname);
#else
        // queried, not assumed: an integrated CUDA device (DGX Spark's
        // GB10) shares one pool with the CPU, and the first field a
        // launcher reads to decide whether RAM and VRAM are two budgets
        // must not be a compile-time constant that says they are
        printf("{\"backend\":\"cuda\",\"name\":\"%s\",\"unified_memory\":%s",
               gname, gpu_unified_memory() ? "true" : "false");
        printf(",\"min_compute_capability\":\"%s\",\"min_gpu\":\"%s\","
               "\"ptx_target\":\"%s\"",
               RUNNER_CUDA_MIN_CC, RUNNER_CUDA_MIN_GPU,
               RUNNER_CUDA_PTX_TARGET);
#endif
        if (vm)
            printf(",\"vram_bytes\":%llu,\"vram_free_bytes\":%llu",
                   (unsigned long long)vtotal, (unsigned long long)vfree);
        // Read this alongside gpu_quants. "This backend has a sparse-MoE
        // path" and "this exact MoE variant's router/layout/activation is
        // implemented" are different claims; gpu_init() still guards the
        // latter at load.
        printf(",\"moe\":%s", gpu_moe_ok() ? "true" : "false");
        printf(",\"eseries\":%s", gpu_eseries_ok() ? "true" : "false");
        printf(",\"kv_q8\":%s", gpu_kv_q8_ok() ? "true" : "false");
        printf(",\"kv_fp4\":%s", gpu_kv_fp4_ok() ? "true" : "false");
        // Metal only: a single allocation above this fails even when the
        // file fits in RAM, so ram_available_bytes alone overstates what
        // the GPU path can load. Same argument as moe/kv_q8: don't let a
        // scheduler read capability across a capacity cliff.
        // Proves the binary carries the kernels currently in the tree.
        // make test-shader-embed compares this to a fresh hash of
        // src/kernels.metal.
        const char *ssha = gpu_shader_source_sha();
        if (ssha) printf(",\"shader_source_sha256\":\"%s\"", ssha);
        size_t mws = 0;
        if (gpu_max_working_set(&mws))
            printf(",\"max_working_set_bytes\":%llu",
                   (unsigned long long)mws);
        printf("}");
    } else
        printf("null");
    // the preset table is a property of the build, not of any one model,
    // so --caps can publish it without loading weights
    printf(",\"sampling_presets\":[");
    for (int i = 0; ; i++) {
        const sampler_preset *sp = sampler_preset_at(i);
        if (!sp) break;
        char esc_src[512];
        json_escape(sp->source, strlen(sp->source), esc_src, sizeof(esc_src));
        printf("%s{\"name\":\"%s\",\"temperature\":%.2f,\"top_p\":%.2f,"
               "\"top_k\":%d,\"min_p\":%.2f,\"repeat_penalty\":%.2f,"
               "\"source\":\"%s\"}",
               i ? "," : "", sp->name, (double)sp->temp, (double)sp->top_p,
               sp->top_k, (double)sp->min_p, (double)sp->repeat_penalty,
               esc_src);
    }
    printf("]");
    // KV cache formats this build can store. Both are available on every
    // backend (CPU and CUDA); q8 needs head_dim % 32 == 0, which is a
    // per-model property and so is reported at load, not here. f16 is the
    // default because q8 is lossy — it does not reproduce f16 output.
    printf(",\"kv_types\":[\"f16\",\"q8\",\"k8v4\",\"fp4\"],\"kv_type_default\":\"f16\"");
    // Placement modes controllers may safely request. cpu_moe means the
    // CUDA backend can keep dense/attention tensors resident while sparse
    // expert FFNs execute from system RAM.
    // cpu_moe_partial advertises the optional layer count / auto fit, so a
    // controller can size the expert split instead of guessing all-or-none.
    printf(",\"tensor_placement\":{\"cpu_moe\":true,"
           "\"cpu_moe_partial\":true}");
    // the fixed -m swap-registry capacity, so a controller bounds the set of
    // models it launches instead of overflowing it (see RUNNER_MAX_MODELS)
    printf(",\"max_models\":%d", RUNNER_MAX_MODELS);
    // One table, two consumers: `quants` is every format the engine can
    // read, and `gpu_quants` is that same table filtered by the active
    // backend's own admission test (gpu_quant_ok -> gpu_type_ok). These
    // were two hand-kept literals under an #ifdef, each carrying a comment
    // asking a human to keep it in step with a switch in another file.
    // They drifted, and --caps advertised Metal kernels that did not
    // exist. Sourcing the answer removes the class of bug, not just the
    // instance: a new format needs one edit, in the backend, and both
    // lists follow.
    static const struct { const char *name; int type; } QUANTS[] = {
        { "F32",    T_F32    }, { "F16",    T_F16    },
        { "BF16",   T_BF16   }, { "Q8_0",   T_Q8_0   },
        { "Q4_0",   T_Q4_0   }, { "Q4_1",   T_Q4_1   },
        { "Q5_0",   T_Q5_0   }, { "Q5_1",   T_Q5_1   },
        { "Q2_K",   T_Q2_K   }, { "Q3_K",   T_Q3_K   },
        { "Q4_K",   T_Q4_K   }, { "Q5_K",   T_Q5_K   },
        { "Q6_K",   T_Q6_K   }, { "IQ4_NL", T_IQ4_NL },
        { "IQ4_XS", T_IQ4_XS }, { "MXFP4",  T_MXFP4  },
        { "IQ1_S",  T_IQ1_S  }, { "IQ1_M",  T_IQ1_M  },
        { "IQ2_XXS", T_IQ2_XXS }, { "IQ2_XS", T_IQ2_XS },
        { "IQ2_S",  T_IQ2_S  },
        { "IQ3_XXS", T_IQ3_XXS }, { "IQ3_S", T_IQ3_S },
        { "NVFP4",  T_NVFP4  },
    };
    const size_t n_quants = sizeof QUANTS / sizeof *QUANTS;
    printf(",\"quants\":[");
    for (size_t qi = 0; qi < n_quants; qi++)
        printf("%s\"%s\"", qi ? "," : "", QUANTS[qi].name);
    // gpu_init() still refuses unsupported architectures and layouts; this
    // is the tensor-format half only, which is what an advisor filters on.
    printf("],\"gpu_quants\":[");
    for (size_t qi = 0, shown_q = 0; qi < n_quants; qi++)
        if (gpu_quant_ok(QUANTS[qi].type))
            printf("%s\"%s\"", shown_q++ ? "," : "", QUANTS[qi].name);
    printf("]");
    // the architectures model_load will admit, so a catalog/advisor can
    // filter unrunnable models by arch without shipping its own copy of the
    // allowlist (single source of truth: model_supported_archs)
    printf(",\"architectures\":[");
    size_t n_arch;
    const char *const *arches = model_supported_archs(&n_arch);
    for (size_t i = 0; i < n_arch; i++)
        printf("%s\"%s\"", i ? "," : "", arches[i]);
    printf("]}\n");
    return 0;
}

// Everything the command line sets, each with its default, before any
// resource exists. Preparation further down overwrites some of it the way a
// flag would (--resume and --verify take their settings from the image or the
// record); the model, tokenizer, engine and loaded records live in main().
typedef struct {
    // what to run on, and the prompt
    const char *model_path, *hf_spec, *prompt, *prompt_file, *system_prompt;
    const char *tmpl_arg, *schema_file;
    int n_predict, n_threads, reserve_cpu_pct, thinking;
    bool n_given, kv_given, seed_given;
    bool interactive, verbose, no_bos, ignore_eos, json_mode;
    // modes that answer without generating
    bool serve, caps, tool_info, bench_json, score, shadow_mode, shadow_yes;
    bool adapt_info;    // --adapt-info: adapter eligibility as JSON
    bool doctor;        // --doctor: one diagnostic report, then exit
    bool doctor_text;   // --doctor-include-text: keep the probe's text
    const char *decide_path;   // --decide FILE (jsonl of /v1/decide requests)
    const char *fit_path;
    // serving
    int port, parallel, ttl;
    long parent_pid;
    bool no_tray, force_uncertified;
    // R1.2.2: per-request receipts in serve mode
    const char *receipts_dir;
    int receipts_keep;
    const char *kv_snapshots;
    const char *sessions_path;   // --sessions DIR (R1.3.5)
    // R8.6: --adapter NAME=PATH, repeatable; served per request as "<model>:NAME"
    const char *adapter_names[16], *adapter_paths[16];
    int n_adapters;
    // speculation and reasoning controls
    const char *draft_path;
    bool draft_required, mtp_on, draft_lookup, loop_guard;
    int draft_k;
    // R4.12.16: server-side default reasoning budget in tokens (0 = off); a
    // request's reasoning_max_tokens overrides it either way.
    int reasoning_budget;
    const char *reasoning_budget_message;   // NULL = the built-in line
    float reasoning_temp;                    // < 0 = off
    // loop_guard: R4.12.18, off by default
    // file transformations
    const char *quant_out, *quant_type, *prune_experts, *remove_sublayer;
    const char *type_plan, *merge_out, *context_out;
    bool type_plan_strict, merge_allow_erased;
    // adapters and training
    const char *lora_path;
    const char *lora_sig;   // R1.2.3: the adapter's OMS bundle
    float lora_scale;
    const char *train_path, *train_out, *dpo_path;
    int train_steps, train_ctx, lora_rank, save_every;
    bool train_eot;
    float train_lr, dpo_beta;
    // records, signing, verification, watermarks, session images
    const char *transcript_path, *transcript_prev, *sign_key, *keygen_path;
    const char *keygen_algo, *lineage_path, *require_eval;
    const char *sign_record, *record_prev, *check_record;
    // R1.2.1: receipt bundles
    const char *export_bundle, *bundle_out, *check_bundle;
    const char *export_pack, *pack_out, *check_pack, *pack_envelope;
    const char *pack_attach[PACK_MAX_ATTACH];
    int n_pack_attach;
    const char *trust_key, *model_sig, *model_pubkey, *sign_model, *model_key;
    bool require_signed, require_signed_model;
    const char *watermark_path, *watermark_keygen, *detect_path;
    const char *verify_path, *session_out, *resume_path;
    int suspend_after;
    bool fork_seed_given;
    uint64_t fork_seed;
    model_params mp;
    // The four sampling knobs are filled in from the model's family preset
    // once the model is known; anything named on the command line is recorded
    // in ov and wins over the preset. smp holds the seed until then.
    sampler smp;
    sampler_override ov;
} cli_opts;

// What --verify read from its record before the load: the parsed record,
// its signature state and public key, and the chain link it names. The
// replay after the load reads all four.
typedef struct {
    jv *rec;
    receipt_sig_state rsig;   // the record's signature state
    char rec_pub[SIGN_PUBHEX_CAP];
    char prev[65];
} cli_verify;

// What a mode runs on once the model is loaded: the model and its tokenizer,
// the engine over them and the prompt token buffer, which cli_cleanup frees
// together, and the facts the load decided that records and reports name.
typedef struct {
    model_t m;
    tokenizer tok;
    engine e;
    int32_t *toks;
    size_t tok_cap;
    char model_sig_json[512];     // load-time OMS verdict for the receipt
    char adapter_sig_json[512];   // and the adapter's (R1.2.3)
    // the measured-envelope verdict the one-shot model loaded under, for its
    // transcript (R1.1.5); unknown on paths that never ran the gate
    int env_state;
    bool env_known;
    char sampling[256];           // the effective sampler, as the load line says
    double load_s;
} cli_run;

static void cli_opts_init(cli_opts *o) {
    memset(o, 0, sizeof *o);
    o->keygen_algo = SIGN_ALGO_ED25519;
    o->n_predict = 256;
    o->port = 8080;
    o->parallel = 1;
    o->ttl = -1;   // -1: 300 for swap mode, never for single
    o->draft_k = 4;
    o->reasoning_temp = -1.0f;
    o->thinking = THINK_DEFAULT;
    o->lora_scale = 1.0f;
    o->train_out = "adapter-out.gguf";
    o->train_steps = 100;
    o->train_ctx = 128;
    o->lora_rank = 8;
    o->train_lr = 1e-4f;
    o->dpo_beta = 0.1f;
    // RUNNER_VRAM_PRIORITY sets the baseline; --vram-priority (parsed below)
    // overrides it, same precedence as every other env/flag pair in runner.
    o->mp.vram_priority = (int)env_i64("RUNNER_VRAM_PRIORITY", -1000, 1000, 0);
}


// The flags, in order. Parsing stops where an early exit stands (--version,
// --help, --tray), so a flag after one is never read; *rc is the process exit
// status whenever this returns false.
static bool cli_parse(cli_opts *o, int argc, char **argv, int *rc) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define NEXT (i + 1 < argc ? argv[++i] : (usage(argv[0]), exit(1), (char*)0))
        if      (!strcmp(a, "-m")) o->model_path = NEXT;
        else if (!strcmp(a, "-hf") || !strcmp(a, "--hf")) o->hf_spec = NEXT;
        else if (!strcmp(a, "-p")) o->prompt = NEXT;
        else if (!strcmp(a, "-f")) o->prompt_file = NEXT;
        else if (!strcmp(a, "-n")) { o->n_predict = (int)int_arg(a, NEXT, -1, INT_MAX); o->n_given = true; }
        else if (!strcmp(a, "-c")) o->mp.n_ctx = (int)int_arg(a, NEXT, 0, INT_MAX);
        else if (!strcmp(a, "-b")) o->mp.n_batch = (int)int_arg(a, NEXT, 0, INT_MAX);
        else if (!strcmp(a, "-t")) o->n_threads = (int)int_arg(a, NEXT, 0, INT_MAX);
        else if (!strcmp(a, "-s")) { o->smp.rng = u64_arg(a, NEXT); o->seed_given = true; }
        else if (!strcmp(a, "-i")) o->interactive = true;
        else if (!strcmp(a, "-v")) o->verbose = true;
        else if (!strcmp(a, "--serve")) o->serve = true;
        // gemma-4 renders a non-thinking prompt by default, matching its
        // reference template; --think opts into the thinking shape
        else if (!strcmp(a, "--think")) o->thinking = THINK_ON;
        else if (!strcmp(a, "--no-think")) o->thinking = THINK_OFF;
        else if (!strcmp(a, "--tray")) { *rc = tray_main(); return false; }
        // The standing opt-out, and no longer a no-op: it suppresses both the
        // bare-invocation launch above and the tray that otherwise follows a
        // --serve or -i session below.
        else if (!strcmp(a, "--no-tray")) o->no_tray = true;
        else if (!strcmp(a, "--force-uncertified")) o->force_uncertified = true;
        else if (!strcmp(a, "--port")) o->port = (int)int_arg(a, NEXT, 1, 65535);
        else if (!strcmp(a, "--parallel")) o->parallel = (int)int_arg(a, NEXT, 1, 16);
        else if (!strcmp(a, "--ttl")) o->ttl = (int)int_arg(a, NEXT, -1, INT_MAX);
        else if (!strcmp(a, "--json")) o->json_mode = true;
        else if (!strcmp(a, "--json-schema")) o->schema_file = NEXT;
        else if (!strcmp(a, "--quantize")) o->quant_out = NEXT;
        else if (!strcmp(a, "--context-surgery")) o->context_out = NEXT;
        else if (!strcmp(a, "--merge-lora")) o->merge_out = NEXT;
        else if (!strcmp(a, "--merge-allow-erased")) o->merge_allow_erased = true;
        else if (!strcmp(a, "--transcript")) o->transcript_path = NEXT;
        else if (!strcmp(a, "--verify")) o->verify_path = NEXT;
        else if (!strcmp(a, "--transcript-prev")) o->transcript_prev = NEXT;
        else if (!strcmp(a, "--sign-key")) o->sign_key = NEXT;
        else if (!strcmp(a, "--lineage")) o->lineage_path = NEXT;
        else if (!strcmp(a, "--require-eval")) o->require_eval = NEXT;
        else if (!strcmp(a, "--keygen")) o->keygen_path = NEXT;
        else if (!strcmp(a, "--sign-record")) o->sign_record = NEXT;
        else if (!strcmp(a, "--record-prev")) o->record_prev = NEXT;
        else if (!strcmp(a, "--check-record")) o->check_record = NEXT;
        else if (!strcmp(a, "--export-bundle")) o->export_bundle = NEXT;
        else if (!strcmp(a, "--bundle-out")) o->bundle_out = NEXT;
        else if (!strcmp(a, "--check-bundle")) o->check_bundle = NEXT;
        else if (!strcmp(a, "--export-pack")) o->export_pack = NEXT;
        else if (!strcmp(a, "--pack-out")) o->pack_out = NEXT;
        else if (!strcmp(a, "--pack-envelope")) o->pack_envelope = NEXT;
        else if (!strcmp(a, "--check-pack")) o->check_pack = NEXT;
        else if (!strcmp(a, "--pack-attach")) {
            const char *f = NEXT;
            if (o->n_pack_attach == PACK_MAX_ATTACH) {
                fprintf(stderr, "error: --pack-attach: at most %d files\n",
                        PACK_MAX_ATTACH);
                *rc = 1;
                return false;
            }
            o->pack_attach[o->n_pack_attach++] = f;
        }
        else if (!strcmp(a, "--receipts")) o->receipts_dir = NEXT;
        else if (!strcmp(a, "--receipts-keep"))
            o->receipts_keep = (int)int_arg(a, NEXT, 0, 100000000);
        else if (!strcmp(a, "--keygen-algo")) o->keygen_algo = NEXT;
        else if (!strcmp(a, "--require-signed")) o->require_signed = true;
        else if (!strcmp(a, "--trust-key")) o->trust_key = NEXT;
        else if (!strcmp(a, "--model-sig")) o->model_sig = NEXT;
        else if (!strcmp(a, "--model-pubkey")) o->model_pubkey = NEXT;
        else if (!strcmp(a, "--sign-model")) o->sign_model = NEXT;
        else if (!strcmp(a, "--model-key")) o->model_key = NEXT;
        else if (!strcmp(a, "--watermark")) o->watermark_path = NEXT;
        else if (!strcmp(a, "--watermark-keygen")) o->watermark_keygen = NEXT;
        else if (!strcmp(a, "--detect-watermark")) o->detect_path = NEXT;
        else if (!strcmp(a, "--kv-snapshots")) o->kv_snapshots = NEXT;
        else if (!strcmp(a, "--sessions")) o->sessions_path = NEXT;
        else if (!strcmp(a, "--session-out")) o->session_out = NEXT;
        else if (!strcmp(a, "--resume")) o->resume_path = NEXT;
        else if (!strcmp(a, "--suspend-after"))
            o->suspend_after = (int)int_arg(a, NEXT, 1, INT_MAX);
        else if (!strcmp(a, "--fork-seed")) {
            o->fork_seed = u64_arg(a, NEXT);
            o->fork_seed_given = true;
        }
        else if (!strcmp(a, "--require-signed-model")) o->require_signed_model = true;
        else if (!strcmp(a, "--quant")) o->quant_type = NEXT;
        else if (!strcmp(a, "--prune-experts")) o->prune_experts = NEXT;
        else if (!strcmp(a, "--remove-sublayer")) o->remove_sublayer = NEXT;
        else if (!strcmp(a, "--type-plan")) o->type_plan = NEXT;
        else if (!strcmp(a, "--type-plan-strict")) o->type_plan_strict = true;
        else if (!strcmp(a, "--temp")) { o->ov.temp = (float)float_arg(a, NEXT, 0, FLT_MAX); o->ov.has_temp = true; }
        else if (!strcmp(a, "--top-k")) { o->ov.top_k = (int)int_arg(a, NEXT, 0, INT_MAX); o->ov.has_top_k = true; }
        else if (!strcmp(a, "--top-p")) { o->ov.top_p = (float)float_arg(a, NEXT, 0, 1); o->ov.has_top_p = true; }
        else if (!strcmp(a, "--min-p")) { o->ov.min_p = (float)float_arg(a, NEXT, 0, 1); o->ov.has_min_p = true; }
        else if (!strcmp(a, "--repeat-penalty")) { o->ov.repeat_penalty = (float)float_arg(a, NEXT, FLT_MIN, FLT_MAX); o->ov.has_repeat_penalty = true; }
        else if (!strcmp(a, "--rope-scale")) o->mp.rope_scale = (float)float_arg(a, NEXT, 0, FLT_MAX);
        else if (!strcmp(a, "--yarn-factor")) o->mp.yarn_factor = (float)float_arg(a, NEXT, 1, FLT_MAX);
        else if (!strcmp(a, "--rope-base")) o->mp.rope_base = (float)float_arg(a, NEXT, 0, FLT_MAX);
        else if (!strcmp(a, "--system")) o->system_prompt = NEXT;
        else if (!strcmp(a, "--chat-template")) o->tmpl_arg = NEXT;
        else if (!strcmp(a, "--no-bos")) o->no_bos = true;
        else if (!strcmp(a, "--ignore-eos")) o->ignore_eos = true;
        else if (!strcmp(a, "--gpu")) {
            const char *v = NEXT;
            if (!strcmp(v, "auto")) o->mp.gpu_mode = GPU_AUTO;
            else if (!strcmp(v, "off")) o->mp.gpu_mode = GPU_OFF;
            else { fprintf(stderr, "error: --gpu expects auto or off\n"); *rc = 1; return false; }
        }
        else if (!strcmp(a, "--kv")) {
            const char *v = NEXT;
            o->kv_given = true;
            o->mp.kv_q8 = o->mp.kv_fp4 = o->mp.kv_split = false;
            if (!strcmp(v, "q8")) o->mp.kv_q8 = true;
            else if (!strcmp(v, "fp4")) o->mp.kv_fp4 = true;
            else if (!strcmp(v, "k8v4")) o->mp.kv_split = true;
            else if (strcmp(v, "f16")) { fprintf(stderr, "error: --kv expects f16, q8, k8v4 or fp4\n"); *rc = 1; return false; }
        }
        else if (!strcmp(a, "--mlock")) o->mp.mlock = true;
        else if (!strcmp(a, "--moe-prefetch")) {
            const char *v = NEXT;
            if (!strcmp(v, "on")) o->mp.moe_prefetch = 1;
            else if (!strcmp(v, "off")) o->mp.moe_prefetch = -1;
            else if (!strcmp(v, "auto")) o->mp.moe_prefetch = 0;
            else { fprintf(stderr, "error: --moe-prefetch expects on, off or auto\n"); *rc = 1; return false; }
        }
        else if (!strcmp(a, "--draft"))   o->draft_path = NEXT;
        else if (!strcmp(a, "--draft-required")) o->draft_required = true;
        else if (!strcmp(a, "--draft-k")) o->draft_k = (int)int_arg(a, NEXT, 1, 15);
        else if (!strcmp(a, "--mtp"))     { o->mp.mtp = true; o->mtp_on = true; }
        else if (!strcmp(a, "--mtp-file")) { o->mp.mtp_path = NEXT; o->mp.mtp = true; o->mtp_on = true; }
        else if (!strcmp(a, "--draft-lookup")) o->draft_lookup = true;
        else if (!strcmp(a, "--reasoning-budget"))
            o->reasoning_budget = (int)int_arg(a, NEXT, 0, INT_MAX);
        else if (!strcmp(a, "--reasoning-budget-message"))
            o->reasoning_budget_message = NEXT;
        else if (!strcmp(a, "--loop-guard")) o->loop_guard = true;
        else if (!strcmp(a, "--bare-recipients")) template_set_bare_recipients(true);
        else if (!strcmp(a, "--reasoning-temp"))
            o->reasoning_temp = (float)float_arg(a, NEXT, 0, 2);
        else if (!strcmp(a, "--bench-json")) o->bench_json = true;
        else if (!strcmp(a, "--doctor")) o->doctor = true;
        else if (!strcmp(a, "--doctor-include-text")) o->doctor = o->doctor_text = true;
        else if (!strcmp(a, "--score")) o->score = true;
        else if (!strcmp(a, "--decide")) o->decide_path = NEXT;
        else if (!strcmp(a, "--lora")) o->lora_path = NEXT;
        else if (!strcmp(a, "--lora-sig")) o->lora_sig = NEXT;
        else if (!strcmp(a, "--adapter")) {
            char *spec = strdup(NEXT);
            char *eq = spec ? strchr(spec, '=') : NULL;
            if (!eq || eq == spec || !eq[1]) {
                fprintf(stderr, "error: --adapter wants NAME=PATH\n");
                *rc = 1;
                return false;
            }
            *eq = 0;
            bool name_ok = strlen(spec) < 64;
            for (const char *c = spec; name_ok && *c; c++)
                name_ok = (*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
                          (*c >= '0' && *c <= '9') || *c == '.' || *c == '_' ||
                          *c == '-';
            if (!name_ok) {
                fprintf(stderr, "error: --adapter NAME must be 1 to 63 "
                        "characters of [A-Za-z0-9._-]\n");
                *rc = 1;
                return false;
            }
            if (o->n_adapters >= 16) {
                fprintf(stderr, "error: at most 16 --adapter\n");
                *rc = 1;
                return false;
            }
            o->adapter_names[o->n_adapters] = spec;   // lives for the process
            o->adapter_paths[o->n_adapters++] = eq + 1;
        }
        else if (!strcmp(a, "--lora-scale"))
            o->lora_scale = (float)float_arg(a, NEXT, 0, FLT_MAX);
        else if (!strcmp(a, "--train")) o->train_path = NEXT;
        else if (!strcmp(a, "--train-dpo")) o->dpo_path = NEXT;
        else if (!strcmp(a, "--dpo-beta"))
            o->dpo_beta = (float)float_arg(a, NEXT, 0, FLT_MAX);
        else if (!strcmp(a, "--train-eot")) o->train_eot = true;
        else if (!strcmp(a, "--train-steps"))
            o->train_steps = (int)int_arg(a, NEXT, 1, INT_MAX);
        else if (!strcmp(a, "--train-out")) o->train_out = NEXT;
        else if (!strcmp(a, "--train-ctx"))
            o->train_ctx = (int)int_arg(a, NEXT, 2, INT_MAX);
        else if (!strcmp(a, "--lr"))
            o->train_lr = (float)float_arg(a, NEXT, 0, 1);
        else if (!strcmp(a, "--lora-rank"))
            o->lora_rank = (int)int_arg(a, NEXT, 1, 512);
        else if (!strcmp(a, "--save-every"))
            o->save_every = (int)int_arg(a, NEXT, 0, INT_MAX);
        else if (!strcmp(a, "--reserve")) o->mp.reserve_vram_pct = o->mp.reserve_ram_pct = (int)int_arg(a, NEXT, 0, 100);
        else if (!strcmp(a, "--reserve-vram")) o->mp.reserve_vram_pct = (int)int_arg(a, NEXT, 0, 100);
        else if (!strcmp(a, "--gpu-layers")) {
            // An explicit 0 means what it says: no layers on the GPU. It used
            // to be the "auto-fit" sentinel, which read as CPU but silently
            // ran full-GPU — that exact misreading produced vacuous
            // CPU-vs-GPU certification evidence (GPU compared with GPU).
            // Auto-fit is the default; omit the flag to get it.
            int n = (int)int_arg(a, NEXT, 0, 100000);
            if (n == 0) o->mp.gpu_mode = GPU_OFF;
            else o->mp.gpu_layers_override = n;
        }
        else if (!strcmp(a, "--cpu-moe")) {
            // Optional argument: a layer count or "auto". A bare --cpu-moe
            // keeps every expert FFN on the host (the original behaviour), so
            // the argument is only consumed when it is unmistakably ours —
            // every other CLI token starts with '-' or is a flag's operand.
            o->mp.cpu_moe = true;
            o->mp.cpu_moe_layers = CPU_MOE_ALL;
            const char *nx = i + 1 < argc ? argv[i + 1] : NULL;
            if (nx && !strcmp(nx, "auto")) {
                o->mp.cpu_moe_layers = CPU_MOE_AUTO;
                i++;
            } else if (nx && nx[0] >= '0' && nx[0] <= '9') {
                o->mp.cpu_moe_layers = (int)int_arg(a, argv[++i], 0, INT_MAX);
            }
        }
        else if (!strcmp(a, "--reserve-ram")) o->mp.reserve_ram_pct = (int)int_arg(a, NEXT, 0, 100);
        else if (!strcmp(a, "--reserve-cpu")) o->reserve_cpu_pct = (int)int_arg(a, NEXT, 0, 100);
        else if (!strcmp(a, "--wait-for-vram")) {
            // optional argument: a bare --wait-for-vram waits the default, and
            // anything that is not a number is left for the next iteration to
            // parse as its own flag
            o->mp.vram_wait_secs = 300;
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9')
                o->mp.vram_wait_secs = (int)int_arg(a, argv[++i], 1, 86400);
        }
        else if (!strcmp(a, "--vram-priority"))
            o->mp.vram_priority = (int)int_arg(a, NEXT, -1000, 1000);
        else if (!strcmp(a, "--yield-on-request")) o->mp.yield_on_request = true;
        else if (!strcmp(a, "--parent-pid")) o->parent_pid = (long)int_arg(a, NEXT, 1, LONG_MAX);
        else if (!strcmp(a, "--caps")) o->caps = true;
        // --shadow-mode: hand off to the Python client's `shadow install`
        // (confirmation, hooks, the /shadow skill and prompt, the model to
        // serve for offloading). The client ships beside the binary in the
        // release archive and in the source tree; nothing is downloaded.
        else if (!strcmp(a, "--shadow-mode")) o->shadow_mode = true;
        else if (!strcmp(a, "--yes")) o->shadow_yes = true;
        // Per-MODEL evidence: the native tool-call protocol resolved from this
        // model's chat template (a build property --caps cannot carry). Loads
        // the model, prints one JSON line, exits.
        else if (!strcmp(a, "--tool-info")) o->tool_info = true;
        else if (!strcmp(a, "--adapt-info")) o->adapt_info = true;
        else if (!strcmp(a, "--fit")) o->fit_path = NEXT;
        else if (!strcmp(a, "--version")) {
            printf("runner %s%s\n", RUNNER_VERSION, RUNNER_BUILD_FLAVOR ? " (" RUNNER_BUILD_FLAVOR_STR ")" : "");
            *rc = 0;
            return false;
        }
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage_to(stdout, argv[0]);
            *rc = 0;
            return false;
        }
        else { fprintf(stderr, "unknown option %s\n", a); usage(argv[0]); *rc = 1; return false; }
    }
    return true;
}

// These options shape render_messages() in the stateful chat loop below.
// A one-shot -p is deliberately raw, while server callers select thinking
// per request; accepting the process flags in either mode would promise a
// prompt change and then silently discard it. Validate before any model or
// fit-check I/O so the caller gets the actionable error they caused.
static bool cli_chat_flags_ok(const cli_opts *o) {
    if (o->system_prompt && (!o->interactive || o->serve)) {
        fprintf(stderr, "error: --system is only valid with interactive chat"
                        " (-i), without --serve\n");
        return false;
    }
    if (o->thinking != THINK_DEFAULT && (!o->interactive || o->serve)) {
        const char *flag = o->thinking == THINK_ON ? "--think" : "--no-think";
        fprintf(stderr, "error: %s is only valid with interactive chat (-i);"
                        " server requests use enable_thinking\n", flag);
        return false;
    }
    return true;
}

// The record and signing utilities need files, never a model, and answer
// before anything else on the command line is considered. True when one
// ran (or refused) and the process exits with *rc.
static bool run_record_utility(const cli_opts *o, int *rc) {
    if (o->sign_record) {
        if (!o->sign_key) {
            fprintf(stderr, "error: --sign-record needs --sign-key FILE\n");
            *rc = 1;
            return true;
        }
        *rc = record_sign(o->sign_record, o->sign_key, o->record_prev) ? 0 : 1;
        return true;
    }
    if (o->check_record) { *rc = record_check(o->check_record, o->trust_key); return true; }
    if (o->lineage_path) { *rc = lineage_walk(o->lineage_path, o->trust_key); return true; }
    if (o->check_bundle) { *rc = bundle_check(o->check_bundle, o->trust_key); return true; }
    if (o->check_pack) { *rc = pack_check(o->check_pack, o->trust_key); return true; }
    if (o->export_pack || o->pack_out) {
        if (!o->export_pack || !o->pack_out) {
            fprintf(stderr, "error: --export-pack RECEIPTS_DIR and --pack-out "
                    "DIR go together\n");
            *rc = 1;
            return true;
        }
        pack_opts po = { .receipts_dir = o->export_pack, .out_dir = o->pack_out,
                         .model_sig = o->model_sig, .model_pubkey = o->model_pubkey,
                         .envelope = o->pack_envelope, .n_attach = o->n_pack_attach,
                         .sign_key = o->sign_key };
        for (int i = 0; i < o->n_pack_attach; i++) po.attach[i] = o->pack_attach[i];
        *rc = pack_export(&po);
        return true;
    }
    if ((o->pack_envelope || o->n_pack_attach) && !o->export_pack) {
        fprintf(stderr, "error: --pack-envelope and --pack-attach go with "
                "--export-pack\n");
        *rc = 1;
        return true;
    }
    if (o->sign_model || o->model_key) {
        if (!o->sign_model || !o->model_key) {
            fprintf(stderr, "error: --sign-model MODEL and --model-key KEY.pem go "
                    "together\n");
            *rc = 1;
            return true;
        }
        char def[4096];
        const char *out = o->model_sig;
        if (!out) {
            if (strlen(o->sign_model) + 5 > sizeof def) {
                fprintf(stderr, "error: --sign-model: path too long\n");
                *rc = 1;
                return true;
            }
            snprintf(def, sizeof def, "%s.sig", o->sign_model);
            out = def;
        }
        oms_sign_info si;
        if (!oms_sign_file(o->sign_model, o->model_key, out, &si)) { *rc = 1; return true; }
        printf("signed: %s (%s/%s, %d file%s, subject sha256 %s, key hint %s)\n",
               out, si.curve, si.hash, si.n_resources,
               si.n_resources == 1 ? "" : "s", si.subject_digest, si.key_hint);
        *rc = 0;
        return true;
    }
    if (o->export_bundle || o->bundle_out) {
        if (!o->export_bundle || !o->bundle_out) {
            fprintf(stderr, "error: --export-bundle RECEIPT and --bundle-out "
                    "DIR go together\n");
            *rc = 1;
            return true;
        }
        bundle_opts bo = { .receipt = o->export_bundle, .out_dir = o->bundle_out,
                           .model_sig = o->model_sig, .model_pubkey = o->model_pubkey,
                           .sign_key = o->sign_key };
        *rc = bundle_export(&bo);
        return true;
    }
    if (o->watermark_keygen) {
        uint8_t key[32];
        char id[17];
        if (!os_random(key, sizeof key)) {
            fprintf(stderr, "error: --watermark-keygen: the OS random generator "
                    "is unavailable\n");
            *rc = 1;
            return true;
        }
        bool ok = wm_key_write(o->watermark_keygen, key, id);
        memset(key, 0, sizeof key);
        if (!ok) {
            fprintf(stderr, "error: --watermark-keygen: cannot create %s (an "
                    "existing key is never overwritten)\n", o->watermark_keygen);
            *rc = 1;
            return true;
        }
        printf("{\"schema_version\":\"xyntetik.runner.watermark_key.v1\","
               "\"key_id\":\"%s\"}\n", id);
        fprintf(stderr, "watermark key -> %s (keep it private; receipts name it "
                "by its id %s)\n", o->watermark_keygen, id);
        *rc = 0;
        return true;
    }
    return false;
}

// --watermark loads the key everything this process marks, replays or
// scores with. --detect-watermark over a transcript record is answered
// here from the record's own token ids; over plain text it needs the
// model's tokenizer, so *detect_text keeps the text for after the load.
// True when the process exits with *rc.
static bool cli_watermark_setup(const cli_opts *o, wm_key *wmk, bool *wm_on,
                                char **detect_text, int *rc) {
    if (o->watermark_path) {
        char werr[256];
        if (!wm_key_load(o->watermark_path, wmk, werr, sizeof werr)) {
            fprintf(stderr, "error: --watermark: %s\n", werr);
            *rc = 1;
            return true;
        }
        *wm_on = true;
    }
    if (o->detect_path) {
        if (!*wm_on) {
            fprintf(stderr, "error: --detect-watermark needs --watermark KEY\n");
            *rc = 1;
            return true;
        }
        size_t dn = 0;
        char *dbuf = read_file(o->detect_path, &dn);
        if (!dbuf) {
            fprintf(stderr, "error: --detect-watermark: cannot read %s\n", o->detect_path);
            *rc = 1;
            return true;
        }
        jv *dj = json_parse(dbuf, dn);
        if (dj && !strcmp(jv_str(jv_get(dj, "schema_version"), ""),
                          "xyntetik.runner.transcript.v1")) {
            // a record carries its own token ids and the prompt as context:
            // no tokenizer, no model
            int np = 0, no = 0;
            int32_t *pt = json_token_ids(jv_get(jv_get(dj, "prompt"), "tokens"), &np);
            int32_t *ot = json_token_ids(jv_get(jv_get(dj, "output"), "tokens"), &no);
            int32_t *all = pt && ot ? malloc(sizeof *all * (size_t)(np + no + 1)) : NULL;
            int drc = 3;
            if (!all) {
                fprintf(stderr, "error: --detect-watermark: %s has malformed token "
                        "arrays\n", o->detect_path);
            } else {
                memcpy(all, pt, sizeof *all * (size_t)np);
                memcpy(all + np, ot, sizeof *all * (size_t)no);
                jv *rw = jv_get(dj, "watermark");
                drc = wm_report(wmk, all, np + no, np, "record",
                                rw ? jv_str(jv_get(rw, "key_id"), "") : NULL);
            }
            free(pt); free(ot); free(all);
            jv_free(dj);
            free(dbuf);
            *rc = drc;
            return true;
        }
        jv_free(dj);
        if (!o->model_path) {
            free(dbuf);
            fprintf(stderr, "error: --detect-watermark: %s is not a transcript "
                    "record; scoring a text needs -m MODEL (its tokenizer)\n",
                    o->detect_path);
            *rc = 1;
            return true;
        }
        *detect_text = dbuf;   // tokenized once the model is loaded
    }
    return false;
}

// --keygen writes a signing key and prints its public half.
static int run_keygen(const cli_opts *o) {
    uint8_t seed[32];
    char pub[SIGN_PUBHEX_CAP];
    if (!sign_algo_known(o->keygen_algo)) {
        fprintf(stderr, "error: --keygen-algo: unknown algorithm %s (ed25519 or "
                "ml-dsa-44)\n", o->keygen_algo);
        return 1;
    }
    if (!os_random(seed, sizeof seed)) {
        fprintf(stderr, "error: --keygen: the OS random generator is unavailable\n");
        return 1;
    }
    if (!signkey_write(o->keygen_path, o->keygen_algo, seed, pub)) {
        fprintf(stderr, "error: --keygen: cannot write %s\n", o->keygen_path);
        return 1;
    }
    memset(seed, 0, sizeof seed);
    printf("{\"schema_version\":\"xyntetik.runner.signkey.v1\",\"algo\":\"%s\","
           "\"public_key\":\"%s\"}\n", o->keygen_algo, pub);
    fprintf(stderr, "signing key -> %s (keep it private; the %s public key is %s)\n",
            o->keygen_path, o->keygen_algo, pub);
    return 0;
}

// A refused draft drops to plain decoding by design, and local CLI modes
// have no machine-readable channel to say so: the note goes to stderr
// beside a zero exit, so a harness collecting stdout records the
// unaccelerated baseline and labels it speculative decoding.
// --draft-required makes that a failed run instead. Serve mode already
// answers the question over the wire, so the flag is refused there rather
// than accepted with no effect.
static bool cli_draft_flags_ok(const cli_opts *o) {
    if (o->draft_required && !o->draft_path) {
        fprintf(stderr, "error: --draft-required needs --draft PATH\n");
        return false;
    }
    if (o->draft_required && o->serve) {
        fprintf(stderr, "error: --draft-required is a local CLI guard; in serve"
                        " mode read draft.active from GET /v1/capabilities\n");
        return false;
    }
    // One draft source per run. The walk takes one proposal per round and
    // the counters attribute it to one source; a second source would either
    // be silently ignored (the accepted-then-ignored shape this project
    // refuses) or need a composition rule nobody has measured.
    if (o->draft_lookup && (o->draft_path || o->mtp_on)) {
        fprintf(stderr, "error: --draft-lookup: one draft source per run; drop "
                        "--draft/--mtp to draft from the context\n");
        return false;
    }
    return true;
}

// Where the model comes from (-m, or -hf fetched to a cached path), the
// prompt (-p, then -f appended into a buffer *owned_prompt owns), whether
// there is anything to do at all, and the seed. True when the process
// exits with *rc.
static bool cli_resolve_inputs(cli_opts *o, const char *prog, bool have_detect_text,
                               char **owned_prompt, int *rc) {
    // -hf resolves to a cached local file before anything reads model_path,
    // so every mode (-p, -i, --serve, --score, ...) sees a plain path
    if (o->hf_spec) {
        if (o->model_path) {
            fprintf(stderr, "error: use -m or -hf, not both\n");
            *rc = 1;
            return true;
        }
        char herr[1024];
        char *fetched = hf_fetch(o->hf_spec, herr, sizeof herr);
        if (!fetched) {
            fprintf(stderr, "error: -hf %s: %s\n", o->hf_spec, herr);
            *rc = 1;
            return true;
        }
        fprintf(stderr, "hf: %s -> %s\n", o->hf_spec, fetched);
        o->model_path = fetched;   // lives for the process
    }
    if (!o->model_path) { usage(prog); *rc = 1; return true; }
    if (o->prompt_file) {
        size_t flen = 0;
        char *fdata = read_file(o->prompt_file, &flen);
        if (!fdata) { fprintf(stderr, "error: cannot read %s\n", o->prompt_file); *rc = 1; return true; }
        size_t plen = o->prompt ? strlen(o->prompt) : 0;
        if (flen > SIZE_MAX - plen - 1) {   // checked concat length
            fprintf(stderr, "error: prompt is too large\n");
            free(fdata);
            *rc = 1;
            return true;
        }
        char *fbuf = malloc(plen + flen + 1);
        if (!fbuf) { free(fdata); fprintf(stderr, "error: out of memory\n"); *rc = 1; return true; }
        if (o->prompt) memcpy(fbuf, o->prompt, plen);
        memcpy(fbuf + plen, fdata, flen);
        fbuf[plen + flen] = 0;
        free(fdata);
        o->prompt = *owned_prompt = fbuf;
    }
    if (!o->prompt && !o->interactive && !o->serve && !o->quant_out && !o->merge_out &&
        !o->context_out &&
        !o->bench_json && !o->doctor && !o->tool_info && !o->adapt_info && !o->train_path && !o->dpo_path && !o->verify_path &&
        !o->decide_path && !have_detect_text && !o->resume_path) {
        fprintf(stderr, "error: need -p PROMPT, -i, or --serve\n");
        usage(prog);
        *rc = 1;
        return true;
    }
    // The sampler's xorshift64 has a fixed point at state 0: it stays 0 and
    // every draw comes back 0.0. So 0 is not a usable seed — but the old
    // `if (rng == 0) rng = time(...)` said that by silently substituting the
    // clock, which turned `-s 0`, a request for a REPRODUCIBLE run, into a
    // random one with nothing on stderr. Only an absent -s takes the clock
    // now; an explicit 0 is refused, because the one thing that must not
    // happen is running it as if it had been honoured.
    if (o->seed_given && o->smp.rng == 0) {
        fprintf(stderr, "error: -s 0 is not a usable seed — the sampler's RNG "
                        "state 0 is a fixed point. Use any other value.\n");
        *rc = 1;
        return true;
    }
    if (!o->seed_given) o->smp.rng = (uint64_t)time(NULL) ^ 0x9E3779B97F4A7C15ull;
    return false;
}

static bool cli_prepare_session(cli_opts *o, session_image *simg, int *rc) {
    // R1.3 session images: suspend (--suspend-after N with --session-out),
    // resume (--resume), fork (--resume with --fork-seed). They image the
    // host KV cache and the solo step loop, so they run on the CPU, without
    // speculation, and outside serve / interactive / verify.
    if (o->session_out || o->suspend_after || o->resume_path || o->fork_seed_given) {
        const char *why = NULL;
        if (o->suspend_after && !o->session_out)
            why = "--suspend-after N needs --session-out FILE";
        else if (o->fork_seed_given && !o->resume_path)
            why = "--fork-seed forks a session and needs --resume FILE";
        else if (o->resume_path && o->prompt)
            why = "--resume continues an image's prompt; give no -p";
        else if (o->serve || o->interactive || o->verify_path || o->transcript_path)
            why = "session images are for one-shot runs (not --serve, -i, "
                  "--verify or --transcript)";
        else if (o->draft_path || o->mtp_on || o->draft_lookup)
            why = "session images run the solo step loop: no --draft, --mtp or "
                  "--draft-lookup";
        else if (o->watermark_path)
            why = "an image does not carry a watermark key, so a resume could not "
                  "continue a marked generation: no --watermark";
        else if (!o->resume_path && o->n_predict < 1)
            why = "session images need a finite generation budget (-n N, N >= 1)";
        else if (!o->resume_path && o->suspend_after >= o->n_predict)
            why = "--suspend-after N must stop before the -n budget is spent";
        if (!why && o->session_out) {
            // refused before the load and the generation, not after them;
            // session_write's O_EXCL still closes the race
            FILE *ex = fopen(o->session_out, "rb");
            if (ex) {
                fclose(ex);
                fprintf(stderr, "error: --session-out %s already exists; an "
                        "image is never overwritten\n", o->session_out);
                *rc = 1;
                return true;
            }
        }
        if (why) { fprintf(stderr, "error: %s\n", why); *rc = 1; return true; }
        o->mp.gpu_mode = GPU_OFF;
    }
    if (o->resume_path) {
        // the image fixes everything that shapes the continuation; a flag
        // that would change it is refused, not silently overridden
        const char *fixed = o->seed_given ? "-s" : o->n_given ? "-n" : o->mp.n_ctx ? "-c"
            : o->kv_given ? "--kv"
            : o->ov.has_temp ? "--temp" : o->ov.has_top_k ? "--top-k"
            : o->ov.has_top_p ? "--top-p" : o->ov.has_min_p ? "--min-p"
            : o->ov.has_repeat_penalty ? "--repeat-penalty" : NULL;
        if (fixed) {
            fprintf(stderr, "error: --resume continues under the image's own "
                    "settings; %s would change them%s\n", fixed,
                    o->seed_given ? " (a different continuation is --fork-seed N)" : "");
            *rc = 1;
            return true;
        }
        if (!session_read(o->resume_path, simg)) { *rc = 1; return true; }
        const session_meta *sm = &simg->meta;
        // the image's shape and sampler, as a replay takes a record's
        o->mp.n_ctx = sm->n_ctx;
        o->mp.kv_q8 = !strcmp(sm->kv_type, "q8");
        o->mp.kv_fp4 = !strcmp(sm->kv_type, "fp4");
        o->mp.kv_split = !strcmp(sm->kv_type, "k8v4");
        o->ov.temp = sm->temp;                     o->ov.has_temp = true;
        o->ov.top_k = sm->top_k;                   o->ov.has_top_k = true;
        o->ov.top_p = sm->top_p;                   o->ov.has_top_p = true;
        o->ov.min_p = sm->min_p;                   o->ov.has_min_p = true;
        o->ov.repeat_penalty = sm->repeat_penalty; o->ov.has_repeat_penalty = true;
        if (o->fork_seed_given && o->fork_seed == 0) {
            fprintf(stderr, "error: --fork-seed must be nonzero\n");
            *rc = 1;
            return true;
        }
        if (!o->fork_seed_given && sm->rng == 0 && sm->temp > 0) {
            fprintf(stderr, "error: --resume %s: a sampled image without an rng "
                    "state\n", o->resume_path);
            *rc = 1;
            return true;
        }
        // a greedy image records no rng (it is never drawn from); the sampler
        // still wants a nonzero state
        o->smp.rng = o->fork_seed_given ? o->fork_seed : sm->rng ? sm->rng : 1;
        if ((o->json_mode && !sm->json_mode) || (o->ignore_eos && !sm->ignore_eos)) {
            fprintf(stderr, "error: --resume: the image was not generated under "
                    "%s\n", o->json_mode && !sm->json_mode ? "--json" : "--ignore-eos");
            session_image_free(simg);
            *rc = 1;
            return true;
        }
        o->json_mode = sm->json_mode;
        o->ignore_eos = sm->ignore_eos;
        o->n_predict = sm->max_new;
        if (sm->schema_sha256[0]) {
            char have[65];
            if (!o->schema_file || !schema_file_digest(o->schema_file, have) ||
                strcmp(have, sm->schema_sha256) != 0) {
                fprintf(stderr, "error: --resume: the image was generated under a "
                        "JSON schema (sha256 %s); give --json-schema naming that "
                        "schema\n", sm->schema_sha256);
                *rc = 1;
                return true;
            }
        } else if (o->schema_file) {
            fprintf(stderr, "error: --resume: the image was not generated under "
                    "a schema; --json-schema would change what it continues\n");
            *rc = 1;
            return true;
        }
        if (o->suspend_after && o->suspend_after <= sm->generated) {
            fprintf(stderr, "error: --suspend-after %d: the image has already "
                    "generated %d tokens\n", o->suspend_after, sm->generated);
            *rc = 1;
            return true;
        }
        if (o->suspend_after >= sm->max_new) {
            fprintf(stderr, "error: --suspend-after %d must stop before the "
                    "image's budget of %d tokens is spent\n", o->suspend_after,
                    sm->max_new);
            *rc = 1;
            return true;
        }
    }

    return false;
}

// ---- notarized inference D2: parse the transcript BEFORE the load so
// the record's profile (ctx, kv, gpu) and config (sampler, seed) shape
// the replay exactly as they shaped the original. Integrity first: the
// chain hash must re-verify from the raw bytes, or nothing else in the
// file is worth believing (UNVERIFIABLE, exit 3).
//
// True when the record is refused and the process exits with *rc.
static bool cli_prepare_verify(cli_opts *o, const wm_key *wmk, bool wm_on,
                               cli_verify *vr, int *rc) {
    if (o->verify_path) {
        size_t vlen = 0;
        char *vbuf = read_file(o->verify_path, &vlen);
        if (!vbuf) {
            fprintf(stderr, "UNVERIFIABLE: cannot read %s\n", o->verify_path);
            *rc = 3;
            return true;
        }
        char *cpos = strstr(vbuf, ",\"chain\"");
        char *clast = cpos;
        while (clast && (clast = strstr(clast + 1, ",\"chain\"")) != NULL) cpos = clast;
        if (!cpos) { fprintf(stderr, "UNVERIFIABLE: no chain in record\n"); free(vbuf); *rc = 3; return true; }
        char want[65];
        envelope_data_sha256(vbuf, (size_t)(cpos - vbuf), want);
        vr->rsig = receipt_signature_check(vbuf, vlen, vr->rec_pub);
        vr->rec = json_parse(vbuf, vlen);
        free(vbuf);
        if (!vr->rec) { fprintf(stderr, "UNVERIFIABLE: malformed record\n"); *rc = 3; return true; }
        const char *sv = jv_str(jv_get(vr->rec, "schema_version"), "");
        const char *ch = jv_str(jv_get(jv_get(vr->rec, "chain"), "hash"), "");
        if (strcmp(sv, "xyntetik.runner.transcript.v1") != 0) {
            fprintf(stderr, "UNVERIFIABLE: schema %s\n", sv); jv_free(vr->rec); *rc = 3; return true;
        }
        if (strcmp(ch, want) != 0) {
            fprintf(stderr, "UNVERIFIABLE: chain hash mismatch (record "
                    "altered after writing)\n");
            jv_free(vr->rec); *rc = 3; return true;
        }
        // Signature and chain link are checked before any replay: a forged
        // or unsigned record must not even get the model loaded for it.
        if (vr->rsig == RSIG_BAD || vr->rsig == RSIG_MALFORMED) {
            fprintf(stderr, "UNVERIFIABLE: %s\n", vr->rsig == RSIG_BAD
                    ? "signature does not verify (record altered or forged)"
                    : "malformed signature object");
            jv_free(vr->rec); *rc = 3; return true;
        }
        if (o->require_signed && vr->rsig != RSIG_OK) {
            fprintf(stderr, "UNVERIFIABLE: --require-signed: the record carries "
                    "no signature\n");
            jv_free(vr->rec); *rc = 3; return true;
        }
        // --trust-key takes the public key as hex, or "sha256:" + the hex
        // SHA-256 of the key bytes (an ML-DSA-44 key is 2624 hex characters;
        // its digest is what fits in a policy file or a command line)
        bool trusted = vr->rsig == RSIG_OK;
        if (trusted && o->trust_key) {
            if (strncmp(o->trust_key, "sha256:", 7) == 0) {
                uint8_t pkb[SIGN_PK_MAX];
                size_t hn = strlen(vr->rec_pub);
                char digest[65] = "";
                for (size_t i = 0; i < hn / 2; i++) {
                    unsigned v = 0;
                    sscanf(vr->rec_pub + 2 * i, "%2x", &v);
                    pkb[i] = (uint8_t)v;
                }
                envelope_data_sha256(pkb, hn / 2, digest);
                trusted = hex_eq_nocase(o->trust_key + 7, digest);
            } else {
                trusted = hex_eq_nocase(o->trust_key, vr->rec_pub);
            }
        }
        if (o->trust_key && !trusted) {
            fprintf(stderr, "UNVERIFIABLE: --trust-key: the record is %s\n",
                    vr->rsig == RSIG_OK ? "signed by a different key" : "unsigned");
            jv_free(vr->rec); *rc = 3; return true;
        }
        snprintf(vr->prev, sizeof vr->prev, "%s",
                 jv_str(jv_get(jv_get(vr->rec, "chain"), "prev"), ""));
        if (o->transcript_prev) {
            char ph[65];
            if (!receipt_chain_hash(o->transcript_prev, ph)) {
                fprintf(stderr, "UNVERIFIABLE: cannot read the chain hash of %s\n",
                        o->transcript_prev);
                jv_free(vr->rec); *rc = 3; return true;
            }
            if (strcmp(ph, vr->prev) != 0) {
                fprintf(stderr, "UNVERIFIABLE: chain link broken: the record's prev "
                        "is not the chain hash of %s\n", o->transcript_prev);
                jv_free(vr->rec); *rc = 3; return true;
            }
        }
        {
            // R1.8.2: a marked record replays only under its own key, and an
            // unmarked one never under a key
            jv *vw = jv_get(vr->rec, "watermark");
            const char *rk = vw ? jv_str(jv_get(vw, "key_id"), "") : NULL;
            const char *why = NULL;
            char wbuf[256];
            if (rk && strcmp(jv_str(jv_get(vw, "scheme"), ""), WM_SCHEME) != 0) {
                snprintf(wbuf, sizeof wbuf, "the output was watermarked with scheme "
                         "\"%.40s\", which this build does not replay",
                         jv_str(jv_get(vw, "scheme"), ""));
                why = wbuf;
            } else if (rk && !wm_on) {
                snprintf(wbuf, sizeof wbuf, "the output was watermarked (key id "
                         "%.16s); replaying it needs that key: verify with "
                         "--watermark naming it", rk);
                why = wbuf;
            } else if (rk && strcmp(rk, wmk->id) != 0) {
                snprintf(wbuf, sizeof wbuf, "--watermark key id %s is not the "
                         "record's watermark key id (%.16s)", wmk->id, rk);
                why = wbuf;
            } else if (!rk && wm_on) {
                why = "--watermark given, but the record's output was not "
                      "watermarked";
            }
            if (why) {
                fprintf(stderr, "UNVERIFIABLE: %s\n", why);
                jv_free(vr->rec);
                *rc = 3;
                return true;
            }
        }
        {
            char why[512];
            if (!replay_constraints_ok(vr->rec, o->json_mode, o->schema_file, o->ignore_eos,
                                       why, sizeof why)) {
                fprintf(stderr, "UNVERIFIABLE: %s\n", why);
                jv_free(vr->rec);
                *rc = 3;
                return true;
            }
        }
        jv *prof = jv_get(vr->rec, "profile"), *cfg = jv_get(vr->rec, "config");
        jv *ctx_v = jv_get(prof, "ctx"), *batch_v = jv_get(prof, "batch");
        jv *threads_v = jv_get(prof, "threads"), *gpu_v = jv_get(prof, "gpu");
        jv *gpu_layers_v = jv_get(prof, "gpu_layers");
        const char *kv = jv_str(jv_get(prof, "kv"), NULL);
        double ctx_d = jv_num(ctx_v, -1), batch_d = jv_num(batch_v, -1);
        double threads_d = jv_num(threads_v, -1);
        double gpu_layers_d = jv_num(gpu_layers_v, -1);
        if (!prof || prof->type != J_OBJ || !cfg || cfg->type != J_OBJ ||
            !ctx_v || ctx_v->type != J_NUM || ctx_d < 1 || ctx_d > INT_MAX ||
            ctx_d != (double)(int)ctx_d || !batch_v || batch_v->type != J_NUM ||
            batch_d < 1 || batch_d > ctx_d || batch_d != (double)(int)batch_d ||
            !threads_v || threads_v->type != J_NUM || threads_d < 1 ||
            threads_d > INT_MAX || threads_d != (double)(int)threads_d ||
            !gpu_v || gpu_v->type != J_BOOL || !kv ||
            (gpu_layers_v && (gpu_layers_v->type != J_NUM ||
             gpu_layers_d < 0 || gpu_layers_d > INT_MAX ||
             gpu_layers_d != (double)(int)gpu_layers_d ||
             (gpu_v->b != (gpu_layers_d > 0)))) ||
            (strcmp(kv, "f16") != 0 && strcmp(kv, "q8") != 0 &&
             strcmp(kv, "fp4") != 0 && strcmp(kv, "k8v4") != 0)) {
            fprintf(stderr, "UNVERIFIABLE: malformed execution profile\n");
            jv_free(vr->rec);
            *rc = 3;
            return true;
        }
        o->mp.n_ctx = (int)ctx_d;
        o->mp.n_batch = (int)batch_d;
        o->n_threads = (int)threads_d;
        o->mp.kv_q8 = strcmp(kv, "q8") == 0;
        o->mp.kv_fp4 = strcmp(kv, "fp4") == 0;
        o->mp.kv_split = strcmp(kv, "k8v4") == 0;
        o->mp.gpu_mode = gpu_v->b ? GPU_AUTO : GPU_OFF;
        if (gpu_layers_v && gpu_layers_d > 0)
            o->mp.gpu_layers_override = (int)gpu_layers_d;
        jv *record_adapter = jv_get(vr->rec, "adapter");
        if (record_adapter && record_adapter->type != J_NULL) {
            jv *scale_v = record_adapter->type == J_OBJ
                            ? jv_get(record_adapter, "scale") : NULL;
            double scale = jv_num(scale_v, -1);
            if (!scale_v || scale_v->type != J_NUM || scale < 0 ||
                scale > FLT_MAX) {
                fprintf(stderr, "UNVERIFIABLE: malformed adapter scale\n");
                jv_free(vr->rec);
                *rc = 3;
                return true;
            }
            o->lora_scale = (float)scale;
        } else if (!record_adapter) {
            fprintf(stderr, "UNVERIFIABLE: missing adapter profile\n");
            jv_free(vr->rec);
            *rc = 3;
            return true;
        }
        // New v1 records carry an exact decimal spelling because this JSON
        // parser stores numbers as doubles.  Keep low legacy seeds working,
        // but refuse to round an older wide seed and call the later mismatch
        // inference divergence: that record is not exactly replayable.
        jv *seed_exact = jv_get(cfg, "seed_u64");
        uint64_t replay_seed = 0;
        bool seed_ok = false;
        if (seed_exact && seed_exact->type == J_STR) {
            seed_ok = parse_u64(seed_exact->str, 1, UINT64_MAX, &replay_seed);
        } else {
            double seed_num = jv_num(jv_get(cfg, "seed"), 0);
            if (seed_num >= 1.0 && seed_num <= 9007199254740992.0) {
                replay_seed = (uint64_t)seed_num;
                seed_ok = seed_num == (double)replay_seed;
            }
        }
        if (!seed_ok) {
            fprintf(stderr, "UNVERIFIABLE: config.seed is not an exact "
                    "nonzero uint64 (wide legacy seeds need seed_u64)\n");
            jv_free(vr->rec);
            *rc = 3;
            return true;
        }
        o->smp.rng = replay_seed;
        // The record's sampling config rides the CLI-override mechanism so
        // it survives sampler_resolve's per-model presets. Apply the same
        // finite/range/integrality rules as the CLI before any narrowing.
        double temp, top_p, min_p, repeat_penalty;
        int top_k, replay_n_predict;
        if (!record_num(cfg, "temp", 0, FLT_MAX, &temp) ||
            !record_int(cfg, "top_k", 0, INT_MAX, &top_k) ||
            !record_num(cfg, "top_p", 0, 1, &top_p) ||
            !record_num(cfg, "min_p", 0, 1, &min_p) ||
            !record_num(cfg, "repeat_penalty", FLT_MIN, FLT_MAX,
                        &repeat_penalty) ||
            !record_int(cfg, "n_predict", -1, INT_MAX,
                        &replay_n_predict)) {
            fprintf(stderr, "UNVERIFIABLE: malformed sampling config\n");
            jv_free(vr->rec);
            *rc = 3;
            return true;
        }
        o->ov.temp = (float)temp;                         o->ov.has_temp = true;
        o->ov.top_k = top_k;                              o->ov.has_top_k = true;
        o->ov.top_p = (float)top_p;                       o->ov.has_top_p = true;
        o->ov.min_p = (float)min_p;                       o->ov.has_min_p = true;
        o->ov.repeat_penalty = (float)repeat_penalty;
        o->ov.has_repeat_penalty = true;
        o->n_predict = replay_n_predict;
    }

    return false;
}

// The thread count: -t, else a share of the cores for --reserve-cpu (set
// earlier), else the platform's default; the model's own thread settings
// follow from it.
static void cli_resolve_threads(cli_opts *o) {
    if (o->n_threads <= 0) {
        // Default to a physical-core proxy (nc/2 under the near-universal 2-way
        // SMT of the many-core x86 boxes this targets), capped so a very large
        // box does not over-spawn. SMT siblings add nothing to a compute-bound
        // decode — measured: throughput plateaus at physical cores and SMT gives
        // no gain. The old min(8,nc) default left big boxes badly underused
        // (measured 32.2s -> 9.9s, a 3.2x speedup, at -t 64 vs -t 8 on a 3B Q4,
        // 64-core box; token-identical across thread counts). Scope a shared box
        // with --reserve-cpu, or pin exactly with -t. On Apple Silicon the
        // platform layer returns the performance-core count instead of the
        // logical/2 proxy, because asymmetric cores are not SMT siblings.
        o->n_threads = plat_default_thread_count();
        // If the model turns out to be CPU-forced (recurrent qwen3.5 path),
        // model_load may raise a defaulted count to this cap; a pinned -t
        // or CPU reservation leaves it 0 and is never overridden.
        int cap = plat_default_thread_count();
        o->mp.cpu_fallback_threads = cap > o->n_threads ? cap : o->n_threads;
    }
    o->mp.verbose = o->verbose;
    o->mp.n_threads = o->serve ? 1 : o->n_threads; // server slots create their own pools
}

// Flags about the model that are refused before it is opened: a YaRN
// factor that is not one, or one beside --rope-scale, and an unknown
// --chat-template, which *tmpl_override receives as a TMPL_* id.
static bool cli_model_flags_ok(const cli_opts *o, int *tmpl_override) {
    if (o->mp.yarn_factor > 0 && o->mp.yarn_factor <= 1.0f) {
        fprintf(stderr, "error: --yarn-factor must be greater than 1\n");
        return false;
    }
    if (o->mp.yarn_factor > 0 && o->mp.rope_scale > 0) {
        fprintf(stderr, "error: --yarn-factor and --rope-scale cannot be "
                        "used together\n");
        return false;
    }

    // --chat-template resolves HERE, not down in the interactive-chat block.
    // It used to resolve only there, and server mode returns above it, so a
    // documented public flag was accepted on the command line and silently
    // discarded on the path most callers take: forcing a template under
    // --serve produced a byte-identical prompt to auto-detection.
    if (o->tmpl_arg) {
        *tmpl_override = template_from_name(o->tmpl_arg);
        if (*tmpl_override < 0) {
            // previously this fell back to auto-detection without a word,
            // which is the same discard in a smaller place
            fprintf(stderr, "error: unknown --chat-template '%s'; see --help "
                            "for the accepted names\n", o->tmpl_arg);
            return false;
        }
        // A multi-model swap set cannot honour one global template; server_run
        // refuses that combination, where the parsed entry count is known.
    }
    return true;
}

// The tray follows a session a person will sit with — a server or an
// interactive chat — and is left running afterwards so the next model can
// be loaded from it. One-shot -p runs and tooling modes (--caps,
// --quantize, --bench-json) deliberately raise nothing: a two-second
// process should not leave a menu-bar icon behind it.
//
// A terminal on EITHER stdin or stdout counts as "a person launched this".
// Requiring both, the way the bare-invocation guard above does, would drop
// the most ordinary case there is — `runner --serve > server.log` — while
// CI and pipes typically have neither and stay unaffected. --no-tray is the
// explicit escape for a wrapper that wants the guarantee in writing.
//
// main() calls it after every argument check a serve or interactive run can
// still fail, the serving configuration, the rewrite flags and the eval
// requirement included, rather than up beside the mode guard where it used to.
// Raised earlier it outlived the process that raised it: `-s 0`, a
// yarn-factor conflict and an unknown --chat-template all refuse AFTER
// that point, and the detached tray they had already spawned held the
// executable open against the next relink. That is the "two-second
// process" the paragraph above says must not leave an icon behind, and it
// was leaving one. The bound stops at argument checks on purpose: a run
// that fails LATER, when a model or a draft will not load, still raises
// one, because moving past that would delay the icon by a whole model
// load for every run that works (suite plan R8.7.6).
static void cli_tray_follow(const cli_opts *o) {
#if defined(__APPLE__) || defined(_WIN32)
    if (!o->no_tray && (o->serve || o->interactive) &&
        (RUNNER_TTY(stdin) || RUNNER_TTY(stdout)))
        tray_ensure_running();
#else
    (void)o->no_tray;
#endif
}

// --context-surgery: rewrite the file's context metadata for a longer
// context under YaRN, and write its provenance record.
static int run_context_surgery(const cli_opts *o, const char *load_path) {
    if (o->quant_out || o->merge_out || o->prune_experts || o->type_plan || o->quant_type ||
        o->remove_sublayer) {
        fprintf(stderr, "error: --context-surgery does not combine with "
                "quantize, merge, prune, or type-plan operations\n");
        return 1;
    }
    if (o->mp.n_ctx <= 0 || o->mp.yarn_factor <= 1.0f) {
        fprintf(stderr, "error: --context-surgery requires -c TARGET and "
                "--yarn-factor FACTOR\n");
        return 1;
    }
    context_surgery_result sr;
    int rc = context_surgery_gguf(load_path, o->context_out,
                                  (uint32_t)o->mp.n_ctx, o->mp.yarn_factor, &sr);
    if (rc != 0) return rc;
    char bsha[65] = "", osha[65] = "";
    if (!envelope_file_sha256(load_path, bsha) ||
        !envelope_file_sha256(o->context_out, osha)) {
        fprintf(stderr, "error: cannot hash context-surgery artifacts "
                "for the provenance record\n");
        return 1;
    }
    sbuf rec = {0};
    sb_lit(&rec, "{\"schema_version\":\"xyntetik.runner.context-surgery.v1\","
                 "\"runner\":\"");
    sb_esc(&rec, RUNNER_VERSION, strlen(RUNNER_VERSION));
    sb_lit(&rec, "\",\"base\":");
    lineage_put_input(&rec, load_path, bsha);
    sb_fmt(&rec, ",\"source\":{\"context_length\":%u,"
                   "\"original_context_length\":%u,\"yarn_factor\":%.9g},",
           sr.source_context, sr.original_context,
           (double)sr.source_factor);
    sb_fmt(&rec, "\"target\":{\"path\":\"");
    sb_esc(&rec, o->context_out, strlen(o->context_out));
    sb_fmt(&rec, "\",\"sha256\":\"%s\",\"context_length\":%d,"
                   "\"original_context_length\":%u,\"yarn_factor\":%.9g},",
           osha, o->mp.n_ctx, sr.original_context, (double)o->mp.yarn_factor);
    sb_fmt(&rec, "\"tensor_payloads_byte_identical\":%s}\n",
           sr.tensors_byte_identical ? "true" : "false");
    bool wrote = write_record(o->context_out, ".context.json",
                              "context-surgery", &rec, o->sign_key);
    free(rec.s);
    return wrote ? 0 : 1;
}

// --merge-lora: fold an adapter into the base weights at the requested
// precision, refused when the output grid would erase the delta, and write
// the provenance record with what survived.
static int run_merge_lora(const cli_opts *o) {
    if (!o->lora_path) {
        fprintf(stderr, "error: --merge-lora requires --lora ADAPTER\n");
        return 1;
    }
    if (o->quant_out || o->prune_experts || o->type_plan || o->remove_sublayer) {
        fprintf(stderr, "error: --merge-lora does not combine with "
                "--quantize, --prune-experts or --type-plan\n");
        return 1;
    }
    // default keep: the merged artifact mirrors the base's own
    // per-tensor precision unless a different output type is asked for
    int tt = o->quant_type ? quantize_type_from_name(o->quant_type) : T_KEEP;
    if (tt == -2) {
        fprintf(stderr, "error: --quant '%s' is not a writable type\n",
                o->quant_type);
        return 1;
    }
    merge_survival sv = {0};
    int rc = merge_lora_gguf(o->model_path, o->lora_path, o->lora_scale,
                             o->merge_out, tt,
                             o->merge_allow_erased ? -1.0f : MERGE_MIN_RETAINED,
                             &sv);
    if (rc == 0) {
        // the D7 discipline extended to merged artifacts: the standalone
        // blob stays auditable — base sha + adapter sha + config ->
        // merged sha, machine-written beside the file
        char bsha[65] = "", asha[65] = "", osha[65] = "";
        if (!envelope_file_sha256(o->model_path, bsha) ||
            !envelope_file_sha256(o->lora_path, asha) ||
            !envelope_file_sha256(o->merge_out, osha)) {
            fprintf(stderr, "error: cannot hash every merge artifact for "
                    "the provenance record\n");
            return 1;
        }
        sbuf rec = {0};
        sb_lit(&rec, "{\"schema_version\":\"xyntetik.runner.merge.v1\","
                     "\"runner\":\"");
        sb_esc(&rec, RUNNER_VERSION, strlen(RUNNER_VERSION));
        sb_lit(&rec, "\",\"base\":");
        lineage_put_input(&rec, o->model_path, bsha);
        sb_lit(&rec, ",\"adapter\":");
        lineage_put_input(&rec, o->lora_path, asha);
        sb_fmt(&rec, ",\"lora_scale\":%g,\"target\":\"",
               (double)o->lora_scale);
        const char *target = o->quant_type ? o->quant_type : "keep";
        sb_esc(&rec, target, strlen(target));
        sb_lit(&rec, "\",\"merged\":{\"path\":\"");
        sb_esc(&rec, o->merge_out, strlen(o->merge_out));
        sb_lit(&rec, "\",\"sha256\":\""); sb_lit(&rec, osha);
        sb_lit(&rec, "\"}");
        // what the output grid kept of the delta (quants.h
        // merge_survival); absent when no adapted tensor was rewritten
        if (sv.ret_den > 0)
            sb_fmt(&rec, ",\"survival\":{\"adapted_bytes\":%llu,"
                   "\"bytes_changed\":%llu,\"delta_retained\":%.6f,"
                   "\"min_retained\":%s}",
                   (unsigned long long)sv.bytes,
                   (unsigned long long)sv.bytes_changed,
                   sv.ret_num / sv.ret_den,
                   o->merge_allow_erased ? "null" : MERGE_MIN_RETAINED_STR);
        sb_lit(&rec, "}\n");
        if (!write_record(o->merge_out, ".merge.json", "merge", &rec, o->sign_key))
            rc = 1;
        free(rec.s);
    }
    return rc;
}

// --quantize, with --quant, --type-plan, --prune-experts or
// --remove-sublayer: rewrite the file and write its provenance record.
static int run_quantize(const cli_opts *o) {
    int tt;
    // --type-plan decides the type PER TENSOR, so a whole-file --quant
    // alongside it is an instruction with nowhere to go: the plan branch
    // in quantize_gguf_plan never reads it, and the summary line then
    // named a type nothing had been written in. Refused rather than
    // dropped, the same call --merge-lora + --type-plan already makes.
    if (o->type_plan_strict && !o->type_plan) {
        fprintf(stderr, "error: --type-plan-strict needs --type-plan\n");
        return 1;
    }
    quantize_set_type_plan_strict(o->type_plan_strict);
    if (o->quant_type && o->type_plan) {
        fprintf(stderr, "error: --quant and --type-plan cannot be used "
                "together: the plan decides every tensor's type, so "
                "--quant would be silently ignored. Put the whole-file "
                "type in the plan's \"default\".\n");
        return 1;
    }
    if (o->quant_type) {
        tt = quantize_type_from_name(o->quant_type);
        if (tt == -2) {
            fprintf(stderr, "error: --quant '%s' is not a writable "
                    "type\n", o->quant_type);
            return 1;
        }
    } else {
        // No --quant given: --prune-experts alone means "just prune,
        // don't touch anyone's precision" (T_KEEP); with neither flag
        // this is plain --quantize, whose long-standing default is q4_0.
        tt = (o->prune_experts || o->remove_sublayer) ? T_KEEP : T_Q4_0;
    }
    int qrc = quantize_gguf_plan(o->model_path, o->quant_out, tt, o->prune_experts,
                                 o->type_plan, o->remove_sublayer);
    if (qrc != 0) return qrc;
    // R17.1: the rewrite leaves its record like merge and training do --
    // the input and output by sha256, every option that shaped the
    // bytes, and the building binary -- so the file's lineage can be
    // walked back (--lineage)
    char bsha[65] = "", osha[65] = "", xsha[65] = "";
    char *exe_path = plat_executable_path();
    bool hash_ok = envelope_file_sha256(o->model_path, bsha) &&
                   envelope_file_sha256(o->quant_out, osha) &&
                   exe_path && envelope_file_sha256(exe_path, xsha);
    free(exe_path);
    if (!hash_ok) {
        fprintf(stderr, "error: cannot hash the rewrite's files for the "
                "provenance record\n");
        return 1;
    }
    const char *target = o->quant_type ? o->quant_type
                       : o->type_plan ? "plan"
                       : (o->prune_experts || o->remove_sublayer) ? "keep" : "q4_0";
    sbuf rec = {0};
    sb_lit(&rec, "{\"schema_version\":\"xyntetik.runner.quant.v1\","
                 "\"runner\":\"");
    sb_esc(&rec, RUNNER_VERSION, strlen(RUNNER_VERSION));
    sb_lit(&rec, "\",\"build\":{\"binary_sha256\":\"");
    sb_lit(&rec, xsha);
    sb_lit(&rec, "\"},\"base\":");
    lineage_put_input(&rec, o->model_path, bsha);
    sb_lit(&rec, ",\"target\":\"");
    sb_esc(&rec, target, strlen(target));
    sb_lit(&rec, "\"");
    if (o->prune_experts) {
        sb_lit(&rec, ",\"prune_experts\":");
        lineage_put_input(&rec, o->prune_experts, NULL);
    }
    if (o->type_plan) {
        sb_lit(&rec, ",\"type_plan\":");
        lineage_put_input(&rec, o->type_plan, NULL);
    }
    if (o->remove_sublayer) {
        sb_lit(&rec, ",\"remove_sublayer\":\"");
        sb_esc(&rec, o->remove_sublayer, strlen(o->remove_sublayer));
        sb_lit(&rec, "\"");
    }
    sb_lit(&rec, ",\"output\":{\"path\":\"");
    sb_esc(&rec, o->quant_out, strlen(o->quant_out));
    sb_lit(&rec, "\",\"sha256\":\""); sb_lit(&rec, osha);
    sb_lit(&rec, "\"}}\n");
    bool wrote = write_record(o->quant_out, ".quant.json", "quantize", &rec, o->sign_key);
    free(rec.s);
    return wrote ? 0 : 1;
}

// Serve-only flags refused without --serve, and the receipt, KV snapshot
// and session stores configured when they are given.
static bool cli_configure_serving(const cli_opts *o) {
    if (o->receipts_dir && !o->serve) {
        fprintf(stderr, "error: --receipts writes one receipt per served "
                "request and needs --serve (use --transcript for a one-shot "
                "run)\n");
        return false;
    }
    if (o->receipts_keep && !o->receipts_dir) {
        fprintf(stderr, "error: --receipts-keep needs --receipts DIR\n");
        return false;
    }
    if (o->receipts_dir) {
        receipts_cfg rcfg = { .dir = o->receipts_dir, .keep = o->receipts_keep,
                              .sign_key = o->sign_key };
        if (!receipts_configure(&rcfg)) return false;
    }
    if (o->kv_snapshots && !o->serve) {
        fprintf(stderr, "error: --kv-snapshots saves and loads named contexts "
                "and needs --serve\n");
        return false;
    }
    if (o->kv_snapshots && !kvsnap_configure(o->kv_snapshots, o->sign_key, o->trust_key)) return false;
    if (o->sessions_path && !o->serve) {
        fprintf(stderr, "error: --sessions keeps the server's session images "
                "and needs --serve (the CLI uses --session-out / --resume)\n");
        return false;
    }
    if (o->sessions_path && !sessions_configure(o->sessions_path)) return false;
    if (o->n_adapters && !o->serve) {
        fprintf(stderr, "error: --adapter routes adapters per request and "
                "needs --serve (use --lora for a one-shot run)\n");
        return false;
    }
    if (o->n_adapters && o->lora_path) {
        fprintf(stderr, "error: --adapter and --lora are exclusive: --lora "
                "installs one adapter for every request; serve it as --adapter "
                "NAME=PATH and name it per request instead\n");
        return false;
    }
    return true;
}

// Every flag that only has meaning while rewriting a file. Reaching here
// means --quantize was not given, and the block above is their only
// reader, so any of them is an instruction that would otherwise be
// silently dropped — the same discard --chat-template was fixed for, in a
// place where the user believes they asked for a different file on disk.
static bool cli_rewrite_flags_ok(const cli_opts *o) {
    const char *orphan = o->prune_experts ? "--prune-experts"
                       : o->remove_sublayer ? "--remove-sublayer"
                       : o->type_plan     ? "--type-plan"
                       : o->quant_type    ? "--quant"
                       : NULL;
    if (orphan) {
        fprintf(stderr, "error: %s requires --quantize OUT (or "
                "--merge-lora OUT for --quant)\n", orphan);
        return false;
    }
    if (o->merge_allow_erased) {
        fprintf(stderr, "error: --merge-allow-erased requires --merge-lora OUT\n");
        return false;
    }
    return true;
}

// R17.2: refuse a model that carries no passing evaluation of the asked
// kind, made for this very file (scripts/eval-record.py writes them)
static bool cli_require_eval_ok(const cli_opts *o, const char *load_path) {
    if (o->require_eval) {
        char why[512];
        if (!lineage_require_eval(load_path, o->require_eval, o->trust_key, why, sizeof why)) {
            fprintf(stderr, "error: --require-eval %s: %s\n", o->require_eval, why);
            return false;
        }
        fprintf(stderr, "eval: %s.eval.%s.json passes for this file\n", load_path, o->require_eval);
    }
    return true;
}

// --tool-info: one JSON line naming this model's native tool-call
// protocol, resolved from its chat template exactly as the chat surface
// would (honouring --chat-template). stdout is JSON-only.
static int run_tool_info(const cli_opts *o, cli_run *rt, int tmpl_override) {
    int ti_tmpl = tmpl_override >= 0 ? tmpl_override
                : template_detect(gguf_get_str(&rt->m.gf,
                                  "tokenizer.chat_template", NULL), &rt->tok);
    bool native = false;
    const char *fam = tool_family_for(ti_tmpl, &native);
    // The template family is named too: which renderer the chat
    // surface would use is the first question a golden pass asks of
    // a file, and a fallback is a finding, not a default. So is the
    // sampling preset a request without sampling fields is served
    // with (resolved as the load does; --temp and friends aside).
    char ti_ident[256];
    sampler_ident(gguf_get_str(&rt->m.gf, "general.name", NULL), rt->m.path,
                  ti_ident, sizeof(ti_ident));
    printf("{\"tool_family\":\"%s\",\"native_tool_protocol\":%s,"
           "\"template\":\"%s\",\"sampling_preset\":\"%s\"}\n",
           fam, native ? "true" : "false", template_name(ti_tmpl),
           sampler_preset_for(rt->m.arch, ti_ident, ti_tmpl)->name);
    cli_cleanup(NULL, NULL, &rt->tok, &rt->m);
    return 0;
}

// --adapt-info (R8.9.1): inference, serving an adapter and training
// one are three capabilities with three admission lists. One JSON
// line says which of the last two this file has and, when not, the
// sentence the real operation would refuse with. stdout is JSON-only.
static int run_adapt_info(cli_run *rt) {
    char why_s[256] = "", why_t[256] = "";
    bool can_s = model_lora_serve_supported(&rt->m, why_s, sizeof why_s);
    bool can_t = model_lora_train_supported(&rt->m, why_t, sizeof why_t);
    char gname[128] = "";
    bool gpu_there = gpu_available(gname, (int)sizeof gname);
    sbuf r = {0};
    sb_lit(&r, "{\"architecture\":\"");
    sb_esc(&r, rt->m.arch, strlen(rt->m.arch));
    sb_fmt(&r, "\",\"adapter_serving\":{\"supported\":%s,\"reason\":",
           can_s ? "true" : "false");
    if (can_s) sb_lit(&r, "null");
    else { sb_lit(&r, "\""); sb_esc(&r, why_s, strlen(why_s)); sb_lit(&r, "\""); }
    // the device applies the adapter on its own blocks where the
    // backend has the kernels; otherwise serving means --gpu off
    sb_fmt(&r, ",\"device_kernels\":%s},\"training\":{\"supported\":%s,\"reason\":",
           !can_s || !gpu_there ? "null"
#ifdef __APPLE__
           : "\"metal\"",
#else
           : "\"cuda\"",
#endif
           can_t ? "true" : "false");
    if (can_t) sb_lit(&r, "null");
    else { sb_lit(&r, "\""); sb_esc(&r, why_t, strlen(why_t)); sb_lit(&r, "\""); }
    sb_lit(&r, ",\"host\":\"cpu\"}}\n");
    if (r.failed) { free(r.s); cli_cleanup(NULL, NULL, &rt->tok, &rt->m); return 1; }
    fwrite(r.s, 1, r.n, stdout);
    free(r.s);
    cli_cleanup(NULL, NULL, &rt->tok, &rt->m);
    return 0;
}

// The load: the model (with its signature, and the adapter's), its tokenizer,
// the modes that answer from the loaded file alone (--tool-info,
// --adapt-info), the measured-envelope gate, the instance registry and
// the sampler the model's family and file resolve to. A swap-mode server
// loads nothing here and resolves the sampler per model later.
//
// A reservation is a budget for the whole server, so the -c 0 auto-fit has
// to know how many slots will divide it. Set before the FIRST load, not
// just for the slots server_run creates: slot 0 is this model, and a slot 0
// that sized itself alone would both over-commit and — because the CUDA
// shared-weight registry keys on context — disagree with its peers and
// force a second upload of the same weights.
//
// True when the process exits with *rc (a refusal, or one of the two modes).
static bool cli_load_model(cli_opts *o, cli_run *rt, const char *load_path, bool registry,
                           int tmpl_override, const oms_policy *signing, int *rc) {
    if (o->train_path) o->mp.gpu_mode = GPU_OFF;   // --train is the CPU path (v1)
    // --adapt-info answers for the model, not for where it happened to load:
    // training is CPU-hosted, and the device half is named separately
    if (o->adapt_info) o->mp.gpu_mode = GPU_OFF;
    if (o->serve) o->mp.n_seq = o->parallel;
    o->mp.lora_path = o->lora_path;
    o->mp.lora_sig = o->lora_sig;
    o->mp.lora_scale = o->lora_scale;
    if (!registry) {
        double t1 = now_s();
        if (!model_load(&rt->m, load_path, &o->mp)) { *rc = 1; return true; }
        oms_result osr;
        if (!oms_check_model(load_path, signing, &osr)) { *rc = 1; return true; }
        if (osr.status[0])
            oms_result_json(&osr, rt->model_sig_json, sizeof rt->model_sig_json);
        // R1.2.3: an adapter changes what is served, so the same trusted key
        // and the same "required" apply to it, with its own bundle
        oms_result aosr;
        memset(&aosr, 0, sizeof aosr);
        if (o->lora_path) {
            oms_policy apol = { o->lora_sig, o->model_pubkey, o->require_signed_model };
            if (!oms_check_artifact(o->lora_path, &apol, "adapter", &aosr)) {
                *rc = 1;
                return true;
            }
            if (aosr.status[0])
                oms_result_json(&aosr, rt->adapter_sig_json, sizeof rt->adapter_sig_json);
        }
        if (!tokenizer_init(&rt->tok, &rt->m.gf)) { *rc = 1; return true; }
        // serve mode: the registry model's own pool is one thread by design
        // (slots create their pools), so the honest number here is the count
        // the slots divide, not the placeholder pool (the lab read "1 threads"
        // as a CPU fallback at one thread, 2026-09-23)
        fprintf(stderr, "loaded %s | %s | %d layers | ctx %d | %d threads%s | %.2fs\n",
                load_path, rt->m.arch, rt->m.n_layer, rt->m.n_ctx,
                o->serve ? o->n_threads : tpool_size(rt->m.tp),
                o->serve ? " (divided across slots)" : "",
                now_s() - t1);
        if (o->tool_info) {
            *rc = run_tool_info(o, rt, tmpl_override);
            return true;
        }
        if (o->adapt_info) {
            *rc = run_adapt_info(rt);
            return true;
        }
        // Measured-envelope sidecar, if one sits next to the model: the
        // load-time three-state gate. Certified/experimental print a banner and
        // load; an OUTSIDE-envelope verdict REFUSES the load with the measured
        // reason unless --force-uncertified is given. Silent when there is no
        // manifest.
        {
            // Resolve against the kernels this model actually loaded. A GPU
            // may be available while --gpu off (or a model-specific fallback)
            // leaves m.gpu NULL; classifying that CPU run as Metal/CUDA would
            // apply a verdict measured for a backend that did no work here.
#ifdef __APPLE__
            const char *env_backend = rt->m.gpu ? "metal" : "cpu";
#else
            const char *env_backend = rt->m.gpu ? "cuda" : "cpu";
#endif
            char env_line[256];
            int env_state = ENV_UNCLASSIFIED;
            bool ok = envelope_gate(load_path, RUNNER_VERSION, env_backend,
                                    o->force_uncertified, env_line,
                                    (int)sizeof env_line, &env_state);
            rt->env_state = env_state;
            rt->env_known = true;
            if (env_line[0]) fprintf(stderr, "%s\n", env_line);
            if (!ok) {
                cli_cleanup(NULL, NULL, &rt->tok, &rt->m);
                *rc = 1;
                return true;
            }
            // The server's /v1/runner/provenance reports the verdicts THIS
            // load ran under, so they are recorded here, where they were made.
            if (o->serve)
                provenance_note_load(&(provenance_load){
                    .model_path = load_path,
                    .adapter_path = o->lora_path, .adapter_scale = o->lora_scale,
                    .signature = &osr, .adapter_signature = &aosr,
                    .envelope_state = env_state, .envelope_detail = env_line });
        }
        // Discovery registry: run modes announce themselves so the tray (or
        // any controller) can list every live runner. Best-effort — failure
        // never affects the run. Utility modes (--quantize/--caps/--bench)
        // stay unregistered.
        if (!o->bench_json && !o->quant_out) {
            const char *bn = strrchr(load_path, '/');
#ifdef _WIN32
            const char *bn2 = strrchr(load_path, '\\');
            if (bn2 && (!bn || bn2 > bn)) bn = bn2;
#endif
            bn = bn ? bn + 1 : load_path;
            const char *const names[] = { bn };
            const char *const paths[] = { load_path };
            instances_register(o->serve ? "serve" : "cli", o->serve ? o->port : 0,
                               names, paths, 1);
            atexit(instances_unregister);
        }
        if (rt->m.mtp_layers)
            fprintf(stderr, "mtp: %d predictor block(s) declared, excluded from "
                    "the backbone; %s\n", rt->m.mtp_layers,
                    rt->m.mtp_ready ? "bound as the draft head (--mtp)"
                                : "not consumed (pass --mtp to draft from it)");
        if (rt->m.agent_profile)
            fprintf(stderr, "agent-profile: protocol=%u tokenizer=%u schema=%s digest=%s features=%llu\n",
                    rt->m.agent_protocol_version, rt->m.agent_tokenizer_version,
                    rt->m.agent_schema_id, rt->m.agent_schema_digest,
                    (unsigned long long)rt->m.n_agent_required_features);
        char ident[256];
        sampler_ident(gguf_get_str(&rt->m.gf, "general.name", NULL), rt->m.path,
                      ident, sizeof(ident));
        int preset_tmpl = tmpl_override >= 0 ? tmpl_override
                        : template_detect(gguf_get_str(&rt->m.gf,
                                          "tokenizer.chat_template", NULL), NULL);
        sampler_file sfile;
        model_sampling_file(&rt->m.gf, &sfile);
        const sampler_preset *sp = sampler_resolve_file(&o->smp, rt->m.arch, ident,
                                                        preset_tmpl, &sfile, &o->ov);
        char sdesc[256];
        sampler_describe(&o->smp, sp, sdesc, sizeof(sdesc));
        fprintf(stderr, "sampling: %s\n", sdesc);
        {
            char fdesc[160];
            sampler_file_describe(&sfile, fdesc, sizeof fdesc);
            if (fdesc[0]) fprintf(stderr, "sampling: %s\n", fdesc);
        }
        snprintf(rt->sampling, sizeof rt->sampling, "%s", sdesc);
        rt->load_s = now_s() - t1;
        // An operator who set a reasoning-channel default must see it here:
        // the line above describes the sampler for calls and answers, and
        // silence about the reasoning turn reads as "there is only one".
        if (o->reasoning_temp >= 0.0f)
            fprintf(stderr, "reasoning: temp %.2f (top_p, min_p, top_k "
                            "inherited from the request or the preset)\n",
                    (double)o->reasoning_temp);
    } else {
        // swap mode: no model yet, so the preset is chosen per model at load
        sampler_resolve(&o->smp, NULL, NULL, -1, &o->ov);
    }
    return false;
}

// --serve: hand the loaded model (or, for a swap set, none) to the server,
// with the watermark key and the per-request adapters it serves.
static int run_serve(cli_opts *o, cli_run *rt, const wm_key *wmk, bool wm_on,
                     bool registry, int tmpl_override, const oms_policy *signing) {
    if (wm_on) server_set_watermark(wmk);
    if (o->n_adapters) server_set_adapters(o->adapter_names, o->adapter_paths,
                                        o->n_adapters, o->lora_scale);
    if (registry) {
        // swap-mode server: models come and go at runtime; the record
        // carries none and readers ask /v1/models live
        instances_register("serve", o->port, NULL, NULL, 0);
        atexit(instances_unregister);
    }
    int rc = server_run(registry ? NULL : &rt->m, registry ? NULL : &rt->tok,
                        o->model_path, &o->mp, o->smp, &o->ov, o->port, o->parallel, o->n_threads,
                        o->ttl, o->draft_path, o->draft_k, o->draft_lookup,
                        o->reasoning_budget, o->reasoning_budget_message,
                        o->reasoning_temp, o->loop_guard, o->ignore_eos,
                        tmpl_override, o->force_uncertified, signing);
    return rc;
}

// The engine over the loaded model: the watermark hook, a draft model, the
// MTP head or the lookup draft, a --json-schema constraint, and the prompt
// token buffer. False when one of them is refused or cannot be had.
static bool cli_engine_setup(cli_opts *o, cli_run *rt, bool wm_on, wm_state *cli_wm) {
    if (!engine_init(&rt->e, &rt->m, &rt->tok, &o->smp)) { // rt->e zeroed with rt_state
        fprintf(stderr, "error: out of memory initializing engine\n");
        return false;
    }
    rt->e.ignore_eos = o->ignore_eos;
    rt->e.json_mode = o->json_mode;
    rt->e.progress = true;
    if (wm_on) {
        rt->e.wm_prepare = wm_prepare;
        rt->e.wm_ud = cli_wm;
    }
    if (o->draft_path) {
        rt->e.dm = spec_draft_load(o->draft_path, &rt->m, &o->mp);
        if (rt->e.dm) {
            fprintf(stderr, "draft: %s (%d tokens per round)\n", o->draft_path, o->draft_k);
            rt->e.draft_k = o->draft_k;
        } else if (o->draft_required) {
            fprintf(stderr, "error: --draft-required: %s was refused above and"
                            " the run would have decoded without it\n",
                    o->draft_path);
            return false;
        }
    }
    if (o->mtp_on) {
        if (!model_mtp_ready(&rt->m)) {
            fprintf(stderr, "error: --mtp: the head is bound but drafts need "
                            "the CPU path (rerun with --gpu off)\n");
            return false;
        }
        rt->e.mtp_on = true;
        rt->e.draft_k = o->draft_k;
        fprintf(stderr, "mtp: drafting from the model's NextN head "
                "(%d tokens per round)\n", o->draft_k);
    }
    if (o->draft_lookup) {
        if (!model_spec_verify_ok(&rt->m)) {
            fprintf(stderr, "error: --draft-lookup: the verify walk needs "
                            "host-readable hidden work (rerun with --gpu off)\n");
            return false;
        }
        rt->e.lookup_on = true;
        rt->e.draft_k = o->draft_k;
        fprintf(stderr, "lookup: drafting from the context (n-grams of %d..%d "
                "tokens, up to %d per round)\n", ENGINE_LOOKUP_N_MIN,
                ENGINE_LOOKUP_N_MAX, o->draft_k);
    }
    if (o->schema_file) {
        size_t ssz = 0;
        char *sbuf = read_file(o->schema_file, &ssz);
        if (!sbuf) { fprintf(stderr, "error: cannot read %s\n", o->schema_file); return false; }
        struct jv *sj = json_parse(sbuf, ssz);
        free(sbuf);
        if (!sj) { fprintf(stderr, "error: %s is not valid JSON\n", o->schema_file); return false; }
        char serr[128];
        rt->e.schema = schema_compile(sj, serr, sizeof(serr));
        jv_free(sj);
        if (!rt->e.schema) { fprintf(stderr, "error: unsupported schema: %s\n", serr); return false; }
        sval_init(&rt->e.sv, rt->e.schema);
    }

    rt->tok_cap = (o->prompt ? strlen(o->prompt) : 0) + (size_t)rt->m.n_ctx + 32;
    // tok_encode narrows the capacity to int, so keep it representable
    if (rt->tok_cap > INT_MAX) rt->tok_cap = INT_MAX;
    rt->toks = malloc(sizeof(int32_t) * rt->tok_cap);
    if (!rt->toks) { fprintf(stderr, "error: out of memory\n"); return false; }
    return true;
}

// AdamW's constants. Both objectives' steps use them and the provenance
// record states them, stringized, so the record's text is the source's.
#define TRAIN_ADAM_BETA1 0.9
#define TRAIN_ADAM_BETA2 0.999
#define TRAIN_ADAM_EPS   1e-8
#define TRAIN_ADAM_WD    0.01
#define TRAIN_STR_(x) #x
#define TRAIN_STR(x)  TRAIN_STR_(x)

static bool train_adam_step(cli_opts *o, cli_run *rt, int step) {
    if (model_lora_adam_step(&rt->m, o->train_lr, (float)TRAIN_ADAM_BETA1,
                             (float)TRAIN_ADAM_BETA2, (float)TRAIN_ADAM_EPS,
                             (float)TRAIN_ADAM_WD, step))
        return true;
    fprintf(stderr, "error: cannot allocate optimizer state\n");
    return false;
}

// What a run trains: the attached adapter, or a fresh one; refused on a
// device-resident model.
static bool train_prepare(cli_opts *o, cli_run *rt) {
    if (rt->m.gpu) {
        // v1 trains on the CPU path only (D8 is CUDA training); refuse
        // loudly rather than train a model whose forward the backward
        // cannot see.
        fprintf(stderr, "error: --train is CPU-only for now — run with "
                "--gpu off\n");
        return false;
    }
    if (!rt->m.lora) {
        uint64_t iseed = o->seed_given ? o->smp.rng : 0;
        if (!model_lora_train_init(&rt->m, o->lora_rank,
                                   2.0f * (float)o->lora_rank, iseed))
            return false;
        fprintf(stderr, "train: fresh rank-%d adapters on every "
                "projection (alpha %g)\n", o->lora_rank,
                (double)(2.0f * o->lora_rank));
    }
    // D8 slice 2, opt-in: run the backward's transposed matvec on the
    // device with the weights cached there. The kernel is gated
    // bit-identical to the CPU chain, so the adapter bytes do not
    // depend on this switch — only the step time does. Failing to init
    // (no CUDA) just stays on the CPU without a word beyond this one.
    {
        const char *tg = getenv("RUNNER_TRAIN_GPU");
        if (tg && *tg && strcmp(tg, "0") != 0 && !gpu_train_init(&rt->m))
            fprintf(stderr, "train-gpu: no usable CUDA device — "
                    "training on the CPU\n");
    }
    return true;
}

// --train-dpo's steps: one preference pair per step, cycled in file order.
static bool train_dpo_steps(cli_opts *o, cli_run *rt, dpo_pair *pairs,
                            int n_pair, int wctx, double *first_loss,
                            double *last_loss) {
    // DPO (arXiv 2305.18290 eq. 7). pi_ref is this same model with the
    // adapter bypassed, so the reference costs no second copy of the
    // weights; see src/dpo.c for why the gradient is a scalar multiple
    // of a difference of two weighted-CE gradients and therefore needs
    // no new kernel. The objective and its gradient are gated against
    // an independent numerical reference by tests/test_dpo_grad.c
    // (independent loss formula, the gradient identity, and a
    // directional derivative against central differences) because an
    // unchecked loss that merely descends is indistinguishable from a
    // checked one until the adapter is useless.
    fprintf(stderr, "train-dpo: %d pair%s, %d steps, lr %g, beta %g, ctx %d\n",
            n_pair, n_pair == 1 ? "" : "s", o->train_steps, (double)o->train_lr,
            (double)o->dpo_beta, wctx);
    double acc_sum = 0, margin_sum = 0;
    for (int step = 1; step <= o->train_steps; step++) {
        dpo_pair *q = &pairs[(step - 1) % n_pair];
        double step_t0 = plat_now();
        dpo_step d;
        if (!dpo_accumulate(&rt->m, (double)o->dpo_beta, q->tw, q->nw, q->mw,
                            q->tl, q->nl, q->ml, &d)) {
            fprintf(stderr, "error: DPO step failed\n");
            return false;
        }
        if (!train_adam_step(o, rt, step)) return false;
        if (step == 1) *first_loss = d.loss;
        *last_loss = d.loss;
        acc_sum += d.acc;
        margin_sum += d.margin;
        double step_s = plat_now() - step_t0;
        // margin and acc are the fields that say whether the objective
        // is doing anything: a run whose loss falls while the margin
        // stays at zero has learned to shrink the reference gap rather
        // than to prefer the chosen completion.
        printf("{\"step\":%d,\"pair\":%d,\"tokens\":%d,"
               "\"loss\":%.6f,\"margin\":%.6f,\"acc\":%.0f,"
               "\"coeff\":%.6g,\"logp_w\":%.4f,\"logp_l\":%.4f,"
               "\"logp_ref_w\":%.4f,\"logp_ref_l\":%.4f,"
               "\"step_s\":%.2f}\n",
               step, (step - 1) % n_pair, q->nw + q->nl,
               d.loss, d.margin, d.acc, d.coeff,
               d.logp_w, d.logp_l, d.logp_ref_w, d.logp_ref_l, step_s);
        fflush(stdout);
        if (o->save_every > 0 && step % o->save_every == 0 &&
            !model_lora_save(&rt->m, o->train_out))
            return false;
    }
    fprintf(stderr, "train-dpo: mean margin %.6f, accuracy %.3f over "
            "%d steps\n", margin_sum / o->train_steps,
            acc_sum / o->train_steps, o->train_steps);
    return true;
}

// --train's steps: weighted cross-entropy, one example per step, cycled in
// file order.
static bool train_ce_steps(cli_opts *o, cli_run *rt, train_ex *exs, int n_ex,
                           int wctx, double *first_loss, double *last_loss) {
    fprintf(stderr, "train: %d example%s, %d steps, lr %g, ctx %d\n",
            n_ex, n_ex == 1 ? "" : "s", o->train_steps, (double)o->train_lr,
            wctx);
    for (int step = 1; step <= o->train_steps; step++) {
        train_ex *ex = &exs[(step - 1) % n_ex];
        double loss = 0;
        double step_t0 = plat_now();
        model_lora_grad_zero(&rt->m);
        if (!model_lora_backward_w(&rt->m, ex->t, ex->n, ex->w, &loss))
            return false;
        if (!train_adam_step(o, rt, step)) return false;
        // Per-token mean over the weighted positions, normalized by the
        // weights' MAGNITUDE. `wsum > 0 ? loss/wsum : loss` silently fell
        // back to the unnormalized total whenever the weights summed
        // negative, which is exactly what a GRPO-style negative advantage
        // produces: a run's reported loss then changed scale between
        // positive- and negative-weight examples, by a factor of the token
        // count, and loss_first/loss_last in the provenance record
        // inherited it. Measured on the fixture: a w=-1 example printed
        // -66.30 where the per-token mean is -16.58. The sign still
        // carries the weight's sign, which is the informative part.
        double wsum = 0;
        if (ex->w) {
            for (int i = 0; i < ex->n - 1; i++) wsum += ex->w[i];
        } else wsum = ex->n - 1;
        double wmag = wsum < 0 ? -wsum : wsum;
        double ml = wmag > 0 ? loss / wmag : loss;
        if (step == 1) *first_loss = ml;
        *last_loss = ml;
        // step_s / tok_s: wall time and TRAINING tokens/sec (forward +
        // backward + optimizer, not inference throughput). Reporting
        // fields only — the reproducibility contract covers the loss
        // and the adapter bytes, never the clock.
        double step_s = plat_now() - step_t0;
        printf("{\"step\":%d,\"example\":%d,\"tokens\":%d,"
               "\"loss\":%.6f,\"step_s\":%.2f,\"tok_s\":%.1f}\n",
               step, (step - 1) % n_ex, ex->n, ml, step_s,
               step_s > 0 ? (double)ex->n / step_s : 0.0);
        fflush(stdout);
        if (o->save_every > 0 && step % o->save_every == 0 &&
            !model_lora_save(&rt->m, o->train_out))
            return false;
    }
    return true;
}

// D7: the provenance record, written beside every adapter — the
// reproducibility claim in checkable form. Two runs with the same
// base/data/seed/config must produce the same adapter_sha256.
static bool train_record(cli_opts *o, cli_run *rt, const char *load_path,
                         int wctx, int eot_id, double first_loss,
                         double last_loss) {
    char bsha[65] = "", dsha[65] = "", asha[65] = "", xsha[65] = "";
    const char *data_path = o->dpo_path ? o->dpo_path : o->train_path;
    bool hash_ok = envelope_file_sha256(load_path, bsha) &&
                   envelope_file_sha256(data_path, dsha) &&
                   envelope_file_sha256(o->train_out, asha);
    // The running binary's own hash + build identity, so a
    // reproduction report can distinguish "same executable" from
    // "same source" mechanically. The determinism contract is
    // ARTIFACT-level: same binary + same inputs -> same adapter
    // bytes. An external rebuild study showed a different ISA
    // profile changes the bytes (libm expf in SiLU/softmax; the
    // fmaf chains are pinned, transcendentals are the compiler's)
    // while behavior stays put — these fields make that boundary
    // visible in every record.
    char *exe_path = plat_executable_path();
    hash_ok = hash_ok && exe_path &&
              envelope_file_sha256(exe_path, xsha);
    free(exe_path);
    if (!hash_ok) {
        fprintf(stderr, "error: cannot hash every training artifact "
                "for the provenance record\n");
        return false;
    }
    sbuf rec = {0};
    sb_lit(&rec, "{\"schema_version\":\"xyntetik.runner.train.v1\","
                 "\"runner\":\"");
    sb_esc(&rec, RUNNER_VERSION, strlen(RUNNER_VERSION));
    sb_lit(&rec, "\",\"build\":{\"binary_sha256\":\"");
    sb_lit(&rec, xsha);
    sb_lit(&rec, "\",\"compiler\":\"");
    sb_esc(&rec, __VERSION__, strlen(__VERSION__));
    sb_lit(&rec, "\",\"os\":\"");
    const char *build_os =
#ifdef _WIN32
            "windows";
#elif defined(__APPLE__)
            "macos";
#else
            "linux";
#endif
    sb_lit(&rec, build_os);
    sb_lit(&rec, "\",\"arch\":\"");
    const char *build_arch = RUNNER_BUILD_ARCH;
    sb_lit(&rec, build_arch);
    sb_lit(&rec, "\"},\"base\":");
    lineage_put_input(&rec, load_path, bsha);
    sb_lit(&rec, ",\"objective\":\"");
    sb_lit(&rec, o->dpo_path ? "dpo" : "weighted-cross-entropy");
    if (o->dpo_path) sb_fmt(&rec, "\",\"dpo_beta\":%g", (double)o->dpo_beta);
    else sb_lit(&rec, "\"");
    sb_lit(&rec, ",\"data\":");
    lineage_put_input(&rec, data_path, dsha);
    sb_fmt(&rec,
            ",\"seed\":%llu,\"lora_rank\":%d,\"alpha\":%g,"
            "\"lr\":%g,\"steps\":%d,\"ctx\":%d,"
            "\"end_of_turn\":%s,\"eot_id\":%d,"
            "\"adamw\":{\"beta1\":" TRAIN_STR(TRAIN_ADAM_BETA1)
            ",\"beta2\":" TRAIN_STR(TRAIN_ADAM_BETA2)
            ",\"eps\":" TRAIN_STR(TRAIN_ADAM_EPS)
            ",\"weight_decay\":" TRAIN_STR(TRAIN_ADAM_WD) "},"
            "\"loss_first\":%.6f,\"loss_last\":%.6f,"
            "\"adapter\":{\"path\":\"",
            (unsigned long long)(o->seed_given ? o->smp.rng : 0),
            o->lora_rank, (double)rt->m.lora_alpha, (double)o->train_lr,
            o->train_steps, wctx, o->train_eot ? "true" : "false", eot_id,
            first_loss, last_loss);
    sb_esc(&rec, o->train_out, strlen(o->train_out));
    sb_lit(&rec, "\",\"sha256\":\""); sb_lit(&rec, asha);
    sb_lit(&rec, "\"}}\n");
    bool ok = write_record(o->train_out, ".train.json", "train", &rec, o->sign_key);
    free(rec.s);
    return ok;
}

// --train / --train-dpo: LoRA training on the CPU path, and the provenance
// record beside the adapter.
static int run_train(cli_opts *o, cli_run *rt, int tmpl_override, const char *load_path) {
    if (o->train_path && o->dpo_path) {
        fprintf(stderr, "error: --train and --train-dpo are different "
                "objectives; pass one\n");
        return 1;
    }
    // --train (adaptation D4): AdamW LoRA training in the serving binary,
    // CPU reference. Two data modes: plain text (tokenize, fixed windows,
    // cycle) and .jsonl lines {"prompt","completion","weight"} — prompt
    // transitions masked to weight 0, completion transitions carry the
    // example weight (the GRPO-style advantage hook). Determinism is the
    // default: fixed init seed unless -s is given, fixed data order, and
    // the whole compute path is the byte-deterministic D3 backward — same
    // data + same seed -> byte-identical adapter file.
    if (!train_prepare(o, rt)) return 1;
    train_ex *exs = NULL;
    int n_ex = 0;
    dpo_pair *pairs = NULL;
    int n_pair = 0;
    int wctx = o->train_ctx < rt->m.n_ctx ? o->train_ctx : rt->m.n_ctx;
    int train_tmpl = tmpl_override >= 0 ? tmpl_override
                     : template_detect(gguf_get_str(&rt->m.gf, "tokenizer.chat_template", NULL), &rt->tok);
    int eot_id = train_eot_id(&rt->tok, train_tmpl);
    if (o->dpo_path) {
        if (!dpo_pairs_load(&rt->tok, o->dpo_path, o->no_bos, wctx, eot_id, o->train_eot,
                            &pairs, &n_pair)) return 1;
    } else if (!train_examples_load(&rt->tok, o->train_path, o->no_bos, wctx, eot_id, o->train_eot,
                                    &exs, &n_ex)) return 1;
    double first_loss = 0, last_loss = 0;
    bool ok = o->dpo_path
        ? train_dpo_steps(o, rt, pairs, n_pair, wctx, &first_loss, &last_loss)
        : train_ce_steps(o, rt, exs, n_ex, wctx, &first_loss, &last_loss);
    ok = ok && model_lora_save(&rt->m, o->train_out) &&
         train_record(o, rt, load_path, wctx, eot_id, first_loss, last_loss);
    if (ok)
        fprintf(stderr, "train: done — first-step loss %.4f, last-step "
                "%.4f, adapter -> %s\n", first_loss, last_loss, o->train_out);
    train_examples_free(exs, n_ex);
    dpo_pairs_free(pairs, n_pair);
    return ok ? 0 : 1;
}

// ---- notarized inference D2: the replay
static int run_verify_replay(cli_opts *o, cli_run *rt, cli_verify *vr, const char *load_path) {
    jv *vm = jv_get(vr->rec, "model"), *va = jv_get(vr->rec, "adapter");
    jv *vprof = jv_get(vr->rec, "profile");
    int expected_ctx = (int)jv_num(jv_get(vprof, "ctx"), -1);
    int expected_batch = (int)jv_num(jv_get(vprof, "batch"), -1);
    int expected_threads = (int)jv_num(jv_get(vprof, "threads"), -1);
    bool expected_gpu = jv_bool(jv_get(vprof, "gpu"), false);
    bool expected_q8 = strcmp(jv_str(jv_get(vprof, "kv"), ""), "q8") == 0;
    bool expected_fp4 = strcmp(jv_str(jv_get(vprof, "kv"), ""), "fp4") == 0;
    bool expected_split = strcmp(jv_str(jv_get(vprof, "kv"), ""), "k8v4") == 0;
    jv *expected_layers_v = jv_get(vprof, "gpu_layers");
    int expected_layers = (int)jv_num(expected_layers_v, -1);
    if (rt->m.n_ctx != expected_ctx || rt->m.n_batch != expected_batch ||
        tpool_size(rt->m.tp) != expected_threads || rt->m.kv_q8 != expected_q8 ||
        rt->m.kv_fp4 != expected_fp4 || rt->m.kv_split != expected_split ||
        (rt->m.gpu != NULL) != expected_gpu ||
        (expected_layers_v && rt->m.gpu_layers != expected_layers)) {
        fprintf(stderr, "UNVERIFIABLE: runtime could not reproduce the "
                "recorded execution profile\n");
        jv_free(vr->rec);
        return 3;
    }
    char sha[65] = "";
    envelope_file_sha256(load_path, sha);
    if (strcmp(sha, jv_str(jv_get(vm, "sha256"), "")) != 0) {
        fprintf(stderr, "UNVERIFIABLE: -m %s sha256 %s does not match "
                "the record's %s\n", load_path, sha,
                jv_str(jv_get(vm, "sha256"), "?"));
        jv_free(vr->rec);
        return 3;
    }
    if (va && va->type != J_NULL) {
        if (!o->lora_path) {
            fprintf(stderr, "UNVERIFIABLE: the record used an adapter "
                    "(sha %s) — pass it with --lora\n",
                    jv_str(jv_get(va, "sha256"), "?"));
            jv_free(vr->rec);
            return 3;
        }
        char asha[65] = "";
        envelope_file_sha256(o->lora_path, asha);
        if (strcmp(asha, jv_str(jv_get(va, "sha256"), "")) != 0) {
            fprintf(stderr, "UNVERIFIABLE: --lora sha256 does not match "
                    "the record's adapter\n");
            jv_free(vr->rec);
            return 3;
        }
    } else if (o->lora_path) {
        fprintf(stderr, "UNVERIFIABLE: --lora given but the record ran "
                "without an adapter\n");
        jv_free(vr->rec);
        return 3;
    }
    jv *jp = jv_get(jv_get(vr->rec, "prompt"), "tokens");
    jv *vout = jv_get(vr->rec, "output");
    jv *jo = jv_get(vout, "tokens");
    const char *expected_text = jv_str(jv_get(vout, "text"), NULL);
    jv *expected_hex_v = jv_get(vout, "bytes_hex");
    const char *expected_hex = NULL;
    size_t expected_bytes_n = 0;
    bool has_exact_bytes = expected_hex_v != NULL;
    if (!jp || jp->type != J_ARR || !jo || jo->type != J_ARR ||
        !expected_text || jp->n < 1 || jp->n >= rt->m.n_ctx ||
        (size_t)jp->n >= rt->tok_cap) {
        fprintf(stderr, "UNVERIFIABLE: malformed token arrays\n");
        jv_free(vr->rec);
        return 3;
    }
    if (has_exact_bytes &&
        !record_hex(expected_hex_v, &expected_hex, &expected_bytes_n)) {
        fprintf(stderr, "UNVERIFIABLE: malformed output.bytes_hex\n");
        jv_free(vr->rec);
        return 3;
    }
    for (int i = 0; i < jp->n; i++) {
        if (!record_token(jp->items[i], rt->m.n_vocab, &rt->toks[i])) {
            fprintf(stderr, "UNVERIFIABLE: invalid prompt token at %d\n",
                    i);
            jv_free(vr->rec);
            return 3;
        }
    }
    for (int i = 0; i < jo->n; i++) {
        int32_t ignored;
        if (!record_token(jo->items[i], rt->m.n_vocab, &ignored)) {
            fprintf(stderr, "UNVERIFIABLE: invalid output token at %d\n",
                    i);
            jv_free(vr->rec);
            return 3;
        }
    }
    double vt0 = now_s();
    float *logits = engine_feed(&rt->e, rt->toks, jp->n);
    if (!logits) {
        fprintf(stderr, "UNVERIFIABLE: prompt exceeds context\n");
        jv_free(vr->rec);
        return 3;
    }
    double gt = 0;
    outcap_t replay_out = {0};
    int n_gen = engine_generate(&rt->e, logits, o->n_predict, outcap_cb,
                                &replay_out, &gt);
    if (replay_out.failed) {
        fprintf(stderr, "UNVERIFIABLE: out of memory capturing replay\n");
        free(replay_out.buf); jv_free(vr->rec);
        return 3;
    }
    (void)vt0;
    // tier: same binary = T1 (bit-exact by the gated contract), else
    // T2 (token replay across builds; libm is the known bit blocker)
    char bsha[65] = "";
    char *verify_exe = plat_executable_path();
    if (verify_exe) envelope_file_sha256(verify_exe, bsha);
    free(verify_exe);
    const char *rbsha = jv_str(jv_get(jv_get(vr->rec, "build"),
                                      "binary_sha256"), "");
    const char *tier = (bsha[0] && strcmp(bsha, rbsha) == 0) ? "T1" : "T2";
    int cmp_n = n_gen < jo->n ? n_gen : jo->n;
    int div_at = -1;
    for (int i = 0; i < cmp_n; i++)
        if ((int32_t)jv_num(jo->items[i], -1) != rt->e.hist[jp->n + i]) {
            div_at = i;
            break;
        }
    if (div_at < 0 && n_gen != jo->n) div_at = cmp_n;
    size_t byte_div_at = SIZE_MAX;
    if (div_at < 0 && has_exact_bytes) {
        size_t byte_cmp_n = replay_out.n < expected_bytes_n
                          ? replay_out.n : expected_bytes_n;
        for (size_t i = 0; i < byte_cmp_n; i++) {
            int expected = hex_nibble(expected_hex[i * 2]) * 16 +
                           hex_nibble(expected_hex[i * 2 + 1]);
            if ((unsigned char)replay_out.buf[i] != expected) {
                byte_div_at = i;
                break;
            }
        }
        if (byte_div_at == SIZE_MAX && replay_out.n != expected_bytes_n)
            byte_div_at = byte_cmp_n;
    }
    int rc;
    if (div_at < 0 && byte_div_at == SIZE_MAX) {
        printf("{\"schema_version\":\"xyntetik.runner.verify.v1\","
               "\"verdict\":\"VERIFIED\",\"tier\":\"%s\","
               "\"tokens\":%d,\"signed\":%s,\"public_key\":\"%s\","
               "\"prev\":\"%s\"}\n", tier, n_gen,
               vr->rsig == RSIG_OK ? "true" : "false", vr->rec_pub, vr->prev);
        fprintf(stderr, "VERIFIED (%s %s): %d output tokens replayed "
                "identically\n", tier,
                strcmp(tier, "T1") == 0 ? "same-binary"
                                        : "token-replay", n_gen);
        rc = 0;
    } else if (div_at >= 0) {
        int want = div_at < jo->n ? (int)jv_num(jo->items[div_at], -1)
                                  : -1;
        int got = div_at < n_gen ? (int)rt->e.hist[jp->n + div_at] : -1;
        printf("{\"schema_version\":\"xyntetik.runner.verify.v1\","
               "\"verdict\":\"DIVERGED\",\"unit\":\"token\","
               "\"at\":%d,"
               "\"expected\":%d,\"got\":%d,\"signed\":%s,"
               "\"public_key\":\"%s\"}\n", div_at, want, got,
               vr->rsig == RSIG_OK ? "true" : "false", vr->rec_pub);
        fprintf(stderr, "DIVERGED at token %d: record has %d, replay "
                "produced %d\n", div_at, want, got);
        rc = 2;
    } else {
        int want = byte_div_at < expected_bytes_n
                 ? hex_nibble(expected_hex[byte_div_at * 2]) * 16 +
                   hex_nibble(expected_hex[byte_div_at * 2 + 1]) : -1;
        int got = byte_div_at < replay_out.n
                ? (unsigned char)replay_out.buf[byte_div_at] : -1;
        printf("{\"schema_version\":\"xyntetik.runner.verify.v1\","
               "\"verdict\":\"DIVERGED\",\"unit\":\"byte\","
               "\"at\":%zu,\"expected\":%d,\"got\":%d,\"signed\":%s,"
               "\"public_key\":\"%s\"}\n",
               byte_div_at, want, got,
               vr->rsig == RSIG_OK ? "true" : "false", vr->rec_pub);
        fprintf(stderr, "DIVERGED at output byte %zu: record has %d, "
                "replay produced %d\n", byte_div_at, want, got);
        rc = 2;
    }
    free(replay_out.buf);
    jv_free(vr->rec);
    return rc;
}

static int run_detect_text(cli_opts *o, cli_run *rt, const wm_key *wmk, char *detect_text) {
    // R1.8.2 over a text: tokenized without BOS by the model's own
    // tokenizer, scored from the first position with a full context
    size_t cap = strlen(detect_text) + 16;
    int32_t *dt = cap <= (size_t)INT_MAX ? malloc(sizeof *dt * cap) : NULL;
    int nt = dt ? tok_encode(&rt->tok, detect_text, dt, (int)cap, false, false) : -1;
    int rc = 3;
    if (nt < 0) fprintf(stderr, "error: --detect-watermark: out of memory\n");
    else rc = wm_report(wmk, dt, nt, WM_CONTEXT, "text", NULL);
    free(dt);
    free(detect_text);
    return rc;
}

static int run_decide(cli_opts *o, cli_run *rt) {
    // Typed decisions (R13.10): the same handler the server route uses,
    // one request per line, one response per line; a bad line is an
    // {"error":...} object on its line and the exit code is 1 at the end
    FILE *df = strcmp(o->decide_path, "-") ? fopen(o->decide_path, "rb") : stdin;
    if (!df) { fprintf(stderr, "error: cannot open %s\n", o->decide_path); return 1; }
    size_t cap = 1 << 16, len = 0; char *line = malloc(cap);
    if (!line) { if (df != stdin) fclose(df); return 1; }
    int rc = 0, lineno = 0, c;
    const char *label = strrchr(o->model_path, '/') ? strrchr(o->model_path, '/') + 1 : o->model_path;
    for (;;) {
        c = fgetc(df);
        if (c != EOF && c != '\n') {
            if (len + 1 >= cap) { cap *= 2; char *nl = realloc(line, cap); if (!nl) { rc = 1; break; } line = nl; }
            line[len++] = (char)c;
            continue;
        }
        if (len > 0) {
            lineno++;
            line[len] = 0;
            jv *req = json_parse(line, len);
            sbuf r = {0};
            const char *derr = NULL;
            int st = req ? decide_handle(&rt->e, req, label, &r, &derr) : 400;
            if (!req) derr = "invalid JSON";
            if (st == 200) printf("%s\n", r.s);
            else { printf("{\"error\":{\"line\":%d,\"status\":%d,\"message\":\"%s\"}}\n", lineno, st, derr ? derr : "decide failed"); rc = 1; }
            fflush(stdout);
            free(r.s);
            if (req) jv_free(req);
            len = 0;
        }
        if (c == EOF) break;
    }
    free(line);
    if (df != stdin) fclose(df);
    return rc;
}

static int run_score(cli_opts *o, cli_run *rt) {
    int n_prompt;
    // Teacher-forced scoring (adaptation D1): per-position
    // log P(token | prefix) over the RAW prompt bytes — no template, no
    // sampling, no gradients. DEFAULT IS THE SOLO PATH: one forward per
    // position, the exact numerics the sampler sees at decode time —
    // measured 2026-08-21, the CPU batched forward is NOT bit-identical
    // to solo (max |dlp| ~1e-6 on the fixtures, diverging from batch row
    // 2), so a chunked scorer would quietly reintroduce the train/infer
    // mismatch this mode exists to abolish. RUNNER_SCORE_CHUNKED=1 opts
    // into the ~10x-faster batched path with that measured envelope;
    // tests/test_score.py pins the delta and the solo determinism.
    if (!o->prompt || !*o->prompt) {
        fprintf(stderr, "error: --score needs a prompt (-p or -f)\n");
        return 1;
    }
    char *p = unescape(o->prompt);
    n_prompt = tok_encode(&rt->tok, p, rt->toks, (int)rt->tok_cap, !o->no_bos, true);
    free(p);
    if (n_prompt < 0) {
        fprintf(stderr, "error: out of memory tokenizing prompt\n");
        return 1;
    }
    if (n_prompt < 2) {
        fprintf(stderr, "error: --score needs at least 2 tokens to score "
                "a transition (got %d)\n", n_prompt);
        return 1;
    }
    if (n_prompt > rt->m.n_ctx) {
        fprintf(stderr, "error: --score prompt is %d tokens, context is "
                "%d\n", n_prompt, rt->m.n_ctx);
        return 1;
    }
    const char *ch_env = getenv("RUNNER_SCORE_CHUNKED");
    bool solo = !(ch_env && *ch_env && strcmp(ch_env, "0") != 0);
    printf("{\"schema_version\":\"xyntetik.runner.score.v1\","
           "\"n_tokens\":%d,\"n_scored\":%d,\"tokens\":[", n_prompt,
           n_prompt - 1);
    for (int i = 0; i < n_prompt; i++)
        printf("%s%d", i ? "," : "", rt->toks[i]);
    printf("],\"logprobs\":[");
    double total = 0;
    int pos = 0, emitted = 0, fail = 0, top1 = 0;
    int32_t *argmaxes = malloc(sizeof(int32_t) * (size_t)n_prompt);
    if (!argmaxes) return 1;
    while (pos < n_prompt - 1 && !fail) {
        int k = n_prompt - 1 - pos;
        int cap = rt->m.spec_batch < rt->m.n_batch ? rt->m.spec_batch : rt->m.n_batch;
        if (k > cap) k = cap;
        if (!solo && k > 1 &&
            model_forward_batch_keep(&rt->m, rt->toks + pos, k, pos)) {
            for (int b = 0; b < k; b++) {
                float *lg = model_spec_row_logits(&rt->m, b);
                // raw-logit log-softmax through the one normalizer every
                // reported logprob uses (engine_logsumexp, engine.h)
                int arg = 0;
                for (int i = 1; i < rt->m.n_vocab; i++)
                    if (lg[i] > lg[arg]) arg = i;
                float lp = lg[rt->toks[pos + b + 1]] -
                           engine_logsumexp(lg, rt->m.n_vocab);
                argmaxes[emitted] = arg;
                printf("%s%.9g", emitted++ ? "," : "", (double)lp);
                total += lp;
                if (arg == rt->toks[pos + b + 1]) top1++;
            }
            pos += k;
        } else {
            float *lg = model_forward(&rt->m, rt->toks[pos], pos);
            if (!lg) { fail = 1; break; }
            int arg = 0;
            for (int i = 1; i < rt->m.n_vocab; i++)
                if (lg[i] > lg[arg]) arg = i;
            float lp = lg[rt->toks[pos + 1]] - engine_logsumexp(lg, rt->m.n_vocab);
            argmaxes[emitted] = arg;
            printf("%s%.9g", emitted++ ? "," : "", (double)lp);
            total += lp;
            if (arg == rt->toks[pos + 1]) top1++;
            pos += 1;
        }
    }
    if (fail) {
        free(argmaxes);
        printf("]}\n");
        fprintf(stderr, "error: forward pass failed at position %d\n", pos);
        return 1;
    }
    // top1 is the self-check anchor: NLL alone cannot tell a working
    // scorer from a broken one, because a wrong number is still a number.
    // The absolute next-token argmax rate is bounded by facts outside this
    // implementation - a token with p > 0.5 IS the argmax, and an argmax
    // token has p >= 1/n_vocab - so a harness can bracket top1 from the
    // logprobs alone and refuse a run that disagrees with itself.
    double nll = -total, mean = nll / (n_prompt - 1);
    // per-position argmax ids (additive field): a second engine's
    // top-1 at the same prefix compares against these
    printf("],\"argmax\":[");
    for (int i = 0; i < emitted; i++)
        printf("%s%d", i ? "," : "", argmaxes[i]);
    free(argmaxes);
    printf("],\"n_vocab\":%d,\"top1\":%d,\"top1_rate\":%.9g,"
           "\"nll_total\":%.9g,\"nll_mean\":%.9g,\"ppl\":%.9g}\n",
           rt->m.n_vocab, top1, (double)top1 / (n_prompt - 1),
           nll, mean, exp(mean));
    return 0;
}

static int run_doctor(cli_opts *o, cli_run *rt, int tmpl_override) {
    double t0;
    int n_prompt;
    // R16.6.2: one explicitly invoked report a user can attach to a
    // question about slowness, nonsense or broken tool calls. It states
    // what this process actually did -- which backend took how many
    // layers, which template and sampler it resolved -- runs one short
    // chat probe through the same renderer, tokenizer and engine a chat
    // request uses, and turns what it saw into findings with a next
    // step. No content leaves the machine and none is recorded unless
    // --doctor-include-text asks for it.
    int dt = tmpl_override >= 0 ? tmpl_override
           : template_detect(gguf_get_str(&rt->m.gf, "tokenizer.chat_template", NULL), &rt->tok);
    bool have_tmpl_text = gguf_get_str(&rt->m.gf, "tokenizer.chat_template", NULL) != NULL;
    bool native = false;
    const char *fam = tool_family_for(dt, &native);
    template_bind_think_tags(dt, &rt->m.think_open, &rt->m.think_close);
    static const char question[] = "What is 2+2? Answer with just the number.";
    chat_msg dmsg[1] = { { .role = "user", .content = question } };
    char *rendered = render_prompt_alloc(dt, dmsg, 1, true, THINK_OFF, NULL,
                                         sizeof question + 1024);
    if (!rendered) {
        fprintf(stderr, "error: out of memory rendering the probe\n");
        return 1;
    }
    n_prompt = tok_encode_prompt(&rt->tok, rendered, rt->toks, (int)rt->tok_cap, !o->no_bos);
    free(rendered);
    bool probe_ran = false, probe_stop = false, probe_has4 = false;
    bool probe_leak = false, probe_empty = true;
    double d_ptime = 0, d_gtime = 0;
    int d_gen = 0;
    outcap_t dout = {0};
    enum { DOCTOR_TOKENS = 48 };
    if (n_prompt > 0 && n_prompt + DOCTOR_TOKENS < rt->m.n_ctx) {
        t0 = now_s();
        float *logits = engine_feed(&rt->e, rt->toks, n_prompt);
        d_ptime = now_s() - t0;
        if (logits) {
            d_gen = engine_generate(&rt->e, logits, DOCTOR_TOKENS, outcap_cb,
                                    &dout, &d_gtime);
            probe_ran = !dout.failed;
            probe_stop = rt->e.hit_stop;
        }
    }
    if (probe_ran && dout.buf) {
        dout.buf[dout.n] = 0;
        for (size_t i = 0; i < dout.n; i++)
            if (!strchr(" \t\r\n", dout.buf[i])) probe_empty = false;
        probe_has4 = strchr(dout.buf, '4') != NULL;
        static const char *const marks[] = {
            "<|im_start|>", "<|im_end|>", "[INST]", "<<SYS>>", "<|eot_id|>",
            "<start_of_turn>", "<|start_header_id|>", "<|end|>", "<|user|>",
            "<SPECIAL_", "<|start_of_role|>" };
        for (size_t i = 0; i < sizeof marks / sizeof *marks; i++)
            if (strstr(dout.buf, marks[i])) probe_leak = true;
    }
    char gname[128] = "";
    bool gpu_there = gpu_available(gname, (int)sizeof gname);
    size_t vfree = 0, vtotal = 0;
    bool vram_ok = gpu_there && gpu_mem_info(&vfree, &vtotal);
    bool asked_gpu = o->mp.gpu_mode != GPU_OFF;
    const char *backend =
#ifdef __APPLE__
        rt->m.gpu ? "metal" : "cpu";
#else
        rt->m.gpu ? "cuda" : "cpu";
#endif
    sbuf r = {0};
    sb_lit(&r, "{\"schema\":\"xyntetik.runner.doctor.v1\",\"runner\":{\"version\":\"" RUNNER_VERSION "\"},");
    // identity: what the file says it is, never where it lives
    const char *bn = strrchr(o->model_path, '/');
#ifdef _WIN32
    const char *bn2 = strrchr(o->model_path, '\\');
    if (bn2 && (!bn || bn2 > bn)) bn = bn2;
#endif
    bn = bn ? bn + 1 : o->model_path;
    sb_lit(&r, "\"model\":{\"file\":\"");
    sb_esc(&r, bn, strlen(bn));
    sb_fmt(&r, "\",\"bytes\":%llu,\"architecture\":\"", (unsigned long long)rt->m.file_size);
    sb_esc(&r, rt->m.arch, strlen(rt->m.arch));
    const char *gn = gguf_get_str(&rt->m.gf, "general.name", "");
    sb_lit(&r, "\",\"name\":\"");
    sb_esc(&r, gn, strlen(gn));
    sb_fmt(&r, "\",\"layers\":%d,\"trained_context\":%d,\"adapter\":%s},",
           rt->m.n_layer, rt->m.n_ctx_train, o->lora_path ? "true" : "false");
    sb_fmt(&r, "\"placement\":{\"requested\":\"%s\",\"backend\":\"%s\","
               "\"gpu_layers\":%d,\"layers\":%d,\"device\":\"",
           asked_gpu ? "auto" : "off", backend, rt->m.gpu ? rt->m.gpu_layers : 0,
           rt->m.n_layer);
    sb_esc(&r, gname, strlen(gname));
    sb_lit(&r, "\"");
    if (vram_ok)
        sb_fmt(&r, ",\"vram_total_bytes\":%llu,\"vram_free_bytes\":%llu",
               (unsigned long long)vtotal, (unsigned long long)vfree);
    sb_fmt(&r, "},\"context\":{\"tokens\":%d,\"kv\":\"%s\"},", rt->m.n_ctx,
           rt->m.kv_fp4 ? "fp4" : rt->m.kv_split ? "k8v4" : rt->m.kv_q8 ? "q8" : "f16");
    sb_fmt(&r, "\"template\":{\"name\":\"%s\",\"recognised\":%s,"
               "\"from_file\":%s,\"tool_family\":\"%s\",\"native_tool_protocol\":%s},",
           template_name(dt), dt == TMPL_LLAMA2_FALLBACK ? "false" : "true",
           have_tmpl_text ? "true" : "false", fam, native ? "true" : "false");
    sb_lit(&r, "\"sampling\":\"");
    sb_esc(&r, rt->sampling, strlen(rt->sampling));
    sb_fmt(&r, "\",\"memory\":{\"ram_total_bytes\":%llu,\"ram_available_bytes\":%llu},",
           (unsigned long long)plat_ram_bytes(),
           (unsigned long long)plat_ram_available_bytes());
    sb_fmt(&r, "\"timings\":{\"load_s\":%.3f,\"threads\":%d", rt->load_s,
           tpool_size(rt->m.tp));
    if (probe_ran)
        sb_fmt(&r, ",\"prompt_tokens\":%d,\"prompt_s\":%.3f,\"prompt_tok_s\":%.2f,"
                   "\"generated_tokens\":%d,\"gen_s\":%.3f,\"gen_tok_s\":%.2f",
               n_prompt, d_ptime, n_prompt / (d_ptime > 0 ? d_ptime : 1e-9),
               d_gen, d_gtime, d_gen / (d_gtime > 0 ? d_gtime : 1e-9));
    sb_fmt(&r, "},\"probe\":{\"ran\":%s,\"thinking\":\"off\",\"stopped_by_itself\":%s,"
               "\"answer_found\":%s,\"empty\":%s,\"template_markup_in_reply\":%s",
           probe_ran ? "true" : "false", probe_stop ? "true" : "false",
           probe_has4 ? "true" : "false", probe_empty ? "true" : "false",
           probe_leak ? "true" : "false");
    if (o->doctor_text && probe_ran && dout.buf) {
        sb_lit(&r, ",\"question\":\"");
        sb_esc(&r, question, sizeof question - 1);
        sb_lit(&r, "\",\"reply\":\"");
        sb_esc(&r, dout.buf, dout.n);
        sb_lit(&r, "\"");
    }
    sb_lit(&r, "},\"findings\":[");
    int nf = 0, worst = 0;   // 0 ok, 1 degraded, 2 broken
#define FINDING(sev, code, detail, next) do { \
        sb_fmt(&r, "%s{\"severity\":\"%s\",\"code\":\"%s\",\"detail\":\"%s\"," \
                   "\"next_step\":\"%s\"}", nf++ ? "," : "", \
               (sev) == 2 ? "broken" : "degraded", code, detail, next); \
        if ((sev) > worst) worst = (sev); } while (0)
    if (dt == TMPL_LLAMA2_FALLBACK)
        FINDING(2, "template_not_recognised",
                "the model's chat template is not one this build renders; llama-2 markup is used, which the model was not trained on",
                "pass --chat-template NAME for a supported family, or report the model so its template can be added");
    if (asked_gpu && gpu_there && !rt->m.gpu)
        FINDING(1, "gpu_not_used",
                "a GPU is present and was not used for this model; the load output above the report names the reason",
                "read the load lines for the refusal; a quantization or architecture without a device kernel runs on the CPU");
    else if (rt->m.gpu && rt->m.gpu_layers < rt->m.n_layer)
        FINDING(1, "partial_offload",
                "only some layers are on the GPU, so every token crosses between device and host",
                "use a smaller quantization or a shorter -c so the whole model fits, or accept the split");
    if (!asked_gpu && gpu_there)
        FINDING(1, "gpu_disabled_by_flag",
                "the run asked for --gpu off although a GPU is present",
                "drop --gpu off to let the model use the device");
    if (!probe_ran)
        FINDING(2, "probe_did_not_run",
                "the probe prompt could not be rendered, tokenized or fed in this context",
                "raise -c, and if it still fails report the model with this file");
    else {
        if (probe_leak)
            FINDING(2, "template_markup_in_reply",
                    "the reply contains chat-template markers, so turn framing or stop tokens do not match the model",
                    "check the template name above against the model card; try --chat-template");
        if (!probe_stop)
            FINDING(1, "did_not_stop",
                    "the model did not end its turn within the probe's 48 tokens",
                    "if replies run on in real use, the stop token or the template is wrong for this model");
        if (probe_empty)
            FINDING(2, "empty_reply",
                    "the model produced no visible text",
                    "check the template and that the file is an instruct or chat model");
        else if (!probe_has4)
            FINDING(1, "wrong_answer",
                    "the reply to a one-step arithmetic question does not contain the answer",
                    "a very small or heavily quantized model can miss it; otherwise suspect the template or the file");
        if (d_gen > 4 && d_gen / (d_gtime > 0 ? d_gtime : 1e-9) < 1.0)
            FINDING(1, "very_slow_decode",
                    "generation is under one token per second",
                    "check available RAM against the model size with --fit; a model larger than RAM pages from disk");
    }
    if (plat_ram_available_bytes() && rt->m.file_size > plat_ram_available_bytes() && !rt->m.gpu)
        FINDING(1, "file_larger_than_available_ram",
                "the model file is larger than the RAM available right now",
                "close other programs, pick a smaller quantization, or run --fit for the exact numbers");
#undef FINDING
    sb_fmt(&r, "],\"verdict\":\"%s\",\"shareable\":\"%s\"}\n",
           worst == 2 ? "broken" : worst == 1 ? "degraded" : "ok",
           o->doctor_text ? "contains the probe question and the model's reply; read before sharing"
                       : "no prompt or reply text; file name and machine sizes only");
    if (r.failed) {
        free(r.s); free(dout.buf);
        fprintf(stderr, "error: out of memory building the report\n");
        return 1;
    }
    fwrite(r.s, 1, r.n, stdout);
    free(r.s);
    free(dout.buf);
    return worst == 2 ? 2 : 0;
}

static int run_bench(cli_opts *o, cli_run *rt) {
    double ptime, gtime, t0;
    int n_prompt, n_gen;
    // A bench that stops at EOS measures the model's chattiness, not the
    // engine: instruct models that answer a bare prompt with instant EOS
    // reported "0.0 tok/s". Decode every requested token.
    rt->e.ignore_eos = true;
    // A ten-token default measured nothing that matters: on the first
    // outside install this bench looked healthy while a real 2,100-token
    // prompt took 89 s to reach its first word, because prefill — not
    // decode — was the wall. Absent an explicit -p/-f, benchmark a
    // realistic prefill instead, clamped to whatever context exists.
    enum { BENCH_PROMPT_TOKENS = 512 };
    char *p = NULL;
    if (o->prompt) {
        p = unescape(o->prompt);
    } else {
        static const char para[] =
            "The lighthouse keeper recorded the weather every morning: "
            "wind direction, cloud cover, the height of the swell against "
            "the north rocks, and the hour the fog lifted. ";
        size_t one = sizeof(para) - 1;
        size_t reps = (size_t)BENCH_PROMPT_TOKENS / 8 + 2;  // >= target tok
        p = malloc(one * reps + 1);
        if (p) {
            p[0] = 0;
            for (size_t r = 0; r < reps; r++) memcpy(p + r * one, para, one);
            p[one * reps] = 0;
        }
    }
    if (!p) {
        fprintf(stderr, "error: out of memory building benchmark prompt\n");
        return 1;
    }
    n_prompt = tok_encode(&rt->tok, p, rt->toks, (int)rt->tok_cap, !o->no_bos, true);
    if (n_prompt < 0) {
        fprintf(stderr, "error: out of memory tokenizing benchmark prompt\n");
        free(p);
        return 1;
    }
    if (!o->prompt) {
        // Truncating the synthesized prompt is safe (it is filler) and is
        // what keeps the default runnable on a small context.
        int want = BENCH_PROMPT_TOKENS;
        // leave the decode phase its tokens, or the rate it reports is
        // measured over a couple of steps on a small-context model
        int room = rt->m.n_ctx - (o->n_predict > 0 ? o->n_predict : 0) - 1;
        if (room < 1) room = rt->m.n_ctx - 1;
        if (want > room) want = room;
        if (n_prompt > want) n_prompt = want;
    }
    if (n_prompt == 0 || n_prompt >= rt->m.n_ctx) {
        fprintf(stderr, "error: benchmark prompt does not fit context\n");
        free(p);
        return 1;
    }
    t0 = now_s();
    float *logits = engine_feed(&rt->e, rt->toks, n_prompt);
    ptime = now_s() - t0;
    if (!logits) {
        fprintf(stderr, "error: benchmark prompt exceeds context\n");
        free(p);
        return 1;
    }
    n_gen = engine_generate(&rt->e, logits, o->n_predict, discard_cb, NULL, &gtime);
    char esc_model[1024];
    json_escape(o->model_path, strlen(o->model_path), esc_model, sizeof(esc_model));
    // prompt_s is time-to-first-token for this prompt length: the number
    // an editor-assistant user actually waits on, and the one a rate alone
    // hides. Both phases report seconds beside their rates.
    printf("{\"model\":\"%s\",\"arch\":\"%s\",\"context\":%d,"
           "\"gpu_layers\":%d,\"layers\":%d,\"prompt_tokens\":%d,"
           "\"generated_tokens\":%d,\"prompt_tok_s\":%.3f,"
           "\"gen_tok_s\":%.3f,\"prompt_s\":%.3f,\"gen_s\":%.3f}\n",
           esc_model, rt->m.arch, rt->m.n_ctx, rt->m.gpu_layers, rt->m.n_layer,
           n_prompt, n_gen,
           n_prompt / (ptime > 0 ? ptime : 1e-9),
           n_gen / (gtime > 0 ? gtime : 1e-9), ptime, gtime);
    free(p);
    return 0;
}

static int run_resume(cli_opts *o, cli_run *rt, session_image *simg, const char *load_path) {
    // R1.3: continue a suspended generation from its image, exactly
    const session_meta *sm = &simg->meta;
    char have[65];
    const char *why = NULL;
    if (!envelope_file_sha256(load_path, have) || strcmp(have, sm->model_sha256))
        why = "-m is not the model the image was made with (model sha256)";
    else if (rt->e.model_key != sm->model_key)
        why = "this engine's model key (context, KV type, geometry) is not the "
              "image's";
    else if (simg->n_vocab != rt->m.n_vocab ||
             simg->state_n != prefix_cache_entry_bytes(&rt->m, sm->n_tokens))
        why = "the image's state does not fit this model";
    if (!why && !engine_state_load(&rt->e, simg->tokens, sm->n_tokens, simg->state))
        why = "this model's KV layout (a ring or tied-V cache) cannot be resumed";
    if (why) {
        fprintf(stderr, "error: --resume %s: %s\n", o->resume_path, why);
        session_image_free(simg);
        return 1;
    }
    char bsha[65] = "";
    char *exe = plat_executable_path();
    if (exe) envelope_file_sha256(exe, bsha);
    free(exe);
    if (strcmp(bsha, sm->binary_sha256) != 0)
        fprintf(stderr, "session: resumed by another binary than the one that "
                "suspended it; the continuation is exact only on that one\n");
    engine_gen_resume(&rt->e, sm->max_new, sm->n_prompt, sm->generated);
    float *lg = malloc(sizeof(float) * (size_t)rt->m.n_vocab);
    if (!lg) { session_image_free(simg); return 1; }
    memcpy(lg, simg->logits, sizeof(float) * (size_t)rt->m.n_vocab);
    fprintf(stderr, "resumed %s: %d tokens, %d of %d generated%s\n",
            o->resume_path, sm->n_tokens, sm->generated, sm->max_new,
            o->fork_seed_given ? " (forked: new rng seed)" : "");
    double st0 = now_s();
    const float *last = NULL;
    bool live = session_steps(&rt->e, lg, o->suspend_after, stdout_cb, NULL, &last);
    bool suspended = live && rt->e.gen_count < rt->e.gen_max;
    int rc = 0;
    if (o->session_out) {
        if (live) {
            if (!session_image_out(o->session_out, &rt->e, load_path, last,
                                   sm->n_prompt, sm->max_new, o->json_mode,
                                   o->ignore_eos, o->schema_file))
                rc = 1;
        } else {
            fprintf(stderr, "error: session: the generation ended before the "
                    "image point; no image written\n");
            rc = 1;
        }
    }
    if (!suspended) engine_gen_end(&rt->e, stdout_cb, NULL, NULL);
    printf("\n");
    double gt = now_s() - st0;
    int n_new = rt->e.gen_count - sm->generated;
    if (rt->e.hit_stop && !o->json_mode) fprintf(stderr, "[end of text]\n");
    fprintf(stderr, "\nresumed gen: %d tok, %.2f tok/s\n", n_new,
            n_new / (gt > 0 ? gt : 1e-9));
    free(lg);
    session_image_free(simg);
    return rc;
}

static int run_oneshot(cli_opts *o, cli_run *rt, const char *load_path, bool wm_on,
                       const wm_key *wmk, const wm_state *cli_wm) {
    double ptime, gtime, t0;
    int n_prompt, n_gen;
    // one-shot completion
    char *p = unescape(o->prompt);
    n_prompt = tok_encode(&rt->tok, p, rt->toks, (int)rt->tok_cap, !o->no_bos, true);
    if (n_prompt < 0) { fprintf(stderr, "error: out of memory tokenizing prompt\n"); free(p); return 1; }
    if (n_prompt == 0) { fprintf(stderr, "error: empty prompt\n"); free(p); return 1; }
    if (n_prompt >= rt->m.n_ctx) {
        fprintf(stderr, "error: prompt is %d tokens but context is %d — "
                "rerun with -c %d or larger\n", n_prompt, rt->m.n_ctx,
                (n_prompt + o->n_predict + 1023) / 1024 * 1024);
        free(p);
        return 1;
    }
    if (o->verbose) {
        fprintf(stderr, "prompt tokens (%d):", n_prompt);
        for (int i = 0; i < n_prompt && i < 64; i++) fprintf(stderr, " %d", rt->toks[i]);
        fprintf(stderr, "\n");
    }
    t0 = now_s();
    float *logits = engine_feed(&rt->e, rt->toks, n_prompt);
    ptime = now_s() - t0;
    if (!logits) { fprintf(stderr, "error: prompt exceeds context\n"); free(p); return 1; }

    if (!o->prompt_file && !o->json_mode && !o->schema_file)
        printf("%s", p); // don't echo file/json/schema prompts
    fflush(stdout);
    // --transcript: the seed recorded is the value the sampler STARTS
    // from (it mutates as it draws), and the output bytes are captured
    // verbatim as they stream
    uint64_t t_seed = o->smp.rng;
    outcap_t ocap = { .echo = true };
    int s_rc = 0;
    if (o->session_out) {
        // R1.3: the step loop, stopped at --suspend-after or the budget,
        // and imaged there while the generation is live
        double st0 = now_s();
        engine_gen_begin(&rt->e, o->n_predict);
        const float *last = NULL;
        bool live = session_steps(&rt->e, logits, o->suspend_after, stdout_cb, NULL, &last);
        bool suspended = live && rt->e.gen_count < rt->e.gen_max;
        if (live) {
            if (!session_image_out(o->session_out, &rt->e, load_path, last, n_prompt,
                                   o->n_predict, o->json_mode, o->ignore_eos, o->schema_file))
                s_rc = 1;
        } else {
            // an ended generation has nothing left to resume; asked for an
            // image and given none is a failure, not a quiet success
            fprintf(stderr, "error: session: the generation ended before the "
                    "image point; no image written\n");
            s_rc = 1;
        }
        // a suspended document stays open: closing it is the resumed
        // run's business, at its own end
        n_gen = suspended ? rt->e.gen_count : engine_gen_end(&rt->e, stdout_cb, NULL, NULL);
        gtime = now_s() - st0;
    } else {
        n_gen = engine_generate(&rt->e, logits, o->n_predict,
                                o->transcript_path ? outcap_cb : stdout_cb,
                                o->transcript_path ? &ocap : NULL, &gtime);
    }
    printf("\n");
    if (rt->e.hit_stop && !o->json_mode) fprintf(stderr, "[end of text]\n");
    fprintf(stderr, "\nprompt: %d tok, %.2f tok/s | gen: %d tok, %.2f tok/s\n",
            n_prompt, n_prompt / (ptime > 0 ? ptime : 1e-9),
            n_gen, n_gen / (gtime > 0 ? gtime : 1e-9));
    model_debug_time_report();
    int t_rc = 0;
    if (o->transcript_path) {
        char prev_hash[65] = "";
        if (o->transcript_prev && !receipt_chain_hash(o->transcript_prev, prev_hash)) {
            fprintf(stderr, "error: --transcript-prev: cannot read the chain "
                            "hash of %s\n", o->transcript_prev);
            t_rc = 1;
            o->transcript_path = NULL;   // do not write an unlinked receipt
        }
        char gname[128] = "";
        if (rt->m.gpu) gpu_available(gname, sizeof gname);
        // R1.1.2: what shaped the output beyond the sampler, so --verify
        // replays it under the same constraints or refuses it
        char cons[256] = "", schema_sha[65] = "";
        size_t cn = 0;
        if (o->schema_file && !schema_file_digest(o->schema_file, schema_sha))
            snprintf(schema_sha, sizeof schema_sha, "unknown");
        if (o->schema_file)
            cn += (size_t)snprintf(cons + cn, sizeof cons - cn,
                                   "{\"kind\":\"json_schema\",\"sha256\":\"%s\"}",
                                   schema_sha);
        if (o->json_mode)
            cn += (size_t)snprintf(cons + cn, sizeof cons - cn,
                                   "%s{\"kind\":\"json_mode\"}", cn ? "," : "");
        if (o->ignore_eos)
            cn += (size_t)snprintf(cons + cn, sizeof cons - cn,
                                   "%s{\"kind\":\"ignore_eos\"}", cn ? "," : "");
        char cons_json[260] = "";
        if (cn) snprintf(cons_json, sizeof cons_json, "[%s]", cons);
        char wm_json[192] = "";
        if (wm_on)
            snprintf(wm_json, sizeof wm_json,
                     "{\"scheme\":\"%s\",\"key_id\":\"%s\",\"layers\":%d,"
                     "\"context\":%d,\"marked_tokens\":%d}", WM_SCHEME,
                     wmk->id, WM_LAYERS, WM_CONTEXT, cli_wm->marked);
        char *transcript_exe = plat_executable_path();
        transcript_info ti = {
            .out_path = o->transcript_path,
            .runner_version = RUNNER_VERSION,
            .executable_path = transcript_exe,
            .compiler = __VERSION__,
            .build_flavor = RUNNER_BUILD_FLAVOR,
#ifdef _WIN32
            .os = "windows",
#elif defined(__APPLE__)
            .os = "macos",
#else
            .os = "linux",
#endif
            .arch = RUNNER_BUILD_ARCH,
            .device = rt->m.gpu ? gname : "cpu",
            .gpu = rt->m.gpu != NULL,
            .gpu_layers = rt->m.gpu_layers,
            .threads = tpool_size(rt->m.tp), .n_ctx = rt->m.n_ctx,
            .n_batch = rt->m.n_batch, .kv_q8 = rt->m.kv_q8, .kv_fp4 = rt->m.kv_fp4,
            .kv_split = rt->m.kv_split,
            .model_path = load_path,
            .adapter_path = o->lora_path, .adapter_scale = o->lora_scale,
            .seed = t_seed,
            .temp = rt->e.smp->temp, .top_p = rt->e.smp->top_p,
            .min_p = rt->e.smp->min_p,
            .repeat_penalty = rt->e.smp->repeat_penalty,
            .top_k = rt->e.smp->top_k, .n_predict = o->n_predict,
            .bos = !o->no_bos,
            .prompt_text = p,
            .prompt_tokens = rt->toks, .n_prompt = n_prompt,
            .output_text = ocap.buf, .output_text_len = ocap.n,
            .output_tokens = rt->e.hist + n_prompt, .n_output = n_gen,
            .hit_stop = rt->e.hit_stop,
            .prev_hash = prev_hash[0] ? prev_hash : NULL,
            .sign_key_path = o->sign_key,
            .spec_source = rt->e.dm ? "model" : rt->e.mtp_on ? "mtp"
                         : rt->e.lookup_on ? "lookup" : NULL,
            .spec_rounds = rt->e.spec_st.rounds,
            .spec_drafted = rt->e.spec_st.drafted,
            .spec_accepted = rt->e.spec_st.accepted,
            .spec_lk_drafted = rt->e.spec_st.lk_drafted,
            .spec_lk_accepted = rt->e.spec_st.lk_accepted,
            .model_sig_json = rt->model_sig_json[0] ? rt->model_sig_json : NULL,
            .adapter_sig_json = rt->adapter_sig_json[0] ? rt->adapter_sig_json : NULL,
            .constraints_json = cons_json[0] ? cons_json : NULL,
            .watermark_json = wm_json[0] ? wm_json : NULL,
            .envelope_known = rt->env_known,
            .envelope_state = rt->env_state,
        };
        if (ocap.failed) {
            fprintf(stderr, "error: transcript: out of memory capturing "
                    "output\n");
            t_rc = 1;
        } else if (transcript_write(&ti))
            fprintf(stderr, "transcript -> %s\n", o->transcript_path);
        else
            t_rc = 1;
        free(transcript_exe);
        free(ocap.buf);
    }
    free(p);
    return t_rc | s_rc;
}

// interactive chat
// resolved and validated up with the serve checks, so it is a TMPL_* here
static int run_chat(cli_opts *o, cli_run *rt, int tmpl_override) {
    double gtime;
    int n_prompt, n_gen;
    int tmpl = -1;
    if (tmpl_override >= 0) tmpl = tmpl_override;
    if (tmpl < 0)
        tmpl = template_detect(gguf_get_str(&rt->m.gf, "tokenizer.chat_template", NULL), &rt->tok);
    template_bind_think_tags(tmpl, &rt->m.think_open, &rt->m.think_close);
    if (!o->system_prompt)
        o->system_prompt = template_default_system(
            tmpl, gguf_get_str(&rt->m.gf, "tokenizer.chat_template", NULL));
    if (!o->system_prompt) o->system_prompt = "You are a helpful assistant.";
    fprintf(stderr, "chat mode (template: %s) — Ctrl-D or /exit to quit\n\n",
            template_name(tmpl));

    bool first_turn = true;
    for (;;) {
        fprintf(stderr, "> ");
        bool line_oom = false;
        char *line = read_line(stdin, &line_oom);
        if (!line) {
            if (line_oom) fprintf(stderr, "[out of memory reading input]\n");
            break;
        }
        if (!strcmp(line, "/exit") || !strcmp(line, "/quit")) { free(line); break; }
        if (!line[0]) { free(line); continue; }

        chat_msg msgs[2];
        int n_msgs = 0;
        if (first_turn && o->system_prompt[0])
            msgs[n_msgs++] = (chat_msg){
                .role = "system", .content = o->system_prompt,
            };
        msgs[n_msgs++] = (chat_msg){ .role = "user", .content = line };
        // Measured, never estimated — the same rule the server holds itself
        // to. render_messages truncates the TAIL, so a fixed buffer that a
        // long --system overruns does not fail the turn, it deletes the
        // question and the generation header and asks something else.
        render_state rs;
        char *rendered = render_prompt_state_alloc(tmpl, msgs, n_msgs, true, o->thinking,
                                                   NULL, strlen(line) +
                                                   strlen(o->system_prompt) + 1024, &rs);
        free(line);   // the render is the only reader; every path below is safe
        if (!rendered) {
            fprintf(stderr, "[out of memory rendering the prompt — turn skipped]\n");
            continue;
        }
        n_prompt = tok_encode_prompt(&rt->tok, rendered, rt->toks, (int)rt->tok_cap,
                                     first_turn && !o->no_bos);
        free(rendered);
        first_turn = false;

        if (n_prompt < 0) {
            fprintf(stderr, "[out of memory tokenizing input — turn skipped]\n");
            continue;
        }
        if (rt->e.pos + n_prompt >= rt->m.n_ctx) {
            fprintf(stderr, "[context full — restart to continue]\n");
            break;
        }
        float *logits = engine_feed(&rt->e, rt->toks, n_prompt);
        if (!logits) { fprintf(stderr, "[context full]\n"); break; }

        jsonv_init(&rt->e.jv); // fresh JSON constraint per turn
        chat_out co = { .in_think = false };
        // The turn starts where the rendered prompt left it (render_state),
        // the seam the server reads too: a prompt that opened a reasoning
        // segment (Harmony's primed analysis channel, the `<think>` of Qwen
        // 3.5 and 3.8, Granite 4.2, Nemotron) is reasoning from its first
        // token, and a content-first split would never see the open marker
        // and print the model's reasoning as the answer.
        if (rs.reasoning_open)
            think_init_reasoning(&co.ts, rt->m.think_open, rt->m.think_close);
        else
            think_init(&co.ts, rt->m.think_open, rt->m.think_close);
        n_gen = engine_generate(&rt->e, logits, o->n_predict, chat_cb, &co, &gtime);
        think_finish(&co.ts, chat_emit, &co);
        if (co.in_think) fputs("\n[/thinking]", stdout);
        think_free(&co.ts);
        printf("\n");
        fprintf(stderr, "[%d tok, %.1f tok/s]\n",
                n_gen, n_gen / (gtime > 0 ? gtime : 1e-9));
    }
    return 0;
}

int main(int argc, char **argv) {
    char *owned_prompt = NULL;
    // the model, tokenizer and engine the modes run on, and what the load
    // decided about them; cli_cleanup frees them on every exit after the load
    cli_run rt_state;
    memset(&rt_state, 0, sizeof rt_state);
    rt_state.env_state = ENV_UNCLASSIFIED;
    cli_run *rt = &rt_state;
    session_image simg;
    memset(&simg, 0, sizeof simg);
    cli_verify vr = { .rec = NULL, .rsig = RSIG_NONE };
    cli_opts opts;
    cli_opts_init(&opts);
    cli_opts *o = &opts;
    int rc;   // the exit status a step that ends the process hands back

    // A bare `runner` — double-clicked, or typed with no arguments — starts
    // the tray on the platforms that have one. The conditions are deliberately
    // narrow: literally zero arguments (so no scripted invocation can ever be
    // affected — scripts always pass something), a real terminal on both
    // stdin and stdout (so pipes and CI probes still get usage text), and a
    // platform with a tray backend (Linux keeps printing usage). --no-tray is
    // the standing opt-out: it suppresses this launch and nothing else, for
    // wrappers that want bare-invocation behavior guaranteed text-only.
#if defined(__APPLE__) || defined(_WIN32)
    if (argc == 1 && RUNNER_TTY(stdin) && RUNNER_TTY(stdout))
        return tray_main();
#endif

    if (!cli_parse(o, argc, argv, &rc)) return rc;

    if (!cli_chat_flags_ok(o)) return 1;
    if (run_record_utility(o, &rc)) return rc;
    // R1.8.1: one key for everything this process marks, replays or scores
    wm_key wmk;
    memset(&wmk, 0, sizeof wmk);
    bool wm_on = false;
    char *detect_text = NULL;
    if (cli_watermark_setup(o, &wmk, &wm_on, &detect_text, &rc)) return rc;
    if (o->keygen_path) return run_keygen(o);
    if (!cli_draft_flags_ok(o)) return 1;
    plat_parent_watch(o->parent_pid);
    if (o->n_threads <= 0 && o->reserve_cpu_pct > 0) {
        o->n_threads = plat_cpu_count() * o->reserve_cpu_pct / 100;
        if (o->n_threads < 1) o->n_threads = 1;
    }
    if (o->fit_path) return run_fit_check(o->fit_path, o->mp.n_ctx, &o->mp);

    if (o->shadow_mode) return run_shadow_mode(o->model_path, o->shadow_yes);
    if (o->caps) return run_caps();
    if (cli_resolve_inputs(o, argv[0], detect_text != NULL, &owned_prompt, &rc))
        return rc;
    if (cli_prepare_session(o, &simg, &rc)) return rc;
    if (cli_prepare_verify(o, &wmk, wm_on, &vr, &rc)) return rc;
    cli_resolve_threads(o);

    f16_init();

    // `-m name=path` with --parallel > 1 is one named model, not a swap set.
    // Swap mode is genuinely single-slot (ensure_resident loads slots[0]
    // alone), but with one entry there is nothing to swap to, so the caller
    // keeps their slots and gives up /unload and --ttl instead. server_run
    // still receives the name=path form and keeps the name for /v1/models;
    // what has to happen HERE is the preload, which needs the bare path.
    const char *eq_one = strchr(o->model_path, '=');
    bool one_named = o->serve && o->parallel > 1 && eq_one && eq_one != o->model_path &&
                     eq_one[1] && !strchr(o->model_path, ',');
    const char *load_path = one_named ? eq_one + 1 : o->model_path;
    bool registry = o->serve && eq_one != NULL && !one_named;

    int tmpl_override = -1;
    if (!cli_model_flags_ok(o, &tmpl_override)) return 1;

    if (o->context_out) return run_context_surgery(o, load_path);
    if (o->merge_out) return run_merge_lora(o);
    if (o->quant_out) return run_quantize(o);
    if (!cli_configure_serving(o)) return 1;
    if (!cli_rewrite_flags_ok(o)) return 1;

    if (!cli_require_eval_ok(o, load_path)) return 1;

    cli_tray_follow(o);

    oms_policy signing = { o->model_sig, o->model_pubkey, o->require_signed_model };
    if (cli_load_model(o, rt, load_path, registry, tmpl_override, &signing, &rc))
        return rc;

    if (o->serve) {
        rc = run_serve(o, rt, &wmk, wm_on, registry, tmpl_override, &signing);
        free(owned_prompt);
        return rc;
    }

    // cli_wm outlives the engine that points at it
    wm_state cli_wm = { .key = wm_on ? &wmk : NULL };
    if (!cli_engine_setup(o, rt, wm_on, &cli_wm)) rc = 1;
    else if (o->train_path || o->dpo_path) rc = run_train(o, rt, tmpl_override, load_path);
    else if (o->verify_path) rc = run_verify_replay(o, rt, &vr, load_path);
    else if (detect_text) rc = run_detect_text(o, rt, &wmk, detect_text);
    else if (o->decide_path) rc = run_decide(o, rt);
    else if (o->score) rc = run_score(o, rt);
    else if (o->doctor) rc = run_doctor(o, rt, tmpl_override);
    else if (o->bench_json) rc = run_bench(o, rt);
    else if (o->resume_path) rc = run_resume(o, rt, &simg, load_path);
    else if (!o->interactive) rc = run_oneshot(o, rt, load_path, wm_on, &wmk, &cli_wm);
    else rc = run_chat(o, rt, tmpl_override);
    // Every failure from here on frees what the success paths free. Written
    // once so a new error return cannot forget it, and so `make debug` is the
    // leak gate the comment on cli_cleanup claims it is: measured under
    // LeakSanitizer on Linux before this, an unreadable --json-schema leaked
    // 237,668 bytes in 19 allocations and an over-long prompt 30,186 in 21,
    // while the successful run was clean.
    cli_cleanup(&rt->e, rt->toks, &rt->tok, &rt->m);
    free(owned_prompt);
    return rc;
}
