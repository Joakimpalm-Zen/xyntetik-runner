// `runner derive` (R17.6.1): make a file from a parent, measure it against a
// named reference, and write its record and its card. docs/derive-design.md
// holds the design and the decisions it was built to.
//
// Every step is an existing mode of this binary, run as a sub-process of
// it: the file is made by `--rewrite` and scored by `--score --score-top`,
// so a derived file is made and measured exactly as those modes make and
// measure one, with their own checks and records. What lives here is the
// order of the steps, the comparison, and the two documents.
//
// The comparison is scripts/gold-logits.py's, so a parent-file number and a
// publisher-reference number mean the same thing: KL over the reference's
// top 20, both sides renormalised, the other side floored at its smallest
// listed logprob minus 2 where it does not list a reference token; top-1
// agreement by argmax; margin-qualified = the reference's top-1 to top-2 gap
// above 0.5 nats. Position pos of the window is score row
// n_special + pos - 1 (the row that predicts the token at pos).
#include "derive.h"
#include "compat.h"
#include "gguf.h"
#include "json.h"
#include "envelope.h"
#include "fpclass.h"
#include "runner.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define DERIVE_TOP 20
// the house bar, stated against the publisher's float32 model only
#define BAR_MEAN_KLD 0.05
#define BAR_MQ_TOP1 97.0

// the recipe: rewrite flags passed through as given, recorded by key
static const struct { const char *flag, *key; } RECIPE[] = {
    { "--quant", "quant" }, { "--type-plan", "type_plan" }, { "--imatrix", "imatrix" },
    { "--prune-experts", "prune_experts" }, { "--remove-sublayer", "remove_sublayer" },
    { "--merge-lora", "merge_lora" },
};
#define N_RECIPE (int)(sizeof RECIPE / sizeof RECIPE[0])

typedef struct {
    const char *parent, *out, *corpus, *reference_hf;
    const char *ref_args[8];   // --reference-arg: passed to gold-logits.py as given
    const char *license;       // --license: for a parent that declares none
    const char *base_model;    // --base-model REPO[@REV]: the publisher's repository
    const char *credits[8];    // --credit TEXT: inputs the card credits
    const char *compares[8];   // --compare LABEL=GOLD.json[:SIDE][:BYTES]
    int n_compares;
    int n_credits;
    int n_ref_args;
    const char *ctx, *threads, *gpu;
    const char *recipe[N_RECIPE];   // by RECIPE index; NULL = not given
    int start, stride, count;
    bool corpus_given;
    // --target-size: a plan ranked from --quant-error (sized_plan)
    long long target_size;
    const char *high, *low, *match;
    bool group_tensor;
} derive_opts;

// what sized_plan decided, for the record
typedef struct {
    char *report, *plan;      // OUT.quant-error.jsonl, OUT.type-plan.json
    long long save_bytes;     // the bytes the demotions had to save
    char **demoted;           // the demoted groups, cheapest first
    int n_demoted;
} sized_result;

typedef struct {
    int positions, qualified;
    double mean_kld, median_kld, max_kld, top1_pct, mq_top1_pct;
    bool have_median, have_qualified;
    const char *reference_class;    // publisher reference only
    const char *reference_dtype;    // its arithmetic, as gold-logits reported it
} derive_measure;

static void derive_usage(FILE *f) {
    fprintf(f,
        "usage: runner derive -m PARENT.gguf -o OUT.gguf RECIPE [options]\n\n"
        "Make a file from PARENT with the rewrite path, measure it against a\n"
        "named reference, and write OUT, OUT.derive.json (the record) and\n"
        "OUT.README.md (a Hugging Face card).\n\n"
        "recipe (at least one; each is the --rewrite flag of the same name):\n"
        "  --quant TYPE  --type-plan PLAN  --imatrix FILE  --prune-experts PLAN\n"
        "  --remove-sublayer LIST  --merge-lora ADAPTER\n"
        "  --target-size BYTES --high T --low T [--match S] [--group block|tensor]\n"
        "                 selective precision: cost every tensor with --quant-error,\n"
        "                 demote the groups that add the least error per byte saved\n"
        "                 from T high to T low until the file fits BYTES, as\n"
        "                 scripts/type-plan-by-sensitivity.py ranks them (with\n"
        "                 --imatrix, the errors are importance-weighted)\n\n"
        "measurement:\n"
        "  --reference hf:DIR  the publisher's model in float32 through\n"
        "                 scripts/gold-logits.py (Python with torch and\n"
        "                 transformers); the only reference a card may state the\n"
        "                 bar against. Default: the parent file in Runner's\n"
        "                 arithmetic, named so on the card, with no bar verdict\n"
        "  --reference-arg ARG  with --reference: pass ARG to gold-logits.py as\n"
        "                 given, repeatable (--upcast-at-use, or --offload-dir DIR\n"
        "                 --max-memory GIB, for a reference too large for float32\n"
        "                 in memory)\n"
        "  --base-model REPO[@REV]  the publisher's repository (and revision): the\n"
        "                 card's base_model, and the reference's name on the card\n"
        "  --credit TEXT  credit an input on the card (repeatable), e.g. an imatrix\n"
        "  --compare LABEL=GOLD.json[:SIDE][:BYTES]  with --reference: another file's\n"
        "                 gold-logits run in the same window (side a or b, its size),\n"
        "                 on the card with the paired tests (repeatable)\n"
        "  --license ID   the licence of a parent that declares no general.license\n"
        "                 (its source's terms; refused if the parent declares another)\n"
        "  --corpus FILE  the text (default tests/fixtures/gold-corpus-2k.txt)\n"
        "  --window S:T:N positions S + n*T for n < N (default 8:15:300)\n"
        "  -c N, -t N, --gpu X  passed to the score runs (--gpu default off)\n\n"
        "The parent must declare general.license: the card copies it.\n\n"
        "  runner derive --card-from OUT.derive.json [-o CARD.md]\n"
        "                 rewrite a card from its record alone (no measurement)\n\n"
        "Environment: RUNNER_GOLD_PYTHON (default python3), RUNNER_GOLD_LOGITS\n"
        "(default scripts/gold-logits.py beside this binary).\n");
}

// files shipped beside the binary, spelled in the platform's separator so
// the path joined to plat_executable_dir() (which ends in '\\' on Windows)
// is one style throughout
#ifdef _WIN32
#define BESIDE_CORPUS "tests\\fixtures\\gold-corpus-2k.txt"
#define BESIDE_GOLD   "scripts\\gold-logits.py"
#else
#define BESIDE_CORPUS "tests/fixtures/gold-corpus-2k.txt"
#define BESIDE_GOLD   "scripts/gold-logits.py"
#endif

static char *join(const char *a, const char *b) {
    size_t n = strlen(a) + strlen(b) + 1;
    char *s = malloc(n);
    if (s) snprintf(s, n, "%s%s", a, b);
    return s;
}

static char *suffixed(const char *path, const char *suffix) { return join(path, suffix); }

static bool parse_window(const char *s, derive_opts *o) {
    int a, b, c;
    char tail;
    if (sscanf(s, "%d:%d:%d%c", &a, &b, &c, &tail) != 3 || a < 1 || b < 1 || c < 1) return false;
    if ((long long)a + (long long)(c - 1) * b + 2 > 100000000LL) return false;   // 1e8 tokens
    o->start = a; o->stride = b; o->count = c;
    return true;
}

static int parse_args(int argc, char **argv, derive_opts *o) {
    memset(o, 0, sizeof *o);
    o->start = 8; o->stride = 15; o->count = 300;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { derive_usage(stdout); return 0; }
        if (i + 1 >= argc) {
            fprintf(stderr, "error: runner derive: %s needs a value (runner derive --help)\n", a);
            return 1;
        }
        const char *v = argv[++i];
        int r = -1;
        for (int k = 0; k < N_RECIPE; k++) if (!strcmp(a, RECIPE[k].flag)) r = k;
        if (r >= 0) o->recipe[r] = v;
        else if (!strcmp(a, "-m")) o->parent = v;
        else if (!strcmp(a, "-o") || !strcmp(a, "--out")) o->out = v;
        else if (!strcmp(a, "--corpus")) o->corpus = v;
        else if (!strcmp(a, "--license")) o->license = v;
        else if (!strcmp(a, "--base-model")) o->base_model = v;
        else if (!strcmp(a, "--compare")) {
            if (o->n_compares == 8) { fprintf(stderr, "error: at most 8 --compare\n"); return 1; }
            o->compares[o->n_compares++] = v;
        }
        else if (!strcmp(a, "--credit")) {
            if (o->n_credits == 8) { fprintf(stderr, "error: at most 8 --credit\n"); return 1; }
            o->credits[o->n_credits++] = v;
        }
        else if (!strcmp(a, "--reference-arg")) {
            if (o->n_ref_args == (int)(sizeof o->ref_args / sizeof o->ref_args[0])) {
                fprintf(stderr, "error: at most 8 --reference-arg\n");
                return 1;
            }
            o->ref_args[o->n_ref_args++] = v;
        }
        else if (!strcmp(a, "-c")) o->ctx = v;
        else if (!strcmp(a, "-t")) o->threads = v;
        else if (!strcmp(a, "--gpu")) o->gpu = v;
        else if (!strcmp(a, "--high")) o->high = v;
        else if (!strcmp(a, "--low")) o->low = v;
        else if (!strcmp(a, "--match")) o->match = v;
        else if (!strcmp(a, "--group")) {
            if (strcmp(v, "block") && strcmp(v, "tensor")) {
                fprintf(stderr, "error: --group is block or tensor (got %s)\n", v);
                return 1;
            }
            o->group_tensor = !strcmp(v, "tensor");
        } else if (!strcmp(a, "--target-size")) {
            char *end;
            o->target_size = strtoll(v, &end, 10);
            if (*end || o->target_size < 1) {
                fprintf(stderr, "error: --target-size is a byte count (got %s)\n", v);
                return 1;
            }
        }
        else if (!strcmp(a, "--reference")) {
            if (strncmp(v, "hf:", 3) || !v[3]) {
                fprintf(stderr, "error: --reference takes hf:DIR, the publisher's model "
                        "directory\n");
                return 1;
            }
            o->reference_hf = v + 3;
        } else if (!strcmp(a, "--window")) {
            if (!parse_window(v, o)) {
                fprintf(stderr, "error: --window is START:STRIDE:COUNT, each at least 1 "
                        "(got %s)\n", v);
                return 1;
            }
        } else {
            fprintf(stderr, "error: runner derive: unknown option %s (runner derive --help)\n", a);
            return 1;
        }
    }
    if (o->n_compares && !o->reference_hf) {
        fprintf(stderr, "error: --compare sets other files beside this one against the publisher "
                "reference and needs --reference hf:DIR\n");
        return 1;
    }
    if (o->n_ref_args && !o->reference_hf) {
        fprintf(stderr, "error: --reference-arg passes arguments to the publisher reference and "
                "needs --reference hf:DIR\n");
        return 1;
    }
    if (!o->parent || !o->out) {
        fprintf(stderr, "error: runner derive needs -m PARENT and -o OUT (runner derive --help)\n");
        return 1;
    }
    if (o->target_size) {
        // the plan is computed, and its types are --high and --low
        if (!o->high || !o->low) {
            fprintf(stderr, "error: --target-size needs --high T and --low T, the candidates' "
                    "type and the type a demoted group gets\n");
            return 1;
        }
        if (o->recipe[0] || o->recipe[1]) {
            fprintf(stderr, "error: --target-size writes its own type plan from --high and "
                    "--low; drop %s\n", o->recipe[0] ? "--quant" : "--type-plan");
            return 1;
        }
    } else if (o->high || o->low || o->match || o->group_tensor) {
        fprintf(stderr, "error: --high, --low, --match and --group shape --target-size's "
                "plan and need --target-size\n");
        return 1;
    }
    bool any = o->target_size > 0;
    for (int k = 0; k < N_RECIPE; k++) any |= o->recipe[k] != NULL;
    if (!any) {
        fprintf(stderr, "error: runner derive needs a recipe (--quant, --type-plan, --imatrix, "
                "--prune-experts, --remove-sublayer, --merge-lora or --target-size): a file "
                "with no change "
                "from its parent has nothing to measure\n");
        return 1;
    }
    return -1;
}

// a sub-run of this binary; its exit status
static int run_self(const char *self, const char **args) {
    const char *argv[48];
    int n = 0;
    argv[n++] = self;
    for (; *args && n < 47; args++) argv[n++] = *args;
    argv[n] = NULL;
    return plat_spawn_wait(argv);
}

// `--score --score-top 20` of model over the window's head, into out
static bool score(const char *self, const derive_opts *o, const char *model,
                  const char *out, int n_tokens) {
    char ntok[16];
    snprintf(ntok, sizeof ntok, "%d", n_tokens);
    const char *a[24];
    int n = 0;
    a[n++] = "-m"; a[n++] = model; a[n++] = "--score"; a[n++] = "--score-top"; a[n++] = "20";
    a[n++] = "-f"; a[n++] = o->corpus; a[n++] = "--score-out"; a[n++] = out;
    a[n++] = "--score-tokens"; a[n++] = ntok;
    a[n++] = "--gpu"; a[n++] = o->gpu ? o->gpu : "off";
    // without -c the context is the window's span, so the default context
    // of a short-context model is never what limits the measurement
    a[n++] = "-c"; a[n++] = o->ctx ? o->ctx : ntok;
    if (o->threads) { a[n++] = "-t"; a[n++] = o->threads; }
    a[n] = NULL;
    int rc = run_self(self, a);
    if (rc != 0) fprintf(stderr, "error: runner derive: scoring %s failed (exit %d)\n", model, rc);
    return rc == 0;
}

static jv *read_json(const char *path) {
    size_t n = 0;
    char *s = plat_read_file(path, (size_t)1 << 30, &n);
    if (!s) { fprintf(stderr, "error: cannot read %s\n", path); return NULL; }
    jv *v = json_parse(s, n);
    free(s);
    if (!v) fprintf(stderr, "error: %s is not JSON\n", path);
    return v;
}

// The window must lie inside the scored text: the last position's token is
// row k + last, so the score needs k + last + 1 tokens.
static bool window_fits(const derive_opts *o, jv *sc) {
    jv *toks = jv_get(sc, "tokens");
    int have = toks && toks->type == J_ARR ? toks->n : 0;
    int k = (int)jv_num(jv_get(sc, "n_special"), 0);
    int need = k + o->start + o->count * o->stride + 1;   // gold-logits.py's span
    if (have < need) {
        fprintf(stderr, "error: the window %d:%d:%d needs %d tokens of the corpus and it "
                "gives %d; give a longer --corpus or a smaller --window\n",
                o->start, o->stride, o->count, need, have);
        return false;
    }
    return true;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

// one top row as (id, logprob) pairs; false when it is not that shape
static bool row_pairs(jv *row, int *ids, double *lps, int *n) {
    if (!row || row->type != J_ARR || row->n > DERIVE_TOP * 4) return false;
    *n = 0;
    for (int i = 0; i < row->n; i++) {
        jv *p = row->items[i];
        if (!p || p->type != J_ARR || p->n != 2) return false;
        if (!p->items[0] || p->items[0]->type != J_NUM || !p->items[1] ||
            p->items[1]->type != J_NUM || !fp_isfinite(p->items[1]->num))
            return false;
        ids[*n] = (int)p->items[0]->num;
        lps[(*n)++] = p->items[1]->num;
    }
    return true;
}

static double logsumexp(const double *x, int n) {
    double m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    double s = 0;
    for (int i = 0; i < n; i++) s += exp(x[i] - m);
    return m + log(s);
}

static int argmax(const double *x, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i;
    return b;
}

// The parent-file comparison: ref and run are the two score documents.
static bool compare_scores(const derive_opts *o, jv *ref, jv *run, derive_measure *m) {
    jv *rt = jv_get(ref, "tokens"), *ft = jv_get(run, "tokens");
    bool same = rt && ft && rt->type == J_ARR && ft->type == J_ARR && rt->n == ft->n;
    for (int i = 0; same && i < rt->n; i++)
        same = jv_num(rt->items[i], -1) == jv_num(ft->items[i], -2);
    int k = (int)jv_num(jv_get(ref, "n_special"), 0);
    if (!same || k != (int)jv_num(jv_get(run, "n_special"), 0)) {
        fprintf(stderr, "error: the parent and the file tokenize the corpus differently; a "
                "rewrite keeps the vocabulary, so this file is not comparable by position\n");
        return false;
    }
    jv *rtop = jv_get(ref, "top"), *ftop = jv_get(run, "top");
    if (!rtop || !ftop || rtop->type != J_ARR || ftop->type != J_ARR) {
        fprintf(stderr, "error: a score file has no top rows (--score-top)\n");
        return false;
    }
    double *kl = malloc(sizeof(double) * (size_t)o->count);
    if (!kl) return false;
    int agree = 0, qual = 0, qual_hit = 0;
    for (int n = 0; n < o->count; n++) {
        int e = k + o->start + n * o->stride - 1;
        int gid[DERIVE_TOP * 4], oid[DERIVE_TOP * 4], ng, no;
        double g[DERIVE_TOP * 4], ov[DERIVE_TOP * 4];
        if (e >= rtop->n || e >= ftop->n || !row_pairs(rtop->items[e], gid, g, &ng) ||
            !row_pairs(ftop->items[e], oid, ov, &no) || ng < 2) {
            fprintf(stderr, "error: score row %d is missing or malformed\n", e);
            free(kl);
            return false;
        }
        double floor_lp = -20.0;
        if (no) {
            floor_lp = ov[0];
            for (int j = 1; j < no; j++) if (ov[j] < floor_lp) floor_lp = ov[j];
            floor_lp -= 2.0;
        }
        double oa[DERIVE_TOP * 4];
        for (int i = 0; i < ng; i++) {
            oa[i] = floor_lp;
            for (int j = 0; j < no; j++) if (oid[j] == gid[i]) { oa[i] = ov[j]; break; }
        }
        double gz = logsumexp(g, ng), oz = logsumexp(oa, ng), d = 0;
        for (int i = 0; i < ng; i++) d += exp(g[i] - gz) * ((g[i] - gz) - (oa[i] - oz));
        kl[n] = d;
        bool hit = no > 0 && gid[argmax(g, ng)] == oid[argmax(ov, no)];
        agree += hit;
        double s[DERIVE_TOP * 4];
        memcpy(s, g, sizeof(double) * (size_t)ng);
        qsort(s, (size_t)ng, sizeof(double), cmp_double);
        if (s[ng - 1] - s[ng - 2] > 0.5) { qual++; qual_hit += hit; }
    }
    double sum = 0, mx = 0;
    for (int n = 0; n < o->count; n++) { sum += kl[n]; if (kl[n] > mx) mx = kl[n]; }
    qsort(kl, (size_t)o->count, sizeof(double), cmp_double);
    m->positions = o->count;
    m->mean_kld = sum / o->count;
    m->max_kld = mx;
    m->median_kld = o->count % 2 ? kl[o->count / 2]
                                 : 0.5 * (kl[o->count / 2 - 1] + kl[o->count / 2]);
    m->have_median = true;
    m->top1_pct = 100.0 * agree / o->count;
    m->qualified = qual;
    m->have_qualified = true;
    m->mq_top1_pct = qual ? 100.0 * qual_hit / qual : 0.0;
    free(kl);
    return true;
}

static const char *gold_python(void) {
    const char *p = getenv("RUNNER_GOLD_PYTHON");
#ifdef _WIN32
    return p && *p ? p : "python";
#else
    return p && *p ? p : "python3";
#endif
}

// The publisher reference's Python, checked before anything is made. A
// RUNNER_GOLD_LOGITS stand-in brings its own needs, so only the shipped
// script is held to torch and transformers.
static bool gold_ready(const char *script, bool shipped) {
    const char *py = gold_python();
    if (!plat_file_readable(script)) {
        fprintf(stderr, "error: --reference needs scripts/gold-logits.py (%s is not readable); "
                "set RUNNER_GOLD_LOGITS to its path\n", script);
        return false;
    }
    const char *probe[] = { py, "-c", shipped ? "import torch, transformers" : "pass", NULL };
    if (plat_spawn_wait(probe) != 0) {
        fprintf(stderr, "error: --reference hf:DIR runs the publisher's model in float32 and "
                "needs Python with torch and transformers (%s could not import them); "
                "install them with `pip install torch transformers accelerate`, or set "
                "RUNNER_GOLD_PYTHON to a Python that has them\n", py);
        return false;
    }
    return true;
}

// a finite number member of obj, or false
static bool num_of(jv *obj, const char *key, double *out) {
    jv *v = jv_get(obj, key);
    if (!v || v->type != J_NUM || !fp_isfinite(v->num)) return false;
    *out = v->num;
    return true;
}

static bool gold_measure(const derive_opts *o, const char *script, const char *file_score,
                         int n_special, const char *gold_out, derive_measure *m) {
    char s[16], t[16], c[16];
    snprintf(s, sizeof s, "%d", o->start);
    snprintf(t, sizeof t, "%d", o->stride);
    snprintf(c, sizeof c, "%d", o->count);
    const char *a[32];
    int n = 0;
    a[n++] = gold_python(); a[n++] = script; a[n++] = "--hf"; a[n++] = o->reference_hf;
    a[n++] = "--corpus"; a[n++] = o->corpus; a[n++] = "--score-a"; a[n++] = file_score;
    a[n++] = "--min-prefix"; a[n++] = s; a[n++] = "--stride"; a[n++] = t;
    a[n++] = "--max-positions"; a[n++] = c; a[n++] = "--top-n"; a[n++] = "20";
    a[n++] = "--out"; a[n++] = gold_out;
    if (o->threads) { a[n++] = "--threads"; a[n++] = o->threads; }
    // Runner prepended BOS to the text it scored; the reference must see the
    // same prefix even where the publisher's tokenizer adds none
    if (n_special > 0) a[n++] = "--force-bos";
    for (int i = 0; i < o->n_ref_args; i++) a[n++] = o->ref_args[i];
    a[n] = NULL;
    int rc = plat_spawn_wait(a);
    if (rc != 0) {
        fprintf(stderr, "error: the publisher reference run failed (exit %d)\n", rc);
        return false;
    }
    jv *g = read_json(gold_out);
    if (!g) return false;
    jv *sum = jv_get(g, "summary"), *side = jv_get(jv_get(sum, "sides"), "a");
    if (!side) {
        fprintf(stderr, "error: %s has no summary.sides.a\n", gold_out);
        jv_free(g);
        return false;
    }
    m->positions = (int)jv_num(jv_get(sum, "positions"), 0);
    bool whole = num_of(side, "mean_kld_vs_gold", &m->mean_kld) &&
                 num_of(side, "max_kld_vs_gold", &m->max_kld) &&
                 num_of(side, "top1_agreement_pct", &m->top1_pct) &&
                 num_of(side, "margin_qualified_top1_pct", &m->mq_top1_pct);
    const char *cls = jv_str(jv_get(sum, "reference_class"), NULL);
    m->reference_class = cls ? strdup(cls) : NULL;
    const char *dt = jv_str(jv_get(sum, "reference_dtype"), NULL);
    m->reference_dtype = dt ? strdup(dt) : NULL;
    jv *rows = jv_get(g, "rows");
    if (rows && rows->type == J_ARR && rows->n) {
        double *kl = malloc(sizeof(double) * (size_t)rows->n);
        int nk = 0;
        m->qualified = 0;
        for (int i = 0; i < rows->n; i++) {
            if (jv_num(jv_get(rows->items[i], "gold_margin"), 0) > 0.5) m->qualified++;
            double v;
            if (kl && num_of(jv_get(rows->items[i], "a"), "kld", &v)) kl[nk++] = v;
        }
        m->have_qualified = true;
        if (kl && nk) {
            qsort(kl, (size_t)nk, sizeof(double), cmp_double);
            m->median_kld = nk % 2 ? kl[nk / 2] : 0.5 * (kl[nk / 2 - 1] + kl[nk / 2]);
            m->have_median = true;
        }
        free(kl);
    }
    jv_free(g);
    if (!whole || m->positions < 1) {
        fprintf(stderr, "error: %s does not carry the measurement\n", gold_out);
        return false;
    }
    return true;
}

static long long file_bytes(const char *path) {
    plat_stat_t st;
    return plat_stat(path, &st) == 0 ? (long long)st.st_size : -1;
}

static void sb_num(sbuf *b, double d) {
    char t[32];
    if (!fp_isfinite(d)) { sb_lit(b, "null"); return; }
    json_format_number(d, t);
    sb_lit(b, t);
}

// a YAML scalar: plain when it is a simple token, else double-quoted
static void yaml_scalar(sbuf *b, const char *s) {
    bool plain = *s != '\0';
    for (const char *c = s; *c && plain; c++)
        plain = (*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
                (*c >= '0' && *c <= '9') || *c == '-' || *c == '.' || *c == '_' || *c == '/';
    if (plain) { sb_lit(b, s); return; }
    sb_str(b, s);
}

static bool write_text(const char *path, const sbuf *b) {
    if (b->failed) { fprintf(stderr, "error: out of memory writing %s\n", path); return false; }
    FILE *f = plat_fopen(path, "wb");
    bool ok = f && fwrite(b->s, 1, b->n, f) == b->n;
    if (f && fclose(f) != 0) ok = false;
    if (!ok) fprintf(stderr, "error: cannot write %s\n", path);
    return ok;
}

typedef struct {
    char license[256], name[256], repo[512];
    int file_type;
    bool license_given;   // from --license: the parent declares none
} parent_meta;

// general.license (required), general.name, the first base model's repo id
// when it is a Hugging Face URL, general.file_type
static bool read_parent(const char *path, const char *given, parent_meta *p) {
    memset(p, 0, sizeof *p);
    gguf_file g;
    if (!gguf_open_header(&g, path)) {
        fprintf(stderr, "error: cannot read the parent %s\n", path);
        return false;
    }
    const char *lic = gguf_get_str(&g, "general.license", NULL);
    const char *name = gguf_get_str(&g, "general.name", NULL);
    const char *url = gguf_get_str(&g, "general.base_model.0.repo_url", NULL);
    p->file_type = (int)gguf_get_u32(&g, "general.file_type", 0xFFFFFFFFu);
    if (lic) snprintf(p->license, sizeof p->license, "%s", lic);
    if (name) snprintf(p->name, sizeof p->name, "%s", name);
    const char *hf = "https://huggingface.co/";
    if (url && !strncmp(url, hf, strlen(hf)))
        snprintf(p->repo, sizeof p->repo, "%s", url + strlen(hf));
    gguf_close(&g);
    // a declared licence is the file's; --license only fills an absent one,
    // so derive never relicenses
    if (*p->license && given && strcmp(given, p->license)) {
        fprintf(stderr, "error: the parent declares general.license %s and --license says %s; "
                "a derived file keeps its parent's license\n", p->license, given);
        return false;
    }
    if (!*p->license && given && *given) {
        snprintf(p->license, sizeof p->license, "%s", given);
        p->license_given = true;
    }
    if (!*p->license) {
        fprintf(stderr, "error: the parent declares no general.license; the card copies the "
                "parent's license, so derive will not make a file it cannot license (give the "
                "source's terms with --license ID)\n");
        return false;
    }
    return true;
}

// One costed candidate group: a block's matching tensors, or one tensor.
typedef struct {
    char *name;
    char **tensors;
    int n, cap;
    double added;             // error at --low minus error at --high
    long long saved;          // bytes at --high minus bytes at --low
    bool weighted, unweighted;
    double cost;
} dgroup;

static int cmp_group(const void *a, const void *b) {
    const dgroup *x = a, *y = b;
    if (x->cost != y->cost) return x->cost < y->cost ? -1 : 1;
    return strcmp(x->name, y->name);
}

static int cmp_str(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

// a report row's {bytes, error} for one type, or false
static bool type_cost(jv *row, const char *type, double *bytes, double *err) {
    jv *t = jv_get(jv_get(row, "types"), type);
    return t && t->type == J_OBJ && num_of(t, "bytes", bytes) && num_of(t, "error", err);
}

// --target-size: run --quant-error over the parent, rank the candidate
// groups as scripts/type-plan-by-sensitivity.py does (added error per saved
// byte, cheapest first, ties by name), demote until the predicted file fits,
// and write the plan. The prediction is the parent's bytes with every
// candidate at --high, from the report's own byte counts against the
// parent's stored tensor sizes.
static bool sized_plan(const char *self, const derive_opts *o, long long pbytes,
                       sized_result *res) {
    res->report = suffixed(o->out, ".quant-error.jsonl");
    res->plan = suffixed(o->out, ".type-plan.json");
    char types[64];
    snprintf(types, sizeof types, "%s,%s", o->high, o->low);
    const char *a[16];
    int n = 0;
    a[n++] = "-m"; a[n++] = o->parent; a[n++] = "--quant-error"; a[n++] = res->report;
    a[n++] = "--quant"; a[n++] = types;
    if (o->recipe[2]) { a[n++] = "--imatrix"; a[n++] = o->recipe[2]; }
    if (o->threads) { a[n++] = "-t"; a[n++] = o->threads; }
    a[n] = NULL;
    int rc = run_self(self, a);
    if (rc != 0) {
        fprintf(stderr, "error: runner derive: --quant-error failed (exit %d)\n", rc);
        return false;
    }
    size_t len = 0;
    char *text = plat_read_file(res->report, 0, &len);
    gguf_file g;
    if (!text || !gguf_open_header(&g, o->parent)) {
        fprintf(stderr, "error: cannot read %s or the parent\n", res->report);
        free(text);
        return false;
    }
    const char *match = o->match ? o->match : "";
    dgroup *gs = NULL;
    int ng = 0, gcap = 0;
    long long delta = 0;   // all candidates at --high, against the parent
    bool ok = true;
    for (char *line = text, *next; ok && line && *line; line = next) {
        next = strchr(line, '\n');
        if (next) *next++ = '\0';
        if (!*line) continue;
        jv *row = json_parse(line, strlen(line));
        const char *tn = jv_str(jv_get(row, "tensor"), NULL);
        if (!row || !tn) { fprintf(stderr, "error: %s: a malformed line\n", res->report); ok = false; }
        else if (strstr(tn, match)) {
            double hb, he, lb, le;
            bool has_h = type_cost(row, o->high, &hb, &he), has_l = type_cost(row, o->low, &lb, &le);
            gguf_tensor *t = gguf_find_tensor(&g, tn);
            if (!has_h || !has_l) {
                fprintf(stderr, "error: %s has no %s entry (the type does not fit its rows, or the "
                        "report did not cost it)\n", tn, has_h ? o->low : o->high);
                ok = false;
            } else if (!t) {
                fprintf(stderr, "error: the report names %s and the parent has no such tensor\n", tn);
                ok = false;
            } else {
                delta += (long long)hb - (long long)t->nbytes;
                // the group: "blk.N" for a ^blk\.(\d+)\. name (the digits as
                // spelled), else the tensor itself
                char key[160];
                size_t dig = 4;
                while (!strncmp(tn, "blk.", 4) && tn[dig] >= '0' && tn[dig] <= '9') dig++;
                if (!o->group_tensor && dig > 4 && tn[dig] == '.')
                    snprintf(key, sizeof key, "%.*s", (int)dig, tn);
                else
                    snprintf(key, sizeof key, "%s", tn);
                int gi = -1;
                for (int k = 0; k < ng; k++) if (!strcmp(gs[k].name, key)) gi = k;
                if (gi < 0) {
                    if (ng == gcap) {
                        gcap = gcap ? gcap * 2 : 64;
                        gs = realloc(gs, sizeof *gs * (size_t)gcap);
                    }
                    memset(&gs[ng], 0, sizeof gs[ng]);
                    gs[ng].name = strdup(key);
                    gi = ng++;
                }
                dgroup *d = &gs[gi];
                if (d->n == d->cap) {
                    d->cap = d->cap ? d->cap * 2 : 8;
                    d->tensors = realloc(d->tensors, sizeof *d->tensors * (size_t)d->cap);
                }
                d->tensors[d->n++] = strdup(tn);
                d->added += le - he;
                d->saved += (long long)hb - (long long)lb;
                if (jv_bool(jv_get(row, "weighted"), false)) d->weighted = true;
                else d->unweighted = true;
            }
        }
        jv_free(row);
    }
    gguf_close(&g);
    free(text);
    if (!ok) return false;
    if (!ng) {
        fprintf(stderr, "error: no tensor in the --quant-error report matches '%s'\n", match);
        return false;
    }
    bool w = false, u = false;
    for (int k = 0; k < ng; k++) { w |= gs[k].weighted; u |= gs[k].unweighted; }
    if (w && u) {
        fprintf(stderr, "error: some candidates' errors are imatrix-weighted and some are not, "
                "and the two do not compare; give an imatrix that covers them or narrow --match\n");
        return false;
    }
    for (int k = 0; k < ng; k++) {
        if (gs[k].saved <= 0) {
            fprintf(stderr, "error: demoting %s to %s saves no bytes\n", gs[k].name, o->low);
            return false;
        }
        gs[k].cost = gs[k].added / (double)gs[k].saved;
    }
    qsort(gs, (size_t)ng, sizeof *gs, cmp_group);
    long long at_high = pbytes + delta;
    res->save_bytes = at_high > o->target_size ? at_high - o->target_size : 0;
    long long saved = 0;
    int nd = 0;
    while (nd < ng && saved < res->save_bytes) saved += gs[nd++].saved;
    if (saved < res->save_bytes) {
        fprintf(stderr, "error: --target-size %lld is out of reach: every candidate at %s still "
                "leaves about %lld bytes\n", o->target_size, o->low, at_high - saved);
        return false;
    }
    sbuf b = {0};
    sb_lit(&b, "{\"default\":");
    sb_str(&b, *match ? "keep" : o->high);
    sb_lit(&b, ",\"rules\":[");
    res->demoted = calloc((size_t)(nd ? nd : 1), sizeof *res->demoted);
    bool first = true;
    for (int k = 0; k < nd; k++) {
        res->demoted[res->n_demoted++] = gs[k].name;
        qsort(gs[k].tensors, (size_t)gs[k].n, sizeof *gs[k].tensors, cmp_str);
        for (int t = 0; t < gs[k].n; t++) {
            sb_lit(&b, first ? "{\"match\":" : ",{\"match\":");
            sb_str(&b, gs[k].tensors[t]);
            sb_lit(&b, ",\"type\":");
            sb_str(&b, o->low);
            sb_lit(&b, "}");
            first = false;
        }
    }
    if (*match) {
        sb_lit(&b, first ? "{\"match\":" : ",{\"match\":");
        sb_str(&b, match);
        sb_lit(&b, ",\"type\":");
        sb_str(&b, o->high);
        sb_lit(&b, "}");
    }
    sb_lit(&b, "]}\n");
    bool wrote = write_text(res->plan, &b);
    free(b.s);
    fprintf(stderr, "derive: demoted %d of %d groups to %s, saving %lld bytes\n",
            nd, ng, o->low, saved);
    return wrote;
}

// A string member as a base name ("" when absent): the card names files, never
// host paths.
static const char *rec_base(jv *obj, const char *key) {
    const char *v = jv_str(jv_get(obj, key), NULL);
    return v ? plat_path_base(v) : "";
}

static double rec_num(jv *obj, const char *key, double dflt) {
    double v;
    return num_of(obj, key, &v) ? v : dflt;
}

// Paired tests, ported from scripts/gold-logits.py so a card's numbers are
// the ones the script prints: McNemar's exact two-sided test on discordant
// pairs, and the normal-approximation Wilcoxon signed-rank on per-position
// differences (ties averaged, zeros dropped; fewer than 5 is no test).
static double mcnemar_exact(int b, int c) {
    int n = b + c, m = b < c ? b : c;
    if (n == 0) return 1.0;
    double tail = 0;
    for (int k = 0; k <= m; k++)
        tail += exp(lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0) - n * log(2.0));
    return tail * 2.0 < 1.0 ? tail * 2.0 : 1.0;
}

static const double *g_absd;
static int cmp_absd(const void *a, const void *b) {
    double x = fabs(g_absd[*(const int *)a]), y = fabs(g_absd[*(const int *)b]);
    return (x > y) - (x < y);
}

static bool wilcoxon_signed_rank(const double *diffs, int n_all, double *z_out, double *p_out) {
    double *d = malloc(sizeof(double) * (size_t)(n_all ? n_all : 1));
    int *ord = malloc(sizeof(int) * (size_t)(n_all ? n_all : 1));
    double *rk = malloc(sizeof(double) * (size_t)(n_all ? n_all : 1));
    int n = 0;
    bool ok = false;
    if (d && ord && rk) {
        for (int i = 0; i < n_all; i++) if (diffs[i] != 0.0) d[n++] = diffs[i];
        if (n >= 5) {
            for (int i = 0; i < n; i++) ord[i] = i;
            g_absd = d;
            qsort(ord, (size_t)n, sizeof(int), cmp_absd);
            for (int i = 0; i < n;) {
                int j = i;
                while (j + 1 < n && fabs(d[ord[j + 1]]) == fabs(d[ord[i]])) j++;
                double r = (i + j) / 2.0 + 1.0;
                for (int k = i; k <= j; k++) rk[ord[k]] = r;
                i = j + 1;
            }
            double w = 0;
            for (int i = 0; i < n; i++) if (d[i] > 0) w += rk[i];
            double mu = n * (n + 1) / 4.0, sd = sqrt(n * (n + 1) * (2.0 * n + 1) / 24.0);
            if (sd > 0) {
                *z_out = (w - mu) / sd;
                *p_out = erfc(fabs(*z_out) / sqrt(2.0));
                ok = true;
            }
        }
    }
    free(d); free(ord); free(rk);
    return ok;
}

// One --compare: LABEL=GOLD.json[:SIDE][:BYTES] against this file's own gold
// run, over the positions both share. Appends one JSON object to `out`.
// diffs are this file's KLD minus the other's, so z > 0 means the other file
// is closer; McNemar counts margin-qualified positions (the reference's top
// two more than 0.5 nats apart) one got right and the other wrong.
static bool compare_json(const char *ours_gold, const char *spec, sbuf *out) {
    const char *eq = strchr(spec, '=');
    if (!eq || eq == spec) {
        fprintf(stderr, "error: --compare is LABEL=GOLD.json[:SIDE][:BYTES] (got %s)\n", spec);
        return false;
    }
    char label[160], path[1024], side[4] = "a";
    long long bytes = -1;
    snprintf(label, sizeof label, "%.*s", (int)(eq - spec), spec);
    snprintf(path, sizeof path, "%s", eq + 1);
    for (int pass = 0; pass < 2; pass++) {   // trailing :BYTES, then :SIDE
        char *colon = strrchr(path, ':');
        if (!colon) break;
        const char *t = colon + 1;
        bool digits = *t != '\0';
        for (const char *q = t; *q; q++) digits &= *q >= '0' && *q <= '9';
        if (digits && bytes < 0) { bytes = atoll(t); *colon = '\0'; }
        else if (!strcmp(t, "a") || !strcmp(t, "b")) { snprintf(side, sizeof side, "%s", t); *colon = '\0'; }
        else break;
    }
    jv *og = read_json(ours_gold), *tg = read_json(path);
    jv *orows = jv_get(og, "rows"), *trows = jv_get(tg, "rows");
    if (!orows || !trows || orows->type != J_ARR || trows->type != J_ARR) {
        fprintf(stderr, "error: --compare needs gold-logits runs with per-position rows (%s, %s)\n",
                ours_gold, path);
        jv_free(og); jv_free(tg);
        return false;
    }
    int cap = orows->n, n = 0, qual = 0, b = 0, c = 0, top_o = 0, q_o = 0;
    double *diffs = malloc(sizeof(double) * (size_t)(cap ? cap : 1));
    double *tk = malloc(sizeof(double) * (size_t)(cap ? cap : 1));
    double sum_t = 0;
    for (int i = 0; diffs && tk && i < orows->n; i++) {
        jv *ro = orows->items[i];
        double pos = rec_num(ro, "pos", -1);
        jv *rt = NULL;
        for (int j = 0; j < trows->n && !rt; j++)
            if (rec_num(trows->items[j], "pos", -2) == pos) rt = trows->items[j];
        jv *so = jv_get(ro, "a"), *st = jv_get(rt, side);
        double ko, kt;
        if (!rt || !num_of(so, "kld", &ko) || !num_of(st, "kld", &kt)) continue;
        bool ao = jv_bool(jv_get(so, "agree"), false), at = jv_bool(jv_get(st, "agree"), false);
        diffs[n] = ko - kt;
        tk[n++] = kt;
        sum_t += kt;
        top_o += at;
        if (rec_num(ro, "gold_margin", 0) > 0.5) {
            qual++;
            q_o += at;
            b += ao && !at;
            c += at && !ao;
        }
    }
    // the paired tests mean something only over the same positions: a run
    // from another window is refused rather than paired over the overlap
    if (!diffs || !tk || n != orows->n || trows->n != orows->n) {
        fprintf(stderr, "error: --compare %s: %d of this file's %d positions pair with its %d rows; "
                "the other file must be measured in the same window\n", label, n, orows->n, trows->n);
        free(diffs); free(tk); jv_free(og); jv_free(tg);
        return false;
    }
    double z = 0, pw = 1;
    bool have_w = n && wilcoxon_signed_rank(diffs, n, &z, &pw);
    double med = 0;
    if (n) {
        qsort(tk, (size_t)n, sizeof(double), cmp_double);
        med = n % 2 ? tk[n / 2] : 0.5 * (tk[n / 2 - 1] + tk[n / 2]);
    }
    double mean = n ? sum_t / n : 0, mqp = qual ? 100.0 * q_o / qual : 0;
    sb_lit(out, "{\"label\":"); sb_str(out, label);
    sb_lit(out, ",\"gold\":"); sb_str(out, plat_path_base(path));
    sb_fmt(out, ",\"side\":\"%s\",\"bytes\":", side);
    if (bytes >= 0) sb_fmt(out, "%lld", bytes); else sb_lit(out, "null");
    sb_fmt(out, ",\"positions\":%d,\"mean_kld\":", n); sb_num(out, mean);
    sb_lit(out, ",\"median_kld\":"); sb_num(out, med);
    sb_lit(out, ",\"top1_agreement_pct\":"); sb_num(out, n ? 100.0 * top_o / n : 0);
    sb_lit(out, ",\"margin_qualified_top1_pct\":"); sb_num(out, mqp);
    sb_lit(out, ",\"wilcoxon_z\":"); if (have_w) sb_num(out, z); else sb_lit(out, "null");
    sb_lit(out, ",\"wilcoxon_p\":"); if (have_w) sb_num(out, pw); else sb_lit(out, "null");
    sb_fmt(out, ",\"mcnemar_this_only\":%d,\"mcnemar_other_only\":%d,\"mcnemar_p\":", b, c);
    sb_num(out, mcnemar_exact(b, c));
    sb_fmt(out, ",\"fails_bar\":%s}", mean > BAR_MEAN_KLD || mqp < BAR_MQ_TOP1 ? "true" : "false");
    free(diffs); free(tk);
    jv_free(og); jv_free(tg);
    return n > 0;
}


// The card from a derive record alone: what the file is and is not first, the
// unflattering metric first, the reference named exactly, base names only.
// derive writes every card through here, and `runner derive --card-from REC`
// rewrites one with a newer runner's wording without measuring again.
// A file the record names, as recorded, or (the record moved off the machine
// that measured it) by the same name beside the record.
static const char *beside_record(const char *rec_path, const char *recorded, char *buf, size_t n) {
    FILE *f = recorded ? plat_fopen(recorded, "rb") : NULL;
    if (f || !recorded) { if (f) fclose(f); return recorded; }
    const char *b = recorded;
    for (const char *q = recorded; *q; q++) if (*q == '/' || *q == '\\') b = q + 1;
    const char *d = rec_path, *e = NULL;
    for (const char *q = rec_path; *q; q++) if (*q == '/' || *q == '\\') e = q;
    if (e) snprintf(buf, n, "%.*s/%s", (int)(e - d), d, b);
    else snprintf(buf, n, "%s", b);
    f = plat_fopen(buf, "rb");
    if (!f) return recorded;
    fclose(f);
    return buf;
}

// The text the measurement read, as the card names it: the project's own
// corpus described, any other by its name.
static void put_corpus(sbuf *c, const char *base) {
    if (!strcmp(base, "gold-corpus-2k.txt"))
        sb_lit(c, "the project's gold corpus (`gold-corpus-2k.txt`: English technical "
               "documentation with code, English prose and one Swedish passage)");
    else
        sb_fmt(c, "`%s`", base);
}

static bool card_from_record(const char *rec_path, const char *card_path,
                             const char *base_fill, const char *const *credit_fill, int n_credit_fill,
                             const char *const *cmp_fill, int n_cmp_fill) {
    jv *r = read_json(rec_path);
    if (!r) return false;
    if (strcmp(jv_str(jv_get(r, "schema_version"), ""), "xyntetik.runner.derive.v1")) {
        fprintf(stderr, "error: %s is not a runner derive record\n", rec_path);
        jv_free(r);
        return false;
    }
    jv *par = jv_get(r, "parent"), *fil = jv_get(r, "file"), *ref = jv_get(r, "reference");
    jv *meas = jv_get(r, "measurement"), *rec = jv_get(r, "recipe"), *win = jv_get(meas, "window");
    bool publisher = !strcmp(jv_str(jv_get(r, "reference_kind"), ""), "publisher-float32");
    const char *ob = rec_base(fil, "path"), *pb = rec_base(par, "path");
    const char *license = jv_str(jv_get(par, "license"), "");
    // the publisher's repository, REPO or REPO@REV: the record's, or one
    // --card-from gives a record that lacks it
    const char *repo = jv_str(jv_get(par, "base_model_repo"), NULL);
    if (!repo) repo = base_fill;
    char repo_id[256] = "", repo_rev[128] = "";
    if (repo) {
        const char *at = strchr(repo, '@');
        snprintf(repo_id, sizeof repo_id, "%.*s", at ? (int)(at - repo) : (int)strlen(repo), repo);
        if (at) snprintf(repo_rev, sizeof repo_rev, "%s", at + 1);
    }
    bool given = !strcmp(jv_str(jv_get(par, "license_source"), ""), "--license");
    int file_type = (int)rec_num(par, "file_type", -1);
    long long pbytes = (long long)rec_num(par, "bytes", -1), fbytes = (long long)rec_num(fil, "bytes", -1);
    int positions = (int)rec_num(meas, "positions", 0);
    double mean = rec_num(meas, "mean_kld", 0), mx = rec_num(meas, "max_kld", 0);
    double top1 = rec_num(meas, "top1_agreement_pct", 0), mqp = rec_num(meas, "margin_qualified_top1_pct", 0);
    jv *bar = jv_get(meas, "bar");
    bool passes = bar && jv_bool(jv_get(bar, "passes"), false);
    sbuf c = {0};
    sb_lit(&c, "---\nlicense: ");
    yaml_scalar(&c, license);
    sb_lit(&c, "\n");
    if (*repo_id) {
        sb_lit(&c, "base_model: ");
        yaml_scalar(&c, repo_id);
        sb_lit(&c, "\n");
    }
    sb_lit(&c, "tags:\n- gguf\n- runner-derive\n---\n\n");
    // what the file is and is not for first, then the weakest number, then
    // the verdict (the card rules: limits first, the unflattering metric
    // before the flattering one)
    const char *corpus = rec_base(meas, "corpus");
    char of[300];
    if (*repo_id) snprintf(of, sizeof of, " of `%s`", repo_id);
    else of[0] = '\0';
    sb_fmt(&c, "`%s` is a %.1f GB GGUF%s, made from `%s` by `runner derive`, for local inference "
           "in GGUF engines. ", ob, fbytes / 1e9, of, pb);
    if (publisher)
        sb_lit(&c, "It is not the publisher's model, only measured against it, and it was measured "
               "on one text only: ");
    else
        sb_lit(&c, "It was measured against the parent file in Runner's arithmetic, not the "
               "publisher's float32 model, so this card states no bar verdict, and on one text "
               "only: ");
    put_corpus(&c, corpus);
    sb_fmt(&c, ", %d positions. No task benchmark was run.\n\n", positions);
    sb_fmt(&c, "Weakest number first: top-1 agreement with the %s is %.2f%%",
           publisher ? "publisher's float32 model" : "parent file", top1);
    if (top1 < 100.0)
        sb_fmt(&c, ": at about one position in %.0f the file's most likely next token differs from "
               "the reference's", 100.0 / (100.0 - top1));
    sb_lit(&c, ".\n\n");
    if (publisher)
        sb_fmt(&c, "This file %s the bar: mean KLD %.4f against the publisher's float32 model "
               "(bar %.2f) and margin-qualified top-1 %.2f%% (bar %.0f%%).\n\n",
               passes ? "passes" : "fails", mean, BAR_MEAN_KLD, mqp, BAR_MQ_TOP1);
    else
        sb_fmt(&c, "Against the parent: mean KLD %.4f, max KLD %.4f.\n\n", mean, mx);
    // other files measured in the same window: the record's, or computed now
    // from --card-from's --compare against the record's own gold run
    jv *cmps = jv_get(r, "comparisons"), *cmp_owned = NULL;
    if ((!cmps || cmps->type != J_ARR || !cmps->n) && n_cmp_fill && publisher) {
        sbuf cj = {0};
        sb_lit(&cj, "[");
        bool okc = true;
        for (int i = 0; okc && i < n_cmp_fill; i++) {
            if (i) sb_lit(&cj, ",");
            char near[1200];
            okc = compare_json(beside_record(rec_path, jv_str(jv_get(meas, "reference_score"), ""), near,
                                             sizeof near), cmp_fill[i], &cj);
        }
        sb_lit(&cj, "]");
        if (!okc) { free(cj.s); free(c.s); jv_free(r); return false; }
        cmp_owned = json_parse(cj.s, cj.n);
        free(cj.s);
        cmps = cmp_owned;
    }
    if (cmps && cmps->type == J_ARR && cmps->n) {
        sb_lit(&c, "## Against other files in the same window\n\n"
               "Each file measured against the same reference at the same positions; paired "
               "tests against this file. Files closer than this one come first.\n\n");
        // three passes: closer than this file, not distinguishable, further
        for (int pass = 0; pass < 3; pass++)
            for (int i = 0; i < cmps->n; i++) {
                jv *x = cmps->items[i];
                double km = rec_num(x, "mean_kld", 0), pw = rec_num(x, "wilcoxon_p", 1);
                double xmq = rec_num(x, "margin_qualified_top1_pct", 0), pm = rec_num(x, "mcnemar_p", 1);
                double xb = rec_num(x, "bytes", -1);
                int cls = pw < 0.05 ? (km < mean ? 0 : 2) : 1;
                if (cls != pass) continue;
                char size[96] = "";
                if (xb > 0)
                    snprintf(size, sizeof size, " (%.2f GB, %.2f GB %s than this file)", xb / 1e9,
                             fabs(xb - fbytes) / 1e9, xb > fbytes ? "larger" : "smaller");
                const char *lab = jv_str(jv_get(x, "label"), "?");
                if (cls == 0)
                    sb_fmt(&c, "- `%s`%s is closer to the float32 model than this file on mean KLD: "
                           "%.4f against %.4f (paired Wilcoxon p %.2g).", lab, size, km, mean, pw);
                else if (cls == 1)
                    sb_fmt(&c, "- `%s`%s and this file are not distinguishable on mean KLD: %.4f "
                           "against %.4f (paired Wilcoxon p %.2g).", lab, size, km, mean, pw);
                else
                    sb_fmt(&c, "- This file is closer to the float32 model than `%s`%s on mean KLD: "
                           "%.4f against %.4f (paired Wilcoxon p %.2g).", lab, size, mean, km, pw);
                sb_fmt(&c, " Margin-qualified top-1: %.2f%% against this file's %.2f%% (%d positions "
                       "only this file got, %d only `%s` got; McNemar p %.2g, %s).", xmq, mqp,
                       (int)rec_num(x, "mcnemar_this_only", 0), (int)rec_num(x, "mcnemar_other_only", 0),
                       lab, pm, pm < 0.05 ? "a difference" : "a tie");
                if (jv_bool(jv_get(x, "fails_bar"), false)) sb_fmt(&c, " `%s` fails the bar.", lab);
                sb_lit(&c, "\n");
            }
        sb_lit(&c, "\n");
    }
    jv_free(cmp_owned);
    sb_lit(&c, "## Size\n\n| | bytes |\n|---|---:|\n");
    sb_fmt(&c, "| parent `%s` | %lld |\n| this file | %lld (%.1f%% of the parent) |\n\n",
           pb, pbytes, fbytes, pbytes > 0 ? 100.0 * (double)fbytes / (double)pbytes : 0.0);
    double target;
    if (num_of(rec, "target_size", &target)) {
        jv *dem = jv_get(rec, "demoted");
        int nd = dem && dem->type == J_ARR ? dem->n : 0;
        sb_fmt(&c, "The recipe's target was %lld bytes: %d group%s demoted from %s to %s, the "
               "ones that add the least round-trip error per byte saved.\n\n", (long long)target,
               nd, nd == 1 ? "" : "s", jv_str(jv_get(rec, "high"), "?"), jv_str(jv_get(rec, "low"), "?"));
    }
    sb_lit(&c, "## Recipe\n\n```sh\n");
    {
        // the reference by repository and revision where the record knows it,
        // not by the local folder it was read from, so a reader can reproduce it
        const char *cmd = jv_str(jv_get(r, "command"), "");
        char from[600];
        snprintf(from, sizeof from, " --reference hf:%s", rec_base(ref, "hf"));
        const char *hit = publisher && repo && *repo ? strstr(cmd, from) : NULL;
        if (hit) {
            sb_put(&c, cmd, (size_t)(hit - cmd));
            sb_fmt(&c, " --reference hf:%s", repo);
            sb_lit(&c, hit + strlen(from));
        } else {
            sb_lit(&c, cmd);
        }
    }
    sb_lit(&c, "\n```\n\n## Fidelity\n\n"
           "| reference | positions | mean KLD | median KLD | max KLD | top-1 | margin-qualified top-1 |\n"
           "|---|---:|---:|---:|---:|---:|---:|\n");
    char med[32] = "not measured", mq[48];
    double medv;
    if (num_of(meas, "median_kld", &medv)) snprintf(med, sizeof med, "%.4f", medv);
    double qn;
    if (num_of(meas, "margin_qualified_positions", &qn))
        snprintf(mq, sizeof mq, "%.2f%% of %d", mqp, (int)qn);
    else
        snprintf(mq, sizeof mq, "%.2f%%", mqp);
    sb_fmt(&c, "| %s | %d | %.4f | %s | %.4f | %.2f%% | %s |\n\n",
           publisher ? "publisher, float32" : "parent file, Runner", positions, mean, med, mx, top1, mq);
    sb_lit(&c, "KL is over the reference's 20 most likely tokens, both sides renormalised; a "
           "position is margin-qualified when the reference's top two are more than 0.5 nats "
           "apart. ");
    sb_fmt(&c, "Positions %d + n x %d for n < %d of `%s`.\n\n## Reference\n\n",
           (int)rec_num(win, "start", 0), (int)rec_num(win, "stride", 0), (int)rec_num(win, "count", 0),
           rec_base(meas, "corpus"));
    if (publisher) {
        // the reference's arithmetic as the instrument reported it: the
        // record's, or (a record from before it carried one) the gold run's
        const char *dt = jv_str(jv_get(ref, "dtype"), NULL);
        jv *gold = NULL;
        if (!dt) {
            const char *gp = jv_str(jv_get(meas, "reference_score"), NULL);
            size_t n = 0;
            char *gs = gp ? plat_read_file(gp, (size_t)1 << 30, &n) : NULL;
            gold = gs ? json_parse(gs, n) : NULL;
            free(gs);
            dt = jv_str(jv_get(jv_get(gold, "summary"), "reference_dtype"), NULL);
        }
        const char *how = "in float32";
        if (dt && strstr(dt, "cast at use") && strstr(dt, "bfloat16"))
            how = "in float32 arithmetic, BF16 weights upcast at use,";
        else if (dt && strstr(dt, "cast at use"))
            how = "in float32 arithmetic, stored weights upcast at use,";
        const char *cls = jv_str(jv_get(ref, "class"), NULL);
        char name[512];
        if (*repo_id && *repo_rev) snprintf(name, sizeof name, "`%s` at `%s`", repo_id, repo_rev);
        else if (*repo_id) snprintf(name, sizeof name, "`%s`", repo_id);
        else snprintf(name, sizeof name, "`%s`", rec_base(ref, "hf"));
        sb_fmt(&c, "The publisher's model %s%s%s%s, run %s by scripts/gold-logits.py. "
               "The bar (mean KLD at most %.2f, margin-qualified top-1 at least %.0f%%) is "
               "stated against this reference only.\n\n",
               name, cls ? " (" : "", cls ? cls : "", cls ? ")" : "", how,
               BAR_MEAN_KLD, BAR_MQ_TOP1);
        jv_free(gold);
    } else {
        bool full = file_type == 0 || file_type == 1 || file_type == 32;
        sb_fmt(&c, "The parent file `%s` (sha256 `%s`), scored by Runner on the CPU. ", pb,
               jv_str(jv_get(par, "sha256"), ""));
        sb_lit(&c, file_type < 0 ? "The parent declares no file type, so whether this "
                                   "reference is itself quantized is not stated here. "
                 : full ? "Its weights are stored at full or half precision. "
                        : "The parent is itself quantized, so this reference is too. ");
        sb_lit(&c, "This measures what the recipe changed against the parent, not how close "
               "either is to the publisher's model; `--reference hf:DIR` measures that.\n\n");
    }
    if (given)
        sb_fmt(&c, "## License\n\n`%s`, given with `--license`: the parent file declares no "
               "`general.license`.\n\n", license);
    else
        sb_fmt(&c, "## License\n\n`%s`, copied from the parent's `general.license`.\n\n", license);
    jv *cr = jv_get(r, "credits");
    int ncr = cr && cr->type == J_ARR ? cr->n : 0;
    if (ncr || n_credit_fill) {
        sb_lit(&c, "## Credits\n\n");
        for (int i = 0; i < ncr; i++) sb_fmt(&c, "- %s\n", jv_str(cr->items[i], ""));
        if (!ncr)
            for (int i = 0; i < n_credit_fill; i++) sb_fmt(&c, "- %s\n", credit_fill[i]);
        sb_lit(&c, "\n");
    }
    sb_fmt(&c, "## Records\n\n`%s.derive.json` holds both files' sha256, the recipe, the window "
           "and the measurement; `%s.quant.json` is the rewrite's own record.\n", ob, ob);
    jv_free(r);
    bool ok = write_text(card_path, &c);
    free(c.s);
    return ok;
}

// the recipe as `runner derive` arguments, file arguments by base name
static void put_command(sbuf *b, const derive_opts *o) {
    sb_fmt(b, "runner derive -m %s -o %s", plat_path_base(o->parent), plat_path_base(o->out));
    for (int k = 0; k < N_RECIPE; k++)
        if (o->recipe[k]) sb_fmt(b, " %s %s", RECIPE[k].flag, plat_path_base(o->recipe[k]));
    if (o->target_size) {
        sb_fmt(b, " --target-size %lld --high %s --low %s", o->target_size, o->high, o->low);
        if (o->match) sb_fmt(b, " --match %s", o->match);
        if (o->group_tensor) sb_lit(b, " --group tensor");
    }
    if (o->license) sb_fmt(b, " --license %s", o->license);
    if (o->base_model) sb_fmt(b, " --base-model %s", o->base_model);
    if (o->corpus_given) sb_fmt(b, " --corpus %s", plat_path_base(o->corpus));
    sb_fmt(b, " --window %d:%d:%d", o->start, o->stride, o->count);
    if (o->reference_hf) sb_fmt(b, " --reference hf:%s", plat_path_base(o->reference_hf));
    for (int i = 0; i < o->n_ref_args; i++)
        sb_fmt(b, " --reference-arg %s", plat_path_base(o->ref_args[i]));
}

// `runner derive --card-from REC [-o CARD]`: the card from a record alone,
// beside it under the file's name unless -o says where
static int card_only(int argc, char **argv) {
    const char *rec = NULL, *out = NULL, *base = NULL;
    const char *credits[8], *cmps[8];
    int nc = 0, ncmp = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--card-from")) rec = argv[i + 1];
        else if (!strcmp(argv[i], "-o") || !strcmp(argv[i], "--out")) out = argv[i + 1];
        else if (!strcmp(argv[i], "--base-model")) base = argv[i + 1];
        else if (!strcmp(argv[i], "--credit") && nc < 8) credits[nc++] = argv[i + 1];
        else if (!strcmp(argv[i], "--compare") && ncmp < 8) cmps[ncmp++] = argv[i + 1];
        else {
            fprintf(stderr, "error: --card-from takes -o CARD, --base-model, --credit and --compare beside it "
                    "(got %s)\n", argv[i]);
            return 1;
        }
    }
    if (!rec || ((argc - 1) % 2)) {
        fprintf(stderr, "error: runner derive --card-from REC.derive.json [-o CARD.md]\n");
        return 1;
    }
    char *card = NULL;
    if (!out) {
        const char *suf = ".derive.json";
        size_t n = strlen(rec), k = strlen(suf);
        if (n <= k || strcmp(rec + n - k, suf)) {
            fprintf(stderr, "error: %s does not end in %s; give the card's path with -o\n", rec, suf);
            return 1;
        }
        card = malloc(n - k + sizeof ".README.md");
        if (!card) return 1;
        memcpy(card, rec, n - k);
        strcpy(card + n - k, ".README.md");
    }
    bool ok = card_from_record(rec, out ? out : card, base, credits, nc, cmps, ncmp);
    if (ok) printf("derive: card %s from %s\n", out ? out : card, rec);
    free(card);
    return ok ? 0 : 1;
}

int derive_main(int argc, char **argv) {
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--card-from")) return card_only(argc, argv);
    derive_opts o;
    int rc = parse_args(argc, argv, &o);
    if (rc >= 0) return rc;
    char *dir = plat_executable_dir();
    char *self = plat_executable_path();
    if (!dir || !self) { fprintf(stderr, "error: cannot find this binary's path\n"); return 1; }
    char *default_corpus = join(dir, BESIDE_CORPUS);
    if (!o.corpus) o.corpus = default_corpus;
    else o.corpus_given = true;
    if (!plat_file_readable(o.corpus)) {
        fprintf(stderr, "error: cannot read the corpus %s; give --corpus FILE\n", o.corpus);
        return 1;
    }
    parent_meta pm;
    if (!read_parent(o.parent, o.license, &pm)) return 1;
    const char *gold_env = getenv("RUNNER_GOLD_LOGITS");
    char *gold_script = gold_env && *gold_env ? strdup(gold_env) : join(dir, BESIDE_GOLD);
    if (o.reference_hf && !gold_ready(gold_script, !(gold_env && *gold_env))) return 1;

    char *quant_rec = suffixed(o.out, ".quant.json");
    char *ref_score = suffixed(o.out, ".parent.score.json");
    char *file_score = suffixed(o.out, ".score.json");
    char *gold_out = suffixed(o.out, ".gold.json");
    char *rec_path = suffixed(o.out, ".derive.json");
    char *card_path = suffixed(o.out, ".README.md");
    // the window's tokens and a BOS: everything the measurement reads
    // gold-logits.py reads its special prefix then ids[:min_prefix +
    // max_positions * stride + 1] and refuses a score file shorter than that
    // span (after its whole float32 forward), so the score runs cover it: a
    // BOS, then start + count * stride + 1 tokens of text
    int n_tokens = 1 + o.start + o.count * o.stride + 1;
    if (o.ctx && atoi(o.ctx) < n_tokens) {
        fprintf(stderr, "error: the window %d:%d:%d reads %d tokens and -c is %s; raise -c or "
                "give a smaller --window\n", o.start, o.stride, o.count, n_tokens, o.ctx);
        return 1;
    }

    // 1. the reference first (default reference): a corpus too short for the
    //    window is refused before anything is made
    jv *ref = NULL;
    if (!o.reference_hf) {
        if (!score(self, &o, o.parent, ref_score, n_tokens) || !(ref = read_json(ref_score)))
            return 1;
        if (!window_fits(&o, ref)) return 1;
    }
    // 2. make the file (a --target-size recipe plans its types first)
    sized_result sz;
    memset(&sz, 0, sizeof sz);
    if (o.target_size && !sized_plan(self, &o, file_bytes(o.parent), &sz)) return 1;
    const char *a[32];
    int n = 0;
    a[n++] = "-m"; a[n++] = o.parent; a[n++] = "--rewrite"; a[n++] = o.out;
    for (int k = 0; k < N_RECIPE; k++)
        if (o.recipe[k]) { a[n++] = RECIPE[k].flag; a[n++] = o.recipe[k]; }
    if (o.target_size) { a[n++] = "--type-plan"; a[n++] = sz.plan; }
    if (o.threads) { a[n++] = "-t"; a[n++] = o.threads; }
    a[n] = NULL;
    rc = run_self(self, a);
    if (rc != 0) {
        fprintf(stderr, "error: runner derive: the rewrite failed (exit %d)\n", rc);
        return 1;
    }
    // 3. measure it
    jv *run = NULL;
    if (!score(self, &o, o.out, file_score, n_tokens) || !(run = read_json(file_score))) return 1;
    derive_measure m;
    memset(&m, 0, sizeof m);
    if (o.reference_hf) {
        if (!window_fits(&o, run)) {
            plat_remove(o.out);
            plat_remove(quant_rec);
            return 1;
        }
        if (!gold_measure(&o, gold_script, file_score, (int)jv_num(jv_get(run, "n_special"), 0),
                          gold_out, &m))
            return 1;
    } else if (!compare_scores(&o, ref, run, &m)) {
        return 1;
    }
    jv_free(ref);
    jv_free(run);

    // 4. the record
    char psha[65], fsha[65];
    if (!envelope_file_sha256(o.parent, psha) || !envelope_file_sha256(o.out, fsha)) {
        fprintf(stderr, "error: cannot hash the parent or the file\n");
        return 1;
    }
    long long pbytes = file_bytes(o.parent), fbytes = file_bytes(o.out);
    bool publisher = o.reference_hf != NULL;
    bool passes = m.mean_kld <= BAR_MEAN_KLD && m.mq_top1_pct >= BAR_MQ_TOP1;
    sbuf r = {0};
    sb_lit(&r, "{\"schema_version\":\"xyntetik.runner.derive.v1\",\"runner_version\":");
    sb_str(&r, RUNNER_VERSION);
    sb_lit(&r, ",\"reference_kind\":");
    sb_str(&r, publisher ? "publisher-float32" : "parent-file");
    sb_lit(&r, ",\"reference\":{");
    if (publisher) {
        sb_lit(&r, "\"hf\":"); sb_str(&r, o.reference_hf);
        sb_lit(&r, ",\"class\":"); sb_str(&r, m.reference_class);
        sb_lit(&r, ",\"dtype\":"); sb_str(&r, m.reference_dtype);
        sb_lit(&r, ",\"script\":\"scripts/gold-logits.py\",\"args\":[");
        for (int i = 0; i < o.n_ref_args; i++) {
            if (i) sb_lit(&r, ",");
            sb_str(&r, o.ref_args[i]);
        }
        sb_lit(&r, "]");
    } else {
        sb_lit(&r, "\"file\":"); sb_str(&r, o.parent);
        sb_fmt(&r, ",\"sha256\":\"%s\",\"arithmetic\":\"runner, cpu\"", psha);
    }
    sb_lit(&r, "},\"comparisons\":[");
    for (int i = 0; i < o.n_compares; i++) {
        if (i) sb_lit(&r, ",");
        if (!compare_json(gold_out, o.compares[i], &r)) return 1;
    }
    sb_lit(&r, "],\"credits\":[");
    for (int i = 0; i < o.n_credits; i++) {
        if (i) sb_lit(&r, ",");
        sb_str(&r, o.credits[i]);
    }
    sb_lit(&r, "],\"command\":");
    sbuf cmd = {0};
    put_command(&cmd, &o);
    sb_put(&cmd, "", 1);
    sb_str(&r, cmd.failed ? NULL : cmd.s);
    sb_lit(&r, ",\"recipe\":{");
    bool first = true;
    for (int k = 0; k < N_RECIPE; k++) {
        if (!o.recipe[k]) continue;
        sb_fmt(&r, "%s\"%s\":", first ? "" : ",", RECIPE[k].key);
        sb_str(&r, o.recipe[k]);
        first = false;
    }
    if (o.target_size) {
        sb_fmt(&r, "%s\"target_size\":%lld,\"high\":", first ? "" : ",", o.target_size);
        sb_str(&r, o.high);
        sb_lit(&r, ",\"low\":"); sb_str(&r, o.low);
        sb_lit(&r, ",\"match\":"); sb_str(&r, o.match ? o.match : "");
        sb_lit(&r, ",\"group\":"); sb_str(&r, o.group_tensor ? "tensor" : "block");
        sb_lit(&r, ",\"quant_error\":"); sb_str(&r, sz.report);
        sb_lit(&r, ",\"type_plan\":"); sb_str(&r, sz.plan);
        sb_fmt(&r, ",\"save_bytes\":%lld,\"demoted\":[", sz.save_bytes);
        for (int k = 0; k < sz.n_demoted; k++) {
            if (k) sb_lit(&r, ",");
            sb_str(&r, sz.demoted[k]);
        }
        sb_lit(&r, "]");
    }
    sb_lit(&r, "},\"parent\":{\"path\":"); sb_str(&r, o.parent);
    sb_fmt(&r, ",\"sha256\":\"%s\",\"bytes\":%lld,\"license\":", psha, pbytes);
    sb_str(&r, pm.license);
    sb_lit(&r, ",\"license_source\":");
    sb_str(&r, pm.license_given ? "--license" : "general.license");
    sb_lit(&r, ",\"base_model_repo\":");
    sb_str(&r, o.base_model ? o.base_model : *pm.repo ? pm.repo : NULL);
    sb_fmt(&r, ",\"file_type\":%d,\"name\":", pm.file_type);
    sb_str(&r, *pm.name ? pm.name : NULL);
    sb_lit(&r, "},\"file\":{\"path\":"); sb_str(&r, o.out);
    sb_fmt(&r, ",\"sha256\":\"%s\",\"bytes\":%lld,\"record\":", fsha, fbytes);
    sb_str(&r, quant_rec);
    sb_lit(&r, "},\"measurement\":{\"corpus\":"); sb_str(&r, o.corpus);
    sb_fmt(&r, ",\"positions\":%d,\"window\":{\"start\":%d,\"stride\":%d,\"count\":%d}",
           m.positions, o.start, o.stride, o.count);
    sb_lit(&r, ",\"mean_kld\":"); sb_num(&r, m.mean_kld);
    sb_lit(&r, ",\"median_kld\":"); if (m.have_median) sb_num(&r, m.median_kld); else sb_lit(&r, "null");
    sb_lit(&r, ",\"max_kld\":"); sb_num(&r, m.max_kld);
    sb_lit(&r, ",\"top1_agreement_pct\":"); sb_num(&r, m.top1_pct);
    sb_lit(&r, ",\"margin_qualified_positions\":");
    if (m.have_qualified) sb_fmt(&r, "%d", m.qualified); else sb_lit(&r, "null");
    sb_lit(&r, ",\"margin_qualified_top1_pct\":"); sb_num(&r, m.mq_top1_pct);
    sb_lit(&r, ",\"reference_score\":"); sb_str(&r, publisher ? gold_out : ref_score);
    sb_lit(&r, ",\"file_score\":"); sb_str(&r, file_score);
    sb_lit(&r, ",\"bar\":");
    if (publisher) {
        sb_fmt(&r, "{\"max_mean_kld\":%g,\"min_margin_qualified_top1_pct\":%g,\"passes\":%s}",
               BAR_MEAN_KLD, BAR_MQ_TOP1, passes ? "true" : "false");
    } else {
        sb_lit(&r, "null");   // no bar verdict against the parent file
    }
    sb_lit(&r, "}}\n");
    if (!write_text(rec_path, &r)) return 1;

    // 5. the card, written from the record just written: the record is the
    //    one source, and --card-from rewrites a card from it alone
    free(cmd.s);
    if (!card_from_record(rec_path, card_path, NULL, NULL, 0, NULL, 0)) return 1;

    printf("derive: %s (%lld bytes, %.1f%% of the parent); mean KLD %.4f, top-1 %.2f%% "
           "against the %s%s; record %s, card %s\n",
           o.out, fbytes, pbytes > 0 ? 100.0 * (double)fbytes / (double)pbytes : 0.0,
           m.mean_kld, m.top1_pct, publisher ? "publisher's float32 model" : "parent file",
           publisher ? (passes ? " (passes the bar)" : " (fails the bar)") : "",
           rec_path, card_path);
    free(r.s);
    free(dir); free(self); free(default_corpus); free(gold_script);
    free(quant_rec); free(ref_score); free(file_score); free(gold_out);
    free(rec_path); free(card_path);
    return 0;
}
