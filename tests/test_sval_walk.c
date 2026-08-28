// Oracle-guided differential walk over the schema validator.
//
// sval_trial's contract is: answer exactly what a full struct copy fed the
// same byte would answer, from every reachable state, out of a scratch whose
// previous contents are irrelevant. The v0.4.0 release gate caught that
// contract broken by live state the copy did not carry (the map seen-key
// guard), and the poison-and-compare unit test had greened on an accidental
// garbage match because it compared answers only.
//
// This test walks the LEGAL state space instead of mutating byte soup: at
// every step it probes all 256 bytes — poisoned scratch, against a full
// copy, comparing answers AND live state — then feeds one randomly chosen
// accepted byte and repeats. Walks are seeded and deterministic; a failure
// prints its (schema, seed, step) coordinates so it can be replayed. On walk
// end it checks the closer's published guarantee: accepted prefix + close
// output parses as JSON.
//
// The default budget is CI-sized (a couple of seconds). RUNNER_SVAL_WALK_SEEDS
// scales the same deterministic sweep up for local soaks.
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "jsonmode.h"
#include "schema.h"

#define MAX_STEPS 160
#define DEFAULT_SEEDS 6

static const char *SCHEMAS[] = {
    // nested typed-additionalProperties maps: the seen guard at two depths
    "{\"type\":\"object\",\"properties\":{"
      "\"m\":{\"type\":\"object\",\"additionalProperties\":"
        "{\"type\":\"object\",\"additionalProperties\":"
          "{\"type\":\"integer\",\"minimum\":-3,\"maximum\":700}}}},"
      "\"required\":[\"m\"]}",
    // flat map with enum values: duplicate refusal plus literal machinery;
    // maxLength-bounded keys keep walks from drowning in key content
    "{\"type\":\"object\",\"additionalProperties\":{\"enum\":[\"p\",\"q\",3]}}",
    // bounded strings: UTF-8 length accounting, escapes, surrogate pairs
    "{\"type\":\"object\",\"properties\":{"
      "\"s\":{\"type\":\"string\",\"minLength\":1,\"maxLength\":4}},"
      "\"required\":[\"s\"]}",
    // numbers against every bound flavor, in an array with item bounds
    "{\"type\":\"object\",\"properties\":{"
      "\"v\":{\"type\":\"array\",\"items\":"
        "{\"type\":\"number\",\"exclusiveMinimum\":0,\"maximum\":99},"
        "\"minItems\":1,\"maxItems\":5},"
      "\"i\":{\"type\":\"integer\",\"minimum\":10,\"exclusiveMaximum\":20}},"
      "\"required\":[\"v\"]}",
    // union of shapes plus a free object: pick_alt and the jsonv submachine
    "{\"type\":\"object\",\"properties\":{"
      "\"d\":{\"type\":[\"string\",\"null\"]},"
      "\"f\":{\"type\":\"object\"}},\"required\":[\"f\"]}",
    // oneOf const scalars: discriminator machinery
    "{\"type\":\"object\",\"properties\":{"
      "\"k\":{\"oneOf\":[{\"const\":\"alpha\"},{\"const\":\"beta\"},"
                        "{\"const\":42}]}},\"required\":[\"k\"]}",
    // closed object: additionalProperties false with required members
    "{\"type\":\"object\",\"properties\":{"
      "\"a\":{\"type\":\"boolean\"},\"b\":{\"type\":\"null\"}},"
      "\"required\":[\"a\",\"b\"],\"additionalProperties\":false}",
};
#define N_SCHEMAS ((int)(sizeof(SCHEMAS) / sizeof(SCHEMAS[0])))

static uint64_t lcg(uint64_t *s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return *s >> 33;
}

static int live_state_equal(const sval *a, const sval *b) {
    if (a->depth != b->depth || a->done != b->done) return 0;
    if (a->last_enum != b->last_enum) return 0;
    int d = a->depth;
    if (d < 0) d = 0;
    if (d > (int)(sizeof(a->stack) / sizeof(a->stack[0])))
        d = (int)(sizeof(a->stack) / sizeof(a->stack[0]));
    if (memcmp(a->stack, b->stack, (size_t)d * sizeof(a->stack[0]))) return 0;
    // the number spelling's NUL terminator is live (number_text_in_bounds
    // reads the C string), so num_len + 1 bytes matter
    if (a->num_len != b->num_len) return 0;
    if (memcmp(a->num_text, b->num_text, (size_t)a->num_len + 1)) return 0;
    if (a->n_seen != b->n_seen) return 0;
    int ns = a->n_seen;
    if (ns > MAP_SEEN_MAX) ns = MAP_SEEN_MAX;
    if (memcmp(a->seen_hash, b->seen_hash,
               (size_t)ns * sizeof(a->seen_hash[0]))) return 0;
    if (memcmp(a->seen_depth, b->seen_depth, (size_t)ns)) return 0;
    const jsonv *ja = &a->any, *jb = &b->any;
    if (ja->depth != jb->depth || ja->st != jb->st || ja->sub != jb->sub ||
        ja->lit != jb->lit || ja->utf8 != jb->utf8 || ja->done != jb->done ||
        ja->esc != jb->esc || ja->esc_hi != jb->esc_hi ||
        ja->khash != jb->khash || ja->kseen_n != jb->kseen_n)
        return 0;
    int jks = ja->kseen_n;
    if (jks > JSON_KEY_SEEN_MAX) jks = JSON_KEY_SEEN_MAX;
    if (memcmp(ja->kseen_hash, jb->kseen_hash,
               (size_t)jks * sizeof(ja->kseen_hash[0]))) return 0;
    if (memcmp(ja->kseen_depth, jb->kseen_depth, (size_t)jks)) return 0;
    int jd = ja->depth;
    if (jd < 0) jd = 0;
    if (jd > (int)sizeof(ja->stack)) jd = (int)sizeof(ja->stack);
    if (memcmp(ja->stack, jb->stack, (size_t)jd)) return 0;
    return 1;
}

static void fail_at(int si, uint64_t seed, int step, int byte, const char *why,
                    const char *acc, size_t n_acc) {
    fprintf(stderr, "sval walk FAILED: schema %d seed %llu step %d byte 0x%02x"
            " (%s)\n  accepted so far (%zu bytes): %.*s\n",
            si, (unsigned long long)seed, step, byte, why,
            n_acc, (int)n_acc, acc);
    exit(1);
}

static void walk(const snode *schema, int si, uint64_t seed) {
    uint64_t rng = seed * 2654435761u + 12345;
    sval v;
    sval_init(&v, schema);
    char accepted[4096];
    size_t n_acc = 0;

    for (int step = 0; step < MAX_STEPS && !v.done; step++) {
        unsigned char ok[256];
        int n_ok = 0;
        for (int c = 1; c < 256; c++) {
            char b = (char)c;
            sval full = v;
            bool want = sval_feed(&full, &b, 1);
            sval before = v;
            sval scratch;
            memset(&scratch, 0xA5, sizeof(scratch));
            bool got = sval_trial(&v, &scratch, &b, 1);
            if (got != want)
                fail_at(si, seed, step, c, "trial != full copy",
                        accepted, n_acc);
            if (want && !live_state_equal(&scratch, &full))
                fail_at(si, seed, step, c, "accepted state diverged",
                        accepted, n_acc);
            if (memcmp(&before, &v, sizeof(v)))
                fail_at(si, seed, step, c, "trial mutated the validator",
                        accepted, n_acc);
            if (want) ok[n_ok++] = (unsigned char)c;
        }

        // the closer's guarantee holds at EVERY frontier, not only where a
        // walk happens to stop: the subnormal minimum fill sat exactly on a
        // frontier the end-only check rarely landed on
        char out[1024];
        sval closing = v;
        int wrote = sval_close(&closing, out, (int)sizeof(out));
        if (wrote < 0 || (size_t)wrote > sizeof(out))
            fail_at(si, seed, step, -1, "close wrote out of bounds",
                    accepted, n_acc);
        bool started = !(v.depth == 1 && v.stack[0].phase == 0 /* P_START */);
        if ((v.done || (wrote > 0 && started)) &&
            n_acc + (size_t)wrote <= sizeof(accepted)) {
            char doc[sizeof(accepted)];
            memcpy(doc, accepted, n_acc);
            memcpy(doc + n_acc, out, (size_t)wrote);
            jv *parsed = json_parse(doc, n_acc + (size_t)wrote);
            if (!parsed)
                fail_at(si, seed, step, -1, "closed document does not parse",
                        doc, n_acc + (size_t)wrote);
            jv_free(parsed);
        }

        if (!n_ok) break;   // fully wedged states exist only via num buffer caps
        char b = (char)ok[lcg(&rng) % (unsigned)n_ok];
        assert(sval_feed(&v, &b, 1));
        if (n_acc < sizeof(accepted)) accepted[n_acc++] = b;
    }
}

int main(void) {
    const char *env = getenv("RUNNER_SVAL_WALK_SEEDS");
    int n_seeds = env ? atoi(env) : DEFAULT_SEEDS;
    if (n_seeds < 1) n_seeds = DEFAULT_SEEDS;

    for (int si = 0; si < N_SCHEMAS; si++) {
        jv *j = json_parse(SCHEMAS[si], strlen(SCHEMAS[si]));
        assert(j);
        char err[256];
        snode *schema = schema_compile(j, err, (int)sizeof(err));
        if (!schema) {
            fprintf(stderr, "schema %d failed to compile: %s\n", si, err);
            return 1;
        }
        for (int seed = 0; seed < n_seeds; seed++)
            walk(schema, si, (uint64_t)seed + 1);
        schema_free(schema);
        jv_free(j);
    }
    printf("sval walk: %d schemas x %d seeds, all probes agree\n",
           N_SCHEMAS, n_seeds);
    return 0;
}
