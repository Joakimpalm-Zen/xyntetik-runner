// envelope_report reads a <model>.envelope.json sidecar and resolves the
// current (version, backend) against it, exact-match, without enforcement.
#include "envelope.h"
#include "json.h"
#include "runner.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Fixture paths, built at run time rather than baked in. They used to be
// "/tmp/..." literals, which on Windows resolve to C:\tmp -- a directory that
// happens to exist on the lab box and does not on a fresh CI runner, so
// `assert(f)` in write_manifest fired and the whole suite stopped at exit 127.
// Every sibling test here (instances, instances_oom, tray_core, vram_rollback)
// already branches on TEMP; this file was the one that did not.
// 256, not 512: every derived path is `char[512]` and the compiler has to be
// able to PROVE the suffix fits (the Windows build is -Werror=format-truncation).
// A 512-byte source into a 512-byte destination is not provable; a 256-byte
// one plus a 17-character suffix is.
static char MODEL[256];
static char RECORD_PATH[256];

static void init_paths(void) {
    const char *base = NULL;
#ifdef _WIN32
    base = getenv("TEMP");
    if (!base || !*base) base = getenv("TMP");
    const char sep = '\\';
#else
    base = getenv("TMPDIR");
    if (!base || !*base) base = "/tmp";
    const char sep = '/';
#endif
    // "." is always writable where the other fixtures already live, and is
    // the honest fallback rather than a path that may not exist
    if (!base || !*base) base = ".";
    snprintf(MODEL, sizeof MODEL, "%s%cxyntetik-envelope-test.gguf", base, sep);
    snprintf(RECORD_PATH, sizeof RECORD_PATH,
             "%s%cxyntetik-transcript-test.json", base, sep);
}

static void write_manifest(const char *body) {
    char path[512];
    snprintf(path, sizeof path, "%s.envelope.json", MODEL);
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s: the fixture directory is not "
                "writable\n", path);
        abort();
    }
    fputs(body, f);
    fclose(f);
}
static void rm_manifest(void) {
    char path[512];
    snprintf(path, sizeof path, "%s.envelope.json", MODEL);
    remove(path);
}

static const char *manifest(const char *version, const char *backend,
                            const char *verdict) {
    static char buf[1024];
    snprintf(buf, sizeof buf,
             "{\"schema_version\":\"xyntetik.runner.envelope.v1\","
             "\"runtime\":{\"version\":\"%s\","
             "\"kernel_set\":{\"backend\":\"%s\"}},"
             "\"verdict\":\"%s\"}",
             version, backend, verdict);
    return buf;
}

// Same, but the manifest also carries an artifact.sha256 (Unit 7).
static const char *manifest_sha(const char *version, const char *backend,
                                const char *verdict, const char *sha) {
    static char buf[1024];
    snprintf(buf, sizeof buf,
             "{\"schema_version\":\"xyntetik.runner.envelope.v1\","
             "\"artifact\":{\"sha256\":\"%s\"},"
             "\"runtime\":{\"version\":\"%s\","
             "\"kernel_set\":{\"backend\":\"%s\"}},"
             "\"verdict\":\"%s\"}",
             sha, version, backend, verdict);
    return buf;
}

// Write a model file (not just its sidecar) so artifact.sha256 has a real file
// to hash. "abc" is the classic FIPS 180-4 SHA-256 test vector.
static void write_model(const char *bytes, size_t n) {
    FILE *f = fopen(MODEL, "wb");
    assert(f);
    fwrite(bytes, 1, n, f);
    fclose(f);
}
static void rm_model(void) { remove(MODEL); }
static const char *ABC_SHA =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
static const char *ZERO_SHA =
    "0000000000000000000000000000000000000000000000000000000000000000";

int main(void) {
    char out[256];
    init_paths();

    // No sidecar -> silent, unclassified (transitional/legacy, not experimental).
    rm_manifest();
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_UNCLASSIFIED);
    assert(out[0] == 0);

    // Exact match + certified -> certified, and it says so.
    write_manifest(manifest(RUNNER_VERSION, "cpu", "certified"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_CERTIFIED);
    assert(strstr(out, "certified") && strstr(out, "measured envelope"));

    // Exact match + outside-envelope -> outside (reported, not enforced).
    write_manifest(manifest(RUNNER_VERSION, "cpu", "outside-envelope"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_OUTSIDE);
    assert(strstr(out, "OUTSIDE"));

    // Exact match + verdict "experimental" -> EXPERIMENTAL: a real measurement
    // that came back inconclusive (NOT the same as one we could not read).
    write_manifest(manifest(RUNNER_VERSION, "cpu", "experimental"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_EXPERIMENTAL);
    assert(strstr(out, "experimental"));

    // Exact-match only: a manifest measured on a DIFFERENT backend does not
    // describe this configuration -> INDETERMINATE (foreign, could not judge).
    write_manifest(manifest(RUNNER_VERSION, "cpu", "certified"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cuda", out, sizeof out) == ENV_INDETERMINATE);

    // ... nor a different runtime version.
    write_manifest(manifest("0.0.0-old", "cpu", "certified"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_INDETERMINATE);

    // Unknown manifest schema is not trusted for this runner -> indeterminate.
    write_manifest("{\"schema_version\":\"other.v9\",\"verdict\":\"certified\"}");
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_INDETERMINATE);

    // A present-but-malformed sidecar is indeterminate, never a crash or a load
    // failure.
    write_manifest("{ this is not json");
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_INDETERMINATE);

    // ---- tool-calling axis (reported-only): summarised in the banner --------
    // A manifest carrying a `tool_calling` block makes envelope_report append a
    // one-line, report-only summary of it. It NEVER changes the resolved state.
    write_manifest("{\"schema_version\":\"xyntetik.runner.envelope.v1\","
                   "\"runtime\":{\"version\":\"" RUNNER_VERSION "\","
                   "\"kernel_set\":{\"backend\":\"cpu\"}},"
                   "\"verdict\":\"certified\","
                   "\"tool_calling\":{"
                   "\"truncation_recovery\":{\"rungs_passed\":6,\"rungs_total\":6},"
                   "\"schema_shape\":{\"held_to_quant\":\"Q4_0\"},"
                   "\"agent_torture\":{\"gate\":\"pass\"},"
                   "\"native_tool_protocol\":{\"tool_family\":\"granite\","
                   "\"native\":true},"
                   "\"gate\":\"pass\"}}");
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_CERTIFIED);
    assert(strstr(out, "tool-calling gate=pass"));
    assert(strstr(out, "truncation 6/6") && strstr(out, "schema-shape@Q4_0"));
    assert(strstr(out, "agent-torture pass") && strstr(out, "native granite"));

    // Hostile/forward-version manifests may carry arbitrarily long reported
    // labels. Reporting with no output buffer must still resolve the state,
    // and a small output buffer must remain terminated rather than advancing
    // later appends beyond the fixed internal summary buffer.
    char long_quant[600];
    memset(long_quant, 'Q', sizeof(long_quant) - 1);
    long_quant[sizeof(long_quant) - 1] = 0;
    char long_manifest[2048];
    snprintf(long_manifest, sizeof long_manifest,
             "{\"schema_version\":\"xyntetik.runner.envelope.v1\","
             "\"runtime\":{\"version\":\"%s\","
             "\"kernel_set\":{\"backend\":\"cpu\"}},"
             "\"verdict\":\"certified\",\"tool_calling\":{"
             "\"schema_shape\":{\"held_to_quant\":\"%s\"},"
             "\"agent_torture\":{\"gate\":\"pass\"}}}",
             RUNNER_VERSION, long_quant);
    write_manifest(long_manifest);
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", NULL, 0) ==
           ENV_CERTIFIED);
    int silent_state = -1;
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, NULL, 0,
                         &silent_state));
    assert(silent_state == ENV_CERTIFIED);
    char small[24];
    memset(small, 'X', sizeof small);
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", small,
                           sizeof small) == ENV_CERTIFIED);
    assert(memchr(small, 0, sizeof small));

    // The same manifest WITHOUT a tool_calling block emits no such line
    // (back-compat: silent for manifests that predate the axis).
    write_manifest(manifest(RUNNER_VERSION, "cpu", "certified"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_CERTIFIED);
    assert(!strstr(out, "tool-calling"));

    // ---- slice 3: the enforcing gate ----------------------------------------
    int st = -1;

    // No manifest -> load, silent, unclassified.
    rm_manifest();
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(out[0] == 0 && st == ENV_UNCLASSIFIED);

    // Certified -> load, with the informational banner.
    write_manifest(manifest(RUNNER_VERSION, "cpu", "certified"));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(st == ENV_CERTIFIED && strstr(out, "certified"));

    // Matching verdict "experimental" -> load, banner, never a refusal.
    write_manifest(manifest(RUNNER_VERSION, "cpu", "experimental"));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(st == ENV_EXPERIMENTAL && strstr(out, "experimental"));

    // Foreign sidecar (measured on another backend) -> INDETERMINATE, loads
    // with a banner, never a refusal.
    write_manifest(manifest(RUNNER_VERSION, "cuda", "certified"));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(st == ENV_INDETERMINATE && out[0]);

    // Outside-envelope -> REFUSE (returns false), and the message tells the
    // user how to override.
    write_manifest(manifest(RUNNER_VERSION, "cpu", "outside-envelope"));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == false);
    assert(st == ENV_OUTSIDE && strstr(out, "refusing") && strstr(out, "--force-uncertified"));

    // ...unless --force-uncertified is set: then it loads with a loud warning.
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", true, out, sizeof out, &st) == true);
    assert(st == ENV_OUTSIDE && strstr(out, "WARNING") && strstr(out, "force-uncertified"));

    // The refusal names the measured reason: the gate check(s) that FAILED.
    write_manifest("{\"schema_version\":\"xyntetik.runner.envelope.v1\","
                   "\"runtime\":{\"version\":\"" RUNNER_VERSION "\","
                   "\"kernel_set\":{\"backend\":\"cpu\"}},"
                   "\"verdict\":\"outside-envelope\","
                   "\"quality\":{\"checks\":{\"cpu_gpu_identity\":\"pass\","
                   "\"ram_fits\":\"fail\"}}}");
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == false);
    assert(strstr(out, "ram_fits"));

    // Fail-open: a malformed manifest is indeterminate, never blocks a load.
    write_manifest("{ this is not json");
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(st == ENV_INDETERMINATE);

    // ---- Unit 7: artifact.sha256 must match the loaded file -----------------
    // A sidecar carries an artifact.sha256; the manifest resolves on
    // (version, backend) only, so nothing else proves it belongs to THIS file.
    // Verify the hash before applying any verdict.
    write_model("abc", 3);

    // Matching sha + outside-envelope -> the verdict STILL applies (refuse).
    write_manifest(manifest_sha(RUNNER_VERSION, "cpu", "outside-envelope", ABC_SHA));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == false);
    assert(st == ENV_OUTSIDE && strstr(out, "refusing"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_OUTSIDE);

    // WRONG sha -> the sidecar names a different artifact: its outside-envelope
    // verdict has NO effect (loads), state INDETERMINATE, and it says why.
    write_manifest(manifest_sha(RUNNER_VERSION, "cpu", "outside-envelope", ZERO_SHA));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(st == ENV_INDETERMINATE && strstr(out, "does not match"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_INDETERMINATE);
    assert(strstr(out, "does not match"));

    // A matching-sha certified sidecar still certifies (the check gates on
    // identity, it does not weaken a legitimate verdict).
    write_manifest(manifest_sha(RUNNER_VERSION, "cpu", "certified", ABC_SHA));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(st == ENV_CERTIFIED && strstr(out, "certified"));

    // Back-compat: a manifest with NO artifact.sha256 behaves exactly as before
    // even when the model file exists — the outside verdict still enforces.
    write_manifest(manifest(RUNNER_VERSION, "cpu", "outside-envelope"));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == false);
    assert(st == ENV_OUTSIDE);

    // Build/device strings come from the compiler and GPU driver, not a JSON
    // grammar. A quote/control byte there must be escaped like paths and
    // prompt text or a successful run writes an unparsable transcript.
    {
        const char *record_path = RECORD_PATH;
        int32_t prompt_tok[] = { 1 };
        int32_t output_tok[] = { 2 };
        transcript_info ti = {
            .out_path = record_path,
            .runner_version = "runner\"test",
            .executable_path = MODEL,
            .compiler = "compiler\nline",
            .os = "test-os", .arch = "test-arch",
            .device = "GPU \"quoted\"",
            .threads = 1, .n_ctx = 8, .n_batch = 1,
            .model_path = MODEL,
            .seed = 1, .repeat_penalty = 1,
            .bos = true, .prompt_text = "p",
            .prompt_tokens = prompt_tok, .n_prompt = 1,
            .output_text = "x", .output_text_len = 1,
            .output_tokens = output_tok, .n_output = 1,
        };
        assert(transcript_write(&ti));
        FILE *rf = fopen(record_path, "rb");
        assert(rf);
        char record[8192];
        size_t rn = fread(record, 1, sizeof record, rf);
        fclose(rf);
        jv *rv = json_parse(record, rn);
        assert(rv);
        assert(strcmp(jv_str(jv_get(rv, "runner"), ""), "runner\"test") == 0);
        jv *build = jv_get(rv, "build");
        jv *profile = jv_get(rv, "profile");
        assert(strcmp(jv_str(jv_get(build, "compiler"), ""),
                      "compiler\nline") == 0);
        assert(strcmp(jv_str(jv_get(profile, "device"), ""),
                      "GPU \"quoted\"") == 0);
        // the default build writes no build.flavor at all
        assert(jv_get(build, "flavor") == NULL);
        jv_free(rv);
        remove(record_path);
        // a T3 build marks its records
        ti.build_flavor = "t3";
        assert(transcript_write(&ti));
        rf = fopen(record_path, "rb");
        assert(rf);
        rn = fread(record, 1, sizeof record, rf);
        fclose(rf);
        rv = json_parse(record, rn);
        assert(rv);
        assert(strcmp(jv_str(jv_get(jv_get(rv, "build"), "flavor"), ""), "t3") == 0);
        jv_free(rv);
        remove(record_path);
    }

    // Signed receipts under both key algorithms: the key file round-trips,
    // the record's signature object names the algo and verifies, and one
    // flipped byte anywhere in the signed span is RSIG_BAD. Both keys derive
    // from the same fixed seed, so the ML-DSA-44 public key here is the ACVP
    // vector's if the seed is theirs (tests/test_mldsa.c pins that); this
    // test pins the receipt plumbing around the primitive.
    {
        static const char *algos[] = { SIGN_ALGO_ED25519, SIGN_ALGO_MLDSA44 };
        static const size_t pk_len[] = { 32, 1312 }, sig_len[] = { 64, 2420 };
        for (int ai = 0; ai < 2; ai++) {
            char key_path[512], record_path[512];
            snprintf(key_path, sizeof key_path, "%s.%s.signkey.json", MODEL, algos[ai]);
            snprintf(record_path, sizeof record_path, "%s.%s.receipt.json", MODEL, algos[ai]);
            uint8_t seed[32];
            for (int i = 0; i < 32; i++) seed[i] = (uint8_t)(i * 7 + ai);
            char pub[SIGN_PUBHEX_CAP];
            assert(signkey_write(key_path, algos[ai], seed, pub));
            assert(strlen(pub) == pk_len[ai] * 2);
            signkey k;
            assert(signkey_load(key_path, &k));
            assert(strcmp(k.algo, algos[ai]) == 0 && k.pk_n == pk_len[ai] && k.sig_n == sig_len[ai]);
            // the same seed derives the same key: deterministic keygen
            signkey k2;
            assert(signkey_load(key_path, &k2) && memcmp(k.pk, k2.pk, k.pk_n) == 0);

            int32_t prompt_tok[] = { 1 };
            int32_t output_tok[] = { 2 };
            transcript_info ti = {
                .out_path = record_path, .sign_key_path = key_path,
                .runner_version = "t", .executable_path = MODEL,
                .compiler = "c", .os = "o", .arch = "a", .device = "d",
                .threads = 1, .n_ctx = 8, .n_batch = 1, .model_path = MODEL,
                .seed = 1, .repeat_penalty = 1, .bos = true, .prompt_text = "p",
                .prompt_tokens = prompt_tok, .n_prompt = 1,
                .output_text = "x", .output_text_len = 1,
                .output_tokens = output_tok, .n_output = 1,
            };
            assert(transcript_write(&ti));
            FILE *rf = fopen(record_path, "rb");
            assert(rf);
            static char record[16384];
            size_t rn = fread(record, 1, sizeof record - 1, rf);
            fclose(rf);
            record[rn] = 0;
            char got_pub[SIGN_PUBHEX_CAP];
            assert(receipt_signature_check(record, rn, got_pub) == RSIG_OK);
            assert(strcmp(got_pub, pub) == 0);
            jv *rv = json_parse(record, rn);
            assert(rv);
            assert(strcmp(jv_str(jv_get(jv_get(rv, "signature"), "algo"), ""), algos[ai]) == 0);
            assert(strlen(jv_str(jv_get(jv_get(rv, "signature"), "sig"), "")) == sig_len[ai] * 2);
            jv_free(rv);
            // tamper inside the signed span (the seed field, before the chain)
            char *seedpos = strstr(record, "\"seed\":1");
            assert(seedpos);
            seedpos[7] = '2';
            assert(receipt_signature_check(record, rn, got_pub) == RSIG_BAD);
            seedpos[7] = '1';
            // tamper in the signature itself
            char *sigpos = strstr(record, "\"sig\":\"");
            assert(sigpos);
            sigpos[7] = sigpos[7] == '0' ? '1' : '0';
            assert(receipt_signature_check(record, rn, got_pub) == RSIG_BAD);
            // an algo the verifier does not know is malformed, not accepted
            char *algopos = strstr(record, ",\"signature\":{\"algo\":\"");
            assert(algopos);
            algopos[22] = 'X';
            assert(receipt_signature_check(record, rn, got_pub) == RSIG_MALFORMED);
            remove(record_path);
            remove(key_path);
        }
        // a key file whose algo is unknown does not load
        char key_path[512];
        snprintf(key_path, sizeof key_path, "%s.bad.signkey.json", MODEL);
        FILE *kf = fopen(key_path, "wb");
        assert(kf);
        fputs("{\"schema_version\":\"xyntetik.runner.signkey.v1\",\"algo\":\"rsa\","
              "\"seed\":\"0000000000000000000000000000000000000000000000000000000000000000\"}", kf);
        fclose(kf);
        signkey k;
        assert(!signkey_load(key_path, &k));
        remove(key_path);
    }

    rm_model();
    rm_manifest();
    printf("test-envelope: OK\n");
    return 0;
}
