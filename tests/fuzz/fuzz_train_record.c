// libFuzzer harness for training record reading, replay pre-flight, and
// unlearning splits (src/train_record.c): an auditor runs `attest replay` and
// `attest unlearn` on records, data, and drop lists supplied by someone else.
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compat.h"
#include "envelope.h"
#include "fuzz_files.h"
#include "json.h"
#include "runner.h"
#include "train_record.h"

static char rec_path[256];
static char parent_path[256];
static char base_path[256];
static char data_path[256];
static char drop_path[256];
static char self_sha[65];
static char base_sha[65];
static char parent_sha[65];

static void cleanup(void) {
    unlink(rec_path);
    unlink(parent_path);
    unlink(base_path);
    unlink(data_path);
    unlink(drop_path);
    fuzz_files_close();
}

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc;
    (void)argv;
    fuzz_files_init();
    snprintf(rec_path, sizeof rec_path, "%s/rec.train.json", fuzz_dir);
    snprintf(parent_path, sizeof parent_path, "%s/parent.train.json", fuzz_dir);
    snprintf(base_path, sizeof base_path, "%s/base.gguf", fuzz_dir);
    snprintf(data_path, sizeof data_path, "%s/data.jsonl", fuzz_dir);
    snprintf(drop_path, sizeof drop_path, "%s/drop.jsonl", fuzz_dir);

    char *exe = plat_executable_path();
    if (!exe || !envelope_file_sha256(exe, self_sha)) memset(self_sha, '0', 64);
    self_sha[64] = 0;
    free(exe);

    static const unsigned char dummy_base[] = "GGUF\x03\x00\x00\x00";
    fuzz_write(base_path, dummy_base, sizeof dummy_base - 1);
    if (!envelope_file_sha256(base_path, base_sha)) memset(base_sha, '0', 64);
    base_sha[64] = 0;

    char parent_rec[2048];
    int pn = snprintf(
        parent_rec, sizeof parent_rec,
        "{\"schema_version\":\"xyntetik.runner.train.v1\",\"runner\":\"" RUNNER_VERSION "\","
        "\"determinism\":\"cpu-same-binary\",\"objective\":\"weighted-cross-entropy\","
        "\"lora_rank\":4,\"alpha\":8,\"lora_scale\":1,\"lr\":0.003,\"steps\":1,\"ctx\":16,"
        "\"seed\":1,\"kv_cache\":\"f16\",\"no_bos\":false,\"end_of_turn\":false,\"eot_id\":-1,"
        "\"adamw\":{\"beta1\":0.9,\"beta2\":0.999,\"eps\":1e-8,\"weight_decay\":0.01},"
        "\"recipe\":{\"schedule\":\"constant\",\"warmup\":0,\"floor\":0,\"batch\":1,\"loss\":\"row-sum\"},"
        "\"base\":{\"path\":\"base.gguf\",\"sha256\":\"%s\"},"
        "\"data\":{\"path\":\"data.jsonl\",\"sha256\":\"%s\"}}",
        base_sha, base_sha);
    assert(pn > 0 && (size_t)pn < sizeof parent_rec);
    fuzz_write(parent_path, (const unsigned char *)parent_rec, (size_t)pn);
    if (!envelope_file_sha256(parent_path, parent_sha)) memset(parent_sha, '0', 64);
    parent_sha[64] = 0;
    atexit(cleanup);
    return 0;
}

static void replace_64_token(char *buf, size_t n, char ch, const char hex[65]) {
    if (n < 64) return;
    for (size_t i = 0; i + 64 <= n; ) {
        bool hit = true;
        for (size_t k = 0; k < 64; k++)
            if (buf[i + k] != ch) { hit = false; break; }
        if (hit) {
            memcpy(buf + i, hex, 64);
            i += 64;
        } else {
            i++;
        }
    }
}

static void probe_record_paths(void) {
    char *empty_env[] = { NULL };
    train_record_host host = {
        .runner_version = RUNNER_VERSION,
        .build_flavor = "default",
        .build_commit = NULL,
        .build_dirty = false,
        .environ_list = empty_env,
    };
    cli_replay *rp = calloc(1, sizeof *rp);
    if (!rp) return;

    train_record_opts ro = {
        .replay_train = rec_path,
        .allow_same_binary = true,
    };
    int rc = 0;
    bool fail = train_record_prepare(&host, &ro, rp, &rc);
    if (!fail) {
        assert(memchr(rp->want_out_path, 0, sizeof rp->want_out_path) != NULL);
        assert(strlen(rp->want_out_sha) == 64);
    } else {
        assert(rc == 1 || rc == 2);
    }

    memset(rp, 0, sizeof *rp);
    memset(&ro, 0, sizeof ro);
    ro.unlearn_audit = rec_path;
    ro.unlearn_parent_data = data_path;
    ro.unlearn_dropped = drop_path;
    ro.allow_same_binary = true;
    rc = 0;
    fail = train_record_prepare(&host, &ro, rp, &rc);
    if (!fail) assert(strlen(rp->want_out_sha) == 64);
    else assert(rc == 1 || rc == 2);
    free(rp);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > 65536) return 0;

    // 1. Direct parser/helper probes on raw fuzzed JSON
    jv *j = json_parse((const char *)data, size);
    if (j) {
        double d = 0;
        int iv = 0;
        (void)train_record_num(j, "lr", 0, 1, &d);
        (void)train_record_int(j, "steps", 1, INT_MAX, &iv);
        assert(train_record_jv_same(j, j));
        jv_free(j);
    }

    // 2. Raw hostile record bytes through replay & unlearn pre-flight
    fuzz_write(rec_path, data, size);
    size_t split = size / 2;
    fuzz_write(data_path, data, split);
    fuzz_write(drop_path, data + split, size - split);

    fuzz_mute();
    probe_record_paths();

    // 3. Substitute valid self/base/parent sha placeholders so fuzzed records
    // also reach deeper hyperparameter, env, eval/kill, and unlearn paths.
    if (size >= 64) {
        char *patched = malloc(size + 1);
        if (patched) {
            memcpy(patched, data, size);
            patched[size] = 0;
            replace_64_token(patched, size, '0', self_sha);
            replace_64_token(patched, size, '1', base_sha);
            replace_64_token(patched, size, '3', parent_sha);
            fuzz_write(rec_path, (const unsigned char *)patched, size);
            free(patched);
            probe_record_paths();
        }
    }

    // 4. Direct unlearn_split on split fuzz bytes
    sbuf kept = {0};
    int count = 0, distinct = 0;
    char why[512];
    (void)train_record_unlearn_split(data_path, drop_path, &kept, &count, &distinct, why, sizeof why);
    free(kept.s);
    fuzz_unmute();
    return 0;
}
