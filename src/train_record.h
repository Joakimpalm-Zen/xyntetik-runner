// Provenance record parsing, replay pre-flight, and unlearning splitter for
// xyntetik.runner.train.v1 records (`runner attest replay`, `runner attest unlearn`,
// and `runner --unlearn`).
#ifndef RUNNER_TRAIN_RECORD_H
#define RUNNER_TRAIN_RECORD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "json.h"

#define TRAIN_RECORD_MAX_BYTES (16u * 1024u * 1024u)

#define TRAIN_ADAM_BETA1 0.9
#define TRAIN_ADAM_BETA2 0.999
#define TRAIN_ADAM_EPS   1e-8
#define TRAIN_ADAM_WD    0.01

typedef struct {
    bool  is_full, want_t3_exact;
    int   want_ctx, want_eot_id, want_resumed_at;
    float want_alpha;
    char  want_out_path[512];
    char  want_out_sha[65];
    char  base_path[4096];
    char  data_path[4096];
    char  from_path[4096];
    char  teacher_dir[4096];
    char  eval_specs[8][8256];
    char  kill_metric[128];
    char  tmp_out[4096];
    // R17.9 step 4, attest replay --segment A..B: the record's checkpoints,
    // checked against its chain, the step-A files beside it, and the segment
    bool  segment;
    int   seg_a, seg_b, seg_every, seg_n;
    char  seg_start[4096];
    struct { int step; long long cursor; char adapter[65], optim[65]; } seg_ck[256];
} cli_replay;

typedef struct {
    const char *runner_version;
    const char *build_flavor;
    const char *build_commit;
    bool        build_dirty;
    char      **environ_list;
} train_record_host;

typedef struct {
    const char *model_path;
    const char *train_path, *train_out, *dpo_path;
    const char *train_teacher, *lora_path, *train_resume;
    const char *trust_key, *sign_key, *stop_file, *train_lp_path;
    const char *replay_train, *replay_data;
    const char *replay_segment;   // attest replay --segment A..B (R17.9 step 4)
    int         replay_stop_at;   // the step a segment replay stops after
    const char *unlearn_parent, *unlearn_drop, *unlearn_audit;
    const char *unlearn_parent_data, *unlearn_dropped, *unlearn_retained;
    int         unlearn_count, unlearn_distinct;
    bool        allow_same_binary, no_bos, train_eot, train_kv_f32;
    bool        train_full, train_gpu, seed_given;
    size_t      train_gpu_budget;
    int         lora_rank, train_steps, train_ctx, replay_resumed_at;
    int         save_every, lr_schedule, lr_warmup, train_batch;
    int         n_eval, kill_at;
    uint64_t    seed;
    float       lora_scale, train_lr, dpo_beta, weight_decay, lr_floor, kill_min_rel;
    const char *eval_spec[8];
    const char *kill_metric;
    struct {
        bool  kv_f32;
        float rope_base, rope_scale, yarn_factor;
    } mp;
} train_record_opts;

bool train_env_counts(const char *name, size_t n);
bool train_env_is_secret(const char *name, size_t n);
bool train_record_num(jv *obj, const char *key, double min, double max, double *out);
bool train_record_int(jv *obj, const char *key, int min, int max, int *out);
bool train_record_jv_same(const jv *a, const jv *b);
bool train_record_unlearn_split(const char *data1, const char *drop, sbuf *kept, int *count,
                                int *distinct, char *why, size_t cap);
bool train_record_fail(int *rc, const char *field, const char *want, const char *got);
bool train_record_prepare(const train_record_host *host, train_record_opts *o, cli_replay *rp, int *rc);

#endif
