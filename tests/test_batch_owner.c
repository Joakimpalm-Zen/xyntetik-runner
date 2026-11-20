// A decode microbatch owns its reference to the device data its members share
// (R3.82, PR 965's re-read).
//
//     ./test-batch-owner-asan     (make test-batch-owner-asan: ASan + UBSan)
//
// gpu_batch borrowed its lead's upload (b->sw = lead->sw) without a reference.
// A failed qwen35 microbatch step releases every member's backend
// (model_batch_decode's recovery, gpu_runtime_failure); when the members were
// the upload's last owners, the last release destroyed it and its context,
// and model_batch_free then read it through the batch (gpu_batch_free's first
// line): a heap use-after-free, reached by test-cuda-batch-q35's fault-multi.
//
// This test includes src/cuda.c and drives the batch's real host-side
// lifetime (batch_host_new, the create path's half before any driver call;
// shared_release, what a member's backend release ends in; gpu_batch_free;
// gpu_batch_release_orphaned) on an upload with no device memory, so it needs
// no driver or device. Under ASan a borrowed upload is a use-after-free and a
// reference never given back is a leak; the reference counts are checked as
// well, so a plain build still fails when the batch takes no reference.
#include "cuda.c"

static int g_fail = 0;

static void ck(int cond, const char *what) {
    printf("%s: %s\n", cond ? "ok" : "FAIL", what);
    if (!cond) g_fail = 1;
}

// one model whose backend holds the upload, as gpu_init leaves it
static gpu_weights *upload(void) {
    gpu_weights *w = calloc(1, sizeof *w);
    if (w) w->refs = 1;
    return w;
}

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);   // the lines before a sanitizer's abort reach the log
    model_t *m = calloc(2, sizeof *m);
    gpu_t *g = calloc(2, sizeof *g);
    if (!m || !g) return 2;
    model_t *seqs[2] = { &m[0], &m[1] };

    // 1. the members release the upload while the batch lives (the recovery
    // of a failed step), then the batch is freed (model_batch_free)
    gpu_weights *w = upload();
    if (!w) return 2;
    w->refs = 2;   // two members' backends
    g[0].sw = g[1].sw = w;
    m[0].gpu = &g[0]; m[1].gpu = &g[1];
    gpu_batch *b = batch_host_new(seqs, 2, &g[0]);
    if (!b) return 2;
    ck(w->refs == 3, "the batch takes its own reference to the members' upload");
    shared_release(w); shared_release(w);
    ck(w->refs == 1, "after both members released their backends the batch still holds the upload");
    gpu_batch_free(b);   // gives the last reference back: the upload is destroyed here, once

    // 2. the lead released its backend: the batch gives its device side back
    // at once, then declines every step, and its free releases nothing twice
    w = upload();
    if (!w) return 2;
    g[0].sw = g[1].sw = w;
    m[0].gpu = &g[0]; m[1].gpu = NULL;
    b = batch_host_new(seqs, 2, &g[0]);
    if (!b) return 2;
    gpu_batch_release_orphaned(b);
    ck(b->sw == w && w->refs == 2, "a batch whose lead still has its backend keeps its device side");
    m[0].gpu = NULL;
    shared_release(w);   // the lead's backend
    ck(w->refs == 1, "the lead's release leaves the batch's reference");
    gpu_batch_release_orphaned(b);   // the last reference: the upload is destroyed here
    ck(b->sw == NULL, "a batch whose lead gave up its backend releases its device side");
    {
        int idx[2] = { 0, 1 }, pos[2] = { 3, 3 };
        int32_t tok[2] = { 1, 2 };
        float *out[2] = { NULL, NULL };
        ck(!gpu_batch_decode(b, idx, tok, pos, 2, out) && !gpu_batch_failed_stepped(b),
           "a released batch declines its step without running it");
    }
    gpu_batch_release_orphaned(b);
    gpu_batch_free(b);

    // 3. a create that fails after the reference was taken (gpu_batch_create's
    // fail path frees the half-made batch): the reference comes back
    w = upload();
    if (!w) return 2;
    g[0].sw = w;
    m[0].gpu = &g[0];
    b = batch_host_new(seqs, 1, &g[0]);
    if (!b) return 2;
    gpu_batch_free(b);
    ck(w->refs == 1, "a batch freed on the partial-create path gives its reference back");
    shared_release(w);

    free(m); free(g);
    printf(g_fail ? "batch-owner: FAILED\n" : "batch-owner: ok\n");
    return g_fail;
}
