// The host MoE bridge against the CPU forward it stands in for.
//
// With --cpu-moe on CUDA a MoE layer's experts run on the host: the device
// runs the mixer, hands the residual rows to model_moe_ffn_cpu(), and resumes
// at the next layer. So the bridge owns the FFN's whole tail (norm, routed
// and shared experts, post-norm, residual scale, the add, the layer's output
// scale) and must compute exactly what forward_layer's FFN computes on the
// CPU. Nothing pinned that: the bridge is reached only on a CUDA host with
// --cpu-moe.
//
// The FFN half of forward_layer is isolated by marking the layer's mixer
// skipped (the state nemotron_h's MoE blocks are in already). Both paths then
// run from the same residual rows over the same primitives, so their outputs
// must be equal bit for bit, on every MoE layer of every fixture given on the
// command line. Hyper-connection models use the premixed bridge and are left
// out here.
//
// model.c is compiled into this test (tests/test_model_load_oom.c's
// pattern) to reach forward_layer.
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../src/model.c"

// The FFN's per-row scratch, filled with a value no computation yields.
static void poison_scratch(model_t *m) {
    size_t rows = (size_t)m->n_batch;
    if (m->shexp_in)
        for (size_t i = 0; i < rows * m->n_embd; i++) m->shexp_in[i] = 1.0e30f;
    for (size_t i = 0; i < rows * m->xdim; i++) m->xb[i] = 1.0e30f;
}

static int check_fixture(const char *path) {
    model_params p;
    memset(&p, 0, sizeof p);
    p.gpu_mode = GPU_OFF;
    p.n_threads = 1;
    p.n_ctx = 64;
    p.n_batch = 4;
    static model_t m;
    if (!model_load(&m, path, &p)) {
        fprintf(stderr, "FAIL: %s: does not load\n", path);
        return 1;
    }
    const int n = 3, E = m.n_embd;
    size_t bytes = sizeof(float) * (size_t)n * E;
    float *x0 = malloc(bytes), *want = malloc(bytes);
    int bad = 0, layers = 0;
    for (int l = 0; l < m.n_layer && x0 && want && !m.hyper_conn; l++) {
        layer_t *ly = &m.layers[l];
        if (!ly->is_moe) continue;
        layers++;
        for (size_t i = 0; i < (size_t)n * E; i++)
            x0[i] = sinf(0.37f * (float)i + 0.11f * (float)l);
        bool skip = ly->skip_mixer;
        ly->skip_mixer = true;
        poison_scratch(&m);
        memcpy(m.x, x0, bytes);
        forward_layer(&m, l, n, 0, 0);
        memcpy(want, m.x, bytes);
        ly->skip_mixer = skip;
        // each path must write every scratch row it reads: what the other
        // left behind would hide one that does not
        poison_scratch(&m);
        memcpy(m.x, x0, bytes);
        if (!model_moe_ffn_cpu(&m, l, n)) {
            fprintf(stderr, "FAIL: %s: layer %d: the bridge refused a MoE layer\n", path, l);
            bad++;
            continue;
        }
        if (memcmp(m.x, want, bytes) != 0) {
            double worst = 0;
            for (size_t i = 0; i < (size_t)n * E; i++) {
                double d = fabs((double)m.x[i] - (double)want[i]);
                if (d > worst) worst = d;
            }
            fprintf(stderr, "FAIL: %s: layer %d: the bridge's residual differs from the "
                    "forward's (worst |diff| %.3g)\n", path, l, worst);
            bad++;
        }
    }
    if (!layers && !m.hyper_conn) {
        fprintf(stderr, "FAIL: %s: no MoE layer to check\n", path);
        bad++;
    }
    free(x0);
    free(want);
    model_free(&m);
    return bad;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: test-moe-host-bridge FIXTURE.gguf...\n");
        return 2;
    }
    int bad = 0;
    for (int i = 1; i < argc; i++) bad += check_fixture(argv[i]);
    if (bad) {
        fprintf(stderr, "test-moe-host-bridge FAILED (%d)\n", bad);
        return 1;
    }
    fprintf(stderr, "test-moe-host-bridge: %d fixtures, every MoE layer's bridge equals "
            "the forward's FFN\n", argc - 1);
    return 0;
}
