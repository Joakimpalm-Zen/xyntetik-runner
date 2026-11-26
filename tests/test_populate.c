// R18.13: Linux post-bind PTE population (MADV_POPULATE_READ).
//
// Cross-platform checks:
//   - model_populate_wanted() admission policy (budget, opt-out, mlock,
//     oversubscribed moe_prefetch);
//   - model_populate_ranges() skips the GGUF header prefix on CPU, yields 0 on
//     header-only and device-resident loads, and selects only host-computed
//     tensors on partial CUDA and --cpu-moe (including RUNNER_MOE_HOT splits).
//
// Linux-only checks (cleanly skipped off Linux):
//   - /proc/self/smaps Rss of the weight mapping is near the mapped data size
//     with populate on (default) and small with RUNNER_POPULATE=0;
//   - reports tiny-fixture load time with and without populate;
//   - --info / header-only (gguf_open_header) and plain gguf_open populate
//     nothing;
//   - a device-resident load populates nothing;
//   - --parallel N populates once (both shared m->W and unshared file-id
//     instances);
//   - a cgroup v2 memory.high tighter than the model suppresses populate.
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else       printf("ok: %s\n", what);
}

// ---- minimal valid 1-layer llama GGUF writer (~6.4 MB, 12 tensors) ---------
// Places small 1D norms right after data_start and three 2 MiB FFN matrices at
// the end so Linux's 64 KiB fault-around on header/norm reads touches <5% of
// the mapping when RUNNER_POPULATE=0, while populate ON brings Rss to ~100%.

typedef struct {
    FILE *f;
    int   err;
} gw;

static void gw_bytes(gw *w, const void *p, size_t n) {
    if (!w->err && n && fwrite(p, 1, n, w->f) != n) w->err = 1;
}
static void gw_u32(gw *w, uint32_t v) { gw_bytes(w, &v, 4); }
static void gw_u64(gw *w, uint64_t v) { gw_bytes(w, &v, 8); }
static void gw_f32(gw *w, float v)    { gw_bytes(w, &v, 4); }
static void gw_str(gw *w, const char *s) {
    uint64_t n = strlen(s);
    gw_u64(w, n);
    gw_bytes(w, s, (size_t)n);
}
static void gw_kv_str(gw *w, const char *k, const char *v) {
    gw_str(w, k); gw_u32(w, GGUF_T_STR); gw_str(w, v);
}
static void gw_kv_u32(gw *w, const char *k, uint32_t v) {
    gw_str(w, k); gw_u32(w, GGUF_T_U32); gw_u32(w, v);
}
static void gw_kv_f32(gw *w, const char *k, float v) {
    gw_str(w, k); gw_u32(w, GGUF_T_F32); gw_f32(w, v);
}

static bool write_tiny_llama_gguf(const char *path) {
    enum {
        N_VOCAB = 64,
        N_EMBD  = 64,
        N_HEAD  = 2,
        N_KV    = 2,
        N_FF    = 8192,  // 3 * 64 * 8192 * 4 = 6.29 MB of FFN weights
        N_CTX   = 128,
        ALIGN   = 32
    };
    struct tdesc { const char *name; uint32_t nd; uint64_t ne[2]; uint64_t nbytes; uint64_t off; };
    struct tdesc ts[] = {
        // Small 1D norms first (read during model_load)
        { "output_norm.weight",       1, { N_EMBD, 0 },       (uint64_t)N_EMBD * 4, 0 },
        { "blk.0.attn_norm.weight",   1, { N_EMBD, 0 },       (uint64_t)N_EMBD * 4, 0 },
        { "blk.0.ffn_norm.weight",    1, { N_EMBD, 0 },       (uint64_t)N_EMBD * 4, 0 },
        // Small 2D attention / embedding weights
        { "token_embd.weight",        2, { N_EMBD, N_VOCAB }, (uint64_t)N_EMBD * N_VOCAB * 4, 0 },
        { "output.weight",            2, { N_EMBD, N_VOCAB }, (uint64_t)N_EMBD * N_VOCAB * 4, 0 },
        { "blk.0.attn_q.weight",      2, { N_EMBD, N_EMBD },  (uint64_t)N_EMBD * N_EMBD * 4, 0 },
        { "blk.0.attn_k.weight",      2, { N_EMBD, N_EMBD },  (uint64_t)N_EMBD * N_EMBD * 4, 0 },
        { "blk.0.attn_v.weight",      2, { N_EMBD, N_EMBD },  (uint64_t)N_EMBD * N_EMBD * 4, 0 },
        { "blk.0.attn_output.weight", 2, { N_EMBD, N_EMBD },  (uint64_t)N_EMBD * N_EMBD * 4, 0 },
        // Large 2D FFN weights (6.29 MB total, never decoded at load)
        { "blk.0.ffn_gate.weight",    2, { N_EMBD, N_FF },    (uint64_t)N_EMBD * N_FF * 4, 0 },
        { "blk.0.ffn_up.weight",      2, { N_EMBD, N_FF },    (uint64_t)N_EMBD * N_FF * 4, 0 },
        { "blk.0.ffn_down.weight",    2, { N_FF, N_EMBD },    (uint64_t)N_FF * N_EMBD * 4, 0 },
    };
    const size_t nt = sizeof ts / sizeof *ts;
    uint64_t cur = 0;
    for (size_t i = 0; i < nt; i++) {
        cur = (cur + ALIGN - 1) & ~(uint64_t)(ALIGN - 1);
        ts[i].off = cur;
        cur += ts[i].nbytes;
    }

    FILE *f = fopen(path, "wb");
    if (!f) return false;
    gw w = { f, 0 };

    // Header: magic "GGUF", version 3, n_tensors, n_kv (9)
    gw_u32(&w, 0x46554747u);
    gw_u32(&w, 3u);
    gw_u64(&w, (uint64_t)nt);
    gw_u64(&w, 9u);

    gw_kv_str(&w, "general.architecture", "llama");
    gw_kv_u32(&w, "general.alignment", ALIGN);
    gw_kv_u32(&w, "llama.block_count", 1);
    gw_kv_u32(&w, "llama.embedding_length", N_EMBD);
    gw_kv_u32(&w, "llama.feed_forward_length", N_FF);
    gw_kv_u32(&w, "llama.attention.head_count", N_HEAD);
    gw_kv_u32(&w, "llama.attention.head_count_kv", N_KV);
    gw_kv_u32(&w, "llama.context_length", N_CTX);
    gw_kv_f32(&w, "llama.attention.layer_norm_rms_epsilon", 1e-5f);

    for (size_t i = 0; i < nt; i++) {
        gw_str(&w, ts[i].name);
        gw_u32(&w, ts[i].nd);
        for (uint32_t d = 0; d < ts[i].nd; d++) gw_u64(&w, ts[i].ne[d]);
        gw_u32(&w, 0u); // F32
        gw_u64(&w, ts[i].off);
    }

    long pos = ftell(f);
    if (pos < 0) { fclose(f); return false; }
    size_t data_start = ((size_t)pos + ALIGN - 1) & ~(size_t)(ALIGN - 1);
    uint8_t zero[ALIGN] = { 0 };
    if (data_start > (size_t)pos) gw_bytes(&w, zero, data_start - (size_t)pos);

    // Fill with non-zero finite F32 values so disk blocks are real, not sparse holes
    float *chunk = malloc(65536);
    if (!chunk) { fclose(f); return false; }
    for (size_t i = 0; i < 65536 / sizeof(float); i++) chunk[i] = 0.01f;

    uint64_t written = 0;
    for (size_t i = 0; i < nt && !w.err; i++) {
        if (ts[i].off > written) {
            gw_bytes(&w, zero, (size_t)(ts[i].off - written));
            written = ts[i].off;
        }
        uint64_t rem = ts[i].nbytes;
        while (rem > 0 && !w.err) {
            size_t step = rem > 65536 ? 65536 : (size_t)rem;
            gw_bytes(&w, chunk, step);
            rem -= step;
            written += step;
        }
    }
    free(chunk);
    fflush(f);
#ifndef _WIN32
    (void)fsync(fileno(f));
#endif
    int cerr = fclose(f);
    return !w.err && cerr == 0;
}

static model_params pop_params(void) {
    model_params p;
    memset(&p, 0, sizeof p);
    p.gpu_mode = GPU_OFF;
    p.n_threads = 1;
    p.n_ctx = 64;
    p.n_batch = 8;
    return p;
}

static void test_policy(void) {
    ck(model_populate_wanted(1000, 2000, false, false, false),
       "policy: fitting unlocked model populates by default");
    ck(model_populate_wanted(2000, 2000, false, false, false),
       "policy: exact budget boundary (bytes == available) populates");
    ck(!model_populate_wanted(2001, 2000, false, false, false),
       "policy: model past available budget does not populate");
    ck(!model_populate_wanted(1000, 2000, false, false, true),
       "policy: RUNNER_POPULATE=0 opt-out suppresses populate");
    ck(!model_populate_wanted(1000, 2000, true, false, false),
       "policy: --mlock suppresses populate");
    ck(!model_populate_wanted(1000, 2000, false, true, false),
       "policy: oversubscribed moe_prefetch suppresses populate");
    ck(!model_populate_wanted(0, 2000, false, false, false),
       "policy: 0 host bytes does not populate");
    ck(!model_populate_wanted(1000, 0, false, false, false),
       "policy: unknown/0 available budget does not populate");
}

static void test_range_planning(const char *path) {
    // 1. Header-only mapping produces 0 ranges
    model_t hm;
    memset(&hm, 0, sizeof hm);
    ck(gguf_open_header(&hm.gf, path), "plan: gguf_open_header opens fixture");
    uint64_t bytes = 999;
    ck(model_populate_ranges(&hm, NULL, 0, &bytes) == 0 && bytes == 0,
       "plan: header-only mapping yields 0 ranges");
    gguf_close(&hm.gf);

    // 2. Pure CPU mapping skips header prefix [0 .. data_offset)
    model_t cm;
    memset(&cm, 0, sizeof cm);
    ck(gguf_open(&cm.gf, path), "plan: gguf_open opens fixture");
    size_t doff = gguf_map_data_offset(&cm.gf, 0);
    ck(doff > 0 && doff < cm.gf.map_size, "plan: data_offset skips GGUF header prefix");
    model_mem_range r[4];
    size_t nr = model_populate_ranges(&cm, r, 4, &bytes);
    ck(nr == 1 && r[0].addr == (const uint8_t *)cm.gf.map + doff &&
       r[0].len == cm.gf.map_size - doff && bytes == cm.gf.map_size - doff,
       "plan: CPU range is [map + data_offset, map + map_size)");
    gguf_close(&cm.gf);

    // 3. Synthetic GPU/partial/--cpu-moe/--cpu-moe+hot planning
    static uint8_t dummy_map[4096];
    gguf_tensor t_emb  = { .data = dummy_map + 0,    .nbytes = 128 };
    gguf_tensor t_out  = { .data = dummy_map + 128,  .nbytes = 128 };
    gguf_tensor t_q0   = { .data = dummy_map + 256,  .nbytes = 64 };
    gguf_tensor t_q1   = { .data = dummy_map + 320,  .nbytes = 64 };
    gguf_tensor t_inp0 = { .data = dummy_map + 384,  .nbytes = 32 };
    gguf_tensor t_ge0  = { .data = dummy_map + 512,  .nbytes = 512, .type = T_F32 };
    gguf_tensor t_ue0  = { .data = dummy_map + 1024, .nbytes = 512, .type = T_F32 };
    gguf_tensor t_de0  = { .data = dummy_map + 1536, .nbytes = 512, .type = T_F32 };
    gguf_tensor t_sh0  = { .data = dummy_map + 2048, .nbytes = 64 };
    gguf_tensor t_g1   = { .data = dummy_map + 2112, .nbytes = 96 };
    layer_t layers[2];
    memset(layers, 0, sizeof layers);
    layers[0].is_moe = true;
    layers[0].n_expert = 2;
    layers[0].wq = &t_q0;
    layers[0].ffn_gate_inp = &t_inp0;
    layers[0].ffn_gate_exps = &t_ge0;
    layers[0].ffn_up_exps = &t_ue0;
    layers[0].ffn_down_exps = &t_de0;
    layers[0].w_gate_shexp = &t_sh0;
    layers[1].is_moe = false;
    layers[1].wq = &t_q1;
    layers[1].w_gate = &t_g1;

    model_t gm;
    memset(&gm, 0, sizeof gm);
    gm.gf.map = dummy_map;
    gm.gf.map_size = sizeof dummy_map;
    gm.gf.mapped_size = sizeof dummy_map;
    gm.gpu = (void *)(uintptr_t)1;
    gm.n_layer = 2;
    gm.n_embd = 8;
    gm.n_ff_exp = 8; // 8 * 8 * 4 = 256 bytes per expert (2 experts = 512 bytes)
    gm.n_expert = 2;
    gm.layers = layers;
    gm.tok_embd = &t_emb;
    gm.output = &t_out;

    // 3a. Full device offload -> 0 ranges
    gm.gpu_layers = 2;
    ck(model_populate_ranges(&gm, NULL, 0, &bytes) == 0 && bytes == 0,
       "plan: full device offload yields 0 host ranges");

    if (!gpu_unified_memory()) {
        // 3b. Partial discrete CUDA (layer 0 on GPU, layer 1 + output on host)
        gm.gpu_layers = 1;
        model_mem_range pr[16];
        size_t pn = model_populate_ranges(&gm, pr, 16, &bytes);
        ck(pn == 3 && bytes == t_q1.nbytes + t_g1.nbytes + t_out.nbytes,
           "plan: partial CUDA selects only CPU layer 1 + output (no layer 0, no tok_embd)");

        // 3c. --cpu-moe full attention offload (5060 Ti case): layer 0 experts on host
        bool moe_host[2] = { true, false };
        gm.gpu_layers = 2;
        gm.cpu_moe = true;
        gm.moe_host = moe_host;
        pn = model_populate_ranges(&gm, pr, 16, &bytes);
        uint64_t want_moe = t_inp0.nbytes + t_ge0.nbytes + t_ue0.nbytes +
                            t_de0.nbytes + t_sh0.nbytes;
        ck(pn == 5 && bytes == want_moe,
           "plan: --cpu-moe selects only host expert bank (skips attention and layer 1)");

        // 3d. --cpu-moe + RUNNER_MOE_HOT: expert 0 in VRAM, expert 1 on host
        int hot_n[2] = { 1, 0 };
        int hot_slot[4] = { 0, -1, -1, -1 }; // layer 0: exp 0 in VRAM, exp 1 on host
        gm.moe_hot_n = hot_n;
        gm.moe_hot_slot = hot_slot;
        pn = model_populate_ranges(&gm, pr, 16, &bytes);
        uint64_t want_hot = t_inp0.nbytes + 256 + 256 + 256 + t_sh0.nbytes;
        ck(pn == 5 && bytes == want_hot,
           "plan: --cpu-moe + RUNNER_MOE_HOT skips VRAM hot expert and populates only host expert");
    } else {
        // Unified-memory GPU (Metal): always device-resident -> 0 ranges
        gm.gpu_layers = 1;
        ck(model_populate_ranges(&gm, NULL, 0, &bytes) == 0 && bytes == 0,
           "plan: unified-memory GPU (Metal) yields 0 host ranges");
    }
}

#ifdef __linux__
// Sum Rss (in bytes) from /proc/self/smaps for all VMAs overlapping [base, base+len).
static int64_t smaps_rss_bytes(const void *base, size_t len) {
    FILE *f = fopen("/proc/self/smaps", "r");
    if (!f) return -1;
    uintptr_t tlo = (uintptr_t)base;
    uintptr_t thi = tlo + len;
    bool in_vma = false;
    uint64_t rss_kb = 0;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        unsigned long vlo = 0, vhi = 0;
        if (line[0] != ' ' && line[0] != '\t' && strchr(line, '-') &&
            sscanf(line, "%lx-%lx ", &vlo, &vhi) == 2) {
            in_vma = ((uintptr_t)vlo < thi && (uintptr_t)vhi > tlo);
            continue;
        }
        if (in_vma) {
            unsigned long kb = 0;
            if (sscanf(line, "Rss: %lu kB", &kb) == 1) rss_kb += kb;
        }
    }
    fclose(f);
    return (int64_t)(rss_kb * 1024ull);
}

static bool write_text_file(const char *path, const char *txt) {
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    size_t n = strlen(txt);
    bool ok = fwrite(txt, 1, n, f) == n;
    return fclose(f) == 0 && ok;
}

static void test_linux_smaps(const char *path) {
    model_params p = pop_params();

    // Prime one-time process init (f16 table, capacity read) so timed loads
    // compare populate ON vs OFF cleanly.
    f16_init();
    (void)plat_ram_available_bytes();

    // 1. Default (populate ON): Rss near mapped data size
    unsetenv("RUNNER_POPULATE");
    model_t m;
    double t0 = plat_now();
    bool ok = model_load(&m, path, &p);
    double ms_on = (plat_now() - t0) * 1000.0;
    ck(ok, "linux: model_load succeeds with populate ON");
    size_t doff = gguf_map_data_offset(&m.gf, 0);
    size_t data_sz = m.gf.map_size > doff ? m.gf.map_size - doff : 0;
    int64_t rss_on = smaps_rss_bytes(m.gf.map, m.gf.map_size);
    uint64_t pop_on = model_populated_bytes(&m);
    printf("populate: tiny fixture (%.2f MiB data): ON  Rss=%.2f MiB (populated=%.2f MiB) load=%.2f ms\n",
           data_sz / 1048576.0, rss_on / 1048576.0, pop_on / 1048576.0, ms_on);
    ck(pop_on == data_sz, "linux: populate ON reports full data_sz populated");
    ck(rss_on >= (int64_t)(data_sz * 9 / 10),
       "linux: /proc/self/smaps Rss is >=90% of mapped data size with populate ON");

    // 2. --parallel 2 (shared m->W): second slot populates 0 bytes
    model_t m2;
    ck(model_load(&m2, path, &p), "linux: second parallel slot loads");
    ck(model_populated_bytes(&m2) == 0,
       "linux: --parallel 2 shared slot populates once (slot 2 populated 0 bytes)");
    model_free(&m2);
    model_free(&m);

    // 3. RUNNER_POPULATE=0: Rss stays small (<25% of mapped data size)
    setenv("RUNNER_POPULATE", "0", 1);
    t0 = plat_now();
    ok = model_load(&m, path, &p);
    double ms_off = (plat_now() - t0) * 1000.0;
    ck(ok, "linux: model_load succeeds with RUNNER_POPULATE=0");
    int64_t rss_off = smaps_rss_bytes(m.gf.map, m.gf.map_size);
    printf("populate: tiny fixture (%.2f MiB data): OFF Rss=%.2f MiB (populated=%llu B) load=%.2f ms\n",
           data_sz / 1048576.0, rss_off / 1048576.0,
           (unsigned long long)model_populated_bytes(&m), ms_off);
    ck(model_populated_bytes(&m) == 0,
       "linux: RUNNER_POPULATE=0 reports 0 populated bytes");
    ck(rss_off >= 0 && rss_off < (int64_t)(data_sz / 4),
       "linux: /proc/self/smaps Rss stays small (<25%) with RUNNER_POPULATE=0");
    model_free(&m);
    unsetenv("RUNNER_POPULATE");

    // 4. --info / header-only (gguf_open_header) and plain gguf_open populate nothing
    gguf_file hg;
    ck(gguf_open_header(&hg, path), "linux: gguf_open_header (--info path) opens fixture");
    int64_t rss_hdr = smaps_rss_bytes(hg.map, hg.map_size);
    printf("populate: header-only (--info) Rss=%.2f MiB\n", rss_hdr / 1048576.0);
    ck(rss_hdr >= 0 && rss_hdr < (int64_t)(data_sz / 4),
       "linux: --info / gguf_open_header populates nothing (<25% Rss)");
    gguf_close(&hg);

    gguf_file pg;
    ck(gguf_open(&pg, path), "linux: plain gguf_open opens fixture");
    int64_t rss_open = smaps_rss_bytes(pg.map, pg.map_size);
    ck(rss_open >= 0 && rss_open < (int64_t)(data_sz / 4),
       "linux: plain gguf_open populates nothing (<25% Rss)");
    gguf_close(&pg);

    // 5. Device-resident load populates nothing
    model_t dm;
    memset(&dm, 0, sizeof dm);
    ck(gguf_open(&dm.gf, path), "linux: map fixture for device-resident check");
    dm.gpu = (void *)(uintptr_t)1;
    dm.n_layer = 1;
    dm.gpu_layers = 1;
    uint64_t pop_dev = model_populate_weights(&dm, false);
    int64_t rss_dev = smaps_rss_bytes(dm.gf.map, dm.gf.map_size);
    printf("populate: device-resident Rss=%.2f MiB (populated=%llu B)\n",
           rss_dev / 1048576.0, (unsigned long long)pop_dev);
    ck(pop_dev == 0 && rss_dev >= 0 && rss_dev < (int64_t)(data_sz / 4),
       "linux: device-resident load populates nothing");
    gguf_close(&dm.gf);

    // 6. Non-shared populate-once (Inkling 2 --parallel N pattern: m->W == NULL)
    model_t u1, u2;
    memset(&u1, 0, sizeof u1);
    memset(&u2, 0, sizeof u2);
    ck(gguf_open(&u1.gf, path) && gguf_open(&u2.gf, path),
       "linux: map two unshared instances");
    u1.file_id_ok = u2.file_id_ok = true;
    u1.file_size = u2.file_size = 123456;
    u1.file_ino  = u2.file_ino  = 7890;
    u1.file_mtime_ns = u2.file_mtime_ns = 111;
    u1.file_ctime_ns = u2.file_ctime_ns = 222;
    ck(model_populate_weights(&u1, false) == data_sz,
       "linux: unshared slot 1 populates data_sz");
    ck(model_populate_weights(&u2, false) == 0,
       "linux: unshared slot 2 of same file populates 0 bytes (populate once)");
    // Freeing slot 1 unregisters the file id so a fresh load populates again.
    model_free(&u2);
    model_free(&u1);

    // 7. Tight cgroup v2 memory.high suppresses populate even with default ON
    char cgdir[] = "/tmp/r1813-cg-XXXXXX";
    if (mkdtemp(cgdir)) {
        char f1[256], f2[256], f3[256], f4[256];
        snprintf(f1, sizeof f1, "%s/cgroup.controllers", cgdir);
        snprintf(f2, sizeof f2, "%s/memory.max", cgdir);
        snprintf(f3, sizeof f3, "%s/memory.high", cgdir);
        snprintf(f4, sizeof f4, "%s/memory.current", cgdir);
        bool cg_ok = write_text_file(f1, "memory\n") &&
                     write_text_file(f2, "max\n") &&
                     write_text_file(f3, "1048576\n") && // 1 MiB < 6.4 MiB fixture
                     write_text_file(f4, "0\n");
        ck(cg_ok, "linux: created tight cgroup v2 memory.high fixture");
        if (cg_ok) {
            setenv("RUNNER_TEST_CGROUP_ROOT", cgdir, 1);
            ck(model_load(&m, path, &p), "linux: model_load under tight cgroup memory.high");
            int64_t rss_cg = smaps_rss_bytes(m.gf.map, m.gf.map_size);
            ck(model_populated_bytes(&m) == 0 && rss_cg >= 0 && rss_cg < (int64_t)(data_sz / 4),
               "linux: cgroup memory.high tighter than model suppresses populate");
            model_free(&m);
            unsetenv("RUNNER_TEST_CGROUP_ROOT");
        }
        remove(f1); remove(f2); remove(f3); remove(f4);
        rmdir(cgdir);
    }
}
#endif

int main(void) {
    const char *path = "test-populate-tmp.gguf";
    if (!write_tiny_llama_gguf(path)) {
        fprintf(stderr, "FAIL: cannot write tiny fixture %s\n", path);
        return 1;
    }

    test_policy();
    test_range_planning(path);

#ifdef __linux__
    test_linux_smaps(path);
#else
    // Still report tiny-fixture load time with and without on non-Linux hosts
    model_params p = pop_params();
    model_t m;
    // Prime one-time process init (f16 table, Metal/capacity probe)
    if (model_load(&m, path, &p)) model_free(&m);
    double t0 = plat_now();
    bool ok1 = model_load(&m, path, &p);
    double ms_on = (plat_now() - t0) * 1000.0;
    ck(ok1 && model_populated_bytes(&m) == 0,
       "non-linux: model_load succeeds and populate is a no-op");
    model_free(&m);
#ifndef _WIN32
    setenv("RUNNER_POPULATE", "0", 1);
#else
    _putenv("RUNNER_POPULATE=0");
#endif
    t0 = plat_now();
    bool ok2 = model_load(&m, path, &p);
    double ms_off = (plat_now() - t0) * 1000.0;
    ck(ok2 && model_populated_bytes(&m) == 0,
       "non-linux: model_load with RUNNER_POPULATE=0 succeeds");
    model_free(&m);
#ifndef _WIN32
    unsetenv("RUNNER_POPULATE");
#else
    _putenv("RUNNER_POPULATE=");
#endif
    printf("populate: tiny fixture load time (non-Linux no-op): default=%.2f ms, RUNNER_POPULATE=0=%.2f ms\n",
           ms_on, ms_off);
    printf("populate: Linux /proc/self/smaps checks skipped cleanly (Linux only)\n");
#endif

    remove(path);
    if (g_fail) {
        fprintf(stderr, "populate: FAILED\n");
        return 1;
    }
    puts("populate: ok");
    return 0;
}
