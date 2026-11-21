// R4.48: the IDENTITY TEST of the test-side Inkling-2-Mini weight provider (tests/inkling2_provider.c) against the
// reference loader of transformers pull request 51050 at 6173a2f6 (tests/inkling2_provider_ref.py: the PR's own
// conversion classes on the same source tensors, read with safetensors' safe_open, float32).
//
//   1. The map: every parameter of the reference model (real-shapes.txt, written by the reference's own class on the
//      meta device) is served, with its shape; any that is not is printed, and the test fails.
//   2. Identity, BIT FOR BIT: every non-expert parameter whole; every expert stack on 8 experts per layer (layers
//      other than the two full ones); all 128 experts of one NVFP4 layer and one MXFP8 layer (3 and 21 by default),
//      both through get_expert and through the whole-stack get. A table per tensor class (the name with the layer
//      number replaced, the source format): tensors, elements, differing elements and bits (must be 0).
//   3. Controls (wrong decodes, which must differ): the NVFP4 factor omitted, the nibble order swapped, the
//      gate_up_proj interleave omitted (on an MXFP8 layer).
//   4. The decode time of one token's experts (8 a layer application), measured per expert, and the memory used.
//
// usage: test-inkling2-provider CKPT.safetensors MAP REAL_SHAPES CKPT_DIR WORKDIR [NVFP4_LAYER MXFP8_LAYER]
//   env INK2P_PYTHON (default python3) and PYTHONPATH (the PR's src) for the
//   reference side, INK2P_REF its script (default tests/inkling2_provider_ref.py).
#define _GNU_SOURCE
#include "inkling2_provider.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// ---------------------------------------------------------------- per-class tallies
typedef struct { char cls[384]; long tensors, experts; uint64_t elems, diff_elems, diff_bits; } row_t;
static row_t rows[256]; static int n_rows;
static row_t *row_for(const char *cls) {
    for (int i = 0; i < n_rows; i++) if (!strcmp(rows[i].cls, cls)) return &rows[i];
    row_t *r = &rows[n_rows++]; memset(r, 0, sizeof *r); snprintf(r->cls, sizeof r->cls, "%s", cls);
    return r;
}
// the class of a parameter: its name with the layer number replaced, and its source format and rank
static void class_of(const ink2p *p, int i, char *out, size_t n) {   // out: 384 bytes
    int nd; const char *ops, *dt; const char *name = ink2p_name(p, i, &nd, NULL, &ops, &dt);
    char pat[200]; size_t k = 0;
    for (const char *c = name; *c && k + 2 < sizeof pat; c++) {
        if (isdigit((unsigned char)*c) && c > name && c[-1] == '.') { while (isdigit((unsigned char)c[1])) c++; pat[k++] = 'N'; }
        else pat[k++] = *c;
    }
    pat[k] = 0;
    const char *fmt = ink2p_is_stack(p, i) ? (!strcmp(dt, "U8") ? "NVFP4" : "MXFP8") : dt;
    snprintf(out, n, "%s [%s %d-D]", pat, fmt, nd);
}

// ---------------------------------------------------------------- the reference stream
static FILE *ref;
static int read_hdr(const char *want, int e, uint64_t want_bytes) {
    char line[512];
    if (!fgets(line, sizeof line, ref)) { fprintf(stderr, "reference stream ended before %s %d\n", want, e); return -1; }
    char name[256]; int ge; unsigned long long nb;
    if (sscanf(line, "D %255s %d %llu", name, &ge, &nb) != 3 || strcmp(name, want) || ge != e || nb != want_bytes) {
        fprintf(stderr, "reference stream: got \"%.200s\", want %s %d %llu bytes\n", line, want, e, (unsigned long long)want_bytes);
        return -1;
    }
    return 0;
}
// compare n floats of ours with the next n*4 bytes of the stream; adds to *de, *db
static unsigned char *chunk; static const size_t CHUNK = 64u << 20;
static int cmp_stream(const float *ours, size_t n, uint64_t *de, uint64_t *db) {
    const unsigned char *o = (const unsigned char *)ours;
    size_t left = n * 4, off = 0;
    while (left) {
        size_t m = left < CHUNK ? left : CHUNK;
        if (fread(chunk, 1, m, ref) != m) { fprintf(stderr, "reference stream short\n"); return -1; }
        for (size_t i = 0; i < m; i += 4) {
            uint32_t a, b; memcpy(&a, o + off + i, 4); memcpy(&b, chunk + i, 4);
            if (a != b) { (*de)++; *db += (uint64_t)__builtin_popcount(a ^ b); }
        }
        off += m; left -= m;
    }
    return 0;
}

// ---------------------------------------------------------------- main
int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s CKPT MAP REAL_SHAPES CKPT_DIR WORKDIR [NVFP4_LAYER MXFP8_LAYER]\n", argv[0]); return 2; }
    const char *ckpt = argv[1], *mapf = argv[2], *shapes = argv[3], *ckdir = argv[4], *work = argv[5];
    int full_nv = argc > 6 ? atoi(argv[6]) : 3, full_mx = argc > 7 ? atoi(argv[7]) : 21;
    char why[512];
    double t_open = now_s();
    ink2p *p = ink2p_open(ckpt, mapf, (size_t)3 << 30, (size_t)512 << 20, why, sizeof why);
    if (!p) { printf("FAIL open: %s\n", why); return 1; }
    printf("provider: %d parameters in the map, opened and checked against the file's header in %.2f s; caches: tensors 3 GiB, "
           "experts 512 MiB (pins: the last %d gets, the last %d get_expert)\n", ink2p_count(p), now_s() - t_open,
           INK2P_PIN_TENSORS, INK2P_PIN_EXPERTS);
    int fails = 0;

    // 1. the map against the reference model's own names and shapes
    {
        FILE *f = fopen(shapes, "r");
        if (!f) { printf("FAIL cannot open %s\n", shapes); return 1; }
        char line[1024]; int n = 0, ok = 0;
        int loop_s = -1, loop_e = -1, passes = -1, n_layer = -1;
        while (fgets(line, sizeof line, f)) {
            char *save = NULL, *t = strtok_r(line, " \n", &save);
            if (!t) continue;
            if (!strcmp(t, "I")) {
                char *k = strtok_r(NULL, " \n", &save), *v = strtok_r(NULL, " \n", &save);
                if (k && v && !strcmp(k, "loop_layer_index_start")) loop_s = atoi(v);
                if (k && v && !strcmp(k, "loop_layer_index_end")) loop_e = atoi(v);
                if (k && v && !strcmp(k, "loop_num_passes")) passes = atoi(v);
                if (k && v && !strcmp(k, "num_hidden_layers")) n_layer = atoi(v);
                continue;
            }
            if (strcmp(t, "T")) continue;
            char *name = strtok_r(NULL, " \n", &save);
            int64_t d[4]; int nd = 0;
            while ((t = strtok_r(NULL, " \n", &save)) && nd < 4) d[nd++] = strtoll(t, NULL, 10);
            n++;
            int i = ink2p_find(p, name);
            if (i < 0) { printf("UNMAPPED %s\n", name); continue; }
            int pnd; int64_t pd[4]; ink2p_name(p, i, &pnd, pd, NULL, NULL);
            int same = pnd == nd;
            for (int k = 0; k < nd && same; k++) same = pd[k] == d[k];
            if (!same) { printf("SHAPE %s: map", name); for (int k = 0; k < pnd; k++) printf(" %lld", (long long)pd[k]);
                         printf(", reference"); for (int k = 0; k < nd; k++) printf(" %lld", (long long)d[k]); printf("\n"); continue; }
            ok++;
        }
        fclose(f);
        int extra = ink2p_count(p) - ok;
        printf("%s map: %d of %d reference parameters mapped with their shapes; %d map entries beyond them\n",
               ok == n && n > 0 && extra == 0 ? "GREEN" : "FAIL", ok, n, extra);
        if (!(ok == n && n > 0 && extra == 0)) fails++;
        // the application counts by expert format, for the timing estimate below
        if (n_layer > 0 && loop_s >= 0) {
            int apps_nv = 0, apps_mx = 0;
            for (int b = 0; b < n_layer; b++) {
                char nm[160]; snprintf(nm, sizeof nm, "model.layers.%d.mlp.experts.gate_up_proj", b);
                int i = ink2p_find(p, nm); const char *dt; ink2p_name(p, i, NULL, NULL, NULL, &dt);
                int m = b >= loop_s && b < loop_e ? passes : 1;
                if (!strcmp(dt, "U8")) apps_nv += m; else apps_mx += m;
            }
            char a[32]; snprintf(a, sizeof a, "%d %d", apps_nv, apps_mx); setenv("INK2P_APPS", a, 1);
        }
    }

    // 2. the manifest: every non-stack parameter whole; 8 experts of each stack on the sampled layers; the two full
    // layers' stacks whole; the controls
    char man[600]; snprintf(man, sizeof man, "%s/manifest.txt", work);
    FILE *mf = fopen(man, "w");
    if (!mf) { printf("FAIL cannot write %s\n", man); return 1; }
    int n_ent = ink2p_count(p);
    for (int i = 0; i < n_ent; i++) if (!ink2p_is_stack(p, i)) fprintf(mf, "W %s\n", ink2p_name(p, i, NULL, NULL, NULL, NULL));
    for (int i = 0; i < n_ent; i++) {
        if (!ink2p_is_stack(p, i)) continue;
        const char *nm = ink2p_name(p, i, NULL, NULL, NULL, NULL);
        int b = atoi(nm + strlen("model.layers."));
        if (b == full_nv || b == full_mx) { fprintf(mf, "S %s\n", nm); continue; }
        fprintf(mf, "X %s ", nm);
        for (int k = 0; k < 8; k++) fprintf(mf, "%s%d", k ? "," : "", (b * 13 + k * 17) % 128);
        fprintf(mf, "\n");
    }
    fprintf(mf, "C nofactor model.layers.%d.mlp.experts.gate_up_proj 5,77\n", full_nv);
    fprintf(mf, "C swapnibble model.layers.%d.mlp.experts.down_proj 5,77\n", full_nv);
    fprintf(mf, "C nointerleave model.layers.%d.mlp.experts.gate_up_proj 7\n", full_mx);
    fprintf(mf, "C nointerleave model.layers.%d.mlp.experts.gate_up_proj 9\n", full_nv);
    fclose(mf);

    const char *py = getenv("INK2P_PYTHON"); if (!py) py = "python3";
    const char *rs = getenv("INK2P_REF"); if (!rs) rs = "tests/inkling2_provider_ref.py";
    char cmd[2048]; snprintf(cmd, sizeof cmd, "nice -n 19 %s %s %s %s %s", py, rs, ckdir, ckpt, man);
    ref = popen(cmd, "r");
    chunk = malloc(CHUNK);
    if (!ref || !chunk) { printf("FAIL cannot start the reference: %s\n", cmd); return 1; }
    double t0 = now_s();
    FILE *mr = fopen(man, "r"); char line[4096]; int bad_stream = 0;
    typedef struct { char kind[256]; uint64_t de, db, el; int n; } ctl_t;
    ctl_t ctl[8]; int n_ctl = 0;
    int done_items = 0;
    while (!bad_stream && fgets(line, sizeof line, mr)) {
        char k[4], a[256], b[256], c[1024];
        int nf = sscanf(line, "%3s %255s %255s %1023s", k, a, b, c);
        if (nf < 2) continue;
        if (k[0] == 'W') {
            int i = ink2p_find(p, a); int nd; int64_t d[4]; ink2p_name(p, i, &nd, d, NULL, NULL);
            size_t n = 1; for (int q = 0; q < nd; q++) n *= (size_t)d[q];
            const ink2_tensor *t = ink2p_get(p, a);
            if (!t || read_hdr(a, -1, n * 4)) { bad_stream = 1; break; }
            char cls[384]; class_of(p, i, cls, sizeof cls); row_t *r = row_for(cls);
            if (cmp_stream(t->data, n, &r->diff_elems, &r->diff_bits)) { bad_stream = 1; break; }
            r->tensors++; r->elems += n;
        } else if (k[0] == 'X' || k[0] == 'S' || k[0] == 'C') {
            const char *nm = k[0] == 'C' ? b : a, *list = k[0] == 'X' ? b : k[0] == 'C' ? c : NULL;
            int i = ink2p_find(p, nm); int nd; int64_t d[4]; ink2p_name(p, i, &nd, d, NULL, NULL);
            size_t per = (size_t)d[1] * d[2];
            int ex[128], n_ex = 0;
            if (list) { for (char *s = (char *)list; *s && n_ex < 128; ) { ex[n_ex++] = (int)strtol(s, &s, 10); if (*s == ',') s++; } }
            else for (int e = 0; e < d[0]; e++) ex[n_ex++] = e;
            char cls[384]; class_of(p, i, cls, sizeof cls);
            row_t *r = NULL, *rs2 = NULL; ctl_t *cc = NULL;
            const ink2_tensor *stack = NULL;
            if (k[0] == 'C') {
                int fl = !strcmp(a, "nofactor") ? INK2P_CTRL_NO_FACTOR : !strcmp(a, "swapnibble") ? INK2P_CTRL_SWAP_NIBBLE : INK2P_CTRL_NO_INTERLEAVE;
                ink2p_set_control(p, fl);
                cc = &ctl[n_ctl++]; memset(cc, 0, sizeof *cc);
                snprintf(cc->kind, sizeof cc->kind, "%s", a);
                char c3[384]; memcpy(c3, cls, sizeof c3); snprintf(cls, sizeof cls, "CONTROL %.40s: %.300s", a, c3);
                r = row_for(cls);
            } else {
                r = row_for(cls);
                if (k[0] == 'S') {
                    ink2p_set_control(p, 0);   // empties the tensor cache before a 2.7 GB stack
                    char c2[512]; snprintf(c2, sizeof c2, "%s via the whole-stack get", cls); rs2 = row_for(c2);
                    stack = ink2p_get(p, nm);
                    if (!stack) { printf("FAIL whole-stack get of %s\n", nm); fails++; }
                }
            }
            r->tensors++; if (rs2) rs2->tensors++;
            for (int q = 0; q < n_ex; q++) {
                const ink2_tensor *t = ink2p_get_expert(p, nm, ex[q]);
                if (!t || t->ne[0] != d[2] || t->ne[1] != d[1] || read_hdr(nm, ex[q], per * 4)) { bad_stream = 1; break; }
                if (stack) {
                    // both paths against one read of the stream: ours (get_expert) and the stack's row e
                    uint64_t de = 0, db = 0, de2 = 0, db2 = 0;
                    const unsigned char *o1 = (const unsigned char *)t->data, *o2 = (const unsigned char *)(stack->data + (size_t)ex[q] * per);
                    size_t left = per * 4, off = 0;
                    while (left) {
                        size_t m = left < CHUNK ? left : CHUNK;
                        if (fread(chunk, 1, m, ref) != m) { bad_stream = 1; break; }
                        for (size_t z = 0; z < m; z += 4) {
                            uint32_t x, y, w2; memcpy(&x, o1 + off + z, 4); memcpy(&w2, o2 + off + z, 4); memcpy(&y, chunk + z, 4);
                            if (x != y) { de++; db += (uint64_t)__builtin_popcount(x ^ y); }
                            if (w2 != y) { de2++; db2 += (uint64_t)__builtin_popcount(w2 ^ y); }
                        }
                        off += m; left -= m;
                    }
                    r->diff_elems += de; r->diff_bits += db; rs2->diff_elems += de2; rs2->diff_bits += db2;
                    rs2->experts++; rs2->elems += per;
                } else if (cmp_stream(t->data, per, &r->diff_elems, &r->diff_bits)) { bad_stream = 1; break; }
                r->experts++; r->elems += per;
            }
            if (stack) ink2p_set_control(p, 0);   // drop the stack
            if (cc) { cc->de = r->diff_elems; cc->db = r->diff_bits; cc->el = r->elems; ink2p_set_control(p, 0); }
        }
        done_items++;
    }
    fclose(mr);
    int prc = pclose(ref);
    double t_id = now_s() - t0;
    if (bad_stream || prc) { printf("FAIL the reference stream broke after %d items (reference exit %d)\n", done_items, prc); fails++; }

    // the table
    printf("\nidentity, provider float32 vs the reference loader's float32 (bit for bit), %d manifest items in %.0f s:\n", done_items, t_id);
    printf("  %-78s %8s %8s %14s %10s %12s\n", "class [source format, rank]", "tensors", "experts", "elements", "diff elts", "diff bits");
    uint64_t tot_el = 0, tot_bad = 0; long tot_t = 0;
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < n_rows; i++) {
            row_t *r = &rows[i]; int is_ctl = !strncmp(r->cls, "CONTROL", 7);
            if (is_ctl != pass) continue;
            printf("  %-78s %8ld %8ld %14llu %10llu %12llu%s\n", r->cls, r->tensors, r->experts, (unsigned long long)r->elems,
                   (unsigned long long)r->diff_elems, (unsigned long long)r->diff_bits,
                   is_ctl ? (r->diff_bits ? "  (differs, as it must)" : "  CONTROL DID NOT DIFFER") : "");
            if (!is_ctl) { tot_el += r->elems; tot_bad += r->diff_bits; if (!strstr(r->cls, "via the whole-stack")) tot_t += r->tensors; }
            else if (!r->diff_bits) fails++;
        }
    int n_cls = 0; for (int i = 0; i < n_rows; i++) if (strncmp(rows[i].cls, "CONTROL", 7) && !strstr(rows[i].cls, "via the whole")) n_cls++;
    printf("%s identity: %d classes, %ld tensors, %llu elements compared, %llu differing bits\n",
           !tot_bad && !bad_stream ? "GREEN" : "FAIL", n_cls, tot_t, (unsigned long long)tot_el, (unsigned long long)tot_bad);
    if (tot_bad) fails++;

    // 4. the decode time of one expert (gate_up_proj + down_proj), cold (first touch of its source pages since this
    // process released them) and warm, NVFP4 and MXFP8; then one token's 8 experts at every application
    {
        double per_e[2] = { 0, 0 };
        for (int f = 0; f < 2; f++) {
            int b = f ? full_mx : full_nv;
            char n13[160], n2[160];
            snprintf(n13, sizeof n13, "model.layers.%d.mlp.experts.gate_up_proj", b);
            snprintf(n2, sizeof n2, "model.layers.%d.mlp.experts.down_proj", b);
            double best = 1e9, first = 0;
            for (int rep = 0; rep < 3; rep++) {
                ink2p_set_control(p, 0);
                double a = now_s();
                for (int e = 40; e < 48; e++) { ink2p_get_expert(p, n13, e); ink2p_get_expert(p, n2, e); }
                double dt = (now_s() - a) / 8;
                if (rep == 0) first = dt;
                if (dt < best) best = dt;
            }
            per_e[f] = best;
            printf("decode %s (layer %d): %.1f ms an expert (gate_up_proj + down_proj, 7.96 M floats), best of 3 runs of 8 experts; "
                   "first run %.1f ms\n", f ? "MXFP8" : "NVFP4", b, best * 1e3, first * 1e3);
        }
        int an = 0, am = 0; const char *ap = getenv("INK2P_APPS"); if (ap) sscanf(ap, "%d %d", &an, &am);
        int K = 8;
        double tok = K * (an * per_e[0] + am * per_e[1]);
        printf("one token, decode on every lookup: %d experts x (%d NVFP4 + %d MXFP8 applications) = %d expert decodes, %.2f s "
               "(single thread); the experts' matvecs themselves are %d x 2 x 7.96 M multiply-adds = %.2f G\n",
               K, an, am, K * (an + am), tok, K * (an + am), K * (an + am) * 2 * 7.962624e6 / 1e9);
    }

    ink2p_stats st; ink2p_get_stats(p, &st);
    struct rusage ru, rc; getrusage(RUSAGE_SELF, &ru); getrusage(RUSAGE_CHILDREN, &rc);
    printf("memory: this process peak RSS %.2f GB (cache peaks: tensors %.2f GB, experts %.2f GB); the reference side "
           "(largest child) peak RSS %.2f GB\n", ru.ru_maxrss / 1e6, st.tensor_peak / 1e9, st.expert_peak / 1e9, rc.ru_maxrss / 1e6);
    printf("provider: %llu gets (%llu hits), %llu get_expert (%llu hits); decoded %.2f G floats as tensors in %.1f s, "
           "%.2f G as experts in %.1f s\n", (unsigned long long)st.gets, (unsigned long long)st.get_hits,
           (unsigned long long)st.experts, (unsigned long long)st.expert_hits, st.decoded_floats_tensor / 1e9,
           st.decode_s_tensor, st.decoded_floats_expert / 1e9, st.decode_s_expert);
    ink2p_close(p);
    free(chunk);
    printf("%s\n", fails ? "RESULT: FAIL" : "RESULT: GREEN");
    return fails ? 1 : 0;
}
