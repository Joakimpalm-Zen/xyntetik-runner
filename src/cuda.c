// CUDA backend: the forward pass on NVIDIA GPUs through the driver API, loaded
// at run time (no toolkit needed; no driver means the CPU). Kernels: kernels.cu,
// embedded as PTX. Weights are copied to VRAM once; the KV cache lives there with
// the host copy authoritative (host rows upload lazily, device rows copy back).
#include "runner.h"
#include "upload_report.h"
#include "kernel_args.h"   // shared with src/kernels.cu — see the header
#include "gpu_tables.h"    // the per-layer tables and mixer weights, one list (R6.68.29)
#include "compat.h"     // plat_ram_available_bytes for the unified-pool budget
#include <stdatomic.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// clock_gettime/CLOCK_MONOTONIC in prof_now: not transitively available on
// every libc, so include it directly rather than relying on another header
#include <time.h>
// stat() identifies the weight file for the shared-upload registry, and the
// registry itself is mutex-guarded: slots load in parallel with serving
#include <pthread.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
typedef HMODULE dl_t;
static dl_t dl_open(void) { return LoadLibraryA("nvcuda.dll"); }
static void *dl_sym(dl_t h, const char *s) { return (void *)GetProcAddress(h, s); }
#else
#include <dlfcn.h>
typedef void *dl_t;
static dl_t dl_open(void) {
    dl_t h = dlopen("libcuda.so.1", RTLD_NOW);
    return h ? h : dlopen("libcuda.so", RTLD_NOW);
}
static void *dl_sym(dl_t h, const char *s) { return dlsym(h, s); }
#endif

#include "kernels_ptx.h"
#include "kernels_ptx80.h"   // the compute_80 TC2 kernels (kernels_sm80.cu, E1)

// ------------------------------------------------- driver API (what we use)

typedef int       CUresult;   // CUDA_SUCCESS == 0
typedef int       CUdevice;
typedef void     *CUcontext;
typedef void     *CUmodule;
typedef void     *CUfunction;
typedef uint64_t  CUdeviceptr;
typedef void     *CUstream;
typedef void     *CUgraph;
typedef void     *CUgraphExec;

static struct {
    dl_t lib;
    CUresult (*Init)(unsigned);
    CUresult (*DeviceGetCount)(int *);
    CUresult (*DeviceGet)(CUdevice *, int);
    CUresult (*DeviceGetName)(char *, int, CUdevice);
    // _v2 is the one that reports a MIG *instance* UUID; the v1 entry point
    // reports the parent card, which is identical for every slice on it.
    CUresult (*DeviceGetUuid_v2)(unsigned char *, CUdevice);
    CUresult (*DeviceGetUuid)(unsigned char *, CUdevice);
    CUresult (*DeviceGetPCIBusId)(char *, int, CUdevice);
    // Windows only (NULL elsewhere): the adapter LUID, which is how the OS
    // video-memory budget is looked up for the same device (compat.c)
    CUresult (*DeviceGetLuid)(char *, unsigned int *, CUdevice);
    CUresult (*PrimaryCtxRetain)(CUcontext *, CUdevice);
    CUresult (*PrimaryCtxRelease)(CUdevice);
    CUresult (*CtxSetCurrent)(CUcontext);
    CUresult (*MemGetInfo)(size_t *, size_t *);
    CUresult (*DeviceGetAttribute)(int *, int, CUdevice);
    CUresult (*MemAlloc)(CUdeviceptr *, size_t);
    CUresult (*MemFree)(CUdeviceptr);
    CUresult (*MemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult (*MemcpyDtoH)(void *, CUdeviceptr, size_t);
    CUresult (*MemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
    CUresult (*MemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    CUresult (*MemsetD8)(CUdeviceptr, unsigned char, size_t);
    CUresult (*MemsetD8Async)(CUdeviceptr, unsigned char, size_t, CUstream);  // optional
    CUresult (*ModuleLoadData)(CUmodule *, const void *);
    CUresult (*ModuleUnload)(CUmodule);
    CUresult (*ModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                             unsigned, unsigned, unsigned,
                             unsigned, void *, void **, void **);
    CUresult (*CtxSynchronize)(void);
    CUresult (*GetErrorString)(CUresult, const char **);
    // optional: a kernel asking for more than 48 KB of dynamic shared
    // memory (TC2) opts in through it; absent, that kernel stays unused
    CUresult (*FuncSetAttribute)(CUfunction, int, int);
    // optional: page-locked host staging (X3's KV gather); absent, pageable
    CUresult (*MemAllocHost)(void **, size_t);
    CUresult (*MemFreeHost)(void *);
    // optional: CUDA graphs for the single-token decode path (Experiment A).
    // Not required for the backend to function — see cu_graphs_ok().
    CUresult (*StreamCreate)(CUstream *, unsigned);
    CUresult (*StreamDestroy)(CUstream);
    CUresult (*StreamSynchronize)(CUstream);
    CUresult (*StreamBeginCapture)(CUstream, int);
    CUresult (*StreamEndCapture)(CUstream, CUgraph *);
    CUresult (*GraphInstantiateWithFlags)(CUgraphExec *, CUgraph, unsigned long long);
    CUresult (*GraphLaunch)(CUgraphExec, CUstream);
    CUresult (*GraphExecDestroy)(CUgraphExec);
    CUresult (*GraphDestroy)(CUgraph);
    // optional: CUDA events for RUNNER_CUDA_PROFILE phase timing
    CUresult (*EventCreate)(void **, unsigned);
    CUresult (*EventRecord)(void *, CUstream);
    CUresult (*EventSynchronize)(void *);
    CUresult (*EventElapsedTime)(float *, void *, void *);
    CUresult (*EventDestroy)(void *);
} cu;

// ---------------------------------------------------- RUNNER_CUDA_PROFILE
// A coarse phase breakdown, off by default: CUDA events between kernel groups
// on the plain (non-graph) path, summed in gpu_free(). `prof` is one
// unsynchronized global: a single-thread development instrument only.
typedef void *CUevent;
enum { PH_NORM = 0, PH_MATVEC, PH_ROPE, PH_ATTN, PH_ELEM, PH_LOGITS, PH_HOSTMOE, PH_N };
static const char *PH_NAME[PH_N] = {
    "norms(rms+qk)", "matvec", "rope", "attention", "elementwise", "logits-mv",
    "host-experts" };   // the stream's idle span around a host-resident expert layer
// two independent accumulator sets: mode 0 = prefill/batch tiles (tn>1, _b
// kernels), mode 1 = single-token decode (tn==1, k_mv_* kernels). Set per
// gpu_forward_batch call from n, so one run yields both clean breakdowns.
enum { M_BATCH = 0, M_SINGLE = 1, M_N = 2 };
static struct {
    int      on;            // RUNNER_CUDA_PROFILE set
    int      inited;        // events allocated
    int      capturing;     // inside graph capture: do not mark
    int      cur_mode;      // M_BATCH or M_SINGLE for the in-flight call
    CUevent  ev[4096];      // per-call event pool
    int      ph[4096];      // phase tag of the group ending at ev[k]
    int      nev;           // events used this call
    double   t_ph[M_N][PH_N];      // accumulated GPU ms per phase
    long long total_launches[M_N]; // exact kernel launches
    long long n_tiles[M_N], n_tokens[M_N];
    double   wall_ms[M_N];
    double   t_stage[M_N], t_copyback[M_N], t_logitscp[M_N], t_sync[M_N];
    double   qpc_hz;
} prof;

static double prof_now(void) {
#ifdef _WIN32
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / prof.qpc_hz;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
#endif
}

// newer drivers export the 64-bit entry points under _v2 names; the plain
// names are the legacy 32-bit-era ABI and must not be used
static void *sym2(const char *name) {
    char v2[64];
    snprintf(v2, sizeof(v2), "%s_v2", name);
    void *p = dl_sym(cu.lib, v2);
    return p ? p : dl_sym(cu.lib, name);
}

// Loaded once per process (cu_load). This body set cu.lib before filling the
// table cu.lib guards, so a second engine initialising at that moment saw
// "loaded" and could call a NULL cu.Init; and a later call answered true even
// when the first had found a required symbol missing.
static bool cu_load_now(void) {
    cu.lib = dl_open();
    if (!cu.lib) return false;
    cu.Init              = dl_sym(cu.lib, "cuInit");
    cu.DeviceGetCount    = dl_sym(cu.lib, "cuDeviceGetCount");
    cu.DeviceGet         = dl_sym(cu.lib, "cuDeviceGet");
    cu.DeviceGetName     = dl_sym(cu.lib, "cuDeviceGetName");
    // all three optional: without them gpu_device_id falls back a step at a
    // time, and an unidentifiable GPU still gets a (coarser) registry
    cu.DeviceGetUuid_v2  = dl_sym(cu.lib, "cuDeviceGetUuid_v2");
    cu.DeviceGetUuid     = dl_sym(cu.lib, "cuDeviceGetUuid");
    cu.DeviceGetPCIBusId = dl_sym(cu.lib, "cuDeviceGetPCIBusId");
    cu.DeviceGetLuid     = dl_sym(cu.lib, "cuDeviceGetLuid");   // optional
    cu.PrimaryCtxRetain  = dl_sym(cu.lib, "cuDevicePrimaryCtxRetain");
    cu.PrimaryCtxRelease = sym2("cuDevicePrimaryCtxRelease");
    cu.CtxSetCurrent     = dl_sym(cu.lib, "cuCtxSetCurrent");
    cu.MemGetInfo        = sym2("cuMemGetInfo");
    // optional: every driver since CUDA 2.0 has it, but treating it as
    // required would brick init over a probe whose absence just means
    // "assume discrete"
    cu.DeviceGetAttribute = dl_sym(cu.lib, "cuDeviceGetAttribute");
    cu.MemAlloc          = sym2("cuMemAlloc");
    cu.MemFree           = sym2("cuMemFree");
    cu.MemcpyHtoD        = sym2("cuMemcpyHtoD");
    cu.MemcpyDtoH        = sym2("cuMemcpyDtoH");
    cu.MemcpyDtoD        = sym2("cuMemcpyDtoD");
    cu.MemcpyDtoDAsync   = sym2("cuMemcpyDtoDAsync");
    cu.MemsetD8          = sym2("cuMemsetD8");
    cu.MemsetD8Async     = sym2("cuMemsetD8Async");
    cu.ModuleLoadData    = dl_sym(cu.lib, "cuModuleLoadData");
    cu.ModuleUnload      = dl_sym(cu.lib, "cuModuleUnload");
    cu.ModuleGetFunction = dl_sym(cu.lib, "cuModuleGetFunction");
    cu.LaunchKernel      = dl_sym(cu.lib, "cuLaunchKernel");
    cu.CtxSynchronize    = dl_sym(cu.lib, "cuCtxSynchronize");
    cu.GetErrorString    = dl_sym(cu.lib, "cuGetErrorString");
    cu.FuncSetAttribute  = dl_sym(cu.lib, "cuFuncSetAttribute");
    cu.MemAllocHost      = sym2("cuMemAllocHost");
    cu.MemFreeHost       = dl_sym(cu.lib, "cuMemFreeHost");
    // optional graph entry points — absence just disables the graph path
    cu.StreamCreate       = dl_sym(cu.lib, "cuStreamCreate");
    cu.StreamDestroy      = sym2("cuStreamDestroy");
    cu.StreamSynchronize  = dl_sym(cu.lib, "cuStreamSynchronize");
    cu.StreamBeginCapture = sym2("cuStreamBeginCapture");
    cu.StreamEndCapture   = sym2("cuStreamEndCapture");
    cu.GraphInstantiateWithFlags = dl_sym(cu.lib, "cuGraphInstantiateWithFlags");
    cu.GraphLaunch        = dl_sym(cu.lib, "cuGraphLaunch");
    cu.GraphExecDestroy   = dl_sym(cu.lib, "cuGraphExecDestroy");
    cu.GraphDestroy       = dl_sym(cu.lib, "cuGraphDestroy");
    cu.EventCreate        = dl_sym(cu.lib, "cuEventCreate");
    cu.EventRecord        = dl_sym(cu.lib, "cuEventRecord");
    cu.EventSynchronize   = dl_sym(cu.lib, "cuEventSynchronize");
    cu.EventElapsedTime   = dl_sym(cu.lib, "cuEventElapsedTime");
    cu.EventDestroy       = sym2("cuEventDestroy");
    return cu.Init && cu.DeviceGetCount && cu.DeviceGet && cu.DeviceGetName &&
           cu.PrimaryCtxRetain && cu.PrimaryCtxRelease && cu.CtxSetCurrent &&
           cu.MemGetInfo && cu.MemAlloc && cu.MemFree && cu.MemcpyHtoD &&
           cu.MemcpyDtoH && cu.MemcpyDtoD && cu.MemsetD8 && cu.ModuleLoadData && cu.ModuleUnload &&
           cu.ModuleGetFunction && cu.LaunchKernel && cu.CtxSynchronize;
}

// ----------------------------------------------------- capture exclusion
// Streams are blocking, so a synchronous call (copy, memset, alloc, free,
// module load, sync) waits on every engine's stream, and CUDA fails it and the
// capture while any stream is being captured. g_capture_mu keeps the two
// apart: held for a capture's window and by each such call (cu_guard_syncing's
// wrappers).
static pthread_mutex_t g_capture_mu = PTHREAD_MUTEX_INITIALIZER;
static _Thread_local bool t_capturing;
static struct {
    CUresult (*MemAlloc)(CUdeviceptr *, size_t);
    CUresult (*MemFree)(CUdeviceptr);
    CUresult (*MemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult (*MemcpyDtoH)(void *, CUdeviceptr, size_t);
    CUresult (*MemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
    CUresult (*MemsetD8)(CUdeviceptr, unsigned char, size_t);
    CUresult (*ModuleLoadData)(CUmodule *, const void *);
    CUresult (*ModuleUnload)(CUmodule);
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                             unsigned, unsigned, unsigned,
                             unsigned, void *, void **, void **);
    CUresult (*CtxSynchronize)(void);
} cu_raw;

// The first synchronous call this thread made while capturing. Such a call
// cannot be recorded: it fails the capture, or would run once and never on a
// replay. The guarded wrappers note it and a failed capture names it.
static _Thread_local const char *t_capture_sync;
#define CAPTURE_NOTE(name) do { if (t_capturing && !t_capture_sync) t_capture_sync = (name); } while (0)

#define SYNC_GUARDED(call) do {                                   \
        bool held_ = !t_capturing;                                \
        if (held_) pthread_mutex_lock(&g_capture_mu);             \
        r_ = (call);                                              \
        if (held_) pthread_mutex_unlock(&g_capture_mu);           \
    } while (0)

// every allocation the engine holds (gpu_live_buffers): the free path is a
// hand-kept list beside the allocation sites, and this count is what shows
// the two have drifted
static _Atomic long g_live_bufs;
static CUresult guarded_MemAlloc(CUdeviceptr *p, size_t n) {
    CAPTURE_NOTE("cuMemAlloc");
    CUresult r_; SYNC_GUARDED(cu_raw.MemAlloc(p, n));
    if (r_ == 0) g_live_bufs++;
    return r_;
}
static CUresult guarded_MemFree(CUdeviceptr p) {
    CAPTURE_NOTE("cuMemFree");
    CUresult r_; SYNC_GUARDED(cu_raw.MemFree(p));
    if (r_ == 0) g_live_bufs--;
    return r_;
}
static CUresult guarded_MemcpyHtoD(CUdeviceptr d, const void *h, size_t n) {
    CAPTURE_NOTE("cuMemcpyHtoD");
    CUresult r_; SYNC_GUARDED(cu_raw.MemcpyHtoD(d, h, n)); return r_;
}
static CUresult guarded_MemcpyDtoH(void *h, CUdeviceptr d, size_t n) {
    CAPTURE_NOTE("cuMemcpyDtoH");
    CUresult r_; SYNC_GUARDED(cu_raw.MemcpyDtoH(h, d, n)); return r_;
}
static CUresult guarded_MemcpyDtoD(CUdeviceptr a, CUdeviceptr b, size_t n) {
    CAPTURE_NOTE("cuMemcpyDtoD");
    CUresult r_; SYNC_GUARDED(cu_raw.MemcpyDtoD(a, b, n)); return r_;
}
static CUresult guarded_MemsetD8(CUdeviceptr d, unsigned char v, size_t n) {
    CAPTURE_NOTE("cuMemsetD8");
    CUresult r_; SYNC_GUARDED(cu_raw.MemsetD8(d, v, n)); return r_;
}
static CUresult guarded_ModuleLoadData(CUmodule *mod, const void *img) {
    CAPTURE_NOTE("cuModuleLoadData");
    CUresult r_; SYNC_GUARDED(cu_raw.ModuleLoadData(mod, img)); return r_;
}
static CUresult guarded_ModuleUnload(CUmodule mod) {
    CAPTURE_NOTE("cuModuleUnload");
    CUresult r_; SYNC_GUARDED(cu_raw.ModuleUnload(mod)); return r_;
}
static CUresult guarded_CtxSynchronize(void) {
    CAPTURE_NOTE("cuCtxSynchronize");
    CUresult r_; SYNC_GUARDED(cu_raw.CtxSynchronize()); return r_;
}
// a launch on the engine's own stream is ordered by that stream alone; one
// on the legacy stream (NULL: the training matvec) synchronizes like a copy
static CUresult guarded_LaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                                     unsigned bx, unsigned by, unsigned bz,
                                     unsigned shmem, void *stream, void **params,
                                     void **extra) {
    if (stream) return cu_raw.LaunchKernel(f, gx, gy, gz, bx, by, bz, shmem, stream, params, extra);
    CAPTURE_NOTE("cuLaunchKernel on the legacy stream");
    CUresult r_;
    SYNC_GUARDED(cu_raw.LaunchKernel(f, gx, gy, gz, bx, by, bz, shmem, stream, params, extra));
    return r_;
}

// One driver entry of cu_guard_syncing: keep the raw pointer and route calls
// through the guarded wrapper of the same name. It pastes a symbol name and
// captures nothing of the function that uses it, so it is defined here, at
// file scope, not inside that function.
#define GUARD(fn) do { cu_raw.fn = cu.fn; if (cu.fn) cu.fn = guarded_##fn; } while (0)
static void cu_guard_syncing(void) {
    GUARD(MemAlloc); GUARD(MemFree); GUARD(MemcpyHtoD); GUARD(MemcpyDtoH);
    GUARD(MemcpyDtoD); GUARD(MemsetD8); GUARD(ModuleLoadData); GUARD(ModuleUnload);
    GUARD(LaunchKernel); GUARD(CtxSynchronize);
}
#undef GUARD

// A capture holds g_capture_mu from its first call to its last. capture_end
// is safe to call twice (the failure paths end a capture "to be sure").
static CUresult capture_begin(CUstream s) {
    pthread_mutex_lock(&g_capture_mu);
    t_capturing = true;
    t_capture_sync = NULL;
    CUresult r = cu.StreamBeginCapture(s, 1);   // THREAD_LOCAL; see above for why that is not enough
    if (r != 0) {
        t_capturing = false;
        pthread_mutex_unlock(&g_capture_mu);
    }
    return r;
}
static CUresult capture_end(CUstream s, CUgraph *graph) {
    CUresult r = cu.StreamEndCapture(s, graph);
    if (t_capturing) {
        t_capturing = false;
        pthread_mutex_unlock(&g_capture_mu);
    }
    return r;
}

static pthread_once_t cu_once = PTHREAD_ONCE_INIT;
static bool cu_ok;
static void cu_load_once(void) {
    cu_ok = cu_load_now();
    if (cu_ok) cu_guard_syncing();
}
static bool cu_load(void) {
    pthread_once(&cu_once, cu_load_once);   // orders the table before every reader
    return cu_ok;
}

static bool cu_graphs_ok(void) {
    return cu.StreamCreate && cu.StreamDestroy && cu.StreamSynchronize &&
           cu.StreamBeginCapture && cu.StreamEndCapture &&
           cu.GraphInstantiateWithFlags && cu.GraphLaunch &&
           cu.GraphExecDestroy && cu.GraphDestroy;
}

static const char *cu_err(CUresult r) {
    const char *s = NULL;
    if (cu.GetErrorString) cu.GetErrorString(r, &s);
    return s ? s : "unknown CUDA error";
}

// RUNNER_REQUIRE_GRAPH_CAPTURE=1 (the CUDA gates): a decode graph that fails
// to capture or launch ends the process with the reason, where the default is
// a fallback to plain launches. A decode that is eager by design says why and
// runs.
static bool graph_capture_required(void) {
    static _Atomic int on = -1;
    if (on < 0) {
        const char *e = getenv("RUNNER_REQUIRE_GRAPH_CAPTURE");
        on = e && *e && strcmp(e, "0") != 0;
    }
    return on == 1;
}
// what: "decode" or "batch"; stage: where it failed; r: the CUDA result there
static void graph_capture_failed(const char *what, const char *stage, CUresult r) {
    const char *op = t_capture_sync;
    fprintf(stderr, "gpu: %s graph capture failed at %s (%s%s%s) \u2014 plain launches\n",
            what, stage, cu_err(r), op ? "; a synchronous " : "", op ? op : "");
    if (graph_capture_required()) {
        fprintf(stderr, "error: RUNNER_REQUIRE_GRAPH_CAPTURE=1: a %s graph that does not "
                "capture is a failure here, not a fallback\n", what);
        exit(3);
    }
}
static void graph_launch_failed(const char *what) {
    fprintf(stderr, "gpu: %s graph launch failed \u2014 plain launches\n", what);
    if (graph_capture_required()) {
        fprintf(stderr, "error: RUNNER_REQUIRE_GRAPH_CAPTURE=1: a %s graph that does not "
                "launch is a failure here, not a fallback\n", what);
        exit(3);
    }
}
// Under the switch, a decode that is eager by design says so once.
static void graph_eager_note(const char *what, const char *why, bool *said) {
    if (!graph_capture_required() || *said || !why) return;
    *said = true;
    fprintf(stderr, "gpu: %s graph off by design: %s\n", what, why);
}

// ------------------------------------------------------------------ backend

// ---------------------------------------------------- shared model weights
// What a forward only reads, identical for two models loaded from one file
// with the same parameters: uploaded once and refcounted, not once per slot.
// Keyed on the file's identity, the geometry and the derived rope tables; a
// mismatch is a miss, and that instance gets its own upload.
// Microbatch width classes: the batched matvecs unroll their column loop, so a
// narrower batch runs the smallest class that covers it. BW_16 serves only a
// speculative verify round's 9 to 16 rows; decode stays within BW_8.
enum { BW_4 = 0, BW_8 = 1, BW_16 = 2, BW_N = 3 };
static inline int batch_width_class(int n) { return n <= 4 ? BW_4 : n <= 8 ? BW_8 : BW_16; }
enum { GPU_SMALL_BATCH = 8 };   // the widest f_gemvb class (BW_8): small-batch routing in enc_mv
enum { KVG_ROWS = 16 };         // rows a forward brings back through the KV gather (R3.4.1)
// The widest microbatch those kernels compute. A wider call does not fail: the
// columns past the class keep stale logits, and the f_mvb fallback indexes a
// fixed register array out of bounds. gpu_batch_decode enforces it and the
// assertion keeps model.c's ceiling from drifting past it.
#define BATCH_MAX_COLS 8
_Static_assert(MODEL_BATCH_MAX <= BATCH_MAX_COLS,
               "MODEL_BATCH_MAX exceeds the widest CUDA microbatch kernel: "
               "add wider f_gemvb width classes before raising it");

typedef struct {
    uint64_t    file_off;
    size_t      nbytes;
    CUdeviceptr device;
} gpu_weight_binding;

typedef struct gpu_weights {
    struct gpu_weights *next;
    int         refs;                   // live gpu_t values pointing here
    // --- identity ---
    char       *path;
    uint64_t    fsize, fino;
    int64_t     fmtime;
    uint32_t    n_maps;        // mapped parts (split GGUF, attached companions)
    uint64_t    mapped_size;   // every part's bytes: with part 1's identity, the key
    int         n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff, n_vocab;
    int         n_ctx, rope_dim, rope_dim_local, kv_q8, v_rmsnorm;
    float       rope_base, rope_mscale;
    float      *rope_inv_freq, *rope_inv_freq_local;  // owned copies, compared
    // --- device state (immutable once built) ---
    CUcontext   ctx;
    CUdevice    dev;
    CUmodule    mod;
    CUfunction  f_rmsnorm, f_qknorm, f_rope, f_store, f_attn, f_silu, f_gelu,
                f_swiglu, f_xielu, f_add, f_scale;
    CUfunction  f_gelu_erf, f_head_gate;   // spark2_5: exact GELU, per-head gate
    CUfunction  f_mla_q, f_mla_kv, f_mla_pack_v;   // deepseek2's naive MLA layout
    CUfunction  f_mla_q_absorb, f_mla_kv_absorb, f_mla_v_absorb;   // and the absorbed form
    CUfunction  f_q35_split, f_q35_gate, f_q35_conv, f_q35_delta;
    CUfunction  f_mamba_conv, f_mamba_ssd, f_mamba_gate_norm, f_relu2;
    // whole-tile twins of the per-token recurrent kernels (C2.3, RUNNER_CUDA_RECUR_SEQ)
    CUfunction  f_q35_conv_seq, f_q35_delta_seq, f_mamba_conv_seq, f_mamba_ssd_seq;
    CUfunction  f_q35_delta1, f_q35_delta_norm;   // the column-split one-token step
    CUfunction  f_colabsmax;            // per-column |x| max for the scaled TC GEMMs
    CUfunction  f_attn_dec, f_attn_merge;   // flash-decoding attention (decode)
    CUfunction  f_attn_tile32, f_attn_tile16, f_attn_tile8; // tiled prefill attention (tile_attn)
    CUfunction  f_attn_dec_gqa, f_attn_merge32; // grouped decode attention (tile_attn)
    CUfunction  f_attn_tc;                  // tensor-core prefill attention (tile_attn)
    CUfunction  f_attn_tcw;                 // its wide-head twin, head_dim 256 to 512
    CUfunction  f_attn_tcw_mla;             // and the absorbed MLA row's (R3.32)
    CUfunction  f_kv_gather;                // one-copy decode KV write-back (X3)
    // Kernel tables indexed by ggml type id, sized past the highest type the
    // loader admits: a CPU-only type must land on a NULL slot and decline,
    // never index past the table.
    #define KT_N (T_Q2_0 + 1)   // ggml type ids are sparse; the highest decoded one bounds the tables
    CUfunction  f_mv[KT_N], f_mvb[KT_N];// indexed by ggml type; _b = tile twin (F32/F16/BF16)
    CUfunction  f_mvt[KT_N];            // transposed matvec (training backward)
    CUfunction  f_lora_a, f_lora_b;     // adapter at inference (D2 on CUDA)
    CUfunction  f_mvcanon[KT_N];        // canonical-order forward (R8.7.1)
    CUfunction  f_gemm_tc[KT_N];        // tensor-core prefill GEMM (opt-in, Q4_K)
    CUfunction  f_gemm2[KT_N];          // pipelined tensor-core GEMM (TC2), by type
    CUmodule    mod80;                  // the compute_80 module, on a CC >= 8.0 device
    CUfunction  f_gemm2_80[KT_N];       // its TC2 kernels (m16n8k16), by type
    // its int8 GEMMs (R3.14, RUNNER_CUDA_I8=1) and their activation quantizer
    CUfunction  f_gemm_i8[KT_N], f_quant_i8;
    CUfunction  f_gemm2_80w[KT_N];      // the same at a 128-column tile
    CUfunction  f_x_half;               // its scaled fp16 activations (k_x_half_scaled)
    CUfunction  f_gemv[KT_N];           // decode coalesced GEMV variants (Q8_0/Q4_K)
    // cross-sequence decode microbatch (Phase 6): per-column position and KV
    CUfunction  f_rope_seq, f_store_seq, f_attn_dec_seq;
    CUfunction  f_attn_dec_gqa_seq, f_attn_merge32_seq; // f_attn_dec_gqa's twins
    CUfunction  f_attn_dec_mla, f_attn_dec_mla_seq;     // the absorbed MLA decode (R3.31)
    // multi-column twins of f_gemv, per microbatch width (see BW_* below):
    // f_gemvb[w][type], w indexing the width class, not the width itself
    CUfunction  f_gemvb[BW_N][KT_N];
    CUfunction  f_moe_q4_0_mvb[BW_N];   // k_mv_q4_0's twins (the by-expert MoE prefill)
    // exact-width twins, [n] for n = 2..8 columns (the codebook i-quants);
    // enc_mv_batch prefers them over the width class
    CUfunction  f_gemvx[9][KT_N];
    // int8-activation codebook GEMVs, [n] for n = 1..8 columns, and the x
    // quantizer (RUNNER_CUDA_Q8X=1, experimental)
    CUfunction  f_q8x[9][KT_N], f_quant_q8x;
    // sparse-MoE device routing + indirect expert matvecs (fused-3D layout)
    CUfunction  f_moe_route, f_moe_actmul, f_moe_sum;
    CUfunction  f_moe_route_replay;     // RUNNER_MOE_REPLAY's router (R3.21.2)
    CUfunction  f_moe_sum_tile, f_moe_place;   // the by-expert prefill (moe_twin)
    CUfunction  f_moe_mv[KT_N];         // indexed by expert tensor type
    // expert-grouped prefill (P2): gather/scatter around the k_gemm family
    CUfunction  f_moe_gather, f_moe_scatter, f_moe_actmul_plain, f_moe_copy;
    CUdeviceptr weights;
    size_t      weights_len;
    size_t      img_off;                // file offset the contiguous image starts at
    bool        cpu_moe;                // sparse tensor-role placement mode
    int         cpu_moe_layers;         // requested host-expert count (share key)
    gpu_weight_binding *bindings;       // packed non-expert tensors in this mode
    int         n_bindings, cap_bindings;
    bool        bound;                  // weights uploaded per tensor (bindings), not as one file prefix
    upload_report rep;                  // progress of the weight upload (upload_report.h)
    size_t      bound_bytes;            // bytes the bindings hold (the banner's number when bound)
    int         gpu_layers;             // split decided by the first loader
    int         n_tab;                  // per-layer table length: n_layer, +1 with a bound MTP block
    int         win;                    // prompt window every instance is built for (model_gpu_window)
    bool        no_id;                  // identity unavailable: listed so the
                                        // split guard can see it, never matched
    CUdeviceptr inv_freq, inv_freq_local, out_norm, dummy;
    CUdeviceptr ones;                   // weightless V RMS norm (gemma4)
    CUdeviceptr *attn_norm, *ffn_norm;  // per layer
    CUdeviceptr *hc_an, *hc_fn;         // qwen4exp per-stream mixer gammas [hc*n_embd]
    CUdeviceptr hc_head_norm;           // qwen4exp head mixer gamma
    // NextN/MTP head on the device (hyper-connection families): the block's
    // own tables sit at index n_layer in the per-layer arrays above, these
    // are the head-only gammas. mtp is part of the share key.
    bool        mtp;
    CUdeviceptr mtp_enorm, mtp_hnorm, mtp_head_norm;
    CUfunction  f_rmsnorm_grouped, f_hc_gate_mean, f_hc_combine, f_silu_scale, f_mtp_cat;
    CUfunction  f_mv_bf16_splitk, f_splitk_reduce, f_axpy_sigmoid;
    CUdeviceptr *pan, *pfn;             // gemma3 sandwich norms, may be 0
    CUdeviceptr *bq, *bk, *bv, *bo;     // per layer, may be 0
    CUdeviceptr *qn, *kn;               // qwen3 per-head q/k norms
    CUdeviceptr *mla_kvn;               // deepseek2: the latent's RMS norm (kv_lora wide)
    CUdeviceptr *mla_kvb;               // and attn_kv_b dequantized (the absorbed form)
    // gemma-4 MoE dual-branch: dense-post/moe-pre/moe-post norms + the router
    // input scale (with 1/sqrt(n_embd) folded in), per MoE layer, may be 0
    CUdeviceptr *g_pn1, *g_prn2, *g_pn2, *g_gis;
    CUdeviceptr *ple_pn;                // gemma4 E-series per-layer post_norm
    // per-expert down-projection scale table per MoE layer (gemma-4), ones
    // when the GGUF has none; and a shared all-ones table for archs without
    // any such scale — k_moe_actmul multiplies unconditionally
    CUdeviceptr *g_dsc;
    CUdeviceptr moe_ones;
    // gpt-oss per layer, may be 0: attention-sink logits [n_head]; router
    // bias [n_expert]; per-expert gate/up ([n_expert][n_ff_exp]) and down
    // ([n_expert][n_embd]) bias tables
    CUdeviceptr *sinks;
    CUdeviceptr *gib, *geb, *ueb, *deb;
    // RUNNER_MOE_HOT (R3.30): per host layer, its hot experts' gate, up and
    // down slices in one allocation ([S] gate | [S] up | [S] down), 0 where
    // the layer has none
    CUdeviceptr *hot_bank;
    CUdeviceptr *ssm_dt, *ssm_a, *ssm_norm;
    // Mamba-2 (granitehybrid/nemotron_h): D skip, conv1d bias, gated RMS
    // norm weight (length ssm.inner, unlike qwen35's ssm_norm at ssm.state)
    CUdeviceptr *ssm_D, *ssm_conv_b, *ssm_gnorm;
} gpu_weights;

// ------------------------------------------------ per-sequence backend state
//
// One per model_t: the KV cache this stream is decoding into, its activation
// scratch, and its own stream/graph. Nothing here is shared, so one sequence
// failing, resetting, or being unloaded cannot touch another's.
enum { SPLITK_PARTIALS = 65536 };   // floats: n_out * splits of one split-K GEMV call

typedef struct {
    gpu_weights *sw;                    // shared weights, refcounted
    CUdeviceptr kc, vc;
    CUdeviceptr x, xb, xb2, q, kt, vt, hb, hb2, att, attn_part, logits;
    // a speculative verify round (model_forward_verify) runs its tile in the
    // one-token decode arithmetic per row (exact_rows); its decode attention
    // reads each row's position and cache from these, as the microbatch's
    // per-column arrays (MVB entries)
    bool        exact_rows;
    CUdeviceptr vcol_pos, vcol_kcp, vcol_vcp;
    CUdeviceptr q8x_q, q8x_d;   // RUNNER_CUDA_Q8X: int8 x [MVB][q8x_cap] and its per-32 scales
    int q8x_cap;
    CUdeviceptr x_hc, hc_xn, hc_lo, hc_gate, hc_inject;  // qwen4exp wide residual + mixer scratch
    CUdeviceptr splitk;                                   // split-K partials, SPLITK_PARTIALS floats
    CUdeviceptr shexp_in, shexp_g, shexp_u, shexp_o, shexp_gate;  // shared expert on the device
    CUdeviceptr mla_buf;   // deepseek2: kv_a, the normed latent, kv_b's output, the packed V
    CUdeviceptr xsc;                    // [win] per-column |x| max for the scaled TC GEMMs
    CUdeviceptr xh;                     // TC2: [win][xh_cap] scaled fp16 activations
    int         xh_cap;                 // the widest GEMM input xh holds
    // What TC2 staged last into xsc / xh (address, shape, and the enc_mv that
    // did it). A call whose next GEMM reads the same unwritten columns sets
    // xh_reuse, and the GEMM skips restaging only if that staging came from the
    // enc_mv right before it. Any other writer of xsc clears xh_src.
    CUdeviceptr xh_src;
    int         xh_n, xh_xs, xh_batch;
    bool        xh_reuse;
    // the int8 route's activations (RUNNER_CUDA_I8): the window's columns as
    // int8 per 32-element block, the block scales and dx * sum(xq)
    CUdeviceptr i8_xq, i8_xd, i8_xs;
    int         i8_cap;
    unsigned long mv_seq, xh_seq;
    // sparse-MoE: router logits [MVB][n_expert], per-slot expert down-out
    // [rows][n_embd] (rows = max(used, MVB); the eager path uses column 0)
    CUdeviceptr moe_logits, moe_eout;
    // device routing results [MVB][used] and per-slot expert hidden scratch
    // [rows][gw] (gw = 2*n_ff_exp for gemma's fused gate_up, n_ff_exp else)
    CUdeviceptr moe_sel, moe_selw, moe_hb, moe_hb2;
    // RUNNER_MOE_HOT: the routing remapped to hot-bank slots [MVB][used], and
    // its host staging: the slots, the normed rows and the host experts' terms
    CUdeviceptr moe_hot_sel;
    int        *h_hot_sel;
    float      *h_hot_x, *h_hot_terms;
    int         hot_xdim;
    // gemma4 E-series: ple holds [token][layer][n_embd_ple] for the batch,
    // uploaded once per forward; ple_g is the per-layer gate scratch.
    CUdeviceptr ple, ple_g;
    // expert-grouped prefill: gathered inputs / down outputs [MVB][n_embd],
    // concatenated per-expert token lists + weights [MVB*used], and the host
    // staging the grouping is built from
    CUdeviceptr moe_gath, moe_dout, moe_gidx, moe_gw;
    // by-expert prefill bit-identical to the fused path (moe_twin): each
    // pair's (token, slot) row and expert, and the tile's slot outputs
    CUdeviceptr moe_gpos, moe_gsel, moe_slots;
    int        *h_moe_gpos, *h_moe_gsel;
    int        *h_moe_sel, *h_moe_gidx;
    float      *h_moe_selw, *h_moe_gw;
    bool        moe_fused_ok;           // every MoE layer runs the indirect path
    int         moe_renorm;             // k_moe_route's renorm: expert_weights_norm
    // RUNNER_MOE_REPLAY / RUNNER_MOE_TRACE on the device routers (R3.21.2):
    // the trace as [replay_npos][n_layer][used] ids, the three counters
    // model.c reports, whether each routing is written to the trace, and the
    // position of the current tile's first column (set_pos, set_rows)
    CUdeviceptr replay_tab, replay_cnt;
    int         replay_npos, replay_nlayer;
    bool        route_trace;
    int         route_pos;
    // RUNNER_MOE_TRACE: the MoE layer in flight's routings, held until its
    // path has the expert outputs whose norms the record carries (moe_trace_*)
    int         tr_n, tr_cap, tr_used, tr_layer;
    size_t      tr_cap_slots;
    int        *tr_pos, *tr_sel;
    float      *tr_w, *tr_norm;
    bool        moe_grouped_ok;         // expert types have width-classed kernels
    bool        moe_twin_ok;            // ... and the by-expert path can be fused-identical
    CUdeviceptr g_scr;                  // gemma-4-MoE dual-branch scratch: 5*n_embd
    CUdeviceptr q35_mix, q35_cv, q35_z, q35_beta, q35_alpha, q35_gate;
    CUdeviceptr q35_hist, q35_state;    // persistent recurrent state per sequence
    CUdeviceptr q35_hist_prev, q35_state_prev; // pre-forward rollback snapshot
    // NextN/MTP head scratch (device): the queue of previous-position wide
    // residuals [n_batch][hcd], the pending row, the per-stream head input
    // [MVB*hc][2E], the chained hidden [hcd] and the normed embedding [MVB][E]
    CUdeviceptr mtp_h, mtp_pending, mtp_cat, mtp_hid, mtp_en, mtp_hn;
    CUdeviceptr mtp_rows;               // [n_batch][hcd]: every tile's final wide residual rows
    // per-row recurrent checkpoints of a verify tile (speculative decode):
    // after row r of the tile, every recurrent layer's fold and conv window
    // land in slot r, so a partially accepted round restores the fold after
    // the last accepted row instead of rolling back and re-folding it
    CUdeviceptr snap_state, snap_hist;
    int         snap_rows;              // slots (0: off)
    int         snap_on;                // the current forward files them (a keep-forward)
    int         snap_valid;             // rows filed by the last keep-forward
    // A Mamba-2 hybrid files rows 0 .. n-2 (its last row's state is the live
    // half) in slots allocated by its first verify round (mamba_snap_ready),
    // one per recurrent layer: snap_ord[l] is layer l's slot, -1 for the
    // others. snap_failed: that allocation failed, so its rounds re-fold.
    int        *snap_ord;
    int         snap_nrec;
    bool        snap_failed;
    unsigned long fwd_count;            // forwards run (recurrent rollback validity)
    unsigned long fwd_snap;             // fwd_count when the rollback point was taken
    // Mamba-2 recurrent buffers (granitehybrid/nemotron_h)
    CUdeviceptr mamba_proj, mamba_xBC, mamba_y;         // per-tile scratch
    CUdeviceptr mamba_conv, mamba_state;                // persistent conv ring + SSD state
    CUdeviceptr mamba_conv_prev;                        // pre-forward rollback snapshot (ring)
    // The SSD state is two halves of mamba_state: a forward reads the committed
    // half (rec_prev) and writes the other (rec_cur), so the pre-forward state a
    // rollback needs survives without a copy. rec_tile0: the forward's first
    // tile is running (later tiles update rec_cur in place).
    int         rec_cur, rec_prev;
    bool        rec_tile0;
    // The loaded adapter on the device: lora_a/lora_b are [n_layer * LW_SLOTS]
    // pointers (0 = no adapter there, or the host hook covers a CPU layer),
    // lora_t the [MVB][lora_r] scratch. Per sequence, not shared: two slots can
    // serve one file with different adapters.
    CUdeviceptr *lora_a, *lora_b;
    int          lora_n;                // table length (n_layer * LW_SLOTS)
    int         *lora_r_of;             // per-slot rank
    float       *lora_scale;            // per-slot (alpha/r) * user scale
    CUdeviceptr  lora_t;
    int          lora_r;                // widest rank bound, 0 = no adapter
    float       *h_x, *h_logits, *h_moe_logits; // host staging
    float       *h_ple, *h_ple_tmp;             // E-series pre-pass staging
    // The leading KV rows [0, kv_rows) the device holds as the host does. A
    // forward leaves [0, pos + n) so, a host write lowers it (gpu_kv_host_wrote),
    // and a forward uploads only [kv_rows, pos): a rewind uploads nothing.
    int          kv_rows;
    // CUDA graphs for the single-token decode path (Experiment A); a
    // double-buffered recurrent state (mamba_db) needs one per buffer parity,
    // the captured scan kernels baking in which half they read and write
    CUstream     stream;
    CUgraph      graph[2];
    CUgraphExec  gexec[2];
    float        graph_temp[2];         // the attention temperature each graph holds
    // the window's first position, then one per later ATT_COLS-column
    // attention sub-tile (set_pos): k_attn finds its sub-tile's start
    // through the pointer it is handed
    CUdeviceptr  pos_dev;
    bool         graph_bad;
    const char  *graph_why;             // why graph_bad was set by design (none: a failure set it)
    bool         graph_why_said;
    int          win;                   // prompt window: columns of every activation buffer
    // X3 (RUNNER_CUDA_KV_GATHER): a decoded token's K and V rows of every
    // offloaded layer gathered on the device (kvg_dev) and brought back in
    // one copy (kvg_host, page-locked when the driver gives it), then
    // scattered into the host cache by the table's host copy
    CUdeviceptr  kvg_tab, kvg_dev, kvg_pos;   // kvg_pos: a forward's row positions (KVG_ROWS)
    void        *kvg_host;
    bool         kvg_pinned;
    size_t       kvg_bytes;
    void        *h_kvg_tab;
} gpu_t;

// Max tokens per matvec tile; must match MVB in kernels.cu (every activation
// buffer is this many columns wide). LORA_R_DEV_MAX is the adapter rank
// k_lora_b's shared vector can stage (LORA_R_MAX in model.c): the runtime
// check refuses a rank past it instead of reading past the array.
#define LORA_R_DEV_MAX 512
#define MVB 64   // activation buffer columns (keep in sync with kernels.cu)
_Static_assert(MVB == MODEL_GPU_MVB, "model.h's MODEL_GPU_MVB is the plan's copy of MVB");
#define MVT 16   // scalar-kernel fixed tile (keep in sync with MVT there)
// Prefill attention scores are staged per (token, head) over the whole
// context, so their scratch is sized by a fixed sub-tile of query columns,
// not by the prompt window: a window of W tokens runs k_attn W/ATT_COLS
// times over the same scratch. Its kernel and arithmetic are unchanged.
#define ATT_COLS 64
// TC2 (kernels.cu G2_*): 128-row x 64-column blocks, 256 threads, K in
// stages of 64, two stages of padded fp16 tiles in dynamic shared memory
#define G2_BM 128
#define G2_BN 64
#define G2_BK 64
#define G2_SMEM ((2 * G2_BM * (G2_BK + 8) + 2 * G2_BN * (G2_BK + 8)) * 2)
#define G2_BN_W 128   // the compute_80 module's 128-column tile (kernels_tc2.cuh)
#define G2_SMEM_W ((2 * G2_BM * (G2_BK + 8) + 2 * G2_BN_W * (G2_BK + 8)) * 2)
// the int8 GEMM (kernels_i8.cuh I8_*, compute_80 module, R3.14): TC2's tile
// with int8 operands, plus the stage's weight scales and mins and the
// activation scales and sums
#define I8_BM 128
#define I8_BN 64
#define I8_BK 64
#define I8_SMEM (2 * I8_BM * (I8_BK + 16) + 2 * I8_BN * (I8_BK + 16) + \
                 (2 * I8_BM * 4 + 2 * I8_BM * 4 + 2 * I8_BN * 2 * 2 + I8_BM * 17) * 4)
// k_attn_tc (kernels.cu AQ_*): Q hi, Q lo and one K/V tile of 64 rows of
// 136 halves, plus 64 row scales, in dynamic shared memory
#define AQ_SMEM (3 * 64 * (128 + 8) * 2 + 64 * 4)
// k_attn_tcw (kernels.cu AW_*): Q hi and lo (16 rows) and a K/V tile (32
// rows) of hd + 8 halves, four 16 x 36 partial S tiles, 16 row scales
#define AW_SMEM(hd) ((2 * 16 + 32) * ((hd) + 8) * 2 + 4 * 16 * 36 * 4 + 16 * 4)
#define CU_FUNC_ATTR_MAX_DYN_SMEM 8   // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
// k_attn_dec_mla's geometry (kernels.cu's MD_*, R3.31) and the dynamic shared
// memory its attribute is set to at load: G <= 16 heads' Q beside one 32-row
// tile of rows up to 640 wide
enum { MD_THREADS = 512, MD_GMAX = 16, MD_NJ = 8, MD_TK = 32 };
#define MD_SMEM_MAX (4 * (MD_GMAX * (640 + 4) + MD_TK * 321 + MD_GMAX * MD_TK + MD_GMAX))
#define CU_DEV_ATTR_CC_MAJOR 75       // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
#define CU_DEV_ATTR_CC_MINOR 76       // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
#define TC_N    64   // tensor-core GEMM token tile (keep in sync)

// flash-decoding KV split count — must match ATTN_SPLITS in kernels.cu; fixed
// (not context-dependent) so the decode CUDA graph stays valid across positions
#define ATTN_SPLITS 8

// output rows per batched-matvec block — must match GEMM_WARPS in kernels.cu
#define GEMM_WARPS 8
#define TC_ROWS    64  // tensor-core GEMM output rows per block (match kernels.cu)

// indirect expert matvec: weight base = w_off + sel[slot]*estride, slot = grid.y
// l_off is a BYTE offset: the cache is fp16, q8_0 or fp4 per cache kind,
// so element indexing is not enough (see the kv storage block in kernels.cu)

bool gpu_available(char *name, int cap) {
    if (!cu_load()) return false;
    if (cu.Init(0) != 0) return false;
    int n = 0;
    if (cu.DeviceGetCount(&n) != 0 || n < 1) return false;
    if (name) {
        CUdevice d;
        if (cu.DeviceGet(&d, 0) != 0 || cu.DeviceGetName(name, cap, d) != 0)
            snprintf(name, cap, "CUDA GPU");
    }
    return true;
}

// Stable identity for the VRAM registry. On this project's dev box the GPU is a
// 1g.24gb MIG slice: `nvidia-smi -L` shows one card UUID and a separate
// MIG-<uuid> for the slice, and only the latter distinguishes slices. So the
// _v2 UUID entry point comes first, the v1 one (parent card) second, the PCI
// bus id third, and a bare device index last.
bool gpu_device_id(char *id, int cap) {
    if (!id || cap <= 0) return false;
    if (!cu_load() || cu.Init(0) != 0) return false;
    int n = 0;
    if (cu.DeviceGetCount(&n) != 0 || n < 1) return false;
    CUdevice d;
    if (cu.DeviceGet(&d, 0) != 0) return false;

    unsigned char u[16];
    CUresult r = cu.DeviceGetUuid_v2 ? cu.DeviceGetUuid_v2(u, d)
               : cu.DeviceGetUuid    ? cu.DeviceGetUuid(u, d)
                                     : (CUresult)1;
    if (r == 0) {
        // the canonical 8-4-4-4-12 rendering, so the string matches what
        // nvidia-smi prints and a human can grep the registry filename for it
        snprintf(id, cap,
                 "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
                 "%02x%02x%02x%02x%02x%02x",
                 u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                 u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
        return true;
    }
    char bus[64];
    if (cu.DeviceGetPCIBusId && cu.DeviceGetPCIBusId(bus, sizeof(bus), d) == 0) {
        snprintf(id, cap, "%s", bus);
        return true;
    }
    snprintf(id, cap, "cuda:0");
    return true;
}

bool gpu_mem_info(size_t *free_bytes, size_t *total_bytes) {
    if (!cu_load() || cu.Init(0) != 0) return false;
    int n = 0;
    if (cu.DeviceGetCount(&n) != 0 || n < 1) return false;
    CUdevice dev;
    CUcontext ctx;
    if (cu.DeviceGet(&dev, 0) != 0) return false;
    if (cu.PrimaryCtxRetain(&ctx, dev) != 0) return false;
    size_t f = 0, t = 0;
    bool ok = cu.CtxSetCurrent(ctx) == 0 && cu.MemGetInfo(&f, &t) == 0;
    cu.PrimaryCtxRelease(dev);
    if (!ok) return false;
    if (free_bytes) *free_bytes = f;
    if (total_bytes) *total_bytes = t;
    return true;
}

static bool cu_dev_integrated(CUdevice dev);

bool gpu_offload_budget(int reserve_vram_pct, uint64_t *budget,
                        uint64_t *headroom) {
    if (!cu_load() || cu.Init(0) != 0) return false;
    int n = 0;
    if (cu.DeviceGetCount(&n) != 0 || n < 1) return false;
    CUdevice dev;
    CUcontext ctx;
    if (cu.DeviceGet(&dev, 0) != 0) return false;
    if (cu_dev_integrated(dev)) return false;   // one pool: RAM is the answer
    if (cu.PrimaryCtxRetain(&ctx, dev) != 0) return false;
    size_t f = 0, t = 0;
    bool ok = cu.CtxSetCurrent(ctx) == 0 && cu.MemGetInfo(&f, &t) == 0;
    cu.PrimaryCtxRelease(dev);
    if (!ok) return false;
    // the same OS budget lookup and arithmetic as the loader (shared_build)
    uint64_t os_budget = 0, os_usage = 0;
    bool os_known = false;
    if (cu.DeviceGetLuid) {
        char luid[8]; unsigned mask = 0;
        if (cu.DeviceGetLuid(luid, &mask, dev) == 0)
            os_known = plat_gpu_os_budget((const unsigned char *)luid,
                                          &os_budget, &os_usage);
    }
    *budget = model_gpu_budget(f, t, os_known, os_budget, os_usage,
                               reserve_vram_pct, headroom);
    return true;
}

// CU_DEVICE_ATTRIBUTE_INTEGRATED: a unified-memory device (DGX Spark) reports
// its one memory pool as VRAM, so VRAM and RAM are one budget there.
// The value is 18, not 17: 17 is KERNEL_EXEC_TIMEOUT, which reads 1 on any GPU
// driving a display and would mark every desktop card as unified.
#define CU_DEV_ATTR_INTEGRATED 18
static bool cu_dev_integrated(CUdevice dev) {
    int v = 0;
    if (!cu.DeviceGetAttribute ||
        cu.DeviceGetAttribute(&v, CU_DEV_ATTR_INTEGRATED, dev) != 0)
        return false;
    return v != 0;
}

bool gpu_unified_memory(void) {
    if (!cu_load() || cu.Init(0) != 0) return false;
    int n = 0;
    if (cu.DeviceGetCount(&n) != 0 || n < 1) return false;
    CUdevice dev;
    if (cu.DeviceGet(&dev, 0) != 0) return false;
    return cu_dev_integrated(dev);
}

bool gpu_os_vidmem(const model_t *m, uint64_t *budget, uint64_t *usage,
                   uint64_t *nonlocal) {
    const gpu_t *g = m ? m->gpu : NULL;
    if (!g || !g->sw || !cu.DeviceGetLuid) return false;
    char luid[8]; unsigned mask = 0;
    if (cu.DeviceGetLuid(luid, &mask, g->sw->dev) != 0) return false;
    return plat_gpu_os_vidmem((const unsigned char *)luid, budget, usage, nonlocal);
}

// The device's compute capability, "major.minor": which kernels a default
// run takes depends on it (TC2 runs its compute_80 kernels from 8.0), so
// --caps reports it beside the build's minimum
const char *gpu_compute_capability(void) {
    static char buf[16];
    if (!cu_load() || cu.Init(0) != 0 || !cu.DeviceGetAttribute) return NULL;
    int n = 0, maj = 0, min = 0;
    CUdevice dev;
    if (cu.DeviceGetCount(&n) != 0 || n < 1 || cu.DeviceGet(&dev, 0) != 0 ||
        cu.DeviceGetAttribute(&maj, CU_DEV_ATTR_CC_MAJOR, dev) != 0 ||
        cu.DeviceGetAttribute(&min, CU_DEV_ATTR_CC_MINOR, dev) != 0)
        return NULL;
    snprintf(buf, sizeof buf, "%d.%d", maj, min);
    return buf;
}

// The CUDA kernels ship as embedded PTX, not shader source; there is no
// equivalent string to hash here. Reporting nothing beats reporting a hash
// of something else.
const char *gpu_shader_source_sha(void) { return NULL; }

bool gpu_max_working_set(size_t *bytes) {
    // CUDA's fit ceiling is device memory, which vram_bytes already reports
    (void)bytes;
    return false;
}

static bool gpu_type_ok(int type) {
    switch (type) {
        case T_F32: case T_F16: case T_BF16: case T_Q8_0: case T_Q4_0:
        case T_Q4_1: case T_Q5_0: case T_Q5_1: case T_Q2_K: case T_Q3_K:
        case T_Q4_K: case T_Q5_K:
        case T_Q6_K: case T_IQ4_NL: case T_IQ4_XS: case T_MXFP4:
        case T_NVFP4: case T_IQ1_S: case T_IQ1_M: case T_IQ2_XXS:
        case T_IQ2_XS: case T_IQ2_S: case T_IQ3_XXS: case T_IQ3_S:
        case T_Q2_0:
            return true;
        default:
            return false;
    }
}

// Types whose EVERY device launch applies the per-tensor scale companion
// (mv_args.scale in the k_mv_* / k_mv_*_b / k_moe_mv_* tails). The coalesced
// GEMV, tiled GEMM and tensor-core kernels have their own tails without it,
// so a companion on a type that can reach them stays on the CPU.
static bool gpu_scale_companion_ok(int type) {
    return type == T_NVFP4;
}

// A tracer for the "no CUDA kernel for this type" diagnostic. Every loader
// format has a CUDA kernel now, so CPU-only CI reaches the message by naming a
// type: RUNNER_CUDA_INIT_INJECT_FAILURE=no-kernel:IQ2_XXS.
static bool inject_no_kernel(int type) {
    const char *inj = getenv("RUNNER_CUDA_INIT_INJECT_FAILURE");
    return inj && !strncmp(inj, "no-kernel:", 10) &&
           !strcmp(inj + 10, ggml_type_name(type));
}

static void tc_scalar_notice(const model_t *m);

static bool gpu_tensor_type_ok(const gguf_tensor *t) {
    if (!t) return true;
    if (t->scale != 1.0f && !gpu_scale_companion_ok(t->type)) {
        // the per-tensor scale companion is applied on the host dot seam; a
        // device kernel that read the raw rows would reintroduce the unscaled
        // NVFP4 defect on one backend only
        fprintf(stderr, "gpu: tensor %s carries a per-tensor scale companion, "
                "which no CUDA kernel for %s applies — using CPU\n", t->name,
                ggml_type_name(t->type));
        return false;
    }
    if (gpu_type_ok(t->type) && !inject_no_kernel(t->type)) return true;
    fprintf(stderr, "gpu: tensor %s uses %s, which has no CUDA kernel — "
            "using CPU\n", t->name, ggml_type_name(t->type));
    return false;
}

// Incremented at the one place the TC GEMM is dispatched, so a gate can ask
// "did it run?" instead of inferring it from the output. Also per weight
// type, so the gate can ask it per format: one total cannot tell "the Q4_K
// tensors ran on the tensor cores" from "every TC-capable tensor did".
// Atomic: every slot dispatches, and a ++ two slots race on loses counts.
static _Atomic unsigned long g_tc_dispatches = 0;
static _Atomic unsigned long g_tc_dispatches_type[KT_N];
unsigned long gpu_tc_dispatches(void) { return g_tc_dispatches; }
unsigned long gpu_tc_dispatches_type(int type) {
    return type >= 0 && type < KT_N ? g_tc_dispatches_type[type] : 0;
}

// The tensor-core prefill GEMMs, by weight type. Read at module load to
// resolve the kernels and by the tolerance gate to learn which types it
// can require a dispatch for; one table so the two agree by construction.
static const struct { int type; const char *name; } TC_KERNELS[] = {
    { T_Q4_K, "k_gemm_q4_K_tc" },
    { T_Q6_K, "k_gemm_q6_K_tc" },
    { T_Q8_0, "k_gemm_q8_0_tc" },
    { T_Q4_0, "k_gemm_q4_0_tc" },
    // codebook i-quants: promoted 2026-09-14 on the same arch list
    // (tc_promoted); the forced-TC leg of tests/test_iquants.py is their gate
    { T_IQ3_S,   "k_gemm_iq3_s_tc" },
    { T_IQ3_XXS, "k_gemm_iq3_xxs_tc" },
    { T_IQ2_S,   "k_gemm_iq2_s_tc" },
    { T_IQ2_XS,  "k_gemm_iq2_xs_tc" },
    { T_IQ2_XXS, "k_gemm_iq2_xxs_tc" },
    { T_IQ1_S,   "k_gemm_iq1_s_tc" },
    { T_IQ1_M,   "k_gemm_iq1_m_tc" },
    { T_IQ4_XS,  "k_gemm_iq4_xs_tc" },
    { T_IQ4_NL,  "k_gemm_iq4_nl_tc" },
    // the K-quants and legacy quants that prefilled on the scalar kernels
    // until 2026-10-21 (kernels_kq_stage.cuh)
    { T_Q5_K, "k_gemm_q5_K_tc" },
    { T_Q3_K, "k_gemm_q3_K_tc" },
    { T_Q2_K, "k_gemm_q2_K_tc" },
    { T_Q4_1, "k_gemm_q4_1_tc" },
    { T_Q5_0, "k_gemm_q5_0_tc" },
    { T_Q5_1, "k_gemm_q5_1_tc" },
    { T_NVFP4, "k_gemm_nvfp4_tc" },   // R4.9.5
};
bool gpu_tc_type_has_kernel(int type) {
    for (size_t i = 0; i < sizeof TC_KERNELS / sizeof *TC_KERNELS; i++)
        if (TC_KERNELS[i].type == type) return true;
    return false;
}

// The --caps answer for this backend, sourced from the admission test above so
// the advertised list and the loader agree by construction.
bool gpu_quant_ok(int type) { return gpu_type_ok(type); }

// Expert tensor types the indirect MoE matvecs (k_moe_mv_*) cover; a type
// outside the set keeps the eager per-expert path.
// MOE_ROUTE_MAX: the experts per token k_moe_route holds; MOE_MAX_USED: the
// picks a device route selects. A model past either is refused a device bank
// (cuda_admit, moe_bank_device_ok): a clamped route would answer among a
// subset of the experts without a word.
enum { MOE_ROUTE_MAX = 512, MOE_MAX_USED = 256 };

static bool moe_indirect_type_ok(int type) {
    switch (type) {
        case T_F32: case T_F16: case T_Q8_0: case T_Q4_0:
        case T_Q4_K: case T_Q5_K: case T_Q6_K: case T_MXFP4:
        case T_NVFP4: case T_IQ3_XXS: case T_IQ4_NL: case T_Q2_0:
        case T_IQ2_S: case T_IQ3_S: case T_IQ4_XS:
        // 2026-10-21: llama-quantize's Q4_K_M of gemma-4-26B-A4B keeps some
        // experts in Q5_0, which sent the whole model to the eager path
        case T_Q5_0: case T_Q4_1: case T_Q5_1:
            return true;
        // IQ2_XS/IQ2_XXS have indirect twins (k_moe_mv_iq2_xs/_xxs) but the
        // fused path measured 6.0 tok/s against 19.1 eager on the IQ2_XS
        // Qwen3.8-Flash-Next file (prefill 3.8 vs 33.9), same split, same
        // memory; until that is understood the eager path stays their default
        case T_IQ2_XS: case T_IQ2_XXS:
            return false;
        default:
            return false;
    }
}

// Expert tensor types with width-classed batched kernels (f_gemvb twins and a
// GEMM). Grouped prefill only pays when the expert matmul's work scales with
// its token count; the fixed-MVB kernels compute every column regardless
// (gemma-4 Q4_0 prefill: 19% slower grouped). Other types stay fused.
// defined with the arithmetic routes and the TC promotion below
static int route_pinned(int bit);
static bool moe_grouped_family(const model_t *m);

static bool moe_grouped_type_ok(int type) {
    switch (type) {
        // Q4_0 joined 2026-10-20: at the runtime prompt window an expert
        // averages ~32 columns, which the tensor-core GEMM covers for Q4_0
        // (excluded while tiles were 16 columns, ~1-2 an expert)
        case T_Q8_0: case T_Q4_K: case T_Q5_K: case T_Q6_K: case T_Q4_0:
            return true;
        default:
            return false;
    }
}

// Per-layer expert placement under tensor-role mode. Outside that mode every
// expert bank is device-resident, so the array is absent and the answer is no.
// test hook state (definition of the setter is beside gpu_tc_force below)
static _Atomic int g_moe_eager_force = -1;

static bool moe_on_host(const model_t *m, int l) {
    return m->cpu_moe && m->moe_host && m->moe_host[l];
}

// RUNNER_MOE_HOT (R3.30): a host layer whose hot experts the device holds
// and runs; the rest of its routed experts stay on the host
static bool moe_hot_on(const model_t *m, int l) {
    return m->moe_hot_n && l >= 0 && l < m->n_layer && m->moe_hot_n[l] > 0 &&
           m->layers[l].is_moe && moe_on_host(m, l);
}

// One hot expert's gate, up and down slices, the bank's slot stride.
static void moe_hot_strides(const layer_t *ly, int n_embd, int nff, uint64_t *gs,
                            uint64_t *us, uint64_t *ds) {
    *gs = (uint64_t)nff * ggml_row_size(ly->ffn_gate_exps->type, n_embd);
    *us = (uint64_t)nff * ggml_row_size(ly->ffn_up_exps->type, n_embd);
    *ds = (uint64_t)n_embd * ggml_row_size(ly->ffn_down_exps->type, nff);
}

// The bytes the hot banks claim on the device, routers included.
static uint64_t moe_hot_bytes(const model_t *m) {
    uint64_t total = 0;
    for (int l = 0; l < m->n_layer; l++) {
        if (!moe_hot_on(m, l)) continue;
        const layer_t *ly = &m->layers[l];
        uint64_t gs, us, ds;
        moe_hot_strides(ly, m->n_embd, m->n_ff_exp, &gs, &us, &ds);
        total += (uint64_t)m->moe_hot_n[l] * (gs + us + ds) + ly->ffn_gate_inp->nbytes;
    }
    return total;
}

static bool moe_any_on_host(const model_t *m) {
    if (!m->cpu_moe) return false;
    for (int l = 0; l < m->n_layer; l++)
        if (m->layers[l].is_moe && moe_on_host(m, l)) return true;
    return false;
}

// Any expert bank the device is expected to run — the trigger for uploading
// expert tensors, validating their types, and allocating MoE scratch.
static bool moe_any_on_device(const model_t *m) {
    if (m->n_expert <= 0) return false;
    for (int l = 0; l < m->n_layer; l++)
        if (m->layers[l].is_moe && (!moe_on_host(m, l) || moe_hot_on(m, l))) return true;
    return false;
}

// Whether the device can run layer l's expert bank. The routed kernels cover a
// SiLU shared expert with its scalar sigmoid gate or without one; a relu2 or
// gate-less one stays on the host, and Granite 4 hybrid keeps its banks there
// by name (its CUDA cells were measured so). Admission and the auto fit ask this.
static bool moe_bank_device_ok(const model_t *m, int l) {
    const layer_t *ly = &m->layers[l];
    // A bank on the device is routed by k_moe_route: softmax and top-k, with
    // no selection bias and no weight scale. A bank whose router needs more
    // (nemotron_h_moe and lfm2moe: a sigmoid with exp_probs_b) stays on the
    // host, where model_moe_ffn_cpu routes it with the CPU's moe_route
    // (R3.43a).
    if (ly->is_moe && (!model_moe_router_is_plain_any_norm(m) || ly->exp_probs_b))
        return false;
    // more experts or picks than a device route takes: it would route among
    // a clamped set (R6.68.67)
    if (ly->is_moe && (ly->n_expert > MOE_ROUTE_MAX || m->n_expert > MOE_ROUTE_MAX ||
                       m->n_expert_used > MOE_MAX_USED))
        return false;
    if (!ly->is_moe || m->n_ff_shexp <= 0) return true;
    if (m->ffn_relu2) return false;
    if (!ly->w_up_shexp) return true;
    if (!ly->w_gate_shexp) return false;
    return ly->ffn_gate_inp_shexp || !m->granite_hybrid;
}

// The fused device-routing MoE path needs the fused-3D expert layout (a
// legacy split model has one arbitrarily-placed tensor per expert, which
// in-kernel base+e*stride addressing cannot express) and indirect kernels
// for every expert tensor type. All-or-nothing per model: one eager layer
// would force the graph off anyway.
static bool moe_fused_eligible(const model_t *m) {
    if (m->n_expert <= 0 || m->n_expert > MOE_ROUTE_MAX) return false;
    if (!moe_any_on_device(m)) return false;
    for (int l = 0; l < m->n_layer; l++) {
        const layer_t *ly = &m->layers[l];
        if (!ly->is_moe || (moe_on_host(m, l) && !moe_hot_on(m, l))) continue;
        if (ly->moe_split) return false;
        if (ly->moe_gemma) {
            if (!moe_indirect_type_ok(ly->ffn_gate_up_exps->type) ||
                !moe_indirect_type_ok(ly->ffn_down_exps->type)) return false;
        } else {
            if (!moe_indirect_type_ok(ly->ffn_gate_exps->type) ||
                !moe_indirect_type_ok(ly->ffn_up_exps->type) ||
                !moe_indirect_type_ok(ly->ffn_down_exps->type)) return false;
        }
    }
    return true;
}

static bool moe_grouped_eligible(const model_t *m) {
    // The moe-grouped route: grouping only pays with many tokens per expert, so
    // it is on by default at the runtime prompt window for the families measured
    // for it (moe_grouped_family). A route pin decides first, then
    // RUNNER_MOE_GROUPED (1 on any family, 0 on none).
    int p = route_pinned(GR_MOE_GROUPED);
    bool on, by_default = false;
    if (p >= 0) on = p;
    else {
        const char *e = getenv("RUNNER_MOE_GROUPED");
        if (e && *e) on = strcmp(e, "0") != 0 && strcmp(e, "off") != 0;
        else on = by_default = moe_grouped_family(m);
    }
    if (!on) return false;
    if (!moe_fused_eligible(m)) return false;
    for (int l = 0; l < m->n_layer; l++) {
        const layer_t *ly = &m->layers[l];
        if (!ly->is_moe) continue;
        // per-expert biases (gpt-oss) are wired through the fused path's
        // actmul/sum only; the grouped scatter has no seat for them, so a
        // biased model keeps the fused prefill (refuse > silently wrong)
        if (ly->ffn_gate_exps_b || ly->ffn_up_exps_b || ly->ffn_down_exps_b)
            return false;
        if (ly->moe_gemma) {
            if (!moe_grouped_type_ok(ly->ffn_gate_up_exps->type) ||
                !moe_grouped_type_ok(ly->ffn_down_exps->type)) return false;
        } else {
            if (!moe_grouped_type_ok(ly->ffn_gate_exps->type) ||
                !moe_grouped_type_ok(ly->ffn_up_exps->type) ||
                !moe_grouped_type_ok(ly->ffn_down_exps->type)) return false;
            // the default covers the expert types its float32-reference row
            // measured (Qwen3-30B-A3B Q4_K_M: Q4_K and Q6_K experts); other
            // admitted types take it when asked for (route pin or env)
            if (by_default &&
                !((ly->ffn_gate_exps->type == T_Q4_K || ly->ffn_gate_exps->type == T_Q6_K) &&
                  (ly->ffn_up_exps->type == T_Q4_K || ly->ffn_up_exps->type == T_Q6_K) &&
                  (ly->ffn_down_exps->type == T_Q4_K || ly->ffn_down_exps->type == T_Q6_K)))
                return false;
        }
    }
    return true;
}

// The by-expert prefill (moe_twin) reproduces the fused path bit for bit: an
// expert's columns go through the width-classed GEMV twins, which carry the
// batch-1 kernel's bits, the one each k_moe_mv_* body copies. So a type needs
// both twins, its k_moe_mv kernel and that batch-1 kernel. Float types stay
// fused: BF16's batch-1 route can be split-K, which k_moe_mv_bf16 does not copy.
static bool moe_twin_type_ok(const gpu_weights *w, int type) {
    // Admitted only where the kernels a type runs have k_moe_mv_<type>'s
    // per-column arithmetic. Q8_0 and the K-quants: the coalesced GEMV, which
    // is also the dense dispatch. MXFP4: k_mv_mxfp4 and its twins. Q4_0:
    // k_moe_mv_q4_0 is k_mv_q4_0's body, not the coalesced k_gemv_q4_0 the dense
    // dispatch picks, so moe_twin_gemm runs k_mv_q4_0 and its own twins.
    switch (type) {
    case T_Q8_0: case T_Q4_K: case T_Q5_K: case T_Q6_K: case T_MXFP4:
        return w->f_gemvb[BW_4][type] && w->f_gemvb[BW_8][type] && w->f_moe_mv[type] &&
               (w->f_gemv[type] || w->f_mv[type]);
    case T_Q4_0:
        return w->f_moe_q4_0_mvb[BW_4] && w->f_moe_q4_0_mvb[BW_8] && w->f_mv[T_Q4_0] &&
               w->f_moe_mv[T_Q4_0];
    default:
        return false;
    }
}
static bool moe_twin_eligible(const model_t *m, const gpu_weights *w) {
    if (!w) return false;
    for (int l = 0; l < m->n_layer; l++) {
        const layer_t *ly = &m->layers[l];
        if (!ly->is_moe || moe_on_host(m, l)) continue;
        if (ly->moe_split) return false;
        if (ly->moe_gemma) {   // the fused gate_up tensor and the down tensor
            if (!ly->ffn_gate_up_exps || !ly->ffn_down_exps ||
                !moe_twin_type_ok(w, ly->ffn_gate_up_exps->type) ||
                !moe_twin_type_ok(w, ly->ffn_down_exps->type))
                return false;
            continue;
        }
        if (!ly->ffn_gate_exps || !ly->ffn_up_exps || !ly->ffn_down_exps ||
            !moe_twin_type_ok(w, ly->ffn_gate_exps->type) ||
            !moe_twin_type_ok(w, ly->ffn_up_exps->type) ||
            !moe_twin_type_ok(w, ly->ffn_down_exps->type))
            return false;
    }
    return true;
}

static void gpu_ctx_free(model_t *m, gpu_t *g);
static void route_replay_flush(gpu_t *g);
static bool gpu_i8_on(void);
static bool gpu_q8x_on(void);

#define CK(call) do { CUresult _r = (call); if (_r != 0) { \
    fprintf(stderr, "gpu: %s failed: %s\n", #call, cu_err(_r)); goto fail; } } while (0)

// deepseek2's per-column MLA scratch (mla_buf), in floats: kv_a [kv_lora +
// rope], the normed latent [kv_lora], then kv_b's output [heads x (nope + v)]
// in the naive form or the raw Q [heads x key_length] in the absorbed one,
// and the packed V [heads x v]; 0 for every other architecture
static size_t mla_cols(const model_t *m) {
    if (!m->deepseek2) return 0;
    size_t r = (size_t)m->mla_kv_lora, rd = (size_t)m->rope_dim, H = (size_t)m->n_head;
    size_t nope = (size_t)(m->mla_qk - m->rope_dim), vd = (size_t)m->mla_v_dim;
    size_t mid = m->mla_absorb ? H * (size_t)m->mla_qk : H * (nope + vd);
    return (r + rd) + r + mid + H * vd;
}

static CUdeviceptr f32_dbuf(const float *src, size_t n, const char *what,
                           int layer) {
    if (!src) return 0;
    if (n > SIZE_MAX / sizeof(float)) {
        if (layer >= 0)
            fprintf(stderr, "gpu: CUDA upload size overflow for %s in blk.%d "
                    "— using CPU\n", what, layer);
        else
            fprintf(stderr, "gpu: CUDA upload size overflow for %s — using "
                    "CPU\n", what);
        return 0;
    }
    CUdeviceptr d = 0;
    CUresult rc = cu.MemAlloc(&d, n * sizeof(float));
    if (rc != 0) {
        if (layer >= 0)
            fprintf(stderr, "gpu: CUDA allocation for %s in blk.%d failed: %s "
                    "— using CPU\n", what, layer, cu_err(rc));
        else
            fprintf(stderr, "gpu: CUDA allocation for %s failed: %s — using "
                    "CPU\n", what, cu_err(rc));
        return 0;
    }
    rc = cu.MemcpyHtoD(d, src, n * sizeof(float));
    if (rc != 0) {
        if (layer >= 0)
            fprintf(stderr, "gpu: CUDA upload for %s in blk.%d failed: %s — "
                    "using CPU\n", what, layer, cu_err(rc));
        else
            fprintf(stderr, "gpu: CUDA upload for %s failed: %s — using CPU\n",
                    what, cu_err(rc));
        cu.MemFree(d);
        return 0;
    }
    return d;
}

// As f32_dbuf, but a NULL src uploads an all-ones buffer instead of returning 0.
// The CPU rmsnorm() runs weightless (multiply by 1) when its weight is NULL, so
// an optional gemma norm that a GGUF omits must become ones on the GPU — a null
// device pointer would instead fault k_rmsnorm (no NULL guard in the kernel).
static CUdeviceptr f32_dbuf_ones(const float *src, size_t n, const char *what,
                                int layer) {
    if (src) return f32_dbuf(src, n, what, layer);
    if (n > SIZE_MAX / sizeof(float)) {
        if (layer >= 0)
            fprintf(stderr, "gpu: CUDA host size overflow for %s in blk.%d — "
                    "using CPU\n", what, layer);
        else
            fprintf(stderr, "gpu: CUDA host size overflow for %s — using "
                    "CPU\n", what);
        return 0;
    }
    float *ones = malloc(n * sizeof(float));
    if (!ones) {
        if (layer >= 0)
            fprintf(stderr, "gpu: CUDA host staging allocation for %s in "
                    "blk.%d failed — using CPU\n", what, layer);
        else
            fprintf(stderr, "gpu: CUDA host staging allocation for %s failed "
                    "— using CPU\n", what);
        return 0;
    }
    for (size_t i = 0; i < n; i++) ones[i] = 1.0f;
    CUdeviceptr d = f32_dbuf(ones, n, what, layer);
    free(ones);
    return d;
}

// one integer naming the cache layout for the shared-weights identity: 0 f16,
// 1 q8_0, 2 fp4, 3 the k8v4 split (K q8_0, V fp4), 4 f32. Two instances with
// different layouts cannot share a device cache. Not the kernels' kind
// numbering (model_kv_kind_k), where 3 is f32.
static int kv_identity(const model_t *m) {
    return m->kv_f32 ? 4 : m->kv_split ? 3 : m->kv_fp4 ? 2 : (int)m->kv_q8;
}

// the units k_store_kv's threads store a row in, for cache kind `kind`
// (kernels.cu kv_units): an fp4 or q8_0 block, else one element (f16, f32)
static int kv_store_units(int kind, int kv_dim) {
    return kind == 3 ? kv_dim : kind == 2 ? kv_dim / 16 : kind ? kv_dim / 32 : kv_dim;
}

bool gpu_kv_q8_ok(void) {
    return true;    // k_store_kv / k_attn / k_attn_dec all read q8_0 blocks
}
bool gpu_kv_fp4_ok(void) {
    return true;    // k_store_kv / k_attn / k_attn_dec read fp4 blocks (kind 2)
}
bool gpu_kv_f32_ok(void) {
    return true;    // and f32 rows (kind 3, --kv f32, R6.33)
}

bool gpu_moe_ok(void) {
    return true;    // expert banks upload and route on the device
}

// A block whose attention or FFN was removed (--remove-sublayer) is walked
// without that branch: fwd_tile skips it per block, and the multi-sequence
// microbatch declines such a model (batch_eligible wants every layer's
// attention weights), so its sequences decode one at a time.
bool gpu_removed_sublayers_ok(void) { return true; }

bool gpu_moe_route_caps(int *max_experts, int *max_used) {
    *max_experts = MOE_ROUTE_MAX;
    *max_used = MOE_MAX_USED;
    return true;
}

// the DeltaNet layers run on the device (gpu_q35_recurrent), adapters too
bool gpu_qwen35_ok(void) { return true; }

bool gpu_eseries_ok(void) {
    // stage_ple ships the per-layer-embedding pre-pass each forward and enc_ple
    // runs the gate/proj/norm branch per layer; gpu_init has no PLE refusal.
    return true;
}

// ------------------------------------------------- shared-weight registry
// Resident uploads, refcounted and keyed on model identity. The lock is held
// across the whole build, so two instances racing on one file cannot both
// pay for the upload.
static gpu_weights *g_shared;
static pthread_mutex_t g_shared_mu = PTHREAD_MUTEX_INITIALIZER;

// File identity beyond the path: a model rebuilt on disk between two loads
// must not be served out of the previous upload. model_file_identity is
// shared with the host-side registry in model.c so one place decides what a
// file IS, and one place reports being unable to tell -- losing it here also
// costs this instance the first one's CPU/GPU split, not just the upload.
static bool file_id(const char *path, uint64_t *size, uint64_t *ino,
                    int64_t *mtime) {
    return model_file_identity(path, "device weights", size, ino, mtime, NULL);
}

// Everything the shared buffers derive from, compared, not assumed from the
// path. The rope tables are not a property of the file alone (YaRN and phi3's
// LongRoPE key off the requested context; --rope-base/--rope-scale override),
// so two loads of one file can want different tables. A miss is not an error.
static bool shared_config_matches(const gpu_weights *w, const model_t *m) {
    if (w->n_layer != m->n_layer || w->n_embd != m->n_embd ||
        w->n_head  != m->n_head  || w->n_head_kv != m->n_head_kv ||
        w->head_dim != m->head_dim || w->n_ff != m->n_ff ||
        w->n_vocab != m->n_vocab || w->n_ctx != m->n_ctx ||
        w->rope_dim != m->rope_dim || w->rope_dim_local != m->rope_dim_local ||
        w->kv_q8 != kv_identity(m) || w->v_rmsnorm != (int)m->v_rmsnorm ||
        w->rope_base != m->rope_base || w->rope_mscale != m->rope_mscale ||
        w->cpu_moe != m->cpu_moe ||
        w->cpu_moe_layers != m->cpu_moe_layers ||
        w->mtp != m->mtp_ready)
        return false;
    if ((w->rope_inv_freq_local != NULL) != (m->rope_inv_freq_local != NULL))
        return false;
    if (memcmp(w->rope_inv_freq, m->rope_inv_freq,
               sizeof(float) * (size_t)(m->rope_dim / 2)) != 0)
        return false;
    if (m->rope_inv_freq_local &&
        memcmp(w->rope_inv_freq_local, m->rope_inv_freq_local,
               sizeof(float) * (size_t)(m->rope_dim_local / 2)) != 0)
        return false;
    return true;
}

static bool shared_matches(const gpu_weights *w, const model_t *m,
                           uint64_t size, uint64_t ino, int64_t mtime) {
    // a no-identity entry is in the list only for the split guard below; its
    // zeroed identity fields would already refuse a real file, but be explicit
    if (w->no_id) return false;
    if (!w->path || !m->path || strcmp(w->path, m->path) != 0) return false;
    if (w->fsize != size || w->fino != ino || w->fmtime != mtime) return false;
    // a multi-part file (split GGUF, --mtp-file companion) is identified by
    // its first part plus the part count and the total mapped bytes
    if (w->n_maps != gguf_map_count(&m->gf) || w->mapped_size != gguf_mapped_size(&m->gf))
        return false;
    return shared_config_matches(w, m);
}

// An instance without a file identity builds privately and re-decides its
// split under the resident instance's VRAM pressure, so two slots of one
// server can answer with different logits. Detected (same path and
// configuration, different split) and said loudly: a refusal would diverge as
// silently on the CPU.
static void split_guard(const gpu_weights *w, const model_t *m) {
    for (const gpu_weights *o = g_shared; o; o = o->next) {
        if (o == w) continue;
        if (!o->path || !m->path || strcmp(o->path, m->path) != 0) continue;
        if (!(w->no_id || o->no_id)) continue;
        if (!shared_config_matches(o, m)) continue;
        if (o->gpu_layers == w->gpu_layers) continue;
        fprintf(stderr,
                "error: %s re-decided the CPU/GPU split without a file "
                "identity: this instance put %d/%d layers on the GPU but a "
                "resident instance of the same path uses %d/%d — two slots "
                "of one server will answer the same request differently\n",
                m->path, w->gpu_layers, w->n_layer,
                o->gpu_layers, o->n_layer);
    }
}

// Would gpu_init for this model reuse a resident upload? The VRAM claim made
// before gpu_init asks, so a second slot (or a draft of the same file) claims
// its own KV cache and scratch, not a second copy of the weights. The same
// match gpu_init makes, under the same lock; a miss is not an error.
bool gpu_shared_weights_resident(const model_t *m) {
    uint64_t size = 0, ino = 0;
    int64_t mtime = 0;
    if (!m || !m->path || !file_id(m->path, &size, &ino, &mtime)) return false;
    bool hit = false;
    pthread_mutex_lock(&g_shared_mu);
    for (const gpu_weights *w = g_shared; w && !hit; w = w->next)
        hit = shared_matches(w, m, size, ino, mtime);
    pthread_mutex_unlock(&g_shared_mu);
    return hit;
}

// A row of gpu_tables.h's table list and the gpu_weights pointer table it
// lives in. No default case: -Wswitch refuses a row without its line here.
static CUdeviceptr **cuda_table(gpu_weights *w, gpu_table t) {
    switch (t) {
    case GT_ATTN_NORM:       return &w->attn_norm;
    case GT_HC_ATTN_NORM:    return &w->hc_an;
    case GT_HC_FFN_NORM:     return &w->hc_fn;
    case GT_FFN_NORM:        return &w->ffn_norm;
    case GT_BQ:              return &w->bq;
    case GT_BK:              return &w->bk;
    case GT_BV:              return &w->bv;
    case GT_BO:              return &w->bo;
    case GT_QNORM:           return &w->qn;
    case GT_KNORM:           return &w->kn;
    case GT_MLA_KV_NORM:     return &w->mla_kvn;
    case GT_MLA_KVB:         return &w->mla_kvb;
    case GT_POST_ATTN_NORM:  return &w->pan;
    case GT_POST_FFN_NORM:   return &w->pfn;
    case GT_PLE_POST_NORM:   return &w->ple_pn;
    case GT_SINKS:           return &w->sinks;
    case GT_MOE_ROUTER_B:    return &w->gib;
    case GT_MOE_GATE_B:      return &w->geb;
    case GT_MOE_UP_B:        return &w->ueb;
    case GT_MOE_DOWN_B:      return &w->deb;
    case GT_SSM_DT:          return &w->ssm_dt;
    case GT_SSM_A:           return &w->ssm_a;
    case GT_SSM_NORM:        return &w->ssm_norm;
    case GT_SSM_GNORM:       return &w->ssm_gnorm;
    case GT_SSM_D:           return &w->ssm_D;
    case GT_SSM_CONV_B:      return &w->ssm_conv_b;
    case GT_G_POST_NORM1:    return &w->g_pn1;
    case GT_G_PRE_NORM2:     return &w->g_prn2;
    case GT_G_POST_NORM2:    return &w->g_pn2;
    case GT_G_ROUTER_SCALE:  return &w->g_gis;
    case GT_G_DOWN_SCALE:    return &w->g_dsc;
    case GT_HOT_BANK:        return &w->hot_bank;
    case GT_COUNT:           break;
    }
    return NULL;
}

// Whether layer l's table of row `d` must exist (the uploader's check): by the
// row's need, over the layer as the binders and the placement left it.
static bool cuda_table_needed(const gpu_table_desc *d, const model_t *m, const layer_t *ly,
                              bool moe_dev) {
    // read as bytes: the rows name host fields of their own pointer types, and
    // a load through a void * lvalue would access them through another type
    // (undefined under the aliasing rules this build optimizes under)
    const void *host = NULL;
    if (d->host != GT_NO_HOST) memcpy(&host, (const char *)ly + d->host, sizeof host);
    switch (d->need) {
    case GT_NEED_HOST:      return host != NULL;
    case GT_NEED_HC:        return m->hyper_conn;
    case GT_NEED_REC:       return ly->recurrent;
    case GT_NEED_REC_DN:    return ly->recurrent && m->qwen35;
    case GT_NEED_REC_M2:    return ly->recurrent && !m->qwen35;
    case GT_NEED_MOE_DEV:   return moe_dev && host != NULL;
    case GT_NEED_GEMMA_MOE: return ly->moe_gemma;
    case GT_NEED_NONE:      return false;
    }
    return false;
}

static void shared_destroy(gpu_weights *w) {
    if (!w) return;
    if (w->ctx) cu.CtxSetCurrent(w->ctx);
    // every table slot, the bound MTP block's (index n_layer) included, then
    // the pointer tables: gpu_tables.h's rows, so a table added there is freed
    for (int t = 0; t < GT_COUNT; t++) {
        CUdeviceptr *tab = *cuda_table(w, (gpu_table)t);
        for (int l = 0; tab && l < w->n_tab; l++)
            if (tab[l]) cu.MemFree(tab[l]);
        free(tab);
    }
    for (int i = 0; i < w->n_bindings; i++)
        if (w->bindings[i].device) cu.MemFree(w->bindings[i].device);
    free(w->bindings);
    // the hyper-connection head gamma and the MTP gammas were missing here
    // until 2026-10-20, like the per-layer tables above: each load of such a
    // file left them in the context every model on the device shares
    CUdeviceptr bufs[] = { w->weights, w->inv_freq, w->inv_freq_local,
                           w->out_norm, w->dummy, w->ones, w->moe_ones,
                           w->hc_head_norm, w->mtp_enorm, w->mtp_hnorm,
                           w->mtp_head_norm };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++)
        if (bufs[i]) cu.MemFree(bufs[i]);
    if (w->mod80) cu.ModuleUnload(w->mod80);
    if (w->mod) cu.ModuleUnload(w->mod);
    if (w->ctx) cu.PrimaryCtxRelease(w->dev);
    free(w->rope_inv_freq); free(w->rope_inv_freq_local);
    free(w->path);
    free(w);
}

static void shared_release(gpu_weights *w) {
    if (!w) return;
    pthread_mutex_lock(&g_shared_mu);
    bool last = --w->refs <= 0;
    if (last)
        // unlink first so a concurrent acquire cannot find a dying entry; the
        // loop is a no-op for an entry that was never listed (no file identity)
        for (gpu_weights **pp = &g_shared; *pp; pp = &(*pp)->next)
            if (*pp == w) { *pp = w->next; break; }
    pthread_mutex_unlock(&g_shared_mu);
    if (last) shared_destroy(w);
}

// Upload one model's immutable device data and decide the CPU/GPU layer split.
// act_bytes/max_hd describe the per-sequence scratch of one instance; they
// feed the fitting decision only.
// A layer's weight bytes cover attention plus the FFN, dense or sparse-MoE
// (router and every expert): the dense tensors are NULL on a MoE layer.
// plan_bytes: a tensor's device bytes in the plan (a per-tensor upload is
// rounded to the driver's allocation granularity).
static size_t plan_bytes(const gguf_tensor *t, bool bound) {
    if (!t) return 0;
    return (size_t)gpu_plan_binding_bytes(t->nbytes, bound);
}

// an expert bank as uploaded: its bytes plus, bound, the rounding of its
// (at most nine) tensors: router, routed gate/up/down or fused gate_up,
// shared gate/up/down and the shared-expert gate
static size_t expert_plan_bytes(const layer_t *ly, int n_expert, bool bound) {
    return model_layer_expert_bytes(ly, n_expert) + (bound ? 9 * GPU_PLAN_GRAN : 0);
}

// A layer's device tensors, listed once: the split plan's bytes, the image's
// stretch, the per-tensor bindings and admission's type check all walk these
// two lists, so none can drift from the others.
// tests/test_cuda_tensor_roles.py holds every tensor field of layer_t to one
// of the lists or a named host-side role.
enum { LAYER_MIXER_MAX = GT_N_MIXER };

// The mixer side, on the device whatever the FFN does: attention or the
// recurrent mixer, the hyper-connection stream mixers (four projections and
// two injects per layer: 1.27 GB on Qwen3.8-Flash-Next), the E-series
// per-layer embedding projections. Absent tensors are left out.
static int layer_mixer_tensors(const layer_t *ly, gguf_tensor **out) {
    // gpu_tables.h's mixer rows, in their order: the fit prices the same list
    const gpu_mixer_desc *rows = gpu_mixer_rows();
    int n = 0;
    for (int i = 0; i < LAYER_MIXER_MAX; i++) {
        gguf_tensor *t = *(gguf_tensor *const *)((const char *)ly + rows[i].field);
        if (t) out[n++] = t;
    }
    return n;
}

// The FFN side as the device holds it: the dense projections, or the
// expert bank (router, routed experts fused or one tensor per expert, the
// shared expert and its gate, Gemma's dense branch). fn sees each present
// tensor in turn; a false from it stops the walk and is returned.
static bool layer_ffn_tensors_each(const layer_t *ly, int n_expert,
                                   bool (*fn)(gguf_tensor *, void *), void *ctx) {
    gguf_tensor *dense[] = { ly->w_gate, ly->w_up, ly->w_down };
    if (!ly->is_moe) {
        for (int i = 0; i < 3; i++)
            if (dense[i] && !fn(dense[i], ctx)) return false;
        return true;
    }
    gguf_tensor *bank[] = { ly->ffn_gate_inp, ly->ffn_gate_up_exps,
                            ly->ffn_gate_exps, ly->ffn_up_exps,
                            ly->ffn_down_exps, ly->w_gate_shexp,
                            ly->w_up_shexp, ly->w_down_shexp,
                            ly->ffn_gate_inp_shexp };
    for (size_t i = 0; i < sizeof bank / sizeof *bank; i++)
        if (bank[i] && !fn(bank[i], ctx)) return false;
    if (ly->moe_split)
        for (int e = 0; e < n_expert; e++) {
            gguf_tensor *one[] = { ly->moe_g[e], ly->moe_u[e], ly->moe_d[e] };
            for (int i = 0; i < 3; i++)
                if (one[i] && !fn(one[i], ctx)) return false;
        }
    if (ly->moe_gemma)
        for (int i = 0; i < 3; i++)
            if (dense[i] && !fn(dense[i], ctx)) return false;
    return true;
}

static size_t layer_weight_bytes(model_t *m, int l, bool host_experts, bool bound) {
    const layer_t *ly = &m->layers[l];
    int n_expert = ly->n_expert;
    if (!bound && !host_experts)
        return (size_t)model_cuda_layer_charge(&m->gf, l, false).file;
    size_t wb = 0;
    gguf_tensor *mix[LAYER_MIXER_MAX];
    int nmix = layer_mixer_tensors(ly, mix);
    for (int i = 0; i < nmix; i++)
        wb += plan_bytes(mix[i], bound);
    if (ly->is_moe && host_experts) {
        // The complete expert FFN (router, routed tensors and a coupled
        // Gemma shared branch) executes on the host. Attention remains here.
    } else if (ly->is_moe) {
        wb += expert_plan_bytes(ly, n_expert, bound);
    } else {
        gguf_tensor *ffn[] = { ly->w_gate, ly->w_up, ly->w_down };
        for (int i = 0; i < 3; i++) wb += plan_bytes(ffn[i], bound);
    }
    return wb;
}

// RUNNER_DEBUG_VRAM=1: the driver's free figure at each init stage, so the
// placement plan can be checked against what the device actually spends
static bool vram_debug_on(void) {
    static _Atomic int on = -1;
    if (on < 0) { const char *e = getenv("RUNNER_DEBUG_VRAM"); on = e && *e && *e != '0'; }
    return on == 1;
}
static void vram_probe(const char *stage) {
    size_t f = 0, t = 0;
    if (vram_debug_on() && cu.MemGetInfo && cu.MemGetInfo(&f, &t) == 0)
        fprintf(stderr, "vram: %-28s free %.3f GB\n", stage, f / 1e9);
}

// RUNNER_DEBUG_VRAM: what the weight upload actually takes (under WDDM a
// device can lose more than the bytes uploaded). For the upload stage every
// cu.MemAlloc goes through va_alloc, which reads the driver's free figure
// around the call. Debug only: off, nothing is swapped or read.
static struct {
    bool on;
    CUresult (*real)(CUdeviceptr *, size_t);
    size_t free0;
    int n;
    uint64_t asked, taken, gap_max, gap_max_asked;
} g_va;

static CUresult va_alloc(CUdeviceptr *p, size_t n) {
    size_t f0 = 0, f1 = 0, t = 0;
    bool known = cu.MemGetInfo(&f0, &t) == 0;
    CUresult rc = g_va.real(p, n);
    if (rc == 0 && known && cu.MemGetInfo(&f1, &t) == 0) {
        uint64_t took = f0 > f1 ? (uint64_t)(f0 - f1) : 0;
        uint64_t gap = took > n ? took - n : 0;
        g_va.n++;
        g_va.asked += n;
        g_va.taken += took;
        if (gap > g_va.gap_max) { g_va.gap_max = gap; g_va.gap_max_asked = n; }
    }
    return rc;
}

static void upload_vram_begin(void) {
    size_t t = 0;
    memset(&g_va, 0, sizeof g_va);
    if (!vram_debug_on() || !cu.MemGetInfo || !cu.MemAlloc ||
        cu.MemGetInfo(&g_va.free0, &t) != 0)
        return;
    g_va.real = cu.MemAlloc;
    cu.MemAlloc = va_alloc;
    g_va.on = true;
}

static void upload_vram_end(void) {
    if (!g_va.on) return;
    cu.MemAlloc = g_va.real;
    g_va.on = false;
    size_t f1 = 0, t = 0;
    if (cu.MemGetInfo(&f1, &t) != 0) return;
    int64_t stage = (int64_t)g_va.free0 - (int64_t)f1;
    int64_t outside = stage - (int64_t)g_va.taken;
    fprintf(stderr, "vram: upload: %d allocations, %.3f GB asked, %.3f GB taken by "
                    "them; largest gap %.1f MB (on a %.1f MB allocation); the stage "
                    "took %.3f GB, %+.3f GB of it outside the allocations "
                    "(bytes: asked %llu, taken %llu, stage %lld)\n",
            g_va.n, g_va.asked / 1e9, g_va.taken / 1e9, g_va.gap_max / 1e6,
            g_va.gap_max_asked / 1e6, stage / 1e9, outside / 1e9,
            (unsigned long long)g_va.asked, (unsigned long long)g_va.taken,
            (long long)stage);
}


static uint64_t tensor_file_off(const model_t *m, const gguf_tensor *t) {
    return (uint64_t)((const uint8_t *)t->data - (const uint8_t *)m->gf.map);
}

// The NextN/MTP head's residual: one row of n_embd, or one per
// hyper-connection stream.
static int mtp_streams(const model_t *m) { return m->hyper_conn ? m->hc_count : 1; }


// Tensor-role placement cannot keep mmap offsets in one compact allocation
// (the skipped experts are holes): each retained tensor is a small binding,
// and matvecs resolve by GGUF file offset.
// Every copy of weight bytes from the file mapping goes through here in
// UPLOAD_CHUNK_BYTES pieces, so a slow source is reported while it runs, not
// taken for a hang (upload_report.h; tests/test_cuda_upload_report.py).
static CUresult htod_reported(gpu_weights *w, CUdeviceptr dst, const void *src, size_t len) {
    size_t done = 0;
    char line[192];
    while (done < len) {
        size_t n = upload_chunk(len, done);
        CUresult rc = cu.MemcpyHtoD(dst + done, (const uint8_t *)src + done, n);
        if (rc != 0) return rc;
        done += n;
        if (upload_report_step(&w->rep, n, plat_now(), line, sizeof line))
            fprintf(stderr, "gpu: %s\n", line);
    }
    return 0;
}

static bool binding_add(gpu_weights *w, const model_t *m, gguf_tensor *t) {
    if (!t) return true;
    uint64_t off = tensor_file_off(m, t);
    for (int i = 0; i < w->n_bindings; i++)
        if (w->bindings[i].file_off == off && w->bindings[i].nbytes >= t->nbytes)
            return true;
    if (w->n_bindings == w->cap_bindings) {
        int cap = w->cap_bindings ? w->cap_bindings * 2 : 64;
        gpu_weight_binding *nb = realloc(w->bindings, sizeof(*nb) * (size_t)cap);
        if (!nb) {
            fprintf(stderr, "gpu: CUDA binding-table growth failed while "
                    "adding tensor %s — using CPU\n", t->name);
            return false;
        }
        w->bindings = nb;
        w->cap_bindings = cap;
    }
    CUdeviceptr d = 0;
    CUresult rc = cu.MemAlloc(&d, t->nbytes);
    if (rc != 0) {
        fprintf(stderr, "gpu: CUDA allocation for tensor %s failed: %s — "
                "using CPU\n", t->name, cu_err(rc));
        return false;
    }
    rc = htod_reported(w, d, t->data, t->nbytes);
    if (rc != 0) {
        fprintf(stderr, "gpu: CUDA upload for tensor %s failed: %s — using "
                "CPU\n", t->name, cu_err(rc));
        cu.MemFree(d);
        return false;
    }
    w->bound_bytes += t->nbytes;
    if (getenv("RUNNER_DEBUG_VRAM") && atoi(getenv("RUNNER_DEBUG_VRAM")) >= 2)
        fprintf(stderr, "vram-bind %s %zu\n", t->name, (size_t)t->nbytes);
    w->bindings[w->n_bindings++] = (gpu_weight_binding){ off, t->nbytes, d };
    w->weights_len += t->nbytes;
    return true;
}

typedef struct { gpu_weights *w; const model_t *m; } binding_ctx;

// One binding over the file stretch [lo, hi) of the main mapping: every
// tensor inside it resolves to it (binding_find), one allocation for all.
static bool binding_add_range(uint64_t lo, uint64_t hi, void *ctx) {
    binding_ctx *c = ctx;
    gpu_weights *w = c->w;
    size_t len = (size_t)(hi - lo);
    if (w->n_bindings == w->cap_bindings) {
        int cap = w->cap_bindings ? w->cap_bindings * 2 : 64;
        gpu_weight_binding *nb = realloc(w->bindings, sizeof(*nb) * (size_t)cap);
        if (!nb) {
            fprintf(stderr, "gpu: CUDA binding-table growth failed — using CPU\n");
            return false;
        }
        w->bindings = nb;
        w->cap_bindings = cap;
    }
    CUdeviceptr d = 0;
    CUresult rc = cu.MemAlloc(&d, len);
    if (rc != 0) {
        fprintf(stderr, "gpu: CUDA allocation for a %.1f MB run of weights failed: %s — "
                "using CPU\n", len / 1e6, cu_err(rc));
        return false;
    }
    rc = htod_reported(c->w, d, (const uint8_t *)c->m->gf.map + lo, len);
    if (rc != 0) {
        fprintf(stderr, "gpu: CUDA upload of a weight run failed: %s — using CPU\n",
                cu_err(rc));
        cu.MemFree(d);
        return false;
    }
    w->bound_bytes += len;
    if (getenv("RUNNER_DEBUG_VRAM") && atoi(getenv("RUNNER_DEBUG_VRAM")) >= 2)
        fprintf(stderr, "vram-bind run at %llu %zu\n", (unsigned long long)lo, len);
    w->bindings[w->n_bindings++] = (gpu_weight_binding){ lo, len, d };
    w->weights_len += len;
    return true;
}


static bool binding_find(const gpu_weights *w, const model_t *m,
                         const gguf_tensor *t, CUdeviceptr *base,
                         uint64_t *relative_off) {
    uint64_t off = tensor_file_off(m, t);
    if (!w->bound) {
        // the image starts at img_off: the first device tensor (shared_plan_upload)
        if (off < w->img_off) return false;
        *base = w->weights;
        *relative_off = off - w->img_off;
        return *relative_off <= w->weights_len &&
               t->nbytes <= w->weights_len - (size_t)*relative_off;
    }
    for (int i = 0; i < w->n_bindings; i++) {
        const gpu_weight_binding *b = &w->bindings[i];
        if (off >= b->file_off && off - b->file_off <= b->nbytes &&
            t->nbytes <= b->nbytes - (size_t)(off - b->file_off)) {
            *base = b->device;
            *relative_off = off - b->file_off;
            return true;
        }
    }
    return false;
}


// shared_build()'s split plan, carried between its stages.
typedef struct {
    size_t vram_budget;   // what the upload may spend
    size_t fixed;         // what it spends whatever the split
    size_t headroom;      // the part of `fixed` that is headroom, once
    size_t instances;     // slots that will share it (m->gpu_instances)
    bool mtp_dev;         // a NextN/MTP block rides along
    int G;                // leading layers on the device
    bool full;            // every layer, the output projection too
    size_t used;          // what the split spends of vram_budget
} shared_plan;

// The file identity and the model facts a later match compares against,
// with the entry's own copies of the rope tables.
static bool shared_record_identity(gpu_weights *w, model_t *m, uint64_t fsize,
                                   uint64_t fino, int64_t fmtime) {
    if (m->path) {
        size_t n = strlen(m->path) + 1;
        w->path = malloc(n);
        if (!w->path) {
            fprintf(stderr, "gpu: CUDA shared-weight path allocation failed — "
                    "using CPU\n");
            return false;
        }
        memcpy(w->path, m->path, n);
    }
    w->fsize = fsize; w->fino = fino; w->fmtime = fmtime;
    w->n_maps = gguf_map_count(&m->gf); w->mapped_size = gguf_mapped_size(&m->gf);
    w->n_layer = m->n_layer; w->n_embd = m->n_embd; w->n_head = m->n_head;
    w->n_head_kv = m->n_head_kv; w->head_dim = m->head_dim; w->n_ff = m->n_ff;
    w->n_vocab = m->n_vocab; w->n_ctx = m->n_ctx;
    w->cpu_moe = m->cpu_moe;
    w->cpu_moe_layers = m->cpu_moe_layers;
    w->mtp = m->mtp_ready;
    w->rope_dim = m->rope_dim; w->rope_dim_local = m->rope_dim_local;
    w->kv_q8 = kv_identity(m); w->v_rmsnorm = (int)m->v_rmsnorm;
    w->rope_base = m->rope_base; w->rope_mscale = m->rope_mscale;
    {   // own copies of the rope tables so a later match can compare against
        // them without reaching into a model_t that may since have been freed
        size_t nb = sizeof(float) * (size_t)(m->rope_dim / 2);
        w->rope_inv_freq = malloc(nb);
        if (!w->rope_inv_freq) {
            fprintf(stderr, "gpu: CUDA shared rope-table allocation failed — "
                    "using CPU\n");
            return false;
        }
        memcpy(w->rope_inv_freq, m->rope_inv_freq, nb);
        if (m->rope_inv_freq_local) {
            size_t lb = sizeof(float) * (size_t)(m->rope_dim_local / 2);
            w->rope_inv_freq_local = malloc(lb);
            if (!w->rope_inv_freq_local) {
                fprintf(stderr, "gpu: CUDA shared local-rope-table allocation "
                        "failed — using CPU\n");
                return false;
            }
            memcpy(w->rope_inv_freq_local, m->rope_inv_freq_local, lb);
        }
    }
    return true;
}

static bool shared_open_context(gpu_weights *w) {
    CK(cu.DeviceGet(&w->dev, 0));
    CK(cu.PrimaryCtxRetain(&w->ctx, w->dev));
    CK(cu.CtxSetCurrent(w->ctx));
    return true;
fail:
    return false;
}

// The device budget this upload may spend, and what it spends whatever the
// split: activation scratch, the embeddings, the headroom, the recurrent
// fold, a bound NextN/MTP block.
static bool shared_plan_budget(gpu_weights *w, model_t *m, size_t act_bytes,
                               shared_plan *P) {
    size_t vram_free = 0, vram_total = 0;
    CK(cu.MemGetInfo(&vram_free, &vram_total));
    vram_probe("plan (budget source)");
    size_t vram_budget = vram_free;
    // Unified memory (integrated device): the "VRAM" the driver reports
    // IS system RAM, so the upload budget must also respect what the OS
    // says is actually free plus reclaimable — the two numbers are views
    // of one pool, and taking the CUDA view alone over-promises when
    // the OS is holding memory the driver cannot see committed.
    if (cu_dev_integrated(w->dev)) {
        vram_budget = (size_t)gpu_plan_unified_cap(vram_budget,
                                                   plat_ram_available_bytes());
        fprintf(stderr, "gpu: unified memory — CPU and GPU share one "
                "pool; offload budget capped at available RAM "
                "(%.1f GiB)\n", (double)vram_budget / (1u << 30));
    }
    // The OS video-memory budget for this process, where the OS keeps one:
    // WDDM pages a process past its budget over PCIe, and the driver's free
    // count does not know it. Matched by LUID (two adapters). model_gpu_budget
    // bounds the driver's view by it and sizes the headroom.
    uint64_t os_budget = 0, os_usage = 0, headroom = 0;
    bool os_known = false;
    if (cu.DeviceGetLuid) {
        char luid[8]; unsigned mask = 0;
        if (cu.DeviceGetLuid(luid, &mask, w->dev) == 0)
            os_known = plat_gpu_os_budget((const unsigned char *)luid,
                                          &os_budget, &os_usage);
    }
    vram_budget = (size_t)model_gpu_budget(vram_budget, vram_total, os_known,
                                           os_budget, os_usage,
                                           m->reserve_vram_pct, &headroom);
    if (os_known)
        fprintf(stderr, "gpu: OS video memory budget %.2f GB for this "
                        "process, %.2f GB in use; offload budget %.2f GB, "
                        "headroom %.2f GB\n", os_budget / 1e9,
                os_usage / 1e9, vram_budget / 1e9, headroom / 1e9);
    else
        fprintf(stderr, "gpu: no OS video memory budget published for "
                        "this device; offload budget %.2f GB from the "
                        "driver's free view, headroom %.2f GB\n",
                vram_budget / 1e9, headroom / 1e9);
    // Fixed device overhead whatever the split: the headroom (CUDA context,
    // PTX JIT, allocator slack, the OS reserve) once, and per instance its
    // activation scratch and state (`inst`). The token embedding is not
    // charged: rows are staged from the host, and a tied head is the full
    // split's.
    size_t fixed = (size_t)headroom;
    // act_bytes already carries the recurrent state and scratch
    // (cuda_instance_dims, model_gpu_recurrent_cost): no second charge here.
    size_t inst = act_bytes;
    // the NextN/MTP block rides along with a full split: its attention,
    // mixers and head projections (its experts stay on the host under
    // --cpu-moe, like any bank the plan leaves there), plus its KV rows
    const bool mtp_dev = m->mtp_ready;
    if (mtp_dev) {
        fixed += layer_weight_bytes(m, m->n_layer, moe_on_host(m, m->n_layer), m->cpu_moe);
        if (m->mtp_eh_proj) fixed += m->mtp_eh_proj->nbytes;
        if (m->mtp_head && m->mtp_head != m->output) fixed += m->mtp_head->nbytes;
        if (m->mtp_hc_head_down) fixed += m->mtp_hc_head_down->nbytes;
        if (m->mtp_hc_head_up) fixed += m->mtp_hc_head_up->nbytes;
        inst += model_kv_boundary_bytes_sum(m, m->n_layer + 1) -
                model_kv_boundary_bytes_sum(m, m->n_layer);
        if (m->qwen35) {   // per-row recurrent checkpoints of a verify tile
            int convdim = 2 * m->ssm.state * m->ssm.groups + m->ssm.inner;
            size_t per_row = sizeof(float) * model_rec_count(m) *
                ((size_t)(m->ssm.conv_kernel - 1) * convdim + (size_t)m->ssm.v_heads * m->ssm.state * m->ssm.state);
            inst += per_row * (size_t)(m->spec_batch < 8 ? m->spec_batch : 8);
        }
    }
    // --parallel N: the first slot decides the split and later slots reuse the
    // upload with their own scratch, state and KV. A plan for one slot let it
    // fill the card and the next was refused or served from the CPU with
    // different logits, so each instance is charged here and its KV below.
    P->instances = m->gpu_instances > 1 ? (size_t)m->gpu_instances : 1;
    fixed += inst * P->instances;
    // Tables are shared by slots, and shared_upload_layer_tables builds
    // them for the whole backbone, including layers whose mixer stays host.
    // Auto expert placement can use any bank: reserve its possible bias
    // tables here so the run-based weight refinement cannot erase them.
    for (int l = 0; l < m->n_layer + (m->mtp_ready ? 1 : 0); l++) {
        bool host = moe_on_host(m, l) && m->cpu_moe_layers != CPU_MOE_AUTO;
        gpu_layer_charge c = model_cuda_layer_charge(&m->gf, l, host);
        if (c.tables > SIZE_MAX - fixed) { fprintf(stderr, "gpu: layer table charge overflow\n"); goto fail; }
        fixed += (size_t)c.tables;
    }
    // and the once-made all-ones expert scale table (moe_ones, n_expert f32)
    if (moe_any_on_device(m)) fixed += sizeof(float) * (size_t)m->n_expert;
    fixed += moe_hot_bytes(m);   // RUNNER_MOE_HOT's banks and their routers
    P->vram_budget = vram_budget;
    P->fixed = fixed;
    P->headroom = (size_t)headroom;
    P->mtp_dev = mtp_dev;
    return true;
fail:
    return false;
}

// What a full split adds to the plan: the output head, once (a tied head is
// the embedding table, which nothing else puts on the device).
static size_t plan_full_extra(const model_t *m, const shared_plan *P) {
    (void)P;
    return m->output->nbytes;
}

static uint64_t plan_runs_bytes(model_t *m, int G, int *runs_out);
// How many leading layers go on the device, and which expert banks: the
// gpu_plan_* decisions on this device's measured costs.
static bool shared_plan_split(gpu_weights *w, model_t *m, shared_plan *P) {
    size_t vram_budget = P->vram_budget;
    size_t fixed = P->fixed;
    // --cpu-moe auto plans from all-host every time: a retry after a
    // failed init (model.c) must not inherit the last attempt's banks
    if (m->cpu_moe && m->cpu_moe_layers == CPU_MOE_AUTO) model_moe_place_host(m, -1);
    // decide how many *leading* layers fit — accumulate each layer's weight
    // bytes plus its KV bytes until the budget runs out; the CPU runs the
    // rest (partial offload)
    // The decisions are model.c's gpu_plan_* (the same ones --fit makes
    // from a header); this measures what each layer costs on this device.
    uint64_t *cost = calloc((size_t)m->n_layer, sizeof(uint64_t));
    uint64_t *bank = calloc((size_t)m->n_layer, sizeof(uint64_t));
    bool *is_moe = calloc((size_t)m->n_layer, sizeof(bool));
    if (!cost || !bank || !is_moe) {
        free(cost); free(bank); free(is_moe);
        fprintf(stderr, "gpu: CUDA split plan allocation failed — using CPU\n");
        return false;
    }
    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        size_t wb = layer_weight_bytes(m, l, moe_on_host(m, l), m->cpu_moe);
        // KV bytes mirror the allocation (model_kv_boundary_bytes): a ringed
        // sliding layer costs its ring rows, a shared-KV layer nothing beyond
        // its owner's rows, and q8_0 rows about 53% of fp16.
        size_t kv = model_kv_boundary_bytes_sum(m, l + 1) -
                    model_kv_boundary_bytes_sum(m, l);
        cost[l] = wb + kv * P->instances;
        // a bank the device cannot run is no candidate for the auto fit
        is_moe[l] = ly->is_moe && moe_bank_device_ok(m, l);
        bank[l] = ly->is_moe ? expert_plan_bytes(ly, ly->n_expert, true) : 0;
    }
    uint64_t used = fixed;
    int G = gpu_plan_prefix(cost, m->n_layer, 0, vram_budget, &used);
    // Partial expert offload: the pass above placed attention (and any dense
    // FFN) for the layers that fit; what budget survives buys whole expert
    // banks, shallowest first. Explicit counts are honoured as given and were
    // accounted above.
    if (m->cpu_moe && m->cpu_moe_layers == CPU_MOE_AUTO) {
        // Hold back what a full split still owes. The forward computes logits
        // on the device from the layer count alone, so a bank that ate the
        // last of the budget would leave every layer on the device and
        // output.weight never uploaded. The head also runs every token, a bank
        // only when routed.
        size_t owed = G == m->n_layer ? plan_full_extra(m, P) : 0;
        // A retry after an out-of-memory init holds back the deepest
        // banks: the plan is an estimate (on a 24 GB MIG slice it
        // undercounted the device by ~2 GB, so a smaller KV that bought a
        // 20th bank failed its recurrent-state allocation and the whole
        // run fell back to the CPU at half speed, 2026-10-08).
        int placed = gpu_plan_expert_banks(is_moe, bank, G, owed, vram_budget,
                                           &used, m->moe_host, m->moe_auto_trim,
                                           m->moe_auto_placed);
        // The greedy pass costs every tensor as its own rounded allocation, but
        // the upload joins tensors into runs and rounds once a run. Re-cost the
        // weights by the runs this placement makes and keep placing banks while
        // that figure and the head fit. A retry after a failed init keeps the
        // greedy plan.
        if (m->moe_auto_trim == 0) {
            uint64_t kvsum = 0;
            for (int l = 0; l < G; l++)
                kvsum += (uint64_t)(model_kv_boundary_bytes_sum(m, l + 1) -
                                    model_kv_boundary_bytes_sum(m, l)) * P->instances;
            uint64_t greedy_w = used > fixed + kvsum ? used - fixed - kvsum : 0;
            int runs = 0;
            uint64_t exact_w = plan_runs_bytes(m, G, &runs);
            if (exact_w && exact_w < greedy_w) {
                used = used - greedy_w + exact_w;
                for (int l = 0; l < G; l++) {
                    if (!is_moe[l] || !m->moe_host[l]) continue;
                    m->moe_host[l] = false;
                    int r2 = 0;
                    uint64_t w2 = plan_runs_bytes(m, G, &r2);
                    uint64_t u2 = used - exact_w + w2;
                    if (!w2 || u2 > vram_budget || owed > vram_budget - u2) {
                        m->moe_host[l] = true;
                        break;
                    }
                    used = u2; exact_w = w2; runs = r2; placed++;
                }
            }
            if (vram_debug_on())
                fprintf(stderr, "vram-plan: headroom+instance %.3f GB, kv %.3f GB, weights %.3f GB "
                                "greedy, %.3f GB by %d runs, banks %d, head owed %.3f GB, used %.3f GB "
                                "(bytes: weights %llu by runs, head %llu)\n",
                        fixed / 1e9, kvsum / 1e9, greedy_w / 1e9, exact_w / 1e9, runs, placed,
                        owed / 1e9, used / 1e9, (unsigned long long)exact_w, (unsigned long long)owed);
            m->moe_auto_placed = placed;
        }
    }
    free(bank); free(is_moe);
    // A full split also needs token_embd and the output weights resident. "The
    // device owns every layer" and "the device owns the output projection"
    // must agree (the forward reads the first, the upload honours the second):
    // if the reserve cannot be met the last layer goes back to the CPU.
    bool gave_back = false;
    bool full = gpu_plan_full_split(&G, m->n_layer, &used, cost, plan_full_extra(m, P),
                                    vram_budget, &gave_back);
    if (gave_back) {
        fprintf(stderr, "gpu: output weights do not fit alongside the "
                        "layer split — running the last layer on the CPU "
                        "so the boundary is consistent\n");
    }
    // An explicit --gpu-layers overrides the fit, below or above it. Above it,
    // say so: a discrete card cannot hold the excess, so the allocation fails
    // or a Windows driver pages it to system memory and runs several times
    // slower without an error (2.9 to 3.6 tok/s against 5.6 on one 8 GB card).
    int natural_G = G;
    if (m->gpu_layers_override > 0) {
        G = m->gpu_layers_override;
        if (G > m->n_layer) G = m->n_layer;
        full = (G == m->n_layer);
        // Past the fitted split, charge the forced layers: otherwise the split
        // line and P->used (which sizes the prompt window) keep the fitted
        // plan's bytes and the window takes memory a forced layer holds. Below
        // the fitted split the fitted figure stands, as an upper bound.
        for (int l = natural_G; l < G; l++) used += cost[l];
        if (G > natural_G)
            fprintf(stderr, "gpu: --gpu-layers %d places %d layers past the %d the "
                            "offload budget holds (%.2f GB); the excess may fail to "
                            "allocate or be paged to system memory, which runs far "
                            "slower. Omit --gpu-layers to use the fitted split.\n",
                    m->gpu_layers_override, G - natural_G, natural_G,
                    (double)vram_budget / 1e9);
    }
    free(cost);
    // Composition warning. Under tensor-role placement the expert banks fill
    // VRAM, so capping the attention split below what fits moves attention to
    // the CPU and frees almost nothing (10.3 tok/s against 14.6 for the layer
    // split alone on a 12 GB card). Said, not refused: the pair stays legal.
    if (m->cpu_moe && m->gpu_layers_override > 0 && G < natural_G)
        fprintf(stderr,
                "gpu: --gpu-layers %d caps the attention split below the "
                "%d layers that fit; under --cpu-moe that mostly moves "
                "attention to the CPU without freeing useful VRAM. Prefer "
                "--cpu-moe N (or auto) to spend the headroom on expert "
                "banks.\n", G, natural_G);
    // The discovery the first outside install could not make: all-host
    // placement left 8.8 GB of a 12 GB card idle and nothing said so.
    if (m->cpu_moe && m->cpu_moe_layers == CPU_MOE_ALL) {
        // Count what the auto fit would actually place, by the same greedy
        // rule, so the advice cannot promise more than it can deliver.
        size_t would_use = used;
        int would_place = 0, moe_layers = 0;
        for (int l = 0; l < G; l++) {
            if (!m->layers[l].is_moe) continue;
            moe_layers++;
            if (!moe_bank_device_ok(m, l)) continue;
            size_t eb = model_layer_expert_bytes(&m->layers[l], m->layers[l].n_expert);
            if (would_use + eb > vram_budget) continue;
            would_use += eb;
            would_place++;
        }
        if (would_place > 0)
            fprintf(stderr,
                    "gpu: %.2f GB of the budget is unused — `--cpu-moe "
                    "auto` would keep %d of %d expert layers on the GPU.\n",
                    (vram_budget - used) / 1e9, would_place, moe_layers);
    }
    if (G == 0) {
        fprintf(stderr,
                "gpu: not even one layer fits %.1f GB %s VRAM — using CPU\n",
                vram_budget / 1e9, m->reserve_vram_pct > 0 ? "reserved" : "free");
        return false;
    }
    int moe_dev = 0, moe_tot = 0;
    for (int l = 0; l < m->n_layer; l++) {
        if (!m->layers[l].is_moe) continue;
        moe_tot++;
        if (!moe_on_host(m, l)) moe_dev++;
    }
    fprintf(stderr, "gpu-split: budget=%.2fGB fixed=%.2fGB G=%d/%d full=%d used=%.2fGB\n", vram_budget/1e9, fixed/1e9, G, m->n_layer, (int)full, used/1e9);
    // the figures --fit predicts, exactly (tests/test_cuda_fit_hybrid.py)
    if (getenv("RUNNER_DEBUG_VRAM"))
        fprintf(stderr, "gpu-split: (bytes: fixed %zu, headroom %zu)\n", fixed, P->headroom);
    // and the device bytes the chosen split holds (R3.58), beside --fit's
    if (getenv("RUNNER_DEBUG_VRAM")) {
        size_t wsum = full ? m->output->nbytes : 0;
        for (int l = 0; l < G; l++)
            wsum += layer_weight_bytes(m, l, moe_on_host(m, l), m->cpu_moe);
        fprintf(stderr, "gpu-split: (bytes: used %llu, weights %zu, layers %d)\n",
                (unsigned long long)used, wsum, G);
    }
    if (P->instances > 1)
        fprintf(stderr, "gpu-split: planned for %zu slots sharing the weights, "
                "each with its own KV cache and scratch\n", P->instances);
    if (m->cpu_moe && moe_tot)
        fprintf(stderr, "gpu-split: experts %d/%d layers on GPU, %d on host%s\n",
                moe_dev, moe_tot, moe_tot - moe_dev,
                m->cpu_moe_layers == CPU_MOE_AUTO ? " (auto fit)" : "");
    w->gpu_layers = full ? m->n_layer : G;
    P->G = G;
    P->full = full;
    // the full-split test's own charge
    P->used = used + (full ? plan_full_extra(m, P) : 0);
    return true;
}

// One contiguous prefix of the file, or one binding per tensor.
// Every tensor the device reads through its weights for a split of G leading
// layers (on a full split also the head, the hyper-connection head and the
// NextN/MTP block). Nothing else is read there: norms, biases, rope factors
// and scales are their own buffers, the token embedding is staged from the
// host, and the per-layer embedding table runs there.
static bool device_tensors_each(model_t *m, int G, bool full, bool mtp_dev,
                                bool (*fn)(gguf_tensor *, void *), void *ctx) {
    for (int l = 0; l < G; l++) {
        layer_t *ly = &m->layers[l];
        gguf_tensor *mix[LAYER_MIXER_MAX];
        int nmix = layer_mixer_tensors(ly, mix);
        for (int i = 0; i < nmix; i++)
            if (!fn(mix[i], ctx)) return false;
        // A dense FFN, and under a partial expert offload a device-
        // resident bank, upload as ordinary bindings, so the same
        // offset-resolved matvec path serves them; a host bank stays home.
        if (!(ly->is_moe && moe_on_host(m, l)) &&
            !layer_ffn_tensors_each(ly, ly->n_expert, fn, ctx))
            return false;
    }
    if (!full) return true;
    gguf_tensor *head[] = { m->output, m->hyper_conn ? m->hc_head_down : NULL,
                            m->hyper_conn ? m->hc_head_up : NULL };
    for (size_t i = 0; i < sizeof head / sizeof *head; i++)
        if (head[i] && !fn(head[i], ctx)) return false;
    if (!mtp_dev) return true;
    layer_t *ly = &m->layers[m->n_layer];
    gguf_tensor *mt[] = { ly->wq, ly->wk, ly->wv, ly->wo, ly->hc_attn_down, ly->hc_attn_up,
                          ly->hc_attn_inject, ly->hc_ffn_down, ly->hc_ffn_up,
                          ly->hc_ffn_inject, m->mtp_eh_proj, m->mtp_hc_head_down,
                          m->mtp_hc_head_up, ly->wqkv, ly->wq_gate, m->mtp_head };
    for (size_t i = 0; i < sizeof mt / sizeof *mt; i++)
        if (mt[i] && !fn(mt[i], ctx)) return false;
    if (!ly->is_moe) {
        gguf_tensor *d[] = { ly->w_gate, ly->w_up, ly->w_down };
        for (int i = 0; i < 3; i++)
            if (d[i] && !fn(d[i], ctx)) return false;
    } else if (!moe_on_host(m, m->n_layer)) {
        gguf_tensor *e[] = { ly->ffn_gate_inp, ly->ffn_gate_exps, ly->ffn_up_exps,
                             ly->ffn_down_exps, ly->w_gate_shexp, ly->w_up_shexp,
                             ly->w_down_shexp, ly->ffn_gate_inp_shexp };
        for (size_t i = 0; i < sizeof e / sizeof *e; i++)
            if (e[i] && !fn(e[i], ctx)) return false;
    } else {
        if ((ly->ffn_gate_inp && !fn(ly->ffn_gate_inp, ctx)) ||
            (ly->ffn_gate_inp_shexp && !fn(ly->ffn_gate_inp_shexp, ctx)))
            return false;
    }
    return true;
}

// The device tensors, collected (device_tensors_each), and their runs in the
// main mapping: tensors that lie together in the file, gaps of up to
// GPU_PLAN_GRAN (the allocator's granularity, so a separate allocation would
// waste as much) riding along. A tensor in another mapping (a split GGUF's
// other part, an --mtp-file head) has no place in a run.
typedef struct {
    const model_t *m;
    gguf_tensor **t;
    int n, cap;
    bool oom;
} tensor_list;

static bool list_add(gguf_tensor *t, void *ctx) {
    tensor_list *L = ctx;
    if (L->n == L->cap) {
        int cap = L->cap ? 2 * L->cap : 256;
        gguf_tensor **nt = realloc(L->t, sizeof(*nt) * (size_t)cap);
        if (!nt) { L->oom = true; return false; }
        L->t = nt;
        L->cap = cap;
    }
    L->t[L->n++] = t;
    return true;
}

static bool in_main_map(const model_t *m, const gguf_tensor *t) {
    const uint8_t *b = (const uint8_t *)m->gf.map, *d = (const uint8_t *)t->data;
    return d >= b && d < b + m->gf.map_size;
}

static int cmp_tensor_data(const void *a, const void *b) {
    const uint8_t *x = (const uint8_t *)(*(gguf_tensor *const *)a)->data;
    const uint8_t *y = (const uint8_t *)(*(gguf_tensor *const *)b)->data;
    return x < y ? -1 : x > y;
}

// The main-mapping tensors of L sorted by file offset (the others moved to
// the end and counted in *n_out), then fn on each of their runs [lo, hi)
// (gpu_plan_runs). Returns the number of runs, or -1 when fn refused one or
// the scratch could not be had.
static int tensor_runs(const model_t *m, tensor_list *L, int *n_out,
                       bool (*fn)(uint64_t lo, uint64_t hi, void *), void *ctx) {
    int k = 0;
    for (int i = 0; i < L->n; i++)
        if (in_main_map(m, L->t[i])) { gguf_tensor *t = L->t[i]; L->t[i] = L->t[k]; L->t[k++] = t; }
    *n_out = L->n - k;
    qsort(L->t, (size_t)k, sizeof *L->t, cmp_tensor_data);
    uint64_t *buf = malloc(sizeof(uint64_t) * 4 * (size_t)(k ? k : 1));
    if (!buf) return -1;
    uint64_t *off = buf, *len = buf + k, *lo = buf + 2 * k, *hi = buf + 3 * k;
    for (int i = 0; i < k; i++) { off[i] = tensor_file_off(m, L->t[i]); len[i] = L->t[i]->nbytes; }
    int runs = gpu_plan_runs(off, len, k, GPU_PLAN_GRAN, lo, hi);
    for (int r = 0; r < runs; r++)
        if (fn && !fn(lo[r], hi[r], ctx)) { free(buf); return -1; }
    free(buf);
    return runs;
}

// The device bytes of the weights for the first G layers as the upload will
// allocate them (R4.26.18): their tensors' runs (tensor_runs), each one
// binding rounded up to the driver's granularity, and a tensor outside the
// main mapping a binding of its own. The head is not included: a full split
// owes it separately (plan_full_extra). 0 when the list cannot be built.
static bool runs_round_add(uint64_t lo, uint64_t hi, void *ctx) {
    *(uint64_t *)ctx += gpu_plan_binding_bytes(hi - lo, true);
    return true;
}
static uint64_t plan_runs_bytes(model_t *m, int G, int *runs_out) {
    tensor_list L = { m, NULL, 0, 0, false };
    device_tensors_each(m, G, false, false, list_add, &L);
    uint64_t total = 0;
    int outside = 0;
    int runs = L.oom ? -1 : tensor_runs(m, &L, &outside, runs_round_add, &total);
    if (runs >= 0)
        for (int i = L.n - outside; i < L.n; i++)
            total += gpu_plan_binding_bytes(L.t[i]->nbytes, true);
    free(L.t);
    if (runs_out) *runs_out = runs < 0 ? 0 : runs + outside;
    return runs < 0 ? 0 : total;
}
typedef struct { uint64_t lo, hi; } span_ctx;
static bool span_take(uint64_t lo, uint64_t hi, void *ctx) {
    span_ctx *c = ctx;
    c->lo = lo; c->hi = hi;
    return true;
}

static void shared_plan_upload(gpu_weights *w, model_t *m, const shared_plan *P) {
    int G = P->G;
    bool full = P->full;
    bool mtp_dev = P->mtp_dev;
    // The image: one contiguous stretch of the file, from the first tensor the
    // device reads to the end of the last (device_tensors_each), so offsets
    // inside it stay valid relative to img_off. Where the device tensors do not
    // lie together in the file they upload as runs (shared_upload_weights).
    tensor_list L = { m, NULL, 0, 0, false };
    device_tensors_each(m, G, full, mtp_dev, list_add, &L);
    int outside = 0;
    span_ctx sp = { 0, 0 };
    int runs = L.oom ? -1 : tensor_runs(m, &L, &outside, span_take, &sp);
    free(L.t);
    bool one = runs == 1 && outside == 0;
    if (!one && !m->cpu_moe && runs > 1)
        fprintf(stderr, "gpu: the device's tensors (%d layers%s) lie in %d stretches of "
                        "the file; uploading them as runs\n", G, full ? " and the head" : "", runs);
    w->bound = m->cpu_moe || !one;
    w->weights_len = w->bound ? 0 : (size_t)(sp.hi - sp.lo);
    w->img_off = w->bound ? 0 : (size_t)sp.lo;
}

// The embedded PTX module and every kernel the encoders launch.
static bool gpu_sm80_on(void);
static bool gpu_tc2_on(void);
static bool tc2_forced(void);
static bool tc2_can_run(const gpu_t *g);
// The compute_80 module loads on a compute capability 8.0+ device unless TC2
// is off or RUNNER_CUDA_SM80=0. The window's charge for TC2's activations
// (shared_build) asks the same question before the load.
static bool sm80_module_wanted(const gpu_weights *w) {
    int ccmaj = 0;
    return gpu_tc2_on() && gpu_sm80_on() && cu.DeviceGetAttribute &&
           cu.DeviceGetAttribute(&ccmaj, CU_DEV_ATTR_CC_MAJOR, w->dev) == 0 && ccmaj >= 8;
}
// R10.29: on the first load after a new card or driver the driver JIT-compiles the embedded PTX
// (measured: 33.4 s on an RTX 5070 for a 370 KB model, then 0.40 s from the driver's cache of
// 47.6 MB), and the process sat there without a word. A load still running after 2 s says why on
// stderr, once per process: a warm load ends first and says nothing. stdout and receipts are untouched.
static atomic_bool g_jit_noted;
static CUresult module_load_noted(CUmodule *mod, const void *img) {
    plat_slow_note *n = atomic_load(&g_jit_noted) ? NULL :
        plat_slow_note_begin(stderr, "gpu: compiling the GPU kernels for this card and driver; this "
                             "happens once per card and driver, later loads take them from the "
                             "driver's cache", 2.0);
    CUresult r = cu.ModuleLoadData(mod, img);
    if (plat_slow_note_end(n)) atomic_store(&g_jit_noted, true);
    return r;
}
static bool shared_load_module(gpu_weights *w) {
    vram_probe("before module load");
    CK(module_load_noted(&w->mod, k_ptx_src));
    vram_probe("after module load");
    struct { CUfunction *f; const char *name; } fns[] = {
        { &w->f_rmsnorm,    "k_rmsnorm" },   { &w->f_qknorm, "k_qknorm" },
        { &w->f_rmsnorm_grouped, "k_rmsnorm_grouped" }, { &w->f_mtp_cat, "k_mtp_cat" },
        { &w->f_hc_gate_mean, "k_hc_gate_mean" }, { &w->f_hc_combine, "k_hc_combine" },
        { &w->f_silu_scale, "k_silu_scale" },
        { &w->f_mv_bf16_splitk, "k_mv_bf16_splitk" }, { &w->f_splitk_reduce, "k_splitk_reduce" },
        { &w->f_axpy_sigmoid, "k_axpy_sigmoid" },
        { &w->f_rope,       "k_rope" },      { &w->f_store,  "k_store_kv" },
        { &w->f_mla_q,      "k_mla_q" },     { &w->f_mla_kv, "k_mla_kv" },
        { &w->f_mla_pack_v, "k_mla_pack_v" },
        { &w->f_mla_q_absorb, "k_mla_q_absorb" }, { &w->f_mla_kv_absorb, "k_mla_kv_absorb" },
        { &w->f_mla_v_absorb, "k_mla_v_absorb" },
        { &w->f_attn,       "k_attn" },      { &w->f_silu,   "k_silu_mul" },
        { &w->f_attn_dec,   "k_attn_dec" },  { &w->f_attn_merge, "k_attn_merge" },
        { &w->f_gelu,       "k_gelu_mul" },  { &w->f_add,    "k_add" },
        { &w->f_gelu_erf,   "k_gelu_erf_mul" },
        { &w->f_head_gate,  "k_head_attn_gate" },
        { &w->f_swiglu,     "k_swiglu_oai_mul" },
        { &w->f_xielu,      "k_xielu" },
        { &w->f_scale,      "k_scale" },
        { &w->f_q35_split,  "k_q35_split_q" },
        { &w->f_q35_gate,   "k_q35_attn_gate" },
        { &w->f_q35_conv,   "k_q35_conv" },
        { &w->f_q35_delta,  "k_q35_delta" },
        { &w->f_mamba_conv, "k_mamba2_conv" },
        { &w->f_mamba_ssd,  "k_mamba2_ssd" },
        { &w->f_mamba_gate_norm, "k_mamba2_gate_norm" },
        { &w->f_relu2,      "k_relu2" },
        { &w->f_colabsmax,  "k_colabsmax" },
        { &w->f_mv[T_F32],  "k_mv_f32" },    { &w->f_mv[T_F16],  "k_mv_f16" },
        { &w->f_mv[T_Q8_0], "k_mv_q8_0" },   { &w->f_mv[T_Q4_0], "k_mv_q4_0" },
        { &w->f_mv[T_Q2_0], "k_mv_q2_0" },
        { &w->f_mv[T_Q4_1], "k_mv_q4_1" },   { &w->f_mv[T_Q5_0], "k_mv_q5_0" },
        { &w->f_mv[T_Q5_1], "k_mv_q5_1" },   { &w->f_mv[T_Q4_K], "k_mv_q4_K" },
        { &w->f_mv[T_Q5_K], "k_mv_q5_K" },   { &w->f_mv[T_Q6_K], "k_mv_q6_K" },
        { &w->f_mv[T_Q3_K], "k_mv_q3_K" },
        { &w->f_lora_a,      "k_lora_a" },
        { &w->f_lora_b,      "k_lora_b" },
        { &w->f_mvcanon[T_F32],  "k_mvcanon_f32" },
        { &w->f_mvcanon[T_F16],  "k_mvcanon_f16" },
        { &w->f_mvcanon[T_Q8_0], "k_mvcanon_q8_0" },
        { &w->f_mvcanon[T_Q4_0], "k_mvcanon_q4_0" },
        { &w->f_mvcanon[T_Q4_K], "k_mvcanon_q4_K" },
        { &w->f_mvcanon[T_Q5_K], "k_mvcanon_q5_K" },
        { &w->f_mvcanon[T_Q6_K], "k_mvcanon_q6_K" },
        { &w->f_mvt[T_F32],  "k_mvt_f32" },
        { &w->f_mvt[T_F16],  "k_mvt_f16" },
        { &w->f_mvt[T_BF16], "k_mvt_bf16" },
        { &w->f_mvt[T_Q8_0], "k_mvt_q8_0" },
        { &w->f_mvt[T_Q4_K], "k_mvt_q4_K" },
        { &w->f_mvt[T_Q4_0], "k_mvt_q4_0" },
        { &w->f_mvt[T_Q6_K], "k_mvt_q6_K" },
        { &w->f_mv[T_Q2_K], "k_mv_q2_K" },
        { &w->f_mv[T_BF16], "k_mv_bf16" },
        { &w->f_mv[T_IQ4_NL], "k_mv_iq4_nl" }, { &w->f_mv[T_IQ4_XS], "k_mv_iq4_xs" },
        { &w->f_mv[T_MXFP4], "k_mv_mxfp4" },
        { &w->f_mv[T_NVFP4], "k_mv_nvfp4" },
        { &w->f_mv[T_IQ2_XXS], "k_mv_iq2_xxs" }, { &w->f_mv[T_IQ2_XS], "k_mv_iq2_xs" },
        { &w->f_mv[T_IQ2_S], "k_mv_iq2_s" },     { &w->f_mv[T_IQ3_XXS], "k_mv_iq3_xxs" },
        { &w->f_mv[T_IQ3_S], "k_mv_iq3_s" },     { &w->f_mv[T_IQ1_S], "k_mv_iq1_s" },
        { &w->f_mv[T_IQ1_M], "k_mv_iq1_m" },
        // the tile twins of the float batch-1 kernels (mvb_is_mv_twin); a
        // quantized type's prompt takes its batch-1 kernel's 16-wide twins or
        // the batch-1 kernel per column (R18.8.11 retired their tiles)
        { &w->f_mvb[T_F32],  "k_mv_f32_b" },  { &w->f_mvb[T_F16],  "k_mv_f16_b" },
        { &w->f_mvb[T_BF16], "k_mv_bf16_b" },
        // decode coalesced GEMV variants (batch==1 fast path for these formats)
        { &w->f_gemv[T_Q8_0], "k_gemv_q8_0" },
        { &w->f_gemv[T_Q4_0], "k_gemv_q4_0" },
        { &w->f_gemv[T_Q4_K], "k_gemv_q4_K" },
        { &w->f_gemv[T_Q5_K], "k_gemv_q5_K" },
        { &w->f_gemv[T_Q3_K], "k_gemv_q3_K" },
        { &w->f_gemv[T_IQ4_XS], "k_gemv_iq4_xs" },
        { &w->f_gemv[T_Q6_K], "k_gemv_q6_K" },
        // cross-sequence decode microbatch: per-column position and KV,
        // plus the multi-column twins of the decode GEMVs above
        { &w->f_rope_seq,      "k_rope_seq" },
        { &w->f_store_seq,     "k_store_kv_seq" },
        { &w->f_attn_dec_seq,  "k_attn_dec_seq" },
        // twins of f_gemv: same per-column arithmetic, x staged in smem,
        // one instantiation per microbatch width class
        { &w->f_gemvb[BW_4][T_BF16], "k_gemvb_bf16_x4" },
        { &w->f_gemvb[BW_8][T_BF16], "k_gemvb_bf16_x8" },
        { &w->f_gemvb[BW_4][T_Q8_0], "k_gemvb_q8_0_x4" },
        { &w->f_gemvb[BW_4][T_Q4_0], "k_gemvb_q4_0_x4" },
        { &w->f_gemvb[BW_4][T_Q4_K], "k_gemvb_q4_K_x4" },
        { &w->f_gemvb[BW_4][T_Q5_K], "k_gemvb_q5_K_x4" },
        { &w->f_gemvb[BW_4][T_Q6_K], "k_gemvb_q6_K_x4" },
        { &w->f_gemvb[BW_8][T_Q8_0], "k_gemvb_q8_0_x8" },
        { &w->f_gemvb[BW_8][T_Q4_0], "k_gemvb_q4_0_x8" },
        { &w->f_gemvb[BW_8][T_Q4_K], "k_gemvb_q4_K_x8" },
        { &w->f_gemvb[BW_8][T_Q5_K], "k_gemvb_q5_K_x8" },
        { &w->f_gemvb[BW_8][T_Q6_K], "k_gemvb_q6_K_x8" },
        { &w->f_gemvb[BW_4][T_Q3_K], "k_gemvb_q3_K_x4" },
        { &w->f_gemvb[BW_8][T_Q3_K], "k_gemvb_q3_K_x8" },
        { &w->f_gemvb[BW_4][T_MXFP4], "k_gemvb_mxfp4_x4" },
        { &w->f_gemvb[BW_8][T_MXFP4], "k_gemvb_mxfp4_x8" },
        { &w->f_gemvb[BW_16][T_BF16], "k_gemvb_bf16_x16" },
        { &w->f_gemvb[BW_16][T_Q8_0], "k_gemvb_q8_0_x16" },
        { &w->f_gemvb[BW_16][T_Q4_0], "k_gemvb_q4_0_x16" },
        { &w->f_gemvb[BW_16][T_Q4_K], "k_gemvb_q4_K_x16" },
        { &w->f_gemvb[BW_16][T_Q5_K], "k_gemvb_q5_K_x16" },
        { &w->f_gemvb[BW_16][T_Q6_K], "k_gemvb_q6_K_x16" },
        { &w->f_gemvb[BW_16][T_Q3_K], "k_gemvb_q3_K_x16" },
        { &w->f_gemvb[BW_16][T_MXFP4], "k_gemvb_mxfp4_x16" },
        { &w->f_gemvb[BW_16][T_IQ2_XXS], "k_gemvb_iq2_xxs_x16" },
        { &w->f_gemvb[BW_16][T_IQ2_XS], "k_gemvb_iq2_xs_x16" },
        { &w->f_gemvb[BW_16][T_IQ2_S], "k_gemvb_iq2_s_x16" },
        { &w->f_gemvb[BW_16][T_IQ3_XXS], "k_gemvb_iq3_xxs_x16" },
        { &w->f_gemvb[BW_16][T_IQ3_S], "k_gemvb_iq3_s_x16" },
        { &w->f_gemvb[BW_16][T_IQ4_XS], "k_gemvb_iq4_xs_x16" },
        { &w->f_moe_q4_0_mvb[BW_4], "k_gemvb_q4_0_mv_x4" },
        { &w->f_moe_q4_0_mvb[BW_8], "k_gemvb_q4_0_mv_x8" },
        { &w->f_gemvb[BW_4][T_IQ2_XXS], "k_gemvb_iq2_xxs_x4" },
        { &w->f_gemvb[BW_8][T_IQ2_XXS], "k_gemvb_iq2_xxs_x8" },
        { &w->f_gemvb[BW_4][T_IQ2_XS], "k_gemvb_iq2_xs_x4" },
        { &w->f_gemvb[BW_8][T_IQ2_XS], "k_gemvb_iq2_xs_x8" },
        { &w->f_gemvb[BW_4][T_IQ2_S], "k_gemvb_iq2_s_x4" },
        { &w->f_gemvb[BW_8][T_IQ2_S], "k_gemvb_iq2_s_x8" },
        { &w->f_gemvb[BW_4][T_IQ3_XXS], "k_gemvb_iq3_xxs_x4" },
        { &w->f_gemvb[BW_8][T_IQ3_XXS], "k_gemvb_iq3_xxs_x8" },
        { &w->f_gemvb[BW_4][T_IQ3_S], "k_gemvb_iq3_s_x4" },
        { &w->f_gemvb[BW_8][T_IQ3_S], "k_gemvb_iq3_s_x8" },
        { &w->f_gemvb[BW_4][T_IQ4_XS], "k_gemvb_iq4_xs_x4" },
        { &w->f_gemvb[BW_8][T_IQ4_XS], "k_gemvb_iq4_xs_x8" },
        { &w->f_gemvx[5][T_Q4_K], "k_gemvx_q4_K_x5" },
        { &w->f_gemvx[6][T_Q4_K], "k_gemvx_q4_K_x6" },
        { &w->f_gemvx[7][T_Q4_K], "k_gemvx_q4_K_x7" },
        { &w->f_gemvx[5][T_Q5_K], "k_gemvx_q5_K_x5" },
        { &w->f_gemvx[6][T_Q5_K], "k_gemvx_q5_K_x6" },
        { &w->f_gemvx[7][T_Q5_K], "k_gemvx_q5_K_x7" },
        { &w->f_gemvx[5][T_Q6_K], "k_gemvx_q6_K_x5" },
        { &w->f_gemvx[6][T_Q6_K], "k_gemvx_q6_K_x6" },
        { &w->f_gemvx[7][T_Q6_K], "k_gemvx_q6_K_x7" },
        { &w->f_gemvx[2][T_IQ2_XXS], "k_gemvx_iq2_xxs_x2" },
        { &w->f_gemvx[3][T_IQ2_XXS], "k_gemvx_iq2_xxs_x3" },
        { &w->f_gemvx[4][T_IQ2_XXS], "k_gemvb_iq2_xxs_x4" },
        { &w->f_gemvx[5][T_IQ2_XXS], "k_gemvx_iq2_xxs_x5" },
        { &w->f_gemvx[6][T_IQ2_XXS], "k_gemvx_iq2_xxs_x6" },
        { &w->f_gemvx[7][T_IQ2_XXS], "k_gemvx_iq2_xxs_x7" },
        { &w->f_gemvx[8][T_IQ2_XXS], "k_gemvb_iq2_xxs_x8" },
        { &w->f_gemvx[2][T_IQ2_XS], "k_gemvx_iq2_xs_x2" },
        { &w->f_gemvx[3][T_IQ2_XS], "k_gemvx_iq2_xs_x3" },
        { &w->f_gemvx[4][T_IQ2_XS], "k_gemvb_iq2_xs_x4" },
        { &w->f_gemvx[5][T_IQ2_XS], "k_gemvx_iq2_xs_x5" },
        { &w->f_gemvx[6][T_IQ2_XS], "k_gemvx_iq2_xs_x6" },
        { &w->f_gemvx[7][T_IQ2_XS], "k_gemvx_iq2_xs_x7" },
        { &w->f_gemvx[8][T_IQ2_XS], "k_gemvb_iq2_xs_x8" },
        { &w->f_gemvx[2][T_IQ2_S], "k_gemvx_iq2_s_x2" },
        { &w->f_gemvx[3][T_IQ2_S], "k_gemvx_iq2_s_x3" },
        { &w->f_gemvx[4][T_IQ2_S], "k_gemvb_iq2_s_x4" },
        { &w->f_gemvx[5][T_IQ2_S], "k_gemvx_iq2_s_x5" },
        { &w->f_gemvx[6][T_IQ2_S], "k_gemvx_iq2_s_x6" },
        { &w->f_gemvx[7][T_IQ2_S], "k_gemvx_iq2_s_x7" },
        { &w->f_gemvx[8][T_IQ2_S], "k_gemvb_iq2_s_x8" },
        { &w->f_gemvx[2][T_IQ3_XXS], "k_gemvx_iq3_xxs_x2" },
        { &w->f_gemvx[3][T_IQ3_XXS], "k_gemvx_iq3_xxs_x3" },
        { &w->f_gemvx[4][T_IQ3_XXS], "k_gemvb_iq3_xxs_x4" },
        { &w->f_gemvx[5][T_IQ3_XXS], "k_gemvx_iq3_xxs_x5" },
        { &w->f_gemvx[6][T_IQ3_XXS], "k_gemvx_iq3_xxs_x6" },
        { &w->f_gemvx[7][T_IQ3_XXS], "k_gemvx_iq3_xxs_x7" },
        { &w->f_gemvx[8][T_IQ3_XXS], "k_gemvb_iq3_xxs_x8" },
        { &w->f_gemvx[2][T_IQ3_S], "k_gemvx_iq3_s_x2" },
        { &w->f_gemvx[3][T_IQ3_S], "k_gemvx_iq3_s_x3" },
        { &w->f_gemvx[4][T_IQ3_S], "k_gemvb_iq3_s_x4" },
        { &w->f_gemvx[5][T_IQ3_S], "k_gemvx_iq3_s_x5" },
        { &w->f_gemvx[6][T_IQ3_S], "k_gemvx_iq3_s_x6" },
        { &w->f_gemvx[7][T_IQ3_S], "k_gemvx_iq3_s_x7" },
        { &w->f_gemvx[8][T_IQ3_S], "k_gemvb_iq3_s_x8" },
        // sparse-MoE device routing + indirect expert matvecs
        { &w->f_moe_route,        "k_moe_route" },
        { &w->f_moe_route_replay, "k_moe_route_replay" },
        { &w->f_moe_actmul,       "k_moe_actmul" },
        { &w->f_moe_sum,          "k_moe_sum" },
        { &w->f_moe_sum_tile,     "k_moe_sum_tile" },
        { &w->f_moe_place,        "k_moe_place" },
        { &w->f_moe_mv[T_F32],    "k_moe_mv_f32" },
        { &w->f_moe_mv[T_F16],    "k_moe_mv_f16" },
        { &w->f_moe_mv[T_Q8_0],   "k_moe_mv_q8_0" },
        { &w->f_moe_mv[T_Q4_0],   "k_moe_mv_q4_0" },
        { &w->f_moe_mv[T_Q4_K],   "k_moe_mv_q4_K" },
        { &w->f_moe_mv[T_Q5_K],   "k_moe_mv_q5_K" },
        { &w->f_moe_mv[T_Q6_K],   "k_moe_mv_q6_K" },
        { &w->f_moe_mv[T_MXFP4],  "k_moe_mv_mxfp4" },
        { &w->f_moe_mv[T_NVFP4],  "k_moe_mv_nvfp4" },
        { &w->f_moe_mv[T_IQ3_XXS], "k_moe_mv_iq3_xxs" },
        { &w->f_moe_mv[T_IQ4_NL],  "k_moe_mv_iq4_nl" },
        { &w->f_moe_mv[T_Q2_0],    "k_moe_mv_q2_0" },
        { &w->f_moe_mv[T_Q5_0],    "k_moe_mv_q5_0" },
        { &w->f_moe_mv[T_Q4_1],    "k_moe_mv_q4_1" },
        { &w->f_moe_mv[T_Q5_1],    "k_moe_mv_q5_1" },
        { &w->f_moe_mv[T_IQ2_S],   "k_moe_mv_iq2_s" },
        { &w->f_moe_mv[T_IQ3_S],   "k_moe_mv_iq3_s" },
        { &w->f_moe_mv[T_IQ4_XS],  "k_moe_mv_iq4_xs" },
        { &w->f_moe_mv[T_IQ2_XS],  "k_moe_mv_iq2_xs" },
        { &w->f_moe_mv[T_IQ2_XXS], "k_moe_mv_iq2_xxs" },
        // expert-grouped prefill glue
        { &w->f_moe_gather,       "k_moe_gather" },
        { &w->f_moe_scatter,      "k_moe_scatter_add" },
        { &w->f_moe_actmul_plain, "k_moe_actmul_plain" },
        { &w->f_moe_copy,         "k_moe_copy_cols" },
        { &w->f_q8x[1][T_IQ3_S], "k_q8x_iq3_s_x1" },
        { &w->f_q8x[2][T_IQ3_S], "k_q8x_iq3_s_x2" },
        { &w->f_q8x[3][T_IQ3_S], "k_q8x_iq3_s_x3" },
        { &w->f_q8x[4][T_IQ3_S], "k_q8x_iq3_s_x4" },
        { &w->f_q8x[5][T_IQ3_S], "k_q8x_iq3_s_x5" },
        { &w->f_q8x[6][T_IQ3_S], "k_q8x_iq3_s_x6" },
        { &w->f_q8x[7][T_IQ3_S], "k_q8x_iq3_s_x7" },
        { &w->f_q8x[8][T_IQ3_S], "k_q8x_iq3_s_x8" },
        { &w->f_q8x[1][T_IQ3_XXS], "k_q8x_iq3_xxs_x1" },
        { &w->f_q8x[2][T_IQ3_XXS], "k_q8x_iq3_xxs_x2" },
        { &w->f_q8x[3][T_IQ3_XXS], "k_q8x_iq3_xxs_x3" },
        { &w->f_q8x[4][T_IQ3_XXS], "k_q8x_iq3_xxs_x4" },
        { &w->f_q8x[5][T_IQ3_XXS], "k_q8x_iq3_xxs_x5" },
        { &w->f_q8x[6][T_IQ3_XXS], "k_q8x_iq3_xxs_x6" },
        { &w->f_q8x[7][T_IQ3_XXS], "k_q8x_iq3_xxs_x7" },
        { &w->f_q8x[8][T_IQ3_XXS], "k_q8x_iq3_xxs_x8" },
        { &w->f_q8x[1][T_IQ4_XS], "k_q8x_iq4_xs_x1" },
        { &w->f_q8x[2][T_IQ4_XS], "k_q8x_iq4_xs_x2" },
        { &w->f_q8x[3][T_IQ4_XS], "k_q8x_iq4_xs_x3" },
        { &w->f_q8x[4][T_IQ4_XS], "k_q8x_iq4_xs_x4" },
        { &w->f_q8x[5][T_IQ4_XS], "k_q8x_iq4_xs_x5" },
        { &w->f_q8x[6][T_IQ4_XS], "k_q8x_iq4_xs_x6" },
        { &w->f_q8x[7][T_IQ4_XS], "k_q8x_iq4_xs_x7" },
        { &w->f_q8x[8][T_IQ4_XS], "k_q8x_iq4_xs_x8" },
        { &w->f_quant_q8x,        "k_quant_q8x" },
        { &w->f_q8x[1][T_Q4_K], "k_q8x_q4_K_x1" },
        { &w->f_q8x[2][T_Q4_K], "k_q8x_q4_K_x2" },
        { &w->f_q8x[3][T_Q4_K], "k_q8x_q4_K_x3" },
        { &w->f_q8x[4][T_Q4_K], "k_q8x_q4_K_x4" },
        { &w->f_q8x[5][T_Q4_K], "k_q8x_q4_K_x5" },
        { &w->f_q8x[6][T_Q4_K], "k_q8x_q4_K_x6" },
        { &w->f_q8x[7][T_Q4_K], "k_q8x_q4_K_x7" },
        { &w->f_q8x[8][T_Q4_K], "k_q8x_q4_K_x8" },
        { &w->f_q8x[1][T_Q8_0], "k_q8x_q8_0_x1" },
        { &w->f_q8x[2][T_Q8_0], "k_q8x_q8_0_x2" },
        { &w->f_q8x[3][T_Q8_0], "k_q8x_q8_0_x3" },
        { &w->f_q8x[4][T_Q8_0], "k_q8x_q8_0_x4" },
        { &w->f_q8x[5][T_Q8_0], "k_q8x_q8_0_x5" },
        { &w->f_q8x[6][T_Q8_0], "k_q8x_q8_0_x6" },
        { &w->f_q8x[7][T_Q8_0], "k_q8x_q8_0_x7" },
        { &w->f_q8x[8][T_Q8_0], "k_q8x_q8_0_x8" },
    };
    for (size_t i = 0; i < sizeof(fns) / sizeof(*fns); i++)
        CK(cu.ModuleGetFunction(fns[i].f, w->mod, fns[i].name));
    // Tensor-core kernels: resolved non-fatally, so an older embedded PTX
    // without them still loads with TC unavailable. This table is the one
    // gpu_tc_type_has_kernel() answers from, so the tolerance gate's idea of
    // "TC-capable" cannot drift from what is registered here.
    for (size_t i = 0; i < sizeof TC_KERNELS / sizeof *TC_KERNELS; i++)
        cu.ModuleGetFunction(&w->f_gemm_tc[TC_KERNELS[i].type], w->mod,
                             TC_KERNELS[i].name);
    // TC2: non-fatal too, and usable only once its dynamic shared memory
    // (G2_SMEM in kernels.cu) is granted
    {
        static const struct { int type; const char *name; } TC2_KERNELS[] = {
            { T_Q4_K, "k_gemm2_q4_K" }, { T_Q6_K, "k_gemm2_q6_K" },
            { T_Q8_0, "k_gemm2_q8_0" }, { T_Q4_0, "k_gemm2_q4_0" },
            { T_IQ3_S, "k_gemm2_iq3_s" }, { T_IQ3_XXS, "k_gemm2_iq3_xxs" },
            { T_IQ2_S, "k_gemm2_iq2_s" }, { T_IQ2_XS, "k_gemm2_iq2_xs" },
            { T_IQ2_XXS, "k_gemm2_iq2_xxs" }, { T_IQ1_S, "k_gemm2_iq1_s" },
            { T_IQ1_M, "k_gemm2_iq1_m" }, { T_IQ4_XS, "k_gemm2_iq4_xs" },
            { T_IQ4_NL, "k_gemm2_iq4_nl" },
            { T_Q5_K, "k_gemm2_q5_K" }, { T_Q3_K, "k_gemm2_q3_K" },
            { T_Q2_K, "k_gemm2_q2_K" }, { T_Q4_1, "k_gemm2_q4_1" },
            { T_Q5_0, "k_gemm2_q5_0" }, { T_Q5_1, "k_gemm2_q5_1" },
            { T_NVFP4, "k_gemm2_nvfp4" },
        };
        cu.ModuleGetFunction(&w->f_x_half, w->mod, "k_x_half_scaled");
        for (size_t i = 0; i < sizeof TC2_KERNELS / sizeof *TC2_KERNELS; i++) {
            CUfunction f = NULL;
            if (!w->f_x_half || !cu.FuncSetAttribute ||
                cu.ModuleGetFunction(&f, w->mod, TC2_KERNELS[i].name) != 0 ||
                cu.FuncSetAttribute(f, CU_FUNC_ATTR_MAX_DYN_SMEM, G2_SMEM) != 0)
                f = NULL;
            w->f_gemm2[TC2_KERNELS[i].type] = f;
        }
        // E1: on a compute capability 8.0+ device, the same GEMM built for
        // compute_80 (one m16n8k16 per 16-deep step), loaded unless TC2 is
        // off. Non-fatal: a module that does not load leaves forced TC2 on
        // the compute_75 kernels and auto TC2 off (tc2_fn).
        if (sm80_module_wanted(w) && w->f_x_half && cu.FuncSetAttribute) {
            if (module_load_noted(&w->mod80, k_ptx80_src) != 0) {
                w->mod80 = NULL;
                fprintf(stderr, "gpu: the compute_80 kernels did not load; TC2 runs "
                        "only when forced, on its compute_75 kernels\n");
            } else {
                static const struct { int type; const char *name; } TC2_80[] = {
                    { T_Q4_K, "k_gemm2_80_q4_K" }, { T_Q6_K, "k_gemm2_80_q6_K" },
                    { T_Q8_0, "k_gemm2_80_q8_0" }, { T_Q4_0, "k_gemm2_80_q4_0" },
                    { T_IQ3_S, "k_gemm2_80_iq3_s" }, { T_IQ3_XXS, "k_gemm2_80_iq3_xxs" },
                    { T_IQ2_S, "k_gemm2_80_iq2_s" }, { T_IQ2_XS, "k_gemm2_80_iq2_xs" },
                    { T_IQ2_XXS, "k_gemm2_80_iq2_xxs" }, { T_IQ1_S, "k_gemm2_80_iq1_s" },
                    { T_IQ1_M, "k_gemm2_80_iq1_m" }, { T_IQ4_XS, "k_gemm2_80_iq4_xs" },
                    { T_IQ4_NL, "k_gemm2_80_iq4_nl" },
                    { T_Q5_K, "k_gemm2_80_q5_K" }, { T_Q3_K, "k_gemm2_80_q3_K" },
                    { T_Q2_K, "k_gemm2_80_q2_K" }, { T_Q4_1, "k_gemm2_80_q4_1" },
                    { T_Q5_0, "k_gemm2_80_q5_0" }, { T_Q5_1, "k_gemm2_80_q5_1" },
                    { T_NVFP4, "k_gemm2_80_nvfp4" },
                };
                for (size_t i = 0; i < sizeof TC2_80 / sizeof *TC2_80; i++) {
                    CUfunction f = NULL;
                    if (cu.ModuleGetFunction(&f, w->mod80, TC2_80[i].name) != 0 ||
                        cu.FuncSetAttribute(f, CU_FUNC_ATTR_MAX_DYN_SMEM, G2_SMEM) != 0)
                        f = NULL;
                    w->f_gemm2_80[TC2_80[i].type] = f;
                }
                // the 128-column twins, for GEMMs over more than 64 columns;
                // a device that refuses their shared memory keeps the above
                static const struct { int type; const char *name; } TC2_80W[] = {
                    { T_Q4_K, "k_gemm2_80w_q4_K" }, { T_Q6_K, "k_gemm2_80w_q6_K" },
                    { T_Q8_0, "k_gemm2_80w_q8_0" }, { T_Q4_0, "k_gemm2_80w_q4_0" },
                    { T_IQ3_S, "k_gemm2_80w_iq3_s" }, { T_IQ3_XXS, "k_gemm2_80w_iq3_xxs" },
                    { T_IQ2_S, "k_gemm2_80w_iq2_s" }, { T_IQ2_XS, "k_gemm2_80w_iq2_xs" },
                    { T_IQ2_XXS, "k_gemm2_80w_iq2_xxs" }, { T_IQ1_S, "k_gemm2_80w_iq1_s" },
                    { T_IQ1_M, "k_gemm2_80w_iq1_m" }, { T_IQ4_XS, "k_gemm2_80w_iq4_xs" },
                    { T_IQ4_NL, "k_gemm2_80w_iq4_nl" },
                    { T_Q5_K, "k_gemm2_80w_q5_K" }, { T_Q3_K, "k_gemm2_80w_q3_K" },
                    { T_Q2_K, "k_gemm2_80w_q2_K" }, { T_Q4_1, "k_gemm2_80w_q4_1" },
                    { T_Q5_0, "k_gemm2_80w_q5_0" }, { T_Q5_1, "k_gemm2_80w_q5_1" },
                    { T_NVFP4, "k_gemm2_80w_nvfp4" },
                };
                for (size_t i = 0; i < sizeof TC2_80W / sizeof *TC2_80W; i++) {
                    CUfunction f = NULL;
                    if (!w->f_gemm2_80[TC2_80W[i].type] ||
                        cu.ModuleGetFunction(&f, w->mod80, TC2_80W[i].name) != 0 ||
                        cu.FuncSetAttribute(f, CU_FUNC_ATTR_MAX_DYN_SMEM, G2_SMEM_W) != 0)
                        f = NULL;
                    w->f_gemm2_80w[TC2_80W[i].type] = f;
                }
                // the int8 GEMMs (R3.14): resolved like the TC2 kernels, so a
                // module without them leaves the route off
                static const struct { int type; const char *name; } I8K[] = {
                    { T_Q4_K, "k_gemm_i8_q4_K" }, { T_Q6_K, "k_gemm_i8_q6_K" },
                    { T_Q8_0, "k_gemm_i8_q8_0" },
                };
                if (cu.ModuleGetFunction(&w->f_quant_i8, w->mod80, "k_quant_i8") != 0)
                    w->f_quant_i8 = NULL;
                for (size_t i = 0; w->f_quant_i8 && i < sizeof I8K / sizeof *I8K; i++) {
                    CUfunction f = NULL;
                    if (cu.ModuleGetFunction(&f, w->mod80, I8K[i].name) != 0 ||
                        cu.FuncSetAttribute(f, CU_FUNC_ATTR_MAX_DYN_SMEM, I8_SMEM) != 0)
                        f = NULL;
                    w->f_gemm_i8[I8K[i].type] = f;
                }
            }
        }
    }
    // the tiled prefill attention: resolved non-fatally like the TC kernels,
    // so a PTX without it keeps k_attn
    cu.ModuleGetFunction(&w->f_attn_tile32, w->mod, "k_attn_tile32");
    cu.ModuleGetFunction(&w->f_attn_dec_gqa, w->mod, "k_attn_dec_gqa");
    // k_attn_tc stages Q in two halves beside the K/V tile: AQ_SMEM (kernels.cu)
    // of dynamic shared memory, past the 48 KB a launch gets without asking
    if (cu.ModuleGetFunction(&w->f_attn_tc, w->mod, "k_attn_tc") != 0 || !cu.FuncSetAttribute ||
        cu.FuncSetAttribute(w->f_attn_tc, CU_FUNC_ATTR_MAX_DYN_SMEM, AQ_SMEM) != 0)
        w->f_attn_tc = NULL;
    if (cu.ModuleGetFunction(&w->f_attn_tcw, w->mod, "k_attn_tcw") != 0 || !cu.FuncSetAttribute ||
        cu.FuncSetAttribute(w->f_attn_tcw, CU_FUNC_ATTR_MAX_DYN_SMEM, AW_SMEM(512)) != 0)
        w->f_attn_tcw = NULL;
    if (cu.ModuleGetFunction(&w->f_attn_tcw_mla, w->mod, "k_attn_tcw_mla") != 0 || !cu.FuncSetAttribute ||
        cu.FuncSetAttribute(w->f_attn_tcw_mla, CU_FUNC_ATTR_MAX_DYN_SMEM, AW_SMEM(576)) != 0)
        w->f_attn_tcw_mla = NULL;
    cu.ModuleGetFunction(&w->f_kv_gather, w->mod, "k_kv_gather");
    cu.ModuleGetFunction(&w->f_q35_conv_seq, w->mod, "k_q35_conv_seq");
    cu.ModuleGetFunction(&w->f_q35_delta_seq, w->mod, "k_q35_delta_seq");
    cu.ModuleGetFunction(&w->f_q35_delta1, w->mod, "k_q35_delta1");
    cu.ModuleGetFunction(&w->f_q35_delta_norm, w->mod, "k_q35_delta_norm");
    cu.ModuleGetFunction(&w->f_mamba_conv_seq, w->mod, "k_mamba2_conv_seq");
    cu.ModuleGetFunction(&w->f_mamba_ssd_seq, w->mod, "k_mamba2_ssd_seq");
    cu.ModuleGetFunction(&w->f_attn_merge32, w->mod, "k_attn_merge32");
    cu.ModuleGetFunction(&w->f_attn_dec_gqa_seq, w->mod, "k_attn_dec_gqa_seq");
    cu.ModuleGetFunction(&w->f_attn_merge32_seq, w->mod, "k_attn_merge32_seq");
    // the absorbed MLA decode stages G heads' Q beside one tile of 576-wide
    // rows: past the 48 KB a launch gets without asking (MD_SMEM_MAX)
    if (cu.ModuleGetFunction(&w->f_attn_dec_mla, w->mod, "k_attn_dec_mla") != 0 || !cu.FuncSetAttribute ||
        cu.FuncSetAttribute(w->f_attn_dec_mla, CU_FUNC_ATTR_MAX_DYN_SMEM, MD_SMEM_MAX) != 0)
        w->f_attn_dec_mla = NULL;
    if (cu.ModuleGetFunction(&w->f_attn_dec_mla_seq, w->mod, "k_attn_dec_mla_seq") != 0 || !cu.FuncSetAttribute ||
        cu.FuncSetAttribute(w->f_attn_dec_mla_seq, CU_FUNC_ATTR_MAX_DYN_SMEM, MD_SMEM_MAX) != 0)
        w->f_attn_dec_mla_seq = NULL;
    cu.ModuleGetFunction(&w->f_attn_tile16, w->mod, "k_attn_tile16");
    cu.ModuleGetFunction(&w->f_attn_tile8, w->mod, "k_attn_tile8");
    return true;
fail:
    return false;
}

static bool shared_upload_weights(gpu_weights *w, model_t *m, const shared_plan *P) {
    int G = P->G;
    bool full = P->full;
    bool mtp_dev = P->mtp_dev;
    // weights: the file bytes the offloaded layers reference (the whole file
    // image for a full split, the planned prefix otherwise), so byte offsets
    // stay valid: every instance sharing this upload indexes it by offsets
    // against its own mmap base. The per-tensor fallback adds up to about the
    // same.
    upload_report_begin(&w->rep, w->weights_len, plat_now());
    if (!w->bound) {
        // One contiguous prefix. When the allocation itself is refused,
        // retry the same split per tensor before giving the model to
        // the CPU: the bindings ask for exactly the planned bytes, and a
        // partial split that was planned to fit usually does.
        CUresult rc = cu.MemAlloc(&w->weights, w->weights_len);
        if (rc != 0) {
            fprintf(stderr, "gpu: cu.MemAlloc(&w->weights, w->weights_len) "
                            "failed: %s (%.2f GB as one prefix); retrying "
                            "the same %d-layer split per tensor\n",
                    cu_err(rc), w->weights_len / 1e9, G);
            w->weights = 0;
            w->bound = true;
            w->weights_len = 0;
            w->img_off = 0;
        } else {
            CK(htod_reported(w, w->weights, (const uint8_t *)m->gf.map + w->img_off,
                             w->weights_len));
        }
    }
    if (w->bound) {
        // Pack only tensors used by the CUDA half (device_tensors_each), in
        // runs: the tensors that lie together share one allocation (a
        // per-tensor allocation each cost the allocator's granularity: a
        // granite-4.2-8b full split held 0.48 GB less free that way), and a
        // tensor in another mapping is bound alone.
        binding_ctx bc = { w, m };
        tensor_list L = { m, NULL, 0, 0, false };
        device_tensors_each(m, G, full, mtp_dev, list_add, &L);
        int outside = 0;
        bool ok = !L.oom && tensor_runs(m, &L, &outside, binding_add_range, &bc) >= 0;
        for (int i = L.n - outside; ok && i < L.n; i++)
            ok = binding_add(w, m, L.t[i]);
        free(L.t);
        if (!ok) goto fail;
    }
    {
        char line[192];
        if (upload_report_end(&w->rep, plat_now(), line, sizeof line))
            fprintf(stderr, "gpu: %s\n", line);
    }
    return true;
fail:
    return false;
}

// The small model-wide constants: rope tables, output and head norms, the
// NextN/MTP norms, the weightless V norm.
static bool shared_upload_constants(gpu_weights *w, model_t *m, int max_hd) {
    CK(cu.MemAlloc(&w->dummy, 4));

    w->inv_freq = f32_dbuf(m->rope_inv_freq, m->rope_dim / 2,
                           "global rope table", -1);
    w->inv_freq_local = f32_dbuf(m->rope_inv_freq_local,
                                 m->rope_dim_local / 2,
                                 "local rope table", -1);
    w->out_norm = f32_dbuf(m->out_norm_w, m->n_embd,
                           "output norm", -1);
    if (m->hyper_conn)
        w->hc_head_norm = f32_dbuf(m->hc_head_norm,
                                   (size_t)m->hc_count * m->n_embd,
                                   "hc head mixer norm", -1);
    if (m->mtp_ready) {
        size_t hcd = (size_t)mtp_streams(m) * m->n_embd;
        // one stream: the head norm is nextn.shared_head_norm, or the
        // backbone's output norm (model.c mtp_run)
        const float *hn = m->mtp_head_norm_w ? m->mtp_head_norm_w : m->hyper_conn ? NULL : m->out_norm_w;
        w->mtp_enorm     = f32_dbuf(m->mtp_enorm_w, m->n_embd, "MTP embedding norm", m->n_layer);
        w->mtp_hnorm     = f32_dbuf(m->mtp_hnorm_w, hcd, "MTP hidden norm", m->n_layer);
        w->mtp_head_norm = f32_dbuf(hn, hcd, "MTP head norm", m->n_layer);
        if (!w->mtp_enorm || !w->mtp_hnorm || !w->mtp_head_norm) goto fail;
    }
    if (!w->inv_freq || (m->hyper_conn ? !w->hc_head_norm : !w->out_norm) ||
        (m->rope_inv_freq_local && !w->inv_freq_local)) goto fail;
    if (m->v_rmsnorm) { // weightless per-head V norm: weight of ones
        float *ones = malloc(sizeof(float) * max_hd);
        if (!ones) {
            fprintf(stderr, "gpu: CUDA V-norm staging allocation failed — "
                    "using CPU\n");
            goto fail;
        }
        for (int i = 0; i < max_hd; i++) ones[i] = 1.0f;
        w->ones = f32_dbuf(ones, max_hd, "weightless V norm", -1);
        free(ones);
        if (!w->ones) goto fail;
    }
    return true;
fail:
    return false;
}

// The per-layer tables of norms, biases and recurrent and MoE constants.
// RUNNER_MOE_HOT: layer l's hot experts into one device allocation, slot by
// slot from the mapped file, and its router beside the other bound tensors.
static bool moe_hot_upload(gpu_weights *w, model_t *m, int l) {
    const layer_t *ly = &m->layers[l];
    int S = m->moe_hot_n[l];
    const int *slot = m->moe_hot_slot + (size_t)l * m->n_expert;
    uint64_t gs, us, ds;
    moe_hot_strides(ly, m->n_embd, m->n_ff_exp, &gs, &us, &ds);
    if (!binding_add(w, m, ly->ffn_gate_inp)) return false;
    size_t bytes = (size_t)S * (size_t)(gs + us + ds);
    CUdeviceptr d = 0;
    CUresult rc = cu.MemAlloc(&d, bytes);
    if (rc != 0) {
        fprintf(stderr, "gpu: CUDA allocation for blk.%d's hot bank (%d experts, "
                "%.2f GB) failed: %s — using CPU\n", l, S, bytes / 1e9, cu_err(rc));
        return false;
    }
    const uint8_t *g = ly->ffn_gate_exps->data, *u = ly->ffn_up_exps->data,
                  *dn = ly->ffn_down_exps->data;
    for (int e = 0; e < ly->n_expert; e++) {
        int k = slot[e];
        if (k < 0) continue;
        if (cu.MemcpyHtoD(d + (size_t)k * gs, g + (size_t)e * gs, gs) != 0 ||
            cu.MemcpyHtoD(d + (size_t)S * gs + (size_t)k * us, u + (size_t)e * us, us) != 0 ||
            cu.MemcpyHtoD(d + (size_t)S * (gs + us) + (size_t)k * ds, dn + (size_t)e * ds,
                          ds) != 0) {
            fprintf(stderr, "gpu: CUDA upload of blk.%d's hot bank failed — using CPU\n", l);
            cu.MemFree(d);
            return false;
        }
    }
    w->hot_bank[l] = d;
    w->bound_bytes += bytes;
    bool first = true;
    for (int k = 0; k < l; k++) if (moe_hot_on(m, k)) first = false;
    if (first) {
        int layers = 0;
        for (int k = 0; k < m->n_layer; k++) layers += moe_hot_on(m, k);
        fprintf(stderr, "gpu-split: RUNNER_MOE_HOT %d expert slots in %d host layers on "
                "the device (%.2f GB)\n", m->moe_hot_total, layers, moe_hot_bytes(m) / 1e9);
    }
    return true;
}

static bool shared_upload_layer_tables(gpu_weights *w, model_t *m) {
    // one slot past the backbone for the NextN/MTP block when it is bound
    const int n_tab = m->n_layer + (m->mtp_ready ? 1 : 0);
    w->n_tab = n_tab;
    // one pointer table per gpu_tables.h row, each n_tab slots
    const gpu_table_desc *rows = gpu_table_rows();
    for (int t = 0; t < GT_COUNT; t++)
        *cuda_table(w, (gpu_table)t) = calloc(n_tab, sizeof(CUdeviceptr));
    for (int t = 0; t < GT_COUNT; t++)
        if (!*cuda_table(w, (gpu_table)t)) {
            fprintf(stderr, "gpu: CUDA shared per-layer pointer-table "
                    "allocation failed (%s) — using CPU\n", rows[t].table_label);
            goto fail;
        }
    if (moe_any_on_device(m)) {
        // all-ones per-expert scale table for archs without a down-
        // projection scale — k_moe_actmul multiplies unconditionally
        w->moe_ones = f32_dbuf_ones(NULL, (size_t)m->n_expert,
                                    "MoE default expert scale", -1);
        if (!w->moe_ones) goto fail;
    }
    for (int l = 0; l < n_tab; l++) {
        layer_t *ly = &m->layers[l];
        w->attn_norm[l] = f32_dbuf(ly->attn_norm_w, m->n_embd,
                                   "attention norm", l);
        if (m->hyper_conn) {
            size_t hcd = (size_t)m->hc_count * m->n_embd;
            w->hc_an[l] = f32_dbuf(ly->hc_attn_norm, hcd, "hc attention mixer norm", l);
            w->hc_fn[l] = f32_dbuf(ly->hc_ffn_norm, hcd, "hc FFN mixer norm", l);
        }
        w->ffn_norm[l]  = f32_dbuf(ly->ffn_norm_w, m->n_embd,
                                   "FFN norm", l);
        w->bq[l] = f32_dbuf(ly->bq, model_q_dim(m, l), "Q bias", l);
        w->bk[l] = f32_dbuf(ly->bk, model_kv_dim(m, l), "K bias", l);
        w->bv[l] = f32_dbuf(ly->bv, model_kv_dim(m, l), "V bias", l);
        w->bo[l] = f32_dbuf(ly->bo, m->n_embd, "attention output bias", l);
        w->qn[l] = f32_dbuf(ly->qnorm_w, model_head_dim(m, l),
                            "Q norm", l);
        w->kn[l] = f32_dbuf(ly->knorm_w, model_head_dim(m, l),
                            "K norm", l);
        w->mla_kvn[l] = f32_dbuf(ly->kv_a_norm_w, (size_t)m->mla_kv_lora,
                                 "latent norm", l);
        // the absorbed form's kv_b, dequantized at load (NULL in the naive form)
        w->mla_kvb[l] = ly->mla_kvb && ly->wkv_b
                        ? f32_dbuf(ly->mla_kvb, (size_t)(ly->wkv_b->ne[0] * ly->wkv_b->ne[1]),
                                   "attn_kv_b f32", l)
                        : 0;
        w->pan[l] = f32_dbuf(ly->post_attn_norm_w, m->n_embd,
                             "post-attention norm", l);
        w->pfn[l] = f32_dbuf(ly->post_ffn_norm_w, m->n_embd,
                             "post-FFN norm", l);
        if (ly->ple_post_norm)
            w->ple_pn[l] = f32_dbuf(ly->ple_post_norm, m->n_embd,
                                    "per-layer embedding post-norm", l);
        if (ly->recurrent) {
            w->ssm_dt[l] = f32_dbuf(ly->ssm_dt, m->ssm.v_heads,
                                    "SSM dt", l);
            w->ssm_a[l] = f32_dbuf(ly->ssm_a, m->ssm.v_heads,
                                   "SSM A", l);
            if (m->qwen35) {
                w->ssm_norm[l] = f32_dbuf(ly->ssm_norm_w, m->ssm.state,
                                          "SSM norm", l);
            } else {   // Mamba-2 (granitehybrid/nemotron_h)
                int cd = m->ssm.inner + 2 * m->ssm.groups * m->ssm.state;
                w->ssm_D[l] = f32_dbuf(ly->ssm_d, m->ssm.v_heads,
                                       "Mamba D", l);
                w->ssm_conv_b[l] = f32_dbuf(ly->ssm_conv1d_b, cd,
                                            "Mamba conv bias", l);
                w->ssm_gnorm[l] = f32_dbuf(ly->ssm_norm_w, m->ssm.inner,
                                           "Mamba gated norm", l);
            }
        }
        // gpt-oss: per-head sink logits + router/per-expert bias tables
        w->sinks[l] = f32_dbuf(ly->attn_sinks, m->n_head,
                                "attention sinks", l);
        if (moe_hot_on(m, l) && !moe_hot_upload(w, m, l)) goto fail;
        if (ly->is_moe && !moe_on_host(m, l)) {
            w->gib[l] = f32_dbuf(ly->ffn_gate_inp_b, ly->n_expert,
                                 "MoE router bias", l);
            w->geb[l] = f32_dbuf(ly->ffn_gate_exps_b,
                                 (size_t)ly->n_expert * m->n_ff_exp,
                                 "MoE gate bias", l);
            w->ueb[l] = f32_dbuf(ly->ffn_up_exps_b,
                                 (size_t)ly->n_expert * m->n_ff_exp,
                                 "MoE up bias", l);
            w->deb[l] = f32_dbuf(ly->ffn_down_exps_b,
                                 (size_t)ly->n_expert * m->n_embd,
                                 "MoE down bias", l);
        }
        if (ly->moe_gemma) {
            // gemma-4 dual-branch norms + router scale. All are optional in
            // the GGUF; the CPU path treats an absent norm as weightless and
            // an absent gate_inp_scale as 1.0, so mirror that here (ones /
            // 1.0-fold) rather than leave a null device ptr — k_rmsnorm has
            // no NULL guard and would fault on device pointer 0.
            w->g_pn1[l]  = f32_dbuf_ones(ly->ffn_post_norm1_w, m->n_embd,
                                         "Gemma post-FFN norm 1", l);
            w->g_prn2[l] = f32_dbuf_ones(ly->ffn_pre_norm2_w, m->n_embd,
                                         "Gemma pre-FFN norm 2", l);
            w->g_pn2[l]  = f32_dbuf_ones(ly->ffn_post_norm2_w, m->n_embd,
                                         "Gemma post-FFN norm 2", l);
            // fold the router's 1/sqrt(n_embd) into the uploaded scale so a
            // plain weighted rmsnorm reproduces the CPU router input exactly
            float inv = 1.0f / sqrtf((float)m->n_embd);
            float *gs = malloc(sizeof(float) * m->n_embd);
            if (!gs) {
                fprintf(stderr, "gpu: CUDA Gemma router-scale staging "
                        "allocation failed in blk.%d — using CPU\n", l);
                goto fail;
            }
            for (int i = 0; i < m->n_embd; i++)
                gs[i] = (ly->gate_inp_scale ? ly->gate_inp_scale[i] : 1.0f) * inv;
            w->g_gis[l] = f32_dbuf(gs, m->n_embd,
                                   "Gemma router input scale", l);
            free(gs);
            // per-expert down scale for the indirect expert path (ones
            // when the GGUF has none — same 1.0 fold the CPU applies)
            w->g_dsc[l] = f32_dbuf_ones(ly->down_exps_scale, ly->n_expert,
                                        "Gemma expert down scale", l);
        }
        bool moe_dev = ly->is_moe && !moe_on_host(m, l);
        // ffn_norm is required only where the layer HAS an FFN: a
        // nemotron_h SSM or attention block carries none (skip_ffn), and
        // demanding its upload here silently pushed every nemotron_h
        // model to the CPU.
        for (int t = 0; t < GT_COUNT; t++)
            if (cuda_table_needed(&rows[t], m, ly, moe_dev) && !(*cuda_table(w, (gpu_table)t))[l]) {
                fprintf(stderr, "gpu: CUDA shared upload is missing %s in "
                        "blk.%d — using CPU\n", rows[t].upload_label, l);
                goto fail;
            }
    }
    return true;
fail:
    return false;
}

// The window asked for: model_params.gpu_window (--verify restoring a
// record), else RUNNER_CUDA_WINDOW (a diagnostic and A/B control). Either is
// a cap, a multiple of 64 and never wider than the plan allows; -b caps it
// the same way. --verify then checks the window it got (gpu_prompt_window).
static int cuda_window_force(const model_t *m) {
    if (m->gpu_window_req > 0) return m->gpu_window_req;
    const char *e = getenv("RUNNER_CUDA_WINDOW");
    return e && *e ? atoi(e) : 0;
}

// the window an instance was built for (transcripts record it)
int gpu_prompt_window(const model_t *m) {
    const gpu_t *g = m ? m->gpu : NULL;
    return g ? g->win : 0;
}

// R3.51: the OS video memory budget and usage on the weights' device right
// now; false where the OS publishes none (WDDM only), and then nothing that
// reads this changes behaviour.
static bool os_vidmem_now(const gpu_weights *w, uint64_t *budget, uint64_t *usage) {
    char luid[8]; unsigned mask = 0;
    return cu.DeviceGetLuid && cu.DeviceGetLuid(luid, &mask, w->dev) == 0 &&
           plat_gpu_os_budget((const unsigned char *)luid, budget, usage);
}

static gpu_weights *shared_build(model_t *m, size_t act_bytes, size_t per_col,
                                 size_t tc2_per_col, int max_hd, uint64_t fsize,
                                 uint64_t fino, int64_t fmtime) {
    gpu_weights *w = calloc(1, sizeof(gpu_weights));
    if (!w) {
        fprintf(stderr, "gpu: CUDA shared-weight state allocation failed — "
                "using CPU\n");
        return NULL;
    }
    shared_plan P = {0};
    if (!shared_record_identity(w, m, fsize, fino, fmtime) ||
        !shared_open_context(w) ||
        !shared_plan_budget(w, m, act_bytes, &P) ||
        !shared_plan_split(w, m, &P))
        goto fail;
    // TC2's fp16 activations take window columns only where TC2 will run:
    // forced, or in auto mode where the compute_80 module is to load
    bool tc2_charged = tc2_forced() || sm80_module_wanted(w);
    if (tc2_charged) per_col += tc2_per_col;
    // The prompt window: wider columns only from what the split left spare,
    // so it can never cost a layer (model_gpu_window). Decided once here and
    // shared by every instance of the upload, like the split itself.
    w->win = model_gpu_window(P.vram_budget > P.used ? P.vram_budget - P.used : 0,
                              per_col, (int)P.instances, m->n_batch,
                              cuda_window_force(m));
    fprintf(stderr, "gpu-split: prompt window %d columns (batch %d, %.2f GB spare, "
            "%zu bytes a column%s)\n", w->win, m->n_batch,
            (P.vram_budget > P.used ? P.vram_budget - P.used : 0) / 1e9, per_col,
            tc2_charged ? " with TC2's activations" : "");
    shared_plan_upload(w, m, &P);
    if (!shared_load_module(w)) goto fail;
    upload_vram_begin();   // RUNNER_DEBUG_VRAM only; a no-op otherwise
    bool uploaded = shared_upload_weights(w, m, &P);
    if (uploaded) {
        vram_probe("upload: after the weights");
        uploaded = shared_upload_constants(w, m, max_hd);
    }
    if (uploaded) {
        vram_probe("upload: after the constants");
        uploaded = shared_upload_layer_tables(w, m);
    }
    upload_vram_end();
    if (!uploaded) goto fail;
    w->refs = 1;
    vram_probe("after weights upload");
    // WDDM lowers this process's budget while its upload fills the card, so
    // the figure the plan used is not the one the load ends under: report the
    // budget as it stands now. Nothing is sized from it (a load 0.3 GB over
    // decoded at full speed; a layer handed back ran three times slower).
    {
        uint64_t ob = 0, ou = 0;
        if (os_vidmem_now(w, &ob, &ou))
            fprintf(stderr, "gpu: OS video memory budget after the weights %.2f GB, "
                            "%.2f GB in use\n", ob / 1e9, ou / 1e9);
    }
    if (getenv("RUNNER_DEBUG_VRAM")) {
        size_t bb = 0;
        for (int i = 0; i < w->n_bindings; i++) bb += w->bindings[i].nbytes;
        size_t br = 0;   // as the driver allocates them: each binding to its granularity
        for (int i = 0; i < w->n_bindings; i++) br += (size_t)gpu_plan_binding_bytes(w->bindings[i].nbytes, true);
        fprintf(stderr, "vram: %d bindings, %.3f GB requested; prefix upload %.3f GB "
                        "(bytes: requested %zu, rounded %zu)\n",
                w->n_bindings, bb / 1e9, w->weights_len / 1e9, bb, br);
    }
    return w;

fail:
    shared_destroy(w);
    return NULL;
}

static gpu_weights *shared_acquire(model_t *m, size_t act_bytes, size_t per_col,
                                   size_t tc2_per_col, int max_hd) {
    uint64_t fsize = 0, fino = 0;
    int64_t  fmtime = 0;
    // Every file gets an identity: part 1's (size, inode, mtime) plus, in
    // shared_matches, the part count and total mapped bytes, so a split GGUF's
    // slots share one device copy (the hazard split_guard describes).
    bool have_id = file_id(m->path, &fsize, &fino, &fmtime);
    gpu_weights *w = NULL;
    pthread_mutex_lock(&g_shared_mu);
    if (have_id)
        for (w = g_shared; w; w = w->next)
            if (shared_matches(w, m, fsize, fino, fmtime)) break;
    if (w) {
        w->refs++;
        fprintf(stderr, "gpu: reusing resident weights (%.1f GB, now shared by "
                "%d instances)\n",
                (w->bound ? w->bound_bytes : w->weights_len) / 1e9, w->refs);
    } else {
        w = shared_build(m, act_bytes, per_col, tc2_per_col, max_hd, fsize, fino, fmtime);
        // an entry with no file identity stays private — shared_matches never
        // returns it — but it is listed anyway so the split guard can compare
        // splits across it; shared_release unlinks it normally at refs 0
        if (w) {
            w->no_id = !have_id;
            split_guard(w, m);
            w->next = g_shared;
            g_shared = w;
        }
    }
    pthread_mutex_unlock(&g_shared_mu);
    return w;
}

// One instance's activation widths and the scratch the plan budgets.
typedef struct {
    int q_dim, kv_dim, max_hd, xdim;
    int q35_convdim, q35_mixdim;   // DeltaNet conv and mix widths
    size_t act_bytes;              // the plan's charge, at GPU_WINDOW_MIN columns
    size_t act_per_col;            // what each further window column allocates
    size_t tc2_per_col;            // TC2's fp16 activations per column, where it runs
} cuda_dims;

// Every refusal that needs no driver: router shapes, per-layer expert counts,
// shared-expert and gate shapes, FFN widths, tensor types. Every weight matmul
// needs a kernel for its quant type (wv may be absent: gemma4's global layers
// reuse the raw K projection as V).
// Generalized MoE routing has no device kernel: k_moe_route is softmax and
// top-k with no bias input, so the backend is refused, not routed wrongly.
// Mamba-2 hybrids run the device SSD scan and conv1d (k_mamba2_*). Dense ones
// offload fully; MoE granitehybrid keeps every bank on the host (its ungated
// shared expert, moe_bank_device_ok).
// qwen35moe composes dense qwen35's trunk, the routed MoE kernels and
// tile_shexp, held by test-cuda-qwen35moe and the real-file cells.
static bool cuda_admit(model_t *m) {
    // deepseek2 runs here in both forms of the multi-head latent attention
    // (R4.4.5 c1 naive, c2 absorbed). The q-lora pair (attn_q_a -> norm ->
    // attn_q_b: DeepSeek-V2 and V3, not V2-Lite) has no device path yet
    if (m->deepseek2) {
        bool qlora = false;
        for (int l = 0; l < m->n_layer; l++) qlora = qlora || m->layers[l].wq_a;
        if (qlora) {
            fprintf(stderr, "gpu: deepseek2 with a q-lora (attn_q_a) is not on "
                    "the CUDA backend yet — using CPU\n");
            return false;
        }
    }
    // LatentMoE (Nemotron 3.5 Super and Bolt): the experts run at the latent
    // width between ffn_latent_down and ffn_latent_up, which no device path
    // or host bank here does. Refused by name, ahead of the router checks
    // that refuse these models today, so a router kernel added later cannot
    // admit them at the wrong width.
    if (m->moe_latent > 0) {
        fprintf(stderr, "gpu: LatentMoE (experts at width %d between "
                "ffn_latent_down and ffn_latent_up) has no device path, "
                "running on CPU\n", m->moe_latent);
        return false;
    }
    // An odd attention head width (R6.68.112, model_gpu_odd_head_why): the
    // loader asks it first under --gpu auto, and so does --info fit; a forced
    // --gpu reaches here without that check
    char head_why[200];
    if (model_gpu_odd_head_why(m, head_why, sizeof head_why)) {
        fprintf(stderr, "gpu: %s, running on CPU\n", head_why);
        return false;
    }
    // Per-layer expert widths (Nemotron 3.5 Bolt, which the rule above already
    // refuses): every device expert table and kernel here takes the model's
    // one width. Refused by name, so that no later path admits a narrower
    // layer's bank at the widest width.
    for (int l = 0; l < m->n_layer; l++)
        if (m->layers[l].is_moe && m->layers[l].n_ff_exp > 0 &&
            m->layers[l].n_ff_exp != m->n_ff_exp) {
            fprintf(stderr, "gpu: per-layer expert widths (blk.%d is %d wide, the "
                    "widest %d) have no device path, running on CPU\n", l,
                    m->layers[l].n_ff_exp, m->n_ff_exp);
            return false;
        }
    // The router and the selection bias matter only to a bank the device runs:
    // a host bank (--cpu-moe) is routed by model_moe_ffn_cpu while the device
    // runs the trunk. So these refuse only when a bank is on the device.
    // Every device route clamps at MOE_ROUTE_MAX experts and MOE_MAX_USED
    // picks, so a router past them would answer among a subset of its experts.
    char route_why[200];
    if (m->n_expert > 0 && moe_any_on_device(m) &&
        model_moe_route_why(m->n_expert, m->n_expert_used, MOE_ROUTE_MAX, MOE_MAX_USED,
                            route_why, sizeof route_why)) {
        fprintf(stderr, "gpu: %s, running on CPU (--cpu-moe keeps the experts and "
                "their routing on the host)\n", route_why);
        return false;
    }
    if (m->n_expert > 0 && moe_any_on_device(m) && !model_moe_router_is_plain_any_norm(m)) {
        fprintf(stderr, "gpu: this model's MoE router (gating=%d groups=%d "
                "norm_w=%d scale=%g) has no device kernel, running on CPU "
                "(--cpu-moe keeps the experts and their routing on the host)\n",
                m->expert_gating, m->n_expert_groups, (int)m->expert_norm_w,
                (double)m->expert_w_scale);
        return false;
    }
    for (int l = 0; l < m->n_layer && m->n_expert > 0; l++)
        if (m->layers[l].exp_probs_b && m->layers[l].is_moe &&
            (!moe_on_host(m, l) || moe_hot_on(m, l))) {
            fprintf(stderr, "gpu: MoE selection bias (exp_probs_b) has no "
                    "device kernel, running on CPU (--cpu-moe keeps the "
                    "experts and their routing on the host)\n");
            return false;
        }
    // A pruned model gives a layer fewer experts than the model declares
    // (model.c holds each count inside [n_expert_used, expert_count]); each
    // layer's tables take that layer's count, and scratch sized by the declared
    // count covers every layer.
    // The shared always-on expert runs as tile_shexp, gated by its scalar
    // sigmoid or not; a shape moe_bank_device_ok refuses sends the model to
    // the CPU unless --cpu-moe keeps those banks on the host.
    if (m->n_ff_shexp > 0 && moe_any_on_device(m)) {
        bool ok_shape = true;
        for (int l = 0; ok_shape && l < m->n_layer; l++)
            if (m->layers[l].is_moe && !moe_on_host(m, l) && !moe_bank_device_ok(m, l))
                ok_shape = false;
        if (!ok_shape) {
            fprintf(stderr, "gpu: this shared-expert shape has no device path — "
                    "running on CPU (--cpu-moe keeps it on the host)\n");
            return false;
        }
    }
    // The dense attention output gate (muse-glimmer) rides qwen35's machinery:
    // only the projection into q35_gate is arch-specific. The gate plus sparse
    // MoE (afmoe) stays on the host: it has never run against the identity gates.
    // spark2_5's per-head gate and exact GELU run on the device for the dense
    // FFN. The MoE kernels' ffn_act selector knows SiLU, tanh-GELU and
    // swiglu_oai only, so an expert FFN with the exact GELU stays on the host.
    if (m->ffn_act == ACT_GELU_ERF && m->n_expert > 0) {
        fprintf(stderr, "gpu: experts with the exact GELU are not on the CUDA "
                "backend yet — using CPU\n");
        return false;
    }
    if (m->attn_out_gate && m->n_expert > 0) {
        // afmoe: per-element attention output gate + sparse MoE has no
        // certified device path; running it would be unverified, so the
        // whole model stays on the host.
        fprintf(stderr, "gpu: gated attention with sparse MoE (afmoe) is not "
                "on the CUDA backend yet — using CPU\n");
        return false;
    }
    if (!gpu_tensor_type_ok(m->output)) return false;
    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        // every tensor the device reads: the mixer side always, the dense
        // FFN unless a host bank replaces it (the expert banks are checked
        // below)
        gguf_tensor *mix[LAYER_MIXER_MAX];
        int nmix = layer_mixer_tensors(ly, mix);
        for (int i = 0; i < nmix; i++)
            if (!gpu_tensor_type_ok(mix[i])) return false;
        if (!(m->cpu_moe && ly->is_moe && moe_on_host(m, l)) &&
            (!gpu_tensor_type_ok(ly->w_gate) || !gpu_tensor_type_ok(ly->w_up) ||
             !gpu_tensor_type_ok(ly->w_down))) return false;
        // The architecture publishes this tiny depthwise kernel as F32 in its
        // GGUF contract. Keeping it F32 avoids a second quant decoder in the
        // stateful kernel; an unexpected export falls back safely.
        if (ly->ssm_conv && ly->ssm_conv->type != T_F32) {
            fprintf(stderr, "gpu: tensor %s uses %s; the CUDA recurrent "
                    "kernel requires F32 — using CPU\n", ly->ssm_conv->name,
                    ggml_type_name(ly->ssm_conv->type));
            return false;
        }
    }
    // MoE layers leave the dense gate/up/down NULL, so the loop above never
    // checks the router or the expert weights, yet enc_mv indexes the KT_N
    // kernel tables by their types. Validate them here: an unsupported type
    // would read past the table or launch a NULL function at decode.
    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        // A host-resident bank never reaches a device kernel; an AUTO fit may
        // still promote this layer later, so only a pinned host layer skips.
        if (!ly->is_moe || (moe_on_host(m, l) && m->cpu_moe_layers != CPU_MOE_AUTO &&
                            !moe_hot_on(m, l)))
            continue;
        if (!gpu_tensor_type_ok(ly->ffn_gate_inp)) return false;
        if (ly->moe_split) {
            for (int e = 0; e < ly->n_expert; e++)
                if (!gpu_tensor_type_ok(ly->moe_g[e]) ||
                    !gpu_tensor_type_ok(ly->moe_u[e]) ||
                    !gpu_tensor_type_ok(ly->moe_d[e])) return false;
        } else {
            if (!gpu_tensor_type_ok(ly->ffn_gate_exps) ||
                !gpu_tensor_type_ok(ly->ffn_up_exps) ||
                !gpu_tensor_type_ok(ly->ffn_down_exps)) return false;
        }
    }
    return true;
}

// The driver library loaded and initialized, and a device present.
static bool cuda_open_driver(void) {
    // Deterministic admission-fallback tracer. This fires before driver
    // discovery so CPU-only CI can hold the diagnostic contract for a failure
    // that normally requires an actual CUDA device plus host allocation
    // pressure to reach.
    const char *init_inject = getenv("RUNNER_CUDA_INIT_INJECT_FAILURE");
    if (init_inject && !strcmp(init_inject, "shared-pointer-tables")) {
        fprintf(stderr, "gpu: CUDA shared per-layer pointer-table allocation "
                "failed (injected) — using CPU\n");
        return false;
    }

    if (!cu_load()) {
        fprintf(stderr, "gpu: CUDA driver library is unavailable or incomplete "
                "— using CPU\n");
        return false;
    }
    CUresult init_rc = cu.Init(0);
    if (init_rc != 0) {
        fprintf(stderr, "gpu: CUDA initialization failed: %s — using CPU\n",
                cu_err(init_rc));
        return false;
    }
    prof.on = getenv("RUNNER_CUDA_PROFILE") != NULL;
    int ndev = 0;
    CUresult count_rc = cu.DeviceGetCount(&ndev);
    if (count_rc != 0) {
        fprintf(stderr, "gpu: CUDA device enumeration failed: %s — using "
                "CPU\n", cu_err(count_rc));
        return false;
    }
    if (ndev < 1) {
        fprintf(stderr, "gpu: CUDA reports no devices — using CPU\n");
        return false;
    }
    return true;
}

// The instance's activation widths (per-layer maxima on heterogeneous
// architectures) and the scratch bytes the split plan budgets for.
// TC2's activation width: the widest matvec input it can receive, in
// halves per window column (the Q8X buffer's sizing). Charged and allocated
// only where TC2 can run, so a device that never runs it holds no TC2 scratch.
static size_t tc2_xh_width(const model_t *m, int xdim, int q35_mixdim) {
    size_t mi = (size_t)xdim;
    if ((size_t)m->n_ff > mi) mi = (size_t)m->n_ff;
    if ((size_t)q35_mixdim > mi) mi = (size_t)q35_mixdim;
    if (2 * (size_t)m->n_embd > mi) mi = 2 * (size_t)m->n_embd;
    if ((size_t)m->ssm.inner > mi) mi = (size_t)m->ssm.inner;
    if ((size_t)m->n_ff_shexp > mi) mi = (size_t)m->n_ff_shexp;
    return mi;
}

static void cuda_instance_dims(const model_t *m, cuda_dims *D) {
    // heterogeneous archs (gemma4) vary q/kv widths per layer: size the
    // shared activation buffers to the maxima
    int q_dim = 0, kv_dim = 0, max_hd = 0;
    for (int l = 0; l < m->n_layer; l++) {
        if (model_q_dim(m, l)    > q_dim)  q_dim  = model_q_dim(m, l);
        if (model_kv_dim(m, l)   > kv_dim) kv_dim = model_kv_dim(m, l);
        if (model_head_dim(m, l) > max_hd) max_hd = model_head_dim(m, l);
    }
    int xdim = q_dim > m->n_embd ? q_dim : m->n_embd;
    int q35_convdim = m->qwen35
                    ? 2 * m->ssm.state * m->ssm.groups + m->ssm.inner : 0;
    int q35_mixdim = q35_convdim > 2 * q_dim ? q35_convdim : 2 * q_dim;
    gpu_act_cost ac = model_gpu_act_cost(m->n_embd, xdim, q_dim, kv_dim, m->n_ff,
                                         m->n_head, m->n_ctx, m->n_vocab);
    size_t act_bytes = ac.fixed + GPU_WINDOW_MIN * ac.per_col;
    // A window column also carries the buffers below that the plan at the
    // narrowest window never charged (each sized in cuda_alloc_*): MoE
    // routing and grouping staging, the shared expert, the hyper-connection
    // streams, the per-layer embeddings and the NextN/MTP rows. A wider
    // window spends only spare budget, so these are counted in full there.
    size_t per_col = ac.per_col;
    // the column maxima (xsc); TC2's fp16 activations (xh) are charged by
    // shared_build, which knows the device
    per_col += sizeof(float);
    D->tc2_per_col = 2 * tc2_xh_width(m, xdim, q35_mixdim);
    if (m->n_expert > 0) {
        size_t used = m->n_expert_used > 0 ? (size_t)m->n_expert_used : 1;
        size_t gw = (size_t)m->n_ff_exp * (m->moe_gemma ? 2 : 1);
        per_col += sizeof(float) * ((size_t)m->n_expert + 4 * used +
                                    3 * (size_t)m->n_embd + 2 * gw +
                                    (m->moe_gemma ? 5 * (size_t)m->n_embd : 0));
        if (m->n_ff_shexp > 0)
            per_col += sizeof(float) * ((size_t)xdim + 2 * (size_t)m->n_ff_shexp +
                                        m->n_embd + 1);
        // the by-expert prefill's slot outputs and pair tables (moe_twin)
        per_col += used * (sizeof(float) * (size_t)m->n_embd + 2 * sizeof(int));
    }
    if (m->hyper_conn)
        per_col += sizeof(float) * (3 * (size_t)m->hc_count * m->n_embd +
                                    m->hc_low_rank + m->hc_count);
    per_col += sizeof(float) * mla_cols(m);   // deepseek2's mla_buf
    if (m->n_embd_ple > 0)
        per_col += 2 * sizeof(float) * ((size_t)m->n_layer + 1) * m->n_embd_ple;
    if (m->mtp_ready) {
        size_t hcd = (size_t)mtp_streams(m) * m->n_embd;
        per_col += sizeof(float) * (3 * hcd + m->n_embd);
    }
    // the recurrent state and the mixer scratch at the narrowest window, the
    // figure --fit charges too (model_gpu_recurrent_cost); a wider window's
    // scratch columns are per_col's
    int rec = m->qwen35 ? MODEL_GPU_REC_DELTANET
            : (m->granite_hybrid || m->nemotron_h) ? MODEL_GPU_REC_MAMBA2 : MODEL_GPU_REC_NONE;
    act_bytes += (size_t)model_gpu_recurrent_cost(rec, (uint64_t)model_rec_count(m),
        (uint64_t)m->ssm.conv_kernel, (uint64_t)m->ssm.inner, (uint64_t)m->ssm.state,
        (uint64_t)m->ssm.groups, (uint64_t)m->ssm.v_heads, (uint64_t)q_dim);
    if (m->qwen35)
        per_col += sizeof(float) * ((size_t)q35_mixdim + q35_convdim +
                                    m->ssm.inner + 2 * m->ssm.v_heads + q_dim);
    if (m->granite_hybrid || m->nemotron_h) {
        int cd = m->ssm.inner + 2 * m->ssm.groups * m->ssm.state;
        int dip = 2 * m->ssm.inner + 2 * m->ssm.groups * m->ssm.state +
                  m->ssm.v_heads;
        per_col += sizeof(float) * ((size_t)dip + cd + m->ssm.inner);
    }
    if (m->attn_out_gate && !m->qwen35) per_col += sizeof(float) * (size_t)q_dim;
    D->q_dim = q_dim;
    D->kv_dim = kv_dim;
    D->max_hd = max_hd;
    D->xdim = xdim;
    D->q35_convdim = q35_convdim;
    D->q35_mixdim = q35_mixdim;
    D->act_bytes = act_bytes;
    D->act_per_col = per_col;
}

// The KV cache for the offloaded layers (and a bound NextN/MTP block).
static bool cuda_alloc_kv(model_t *m, gpu_t *g) {
    // device KV holds only the offloaded layers [0, gpu_layers) -- the
    // raw cumulative boundary, not layer gpu_layers's own (possibly
    // redirected) storage location. See model_kv_boundary_bytes().
    // K and V are sized separately: under --kv k8v4 the two caches
    // have different row formats (K q8_0, V fp4).
    const bool mtp_dev = m->mtp_ready && m->gpu_layers == m->n_layer;
    size_t k_bytes = model_k_boundary_bytes(m, m->gpu_layers + (mtp_dev ? 1 : 0));
    size_t v_bytes = model_v_boundary_bytes(m, m->gpu_layers + (mtp_dev ? 1 : 0));
    // A hybrid's stateful layers own no KV rows, so a split whose device
    // layers are all stateful has 0 bytes to hold, and cuMemAlloc refuses 0.
    // The pointers still go to launches and 0-byte copies, so the cache is a
    // small allocation filled with 0xFF (NaN in every KV format).
    enum { KV_EMPTY_BYTES = 256 };
    size_t k_alloc = k_bytes ? k_bytes : KV_EMPTY_BYTES, v_alloc = v_bytes ? v_bytes : KV_EMPTY_BYTES;
    CK(cu.MemAlloc(&g->kc, k_alloc));
    CK(cu.MemsetD8(g->kc, k_bytes ? 0 : 0xFF, k_alloc));
    if (model_kv_v_shared(m)) {   // the absorbed MLA form's V is its K row
        g->vc = g->kc;
        return true;
    }
    CK(cu.MemAlloc(&g->vc, v_alloc));
    CK(cu.MemsetD8(g->vc, v_bytes ? 0 : 0xFF, v_alloc));
    return true;
fail:
    return false;
}

// Per-column activation scratch, the hyper-connection and NextN/MTP
// buffers, attention partials and the logits.
static bool cuda_alloc_activations(model_t *m, gpu_t *g, const cuda_dims *D) {
    int q_dim = D->q_dim;
    int kv_dim = D->kv_dim;
    int max_hd = D->max_hd;
    int xdim = D->xdim;
    int q35_mixdim = D->q35_mixdim;
    const bool mtp_dev = m->mtp_ready && m->gpu_layers == m->n_layer;
    if (m->n_embd_ple > 0) {
        // [token][layer][n_embd_ple] for a whole tile, plus the per-layer
        // gate scratch. The pre-pass itself runs on the host (its
        // per-token row gather+dequant from the quantized token table has
        // no device kernel; see model_ple_prepass), so only its result
        // crosses the bus — once per forward, not once per layer.
        size_t per_tok = (size_t)m->n_layer * m->n_embd_ple;
        CK(cu.MemAlloc(&g->ple,   sizeof(float) * (size_t)g->win * per_tok));
        CK(cu.MemAlloc(&g->ple_g, sizeof(float) * (size_t)g->win * m->n_embd_ple));
    }
    CK(cu.MemAlloc(&g->x,      sizeof(float) * (size_t)g->win * m->n_embd));
    CK(cu.MemAlloc(&g->xb,     sizeof(float) * (size_t)g->win * xdim));
    CK(cu.MemAlloc(&g->xb2,    sizeof(float) * (size_t)g->win * xdim));
    if (mtp_dev) {
        size_t hcd = (size_t)mtp_streams(m) * m->n_embd;
        CK(cu.MemAlloc(&g->mtp_h,       sizeof(float) * (size_t)m->n_batch * hcd));
        CK(cu.MemAlloc(&g->mtp_pending, sizeof(float) * hcd));
        CK(cu.MemAlloc(&g->mtp_cat,     sizeof(float) * (size_t)g->win * hcd * 2));
        CK(cu.MemAlloc(&g->mtp_hid,     sizeof(float) * hcd));
        CK(cu.MemAlloc(&g->mtp_en,      sizeof(float) * (size_t)g->win * m->n_embd));
        CK(cu.MemAlloc(&g->mtp_hn,      sizeof(float) * (size_t)g->win * hcd));
        CK(cu.MemAlloc(&g->mtp_rows,    sizeof(float) * (size_t)m->n_batch * hcd));
        CK(cu.MemsetD8(g->mtp_pending, 0, sizeof(float) * hcd));
        if (m->qwen35 && cu.MemcpyDtoDAsync) {
            int convdim = 2 * m->ssm.state * m->ssm.groups + m->ssm.inner;
            size_t hist_l = model_rec_count(m) * (m->ssm.conv_kernel - 1) * convdim;
            size_t state_l = model_rec_count(m) * m->ssm.v_heads * m->ssm.state * m->ssm.state;
            g->snap_rows = m->spec_batch < 8 ? m->spec_batch : 8;
            CK(cu.MemAlloc(&g->snap_state, sizeof(float) * state_l * g->snap_rows));
            CK(cu.MemAlloc(&g->snap_hist,  sizeof(float) * (hist_l ? hist_l : 1) * g->snap_rows));
        }
    }
    if (m->hyper_conn) {
        size_t hcd = (size_t)m->hc_count * m->n_embd;
        CK(cu.MemAlloc(&g->x_hc,      sizeof(float) * (size_t)g->win * hcd));
        CK(cu.MemAlloc(&g->hc_xn,     sizeof(float) * (size_t)g->win * hcd));
        CK(cu.MemAlloc(&g->hc_gate,   sizeof(float) * (size_t)g->win * hcd));
        CK(cu.MemAlloc(&g->hc_lo,     sizeof(float) * (size_t)g->win * m->hc_low_rank));
        CK(cu.MemAlloc(&g->hc_inject, sizeof(float) * (size_t)g->win * m->hc_count));
        CK(cu.MemAlloc(&g->splitk,    sizeof(float) * SPLITK_PARTIALS));
    }
    CK(cu.MemAlloc(&g->q,      sizeof(float) * (size_t)g->win * q_dim));
    CK(cu.MemAlloc(&g->kt,     sizeof(float) * (size_t)g->win * kv_dim));
    CK(cu.MemAlloc(&g->vt,     sizeof(float) * (size_t)g->win * kv_dim));
    if (m->deepseek2)
        CK(cu.MemAlloc(&g->mla_buf, sizeof(float) * (size_t)g->win * mla_cols(m)));
    // hb/hb2 also hold gemma-4 MoE's fused gate_up (2*n_ff_exp) for one
    // token; win*n_ff dwarfs that for real models, but size for the max so a
    // pathological n_ff_exp can never overrun the buffer.
    size_t hb_elems = (size_t)g->win * m->n_ff;
    if (m->moe_gemma && 2 * (size_t)m->n_ff_exp > hb_elems)
        hb_elems = 2 * (size_t)m->n_ff_exp;
    CK(cu.MemAlloc(&g->hb,     sizeof(float) * hb_elems));
    CK(cu.MemAlloc(&g->hb2,    sizeof(float) * hb_elems));
    if (gpu_q8x_on()) {
        // the widest matvec input: FFN down (n_ff), the attention/recurrent
        // projections (xdim, the DeltaNet mix), the MTP concat (2 n_embd)
        int mi = xdim;
        if (m->n_ff > mi) mi = m->n_ff;
        if (q35_mixdim > mi) mi = q35_mixdim;
        if (2 * m->n_embd > mi) mi = 2 * m->n_embd;
        if (m->ssm.inner > mi) mi = m->ssm.inner;
        g->q8x_cap = mi;
        CK(cu.MemAlloc(&g->q8x_q, (size_t)MVB * mi));
        CK(cu.MemAlloc(&g->q8x_d, sizeof(float) * (size_t)MVB * (mi / 32 + 1)));
    }
    if (m->n_ff_shexp > 0 && moe_any_on_device(m)) {
        CK(cu.MemAlloc(&g->shexp_in,   sizeof(float) * (size_t)g->win * xdim));
        CK(cu.MemAlloc(&g->shexp_g,    sizeof(float) * (size_t)g->win * m->n_ff_shexp));
        CK(cu.MemAlloc(&g->shexp_u,    sizeof(float) * (size_t)g->win * m->n_ff_shexp));
        CK(cu.MemAlloc(&g->shexp_o,    sizeof(float) * (size_t)g->win * m->n_embd));
        CK(cu.MemAlloc(&g->shexp_gate, sizeof(float) * (size_t)g->win));
    }
    CK(cu.MemAlloc(&g->att,    sizeof(float) * ATT_COLS * (size_t)m->n_head * m->n_ctx));
    CK(cu.MemAlloc(&g->vcol_pos, sizeof(int) * MVB));
    CK(cu.MemAlloc(&g->vcol_kcp, sizeof(uint64_t) * MVB));
    CK(cu.MemAlloc(&g->vcol_vcp, sizeof(uint64_t) * MVB));
    // flash-decoding partials: per (token, head, split) -> hd weighted-V + max + sum
    CK(cu.MemAlloc(&g->attn_part, sizeof(float) * MVB * (size_t)m->n_head *
                                  ATTN_SPLITS * (max_hd + 2)));
    // spec_batch rows: the speculative verify reads every row's head
    CK(cu.MemAlloc(&g->logits, sizeof(float) * (size_t)(m->spec_batch > 0 ? m->spec_batch : 1) * m->n_vocab));
    CK(cu.MemAlloc(&g->xsc,    sizeof(float) * (size_t)(g->win > TC_N ? g->win : TC_N)));
    if (g->sw->f_x_half && tc2_can_run(g)) {
        // TC2's activations, in half for every window column
        g->xh_cap = (int)tc2_xh_width(m, xdim, q35_mixdim);
        CK(cu.MemAlloc(&g->xh, 2 * (size_t)g->win * (size_t)g->xh_cap));
        if (gpu_i8_on() && g->xh_cap % 32 == 0) {
            g->i8_cap = g->xh_cap;
            size_t nb = (size_t)g->win * (size_t)(g->i8_cap / 32);
            CK(cu.MemAlloc(&g->i8_xq, (size_t)g->win * (size_t)g->i8_cap));
            CK(cu.MemAlloc(&g->i8_xd, sizeof(float) * nb));
            CK(cu.MemAlloc(&g->i8_xs, sizeof(float) * nb));
        }
    }
    CK(cu.MemAlloc(&g->pos_dev, sizeof(int) * (size_t)((g->win + ATT_COLS - 1) / ATT_COLS)));
    return true;
fail:
    return false;
}

// The recurrent state (DeltaNet, Mamba-2) and the attention gate scratch.
static bool cuda_alloc_recurrent(model_t *m, gpu_t *g, const cuda_dims *D) {
    int q_dim = D->q_dim;
    int q35_convdim = D->q35_convdim;
    int q35_mixdim = D->q35_mixdim;
    if (m->qwen35) {
        size_t hist_elems = model_rec_count(m) * (m->ssm.conv_kernel - 1) *
                            q35_convdim;
        size_t hist_alloc = hist_elems ? hist_elems : 1;
        size_t state_elems = model_rec_count(m) * m->ssm.v_heads *
                             m->ssm.state * m->ssm.state;
        CK(cu.MemAlloc(&g->q35_mix, sizeof(float) * (size_t)g->win * q35_mixdim));
        CK(cu.MemAlloc(&g->q35_cv, sizeof(float) * (size_t)g->win * q35_convdim));
        CK(cu.MemAlloc(&g->q35_z, sizeof(float) * (size_t)g->win * m->ssm.inner));
        CK(cu.MemAlloc(&g->q35_beta, sizeof(float) * (size_t)g->win * m->ssm.v_heads));
        CK(cu.MemAlloc(&g->q35_alpha, sizeof(float) * (size_t)g->win * m->ssm.v_heads));
        CK(cu.MemAlloc(&g->q35_gate, sizeof(float) * (size_t)g->win * q_dim));
        CK(cu.MemAlloc(&g->q35_hist, sizeof(float) * hist_alloc));
        CK(cu.MemAlloc(&g->q35_state, sizeof(float) * state_elems));
        CK(cu.MemAlloc(&g->q35_hist_prev, sizeof(float) * hist_alloc));
        CK(cu.MemAlloc(&g->q35_state_prev, sizeof(float) * state_elems));
        CK(cu.MemsetD8(g->q35_hist, 0, sizeof(float) * hist_alloc));
        CK(cu.MemsetD8(g->q35_state, 0, sizeof(float) * state_elems));
        CK(cu.MemsetD8(g->q35_hist_prev, 0, sizeof(float) * hist_alloc));
        CK(cu.MemsetD8(g->q35_state_prev, 0, sizeof(float) * state_elems));
    }
    if (m->granite_hybrid || m->nemotron_h) {
        int cd = m->ssm.inner + 2 * m->ssm.groups * m->ssm.state;
        int dip = 2 * m->ssm.inner + 2 * m->ssm.groups * m->ssm.state +
                  m->ssm.v_heads;
        size_t conv_elems = model_rec_count(m) * (m->ssm.conv_kernel - 1) * cd;
        size_t conv_alloc = conv_elems ? conv_elems : 1;
        size_t state_elems = model_rec_count(m) * m->ssm.inner * m->ssm.state;
        CK(cu.MemAlloc(&g->mamba_proj,  sizeof(float) * (size_t)g->win * dip));
        CK(cu.MemAlloc(&g->mamba_xBC,   sizeof(float) * (size_t)g->win * cd));
        CK(cu.MemAlloc(&g->mamba_y,     sizeof(float) * (size_t)g->win * m->ssm.inner));
        CK(cu.MemAlloc(&g->mamba_conv,       sizeof(float) * conv_alloc));
        CK(cu.MemAlloc(&g->mamba_state,      2 * sizeof(float) * state_elems));
        CK(cu.MemAlloc(&g->mamba_conv_prev,  sizeof(float) * conv_alloc));
        CK(cu.MemsetD8(g->mamba_conv, 0, sizeof(float) * conv_alloc));
        CK(cu.MemsetD8(g->mamba_state, 0, 2 * sizeof(float) * state_elems));
        CK(cu.MemsetD8(g->mamba_conv_prev, 0, sizeof(float) * conv_alloc));
        g->rec_cur = g->rec_prev = 0;
    }
    // gated attention without the rest of qwen35 (afmoe, muse-glimmer):
    // only the gate scratch is needed
    if (m->attn_out_gate && !m->qwen35)
        CK(cu.MemAlloc(&g->q35_gate, sizeof(float) * (size_t)g->win * q_dim));
    return true;
fail:
    return false;
}

// Sparse-MoE scratch, device and host, and which MoE paths apply.
static bool cuda_alloc_moe(model_t *m, gpu_t *g) {
    if (m->n_expert > 0) {
        // sparse-MoE buffers, sized for both the fused indirect path
        // (device routing for a whole tile, per-slot expert scratch) and
        // the eager fallback (which uses column/slot 0 only)
        int used_c = m->n_expert_used;
        if (used_c < 1) used_c = 1;
        if (used_c > MOE_MAX_USED) used_c = MOE_MAX_USED;
        int rows = used_c > g->win ? used_c : g->win;
        size_t gw = (size_t)m->n_ff_exp * (m->moe_gemma ? 2 : 1);
        if (gw < 1) gw = 1;
        CK(cu.MemAlloc(&g->moe_logits,
                       sizeof(float) * (size_t)g->win * m->n_expert));
        CK(cu.MemAlloc(&g->moe_eout,
                       sizeof(float) * (size_t)rows * m->n_embd));
        CK(cu.MemAlloc(&g->moe_sel,  sizeof(int)   * (size_t)g->win * used_c));
        CK(cu.MemAlloc(&g->moe_selw, sizeof(float) * (size_t)g->win * used_c));
        CK(cu.MemAlloc(&g->moe_hb,   sizeof(float) * (size_t)rows * gw));
        CK(cu.MemAlloc(&g->moe_hb2,  sizeof(float) * (size_t)rows * gw));
        // expert-grouped prefill staging (device + host mirrors)
        CK(cu.MemAlloc(&g->moe_gath, sizeof(float) * (size_t)g->win * m->n_embd));
        CK(cu.MemAlloc(&g->moe_dout, sizeof(float) * (size_t)g->win * m->n_embd));
        CK(cu.MemAlloc(&g->moe_gidx, sizeof(int)   * (size_t)g->win * used_c));
        CK(cu.MemAlloc(&g->moe_gw,   sizeof(float) * (size_t)g->win * used_c));
        g->h_moe_sel  = malloc(sizeof(int)   * (size_t)g->win * used_c);
        g->h_moe_selw = malloc(sizeof(float) * (size_t)g->win * used_c);
        g->h_moe_gidx = malloc(sizeof(int)   * (size_t)g->win * used_c);
        g->h_moe_gw   = malloc(sizeof(float) * (size_t)g->win * used_c);
        if (!g->h_moe_sel || !g->h_moe_selw || !g->h_moe_gidx ||
            !g->h_moe_gw) {
            fprintf(stderr, "gpu: CUDA MoE host staging allocation failed "
                    "— using CPU\n");
            goto fail;
        }
        if (m->moe_hot_n) {
            int hx = m->n_embd;
            for (int l = 0; l < m->n_layer; l++)
                if (model_q_dim(m, l) > hx) hx = model_q_dim(m, l);
            g->hot_xdim = hx;
            CK(cu.MemAlloc(&g->moe_hot_sel, sizeof(int) * (size_t)g->win * used_c));
            g->h_hot_sel   = malloc(sizeof(int)   * (size_t)g->win * used_c);
            g->h_hot_x     = malloc(sizeof(float) * (size_t)g->win * hx);
            g->h_hot_terms = malloc(sizeof(float) * (size_t)g->win * used_c * m->n_embd);
            if (!g->h_hot_sel || !g->h_hot_x || !g->h_hot_terms) {
                fprintf(stderr, "gpu: CUDA RUNNER_MOE_HOT host staging allocation "
                        "failed — using CPU\n");
                goto fail;
            }
        }
        g->moe_renorm = m->expert_norm_w ? 1 : 0;
        if (model_moe_replay_on()) {
            int np = 0;
            int *tab = model_moe_replay_table(m->n_layer, m->n_expert_used, &np);
            size_t cells = (size_t)np * m->n_layer * m->n_expert_used;
            if (!tab) goto fail;
            g->replay_npos = np;
            g->replay_nlayer = m->n_layer;
            if (cu.MemAlloc(&g->replay_tab, sizeof(int) * (cells ? cells : 1)) != 0 ||
                cu.MemAlloc(&g->replay_cnt, sizeof(unsigned) * 3) != 0 ||
                (cells && cu.MemcpyHtoD(g->replay_tab, tab, sizeof(int) * cells) != 0) ||
                cu.MemsetD8(g->replay_cnt, 0, sizeof(unsigned) * 3) != 0) {
                free(tab);
                goto fail;
            }
            free(tab);
        }
        g->route_trace = model_moe_trace_on();
        g->moe_fused_ok = moe_fused_eligible(m);
        // RUNNER_MOE_EAGER: debug escape to the v0.1.4 host-routing
        // path — the reference side of the RUNNER_DEBUG_MOE bit
        // comparison, and the fused-vs-eager A/B instrument.
        int eager = g_moe_eager_force;   // one read: a test may set it meanwhile
        if (eager >= 0) {
            if (eager) g->moe_fused_ok = false;
        } else if (getenv("RUNNER_MOE_EAGER")) {
            g->moe_fused_ok = false;
        }
        g->moe_grouped_ok = moe_grouped_eligible(m);
        g->moe_twin_ok = g->moe_fused_ok && moe_twin_eligible(m, g->sw);
        if (g->moe_twin_ok) {
            CK(cu.MemAlloc(&g->moe_gpos,  sizeof(int) * (size_t)g->win * used_c));
            CK(cu.MemAlloc(&g->moe_gsel,  sizeof(int) * (size_t)g->win * used_c));
            CK(cu.MemAlloc(&g->moe_slots,
                           sizeof(float) * (size_t)g->win * used_c * m->n_embd));
            g->h_moe_gpos = malloc(sizeof(int) * (size_t)g->win * used_c);
            g->h_moe_gsel = malloc(sizeof(int) * (size_t)g->win * used_c);
            if (!g->h_moe_gpos || !g->h_moe_gsel) g->moe_twin_ok = false;
        }
    }
    if (m->moe_gemma)
        // gemma-4-MoE dual-branch scratch: xn, xn2, rin, mout — window
        // columns each so the grouped prefill can batch the whole tile
        CK(cu.MemAlloc(&g->g_scr,
                       sizeof(float) * 5 * (size_t)g->win * m->n_embd));
    return true;
fail:
    return false;
}

// Whether a qwen35 model's one-token decode may replay a captured graph
// (RUNNER_CUDA_GRAPH_RECUR=0 keeps it eager, an A/B control).
static bool cuda_graph_recur_on(void) {
    static _Atomic int on = -1;
    if (on < 0) { const char *e = getenv("RUNNER_CUDA_GRAPH_RECUR"); on = !(e && *e && strcmp(e, "0") == 0); }
    return on == 1;
}

// An attention layer with no V projection (gemma4's global layers) builds V
// as zero plus K. On the instance's stream the zeroing is a memset node a
// decode graph can capture, where the synchronous cuMemsetD8 forced an eager
// decode; same bytes either way. A driver without cuMemsetD8Async keeps the
// synchronous call. Other layers without wv keep their own rules.
static bool v_from_k_capturable(const model_t *m, int l) {
    const layer_t *ly = &m->layers[l];
    return cu.MemsetD8Async && !ly->wv && !ly->recurrent && !ly->skip_mixer &&
           !(m->l_no_attn && m->l_no_attn[l]);
}

// The stream graphs are captured on, and whether this model can be captured.
static void cuda_plan_graphs(model_t *m, gpu_t *g) {
    if (cu_graphs_ok() && cu.StreamCreate(&g->stream, 0) != 0)
        g->stream = NULL;
    // What keeps a decode graph off. gemma4's V copy without an async memset
    // is synchronous. MoE with device routing (moe_fused_ok) captures; the
    // eager fallback reads router logits back mid-forward and cannot.
    // Recurrent steps capture where they are launches on buffers fixed for the
    // instance, advanced in place, with the rollback copy taken before the
    // forward: qwen35 (with routed experts too: its banks hold no recurrent
    // state) and the dense Mamba-2 hybrids. qwen4exp (host steps between
    // layers) and the Mamba-2 hybrids with experts stay eager.
    // RUNNER_CUDA_GRAPH_RECUR=0 keeps every recurrent family eager.
    bool mamba_dense = (m->granite_hybrid || m->nemotron_h) && m->n_expert == 0;
    bool recur_graph = (m->qwen35 || mamba_dense) && !m->hyper_conn && cuda_graph_recur_on();
    // each rule names itself (graph_why), so a gate that requires the graph
    // (RUNNER_REQUIRE_GRAPH_CAPTURE) can tell eager by design from a failure
    // The three per-layer rules, the same union as before; which one names the
    // layer goes most specific first, so a recurrent layer (no V projection by
    // nature) is called recurrent and not V-less.
    const char *why = NULL;
    for (int l = 0; l < m->gpu_layers && !why; l++) {
        const layer_t *ly = &m->layers[l];
        bool vless = !ly->wv && !(ly->recurrent && recur_graph) &&
                     !(ly->skip_mixer && recur_graph) && !v_from_k_capturable(m, l);
        if (ly->recurrent && !recur_graph)
            why = "a recurrent layer of a family whose decode is not captured (or RUNNER_CUDA_GRAPH_RECUR=0)";
        else if (ly->is_moe && !g->moe_fused_ok)
            why = "MoE routing read back to the host mid-forward (the eager MoE path)";
        else if (vless && ly->skip_mixer)
            why = "an MLP-only block of a hybrid whose decode is not captured";
        else if (vless && m->l_no_attn && m->l_no_attn[l])
            why = "a block whose attention was removed (--remove-sublayer)";
        else if (vless)
            why = "a layer without a V projection that is not built on the stream (a synchronous memset)";
    }
    // RUNNER_DEBUG_MOE prints the first routings' choices, read back to the
    // host inside the fused decode forward: a capture cannot record that
    // (the R6.68.15 audit), so the debug run decodes eagerly
    if (!why && m->n_expert > 0 && moe_any_on_device(m) && getenv("RUNNER_DEBUG_MOE"))
        why = "RUNNER_DEBUG_MOE reads the routing back to the host";
    if (!why && m->qwen35 && !recur_graph)
        why = "a qwen35 file whose decode is not captured (hyper-connections, or RUNNER_CUDA_GRAPH_RECUR=0)";
    if (!why && moe_any_on_host(m)) why = "experts on the host";
    // RUNNER_MOE_REPLAY / RUNNER_MOE_TRACE: each routing takes its position
    // from the host, and a trace reads it back, so neither is captured
    if (!why && (g->replay_tab || g->route_trace)) why = "RUNNER_MOE_REPLAY or RUNNER_MOE_TRACE";
    if (why) { g->graph_bad = true; g->graph_why = why; }
}

// Host staging for the per-layer embeddings, the residual and the logits.
static bool cuda_alloc_host_staging(model_t *m, gpu_t *g) {
    if (m->n_embd_ple > 0) {
        size_t per_tok = (size_t)m->n_layer * m->n_embd_ple;
        g->h_ple     = malloc(sizeof(float) * (size_t)g->win * per_tok);
        g->h_ple_tmp = malloc(sizeof(float) * per_tok);
        if (!g->h_ple || !g->h_ple_tmp) {
            fprintf(stderr, "gpu: CUDA per-layer embedding host staging "
                    "allocation failed — using CPU\n");
            goto fail;
        }
    }
    g->h_x      = malloc(sizeof(float) * (size_t)g->win * m->n_embd);
    g->h_logits = malloc(sizeof(float) * (size_t)(m->spec_batch > 0 ? m->spec_batch : 1) * m->n_vocab);
    if (m->n_expert > 0)
        g->h_moe_logits = malloc(sizeof(float) * (size_t)m->n_expert * (size_t)(GPU_SMALL_BATCH > 1 ? GPU_SMALL_BATCH : 1));
    if (!g->h_x || !g->h_logits ||
        (m->n_expert > 0 && !g->h_moe_logits)) {
        fprintf(stderr, "gpu: CUDA host activation staging allocation "
                "failed — using CPU\n");
        goto fail;
    }
    return true;
fail:
    return false;
}

static void cuda_report_init(model_t *m, gpu_t *g, size_t act_bytes) {
    char name[128] = "CUDA GPU";
    cu.DeviceGetName(name, sizeof(name), g->sw->dev);
    if (m->cpu_moe) {
        int on_host = 0, moe_tot = 0;
        for (int l = 0; l < m->n_layer; l++) {
            if (!m->layers[l].is_moe) continue;
            moe_tot++;
            if (moe_on_host(m, l)) on_host++;
        }
        fprintf(stderr, "gpu: CUDA backend on %s (%d/%d attention layers, "
                "%.1f GB in VRAM; %d/%d expert layers on CPU)\n", name,
                m->gpu_layers, m->n_layer, g->sw->bound_bytes / 1e9,
                on_host, moe_tot);
    }
    else if (m->gpu_layers < m->n_layer)
        fprintf(stderr, "gpu: CUDA backend on %s (%d/%d layers, %.1f GB in "
                "VRAM; CPU runs the rest)\n", name, m->gpu_layers, m->n_layer,
                (g->sw->bound ? g->sw->bound_bytes : g->sw->weights_len) / 1e9);
    else
        fprintf(stderr, "gpu: CUDA backend on %s (%.1f GB weights in VRAM)\n",
                name, g->sw->weights_len / 1e9);
    // the number Phase 5 is judged on: with weights shared, a second slot
    // should cost only its KV cache and activation scratch
    tc_scalar_notice(m);
    size_t vfree = 0, vtotal = 0;
    if (cu.MemGetInfo(&vfree, &vtotal) == 0)
        fprintf(stderr, "gpu: VRAM %.2f GB free of %.2f GB after init "
                "(kv %.2f GB + scratch %.2f GB this instance)\n",
                vfree / 1e9, vtotal / 1e9,
                (double)model_kv_boundary_bytes_sum(m, m->gpu_layers) / 1e9,
                act_bytes / 1e9);
}

// Release the host pages of what the device now holds. An upload copies its
// tensors out of the read-only file mapping and every page it touched stayed
// resident (about 5 GB of host RSS for a 4.7 GB fully offloaded file). Same
// walk as the bound upload, plus the token embedding (read a row per token).
// No value changes: the map is PROT_READ + MAP_PRIVATE, so a later CPU read
// faults the file's bytes back in. weights_locked is checked: --mlock wins.
static bool release_host_pages(gguf_tensor *t, void *ctx) {
    (void)ctx;
    if (t && t->data) plat_release_pages(t->data, t->nbytes);
    return true;
}

// RUNNER_MOE_HOT: the slices moe_hot_upload copied into layer l's bank. The
// host stays home for the other experts, so its tensors keep their pages and
// only a listed expert's go. Its slices are the bank's bytes and the host path
// never reads them (model_moe_hot_terms runs the misses), so nothing here
// changes a value either. Returns the experts released.
static int moe_hot_release_host(model_t *m, const layer_t *ly, int l) {
    const int *slot = m->moe_hot_slot + (size_t)l * m->n_expert;
    uint64_t gs, us, ds;
    moe_hot_strides(ly, m->n_embd, m->n_ff_exp, &gs, &us, &ds);
    const uint8_t *g = ly->ffn_gate_exps->data, *u = ly->ffn_up_exps->data,
                  *dn = ly->ffn_down_exps->data;
    int n = 0;
    for (int e = 0; e < ly->n_expert; e++) {
        if (slot[e] < 0) continue;
        plat_release_pages(g + (size_t)e * gs, gs);
        plat_release_pages(u + (size_t)e * us, us);
        plat_release_pages(dn + (size_t)e * ds, ds);
        n++;
    }
    return n;
}

static void cuda_release_host_weights(model_t *m) {
    if (m->weights_locked) return;
    int hot = 0;
    for (int l = 0; l < m->gpu_layers && l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        gguf_tensor *mix[LAYER_MIXER_MAX];
        int nmix = layer_mixer_tensors(ly, mix);
        for (int i = 0; i < nmix; i++) release_host_pages(mix[i], NULL);
        if (!(ly->is_moe && moe_on_host(m, l)))
            layer_ffn_tensors_each(ly, ly->n_expert, release_host_pages, NULL);
        else if (moe_hot_on(m, l))
            hot += moe_hot_release_host(m, ly, l);
    }
    if (m->gpu_layers >= m->n_layer) release_host_pages(m->output, NULL);
    release_host_pages(m->tok_embd, NULL);
    if (hot)
        fprintf(stderr, "gpu-split: RUNNER_MOE_HOT gave back the host copies of %d experts\n", hot);
}

static bool kv_gather_init(gpu_t *g, model_t *m);

bool gpu_init(model_t *m) {
    if (!cuda_admit(m) || !cuda_open_driver()) return false;

    gpu_t *g = calloc(1, sizeof(gpu_t));
    if (!g) {
        fprintf(stderr, "gpu: CUDA backend state allocation failed — using "
                "CPU\n");
        return false;
    }

    cuda_dims D;
    cuda_instance_dims(m, &D);
    // the weights (and the CPU/GPU split they imply) come from the registry:
    // a second instance of the same file reuses the first one's upload
    g->sw = shared_acquire(m, D.act_bytes, D.act_per_col, D.tc2_per_col, D.max_hd);
    if (!g->sw) goto fail;
    m->gpu_layers = g->sw->gpu_layers;
    g->win = g->sw->win;
    CK(cu.CtxSetCurrent(g->sw->ctx));
    if (!cuda_alloc_kv(m, g) ||
        !kv_gather_init(g, m) ||
        !cuda_alloc_activations(m, g, &D) ||
        !cuda_alloc_recurrent(m, g, &D) ||
        !cuda_alloc_moe(m, g))
        goto fail;
    cuda_plan_graphs(m, g);
    if (!cuda_alloc_host_staging(m, g)) goto fail;
    cuda_report_init(m, g, D.act_bytes);

    vram_probe("after instance buffers");
    {   // R3.51, RUNNER_DEBUG_VRAM only: usage against the OS budget once
        // everything is allocated. Over it, Windows may move part of the
        // process's video memory to system memory; whether that slows decode
        // was not shown (0.3 GB over ran at full speed on an RTX 5070), so it
        // is a debug fact, not a default warning.
        uint64_t ob = 0, ou = 0;
        if (vram_debug_on() && os_vidmem_now(g->sw, &ob, &ou))
            fprintf(stderr, "vram: after init, OS usage %.3f of %.3f GB budget (%+.3f)%s\n",
                    ou / 1e9, ob / 1e9, ((double)ou - (double)ob) / 1e9,
                    ou > ob ? "; over it, Windows may move part of this process's video "
                              "memory to system memory" : "");
    }
    m->gpu = g;
    // only now: a load that fails above runs on the CPU, from these pages
    cuda_release_host_weights(m);
    return true;

fail:
    gpu_ctx_free(m, g);
    return false;

}

// ----------------------------------------------------------------- launches

// One-shot diagnostic for the GPU giving up mid-forward. Every enc_* helper
// returns a bare false, which covers a launch error, a missing kernel and a
// weight never uploaded alike, so the first failure names itself. Once per
// process: a failing forward fails for every layer.
static void gpu_first_error(const char *fmt, ...) {
    static _Atomic bool reported;   // any slot may fail first (compat.h)
    if (!plat_once(&reported)) return;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "gpu: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

static bool launch(gpu_t *g, CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                   unsigned bx, void **params) {
    if (prof.on) prof.total_launches[prof.cur_mode]++;
    if (!f) { gpu_first_error("no kernel bound for this operation"); return false; }
    CUresult r = cu.LaunchKernel(f, gx, gy, gz, bx, 1, 1, 0, g->stream, params, NULL);
    if (r != 0)
        gpu_first_error("cuLaunchKernel failed: %s (grid %ux%ux%u, block %u)",
                        cu_err(r), gx, gy, gz, bx);
    return r == 0;
}

// launch() with dynamic shared memory (bytes <= 48 KB: no opt-in attribute)
static bool launch_smem(gpu_t *g, CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                        unsigned bx, unsigned smem, void **params) {
    if (prof.on) prof.total_launches[prof.cur_mode]++;
    if (!f) { gpu_first_error("no kernel bound for this operation"); return false; }
    CUresult r = cu.LaunchKernel(f, gx, gy, gz, bx, 1, 1, smem, g->stream, params, NULL);
    if (r != 0)
        gpu_first_error("cuLaunchKernel failed: %s (grid %ux%ux%u, block %u, smem %u)",
                        cu_err(r), gx, gy, gz, bx, smem);
    return r == 0;
}

static void prof_report(void);

// lazily allocate the event pool the first time profiling runs
static void prof_init(void) {
    if (prof.inited || !prof.on) return;
    if (!cu.EventCreate || !cu.EventRecord || !cu.EventElapsedTime ||
        !cu.EventSynchronize) { prof.on = 0; return; }
#ifdef _WIN32
    LARGE_INTEGER f; QueryPerformanceFrequency(&f); prof.qpc_hz = (double)f.QuadPart;
#else
    prof.qpc_hz = 1.0;
#endif
    for (int i = 0; i < 4096; i++)
        if (cu.EventCreate(&prof.ev[i], 0) != 0) {
            // free the events already created so a mid-loop failure doesn't
            // orphan them (prof_report won't run: inited stays 0)
            for (int j = 0; j < i; j++) {
                if (cu.EventDestroy) cu.EventDestroy(prof.ev[j]);
                prof.ev[j] = NULL;
            }
            prof.on = 0;
            return;
        }
    prof.inited = 1;
    atexit(prof_report);   // CLI mode returns without model_free/gpu_free
}

// record an event tagged with the phase of the kernel group just launched;
// no-op unless profiling is on and we are not mid graph-capture
static void prof_mark(gpu_t *g, int phase) {
    if (!prof.on || prof.capturing || !prof.inited) return;
    if (prof.nev >= 4096) return;
    prof.ph[prof.nev] = phase;
    cu.EventRecord(prof.ev[prof.nev], g->stream);
    prof.nev++;
}

// after the call's stream/ctx has been synchronized, fold the per-group
// GPU deltas into the phase accumulators and reset the pool
static void prof_flush(void) {
    if (!prof.on || !prof.inited) return;
    for (int k = 1; k < prof.nev; k++) {
        float ms = 0;
        if (cu.EventElapsedTime(&ms, prof.ev[k - 1], prof.ev[k]) == 0 &&
            prof.ph[k] >= 0 && prof.ph[k] < PH_N)
            prof.t_ph[prof.cur_mode][prof.ph[k]] += ms;
    }
    prof.nev = 0;
}

static bool enc_rmsnorm(gpu_t *g, CUdeviceptr x, CUdeviceptr y, CUdeviceptr w,
                        int n, float eps, int batch, int xs, int ys) {
    void *p[] = { &x, &y, &w, &n, &eps, &xs, &ys };
    return launch(g, g->sw->f_rmsnorm, 1, batch, 1, 256, p);
}

// Tensor-core GEMM (RUNNER_CUDA_TC). Combos in tc_promoted() run TC by
// default, promoted per (type, arch) by the tolerance gate
// (tests/test_tc_tol.c); the rest stay scalar. =1 forces it on everywhere
// (how a candidate is measured), =0/off forces it off: the value is parsed.
enum { TC_ENV_UNSET = -2 };
static _Atomic int g_tc_state = TC_ENV_UNSET;   // -2 env unread, -1 per-combo, 0/1 forced

// Test hook (the TC tolerance gate compares scalar and TC in one process,
// so an env var read once at first launch cannot express "both"). Pass -1
// to return to the env/promotion default.
void gpu_tc_force(int on) {
    g_tc_state = on < 0 ? TC_ENV_UNSET : (on != 0);
}


// The fast decode matvec is a Metal lever: CUDA decode reaches its throughput
// through the batched/TC paths instead, and the CUDA matvec is not under the
// same byte-identity pin that forced Metal to grow a second kernel. Present so
// the mv gate links and can say "never dispatched here".
void gpu_mv_force(int on) { (void)on; }
unsigned long gpu_mv_dispatches(void) { return 0; }
unsigned long gpu_mv2_dispatches(void) { return 0; }

// Metal-only lever; present so the gate links and reports 'never here'.
void gpu_attn_coop_force(int on) { (void)on; }
unsigned long gpu_attn_coop_dispatches(void) { return 0; }
unsigned long gpu_attn_tile_dispatches(void) { return 0; }
static _Atomic unsigned long g_attn_tiled_dispatches;
unsigned long gpu_attn_tiled_dispatches(void) { return g_attn_tiled_dispatches; }
unsigned long gpu_attn_gqa_dispatches(void) { return 0; }

static _Atomic unsigned long g_splitk_dispatches;
unsigned long gpu_splitk_dispatches(void) { return g_splitk_dispatches; }
long gpu_live_buffers(void) { return cu_ok ? g_live_bufs : -1; }

// Same hook for the MoE routing path: the fused-vs-eager tolerance gate has to
// run both inside one process, which an env var read at first launch cannot
// express. -1 restores the env default (RUNNER_MOE_EAGER).
void gpu_moe_eager_force(int on) { g_moe_eager_force = on < 0 ? -1 : (on != 0); }

// Promoted (type, arch) combos: each has its own measured gate row on real
// weights; an unmeasured arch is absent, not implied. qwen3moe passed its
// gate but stays opt-in here (MoE routing amplifies fp16 noise) and runs as
// the moe-tc route below. Evidence: docs/tc-promotion-evidence.md.
// The codebook family (see tc_promoted below).
static bool tc_codebook(int type) {
    switch (type) {
        case T_IQ1_S: case T_IQ1_M: case T_IQ2_XXS: case T_IQ2_XS: case T_IQ2_S:
        case T_IQ3_XXS: case T_IQ3_S: case T_IQ4_XS: case T_IQ4_NL:
            return true;
        default:
            return false;
    }
}

// The six types whose tensor-core kernels arrived 2026-10-21
// (kernels_kq_stage.cuh): K-quants and legacy quants that prefilled on the
// scalar GEMM or a column tile at a time until then.
static bool tc_kq_type(int type) {
    return type == T_Q5_K || type == T_Q3_K || type == T_Q2_K ||
           type == T_Q4_1 || type == T_Q5_0 || type == T_Q5_1;
}

// The types the tensor-core GEMM is promoted for (the per-type rows in
// tc_promoted below).
static bool tc_type_promotable(int type) {
    return type == T_Q4_K || type == T_Q6_K || type == T_Q8_0 || type == T_Q4_0 ||
           tc_codebook(type) || tc_kq_type(type);
}

// MoE families whose tolerance route is their default: each measured against
// the publisher's float32 model over 2,000 positions (mean KL within 1.01x of
// the current route, and neither a paired Wilcoxon nor McNemar worse at
// p < 0.05). They are named routes: a record names them and --verify pins
// them. gemma4 missed the bound (1.034x) and stays opt-in. Rows:
// docs/tc-promotion-evidence.md.
static bool moe_tc_family(const model_t *m) {
    return m && (strcmp(m->arch, "qwen3moe") == 0 || strcmp(m->arch, "gpt-oss") == 0);
}
static bool moe_grouped_family(const model_t *m) {
    return m && strcmp(m->arch, "qwen3moe") == 0;
}

static int tc_nvfp4_env(void);
static bool tc_promoted(const model_t *m, int type) {
    // Each type below is promoted on tolerance rows from test_tc_tol on real
    // weights (64 teacher-forced positions and a free-running arm), never on an
    // exactness argument: a kernel once reported bit-identity while computing
    // 16 of its 64 columns. The rows, per type and arch, are in
    // docs/tc-promotion-evidence.md.
    // NVFP4: RUNNER_CUDA_TC_NVFP4=0 holds it off; =1 forces it on any
    // architecture, the trusted list below deciding otherwise.
    if (type == T_NVFP4) {
        if (tc_nvfp4_env() == 0)
            return false;
    } else if (!tc_type_promotable(type))
        return false;
    // The tc_kq_type six (float32-reference rows, docs/tc-promotion-evidence.md):
    // their stagers equal dequant_row bit for bit on the host (test-kq-stage)
    // and test-cuda-tc-kq holds each on the TC, TC2 and compute_80 kernels.
    // qwen4exp keeps them on the scalar GEMM: its admission covers what the TC
    // prefill touched on its release file, which these types did not.
    if (tc_kq_type(type) && strcmp(m->arch, "qwen4exp") == 0)
        return false;
    // The trusted architectures: each joined on its own test_tc_tol rows, which
    // gate the whole model (teacher-forced and free-running), so whatever subset
    // of its projections the TC prefill reaches is inside the measured envelope.
    // qwen3moe is absent: it fails the free-running arm (MoE amplification).
    // qwen4exp is covered for what the TC prefill touched on its release file
    // with the experts on the host; a device-expert row is still owed.
    // Rows: docs/tc-promotion-evidence.md.
    static const char *archs[] = { "llama", "phi3", "gemma4", "qwen3",
                                   "qwen35", "qwen4exp",
                                   "mistral3", "gemma3", "granite",
                                   "muse-glimmer", "qwen2",
                                   "apertus", "nemotron_h", "granitehybrid" };
    for (size_t i = 0; i < sizeof(archs) / sizeof(*archs); i++)
        if (strcmp(m->arch, archs[i]) == 0) return true;
    // Promoted for exactly the types its row's file exercised, each on a
    // float32-reference row and a speed row (docs/tc-promotion-evidence.md):
    // spark2_5 Q4_K/Q6_K, clef Q8_0/Q4_K. k2-horizon and stablelm run on the CPU
    // on a CUDA build (grouped RMSNorm, LayerNorm), so they have no route here.
    static const struct { const char *arch; int types[3]; } scoped[] = {
        { "spark2_5", { T_Q4_K, T_Q6_K, -1 } },
        { "clef",     { T_Q4_K, T_Q8_0, -1 } },
    };
    for (size_t i = 0; i < sizeof(scoped) / sizeof(*scoped); i++) {
        if (strcmp(m->arch, scoped[i].arch) != 0) continue;
        for (int k = 0; k < 3 && scoped[i].types[k] >= 0; k++)
            if (scoped[i].types[k] == type) return true;
    }
    return false;
}

// NVFP4's tensor-core route, promoted on the trusted architectures
// (tc_promoted). RUNNER_CUDA_TC_NVFP4 decides for NVFP4 tensors alone: 0 or
// off holds the route off, any other value forces it on, unset (-1) leaves
// it to tc_promoted. RUNNER_CUDA_TC, when set, decides for every type.
static int tc_nvfp4_env(void) {
    static _Atomic int on = -2;
    if (on == -2) {
        const char *e = getenv("RUNNER_CUDA_TC_NVFP4");
        on = !e || !*e ? -1 : strcmp(e, "0") != 0 && strcmp(e, "off") != 0;
    }
    return on;
}

static bool tc_on(const model_t *m, int type) {
    int st = g_tc_state;   // one read; gpu_tc_force may set it from a test meanwhile
    if (st == TC_ENV_UNSET) {
        const char *e = getenv("RUNNER_CUDA_TC");
        int v = !e || !*e ? -1                             // per-combo default
              : strcmp(e, "0") != 0 && strcmp(e, "off") != 0;
        // only from unread: a forced value set meanwhile stands
        st = atomic_compare_exchange_strong(&g_tc_state, &(int){ TC_ENV_UNSET }, v) ? v : g_tc_state;
    }
    if (type == T_NVFP4 && st < 0 && tc_nvfp4_env() == 1) return true;
    if (moe_tc_family(m)) {
        // a route pin decides, then RUNNER_CUDA_TC, then the default (on)
        int p = route_pinned(GR_MOE_TC);
        if (p >= 0) return p && tc_type_promotable(type);
        if (st >= 0) return st != 0;
        return tc_type_promotable(type);
    }
    if (st >= 0) return st != 0;
    return tc_promoted(m, type);
}

// Whether the default (RUNNER_CUDA_TC unset) takes the tensor cores for any
// weight matrix of this model: the route line's "tc=taken", else "tc=off". It
// read "tc=promoted" for every non-MoE architecture before, listed or not, so
// an unpromoted model's line and its records claimed a tensor-core prefill that
// never ran; --verify reads that older word as "not recorded" (main.c).
static bool tc_default_takes(const model_t *m) {
    for (uint64_t i = 0; i < m->gf.n_tensors; i++) {
        const gguf_tensor *t = &m->gf.tensors[i];
        if (t->n_dims == 2 && tc_promoted(m, t->type)) return true;
    }
    return false;
}

// The route's tensor-core policy for one weight type, the overrides included:
// what a record's profile.route.tc_types lists (model_route_receipt_json).
bool gpu_tc_takes(const model_t *m, int type) { return m && tc_on(m, type); }

// Said once at load: a codebook i-quant on an architecture whose tensor-core
// prefill is not promoted has only the scalar batched kernel (2.6 tok/s
// against 85.3 with the tensor cores on one 30B IQ3_XXS file). The user is
// told what they are getting and what the alternatives are.
static void tc_scalar_notice(const model_t *m) {
    if (g_tc_state == TC_ENV_UNSET) (void)tc_on(m, T_Q4_K);   // read the env once
    int n = 0, type = -1;
    for (uint64_t i = 0; i < m->gf.n_tensors; i++) {
        const gguf_tensor *t = &m->gf.tensors[i];
        if (t->n_dims != 2 || !tc_codebook(t->type) || tc_on(m, t->type)) continue;
        if (!n) type = t->type;
        n++;
    }
    if (!n) return;
    fprintf(stderr, "gpu: %d weight matrices are codebook i-quants (%s, ...) and "
            "the tensor-core prefill is %s for '%s': prompts are prefilled by the "
            "scalar kernel, which can be slower than the CPU. %s\n", n,
            ggml_type_name(type),
            g_tc_state == 0 ? "turned off (RUNNER_CUDA_TC=0)" : "not promoted",
            m->arch,
            g_tc_state == 0 ? "Unset RUNNER_CUDA_TC to use the promoted kernels."
                            : "RUNNER_CUDA_TC=1 uses the tensor cores (not yet "
                              "gated for this architecture); --gpu off prefills "
                              "on the host.");
}

// A kernel with a fixed compile-time column tile cannot take the full prefill
// batch, so a wider batch runs as ceil(batch/tile) launches of the proven
// shape. The scalar Q8_0 prefill GEMM (retired in R18.8.11) did this for its
// 8-column tile; this
// generalises it so the buffer width and the tile width can move apart.
// `scale` is passed as a sixth kernel parameter only when `scaled` is set:
// the NVFP4 kernels take the per-tensor companion that way, every other
// kernel keeps its five-parameter signature and its committed PTX.
// The types whose batch-1 matvec gives each lane a whole quant block and
// stages the activations once per block of rows (X21, X21b): launched eight
// rows a block. The others keep four.
static bool lane_staged_type(int type) {
    switch (type) {
    case T_Q4_0: case T_MXFP4: case T_IQ4_NL: case T_NVFP4: case T_Q2_K:
    case T_IQ1_S: case T_IQ1_M: case T_Q4_1: case T_Q5_0: case T_Q5_1: case T_Q2_0:
        return true;
    default:
        return false;
    }
}

// ytile: the kernel takes its column tile from blockIdx.y (the _b tile
// matvecs, MV_HEAD_B), so a batch wider than one tile is one launch
static bool launch_tiled(gpu_t *g, CUfunction f, unsigned grid, unsigned block,
                         CUdeviceptr weights, CUdeviceptr x, CUdeviceptr y,
                         mv_args a, CUdeviceptr bias, int tile,
                         bool scaled, float scale, bool ytile) {
    if (a.batch <= tile || ytile) {
        void *p[] = { &weights, &x, &y, &a, &bias, &scale };
        // one grid row for a batch up to a tile, an empty one included (no
        // caller passes 0 columns; a 0-row grid would be a launch error where
        // the one-row launch is the no-op it was before ytile)
        unsigned rows = a.batch > tile ? (unsigned)((a.batch + tile - 1) / tile) : 1;
        return launch(g, f, grid, rows, 1, block, p);
    }
    for (int off = 0; off < a.batch; off += tile) {
        mv_args ai = a;
        ai.batch = (a.batch - off) < tile ? (a.batch - off) : tile;
        CUdeviceptr xi = x + (CUdeviceptr)((uint64_t)off * (uint64_t)a.xs * sizeof(float));
        CUdeviceptr yi = y + (CUdeviceptr)((uint64_t)off * (uint64_t)a.ys * sizeof(float));
        void *pi[] = { &weights, &xi, &yi, &ai, &bias, &scale };
        if (!launch(g, f, grid, 1, 1, block, pi)) return false;
    }
    (void)scaled;   // the extra entry is harmless for a five-parameter kernel
    return true;
}

// Every tensor-core GEMM stages each token column scaled by 2^14 / max|x|
// (see TC_GEMM_BLK_X in kernels.cu; the codebook family from the start,
// Q4_K/Q6_K/Q8_0/Q4_0 since the same day's follow-up), so every tile of
// TC_N columns is preceded by k_colabsmax into g->xsc, and the kernel takes
// that pointer as its trailing parameter.
static bool launch_tiled_xscaled(gpu_t *g, CUfunction f, unsigned grid,
                                 unsigned block, CUdeviceptr weights,
                                 CUdeviceptr x, CUdeviceptr y, mv_args a,
                                 CUdeviceptr bias) {
    for (int off = 0; off < a.batch; off += TC_N) {
        mv_args ai = a;
        ai.batch = (a.batch - off) < TC_N ? (a.batch - off) : TC_N;
        CUdeviceptr xi = x + (CUdeviceptr)((uint64_t)off * (uint64_t)a.xs * sizeof(float));
        CUdeviceptr yi = y + (CUdeviceptr)((uint64_t)off * (uint64_t)a.ys * sizeof(float));
        void *pm[] = { &xi, &g->xsc, &ai.n_in, &ai.xs, &ai.batch };
        g->xh_src = 0;                   // xsc no longer holds TC2's staging
        if (!launch(g, g->sw->f_colabsmax, (unsigned)ai.batch, 1, 1, 256, pm))
            return false;
        void *pi[] = { &weights, &xi, &yi, &ai, &bias, &g->xsc };
        if (!launch(g, f, grid, 1, 1, block, pi)) return false;
    }
    return true;
}

static bool enc_mv_batch(gpu_t *g, model_t *m, gguf_tensor *w, CUdeviceptr x,
                         CUdeviceptr y, int n_in, int n_out, CUdeviceptr bias,
                         int batch, int xs, int ys);
// RUNNER_CUDA_Q8X=1: the int8-activation codebook GEMVs (EXPERIMENTAL; not
// the float route's bits, tolerance-gated against the CPU)
static bool gpu_q8x_on(void) {
    static _Atomic int on = -1;
    if (on < 0) { const char *e = getenv("RUNNER_CUDA_Q8X"); on = e && *e && *e != '0'; }
    return on == 1;
}

// The arithmetic routes' pin (gpu.h GR_*): -1 leaves each route to its
// RUNNER_CUDA_* diagnostic and its default; a mask set by --gpu-route or by
// --verify from a record decides every one of them instead.
static _Atomic int g_route_pin = -1;
void gpu_route_pin(int mask) { g_route_pin = mask < 0 ? -1 : (mask & GR_ALL); }
// RUNNER_CUDA_ROUTE: the same set as an environment setting, under the
// command line's pin and over the per-route variables, so a harness that
// compares CPU and GPU bytes pins every arithmetic route with one variable
// (RUNNER_CUDA_ROUTE=reference beside RUNNER_CUDA_TC=0). An unknown value is
// said once and ignored.
static int route_env(void) {
    static _Atomic int env = -2;    // -2 unread, -1 unset or invalid
    if (env == -2) {
        const char *e = getenv("RUNNER_CUDA_ROUTE");
        int v = e && *e ? gpu_route_parse(e) : -1;
        if (e && *e && v < 0)
            fprintf(stderr, "gpu: RUNNER_CUDA_ROUTE=%s is not reference or a list of tc2, "
                    "sm80, attn-tile, attn-dec, attn-tc, moe-tc, moe-grouped; ignored\n", e);
        env = v;
    }
    return env;
}
static int route_pinned(int bit) {
    int p = g_route_pin;
    if (p < 0) p = route_env();
    return p < 0 ? -1 : (p & bit) != 0;
}

// The pipelined tensor-core GEMM (TC2, k_gemm2_*) wherever the TC route is
// on for the type. Three modes. Auto, the default: TC2 runs only through the
// compute_80 module's kernels (a compute capability 8.0+ device), whose fp32
// sums are the tensor-core GEMM's own, so its outputs are byte for byte the
// TC route's (chunked --score on sm_86 and sm_120); elsewhere the TC kernels
// run. Forced (RUNNER_CUDA_TC2=1, or tc2 in a route pin): the same where the
// compute_80 module loaded, and the compute_75 kernels where it did not
// (below 8.0, or RUNNER_CUDA_SM80=0), which round between the two halves of
// each 16-deep step and so differ from the TC route. Off (RUNNER_CUDA_TC2=0,
// or a pin without tc2).
enum { TC2_OFF = 0, TC2_FORCED = 1, TC2_AUTO = 2 };
static int tc2_mode(void) {
    if (route_pinned(GR_TC2) >= 0) return route_pinned(GR_TC2) ? TC2_FORCED : TC2_OFF;
    static _Atomic int mode = -1;
    if (mode < 0) {
        const char *e = getenv("RUNNER_CUDA_TC2");
        mode = !e || !*e ? TC2_AUTO : strcmp(e, "0") == 0 ? TC2_OFF : TC2_FORCED;
    }
    return mode;
}
// TC2 may run in this process (the module load and the window's charge ask)
static bool gpu_tc2_on(void) { return tc2_mode() != TC2_OFF; }
static bool tc2_forced(void) { return tc2_mode() == TC2_FORCED; }
static _Atomic unsigned long g_tc2_dispatches;
unsigned long gpu_tc2_dispatches(void) { return g_tc2_dispatches; }
static _Atomic unsigned long g_i8_dispatches;
unsigned long gpu_i8_dispatches(void) { return g_i8_dispatches; }
static _Atomic unsigned long g_tc2_reused;
unsigned long gpu_tc2_reused(void) { return g_tc2_reused; }
// RUNNER_CUDA_XH_REUSE=0: every TC2 GEMM stages its own columns (the A/B)
// RUNNER_CUDA_I8=1: the int8 tensor-core prefill (R3.14) for the types it
// has kernels for, on a compute capability 8.0+ device. A tolerance route
// (int8 activations per 32-element block), opt-in until a type passes its
// gates.
static bool gpu_i8_on(void) {
    static _Atomic int on = -1;
    if (on < 0) {
        const char *e = getenv("RUNNER_CUDA_I8");
        on = e && *e && strcmp(e, "0") != 0 && strcmp(e, "off") != 0;
    }
    return on > 0;
}

static bool xh_reuse_on(void) {
    static _Atomic int on = -1;
    if (on < 0) { const char *e = getenv("RUNNER_CUDA_XH_REUSE"); on = !(e && *e == '0'); }
    return on == 1;
}

// E1: RUNNER_CUDA_SM80=0 keeps TC2 on its compute_75 kernels on a device
// that could run the compute_80 ones (a diagnostic and A/B control)
static bool gpu_sm80_on(void) {
    if (route_pinned(GR_SM80) >= 0) return route_pinned(GR_SM80);
    static _Atomic int on = -1;
    if (on < 0) {
        const char *e = getenv("RUNNER_CUDA_SM80");
        on = !(e && e[0] == '0' && e[1] == 0);
    }
    return on == 1;
}
static _Atomic unsigned long g_tc2_80_dispatches;
unsigned long gpu_tc2_80_dispatches(void) { return g_tc2_80_dispatches; }
// the compute_80 module's 128-column tile for a GEMM over more than 64
// columns: byte-identical to the 64-column kernels, so on wherever they run;
// RUNNER_CUDA_TC2_WIDE=0 keeps the 64-column tile (an A/B control),
// gpu_tc2_wide_force overrides for a test (-1 returns to the switch)
static _Atomic int g_tc2_wide_force = -1;
void gpu_tc2_wide_force(int on) { g_tc2_wide_force = on < 0 ? -1 : on != 0; }
static bool gpu_tc2_wide_on(void) {
    if (g_tc2_wide_force >= 0) return g_tc2_wide_force == 1;
    static _Atomic int on = -1;
    if (on < 0) { const char *e = getenv("RUNNER_CUDA_TC2_WIDE"); on = !(e && *e && strcmp(e, "0") == 0); }
    return on == 1;
}
static _Atomic unsigned long g_tc2_wide_dispatches;
unsigned long gpu_tc2_wide_dispatches(void) { return g_tc2_wide_dispatches; }

// The TC2 kernel for the type on this instance, or NULL for the TC route.
// With the compute_80 module loaded (CC >= 8.0, RUNNER_CUDA_SM80 not 0) only
// its kernels, forced or auto, so a run recorded as tc2,sm80 replays the same
// kernels whichever mode wrote it; without it the compute_75 ones, forced only.
static CUfunction tc2_fn(const gpu_t *g, int type) {
    int mode = tc2_mode();
    if (mode == TC2_OFF) return NULL;
    if (g->sw->mod80) return g->sw->f_gemm2_80[type];
    return mode == TC2_FORCED ? g->sw->f_gemm2[type] : NULL;
}
// whether any TC2 kernel can run on this instance (its activation buffer)
static bool tc2_can_run(const gpu_t *g) {
    int mode = tc2_mode();
    return mode == TC2_FORCED || (mode == TC2_AUTO && g->sw && g->sw->mod80);
}

// C2.3: a prompt tile's recurrent steps (Gated DeltaNet conv + delta,
// Mamba-2 conv + scan) as one launch per kernel over the whole tile
// (k_*_seq) instead of one launch per token. Byte-equal to the per-token
// kernels, so on by default; RUNNER_CUDA_RECUR_SEQ=0 keeps the per-token
// launches, gpu_recur_seq_force overrides for a test (-1 returns to the
// switch).
#define RS_MAX_HIST 7     // kernels.cu: conv history taps a thread keeps
#define RS_MAX_SD   128   // kernels.cu: state row a thread keeps
static _Atomic int g_recur_seq_force = -1;
void gpu_recur_seq_force(int on) { g_recur_seq_force = on < 0 ? -1 : on != 0; }
static bool recur_seq_on(void) {
    if (g_recur_seq_force >= 0) return g_recur_seq_force == 1;
    static _Atomic int on = -1;
    if (on < 0) {
        const char *e = getenv("RUNNER_CUDA_RECUR_SEQ");
        on = !(e && *e && strcmp(e, "0") == 0);
    }
    return on == 1;
}
static _Atomic unsigned long g_recur_seq_dispatches;
unsigned long gpu_recur_seq_dispatches(void) { return g_recur_seq_dispatches; }
// whether a prompt tile of this model's recurrent layers takes the whole-tile
// kernels: both dispatch sites and the route detail ask it (a speculative
// keep forward still takes the per-token loop, which files the snapshots)
static bool recur_seq_admits(const gpu_t *g, const model_t *m) {
    if (!recur_seq_on() || m->ssm.conv_kernel - 1 > RS_MAX_HIST || m->ssm.state > RS_MAX_SD)
        return false;
    return m->qwen35 ? g->sw->f_q35_conv_seq && g->sw->f_q35_delta_seq
                     : m->ssm.v_heads > 0 && g->sw->f_mamba_conv_seq && g->sw->f_mamba_ssd_seq;
}


// The batch-1 routes with no width-classed twin: the int8-activation GEMV
// (RUNNER_CUDA_Q8X) and BF16's split-K for a few long rows. enc_mv takes
// them in this order; a row of an exact-rows tile (a speculative verify
// round) on such a tensor runs the batch-1 call itself.
static bool mv_route_q8x(const gpu_t *g, const gguf_tensor *w, int n_in, int batch) {
    return batch >= 1 && batch <= 8 && w->scale == 1.0f && g->q8x_q &&
           n_in <= g->q8x_cap && n_in % 256 == 0 && g->sw->f_q8x[batch][w->type];
}
static bool mv_route_splitk(const gpu_t *g, const gguf_tensor *w, int n_in, int n_out) {
    return w->type == T_BF16 && w->scale == 1.0f && g->splitk &&
           n_out <= 1024 && n_in >= 2048 && g->sw->f_mv_bf16_splitk;
}
// The route the width-classed twins (f_gemvb, and the codebook i-quants'
// in-between f_gemvx widths) reproduce: the batch-1 kernel enc_mv picks for
// an unscaled tensor off the two routes above, f_gemv where a type has one and
// f_mv otherwise. A verify round's exact rows and the decode microbatch
// (batch_mv_twin_ok) both ask here.
static bool mv_twin_route(const gpu_t *g, const gguf_tensor *w, int n_in, int n_out) {
    return w->scale == 1.0f && !mv_route_q8x(g, w, n_in, 1) &&
           !mv_route_splitk(g, w, n_in, n_out);
}
static inline bool mvb_is_mv_twin(int type);
// Whether a prompt's matvec (batch > GPU_SMALL_BATCH) of w on the scalar
// route runs the batch-1 GEMV's 16-wide twins (enc_mv; scalar_mv_route
// answers the same for test-prefill-bits' route table). Every type whose batch-1
// route has a loaded 16-wide twin takes it, by capability rather than by a
// list (R6.57; the list was R6.35 Q4_0, R6.41 Q4_K/Q5_K/Q6_K, R6.50 Q8_0,
// R6.54 Q3_K): the tile kernels of the others (k_gemm_*, and f_mvb's
// k_mv_*_b, which folded the block scale per term, retired in R18.8.11)
// summed in another order. Not F32, F16 or BF16, whose f_mvb already is
// their batch-1 kernel's twin.
static bool prefill_takes_twins(const gpu_t *g, const gguf_tensor *w, int n_in, int n_out) {
    return !mvb_is_mv_twin(w->type) && mv_twin_route(g, w, n_in, n_out) &&
           g->sw->f_gemvb[BW_16][w->type];
}
// Whether a matvec of w over batch (> 1) columns on the scalar route runs the
// batch-1 kernel once per column (R6.58): every type with neither a 16-wide
// twin on its batch-1 route (prefill_takes_twins) nor a tile kernel that is
// the batch-1 kernel's twin (mvb_is_mv_twin). Their tiles (k_gemm_*, f_mvb's
// k_mv_*_b, retired in R18.8.11) summed in another order; the batch-1 call
// per column is its bits by
// construction, and prefilled 7 to 10x faster than the tile on Q5_0, Q5_1 and
// IQ4_NL files. Over 8 columns, and 2 to 8 of a scaled tensor (NVFP4's
// companion), which the small-batch branch does not take.
static bool prefill_per_column(const gpu_t *g, const gguf_tensor *w, int n_in, int n_out, int batch) {
    if (batch < 2 || mvb_is_mv_twin(w->type)) return false;
    if (batch <= GPU_SMALL_BATCH) return w->scale != 1.0f;
    return !prefill_takes_twins(g, w, n_in, n_out);
}
// Where a matvec of w over batch (> 1) columns goes on the scalar route, in
// enc_mv's order below its tensor-core branches: the small-batch branch, the
// 16-wide twins, the batch-1 kernel per column, the float tile twin f_mvb
// (gpu.h's GPU_PREFILL_* answers; test-prefill-bits' route table asks it).
// No route answers GPU_PREFILL_TILE since R18.8.11 retired the quantized
// tiles; the table still refuses it. The experimental int8-activation route
// (RUNNER_CUDA_Q8X) is not modelled.
static int scalar_mv_route(const gpu_t *g, const gguf_tensor *w, int n_in, int n_out, int batch) {
    const gpu_weights *sw = g->sw;
    if (batch < 2 || (!sw->f_mv[w->type] && !sw->f_gemv[w->type])) return GPU_PREFILL_NONE;
    if (batch <= GPU_SMALL_BATCH && w->scale == 1.0f)
        return sw->f_gemvb[batch_width_class(batch)][w->type] ? GPU_PREFILL_TWINS
                                                              : GPU_PREFILL_PER_COLUMN;
    if (batch > GPU_SMALL_BATCH && prefill_takes_twins(g, w, n_in, n_out)) return GPU_PREFILL_TWINS;
    if (prefill_per_column(g, w, n_in, n_out, batch)) return GPU_PREFILL_PER_COLUMN;
    return mvb_is_mv_twin(w->type) && sw->f_mvb[w->type] ? GPU_PREFILL_TILE_TWIN : GPU_PREFILL_NONE;
}
// gpu.h's questions for the route table, asked of a dense 4096 x 4096 tensor
// of the type: is the 16-wide twin of its batch-1 route loaded, and where does
// a matvec of it over batch columns go (scaled: with a per-tensor companion)
static bool prefill_probe(const model_t *m, int type, gguf_tensor *w) {
    const gpu_t *g = m->gpu;
    if (!g || !g->sw || type < 0 || type >= KT_N) return false;
    memset(w, 0, sizeof(*w));
    w->type = type;
    w->scale = 1.0f;
    return true;
}
bool gpu_prefill_twin16(const model_t *m, int type) {
    gguf_tensor w;
    if (!prefill_probe(m, type, &w)) return false;
    const gpu_t *g = m->gpu;
    return mv_twin_route(g, &w, 4096, 4096) && g->sw->f_gemvb[BW_16][type];
}
int gpu_prefill_route(const model_t *m, int type, int batch, bool scaled) {
    gguf_tensor w;
    if (!prefill_probe(m, type, &w)) return GPU_PREFILL_NONE;
    if (scaled) w.scale = 0.5f;
    const gpu_t *g = m->gpu;
    return scalar_mv_route(g, &w, 4096, 4096, batch);
}
static bool batch_mv_twin_ok(const gpu_t *g, const gguf_tensor *t);

static bool enc_scale(gpu_t *g, CUdeviceptr x, float s, int n, int batch, int xs);
static bool enc_mv(gpu_t *g, model_t *m, gguf_tensor *w, CUdeviceptr x,
                   CUdeviceptr y, int n_in, int n_out, CUdeviceptr bias,
                   int batch, int xs, int ys) {
    unsigned long seq = ++g->mv_seq;
    // Exact rows (a speculative verify round, model_forward_verify): every
    // column the batch-1 call's bits. Up to sixteen columns a pass through
    // the width-classed twins where the batch-1 route is the GEMV they twin
    // (the decode microbatch's contract; eight where a type has no 16-wide
    // twin), else the batch-1 call a column.
    if (batch > 1 && g->exact_rows) {
        bool twin = mv_twin_route(g, w, n_in, n_out);
        int step = g->sw->f_gemvb[BW_16][w->type] ? 16 : GPU_SMALL_BATCH;
        for (int c0 = 0; c0 < batch; c0 += step) {
            int cn = batch - c0 < step ? batch - c0 : step;
            CUdeviceptr xc = x + (size_t)c0 * xs * sizeof(float);
            CUdeviceptr yc = y + (size_t)c0 * ys * sizeof(float);
            if (cn > 1 && twin && g->sw->f_gemvb[batch_width_class(cn)][w->type]) {
                if (!enc_mv_batch(g, m, w, xc, yc, n_in, n_out, bias, cn, xs, ys))
                    return false;
                continue;
            }
            for (int c = 0; c < cn; c++)
                if (!enc_mv(g, m, w, xc + (size_t)c * xs * sizeof(float),
                            yc + (size_t)c * ys * sizeof(float), n_in, n_out, bias,
                            1, xs, ys))
                    return false;
        }
        return true;
    }
    CUdeviceptr weights = 0;
    uint64_t w_off = 0;
    if (!binding_find(g->sw, m, w, &weights, &w_off)) {
        gpu_first_error("tensor '%s' is not resident on the device "
                        "(no binding covers it)", w->name);
        return false;
    }
    mv_args a = { n_in, n_out, w_off, bias != 0, batch, xs, ys };
    CUdeviceptr b = bias ? bias : g->sw->dummy;
    void *p[] = { &weights, &x, &y, &a, &b };
    // RUNNER_CUDA_Q8X: quantize x to int8 once (per 32), then the dp4a GEMV;
    // batch 1..8 of IQ3_S / IQ3_XXS / IQ4_XS
    if (mv_route_q8x(g, w, n_in, batch)) {
        int nbk = n_in / 32 * batch;
        void *pq[] = { &x, &g->q8x_q, &g->q8x_d, &n_in, &batch, &xs };
        if (!launch(g, g->sw->f_quant_q8x, (nbk + 3) / 4, 1, 1, 128, pq)) return false;
        void *pm[] = { &weights, &g->q8x_q, &g->q8x_d, &y, &a, &b };
        unsigned rpb = batch == 1 ? 8 : 4;
        return launch(g, g->sw->f_q8x[batch][w->type], (n_out + rpb - 1) / rpb, 1, 1, 128, pm);
    }
    // Small batch (1 < batch <= GPU_SMALL_BATCH): a speculative verify tile, a
    // short prompt, a few rows of one expert. The tile kernels are shaped for
    // 64-row tiles and barely get cheaper with fewer rows. The width-classed
    // twins stream each weight once and reproduce the batch-1 bits per column; a
    // type without a twin takes the batch-1 kernel once per column.
    if (batch > 1 && batch <= GPU_SMALL_BATCH && w->scale == 1.0f) {
        if (g->sw->f_gemvb[batch_width_class(batch)][w->type])
            return enc_mv_batch(g, m, w, x, y, n_in, n_out, bias, batch, xs, ys);
        for (int c = 0; c < batch; c++)
            if (!enc_mv(g, m, w, x + (size_t)c * xs * sizeof(float),
                        y + (size_t)c * ys * sizeof(float), n_in, n_out, bias,
                        1, xs, ys))
                return false;
        return true;
    }
    // Prefill (batch>1) on the int8 route (RUNNER_CUDA_I8, R3.14), the
    // tensor-core route's int8 variant: where the TC route is on for the type
    // (tc_on, so a forced-off reference stays scalar), the window's columns
    // quantized to int8 per 32-element block once, then one launch of the
    // type's int8 GEMM over every column. Counted as a tensor-core dispatch.
    if (batch > 1 && w->scale == 1.0f && g->i8_xq && g->sw->f_quant_i8 && tc_on(m, w->type) &&
        g->sw->f_gemm_i8[w->type] && n_in % I8_BK == 0 && n_in <= g->i8_cap &&
        batch <= g->win) {
        int nbk = n_in / 32 * batch;
        void *pq[] = { &x, &g->i8_xq, &g->i8_xd, &g->i8_xs, &n_in, &batch, &xs };
        if (!launch(g, g->sw->f_quant_i8, (unsigned)((nbk + 7) / 8), 1, 1, 256, pq))
            return false;
        void *pg[] = { &weights, &g->i8_xq, &g->i8_xd, &g->i8_xs, &y, &a, &b };
        g_i8_dispatches++;
        g_tc_dispatches++;
        g_tc_dispatches_type[w->type]++;
        return launch_smem(g, g->sw->f_gemm_i8[w->type], (unsigned)((n_out + I8_BM - 1) / I8_BM),
                           (unsigned)((batch + I8_BN - 1) / I8_BN), 1, 256, I8_SMEM, pg);
    }
    // Prefill (batch>1) on the tensor-core route, the pipelined TC2 GEMM where
    // it has a kernel for the type: the window's column maxima and its fp16
    // activations once, then one launch over every column (kernels.cu TC2).
    // NVFP4's two-level files carry a per-tensor scale (the X.scale companion):
    // the tensor-core GEMMs read the blocks alone, so their output takes the
    // scale after, where no bias has to be ordered against it (R4.9.5)
    bool tc_scale_ok = w->scale == 1.0f || (w->type == T_NVFP4 && !bias);
    CUfunction f2 = batch > 1 && tc_scale_ok && g->xh ? tc2_fn(g, w->type) : NULL;
    if (f2 && tc_on(m, w->type) && n_in % G2_BK == 0 &&
        n_in <= g->xh_cap && batch <= g->win) {
        if (g->xh_reuse && xh_reuse_on() && g->xh_seq == seq - 1 && g->xh_src == x &&
            g->xh_n == n_in && g->xh_xs == xs && g->xh_batch == batch) {
            g_tc2_reused++;
        } else {
            void *pm[] = { &x, &g->xsc, &n_in, &xs, &batch };
            if (!launch(g, g->sw->f_colabsmax, (unsigned)batch, 1, 1, 256, pm)) return false;
            void *ph[] = { &x, &g->xsc, &g->xh, &n_in, &xs, &batch };
            if (!launch(g, g->sw->f_x_half, (unsigned)((n_in + 255) / 256), (unsigned)batch,
                        1, 256, ph))
                return false;
            g->xh_src = x; g->xh_n = n_in; g->xh_xs = xs; g->xh_batch = batch;
        }
        g->xh_seq = seq;
        g_tc_dispatches++;
        g_tc_dispatches_type[w->type]++;
        g_tc2_dispatches++;
        int bn = G2_BN;
        unsigned smem = G2_SMEM;
        if (f2 == g->sw->f_gemm2_80[w->type]) {
            g_tc2_80_dispatches++;
            // the 128-column twin for a GEMM over 64 columns: same outputs
            if (batch > G2_BN && g->sw->f_gemm2_80w[w->type] && gpu_tc2_wide_on()) {
                f2 = g->sw->f_gemm2_80w[w->type];
                bn = G2_BN_W;
                smem = G2_SMEM_W;
                g_tc2_wide_dispatches++;
            }
        }
        void *pg[] = { &weights, &g->xh, &y, &a, &b, &g->xsc };
        if (!launch_smem(g, f2, (unsigned)((n_out + G2_BM - 1) / G2_BM),
                         (unsigned)((batch + bn - 1) / bn), 1, 256, smem, pg))
            return false;
        return w->scale == 1.0f || enc_scale(g, y, w->scale, n_out, batch, ys);
    }
    // Prefill (batch>1), tensor-core GEMM when promoted for this (type, arch)
    // or forced by RUNNER_CUDA_TC: the block dequantizes a 64-row fp16 weight
    // tile once and its four warps' MMAs share it (TC_ROWS/block, 128 threads).
    if (batch > 1 && tc_scale_ok && tc_on(m, w->type) && g->sw->f_gemm_tc[w->type]) {
        g_tc_dispatches++;
        g_tc_dispatches_type[w->type]++;
        if (!launch_tiled_xscaled(g, g->sw->f_gemm_tc[w->type],
                                  (n_out + TC_ROWS - 1) / TC_ROWS, 128,
                                  weights, x, y, a, b))
            return false;
        return w->scale == 1.0f || enc_scale(g, y, w->scale, n_out, batch, ys);
    }
    // Prefill (batch > GPU_SMALL_BATCH) on the scalar route of a tensor whose
    // batch-1 route has a loaded 16-wide twin (prefill_takes_twins), ahead of
    // the tiled GEMM. The tile kernels sum in another order than the batch-1
    // kernels, so a prompt's rows would not carry a lone decode's bits, and
    // under a q8_0 KV cache that difference crosses quantization steps. The
    // 16-wide twin runs 16 columns a pass, a lone last column the batch-1
    // kernel. test-prefill-bits holds the rows a prompt stores and the route
    // table.
    if (batch > GPU_SMALL_BATCH && prefill_takes_twins(g, w, n_in, n_out)) {
        for (int c0 = 0; c0 < batch; c0 += 16) {
            int cn = batch - c0 < 16 ? batch - c0 : 16;
            CUdeviceptr xc = x + (size_t)c0 * xs * sizeof(float);
            CUdeviceptr yc = y + (size_t)c0 * ys * sizeof(float);
            if (!(cn == 1 ? enc_mv(g, m, w, xc, yc, n_in, n_out, bias, 1, xs, ys)
                          : enc_mv_batch(g, m, w, xc, yc, n_in, n_out, bias, cn, xs, ys)))
                return false;
        }
        return true;
    }
    // Prefill on the scalar route of a type with no twin of its batch-1 kernel
    // (prefill_per_column), and a scaled tensor's 2 to 8 columns: the batch-1
    // kernel once per column, each column its bits. The tile kernels below sum
    // in another order and ran 7 to 10x slower.
    if (prefill_per_column(g, w, n_in, n_out, batch)) {
        for (int c = 0; c < batch; c++)
            if (!enc_mv(g, m, w, x + (size_t)c * xs * sizeof(float),
                        y + (size_t)c * ys * sizeof(float), n_in, n_out, bias, 1, xs, ys))
                return false;
        return true;
    }
    // Decode (batch==1), few long BF16 rows (n_out <= 1024, n_in >= 2048):
    // split-K, see k_mv_bf16_splitk. splits is a power of two dividing n_in/32
    // with n_out*splits around 4,096 blocks, bounded by the partials buffer.
    if (batch == 1 && mv_route_splitk(g, w, n_in, n_out)) {
        int splits = 1;
        while (splits < 64 && n_out * splits * 2 <= 4096 &&
               (n_in % (splits * 2 * 32)) == 0 &&
               (size_t)n_out * splits * 2 <= SPLITK_PARTIALS) splits *= 2;
        void *ps[] = { &weights, &x, &g->splitk, &a, &splits };
        g_splitk_dispatches++;
        if (!launch(g, g->sw->f_mv_bf16_splitk, n_out, splits, 1, 256, ps)) return false;
        int hb = bias != 0;
        void *pr[] = { &g->splitk, &y, &b, &n_out, &splits, &hb };
        return launch(g, g->sw->f_splitk_reduce, (n_out + 255) / 256, 1, 1, 256, pr);
    }
    // Decode (batch==1) uses the coalesced lane-per-element GEMV where available
    // (Q8_0/Q4_K); same 4-rows/block shape, so capture-compatible with no
    // host-side branching on per-token state.
    if (batch == 1 && w->scale == 1.0f && g->sw->f_gemv[w->type])
        return launch(g, g->sw->f_gemv[w->type], (n_out + 3) / 4, 1, 1, 128, p);
    // 128 threads = 4 warps = 4 rows per block; the tile variant applies each
    // decoded weight to all columns, the single variant is faster at batch 1.
    // Over one column only F32, F16 and BF16 reach here (their f_mvb is the
    // batch-1 kernel's twin): every other type took a branch above.
    CUfunction f = batch > 1 ? g->sw->f_mvb[w->type] : g->sw->f_mv[w->type];
    if (batch > 1 && !f) {
        gpu_first_error("tensor '%s' has no tile twin for %d columns; its prompt "
                        "should have taken the per-column route", w->name, batch);
        return false;
    }
    // the lane-level i-quant batch-1 kernels take IQ_ROWS (2) rows per warp
    unsigned rpb = 4, block = 128;
    if (batch == 1 && (w->type == T_IQ2_XXS || w->type == T_IQ2_XS || w->type == T_IQ2_S ||
                       w->type == T_IQ3_XXS || w->type == T_IQ3_S || w->type == T_IQ4_XS))
        rpb = 4 * 2;
    // the one-lane-per-block batch-1 kernels stage the column's activations
    // once per block of rows (kernels.cu, LANE_STAGE_XN), so eight rows a
    // block share each copy; every row is computed the same either way
    if (batch == 1 && lane_staged_type(w->type)) { rpb = 8; block = 256; }
    return launch_tiled(g, f, (n_out + rpb - 1) / rpb, block, weights, x, y, a, b,
                        batch > 1 ? MVT : MVB,
                        w->type == T_NVFP4, w->scale, batch > 1);
}

// ------------------------------------------- adapter at a projection site
// y += scale * B(Ax), the device twin of model.c's LORA_HOOK. The host hook
// runs beside the CPU projections; on a GPU-resident layer the activations
// never reach the host, so the same delta runs here from device-resident A
// and B. Returns true unchanged when this (layer, slot) carries no adapter,
// which is every site of an unadapted model and the whole cost of the check.
//
// Capture-safe: the only host-side branch is on the rank, which is fixed for
// the model's lifetime, so a decode graph captured with an adapter replays
// with the same launches.
static bool enc_lora(gpu_t *g, int l, int slot, CUdeviceptr x, CUdeviceptr y,
                     int n_in, int n_out, int batch, int xs, int ys) {
    if (!g->lora_a) return true;
    size_t idx = (size_t)l * LW_SLOTS + (size_t)slot;
    CUdeviceptr A = g->lora_a[idx], B = g->lora_b[idx];
    if (!A || !B) return true;
    int r = g->lora_r_of[idx];
    float scale = g->lora_scale[idx];
    void *pa[] = { &A, &x, &g->lora_t, &n_in, &r, &xs };
    if (!launch(g, g->sw->f_lora_a, (unsigned)r, (unsigned)batch, 1, 128, pa))
        return false;
    void *pb[] = { &B, &g->lora_t, &y, &n_out, &r, &ys, &scale };
    return launch(g, g->sw->f_lora_b, (unsigned)((n_out + 127) / 128),
                  (unsigned)batch, 1, 128, pb);
}

static bool enc_qknorm(gpu_t *g, model_t *m, CUdeviceptr v, CUdeviceptr w,
                       int n_heads, int hd, int batch, int vs) {
    float eps = m->rms_eps;
    void *p[] = { &v, &w, &hd, &eps, &vs };
    return launch(g, g->sw->f_qknorm, n_heads, batch, 1, 64, p);
}

static bool enc_rope(gpu_t *g, model_t *m, CUdeviceptr v, int n_heads,
                     int batch, int vs, int l) {
    // sliding-window layers rope at their own base with no YaRN scale;
    // heterogeneous archs (gemma4) also rotate fewer dims on local layers
    bool local = model_is_swa(m, l);
    rope_args a = { model_head_dim(m, l), n_heads, model_rope_dim(m, l) / 2,
                    m->rope_neox, model_rope_mscale(m, l) };
    CUdeviceptr fr = local ? g->sw->inv_freq_local : g->sw->inv_freq;
    void *p[] = { &v, &fr, &a, &g->pos_dev, &vs };
    return launch(g, g->sw->f_rope, (a.half_dim + 31) / 32, n_heads, batch, 32, p);
}

static bool enc_scale(gpu_t *g, CUdeviceptr x, float s, int n, int batch, int xs) {
    void *p[] = { &x, &s, &n, &xs };
    return launch(g, g->sw->f_scale, (n + 255) / 256, batch, 1, 256, p);
}

// Rope and the attention temperature, mirroring the CPU tail. llama 4's NoPE
// layer skips rope and instead scales Q by a per-token factor that grows with
// position; mistral3 scales every layer's Q by it after the rotation
// (model_temp_applies). The factor differs per token, so this is one
// enc_scale per column rather than one for the tile. Only reached by a model
// that declares the knob.
static bool enc_rope_or_temp(gpu_t *g, model_t *m, CUdeviceptr q, int n_heads,
                             int tn, int q_dim, int l, int pos,
                             bool (*rope)(gpu_t *, model_t *, CUdeviceptr, int, int, int, int)) {
    if (model_layer_ropes(m, l) && !rope(g, m, q, n_heads, tn, q_dim, l)) return false;
    if (!model_temp_applies(m, l)) return true;
    for (int b = 0; b < tn; b++) {
        float ts = model_attn_temp(m, pos + b);
        if (ts == 1.0f) continue;
        CUdeviceptr col = q + (CUdeviceptr)((size_t)b * q_dim * sizeof(float));
        if (!enc_scale(g, col, ts, q_dim, 1, q_dim)) return false;
    }
    return true;
}


static bool enc_add(gpu_t *g, CUdeviceptr x, CUdeviceptr d, int n,
                    int batch, int xs, int ds) {
    void *p[] = { &x, &d, &n, &xs, &ds };
    return launch(g, g->sw->f_add, (n + 255) / 256, batch, 1, 256, p);
}

// gemma4 E-series per-layer embedding branch, mirroring the CPU tail in
// model.c: gate the post-FFN residual into the PLE width, multiply it by this
// layer's slice of the per-token table, project back and RMS-norm, then add.
// Runs after the FFN residual and before the layer output scale.
//
// xb is free here (its FFN output already went into x) and doubles as the
// projection destination, matching what the CPU path reuses.
static bool enc_ple(gpu_t *g, model_t *m, const layer_t *ly, int l,
                    int tn, int xdim) {
    if (!ly->ple_gate) return true;
    int P = m->n_embd_ple, n_embd = m->n_embd;
    size_t per_tok = (size_t)m->n_layer * P;
    if (!enc_mv(g, m, ly->ple_gate, g->x, g->ple_g, n_embd, P, 0,
                tn, n_embd, P))
        return false;
    // k_gelu_mul is gelu(a)*b elementwise in place on a — the same tanh
    // approximation gated_act(ACT_GELU, ...) uses on the host. The slice
    // stride is per_tok, so each token reads its own layer-l window.
    for (int b = 0; b < tn; b++) {
        CUdeviceptr gate = g->ple_g + (CUdeviceptr)((size_t)b * P * sizeof(float));
        CUdeviceptr slice = g->ple +
            (CUdeviceptr)(((size_t)b * per_tok + (size_t)l * P) * sizeof(float));
        void *p[] = { &gate, &slice, &P };
        if (!launch(g, g->sw->f_gelu, (P + 255) / 256, 1, 1, 256, p))
            return false;
    }
    if (!enc_mv(g, m, ly->ple_proj, g->ple_g, g->xb, P, n_embd, 0,
                tn, P, xdim))
        return false;
    if (!enc_rmsnorm(g, g->xb, g->xb, g->sw->ple_pn[l], n_embd, m->rms_eps,
                     tn, xdim, xdim))
        return false;
    return enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);
}

static bool enc_actmul(gpu_t *g, model_t *m, CUdeviceptr a, CUdeviceptr b, int n) {
    void *p[] = { &a, &b, &n };
    CUfunction f = m->ffn_act == ACT_SWIGLU_OAI ? g->sw->f_swiglu
                 : m->ffn_act == ACT_GELU       ? g->sw->f_gelu
                 : m->ffn_act == ACT_GELU_ERF   ? g->sw->f_gelu_erf
                                                : g->sw->f_silu;
    return launch(g, f, (n + 255) / 256, 1, 1, 256, p);
}

// Apertus's dense FFN is ungated: up -> xIELU -> down. The four effective
// parameters were transformed once at model load and are passed by value for
// the current layer, matching the CPU xielu() helper without device tables.
static bool enc_relu2(gpu_t *g, CUdeviceptr x, int n) {
    void *p[] = { &x, &n };
    return launch(g, g->sw->f_relu2, (n + 255) / 256, 1, 1, 256, p);
}

static bool enc_xielu(gpu_t *g, CUdeviceptr x, int n,
                      float an, float ap, float b, float eps) {
    void *p[] = { &x, &n, &an, &ap, &b, &eps };
    return launch(g, g->sw->f_xielu, (n + 255) / 256, 1, 1, 256, p);
}

// Per-sequence teardown. The shared weights are not touched here beyond
// dropping this instance's reference: another slot may still be decoding
// against them, and only the last release frees them.
// The adapter's device copy, released by BOTH the public unbind (an adapter
// dropped from a live model) and context teardown (the model going away, where
// m->gpu is already NULL and the public entry point cannot reach it).
static void lora_dev_release(gpu_t *g) {
    for (int i = 0; i < g->lora_n; i++) {
        if (g->lora_a && g->lora_a[i]) cu.MemFree(g->lora_a[i]);
        if (g->lora_b && g->lora_b[i]) cu.MemFree(g->lora_b[i]);
    }
    if (g->lora_t) cu.MemFree(g->lora_t);
    free(g->lora_a);     g->lora_a = NULL;
    free(g->lora_b);     g->lora_b = NULL;
    free(g->lora_r_of);  g->lora_r_of = NULL;
    free(g->lora_scale); g->lora_scale = NULL;
    g->lora_t = 0;
    g->lora_r = 0;
    g->lora_n = 0;
}

static void gpu_ctx_free(model_t *m, gpu_t *g) {
    (void)m;
    if (!g) return;
    if (g->sw && g->sw->ctx) cu.CtxSetCurrent(g->sw->ctx);
    route_replay_flush(g);
    lora_dev_release(g);
    for (int i = 0; i < 2; i++) {
        if (g->gexec[i] && cu.GraphExecDestroy) cu.GraphExecDestroy(g->gexec[i]);
        if (g->graph[i] && cu.GraphDestroy) cu.GraphDestroy(g->graph[i]);
    }
    if (g->stream && cu.StreamDestroy) cu.StreamDestroy(g->stream);
    CUdeviceptr bufs[] = { g->kc, g->vc == g->kc ? 0 : g->vc, g->x, g->xb, g->xb2,
                           g->q, g->kt, g->vt, g->hb, g->hb2, g->att,
                           g->attn_part, g->logits, g->xsc, g->xh, g->pos_dev, g->q8x_q, g->q8x_d,
                           g->moe_logits, g->moe_eout, g->moe_sel,
                           g->moe_selw, g->moe_hb, g->moe_hb2, g->moe_hot_sel,
                           g->moe_gath, g->moe_dout, g->moe_gidx, g->moe_gw,
                           g->moe_gpos, g->moe_gsel, g->moe_slots,
                           g->g_scr, g->ple, g->ple_g,
                           g->q35_mix, g->q35_cv, g->q35_z, g->q35_beta,
                           g->q35_alpha, g->q35_gate, g->q35_hist,
                           g->q35_state, g->q35_hist_prev, g->q35_state_prev,
                           g->mtp_h, g->mtp_pending, g->mtp_cat, g->mtp_hid,
                           g->mtp_en, g->mtp_hn, g->mtp_rows, g->snap_state, g->snap_hist,
                           g->mamba_proj, g->mamba_xBC, g->mamba_y,
                           g->mamba_conv, g->mamba_state,
                           g->mamba_conv_prev,
                           // the hyper-connection streams, the split-K partials
                           // and the shared expert: missing from this list until
                           // 2026-10-20, so every such instance leaked them into
                           // the shared context (tests/test_cuda_teardown.c)
                           g->x_hc, g->hc_xn, g->hc_gate, g->hc_lo, g->hc_inject,
                           g->splitk, g->shexp_in, g->shexp_g, g->shexp_u,
                           g->shexp_o, g->shexp_gate, g->mla_buf, g->kvg_tab, g->kvg_dev, g->kvg_pos,
                           g->vcol_pos, g->vcol_kcp, g->vcol_vcp,
                           g->replay_tab, g->replay_cnt, g->i8_xq, g->i8_xd, g->i8_xs };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++)
        if (bufs[i]) cu.MemFree(bufs[i]);
    free(g->h_x); free(g->h_logits); free(g->h_moe_logits);
    free(g->h_ple); free(g->h_ple_tmp);
    free(g->h_moe_sel); free(g->h_moe_selw);
    free(g->h_hot_sel); free(g->h_hot_x); free(g->h_hot_terms);
    free(g->h_moe_gidx); free(g->h_moe_gw);
    free(g->tr_pos); free(g->tr_sel); free(g->tr_w); free(g->tr_norm);
    free(g->h_moe_gpos); free(g->h_moe_gsel);
    if (g->kvg_host) {
        if (g->kvg_pinned && cu.MemFreeHost) cu.MemFreeHost(g->kvg_host);
        else free(g->kvg_host);
    }
    free(g->h_kvg_tab);
    free(g->snap_ord);
    shared_release(g->sw);
    free(g);
}

static void prof_report_mode(int md, const char *label) {
    if (prof.n_tiles[md] == 0) return;
    double gpu_sum = 0;
    for (int i = 0; i < PH_N; i++) gpu_sum += prof.t_ph[md][i];
    double wall = prof.wall_ms[md];
    double nt = (double)prof.n_tiles[md], ntok = (double)prof.n_tokens[md];
    // On the plain path a synchronous cuMemcpyHtoD (inside stage_x) blocks on
    // g->stream, so the GPU-execution wait is charged to whichever host timer
    // issues the next blocking copy — it is NOT extra time. The GPU event sum
    // is the authoritative busy time; residual = wall - GPU-busy is the true
    // launch/host overhead (host copy timers overlap GPU and are shown raw).
    double residual = wall - gpu_sum;
    fprintf(stderr,
      "\n==== RUNNER_CUDA_PROFILE [%s] ====\n"
      "note: CUDA graphs are DISABLED while profiling — a replayed graph never\n"
      "      re-enters the instrumented launch path, so decode would report\n"
      "      0.0 ms for every phase. Decode wall time is therefore pessimistic\n"
      "      (it carries the launch overhead graphs remove); the phase SPLIT is\n"
      "      what these numbers are for.\n"
      "tiles=%lld tokens=%lld launches=%lld (%.1f/tile, %.1f/token)\n"
      "wall: %.1f ms | %.4f ms/tile | %.4f ms/token\n"
      "-- GPU phase (CUDA events, authoritative) --\n",
      label, prof.n_tiles[md], prof.n_tokens[md], prof.total_launches[md],
      prof.total_launches[md] / nt, prof.total_launches[md] / ntok,
      wall, wall / nt, wall / ntok);
    for (int i = 0; i < PH_N; i++)
        fprintf(stderr, "  %-14s %9.1f ms  %5.1f%%  %.4f ms/tile  %.4f ms/tok\n",
                PH_NAME[i], prof.t_ph[md][i],
                100.0 * prof.t_ph[md][i] / (gpu_sum ? gpu_sum : 1),
                prof.t_ph[md][i] / nt, prof.t_ph[md][i] / ntok);
    fprintf(stderr,
      "  %-14s %9.1f ms  (%.1f%% of wall)  %.4f ms/tile\n"
      "-- host-side raw QPC (overlaps GPU via sync memcpy) --\n"
      "  stage_x=%.1f  kv_copyback=%.1f  logitsDtoH=%.1f  sync-wait=%.1f ms\n"
      "  residual(wall-GPUbusy)=%.1f ms (%.1f%% of wall) = launch+host overhead\n",
      "GPU-TOTAL", gpu_sum, 100.0 * gpu_sum / (wall ? wall : 1), gpu_sum / nt,
      prof.t_stage[md], prof.t_copyback[md], prof.t_logitscp[md], prof.t_sync[md],
      residual, 100.0 * residual / (wall ? wall : 1));
}

static void prof_report(void) {
    if (!prof.on || !prof.inited) return;
    prof_report_mode(M_BATCH,  "PREFILL tile tn>1 (_b kernels)");
    prof_report_mode(M_SINGLE, "DECODE  tn==1 (k_mv_* kernels)");
    fprintf(stderr, "=============================\n");
    for (int i = 0; i < 4096; i++)
        if (prof.ev[i] && cu.EventDestroy) cu.EventDestroy(prof.ev[i]);
    prof.inited = 0;
}

void gpu_free(model_t *m) {
    prof_report();
    gpu_t *g = m->gpu;
    m->gpu = NULL;
    gpu_ctx_free(m, g);
}

// Move the device-resident recurrent layers' state between the device
// buffers and the host buffers (m->ssm_conv_state / m->ssm_state_mem), layer
// by layer at the same offsets, for the layers this context runs. `prev`
// selects the pre-forward rollback copy (the fallback path wants the state
// the failed forward started from); otherwise the live state. Per-layer sizes
// follow the allocation in gpu_init exactly: qwen35 keeps a [v_heads x state
// x state] DeltaNet matrix, Mamba-2 an [inner x state] one, and both keep the
// (conv_kernel - 1) x conv_dim ring. The host layout is the same (model.c
// recurrent_conv_bytes / recurrent_state_bytes).
// The base of half h of the double-buffered Mamba-2 SSD state.
static CUdeviceptr mamba_half(const gpu_t *g, const model_t *m, int h) {
    return g->mamba_state +
           (CUdeviceptr)h * sizeof(float) * model_rec_count(m) * m->ssm.inner * m->ssm.state;
}

static bool recurrent_xfer(gpu_t *g, model_t *m, bool to_host, bool prev) {
    CUdeviceptr dconv, dstate;
    size_t conv_layer, state_layer;
    if (m->qwen35) {
        int convdim = 2 * m->ssm.state * m->ssm.groups + m->ssm.inner;
        conv_layer  = (size_t)(m->ssm.conv_kernel - 1) * convdim;
        state_layer = (size_t)m->ssm.v_heads * m->ssm.state * m->ssm.state;
        dconv  = prev ? g->q35_hist_prev  : g->q35_hist;
        dstate = prev ? g->q35_state_prev : g->q35_state;
    } else if ((m->granite_hybrid || m->nemotron_h) && g->mamba_state) {
        int cd = m->ssm.inner + 2 * m->ssm.groups * m->ssm.state;
        conv_layer  = (size_t)(m->ssm.conv_kernel - 1) * cd;
        state_layer = (size_t)m->ssm.inner * m->ssm.state;
        dconv  = prev ? g->mamba_conv_prev  : g->mamba_conv;
        dstate = mamba_half(g, m, prev ? g->rec_prev : g->rec_cur);
    } else {
        return true;   // no device-resident fold in this context
    }
    for (int l = 0; l < m->gpu_layers; l++) {
        if (!m->layers[l].recurrent) continue;
        float *hconv  = m->ssm_conv_state + model_rec_slot(m, l) * conv_layer;
        float *hstate = m->ssm_state_mem + model_rec_slot(m, l) * state_layer;
        CUdeviceptr dc = dconv + model_rec_slot(m, l) * conv_layer * sizeof(float);
        CUdeviceptr ds = dstate + model_rec_slot(m, l) * state_layer * sizeof(float);
        if (to_host) {
            if (conv_layer && cu.MemcpyDtoH(hconv, dc, conv_layer * sizeof(float)) != 0)
                return false;
            if (cu.MemcpyDtoH(hstate, ds, state_layer * sizeof(float)) != 0)
                return false;
        } else {
            if (conv_layer && cu.MemcpyHtoD(dc, hconv, conv_layer * sizeof(float)) != 0)
                return false;
            if (cu.MemcpyHtoD(ds, hstate, state_layer * sizeof(float)) != 0)
                return false;
        }
    }
    return true;
}

// Turn mark (model_recurrent_mark): the live fold, both directions. The host
// buffers are the seam the mark copies through, so a mark on an offloaded
// recurrent slot costs one fold-sized PCIe copy per request at the prompt
// boundary and one more on the resume, both far below re-folding the prompt.
bool gpu_recurrent_download(model_t *m) {
    gpu_t *g = m ? m->gpu : NULL;
    if (!g) return true;
    if (!g->sw || !g->sw->ctx || cu.CtxSetCurrent(g->sw->ctx) != 0) return false;
    return recurrent_xfer(g, m, true, false);
}

bool gpu_recurrent_upload(model_t *m) {
    gpu_t *g = m ? m->gpu : NULL;
    if (!g) return true;
    if (!g->sw || !g->sw->ctx || cu.CtxSetCurrent(g->sw->ctx) != 0) return false;
    return recurrent_xfer(g, m, false, false);
}

// A runtime GPU failure on CUDA is fully recoverable on the host: the host KV
// cache is the authoritative copy and the device one is a mirror, so there is
// nothing to rescue before releasing. Freeing rather than orphaning matters
// now that weights are shared — an abandoned context would keep every other
// slot's copy of them alive too.
static bool fwd_tile(gpu_t *g, model_t *m, const int32_t *tokens, int tn,
                     int pos, bool want_logits, int l0, int l1);
static bool gpu_hc_mix(gpu_t *g, model_t *m, CUdeviceptr w_norm,
                       gguf_tensor *w_down, gguf_tensor *w_up, gguf_tensor *w_inject,
                       CUdeviceptr x_hc_rows, CUdeviceptr y, int ys, int tn);
// ---- NextN/MTP head on the device ----------------------------------------
// The CPU head's twin (model.c mtp_run): per queued pair (h, tok) the input
// per stream is [enorm(embed(tok)) ; hnorm_c(h_c)], projected by eh_proj into
// the block's wide residual, the block runs through fwd_tile at index
// n_layer (its own KV region, its experts where --cpu-moe left them), the
// next chained step reads the wide residual and the logits read it through
// the block's own stream mixer. Hidden rows never leave the device: the
// trunk's x_hc rows are copied into the queue (gpu_mtp_note_row) and the
// chained hidden stays in mtp_hid.
bool gpu_mtp_bound(const model_t *m) {
    gpu_t *g = m ? m->gpu : NULL;
    return g && g->mtp_h && m->mtp_ready && m->gpu_layers >= m->n_layer;
}

bool gpu_mtp_note_row(model_t *m, int row) {
    gpu_t *g = m->gpu;
    if (!gpu_mtp_bound(m) || row < 0 || row >= m->n_batch) return false;
    size_t hcd = (size_t)mtp_streams(m) * m->n_embd * sizeof(float);
    if (cu.CtxSetCurrent(g->sw->ctx) != 0) return false;
    // mtp_rows holds every tile of the last forward (x_hc only the last tile)
    return cu.MemcpyDtoD(g->mtp_pending, g->mtp_rows + (size_t)row * hcd, hcd) == 0;
}

// after a trunk tile: file its final residual rows (wide under hyper-connections) for the head
static bool mtp_save_rows(gpu_t *g, model_t *m, int i, int tn) {
    if (!g->mtp_rows) return true;
    size_t hcd = (size_t)mtp_streams(m) * m->n_embd * sizeof(float);
    if (i + tn > m->n_batch) return true;
    if (cu.StreamSynchronize(g->stream) != 0) return false;   // the tile's kernels first
    return cu.MemcpyDtoD(g->mtp_rows + (size_t)i * hcd, m->hyper_conn ? g->x_hc : g->x,
                         (size_t)tn * hcd) == 0;
}

bool gpu_mtp_queue_pending(model_t *m, int slot) {
    gpu_t *g = m->gpu;
    if (!gpu_mtp_bound(m) || slot < 0 || slot >= m->n_batch) return false;
    size_t hcd = (size_t)mtp_streams(m) * m->n_embd * sizeof(float);
    if (cu.CtxSetCurrent(g->sw->ctx) != 0) return false;
    return cu.MemcpyDtoD(g->mtp_h + (size_t)slot * hcd, g->mtp_pending, hcd) == 0;
}

bool gpu_mtp_reset_pending(model_t *m) {
    gpu_t *g = m->gpu;
    if (!gpu_mtp_bound(m)) return false;
    size_t hcd = (size_t)mtp_streams(m) * m->n_embd * sizeof(float);
    if (cu.CtxSetCurrent(g->sw->ctx) != 0) return false;
    return cu.MemsetD8(g->mtp_pending, 0, hcd) == 0;
}

// Run n queued pairs (h from mtp_h, or the chained mtp_hid when chained) at
// positions pos..pos+n-1; want_logits leaves the last row's head logits in
// m->mtp_logits (host) and the last wide residual in mtp_hid.
static bool set_pos(gpu_t *g, int pos, int tn);
bool gpu_mtp_run(model_t *m, bool chained, int slot0, const int32_t *tok, int n, int pos,
                 bool want_logits) {
    gpu_t *g = m->gpu;
    if (!gpu_mtp_bound(m) || n < 1 || n > g->win || (chained && n != 1) ||
        slot0 < 0 || slot0 + n > m->n_batch) return false;
    int E = m->n_embd, hc = mtp_streams(m), hcd = hc * E, l = m->n_layer;
    // the block's residual: the wide x_hc, or the narrow x of a plain model
    CUdeviceptr res = m->hyper_conn ? g->x_hc : g->x;
    if (cu.CtxSetCurrent(g->sw->ctx) != 0) return false;
    // embeddings of the pair tokens, dequantized on the host like stage_x
    size_t ers = ggml_row_size(m->mtp_embd->type, E);
    for (int b = 0; b < n; b++) {
        float *hx = g->h_x + (size_t)b * E;
        int32_t id = tok[b];
        if (id < 0 || id >= m->n_vocab) id = 0;
        dequant_row(m->mtp_embd->type, (uint8_t *)m->mtp_embd->data + (size_t)id * ers, hx, E);
        if (m->mtp_embd->scale != 1.0f) for (int i = 0; i < E; i++) hx[i] *= m->mtp_embd->scale;
    }
    if (cu.MemcpyHtoD(g->mtp_en, g->h_x, sizeof(float) * (size_t)n * E) != 0 ||
        !set_pos(g, pos, n))   // the block's rows start at pos
        return false;
    CUdeviceptr hsrc = chained ? g->mtp_hid : g->mtp_h + (size_t)slot0 * hcd * sizeof(float);
    // the normed embedding goes through xb (not in place: the norm kernel's
    // read and write loops are not guaranteed ordered against each other)
    if (!enc_rmsnorm(g, g->mtp_en, g->xb, g->sw->mtp_enorm, E, m->rms_eps, n, E, E))
        return false;
    {
        float eps = m->rms_eps; int xs = hcd;
        void *pn[] = { &hsrc, &g->mtp_hn, &g->sw->mtp_hnorm, &E, &hc, &eps, &xs, &xs };
        if (!launch(g, g->sw->f_rmsnorm_grouped, hc, n, 1, 256, pn)) return false;
    }
    {
        void *pc[] = { &g->xb, &g->mtp_hn, &g->mtp_cat, &E, &hc };
        if (!launch(g, g->sw->f_mtp_cat, hc, n, 1, 256, pc)) return false;
    }
    // eh_proj over the n*hc stream rows straight into the block's wide residual,
    // a tile of columns at a time (the batched matvecs take at most MVB columns)
    for (int r0 = 0; r0 < n * hc; r0 += MVB) {
        int rc = n * hc - r0 < MVB ? n * hc - r0 : MVB;
        if (!enc_mv(g, m, m->mtp_eh_proj, g->mtp_cat + (size_t)r0 * 2 * E * sizeof(float),
                    res + (size_t)r0 * E * sizeof(float), 2 * E, E, 0, rc, 2 * E, E))
            return false;
    }
    if (!fwd_tile(g, m, tok, n, pos, false, l, l + 1)) return false;
    CUdeviceptr last = res + (size_t)(n - 1) * hcd * sizeof(float);
    if (!m->hyper_conn) {
        // one stream (model.c mtp_run): the head norm writes the hidden the
        // next chained step pairs with, and the LM head reads it
        if (!want_logits) return cu.StreamSynchronize(g->stream) == 0;
        if (!enc_rmsnorm(g, last, g->mtp_hid, g->sw->mtp_head_norm, E, m->rms_eps, 1, E, E) ||
            !enc_mv(g, m, m->mtp_head, g->mtp_hid, g->logits, E, m->n_vocab, 0, 1, 0, 0) ||
            cu.StreamSynchronize(g->stream) != 0 ||
            cu.MemcpyDtoH(m->mtp_logits, g->logits, sizeof(float) * (size_t)m->n_vocab) != 0)
            return false;
        return true;
    }
    // the copies below are plain (default-stream) copies: wait for the block
    if (cu.StreamSynchronize(g->stream) != 0) return false;
    if (cu.MemcpyDtoD(g->mtp_hid, last, (size_t)hcd * sizeof(float)) != 0) return false;
    if (want_logits) {
        if (!gpu_hc_mix(g, m, g->sw->mtp_head_norm, m->mtp_hc_head_down, m->mtp_hc_head_up,
                        NULL, last, g->xb, 0, 1)) return false;
        if (!enc_mv(g, m, m->mtp_head, g->xb, g->logits, E, m->n_vocab, 0, 1, 0, 0)) return false;
        if (cu.StreamSynchronize(g->stream) != 0) return false;
        if (cu.MemcpyDtoH(m->mtp_logits, g->logits, sizeof(float) * (size_t)m->n_vocab) != 0)
            return false;
    } else if (cu.StreamSynchronize(g->stream) != 0) {
        return false;
    }
    // the block's KV rows are read only by the block itself on the device; a
    // CPU fallback mid-run drops them and the head re-syncs (model_mtp_reset)
    return true;
}

// Speculative rollback of the device fold: the forward that advanced it
// copied the pre-forward state to the *_prev buffers, so one forward after
// the snapshot the rollback is a device-to-device copy. More than one
// forward (or none) since the snapshot means the copy is stale: refuse, and
// the caller rebuilds the fold from the history.
bool gpu_recurrent_rollback_mark(model_t *m) {
    gpu_t *g = m ? m->gpu : NULL;
    if (!g) return false;
    g->fwd_snap = g->fwd_count;
    return true;
}

// Restore the device fold to its state after row r of the last keep-forward
// (r = 0 is after the pending token, r = acc after the last accepted draft):
// no re-fold. false = no checkpoint for that row (caller rolls back instead).
bool gpu_recurrent_restore_row(model_t *m, int r) {
    gpu_t *g = m ? m->gpu : NULL;
    if (!g || !g->snap_rows || r < 0 || r >= g->snap_valid) return false;
    if (cu.CtxSetCurrent(g->sw->ctx) != 0) return false;
    if ((m->granite_hybrid || m->nemotron_h) && g->mamba_state && g->snap_ord) {
        // the round's last row wrote the live half; an earlier row's state
        // and conv window come back from its slot into the half the next
        // forward reads (rec_cur) and the ring
        if (r < g->snap_valid - 1) {
            int cd = m->ssm.inner + 2 * m->ssm.groups * m->ssm.state;
            size_t sl = (size_t)m->ssm.inner * m->ssm.state;
            size_t hl = (size_t)(m->ssm.conv_kernel - 1) * cd;
            for (int l = 0; l < m->n_layer; l++) {
                int o = g->snap_ord[l];
                if (o < 0) continue;
                size_t slot = (size_t)r * g->snap_nrec + o;
                if (cu.MemcpyDtoDAsync(mamba_half(g, m, g->rec_cur) + model_rec_slot(m, l) * sl * sizeof(float),
                                       g->snap_state + slot * sl * sizeof(float),
                                       sl * sizeof(float), g->stream) != 0 ||
                    (hl && cu.MemcpyDtoDAsync(g->mamba_conv + model_rec_slot(m, l) * hl * sizeof(float),
                                              g->snap_hist + slot * hl * sizeof(float),
                                              hl * sizeof(float), g->stream) != 0))
                    return false;
            }
            if (cu.StreamSynchronize(g->stream) != 0) return false;
        }
        g->snap_valid = 0;
        return true;
    }
    if (!m->qwen35) return false;
    int convdim = 2 * m->ssm.state * m->ssm.groups + m->ssm.inner;
    size_t hb = sizeof(float) * model_rec_count(m) * (m->ssm.conv_kernel - 1) * convdim;
    size_t sb = sizeof(float) * model_rec_count(m) * m->ssm.v_heads * m->ssm.state * m->ssm.state;
    if (cu.StreamSynchronize(g->stream) != 0) return false;
    if ((hb && cu.MemcpyDtoD(g->q35_hist, g->snap_hist + (size_t)r * hb, hb) != 0) ||
        cu.MemcpyDtoD(g->q35_state, g->snap_state + (size_t)r * sb, sb) != 0)
        return false;
    g->snap_valid = 0;
    return true;
}

bool gpu_recurrent_rollback(model_t *m) {
    gpu_t *g = m ? m->gpu : NULL;
    if (!g || g->fwd_count != g->fwd_snap + 1) return false;
    if (cu.CtxSetCurrent(g->sw->ctx) != 0) return false;
    if (m->qwen35) {
        int convdim = 2 * m->ssm.state * m->ssm.groups + m->ssm.inner;
        size_t hist_bytes = sizeof(float) * model_rec_count(m) * (m->ssm.conv_kernel - 1) * convdim;
        size_t state_bytes = sizeof(float) * model_rec_count(m) * m->ssm.v_heads * m->ssm.state * m->ssm.state;
        if ((hist_bytes && cu.MemcpyDtoD(g->q35_hist, g->q35_hist_prev, hist_bytes) != 0) ||
            cu.MemcpyDtoD(g->q35_state, g->q35_state_prev, state_bytes) != 0)
            return false;
    } else if ((m->granite_hybrid || m->nemotron_h) && g->mamba_state) {
        int cd = m->ssm.inner + 2 * m->ssm.groups * m->ssm.state;
        size_t conv_bytes = sizeof(float) * model_rec_count(m) * (m->ssm.conv_kernel - 1) * cd;
        if (cu.MemcpyDtoD(g->mamba_conv, g->mamba_conv_prev, conv_bytes) != 0)
            return false;
        g->rec_cur = g->rec_prev;   // the forward's input half, untouched by it
    }
    // the restored state is now also the pre-forward state of the next forward
    g->fwd_snap = g->fwd_count;
    return cu.CtxSynchronize() == 0;
}

void gpu_disable(model_t *m) {
    gpu_t *g = m ? m->gpu : NULL;
    // KV is mirrored after every step. The recurrent fold is much larger, so
    // rescue it only on this rare fallback path instead of copying tens of
    // megabytes over PCIe after every generated token: the pre-forward
    // rollback copy is the state the failed forward started from.
    if (g && g->sw && g->sw->ctx && cu.CtxSetCurrent(g->sw->ctx) == 0)
        recurrent_xfer(g, m, true, true);
    gpu_free(m);
}

// X3: one gather of a decoded token's KV rows, one copy back. It moves the
// same bytes to the same host slots as the per-layer copies, so it is on by
// default; RUNNER_CUDA_KV_GATHER=0 keeps the per-layer copies.
static bool gpu_kv_gather_on(void) {
    static _Atomic int on = -1;
    if (on < 0) {
        const char *e = getenv("RUNNER_CUDA_KV_GATHER");
        on = !(e && *e && strcmp(e, "0") == 0);
    }
    return on == 1;
}
static _Atomic unsigned long g_kvg_tokens;
unsigned long gpu_kv_gather_tokens(void) { return g_kvg_tokens; }

// The per-layer table and the two staging buffers, at load; false only on a
// CUDA error (no switch, no kernel or nothing offloaded leaves it unused).
static bool kv_gather_init(gpu_t *g, model_t *m) {
    if (!gpu_kv_gather_on() || !g->sw->f_kv_gather || m->gpu_layers < 1) return true;
    int n = m->gpu_layers;
    kvg_entry *t = calloc((size_t)n, sizeof(kvg_entry));
    if (!t) return true;
    size_t at = 0;
    for (int l = 0; l < n; l++) {
        t[l].koff = model_k_byte_off(m, l);
        t[l].voff = model_v_byte_off(m, l);
        t[l].krow = (unsigned)model_k_row_bytes(m, l);
        t[l].vrow = (unsigned)model_v_row_bytes_own(m, l);   // 0: V is K's table
        t[l].ring = model_kv_is_ring(m, l) ? m->kv_ring : 0;
        t[l].stage = (unsigned)at;
        at += t[l].krow + t[l].vrow;
    }
    g->h_kvg_tab = t;
    g->kvg_bytes = at;
    // KVG_ROWS rows of staging: a forward of up to that many rows (a verify
    // round, a short prompt) gathers each row's slice into its own row
    size_t st = at * KVG_ROWS;
    if (cu.MemAlloc(&g->kvg_tab, sizeof(kvg_entry) * (size_t)n) != 0 ||
        cu.MemcpyHtoD(g->kvg_tab, t, sizeof(kvg_entry) * (size_t)n) != 0 ||
        cu.MemAlloc(&g->kvg_dev, st ? st : 1) != 0 ||
        cu.MemAlloc(&g->kvg_pos, sizeof(int) * KVG_ROWS) != 0)
        return false;
    if (cu.MemAllocHost && cu.MemAllocHost(&g->kvg_host, st ? st : 1) == 0) g->kvg_pinned = true;
    else g->kvg_host = malloc(st ? st : 1);
    return g->kvg_host != NULL;
}

static bool kv_gather_launch(gpu_t *g, model_t *m) {
    void *p[] = { &g->kc, &g->vc, &g->kvg_tab, &g->pos_dev, &g->kvg_dev };
    return launch(g, g->sw->f_kv_gather, (unsigned)m->gpu_layers, 2, 1, 256, p);
}

// Whether an n-row forward's KV rows come back through the gather: up to
// KVG_ROWS rows of one tile. Longer forwards move each layer's rows as
// spans large enough that the per-layer copies cost little per row.
static bool kv_gather_rows_ok(const gpu_t *g, int n, bool partial) {
    return g->kvg_dev && !partial && n >= 1 && n <= KVG_ROWS && n <= g->win;
}

// R3.4.1: an n-row forward's rows, each into its own staging row, from the
// row positions in kvg_pos. It replaces two synchronous copies per layer
// (and span) with one position upload, n launches of the decode gather and
// one copy back; the bytes and the host slots are the per-layer copies'.
static bool kv_gather_rows_launch(gpu_t *g, model_t *m, int pos, int n) {
    // Callers use one row or select through kv_gather_rows_ok; keep the
    // staging extent explicit here as well, including sanitizer builds.
    if (n < 1 || n > KVG_ROWS) return false;
    int hp[KVG_ROWS] = {0};
    for (int r = 0; r < n; r++) hp[r] = pos + r;
    if (cu.MemcpyHtoD(g->kvg_pos, hp, sizeof(int) * (size_t)n) != 0) return false;
    for (int r = 0; r < n; r++) {
        CUdeviceptr pp = g->kvg_pos + (size_t)r * sizeof(int);
        CUdeviceptr out = g->kvg_dev + (size_t)r * g->kvg_bytes;
        void *p[] = { &g->kc, &g->vc, &g->kvg_tab, &pp, &out };
        if (!launch(g, g->sw->f_kv_gather, (unsigned)m->gpu_layers, 2, 1, 256, p)) return false;
    }
    return true;
}

// after the forward synchronized: the rows of positions [pos, pos + n), one
// copy, then into the host cache at their slots (the host cache stays
// authoritative)
static bool kv_gather_finish(gpu_t *g, model_t *m, int pos, int n) {
    if (cu.MemcpyDtoH(g->kvg_host, g->kvg_dev, g->kvg_bytes * (size_t)n) != 0) return false;
    const kvg_entry *t = g->h_kvg_tab;
    for (int r = 0; r < n; r++) {
        const uint8_t *row = (const uint8_t *)g->kvg_host + (size_t)r * g->kvg_bytes;
        for (int l = 0; l < m->gpu_layers; l++) {
            size_t slot = (size_t)(t[l].ring > 0 ? (pos + r) % t[l].ring : pos + r);
            const uint8_t *src = row + t[l].stage;
            memcpy((uint8_t *)m->kcache + t[l].koff + slot * t[l].krow, src, t[l].krow);
            memcpy((uint8_t *)m->vcache + t[l].voff + slot * t[l].vrow, src + t[l].krow,
                   t[l].vrow);
        }
    }
    g_kvg_tokens += (unsigned long)n;
    return true;
}

// The rows positions [lo, hi) of layer l occupy, as at most two spans of its
// storage. A flat layer's are the rows themselves. A ring layer's are the
// slots those positions write, p % kv_ring: contiguous unless they wrap, and
// the whole ring once the span is as long as it. Both sides address the
// slots the same way, so a span is the same bytes on each.
static int kv_spans(const model_t *m, int l, int lo, int hi, int s_lo[2], int s_hi[2]) {
    if (lo >= hi) return 0;
    if (!model_kv_is_ring(m, l)) { s_lo[0] = lo; s_hi[0] = hi; return 1; }
    int r = m->kv_ring;
    if (hi - lo >= r) { s_lo[0] = 0; s_hi[0] = r; return 1; }
    int a = lo % r, b = (hi - 1) % r + 1;   // slots [a, b), or [a, r) + [0, b)
    if (a < b) { s_lo[0] = a; s_hi[0] = b; return 1; }
    s_lo[0] = a; s_hi[0] = r;
    s_lo[1] = 0; s_hi[1] = b;
    return 2;
}

// one span of layer l's K and V rows between the host cache and the device
static bool kv_move(gpu_t *g, model_t *m, int l, int lo, int hi, bool up) {
    size_t krow = model_k_row_bytes(m, l), vrow = model_v_row_bytes_own(m, l);
    size_t koff = model_k_byte_off(m, l) + (size_t)lo * krow;
    size_t voff = model_v_byte_off(m, l) + (size_t)lo * vrow;
    size_t kb = (size_t)(hi - lo) * krow, vb = (size_t)(hi - lo) * vrow;
    // vb is 0 when V is K's table: the K copy already moved its rows
    if (up)
        return cu.MemcpyHtoD(g->kc + koff, (uint8_t *)m->kcache + koff, kb) == 0 &&
               (vb == 0 || cu.MemcpyHtoD(g->vc + voff, (uint8_t *)m->vcache + voff, vb) == 0);
    return cu.MemcpyDtoH((uint8_t *)m->kcache + koff, g->kc + koff, kb) == 0 &&
           (vb == 0 || cu.MemcpyDtoH((uint8_t *)m->vcache + voff, g->vc + voff, vb) == 0);
}

// upload host KV rows [lo, hi) for every device layer: rows the device does
// not hold as the host does (gpu_forward_batch's kv_rows)
static bool kv_upload(gpu_t *g, model_t *m, int lo, int hi) {
    if (lo >= hi) return true;
    for (int l = 0; l < m->gpu_layers; l++) {
        int a[2], b[2], ns = kv_spans(m, l, lo, hi, a, b);
        for (int s = 0; s < ns; s++)
            if (!kv_move(g, m, l, a[s], b[s], true)) return false;
    }
    return true;
}

// copy device KV rows [lo, hi) back to the (authoritative) host cache
static bool kv_copyback(gpu_t *g, model_t *m, int lo, int hi) {
    if (lo >= hi) return true;
    for (int l = 0; l < m->gpu_layers; l++) {
        int a[2], b[2], ns = kv_spans(m, l, lo, hi, a, b);
        for (int s = 0; s < ns; s++)
            if (!kv_move(g, m, l, a[s], b[s], false)) return false;
    }
    return true;
}

// stage the tile's token embeddings into g->x (host dequant + one HtoD);
// kept outside fwd_tile so graph capture records kernels only
// E-series: run the per-layer-embedding pre-pass on the host over the freshly
// staged embeddings and ship its result. Once per forward, so the transfer is
// n_tokens * n_layer * n_embd_ple floats — 43 KB for a single E4B token.
static bool stage_ple(gpu_t *g, model_t *m, const int32_t *tokens, int tn) {
    if (m->n_embd_ple <= 0) return true;
    size_t per_tok = (size_t)m->n_layer * m->n_embd_ple;
    model_ple_prepass(m, tokens, tn, g->h_x, g->h_ple, g->h_ple_tmp);
    return cu.MemcpyHtoD(g->ple, g->h_ple,
                         sizeof(float) * (size_t)tn * per_tok) == 0;
}

static bool stage_x(gpu_t *g, model_t *m, const int32_t *tokens, int tn) {
    size_t ers = ggml_row_size(m->tok_embd->type, m->n_embd);
    for (int b = 0; b < tn; b++) {
        float *hx = g->h_x + (size_t)b * m->n_embd;
        dequant_row(m->tok_embd->type,
                    (uint8_t *)m->tok_embd->data + (size_t)tokens[b] * ers,
                    hx, m->n_embd);
        if (m->tok_embd->scale != 1.0f)
            for (int i = 0; i < m->n_embd; i++) hx[i] *= m->tok_embd->scale;
        model_embd_transform(m, hx);
    }
    if (cu.MemcpyHtoD(g->x, g->h_x, sizeof(float) * tn * m->n_embd) != 0)
        return false;
    if (m->hyper_conn) {
        // the wide residual starts as hc identical copies of the embedding
        size_t hcd = (size_t)m->hc_count * m->n_embd;
        for (int b = 0; b < tn; b++)
            for (int c = 0; c < m->hc_count; c++)
                memcpy(m->x_hc + (size_t)b * hcd + (size_t)c * m->n_embd,
                       g->h_x + (size_t)b * m->n_embd, sizeof(float) * (size_t)m->n_embd);
        if (cu.MemcpyHtoD(g->x_hc, m->x_hc, sizeof(float) * tn * hcd) != 0)
            return false;
    }
    return stage_ple(g, m, tokens, tn);
}

// encode a tile of up to win tokens: matvecs read each weight once for the
// whole tile, and every other kernel takes the token column from the launch
// grid — a tile costs the same number of launches as a single token, which
// matters because WDDM launch overhead dominates small-kernel dispatch. The
// vocab-logits matvec (the single most expensive launch) runs only when the
// caller wants logits, and only for the tile's last token. The caller stages
// x (and, for graph capture, uploads pos) before calling this.
// ----------------------------------------------- sparse-MoE (device routing)
// Encoders for the fused indirect MoE path: routing and expert selection stay
// entirely on the device, so a MoE decode step has no host round-trip and is
// CUDA-graph capturable (the moe-gpu-routing spec's P1).

// Device softmax -> top-k -> renormalize (g->moe_renorm: off under
// expert_weights_norm false) for `tokens` routed columns. The
// kernel is serial per token and mirrors moe_route() bit for bit (see the
// contract comment in kernels.cu).
// Under RUNNER_MOE_REPLAY the replay twin routes instead (the recorded choice
// where the trace holds one, k_moe_route's arithmetic elsewhere); under
// RUNNER_MOE_TRACE the tile's routing is read back and written to the trace.
// layer and col0 (the first routed column within the tile) place each column
// at its position.
// RUNNER_MOE_TRACE on the device paths. A record carries, besides the
// routing, each selected expert's output norm: the L2 norm of its own FFN
// output after the down projection and bias, before the routing weight (and
// gemma-4's down scale), as model.c writes it. enc_moe_route and
// route_trace_host hold a layer's routings here; the path fills in the norms
// where it has the expert outputs, and gpu_moe_ffn / gpu_gemma_moe_ffn write
// the records. A trace run is eager (graph_bad), and every norm costs a sync
// and a copy back. A norm left uncomputed (-1) writes an empty "norms" array.
static void moe_trace_flush(gpu_t *g) {
    for (int r = 0; r < g->tr_n; r++) {
        size_t o = (size_t)r * g->tr_used;
        bool all = true;
        for (int s = 0; s < g->tr_used; s++)
            if (g->tr_norm[o + s] < 0.0f) all = false;   // -1: not computed
        model_moe_trace_route(g->tr_pos[r], g->tr_layer, g->tr_sel + o, g->tr_w + o,
                              all ? g->tr_norm + o : NULL, g->tr_used);
    }
    g->tr_n = 0;
}

static bool moe_trace_add(gpu_t *g, int pos, int layer, const int *sel,
                          const float *w, int used) {
    if (g->tr_n && (layer != g->tr_layer || used != g->tr_used)) moe_trace_flush(g);
    if (g->tr_n + 1 > g->tr_cap || (size_t)(g->tr_n + 1) * used > g->tr_cap_slots) {
        int cap = 2 * (g->tr_n + 1);
        size_t slots = (size_t)cap * used;
        int *p = realloc(g->tr_pos, sizeof(int) * (size_t)cap);
        if (p) g->tr_pos = p;
        int *q = p ? realloc(g->tr_sel, sizeof(int) * slots) : NULL;
        if (q) g->tr_sel = q;
        float *v = q ? realloc(g->tr_w, sizeof(float) * slots) : NULL;
        if (v) g->tr_w = v;
        float *nv = v ? realloc(g->tr_norm, sizeof(float) * slots) : NULL;
        if (!nv) return false;
        g->tr_norm = nv;
        g->tr_cap = cap;
        g->tr_cap_slots = slots;
    }
    size_t o = (size_t)g->tr_n * used;
    g->tr_layer = layer;
    g->tr_used = used;
    g->tr_pos[g->tr_n] = pos;
    memcpy(g->tr_sel + o, sel, sizeof(int) * (size_t)used);
    memcpy(g->tr_w + o, w, sizeof(float) * (size_t)used);
    for (int s = 0; s < used; s++) g->tr_norm[o + s] = -1.0f;
    g->tr_n++;
    return true;
}

// ||v / sc + b||: an output row the device scaled by sc (1 where it did not),
// with the down bias b (NULL where the row has it or the model has none);
// -1, the "not computed" mark, where sc is 0
static float moe_out_norm(const float *v, int n, float sc, const float *b) {
    if (sc == 0.0f) return -1.0f;
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = v[i] / sc + (b ? b[i] : 0.0f);
        ss += d * d;
    }
    return sqrtf(ss);
}

// The records rec0 .. rec0 + n - 1 (tile columns), slot-major on the device:
// row (r, s) at src + (r * used + s) * es floats, each scaled by its routing
// weight times dscale[e] (the gemma-4 down scale, NULL elsewhere), the down
// bias not yet added.
static bool moe_trace_norms_slots(gpu_t *g, const layer_t *ly, CUdeviceptr src,
                                  int rec0, int n, int n_embd, int es,
                                  const float *dscale) {
    if (!g->route_trace || n <= 0) return true;
    int used = g->tr_used;
    size_t rows = (size_t)n * used;
    float *h = malloc(sizeof(float) * rows * (size_t)es);
    if (!h) return false;
    bool ok = cu.StreamSynchronize(g->stream) == 0 &&
              cu.MemcpyDtoH(h, src, sizeof(float) * rows * (size_t)es) == 0;
    for (size_t r = 0; ok && r < rows; r++) {
        size_t k = (size_t)rec0 * used + r;
        int e = g->tr_sel[k];
        float sc = g->tr_w[k] * (dscale ? dscale[e] : 1.0f);
        g->tr_norm[k] = moe_out_norm(h + r * (size_t)es, n_embd, sc,
                                     ly->ffn_down_exps_b
                                         ? ly->ffn_down_exps_b + (size_t)e * n_embd : NULL);
    }
    free(h);
    return ok;
}

// Expert e's unscaled output rows for the tile columns tok[0 .. cnt), packed
// at n_embd (the grouped paths), the down bias not yet added.
static bool moe_trace_norms_rows(gpu_t *g, const layer_t *ly, CUdeviceptr src,
                                 const int *tok, int cnt, int e, int n_embd) {
    if (!g->route_trace || cnt <= 0) return true;
    int used = g->tr_used;
    float *h = malloc(sizeof(float) * (size_t)cnt * n_embd);
    if (!h) return false;
    bool ok = cu.StreamSynchronize(g->stream) == 0 &&
              cu.MemcpyDtoH(h, src, sizeof(float) * (size_t)cnt * n_embd) == 0;
    const float *b = ly->ffn_down_exps_b ? ly->ffn_down_exps_b + (size_t)e * n_embd : NULL;
    for (int r = 0; ok && r < cnt; r++)
        for (int s = 0; s < used; s++) {
            size_t k = (size_t)tok[r] * used + s;
            if (g->tr_sel[k] == e)
                g->tr_norm[k] = moe_out_norm(h + (size_t)r * n_embd, n_embd, 1.0f, b);
        }
    free(h);
    return ok;
}

// One slot's output row on the eager paths: record rec, slot s, scaled by sc
// (1 where the weight was applied after the down projection), bias included.
static bool moe_trace_norm_one(gpu_t *g, CUdeviceptr src, int rec, int s,
                               int n_embd, float sc) {
    if (!g->route_trace) return true;
    float *h = malloc(sizeof(float) * (size_t)n_embd);
    if (!h) return false;
    bool ok = cu.StreamSynchronize(g->stream) == 0 &&
              cu.MemcpyDtoH(h, src, sizeof(float) * (size_t)n_embd) == 0;
    if (ok) g->tr_norm[(size_t)rec * g->tr_used + s] = moe_out_norm(h, n_embd, sc, NULL);
    free(h);
    return ok;
}

static bool enc_moe_route(gpu_t *g, CUdeviceptr logits, CUdeviceptr sel,
                          CUdeviceptr selw, int ne, int used, int tokens,
                          int ls, int layer, int col0) {
    bool ok;
    if (g->replay_tab) {
        int pos0 = g->route_pos + col0;
        void *p[] = { &logits, &sel, &selw, &ne, &used, &tokens, &ls, &g->replay_tab,
                      &g->replay_npos, &g->replay_nlayer, &pos0, &layer, &g->replay_cnt,
                      &g->moe_renorm };
        ok = launch(g, g->sw->f_moe_route_replay, (unsigned)(tokens + 31) / 32, 1, 1, 32, p);
    } else {
        void *p[] = { &logits, &sel, &selw, &ne, &used, &tokens, &ls, &g->moe_renorm };
        ok = launch(g, g->sw->f_moe_route, (unsigned)(tokens + 31) / 32, 1, 1, 32, p);
    }
    if (!ok || !g->route_trace) return ok;
    if (cu.StreamSynchronize(g->stream) != 0 ||
        cu.MemcpyDtoH(g->h_moe_sel, sel, sizeof(int) * (size_t)tokens * used) != 0 ||
        cu.MemcpyDtoH(g->h_moe_selw, selw, sizeof(float) * (size_t)tokens * used) != 0)
        return false;
    for (int t = 0; t < tokens; t++)
        if (!moe_trace_add(g, g->route_pos + col0 + t, layer, g->h_moe_sel + (size_t)t * used,
                           g->h_moe_selw + (size_t)t * used, used))
            return false;
    return true;
}

// The device's replay counts into model.c's exit report, and reset.
static void route_replay_flush(gpu_t *g) {
    unsigned c[3];
    if (!g->replay_cnt || cu.MemcpyDtoH(c, g->replay_cnt, sizeof c) != 0) return;
    if (c[0] || c[1] || c[2]) {
        model_moe_replay_count(c[0], c[1], c[2]);
        cu.MemsetD8(g->replay_cnt, 0, sizeof c);
    }
}

// The eager paths route on the host: RUNNER_MOE_REPLAY's recorded experts for
// column t (NULL where the trace has none), and the routing written to
// RUNNER_MOE_TRACE with the weights renormalised as applied.
static const int *route_forced_host(const gpu_t *g, int t, int layer, int used) {
    return g->replay_tab ? model_moe_replay_ids(g->route_pos + t, layer, used) : NULL;
}

static bool route_trace_host(gpu_t *g, int t, int layer, const int *sel,
                             const float *selw, float denom, int used) {
    if (!g->route_trace) return true;
    float w[256];
    for (int s = 0; s < used; s++) w[s] = selw[s] / denom;
    return moe_trace_add(g, g->route_pos + t, layer, sel, w, used);
}

// One launch computes every selected expert's matvec: grid.y = expert slot,
// weight base resolved in-kernel from sel[slot] and the fused-3D expert
// byte stride (the moe_expert_weight arithmetic, moved into kernel args).
// base is the whole fused tensor, so binding_find's bounds check covers
// every expert's slice at once.
static bool enc_moe_mv_at(gpu_t *g, gguf_tensor *base, CUdeviceptr weights,
                          uint64_t w_off, uint64_t estride, CUdeviceptr x,
                          CUdeviceptr y, CUdeviceptr sel, int n_in, int n_out,
                          int nslots, int xs, int ys);

static bool enc_moe_mv(gpu_t *g, model_t *m, gguf_tensor *base,
                       uint64_t estride, CUdeviceptr x, CUdeviceptr y,
                       CUdeviceptr sel, int n_in, int n_out, int nslots,
                       int xs, int ys) {
    CUdeviceptr weights = 0;
    uint64_t w_off = 0;
    if (!binding_find(g->sw, m, base, &weights, &w_off)) {
        gpu_first_error("expert tensor '%s' is not resident on the device "
                        "(no binding covers it)", base->name);
        return false;
    }
    return enc_moe_mv_at(g, base, weights, w_off, estride, x, y, sel, n_in, n_out,
                         nslots, xs, ys);
}

// enc_moe_mv's launch on an expert stack at weights + w_off, estride bytes an
// expert: the bound tensor's, or a RUNNER_MOE_HOT bank's slots.
static bool enc_moe_mv_at(gpu_t *g, gguf_tensor *base, CUdeviceptr weights,
                          uint64_t w_off, uint64_t estride, CUdeviceptr x,
                          CUdeviceptr y, CUdeviceptr sel, int n_in, int n_out,
                          int nslots, int xs, int ys) {
    CUfunction f = g->sw->f_moe_mv[base->type];
    if (!f) {
        gpu_first_error("no indirect MoE kernel for quant type %s",
                        ggml_type_name(base->type));
        return false;
    }
    moe_args a = { n_in, n_out, w_off, estride, xs, ys };
    // k_moe_mv_nvfp4 takes the per-tensor companion as a sixth parameter;
    // the others read five and ignore the trailing entry
    float scale = base->scale;
    void *p[] = { &weights, &x, &y, &a, &sel, &scale };
    // the one-lane-per-block types stage the slot's activations once per
    // block of rows (kernels.cu, MOE_LANE_HEAD), so eight rows a block share
    // each copy; each warp's row is computed the same either way
    // (IQ4_XS: its MoE copy is lane-per-block and staged, its dense batch-1
    // kernel is not, so it is named here and not in lane_staged_type)
    int rows = lane_staged_type(base->type) || base->type == T_IQ4_XS ? 8 : 4;
    return launch(g, f, (unsigned)((n_out + rows - 1) / rows), nslots, 1,
                  (unsigned)(32 * rows), p);
}

// Gated activation + routing-weight fold for every slot in one launch.
// dscale is the per-expert down-scale table (gemma-4) or the shared ones
// table; the arithmetic matches the eager actmul+scale launch pair exactly.
// gb/ub are the per-expert gate/up bias tables (gpt-oss) or 0 — the bias
// lands before the activation, mirroring moe_ffn_token on the CPU.
static bool enc_moe_actmul(gpu_t *g, model_t *m, CUdeviceptr gbuf,
                           CUdeviceptr ubuf, int nff, int gss, int uss,
                           CUdeviceptr selw, CUdeviceptr dscale,
                           CUdeviceptr sel, int nslots,
                           CUdeviceptr gb, CUdeviceptr ub) {
    int act = m->ffn_act;  // ACT_* values match the kernel's selector 1:1
    void *p[] = { &gbuf, &ubuf, &nff, &gss, &uss, &act, &selw, &dscale, &sel,
                  &gb, &ub };
    return launch(g, g->sw->f_moe_actmul, (nff + 255) / 256, nslots, 1, 256, p);
}

// Sum the per-slot down outputs into the token's FFN output, ascending slot
// order — same accumulation order as the eager write-then-add sequence.
// db is the per-expert down-bias table (gpt-oss) or 0; each slot adds
// selw*db[sel] because the routing weight was already folded into the hidden
// before the down matvec (w*(down+db) distributed).
static bool enc_moe_sum(gpu_t *g, CUdeviceptr out, CUdeviceptr eout, int n,
                        int nslots, int es, CUdeviceptr db, CUdeviceptr sel,
                        CUdeviceptr selw) {
    void *p[] = { &out, &eout, &n, &nslots, &es, &db, &sel, &selw };
    return launch(g, g->sw->f_moe_sum, (n + 255) / 256, 1, 1, 256, p);
}

// Fused device-routing MoE FFN (Mixtral/Qwen3 topology, fused-3D layout).
// Reads the RMS-normed input from g->xb (stride xdim) and writes the FFN
// output back into g->xb. Per layer: one router matvec per token (batch-1,
// the certified per-token GEMV arithmetic), one routing kernel for the
// tile, then per token gate/up/actmul/down/sum — six launches a token and
// zero host round-trips, vs the eager path's ~49 launches and one
// sync+DtoH per token.
static inline bool mvb_is_mv_twin(int type);

// The router over a prompt tile, every column the batch-1 call's bits. For
// F32 and F16 (and BF16 off its split-K route) the batch-1 call is f_mv, and
// f_mvb is its twin (mvb_is_mv_twin): one launch over the tile, sixteen
// columns per grid row. enc_mv(batch = tn) would not do it below nine
// columns, where these types have no f_gemvb and its small-batch branch runs
// the batch-1 kernel once per column. Any other type keeps one batch-1 call
// per column.
static bool enc_router(gpu_t *g, model_t *m, gguf_tensor *w, CUdeviceptr x, int xs,
                       CUdeviceptr y, int n_in, int n_out, CUdeviceptr bias, int tn) {
    if (tn > 1 && w->scale == 1.0f && mvb_is_mv_twin(w->type) && g->sw->f_mvb[w->type] &&
        !mv_route_splitk(g, w, n_in, n_out)) {
        CUdeviceptr weights = 0;
        uint64_t w_off = 0;
        if (!binding_find(g->sw, m, w, &weights, &w_off)) {
            gpu_first_error("tensor '%s' is not resident on the device "
                            "(no binding covers it)", w->name);
            return false;
        }
        mv_args a = { n_in, n_out, w_off, bias != 0, tn, xs, n_out };
        return launch_tiled(g, g->sw->f_mvb[w->type], (unsigned)((n_out + 3) / 4), 128,
                            weights, x, y, a, bias ? bias : g->sw->dummy, MVT,
                            false, w->scale, true);
    }
    for (int t = 0; t < tn; t++)
        if (!enc_mv(g, m, w, x + (size_t)t * xs * sizeof(float),
                    y + (size_t)t * n_out * sizeof(float), n_in, n_out, bias, 1, xs, n_out))
            return false;
    return true;
}

static bool gpu_moe_ffn_fused(gpu_t *g, model_t *m, const layer_t *ly,
                              int tn, int xdim) {
    int n_embd = m->n_embd, ne = ly->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp;
    int l = (int)(ly - m->layers);
    if (used > MOE_MAX_USED) used = MOE_MAX_USED;
    if (ne  > MOE_ROUTE_MAX) ne  = MOE_ROUTE_MAX;
    uint64_t gstride = (uint64_t)nff *
                       ggml_row_size(ly->ffn_gate_exps->type, n_embd);
    uint64_t ustride = (uint64_t)nff *
                       ggml_row_size(ly->ffn_up_exps->type, n_embd);
    uint64_t dstride = (uint64_t)n_embd *
                       ggml_row_size(ly->ffn_down_exps->type, nff);
    // gib: the gpt-oss router bias, which applies to the LOGITS before
    // gating (model.c moe_route). k_moe_route takes no bias input, so it
    // has to ride the router matvec, as it already does on the eager and
    // grouped paths and in metal.m.
    if (!enc_router(g, m, ly->ffn_gate_inp, g->xb, xdim, g->moe_logits, n_embd, ne,
                    g->sw->gib[l], tn))
        return false;
    if (!enc_moe_route(g, g->moe_logits, g->moe_sel, g->moe_selw, ne, used,
                       tn, ne, l, 0))
        return false;
    // RUNNER_DEBUG_MOE: fused-path twin of the eager dump above
    if (getenv("RUNNER_DEBUG_MOE")) {
        static _Atomic unsigned dumped;   // ++ is atomic: two dumps, whichever slots
        if (dumped++ < 2) {
            cu.StreamSynchronize(g->stream);
            int hs[256]; float hw[256];
            cu.MemcpyDtoH(hs, g->moe_sel, sizeof(int) * used);
            cu.MemcpyDtoH(hw, g->moe_selw, sizeof(float) * used);
            fprintf(stderr, "FUSED sel :");
            for (int s = 0; s < used; s++) fprintf(stderr, " %d", hs[s]);
            fprintf(stderr, "\nFUSED selw:");
            for (int s = 0; s < used; s++) {
                unsigned u; memcpy(&u, &hw[s], 4);
                fprintf(stderr, " %08x", u);
            }
            fprintf(stderr, "\n");
        }
    }
    for (int t = 0; t < tn; t++) {
        CUdeviceptr xin  = g->xb + (size_t)t * xdim * sizeof(float);
        CUdeviceptr sel  = g->moe_sel  + (size_t)t * used * sizeof(int);
        CUdeviceptr selw = g->moe_selw + (size_t)t * used * sizeof(float);
        bool ok = enc_moe_mv(g, m, ly->ffn_gate_exps, gstride, xin, g->moe_hb,
                             sel, n_embd, nff, used, 0, nff)
               && enc_moe_mv(g, m, ly->ffn_up_exps, ustride, xin, g->moe_hb2,
                             sel, n_embd, nff, used, 0, nff)
               && enc_moe_actmul(g, m, g->moe_hb, g->moe_hb2, nff, nff, nff,
                                 selw, g->sw->moe_ones, sel, used,
                                 g->sw->geb[l], g->sw->ueb[l])
               && enc_moe_mv(g, m, ly->ffn_down_exps, dstride, g->moe_hb,
                             g->moe_eout, sel, nff, n_embd, used, nff, n_embd)
               && moe_trace_norms_slots(g, ly, g->moe_eout, t, 1, n_embd, n_embd, NULL)
               && enc_moe_sum(g, xin, g->moe_eout, n_embd, used, n_embd,
                              g->sw->deb[l], sel, selw);
        if (!ok) return false;
    }
    return true;
}

// ------------------------------------------ expert-grouped prefill (P2)
// CUDA port of the CPU cabdad1 grouping: route the tile on device, read the
// tile's routing back ONCE (prefill is never graph-captured, so the single
// sync+DtoH is legal), build per-expert token lists on the host, then run
// each active expert once over its token set with the batched GEMM kernels
// — each expert's weights stream through the SMs once per tile instead of
// once per (token, slot). Accumulation follows the CPU grouped path:
// ascending expert index, routing weight applied at the scatter.

// Read the tile's routing and build the concatenated per-expert token
// lists (ascending expert), uploading them to moe_gidx/moe_gw. Fills
// e_start/e_cnt (ne entries); when fold_dscale is set the per-expert down
// scale (gemma-4) is folded into the scatter weight, exactly as the CPU
// grouped/eager paths compute sc = selw * down_exps_scale[e]. Returns
// false on a CUDA error.
static bool moe_build_groups(gpu_t *g, const layer_t *ly, int tn, int used,
                             int ne, bool fold_dscale, int *e_start,
                             int *e_cnt) {
    if (cu.StreamSynchronize(g->stream) != 0 ||
        cu.MemcpyDtoH(g->h_moe_sel, g->moe_sel,
                      sizeof(int) * (size_t)tn * used) != 0 ||
        cu.MemcpyDtoH(g->h_moe_selw, g->moe_selw,
                      sizeof(float) * (size_t)tn * used) != 0)
        return false;
    int total = 0;
    for (int e = 0; e < ne; e++) {
        e_start[e] = total;
        for (int b = 0; b < tn; b++) {
            const int   *sel  = g->h_moe_sel  + (size_t)b * used;
            const float *selw = g->h_moe_selw + (size_t)b * used;
            for (int s = 0; s < used; s++) {
                if (sel[s] != e) continue;
                g->h_moe_gidx[total] = b;
                g->h_moe_gw[total] = selw[s] *
                    (fold_dscale && ly->down_exps_scale
                         ? ly->down_exps_scale[e] : 1.0f);
                total++;
                break;                          // top-k experts are distinct
            }
        }
        e_cnt[e] = total - e_start[e];
    }
    return cu.MemcpyHtoD(g->moe_gidx, g->h_moe_gidx,
                         sizeof(int) * (size_t)total) == 0 &&
           cu.MemcpyHtoD(g->moe_gw, g->h_moe_gw,
                         sizeof(float) * (size_t)total) == 0;
}

// defined in the decode-microbatch section below; used here for its
// width-classed kernel selection
static bool enc_mv_batch(gpu_t *g, model_t *m, gguf_tensor *w, CUdeviceptr x,
                         CUdeviceptr y, int n_in, int n_out, CUdeviceptr bias,
                         int batch, int xs, int ys);

// Batched matmul for one expert's token set, cnt-aware. The fixed-width
// GEMM kernels compute all MVB columns whatever the batch, so at the small
// counts grouping produces (~1-2 tokens/expert on a 16-token tile) they do
// ~10x the useful FMA work and lose to the per-token path (measured on the
// 3070). Pick the narrowest kernel that covers cnt: the batch-1 GEMV at
// cnt==1, the width-classed f_gemvb twins up to 8, the full GEMM beyond —
// and the TC GEMM whenever it is promoted/forced for this (type, arch),
// since tensor-core MACs make the unused columns nearly free.
// enc_mv for a GEMM whose input columns the previous TC2 GEMM staged, with
// nothing written to them since (the caller's knowledge, scoped to this call)
static bool enc_mv_same_x(gpu_t *g, model_t *m, gguf_tensor *w, CUdeviceptr x,
                          CUdeviceptr y, int n_in, int n_out, CUdeviceptr bias,
                          int batch, int xs, int ys) {
    g->xh_reuse = true;
    bool ok = enc_mv(g, m, w, x, y, n_in, n_out, bias, batch, xs, ys);
    g->xh_reuse = false;
    return ok;
}

static bool enc_moe_gemm(gpu_t *g, model_t *m, gguf_tensor *w, CUdeviceptr x,
                         CUdeviceptr y, int n_in, int n_out, int cnt,
                         int xs, int ys) {
    // a verify round's rows never reach the grouped prefill (gpu_moe_ffn), and
    // if one did, enc_mv's exact block gives every column the batch-1 bits
    if (cnt == 1 || g->exact_rows)
        return enc_mv(g, m, w, x, y, n_in, n_out, 0, cnt, xs, ys);
    if (cnt > 8 || (tc_on(m, w->type) && g->sw->f_gemm_tc[w->type]))
        return enc_mv(g, m, w, x, y, n_in, n_out, 0, cnt, xs, ys);
    // enc_mv_batch REFUSES a type with no width-classed twin, because the
    // decode microbatch owes each sequence the bits a lone step would produce.
    // Grouped expert prefill owes no such thing -- it is tolerance-gated
    // (test-moe-tol), not identity-gated -- so here the tile kernel is still a
    // legitimate fallback and enc_mv picks it. moe_grouped_eligible() admits
    // only Q8_0/Q4_K/Q5_K/Q6_K/Q4_0 today, all of which have twins, so this branch
    // is unreachable; it exists so widening that set cannot silently turn into
    // a hard failure at the first expert.
    if (!g->sw->f_gemvb[batch_width_class(cnt)][w->type])
        return enc_mv(g, m, w, x, y, n_in, n_out, 0, cnt, xs, ys);
    return enc_mv_batch(g, m, w, x, y, n_in, n_out, 0, cnt, xs, ys);
}

static bool enc_moe_gather(gpu_t *g, CUdeviceptr dst, CUdeviceptr src,
                           CUdeviceptr idx, int n, int cnt, int ss, int ds) {
    void *p[] = { &dst, &src, &idx, &n, &ss, &ds };
    return launch(g, g->sw->f_moe_gather, (n + 255) / 256, cnt, 1, 256, p);
}

static bool enc_moe_scatter(gpu_t *g, CUdeviceptr out, CUdeviceptr src,
                            CUdeviceptr idx, CUdeviceptr wts, int n, int cnt,
                            int os, int ss) {
    void *p[] = { &out, &src, &idx, &wts, &n, &os, &ss };
    return launch(g, g->sw->f_moe_scatter, (n + 255) / 256, cnt, 1, 256, p);
}

// Grouped Mixtral/Qwen3 tile: reads the RMS-normed inputs from g->xb
// (stride xdim), writes the FFN outputs back into g->xb, like the other
// gpu_moe_ffn_* variants.
static bool gpu_moe_ffn_grouped(gpu_t *g, model_t *m, const layer_t *ly,
                                int tn, int xdim) {
    int n_embd = m->n_embd, ne = ly->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp;
    int l = (int)(ly - m->layers);
    if (used > MOE_MAX_USED) used = MOE_MAX_USED;
    if (ne  > MOE_ROUTE_MAX) ne  = MOE_ROUTE_MAX;
    if (!enc_router(g, m, ly->ffn_gate_inp, g->xb, xdim, g->moe_logits, n_embd, ne,
                    g->sw->gib[l], tn))
        return false;
    if (!enc_moe_route(g, g->moe_logits, g->moe_sel, g->moe_selw, ne, used,
                       tn, ne, l, 0))
        return false;
    int e_start[MOE_ROUTE_MAX], e_cnt[MOE_ROUTE_MAX];   // by expert: ne, up to MOE_ROUTE_MAX
    if (!moe_build_groups(g, ly, tn, used, ne, false, e_start, e_cnt))
        return false;
    if (cu.MemsetD8(g->moe_eout, 0, sizeof(float) * (size_t)tn * n_embd) != 0)
        return false;
    for (int e = 0; e < ne; e++) {
        int cnt = e_cnt[e];
        if (cnt == 0) continue;
        CUdeviceptr idx = g->moe_gidx + (size_t)e_start[e] * sizeof(int);
        CUdeviceptr wts = g->moe_gw   + (size_t)e_start[e] * sizeof(float);
        gguf_tensor gv = moe_expert_weight(ly, 0, e, n_embd, nff);
        gguf_tensor uv = moe_expert_weight(ly, 1, e, n_embd, nff);
        gguf_tensor dv = moe_expert_weight(ly, 2, e, n_embd, nff);
        bool ok = enc_moe_gather(g, g->moe_gath, g->xb, idx, n_embd, cnt,
                                 xdim, n_embd)
               && enc_moe_gemm(g, m, &gv, g->moe_gath, g->moe_hb, n_embd,
                               nff, cnt, n_embd, nff);
        // up reads the gathered columns gate just staged
        g->xh_reuse = true;
        ok = ok && enc_moe_gemm(g, m, &uv, g->moe_gath, g->moe_hb2, n_embd,
                                nff, cnt, n_embd, nff);
        g->xh_reuse = false;
        ok = ok
               && enc_actmul(g, m, g->moe_hb, g->moe_hb2, cnt * nff)
               && enc_moe_gemm(g, m, &dv, g->moe_hb, g->moe_dout, nff,
                               n_embd, cnt, nff, n_embd)
               && moe_trace_norms_rows(g, ly, g->moe_dout, g->h_moe_gidx + e_start[e],
                                       cnt, e, n_embd)
               && enc_moe_scatter(g, g->moe_eout, g->moe_dout, idx, wts,
                                  n_embd, cnt, n_embd, n_embd);
        if (!ok) return false;
    }
    void *p[] = { &g->xb, &g->moe_eout, &n_embd, &xdim, &n_embd };
    return launch(g, g->sw->f_moe_copy, (n_embd + 255) / 256, tn, 1, 256, p);
}

// Grouped gemma-4 dual-branch tile: the dense shared branch batches across
// the whole tile with the ordinary tn-wide kernels (it was per-token
// before), and the routed branch groups by expert over the fused gate_up
// tensor. Reads g->x (un-normed residual), writes dense+routed into g->xb,
// same contract as the other gemma variants.
static bool gpu_gemma_moe_ffn_grouped(gpu_t *g, model_t *m, const layer_t *ly,
                                      int l, int tn, int xdim) {
    int n_embd = m->n_embd, ne = ly->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp, dff = ly->n_ff;
    if (used > MOE_MAX_USED) used = MOE_MAX_USED;
    if (ne  > MOE_ROUTE_MAX) ne  = MOE_ROUTE_MAX;
    float eps = m->rms_eps;
    CUdeviceptr xn_b  = g->g_scr;
    CUdeviceptr xn2_b = g->g_scr + (size_t)g->win * n_embd * sizeof(float);
    CUdeviceptr rin_b = g->g_scr + (size_t)2 * g->win * n_embd * sizeof(float);
    // --- dense shared MLP for the whole tile
    bool ok = enc_rmsnorm(g, g->x, xn_b, g->sw->ffn_norm[l], n_embd, eps,
                          tn, n_embd, n_embd)
           && enc_mv(g, m, ly->w_gate, xn_b, g->hb,  n_embd, dff, 0, tn,
                     n_embd, dff)
           && enc_mv_same_x(g, m, ly->w_up, xn_b, g->hb2, n_embd, dff, 0, tn,
                            n_embd, dff)
           && enc_actmul(g, m, g->hb, g->hb2, tn * dff)
           && enc_mv(g, m, ly->w_down, g->hb, g->xb, dff, n_embd, 0, tn,
                     dff, xdim)
           && enc_rmsnorm(g, g->xb, g->xb, g->sw->g_pn1[l], n_embd, eps,
                          tn, xdim, xdim);
    if (!ok) return false;
    // --- routed branch: norms + router for the tile, then group by expert
    ok = enc_rmsnorm(g, g->x, xn2_b, g->sw->g_prn2[l], n_embd, eps, tn,
                     n_embd, n_embd)
      && enc_rmsnorm(g, g->x, rin_b, g->sw->g_gis[l], n_embd, eps, tn,
                     n_embd, n_embd);
    ok = ok && enc_router(g, m, ly->ffn_gate_inp, rin_b, n_embd, g->moe_logits,
                          n_embd, ne, 0, tn);
    ok = ok && enc_moe_route(g, g->moe_logits, g->moe_sel, g->moe_selw, ne,
                             used, tn, ne, l, 0);
    if (!ok) return false;
    int e_start[MOE_ROUTE_MAX], e_cnt[MOE_ROUTE_MAX];   // by expert: ne, up to MOE_ROUTE_MAX
    if (!moe_build_groups(g, ly, tn, used, ne, true, e_start, e_cnt))
        return false;
    if (cu.MemsetD8(g->moe_eout, 0, sizeof(float) * (size_t)tn * n_embd) != 0)
        return false;
    int act = m->ffn_act;  // ACT_* values match the kernel's selector 1:1
    for (int e = 0; e < ne; e++) {
        int cnt = e_cnt[e];
        if (cnt == 0) continue;
        CUdeviceptr idx = g->moe_gidx + (size_t)e_start[e] * sizeof(int);
        CUdeviceptr wts = g->moe_gw   + (size_t)e_start[e] * sizeof(float);
        // fused gate_up expert slice, nbytes clamped (see the eager path)
        gguf_tensor guv = *ly->ffn_gate_up_exps;
        size_t rs = ggml_row_size(guv.type, n_embd);
        guv.data = (uint8_t *)ly->ffn_gate_up_exps->data +
                   (size_t)e * (2 * (size_t)nff) * rs;
        guv.nbytes = (2 * (size_t)nff) * rs;
        gguf_tensor dv = moe_expert_weight(ly, 2, e, n_embd, nff);
        CUdeviceptr up = g->moe_hb + (size_t)nff * sizeof(float);
        int gss = 2 * nff;
        void *pa[] = { &g->moe_hb, &up, &nff, &gss, &gss, &act };
        bool eok = enc_moe_gather(g, g->moe_gath, xn2_b, idx, n_embd, cnt,
                                  n_embd, n_embd)
                && enc_moe_gemm(g, m, &guv, g->moe_gath, g->moe_hb, n_embd,
                                2 * nff, cnt, n_embd, 2 * nff)
                && launch(g, g->sw->f_moe_actmul_plain, (nff + 255) / 256,
                          cnt, 1, 256, pa)
                && enc_moe_gemm(g, m, &dv, g->moe_hb, g->moe_dout, nff,
                                n_embd, cnt, 2 * nff, n_embd)
                && moe_trace_norms_rows(g, ly, g->moe_dout, g->h_moe_gidx + e_start[e],
                                        cnt, e, n_embd)
                && enc_moe_scatter(g, g->moe_eout, g->moe_dout, idx, wts,
                                   n_embd, cnt, n_embd, n_embd);
        if (!eok) return false;
    }
    // post_ffw_norm_2 on the routed accumulator, then out (=dense) += routed
    return enc_rmsnorm(g, g->moe_eout, g->moe_eout, g->sw->g_pn2[l], n_embd,
                       eps, tn, n_embd, n_embd)
        && enc_add(g, g->xb, g->moe_eout, n_embd, tn, xdim, n_embd);
}

// Eager fallback (legacy split expert layout, or an expert tensor type the
// indirect kernels do not cover). Routing — softmax over ALL experts, top-k,
// renormalize (Mixtral/Qwen3; not under expert_weights_norm false) — runs
// on the host from router logits read
// back per token; the per-expert SwiGLU matmuls run on the GPU by pointing
// enc_mv at each expert's slice. Host-dependent routing is why this path
// forces the eager (graph_bad) decode.
static bool gpu_moe_ffn_eager(gpu_t *g, model_t *m, const layer_t *ly, int tn, int xdim) {
    int n_embd = m->n_embd, ne = ly->n_expert, used = m->n_expert_used, nff = m->n_ff_exp;
    int l = (int)(ly - m->layers);
    // sel[]/selw[] below are MOE_MAX_USED wide and the router logits
    // MOE_ROUTE_MAX wide. The expert count bounds the router, not the
    // selection: clamping ne to the selection width routes a 512-expert model
    // among its first 256 experts only.
    if (used > MOE_MAX_USED) used = MOE_MAX_USED;
    if (ne  > MOE_ROUTE_MAX) ne  = MOE_ROUTE_MAX;
    // A small tile (a speculative verify) routes all its rows in one launch
    // and reads the logits back once, not one sync and copy per row. Only with
    // a width-classed twin of the router's batch-1 kernel, so each row's
    // logits are the bits the per-row launch produces; wider tiles keep the
    // per-row route.
    bool batched_router = tn > 1 && tn <= GPU_SMALL_BATCH &&
                          batch_mv_twin_ok(g, ly->ffn_gate_inp) &&
                          g->sw->f_gemvb[batch_width_class(tn)][ly->ffn_gate_inp->type];
    if (batched_router) {
        if (!enc_mv(g, m, ly->ffn_gate_inp, g->xb, g->moe_logits, n_embd, ne,
                    g->sw->gib[l], tn, xdim, ne))
            return false;
        if (cu.StreamSynchronize(g->stream) != 0) return false;
        if (cu.MemcpyDtoH(g->h_moe_logits, g->moe_logits,
                          sizeof(float) * (size_t)tn * ne) != 0)
            return false;
    }
    for (int t = 0; t < tn; t++) {
        CUdeviceptr xin  = g->xb  + (size_t)t * xdim * sizeof(float);
        CUdeviceptr aout = g->xb2 + (size_t)t * xdim * sizeof(float);
        float *lg = g->h_moe_logits + (batched_router ? (size_t)t * ne : 0);
        if (!batched_router) {
            if (!enc_mv(g, m, ly->ffn_gate_inp, xin, g->moe_logits, n_embd, ne,
                        g->sw->gib[l], 1, xdim, ne))
                return false;
            if (cu.StreamSynchronize(g->stream) != 0) return false;
            if (cu.MemcpyDtoH(g->h_moe_logits, g->moe_logits, sizeof(float) * (size_t)ne) != 0)
                return false;
        }
        // softmax over all experts, then top-k with renormalized weights
        float mx = lg[0];
        for (int e = 1; e < ne; e++)
            if (lg[e] > mx) mx = lg[e];
        float ssum = 0.0f;
        for (int e = 0; e < ne; e++) { float p = expf(lg[e] - mx); lg[e] = p; ssum += p; }
        for (int e = 0; e < ne; e++) lg[e] /= ssum;
        int sel[MOE_MAX_USED];
        float selw[MOE_MAX_USED], denom = 0.0f;
        const int *forced = route_forced_host(g, t, l, used);
        unsigned long differed = 0;
        for (int s = 0; s < used; s++) {
            int best = 0;
            float bp = -1.0f;
            for (int e = 0; e < ne; e++)
                if (lg[e] > bp) { bp = lg[e]; best = e; }
            if (forced) {       // RUNNER_MOE_REPLAY: the recorded choice
                differed += forced[s] != best;
                if (forced[s] < ne) { best = forced[s]; bp = lg[best]; }
            }
            sel[s] = best;
            selw[s] = bp;
            denom += bp;
            lg[best] = -1.0f;
        }
        if (forced) model_moe_replay_count((unsigned long)used, differed, 0);
        // expert_weights_norm false (deepseek2): the softmax weights as they
        // are; x / 1.0f is exact, so the weights below are selw unchanged
        if (!m->expert_norm_w) denom = 1.0f;
        if (!route_trace_host(g, t, l, sel, selw, denom, used)) return false;
        // RUNNER_DEBUG_MOE: dump the first two routings' selection + weights
        // as hex — the discriminator that isolated the 2026-07-29 identity
        // regression (device-vs-host expf + reciprocal-math, both 1-ulp).
        // Paired with the same dump in the fused path; bit-compare them.
        if (getenv("RUNNER_DEBUG_MOE")) {
            static _Atomic unsigned dumped;   // ++ is atomic: two dumps, whichever slots
            if (t == 0 && dumped++ < 2) {
                fprintf(stderr, "EAGER sel :");
                for (int s = 0; s < used; s++) fprintf(stderr, " %d", sel[s]);
                fprintf(stderr, "\nEAGER selw:");
                for (int s = 0; s < used; s++) {
                    float w_ = selw[s] / denom;
                    unsigned u; memcpy(&u, &w_, 4);
                    fprintf(stderr, " %08x", u);
                }
                fprintf(stderr, "\n");
            }
        }
        for (int s = 0; s < used; s++) {
            int e = sel[s];
            float w = selw[s] / denom;
            gguf_tensor gv = moe_expert_weight(ly, 0, e, n_embd, nff);
            gguf_tensor uv = moe_expert_weight(ly, 1, e, n_embd, nff);
            gguf_tensor dv = moe_expert_weight(ly, 2, e, n_embd, nff);
            // gpt-oss per-expert bias slices; 0 for every other arch, which
            // keeps the certified unbiased launch sequence byte-for-byte
            CUdeviceptr gb = g->sw->geb[l]
                ? g->sw->geb[l] + (size_t)e * nff * sizeof(float) : 0;
            CUdeviceptr ub = g->sw->ueb[l]
                ? g->sw->ueb[l] + (size_t)e * nff * sizeof(float) : 0;
            CUdeviceptr db = g->sw->deb[l]
                ? g->sw->deb[l] + (size_t)e * n_embd * sizeof(float) : 0;
            bool ok = enc_mv(g, m, &gv, xin, g->hb, n_embd, nff, gb, 1, xdim, nff)
                   && enc_mv(g, m, &uv, xin, g->hb2, n_embd, nff, ub, 1, xdim, nff)
                   && enc_actmul(g, m, g->hb, g->hb2, nff);
            // Down bias must land BEFORE the routing weight scales the
            // output (CPU: out += w*(down(h)+db)). Folding w into the hidden
            // first would leave the bias unscaled, so the biased sequence
            // rides the bias through the down matvec tail and scales after.
            if (!ok) return false;
            if (!db && !enc_scale(g, g->hb, w, nff, 1, nff))   // fold weight into the hidden
                return false;
            CUdeviceptr dst = s == 0 ? aout : g->moe_eout;
            if (!enc_mv(g, m, &dv, g->hb, dst, nff, n_embd, db, 1, nff, n_embd) ||
                !moe_trace_norm_one(g, dst, t, s, n_embd, db ? 1.0f : w))
                return false;
            if (db && !enc_scale(g, dst, w, n_embd, 1, n_embd))
                return false;
            if (s > 0 && !enc_add(g, aout, g->moe_eout, n_embd, 1, n_embd, n_embd))
                return false;
        }
    }
    // move the accumulated per-token outputs (in xb2) back into xb for the
    // residual add; sync first so the expert kernels have completed
    if (cu.StreamSynchronize(g->stream) != 0) return false;
    return cu.MemcpyDtoD(g->xb, g->xb2, (size_t)tn * xdim * sizeof(float)) == 0;
}

// ------------------------------------- by-expert prefill, fused-identical
// The fused path streams every token's experts' weights per token, which
// bounds its prefill. This path runs the same arithmetic grouped by expert:
// the routing is read back once and each active expert's columns go through
// the width-classed GEMV twins, eight at a time, each column the batch-1
// kernel's bits. Activation, placement and the slot-order sum are the fused
// path's, so every value and every addition's order is too.
// On by default where it measured faster (gemma4 MoE and gpt-oss, +11 to
// +14% prefill), not for qwen3moe (8 to 9% slower) or unmeasured families.
// RUNNER_MOE_TWIN=1 takes it on any family, 0 keeps the fused path on all;
// gpu_moe_twin_force overrides both for a test (-1 returns to the switch).
static _Atomic int g_moe_twin_force = -1;
void gpu_moe_twin_force(int on) { g_moe_twin_force = on < 0 ? -1 : on != 0; }
static bool moe_twin_on(const model_t *m) {
    if (g_moe_twin_force >= 0) return g_moe_twin_force == 1;
    static _Atomic int env = -2;
    if (env == -2) {
        const char *e = getenv("RUNNER_MOE_TWIN");
        env = !(e && *e) ? -1 : strcmp(e, "0") != 0;
    }
    if (env >= 0) return env == 1;
    return strcmp(m->arch, "gemma4") == 0 || strcmp(m->arch, "gpt-oss") == 0;
}
static _Atomic unsigned long g_moe_twin_tiles;
unsigned long gpu_moe_twin_tiles(void) { return g_moe_twin_tiles; }

// one projection over cnt columns with each column's batch-1 bits: eight at
// a time through the width-classed twins where the type has them (and is not
// a float type, whose batch-1 route can be split-K), else one batch-1 call a
// column. `expert`: w is an expert slice, which the fused path runs through
// k_moe_mv_<type>; otherwise a dense tensor (gemma-4's shared MLP), which it
// runs through enc_mv's batch-1 dispatch. The two differ only for Q4_0:
// k_moe_mv_q4_0 is k_mv_q4_0's lane-per-block body, while enc_mv picks the
// coalesced k_gemv_q4_0, whose own twins (k_gemvb_q4_0_x*) give its bits.
static bool moe_twin_gemm(gpu_t *g, model_t *m, gguf_tensor *w, bool expert,
                          CUdeviceptr x, CUdeviceptr y, int n_in, int n_out,
                          int cnt, int xs, int ys) {
    bool lane_q40 = expert && w->type == T_Q4_0;
    bool twins = w->type != T_Q4_0 || lane_q40
               ? moe_twin_type_ok(g->sw, w->type)
               : g->sw->f_gemvb[BW_4][T_Q4_0] && g->sw->f_gemvb[BW_8][T_Q4_0] &&
                 g->sw->f_gemv[T_Q4_0];
    if (!twins) {
        for (int c = 0; c < cnt; c++)
            if (!enc_mv(g, m, w, x + (size_t)c * xs * sizeof(float),
                        y + (size_t)c * ys * sizeof(float), n_in, n_out, 0, 1, xs, ys))
                return false;
        return true;
    }
    CUdeviceptr weights = 0, nob = g->sw->dummy;
    uint64_t w_off = 0;
    if (lane_q40 && !binding_find(g->sw, m, w, &weights, &w_off)) {
        gpu_first_error("tensor '%s' is not resident on the device "
                        "(no binding covers it)", w->name);
        return false;
    }
    for (int c0 = 0; c0 < cnt; c0 += 8) {
        int wd = cnt - c0 < 8 ? cnt - c0 : 8;
        CUdeviceptr xc = x + (size_t)c0 * xs * sizeof(float);
        CUdeviceptr yc = y + (size_t)c0 * ys * sizeof(float);
        bool ok;
        if (lane_q40) {
            // k_moe_mv_q4_0's own body: k_mv_q4_0 for one column (4 rows a
            // block; enc_mv launches it at 8, and a row's bits do not depend
            // on the rows a block holds), its twins for 2 to 8
            mv_args a = { n_in, n_out, w_off, false, wd, xs, ys };
            void *p[] = { &weights, &xc, &yc, &a, &nob };
            ok = wd == 1
               ? launch(g, g->sw->f_mv[T_Q4_0], (unsigned)((n_out + 3) / 4), 1, 1, 128, p)
               : launch(g, g->sw->f_moe_q4_0_mvb[batch_width_class(wd)],
                        (unsigned)((n_out + GEMM_WARPS - 1) / GEMM_WARPS), 1, 1,
                        GEMM_WARPS * 32, p);
        } else {
            ok = wd == 1 ? enc_mv(g, m, w, xc, yc, n_in, n_out, 0, 1, xs, ys)
                         : enc_mv_batch(g, m, w, xc, yc, n_in, n_out, 0, wd, xs, ys);
        }
        if (!ok) return false;
    }
    return true;
}

static bool gpu_moe_ffn_twin(gpu_t *g, model_t *m, const layer_t *ly, int tn,
                             int xdim) {
    int n_embd = m->n_embd, ne = ly->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp;
    int l = (int)(ly - m->layers);
    if (used > MOE_MAX_USED) used = MOE_MAX_USED;
    if (ne  > MOE_ROUTE_MAX) ne  = MOE_ROUTE_MAX;
    // routing as the fused path does it: the router with each column's
    // batch-1 bits (the gpt-oss router bias riding it), then the routing
    // kernel for the tile
    if (!enc_router(g, m, ly->ffn_gate_inp, g->xb, xdim, g->moe_logits, n_embd, ne,
                    g->sw->gib[l], tn))
        return false;
    if (!enc_moe_route(g, g->moe_logits, g->moe_sel, g->moe_selw, ne, used,
                       tn, ne, l, 0))
        return false;
    // the tile's routing, once (prefill is never captured into a graph), as
    // each expert's (token, slot) pairs in ascending token order
    if (cu.StreamSynchronize(g->stream) != 0 ||
        cu.MemcpyDtoH(g->h_moe_sel, g->moe_sel, sizeof(int) * (size_t)tn * used) != 0 ||
        cu.MemcpyDtoH(g->h_moe_selw, g->moe_selw, sizeof(float) * (size_t)tn * used) != 0)
        return false;
    int e_start[MOE_ROUTE_MAX], e_cnt[MOE_ROUTE_MAX], total = 0;
    for (int e = 0; e < ne; e++) {
        e_start[e] = total;
        for (int t = 0; t < tn; t++)
            for (int sl = 0; sl < used; sl++) {
                if (g->h_moe_sel[(size_t)t * used + sl] != e) continue;
                g->h_moe_gidx[total] = t;
                g->h_moe_gpos[total] = t * used + sl;
                g->h_moe_gsel[total] = e;
                g->h_moe_gw[total] = g->h_moe_selw[(size_t)t * used + sl];
                total++;
                break;                          // a token's experts are distinct
            }
        e_cnt[e] = total - e_start[e];
    }
    if (cu.MemcpyHtoD(g->moe_gidx, g->h_moe_gidx, sizeof(int) * (size_t)total) != 0 ||
        cu.MemcpyHtoD(g->moe_gpos, g->h_moe_gpos, sizeof(int) * (size_t)total) != 0 ||
        cu.MemcpyHtoD(g->moe_gsel, g->h_moe_gsel, sizeof(int) * (size_t)total) != 0 ||
        cu.MemcpyHtoD(g->moe_gw, g->h_moe_gw, sizeof(float) * (size_t)total) != 0)
        return false;
    for (int e = 0; e < ne; e++) {
        int cnt = e_cnt[e];
        if (cnt == 0) continue;
        CUdeviceptr idx = g->moe_gidx + (size_t)e_start[e] * sizeof(int);
        CUdeviceptr pos = g->moe_gpos + (size_t)e_start[e] * sizeof(int);
        CUdeviceptr esl = g->moe_gsel + (size_t)e_start[e] * sizeof(int);
        CUdeviceptr wts = g->moe_gw   + (size_t)e_start[e] * sizeof(float);
        gguf_tensor gv = moe_expert_weight(ly, 0, e, n_embd, nff);
        gguf_tensor uv = moe_expert_weight(ly, 1, e, n_embd, nff);
        gguf_tensor dv = moe_expert_weight(ly, 2, e, n_embd, nff);
        void *pp[] = { &g->moe_slots, &g->moe_dout, &pos, &n_embd, &n_embd };
        bool ok = enc_moe_gather(g, g->moe_gath, g->xb, idx, n_embd, cnt, xdim, n_embd)
               && moe_twin_gemm(g, m, &gv, true, g->moe_gath, g->moe_hb, n_embd, nff, cnt, n_embd, nff)
               && moe_twin_gemm(g, m, &uv, true, g->moe_gath, g->moe_hb2, n_embd, nff, cnt, n_embd, nff)
               && enc_moe_actmul(g, m, g->moe_hb, g->moe_hb2, nff, nff, nff, wts,
                                 g->sw->moe_ones, esl, cnt, g->sw->geb[l], g->sw->ueb[l])
               && moe_twin_gemm(g, m, &dv, true, g->moe_hb, g->moe_dout, nff, n_embd, cnt, nff, n_embd)
               && launch(g, g->sw->f_moe_place, (n_embd + 255) / 256, cnt, 1, 256, pp);
        if (!ok) return false;
    }
    // every token's slots summed in slot order into its column of xb
    if (!moe_trace_norms_slots(g, ly, g->moe_slots, 0, tn, n_embd, n_embd, NULL))
        return false;
    void *ps[] = { &g->xb, &g->moe_slots, &n_embd, &used, &n_embd, &g->sw->deb[l],
                   &g->moe_sel, &g->moe_selw, &xdim };
    g_moe_twin_tiles++;
    return launch(g, g->sw->f_moe_sum_tile, (n_embd + 255) / 256, tn, 1, 256, ps);
}

static bool gpu_moe_ffn_path(gpu_t *g, model_t *m, const layer_t *ly, int tn,
                             int xdim) {
    if (g->moe_fused_ok) {
        // grouped: tolerance arithmetic, never a verify round's rows
        if (tn > 1 && g->moe_grouped_ok && !g->exact_rows)
            return gpu_moe_ffn_grouped(g, m, ly, tn, xdim);
        // by expert once the tile averages four or more tokens an expert, the
        // fused path's own bits either way (the int8 activation route has no
        // twins, so it keeps the fused path)
        if (tn > 8 && g->moe_twin_ok && moe_twin_on(m) && !gpu_q8x_on() &&
            (long)tn * m->n_expert_used >= 4L * ly->n_expert)
            return gpu_moe_ffn_twin(g, m, ly, tn, xdim);
        return gpu_moe_ffn_fused(g, m, ly, tn, xdim);
    }
    return gpu_moe_ffn_eager(g, m, ly, tn, xdim);
}

// The tile's trace records start at its column 0 (moe_trace_norms_*), and
// are written once its path has filled in their norms.
// RUNNER_MOE_HOT_PROF: where a mixed layer's time goes, per layer, printed
// at exit. route = the router, its sync and the routing's read-back and
// remap; host = the host experts, computed while the device runs the hot
// bank; wait = the sync after both; fill = the host terms' upload and the
// sum. Single-thread instrument, like RUNNER_CUDA_PROFILE.
static struct {
    int    on, inited, nl;
    double *t[4];          // per layer: route, host, wait, fill (ms)
    long  *calls, *tok, *hit, *miss;
} hotp;

static double hot_now(void) {
#ifdef _WIN32
    static LARGE_INTEGER f;
    LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
#endif
}

static void hot_prof_report(void) {
    double sum[4] = {0};
    long calls = 0, tok = 0, hit = 0, miss = 0;
    for (int l = 0; l < hotp.nl; l++) {
        if (!hotp.calls[l]) continue;
        fprintf(stderr, "moe-hot: blk.%d %ld calls %ld tokens, per call route %.3f host %.3f "
                "wait %.3f fill %.3f ms; hit %.3f of %ld routings\n", l, hotp.calls[l],
                hotp.tok[l], hotp.t[0][l] / hotp.calls[l], hotp.t[1][l] / hotp.calls[l],
                hotp.t[2][l] / hotp.calls[l], hotp.t[3][l] / hotp.calls[l],
                (double)hotp.hit[l] / (double)(hotp.hit[l] + hotp.miss[l]),
                hotp.hit[l] + hotp.miss[l]);
        for (int k = 0; k < 4; k++) sum[k] += hotp.t[k][l];
        calls += hotp.calls[l]; tok += hotp.tok[l]; hit += hotp.hit[l]; miss += hotp.miss[l];
    }
    if (calls)
        fprintf(stderr, "moe-hot: all layers %ld calls %ld tokens, ms route %.1f host %.1f "
                "wait %.1f fill %.1f; hit %.3f of %ld routings\n", calls, tok, sum[0], sum[1],
                sum[2], sum[3], (double)hit / (double)(hit + miss), hit + miss);
}

static void hot_prof_init(const model_t *m) {
    if (hotp.inited) return;
    hotp.inited = 1;
    hotp.on = getenv("RUNNER_MOE_HOT_PROF") != NULL;
    if (!hotp.on) return;
    hotp.nl = m->n_layer;
    for (int k = 0; k < 4; k++) hotp.t[k] = calloc((size_t)m->n_layer, sizeof(double));
    hotp.calls = calloc((size_t)m->n_layer, sizeof(long));
    hotp.tok   = calloc((size_t)m->n_layer, sizeof(long));
    hotp.hit   = calloc((size_t)m->n_layer, sizeof(long));
    hotp.miss  = calloc((size_t)m->n_layer, sizeof(long));
    if (!hotp.t[0] || !hotp.t[1] || !hotp.t[2] || !hotp.t[3] || !hotp.calls ||
        !hotp.tok || !hotp.hit || !hotp.miss) {
        hotp.on = 0;
        return;
    }
    atexit(hot_prof_report);
}

// RUNNER_MOE_HOT (R3.30): a host layer whose hot experts sit in a device bank.
// The device routes as the fused path does; the routing comes back, each slot
// is remapped to its bank slot (a host expert's to slot 0, overwritten
// below), and the device runs the fused path's gate, up, activation and down
// kernels on the bank while the host runs the other experts with the CPU
// path's own grouped expert. Their terms (routing weight times down output)
// replace those slots' rows, and k_moe_sum adds the slots in the router's
// order. Reads the normed rows in g->xb and leaves the FFN output there, as
// gpu_moe_ffn_fused does. A slot's bytes are its backend's; the layer's sum
// is neither placement's, and gpu-identity's tolerance is its gate.
static bool gpu_moe_ffn_hot(gpu_t *g, model_t *m, const layer_t *ly, int tn, int xdim) {
    gpu_weights *w = g->sw;
    int n_embd = m->n_embd, ne = ly->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp;
    int l = (int)(ly - m->layers);
    const int *slot = m->moe_hot_slot + (size_t)l * m->n_expert;
    int S = m->moe_hot_n[l];
    uint64_t gs, us, ds;
    moe_hot_strides(ly, n_embd, nff, &gs, &us, &ds);
    CUdeviceptr bank = w->hot_bank[l];
    hot_prof_init(m);
    double t0 = hotp.on ? hot_now() : 0;
    if (xdim > g->hot_xdim) return false;
    if (!enc_router(g, m, ly->ffn_gate_inp, g->xb, xdim, g->moe_logits, n_embd, ne,
                    w->gib[l], tn) ||
        !enc_moe_route(g, g->moe_logits, g->moe_sel, g->moe_selw, ne, used, tn, ne, l, 0))
        return false;
    size_t nsl = (size_t)tn * used;
    if (cu.StreamSynchronize(g->stream) != 0 ||
        cu.MemcpyDtoH(g->h_moe_sel, g->moe_sel, sizeof(int) * nsl) != 0 ||
        cu.MemcpyDtoH(g->h_moe_selw, g->moe_selw, sizeof(float) * nsl) != 0)
        return false;
    long misses = 0;
    for (size_t i = 0; i < nsl; i++) {
        int e = g->h_moe_sel[i];
        int k = e >= 0 && e < ne ? slot[e] : -1;
        g->h_hot_sel[i] = k >= 0 ? k : 0;
        misses += k < 0;
    }
    if (cu.MemcpyHtoD(g->moe_hot_sel, g->h_hot_sel, sizeof(int) * nsl) != 0 ||
        (misses && cu.MemcpyDtoH(g->h_hot_x, g->xb, sizeof(float) * (size_t)tn * xdim) != 0))
        return false;
    double t1 = hotp.on ? hot_now() : 0, t2 = t1, wait = 0;
    // moe_eout holds max(used, MVB) rows: a chunk of tokens' slots at a time
    int rows = used > g->win ? used : g->win;
    int chunk = rows / used;
    if (chunk < 1) chunk = 1;
    bool host_done = false;
    for (int t0c = 0; t0c < tn; t0c += chunk) {
        int t1c = t0c + chunk < tn ? t0c + chunk : tn;
        for (int t = t0c; t < t1c; t++) {
            CUdeviceptr xin  = g->xb + (size_t)t * xdim * sizeof(float);
            CUdeviceptr hsel = g->moe_hot_sel + (size_t)t * used * sizeof(int);
            CUdeviceptr sel  = g->moe_sel  + (size_t)t * used * sizeof(int);
            CUdeviceptr selw = g->moe_selw + (size_t)t * used * sizeof(float);
            CUdeviceptr eout = g->moe_eout + (size_t)(t - t0c) * used * n_embd * sizeof(float);
            if (!enc_moe_mv_at(g, ly->ffn_gate_exps, bank, 0, gs, xin, g->moe_hb, hsel,
                               n_embd, nff, used, 0, nff) ||
                !enc_moe_mv_at(g, ly->ffn_up_exps, bank, (uint64_t)S * gs, us, xin,
                               g->moe_hb2, hsel, n_embd, nff, used, 0, nff) ||
                !enc_moe_actmul(g, m, g->moe_hb, g->moe_hb2, nff, nff, nff, selw,
                                w->moe_ones, sel, used, 0, 0) ||
                !enc_moe_mv_at(g, ly->ffn_down_exps, bank, (uint64_t)S * (gs + us), ds,
                               g->moe_hb, eout, hsel, nff, n_embd, used, nff, n_embd))
                return false;
        }
        // the host experts of the whole tile, once, while the first chunk runs
        if (!host_done) {
            if (misses && !model_moe_hot_terms(m, l, tn, g->h_hot_x, xdim, g->h_moe_sel,
                                                g->h_moe_selw, g->h_hot_terms))
                return false;
            host_done = true;
            if (hotp.on) t2 = hot_now();
        }
        double ts = hotp.on ? hot_now() : 0;
        if (cu.StreamSynchronize(g->stream) != 0) return false;
        if (hotp.on) wait += hot_now() - ts;
        for (int t = t0c; misses && t < t1c; t++)
            for (int sl = 0; sl < used; sl++) {
                size_t i = (size_t)t * used + sl;
                int e = g->h_moe_sel[i];
                if (e >= 0 && e < ne && slot[e] >= 0) continue;
                CUdeviceptr row = g->moe_eout +
                    ((size_t)(t - t0c) * used + sl) * n_embd * sizeof(float);
                if (cu.MemcpyHtoD(row, g->h_hot_terms + i * n_embd,
                                  sizeof(float) * (size_t)n_embd) != 0)
                    return false;
            }
        for (int t = t0c; t < t1c; t++) {
            CUdeviceptr xin  = g->xb + (size_t)t * xdim * sizeof(float);
            CUdeviceptr sel  = g->moe_sel  + (size_t)t * used * sizeof(int);
            CUdeviceptr selw = g->moe_selw + (size_t)t * used * sizeof(float);
            CUdeviceptr eout = g->moe_eout + (size_t)(t - t0c) * used * n_embd * sizeof(float);
            if (!enc_moe_sum(g, xin, eout, n_embd, used, n_embd, 0, sel, selw)) return false;
        }
    }
    if (hotp.on) {
        double t4 = hot_now();
        hotp.t[0][l] += t1 - t0;
        hotp.t[1][l] += t2 - t1;
        hotp.t[2][l] += wait;
        hotp.t[3][l] += (t4 - t2) - wait;
        hotp.calls[l]++;
        hotp.tok[l] += tn;
        hotp.hit[l] += (long)nsl - misses;
        hotp.miss[l] += misses;
    }
    return true;
}

static bool gpu_moe_ffn(gpu_t *g, model_t *m, const layer_t *ly, int tn,
                        int xdim) {
    if (g->route_trace) moe_trace_flush(g);
    bool ok = gpu_moe_ffn_path(g, m, ly, tn, xdim);
    if (g->route_trace) moe_trace_flush(g);
    return ok;
}

// gemma-4-MoE dual-branch FFN on GPU: a dense GELU shared expert AND a routed
// top-k GELU expert set, each with its own pre/post RMSNorm sandwich, summed.
// Reads the post-attention residual from g->x (UN-normed — each branch norms it
// itself), writes mlp+moe_out into g->xb; the caller then applies post_ffw_norm
// (pfn) and the residual add. Mirrors the CPU gemma_moe_ffn token-for-token.
//
// Fused variant: routing and expert selection on device (k_moe_route +
// indirect matvecs over the fused gate_up tensor), no host round-trip and no
// stream sync — capture-clean, like gpu_moe_ffn_fused.
static bool gpu_gemma_moe_ffn_fused(gpu_t *g, model_t *m, const layer_t *ly,
                                    int l, int tn, int xdim) {
    int n_embd = m->n_embd, ne = ly->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp, dff = ly->n_ff;  // dff = dense shared-FFN size
    if (used > MOE_MAX_USED) used = MOE_MAX_USED;
    if (ne  > MOE_ROUTE_MAX) ne  = MOE_ROUTE_MAX;
    float eps = m->rms_eps;
    // fused gate_up expert block: [n_embd -> 2*nff] per expert
    uint64_t gustride = (uint64_t)(2 * (size_t)nff) *
                        ggml_row_size(ly->ffn_gate_up_exps->type, n_embd);
    uint64_t dstride  = (uint64_t)n_embd *
                        ggml_row_size(ly->ffn_down_exps->type, nff);
    CUdeviceptr xn   = g->g_scr + (size_t)0 * n_embd * sizeof(float);
    CUdeviceptr xn2  = g->g_scr + (size_t)1 * n_embd * sizeof(float);
    CUdeviceptr rin  = g->g_scr + (size_t)2 * n_embd * sizeof(float);
    CUdeviceptr mout = g->g_scr + (size_t)3 * n_embd * sizeof(float);
    for (int t = 0; t < tn; t++) {
        // g->x (the residual stream) is n_embd-strided; g->xb is xdim-strided
        CUdeviceptr attn = g->x  + (size_t)t * n_embd * sizeof(float);
        CUdeviceptr out  = g->xb + (size_t)t * xdim   * sizeof(float);
        // --- dense shared MLP: rmsnorm(ffn_norm) -> GELU SwiGLU -> post_ffw_norm_1.
        //     Written straight into `out`; the routed branch is added on top below.
        bool ok = enc_rmsnorm(g, attn, xn, g->sw->ffn_norm[l], n_embd, eps, 1, xdim, n_embd)
               && enc_mv(g, m, ly->w_gate, xn, g->hb,  n_embd, dff, 0, 1, n_embd, dff)
               && enc_mv(g, m, ly->w_up,   xn, g->hb2, n_embd, dff, 0, 1, n_embd, dff)
               && enc_actmul(g, m, g->hb, g->hb2, dff)
               && enc_mv(g, m, ly->w_down, g->hb, out, dff, n_embd, 0, 1, dff, xdim)
               && enc_rmsnorm(g, out, out, g->sw->g_pn1[l], n_embd, eps, 1, xdim, xdim);
        if (!ok) return false;
        // --- routed experts: pre-norm the branch input (pre_norm2); the router
        //     runs on a SEPARATE weightless-rmsnorm(attn) with (1/sqrt(n_embd) *
        //     gate_inp_scale) folded into the uploaded g_gis weight. Selection
        //     runs on device; per-slot scale = selw * down_exps_scale[expert].
        CUdeviceptr lg   = g->moe_logits + (size_t)t * ne * sizeof(float);
        CUdeviceptr sel  = g->moe_sel  + (size_t)t * used * sizeof(int);
        CUdeviceptr selw = g->moe_selw + (size_t)t * used * sizeof(float);
        CUdeviceptr up   = g->moe_hb + (size_t)nff * sizeof(float);
        ok = enc_rmsnorm(g, attn, xn2, g->sw->g_prn2[l], n_embd, eps, 1, xdim, n_embd)
          && enc_rmsnorm(g, attn, rin, g->sw->g_gis[l],  n_embd, eps, 1, xdim, n_embd)
          && enc_mv(g, m, ly->ffn_gate_inp, rin, lg, n_embd, ne, 0, 1, n_embd, ne)
          && enc_moe_route(g, lg, sel, selw, ne, used, 1, ne, l, t)
          && enc_moe_mv(g, m, ly->ffn_gate_up_exps, gustride, xn2, g->moe_hb,
                        sel, n_embd, 2 * nff, used, 0, 2 * nff)
          && enc_moe_actmul(g, m, g->moe_hb, up, nff, 2 * nff, 2 * nff,
                            selw, g->sw->g_dsc[l], sel, used, 0, 0)
          && enc_moe_mv(g, m, ly->ffn_down_exps, dstride, g->moe_hb,
                        g->moe_eout, sel, nff, n_embd, used, 2 * nff, n_embd)
          && moe_trace_norms_slots(g, ly, g->moe_eout, t, 1, n_embd, n_embd,
                                   ly->down_exps_scale)
          && enc_moe_sum(g, mout, g->moe_eout, n_embd, used, n_embd, 0, 0, 0);
        if (!ok) return false;
        // post_ffw_norm_2 on the routed branch, then out (=dense) += routed
        if (!enc_rmsnorm(g, mout, mout, g->sw->g_pn2[l], n_embd, eps, 1, n_embd, n_embd) ||
            !enc_add(g, out, mout, n_embd, 1, xdim, n_embd))
            return false;
    }
    return true;
}

static bool gpu_gemma_moe_ffn_eager(gpu_t *g, model_t *m, const layer_t *ly, int l,
                              int tn, int xdim) {
    int n_embd = m->n_embd, ne = ly->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp, dff = ly->n_ff;  // dff = dense shared-FFN size
    if (used > MOE_MAX_USED) used = MOE_MAX_USED;
    if (ne  > MOE_ROUTE_MAX) ne  = MOE_ROUTE_MAX;
    float eps = m->rms_eps;
    CUdeviceptr xn   = g->g_scr + (size_t)0 * n_embd * sizeof(float);
    CUdeviceptr xn2  = g->g_scr + (size_t)1 * n_embd * sizeof(float);
    CUdeviceptr rin  = g->g_scr + (size_t)2 * n_embd * sizeof(float);
    CUdeviceptr mout = g->g_scr + (size_t)3 * n_embd * sizeof(float);
    for (int t = 0; t < tn; t++) {
        // g->x (the residual stream) is n_embd-strided; g->xb is xdim-strided
        CUdeviceptr attn = g->x  + (size_t)t * n_embd * sizeof(float);
        CUdeviceptr out  = g->xb + (size_t)t * xdim   * sizeof(float);
        // --- dense shared MLP: rmsnorm(ffn_norm) -> GELU SwiGLU -> post_ffw_norm_1.
        //     Written straight into `out`; the routed branch is added on top below.
        bool ok = enc_rmsnorm(g, attn, xn, g->sw->ffn_norm[l], n_embd, eps, 1, xdim, n_embd)
               && enc_mv(g, m, ly->w_gate, xn, g->hb,  n_embd, dff, 0, 1, n_embd, dff)
               && enc_mv(g, m, ly->w_up,   xn, g->hb2, n_embd, dff, 0, 1, n_embd, dff)
               && enc_actmul(g, m, g->hb, g->hb2, dff)
               && enc_mv(g, m, ly->w_down, g->hb, out, dff, n_embd, 0, 1, dff, xdim)
               && enc_rmsnorm(g, out, out, g->sw->g_pn1[l], n_embd, eps, 1, xdim, xdim);
        if (!ok) return false;
        // --- routed experts: pre-norm the branch input (pre_norm2); the router
        //     runs on a SEPARATE weightless-rmsnorm(attn) with (1/sqrt(n_embd) *
        //     gate_inp_scale) folded into the uploaded g_gis weight
        ok = enc_rmsnorm(g, attn, xn2, g->sw->g_prn2[l], n_embd, eps, 1, xdim, n_embd)
          && enc_rmsnorm(g, attn, rin, g->sw->g_gis[l],  n_embd, eps, 1, xdim, n_embd)
          && enc_mv(g, m, ly->ffn_gate_inp, rin, g->moe_logits, n_embd, ne, 0, 1, n_embd, ne);
        if (!ok) return false;
        if (cu.StreamSynchronize(g->stream) != 0) return false;
        if (cu.MemcpyDtoH(g->h_moe_logits, g->moe_logits, sizeof(float) * (size_t)ne) != 0)
            return false;
        // softmax over all experts, then top-k with renormalized weights
        float *lg = g->h_moe_logits;
        float mx = lg[0];
        for (int e = 1; e < ne; e++) if (lg[e] > mx) mx = lg[e];
        float ssum = 0.0f;
        for (int e = 0; e < ne; e++) { float p = expf(lg[e] - mx); lg[e] = p; ssum += p; }
        for (int e = 0; e < ne; e++) lg[e] /= ssum;
        int sel[MOE_MAX_USED];
        float selw[MOE_MAX_USED], denom = 0.0f;
        const int *forced = route_forced_host(g, t, l, used);
        unsigned long differed = 0;
        for (int s = 0; s < used; s++) {
            int best = 0; float bp = -1.0f;
            for (int e = 0; e < ne; e++) if (lg[e] > bp) { bp = lg[e]; best = e; }
            if (forced) {       // RUNNER_MOE_REPLAY: the recorded choice
                differed += forced[s] != best;
                if (forced[s] < ne) { best = forced[s]; bp = lg[best]; }
            }
            sel[s] = best; selw[s] = bp; denom += bp; lg[best] = -1.0f;
        }
        if (forced) model_moe_replay_count((unsigned long)used, differed, 0);
        // expert_weights_norm false (deepseek2): the softmax weights as they
        // are; x / 1.0f is exact, so the weights below are selw unchanged
        if (!m->expert_norm_w) denom = 1.0f;
        if (!route_trace_host(g, t, l, sel, selw, denom, used)) return false;
        for (int s = 0; s < used; s++) {
            int e = sel[s];
            float sc = (selw[s] / denom) *
                       (ly->down_exps_scale ? ly->down_exps_scale[e] : 1.0f);
            // fused gate_up expert weight: [n_embd -> 2*nff], gate=[0,nff) up=[nff,2nff)
            gguf_tensor guv = *ly->ffn_gate_up_exps;
            size_t rs = ggml_row_size(guv.type, n_embd);
            guv.data = (uint8_t *)ly->ffn_gate_up_exps->data +
                       (size_t)e * (2 * (size_t)nff) * rs;
            guv.nbytes = (2 * (size_t)nff) * rs;  // clamp to the slice (see moe_expert_weight)
            gguf_tensor dv = moe_expert_weight(ly, 2, e, n_embd, nff);
            CUdeviceptr up = g->hb + (size_t)nff * sizeof(float);
            ok = enc_mv(g, m, &guv, xn2, g->hb, n_embd, 2 * nff, 0, 1, n_embd, 2 * nff)
              && enc_actmul(g, m, g->hb, up, nff)         // GELU(gate)*up -> hb[0,nff)
              && enc_scale(g, g->hb, sc, nff, 1, nff);    // fold selw*down_scale
            if (!ok) return false;
            CUdeviceptr dst = s == 0 ? mout : g->moe_eout;
            if (!enc_mv(g, m, &dv, g->hb, dst, nff, n_embd, 0, 1, nff, n_embd) ||
                !moe_trace_norm_one(g, dst, t, s, n_embd, sc) ||
                (s > 0 && !enc_add(g, mout, g->moe_eout, n_embd, 1, n_embd, n_embd)))
                return false;
        }
        // post_ffw_norm_2 on the routed branch, then out (=dense) += routed
        if (!enc_rmsnorm(g, mout, mout, g->sw->g_pn2[l], n_embd, eps, 1, n_embd, n_embd) ||
            !enc_add(g, out, mout, n_embd, 1, xdim, n_embd))
            return false;
    }
    return cu.StreamSynchronize(g->stream) == 0;
}

// gemma-4's dual-branch tile, fused-identical (moe_twin): the fused path's
// per-token sequence with every per-column step run over the tile (the RMS
// norms, the add and the routing kernel compute each column as they do one
// column), the dense shared MLP and the experts' projections through
// moe_twin_gemm (each column the batch-1 bits), the routed experts grouped
// by expert as in gpu_moe_ffn_twin, and each token's slots summed in slot
// order before post_ffw_norm_2.
static bool gpu_gemma_moe_ffn_twin(gpu_t *g, model_t *m, const layer_t *ly,
                                   int l, int tn, int xdim) {
    int n_embd = m->n_embd, ne = ly->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp, dff = ly->n_ff;
    if (used > MOE_MAX_USED) used = MOE_MAX_USED;
    if (ne  > MOE_ROUTE_MAX) ne  = MOE_ROUTE_MAX;
    float eps = m->rms_eps;
    CUdeviceptr xn_b   = g->g_scr;
    CUdeviceptr xn2_b  = g->g_scr + (size_t)g->win * n_embd * sizeof(float);
    CUdeviceptr rin_b  = g->g_scr + (size_t)2 * g->win * n_embd * sizeof(float);
    CUdeviceptr mout_b = g->g_scr + (size_t)3 * g->win * n_embd * sizeof(float);
    // --- dense shared MLP for the tile, written into xb, then post_ffw_norm_1
    bool ok = enc_rmsnorm(g, g->x, xn_b, g->sw->ffn_norm[l], n_embd, eps, tn, n_embd, n_embd)
           && moe_twin_gemm(g, m, ly->w_gate, false, xn_b, g->hb,  n_embd, dff, tn, n_embd, dff)
           && moe_twin_gemm(g, m, ly->w_up,   false, xn_b, g->hb2, n_embd, dff, tn, n_embd, dff)
           && enc_actmul(g, m, g->hb, g->hb2, tn * dff)
           && moe_twin_gemm(g, m, ly->w_down, false, g->hb, g->xb, dff, n_embd, tn, dff, xdim)
           && enc_rmsnorm(g, g->xb, g->xb, g->sw->g_pn1[l], n_embd, eps, tn, xdim, xdim);
    // --- routed branch: the two norms, the router (each column's batch-1
    //     bits), routing
    ok = ok && enc_rmsnorm(g, g->x, xn2_b, g->sw->g_prn2[l], n_embd, eps, tn, n_embd, n_embd)
            && enc_rmsnorm(g, g->x, rin_b, g->sw->g_gis[l], n_embd, eps, tn, n_embd, n_embd);
    ok = ok && enc_router(g, m, ly->ffn_gate_inp, rin_b, n_embd, g->moe_logits,
                          n_embd, ne, 0, tn);
    ok = ok && enc_moe_route(g, g->moe_logits, g->moe_sel, g->moe_selw, ne, used, tn, ne,
                             l, 0);
    if (!ok) return false;
    if (cu.StreamSynchronize(g->stream) != 0 ||
        cu.MemcpyDtoH(g->h_moe_sel, g->moe_sel, sizeof(int) * (size_t)tn * used) != 0 ||
        cu.MemcpyDtoH(g->h_moe_selw, g->moe_selw, sizeof(float) * (size_t)tn * used) != 0)
        return false;
    int e_start[MOE_ROUTE_MAX], e_cnt[MOE_ROUTE_MAX], total = 0;
    for (int e = 0; e < ne; e++) {
        e_start[e] = total;
        for (int t = 0; t < tn; t++)
            for (int sl = 0; sl < used; sl++) {
                if (g->h_moe_sel[(size_t)t * used + sl] != e) continue;
                g->h_moe_gidx[total] = t;
                g->h_moe_gpos[total] = t * used + sl;
                g->h_moe_gsel[total] = e;
                g->h_moe_gw[total] = g->h_moe_selw[(size_t)t * used + sl];
                total++;
                break;
            }
        e_cnt[e] = total - e_start[e];
    }
    if (cu.MemcpyHtoD(g->moe_gidx, g->h_moe_gidx, sizeof(int) * (size_t)total) != 0 ||
        cu.MemcpyHtoD(g->moe_gpos, g->h_moe_gpos, sizeof(int) * (size_t)total) != 0 ||
        cu.MemcpyHtoD(g->moe_gsel, g->h_moe_gsel, sizeof(int) * (size_t)total) != 0 ||
        cu.MemcpyHtoD(g->moe_gw, g->h_moe_gw, sizeof(float) * (size_t)total) != 0)
        return false;
    for (int e = 0; e < ne; e++) {
        int cnt = e_cnt[e];
        if (cnt == 0) continue;
        CUdeviceptr idx = g->moe_gidx + (size_t)e_start[e] * sizeof(int);
        CUdeviceptr pos = g->moe_gpos + (size_t)e_start[e] * sizeof(int);
        CUdeviceptr esl = g->moe_gsel + (size_t)e_start[e] * sizeof(int);
        CUdeviceptr wts = g->moe_gw   + (size_t)e_start[e] * sizeof(float);
        // the expert's slice of the fused gate_up tensor, as the grouped path cuts it
        gguf_tensor guv = *ly->ffn_gate_up_exps;
        size_t rs = ggml_row_size(guv.type, n_embd);
        guv.data = (uint8_t *)ly->ffn_gate_up_exps->data + (size_t)e * (2 * (size_t)nff) * rs;
        guv.nbytes = (2 * (size_t)nff) * rs;
        gguf_tensor dv = moe_expert_weight(ly, 2, e, n_embd, nff);
        CUdeviceptr up = g->moe_hb + (size_t)nff * sizeof(float);
        void *pp[] = { &g->moe_slots, &g->moe_dout, &pos, &n_embd, &n_embd };
        bool eok = enc_moe_gather(g, g->moe_gath, xn2_b, idx, n_embd, cnt, n_embd, n_embd)
                && moe_twin_gemm(g, m, &guv, true, g->moe_gath, g->moe_hb, n_embd, 2 * nff, cnt,
                                 n_embd, 2 * nff)
                && enc_moe_actmul(g, m, g->moe_hb, up, nff, 2 * nff, 2 * nff, wts,
                                  g->sw->g_dsc[l], esl, cnt, 0, 0)
                && moe_twin_gemm(g, m, &dv, true, g->moe_hb, g->moe_dout, nff, n_embd, cnt,
                                 2 * nff, n_embd)
                && launch(g, g->sw->f_moe_place, (n_embd + 255) / 256, cnt, 1, 256, pp);
        if (!eok) return false;
    }
    // every token's slots in slot order (no down bias on gemma), then
    // post_ffw_norm_2 and dense += routed, all per column
    if (!moe_trace_norms_slots(g, ly, g->moe_slots, 0, tn, n_embd, n_embd,
                               ly->down_exps_scale))
        return false;
    CUdeviceptr zero = 0;
    void *ps[] = { &mout_b, &g->moe_slots, &n_embd, &used, &n_embd, &zero,
                   &g->moe_sel, &g->moe_selw, &n_embd };
    g_moe_twin_tiles++;
    return launch(g, g->sw->f_moe_sum_tile, (n_embd + 255) / 256, tn, 1, 256, ps)
        && enc_rmsnorm(g, mout_b, mout_b, g->sw->g_pn2[l], n_embd, eps, tn, n_embd, n_embd)
        && enc_add(g, g->xb, mout_b, n_embd, tn, xdim, n_embd);
}

static bool gpu_gemma_moe_ffn_path(gpu_t *g, model_t *m, const layer_t *ly, int l,
                                   int tn, int xdim) {
    if (g->moe_fused_ok) {
        if (tn > 1 && g->moe_grouped_ok && !g->exact_rows)
            return gpu_gemma_moe_ffn_grouped(g, m, ly, l, tn, xdim);
        if (tn > 8 && g->moe_twin_ok && moe_twin_on(m) && !gpu_q8x_on() &&
            (long)tn * m->n_expert_used >= 4L * ly->n_expert)
            return gpu_gemma_moe_ffn_twin(g, m, ly, l, tn, xdim);
        return gpu_gemma_moe_ffn_fused(g, m, ly, l, tn, xdim);
    }
    return gpu_gemma_moe_ffn_eager(g, m, ly, l, tn, xdim);
}

// as gpu_moe_ffn: the tile's trace records, written with their norms
static bool gpu_gemma_moe_ffn(gpu_t *g, model_t *m, const layer_t *ly, int l,
                              int tn, int xdim) {
    if (g->route_trace) moe_trace_flush(g);
    bool ok = gpu_gemma_moe_ffn_path(g, m, ly, l, tn, xdim);
    if (g->route_trace) moe_trace_flush(g);
    return ok;
}

// Qwen3.5 Gated DeltaNet layer. Learned projections use the same quant-aware
// matvec dispatch as dense attention; only convolution and the recurrent state
// transition are architecture-specific CUDA kernels. Prompt columns are
// advanced in order because both history and state are causal.
static bool gpu_q35_recurrent(gpu_t *g, model_t *m, const layer_t *ly, int l,
                              int tn, int xdim) {
    int sk = m->ssm.state, ng = m->ssm.groups, nh = m->ssm.v_heads;
    int inner = m->ssm.inner, convdim = 2 * sk * ng + inner;
    int histn = m->ssm.conv_kernel - 1;
    CUdeviceptr weight_base = 0;
    uint64_t weight_off = 0;
    if (!binding_find(g->sw, m, ly->ssm_conv, &weight_base, &weight_off))
        return false;
    CUdeviceptr conv_w = weight_base + weight_off;
    int tn_done = 0;   // tokens the whole-tile kernels already advanced

    // each projection, then its adapter delta on its own output, as the
    // CPU walk (qwen35_linear) applies them
    bool ok = enc_mv(g, m, ly->wqkv, g->xb, g->q35_mix,
                     m->n_embd, convdim, 0, tn, xdim, convdim)
           && enc_lora(g, l, LW_QKV, g->xb, g->q35_mix, m->n_embd, convdim,
                       tn, xdim, convdim)
           && enc_mv(g, m, ly->wq_gate, g->xb, g->q35_z,
                     m->n_embd, inner, 0, tn, xdim, inner)
           && enc_lora(g, l, LW_ZGATE, g->xb, g->q35_z, m->n_embd, inner,
                       tn, xdim, inner)
           && enc_mv(g, m, ly->ssm_beta, g->xb, g->q35_beta,
                     m->n_embd, nh, 0, tn, xdim, nh)
           && enc_lora(g, l, LW_BETA, g->xb, g->q35_beta, m->n_embd, nh,
                       tn, xdim, nh)
           && enc_mv(g, m, ly->ssm_alpha, g->xb, g->q35_alpha,
                     m->n_embd, nh, 0, tn, xdim, nh)
           && enc_lora(g, l, LW_ALPHA, g->xb, g->q35_alpha, m->n_embd, nh,
                       tn, xdim, nh);
    if (!ok) return false;

    size_t hist_layer = (size_t)histn * convdim;
    size_t state_layer = (size_t)nh * sk * sk;
    CUdeviceptr hist = g->q35_hist + model_rec_slot(m, l) * hist_layer * sizeof(float);
    CUdeviceptr state = g->q35_state + model_rec_slot(m, l) * state_layer * sizeof(float);
    // the whole tile in one launch per kernel, when no row needs its own
    // checkpoint (speculative keep rows take the loop below, a row a pass)
    bool seq = recur_seq_admits(g, m);
    bool q35_split_ok = seq && sk % Q35D1_COLS == 0 && sk <= Q35D1_MAX_SD &&
                        g->sw->f_q35_delta1 && g->sw->f_q35_delta_norm;
    int kernel = m->ssm.conv_kernel;
    float eps = m->rms_eps;
    int gate_sigmoid = m->hyper_conn ? 1 : 0;   // Qwen3.8's DeltaNet gate
    if (tn > 1 && !g->snap_on && seq) {
        void *pc[] = { &g->q35_mix, &g->q35_cv, &conv_w, &hist, &convdim, &kernel, &tn };
        void *pd[] = { &g->q35_cv, &g->q35_z, &g->q35_beta, &g->q35_alpha,
                       &g->sw->ssm_dt[l], &g->sw->ssm_a[l], &g->sw->ssm_norm[l],
                       &state, &g->xb2, &sk, &ng, &nh, &eps, &gate_sigmoid, &tn,
                       &convdim, &inner, &xdim };
        if (!launch(g, g->sw->f_q35_conv_seq, (convdim + 255) / 256, 1, 1, 256, pc) ||
            !launch(g, g->sw->f_q35_delta_seq, nh, 1, 1, 128, pd))
            return false;
        g_recur_seq_dispatches++;
        tn_done = tn;
    }
    for (int t = tn_done; t < tn; t++) {
        CUdeviceptr mix = g->q35_mix + (size_t)t * convdim * sizeof(float);
        CUdeviceptr cv = g->q35_cv + (size_t)t * convdim * sizeof(float);
        CUdeviceptr z = g->q35_z + (size_t)t * inner * sizeof(float);
        CUdeviceptr beta = g->q35_beta + (size_t)t * nh * sizeof(float);
        CUdeviceptr alpha = g->q35_alpha + (size_t)t * nh * sizeof(float);
        CUdeviceptr out = g->xb2 + (size_t)t * xdim * sizeof(float);
        if (seq) {
            // a one-token decode step, or a checkpointed row of a verify
            // round: the whole-tile conv over one token, then the step with
            // each head's columns over several blocks (k_q35_delta1 and its
            // norm, k_q35_delta's arithmetic) where the state width allows,
            // else the whole-tile kernel over one token (GPU_ID_RECUR's
            // identity).
            int one = 1;
            void *pc[] = { &mix, &cv, &conv_w, &hist, &convdim, &kernel, &one };
            if (!launch(g, g->sw->f_q35_conv_seq, (convdim + 255) / 256, 1, 1, 256, pc))
                return false;
            if (q35_split_ok) {
                void *p1[] = { &cv, &beta, &alpha, &g->sw->ssm_dt[l], &g->sw->ssm_a[l],
                               &state, &out, &sk, &ng, &nh, &eps };
                void *p2[] = { &z, &g->sw->ssm_norm[l], &out, &sk, &nh, &eps, &gate_sigmoid };
                if (!launch(g, g->sw->f_q35_delta1, nh, (unsigned)(sk / Q35D1_COLS), 1, 128, p1) ||
                    !launch(g, g->sw->f_q35_delta_norm, nh, 1, 1, 128, p2))
                    return false;
            } else {
                void *pd[] = { &cv, &z, &beta, &alpha, &g->sw->ssm_dt[l],
                               &g->sw->ssm_a[l], &g->sw->ssm_norm[l], &state, &out,
                               &sk, &ng, &nh, &eps, &gate_sigmoid, &one,
                               &convdim, &inner, &xdim };
                if (!launch(g, g->sw->f_q35_delta_seq, nh, 1, 1, 128, pd))
                    return false;
            }
            g_recur_seq_dispatches++;
        } else {
            void *pc[] = { &mix, &cv, &conv_w, &hist, &convdim, &kernel };
            if (!launch(g, g->sw->f_q35_conv, (convdim + 255) / 256,
                        1, 1, 256, pc)) return false;
            void *pd[] = { &cv, &z, &beta, &alpha, &g->sw->ssm_dt[l],
                           &g->sw->ssm_a[l], &g->sw->ssm_norm[l], &state,
                           &out, &sk, &ng, &nh, &eps, &gate_sigmoid };
            if (!launch(g, g->sw->f_q35_delta, nh, 1, 1, Q35D_COLS, pd)) return false;
        }
        if (g->snap_on && t < g->snap_rows) {
            // checkpoint after this row, in stream order behind the kernels
            size_t hb = sizeof(float) * model_rec_count(m) * hist_layer;
            size_t sb = sizeof(float) * model_rec_count(m) * state_layer;
            if ((hist_layer && cu.MemcpyDtoDAsync(g->snap_hist + (size_t)t * hb + model_rec_slot(m, l) * hist_layer * sizeof(float),
                                                   hist, hist_layer * sizeof(float), g->stream) != 0) ||
                cu.MemcpyDtoDAsync(g->snap_state + (size_t)t * sb + model_rec_slot(m, l) * state_layer * sizeof(float),
                                   state, state_layer * sizeof(float), g->stream) != 0)
                return false;
        }
    }
    return enc_mv(g, m, ly->ssm_out, g->xb2, g->xb, inner, m->n_embd,
                  0, tn, xdim, xdim)
        && enc_lora(g, l, LW_SSMO, g->xb2, g->xb, inner, m->n_embd,
                    tn, xdim, xdim);
}

// One Mamba-2 recurrent layer (granitehybrid / nemotron_h). Mirrors the CPU
// mamba2_ssd_step: in_proj -> per-token (conv1d+silu, SSD scan, silu-gate +
// grouped RMSNormGated) -> out_proj. Conv ring and SSD state are persistent
// device buffers advanced in place; tokens run in order because the state is
// causal. Recurrent layers force graph_bad, so this always runs eagerly.
static bool gpu_mamba2_recurrent(gpu_t *g, model_t *m, const layer_t *ly,
                                 int l, int tn, int xdim) {
    int nh = m->ssm.v_heads, ng = m->ssm.groups, ds = m->ssm.state;
    int inner = m->ssm.inner, hd = inner / nh, gsz = nh / ng;
    int conv_dim = inner + 2 * ng * ds;
    int d_in_proj = 2 * inner + 2 * ng * ds + nh;
    int histn = m->ssm.conv_kernel - 1, kk = m->ssm.conv_kernel;
    int per_g = inner / ng;
    float eps = m->rms_eps;
    CUdeviceptr conv_base = 0; uint64_t conv_off = 0;
    if (!binding_find(g->sw, m, ly->ssm_conv, &conv_base, &conv_off))
        return false;
    CUdeviceptr conv_w = conv_base + conv_off;
    // in_proj: attn-normed input (g->xb) -> zxBCdt (g->mamba_proj)
    if (!enc_mv(g, m, ly->ssm_in, g->xb, g->mamba_proj,
                m->n_embd, d_in_proj, 0, tn, xdim, d_in_proj))
        return false;
    CUdeviceptr ring  = g->mamba_conv +
                        model_rec_slot(m, l) * histn * conv_dim * sizeof(float);
    // the layer's SSD state: read from the forward's input half on its first
    // tile (later tiles continue in the half it writes), written to rec_cur
    size_t soff = model_rec_slot(m, l) * nh * hd * ds * sizeof(float);
    CUdeviceptr state = mamba_half(g, m, g->rec_cur) + soff;
    CUdeviceptr state_in = mamba_half(g, m, g->rec_tile0 ? g->rec_prev : g->rec_cur) + soff;
    CUdeviceptr Aw = g->sw->ssm_a[l], dtw = g->sw->ssm_dt[l];
    CUdeviceptr Dw = g->sw->ssm_D[l], cb = g->sw->ssm_conv_b[l];
    CUdeviceptr gn = g->sw->ssm_gnorm[l];
    // a verify round filing its rows (mamba_snap_ready): rows before the
    // last leave their state and conv window in their slots (row t's at
    // slot t * snap_nrec + this layer's ordinal), the last row writes the
    // live half, on the tile kernels or the per-token ones alike
    bool snap = g->snap_on && g->snap_ord && g->snap_ord[l] >= 0;
    size_t sl = (size_t)nh * hd * ds, hl = (size_t)histn * conv_dim;
    CUdeviceptr ssnap = snap ? g->snap_state + (size_t)g->snap_ord[l] * sl * sizeof(float) : 0;
    CUdeviceptr hsnap = snap && hl ? g->snap_hist + (size_t)g->snap_ord[l] * hl * sizeof(float) : 0;
    int sstride = snap ? (int)((size_t)g->snap_nrec * sl) : 0;
    int hstride = snap ? (int)((size_t)g->snap_nrec * hl) : 0;
    if (tn > 1 && recur_seq_admits(g, m)) {
        // conv and scan over the whole tile, one launch each; the gate and
        // norm read only their own token's row, so one launch with a grid
        // row per token keeps each token's bytes
        void *pc[] = { &g->mamba_proj, &g->mamba_xBC, &conv_w, &cb, &ring,
                       &conv_dim, &inner, &d_in_proj, &kk, &histn, &tn, &hsnap, &hstride };
        void *ps[] = { &g->mamba_xBC, &g->mamba_proj, &Aw, &dtw, &Dw, &state_in, &state,
                       &g->mamba_y, &inner, &hd, &ds, &ng, &gsz, &conv_dim,
                       &d_in_proj, &tn, &ssnap, &sstride };
        if (!launch(g, g->sw->f_mamba_conv_seq, (conv_dim + 255) / 256, 1, 1, 256, pc) ||
            !launch(g, g->sw->f_mamba_ssd_seq, nh, 1, 1, hd, ps))
            return false;
        int t0 = 0;
        void *pg[] = { &g->mamba_proj, &g->mamba_y, &gn, &inner, &ng, &per_g,
                       &eps, &d_in_proj, &t0 };
        if (!launch(g, g->sw->f_mamba_gate_norm, ng, tn, 1, 256, pg))
            return false;
        g_recur_seq_dispatches++;
        return enc_mv(g, m, ly->ssm_out, g->mamba_y, g->xb, inner, m->n_embd,
                      0, tn, inner, xdim);
    }
    for (int t = 0; t < tn; t++) {
        void *pc[] = { &g->mamba_proj, &g->mamba_xBC, &conv_w, &cb, &ring,
                       &conv_dim, &inner, &d_in_proj, &kk, &histn, &t };
        if (!launch(g, g->sw->f_mamba_conv, (conv_dim + 255) / 256, 1, 1,
                    256, pc)) return false;
        // a filed row writes its slot (the window copied after its conv
        // step), and token t > 0 continues from the row token t - 1 wrote
        bool filed = snap && t < tn - 1;
        if (filed && hl &&
            cu.MemcpyDtoDAsync(hsnap + (size_t)t * hstride * sizeof(float), ring,
                               hl * sizeof(float), g->stream) != 0)
            return false;
        CUdeviceptr sin = t == 0 ? state_in :
                          snap ? ssnap + (size_t)(t - 1) * sstride * sizeof(float) : state;
        CUdeviceptr sout = filed ? ssnap + (size_t)t * sstride * sizeof(float) : state;
        void *ps[] = { &g->mamba_xBC, &g->mamba_proj, &Aw, &dtw, &Dw, &sin, &sout,
                       &g->mamba_y, &inner, &hd, &ds, &ng, &gsz, &conv_dim,
                       &d_in_proj, &t };
        if (!launch(g, g->sw->f_mamba_ssd, nh, 1, 1, hd, ps)) return false;
        void *pg[] = { &g->mamba_proj, &g->mamba_y, &gn, &inner, &ng, &per_g,
                       &eps, &d_in_proj, &t };
        if (!launch(g, g->sw->f_mamba_gate_norm, ng, 1, 1, 256, pg))
            return false;
    }
    // out_proj: y (g->mamba_y, stride inner) -> g->xb (stride xdim)
    return enc_mv(g, m, ly->ssm_out, g->mamba_y, g->xb, inner, m->n_embd,
                  0, tn, inner, xdim);
}

// ---- qwen4exp hyper-connections on the device (twins of model.c hc_mix /
// hc_combine; the same ops in the same order, the reductions per stream)
// mixed (n_embd, stride ys) = mean_c xn_c * sigmoid(W_up silu(W_down xn / hc));
// the inject logits (hc per row) are left in g->hc_inject for hc_combine.
static bool gpu_hc_mix(gpu_t *g, model_t *m, CUdeviceptr w_norm,
                       gguf_tensor *w_down, gguf_tensor *w_up, gguf_tensor *w_inject,
                       CUdeviceptr x_hc_rows, CUdeviceptr y, int ys, int tn) {
    int E = m->n_embd, hc = m->hc_count, hcd = hc * E, lr = m->hc_low_rank;
    float eps = m->rms_eps;
    int xs = hcd;
    void *pn[] = { &x_hc_rows, &g->hc_xn, &w_norm, &E, &hc, &eps, &xs, &xs };
    if (!launch(g, g->sw->f_rmsnorm_grouped, hc, tn, 1, 256, pn)) return false;
    if (!enc_mv(g, m, w_down, g->hc_xn, g->hc_lo, hcd, lr, 0, tn, hcd, lr)) return false;
    {
        float inv_hc = 1.0f / (float)hc;
        int n = lr;
        void *ps[] = { &g->hc_lo, &inv_hc, &n, &n };
        if (!launch(g, g->sw->f_silu_scale, (lr + 255) / 256, tn, 1, 256, ps)) return false;
    }
    if (!enc_mv(g, m, w_up, g->hc_lo, g->hc_gate, lr, hcd, 0, tn, lr, hcd)) return false;
    {
        void *pm[] = { &g->hc_xn, &g->hc_gate, &y, &E, &hc, &xs, &ys };
        if (!launch(g, g->sw->f_hc_gate_mean, (E + 255) / 256, tn, 1, 256, pm)) return false;
    }
    if (w_inject &&
        !enc_mv(g, m, w_inject, g->hc_xn, g->hc_inject, hcd, hc, 0, tn, hcd, hc)) return false;
    return true;
}

static bool gpu_hc_combine(gpu_t *g, model_t *m, CUdeviceptr block_out, int os, int tn) {
    int E = m->n_embd, hc = m->hc_count, hcd = hc * E;
    void *pc[] = { &g->x_hc, &block_out, &g->hc_inject, &E, &hc, &hcd, &os };
    return launch(g, g->sw->f_hc_combine, (E + 255) / 256, tn, 1, 256, pc);
}

// Q, through the DeltaNet split on qwen35 (Q and the output gate share one
// projection), else with its adapter delta and the output gate.
static bool tile_q(gpu_t *g, model_t *m, layer_t *ly, int l, int tn,
                   int n_embd, int xdim, int hd, int q_dim) {
    bool ok = true;
    if (m->qwen35) {
        // Q and the output gate share the projection, interleaved per head;
        // the adapter spans the whole output (attn_q's rows), before the split
        ok = ok && enc_mv(g, m, ly->wq, g->xb, g->q35_mix,
                          n_embd, 2 * q_dim, g->sw->bq[l], tn, xdim,
                          2 * q_dim)
                && enc_lora(g, l, LW_Q, g->xb, g->q35_mix, n_embd, 2 * q_dim,
                            tn, xdim, 2 * q_dim);
        int heads = m->n_head, packed_stride = 2 * q_dim, q_stride = q_dim;
        void *ps[] = { &g->q35_mix, &g->q, &g->q35_gate, &heads, &hd,
                       &packed_stride, &q_stride };
        ok = ok && launch(g, g->sw->f_q35_split, (q_dim + 255) / 256,
                          tn, 1, 256, ps);
    } else {
        ok = ok && enc_mv(g, m, ly->wq, g->xb, g->q, n_embd, q_dim,
                          g->sw->bq[l], tn, xdim, q_dim)
                && enc_lora(g, l, LW_Q, g->xb, g->q, n_embd, q_dim,
                            tn, xdim, q_dim);
        // the output gate projects the same normed input as Q, and xb is
        // overwritten by wo below, so this must run before attention
        // spark2_5's gate projects to one value per head, not per output
        if (m->attn_out_gate && ly->wq_gate)
            ok = ok && enc_mv_same_x(g, m, ly->wq_gate, g->xb, g->q35_gate,
                                     n_embd, m->attn_gate_heads ? m->n_head : q_dim,
                                     0, tn, xdim, q_dim);
    }
    return ok;
}

// Zero a device buffer in stream order (capturable), or synchronously where
// the driver has no cuMemsetD8Async (see v_from_k_capturable).
static bool enc_zero(gpu_t *g, CUdeviceptr p, size_t bytes) {
    if (cu.MemsetD8Async && g->stream) return cu.MemsetD8Async(p, 0, bytes, g->stream) == 0;
    return cu.MemsetD8(p, 0, bytes) == 0;
}

// K and V with their adapter deltas; gemma4 global layers copy K as V.
static bool tile_kv_proj(gpu_t *g, model_t *m, layer_t *ly, int l, int tn,
                         int n_embd, int xdim, int kv_dim) {
    bool ok = true;
    // xb is Q's input (tile_q, just before): K and V reuse its staging
    ok = ok && enc_mv_same_x(g, m, ly->wk, g->xb, g->kt, n_embd, kv_dim, g->sw->bk[l], tn, xdim, kv_dim)
            && enc_lora(g, l, LW_K, g->xb, g->kt, n_embd, kv_dim, tn, xdim, kv_dim);
    if (ly->wv) {
        ok = ok && enc_mv_same_x(g, m, ly->wv, g->xb, g->vt, n_embd, kv_dim, g->sw->bv[l], tn, xdim, kv_dim)
                && enc_lora(g, l, LW_V, g->xb, g->vt, n_embd, kv_dim, tn, xdim, kv_dim);
    } else {
        // gemma4 global layers have no V projection: V is the raw K
        // (zero + add = device-side copy with the kernels we have)
        ok = ok && enc_zero(g, g->vt, sizeof(float) * (size_t)tn * kv_dim);
        ok = ok && enc_add(g, g->vt, g->kt, kv_dim, tn, kv_dim, kv_dim);
    }
    return ok;
}

// deepseek2's Q, K and V for every column in the naive multi-head latent
// attention (R4.4.5 c1), the device twin of model.c's mla_project: Q in the
// publisher's per-head order into kt, then as [pe | nope] into q; the latent
// and the shared rope key from attn_kv_a_mqa, the latent normed, kv_b's
// per-head [k_nope | v]; then K [k_pe | k_nope] into kt and V [v | 0] into
// vt. Here kv_dim == q_dim (every head owns its K). The q-lora form and the
// absorbed form are refused at admission (cuda_admit, model.c).
static bool tile_mla_project(gpu_t *g, model_t *m, layer_t *ly, int l, int tn,
                             int n_embd, int xdim) {
    int H = m->n_head, hd = model_head_dim(m, l), rd = m->rope_dim, r = m->mla_kv_lora;
    int vd = m->mla_v_dim, kvb = H * (hd - rd + vd);
    int q_dim = model_q_dim(m, l), kv_dim = model_kv_dim(m, l);
    CUdeviceptr kva = g->mla_buf;
    CUdeviceptr lat = kva + sizeof(float) * (size_t)g->win * (size_t)(r + rd);
    CUdeviceptr kvo = lat + sizeof(float) * (size_t)g->win * (size_t)r;
    if (ly->wq_a) return false;
    if (m->mla_absorb) {
        // the absorbed form (R4.4.5 c2): the raw Q [heads x key_length] into
        // the scratch (kvo's place), then per head [q_pe | k_b^T q_nope] into
        // q (hd = rope + kv_lora wide); one KV head [k_pe | latent] into kt
        // and vt (kv_dim = hd)
        int qk = m->mla_qk, kvw = qk - rd + vd, qr_w = H * qk;
        bool ok = enc_mv(g, m, ly->wq, g->xb, kvo, n_embd, qr_w, 0, tn, xdim, qr_w)
               && enc_mv_same_x(g, m, ly->wkv_a, g->xb, kva, n_embd, r + rd, 0, tn, xdim, r + rd)
               && enc_rmsnorm(g, kva, lat, g->sw->mla_kvn[l], r, m->rms_eps, tn, r + rd, r);
        void *pq[] = { &kvo, &g->sw->mla_kvb[l], &g->q, &H, &qk, &rd, &r, &kvw, &q_dim };
        ok = ok && launch(g, g->sw->f_mla_q_absorb, (unsigned)((hd + 255) / 256), H, tn, 256, pq);
        void *pk[] = { &kva, &lat, &g->kt, &g->vt, &r, &rd, &kv_dim };
        ok = ok && launch(g, g->sw->f_mla_kv_absorb, (unsigned)((hd + 255) / 256), tn, 1, 256, pk);
        return ok;
    }
    if (kv_dim != q_dim) return false;
    bool ok = enc_mv(g, m, ly->wq, g->xb, g->kt, n_embd, q_dim, 0, tn, xdim, kv_dim)
           && enc_mv_same_x(g, m, ly->wkv_a, g->xb, kva, n_embd, r + rd, 0, tn, xdim, r + rd)
           && enc_rmsnorm(g, kva, lat, g->sw->mla_kvn[l], r, m->rms_eps, tn, r + rd, r)
           && enc_mv(g, m, ly->wkv_b, lat, kvo, r, kvb, 0, tn, r, kvb);
    unsigned gx = (unsigned)((H * hd + 255) / 256);
    void *pq[] = { &g->kt, &g->q, &H, &hd, &rd, &q_dim };
    ok = ok && launch(g, g->sw->f_mla_q, gx, tn, 1, 256, pq);
    void *pk[] = { &kva, &kvo, &g->kt, &g->vt, &H, &hd, &rd, &r, &vd, &kv_dim };
    ok = ok && launch(g, g->sw->f_mla_kv, gx, tn, 1, 256, pk);
    return ok;
}

// The V and K norms, K's rotation, and the store into the cache.
static bool tile_kv_store(gpu_t *g, model_t *m, int l, int tn, int hd,
                          int n_kv, int kv_dim) {
    bool ok = true;
    if (m->v_rmsnorm)
        // weightless per-head RMS norm on V (pre-K-norm values)
        ok = ok && enc_qknorm(g, m, g->vt, g->sw->ones, n_kv, hd, tn, kv_dim);
    if (g->sw->kn[l]) ok = ok && enc_qknorm(g, m, g->kt, g->sw->kn[l], n_kv, hd, tn, kv_dim);
    if (model_layer_ropes(m, l))
        ok = ok && enc_rope(g, m, g->kt, n_kv, tn, kv_dim, l);

    {
        // cache rows for consecutive positions are contiguous, so the
        // kernel indexes both source column and destination row by grid.y
        uint64_t l_off = model_k_byte_off(m, l), v_off = model_v_byte_off(m, l);
        int q8 = model_kv_kind_k(m), vq8 = model_kv_kind_v(m);   // cache KINDS
        // one thread per stored unit: per value for fp16, per 32-value
        // q8_0 block or 16-value fp4 block (the block shares one scale);
        // the grid covers the larger of the K and V unit counts
        int ku = kv_store_units(q8, kv_dim), vu = kv_store_units(vq8, kv_dim);
        int units = ku > vu ? ku : vu;
        int kring = model_kv_is_ring(m, l) ? m->kv_ring : 0;
        // a shared V (the absorbed MLA form) is K's table: its store writes
        // the rotated K row over itself, the same bytes in the same thread,
        // where vt's unrotated copy would overwrite k_pe
        CUdeviceptr *vsrc = model_kv_v_shared(m) ? &g->kt : &g->vt;
        void *ps[] = { &g->kt, vsrc, &g->kc, &g->vc, &kv_dim, &l_off,
                       &g->pos_dev, &q8, &kring, &v_off, &vq8 };
        ok = ok && launch(g, g->sw->f_store, (units + 63) / 64, tn, 1, 64, ps);
    }
    return ok;
}

// RUNNER_CUDA_ATTN_TILE: the tiled prefill attention (k_attn_tile*), on by
// default since its float32-reference rows were never further from the
// reference than k_attn's; 0 keeps k_attn. A tolerance route: recorded as
// attn-tile, and --gpu-route reference runs k_attn. Head widths 257 to 512
// (gemma-4's global layers) take k_attn_tile8.

static bool attn_tile_on(void) {
    if (route_pinned(GR_ATTN_TILE) >= 0) return route_pinned(GR_ATTN_TILE);
    static _Atomic int on = -1;
    if (on < 0) {
        const char *e = getenv("RUNNER_CUDA_ATTN_TILE");
        on = !(e && *e && strcmp(e, "0") == 0);
    }
    return on == 1;
}

// The tiled prefill attention over a whole tile of tn query columns, or
// false (nothing launched) when it does not apply: switch off, kernel
// absent from the PTX, a head width past 512, an odd head width, or a GQA
// ratio that does not divide. Its launch shape and shared footprint are
// k_attn_tile's (see kernels.cu): 32 query rows of 32-key tiles at head_dim
// <= 128, 16 of 16 at <= 256, 8 of 8 at <= 512, the rows a block's heads
// times its strip of tokens.
static bool tile_attn_tiled(gpu_t *g, model_t *m, int tn, int hd, int n_kv,
                            const attn_args *aa, CUdeviceptr sinks, bool *ok) {
    if (!attn_tile_on() || hd % 2 || n_kv < 1 || m->n_head % n_kv) return false;
    int R = hd <= 128 ? 32 : hd <= 256 ? 16 : hd <= 512 ? 8 : 0;
    CUfunction f = R == 32 ? g->sw->f_attn_tile32 : R == 16 ? g->sw->f_attn_tile16 :
                   R == 8 ? g->sw->f_attn_tile8 : NULL;
    if (!f) return false;
    int G = m->n_head / n_kv;
    int hb = G < R ? G : R, qs = R / hb, ngh = (G + hb - 1) / hb;
    int kw = (int)kv_bytes_of_kind(model_kv_kind_k(m), (size_t)hd);
    int vw = (int)kv_bytes_of_kind(model_kv_kind_v(m), (size_t)hd);
    kw = ((kw + 3) / 4) | 1;
    vw = ((vw + 3) / 4) | 1;
    unsigned smem = (unsigned)sizeof(float) *
                    (unsigned)(R * (hd + 1) + R * (kw > vw ? kw : vw) + R * R + 2 * R);
    if (smem > 48 * 1024) return false;
    attn_args a = *aa;
    void *pa[] = { &g->q, &g->kc, &g->vc, &g->xb2, &a, &g->pos_dev, &sinks,
                   &tn, &hb, &qs };
    *ok = launch_smem(g, f, (unsigned)(n_kv * ngh), (unsigned)((tn + qs - 1) / qs), 1,
                      256, smem, pa);
    if (*ok) g_attn_tiled_dispatches++;
    return true;
}

// RUNNER_CUDA_ATTN_TC: the tensor-core prefill attention (k_attn_tc) for fp16
// caches, ahead of the tiled kernel; it carries Q and P in two halves, which
// reads the same as k_attn where single halves read further from the reference.
// On by default for the families whose long-context float32-reference rows
// passed against attn-tile (about 3x the tile's prefill at 8K). gpt-oss
// failed its row and windowed files have none, so they keep the tile; 1 / 0
// force it on / off everywhere. Rows: docs/tc-promotion-evidence.md.
static bool attn_tc_family(const model_t *m) {
    static const char *const fam[] = { "llama", "qwen2", "qwen3", "qwen3moe", "granite", "phi3" };
    if (!m || m->swa_window > 0) return false;
    for (size_t i = 0; i < sizeof fam / sizeof *fam; i++)
        if (strcmp(m->arch, fam[i]) == 0) return true;
    return false;
}
static bool attn_tc_on(const model_t *m) {
    if (route_pinned(GR_ATTN_TC) >= 0) return route_pinned(GR_ATTN_TC);
    static _Atomic int on = -1;   // -1 unread, 2 unset (the family default)
    if (on < 0) {
        const char *e = getenv("RUNNER_CUDA_ATTN_TC");
        on = !e || !*e ? 2 : strcmp(e, "0") != 0;
    }
    return on == 2 ? attn_tc_family(m) : on == 1;
}
static _Atomic unsigned long g_attn_tc_dispatches;
unsigned long gpu_attn_tc_dispatches(void) { return g_attn_tc_dispatches; }

// The tensor-core prefill attention over a whole tile, or false when it does
// not apply: switch off, kernel absent, a quantized K or V cache, or a head
// width that is not a multiple of 16 up to 128. 64 query rows a block.
static bool tile_attn_tc(gpu_t *g, model_t *m, int tn, int hd, int n_kv,
                         const attn_args *aa, CUdeviceptr sinks, bool *ok) {
    if (!attn_tc_on(m) ||
        !g->sw->f_attn_tc || model_kv_kind_k(m) || model_kv_kind_v(m) ||
        hd % 16 || hd > 128 || n_kv < 1 || m->n_head % n_kv)
        return false;
    int G = m->n_head / n_kv, R = 64;
    int hb = G < R ? G : R, qs = R / hb, ngh = (G + hb - 1) / hb;
    attn_args a = *aa;
    void *pa[] = { &g->q, &g->kc, &g->vc, &g->xb2, &a, &g->pos_dev, &sinks,
                   &tn, &hb, &qs };
    *ok = launch_smem(g, g->sw->f_attn_tc, (unsigned)(n_kv * ngh), (unsigned)((tn + qs - 1) / qs),
                      1, 128, AQ_SMEM, pa);
    if (*ok) { g_attn_tc_dispatches++; g_attn_tiled_dispatches++; }
    return true;
}

// RUNNER_CUDA_ATTN_TC_WIDE: the tensor-core prefill attention for head widths
// 256 to 512 (k_attn_tcw), where k_attn_tc does not reach. On by default for
// gemma-4's E-series (its 512-wide global layers), qwen35 at head width 256,
// and deepseek2's absorbed row through k_attn_tcw_mla; 0 keeps the tile
// there, 1 forces it on every model it admits.
// The absorbed MLA row k_attn_tcw_mla takes, as the column its latent starts
// at, or -1: one table (V is the K row, model_kv_v_shared), a rope slice of
// whole 8-wide groups, quarters of the row that are whole k16 steps, and a
// latent of whole 128-wide quarters up to 512, within AW_SMEM(576).
static int tcw_mla_vlo(const model_t *m, int hd) {
    int rd = m->rope_dim, dv = hd - rd;
    if (!model_kv_v_shared(m) || rd < 8 || rd % 8 || hd > 576) return -1;
    return (hd / 4) % 16 == 0 && dv % 128 == 0 && dv <= 512 ? rd : -1;
}

static bool attn_tcw_family(const model_t *m) {
    if (!m) return false;
    if (strcmp(m->arch, "gemma4") == 0) return m->n_embd_ple > 0;
    if (m->deepseek2) return tcw_mla_vlo(m, model_head_dim(m, 0)) >= 0;
    return strcmp(m->arch, "qwen35") == 0 && m->head_dim == 256;
}
static bool attn_tcw_on(const model_t *m) {
    static _Atomic int on = -1;   // -1 unread, 2 unset (the family default)
    if (on < 0) {
        const char *e = getenv("RUNNER_CUDA_ATTN_TC_WIDE");
        on = !e || !*e ? 2 : strcmp(e, "0") != 0;
    }
    return on == 2 ? attn_tcw_family(m) : on == 1;
}
static _Atomic unsigned long g_attn_tcw_dispatches;
unsigned long gpu_attn_tcw_dispatches(void) { return g_attn_tcw_dispatches; }
static _Atomic unsigned long g_attn_tcw_mla_dispatches;
unsigned long gpu_attn_tcw_mla_dispatches(void) { return g_attn_tcw_mla_dispatches; }

// The wide-head tensor-core prefill attention over a whole tile, or false
// when it does not apply: switch off, kernel absent, a quantized K or V
// cache, sinks, or a head width that is not 256, 384 or 512 (whole 32-wide
// quarters a warp) and not an absorbed MLA row k_attn_tcw_mla takes. 16
// query rows a block.
static bool tile_attn_tcw(gpu_t *g, model_t *m, int tn, int hd, int n_kv,
                          const attn_args *aa, CUdeviceptr sinks, bool *ok) {
    if (!attn_tcw_on(m) || model_kv_kind_k(m) || model_kv_kind_v(m) || sinks ||
        n_kv < 1 || m->n_head % n_kv)
        return false;
    int vlo = g->sw->f_attn_tcw_mla ? tcw_mla_vlo(m, hd) : -1;
    if (vlo < 0 && (!g->sw->f_attn_tcw || hd % 128 || hd < 256 || hd > 512))
        return false;
    int G = m->n_head / n_kv, R = 16;
    int hb = G < R ? G : R, qs = R / hb, ngh = (G + hb - 1) / hb;
    attn_args a = *aa;
    void *pa[] = { &g->q, &g->kc, &g->vc, &g->xb2, &a, &g->pos_dev, &sinks,
                   &tn, &hb, &qs, &vlo };
    *ok = launch_smem(g, vlo >= 0 ? g->sw->f_attn_tcw_mla : g->sw->f_attn_tcw,
                      (unsigned)(n_kv * ngh), (unsigned)((tn + qs - 1) / qs),
                      1, 128, AW_SMEM(hd), pa);
    if (*ok) {
        g_attn_tcw_dispatches++; g_attn_tiled_dispatches++;
        if (vlo >= 0) g_attn_tcw_mla_dispatches++;
    }
    return true;
}

// RUNNER_CUDA_ATTN_DEC: the grouped decode attention (k_attn_dec_gqa, one
// block per KV head and slice serving its query heads), on by default since
// its float32-reference rows matched k_attn_dec's (the same three models);
// 0 keeps k_attn_dec. Recorded as attn-dec.
static bool attn_dec_on(void) {
    if (route_pinned(GR_ATTN_DEC) >= 0) return route_pinned(GR_ATTN_DEC);
    static _Atomic int on = -1;
    if (on < 0) {
        const char *e = getenv("RUNNER_CUDA_ATTN_DEC");
        on = !(e && *e && strcmp(e, "0") == 0);
    }
    return on == 1;
}
static _Atomic unsigned long g_attn_dec_gqa_dispatches;
unsigned long gpu_attn_dec_gqa_dispatches(void) { return g_attn_dec_gqa_dispatches; }

// The arithmetic routes in force (gpu.h): what a record states and --verify
// checks after it has pinned them. With an instance, sm80 counts only when
// its compute_80 module actually loaded (a device under 8.0 runs TC2's
// compute_75 kernels whatever was asked).
int gpu_route_active(const model_t *m) {
    const gpu_t *g = m ? m->gpu : NULL;
    // TC2 as this instance runs it: auto counts only where the compute_80
    // kernels loaded (there it is the TC route's arithmetic, recorded so a
    // replay elsewhere does not substitute the compute_75 rounding)
    int mode = tc2_mode();
    bool loaded80 = g && g->sw && g->sw->mod80;
    bool tc2 = mode == TC2_FORCED || (mode == TC2_AUTO && loaded80);
    bool sm80 = tc2 && gpu_sm80_on() && (!g || loaded80);
    // moe-tc counts only for its families (elsewhere the TC route's own
    // bits record it), where the TC route is on for the family (tc_on);
    // moe-grouped wherever the instance took the grouped prefill (or,
    // without one, where it would): by default only its family, but a pin
    // or RUNNER_MOE_GROUPED=1 admits any MoE model whose expert types it
    // takes, and a record that left it out replayed another path
    bool mtc = moe_tc_family(m) && tc_on(m, T_Q4_K);
    bool mgr = g ? g->moe_grouped_ok : moe_grouped_eligible(m);
    return (tc2 ? GR_TC2 : 0) | (sm80 ? GR_SM80 : 0) |
           (attn_tile_on() ? GR_ATTN_TILE : 0) | (attn_dec_on() ? GR_ATTN_DEC : 0) |
           (attn_tc_on(m) ? GR_ATTN_TC : 0) | (mtc ? GR_MOE_TC : 0) | (mgr ? GR_MOE_GROUPED : 0);
}

// The dynamic shared memory k_attn_dec_gqa needs for a layer's geometry, or
// 0 when the grouped decode attention does not apply there (off, or a shape
// it does not take: k_attn_dec then). One answer for the solo decode and the
// microbatch (fwd_batch), so a layer runs the same kernel arithmetic in both.
static unsigned attn_dec_gqa_smem(const model_t *m, int hd, int n_kv) {
    enum { TK = 32, GMAX = 8 };
    if (!attn_dec_on() || hd % 2 || hd > 256 || n_kv < 1 || m->n_head % n_kv ||
        m->n_head / n_kv > GMAX)
        return 0;
    int kw = (int)kv_bytes_of_kind(model_kv_kind_k(m), (size_t)hd);
    int vw = (int)kv_bytes_of_kind(model_kv_kind_v(m), (size_t)hd);
    kw = ((kw + 3) / 4) | 1;
    vw = ((vw + 3) / 4) | 1;
    unsigned smem = (unsigned)sizeof(float) *
                    (unsigned)(GMAX * (hd + 4) + 2 * TK * (kw > vw ? kw : vw) + GMAX * TK + GMAX);
    // the 48 KB a launch gets without opting in: f16 at head_dim 256 needs
    // 42,400 B; f32 rows (--kv f32) fit up to head_dim 164 and past it take
    // k_attn_dec
    return smem <= 48 * 1024 ? smem : 0;
}

// The grouped decode attention for the single query column, or false
// (nothing launched) when it does not apply. 32 fixed slices, so the grid
// and the graph that captures it do not depend on the position.
static bool tile_attn_dec_gqa(gpu_t *g, model_t *m, int hd, int n_kv,
                              const attn_args *aa, CUdeviceptr sinks, bool *ok) {
    enum { SPLITS = 32 };
    unsigned smem = attn_dec_gqa_smem(m, hd, n_kv);
    if (!smem || !g->sw->f_attn_dec_gqa || !g->sw->f_attn_merge32) return false;
    attn_args a = *aa;
    void *pd[] = { &g->q, &g->kc, &g->vc, &g->attn_part, &a, &g->pos_dev };
    void *pm[] = { &g->xb2, &g->attn_part, &a, &g->pos_dev, &sinks };
    *ok = launch_smem(g, g->sw->f_attn_dec_gqa, (unsigned)n_kv, SPLITS, 1, 256, smem, pd) &&
          launch(g, g->sw->f_attn_merge32, (unsigned)m->n_head, 1, 1, 128, pm);
    if (*ok) g_attn_dec_gqa_dispatches++;
    return true;
}

// The absorbed MLA decode (k_attn_dec_mla, R3.31): its dynamic shared memory
// for a layer's geometry, or 0 when it does not apply (decode attention off,
// the kernel absent from the PTX, not the shared one-table layout, or a shape
// it does not take: k_attn_dec then). One answer for the solo decode and the
// verify rows, so a row runs its solo arithmetic. MD_* (above) mirror
// kernels.cu.
static unsigned attn_dec_mla_smem(const gpu_t *g, const model_t *m, int hd, int n_kv) {
    if (!attn_dec_on() || !g->sw->f_attn_dec_mla || !g->sw->f_attn_merge32 ||
        !model_kv_v_shared(m) || n_kv < 1 || m->n_head % n_kv)
        return 0;
    int G = m->n_head / n_kv, vlo = m->rope_dim;
    if (G > MD_GMAX || hd > 640 || hd % 2 || vlo % 2 || vlo < 0 || vlo >= hd ||
        G * (hd - vlo) / 2 > MD_NJ * MD_THREADS)
        return 0;
    int kb = (int)kv_bytes_of_kind(model_kv_kind_k(m), (size_t)hd);
    int bws = ((kb + 3) / 4) | 1;
    unsigned smem = (unsigned)sizeof(float) *
                    (unsigned)(MD_GMAX * (hd + 4) + MD_TK * bws + MD_GMAX * MD_TK + MD_GMAX);
    // MD_SMEM_MAX is what the kernel opted into, sized for f16 rows up to 640
    // wide; f32 rows (--kv f32) past it take k_attn_dec
    return smem <= MD_SMEM_MAX ? smem : 0;
}
static _Atomic unsigned long g_attn_dec_mla_dispatches;
unsigned long gpu_attn_dec_mla_dispatches(void) { return g_attn_dec_mla_dispatches; }

// The absorbed MLA decode for the single query column, or false (nothing
// launched) when it does not apply; its partials are k_attn_dec_gqa's, merged
// by k_attn_merge32
static bool tile_attn_dec_mla(gpu_t *g, model_t *m, int hd, int n_kv,
                              const attn_args *aa, CUdeviceptr sinks, bool *ok) {
    unsigned smem = attn_dec_mla_smem(g, m, hd, n_kv);
    if (!smem) return false;
    attn_args a = *aa;
    int vlo = m->rope_dim;
    void *pd[] = { &g->q, &g->kc, &g->attn_part, &a, &g->pos_dev, &vlo };
    void *pm[] = { &g->xb2, &g->attn_part, &a, &g->pos_dev, &sinks };
    *ok = launch_smem(g, g->sw->f_attn_dec_mla, (unsigned)n_kv, 32, 1, MD_THREADS, smem, pd) &&
          launch(g, g->sw->f_attn_merge32, (unsigned)m->n_head, 1, 1, 128, pm);
    if (*ok) g_attn_dec_mla_dispatches++;
    return true;
}

// Which decode attention a microbatch column runs on a layer: the twin of
// what tile_attn runs for the same column alone. 1: k_attn_dec_gqa_seq (the
// solo decode takes k_attn_dec_gqa there), 0: k_attn_dec_seq (k_attn_dec),
// -1: the solo kernel has no twin in this module, and the layer cannot batch.
// The partials of a column sit at its own offset in attn_part, which the
// grouped split count fills fastest: see the assertion beside it.
_Static_assert(MODEL_BATCH_MAX * 32 <= MVB * ATTN_SPLITS,
               "attn_part holds a full microbatch of grouped decode partials");
static int batch_attn_gqa(const gpu_t *g, const model_t *m, int hd, int n_kv) {
    if (!attn_dec_gqa_smem(m, hd, n_kv) || !g->sw->f_attn_dec_gqa || !g->sw->f_attn_merge32)
        return 0;
    return g->sw->f_attn_dec_gqa_seq && g->sw->f_attn_merge32_seq ? 1 : -1;
}

static _Atomic unsigned long g_exact_verify_tiles;
unsigned long gpu_exact_verify_tiles(void) { return g_exact_verify_tiles; }

// A verify round's attention (exact_rows): every row through the decode
// attention a solo step at its position runs (the grouped kernel where
// tile_attn_dec_gqa takes the layer, else k_attn_dec), as that kernel's
// multi-column twin over the rows' own positions and the one cache
// (vcol_*). Each layer stored all the tile's rows before any row attends,
// and a row reads only up to its own position, so it sees exactly the rows
// its solo step would. Sixteen rows a launch: the grouped partials of
// sixteen fill attn_part.
static bool tile_attn_rows(gpu_t *g, model_t *m, int tn, int hd, int n_kv,
                           const attn_args *aa, CUdeviceptr sinks) {
    int gq = batch_attn_gqa(g, m, hd, n_kv);
    if (gq < 0 || !g->sw->f_attn_dec_seq) {
        gpu_first_error("the decode attention has no multi-row twin in this "
                        "module; refusing the exact verify tile");
        return false;
    }
    attn_args a = *aa;
    bool ok = true;
    unsigned mla = attn_dec_mla_smem(g, m, hd, n_kv);
    if (mla && !g->sw->f_attn_dec_mla_seq) {
        gpu_first_error("the absorbed MLA decode has no multi-row twin in this "
                        "module; refusing the exact verify tile");
        return false;
    }
    int vlo = m->rope_dim;
    for (int c0 = 0; c0 < tn && ok; c0 += 16) {
        unsigned cn = (unsigned)(tn - c0 < 16 ? tn - c0 : 16);
        CUdeviceptr q   = g->q + (size_t)c0 * a.qs * sizeof(float);
        CUdeviceptr out = g->xb2 + (size_t)c0 * a.os * sizeof(float);
        CUdeviceptr pos = g->vcol_pos + (size_t)c0 * sizeof(int);
        CUdeviceptr kcp = g->vcol_kcp + (size_t)c0 * sizeof(uint64_t);
        CUdeviceptr vcp = g->vcol_vcp + (size_t)c0 * sizeof(uint64_t);
        void *pm[] = { &out, &g->attn_part, &a, &pos, &sinks };
        if (mla) {
            void *pg[] = { &q, &kcp, &g->attn_part, &a, &pos, &vlo };
            ok = launch_smem(g, g->sw->f_attn_dec_mla_seq, (unsigned)n_kv, 32, cn, MD_THREADS, mla, pg) &&
                 launch(g, g->sw->f_attn_merge32_seq, (unsigned)m->n_head, cn, 1, 128, pm);
        } else if (gq) {
            void *pg[] = { &q, &kcp, &vcp, &g->attn_part, &a, &pos };
            ok = launch_smem(g, g->sw->f_attn_dec_gqa_seq, (unsigned)n_kv, 32, cn, 256,
                             attn_dec_gqa_smem(m, hd, n_kv), pg) &&
                 launch(g, g->sw->f_attn_merge32_seq, (unsigned)m->n_head, cn, 1, 128, pm);
        } else {
            void *pd[] = { &q, &kcp, &vcp, &g->att, &g->attn_part, &a, &pos };
            ok = launch(g, g->sw->f_attn_dec_seq, (unsigned)m->n_head, ATTN_SPLITS, cn, 128, pd) &&
                 launch(g, g->sw->f_attn_merge, (unsigned)m->n_head, cn, 1, 128, pm);
        }
    }
    return ok;
}

// RUNNER_CUDA_ATTN_DUMP=<dir> (a development switch) writes, for every prompt
// tile's attention, the kernel's inputs and output as the device holds them:
// Q, this layer's K and V rows up to the tile's last position, the output,
// and a JSON line of geometry and which kernel ran.
// scripts/attn-dump-compare.py reads it. Unset, a tile pays one flag test.
static bool attn_dump(gpu_t *g, model_t *m, int l, int tn, int hd, int n_kv,
                      int q_dim, int xdim, const attn_args *aa, unsigned long tcw0) {
    static const char *_Atomic dir = NULL;
    static _Atomic int read = 0, seq = 0;
    if (!read) { dir = getenv("RUNNER_CUDA_ATTN_DUMP"); read = 1; }
    if (!dir || !*dir || tn < 2) return true;
    if (cu.StreamSynchronize(g->stream) != 0) return false;
    int pos = 0;
    if (cu.MemcpyDtoH(&pos, g->pos_dev, sizeof pos) != 0) return false;
    int kv_dim = n_kv * hd, rows = pos + tn;
    bool fp16 = !aa->q8 && !aa->vq8 && !aa->ring;
    size_t qb = sizeof(float) * (size_t)tn * q_dim, kb = (size_t)rows * kv_dim * 2;
    float *qh = malloc(qb), *oh = malloc(qb), *row = malloc(sizeof(float) * (size_t)xdim);
    unsigned char *kh = fp16 ? malloc(kb) : NULL, *vh = fp16 ? malloc(kb) : NULL;
    bool ok = qh && oh && row && (!fp16 || (kh && vh));
    ok = ok && cu.MemcpyDtoH(qh, g->q, qb) == 0;
    for (int t = 0; ok && t < tn; t++) {
        ok = cu.MemcpyDtoH(row, g->xb2 + (size_t)t * xdim * sizeof(float),
                           sizeof(float) * (size_t)xdim) == 0;
        if (ok) memcpy(oh + (size_t)t * q_dim, row, sizeof(float) * (size_t)q_dim);
    }
    if (ok && fp16)
        ok = cu.MemcpyDtoH(kh, g->kc + aa->l_off, kb) == 0 &&
             cu.MemcpyDtoH(vh, g->vc + aa->v_off, kb) == 0;
    char path[1024];
    int n = seq++;
    const char *ext[] = { "q", "o", "k", "v" };
    const void *buf[] = { qh, oh, kh, vh };
    size_t len[] = { qb, qb, kb, kb };
    for (int i = 0; ok && i < (fp16 ? 4 : 2); i++) {
        snprintf(path, sizeof path, "%s/c%04d-L%02d.%s", dir, n, l, ext[i]);
        FILE *f = plat_fopen(path, "wb");
        ok = f && fwrite(buf[i], 1, len[i], f) == len[i];
        if (f) fclose(f);
    }
    snprintf(path, sizeof path, "%s/meta.jsonl", dir);
    FILE *f = ok ? plat_fopen(path, "a") : NULL;
    if (f) {
        fprintf(f, "{\"call\":%d,\"layer\":%d,\"pos\":%d,\"tn\":%d,\"head_dim\":%d,"
                "\"n_head\":%d,\"n_kv\":%d,\"q_dim\":%d,\"scale\":%.9g,\"window\":%d,"
                "\"ring\":%d,\"kv_fp16\":%s,\"tcw\":%lu}\n", n, l, pos, tn, hd, m->n_head,
                n_kv, q_dim, (double)aa->scale, aa->window, aa->ring, fp16 ? "true" : "false",
                g_attn_tcw_dispatches - tcw0);
        fclose(f);
    }
    free(qh); free(oh); free(row); free(kh); free(vh);
    if (!ok) fprintf(stderr, "attn-dump: failed at call %d layer %d\n", n, l);
    return ok;
}

// Attention over the cache: flash-decoding at a single token (split, then
// merge), the tiled prefill kernel or k_attn for a prompt tile.
static bool tile_attn(gpu_t *g, model_t *m, int l, int tn, int hd, int n_kv,
                      int q_dim, int xdim, bool local) {
    bool ok = true;
    {
        attn_args aa = { hd, m->n_head, n_kv, m->n_ctx,
                         (uint64_t)model_k_byte_off(m, l),
                         model_attn_scale(m, l), q_dim, xdim,
                         local ? m->swa_window : 0, model_kv_kind_k(m),
                         model_kv_is_ring(m, l) ? m->kv_ring : 0,
                         (uint64_t)model_v_byte_off(m, l), model_kv_kind_v(m) };
        // Decode (tn==1, one query, long KV): flash-decoding — split the KV
        // range across ATTN_SPLITS blocks/head (higher occupancy, coalesced)
        // then merge partials. Prefill (tn>1) already runs n_head*tn blocks,
        // so it keeps the single-pass k_attn. tn is the tile size (constant
        // at graph-capture time), not per-token state, so this branch is
        // capture-safe.
        // gpt-oss attention sinks for this layer (0 elsewhere): the sink
        // joins the softmax at the normalization point — inside k_attn's
        // single reduction, or at k_attn_merge's global merge (the split
        // partials carry no normalization, so k_attn_dec is untouched)
        CUdeviceptr sinks = g->sw->sinks[l];
        unsigned long tcw0 = g_attn_tcw_dispatches;
        if (tn > 1 && g->exact_rows) {
            ok = tile_attn_rows(g, m, tn, hd, n_kv, &aa, sinks);
        } else if (tn == 1 && tile_attn_dec_mla(g, m, hd, n_kv, &aa, sinks, &ok)) {
            // ran (or failed): the absorbed MLA slices, then their merge
        } else if (tn == 1 && tile_attn_dec_gqa(g, m, hd, n_kv, &aa, sinks, &ok)) {
            // ran (or failed): grouped slices, then their merge
        } else if (tn == 1) {
            void *pd[] = { &g->q, &g->kc, &g->vc, &g->att, &g->attn_part,
                           &aa, &g->pos_dev };
            ok = ok && launch(g, g->sw->f_attn_dec, m->n_head, ATTN_SPLITS, 1, 128, pd);
            void *pm[] = { &g->xb2, &g->attn_part, &aa, &g->pos_dev, &sinks };
            ok = ok && launch(g, g->sw->f_attn_merge, m->n_head, 1, 1, 128, pm);
        } else if (tile_attn_tcw(g, m, tn, hd, n_kv, &aa, sinks, &ok)) {
            // ran (or failed) on the tensor cores, a wide head
        } else if (tile_attn_tc(g, m, tn, hd, n_kv, &aa, sinks, &ok)) {
            // ran (or failed) on the tensor cores over the whole tile
        } else if (tile_attn_tiled(g, m, tn, hd, n_kv, &aa, sinks, &ok)) {
            // ran (or failed) as one launch over the whole tile
        } else {
            // one launch per ATT_COLS query columns over the shared score
            // scratch; set_pos left each sub-tile's start at pos_dev[s/ATT_COLS]
            for (int s = 0; s < tn && ok; s += ATT_COLS) {
                int sn = tn - s < ATT_COLS ? tn - s : ATT_COLS;
                CUdeviceptr qs = g->q + (size_t)s * q_dim * sizeof(float);
                CUdeviceptr os = g->xb2 + (size_t)s * xdim * sizeof(float);
                CUdeviceptr ps = g->pos_dev + (size_t)(s / ATT_COLS) * sizeof(int);
                void *pa[] = { &qs, &g->kc, &g->vc, &g->att, &os, &aa, &ps, &sinks };
                ok = launch(g, g->sw->f_attn, m->n_head, sn, 1, 128, pa);
            }
        }
        if (ok && tn > 1) ok = attn_dump(g, m, l, tn, hd, n_kv, q_dim, xdim, &aa, tcw0);
    }
    return ok;
}

// The output gate and the output projection with its adapter delta.
static bool tile_attn_out(gpu_t *g, model_t *m, layer_t *ly, int l, int tn,
                          int n_embd, int q_dim, int xdim) {
    bool ok = true;
    if (m->attn_gate_heads && ly->wq_gate) {
        // one sigmoid per head over its head_dim outputs (spark2_5)
        int gate_stride = q_dim, hd = model_head_dim(m, l);
        void *pg[] = { &g->xb2, &g->q35_gate, &q_dim, &hd, &xdim, &gate_stride };
        ok = ok && launch(g, g->sw->f_head_gate, (q_dim + 255) / 256,
                          tn, 1, 256, pg);
    } else if (m->qwen35 || (m->attn_out_gate && ly->wq_gate)) {
        int gate_stride = q_dim;
        void *pg[] = { &g->xb2, &g->q35_gate, &q_dim, &xdim, &gate_stride };
        ok = ok && launch(g, g->sw->f_q35_gate, (q_dim + 255) / 256,
                          tn, 1, 256, pg);
    }

    if (m->deepseek2) {
        // each head's first mla_v_dim outputs (the rest is V's zero pad),
        // packed for attn_output as model.c's attn_output packs them
        int H = m->n_head, hd = model_head_dim(m, l), vd = m->mla_v_dim, pw = H * vd;
        int r = m->mla_kv_lora, rd = m->rope_dim, kvb = H * (hd - rd + vd);
        int mid = m->mla_absorb ? H * m->mla_qk : kvb;
        CUdeviceptr pv = g->mla_buf + sizeof(float) * (size_t)g->win *
                         (size_t)(r + rd + r + mid);
        if (m->mla_absorb) {
            // each head's output latent through its v-rows (R4.4.5 c2)
            int qn = m->mla_qk - rd;
            void *pa[] = { &g->xb2, &g->sw->mla_kvb[l], &pv, &H, &rd, &r, &qn, &vd, &xdim };
            ok = launch(g, g->sw->f_mla_v_absorb, (unsigned)((vd + 7) / 8), H, tn, 256, pa);
        } else {
            void *pp[] = { &g->xb2, &pv, &H, &hd, &vd, &xdim };
            ok = launch(g, g->sw->f_mla_pack_v, (unsigned)((pw + 255) / 256), tn, 1, 256, pp);
        }
        return ok && enc_mv(g, m, ly->wo, pv, g->xb, pw, n_embd, g->sw->bo[l], tn, pw, xdim);
    }
    ok = ok && enc_mv(g, m, ly->wo, g->xb2, g->xb, q_dim, n_embd, g->sw->bo[l], tn, xdim, xdim)
            && enc_lora(g, l, LW_O, g->xb2, g->xb, q_dim, n_embd, tn, xdim, xdim);
    return ok;
}

// A branch's tail, the same after attention and after the FFN: its post-norm
// (when the layer has one), the granite muP scale, and the residual add, or
// the hyper-connection streams' combine.
static bool tile_branch_out(gpu_t *g, model_t *m, CUdeviceptr post_norm, int tn,
                            int xdim) {
    int n_embd = m->n_embd;
    bool ok = true;
    if (post_norm)
        ok = ok && enc_rmsnorm(g, g->xb, g->xb, post_norm, n_embd, m->post_norm_eps,
                               tn, xdim, xdim);
    if (m->resid_scale != 1.0f)  // granite muP branch scale
        ok = ok && enc_scale(g, g->xb, m->resid_scale, n_embd, tn, xdim);
    if (m->hyper_conn) ok = ok && gpu_hc_combine(g, m, g->xb, xdim, tn);
    else ok = ok && enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);
    return ok;
}

// The shared expert, the device twin of shexp_add, added into the routed
// experts' output.
static bool tile_shexp(gpu_t *g, model_t *m, layer_t *ly, int tn, int xdim) {
    int n_embd = m->n_embd;
    bool ok = true;
    // the device twin of shexp_add:
    // out += sigmoid(gate_inp_shexp . in) * down(act(gate(in)) * up(in)),
    // or without the sigmoid where the file has no gate_inp_shexp
    int nf = m->n_ff_shexp;
    ok = ok && enc_mv(g, m, ly->w_up_shexp, g->shexp_in, g->shexp_u, n_embd, nf, 0, tn, xdim, nf)
            && enc_mv(g, m, ly->w_gate_shexp, g->shexp_in, g->shexp_g, n_embd, nf, 0, tn, xdim, nf)
            && enc_actmul(g, m, g->shexp_g, g->shexp_u, tn * nf)
            && enc_mv(g, m, ly->w_down_shexp, g->shexp_g, g->shexp_o, nf, n_embd, 0, tn, nf, n_embd);
    if (!ly->ffn_gate_inp_shexp)    // DeepSeek-style, ungated (R4.4.5 c0b)
        return ok && enc_add(g, g->xb, g->shexp_o, n_embd, tn, xdim, n_embd);
    ok = ok && enc_mv(g, m, ly->ffn_gate_inp_shexp, g->shexp_in, g->shexp_gate, n_embd, 1, 0, tn, xdim, 1);
    if (ok) {
        int vs = n_embd;
        void *pa[] = { &g->xb, &g->shexp_o, &g->shexp_gate, &n_embd, &xdim, &vs };
        ok = launch(g, g->sw->f_axpy_sigmoid, (n_embd + 255) / 256, tn, 1, 256, pa);
    }
    return ok;
}

// The dense FFN's input projections: up alone for a gate-less FFN (relu2,
// xIELU), else gate and up, each with its adapter delta. The layer's own
// width (gemma-4 E2B varies it per layer); rows keep the model-wide stride
// m->n_ff, the widest layer, which the buffers are sized by.
static bool tile_ffn_in(gpu_t *g, model_t *m, layer_t *ly, int l, int tn, int xdim) {
    int n_embd = m->n_embd, nff = ly->n_ff;
    bool ok = true;
    if (!ly->w_gate) {
        ok = ok && enc_mv(g, m, ly->w_up, g->xb, g->hb,
                          n_embd, nff, 0, tn, xdim, m->n_ff)
                && enc_lora(g, l, LW_UP, g->xb, g->hb, n_embd, nff,
                            tn, xdim, m->n_ff);
    } else {
        ok = ok && enc_mv(g, m, ly->w_gate, g->xb, g->hb,
                          n_embd, nff, 0, tn, xdim, m->n_ff)
                && enc_lora(g, l, LW_GATE, g->xb, g->hb, n_embd, nff,
                            tn, xdim, m->n_ff);
        ok = ok && enc_mv_same_x(g, m, ly->w_up, g->xb, g->hb2,
                                 n_embd, nff, 0, tn, xdim, m->n_ff)
                && enc_lora(g, l, LW_UP, g->xb, g->hb2, n_embd, nff,
                            tn, xdim, m->n_ff);
    }
    return ok;
}

// The dense FFN's activation: relu2 or xIELU on a gate-less FFN, else the
// gated product. Over whole m->n_ff rows: on a narrower layer the columns
// past its width hold stale values nothing reads.
static bool tile_ffn_act(gpu_t *g, model_t *m, layer_t *ly, int l, int tn) {
    if (!ly->w_gate) {
        if (m->ffn_relu2)
            return enc_relu2(g, g->hb, tn * m->n_ff);
        return enc_xielu(g, g->hb, tn * m->n_ff,
                         m->xielu_an[l], m->xielu_ap[l],
                         m->xielu_b[l], m->xielu_eps[l]);
    }
    return enc_actmul(g, m, g->hb, g->hb2, tn * m->n_ff);
}

static bool fwd_tile(gpu_t *g, model_t *m, const int32_t *tokens, int tn,
                     int pos, bool want_logits, int l0, int l1) {
    (void)tokens; (void)pos;
    int n_embd = m->n_embd;
    int xdim = n_embd;
    for (int l = 0; l < m->n_layer; l++)
        if (model_q_dim(m, l) > xdim) xdim = model_q_dim(m, l);

    bool ok = true;
    prof_mark(g, -1);   // tile anchor: delta to the first group is charged below
    for (int l = l0; l < l1 && ok; l++) {
        layer_t *ly = &m->layers[l];
        bool local  = model_is_swa(m, l);
        int hd      = model_head_dim(m, l);
        int n_kv    = model_n_head_kv(m, l);
        int q_dim   = model_q_dim(m, l);
        int kv_dim  = model_kv_dim(m, l);

        // --remove-sublayer attn:N: the block has no attention, so its whole
        // branch (pre-norm, attention, post-norm, residual add) is skipped and
        // the residual passes through. It jumps past the branch only, unlike
        // the nemotron_h jump below, so the FFN dispatch that follows (host or
        // device experts, gemma's dual-branch FFN) still runs as it would.
        // The block's KV is not allocated: its per-layer KV width is 0.
        if (m->l_no_attn && m->l_no_attn[l]) goto attention_removed;
        // nemotron_h MLP-only block: its device FFN, or, for a nemotron_h_moe
        // block whose experts are on the host (--cpu-moe, R3.43a), the host
        // branch below that routes and runs them (model_moe_ffn_cpu)
        if (ly->skip_mixer) {
            if (ly->is_moe && moe_on_host(m, l) && !moe_hot_on(m, l)) goto attention_removed;
            goto mamba_mlp;
        }
        if (m->hyper_conn) {
            size_t hcd = (size_t)m->hc_count * n_embd;
            if (ly->ple4) {
                // the n-gram PLE block runs on the host over the wide
                // residual rows (one layer; its table is host-resident)
                if (cu.StreamSynchronize(g->stream) != 0 ||
                    cu.MemcpyDtoH(m->x_hc, g->x_hc, sizeof(float) * tn * hcd) != 0 ||
                    !model_ple4_block_host(m, l, tokens, tn, pos) ||
                    cu.MemcpyHtoD(g->x_hc, m->x_hc, sizeof(float) * tn * hcd) != 0)
                    return false;
            }
            ok = ok && gpu_hc_mix(g, m, g->sw->hc_an[l], ly->hc_attn_down, ly->hc_attn_up,
                                  ly->hc_attn_inject, g->x_hc, g->xb, xdim, tn);
        } else
        ok = ok && enc_rmsnorm(g, g->x, g->xb, g->sw->attn_norm[l], n_embd, m->rms_eps,
                               tn, n_embd, xdim);
        prof_mark(g, PH_NORM);
        if (ly->recurrent) {
            ok = ok && (m->qwen35
                        ? gpu_q35_recurrent(g, m, ly, l, tn, xdim)
                        : gpu_mamba2_recurrent(g, m, ly, l, tn, xdim));
            prof_mark(g, PH_MATVEC);
            goto attention_done;
        }
        // deepseek2 builds Q, K and V together from the latent pair
        if (m->deepseek2) ok = ok && tile_mla_project(g, m, ly, l, tn, n_embd, xdim);
        else ok = ok && tile_q(g, m, ly, l, tn, n_embd, xdim, hd, q_dim);
        // gemma4 E-series shared-KV layers project Q as usual but compute no
        // K/V: they attend over the cache an earlier layer filled, which the
        // aliased model_kv_byte_off() below already resolves to.
        bool owns_kv = model_kv_owner(m, l) == l;
        if (!owns_kv) goto kv_done;
        if (!m->deepseek2) ok = ok && tile_kv_proj(g, m, ly, l, tn, n_embd, xdim, kv_dim);
        prof_mark(g, PH_MATVEC);
        ok = ok && tile_kv_store(g, m, l, tn, hd, n_kv, kv_dim);
    kv_done:
        if (g->sw->qn[l]) ok = ok && enc_qknorm(g, m, g->q,  g->sw->qn[l], m->n_head, hd, tn, q_dim);
        prof_mark(g, PH_NORM);
        ok = ok && enc_rope_or_temp(g, m, g->q, m->n_head, tn, q_dim, l,
                                    pos, enc_rope);
        prof_mark(g, PH_ROPE);

        ok = ok && tile_attn(g, m, l, tn, hd, n_kv, q_dim, xdim, local);
        prof_mark(g, PH_ATTN);

        ok = ok && tile_attn_out(g, m, ly, l, tn, n_embd, q_dim, xdim);
        prof_mark(g, PH_MATVEC);
    attention_done:
        ok = ok && tile_branch_out(g, m, g->sw->pan[l], tn, xdim);
        prof_mark(g, PH_ELEM);
    attention_removed:
        // nemotron_h SSM/attention blocks, and --remove-sublayer mlp:N
        if (ly->skip_ffn) goto mamba_tail;

        if (moe_hot_on(m, l) && g->sw->hot_bank[l]) {
            // RUNNER_MOE_HOT: the device norms and routes; the hot bank and
            // the host share the experts (gpu_moe_ffn_hot)
            ok = ok && enc_rmsnorm(g, g->x, g->xb, g->sw->ffn_norm[l], n_embd, m->rms_eps,
                                   tn, n_embd, xdim);
            prof_mark(g, PH_NORM);
            if (ok && g->route_trace) moe_trace_flush(g);
            ok = ok && gpu_moe_ffn_hot(g, m, ly, tn, xdim);
            if (ok && g->route_trace) moe_trace_flush(g);
            prof_mark(g, PH_HOSTMOE);
            goto ffn_done;
        }
        if (ly->is_moe && moe_on_host(m, l) && m->hyper_conn) {
            // qwen4exp: the device mixes the FFN input, the host runs the
            // routed + shared experts on that tile, the device scatters the
            // FFN output into the four streams
            ok = ok && gpu_hc_mix(g, m, g->sw->hc_fn[l], ly->hc_ffn_down, ly->hc_ffn_up,
                                  ly->hc_ffn_inject, g->x_hc, g->xb, xdim, tn);
            if (!ok) return false;
            prof_mark(g, PH_MATVEC);
            if (cu.StreamSynchronize(g->stream) != 0 ||
                cu.MemcpyDtoH(m->xb, g->xb, sizeof(float) * (size_t)tn * xdim) != 0 ||
                !model_moe_ffn_cpu_premixed(m, l, tn) ||
                cu.MemcpyHtoD(g->xb, m->xb, sizeof(float) * (size_t)tn * xdim) != 0)
                return false;
            prof_mark(g, PH_HOSTMOE);
            ok = ok && gpu_hc_combine(g, m, g->xb, xdim, tn);
            prof_mark(g, PH_MATVEC);
            continue;
        }
        if (ly->is_moe && moe_on_host(m, l)) {
            // Tensor-role boundary: attention has updated the residual and KV
            // on-device. Move only the small activation tile to the host, run
            // the sparse expert FFN against mmap-resident weights, and resume
            // the next layer on CUDA. This is the 30B-on-small-VRAM path.
            prof_mark(g, PH_MATVEC);
            if (cu.StreamSynchronize(g->stream) != 0 ||
                cu.MemcpyDtoH(m->x, g->x,
                              sizeof(float) * (size_t)tn * n_embd) != 0 ||
                !model_moe_ffn_cpu(m, l, tn) ||
                cu.MemcpyHtoD(g->x, m->x,
                              sizeof(float) * (size_t)tn * n_embd) != 0)
                return false;
            prof_mark(g, PH_HOSTMOE);
            continue;
        }

        if (ly->moe_gemma) {
            // dual-branch FFN norms itself off g->x (the un-normed residual)
            ok = ok && gpu_gemma_moe_ffn(g, m, ly, l, tn, xdim);
            prof_mark(g, PH_MATVEC);
            goto ffn_done;
        }
    mamba_mlp:
        if (m->hyper_conn)
            ok = ok && gpu_hc_mix(g, m, g->sw->hc_fn[l], ly->hc_ffn_down, ly->hc_ffn_up,
                                  ly->hc_ffn_inject, g->x_hc, g->xb, xdim, tn);
        else
        ok = ok && enc_rmsnorm(g, g->x, g->xb, g->sw->ffn_norm[l], n_embd, m->rms_eps,
                               tn, n_embd, xdim);
        prof_mark(g, PH_NORM);
        if (ly->is_moe) {
            // The shared expert reads the normed input the routed experts
            // then overwrite, so it is copied first, in stream order on the
            // instance's stream: a decode graph records the copy, which the
            // synchronous cuMemcpyDtoD could not be. A driver without the
            // async call keeps the synchronous one (and the eager decode).
            bool shexp = ly->w_up_shexp && g->shexp_in;
            size_t shexp_bytes = sizeof(float) * (size_t)tn * xdim;
            if (shexp && (cu.MemcpyDtoDAsync
                              ? cu.MemcpyDtoDAsync(g->shexp_in, g->xb, shexp_bytes, g->stream)
                              : cu.MemcpyDtoD(g->shexp_in, g->xb, shexp_bytes)) != 0)
                return false;
            ok = ok && gpu_moe_ffn(g, m, ly, tn, xdim);  // writes FFN output into g->xb
            if (ok && shexp) ok = tile_shexp(g, m, ly, tn, xdim);
            prof_mark(g, PH_MATVEC);
        } else {
            ok = ok && tile_ffn_in(g, m, ly, l, tn, xdim);
            prof_mark(g, PH_MATVEC);
            ok = ok && tile_ffn_act(g, m, ly, l, tn);
            prof_mark(g, PH_ELEM);
            ok = ok && enc_mv(g, m, ly->w_down, g->hb, g->xb, ly->n_ff, n_embd, 0, tn, m->n_ff, xdim)
                    && enc_lora(g, l, LW_DOWN, g->hb, g->xb, ly->n_ff, n_embd, tn, m->n_ff, xdim);
            prof_mark(g, PH_MATVEC);
        }
    ffn_done:
        ok = ok && tile_branch_out(g, m, g->sw->pfn[l], tn, xdim);
    mamba_tail:
        ok = ok && enc_ple(g, m, ly, l, tn, xdim);
        if (ly->out_scale != 1.0f && ly->out_scale != 0.0f)
            // gemma4: whole-layer output scalar, applied after both residuals
            ok = ok && enc_scale(g, g->x, ly->out_scale, n_embd, tn, n_embd);
        prof_mark(g, PH_ELEM);
    }

    if (m->spec_want_all >= tn && tn <= m->spec_batch && l1 >= m->n_layer) {
        // speculative verify (model_forward_batch_keep): every row's head,
        // row r's logits at g->logits + r * n_vocab (gpu_spec_logits)
        if (m->hyper_conn)
            ok = ok && gpu_hc_mix(g, m, g->sw->hc_head_norm, m->hc_head_down, m->hc_head_up,
                                  NULL, g->x_hc, g->xb, xdim, tn);
        else
            ok = ok && enc_rmsnorm(g, g->x, g->xb, g->sw->out_norm, n_embd, m->rms_eps,
                                   tn, n_embd, xdim);
        prof_mark(g, PH_NORM);
        ok = ok && enc_mv(g, m, m->output, g->xb, g->logits, n_embd, m->n_vocab, 0, tn, xdim, m->n_vocab);
        prof_mark(g, PH_LOGITS);
    } else if (want_logits) {
        CUdeviceptr xlast = g->x + (size_t)(tn - 1) * n_embd * sizeof(float);
        if (m->hyper_conn) {
            // the final mixer is the output norm: last row of the wide residual
            CUdeviceptr xhl = g->x_hc + (size_t)(tn - 1) * m->hc_count * n_embd * sizeof(float);
            ok = ok && gpu_hc_mix(g, m, g->sw->hc_head_norm, m->hc_head_down, m->hc_head_up,
                                  NULL, xhl, g->xb, 0, 1);
        } else
        ok = ok && enc_rmsnorm(g, xlast, g->xb, g->sw->out_norm, n_embd, m->rms_eps,
                               1, 0, 0);
        prof_mark(g, PH_NORM);
        ok = ok && enc_mv(g, m, m->output, g->xb, g->logits, n_embd, m->n_vocab, 0, 1, 0, 0);
        prof_mark(g, PH_LOGITS);
    }
    return ok;
}

// ==========================================================================
// Cross-sequence decode microbatch
// ==========================================================================
// One decode step for N independent sequences in one pass over the weights.
// Almost nothing new is allocated: the batch adds the per-column positions,
// the two per-column KV base pointers and N rows of logits, and borrows the
// lead sequence's scratch, stream and graph slot. So a sequence must not be
// decoded both ways at once (the public API says so).
// Sequences are named per call, so membership can change every step
// (continuous batching): positions and KV pointers live in device memory, so
// changing them does not invalidate a graph, captured per microbatch width.
struct gpu_batch {
    model_t   **seqs;              // borrowed: the caller owns these
    int         n;                 // sequences enrolled at create time
    gpu_t      *lead;              // whose scratch/stream/KV machinery is used
    gpu_weights *sw;
    int         n_vocab, n_embd, xdim;
    CUdeviceptr logits_n;          // [MVB][n_vocab]
    CUdeviceptr pos_d;             // [MVB] per-column position
    CUdeviceptr kcp_d, vcp_d;      // [MVB] per-column KV base pointers
    float      *h_logits;          // [MVB][n_vocab] host staging
    int         h_pos[MVB];
    uint64_t    h_kcp[MVB], h_vcp[MVB];
    // one captured graph per microbatch width: grid shapes and mv_args.batch
    // are baked in at capture, but positions and KV pointers are not, so a
    // graph stays valid across steps and across a change of membership
    CUgraph     graph[MVB + 1];
    CUgraphExec gexec[MVB + 1];
    bool        graph_bad;
    const char *graph_why;      // why graph_bad was set by design (none: a failure set it)
    bool        graph_why_said;
};

// enc_mv_batch reproduces the batch-1 result by launching the multi-column
// twin of the kernel the batch-1 path uses. A type with a decode GEMV but no
// width-classed twin has no such kernel: refuse the microbatch at create time
// and let the caller decode those sequences one at a time. The set is empty
// today; the check stays for the next type that grows a decode GEMV.
// Where there is no decode GEMV the batch-1 path runs f_mv, and f_mvb is its
// twin only where the _b kernel applies the same per-element weight in the
// same order: F32, F16 and BF16. Every quantized _b kernel folds the block
// scale into each term while its batch-1 partner keeps it outside the block
// sum: the same values in another association, so different bits. Those
// types batch via f_gemvb or decode one sequence at a time; it is not fixed
// in the kernel, because k_mv_*_b is also the prefill tile kernel and
// changing its arithmetic changes certified prefill.
// Q2_K's and Q3_K's _b tiles are term-exact too but are not admitted: they
// measured 0.17x to 0.63x of sequential decode. Q3_K batches through its
// k_gemv twin; Q2_K decodes its sequences one at a time.
static inline bool mvb_is_mv_twin(int type) {
    return type == T_F32 || type == T_F16 || type == T_BF16;
}

// Every f_gemvb is the twin of the batch-1 kernel enc_mv picks: f_gemv where a
// type has one (the K-quants, Q8_0, Q4_0, IQ4_XS), f_mv otherwise (BF16, MXFP4
// and the codebook i-quants' lane kernels).
static bool batch_mv_twin_ok(const gpu_t *g, const gguf_tensor *t) {
    if (!t) return true;
    int type = t->type;
    if (type < 0 || type >= KT_N) return false;
    if (!mv_twin_route(g, t, (int)t->ne[0], (int)t->ne[1])) return false;
    const gpu_weights *sw = g->sw;
    // the microbatch is at most MODEL_BATCH_MAX (8) columns: BW_4 and BW_8
    if (sw->f_gemvb[BW_4][type] && sw->f_gemvb[BW_8][type]) return true;
    return !sw->f_gemv[type] && mvb_is_mv_twin(type) && sw->f_mvb[type];
}

// Every member must be decodable in one launch against one weight upload:
// same shared weights, full offload (a partial split has to hand the boundary
// activation back to the CPU per sequence, which is a different kernel shape),
// and the batched kernels present. A no here is not a failure — the caller
// decodes these sequences one at a time, exactly as before.
static bool batch_eligible(model_t **seqs, int n, gpu_t **lead_out) {
    if (n < 1) return false;
    if (seqs[0] && seqs[0]->hyper_conn) return false;   // qwen4exp: no microbatch path yet
    gpu_t *lead = NULL;
    for (int i = 0; i < n; i++) {
        model_t *m = seqs[i];
        if (!m || !m->gpu) return false;
        gpu_t *g = m->gpu;
        if (m->gpu_layers < m->n_layer) return false;
        // fwd_batch runs a dense FFN unconditionally; MoE (sparse experts or
        // the gemma-4 dual-branch) has no path there, and it neither projects
        // the attention output gate nor skips rope on nope_on_full layers —
        // keep all of those on fwd_tile. The recurrent families (qwen35 and
        // the Mamba-2 hybrids) are declined as a family: fwd_batch has no
        // mixer path and would hand a recurrent layer's NULL attention
        // weights to enc_mv_batch as dense tensors. NoPE-stepped models are
        // declined here too (enc_rope_batch ropes every layer), and so is any
        // attention temperature (mistral3's scales every layer's Q by a
        // per-position factor fwd_batch does not apply); the same check in
        // gpu_batch_decode stays as the runtime backstop.
        if (m->n_expert > 0 || m->moe_gemma || m->qwen35 || m->deepseek2 ||
            m->granite_hybrid || m->nemotron_h ||
            m->attn_out_gate || m->nope_on_full ||
            m->no_rope_layer_step > 0 || m->attn_temp_scale != 0.0f) return false;
        // fwd_batch has its own projection sites and does not carry the
        // adapter hooks (R8.7.2 wired fwd_tile only), so an adapted model
        // declines the multi-sequence batch and decodes sequentially rather
        // than losing its adapter on the batched path.
        if (m->lora) return false;
        if (!g->sw->f_rope_seq || !g->sw->f_store_seq || !g->sw->f_attn_dec_seq)
            return false;
        // every tensor fwd_batch hands to enc_mv_batch must have a real twin
        if (!batch_mv_twin_ok(g, m->output)) return false;
        for (int l = 0; l < m->n_layer; l++) {
            const layer_t *ly = &m->layers[l];
            // the decode attention the layer runs alone needs its twin here
            if (batch_attn_gqa(g, m, model_head_dim(m, l), model_n_head_kv(m, l)) < 0)
                return false;
            // fwd_batch implements the gated dense transformer loop only.
            // Apertus is dense but deliberately has no w_gate (its FFN is
            // up -> xIELU -> down); admitting it would hand NULL to
            // enc_mv_batch below. Keep it on the bit-identical sequential GPU
            // path until the batched loop has an explicit ungated branch.
            if (!ly->wq || !ly->wo || !ly->w_gate ||
                !ly->w_up || !ly->w_down ||
                (model_kv_owner(m, l) == l && !ly->wk))
                return false;
            const gguf_tensor *ws[] = { ly->wq, ly->wk, ly->wv, ly->wo,
                                        ly->w_gate, ly->w_up, ly->w_down };
            for (size_t k = 0; k < sizeof(ws) / sizeof(*ws); k++)
                if (!batch_mv_twin_ok(g, ws[k])) return false;
        }
        if (!lead) lead = g;
        // one upload, one geometry: a member from a different registry entry
        // would index a different weight blob with this one's offsets
        else if (g->sw != lead->sw) return false;
    }
    *lead_out = lead;
    return true;
}

// D8 slice 4: the position grid. The mvt kernels parallelize over n_in
// alone, so a 1536-wide projection is six 256-thread blocks and most of the
// device idles; positions are the free second dimension, and the kernel's
// strided t loop makes ANY gridDim.y produce the same bytes, so this is a
// scheduling choice and never a numerical one. CUDA's grid.y limit is
// 65535; a window past that is covered by the stride.
static unsigned mvt_grid_y(int batch) {
    return batch > 65535 ? 65535u : (unsigned)(batch < 1 ? 1 : batch);
}

// ------------------------------------- adapter residency (D2 on CUDA)
// Called once, after model_lora_load has parsed the adapter and before any
// forward. Uploads A and B for every GPU-resident (layer, slot) that carries
// one and allocates the inner-projection scratch; a host layer is skipped
// (its host hook applies the same delta). Fails loudly: an unadapted model
// served quietly looks exactly like a weak adapter.
// A decode graph records the forward it captured, adapter launches included
// or not, so bind and unbind drop any graph captured before them and the
// next decode captures again.
static void lora_graphs_drop(gpu_t *g) {
    for (int i = 0; i < 2; i++) {
        if (g->gexec[i]) { cu.GraphExecDestroy(g->gexec[i]); g->gexec[i] = NULL; }
        if (g->graph[i]) { cu.GraphDestroy(g->graph[i]); g->graph[i] = NULL; }
    }
}

bool gpu_lora_bind(model_t *m) {
    gpu_t *g = m->gpu;
    if (!g) return false;
    if (!g->sw->f_lora_a || !g->sw->f_lora_b) {
        fprintf(stderr, "error: this build's embedded kernels have no "
                "adapter path; run with --gpu off\n");
        return false;
    }
    if (cu.CtxSetCurrent(g->sw->ctx) != 0) return false;
    size_t n = (size_t)m->n_layer * LW_SLOTS;
    g->lora_n = (int)n;
    g->lora_a = calloc(n, sizeof(*g->lora_a));
    g->lora_b = calloc(n, sizeof(*g->lora_b));
    g->lora_r_of = calloc(n, sizeof(*g->lora_r_of));
    g->lora_scale = calloc(n, sizeof(*g->lora_scale));
    if (!g->lora_a || !g->lora_b || !g->lora_r_of || !g->lora_scale) {
        gpu_lora_unbind(m);
        return false;
    }
    int G = m->gpu_layers < m->n_layer ? m->gpu_layers : m->n_layer;
    size_t bytes = 0;
    for (int l = 0; l < G; l++) {
        for (int sl = 0; sl < LW_SLOTS; sl++) {
            const float *a = NULL, *b = NULL;
            int r = 0, n_in = 0, n_out = 0;
            float scale = 0.0f;
            if (!model_lora_slot(m, l, sl, &a, &b, &r, &scale, &n_in, &n_out))
                continue;
            if (r > LORA_R_DEV_MAX) {
                fprintf(stderr, "error: adapter rank %d exceeds the device "
                        "path's %d; run with --gpu off\n", r, LORA_R_DEV_MAX);
                gpu_lora_unbind(m);
                return false;
            }
            size_t na = sizeof(float) * (size_t)r * (size_t)n_in;
            size_t nb = sizeof(float) * (size_t)n_out * (size_t)r;
            size_t idx = (size_t)l * LW_SLOTS + (size_t)sl;
            if (cu.MemAlloc(&g->lora_a[idx], na) != 0 ||
                cu.MemcpyHtoD(g->lora_a[idx], a, na) != 0 ||
                cu.MemAlloc(&g->lora_b[idx], nb) != 0 ||
                cu.MemcpyHtoD(g->lora_b[idx], b, nb) != 0) {
                fprintf(stderr, "error: cannot place the adapter in VRAM "
                        "(%.1f MB placed); run with --gpu off\n", bytes / 1e6);
                gpu_lora_unbind(m);
                return false;
            }
            g->lora_r_of[idx] = r;
            g->lora_scale[idx] = scale;
            if (r > g->lora_r) g->lora_r = r;
            bytes += na + nb;
        }
    }
    if (!g->lora_r) {          // adapter loaded, nothing on the device half
        gpu_lora_unbind(m);
        return true;
    }
    if (cu.MemAlloc(&g->lora_t,
                    sizeof(float) * (size_t)g->win * (size_t)g->lora_r) != 0) {
        fprintf(stderr, "error: cannot place the adapter scratch in VRAM; "
                "run with --gpu off\n");
        gpu_lora_unbind(m);
        return false;
    }
    lora_graphs_drop(g);       // a graph captured before the adapter lacks its launches
    fprintf(stderr, "gpu: adapter on device for %d block%s (%.1f MB VRAM)\n",
            G, G == 1 ? "" : "s", bytes / 1e6);
    return true;
}

void gpu_lora_unbind(model_t *m) {
    gpu_t *g = m ? m->gpu : NULL;
    if (!g) return;
    if (g->lora_a || g->lora_b || g->lora_t || g->gexec[0] || g->gexec[1])
        cu.CtxSetCurrent(g->sw->ctx);
    lora_graphs_drop(g);       // a graph with the adapter's launches reads what is freed next
    lora_dev_release(g);
}

// ------------------- canonical-order forward matvec (R8.7.1 slice 1)
// y[row] = dot(W[row], x) in EXACTLY the association vec_dot uses under
// RUNNER_CANON_KERNELS, so the two agree bit for bit rather than closely.
// Synchronous and allocating its own scratch: this is the correctness and
// throughput primitive, not the integrated training forward, the same way
// gpu_mvt was before slices 2 to 4 consumed it.
//
// false = no backend, no kernel for this type (F32, F16, Q8_0, and since
// R12.1b Q4_0, Q4_K, Q5_K and Q6_K have a canonical order), or the tensor is
// not device-resident.
bool gpu_mvcanon(model_t *m, const gguf_tensor *w, const float *x, float *y,
                 int n_in, int n_out) {
    gpu_t *g = m->gpu;
    if (!g || !w || w->type >= KT_N || !g->sw->f_mvcanon[w->type]) return false;
    if (w->scale != 1.0f) return false;   // a scaled tensor is not this path
    CUdeviceptr weights = 0;
    uint64_t w_off = 0;
    if (!binding_find(g->sw, m, w, &weights, &w_off)) return false;
    if (cu.CtxSetCurrent(g->sw->ctx) != 0) return false;
    CUdeviceptr d_x = 0, d_y = 0;
    if (cu.MemAlloc(&d_x, sizeof(float) * (size_t)n_in) != 0) return false;
    if (cu.MemAlloc(&d_y, sizeof(float) * (size_t)n_out) != 0) {
        cu.MemFree(d_x);
        return false;
    }
    int row_bytes = (int)ggml_row_size(w->type, n_in);
    bool ok = cu.MemcpyHtoD(d_x, x, sizeof(float) * (size_t)n_in) == 0;
    if (ok) {
        void *p[] = { &weights, &d_x, &d_y, &n_in, &n_out, &w_off, &row_bytes };
        // 8 lanes per row, 256 threads = 32 rows per block
        int threads = 256, rows = threads / 8;
        unsigned blocks = (unsigned)((n_out + rows - 1) / rows);
        ok = launch(g, g->sw->f_mvcanon[w->type], blocks, 1, 1, threads, p) &&
             cu.CtxSynchronize() == 0 &&
             cu.MemcpyDtoH(y, d_y, sizeof(float) * (size_t)n_out) == 0;
    }
    cu.MemFree(d_x);
    cu.MemFree(d_y);
    return ok;
}

// ------------------- transposed matvec for training (adaptation D8 slice 1)
// dx[t][i] += sum_j W[j][i]*dy[t][j] on the device, with the accumulator
// chain identical to the CPU trainer (start from dx, serial j, skip zero
// dy). Synchronous, allocates its scratch per call: this is the correctness
// and throughput primitive, not yet the integrated training path.
bool gpu_mvt(model_t *m, const gguf_tensor *w, const float *dy, float *dx,
             int n_in, int n_out, int batch) {
    gpu_t *g = m->gpu;
    if (!g || !w || w->type >= KT_N || !g->sw->f_mvt[w->type]) return false;
    CUdeviceptr weights = 0;
    uint64_t w_off = 0;
    if (!binding_find(g->sw, m, w, &weights, &w_off)) return false;
    if (cu.CtxSetCurrent(g->sw->ctx) != 0) return false;
    CUdeviceptr d_dy = 0, d_dx = 0;
    size_t ny = sizeof(float) * (size_t)batch * n_out;
    size_t nx = sizeof(float) * (size_t)batch * n_in;
    if (cu.MemAlloc(&d_dy, ny) != 0) return false;
    if (cu.MemAlloc(&d_dx, nx) != 0) { cu.MemFree(d_dy); return false; }
    bool ok = cu.MemcpyHtoD(d_dy, dy, ny) == 0 &&
              cu.MemcpyHtoD(d_dx, dx, nx) == 0;
    if (ok) {
        mv_args a = { n_in, n_out, w_off, 0, batch, n_out, n_in };
        CUdeviceptr bias = g->sw->dummy;
        void *p[] = { &weights, &d_dy, &d_dx, &a, &bias };
        int threads = 256;
        int blocks = (n_in + threads - 1) / threads;
        ok = launch(g, g->sw->f_mvt[w->type], blocks, mvt_grid_y(batch), 1,
                    threads, p) &&
             cu.CtxSynchronize() == 0 &&
             cu.MemcpyDtoH(dx, d_dx, nx) == 0;
    }
    cu.MemFree(d_dy);
    cu.MemFree(d_dx);
    return ok;
}

// ------------------- training GPU context
// --train keeps the model CPU-resident, but the backward's dominant
// primitive has a device twin that is bit-identical to the CPU chain. This
// context is its own CUDA state, independent of the serving backend: each
// weight tensor is uploaded once on first use (per-call uploads drowned the
// kernel win) and the dy/dx scratch is reused. A tensor that cannot go to
// the device stays on the CPU path, which produces the same bytes.

// A cached upload is known by the bytes it holds, not by the descriptor
// that pointed at them: a MoE expert reaches the backward as a view of the
// fused expert tensor (moe_expert_weight), built per call, so every expert
// of a layer arrives in a descriptor at the same address. Keyed by that
// address, the first expert's weights served every later expert.
typedef struct {
    const void *data;
    size_t      nbytes;
    CUdeviceptr dev;
} tg_slot;

// R8.21: the DeltaNet recurrence's device buffers (train_gpu_t.dn)
enum { DN_CV, DN_BETA, DN_DECAY, DN_STATE, DN_Y, DN_CKPT, DN_SB, DN_DY, DN_DQ,
       DN_DK, DN_DV, DN_DBETA, DN_DDEC, DN_NBUF };
// the attention backward's device buffers (train_gpu_t.ab)
enum { AB_K, AB_V, AB_Q, AB_DAO, AB_DQ, AB_DK, AB_DV, AB_P, AB_DS, AB_NBUF };

// A forward layer's recurrent state, kept on the device between the rounds
// of a step (known by its host buffer). at says which copy is current: the
// host's (upload it before the next round), the device's (the host's is
// stale until gpu_train_dn_pull), zero (a reset: clear the device copy), or
// lost (a device failure after the device copy was the only current one).
enum { TDN_AT_HOST, TDN_AT_DEV, TDN_AT_ZERO, TDN_AT_LOST };
typedef struct {
    const float *host;
    CUdeviceptr  dev;
    size_t       bytes;
    int          at;
} tdn_slot;

typedef struct {
    CUcontext   ctx;              // primary context, retained on device 0
    CUdevice    dev;
    CUmodule    mod;
    // R8.19 (kernels_train.cuh): the transposed product, the CPU chain's
    // bits, and the batch-invariant forward product, for every type the
    // CPU trainer decodes
    CUfunction  f_tmvt[KT_N], f_tmm[KT_N], f_tmmw[KT_N];
    // R8.20 attempt 3: the register-blocked forward product and its narrow
    // twin (one arithmetic, so either gives a column the same bits)
    CUfunction  f_tmmr[KT_N], f_tmmrn[KT_N];
    // the transposed product's register-blocked twins, --train-gpu only
    CUfunction  f_tmvtr[KT_N], f_tmvtrn[KT_N];
    // R8.21: the DeltaNet recurrence (k_tdn_fw, k_tdn_bw) and its buffers.
    // A recompute leaves its inputs and checkpoints there for the same
    // layer's reverse sweep; dn_owner names the recompute they belong to
    CUfunction  f_tdn_fw, f_tdn_bw;
    CUdeviceptr dn[DN_NBUF];
    size_t      dn_cap[DN_NBUF];
    const float *dn_owner;
    int         dn_T;
    bool        dn_noted;
    double      t_dn;             // RUNNER_TRAIN_PROF: wall inside the calls
    long        n_dn, n_dn_failed;   // the record: calls run, calls failed
    // the attention backward (k_tab_p, k_tab_dq, k_tab_dkv) and its buffers
    CUfunction  f_tab_p, f_tab_dq, f_tab_dkv;
    CUdeviceptr ab[AB_NBUF];
    size_t      ab_cap[AB_NBUF];
    long        n_ab, n_ab_failed;
    bool        ab_noted;
    double      t_ab;
    tdn_slot   *dn_slot;          // the forward's states, one a layer
    int         n_dn_slot, cap_dn_slot;
    CUdeviceptr dummy;            // zeroed no-bias operand
    tg_slot    *cache;
    int         n_cache, cap_cache;
    CUdeviceptr d_dy, d_dx;
    size_t      cap_dy, cap_dx;
    float      *h_x, *h_y;        // the forward's packed host rows
    size_t      cap_hx, cap_hy;
    // RUNNER_TRAIN_PROF: wall time inside the two calls (transfers and the
    // sync included) and their counts, since the last gpu_train_prof
    double      t_mm, t_mvt, k_mm, k_mvt;   // k_: the launch and its sync alone
    long        n_mm, n_mvt;
    size_t      vram_used, vram_budget;
    bool        budget_noted, budget_set;
    // one device arena of min(budget, the model's tensor bytes), reserved at
    // the start: a tensor fits or not by the budget and the order of first
    // use alone, never by what else holds memory later in the run
    CUdeviceptr arena;
    size_t      arena_size;
    // the tensors that stayed on the CPU (by host address), for the record:
    // which tensors fit decides the adapter's bits (their products take the
    // CPU's order), and the free VRAM at the run's start decides which fit
    const void **cpu_kept;
    int         n_cpu, cap_cpu;
    size_t      cpu_bytes;
} train_gpu_t;

bool gpu_train_init(model_t *m) {
    if (m->train_gpu) return true;
    m->train_gpu_err = 1;   // until a device is up: none usable
    if (!cu_load() || cu.Init(0) != 0) return false;
    int n = 0;
    if (cu.DeviceGetCount(&n) != 0 || n < 1) return false;
    train_gpu_t *t = calloc(1, sizeof(*t));
    if (!t) return false;
    if (cu.DeviceGet(&t->dev, 0) != 0 ||
        cu.PrimaryCtxRetain(&t->ctx, t->dev) != 0) {
        free(t);
        return false;
    }
    if (cu.CtxSetCurrent(t->ctx) != 0 ||
        module_load_noted(&t->mod, k_ptx_src) != 0 ||
        cu.MemAlloc(&t->dummy, 4) != 0 ||
        cu.MemsetD8(t->dummy, 0, 4) != 0) {
        cu.PrimaryCtxRelease(t->dev);
        free(t);
        return false;
    }
    static const struct { int type; const char *mvt, *mm, *mmw; } FNS[] = {
        { T_F32, "k_tmvt_f32", "k_tmm_f32", "k_tmmw_f32" },
        { T_F16, "k_tmvt_f16", "k_tmm_f16", "k_tmmw_f16" },
        { T_BF16, "k_tmvt_bf16", "k_tmm_bf16", "k_tmmw_bf16" },
        { T_Q8_0, "k_tmvt_q8_0", "k_tmm_q8_0", "k_tmmw_q8_0" },
        { T_Q4_0, "k_tmvt_q4_0", "k_tmm_q4_0", "k_tmmw_q4_0" },
        { T_Q4_1, "k_tmvt_q4_1", "k_tmm_q4_1", "k_tmmw_q4_1" },
        { T_Q5_0, "k_tmvt_q5_0", "k_tmm_q5_0", "k_tmmw_q5_0" },
        { T_Q5_1, "k_tmvt_q5_1", "k_tmm_q5_1", "k_tmmw_q5_1" },
        { T_Q2_K, "k_tmvt_q2_K", "k_tmm_q2_K", "k_tmmw_q2_K" },
        { T_Q3_K, "k_tmvt_q3_K", "k_tmm_q3_K", "k_tmmw_q3_K" },
        { T_Q4_K, "k_tmvt_q4_K", "k_tmm_q4_K", "k_tmmw_q4_K" },
        { T_Q5_K, "k_tmvt_q5_K", "k_tmm_q5_K", "k_tmmw_q5_K" },
        { T_Q6_K, "k_tmvt_q6_K", "k_tmm_q6_K", "k_tmmw_q6_K" },
        { T_IQ4_NL, "k_tmvt_iq4_nl", "k_tmm_iq4_nl", "k_tmmw_iq4_nl" },
        { T_IQ4_XS, "k_tmvt_iq4_xs", "k_tmm_iq4_xs", "k_tmmw_iq4_xs" },
        { T_IQ2_XXS, "k_tmvt_iq2_xxs", "k_tmm_iq2_xxs", "k_tmmw_iq2_xxs" },
        { T_IQ2_XS, "k_tmvt_iq2_xs", "k_tmm_iq2_xs", "k_tmmw_iq2_xs" },
        { T_IQ2_S, "k_tmvt_iq2_s", "k_tmm_iq2_s", "k_tmmw_iq2_s" },
        { T_IQ3_XXS, "k_tmvt_iq3_xxs", "k_tmm_iq3_xxs", "k_tmmw_iq3_xxs" },
        { T_IQ3_S, "k_tmvt_iq3_s", "k_tmm_iq3_s", "k_tmmw_iq3_s" },
        { T_IQ1_S, "k_tmvt_iq1_s", "k_tmm_iq1_s", "k_tmmw_iq1_s" },
        { T_IQ1_M, "k_tmvt_iq1_m", "k_tmm_iq1_m", "k_tmmw_iq1_m" },
        { T_MXFP4, "k_tmvt_mxfp4", "k_tmm_mxfp4", "k_tmmw_mxfp4" },
        { T_NVFP4, "k_tmvt_nvfp4", "k_tmm_nvfp4", "k_tmmw_nvfp4" },
        { T_Q2_0, "k_tmvt_q2_0", "k_tmm_q2_0", "k_tmmw_q2_0" },
    };
    if (cu.ModuleGetFunction(&t->f_tab_p, t->mod, "k_tab_p") != 0 ||
        cu.ModuleGetFunction(&t->f_tab_dq, t->mod, "k_tab_dq") != 0 ||
        cu.ModuleGetFunction(&t->f_tab_dkv, t->mod, "k_tab_dkv") != 0)
        t->f_tab_p = t->f_tab_dq = t->f_tab_dkv = NULL;
    if (cu.ModuleGetFunction(&t->f_tdn_fw, t->mod, "k_tdn_fw") != 0 ||
        cu.ModuleGetFunction(&t->f_tdn_bw, t->mod, "k_tdn_bw") != 0)
        t->f_tdn_fw = t->f_tdn_bw = NULL;
    for (size_t i = 0; i < sizeof(FNS) / sizeof(*FNS); i++) {
        if (cu.ModuleGetFunction(&t->f_tmvt[FNS[i].type], t->mod,
                                 FNS[i].mvt) != 0)
            t->f_tmvt[FNS[i].type] = NULL;
        if (cu.ModuleGetFunction(&t->f_tmm[FNS[i].type], t->mod,
                                 FNS[i].mm) != 0)
            t->f_tmm[FNS[i].type] = NULL;
        if (cu.ModuleGetFunction(&t->f_tmmw[FNS[i].type], t->mod,
                                 FNS[i].mmw) != 0)
            t->f_tmmw[FNS[i].type] = NULL;
        char nm[48];
        snprintf(nm, sizeof nm, "k_tmmr_%s", FNS[i].mm + 6);
        if (cu.ModuleGetFunction(&t->f_tmmr[FNS[i].type], t->mod, nm) != 0)
            t->f_tmmr[FNS[i].type] = NULL;
        snprintf(nm, sizeof nm, "k_tmmrn_%s", FNS[i].mm + 6);
        if (cu.ModuleGetFunction(&t->f_tmmrn[FNS[i].type], t->mod, nm) != 0)
            t->f_tmmrn[FNS[i].type] = NULL;
        snprintf(nm, sizeof nm, "k_tmvtr_%s", FNS[i].mm + 6);
        if (cu.ModuleGetFunction(&t->f_tmvtr[FNS[i].type], t->mod, nm) != 0)
            t->f_tmvtr[FNS[i].type] = NULL;
        snprintf(nm, sizeof nm, "k_tmvtrn_%s", FNS[i].mm + 6);
        if (cu.ModuleGetFunction(&t->f_tmvtrn[FNS[i].type], t->mod, nm) != 0)
            t->f_tmvtrn[FNS[i].type] = NULL;
    }
    size_t free_b = 0, total_b = 0;
    if (cu.MemGetInfo(&free_b, &total_b) != 0) free_b = 0;
    // leave a margin for the driver and anyone sharing the card. The budget
    // decides which weights the device holds and so the adapter's bits:
    // --train-gpu-budget names it, else the free memory measured here does,
    // and the record carries it either way
    size_t avail = free_b > (512u << 20) ? free_b - (512u << 20) : 0;
    if (m->train_gpu_budget) {
        if (m->train_gpu_budget > avail) {
            fprintf(stderr, "error: --train-gpu-budget %llu is more than "
                    "this device has free (%llu bytes, less a 512 MB "
                    "margin)\n", (unsigned long long)m->train_gpu_budget,
                    (unsigned long long)avail);
            m->train_gpu_err = 2;
            cu.MemFree(t->dummy);
            cu.ModuleUnload(t->mod);
            cu.PrimaryCtxRelease(t->dev);
            free(t);
            return false;
        }
        t->vram_budget = (size_t)m->train_gpu_budget;
        t->budget_set = true;
    } else {
        t->vram_budget = avail;
    }
    // the arena: no more than the model's tensors can use (a caller with
    // no model file, test-train-gpu, gets the budget it set)
    size_t need = 0;
    for (uint64_t i = 0; i < m->gf.n_tensors; i++)
        need += ((size_t)m->gf.tensors[i].nbytes + 255) & ~(size_t)255;
    if (!m->gf.n_tensors) need = t->vram_budget;
    t->arena_size = need < t->vram_budget ? need : t->vram_budget;
    if (t->arena_size && cu.MemAlloc(&t->arena, t->arena_size) != 0) {
        fprintf(stderr, "error: --train-gpu could not reserve %llu bytes "
                "on the device\n", (unsigned long long)t->arena_size);
        m->train_gpu_err = 2;
        cu.MemFree(t->dummy);
        cu.ModuleUnload(t->mod);
        cu.PrimaryCtxRelease(t->dev);
        free(t);
        return false;
    }
    char name[128] = "CUDA GPU";
    gpu_available(name, sizeof(name));
    fprintf(stderr, "train-gpu: %s, weights cached on first use (budget "
            "%.1f GB%s, %.1f GB reserved)\n", name, t->vram_budget / 1e9,
            t->budget_set ? ", --train-gpu-budget" : ", the free memory",
            t->arena_size / 1e9);
    m->train_gpu = t;
    m->train_gpu_err = 0;
    return true;
}

void gpu_train_free(model_t *m) {
    train_gpu_t *t = m->train_gpu;
    if (!t) return;
    if (cu.CtxSetCurrent(t->ctx) == 0) {
        if (t->arena) cu.MemFree(t->arena);
        if (t->d_dy) cu.MemFree(t->d_dy);
        if (t->d_dx) cu.MemFree(t->d_dx);
        for (int i = 0; i < DN_NBUF; i++)
            if (t->dn[i]) cu.MemFree(t->dn[i]);
        for (int i = 0; i < t->n_dn_slot; i++) cu.MemFree(t->dn_slot[i].dev);
        for (int i = 0; i < AB_NBUF; i++)
            if (t->ab[i]) cu.MemFree(t->ab[i]);
        cu.MemFree(t->dummy);
        cu.ModuleUnload(t->mod);
    }
    cu.PrimaryCtxRelease(t->dev);
    free(t->cache);
    free(t->dn_slot);
    free(t->cpu_kept);
    free(t->h_x);
    free(t->h_y);
    free(t);
    m->train_gpu = NULL;
}

// a tensor the budget kept on the CPU, counted once
static void tg_kept(train_gpu_t *t, const gguf_tensor *w) {
    for (int i = 0; i < t->n_cpu; i++)
        if (t->cpu_kept[i] == w->data) return;
    if (t->n_cpu == t->cap_cpu) {
        int cap = t->cap_cpu ? t->cap_cpu * 2 : 64;
        const void **c = realloc(t->cpu_kept, sizeof(*c) * (size_t)cap);
        if (!c) return;
        t->cpu_kept = c;
        t->cap_cpu = cap;
    }
    t->cpu_kept[t->n_cpu++] = w->data;
    t->cpu_bytes += w->nbytes;
}

void gpu_train_placement(model_t *m, int *on_dev, size_t *dev_bytes,
                         int *on_cpu, size_t *cpu_bytes, size_t *budget,
                         bool *budget_set) {
    train_gpu_t *t = m->train_gpu;
    *on_dev = t ? t->n_cache : 0;
    *dev_bytes = t ? t->vram_used : 0;
    *on_cpu = t ? t->n_cpu : 0;
    *cpu_bytes = t ? t->cpu_bytes : 0;
    *budget = t ? t->vram_budget : 0;
    *budget_set = t && t->budget_set;
}

static bool tg_scratch(CUdeviceptr *buf, size_t *cap, size_t need) {
    if (*cap >= need) return true;
    if (*buf) cu.MemFree(*buf);
    *buf = 0;
    *cap = 0;
    if (cu.MemAlloc(buf, need) != 0) return false;
    *cap = need;
    return true;
}

// The tensor's device copy, uploaded on its first use and cached for the
// run; 0 when it does not fit the budget (it then stays on the CPU, for the
// whole run: the budget only fills). Known by host address and size, so the
// host bytes must outlive the context (a mapped model file's do); a buffer
// freed and handed out again at the same address would hit the old copy.
static CUdeviceptr tg_weights(train_gpu_t *t, const gguf_tensor *w) {
    for (int i = 0; i < t->n_cache; i++)
        if (t->cache[i].data == w->data && t->cache[i].nbytes == w->nbytes)
            return t->cache[i].dev;
    size_t span = ((size_t)w->nbytes + 255) & ~(size_t)255;
    if (t->vram_used + span > t->arena_size) {
        if (!t->budget_noted) {
            t->budget_noted = true;
            fprintf(stderr, "train-gpu: VRAM budget reached after %.1f "
                    "GB, remaining tensors train on the CPU (the adapter's "
                    "bits depend on this split; the record names it)\n",
                    t->vram_used / 1e9);
        }
        tg_kept(t, w);
        return 0;
    }
    if (t->n_cache == t->cap_cache) {
        int cap = t->cap_cache ? t->cap_cache * 2 : 64;
        tg_slot *c = realloc(t->cache, sizeof(tg_slot) * (size_t)cap);
        if (!c) return 0;
        t->cache = c;
        t->cap_cache = cap;
    }
    CUdeviceptr weights = t->arena + t->vram_used;
    if (cu.MemcpyHtoD(weights, w->data, w->nbytes) != 0) return 0;
    t->cache[t->n_cache].data = w->data;
    t->cache[t->n_cache].nbytes = w->nbytes;
    t->cache[t->n_cache].dev = weights;
    t->n_cache++;
    t->vram_used += span;
    return weights;
}

static bool train_mvt(train_gpu_t *t, const gguf_tensor *w, const float *dy,
                      float *dx, int n_in, int n_out, int nb, bool dev);

bool gpu_train_mvt(model_t *m, const gguf_tensor *w, const float *dy,
                   float *dx, int n_in, int n_out, int nb) {
    train_gpu_t *t = m->train_gpu;
    // the kernel reads raw rows: a scaled tensor (stored * scale) is the
    // CPU chain's, which multiplies the decoded row first
    if (!t || !w || w->scale != 1.0f || w->type >= KT_N || !t->f_tmvt[w->type])
        return false;
    double t0 = plat_now();
    bool ok = train_mvt(t, w, dy, dx, n_in, n_out, nb, m->train_dev);
    t->t_mvt += plat_now() - t0;
    t->n_mvt += ok;
    return ok;
}

static bool train_mvt(train_gpu_t *t, const gguf_tensor *w, const float *dy,
                      float *dx, int n_in, int n_out, int nb, bool dev) {
    if (cu.CtxSetCurrent(t->ctx) != 0) return false;
    CUdeviceptr weights = tg_weights(t, w);
    if (!weights) return false;
    size_t ny = sizeof(float) * (size_t)nb * n_out;
    size_t nx = sizeof(float) * (size_t)nb * n_in;
    if (!tg_scratch(&t->d_dy, &t->cap_dy, ny) ||
        !tg_scratch(&t->d_dx, &t->cap_dx, nx))
        return false;
    if (cu.MemcpyHtoD(t->d_dy, dy, ny) != 0 ||
        cu.MemcpyHtoD(t->d_dx, dx, nx) != 0)
        return false;
    mv_args a = { n_in, n_out, 0, 0, nb, n_out, n_in };
    void *p[] = { &weights, &t->d_dy, &t->d_dx, &a, &t->dummy };
    // --train-gpu takes the register-blocked twins (k_tmvtr past 32 columns,
    // k_tmvtrn up to them); the assist (RUNNER_TRAIN_GPU=1) keeps k_tmvt, the
    // CPU chain's bits. RUNNER_TRAIN_MVT_KERNEL=chain takes k_tmvt in device
    // mode too, for comparison (read per call, like RUNNER_TRAIN_MM_KERNEL).
    const char *ke = getenv("RUNNER_TRAIN_MVT_KERNEL");
    bool tile = dev && !(ke && !strcmp(ke, "chain")) && t->f_tmvtr[w->type] &&
                t->f_tmvtrn[w->type];
    CUfunction f;
    unsigned blocks, rows, threads;
    if (tile) {
        int ncb = nb > 2 * TMR_BNN ? TMR_BN : TMR_BNN;
        f = ncb == TMR_BN ? t->f_tmvtr[w->type] : t->f_tmvtrn[w->type];
        blocks = (unsigned)((n_in + TMR_BM - 1) / TMR_BM);
        rows = (unsigned)((nb + ncb - 1) / ncb);
        threads = TMR_KS * 64;
    } else {
        f = t->f_tmvt[w->type];
        blocks = (unsigned)((n_in + TV_IT - 1) / TV_IT);
        rows = (unsigned)((nb + TV_TT - 1) / TV_TT);
        threads = TV_IT * TV_TT / 8;
    }
    double k0 = plat_now();
    if (cu.LaunchKernel(f, blocks, rows, 1, threads, 1, 1, 0, NULL, p, NULL) != 0 ||
        cu.CtxSynchronize() != 0)
        return false;
    t->k_mvt += plat_now() - k0;
    return cu.MemcpyDtoH(dx, t->d_dx, nx) == 0;
}

static bool tg_host(float **buf, size_t *cap, size_t n) {
    if (*cap >= n) return true;
    float *b = realloc(*buf, sizeof(float) * n);
    if (!b) return false;
    *buf = b;
    *cap = n;
    return true;
}

static bool train_mm(train_gpu_t *t, const gguf_tensor *w, const float *x,
                     int x_stride, float *y, int y_stride, int n_in, int n_out,
                     const float *bias, int nb);

bool gpu_train_mm(model_t *m, const gguf_tensor *w, const float *x,
                  int x_stride, float *y, int y_stride, int n_in, int n_out,
                  const float *bias, int nb) {
    train_gpu_t *t = m->train_gpu;
    if (!t || !w || nb < 1 || w->scale != 1.0f || w->type >= KT_N ||
        !t->f_tmm[w->type])
        return false;
    double t0 = plat_now();
    bool ok = train_mm(t, w, x, x_stride, y, y_stride, n_in, n_out, bias, nb);
    t->t_mm += plat_now() - t0;
    t->n_mm += ok;
    return ok;
}

void gpu_train_prof(model_t *m, double *t_mm, long *n_mm, double *t_mvt,
                    long *n_mvt, double *k_mm, double *k_mvt) {
    train_gpu_t *t = m->train_gpu;
    *t_mm = t ? t->t_mm : 0;
    *n_mm = t ? t->n_mm : 0;
    *t_mvt = t ? t->t_mvt : 0;
    *n_mvt = t ? t->n_mvt : 0;
    *k_mm = t ? t->k_mm : 0;
    *k_mvt = t ? t->k_mvt : 0;
    if (t) {
        t->t_mm = t->t_mvt = t->k_mm = t->k_mvt = 0;
        t->n_mm = t->n_mvt = 0;
    }
}

double gpu_train_dev_wall(model_t *m) {
    train_gpu_t *t = m->train_gpu;
    return t ? t->t_mm + t->t_mvt + t->t_dn + t->t_ab : 0;
}

// ------------------- R8.21: the DeltaNet recurrence of a training step
// (kernels_train.cuh k_tdn_fw / k_tdn_bw). The shapes the kernels take: a
// head as wide as its key group, a whole number of TDN_ROWS rows, at most
// TDN_MAX_HV; anything else stays on the CPU (decided by the model's shape,
// so the same for every call of a run).
static bool tdn_shape_ok(const tdn_args *a) {
    return a->T >= 1 && a->C >= 1 && a->nh >= 1 && a->ng >= 1 &&
           a->hv == a->sk && a->hv >= TDN_ROWS && a->hv % TDN_ROWS == 0 &&
           a->hv <= TDN_MAX_HV;
}

bool gpu_train_dn_ok(model_t *m, const tdn_args *a) {
    train_gpu_t *t = m->train_gpu;
    return t && t->f_tdn_fw && t->f_tdn_bw && tdn_shape_ok(a);
}

// A call that failed on the device leaves that part of the step to the
// CPU, which the caller does; say so once, so a run whose values mix the
// two is never silent about it.
static void tdn_note(train_gpu_t *t, const char *what) {
    if (t->dn_noted) return;
    t->dn_noted = true;
    fprintf(stderr, "train-gpu: the DeltaNet recurrence's %s failed on the "
            "device and ran on the CPU (this run's values mix the two)\n", what);
}

static bool tdn_buf(train_gpu_t *t, int i, size_t need) {
    return tg_scratch(&t->dn[i], &t->dn_cap[i], need);
}

// the device copy of the forward state at host, made on first use (the
// host's copy current); a host buffer of another size at the same address
// is a new state
static tdn_slot *tdn_slot_for(train_gpu_t *t, const float *host, size_t bytes) {
    for (int i = 0; i < t->n_dn_slot; i++) {
        tdn_slot *s = &t->dn_slot[i];
        if (s->host != host) continue;
        if (s->bytes != bytes) {
            cu.MemFree(s->dev);
            s->dev = 0;
            if (cu.MemAlloc(&s->dev, bytes) != 0) {
                *s = t->dn_slot[--t->n_dn_slot];   // dropped
                return NULL;
            }
            s->bytes = bytes;
            s->at = TDN_AT_HOST;
        }
        return s;
    }
    if (t->n_dn_slot == t->cap_dn_slot) {
        int cap = t->cap_dn_slot ? t->cap_dn_slot * 2 : 64;
        tdn_slot *s = realloc(t->dn_slot, sizeof(*s) * (size_t)cap);
        if (!s) return NULL;
        t->dn_slot = s;
        t->cap_dn_slot = cap;
    }
    tdn_slot *s = &t->dn_slot[t->n_dn_slot];
    if (cu.MemAlloc(&s->dev, bytes) != 0) return NULL;
    s->host = host;
    s->bytes = bytes;
    s->at = TDN_AT_HOST;
    t->n_dn_slot++;
    return s;
}

bool gpu_train_dn_pull(model_t *m) {
    train_gpu_t *t = m->train_gpu;
    if (!t || !t->n_dn_slot) return true;
    if (cu.CtxSetCurrent(t->ctx) != 0) return false;
    bool ok = true;
    for (int i = 0; i < t->n_dn_slot; i++) {
        tdn_slot *s = &t->dn_slot[i];
        if (s->at == TDN_AT_LOST) ok = false;
        if (s->at != TDN_AT_DEV) continue;
        if (cu.MemcpyDtoH((void *)s->host, s->dev, s->bytes) == 0) s->at = TDN_AT_HOST;
        else ok = false;
    }
    return ok;
}

void gpu_train_dn_reset(model_t *m) {
    train_gpu_t *t = m->train_gpu;
    for (int i = 0; t && i < t->n_dn_slot; i++) t->dn_slot[i].at = TDN_AT_ZERO;
}

static bool train_dn_fw(train_gpu_t *t, const tdn_args *a, const float *cv,
                        const float *beta, const float *decay, float *state,
                        float *y) {
    if (cu.CtxSetCurrent(t->ctx) != 0) return false;
    size_t T = (size_t)a->T, hh = (size_t)a->hv * (size_t)a->hv;
    size_t ncv = sizeof(float) * T * (size_t)a->convdim;
    size_t ng = sizeof(float) * T * (size_t)a->nh;
    size_t ny = sizeof(float) * T * (size_t)a->inner;
    size_t ns = sizeof(float) * (size_t)a->nh * hh;
    size_t nseg = (T + (size_t)a->C - 1) / (size_t)a->C;
    size_t nqk = sizeof(float) * (size_t)a->nh * T * (size_t)a->sk;
    bool rec = !state;   // the recompute: from zero, checkpoints kept
    t->dn_owner = NULL;
    // a forward round's state stays on the device for the next round
    tdn_slot *sl = rec ? NULL : tdn_slot_for(t, state, ns);
    if (!rec && !sl) return false;
    if (!tdn_buf(t, DN_CV, ncv) || !tdn_buf(t, DN_BETA, ng) ||
        !tdn_buf(t, DN_DECAY, ng) || (rec && !tdn_buf(t, DN_STATE, ns)) ||
        !tdn_buf(t, DN_Y, ny))
        return false;
    // the recompute takes the reverse sweep's buffers too, so a sweep that
    // follows has them
    if (rec && (!tdn_buf(t, DN_CKPT, nseg * ns) ||
                !tdn_buf(t, DN_SB, (size_t)a->C * ns) ||
                !tdn_buf(t, DN_DY, ny) || !tdn_buf(t, DN_DQ, nqk) ||
                !tdn_buf(t, DN_DK, nqk) || !tdn_buf(t, DN_DV, ny) ||
                !tdn_buf(t, DN_DBETA, ng) || !tdn_buf(t, DN_DDEC, ng)))
        return false;
    if (cu.MemcpyHtoD(t->dn[DN_CV], cv, ncv) != 0 ||
        cu.MemcpyHtoD(t->dn[DN_BETA], beta, ng) != 0 ||
        cu.MemcpyHtoD(t->dn[DN_DECAY], decay, ng) != 0)
        return false;
    int prior = sl ? sl->at : TDN_AT_HOST;
    if (prior == TDN_AT_LOST ||
        (sl && prior == TDN_AT_HOST && cu.MemcpyHtoD(sl->dev, state, ns) != 0) ||
        (sl && prior == TDN_AT_ZERO && cu.MemsetD8(sl->dev, 0, ns) != 0))
        return false;
    CUdeviceptr st = sl ? sl->dev : t->dn[DN_STATE];
    CUdeviceptr ck = rec ? t->dn[DN_CKPT] : 0;
    tdn_args aa = *a;
    int load = !rec;
    void *p[] = { &t->dn[DN_CV], &t->dn[DN_BETA], &t->dn[DN_DECAY], &st,
                  &t->dn[DN_Y], &ck, &aa, &load };
    unsigned gy = (unsigned)((a->hv + TDN_ROWS * TDN_FW_WARPS - 1) /
                             (TDN_ROWS * TDN_FW_WARPS));
    if (cu.LaunchKernel(t->f_tdn_fw, (unsigned)a->nh, gy, 1,
                        TDN_FW_WARPS * 32, 1, 1, 0, NULL, p, NULL) != 0 ||
        cu.CtxSynchronize() != 0 ||
        cu.MemcpyDtoH(y, t->dn[DN_Y], ny) != 0) {
        // the kernel may have run: a state only the device held is gone; a
        // host-current or reset state is still the host's
        if (sl) sl->at = prior == TDN_AT_DEV ? TDN_AT_LOST : TDN_AT_HOST;
        return false;
    }
    if (sl) sl->at = TDN_AT_DEV;
    if (rec) {
        t->dn_owner = cv;
        t->dn_T = a->T;
    }
    return true;
}

bool gpu_train_dn_fw(model_t *m, const tdn_args *a, const float *cv,
                     const float *beta, const float *decay, float *state,
                     float *y) {
    train_gpu_t *t = m->train_gpu;
    if (!gpu_train_dn_ok(m, a)) return false;
    double t0 = plat_now();
    bool ok = train_dn_fw(t, a, cv, beta, decay, state, y);
    t->t_dn += plat_now() - t0;
    if (ok) t->n_dn++;
    else {
        t->n_dn_failed++;
        tdn_note(t, state ? "forward" : "recompute");
    }
    return ok;
}

static bool train_dn_bw(train_gpu_t *t, const tdn_args *a, const float *dy,
                        float *dq, float *dk, float *dv, float *dbeta,
                        float *ddecay) {
    if (cu.CtxSetCurrent(t->ctx) != 0) return false;
    size_t T = (size_t)a->T;
    size_t ny = sizeof(float) * T * (size_t)a->inner;
    size_t ng = sizeof(float) * T * (size_t)a->nh;
    size_t nqk = sizeof(float) * (size_t)a->nh * T * (size_t)a->sk;
    if (cu.MemcpyHtoD(t->dn[DN_DY], dy, ny) != 0) return false;
    tdn_args aa = *a;
    void *p[] = { &t->dn[DN_CV], &t->dn[DN_BETA], &t->dn[DN_DECAY],
                  &t->dn[DN_CKPT], &t->dn[DN_SB], &t->dn[DN_DY], &t->dn[DN_DQ],
                  &t->dn[DN_DK], &t->dn[DN_DV], &t->dn[DN_DBETA],
                  &t->dn[DN_DDEC], &aa };
    return cu.LaunchKernel(t->f_tdn_bw, (unsigned)a->nh, 1, 1,
                           (unsigned)(a->hv / TDN_ROWS * 32), 1, 1, 0, NULL, p,
                           NULL) == 0 &&
           cu.CtxSynchronize() == 0 &&
           cu.MemcpyDtoH(dq, t->dn[DN_DQ], nqk) == 0 &&
           cu.MemcpyDtoH(dk, t->dn[DN_DK], nqk) == 0 &&
           cu.MemcpyDtoH(dv, t->dn[DN_DV], ny) == 0 &&
           cu.MemcpyDtoH(dbeta, t->dn[DN_DBETA], ng) == 0 &&
           cu.MemcpyDtoH(ddecay, t->dn[DN_DDEC], ng) == 0;
}

bool gpu_train_dn_bw(model_t *m, const tdn_args *a, const float *cv,
                     const float *dy, float *dq, float *dk, float *dv,
                     float *dbeta, float *ddecay) {
    train_gpu_t *t = m->train_gpu;
    // only on the device recompute of these very inputs: its checkpoints
    // are the states the sweep starts from
    if (!gpu_train_dn_ok(m, a) || !cv || t->dn_owner != cv || t->dn_T != a->T)
        return false;
    double t0 = plat_now();
    bool ok = train_dn_bw(t, a, dy, dq, dk, dv, dbeta, ddecay);
    t->dn_owner = NULL;
    t->t_dn += plat_now() - t0;
    if (ok) t->n_dn++;
    else {
        t->n_dn_failed++;
        tdn_note(t, "reverse sweep");
    }
    return ok;
}

// ------------------- the attention backward of a training step
// (kernels_train.cuh k_tab_*): the window in blocks of TAB_TB positions, each
// block's probabilities and score gradients, then dq for its rows and dK /
// dV's terms from them, the blocks in ascending order.
static bool tab_buf(train_gpu_t *t, int i, size_t need) {
    return tg_scratch(&t->ab[i], &t->ab_cap[i], need);
}

static bool train_attn_bw(train_gpu_t *t, const tab_args *a, const void *kc,
                          const void *vc, const float *q, const float *dao,
                          float *dq, float *dk, float *dv) {
    if (cu.CtxSetCurrent(t->ctx) != 0) return false;
    size_t T = (size_t)a->T, H = (size_t)a->n_head;
    size_t nk = T * a->k_row_b, nv = T * a->v_row_b;
    size_t nq = sizeof(float) * T * (size_t)a->q_dim;
    size_t nkv = sizeof(float) * T * (size_t)a->kv_dim;
    size_t tb = T < TAB_TB ? T : TAB_TB;
    size_t np = sizeof(float) * tb * H * T;
    if (!tab_buf(t, AB_K, nk) || !tab_buf(t, AB_V, nv) || !tab_buf(t, AB_Q, nq) ||
        !tab_buf(t, AB_DAO, nq) || !tab_buf(t, AB_DQ, nq) || !tab_buf(t, AB_DK, nkv) ||
        !tab_buf(t, AB_DV, nkv) || !tab_buf(t, AB_P, np) || !tab_buf(t, AB_DS, np))
        return false;
    if (cu.MemcpyHtoD(t->ab[AB_K], kc, nk) != 0 || cu.MemcpyHtoD(t->ab[AB_V], vc, nv) != 0 ||
        cu.MemcpyHtoD(t->ab[AB_Q], q, nq) != 0 || cu.MemcpyHtoD(t->ab[AB_DAO], dao, nq) != 0 ||
        cu.MemcpyHtoD(t->ab[AB_DQ], dq, nq) != 0 || cu.MemcpyHtoD(t->ab[AB_DK], dk, nkv) != 0 ||
        cu.MemcpyHtoD(t->ab[AB_DV], dv, nkv) != 0)
        return false;
    tab_args aa = *a;
    unsigned nkvh = (unsigned)(a->n_head / a->kv_mul);
    for (int tb0 = 0; tb0 < a->T; tb0 += TAB_TB) {
        int tbn = a->T - tb0 < TAB_TB ? a->T - tb0 : TAB_TB;
        void *pp[] = { &t->ab[AB_K], &t->ab[AB_V], &t->ab[AB_Q], &t->ab[AB_DAO],
                       &t->ab[AB_P], &t->ab[AB_DS], &aa, &tb0 };
        void *pq[] = { &t->ab[AB_K], &t->ab[AB_DS], &t->ab[AB_DQ], &aa, &tb0 };
        void *pk[] = { &t->ab[AB_Q], &t->ab[AB_DAO], &t->ab[AB_P], &t->ab[AB_DS],
                       &t->ab[AB_DK], &t->ab[AB_DV], &aa, &tb0, &tbn };
        if (cu.LaunchKernel(t->f_tab_p, (unsigned)tbn, (unsigned)a->n_head, 1,
                            TAB_THREADS, 1, 1, 0, NULL, pp, NULL) != 0 ||
            cu.LaunchKernel(t->f_tab_dq, (unsigned)tbn, (unsigned)a->n_head, 1,
                            TAB_THREADS, 1, 1, 0, NULL, pq, NULL) != 0 ||
            cu.LaunchKernel(t->f_tab_dkv, (unsigned)(tb0 + tbn), nkvh, 1,
                            TAB_THREADS, 1, 1, 0, NULL, pk, NULL) != 0)
            return false;
    }
    // all three come back before any lands, so a failed copy leaves the
    // caller's sums as they were for the CPU to add into
    size_t nh = (nq + 2 * nkv) / sizeof(float);
    if (!tg_host(&t->h_y, &t->cap_hy, nh)) return false;
    float *hq = t->h_y, *hk = hq + nq / sizeof(float), *hv = hk + nkv / sizeof(float);
    if (cu.CtxSynchronize() != 0 || cu.MemcpyDtoH(hq, t->ab[AB_DQ], nq) != 0 ||
        cu.MemcpyDtoH(hk, t->ab[AB_DK], nkv) != 0 ||
        cu.MemcpyDtoH(hv, t->ab[AB_DV], nkv) != 0)
        return false;
    memcpy(dq, hq, nq);
    memcpy(dk, hk, nkv);
    memcpy(dv, hv, nkv);
    return true;
}

bool gpu_train_attn_bw(model_t *m, const tab_args *a, const void *kc,
                       const void *vc, const float *q, const float *dao,
                       float *dq, float *dk, float *dv) {
    train_gpu_t *t = m->train_gpu;
    if (!t || !t->f_tab_p || a->T < 1 || a->hd < 1 || a->hd > TAB_MAX_HD ||
        a->kv_mul < 1 || a->n_head % a->kv_mul)
        return false;
    double t0 = plat_now();
    bool ok = train_attn_bw(t, a, kc, vc, q, dao, dq, dk, dv);
    t->t_ab += plat_now() - t0;
    if (ok) t->n_ab++;
    else {
        t->n_ab_failed++;
        if (!t->ab_noted) {
            t->ab_noted = true;
            fprintf(stderr, "train-gpu: the attention backward failed on the "
                    "device and ran on the CPU (this run's values mix the two)\n");
        }
    }
    return ok;
}

void gpu_train_attn_counts(model_t *m, long *dev, long *failed) {
    train_gpu_t *t = m->train_gpu;
    *dev = t ? t->n_ab : 0;
    *failed = t ? t->n_ab_failed : 0;
}

void gpu_train_dn_counts(model_t *m, long *dev, long *failed) {
    train_gpu_t *t = m->train_gpu;
    *dev = t ? t->n_dn : 0;
    *failed = t ? t->n_dn_failed : 0;
}

static bool train_mm(train_gpu_t *t, const gguf_tensor *w, const float *x,
                     int x_stride, float *y, int y_stride, int n_in, int n_out,
                     const float *bias, int nb) {
    if (cu.CtxSetCurrent(t->ctx) != 0) return false;
    CUdeviceptr weights = tg_weights(t, w);
    if (!weights) return false;
    size_t nx = (size_t)nb * n_in, ny = (size_t)nb * n_out;
    if (!tg_scratch(&t->d_dx, &t->cap_dx, sizeof(float) * nx) ||
        !tg_scratch(&t->d_dy, &t->cap_dy, sizeof(float) * ny) ||
        !tg_host(&t->h_y, &t->cap_hy, ny))
        return false;
    const float *xp = x;
    if (x_stride != n_in) {
        if (!tg_host(&t->h_x, &t->cap_hx, nx)) return false;
        for (int c = 0; c < nb; c++)
            memcpy(t->h_x + (size_t)c * n_in, x + (size_t)c * x_stride,
                   sizeof(float) * (size_t)n_in);
        xp = t->h_x;
    }
    if (cu.MemcpyHtoD(t->d_dx, xp, sizeof(float) * nx) != 0) return false;
    mv_args a = { n_in, n_out, 0, 0, nb, n_in, n_out };
    void *p[] = { &weights, &t->d_dx, &t->d_dy, &a, &t->dummy };
    // The register-blocked kernels (k_tmmr, k_tmmrn past 32 columns and up
    // to them); RUNNER_TRAIN_MM_KERNEL=warp takes the per-warp ones instead
    // (k_tmm, k_tmmw past one 16-column tile), the arithmetic before R8.20
    // attempt 3, for comparison. Within a family every column gets the same
    // bits from either kernel; the two families differ, so one family serves
    // a whole run (read per call so test-train-gpu can compare them).
    const char *ke = getenv("RUNNER_TRAIN_MM_KERNEL");
    bool warp = (ke && !strcmp(ke, "warp")) || !t->f_tmmr[w->type] ||
                !t->f_tmmrn[w->type];
    CUfunction f;
    unsigned blocks, cols, threads;
    if (warp) {
        bool wide = nb > TMM_NC && t->f_tmmw[w->type];
        int ncb = wide ? TMM_NCW : TMM_NC;
        f = wide ? t->f_tmmw[w->type] : t->f_tmm[w->type];
        blocks = (unsigned)((n_out + TMM_WARPS * TMM_RW - 1) / (TMM_WARPS * TMM_RW));
        cols = (unsigned)((nb + ncb - 1) / ncb);
        threads = TMM_WARPS * 32;
    } else {
        int ncb = nb > 2 * TMR_BNN ? TMR_BN : TMR_BNN;
        f = ncb == TMR_BN ? t->f_tmmr[w->type] : t->f_tmmrn[w->type];
        blocks = (unsigned)((n_out + TMR_BM - 1) / TMR_BM);
        cols = (unsigned)((nb + ncb - 1) / ncb);
        threads = TMR_KS * 64;
    }
    double k0 = plat_now();
    if (cu.LaunchKernel(f, blocks, cols, 1, threads, 1, 1, 0, NULL, p, NULL) != 0 ||
        cu.CtxSynchronize() != 0)
        return false;
    t->k_mm += plat_now() - k0;
    if (cu.MemcpyDtoH(t->h_y, t->d_dy, sizeof(float) * ny) != 0)
        return false;
    for (int c = 0; c < nb; c++) {
        const float *src = t->h_y + (size_t)c * n_out;
        float *dst = y + (size_t)c * y_stride;
        if (bias)
            for (int j = 0; j < n_out; j++) dst[j] = src[j] + bias[j];
        else
            memcpy(dst, src, sizeof(float) * (size_t)n_out);
    }
    return true;
}

// Discrete VRAM: the hidden states and per-row logits live device-side and
// the CPU verify walk cannot read them without copies the design does not
// want on this path. Full-offload speculative decoding stays refused here.
// Speculative verify on CUDA: a full split emits every row's head into
// g->logits (fwd_tile, spec_want_all) and gpu_forward_batch brings the rows
// to h_logits; the walk reads them by row. A partial split hands the
// boundary rows to the CPU, which serves the walk itself.
bool gpu_spec_keep_ok(const model_t *m) {
    gpu_t *g = m ? m->gpu : NULL;
    return g && m->gpu_layers >= m->n_layer && m->spec_batch <= MVB;
}

// cuda_fwd_tiles copies each tile's final residual to the host, the copy a
// partial split makes at its boundary. The wide hyper-connection residual
// is not handed.
bool gpu_hidden_rows_ok(const model_t *m) {
    return m && m->gpu && !m->hyper_conn;
}

// The selections enc_mv and the MoE path make for this instance (plan P1;
// docs/kernel-routes.md has each one's obligation): whether one-token decode
// replays its graph, the tensor-core GEMM's mode (promoted per type and
// architecture, forced on, or off), the int8-activation GEMVs, the
// small-batch twins, and the MoE prefill path (eager, fused, by-expert where
// a tile admits it, or grouped).
const char *gpu_route_detail(const model_t *m) {
    const gpu_t *g = m ? m->gpu : NULL;
    if (!g) return "";
    static _Thread_local char buf[240];
    if (g_tc_state == TC_ENV_UNSET) (void)tc_on(m, T_Q4_K);   // reads the env once
    // window: the prompt columns one pass of the device layers takes
    // (model_gpu_window); tc_cols: the column tile each tensor-core GEMM
    // launch covers inside it, so a window of 512 is 8 such launches
    // tc: the promoted, forced or off state, then "+nvfp4" where
    // RUNNER_CUDA_TC_NVFP4=1 forces NVFP4's route and "-nvfp4" where =0 holds
    // it off; i8: the int8 tensor-core prefill (RUNNER_CUDA_I8) armed, its
    // kernels loaded and its buffers made, so a receipt names that route;
    // " attn_wide=on" where k_attn_tcw takes some layer of this model
    bool wide = false;
    if (attn_tcw_on(m) && !model_kv_kind_k(m) && !model_kv_kind_v(m))
        for (int l = 0; l < m->n_layer && !wide; l++) {
            int hd = model_head_dim(m, l);
            wide = !(g->sw->sinks && g->sw->sinks[l]) &&
                   ((g->sw->f_attn_tcw && hd % 128 == 0 && hd >= 256 && hd <= 512) ||
                    (g->sw->f_attn_tcw_mla && tcw_mla_vlo(m, hd) >= 0));
        }
    snprintf(buf, sizeof buf, "graph=%s tc=%s%s tc2=%s q8x=%s i8=%s small_batch=%s attn_tile=%s%s "
             "window=%d tc_cols=%d%s%s%s",
             g->graph_bad ? "off" : g->gexec[0] || g->gexec[1] ? "replay" : "none",
             g_tc_state < 0 || moe_tc_family(m)
                 ? (moe_tc_family(m) ? (tc_on(m, T_Q4_K) ? "moe-tc" : "off")
                                     : tc_default_takes(m) ? "taken" : "off")
                 : g_tc_state ? "forced" : "off",
             g_tc_state >= 0 || tc_nvfp4_env() < 0 ? "" : tc_nvfp4_env() ? "+nvfp4" : "-nvfp4",
             !g->xh ? "off" : g->sw->mod80 ? "on-sm80" : "on",
             gpu_q8x_on() ? "on" : "off", g->i8_xq && g->sw->f_quant_i8 ? "on" : "off",
             "on",   // small_batch: always since R18.8.11; the token stays for receipts
             attn_tc_on(m) ? (attn_dec_on() ? "tc+dec" : "tc")
             : attn_tile_on() ? (attn_dec_on() ? "on+dec" : "on") : (attn_dec_on() ? "dec" : "off"),
             wide ? " attn_wide=on" : "",
             g->win, g->xh ? G2_BN : TC_N,
             m->n_expert > 0 ? " moe=" : "",
             m->n_expert > 0 ? (!g->moe_fused_ok ? "eager" : g->moe_grouped_ok ? "grouped"
                                : g->moe_twin_ok && moe_twin_on(m) ? "by-expert" : "fused") : "",
             recur_seq_admits(g, m) ? " recur=tile" : "");
    return buf;
}
void gpu_kv_host_wrote(model_t *m, int lo) {
    gpu_t *g = m ? m->gpu : NULL;
    if (g && lo < g->kv_rows) g->kv_rows = lo < 0 ? 0 : lo;
}

float *gpu_spec_logits(model_t *m, int row) {
    gpu_t *g = m ? m->gpu : NULL;
    if (!g || !g->h_logits || row < 0 || row >= m->spec_batch) return NULL;
    return g->h_logits + (size_t)row * m->n_vocab;
}

gpu_batch *gpu_batch_create(model_t **seqs, int n) {
    gpu_t *lead = NULL;
    if (!batch_eligible(seqs, n, &lead)) return NULL;
    model_t *m0 = seqs[0];

    gpu_batch *b = calloc(1, sizeof(gpu_batch));
    if (!b) {
        fprintf(stderr, "gpu: CUDA batch state allocation failed — using "
                "sequential GPU forwards\n");
        return NULL;
    }
    b->seqs = malloc(sizeof(model_t *) * (size_t)n);
    if (!b->seqs) {
        fprintf(stderr, "gpu: CUDA batch sequence-table allocation failed — "
                "using sequential GPU forwards\n");
        free(b);
        return NULL;
    }
    memcpy(b->seqs, seqs, sizeof(model_t *) * (size_t)n);
    b->n = n;
    b->lead = lead;
    b->sw = lead->sw;
    b->n_vocab = m0->n_vocab;
    b->n_embd  = m0->n_embd;
    b->xdim = m0->n_embd;
    for (int l = 0; l < m0->n_layer; l++)
        if (model_q_dim(m0, l) > b->xdim) b->xdim = model_q_dim(m0, l);

    CUresult ctx_rc = cu.CtxSetCurrent(b->sw->ctx);
    if (ctx_rc != 0) {
        fprintf(stderr, "gpu: CUDA batch context selection failed: %s — "
                "using sequential GPU forwards\n", cu_err(ctx_rc));
        goto fail;
    }
    CK(cu.MemAlloc(&b->logits_n, sizeof(float) * (size_t)MVB * b->n_vocab));
    CK(cu.MemAlloc(&b->pos_d, sizeof(int) * MVB));
    CK(cu.MemAlloc(&b->kcp_d, sizeof(uint64_t) * MVB));
    CK(cu.MemAlloc(&b->vcp_d, sizeof(uint64_t) * MVB));
    b->h_logits = malloc(sizeof(float) * (size_t)MVB * b->n_vocab);
    if (!b->h_logits) {
        fprintf(stderr, "gpu: CUDA batch host-logit allocation failed — using "
                "sequential GPU forwards\n");
        goto fail;
    }
    // no stream means no graphs; plain launches still work, just slower.
    // gemma4's V-copy path (ly->wv == NULL) uses a synchronous MemsetD8 that
    // cannot be captured, same as the solo decode graph refuses it.
    b->graph_bad = lead->stream == NULL || !cu_graphs_ok();
    if (b->graph_bad) b->graph_why = "no stream to capture on (or the driver has no graph API)";
    for (int l = 0; l < m0->n_layer; l++)
        if (!m0->layers[l].wv && !v_from_k_capturable(m0, l)) {
            b->graph_bad = true;
            if (!b->graph_why)
                b->graph_why = "a layer without a V projection that is not built on the stream (a synchronous memset)";
        }
    return b;

fail:
    gpu_batch_free(b);
    return NULL;
}

bool gpu_batch_tiles_used(const gpu_batch *b) { (void)b; return false; }

void gpu_batch_free(gpu_batch *b) {
    if (!b) return;
    if (b->sw && b->sw->ctx) cu.CtxSetCurrent(b->sw->ctx);
    for (int i = 0; i <= MVB; i++) {
        if (b->gexec[i] && cu.GraphExecDestroy) cu.GraphExecDestroy(b->gexec[i]);
        if (b->graph[i] && cu.GraphDestroy) cu.GraphDestroy(b->graph[i]);
    }
    CUdeviceptr bufs[] = { b->logits_n, b->pos_d, b->kcp_d, b->vcp_d };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++)
        if (bufs[i]) cu.MemFree(bufs[i]);
    free(b->h_logits);
    free(b->seqs);
    free(b);
}

// Matvec for a decode microbatch. The selection rule is the whole correctness
// argument: pick the MULTI-COLUMN TWIN of whatever kernel gpu_forward_batch
// would have picked at batch 1 — f_gemvb where f_gemv exists, f_mvb otherwise
// — so each column reproduces the lone-token result bit for bit. It must never
// reach for f_gemm: the prefill GEMM is faster and reassociates the reduction,
// which is exactly the trade this path exists to refuse.
static bool enc_mv_batch(gpu_t *g, model_t *m, gguf_tensor *w, CUdeviceptr x,
                         CUdeviceptr y, int n_in, int n_out, CUdeviceptr bias,
                         int batch, int xs, int ys) {
    CUdeviceptr weights = 0;
    uint64_t w_off = 0;
    if (!binding_find(g->sw, m, w, &weights, &w_off)) {
        gpu_first_error("tensor '%s' is not resident on the device "
                        "(no binding covers it)", w->name);
        return false;
    }
    mv_args a = { n_in, n_out, w_off, bias != 0, batch, xs, ys };
    CUdeviceptr bp = bias ? bias : g->sw->dummy;
    void *p[] = { &weights, &x, &y, &a, &bp };
    // GEMM-shaped twins stage x in shared memory and give 8 rows per block;
    // without that, MVB scattered global x-loads per decoded weight make a
    // microbatch L1-bound and it loses to running the sequences separately.
    CUfunction f = batch >= 2 && batch <= 8 && g->sw->f_gemvx[batch][w->type]
                 ? g->sw->f_gemvx[batch][w->type]
                 : g->sw->f_gemvb[batch_width_class(batch)][w->type];
    // rows a warp: the codebook i-quant twins take IQ_TWR (2) (kernels.cu),
    // the Q3_K, IQ4_XS, Q8_0, Q4_0 and BF16 ones KQ_TWR / DQ_TWR of their
    // class width (kernel_args.h)
    bool iq2r = w->type == T_IQ2_XXS || w->type == T_IQ2_XS || w->type == T_IQ2_S ||
                w->type == T_IQ3_XXS || w->type == T_IQ3_S;
    int cls_nc = 4 << batch_width_class(batch);
    // Q4_K, Q5_K and Q6_K take rows a warp and warps a block (KQ_DBR, KQ_DBW)
    // by the width the kernel was built for: the exact 5 to 7, or the class
    bool kq = w->type == T_Q4_K || w->type == T_Q5_K || w->type == T_Q6_K;
    bool exact = batch >= 2 && batch <= 8 && f && f == g->sw->f_gemvx[batch][w->type];
    int kq_nc = exact ? batch : cls_nc;
    unsigned rpw = iq2r ? 2
                 : kq ? KQ_DBR(kq_nc)
                 : w->type == T_Q3_K || w->type == T_IQ4_XS ? KQ_TWR(cls_nc)
                 : w->type == T_Q8_0 || w->type == T_Q4_0 || w->type == T_BF16
                   ? DQ_TWR(cls_nc) : 1;
    unsigned wpb = kq ? KQ_DBW(kq_nc) : GEMM_WARPS;
    unsigned rpb = wpb * rpw;
    if (f) return launch(g, f, (n_out + rpb - 1) / rpb, 1, 1, wpb * 32, p);
    // No width-classed twin. f_mvb is the right answer only where it really is
    // f_mv's twin (see mvb_is_mv_twin) — for a quantized type it is a different
    // association and therefore different bits, which is exactly what this path
    // exists to refuse. batch_eligible() declines such a model at
    // gpu_batch_create time and the caller decodes its sequences one at a time,
    // so reaching here means that check has drifted: name the tensor rather
    // than silently return different numbers.
    if (mvb_is_mv_twin(w->type) && g->sw->f_mvb[w->type])
        return launch(g, g->sw->f_mvb[w->type], (n_out + 3) / 4, 1, 1, 128, p);
    gpu_first_error("tensor '%s' has no batched kernel that is bitwise the "
                    "batch-1 kernel; refusing the microbatch", w->name);
    return false;
}

static bool enc_rope_batch(gpu_batch *B, model_t *m, CUdeviceptr v, int n_heads,
                           int batch, int vs, int l) {
    gpu_t *g = B->lead;
    bool local = model_is_swa(m, l);
    rope_args a = { model_head_dim(m, l), n_heads, model_rope_dim(m, l) / 2,
                    m->rope_neox, model_rope_mscale(m, l) };
    CUdeviceptr fr = local ? g->sw->inv_freq_local : g->sw->inv_freq;
    void *p[] = { &v, &fr, &a, &B->pos_d, &vs };
    return launch(g, g->sw->f_rope_seq, (a.half_dim + 31) / 32, n_heads, batch, 32, p);
}

// The layer loop, structurally fwd_tile with tn = the microbatch width. Only
// the position-bearing and KV-bearing kernels differ; every other launch is
// the same call with a wider batch, and each already treats its columns
// independently. Logits are produced for EVERY column, not just the last —
// a microbatch has N answers, not one.
static bool fwd_batch(gpu_batch *B, model_t *m, int tn) {
    gpu_t *g = B->lead;
    int n_embd = B->n_embd, xdim = B->xdim;
    bool ok = true;

    for (int l = 0; l < m->n_layer && ok; l++) {
        layer_t *ly = &m->layers[l];
        bool local  = model_is_swa(m, l);
        int hd      = model_head_dim(m, l);
        int n_kv    = model_n_head_kv(m, l);
        int q_dim   = model_q_dim(m, l);
        int kv_dim  = model_kv_dim(m, l);

        ok = ok && enc_rmsnorm(g, g->x, g->xb, g->sw->attn_norm[l], n_embd,
                               m->rms_eps, tn, n_embd, xdim);
        ok = ok && enc_mv_batch(g, m, ly->wq, g->xb, g->q,  n_embd, q_dim,  g->sw->bq[l], tn, xdim, q_dim);
        // shared-KV layers: Q only, then attend over the owning layer's rows
        if (model_kv_owner(m, l) == l) {
        ok = ok && enc_mv_batch(g, m, ly->wk, g->xb, g->kt, n_embd, kv_dim, g->sw->bk[l], tn, xdim, kv_dim);
        if (ly->wv) {
            ok = ok && enc_mv_batch(g, m, ly->wv, g->xb, g->vt, n_embd, kv_dim, g->sw->bv[l], tn, xdim, kv_dim);
        } else {
            ok = ok && enc_zero(g, g->vt, sizeof(float) * (size_t)tn * kv_dim);
            ok = ok && enc_add(g, g->vt, g->kt, kv_dim, tn, kv_dim, kv_dim);
        }
        if (m->v_rmsnorm)
            ok = ok && enc_qknorm(g, m, g->vt, g->sw->ones, n_kv, hd, tn, kv_dim);
        if (g->sw->kn[l]) ok = ok && enc_qknorm(g, m, g->kt, g->sw->kn[l], n_kv, hd, tn, kv_dim);
        ok = ok && enc_rope_batch(B, m, g->kt, n_kv, tn, kv_dim, l);

        {   // each column stores into its own sequence's cache at its own row
            uint64_t l_off = model_k_byte_off(m, l), v_off = model_v_byte_off(m, l);
            int q8 = model_kv_kind_k(m), vq8 = model_kv_kind_v(m);   // cache KINDS
            int ku = kv_store_units(q8, kv_dim), vu = kv_store_units(vq8, kv_dim);
            int units = ku > vu ? ku : vu;
            int kring = model_kv_is_ring(m, l) ? m->kv_ring : 0;
            void *ps[] = { &g->kt, &g->vt, &B->kcp_d, &B->vcp_d, &kv_dim,
                           &l_off, &B->pos_d, &q8, &kring, &v_off, &vq8 };
            ok = ok && launch(g, g->sw->f_store_seq, (units + 63) / 64, tn, 1, 64, ps);
        }
        }
        if (g->sw->qn[l]) ok = ok && enc_qknorm(g, m, g->q,  g->sw->qn[l], m->n_head, hd, tn, q_dim);
        ok = ok && enc_rope_batch(B, m, g->q, m->n_head, tn, q_dim, l);
        {   // flash-decoding over N sequences: grid (head, split, sequence).
            // The extra grid dimension is free occupancy — a solo decode step
            // leaves most of the GPU idle, which is the headroom batching eats.
            attn_args aa = { hd, m->n_head, n_kv, m->n_ctx,
                             (uint64_t)model_k_byte_off(m, l),
                             model_attn_scale(m, l), q_dim, xdim,
                             local ? m->swa_window : 0, model_kv_kind_k(m),
                             model_kv_is_ring(m, l) ? m->kv_ring : 0,
                             (uint64_t)model_v_byte_off(m, l), model_kv_kind_v(m) };
            CUdeviceptr sinks = g->sw->sinks[l];
            void *pm[] = { &g->xb2, &g->attn_part, &aa, &B->pos_d, &sinks };
            if (batch_attn_gqa(g, m, hd, n_kv) > 0) {
                // the grouped decode attention, as tile_attn runs it alone
                void *pg[] = { &g->q, &B->kcp_d, &B->vcp_d, &g->attn_part, &aa, &B->pos_d };
                ok = ok && launch_smem(g, g->sw->f_attn_dec_gqa_seq, (unsigned)n_kv, 32,
                                       (unsigned)tn, 256, attn_dec_gqa_smem(m, hd, n_kv), pg);
                ok = ok && launch(g, g->sw->f_attn_merge32_seq, m->n_head, tn, 1, 128, pm);
            } else {
                void *pd[] = { &g->q, &B->kcp_d, &B->vcp_d, &g->att, &g->attn_part,
                               &aa, &B->pos_d };
                ok = ok && launch(g, g->sw->f_attn_dec_seq, m->n_head, ATTN_SPLITS, tn, 128, pd);
                // k_attn_merge reads neither position nor cache, so the
                // unbatched kernel serves here unchanged
                ok = ok && launch(g, g->sw->f_attn_merge, m->n_head, tn, 1, 128, pm);
            }
        }

        ok = ok && enc_mv_batch(g, m, ly->wo, g->xb2, g->xb, q_dim, n_embd, g->sw->bo[l], tn, xdim, xdim);
        if (g->sw->pan[l])
            ok = ok && enc_rmsnorm(g, g->xb, g->xb, g->sw->pan[l], n_embd, m->post_norm_eps, tn, xdim, xdim);
        if (m->resid_scale != 1.0f)
            ok = ok && enc_scale(g, g->xb, m->resid_scale, n_embd, tn, xdim);
        ok = ok && enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);

        ok = ok && enc_rmsnorm(g, g->x, g->xb, g->sw->ffn_norm[l], n_embd, m->rms_eps, tn, n_embd, xdim);
        ok = ok && enc_mv_batch(g, m, ly->w_gate, g->xb, g->hb,  n_embd, ly->n_ff, 0, tn, xdim, m->n_ff);
        ok = ok && enc_mv_batch(g, m, ly->w_up,   g->xb, g->hb2, n_embd, ly->n_ff, 0, tn, xdim, m->n_ff);
        ok = ok && enc_actmul(g, m, g->hb, g->hb2, tn * m->n_ff);
        ok = ok && enc_mv_batch(g, m, ly->w_down, g->hb, g->xb, ly->n_ff, n_embd, 0, tn, m->n_ff, xdim);
        if (g->sw->pfn[l])
            ok = ok && enc_rmsnorm(g, g->xb, g->xb, g->sw->pfn[l], n_embd, m->post_norm_eps, tn, xdim, xdim);
        if (m->resid_scale != 1.0f)
            ok = ok && enc_scale(g, g->xb, m->resid_scale, n_embd, tn, xdim);
        ok = ok && enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);
        ok = ok && enc_ple(g, m, ly, l, tn, xdim);
        if (ly->out_scale != 1.0f && ly->out_scale != 0.0f)
            ok = ok && enc_scale(g, g->x, ly->out_scale, n_embd, tn, n_embd);
    }

    ok = ok && enc_rmsnorm(g, g->x, g->xb, g->sw->out_norm, n_embd, m->rms_eps,
                           tn, n_embd, xdim);
    ok = ok && enc_mv_batch(g, m, m->output, g->xb, B->logits_n, n_embd,
                            B->n_vocab, 0, tn, xdim, B->n_vocab);
    return ok;
}

// Stage the microbatch's token embeddings: column i is sequence idx[i]'s token.
// Identical arithmetic to stage_x, just gathering from N different callers.
static bool stage_x_batch(gpu_batch *B, model_t *m, const int32_t *tok, int n) {
    gpu_t *g = B->lead;
    size_t ers = ggml_row_size(m->tok_embd->type, m->n_embd);
    for (int b = 0; b < n; b++) {
        float *hx = g->h_x + (size_t)b * m->n_embd;
        dequant_row(m->tok_embd->type,
                    (uint8_t *)m->tok_embd->data + (size_t)tok[b] * ers,
                    hx, m->n_embd);
        model_embd_transform(m, hx);
    }
    if (cu.MemcpyHtoD(g->x, g->h_x, sizeof(float) * (size_t)n * m->n_embd) != 0)
        return false;
    return stage_ple(g, m, tok, n);
}

bool gpu_batch_decode(gpu_batch *b, const int *idx, const int32_t *tok,
                      const int *pos, int n, float **out) {
    if (!b || n < 1 || n > BATCH_MAX_COLS) return false;
    // idx[] is caller-supplied and selects members of b->seqs[0, b->n): a bogus
    // index would dereference a wild model_t* below. Reject the whole batch
    // rather than launch against garbage — the caller then decodes sequentially.
    for (int i = 0; i < n; i++)
        if (idx[i] < 0 || idx[i] >= b->n) return false;
    // NoPE layers, and mistral3's every layer, need a per-token Q scale
    // derived from each column's position, and this path keeps positions on
    // the device. Decline rather than scale with the wrong factor; the caller
    // decodes sequentially, which is the same retreat it already takes for a
    // bad index.
    if (b->n > 0 && b->seqs[idx[0]] &&
        (b->seqs[idx[0]]->no_rope_layer_step > 0 ||
         b->seqs[idx[0]]->nope_on_full ||
         b->seqs[idx[0]]->attn_temp_scale != 0.0f))
        return false;
    // A microbatch of one is just a decode step with extra staging, and the
    // solo path is better at it: narrower matvec kernels, x straight from
    // global, and a graph that is already warm. Hand it over rather than
    // charge a lightly loaded server for machinery it is not using.
    if (n == 1) {
        model_t *m = b->seqs[idx[0]];
        float *lg = NULL;
        if (!gpu_forward_batch(m, tok, 1, pos[0], true, &lg) || !lg) return false;
        out[0] = lg;
        return true;
    }
    gpu_t *g = b->lead;
    model_t *m0 = b->seqs[0];
    // The lead supplies the scratch and the graphs. If it fell back to the CPU
    // since this batch was created its buffers are gone, and the batch has to
    // decline rather than launch against freed memory — the caller then decodes
    // sequentially, which is where that sequence already is.
    if (m0->gpu != g) return false;
    if (cu.CtxSetCurrent(b->sw->ctx) != 0) return false;

    // Bring each sequence's device KV in line with its host cache. This is the
    // same contract gpu_forward_batch keeps — the host copy is authoritative —
    // applied per member: a sequence that decoded its previous token in this
    // same slot is already contiguous and costs nothing, while one that was
    // just prefilled on the CPU, rewound, or swapped in resyncs [0, pos).
    for (int i = 0; i < n; i++) {
        model_t *m = b->seqs[idx[i]];
        gpu_t   *gi = m->gpu;
        if (!gi || gi->sw != b->sw || m->gpu_layers < m->n_layer) return false;
        if (pos[i] < 0 || pos[i] >= m->n_ctx) return false;
        if (pos[i] > gi->kv_rows && !kv_upload(gi, m, gi->kv_rows, pos[i])) return false;
        b->h_pos[i] = pos[i];
        b->h_kcp[i] = (uint64_t)gi->kc;
        b->h_vcp[i] = (uint64_t)gi->vc;
    }
    // pad the unused columns with column 0's sequence: the kernels compute
    // them (grid dimensions are captured into the graph) and the results are
    // discarded, but they must address real memory
    for (int i = n; i < MVB; i++) {
        b->h_pos[i] = b->h_pos[0];
        b->h_kcp[i] = b->h_kcp[0];
        b->h_vcp[i] = b->h_vcp[0];
    }

    if (!stage_x_batch(b, m0, tok, n) ||
        cu.MemcpyHtoD(b->pos_d, b->h_pos, sizeof(int) * MVB) != 0 ||
        cu.MemcpyHtoD(b->kcp_d, b->h_kcp, sizeof(uint64_t) * MVB) != 0 ||
        cu.MemcpyHtoD(b->vcp_d, b->h_vcp, sizeof(uint64_t) * MVB) != 0)
        return false;

    // Capture once per microbatch width. Launch overhead is the thing batching
    // is fighting — a solo decode step is already graph-captured, so a batched
    // step issuing ~500 plain launches would hand back much of what it won.
    bool ran = false;
    if (b->graph_bad) graph_eager_note("batch", b->graph_why, &b->graph_why_said);
    if (!b->graph_bad && !prof.on) {   // see the note on the solo path above
        if (!b->gexec[n]) {
            prof.capturing = 1;
            CUresult r = 0;
            const char *stage = NULL;
            if ((r = capture_begin(g->stream)) != 0) stage = "the start";
            else if (!fwd_batch(b, m0, n)) stage = "the forward";
            else if ((r = capture_end(g->stream, &b->graph[n])) != 0) stage = "the end";
            else if ((r = cu.GraphInstantiateWithFlags(&b->gexec[n], b->graph[n], 0)) != 0)
                stage = "instantiation";
            if (stage) {
                CUgraph junk = NULL;
                CUresult re = capture_end(g->stream, &junk);
                if (junk && junk != b->graph[n]) cu.GraphDestroy(junk);
                b->graph_bad = true;
                graph_capture_failed("batch", stage, r ? r : re);
            }
            prof.capturing = 0;
        }
        if (b->gexec[n])
            ran = cu.GraphLaunch(b->gexec[n], g->stream) == 0 &&
                  cu.StreamSynchronize(g->stream) == 0;
        if (!ran && !b->graph_bad) {
            b->graph_bad = true;
            graph_launch_failed("batch");
        }
    }
    if (!ran) {
        if (!fwd_batch(b, m0, n)) {
            fprintf(stderr, "gpu: batched decode failed — falling back\n");
            return false;
        }
        if (cu.CtxSynchronize() != 0) return false;
    }

    // Each sequence's newly written KV row goes back to its own host cache, so
    // any member can leave the batch and continue on the CPU or in a solo
    // forward without noticing it was ever batched. Through each member's
    // gather (R3.4.1): its row gathered on the device, one wait for all of
    // them, one copy back each, instead of two copies per layer per member.
    bool gathered = false;
    for (int i = 0; i < n; i++) {
        model_t *m = b->seqs[idx[i]];
        if (kv_gather_rows_ok(m->gpu, 1, false)) {
            if (!kv_gather_rows_launch(m->gpu, m, pos[i], 1)) return false;
            gathered = true;
        }
    }
    if (gathered && cu.CtxSynchronize() != 0) return false;
    for (int i = 0; i < n; i++) {
        model_t *m = b->seqs[idx[i]];
        gpu_t   *gi = m->gpu;
        if (kv_gather_rows_ok(gi, 1, false) ? !kv_gather_finish(gi, m, pos[i], 1)
                                            : !kv_copyback(gi, m, pos[i], pos[i] + 1))
            return false;
        gi->kv_rows = pos[i] + 1;
    }
    if (cu.MemcpyDtoH(b->h_logits, b->logits_n,
                      sizeof(float) * (size_t)n * b->n_vocab) != 0)
        return false;
    for (int i = 0; i < n; i++)
        out[i] = b->h_logits + (size_t)i * b->n_vocab;
    return true;
}

// Fault-injection hook, debug builds and torture runs only.
// RUNNER_CUDA_INJECT_FAILURE=N fails the Nth device forward and returns its
// ordinal; unset or 0 disables it, and a non-numeric value means 1.
// N > 1 is the case worth having: the first device forward is the prefill
// tile, and failing at decode is what exercises the rollback (the recurrent
// snapshot, a device KV the host must resync).
// Counted across every slot; the budget is taken with a compare-and-swap so
// two slots never both spend its last failure.
static int inject_fail_ordinal(void) {
    static _Atomic int budget = -1;  // -1 unread, then the calls left until the failure
    static _Atomic int seen = 0;
    if (budget < 0) {
        const char *s = getenv("RUNNER_CUDA_INJECT_FAILURE");
        int v = s ? (atoi(s) > 0 ? atoi(s) : 1) : 0;
        atomic_compare_exchange_strong(&budget, &(int){ -1 }, v);
    }
    for (int left = budget; left > 0; ) {
        if (atomic_compare_exchange_weak(&budget, &left, left - 1)) {
            int n = ++seen;
            return left == 1 ? n : 0;
        }
    }
    return 0;
}

// The device copy of a tile's positions: its first row's, then the start of
// every later ATT_COLS-column attention sub-tile. Every writer of pos_dev
// goes through here so a sub-tile can never read a stale start.
static bool set_pos(gpu_t *g, int pos, int tn) {
    g->route_pos = pos;
    int hp[GPU_WINDOW_MAX / ATT_COLS];
    int n = (tn + ATT_COLS - 1) / ATT_COLS;
    if (n < 1) n = 1;
    if (n > (g->win + ATT_COLS - 1) / ATT_COLS) return false;
    for (int j = 0; j < n; j++) hp[j] = pos + j * ATT_COLS;
    return cu.MemcpyHtoD(g->pos_dev, hp, sizeof(int) * (size_t)n) == 0;
}

// Whether an n-row keep-forward on a Mamba-2 hybrid files its rows: rows
// 0 .. n-2 write their SSD state into per-row slots and the last row the
// live half, so a partially accepted round restores the state after its last
// kept row (gpu_recurrent_restore_row), not by rolling back and re-folding.
// The slots are allocated by the first verify round, so plain decoding never
// holds them; a failed allocation leaves the re-fold.
static bool mamba_snap_ready(gpu_t *g, model_t *m, int n) {
    int rows = (m->spec_batch < 8 ? m->spec_batch : 8) - 1;
    if (n < 2 || n - 1 > rows || g->snap_failed || m->gpu_layers < m->n_layer ||
        !cu.MemcpyDtoDAsync)
        return false;
    if (g->snap_rows) return true;
    int cd = m->ssm.inner + 2 * m->ssm.groups * m->ssm.state;
    size_t sl = (size_t)m->ssm.inner * m->ssm.state;
    size_t hl = (size_t)(m->ssm.conv_kernel - 1) * cd;
    int nrec = 0;
    g->snap_ord = malloc(sizeof(int) * (size_t)m->n_layer);
    if (g->snap_ord)
        for (int l = 0; l < m->n_layer; l++)
            g->snap_ord[l] = m->layers[l].recurrent ? nrec++ : -1;
    size_t slots = (size_t)rows * nrec;
    if ((size_t)nrec * sl > INT_MAX || (size_t)nrec * hl > INT_MAX) {
        // the tile kernels take a row's stride (nrec slots) as an int
        free(g->snap_ord);
        g->snap_ord = NULL;
        g->snap_failed = true;
        fprintf(stderr, "gpu: a verify row's recurrent state passes 2^31 floats; a "
                "partially accepted speculative round re-folds its kept rows\n");
        return false;
    }
    if (!g->snap_ord || !nrec ||
        cu.MemAlloc(&g->snap_state, sizeof(float) * sl * slots) != 0 ||
        cu.MemAlloc(&g->snap_hist, sizeof(float) * (hl ? hl : 1) * slots) != 0) {
        if (g->snap_state) cu.MemFree(g->snap_state);
        free(g->snap_ord);
        g->snap_state = 0;
        g->snap_ord = NULL;
        g->snap_failed = true;
        fprintf(stderr, "gpu: no room for the verify checkpoints (%.0f MB); a partially "
                "accepted speculative round re-folds its kept rows\n",
                (double)sizeof(float) * (sl + hl) * (double)slots / 1e6);
        return false;
    }
    g->snap_nrec = nrec;
    g->snap_rows = rows;
    return true;
}

// The recurrent state reset at a new sequence and snapshot before every
// forward (the rollback a failed or rejected forward restores), the
// keep-forward's checkpoint flags.
static bool cuda_fwd_prepare(gpu_t *g, model_t *m, int n, int pos) {
    if (cu.CtxSetCurrent(g->sw->ctx) != 0) return false;
    g->xh_src = 0;
    if (m->qwen35 && pos == 0) {
        int convdim = 2 * m->ssm.state * m->ssm.groups + m->ssm.inner;
        size_t hist_elems = model_rec_count(m) * (m->ssm.conv_kernel - 1) * convdim;
        size_t state_elems = model_rec_count(m) * m->ssm.v_heads *
                             m->ssm.state * m->ssm.state;
        if ((hist_elems && cu.MemsetD8(g->q35_hist, 0,
                                      hist_elems * sizeof(float)) != 0) ||
            cu.MemsetD8(g->q35_state, 0, state_elems * sizeof(float)) != 0)
            return false;
    }
    if (m->qwen35) {
        int convdim = 2 * m->ssm.state * m->ssm.groups + m->ssm.inner;
        size_t hist_bytes = sizeof(float) * model_rec_count(m) *
                            (m->ssm.conv_kernel - 1) * convdim;
        size_t state_bytes = sizeof(float) * model_rec_count(m) *
                             m->ssm.v_heads * m->ssm.state * m->ssm.state;
        if ((hist_bytes && cu.MemcpyDtoD(g->q35_hist_prev, g->q35_hist,
                                        hist_bytes) != 0) ||
            cu.MemcpyDtoD(g->q35_state_prev, g->q35_state, state_bytes) != 0)
            return false;
    }
    g->fwd_count++;
    // a keep-forward (speculative verify) files per-row checkpoints
    g->snap_on = m->spec_want_all >= n &&
                 (m->qwen35 ? g->snap_rows > 0 && n <= g->snap_rows
                            : (m->granite_hybrid || m->nemotron_h) && g->mamba_state &&
                              mamba_snap_ready(g, m, n));
    g->snap_valid = 0;
    if ((m->granite_hybrid || m->nemotron_h) && g->mamba_state) {
        int cd = m->ssm.inner + 2 * m->ssm.groups * m->ssm.state;
        size_t conv_bytes = sizeof(float) * model_rec_count(m) *
                            (m->ssm.conv_kernel - 1) * cd;
        size_t state_bytes = sizeof(float) * model_rec_count(m) *
                             m->ssm.inner * m->ssm.state;
        int src = g->rec_cur;
        if (pos == 0) {
            if ((conv_bytes && cu.MemsetD8(g->mamba_conv, 0, conv_bytes) != 0) ||
                cu.MemsetD8(mamba_half(g, m, src), 0, state_bytes) != 0) return false;
        }
        if (conv_bytes && cu.MemcpyDtoD(g->mamba_conv_prev, g->mamba_conv,
                                        conv_bytes) != 0)
            return false;
        // the state is double-buffered instead of copied: read src, write
        // the other half (see gpu_t.rec_cur)
        g->rec_prev = src;
        g->rec_cur = 1 - src;
        g->rec_tile0 = true;
    }
    return true;
}

// A single-token decode through the captured graph (capturing it on first
// use): 1 served, 0 not served (the plain path runs), -1 failed.
static int cuda_fwd_graph_decode(gpu_t *g, model_t *m, const int32_t *tokens,
                                 int pos, float **logits, int m_, double t_fwd0) {
    double ts0 = prof.on ? prof_now() : 0;
    if (!stage_x(g, m, tokens, 1) || !set_pos(g, pos, 1)) return -1;
    if (prof.on) prof.t_stage[m_] += prof_now() - ts0;
    // a double-buffered Mamba-2 state alternates halves every forward: one
    // graph per parity (the half this forward writes)
    int gi = (m->granite_hybrid || m->nemotron_h) && g->mamba_state ? g->rec_cur : 0;
    // The attention temperature is a host value the graph holds:
    // enc_rope_or_temp computes it from pos and passes it to k_scale. It
    // steps every floor_scale positions, so a graph captured before a step
    // would replay the old factor: capture again when the factor moves.
    float temp = model_attn_temp(m, pos);
    if (g->gexec[gi] && g->graph_temp[gi] != temp) {
        cu.GraphExecDestroy(g->gexec[gi]);
        cu.GraphDestroy(g->graph[gi]);
        g->gexec[gi] = NULL;
        g->graph[gi] = NULL;
    }
    if (!g->gexec[gi]) {
        g->graph_temp[gi] = temp;
        prof.capturing = 1;
        // capture records the launch sequence without executing it; the
        // recorded graph is position-invariant because pos lives in device
        // memory, so one capture serves every decode token.
        // THREAD_LOCAL (1), not GLOBAL (0): GLOBAL forbids synchronous CUDA
        // calls from every other thread for the capture window, so another
        // slot's forward would fail and drop that slot to the CPU.
        // capture_begin's exclusion is the other half (see above).
        CUresult r = 0;
        const char *stage = NULL;
        if ((r = capture_begin(g->stream)) != 0) stage = "the start";
        else if (!fwd_tile(g, m, tokens, 1, pos, true, 0, m->gpu_layers)) stage = "the forward";
        else if (g->kvg_dev && !kv_gather_launch(g, m)) stage = "the KV gather";
        else if ((r = capture_end(g->stream, &g->graph[gi])) != 0) stage = "the end";
        else if ((r = cu.GraphInstantiateWithFlags(&g->gexec[gi], g->graph[gi], 0)) != 0)
            stage = "instantiation";
        if (stage) {
            CUgraph junk = NULL;
            CUresult re = capture_end(g->stream, &junk); // ensure capture ended
            if (junk && junk != g->graph[gi]) cu.GraphDestroy(junk);
            g->graph_bad = true;
            graph_capture_failed("decode", stage, r ? r : re);
        }
        prof.capturing = 0;
    }
    if (g->gexec[gi]) {
        double tg0 = prof.on ? prof_now() : 0;
        bool gl = cu.GraphLaunch(g->gexec[gi], g->stream) == 0 &&
                  cu.StreamSynchronize(g->stream) == 0;
        if (prof.on) prof.t_sync[m_] += prof_now() - tg0;
        if (gl) {
            if (!mtp_save_rows(g, m, 0, 1)) return -1;
            double tc0 = prof.on ? prof_now() : 0;
            if (g->kvg_dev ? !kv_gather_finish(g, m, pos, 1) : !kv_copyback(g, m, pos, pos + 1))
                return -1;
            if (prof.on) prof.t_copyback[m_] += prof_now() - tc0;
            g->kv_rows = pos + 1;
            double tl0 = prof.on ? prof_now() : 0;
            if (cu.MemcpyDtoH(g->h_logits, g->logits, sizeof(float) * m->n_vocab) != 0)
                return -1;
            if (prof.on) {
                prof.t_logitscp[m_] += prof_now() - tl0;
                prof.n_tokens[m_] += 1; prof.n_tiles[m_] += 1;
                prof.wall_ms[m_] += prof_now() - t_fwd0;
                prof_flush();
            }
            // Steady-state decode returns here, so the hook has to be
            // checked on this path too — otherwise no injected failure can
            // ever land after prefill, which is the only interesting place
            // for one to land.
            int ginj = inject_fail_ordinal();
            if (ginj) {
                fprintf(stderr, "gpu: injected CUDA runtime failure at "
                                "device forward %d — falling back to CPU\n",
                        ginj);
                return -1;
            }
            if (logits) *logits = g->h_logits;
            return 1;
        }
    }
    if (!g->graph_bad) { g->graph_bad = true; graph_launch_failed("decode"); }
    return 0;
}

// A verify round's rows (tile_attn_rows): position pos + c and the
// instance's own cache for row c; the unused entries repeat row 0's.
static bool set_rows(gpu_t *g, int pos, int tn) {
    g->route_pos = pos;
    int hp[MVB];
    uint64_t hk[MVB], hv[MVB];
    for (int c = 0; c < MVB; c++) {
        hp[c] = pos + (c < tn ? c : 0);
        hk[c] = (uint64_t)g->kc;
        hv[c] = (uint64_t)g->vc;
    }
    return cu.MemcpyHtoD(g->vcol_pos, hp, sizeof hp) == 0 &&
           cu.MemcpyHtoD(g->vcol_kcp, hk, sizeof hk) == 0 &&
           cu.MemcpyHtoD(g->vcol_vcp, hv, sizeof hv) == 0;
}

static bool cuda_fwd_tiles_run(gpu_t *g, model_t *m, const int32_t *tokens, int n,
                               int pos, bool want_logits, bool partial, int m_);

// The batch in tiles of the instance's prompt window (model_gpu_window):
// stage, run the device layers, hand a partial split's boundary activation
// to the host.
static bool cuda_fwd_tiles(gpu_t *g, model_t *m, const int32_t *tokens, int n,
                           int pos, bool want_logits, bool partial, int m_) {
    // a speculative verify round (model_forward_verify) or the walk's re-fold
    // of its kept rows (model_forward_refold) in one tile: its rows in the
    // one-token decode arithmetic (enc_mv, tile_attn_rows, the MoE dispatch),
    // so each row's bits are a solo step's at its position
    g->exact_rows = m->spec_exact && n > 1 && n <= g->win && n <= MVB;
    bool ok = cuda_fwd_tiles_run(g, m, tokens, n, pos, want_logits, partial, m_);
    g->exact_rows = false;
    return ok;
}

static bool cuda_fwd_tiles_run(gpu_t *g, model_t *m, const int32_t *tokens, int n,
                               int pos, bool want_logits, bool partial, int m_) {
    for (int i = 0; i < n; i += g->win) {
        int tn = n - i < g->win ? n - i : g->win;
        bool last = i + tn == n;
        int p = pos + i;
        // device runs layers [0, gpu_layers); logits only when it owns the whole
        // model (partial hands the boundary activation back to the CPU)
        double ts0 = prof.on ? prof_now() : 0;
        bool sok = stage_x(g, m, tokens + i, tn) && set_pos(g, p, tn) &&
                   (!g->exact_rows || set_rows(g, p, tn));
        if (sok && g->exact_rows) g_exact_verify_tiles++;
        if (prof.on) { prof.t_stage[m_] += prof_now() - ts0; prof.n_tiles[m_] += 1;
                       prof.n_tokens[m_] += tn; }
        if (!sok ||
            !fwd_tile(g, m, tokens + i, tn, p,
                      want_logits && last && !partial, 0, m->gpu_layers)) {
            fprintf(stderr, "gpu: kernel launch failed — falling back to CPU\n");
            return false;
        }
        g->rec_tile0 = false;
        if (!mtp_save_rows(g, m, i, tn)) return false;
        // one tile when the gather takes the rows (n <= KVG_ROWS <= win)
        if (kv_gather_rows_ok(g, n, partial) &&
            !(n == 1 ? kv_gather_launch(g, m) : kv_gather_rows_launch(g, m, p, n)))
            return false;
        int inj = inject_fail_ordinal();
        if (inj) {
            // Deterministic ownership/state-machine hook: fail after recurrent
            // state changed, so the caller must restore the pre-forward device
            // snapshot before recomputing this same tile on the CPU.
            fprintf(stderr, "gpu: injected CUDA runtime failure at device "
                            "forward %d — falling back to CPU\n", inj);
            return false;
        }
        // partial: copy this tile's post-boundary activation to the host
        // buffer so the CPU layer loop can continue from gpu_layers. Under
        // hyper-connections the residual is the wide x_hc (hc * n_embd per
        // row), not x: handing x over leaves the CPU layers a stale stream.
        if (partial) {
            if (m->hyper_conn) {
                size_t hcd = (size_t)m->hc_count * m->n_embd;
                if (cu.MemcpyDtoH(m->x_hc + (size_t)i * hcd, g->x_hc,
                                  sizeof(float) * tn * hcd) != 0)
                    return false;
            } else if (cu.MemcpyDtoH((uint8_t *)m->x + (size_t)i * m->n_embd * sizeof(float),
                                     g->x, sizeof(float) * tn * m->n_embd) != 0) {
                return false;
            }
        } else if (m->want_hidden && !m->hyper_conn) {
            // model_hidden_states: every row's final residual, the tile's
            // kernels finished first (as mtp_save_rows waits for them)
            if (cu.StreamSynchronize(g->stream) != 0 ||
                cu.MemcpyDtoH((uint8_t *)m->x + (size_t)i * m->n_embd * sizeof(float),
                              g->x, sizeof(float) * tn * m->n_embd) != 0)
                return false;
        }
    }
    return true;
}

// The forward synchronized, the KV rows copied back (the host cache stays
// authoritative), the logits read.
static bool cuda_fwd_finish(gpu_t *g, model_t *m, int n, int pos, bool want_logits,
                            bool partial, float **logits, int m_, double t_fwd0) {
    double tsy0 = prof.on ? prof_now() : 0;
    if (cu.CtxSynchronize() != 0) {
        fprintf(stderr, "gpu: forward failed — falling back to CPU\n");
        return false;
    }
    if (prof.on) { prof.t_sync[m_] += prof_now() - tsy0; prof_flush(); }
    route_replay_flush(g);

    if (g->snap_on) { g->snap_valid = n; g->snap_on = 0; }
    // keep the host KV cache authoritative: copy the offloaded layers' rows
    // back so the CPU path can take over at any point
    double tc0 = prof.on ? prof_now() : 0;
    if (kv_gather_rows_ok(g, n, partial) ? !kv_gather_finish(g, m, pos, n)
                                         : !kv_copyback(g, m, pos, pos + n))
        return false;
    if (prof.on) prof.t_copyback[m_] += prof_now() - tc0;
    g->kv_rows = pos + n;

    if (want_logits && !partial) {
        double tl0 = prof.on ? prof_now() : 0;
        if (cu.MemcpyDtoH(g->h_logits, g->logits, sizeof(float) * m->n_vocab) != 0)
            return false;
        if (prof.on) prof.t_logitscp[m_] += prof_now() - tl0;
        if (logits) *logits = g->h_logits;
    } else if (m->spec_want_all >= n && n <= m->spec_batch && !partial) {
        // the verify walk's rows, read lazily by gpu_spec_logits
        if (cu.MemcpyDtoH(g->h_logits, g->logits, sizeof(float) * (size_t)n * m->n_vocab) != 0)
            return false;
    }
    if (prof.on) prof.wall_ms[m_] += prof_now() - t_fwd0;
    return true;
}

bool gpu_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                       bool want_logits, float **logits) {
    gpu_t *g = m->gpu;
    if (logits) *logits = NULL;
    if (n < 1) return true;
    // A prior recoverable GPU failure runs gpu_disable(), which frees the
    // context and clears m->gpu. Decline rather than deref a NULL g (the
    // batch path re-checks m0->gpu for the same reason).
    if (!g) return false;

    if (!cuda_fwd_prepare(g, m, n, pos)) return false;
    if (prof.on && !prof.inited) prof_init();
    prof.cur_mode = (n == 1) ? M_SINGLE : M_BATCH;
    int m_ = prof.cur_mode;
    double t_fwd0 = prof.on ? prof_now() : 0;

    // the rows this forward reads that the device does not hold as the host
    // does (rows past its last forward, a snapshot the host installed):
    // [kv_rows, pos).
    // A position at or below kv_rows needs none: the rows under it are the
    // host's, and the forward writes its own.
    if (pos > g->kv_rows && !kv_upload(g, m, g->kv_rows, pos)) return false;

    bool partial = m->gpu_layers < m->n_layer;
    // !prof.on: a replayed CUDA graph never re-enters the instrumented launch
    // path, so its phase counters stay 0: profiling runs decode eagerly. The
    // absolute decode wall is then pessimistic; the phase split is what the
    // profile is for.
    // !snap_on: a keep-forward's per-row checkpoints are copies the decode
    // graph does not hold
    if (n == 1 && want_logits && !partial && !g->graph_bad && g->stream &&
        !g->snap_on && !prof.on && !getenv("RUNNER_CUDA_GRAPH_OFF")) {
        int served = cuda_fwd_graph_decode(g, m, tokens, pos, logits, m_, t_fwd0);
        if (served) return served > 0;
    } else if (n == 1 && want_logits && !partial && graph_capture_required()) {
        // eager by design: say why once (graph_why); a graph_bad no rule
        // named was set by a failure, which ended the process under the switch
        const char *why = g->graph_bad ? g->graph_why
                        : !g->stream ? "no stream to capture on (or the driver has no graph API)"
                        : g->snap_on ? "a keep-forward's per-row checkpoints"
                        : prof.on ? "the profiler runs decode eagerly"
                        : "RUNNER_CUDA_GRAPH_OFF";
        graph_eager_note("decode", why, &g->graph_why_said);
    }

    if (!cuda_fwd_tiles(g, m, tokens, n, pos, want_logits, partial, m_)) return false;
    return cuda_fwd_finish(g, m, n, pos, want_logits, partial, logits, m_, t_fwd0);
}
