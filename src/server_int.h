// Shared server state: what the slot workers, the scheduler, the residency
// layer and the routes all touch.
//
// This header does not make SV less of a global -- it makes it a NAMED global
// with a declared type, so the modules split out of server.c under RNR-019 can
// be separate translation units at all. Turning SV into a context passed down
// from server_run is the remaining half of that finding and a real behavioural
// change, not a move; it is deliberately not attempted here.
#ifndef RUNNER_SERVER_INT_H
#define RUNNER_SERVER_INT_H

#include <pthread.h>
#include <stdatomic.h>

#include "runner.h"
#include "http.h"
#include "oms.h"
#include "watermark.h"

typedef struct slot_s slot_t;
const char *slot_default_system(const slot_t *s, bool tools);
// When a server stop has been requested: the moment active requests stop
// being waited for (the first stop request plus the drain grace), else 0.
// Read by each request's stop predicate, on the same clock as now_s().
double server_drain_deadline(void);
extern const char *server_template_text_override;

struct slot_s {
    model_t   *m;         // slot 0 borrows the preloaded model
    tokenizer *tok;       // shared (read-only)
    sampler    smp;
    sampler    smp_base;  // pristine server defaults (per-request resets)
    engine     e;
    int        id;
    int        tmpl;
    pthread_t  th;
    // Stage timing for the request this slot is serving: when the worker
    // took it off the accept queue (its clock start) and how long it had
    // waited there. Zero for a request the accept loop answered itself.
    double     req_t0, queue_wait_s;
    // R8.6: the per-request adapter this slot's model borrows now (index
    // into SV.adapters), -1 for none
    int        adapter;
    // R1.8.1: this slot's watermark hook state (the key is SV.wm_key)
    wm_state   wm;
    // R3.7.4: the current request's draft hints, tokenized (runs separated
    // by -1). Owned by the slot and reused, so no request path frees it.
    int32_t   *hint_buf;
    int        hint_cap;
    // R10.12.7: what this slot is doing, for /health. Written by the slot's
    // worker, read by the accept thread, so every field is atomic. rq_phase:
    // 0 idle, 1 prefill, 2 generating, 3 loading its model (R10.12.3.3). rq_done counts prompt tokens that are
    // in the KV (reused ones included), rq_gen tokens generated.
    atomic_int rq_phase, rq_prompt, rq_done, rq_gen;
    // R10.12.3.2: the request a cancel can name, by its response id or the
    // client's X-Request-Id. rq_id, rq_cid and rq_live are guarded by the
    // server's request-id lock; cid_next is the header of the request this
    // slot is reading. rq_cancel is set by a cancel and read by the
    // request's stop check. rq_ticket (R6.68.8), under the same lock, is
    // what the cancels of the live request wait on: its end is recorded
    // there, with its own token count, and not on the slot, where the next
    // request would overwrite it.
    char        rq_id[48], rq_cid[65], cid_next[65];
    bool        rq_live;
    atomic_int  rq_cancel;
    struct cancel_ticket *rq_ticket;
};

// A per-request adapter (R8.6): loaded once at startup, borrowed by a slot's
// model for the requests that name it as "<model>:<name>".
#define RUNNER_MAX_ADAPTERS 16
typedef struct {
    char      name[64];
    char      path[1024];
    char      sha256[65];
    char      sig_json[512];   // load-time OMS verdict, "" = none (R1.2.3)
    lora_set *set;
} adapter_entry;

typedef struct {
    sock_t fds[512];
    double at[512];   // when each fd was queued (now_s), for the queue wait
    int  head, tail, count, limit;
    bool shutdown;
    pthread_mutex_t mu;
    pthread_cond_t cv;
} fdqueue;

// model registry for swap mode (-m "name=path,name2=path2"): one model
// resident at a time (llama-swap semantics), loaded on demand, unloaded
// after --ttl idle seconds
typedef struct {
    // A basename is up to 255 bytes on every real filesystem, and single-model
    // serve copies the model file's basename in here (the parallel==1 join
    // below), so anything shorter TRUNCATES the served model id rather than
    // refusing it. 64 did exactly that to a 68-character checkpoint name on
    // the lab box (2026-09-24): the id came back cut at 63 characters, which
    // happened to land where ".gguf" began, and every client that named the
    // file got a 404 from a server whose /v1/models listed a name no file
    // had. An operator-chosen registry name is still refused when it does not
    // fit, with the limit named.
    char name[256];
    char path[1024];
    int  tmpl;
} reg_entry;

typedef struct {
    slot_t    *slots;
    int        n_slots;
    fdqueue    q;
    const char *model_name;
    int        n_predict_cap;
    // R4.12.16: default reasoning budget in tokens (0 = off), from
    // --reasoning-budget. A request's reasoning_max_tokens overrides it.
    int        reasoning_budget;
    // --reasoning-budget-message: the server's transition sentence, NULL for
    // the built-in default. A request's reasoning_budget_message wins.
    const char *reasoning_budget_message;
    // --reasoning-temp: the server's default reasoning-channel temperature.
    // The BOOL is load-bearing. SV is zero-initialised in every context that
    // does not run server_run -- the socketpair unit tests among them -- and
    // a float sentinel read 0.0 as "on at temperature zero", which switched
    // the feature on for every request and then refused models that declare
    // no reasoning channel (CI caught it, 2026-09-26). 0 can mean off for
    // reasoning_budget because it is a count; a temperature of 0 is a
    // legitimate setting, so it needs its own flag.
    // --loop-guard: the server's default for the loop guard (off).
    bool       loop_guard;
    bool       reasoning_temp_set;
    float      reasoning_temp;
    atomic_int ctx_size;
    // A request's id is 128 bits of SHA-256(id_seed || req_counter): see request_id.
    atomic_ullong req_counter;
    uint8_t id_seed[32];
    // swap mode
    reg_entry   reg[RUNNER_MAX_MODELS];
    int         n_reg;
    bool        single;        // single-model serve wrapped as a 1-entry registry
    bool        borrowed;      // slot 0's model/tok containers owned by the caller
    atomic_int  resident;      // index into reg, -1 = none
    double      last_used;
    int         ttl;
    model_params mp;
    pthread_mutex_t swap_mu;
    atomic_int  active_requests;
    // Inference requests admitted since start, the monotonic counterpart of
    // active_requests: incremented and decremented at the same two points, so
    // "how many are running" and "how many have run" cannot describe different
    // sets of requests. The management and telemetry routes are not counted,
    // for the reason /health already documents about itself -- a dashboard
    // polling on a timer must not measure its own polling.
    atomic_ullong total_requests;
    // the server's sampling seed (-s, or a time-derived one), the root of every
    // unseeded request's stream (server_request_stream), and how many requests
    // have started sampling: an unseeded request's place in that order picks
    // its stream, so the serving slot does not, and a request that sampled
    // nothing (refused, greedy) moves no other request's draws
    uint64_t    base_rng;
    atomic_ullong sampled_requests;
    // Loading state machine: swap_to releases swap_mu for the duration of the
    // actual model load (minutes for big files, up to --wait-for-vram longer),
    // so /unload and shutdown are never blocked behind a load. `loading` and
    // `pending_unload` are protected by swap_mu; `load_cancel` is a lock-free
    // atomic flag the vram wait polls (volatile is not cross-thread
    // synchronization — RNR-008).
    bool         loading;         // a swap_to load is in flight, lock released
    bool         pending_unload;  // /unload arrived while busy; honoured at the next safe point
    atomic_int   load_cancel;     // aborts a --wait-for-vram queue wait, and discards the load
    sock_t       load_owner;      // the request a load runs for; the loading thread's alone
    // R10.12.3.3: a cancel by id reached the request while it was loading its
    // model. Set by the cancel, cleared when a request takes or gives back its
    // ids (server.c); the registry's wait only reads it, as it reads
    // load_owner, so the registry never needs the server's code. Registry
    // mode is single-slot, so one request owns the flag at a time.
    atomic_int   load_owner_cancelled;
    atomic_bool shutdown;
    pthread_t   reaper_th;
    bool        reaper_started;
    model_t    *draft;        // per-slot draft would be plural; single-model
    int         draft_k;      // serve has exactly one slot (see swap_to)
    // the --draft argv string, kept only when its startup load succeeded, so
    // a swap_to() after /unload can bring the draft back with the target
    const char *draft_path;
    // RI-2: what the OPERATOR asked for, kept separately from what happened.
    // draft_path above is set only on a successful load, so it cannot answer
    // "was a draft requested and refused" -- which is exactly the question a
    // benchmark harness has to ask before it reports a number as speculative
    // decoding. draft_note names the reason when requested && !active.
    bool        draft_requested;
    // R3.39.4: --draft off, so no request takes a source by default either
    bool        draft_off;
    // Successful single-model startup source, retained for reloads. Live
    // capabilities instead inspect each resident slot's actual engine.
    const char *draft_source;
    const char *draft_note;
    // sampling defaults come from the served model's family preset; the CLI
    // overrides are kept so a swapped-in model can be re-resolved against them
    sampler_override ov;
    const char *preset_name;
    // default wall-clock ceiling per generation, 0 = none (RUNNER_REQUEST_TIMEOUT)
    double      req_timeout;
    bool        ignore_eos;   // process-level --ignore-eos, including reloads
    // --chat-template resolved to a TMPL_* value, or -1 for auto-detection.
    // Read by registry.c on every load, so the override outlives a /unload or
    // a --ttl expiry; main.c refuses the flag with a swap set, so this is only
    // ever the single served model's own template.
    int         tmpl_override;
    // Measured-envelope enforcement for swap-mode loads. A model swapped in on
    // demand whose sidecar records an outside-envelope verdict for this runtime
    // is REFUSED per-request (never a process exit — the server keeps serving
    // its other models), unless force_uncertified was set at startup.
    bool        force_uncertified;
    oms_policy  signing;      // same policy on initial and subsequent loads
    adapter_entry adapters[RUNNER_MAX_ADAPTERS];
    int           n_adapters;
    // R1.8.1: every sampled generation is marked with this key when wm_on
    bool          wm_on;
    wm_key        wm_key;
} server_state;

extern server_state SV;

// Cumulative work done by this process, for a capacity dashboard. The state
// lives in completion.c, beside the single place every surface's generation
// passes through -- not in server_state, because the tools that link the
// completion path without the server (test-responses-sm) must still resolve
// the recording call.
//
// Raw monotonic totals on purpose: a tok/s field would bake in an averaging
// window the runner has no business choosing, and would be wrong for every
// consumer whose window differs. The engine measures; the suite presents.
//
// One record per finished request, passed as a value rather than a widening
// parameter list: every field here is a fact about the SAME turn, and a
// six-argument call is how a caller comes to pass the cached count where the
// generated one belongs.
typedef struct {
    int    n_prompt, n_gen;
    // prompt rows this request did not have to prefill, from either cache
    // tier. The same number the wire reports as cached_tokens.
    int    n_cached;
    // The speculative walk's own counters, and only when the walk actually
    // ran: engine.spec_st is reset by engine_generate_spec, so a solo turn on
    // a slot that speculated earlier still holds the earlier numbers and
    // reading them unconditionally would count one walk many times.
    int    spec_rounds, spec_drafted, spec_accepted;
    double gen_seconds;
} work_record;

typedef struct {
    unsigned long long prompt_tokens, gen_tokens, cached_tokens;
    unsigned long long spec_rounds, spec_drafted, spec_accepted;
    double gen_seconds;
} work_totals;

void server_record_work(const work_record *w);
// R10.12.3.2: a completion request takes its ids (false: another active
// request holds the same X-Request-Id, nothing is registered), and the slot
// gives them back when the request has ended, whatever way it ended.
bool server_request_begin(slot_t *s, const char *response_id);
// R10.12.3.3: the same claim made before the request's model is swapped in,
// with no response id yet, so a cancel by X-Request-Id reaches it during the
// load; loading marks it phase 3 until the swap returns. server_request_begin
// on a claimed slot only adds the response id.
bool server_request_claim(slot_t *s, bool loading);
void server_request_end(slot_t *s);
void server_work_totals(work_totals *out);

// swap_to results below 0: the name matched no registry entry (a caller
// typo -- 400) vs the entry exists but its model failed to load (a broken
// model -- 5xx) vs the load was discarded because /unload or shutdown arrived
// while it ran (nobody's fault -- 503). Callers must tell them apart.
#define SWAP_UNKNOWN     (-1)
#define SWAP_LOAD_FAILED (-2)
#define SWAP_ABORTED     (-3)
// The entry exists and its model loads fine, but its measured-envelope sidecar
// records an outside-envelope verdict for this runtime -- a deliberate policy
// refusal (409), distinct from a broken model (5xx). --force-uncertified at
// startup suppresses it.
#define SWAP_ENVELOPE_REFUSED (-4)
#define SWAP_SIGNATURE_REFUSED (-5)
// The request named what it expects to find resident (`expect_resident`) and
// that is not what is resident: nothing was loaded, unloaded or swapped, and
// no pending unload was cancelled (409). UNKNOWN is the same refusal when the
// expectation could not be checked yet (the model digest is still being
// taken, or the file changed on disk).
#define SWAP_EXPECT_MISMATCH (-6)
#define SWAP_EXPECT_UNKNOWN  (-7)

// What a request expects to find resident. A request that carries one is
// served by the resident model or refused; it never causes a load.
typedef struct {
    bool               set;
    unsigned long long generation;   // 0 = not asked
    char               sha256[65];   // "" = not asked
} resident_expect;
// Parse `expect_resident` off a request body. False = malformed (a 400 was
// sent); *ex stays unset when the field is absent.
bool request_expect_resident(sock_t fd, jv *req, resident_expect *ex);

// ---- residency and admission (registry.c) ----

int  resident_load(void);
void resident_store(int v);
int  context_load(void);
void context_store(int value);
// Both assume SV.swap_mu is held by the caller.
void unload_resident(void);
void unload_draft(void);
void handle_unload(sock_t fd);
// Resolve + load the requested model; returns the registry index or a SWAP_*.
int  swap_to(const char *want);
// `owner` is the requesting client, or SOCK_INVALID: a load with no client to
// lose is stopped only by /unload and shutdown.
int  swap_to_expect(const char *want, const resident_expect *ex, sock_t owner);
bool validate_single_model_request(sock_t fd, jv *req);
// The same expectation on a server whose slots hold the model directly.
int  single_model_expect(const resident_expect *ex);
// R8.6: split an adapter suffix off the request's "model" ("base:name"),
// leaving the base for the usual validation; returns the adapter index, -1
// for none. A suffix naming no loaded adapter is left in place, so the model
// check answers it as an unknown model.
int  request_adapter(jv *req);
// Borrow adapter idx (-1: none) on the slot's model for this request, and
// re-key the slot's prefix identity when it changes. False: answered 500.
bool slot_use_adapter(slot_t *s, int idx, sock_t fd);
// R2.13: answers 400 and returns true for a chat_template_kwargs key the
// model's own template reads and Runner cannot render on this surface.
bool template_kwarg_refused(slot_t *s, sock_t fd, jv *req, int surface);
bool init_swap_runtime(const model_params *mp, int n_threads, int ttl);
// Admission: q_push sheds with 503 when the queue is full, q_pop blocks until
// a connection arrives or the queue is shut down (SOCK_INVALID then).
void   q_push(sock_t fd);
// Pop the next connection; *waited (optional) receives the seconds it sat
// in the queue, the request's queue-wait stage.
sock_t q_pop(double *waited);
// Drain the queue at shutdown: every waiting connection is told 503
// rather than dropped, and the workers blocked in q_pop are woken.
void   queue_shutdown(void);

#endif // RUNNER_SERVER_INT_H
