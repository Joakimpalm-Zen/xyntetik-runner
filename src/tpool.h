// Fixed worker pool: run one index range function over N items.
#ifndef RUNNER_TPOOL_H
#define RUNNER_TPOOL_H


typedef void (*tp_fn)(void *ctx, int i0, int i1); // process items [i0, i1)
typedef struct tpool tpool;
tpool *tpool_create(int n_threads);
void   tpool_run(tpool *tp, tp_fn fn, void *ctx, int n_items);
void   tpool_destroy(tpool *tp);
// A pool shared by several model instances (the server's slots): each holder
// retains it once and destroys it once; the last destroy frees the threads.
// tpool_run serializes runs, so holders may call it from their own threads
// and their forwards interleave one matvec at a time.
void   tpool_retain(tpool *tp);
// The size of the one pool every server slot shares: the server's thread
// count, at least `floor` (a model's CPU-forced fallback count), and one CPU
// short of the affinity mask when more than one slot shares it (the slots'
// own threads and the HTTP thread need a CPU; 16 pool threads on 16 cores
// halved a lone request, 15 did not, 2026-10-08). The one place the policy
// lives; server.c calls it and tests/test_thread_default.c pins it.
int    tpool_shared_threads(int n_threads, int parallel, int cpus, int floor);
int    tpool_size(const tpool *tp);  // workers incl. the calling thread

#endif // RUNNER_TPOOL_H
