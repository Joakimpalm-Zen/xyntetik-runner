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
int    tpool_size(const tpool *tp);  // workers incl. the calling thread

#endif // RUNNER_TPOOL_H
