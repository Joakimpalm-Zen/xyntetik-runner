// Progress for a long weight upload to a device.
//
// The weights go to the device straight from the model file's mapping, so the
// copy runs at whatever speed the storage faults the pages in. On a warm page
// cache that is seconds; on a cold file over a throttled disk it is minutes,
// and until 2026-10-28 nothing said so: a bf16 4B (8.05 GB) on the lab's
// IOPS-capped array faulted in at under 15 MB/s, the process printed its split
// lines and then nothing for nine minutes, and the run was killed as a hang
// (reported as "--score with --lora hangs on CUDA"; the adapter never loaded,
// the upload was still reading the file).
//
// The policy here is backend-free so a test can drive it with a synthetic
// clock: the first UPLOAD_REPORT_QUIET_S are silent (a normal load says
// nothing new), then one line every UPLOAD_REPORT_EVERY_S with the bytes
// placed, the rate from the file and, when the total is known, the time left.
// The backend copies in UPLOAD_CHUNK_BYTES pieces so it can call in between.
#ifndef UPLOAD_REPORT_H
#define UPLOAD_REPORT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#define UPLOAD_REPORT_QUIET_S 2.0
#define UPLOAD_REPORT_EVERY_S 5.0
#define UPLOAD_CHUNK_BYTES    ((size_t)64u << 20)

typedef struct {
    size_t total;       // bytes the whole upload will place; 0 = unknown
    size_t done;        // bytes placed so far
    double t0;          // clock at the start (seconds, any monotonic origin)
    double t_last;      // clock at the last line, or t0
    int    lines;       // lines spoken so far
} upload_report;

static inline void upload_report_begin(upload_report *r, size_t total, double now) {
    r->total = total;
    r->done = 0;
    r->t0 = now;
    r->t_last = now;
    r->lines = 0;
}

// The next chunk of a copy: at most UPLOAD_CHUNK_BYTES, never past `len`.
static inline size_t upload_chunk(size_t len, size_t done) {
    size_t left = len - done;
    return left < UPLOAD_CHUNK_BYTES ? left : UPLOAD_CHUNK_BYTES;
}

// Account `bytes` placed at `now`. Returns true when a line is due and writes
// it to `line` (without the "gpu: " prefix or a newline). The first line waits
// out the quiet time; later lines come every UPLOAD_REPORT_EVERY_S.
static inline bool upload_report_step(upload_report *r, size_t bytes, double now,
                                      char *line, size_t cap) {
    r->done += bytes;
    double since = now - (r->lines ? r->t_last : r->t0);
    double due = r->lines ? UPLOAD_REPORT_EVERY_S : UPLOAD_REPORT_QUIET_S;
    if (since < due) return false;
    double elapsed = now - r->t0;
    double rate = elapsed > 0 ? (double)r->done / elapsed : 0.0;   // bytes/s
    int n;
    if (r->total && rate > 0 && r->done <= r->total) {
        double left = (double)(r->total - r->done) / rate;
        n = snprintf(line, cap, "weights %.2f of %.2f GB on the device after %.0f s "
                     "(%.0f MB/s from the file; about %.0f s more)",
                     r->done / 1e9, r->total / 1e9, elapsed, rate / 1e6, left);
    } else {
        n = snprintf(line, cap, "weights %.2f GB on the device after %.0f s "
                     "(%.0f MB/s from the file)", r->done / 1e9, elapsed, rate / 1e6);
    }
    (void)n;
    r->t_last = now;
    r->lines++;
    return true;
}

// The closing line when the upload spoke at all: how long it took in the end.
static inline bool upload_report_end(const upload_report *r, double now,
                                     char *line, size_t cap) {
    if (!r->lines) return false;
    double elapsed = now - r->t0;
    snprintf(line, cap, "weights on the device: %.2f GB in %.0f s (%.0f MB/s from the file)",
             r->done / 1e9, elapsed, elapsed > 0 ? r->done / 1e6 / elapsed : 0.0);
    return true;
}

#endif
