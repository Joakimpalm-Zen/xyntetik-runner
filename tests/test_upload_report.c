// The weight-upload progress policy (src/upload_report.h), driven with a
// synthetic clock. GPU-free on purpose: the behaviour under test is what was
// missing on 2026-10-28, when an 8 GB bf16 upload from a cold file on a
// throttled disk printed nothing for nine minutes and was killed as a hang.
#include "upload_report.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#define GB ((size_t)1000 * 1000 * 1000)

// Copy `total` bytes in the backend's chunks at `rate` bytes per second,
// counting the lines the policy asks for. Returns the number of lines.
static int drive(size_t total, double rate, char *first, size_t cap, double *end_t) {
    upload_report r;
    char line[192];
    double now = 100.0;                       // any monotonic origin
    upload_report_begin(&r, total, now);
    int lines = 0;
    size_t done = 0;
    while (done < total) {
        size_t n = upload_chunk(total, done);
        assert(n > 0 && n <= UPLOAD_CHUNK_BYTES && done + n <= total);
        now += (double)n / rate;
        done += n;
        if (upload_report_step(&r, n, now, line, sizeof line)) {
            if (lines == 0 && first) snprintf(first, cap, "%s", line);
            lines++;
        }
    }
    assert(r.done == total);
    if (end_t) *end_t = now - 100.0;
    return lines;
}

int main(void) {
    char first[192];
    // 1. A warm load says nothing: 8 GB at 10 GB/s is under the quiet time.
    assert(drive(8 * GB, 10e9, first, sizeof first, NULL) == 0);
    {
        upload_report r; char line[192];
        upload_report_begin(&r, 8 * GB, 0.0);
        assert(!upload_report_end(&r, 0.8, line, sizeof line));
    }
    // 2. The incident: 8.05 GB at 15 MB/s. The first line comes after the quiet
    //    time with the rate and the time left; one more every EVERY_S after.
    //    Lines fall on chunk boundaries (a 64 MB chunk is 4.5 s at this rate),
    //    so the first comes at the first boundary past the quiet time.
    double end_t = 0, chunk_t = (double)UPLOAD_CHUNK_BYTES / 15e6;
    int lines = drive(8050 * GB / 1000, 15e6, first, sizeof first, &end_t);
    double after = 0;
    assert(sscanf(strstr(first, "after "), "after %lf s", &after) == 1);
    assert(after >= UPLOAD_REPORT_QUIET_S && after <= UPLOAD_REPORT_QUIET_S + chunk_t);
    assert(strstr(first, "of 8.05 GB on the device after ") != NULL);
    assert(strstr(first, "(15 MB/s from the file; about ") != NULL);
    assert(strstr(first, " s more)") != NULL);
    // 536 s of copy: a line at least every EVERY_S, at most every EVERY_S plus a chunk
    int most = 1 + (int)((end_t - UPLOAD_REPORT_QUIET_S) / UPLOAD_REPORT_EVERY_S);
    int least = (int)((end_t - UPLOAD_REPORT_QUIET_S) / (UPLOAD_REPORT_EVERY_S + chunk_t));
    assert(lines >= least && lines <= most);
    // 3. Unknown total: the line still names the bytes and the rate, no estimate.
    {
        upload_report r; char line[192];
        upload_report_begin(&r, 0, 10.0);
        bool said = upload_report_step(&r, 3 * GB, 13.0, line, sizeof line);
        assert(said);
        assert(strstr(line, "weights 3.00 GB on the device after 3 s (1000 MB/s from the file)") != NULL);
        assert(strstr(line, "more") == NULL);
    }
    // 4. The closing line only after the policy spoke, with the whole-upload rate.
    {
        upload_report r; char line[192];
        upload_report_begin(&r, 2 * GB, 0.0);
        assert(!upload_report_step(&r, GB, 1.0, line, sizeof line));
        assert(upload_report_step(&r, GB, 2.5, line, sizeof line));
        assert(upload_report_end(&r, 2.5, line, sizeof line));
        assert(strstr(line, "weights on the device: 2.00 GB in 2 s (800 MB/s from the file)") != NULL);
    }
    // 5. Chunking covers an odd length exactly, last piece short.
    {
        size_t len = 3 * UPLOAD_CHUNK_BYTES + 12345, done = 0; int chunks = 0;
        while (done < len) { done += upload_chunk(len, done); chunks++; }
        assert(done == len && chunks == 4);
    }
    printf("upload report: warm load silent, slow load reported every %.0f s after %.0f s, "
           "unknown total named, closing line, chunks exact\n",
           UPLOAD_REPORT_EVERY_S, UPLOAD_REPORT_QUIET_S);
    return 0;
}
