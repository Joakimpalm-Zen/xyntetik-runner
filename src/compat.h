// Platform abstraction: memory-mapped files, CPU count, paths, clocks.
// POSIX (Linux/macOS/BSD) and Windows (MinGW-w64 / MSYS2).
#ifndef RUNNER_COMPAT_H
#define RUNNER_COMPAT_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>     // FILE: plat_env_file

// printf-style argument checking against the runtime that formats. MinGW's
// GCC reads format(printf) as Microsoft's rules, which have no %z, so every
// %zu in a helper that forwards to vsnprintf warned and misaligned the rest
// of its arguments; UCRT's vsnprintf prints %zu correctly. The MinGW headers
// name the archetype of the runtime they selected in __MINGW_PRINTF_FORMAT
// (gnu_printf for UCRT), which is what the libc printf itself is checked by.
#ifdef __MINGW32__
#include <stdio.h>
#endif
#if defined(__MINGW32__) && defined(__MINGW_PRINTF_FORMAT)
#define RUNNER_PRINTF(fmt, args) __attribute__((format(__MINGW_PRINTF_FORMAT, fmt, args)))
#else
#define RUNNER_PRINTF(fmt, args) __attribute__((format(printf, fmt, args)))
#endif

static inline bool checked_u64_add(uint64_t a, uint64_t b, uint64_t *out) {
    if (b > UINT64_MAX - a) return false;
    *out = a + b;
    return true;
}

static inline bool checked_u64_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

// map a regular file read-only; returns NULL on failure (missing, empty,
// directory, ...). The mapping outlives any internal handles.
void  *plat_mmap_ro(const char *path, size_t *size);
// Ask the OS to fault a whole range in as one unit. Advisory: it can never
// change what a later read returns, only how many faults that read costs.
void   plat_willneed(const void *addr, size_t len);
// Whether plat_willneed actually does anything on this build+OS. POSIX always
// has madvise; Windows needs PrefetchVirtualMemory, which is resolved at
// runtime and absent before Windows 8. Callers use this so nothing announces
// a prefetch that silently is not happening — an A/B run against a banner
// that lies is worse than no A/B, which this project has paid for once
// already (the Metal shader library that failed to compile while --caps still
// advertised the backend).
bool   plat_willneed_available(void);
// Hand the pages of a range of a READ-ONLY file mapping back to the OS: they
// leave this process's resident set, and a later read faults them back in
// from the file (normally the page cache) with the same bytes. Only whole
// pages inside the range go, so a neighbour sharing a boundary page keeps
// it. Advisory like plat_willneed: it changes residency, never a value.
// Never on a locked range (--mlock): Linux refuses it there, and the Windows
// call that trims the working set is the one that unlocks.
void   plat_release_pages(const void *addr, size_t len);
void   plat_munmap(void *p, size_t size);

int         plat_cpu_count(void);
// Ceiling on the DEFAULT thread count. Distinct from TP_MAX, which clamps an
// explicit -t: a caller who asks for 64 still gets 64.
//
// The halving rule below approximates physical cores on an SMT machine and is
// unmeasured; this ceiling is the measured part. Two independent sweeps found
// decode PEAKS at 32 threads and regresses above it - tpbench on a 64-core
// Zen 5 box (2026-08-13) and the 128-core Linux host sweep (2026-08-29), where
// both models peaked at t=32 and the previous default of 64 cost -41.0%
// decode and -55.3% prefill on the smaller one. The heuristic was not wrong,
// it lacked a ceiling.
#define PLAT_THREAD_DEFAULT_MAX 32
int         plat_default_thread_count(void);
// The policy as a pure function of the logical CPU count, so the many-core
// regime is testable on a machine that does not have many cores.
int         plat_thread_default_for(int logical_cpus);
uint64_t    plat_ram_bytes(void);

// --- weight residency -----------------------------------------------------
//
// runner mmaps its weights and the OS is free to evict them. On a machine
// under real memory pressure that turns every token into SSD paging, and it is
// invisible: tok/s, /health and --caps all stay green while a five-token reply
// takes a minute. Reported from a 16 GB M2 Pro driving an IDE, where a 1.2k
// prompt produced zero tokens in 300 s at 0% CPU. These four are what let
// runner say so.

// Wire the mapping into RAM. OPT-IN and expected to fail: locking 5 GB on a
// 16 GB machine can create the pressure it was meant to avoid, so a false
// return is a normal outcome to report, not an error to abort on.
bool        plat_mlock(void *p, size_t size);
void        plat_munlock(void *p, size_t size);

// Fraction of the mapping currently in physical memory, or -1.0 where the
// platform will not say. This is the direct answer to "is the model resident",
// which model *file size* cannot give: residency is a property of the moment.
double      plat_resident_fraction(const void *p, size_t size);
// Resident set of THIS process, in bytes, and the high-water mark; 0 when the
// platform will not say. Distinct from plat_resident_fraction, which asks how
// much of one MAPPING is in memory: a supervisor budgeting RAM across several
// runners needs the process total, including the KV cache, activations and
// allocator overhead that no mapping accounts for.
uint64_t    plat_proc_rss_bytes(void);
uint64_t    plat_proc_peak_rss_bytes(void);

// Major faults this process has taken — page-ins that went to disk. The delta
// across a request is the exact mechanism behind a paging stall, which beats
// inferring one from wall-clock: a slow request that took no major faults was
// slow for some other reason. 0 where unavailable.
uint64_t    plat_major_faults(void);

// The OS video-memory budget for THIS process on the adapter identified by
// an 8-byte LUID (IDXGIAdapter3::QueryVideoMemoryInfo, local segment):
// *budget is what the process may keep resident before the OS starts paging
// it, *usage what it holds now. True only where the OS publishes one
// (WDDM); false on every other platform and when the query is unavailable,
// which the caller treats as "budget unknown", never as zero. The driver's
// own free-memory count knows nothing of this: on a 12 GB card shared with
// the desktop it promised 11.7 GB, and WDDM then paged the process over PCIe
// (2026-09-14 Windows report).
bool        plat_gpu_os_budget(const unsigned char luid[8], uint64_t *budget,
                               uint64_t *usage);
// The page-fault counter plat_major_faults reads: "major" where the OS
// separates page-ins from disk (POSIX ru_majflt), "all" where it does not
// (Windows PageFaultCount counts soft faults too). Reported beside the
// number so a reader knows what it means on the machine it came from.
const char *plat_page_fault_counter(void);

// Physically available RAM: free plus what the OS would reclaim on demand. NOT
// total RAM, because "model fits in RAM" is the test that passes right before
// the machine starts thrashing. 0 where unavailable.
uint64_t    plat_ram_available_bytes(void);
bool        plat_file_readable(const char *path);
double      plat_now(void);        // monotonic seconds

// Install a completed same-filesystem temporary file at `path`, replacing an
// existing regular file. POSIX rename() has replacement semantics; Windows
// needs MoveFileEx(..., REPLACE_EXISTING) for the same operation.
bool        plat_replace_file(const char *tmp_path, const char *path);

// An openable path for the executable image of this process, independent of
// argv[0] and the current working directory. Caller frees; NULL on failure.
char       *plat_executable_path(void);

// exit(0) this process when pid dies — supervisors pass their own pid so a
// SIGKILLed xyntetik-thane cannot leave an orphaned runner. pid <= 0 = no-op.
void        plat_parent_watch(long pid);

void        plat_sleep_ms(int ms);
long        plat_pid_self(void);

// Is file descriptor `fd` a terminal a person is at? On Windows a console,
// not merely a character device: _isatty() is also true for NUL, which is
// where services, process managers and subprocess.DEVNULL put stdin.
bool        plat_is_terminal(int fd);

// Is `pid` a process that still exists? Used to reap registry entries left by
// runners that died without deregistering (SIGKILL, crash, OOM killer).
bool        plat_pid_alive(long pid);

// Opaque process-creation stamp, false when the platform cannot report one.
// The unit is per-platform (Linux: seconds since boot; Windows/macOS: seconds
// since the epoch): values are only ever compared for equality against stamps
// recorded on the same machine, so files that store them must not outlive a
// boot. Pids are recycled, so an entry whose pid is alive but whose stamp has
// moved belongs to an unrelated process and is still dead for reaping purposes.
bool        plat_pid_start_time(long pid, uint64_t *out);

// Durable-but-transient scratch directory: $XDG_RUNTIME_DIR, else the platform
// temp directory, else /tmp. Never NULL.
const char *plat_runtime_dir(void);

// Read-modify-write `path` under an exclusive cross-process lock.
//
// The whole file is read, handed to `fn`, and whatever `fn` returns (a malloc'd
// NUL-terminated string, or NULL to leave the file untouched) is written back;
// the lock is held across all of it, so concurrent callers serialise. `in` is
// "" for a file that does not exist yet.
//
// Returns false when the file could not be opened/read or the complete update
// could not be written. When the platform has no working lock the
// read-modify-write still happens unlocked — an unlockable filesystem must not
// stop a runner from starting.
typedef char *(*plat_rmw_fn)(const char *in, size_t in_len, void *ud);
bool        plat_file_rmw(const char *path, plat_rmw_fn fn, void *ud);

// Strict numeric parsing shared by CLI flags and environment overrides.
// parse_* return false (leaving *out untouched) on empty/garbage/overflow/sign/
// non-finite input. env_* read getenv(name): absent keeps `cur`, invalid warns
// to stderr and keeps `cur` — never a silent zero.
bool      parse_i64(const char *s, long long lo, long long hi, long long *out);
bool      parse_u64(const char *s, uint64_t lo, uint64_t hi, uint64_t *out);
bool      parse_f64(const char *s, double lo, double hi, double *out);
long long env_i64(const char *name, long long lo, long long hi, long long cur);
uint64_t  env_u64(const char *name, uint64_t lo, uint64_t hi, uint64_t cur);
double    env_f64(const char *name, double lo, double hi, double cur);

// Process-wide state any thread may reach first. A plain static set on first
// use is a data race when two threads get there together (two slots, or an
// embedder's two engines, starting cold), so these two are the shared forms.
//
// plat_once: true for exactly one caller, the first; *flag starts false. For
// a notice said once per process.
bool plat_once(_Atomic bool *flag);
// plat_env_file: the file an environment variable names, opened for append
// by the first caller and kept; NULL when the variable is unset or the open
// failed (which is warned once). Zero-initialise the struct.
typedef struct { _Atomic int state; FILE *fp; } plat_env_file;
FILE *plat_env_file_get(plat_env_file *f, const char *env);

#endif
