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

// --- capacity: what this process may use ----------------------------------
//
// The host's numbers narrowed by the limits a container puts on this
// process. A cgroup's CPU quota (cpu.max) and memory limit (memory.max) are
// invisible to the affinity mask, the physical page count and
// /proc/meminfo: on the Linux CUDA lab box, a container held to 26 CPUs and
// 80 GiB on a 128-CPU, 268 GB host reported the host's numbers, and every
// default sized from them over-committed. plat_cpu_count, plat_ram_bytes and
// plat_ram_available_bytes return the effective fields below.
typedef struct {
    int         cpu_cores;            // min(host, ceil(quota / period))
    uint64_t    ram_bytes;            // min(host, the memory limit)
    uint64_t    ram_available_bytes;  // min(host, what the limit leaves)
    int         host_cpu_cores;       // the affinity mask, else the online count
    uint64_t    host_ram_bytes;       // physical memory
    uint64_t    host_ram_available_bytes;   // the OS's own available figure
    // "cgroup2" or "cgroup1" when a hierarchy was read (with or without a
    // limit in it), "none" when there is none (macOS, Windows)
    const char *limits_source;
    int         cpu_quota;            // tightest quota in CPUs, rounded up; 0 = none
    uint64_t    memory_max_bytes;     // tightest memory limit; 0 = none
} plat_capacity;

void        plat_capacity_get(plat_capacity *out);
// The reading behind plat_capacity_get from explicit sources, so a test can
// point it at a fixture tree. `cgroup_root` is the cgroup mount
// (/sys/fs/cgroup; NULL or absent: no hierarchy). `self_cgroup` is the
// process's membership file (/proc/self/cgroup); NULL, or no usable line in
// it, makes the mount root the process's cgroup. Each level from the
// process's cgroup up to the mount is read and the tightest limit wins, as
// the kernel enforces them. Available memory under a limit is the headroom
// (limit less usage) plus the cgroup's reclaimable file cache and slab, the
// same terms MemAvailable counts for the host.
void        plat_capacity_at(const char *cgroup_root, const char *self_cgroup,
                             int host_cpu_cores, uint64_t host_ram_bytes,
                             uint64_t host_ram_available_bytes, plat_capacity *out);

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
// The VM page size mlock and munlock round to (4 KB on x86, 16 KB on Apple
// Silicon); 0 when the platform will not say.
size_t      plat_page_size(void);

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
// the machine starts thrashing. Inside a memory limit, what the limit leaves
// (plat_capacity). 0 where unavailable.
uint64_t    plat_ram_available_bytes(void);
bool        plat_file_readable(const char *path);
// A whole regular file as a malloc'd buffer with a NUL after the last byte
// (*n_out gets the byte count without it). NULL when the file cannot be
// opened or measured, is larger than max_bytes (0 = no cap), cannot be
// allocated, or does not read back in full: a short or failed read is an
// error, never a shorter file.
char       *plat_read_file(const char *path, size_t max_bytes, size_t *n_out);
// The last path component: what follows the last '/' or '\\' (both count on
// every platform, so a path spelled the other platform's way still names
// its file), or the whole string when it has neither.
const char *plat_path_base(const char *path);
// mkdir that counts an existing directory as success (mode 0755 on POSIX)
bool        plat_mkdir_exist_ok(const char *path);
double      plat_now(void);        // monotonic seconds

// The ANSI code page of this process: the one Windows' narrow (-A) APIs and
// the C runtime read and write text in (GetACP(); 65001 is UTF-8). With the
// manifest runner.exe carries (src/runner.manifest) it is 65001 on Windows 10
// version 1903 and later; a lower number means that manifest was missing or
// not honoured, which --caps reports so the regression is visible. 0 on other
// platforms, where arguments, file names and the environment are bytes.
unsigned    plat_code_page(void);
// Whether this executable carries a manifest that asks for the UTF-8 code
// page, read back from its own resources; false on other platforms.
bool        plat_utf8_manifest(void);

// --- UTF-8 at the operating-system boundary --------------------------------
//
// Every string runner holds is UTF-8: prompts, file names, environment
// values. POSIX passes bytes through, so on Linux and macOS the calls below
// are the libc calls themselves. Windows' narrow C runtime and -A APIs read
// text in the process's ANSI code page instead. runner.exe's manifest makes
// that UTF-8 on Windows 10 1903 and later; on an older Windows, or with the
// manifest stripped, a cp1252 system read the two UTF-8 bytes of an accented
// letter as two cp1252 characters and an ideograph as "?". So every file
// name in src/ goes through these doors, which on Windows convert UTF-8 to
// UTF-16 and call the wide entry points: what a name means no longer
// depends on the code page. tests/test_utf8_io.py holds both builds to it,
// and holds src/ to these doors.

// First thing in main(). On Windows: argv becomes the command line as the
// OS holds it (UTF-16, split by the C runtime's own rules) in UTF-8, the C
// runtime's environment table is rewritten in UTF-8 where it was not
// already, and a console on stdout or stderr is switched to the UTF-8 output
// code page until exit (redirected output is bytes, and needs nothing).
// Nothing elsewhere.
void plat_utf8_startup(int *argc, char ***argv);

// The doors convert in buffers of their own and hand nothing back to free,
// apart from plat_executable_path and plat_console_read_line. That matters:
// the tests/test_*_oom.c gates compile model.c, instances.c and others with
// counting allocators, and read a free() there of memory compat.c allocated
// as a release with no allocation behind it. None of those modules calls the
// two that do hand memory back.
#ifdef _WIN32
// CreateFileW with a UTF-8 name: the arguments and the result are
// CreateFile's (a HANDLE, INVALID_HANDLE_VALUE on failure with GetLastError()
// as the call set it), spelled without <windows.h>.
void *plat_create_file(const char *path, unsigned long access, unsigned long share,
                       void *security, unsigned long disposition, unsigned long flags);

// 64-bit sizes and times: MinGW's stat() keeps st_size in 32 bits and fails
// with EOVERFLOW on every file over 2 GiB, which is every real model.
typedef struct _stat64 plat_stat_t;
FILE *plat_fopen(const char *path, const char *mode);
// reopen `stream` (stdout) on path, as freopen
FILE *plat_freopen(const char *path, const char *mode, FILE *stream);
int   plat_open(const char *path, int flags, int mode);
int   plat_remove(const char *path);
int   plat_rename(const char *from, const char *to);
int   plat_mkdir(const char *path, int mode);     // mode: POSIX only
int   plat_rmdir(const char *path);
// stat() as MinGW's POSIX one behaves: a trailing separator names the
// directory itself ("dir/" is "dir"), except at a root
int   plat_stat(const char *path, plat_stat_t *st);
int   plat_setenv(const char *name, const char *value);
// One line typed at the console that is stdin, without its line ending, as
// malloc'd UTF-8; NULL at the end of input (Ctrl+Z first on a line, as the C
// runtime reads a console, or a read that fails), with *oom set when memory
// ran out. Read in UTF-16 (ReadConsoleW): the narrow read gives the
// console's input code page, 437 or 850 out of the box, whatever the
// process's own code page is, so a typed "e acute" arrived as one cp850 byte.
// Only for a stdin that plat_is_terminal(0) calls a console.
char *plat_console_read_line(bool *oom);
#else
typedef struct stat plat_stat_t;
#define plat_fopen(path, mode)        fopen(path, mode)
#define plat_freopen(path, mode, f)   freopen(path, mode, f)
#define plat_open(path, flags, mode)  open(path, flags, mode)
#define plat_remove(path)             remove(path)
#define plat_rename(from, to)         rename(from, to)
#define plat_mkdir(path, mode)        mkdir(path, mode)
#define plat_rmdir(path)              rmdir(path)
#define plat_stat(path, st)           stat(path, st)
#define plat_setenv(name, value)      setenv(name, value, 1)
#endif

// The absolute form of `path` in `out`; false when it fails or does not fit.
// The platform's own rules, as before: POSIX realpath() resolves links and
// needs the file to exist, Windows _fullpath() only joins and normalises.
bool plat_realpath(const char *path, char *out, size_t cap);

// A directory's entry names, "." and ".." included, in UTF-8. A name is
// valid until the next plat_readdir on the same handle.
typedef struct plat_dir plat_dir;
plat_dir   *plat_opendir(const char *path);
const char *plat_readdir(plat_dir *d);
void        plat_closedir(plat_dir *d);

// Run argv[0], looked up on PATH, with argv (NULL-terminated), and wait: its
// exit code, or -1 when it could not be started. On Windows every argument
// is quoted by the C runtime's rules first, because the spawn functions join
// argv with spaces and quote nothing: "C:\Program Files\x" reached the
// child as two arguments.
int plat_spawn_wait(const char *const argv[]);

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
