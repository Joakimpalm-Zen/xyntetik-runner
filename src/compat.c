#define _CRT_RAND_S   // rand_s, for plat_random: before the first system header
// sched_getaffinity (machine_cpu_count counts the mask this process may run on)
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include "compat.h"
#include "fpclass.h"

#include <stdatomic.h>

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#include <psapi.h>

static wchar_t *utf8_to_wide(const char *s);   // the UTF-8 doors, below
static char *wide_to_utf8(const wchar_t *w);

void *plat_mmap_ro(const char *path, size_t *size) {
    wchar_t *wp = utf8_to_wide(path);
    HANDLE f = wp ? CreateFileW(wp, GENERIC_READ, FILE_SHARE_READ, NULL,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL)
                  : INVALID_HANDLE_VALUE;
    free(wp);
    if (f == INVALID_HANDLE_VALUE) return NULL;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz) || sz.QuadPart <= 0) { CloseHandle(f); return NULL; }
    HANDLE m = CreateFileMappingA(f, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(f); // the mapping keeps the file alive
    if (!m) return NULL;
    void *p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(m); // the view keeps the mapping alive
    if (!p) return NULL;
    *size = (size_t)sz.QuadPart;
    return p;
}

// PrefetchVirtualMemory is Windows' madvise(MADV_WILLNEED): it asks the
// memory manager to bring a range in as few large I/Os instead of letting it
// arrive as a storm of individual faults.
//
// This was a no-op until 2026-08-07, which quietly made Windows the only
// platform still paying fault-by-fault for MoE experts. Measured on the
// gemma-4-26B routing traces, an expert reached through ~16 KB faults costs
// 49,440 requests/token against 480 when the same bytes are asked for as
// whole blocks — a 5.43x difference in cold expert I/O. Every other platform
// has had the block path since v0.1.11.
//
// Resolved at runtime rather than linked: the symbol needs Windows 8 /
// Server 2012, and some MinGW SDK headers do not declare it at all. Missing
// symbol means the old no-op, which is always safe — this advice can change
// how fast bytes arrive, never which bytes arrive.
typedef struct { PVOID VirtualAddress; SIZE_T NumberOfBytes; } runner_mem_range;
typedef BOOL (WINAPI *runner_pvm_fn)(HANDLE, ULONG_PTR, runner_mem_range *, ULONG);

static runner_pvm_fn runner_pvm(void) {
    static runner_pvm_fn pvm;
    static LONG resolved;              // 0 = untried, 1 = done
    if (!InterlockedCompareExchange(&resolved, 1, 0)) {
        HMODULE k32 = GetModuleHandleA("kernel32.dll");
        if (k32)
            pvm = (runner_pvm_fn)(void *)GetProcAddress(k32, "PrefetchVirtualMemory");
    }
    return pvm;
}

bool plat_willneed_available(void) { return runner_pvm() != NULL; }

void plat_willneed(const void *addr, size_t len) {
    if (!addr || !len) return;
    runner_pvm_fn pvm = runner_pvm();
    // Benign race by construction: a second thread arriving mid-resolution
    // sees pvm still NULL and skips one advisory call. The cost of that is
    // one expert arriving by faults instead of blocks, which is the behaviour
    // this whole function is an optimisation over — never a correctness
    // question. Worth less than a lock on the FFN hot path.
    if (!pvm) return;                  // pre-Win8, or SDK without it
    runner_mem_range r;
    r.VirtualAddress = (PVOID)(uintptr_t)addr;
    r.NumberOfBytes = len;
    // Flags is reserved and must be 0. The return value is deliberately
    // ignored: a refused prefetch is a slower read, not a failed one, and the
    // caller has nothing useful to do about it either way.
    (void)pvm(GetCurrentProcess(), 1, &r, 0);
}

// VirtualUnlock on pages that are NOT locked removes them from the working
// set (documented behaviour; it then reports ERROR_NOT_LOCKED, which is the
// expected answer here, so the return value carries nothing). A mapped view
// cannot be decommitted in part, and this trim is what the memory manager
// offers instead: the pages stay in the standby list and fault back in soft.
void plat_release_pages(const void *addr, size_t len) {
    if (!addr || !len) return;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    uintptr_t psz = si.dwPageSize ? si.dwPageSize : 4096;
    uintptr_t lo = ((uintptr_t)addr + psz - 1) & ~(psz - 1);
    uintptr_t hi = ((uintptr_t)addr + len) & ~(psz - 1);
    if (hi > lo) (void)VirtualUnlock((void *)lo, (SIZE_T)(hi - lo));
}

void plat_munmap(void *p, size_t size) {
    (void)size;
    if (p) UnmapViewOfFile(p);
}

static int machine_cpu_count(void) {
    // the process affinity mask first (a job object or `start /affinity`
    // narrows it), the machine's count as the fallback; see the POSIX twin
    DWORD_PTR pm = 0, sm = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm) && pm) {
        int n = 0;
        for (DWORD_PTR b = pm; b; b &= b - 1) n++;
        if (n > 0) return n;
    }
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1;
}

int plat_default_thread_count(void) {
    return plat_thread_default_for(plat_cpu_count());
}

static uint64_t machine_ram_bytes(void) {
    MEMORYSTATUSEX ms = { .dwLength = sizeof(ms) };
    GlobalMemoryStatusEx(&ms);
    return ms.ullTotalPhys;
}

// Bytes this process holds locked, which the working-set minimum must cover.
static volatile LONG64 g_win_locked;

bool plat_mlock(void *p, size_t size) {
    if (!p || !size) return false;
    // The default working-set maximum is far below a model, so raise it first
    // or VirtualLock fails on anything worth locking. Both calls are allowed
    // to fail: this is opt-in and reported, never fatal. The minimum covers
    // everything this process holds locked plus this range: --mlock locks a
    // model as several ranges, and sizing each call alone would let the
    // second shrink the budget the first is counted against.
    SIZE_T lo = 0, hi = 0;
    HANDLE self = GetCurrentProcess();
    if (GetProcessWorkingSetSize(self, &lo, &hi)) {
        SIZE_T want = (SIZE_T)g_win_locked + (SIZE_T)size + (64u << 20);   // headroom for everything else
        if (lo < want) SetProcessWorkingSetSize(self, want, hi > want ? hi : want);
    }
    if (!VirtualLock(p, size)) {
        errno = GetLastError() == ERROR_WORKING_SET_QUOTA ? EAGAIN : ENOMEM;
        return false;
    }
    InterlockedExchangeAdd64(&g_win_locked, (LONG64)size);
    return true;
}

void plat_munlock(void *p, size_t size) {
    if (p && size && VirtualUnlock(p, size))
        InterlockedExchangeAdd64(&g_win_locked, -(LONG64)size);
}

size_t plat_page_size(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
}

double plat_resident_fraction(const void *p, size_t size) {
    if (!p || !size) return -1.0;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    size_t psz = si.dwPageSize ? si.dwPageSize : 4096;
    size_t pages = (size + psz - 1) / psz;
    enum { WINDOW = 4096 };
    PSAPI_WORKING_SET_EX_INFORMATION *info =
        malloc(sizeof(*info) * WINDOW);
    if (!info) return -1.0;
    size_t resident = 0, done = 0;
    while (done < pages) {
        size_t n = pages - done < WINDOW ? pages - done : WINDOW;
        for (size_t i = 0; i < n; i++)
            info[i].VirtualAddress = (PVOID)((const char *)p + (done + i) * psz);
        if (!QueryWorkingSetEx(GetCurrentProcess(), info,
                               (DWORD)(sizeof(*info) * n))) {
            free(info);
            return -1.0;
        }
        for (size_t i = 0; i < n; i++)
            if (info[i].VirtualAttributes.Valid) resident++;
        done += n;
    }
    free(info);
    return pages ? (double)resident / (double)pages : -1.0;
}

uint64_t plat_major_faults(void) {
    // Windows does not separate hard from soft faults here, so this counts
    // both. The delta is still the right shape of signal, but a nonzero value
    // means less than it does on POSIX -- said plainly rather than papered over.
    PROCESS_MEMORY_COUNTERS pmc = { .cb = sizeof(pmc) };
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return 0;
    return (uint64_t)pmc.PageFaultCount;
}

const char *plat_page_fault_counter(void) { return "all"; }

// DXGI without linking dxgi.lib or dragging in its headers: the factory
// entry point is loaded by name and the three COM interfaces are called
// through their vtables. IDXGIAdapter3 (dxgi1_4) adds QueryVideoMemoryInfo,
// which reports the process's WDDM budget and current usage per memory
// segment group; LOCAL is the card's own memory.
typedef struct { unsigned char b[16]; } win_guid;
// {770AAE78-F26F-4DBA-A829-253C83D1B387} and {645967A4-1392-4310-A798-8053CE3E93FD},
// spelled in memory order (Data1..Data3 little-endian, Data4 as written)
static const win_guid IID_FACTORY1 =
    {{0x78,0xae,0x0a,0x77,0x6f,0xf2,0xba,0x4d,0xa8,0x29,0x25,0x3c,0x83,0xd1,0xb3,0x87}};
static const win_guid IID_ADAPTER3 =
    {{0xa4,0x67,0x59,0x64,0x92,0x13,0x10,0x43,0xa7,0x98,0x80,0x53,0xce,0x3e,0x93,0xfd}};
typedef struct { unsigned long long Budget, CurrentUsage,
                 AvailableForReservation, CurrentReservation; } win_vmem_info;
typedef struct {
    unsigned short Description[128];
    unsigned VendorId, DeviceId, SubSysId, Revision;
    size_t DedicatedVideoMemory, DedicatedSystemMemory, SharedSystemMemory;
    unsigned char AdapterLuid[8];
    unsigned Flags;
} win_adapter_desc1;
typedef struct win_com { struct win_com_vtbl *vt; } win_com;
struct win_com_vtbl {
    long (__stdcall *QueryInterface)(win_com *, const win_guid *, void **);
    unsigned long (__stdcall *AddRef)(win_com *);
    unsigned long (__stdcall *Release)(win_com *);
    void *slot[9];   // IDXGIObject (4) + IDXGIFactory (5), unused here
    long (__stdcall *EnumAdapters1)(win_com *, unsigned, win_com **);   // IDXGIFactory1[12]
};
// IDXGIAdapter3's vtable: IUnknown(3) + IDXGIObject(4) + IDXGIAdapter(3) +
// IDXGIAdapter1(1: GetDesc1) + IDXGIAdapter2(1) + IDXGIAdapter3(...:
// RegisterHardwareContentProtectionTeardownStatusEvent,
// UnregisterHardwareContentProtectionTeardownStatus,
// QueryVideoMemoryInfo, ...)
struct win_adapter_vtbl {
    void *iunknown[3];
    void *object[4];
    void *adapter[3];
    long (__stdcall *GetDesc1)(win_com *, win_adapter_desc1 *);
    void *adapter2[1];
    void *teardown[2];
    long (__stdcall *QueryVideoMemoryInfo)(win_com *, unsigned, int, win_vmem_info *);
};

// One QueryVideoMemoryInfo pass on the adapter with this LUID: the LOCAL
// segment group always, the NON_LOCAL one (system memory holding the
// process's allocations, Task Manager's "shared GPU memory") when asked.
static bool dxgi_vidmem(const unsigned char luid[8], win_vmem_info *local,
                        win_vmem_info *nonlocal) {
    typedef long (__stdcall *create_fn)(const win_guid *, void **);
    HMODULE dxgi = LoadLibraryA("dxgi.dll");
    if (!dxgi) return false;
    create_fn create = (create_fn)(void *)GetProcAddress(dxgi, "CreateDXGIFactory1");
    if (!create) { FreeLibrary(dxgi); return false; }
    win_com *factory = NULL;
    bool found = false;
    if (create(&IID_FACTORY1, (void **)&factory) == 0 && factory) {
        for (unsigned i = 0; !found; i++) {
            win_com *adapter = NULL;
            if (factory->vt->EnumAdapters1(factory, i, &adapter) != 0 || !adapter) break;
            win_com *a3 = NULL;
            if (adapter->vt->QueryInterface(adapter, &IID_ADAPTER3, (void **)&a3) == 0 && a3) {
                struct win_adapter_vtbl *vt = (struct win_adapter_vtbl *)a3->vt;
                win_adapter_desc1 desc;
                memset(&desc, 0, sizeof(desc));
                if (vt->GetDesc1(a3, &desc) == 0 &&
                    memcmp(desc.AdapterLuid, luid, 8) == 0) {
                    memset(local, 0, sizeof(*local));
                    found = vt->QueryVideoMemoryInfo(a3, 0, /*LOCAL*/ 0, local) == 0;
                    if (found && nonlocal) {
                        memset(nonlocal, 0, sizeof(*nonlocal));
                        found = vt->QueryVideoMemoryInfo(a3, 0, /*NON_LOCAL*/ 1,
                                                         nonlocal) == 0;
                    }
                }
                a3->vt->Release(a3);
            }
            adapter->vt->Release(adapter);
        }
        factory->vt->Release(factory);
    }
    FreeLibrary(dxgi);
    return found;
}

bool plat_gpu_os_budget(const unsigned char luid[8], uint64_t *budget,
                        uint64_t *usage) {
    win_vmem_info local;
    if (!dxgi_vidmem(luid, &local, NULL)) return false;
    *budget = local.Budget;
    *usage  = local.CurrentUsage;
    return true;
}

bool plat_gpu_os_vidmem(const unsigned char luid[8], uint64_t *budget,
                        uint64_t *usage, uint64_t *nonlocal) {
    win_vmem_info local, shared;
    if (!dxgi_vidmem(luid, &local, &shared)) return false;
    *budget   = local.Budget;
    *usage    = local.CurrentUsage;
    *nonlocal = shared.CurrentUsage;
    return true;
}

uint64_t plat_proc_rss_bytes(void) {
    PROCESS_MEMORY_COUNTERS pmc = { .cb = sizeof(pmc) };
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return 0;
    return (uint64_t)pmc.WorkingSetSize;
}

uint64_t plat_proc_peak_rss_bytes(void) {
    PROCESS_MEMORY_COUNTERS pmc = { .cb = sizeof(pmc) };
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return 0;
    return (uint64_t)pmc.PeakWorkingSetSize;
}

static uint64_t machine_ram_available_bytes(void) {
    MEMORYSTATUSEX ms = { .dwLength = sizeof(ms) };
    if (!GlobalMemoryStatusEx(&ms)) return 0;
    return ms.ullAvailPhys;
}

bool plat_file_readable(const char *path) {
    wchar_t *wp = utf8_to_wide(path);
    bool ok = wp && _waccess(wp, 4) == 0;
    free(wp);
    return ok;
}

double plat_now(void) {
    static LARGE_INTEGER freq;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}

static DWORD WINAPI parent_wait(LPVOID h) {
    WaitForSingleObject((HANDLE)h, INFINITE);
    fprintf(stderr, "parent process exited — shutting down\n");
    // _exit(0) maps to ExitProcess, which runs DLL_PROCESS_DETACH after
    // killing peer threads at arbitrary points; if a slot thread died
    // mid-CUDA-call, nvcuda's DllMain can deadlock on it, leaving a zombie
    // process holding VRAM. TerminateProcess skips DLL rundown entirely.
    TerminateProcess(GetCurrentProcess(), 0);
    return 0; // unreachable: the process is gone by the time we'd get here
}

void plat_sleep_ms(int ms) {
    if (ms > 0) Sleep((DWORD)ms);
}

long plat_pid_self(void) {
    return (long)GetCurrentProcessId();
}

bool plat_is_terminal(int fd) {
    // GetConsoleMode succeeds only on a console handle; NUL, a pipe and a
    // file all fail it, which _isatty cannot tell apart for NUL
    intptr_t h = _get_osfhandle(fd);
    DWORD mode;
    return h != -1 && GetConsoleMode((HANDLE)h, &mode) != 0;
}

bool plat_pid_alive(long pid) {
    if (pid <= 0) return false;
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
    // ERROR_ACCESS_DENIED means the process exists but belongs to someone else:
    // still alive, and its reservation is still real
    if (!h) return GetLastError() == ERROR_ACCESS_DENIED;
    DWORD code = 0;
    bool alive = GetExitCodeProcess(h, &code) && code == STILL_ACTIVE;
    CloseHandle(h);
    return alive;
}

bool plat_pid_start_time(long pid, uint64_t *out) {
    if (pid <= 0) return false;
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
    if (!h) return false;
    FILETIME create, exit_t, kern, user;
    bool ok = GetProcessTimes(h, &create, &exit_t, &kern, &user);
    CloseHandle(h);
    if (!ok) return false;
    // FILETIME is 100ns ticks since 1601-01-01; 11644473600 seconds to Unix
    uint64_t ticks = ((uint64_t)create.dwHighDateTime << 32) | create.dwLowDateTime;
    *out = ticks / 10000000ull - 11644473600ull;
    return true;
}

const char *plat_runtime_dir(void) {
    const char *d = getenv("XDG_RUNTIME_DIR");
    if (d && *d) return d;
    if ((d = getenv("TEMP")) && *d) return d;
    if ((d = getenv("TMP")) && *d) return d;
    return ".";
}

bool plat_file_rmw(const char *path, plat_rmw_fn fn, void *ud) {
    wchar_t *wp = utf8_to_wide(path);
    HANDLE f = wp ? CreateFileW(wp, GENERIC_READ | GENERIC_WRITE,
                                FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                                OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL)
                  : INVALID_HANDLE_VALUE;
    free(wp);
    if (f == INVALID_HANDLE_VALUE) return false;

    OVERLAPPED ov = {0};
    // LockFileEx is advisory-by-convention here: every writer of this file goes
    // through plat_file_rmw. A failure is not fatal — see the header note.
    bool locked = LockFileEx(f, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &ov);

    // A file that exists but cannot be read back — oversized, unreadable, or
    // malloc failed — fails the whole rmw. Running fn against an empty view
    // and writing that back would truncate every other process's entries.
    char *buf = NULL;
    size_t len = 0;
    LARGE_INTEGER sz;
    bool readable = GetFileSizeEx(f, &sz) && sz.QuadPart < (1 << 20);
    if (readable && sz.QuadPart > 0) {
        len = (size_t)sz.QuadPart;
        buf = malloc(len + 1);
        size_t off = 0;
        while (buf && off < len) {
            DWORD got = 0;
            if (!ReadFile(f, buf + off, (DWORD)(len - off), &got, NULL) ||
                got == 0) {
                readable = false;
                break;
            }
            off += got;
        }
        if (!buf) readable = false;
        if (readable) buf[len] = 0;
    }
    if (!readable) {
        free(buf);
        if (locked) {
            OVERLAPPED uo = {0};
            UnlockFileEx(f, 0, MAXDWORD, MAXDWORD, &uo);
        }
        CloseHandle(f);
        return false;
    }

    char *out = fn(buf ? buf : "", len, ud);
    free(buf);

    bool written = true;
    if (out) {
        const char *inject = getenv("RUNNER_FILE_RMW_WRITE_FAIL");
        size_t n = strlen(out);
        if ((inject && *inject && strcmp(inject, "0") != 0) ||
            n > MAXDWORD) {
            written = false;
        }
        LARGE_INTEGER zero = {0};
        if (written && !SetFilePointerEx(f, zero, NULL, FILE_BEGIN))
            written = false;
        size_t off = 0;
        while (written && off < n) {
            DWORD wrote = 0;
            if (!WriteFile(f, out + off, (DWORD)(n - off), &wrote, NULL) ||
                wrote == 0) {
                written = false;
                break;
            }
            off += wrote;
        }
        // Truncate only after the complete replacement is present. A short
        // write can still damage its prefix, but must not also discard the
        // untouched tail or be reported as success.
        if (written && !SetEndOfFile(f)) written = false;
        // Callers keep live-process VRAM accounting or disposable yield
        // sentinels, not durable data. Power loss ends their owners; readers
        // tolerate torn rows and reap stale owners. Do not stall the lock on
        // a durability flush. In-place writes were never crash-atomic.
        free(out);
    }

    if (locked) {
        OVERLAPPED uo = {0};
        UnlockFileEx(f, 0, MAXDWORD, MAXDWORD, &uo);
    }
    CloseHandle(f);
    return written;
}

void plat_parent_watch(long pid) {
    if (pid <= 0) return;
    HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, (DWORD)pid);
    if (!h) {
        // unobservable parent == already-dead parent: refusing to run
        // unwatched is the whole point of the flag. This runs before model
        // load (no CUDA threads exist yet), so a plain _exit is fine here.
        fprintf(stderr, "error: --parent-pid %ld is not observable — exiting\n", pid);
        _exit(0);
    }
    HANDLE th = CreateThread(NULL, 0, parent_wait, h, 0, NULL);
    if (th) {
        CloseHandle(th);
    } else {
        // same rationale as the unobservable-pid branch above: this also
        // runs pre-model-load, so no CUDA threads exist yet and a plain
        // _exit is fine — but continuing unwatched would silently break
        // the flag's contract, so fail hard instead
        fprintf(stderr, "error: --parent-pid watcher thread failed to start — exiting\n");
        _exit(0);
    }
}

#else // POSIX

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>

#ifdef __linux__
#include <poll.h>
#include <sched.h>
#include <sys/syscall.h>
#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434   // one number on every architecture since Linux 5.3
#endif
#endif

#include <sys/mman.h>
#include <sys/resource.h>
#ifdef __linux__
#include <sys/sysmacros.h>   // major, minor (R3.41's readahead window)
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach-o/dyld.h>
#endif

#ifdef __linux__
// The readahead one MADV_WILLNEED call gets on Linux (R3.41): the kernel clamps
// a hint to the device's max(io_pages, ra_pages) (force_page_cache_ra), so a
// longer range is hinted one window at a time. The window is read from sysfs
// for each mapped file's device and the smallest one seen is kept, so no hint
// is ever wider than a device it may land on; 0 until a file is mapped.
static _Atomic size_t g_willneed_window;

static size_t block_queue_kb(unsigned maj, unsigned mnr, const char *knob) {
    char path[96];
    // a partition has no queue of its own: its parent's is one level up
    static const char *const fmt[] = { "/sys/dev/block/%u:%u/queue/%s",
                                       "/sys/dev/block/%u:%u/../queue/%s" };
    for (int i = 0; i < 2; i++) {
        snprintf(path, sizeof path, fmt[i], maj, mnr, knob);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        unsigned long kb = 0;
        int ok = fscanf(f, "%lu", &kb) == 1;
        fclose(f);
        if (ok) return (size_t)kb;
    }
    return 0;
}

static void willneed_window_note(const struct stat *st) {
    unsigned maj = major(st->st_dev), mnr = minor(st->st_dev);
    size_t ra = block_queue_kb(maj, mnr, "read_ahead_kb");
    size_t io = block_queue_kb(maj, mnr, "max_sectors_kb");
    size_t kb = ra > io ? ra : io;
    size_t w = (kb ? kb : 128) << 10;   // no block device (tmpfs, overlay): the kernel's default
    size_t cur = g_willneed_window;
    while ((cur == 0 || w < cur) &&
           !atomic_compare_exchange_weak(&g_willneed_window, &cur, w)) {
    }
}

size_t plat_willneed_window(void) {
    size_t w = g_willneed_window;
    return w ? w : (size_t)128 << 10;
}
#endif

void *plat_mmap_ro(const char *path, size_t *size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0) {
        close(fd);
        return NULL;
    }
#ifdef __linux__
    willneed_window_note(&st);
#endif
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); // the mapping persists after close
    if (p == MAP_FAILED) return NULL;
    *size = (size_t)st.st_size;
    return p;
}

// madvise(MADV_WILLNEED) turns "fault this in 16 KB at a time, synchronously"
// into readahead the kernel issues at once. That is the point on a model larger
// than RAM: a 3.35 MB expert costs ~200 faults through the fault path and a few
// large requests through this one. Advisory and best-effort by definition: if
// the kernel ignores it the weights still read correctly, just slower, so no
// output can depend on whether it worked.
//
// Linux takes it only in a form it accepts (R3.41, measured on 6.12):
// - The start must be page-aligned, or madvise fails with EINVAL. GGUF tensor
//   data is 32-byte aligned, so an expert's start almost never is: before this
//   fix every expert hint was refused, and --moe-prefetch on did nothing.
//   The start is truncated to its page.
// - One call reads only the device's readahead window (2 MB on the lab's
//   volume, 128 KB by default), so a 4.41 MB gpt-oss expert came in 47.6%. The
//   range is hinted one window at a time (plat_willneed_window).
// macOS is left as it was: it was measured as a win there, and its madvise is
// not known to refuse an unaligned start.
void plat_willneed(const void *addr, size_t len) {
    if (!addr || !len) return;
#ifdef __linux__
    static _Atomic long page;
    long ps = page;
    if (!ps) page = ps = sysconf(_SC_PAGESIZE);
    uintptr_t lo = (uintptr_t)addr & ~((uintptr_t)ps - 1), hi = (uintptr_t)addr + len;
    size_t w = plat_willneed_window();
    for (uintptr_t a = lo; a < hi; a += w)
        (void)madvise((void *)a, hi - a < w ? (size_t)(hi - a) : w, MADV_WILLNEED);
#else
    (void)madvise((void *)(uintptr_t)addr, len, MADV_WILLNEED);
#endif
}

bool plat_willneed_available(void) { return true; }   // madvise is always there

// MADV_DONTNEED on a PROT_READ, MAP_PRIVATE file mapping (plat_mmap_ro's)
// drops the pages from this process; the next read maps the file's bytes
// again. On a private mapping that had been WRITTEN it would discard the
// writes, which is why this is only ever for plat_mmap_ro's read-only maps.
void plat_release_pages(const void *addr, size_t len) {
    long ps = sysconf(_SC_PAGESIZE);
    if (!addr || !len || ps <= 0) return;
    uintptr_t psz = (uintptr_t)ps;
    uintptr_t lo = ((uintptr_t)addr + psz - 1) & ~(psz - 1);
    uintptr_t hi = ((uintptr_t)addr + len) & ~(psz - 1);
    if (hi > lo) (void)madvise((void *)lo, hi - lo, MADV_DONTNEED);
}

void plat_munmap(void *p, size_t size) {
    if (p) munmap(p, size);
}

// The CPUs this process may run on, not the machine's: under taskset, a
// cgroup cpuset or a container the two differ, and a thread default sized
// from the machine oversubscribes the mask. Measured 2026-10-07 on a 128-cpu
// box with a 32-cpu mask (16 cores + SMT siblings): 32 pool threads decoded
// Qwen3.8-Flash-Next at 1.7 tok/s where 16 gave 9.3, a spinning sibling
// starving the straggler at every barrier. A CFS quota (cpu.max) narrows
// nothing here: the mask still lists every CPU, and plat_capacity_at applies
// the quota on top.
static int machine_cpu_count(void) {
#ifdef __linux__
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        int n = CPU_COUNT(&mask);
        if (n > 0) return n;
    }
#endif
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}

int plat_default_thread_count(void) {
    int nc = plat_cpu_count();
#ifdef __APPLE__
    bool perf_known = false;
    int perf = 0;
    size_t len = sizeof(perf);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &len, NULL, 0) == 0 &&
        len == sizeof(perf) && perf > 0) {
        // never more than the CPUs this process may use; on a Mac the
        // performance cores are always the fewer, so this only binds where
        // a limit has narrowed plat_cpu_count below them
        if (perf < nc) nc = perf;
        perf_known = true;
    }
#endif
#ifdef __APPLE__
    // Asymmetric cores: the performance cores ARE the answer, so the halving
    // rule is skipped. The ceiling still applies, though no Apple part reaches
    // it today.
    if (perf_known) {
        int n = nc > PLAT_THREAD_DEFAULT_MAX ? PLAT_THREAD_DEFAULT_MAX : nc;
        return n > 0 ? n : 1;
    }
#endif
    return plat_thread_default_for(nc);
}

static uint64_t machine_ram_bytes(void) {
    long pages = sysconf(_SC_PHYS_PAGES);
    long psz   = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && psz > 0 ? (uint64_t)pages * (uint64_t)psz : 0;
}

bool plat_mlock(void *p, size_t size) {
    if (!p || !size) return false;
    return mlock(p, size) == 0;
}

void plat_munlock(void *p, size_t size) {
    if (p && size) munlock(p, size);
}

size_t plat_page_size(void) {
    long ps = sysconf(_SC_PAGESIZE);
    return ps > 0 ? (size_t)ps : 0;
}

double plat_resident_fraction(const void *p, size_t size) {
    if (!p || !size) return -1.0;
    long psz = sysconf(_SC_PAGE_SIZE);
    if (psz <= 0) return -1.0;
    size_t pages = (size + (size_t)psz - 1) / (size_t)psz;
    // Walk the mapping in windows rather than allocating one byte per page: a
    // 5 GB model is 1.28M pages, and asking about residency should not itself
    // be a megabyte of allocation.
    enum { WINDOW = 64u << 10 };            // pages described per call
    unsigned char *vec = malloc(WINDOW);
    if (!vec) return -1.0;
    size_t resident = 0, done = 0;
    while (done < pages) {
        size_t n = pages - done < WINDOW ? pages - done : WINDOW;
        const char *base = (const char *)p + done * (size_t)psz;
        size_t span = n * (size_t)psz;
        if (span > size - done * (size_t)psz) span = size - done * (size_t)psz;
#ifdef __APPLE__
        if (mincore((void *)(uintptr_t)base, span, (char *)vec) != 0) {
#else
        if (mincore((void *)(uintptr_t)base, span, vec) != 0) {
#endif
            free(vec);
            return -1.0;
        }
        for (size_t i = 0; i < n; i++) if (vec[i] & 1) resident++;
        done += n;
    }
    free(vec);
    return pages ? (double)resident / (double)pages : -1.0;
}

uint64_t plat_major_faults(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
    return (uint64_t)ru.ru_majflt;
}

const char *plat_page_fault_counter(void) { return "major"; }

bool plat_gpu_os_budget(const unsigned char luid[8], uint64_t *budget,
                        uint64_t *usage) {
    (void)luid; (void)budget; (void)usage;
    return false;   // no OS video-memory budget on POSIX
}

bool plat_gpu_os_vidmem(const unsigned char luid[8], uint64_t *budget,
                        uint64_t *usage, uint64_t *nonlocal) {
    (void)luid; (void)budget; (void)usage; (void)nonlocal;
    return false;   // as plat_gpu_os_budget: WDDM only
}

// getrusage gives the PEAK on both, and disagrees about units: ru_maxrss is
// kilobytes on Linux and bytes on the BSDs including macOS. Getting that wrong
// is a factor of 1024 in a number an operator sizes machines with, so the two
// are spelled out rather than assumed.
uint64_t plat_proc_peak_rss_bytes(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#ifdef __linux__
    return (uint64_t)ru.ru_maxrss * 1024ull;
#else
    return (uint64_t)ru.ru_maxrss;
#endif
}

// CURRENT resident set, which getrusage does not carry at all.
uint64_t plat_proc_rss_bytes(void) {
#ifdef __APPLE__
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) != KERN_SUCCESS) return 0;
    return (uint64_t)info.resident_size;
#elif defined(__linux__)
    // statm field 2 is resident pages. /proc is the only portable-enough
    // source; a kernel without it reports 0 rather than a guess.
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long long total = 0, resident = 0;
    int got = fscanf(f, "%llu %llu", &total, &resident);
    fclose(f);
    if (got != 2) return 0;
    long psz = sysconf(_SC_PAGESIZE);
    return psz > 0 ? (uint64_t)resident * (uint64_t)psz : 0;
#else
    return plat_proc_peak_rss_bytes();   // best available; peak >= current
#endif
}

static uint64_t machine_ram_available_bytes(void) {
#ifdef __APPLE__
    // free + inactive + purgeable is what the kernel can hand back without
    // swapping; vm.page_free_count alone reads near zero on a healthy Mac and
    // would make every model look unloadable.
    mach_port_t host = mach_host_self();
    vm_size_t psz = 0;
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_page_size(host, &psz) != KERN_SUCCESS) return 0;
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &count)
        != KERN_SUCCESS) return 0;
    uint64_t pages = (uint64_t)vm.free_count + vm.inactive_count + vm.purgeable_count;
    return pages * (uint64_t)psz;
#else
    // MemAvailable is the kernel's own estimate of what a new allocation can
    // get without swapping. MemFree is not the same thing and is usually much
    // smaller, since the page cache is counted as used.
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof line, f)) {
            unsigned long long kb;
            if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
                fclose(f);
                return (uint64_t)kb * 1024u;
            }
        }
        fclose(f);
    }
    long avail = sysconf(_SC_AVPHYS_PAGES), psz = sysconf(_SC_PAGE_SIZE);
    return avail > 0 && psz > 0 ? (uint64_t)avail * (uint64_t)psz : 0;
#endif
}

bool plat_file_readable(const char *path) {
    return access(path, R_OK) == 0;
}

double plat_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void plat_sleep_ms(int ms) {
    if (ms <= 0) return;
    struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

long plat_pid_self(void) {
    return (long)getpid();
}

bool plat_is_terminal(int fd) {
    return isatty(fd) == 1;
}

bool plat_pid_alive(long pid) {
    if (pid <= 0) return false;
    // EPERM means the process exists but is owned by another user: alive, and
    // its reservation is still real. Only ESRCH proves it is gone.
    if (kill((pid_t)pid, 0) != 0 && errno == ESRCH) return false;
    // A zombie answers kill(pid, 0) too — it is a process-table entry, not a
    // process. It holds no VRAM, no weights and no context, and no signal can
    // move it further; only its parent's wait() can, and that may never come
    // (a supervisor that does not reap, a container whose PID 1 is not an
    // init). This function's whole job is deciding whether a dead owner's
    // record can be reaped, so counting a zombie alive keeps a reservation
    // standing for a process that has already died, for an unbounded time.
#ifdef __APPLE__
    struct kinfo_proc kp;
    size_t len = sizeof kp;
    int mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_PID, (int)pid };
    if (sysctl(mib, 4, &kp, &len, NULL, 0) == 0 && len >= sizeof kp &&
        kp.kp_proc.p_stat == SZOMB)
        return false;
#elif defined(__linux__)
    char sp[64], buf[512];
    snprintf(sp, sizeof sp, "/proc/%ld/stat", pid);
    FILE *f = fopen(sp, "rb");
    if (f) {
        size_t rn = fread(buf, 1, sizeof buf - 1, f);
        fclose(f);
        buf[rn] = 0;
        // state is the field after the parenthesised comm, which can itself
        // contain spaces and brackets — so scan from the LAST ')'
        char *rp = strrchr(buf, ')');
        if (rp && rp[1] == ' ' && (rp[2] == 'Z' || rp[2] == 'X')) return false;
    }
#endif
    return true;
}

bool plat_pid_start_time(long pid, uint64_t *out) {
#ifdef __linux__
    if (pid <= 0) return false;
    char path[64];
    snprintf(path, sizeof(path), "/proc/%ld/stat", pid);
    FILE *f = fopen(path, "r");
    if (!f) return false;
    char buf[2048];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (n == 0) return false;
    buf[n] = 0;
    // field 22 is starttime in clock ticks since boot. comm (field 2) is
    // parenthesised and may itself contain spaces, so scan from the LAST ')'.
    char *p = strrchr(buf, ')');
    if (!p) return false;
    p++;
    for (int field = 3; field <= 21; field++) {
        while (*p == ' ') p++;
        while (*p && *p != ' ') p++;
        if (!*p) return false;
    }
    unsigned long long ticks = strtoull(p, NULL, 10);
    long hz = sysconf(_SC_CLK_TCK);
    if (hz <= 0) return false;
    // relative to boot is enough: entries only ever compare against themselves
    *out = (uint64_t)(ticks / (unsigned long long)hz);
    return true;
#elif defined(__APPLE__)
    if (pid <= 0) return false;
    struct kinfo_proc kp;
    size_t len = sizeof(kp);
    int mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_PID, (int)pid };
    if (sysctl(mib, 4, &kp, &len, NULL, 0) != 0 || len < sizeof(kp)) return false;
    *out = (uint64_t)kp.kp_proc.p_starttime.tv_sec;
    return true;
#else
    (void)pid; (void)out;
    return false;   // other BSDs: pid liveness alone (see plat_pid_alive)
#endif
}

const char *plat_runtime_dir(void) {
    const char *d = getenv("XDG_RUNTIME_DIR");
    if (d && *d) return d;
    if ((d = getenv("TMPDIR")) && *d) return d;
    return "/tmp";
}

bool plat_file_rmw(const char *path, plat_rmw_fn fn, void *ud) {
    // O_NOFOLLOW: the path can live in a world-writable fallback dir (/tmp)
    // under a predictable name, so a planted symlink must not redirect the
    // write. Refusing (best-effort accounting) beats following it.
    int fd = open(path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) return false;

    // flock is advisory-by-convention here: every writer of this file goes
    // through plat_file_rmw. Some network filesystems refuse it; a failure is
    // not fatal, because an unlockable filesystem must not stop a runner.
    bool locked = flock(fd, LOCK_EX) == 0;

    // A file that exists but cannot be read back — oversized, unreadable, or
    // malloc failed — fails the whole rmw. Running fn against an empty view
    // and writing that back would truncate every other process's entries.
    char *buf = NULL;
    size_t len = 0;
    struct stat st;
    bool readable = fstat(fd, &st) == 0 && st.st_size < (1 << 20);
    if (readable && st.st_size > 0) {
        len = (size_t)st.st_size;
        buf = malloc(len + 1);
        size_t off = 0;
        while (buf && off < len) {
            ssize_t got = pread(fd, buf + off, len - off, (off_t)off);
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) { readable = false; break; }
            off += (size_t)got;
        }
        if (!buf) readable = false;
        if (readable) buf[len] = 0;
    }
    if (!readable) {
        free(buf);
        if (locked) flock(fd, LOCK_UN);
        close(fd);
        return false;
    }

    char *out = fn(buf ? buf : "", len, ud);
    free(buf);

    bool written = true;
    if (out) {
        size_t n = strlen(out);
        const char *inject = getenv("RUNNER_FILE_RMW_WRITE_FAIL");
        if (inject && *inject && strcmp(inject, "0") != 0) written = false;
        size_t off = 0;
        while (written && off < n) {
            ssize_t w = pwrite(fd, out + off, n - off, (off_t)off);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) { written = false; break; }
            off += (size_t)w;
        }
        // Preserve the old tail until the full replacement has landed.
        if (written && ftruncate(fd, (off_t)n) != 0) written = false;
        // Only live-process VRAM accounting and disposable yield sentinels
        // use this path. Readers tolerate torn rows and reap stale owners;
        // power loss ends the owners too. An fsync under the lock buys no
        // useful durability, and cannot make an in-place update crash-atomic.
        free(out);
    }

    if (locked) flock(fd, LOCK_UN);
    close(fd);
    return written;
}

// The watched process, set once before the watcher thread starts.
static long parent_watched = 0;
#ifdef __linux__
static int parent_pidfd = -1;
#endif

static void *parent_poll(void *arg) {
    (void)arg;
    long pid = parent_watched;
#ifdef __linux__
    // A pidfd becomes readable when the watched PROCESS has exited, at once
    // and with no window for its pid to be reused. This replaced
    // PR_SET_PDEATHSIG (R4.12.25), which the kernel fires when the THREAD
    // that forked the Runner exits: a supervisor that launched from a
    // short-lived thread had its Runner killed seconds after it was ready,
    // and it fired on the direct parent even when --parent-pid named a
    // grandparent. An unexpected poll error falls through to the 2 s poll.
    if (parent_pidfd >= 0) {
        struct pollfd pf = { .fd = parent_pidfd, .events = POLLIN };
        int r;
        while ((r = poll(&pf, 1, -1)) < 0 && errno == EINTR) {}
        if (r > 0) {
            fprintf(stderr, "parent %ld exited — shutting down\n", pid);
            _exit(0);
        }
    }
#endif
    for (;;) {
        struct timespec ts = { 2, 0 };
        nanosleep(&ts, NULL);
        if (kill((pid_t)pid, 0) != 0 && errno == ESRCH) {
            fprintf(stderr, "parent %ld exited — shutting down\n", pid);
            _exit(0);
        }
    }
    return NULL;
}

void plat_parent_watch(long pid) {
    if (pid <= 0) return;
    parent_watched = pid;
#ifdef __linux__
    // Linux 5.3 and later; an older kernel (ENOSYS) or a refusal keeps the
    // 2 s poll alone. A pid that is already gone is refused here, as on
    // Windows: the flag's contract is to never run unwatched.
    long fd = syscall(SYS_pidfd_open, (pid_t)pid, 0);
    if (fd < 0 && errno == ESRCH) {
        fprintf(stderr, "error: --parent-pid %ld is not running — exiting\n", pid);
        _exit(0);
    }
    parent_pidfd = fd >= 0 ? (int)fd : -1;
#endif
    pthread_t th;
    pthread_attr_t at;
    bool attr_ok = pthread_attr_init(&at) == 0;
    if (!attr_ok || pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED) != 0 ||
        pthread_create(&th, &at, parent_poll, NULL) != 0) {
        // this runs pre-model-load (no CUDA threads exist yet), so a plain
        // _exit is fine — but continuing unwatched would silently break
        // the flag's contract, so fail hard instead
        fprintf(stderr, "error: --parent-pid watcher thread failed to start — exiting\n");
        if (attr_ok) pthread_attr_destroy(&at);
        _exit(0);
    }
    pthread_attr_destroy(&at);
}

#endif

// ------------------------------------------------- UTF-8 at the OS boundary
//
// See compat.h. Windows converts at each door; POSIX passes bytes through,
// so only the calls that are not plain libc (realpath's allocation, the
// directory walk, the spawn) have bodies there.

#ifdef _WIN32
#include <direct.h>
#include <errno.h>
#include <fcntl.h>
#include <process.h>
#include <sys/stat.h>

// UTF-8 <-> UTF-16, malloc'd and freed here; NULL when out of memory. A
// byte sequence that is not UTF-8 becomes U+FFFD, as it does in the -A APIs
// under the UTF-8 code page.
static wchar_t *utf8_to_wide(const char *s) {
    if (!s) { errno = EINVAL; return NULL; }
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    wchar_t *w = n > 0 ? malloc((size_t)n * sizeof *w) : NULL;
    if (!w || MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n) != n) {
        free(w);
        errno = ENOMEM;
        return NULL;
    }
    return w;
}

static char *wide_to_utf8(const wchar_t *w) {
    if (!w) { errno = EINVAL; return NULL; }
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    char *s = n > 0 ? malloc((size_t)n) : NULL;
    if (!s || WideCharToMultiByte(CP_UTF8, 0, w, -1, s, n, NULL, NULL) != n) {
        free(s);
        errno = ENOMEM;
        return NULL;
    }
    return s;
}

// free() that leaves errno and the last Win32 error as the wide call set
// them: callers report strerror(errno), and some read GetLastError()
// (ERROR_ALREADY_EXISTS after a mkdir), from the call, not from cleanup
static void free_keep_errno(void *a, void *b) {
    int e = errno;
    DWORD le = GetLastError();
    free(a);
    free(b);
    errno = e;
    SetLastError(le);
}

FILE *plat_fopen(const char *path, const char *mode) {
    wchar_t *wp = utf8_to_wide(path), *wm = wp ? utf8_to_wide(mode) : NULL;
    FILE *f = wm ? _wfopen(wp, wm) : NULL;
    free_keep_errno(wp, wm);
    return f;
}

FILE *plat_freopen(const char *path, const char *mode, FILE *stream) {
    wchar_t *wp = utf8_to_wide(path), *wm = wp ? utf8_to_wide(mode) : NULL;
    FILE *f = wm ? _wfreopen(wp, wm, stream) : NULL;
    free_keep_errno(wp, wm);
    return f;
}

int plat_open(const char *path, int flags, int mode) {
    wchar_t *wp = utf8_to_wide(path);
    int fd = wp ? _wopen(wp, flags, mode) : -1;
    free_keep_errno(wp, NULL);
    return fd;
}

int plat_remove(const char *path) {
    wchar_t *wp = utf8_to_wide(path);
    int r = wp ? _wremove(wp) : -1;
    free_keep_errno(wp, NULL);
    return r;
}

int plat_rename(const char *from, const char *to) {
    wchar_t *wf = utf8_to_wide(from), *wt = wf ? utf8_to_wide(to) : NULL;
    int r = wt ? _wrename(wf, wt) : -1;
    free_keep_errno(wf, wt);
    return r;
}

int plat_mkdir(const char *path, int mode) {
    (void)mode;
    wchar_t *wp = utf8_to_wide(path);
    int r = wp ? _wmkdir(wp) : -1;
    free_keep_errno(wp, NULL);
    return r;
}

int plat_rmdir(const char *path) {
    wchar_t *wp = utf8_to_wide(path);
    int r = wp ? _wrmdir(wp) : -1;
    free_keep_errno(wp, NULL);
    return r;
}

void *plat_create_file(const char *path, unsigned long access, unsigned long share,
                       void *security, unsigned long disposition, unsigned long flags) {
    wchar_t *wp = utf8_to_wide(path);
    HANDLE h = wp ? CreateFileW(wp, access, share, (SECURITY_ATTRIBUTES *)security,
                                disposition, flags, NULL)
                  : INVALID_HANDLE_VALUE;
    free_keep_errno(wp, NULL);
    return h;
}

int plat_stat(const char *path, plat_stat_t *st) {
    wchar_t *wp = utf8_to_wide(path);
    if (!wp) return -1;
    // the C runtime's stat refuses "dir\" for anything but a root; MinGW's
    // POSIX stat() strips the separators first, and the callers were
    // written against that one
    size_t n = wcslen(wp);
    while (n > 1 && (wp[n - 1] == L'/' || wp[n - 1] == L'\\') &&
           !(n == 3 && wp[1] == L':'))
        wp[--n] = 0;
    int r = _wstat64(wp, st);
    free_keep_errno(wp, NULL);
    return r;
}

// The C runtime's narrow environment table (what getenv reads, and what
// model.c walks for RUNNER_* routes), rebuilt in UTF-8 from the wide table
// when the two disagree. They agree whenever the code page is UTF-8; the
// wide one is what the OS holds. The old table is left alone, not freed:
// main()'s envp may still point at it.
static void env_narrow_from_wide(void) {
    wchar_t **w = _wenviron;
    char **old = _environ;
    if (!w) return;
    size_t n = 0;
    while (w[n]) n++;
    char **u = calloc(n + 1, sizeof *u);
    if (!u) return;
    bool same = old != NULL;
    for (size_t i = 0; i < n; i++) {
        if (!(u[i] = wide_to_utf8(w[i]))) {
            for (size_t k = 0; k < i; k++) free(u[k]);
            free(u);
            return;
        }
        if (same && (!old[i] || strcmp(old[i], u[i]) != 0)) same = false;
    }
    if (same && !old[n]) {
        for (size_t k = 0; k < n; k++) free(u[k]);
        free(u);
        return;
    }
    _environ = u;
}

int plat_setenv(const char *name, const char *value) {
    wchar_t *wn = utf8_to_wide(name), *wv = wn ? utf8_to_wide(value) : NULL;
    // _wputenv_s sets the wide table and the OS block; the narrow copy it
    // derives is in the ANSI code page, so it is rebuilt after
    errno_t e = wv ? _wputenv_s(wn, wv) : ENOMEM;
    free(wn);
    free(wv);
    if (e) { errno = e; return -1; }
    env_narrow_from_wide();
    return 0;
}

static bool g_console_eof;   // a Ctrl+Z was read: every later read is the end

char *plat_console_read_line(bool *oom) {
    if (g_console_eof) return NULL;
    HANDLE in = (HANDLE)_get_osfhandle(0);
    size_t cap = 256, n = 0;
    wchar_t *w = malloc(cap * sizeof *w);
    if (!w) { *oom = true; return NULL; }
    bool ended = false;     // the line's newline arrived
    for (;;) {
        if (cap - n < 129) {
            if (cap > ((size_t)1 << 28)) { free(w); *oom = true; return NULL; }
            wchar_t *g = realloc(w, 2 * cap * sizeof *w);
            if (!g) { free(w); *oom = true; return NULL; }
            w = g;
            cap *= 2;
        }
        DWORD got = 0;
        if (!ReadConsoleW(in, w + n, 128, &got, NULL) || got == 0) break;
        n += got;
        if (w[n - 1] == L'\n') { ended = true; break; }
    }
    if (ended) {
        n--;
        if (n && w[n - 1] == L'\r') n--;
    }
    // Ctrl+Z is the end of input on a console; what was typed before it on
    // the same line is the last line
    for (size_t i = 0; i < n; i++)
        if (w[i] == 0x1A) { n = i; ended = false; g_console_eof = true; break; }
    if (n == 0 && !ended) { free(w); return NULL; }
    w[n] = 0;
    char *u = wide_to_utf8(w);
    free(w);
    if (!u) *oom = true;
    return u;
}

// The C runtime's own split of the wide command line, the one it made of
// the narrow line for main(): __wgetmainargs is what MinGW's startup calls
// for wmain(), and _dowildcard is the wildcard setting it used for main().
typedef struct { int newmode; } plat_startupinfo;
extern int __wgetmainargs(int *argc, wchar_t ***argv, wchar_t ***env, int glob,
                          plat_startupinfo *si);
extern int _dowildcard;

static UINT g_console_out_cp;
static void console_out_cp_restore(void) { SetConsoleOutputCP(g_console_out_cp); }

void plat_utf8_startup(int *argc, char ***argv) {
    int wargc = 0;
    wchar_t **wargv = NULL, **wenv = NULL;
    plat_startupinfo si = { 0 };
    // the same split of the same line, so the counts agree; a disagreement
    // would mean this is not the line main() was given, and then the
    // narrow argv stays
    if (__wgetmainargs(&wargc, &wargv, &wenv, _dowildcard, &si) == 0 && wargv &&
        wargc == *argc) {
        char **u = calloc((size_t)wargc + 1, sizeof *u);
        bool ok = u != NULL;
        for (int i = 0; ok && i < wargc; i++) ok = (u[i] = wide_to_utf8(wargv[i])) != NULL;
        if (ok) {
            *argv = u;
            __argv = u;
        } else if (u) {
            for (int i = 0; i < wargc; i++) free(u[i]);
            free(u);
        }
    }
    env_narrow_from_wide();
    // The engine writes UTF-8. A console draws the bytes it is given in its
    // output code page (437 or 850 by default), which turned every
    // non-ASCII character into two or three wrong ones; a pipe or a file
    // takes the bytes as they are. The console is shared with the shell
    // that started us, so its code page is put back at exit.
    if (plat_is_terminal(1) || plat_is_terminal(2)) {
        UINT cp = GetConsoleOutputCP();
        if (cp && cp != CP_UTF8 && SetConsoleOutputCP(CP_UTF8)) {
            g_console_out_cp = cp;
            atexit(console_out_cp_restore);
        }
    }
}

bool plat_realpath(const char *path, char *out, size_t cap) {
    wchar_t *wp = utf8_to_wide(path);
    wchar_t *abs = wp ? _wfullpath(NULL, wp, 0) : NULL;
    char *r = abs ? wide_to_utf8(abs) : NULL;
    bool ok = r && strlen(r) < cap;
    if (ok) memcpy(out, r, strlen(r) + 1);
    free(r);
    free_keep_errno(wp, abs);
    return ok;
}

struct plat_dir {
    HANDLE h;
    bool pending;          // fd holds an entry plat_readdir has not returned
    WIN32_FIND_DATAW fd;
    char *name;
};

int plat_stat_nofollow(const char *path, plat_stat_t *st, bool *link) {
    wchar_t *wp = utf8_to_wide(path);
    DWORD a = wp ? GetFileAttributesW(wp) : INVALID_FILE_ATTRIBUTES;
    DWORD e = GetLastError();
    free(wp);
    *link = false;
    if (a == INVALID_FILE_ATTRIBUTES) {
        errno = e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND ? ENOENT
              : e == ERROR_ACCESS_DENIED ? EACCES : EIO;
        return -1;
    }
    *link = (a & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
    if (*link) { memset(st, 0, sizeof *st); return 0; }
    return plat_stat(path, st);
}

plat_dir *plat_opendir(const char *path) {
    size_t n = strlen(path);
    char *pat = malloc(n + 3);
    if (!pat) { errno = ENOMEM; return NULL; }
    memcpy(pat, path, n);
    if (n && pat[n - 1] != '/' && pat[n - 1] != '\\') pat[n++] = '\\';
    pat[n++] = '*';
    pat[n] = 0;
    wchar_t *wp = utf8_to_wide(pat);
    free(pat);
    plat_dir *d = wp ? calloc(1, sizeof *d) : NULL;
    if (!d) { free(wp); errno = ENOMEM; return NULL; }
    d->h = FindFirstFileW(wp, &d->fd);
    free(wp);
    if (d->h == INVALID_HANDLE_VALUE) {
        DWORD e = GetLastError();
        free(d);
        errno = e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND ? ENOENT
              : e == ERROR_DIRECTORY ? ENOTDIR
              : e == ERROR_ACCESS_DENIED ? EACCES : EIO;
        return NULL;
    }
    d->pending = true;
    return d;
}

const char *plat_readdir(plat_dir *d) {
    if (!d->pending && !FindNextFileW(d->h, &d->fd)) {
        errno = GetLastError() == ERROR_NO_MORE_FILES ? 0 : EIO;
        return NULL;
    }
    d->pending = false;
    free(d->name);
    d->name = wide_to_utf8(d->fd.cFileName);
    if (!d->name) errno = ENOMEM;
    return d->name;
}

void plat_closedir(plat_dir *d) {
    if (!d) return;
    FindClose(d->h);
    free(d->name);
    free(d);
}

// Quote one argument by the C runtime's rules: wrap in double quotes, double
// the backslashes that precede a quote, escape the quote.
static char *win_quote(const char *arg) {
    bool plain = *arg != 0;
    for (const char *p = arg; *p && plain; p++)
        if (*p == ' ' || *p == '\t' || *p == '"') plain = false;
    if (plain) return strdup(arg);
    size_t n = strlen(arg);
    char *q = malloc(2 * n + 3), *w = q;
    if (!q) return NULL;
    *w++ = '"';
    for (size_t i = 0; i < n; i++) {
        size_t bs = 0;
        while (i < n && arg[i] == '\\') { bs++; i++; }
        if (i == n) { for (size_t k = 0; k < 2 * bs; k++) *w++ = '\\'; break; }
        if (arg[i] == '"') { for (size_t k = 0; k < 2 * bs + 1; k++) *w++ = '\\'; *w++ = '"'; }
        else { for (size_t k = 0; k < bs; k++) *w++ = '\\'; *w++ = arg[i]; }
    }
    *w++ = '"'; *w = 0;
    return q;
}

int plat_spawn_wait_stdin(const char *const argv[], const char *input, size_t n) {
    if (!input) return plat_spawn_wait(argv);
    // The input goes into a pipe before the child starts (it fits the pipe,
    // so the write cannot block), the pipe's read end becomes descriptor 0
    // for the length of the spawn, which the child inherits, and the
    // caller's own standard input comes back after it.
    int fds[2];
    if (n > PLAT_SPAWN_STDIN_MAX || _pipe(fds, 2 * PLAT_SPAWN_STDIN_MAX, _O_BINARY) != 0)
        return -1;
    int wrote = n ? _write(fds[1], input, (unsigned)n) : 0;
    _close(fds[1]);
    int saved = _dup(0);   // -1 when the process has no standard input
    if (wrote != (int)n || _dup2(fds[0], 0) != 0) {
        _close(fds[0]);
        if (saved >= 0) _close(saved);
        return -1;
    }
    _close(fds[0]);
    int rc = plat_spawn_wait(argv);
    if (saved >= 0) { _dup2(saved, 0); _close(saved); }
    else _close(0);
    return rc;
}

int plat_spawn_wait(const char *const argv[]) {
    int n = 0;
    while (argv[n]) n++;
    wchar_t **w = calloc((size_t)n + 1, sizeof *w);
    wchar_t *file = w && n ? utf8_to_wide(argv[0]) : NULL;
    bool ok = file != NULL;
    for (int i = 0; i < n && ok; i++) {
        char *q = win_quote(argv[i]);
        ok = q && (w[i] = utf8_to_wide(q)) != NULL;
        free(q);
    }
    intptr_t rc = ok ? _wspawnvp(_P_WAIT, file, (const wchar_t *const *)w) : -1;
    for (int i = 0; w && i < n; i++) free(w[i]);
    free(w);
    free(file);
    return rc == -1 ? -1 : (int)rc;
}

static bool spawn_detached(const char *executable, const char *arg) {
    char *a = win_quote(executable), *b = win_quote(arg);
    size_t n = a && b ? strlen(a) + strlen(b) + 2 : 0;
    char *cmd = n ? malloc(n) : NULL;
    if (cmd) snprintf(cmd, n, "%s %s", a, b);
    wchar_t *file = utf8_to_wide(executable);
    wchar_t *wide = cmd ? utf8_to_wide(cmd) : NULL;
    STARTUPINFOW si = {0};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi;
    bool ok = file && wide && CreateProcessW(file, wide, NULL, NULL, FALSE,
                        DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, NULL, NULL, &si, &pi);
    if (ok) { CloseHandle(pi.hThread); CloseHandle(pi.hProcess); }
    free(a); free(b); free(cmd); free(file); free(wide);
    return ok;
}

#else // POSIX

#include <dirent.h>
#include <sys/wait.h>

void plat_utf8_startup(int *argc, char ***argv) { (void)argc; (void)argv; }

bool plat_realpath(const char *path, char *out, size_t cap) {
    // realpath's own allocation: glibc's fortified build aborts on a caller
    // buffer smaller than PATH_MAX however short the result is
    char *r = realpath(path, NULL);
    bool ok = r && strlen(r) < cap;
    if (ok) memcpy(out, r, strlen(r) + 1);
    free(r);
    return ok;
}

struct plat_dir { DIR *d; };

int plat_stat_nofollow(const char *path, plat_stat_t *st, bool *link) {
    int rc = lstat(path, st);
    *link = rc == 0 && S_ISLNK(st->st_mode);
    return rc;
}

plat_dir *plat_opendir(const char *path) {
    DIR *dp = opendir(path);
    if (!dp) return NULL;
    plat_dir *d = malloc(sizeof *d);
    if (!d) { closedir(dp); errno = ENOMEM; return NULL; }
    d->d = dp;
    return d;
}

const char *plat_readdir(plat_dir *d) {
    errno = 0;
    struct dirent *e = readdir(d->d);
    return e ? e->d_name : NULL;
}

void plat_closedir(plat_dir *d) {
    if (!d) return;
    closedir(d->d);
    free(d);
}

int plat_spawn_wait_stdin(const char *const argv[], const char *input, size_t n) {
    // The input is written into a pipe before the fork: it fits the pipe, so
    // the write cannot block, and with both ends still open here it cannot
    // raise SIGPIPE whatever becomes of the child.
    int p[2] = { -1, -1 };
    if (input) {
        if (n > PLAT_SPAWN_STDIN_MAX || pipe(p) != 0) return -1;
        ssize_t w = n ? write(p[1], input, n) : 0;
        close(p[1]);
        if (w != (ssize_t)n) { close(p[0]); return -1; }
    }
    pid_t pid = fork();
    if (pid < 0) {
        if (input) close(p[0]);
        return -1;
    }
    if (pid == 0) {
        if (input && (dup2(p[0], 0) < 0 || close(p[0]) != 0)) _exit(127);
        execvp(argv[0], (char *const *)argv);
        _exit(127);
    }
    if (input) close(p[0]);
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status) == 127 ? -1 : WEXITSTATUS(status);
    return -1;
}

int plat_spawn_wait(const char *const argv[]) {
    return plat_spawn_wait_stdin(argv, NULL, 0);
}

static void *reap_detached(void *arg) {
    // Only an interruptible child wait, never filesystem I/O. Nothing joins
    // this thread at exit, and a child stuck in unlink is a separate process.
    while (waitpid((pid_t)(intptr_t)arg, NULL, 0) < 0 && errno == EINTR) {}
    return NULL;
}

static bool spawn_detached(const char *executable, const char *arg) {
    long last_fd = sysconf(_SC_OPEN_MAX);
    if (last_fd < 0 || last_fd > INT_MAX) last_fd = 65536;
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        // After a multithreaded fork, use only async-signal-safe operations
        // until exec. Do not carry model handles, leases or output pipes.
        setsid();
        int nullfd = open("/dev/null", O_RDWR);
        if (nullfd < 0) _exit(127);
        for (int i = 0; i < 3; i++) if (dup2(nullfd, i) < 0) _exit(127);
        bool closed = false;
#if defined(__linux__) && defined(SYS_close_range)
        closed = syscall(SYS_close_range, 3u, ~0u, 0u) == 0;
#endif
        if (!closed) for (long fd = 3; fd < last_fd; fd++) close((int)fd);
        char *const argv[] = {(char *)executable, (char *)arg, NULL};
        execv(executable, argv);
        _exit(127);   // exec failure is best-effort: the caller never waits
    }
    pthread_t reaper;
    if (pthread_create(&reaper, NULL, reap_detached, (void *)(intptr_t)pid) == 0)
        pthread_detach(reaper);
    // If reaper allocation failed, the child still runs; the OS reaps it
    // when this process exits. Never turn cleanup into a startup wait.
    return true;
}

#endif

bool plat_spawn_self_detached(const char *arg) {
    char *exe = plat_executable_path();
    bool ok = exe && spawn_detached(exe, arg);
    free(exe);
    return ok;
}

unsigned plat_code_page(void) {
#ifdef _WIN32
    return (unsigned)GetACP();
#else
    return 0;
#endif
}

bool plat_utf8_manifest(void) {
#ifdef _WIN32
    // resource type 24 (RT_MANIFEST), name 1: the manifest the loader reads
    // for an executable (CREATEPROCESS_MANIFEST_RESOURCE_ID)
    HRSRC r = FindResourceW(NULL, MAKEINTRESOURCEW(1), MAKEINTRESOURCEW(24));
    HGLOBAL g = r ? LoadResource(NULL, r) : NULL;
    const char *p = g ? (const char *)LockResource(g) : NULL;
    DWORD n = r ? SizeofResource(NULL, r) : 0;
    if (!p || !n) return false;
    // <activeCodePage ...>UTF-8</activeCodePage>, the element's text being
    // the first thing after the end of its start tag
    static const char tag[] = "activeCodePage";
    static const char val[] = "UTF-8";
    for (DWORD i = 0; i + sizeof tag - 1 <= n; i++) {
        if (memcmp(p + i, tag, sizeof tag - 1) != 0) continue;
        DWORD j = i + (DWORD)sizeof tag - 1;
        while (j < n && p[j] != '>' && p[j] != '<') j++;
        if (j < n && p[j] == '>' && n - (j + 1) >= sizeof val - 1 &&
            _strnicmp(p + j + 1, val, sizeof val - 1) == 0)
            return true;
    }
    return false;
#else
    return false;
#endif
}

bool plat_replace_file(const char *tmp_path, const char *path) {
#ifdef _WIN32
    wchar_t *wt = utf8_to_wide(tmp_path), *wp = wt ? utf8_to_wide(path) : NULL;
    bool ok = wp && MoveFileExW(wt, wp, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
    free(wt);
    free(wp);
    return ok;
#else
    return rename(tmp_path, path) == 0;
#endif
}

bool plat_random(uint8_t *out, size_t n) {
#ifdef _WIN32
    for (size_t i = 0; i < n; i += 4) {
        unsigned v;
        if (rand_s(&v) != 0) return false;
        for (size_t k = 0; k < 4 && i + k < n; k++)
            out[i + k] = (uint8_t)(v >> (8 * k));
    }
    return true;
#else
    FILE *f = plat_fopen("/dev/urandom", "rb");
    if (!f) return false;
    size_t r = fread(out, 1, n, f);
    fclose(f);
    return r == n;
#endif
}

// The size of an open file, 64-bit on every platform, with the position left
// at the start. ftell() returns long, which is 32 bits on Windows: a file past
// 2 GiB read there as an error (R6.68.5).
bool plat_file_size(FILE *f, uint64_t *out) {
#ifdef _WIN32
    if (_fseeki64(f, 0, SEEK_END) != 0) return false;
    __int64 end = _ftelli64(f);
    if (end < 0 || _fseeki64(f, 0, SEEK_SET) != 0) return false;
#else
    if (fseeko(f, 0, SEEK_END) != 0) return false;
    off_t end = ftello(f);
    if (end < 0 || fseeko(f, 0, SEEK_SET) != 0) return false;
#endif
    *out = (uint64_t)end;
    return true;
}

char *plat_read_file(const char *path, size_t max_bytes, size_t *n_out) {
    FILE *f = plat_fopen(path, "rb");
    if (!f) return NULL;
    uint64_t sz = 0;
    // the size must fit a buffer with its terminator, and the cap when one is set
    if (!plat_file_size(f, &sz) || sz >= (uint64_t)SIZE_MAX ||
        (max_bytes && sz > (uint64_t)max_bytes)) {
        fclose(f);
        return NULL;
    }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)sz, f);
    bool err = ferror(f) != 0;
    fclose(f);
    if (err || got != (size_t)sz) { free(buf); return NULL; }
    buf[got] = 0;
    if (n_out) *n_out = got;
    return buf;
}

// The process's physical footprint, where the operating system keeps one: on
// macOS the kernel's phys_footprint (TASK_VM_INFO), the figure Activity Monitor
// and footprint(1) show. Unlike the resident size it counts the GPU driver's
// memory for this process (Metal buffers, IOAccelerator) and compressed pages:
// a serving process read 13 MB resident at a 638 MB footprint, 598 MB of it
// Metal, so a device-memory leak there showed nowhere else (R6.68.12). False
// elsewhere: Linux and Windows have no one figure that adds device memory to
// RAM, and their GPU memory is reported apart, so the field is left out
// rather than repeating the resident set.
bool plat_proc_phys_footprint_bytes(uint64_t *out) {
#ifdef __APPLE__
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS ||
        count < TASK_VM_INFO_REV1_COUNT)   // phys_footprint arrived in revision 1
        return false;
    *out = (uint64_t)info.phys_footprint;
    return true;
#else
    (void)out;
    return false;
#endif
}

bool plat_mkdir_exist_ok(const char *path) {
    if (plat_mkdir(path, 0755) == 0 || errno == EEXIST) return true;
#ifdef _WIN32
    return GetLastError() == ERROR_ALREADY_EXISTS;
#else
    return false;
#endif
}

const char *plat_path_base(const char *path) {
    const char *b = path;
    for (const char *c = path; *c; c++)
        if (*c == '/' || *c == '\\') b = c + 1;
    return b;
}

char *plat_executable_path(void) {
#ifdef __linux__
    // Opening this magic link reads the running image even when argv[0] was a
    // bare PATH lookup (and remains meaningful after the original is unlinked).
    return strdup("/proc/self/exe");
#elif defined(_WIN32)
    // wide, then UTF-8: the narrow call spells each character the ANSI code
    // page lacks as "?", and the path then names no file
    for (size_t cap = 1024; cap <= (1u << 20); cap *= 2) {
        wchar_t *w = malloc(cap * sizeof *w);
        if (!w) return NULL;
        DWORD n = GetModuleFileNameW(NULL, w, (DWORD)cap);
        char *u = n > 0 && n < cap ? wide_to_utf8(w) : NULL;
        free(w);
        if (u || n == 0 || n < cap) return u;
    }
    return NULL;
#else
    size_t cap = 1024;
    for (;;) {
        if (cap > (1u << 20)) return NULL;
        char *path = malloc(cap);
        if (!path) return NULL;
#if defined(__APPLE__)
        uint32_t needed = (uint32_t)cap;
        if (_NSGetExecutablePath(path, &needed) == 0) {
            char *absolute = realpath(path, NULL);
            if (absolute) { free(path); return absolute; }
            return path;
        }
        free(path);
        cap = needed > cap ? needed : cap * 2;
#else
        free(path);
        return NULL;
#endif
    }
#endif
}

char *plat_executable_dir(void) {
#ifdef __linux__
    // the link's target, not the link: "/proc/self/exe" opens the image but
    // its own directory is /proc/self
    char *p = NULL;
    for (size_t cap = 1024; cap <= (1u << 20) && !p; cap *= 2) {
        char *b = malloc(cap);
        if (!b) return NULL;
        ssize_t n = readlink("/proc/self/exe", b, cap);
        if (n < 0) { free(b); return NULL; }
        if ((size_t)n < cap) { b[n] = '\0'; p = b; }
        else free(b);
    }
#else
    char *p = plat_executable_path();   // absolute on macOS and Windows
#endif
    if (!p) return NULL;
    p[plat_path_base(p) - p] = '\0';
    return p;
}

// ------------------------------------------------------------------ capacity
//
// The cgroup files read here are the kernel's documented interface
// (Documentation/admin-guide/cgroup-v2.rst, cgroup-v1/memory.rst and
// scheduler/sched-bwc.rst): v2 cpu.max is "$MAX $PERIOD" with "max" for no
// quota and memory.max a byte count or "max"; v1 has cpu.cfs_quota_us (-1
// for none) over cpu.cfs_period_us, and memory.limit_in_bytes.

#define CG_PATH_MAX 4096

// One small cgroup file with its trailing whitespace trimmed. False when it
// cannot be opened, which is how an absent level or controller reads.
static bool cg_text(const char *dir, const char *name, char *buf, size_t n) {
    char path[CG_PATH_MAX + 64];
    int w = snprintf(path, sizeof path, "%s/%s", dir, name);
    if (w < 0 || (size_t)w >= sizeof path) return false;
    FILE *f = plat_fopen(path, "r");
    if (!f) return false;
    size_t got = fread(buf, 1, n - 1, f);
    fclose(f);
    buf[got] = '\0';
    while (got && (buf[got - 1] == '\n' || buf[got - 1] == '\r' ||
                   buf[got - 1] == ' ' || buf[got - 1] == '\t'))
        buf[--got] = '\0';
    return true;
}

// A number at least `lo`; false for "max" and for text the kernel does not
// write, which then carries no limit rather than a guessed one.
static bool cg_num(const char *dir, const char *name, uint64_t lo, uint64_t *out) {
    char b[64];
    return cg_text(dir, name, b, sizeof b) && strcmp(b, "max") &&
           parse_u64(b, lo, UINT64_MAX, out);
}

// v1 writes "no limit" as LONG_MAX rounded down to a page
// (9223372036854771712 with 4 KiB pages). No machine has 2^62 bytes, so a
// figure at or above it is that sentinel for every page size.
#define CG1_UNLIMITED (UINT64_C(1) << 62)

// Quota over period in CPUs, rounded up and never below one: 1.5 CPUs of
// bandwidth is work for two threads, and a pool of one would leave the half
// unused.
static int cg_cpus(uint64_t quota, uint64_t period) {
    uint64_t n = quota / period + (quota % period != 0);
    if (n < 1) n = 1;
    return n > (uint64_t)INT_MAX ? INT_MAX : (int)n;
}

static int cg2_quota(const char *dir) {
    char b[96];
    if (!cg_text(dir, "cpu.max", b, sizeof b)) return 0;
    char *sp = strchr(b, ' ');
    if (!sp) return 0;
    *sp = '\0';
    uint64_t q, p;
    if (!strcmp(b, "max") || !parse_u64(b, 1, UINT64_MAX, &q) ||
        !parse_u64(sp + 1, 1, UINT64_MAX, &p)) return 0;
    return cg_cpus(q, p);
}

static int cg1_quota(const char *dir) {
    uint64_t q, p;
    // -1 is no quota, and parse_u64 refuses the sign
    if (!cg_num(dir, "cpu.cfs_quota_us", 1, &q) ||
        !cg_num(dir, "cpu.cfs_period_us", 1, &p)) return 0;
    return cg_cpus(q, p);
}

// What the kernel can take back from this cgroup's own charge without
// swapping: the file LRU pages and reclaimable slab, the terms MemAvailable
// counts for the whole machine. shmem sits on the anon lists and is in
// neither. Without these, the page cache the model file itself leaves
// charged to the container would read as used. v1 has no slab line, and its
// hierarchical total_ keys are preferred because usage_in_bytes counts the
// children too.
static uint64_t cg_reclaimable(const char *dir, int v) {
    char path[CG_PATH_MAX + 64];
    int w = snprintf(path, sizeof path, "%s/memory.stat", dir);
    if (w < 0 || (size_t)w >= sizeof path) return 0;
    FILE *f = plat_fopen(path, "r");
    if (!f) return 0;
    uint64_t v2 = 0, total = 0, local = 0;
    bool have_total = false;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        char *sp = strchr(line, ' ');
        if (!sp) continue;
        *sp = '\0';
        char *val = sp + 1;
        val[strcspn(val, "\r\n")] = '\0';
        uint64_t x, *into = NULL;
        if (!parse_u64(val, 0, UINT64_MAX, &x)) continue;
        if (v == 2) {
            if (!strcmp(line, "active_file") || !strcmp(line, "inactive_file") ||
                !strcmp(line, "slab_reclaimable")) into = &v2;
        } else if (!strcmp(line, "total_active_file") ||
                   !strcmp(line, "total_inactive_file")) {
            into = &total;
            have_total = true;
        } else if (!strcmp(line, "active_file") || !strcmp(line, "inactive_file"))
            into = &local;
        if (into && !checked_u64_add(*into, x, into)) *into = UINT64_MAX;
    }
    fclose(f);
    return v == 2 ? v2 : have_total ? total : local;
}

// The running bound over the levels walked so far: the tightest of each.
typedef struct {
    int      cpus;        // CPUs, 0 = no quota seen
    uint64_t mem_max;     // bytes, 0 = no limit seen
    uint64_t mem_left;    // what the tightest-left level leaves, valid with have_left
    bool     have_left;
} cg_bound;

static void cg_fold(const char *dir, int v, bool cpu, bool mem, cg_bound *b) {
    if (cpu) {
        int n = v == 2 ? cg2_quota(dir) : cg1_quota(dir);
        if (n > 0 && (!b->cpus || n < b->cpus)) b->cpus = n;
    }
    uint64_t max, used;
    if (!mem || !cg_num(dir, v == 2 ? "memory.max" : "memory.limit_in_bytes", 1, &max) ||
        (v == 1 && max >= CG1_UNLIMITED)) return;
    // headroom under this level's limit (siblings' usage included, since an
    // ancestor's charge covers its whole subtree), plus what it can reclaim
    uint64_t left = max;
    if (cg_num(dir, v == 2 ? "memory.current" : "memory.usage_in_bytes", 0, &used)) {
        left = used < max ? max - used : 0;
        uint64_t r = cg_reclaimable(dir, v);
        left = r > max - left ? max : left + r;
    }
    if (!b->mem_max || max < b->mem_max) b->mem_max = max;
    if (!b->have_left || left < b->mem_left) { b->mem_left = left; b->have_left = true; }
}

// Every level from the process's cgroup `self` (a path under the mount such
// as "/system.slice/x.scope", or "" for the mount root) up to the mount
// `base`, since the kernel enforces each ancestor's limit too. A level that
// does not exist under this mount simply has no files.
static void cg_walk(const char *base, const char *self, int v, bool cpu, bool mem,
                    cg_bound *b) {
    char dir[CG_PATH_MAX];
    size_t bl = strlen(base);
    int w = snprintf(dir, sizeof dir, "%s%s", base, self);
    if (w < 0 || (size_t)w >= sizeof dir) {
        if (bl >= sizeof dir) return;
        memcpy(dir, base, bl + 1);
    }
    for (;;) {
        cg_fold(dir, v, cpu, mem, b);
        if (strlen(dir) <= bl) break;
        char *slash = strrchr(dir + bl, '/');
        if (!slash) break;
        *slash = '\0';
    }
}

static bool cg_list_has(const char *list, const char *name) {
    size_t nl = strlen(name);
    for (const char *p = list;;) {
        const char *e = strchr(p, ',');
        size_t len = e ? (size_t)(e - p) : strlen(p);
        if (len == nl && !strncmp(p, name, nl)) return true;
        if (!e) return false;
        p = e + 1;
    }
}

static bool cg_climbs(const char *p) {
    for (const char *s = p; (s = strstr(s, "/..")) != NULL; s += 3)
        if (s[3] == '/' || s[3] == '\0') return true;
    return false;
}

// The process's cgroup under one hierarchy, from /proc/self/cgroup: the v2
// "0::/path" line when `controller` is NULL, else the v1 line whose
// controller list names it ("4:cpu,cpuacct:/path"). "" (the mount root)
// when there is no file or no such line, and for a path that climbs out of
// the mount with "..", which is how a cgroup namespace shows a cgroup
// outside it.
static void cg_self(const char *file, const char *controller, char *out, size_t n) {
    out[0] = '\0';
    FILE *f = file ? plat_fopen(file, "r") : NULL;
    if (!f) return;
    char line[CG_PATH_MAX + 256];
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = '\0';
        char *c1 = strchr(line, ':'), *c2 = c1 ? strchr(c1 + 1, ':') : NULL;
        if (!c2) continue;
        *c1 = *c2 = '\0';
        const char *ctrls = c1 + 1, *path = c2 + 1;
        if (controller ? !cg_list_has(ctrls, controller) : (strcmp(line, "0") || *ctrls))
            continue;
        if (path[0] == '/' && strcmp(path, "/") && !cg_climbs(path) && strlen(path) < n)
            memcpy(out, path, strlen(path) + 1);
        break;
    }
    fclose(f);
}

void plat_capacity_at(const char *root, const char *self_file, int host_cpus,
                      uint64_t host_ram, uint64_t host_avail, plat_capacity *out) {
    memset(out, 0, sizeof *out);
    out->host_cpu_cores = host_cpus > 0 ? host_cpus : 1;
    out->host_ram_bytes = host_ram;
    out->host_ram_available_bytes = host_avail;
    out->limits_source = "none";
    cg_bound b = { 0 };
    char self[CG_PATH_MAX], base[CG_PATH_MAX], probe[8];
    if (root && strlen(root) + 32 < sizeof base) {
        if (cg_text(root, "cgroup.controllers", probe, sizeof probe)) {
            out->limits_source = "cgroup2";
            cg_self(self_file, NULL, self, sizeof self);
            cg_walk(root, self, 2, true, true, &b);
        } else {
            // v1 mounts each controller apart; cpu is "cpu" or the combined
            // "cpu,cpuacct" ("cpu" is usually a symlink to it)
            static const char *const cpu_dirs[] = { "cpu", "cpu,cpuacct", "cpuacct,cpu" };
            for (size_t i = 0; i < sizeof cpu_dirs / sizeof *cpu_dirs; i++) {
                snprintf(base, sizeof base, "%s/%s", root, cpu_dirs[i]);
                if (!cg_text(base, "cpu.cfs_period_us", probe, sizeof probe)) continue;
                out->limits_source = "cgroup1";
                cg_self(self_file, "cpu", self, sizeof self);
                cg_walk(base, self, 1, true, false, &b);
                break;
            }
            snprintf(base, sizeof base, "%s/memory", root);
            if (cg_text(base, "memory.limit_in_bytes", probe, sizeof probe)) {
                out->limits_source = "cgroup1";
                cg_self(self_file, "memory", self, sizeof self);
                cg_walk(base, self, 1, false, true, &b);
            }
        }
    }
    out->cpu_quota = b.cpus;
    out->memory_max_bytes = b.mem_max;
    out->cpu_cores = b.cpus && b.cpus < out->host_cpu_cores ? b.cpus : out->host_cpu_cores;
    // a host figure of 0 is "unknown", so the limit's figure stands alone
    out->ram_bytes = b.mem_max && (!host_ram || b.mem_max < host_ram) ? b.mem_max : host_ram;
    out->ram_available_bytes =
        b.have_left && (!host_avail || b.mem_left < host_avail) ? b.mem_left : host_avail;
}

void plat_capacity_get(plat_capacity *out) {
    // RUNNER_TEST_CGROUP_ROOT (test-only) reads a fixture tree as this
    // process's own cgroup, on any platform
    const char *root = getenv("RUNNER_TEST_CGROUP_ROOT"), *self = NULL;
    if (!root || !*root) {
#ifdef __linux__
        root = "/sys/fs/cgroup";
        self = "/proc/self/cgroup";
#else
        root = NULL;
#endif
    }
    plat_capacity_at(root, self, machine_cpu_count(), machine_ram_bytes(),
                     machine_ram_available_bytes(), out);
}

int plat_cpu_count(void) {
    plat_capacity c;
    plat_capacity_get(&c);
    return c.cpu_cores;
}

uint64_t plat_ram_bytes(void) {
    plat_capacity c;
    plat_capacity_get(&c);
    return c.ram_bytes;
}

uint64_t plat_ram_available_bytes(void) {
    plat_capacity c;
    plat_capacity_get(&c);
    return c.ram_available_bytes;
}

// --------------------------------------------------------- strict numeric parse
//
// One strict parser shared by CLI flags and environment overrides (RNR-021):
// rejects empty input, trailing garbage, overflow, sign errors, and (for
// doubles) non-finite values, instead of the silent atoi/atof/strtoull=0 that
// let a typo disable a deadline or overflow a byte budget. Finiteness is tested
// on the IEEE bits (fpclass.h), not isfinite(), because the build uses
// -ffast-math.
#include <errno.h>

// One policy, both platforms, and a pure function so the many-core regime can
// be tested on a machine that has few cores. See PLAT_THREAD_DEFAULT_MAX in
// compat.h for why the ceiling is where it is.
//
// Only machines with MORE THAN 64 logical CPUs change behaviour here: at 64 the
// halving already lands on the ceiling, and everything below it is untouched.
// That is deliberate - the ceiling was measured in the many-core regime and
// nowhere else, so it moves nothing that was not measured.
int plat_thread_default_for(int nc) {
    int n = nc >= 4 ? nc / 2 : nc;
    if (n > PLAT_THREAD_DEFAULT_MAX) n = PLAT_THREAD_DEFAULT_MAX;
    return n > 0 ? n : 1;
}

bool parse_i64(const char *s, long long lo, long long hi, long long *out) {
    if (!s || !*s) return false;
    char *end = NULL;
    errno = 0;
    long long v = strtoll(s, &end, 10);
    if (errno || end == s || *end || v < lo || v > hi) return false;
    *out = v;
    return true;
}

bool parse_u64(const char *s, uint64_t lo, uint64_t hi, uint64_t *out) {
    // strtoull silently NEGATES a leading '-' and does not set errno for it,
    // so the sign has to be refused here. It skips leading whitespace first,
    // which is why this walks past it rather than testing s[0]: " -1" reached
    // strtoull with the guard satisfied and came back 18446744073709551615.
    if (!s) return false;
    const char *p = s;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' ||
           *p == '\f' || *p == '\v') p++;
    if (!*p || *p == '-') return false;
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || end == s || *end || v < lo || v > hi) return false;
    *out = (uint64_t)v;
    return true;
}

bool parse_f64(const char *s, double lo, double hi, double *out) {
    if (!s || !*s) return false;
    char *end = NULL;
    errno = 0;
    double v = strtod(s, &end);
    if (errno || end == s || *end || !fp_isfinite(v) || v < lo || v > hi) return false;
    *out = v;
    return true;
}

// Environment overrides: an absent variable keeps `cur`; an invalid one warns
// at startup and keeps `cur` — never a silent 0.
long long env_i64(const char *name, long long lo, long long hi, long long cur) {
    const char *s = getenv(name);
    long long v;
    if (!s) return cur;
    if (!parse_i64(s, lo, hi, &v)) {
        fprintf(stderr, "warning: %s='%s' is not an integer in [%lld, %lld] — ignoring\n",
                name, s, lo, hi);
        return cur;
    }
    return v;
}

uint64_t env_u64(const char *name, uint64_t lo, uint64_t hi, uint64_t cur) {
    const char *s = getenv(name);
    uint64_t v;
    if (!s) return cur;
    if (!parse_u64(s, lo, hi, &v)) {
        fprintf(stderr, "warning: %s='%s' is not an unsigned integer in range — ignoring\n",
                name, s);
        return cur;
    }
    return v;
}

double env_f64(const char *name, double lo, double hi, double cur) {
    const char *s = getenv(name);
    double v;
    if (!s) return cur;
    if (!parse_f64(s, lo, hi, &v)) {
        fprintf(stderr, "warning: %s='%s' is not a finite number in [%g, %g] — ignoring\n",
                name, s, lo, hi);
        return cur;
    }
    return v;
}

bool plat_once(_Atomic bool *flag) {
    return !atomic_exchange_explicit(flag, true, memory_order_relaxed);
}

// state: 0 unread, 1 one caller is opening it, 2 settled (fp final). The
// release store publishes fp to the acquire loads; a caller that loses the
// race waits the microseconds the open takes.
FILE *plat_env_file_get(plat_env_file *f, const char *env) {
    if (atomic_load_explicit(&f->state, memory_order_acquire) == 2) return f->fp;
    int unread = 0;
    if (atomic_compare_exchange_strong(&f->state, &unread, 1)) {
        const char *path = getenv(env);
        if (path && *path) {
            f->fp = plat_fopen(path, "a");
            if (!f->fp)
                fprintf(stderr, "warning: %s=%s: could not open for append\n", env, path);
        }
        atomic_store_explicit(&f->state, 2, memory_order_release);
        return f->fp;
    }
    while (atomic_load_explicit(&f->state, memory_order_acquire) != 2) plat_sleep_ms(1);
    return f->fp;
}

// ------------------------------------------------- a note for a silent wait
//
// See compat.h. The watcher waits on an event (Windows) or a condition
// variable (POSIX) with a timeout; the operation's end signals it first when
// it is quick. The line is said under the same lock end takes, so the answer
// end gives is exact.

#ifdef _WIN32
struct plat_slow_note {
    HANDLE thread, done;
    CRITICAL_SECTION mu;
    FILE *out;
    char line[512];
    DWORD after_ms;
    bool said;
};
static DWORD WINAPI slow_note_watch(LPVOID arg) {
    plat_slow_note *n = arg;
    if (WaitForSingleObject(n->done, n->after_ms) == WAIT_TIMEOUT) {
        EnterCriticalSection(&n->mu);
        if (WaitForSingleObject(n->done, 0) == WAIT_TIMEOUT) {
            fprintf(n->out, "%s\n", n->line);
            fflush(n->out);
            n->said = true;
        }
        LeaveCriticalSection(&n->mu);
    }
    return 0;
}
plat_slow_note *plat_slow_note_begin(FILE *out, const char *line, double after_s) {
    plat_slow_note *n = calloc(1, sizeof *n);
    if (!n) return NULL;
    n->out = out;
    snprintf(n->line, sizeof n->line, "%s", line);
    n->after_ms = after_s > 0 ? (DWORD)(after_s * 1000.0) : 0;
    InitializeCriticalSection(&n->mu);
    n->done = CreateEventW(NULL, TRUE, FALSE, NULL);
    n->thread = n->done ? CreateThread(NULL, 0, slow_note_watch, n, 0, NULL) : NULL;
    if (!n->thread) {
        if (n->done) CloseHandle(n->done);
        DeleteCriticalSection(&n->mu);
        free(n);
        return NULL;
    }
    return n;
}
bool plat_slow_note_end(plat_slow_note *n) {
    if (!n) return false;
    EnterCriticalSection(&n->mu);
    SetEvent(n->done);
    LeaveCriticalSection(&n->mu);
    WaitForSingleObject(n->thread, INFINITE);
    bool said = n->said;
    CloseHandle(n->thread);
    CloseHandle(n->done);
    DeleteCriticalSection(&n->mu);
    free(n);
    return said;
}
#else
#include <pthread.h>
#include <time.h>
struct plat_slow_note {
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    FILE *out;
    char line[512];
    struct timespec at;
    bool done, said;
};
static void *slow_note_watch(void *arg) {
    plat_slow_note *n = arg;
    pthread_mutex_lock(&n->mu);
    while (!n->done)
        if (pthread_cond_timedwait(&n->cv, &n->mu, &n->at) == ETIMEDOUT) break;
    if (!n->done) {
        fprintf(n->out, "%s\n", n->line);
        fflush(n->out);
        n->said = true;
    }
    pthread_mutex_unlock(&n->mu);
    return NULL;
}
plat_slow_note *plat_slow_note_begin(FILE *out, const char *line, double after_s) {
    plat_slow_note *n = calloc(1, sizeof *n);
    if (!n) return NULL;
    n->out = out;
    snprintf(n->line, sizeof n->line, "%s", line);
    clock_gettime(CLOCK_REALTIME, &n->at);
    if (after_s < 0) after_s = 0;
    long long ns = (long long)n->at.tv_nsec + (long long)((after_s - (long long)after_s) * 1e9);
    n->at.tv_sec += (time_t)after_s + (time_t)(ns / 1000000000LL);
    n->at.tv_nsec = (long)(ns % 1000000000LL);
    if (pthread_mutex_init(&n->mu, NULL) != 0) { free(n); return NULL; }
    if (pthread_cond_init(&n->cv, NULL) != 0) { pthread_mutex_destroy(&n->mu); free(n); return NULL; }
    if (pthread_create(&n->thread, NULL, slow_note_watch, n) != 0) {
        pthread_cond_destroy(&n->cv);
        pthread_mutex_destroy(&n->mu);
        free(n);
        return NULL;
    }
    return n;
}
bool plat_slow_note_end(plat_slow_note *n) {
    if (!n) return false;
    pthread_mutex_lock(&n->mu);
    n->done = true;
    pthread_cond_signal(&n->cv);
    pthread_mutex_unlock(&n->mu);
    pthread_join(n->thread, NULL);
    bool said = n->said;
    pthread_cond_destroy(&n->cv);
    pthread_mutex_destroy(&n->mu);
    free(n);
    return said;
}
#endif
