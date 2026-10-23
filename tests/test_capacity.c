// What this process may use, against the cgroup limits that bound it.
//
// Found on the Linux CUDA lab box: a container with cpu.max
// "2600000 100000" (26 CPUs) and memory.max 80 GiB on a 128-CPU, 268 GB
// host. --caps reported 128 CPUs and 268 GB, because the affinity mask does
// not carry a CFS quota and the physical page count and /proc/meminfo
// describe the host; the default thread count, --fit, the resident-model
// checks and the batch default all sized from those numbers.
//
// plat_capacity_at reads a cgroup tree from explicit paths and takes the
// host numbers as arguments, so every expected value below is computed by
// hand from the fixture's own text: the kernel's documented file formats
// are the anchor, not another run of this code.
#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#define test_mkdir(p) _mkdir(p)
#define test_rmdir(p) _rmdir(p)
#define test_pid() ((long)_getpid())
#else
#include <sys/stat.h>
#include <unistd.h>
#define test_mkdir(p) mkdir(p, 0755)
#define test_rmdir(p) rmdir(p)
#define test_pid() ((long)getpid())
#endif

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        printf("ok: %s\n", what);
}

#define GiB(x) ((uint64_t)(x) << 30)
#define MiB(x) ((uint64_t)(x) << 20)

// The lab box's host, as it reported itself
#define HOST_CPUS  128
#define HOST_RAM   268000000000ull
#define HOST_AVAIL 235000000000ull

// Every path the test creates, removed in reverse at the end.
static char g_made[256][1024];
static int  g_is_dir[256];
static int  g_n_made = 0;
static char g_base[768];

static void remember(const char *p, int is_dir) {
    if (g_n_made >= 256) return;
    snprintf(g_made[g_n_made], sizeof g_made[0], "%s", p);
    g_is_dir[g_n_made++] = is_dir;
}

// Write `text` at <base>/<tree>/<rel>, creating each directory on the way.
static void put(const char *tree, const char *rel, const char *text) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s/%s", g_base, tree, rel);
    for (char *p = path + strlen(g_base) + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (test_mkdir(path) == 0) remember(path, 1);
        *p = '/';
    }
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "FAIL: cannot write %s\n", path); g_fail = 1; return; }
    fputs(text, f);
    fclose(f);
    remember(path, 0);
}

static const char *tree(const char *name, char *buf, size_t n) {
    snprintf(buf, n, "%s/%s", g_base, name);
    return buf;
}

static void read_tree(const char *name, const char *self, int cpus,
                      uint64_t ram, uint64_t avail, plat_capacity *c) {
    char root[1024], self_path[1024];
    const char *sp = NULL;
    if (self) { snprintf(self_path, sizeof self_path, "%s/%s", g_base, self); sp = self_path; }
    memset(c, 0x5a, sizeof *c);   // nothing may survive from before the call
    plat_capacity_at(tree(name, root, sizeof root), sp, cpus, ram, avail, c);
}

static void host_echoed(const plat_capacity *c, int cpus, uint64_t ram,
                        uint64_t avail, const char *what) {
    char m[160];
    snprintf(m, sizeof m, "%s: the host numbers are reported as given", what);
    ck(c->host_cpu_cores == cpus && c->host_ram_bytes == ram &&
       c->host_ram_available_bytes == avail, m);
}

int main(void) {
    snprintf(g_base, sizeof g_base, "%s/runner-test-capacity-%ld",
             plat_runtime_dir(), test_pid());
    if (test_mkdir(g_base) == 0) remember(g_base, 1);
    plat_capacity c;

    // --- cgroup v2, the lab box's container --------------------------------
    // cpu.max is "$MAX $PERIOD" in microseconds; 2600000 / 100000 = 26 CPUs.
    // memory.max 85899345920 = 80 GiB; memory.current 10 GiB charged.
    put("v2-lab", "cgroup.controllers", "cpuset cpu io memory hugetlb pids rdma misc\n");
    put("v2-lab", "cpu.max", "2600000 100000\n");
    put("v2-lab", "memory.max", "85899345920\n");
    put("v2-lab", "memory.current", "10737418240\n");
    read_tree("v2-lab", NULL, HOST_CPUS, HOST_RAM, HOST_AVAIL, &c);
    ck(!strcmp(c.limits_source, "cgroup2"), "v2: the source is cgroup2");
    ck(c.cpu_quota == 26, "v2: cpu.max 2600000 100000 is a 26-CPU quota");
    ck(c.cpu_cores == 26, "v2: 26 of 128 CPUs are this process's");
    ck(c.memory_max_bytes == GiB(80), "v2: memory.max 85899345920 is 80 GiB");
    ck(c.ram_bytes == GiB(80), "v2: the limit replaces the host's 268 GB");
    ck(c.ram_available_bytes == GiB(80) - GiB(10),
       "v2: available is the limit less memory.current, 70 GiB");
    host_echoed(&c, HOST_CPUS, HOST_RAM, HOST_AVAIL, "v2");

    // --- "max" is no quota and no limit ------------------------------------
    put("v2-max", "cgroup.controllers", "cpu memory\n");
    put("v2-max", "cpu.max", "max 100000\n");
    put("v2-max", "memory.max", "max\n");
    put("v2-max", "memory.current", "4294967296\n");
    read_tree("v2-max", NULL, HOST_CPUS, HOST_RAM, HOST_AVAIL, &c);
    ck(!strcmp(c.limits_source, "cgroup2"), "max: the hierarchy was read");
    ck(c.cpu_quota == 0 && c.cpu_cores == HOST_CPUS,
       "max: cpu.max \"max 100000\" leaves the affinity count, 128");
    ck(c.memory_max_bytes == 0 && c.ram_bytes == HOST_RAM &&
       c.ram_available_bytes == HOST_AVAIL,
       "max: memory.max \"max\" leaves the host's RAM and MemAvailable");

    // --- a fractional quota rounds up, never below one ----------------------
    put("v2-frac", "cgroup.controllers", "cpu\n");
    put("v2-frac", "cpu.max", "150000 100000\n");
    read_tree("v2-frac", NULL, HOST_CPUS, HOST_RAM, HOST_AVAIL, &c);
    ck(c.cpu_quota == 2 && c.cpu_cores == 2, "cpu.max 150000 100000 (1.5 CPUs) -> 2");
    put("v2-half", "cgroup.controllers", "cpu\n");
    put("v2-half", "cpu.max", "50000 100000\n");
    read_tree("v2-half", NULL, HOST_CPUS, HOST_RAM, HOST_AVAIL, &c);
    ck(c.cpu_quota == 1 && c.cpu_cores == 1, "cpu.max 50000 100000 (0.5 CPUs) -> 1");

    // --- the affinity mask still wins when it is the narrower --------------
    read_tree("v2-lab", NULL, 16, HOST_RAM, HOST_AVAIL, &c);
    ck(c.cpu_quota == 26 && c.cpu_cores == 16,
       "a 16-CPU mask under a 26-CPU quota is 16; the quota is still reported");

    // --- reclaimable page cache counts as available, as MemAvailable does --
    // 8 GiB limit, 7 GiB charged, of which 2 GiB active_file, 1 GiB
    // inactive_file and 256 MiB slab_reclaimable can be reclaimed: 1 GiB of
    // headroom plus 3.25 GiB. shmem sits on the anon lists and is not
    // counted (the 512 MiB below would make it 4.75 GiB if it were).
    put("v2-stat", "cgroup.controllers", "memory\n");
    put("v2-stat", "memory.max", "8589934592\n");
    put("v2-stat", "memory.current", "7516192768\n");
    put("v2-stat", "memory.stat",
        "anon 2147483648\nfile 4831838208\nkernel 268435456\nshmem 536870912\n"
        "inactive_anon 1073741824\nactive_anon 1610612736\n"
        "inactive_file 1073741824\nactive_file 2147483648\n"
        "unevictable 0\nslab_reclaimable 268435456\nslab_unreclaimable 33554432\n");
    read_tree("v2-stat", NULL, HOST_CPUS, HOST_RAM, HOST_AVAIL, &c);
    ck(c.ram_available_bytes == GiB(1) + GiB(3) + MiB(256),
       "memory.stat: headroom 1 GiB + active_file + inactive_file + slab_reclaimable = 4.25 GiB");
    read_tree("v2-stat", NULL, HOST_CPUS, HOST_RAM, GiB(2), &c);
    ck(c.ram_available_bytes == GiB(2), "a host MemAvailable below the cgroup's headroom wins");
    read_tree("v2-stat", NULL, HOST_CPUS, HOST_RAM, 0, &c);
    ck(c.ram_available_bytes == GiB(1) + GiB(3) + MiB(256),
       "an unknown host MemAvailable (0) leaves the cgroup's figure");

    // --- over the limit is zero headroom, not a wrapped subtraction --------
    put("v2-over", "cgroup.controllers", "memory\n");
    put("v2-over", "memory.max", "1073741824\n");
    put("v2-over", "memory.current", "1100000000\n");
    put("v2-over", "memory.stat", "inactive_file 104857600\nactive_file 0\n");
    read_tree("v2-over", NULL, HOST_CPUS, HOST_RAM, HOST_AVAIL, &c);
    ck(c.ram_available_bytes == MiB(100), "usage past memory.max leaves only the reclaimable 100 MiB");

    // --- the process's own cgroup, under a host cgroup namespace ------------
    // /proc/self/cgroup names the cgroup; an ancestor's limit binds too, so
    // every level from it up to the mount is read and the tightest wins:
    // cpu 4 at the scope, 2 at the slice; memory 16 GiB with 4 GiB charged
    // at the scope (12 GiB headroom), 12 GiB with 10 GiB charged at the
    // slice (2 GiB headroom, siblings included).
    put("v2-nest", "cgroup.controllers", "cpu memory\n");
    put("v2-nest", "system.slice/docker-0123abcd.scope/cpu.max", "400000 100000\n");
    put("v2-nest", "system.slice/docker-0123abcd.scope/memory.max", "17179869184\n");
    put("v2-nest", "system.slice/docker-0123abcd.scope/memory.current", "4294967296\n");
    put("v2-nest", "system.slice/cpu.max", "200000 100000\n");
    put("v2-nest", "system.slice/memory.max", "12884901888\n");
    put("v2-nest", "system.slice/memory.current", "10737418240\n");
    // the mount root's own quota: looser than the slice, and all that a
    // reader without the membership file sees
    put("v2-nest", "cpu.max", "2600000 100000\n");
    put("self-nest", "cgroup", "0::/system.slice/docker-0123abcd.scope\n");
    read_tree("v2-nest", "self-nest/cgroup", HOST_CPUS, HOST_RAM, HOST_AVAIL, &c);
    ck(c.cpu_quota == 2 && c.cpu_cores == 2, "nested: the slice's 2-CPU quota binds the scope");
    ck(c.memory_max_bytes == GiB(12) && c.ram_bytes == GiB(12), "nested: the tighter 12 GiB limit");
    ck(c.ram_available_bytes == GiB(2), "nested: the slice's 2 GiB headroom binds");
    // without the membership file the mount root is the process's cgroup
    read_tree("v2-nest", NULL, HOST_CPUS, HOST_RAM, HOST_AVAIL, &c);
    ck(c.cpu_cores == 26 && c.memory_max_bytes == 0, "nested: no self file reads the root alone");
    // a membership path that is not under the mount falls back to the root
    put("self-gone", "cgroup", "0::/not/here\n");
    read_tree("v2-nest", "self-gone/cgroup", HOST_CPUS, HOST_RAM, HOST_AVAIL, &c);
    ck(c.cpu_cores == 26, "nested: a missing cgroup directory walks up to the root");
    put("self-up", "cgroup", "0::/../../outside\n");
    read_tree("v2-nest", "self-up/cgroup", HOST_CPUS, HOST_RAM, HOST_AVAIL, &c);
    ck(c.cpu_cores == 26, "nested: a path that climbs out of the mount reads the root only");

    // --- cgroup v1 ----------------------------------------------------------
    // Per-controller mounts. Docker bind-mounts the container's own cgroup
    // at each, while /proc/self/cgroup still names the host-side path, so
    // the walk up from that path lands on the mount root, which is the
    // container. cfs 2600000 / 100000 = 26; limit 80 GiB, usage 10 GiB, of
    // which 1 GiB + 1 GiB is file cache.
    put("v1", "cpu,cpuacct/cpu.cfs_quota_us", "2600000\n");
    put("v1", "cpu,cpuacct/cpu.cfs_period_us", "100000\n");
    put("v1", "memory/memory.limit_in_bytes", "85899345920\n");
    put("v1", "memory/memory.usage_in_bytes", "10737418240\n");
    put("v1", "memory/memory.stat",
        "cache 2147483648\nrss 8589934592\nactive_file 0\ninactive_file 0\n"
        "total_cache 2147483648\ntotal_rss 8589934592\n"
        "total_inactive_file 1073741824\ntotal_active_file 1073741824\n");
    put("self-v1", "cgroup",
        "12:memory:/docker/0123abcd\n11:pids:/docker/0123abcd\n"
        "4:cpu,cpuacct:/docker/0123abcd\n3:cpuset:/docker/0123abcd\n"
        "1:name=systemd:/docker/0123abcd\n0::/system.slice/containerd.service\n");
    read_tree("v1", "self-v1/cgroup", HOST_CPUS, HOST_RAM, HOST_AVAIL, &c);
    ck(!strcmp(c.limits_source, "cgroup1"), "v1: the source is cgroup1");
    ck(c.cpu_quota == 26 && c.cpu_cores == 26, "v1: cfs_quota_us 2600000 / cfs_period_us 100000 = 26");
    ck(c.memory_max_bytes == GiB(80) && c.ram_bytes == GiB(80), "v1: limit_in_bytes 80 GiB");
    ck(c.ram_available_bytes == GiB(80) - GiB(10) + GiB(2),
       "v1: limit less usage_in_bytes plus total_active_file + total_inactive_file = 72 GiB");
    host_echoed(&c, HOST_CPUS, HOST_RAM, HOST_AVAIL, "v1");

    // cfs_quota_us -1 is no quota; v1 writes "no limit" as LONG_MAX rounded
    // down to a page, 9223372036854771712 with 4 KiB pages
    put("v1-open", "cpu/cpu.cfs_quota_us", "-1\n");
    put("v1-open", "cpu/cpu.cfs_period_us", "100000\n");
    put("v1-open", "memory/memory.limit_in_bytes", "9223372036854771712\n");
    put("v1-open", "memory/memory.usage_in_bytes", "4294967296\n");
    read_tree("v1-open", NULL, HOST_CPUS, HOST_RAM, HOST_AVAIL, &c);
    ck(!strcmp(c.limits_source, "cgroup1"), "v1 unlimited: the hierarchy was read");
    ck(c.cpu_quota == 0 && c.cpu_cores == HOST_CPUS, "v1 unlimited: cfs_quota_us -1 is no quota");
    ck(c.memory_max_bytes == 0 && c.ram_bytes == HOST_RAM && c.ram_available_bytes == HOST_AVAIL,
       "v1 unlimited: the page-rounded LONG_MAX sentinel is no limit");

    // --- text the kernel does not write is no information ------------------
    put("v2-bad", "cgroup.controllers", "cpu memory\n");
    put("v2-bad", "cpu.max", "lots 100000\n");
    put("v2-bad", "memory.max", "-1\n");
    read_tree("v2-bad", NULL, HOST_CPUS, HOST_RAM, HOST_AVAIL, &c);
    ck(c.cpu_cores == HOST_CPUS && c.ram_bytes == HOST_RAM, "malformed files are ignored, not guessed at");

    // --- no hierarchy: the host numbers, source none ------------------------
    read_tree("absent", NULL, HOST_CPUS, HOST_RAM, HOST_AVAIL, &c);
    ck(!strcmp(c.limits_source, "none"), "absent: no cgroup files means source none");
    ck(c.cpu_cores == HOST_CPUS && c.ram_bytes == HOST_RAM &&
       c.ram_available_bytes == HOST_AVAIL && c.cpu_quota == 0 && c.memory_max_bytes == 0,
       "absent: the host numbers are the effective ones");
    memset(&c, 0x5a, sizeof c);
    plat_capacity_at(NULL, NULL, 8, GiB(16), GiB(9), &c);
    ck(!strcmp(c.limits_source, "none") && c.cpu_cores == 8 && c.ram_bytes == GiB(16) &&
       c.ram_available_bytes == GiB(9), "no cgroup root: the host numbers");

    // --- the live reading agrees with the three accessors ------------------
    plat_capacity live;
    plat_capacity_get(&live);
    ck(live.cpu_cores >= 1 && live.cpu_cores <= live.host_cpu_cores,
       "live: the effective CPU count is within the host's");
    ck(live.ram_bytes <= live.host_ram_bytes || !live.host_ram_bytes,
       "live: the effective RAM is within the host's");
    ck(plat_cpu_count() == live.cpu_cores, "live: plat_cpu_count is the effective count");
    ck(plat_ram_bytes() == live.ram_bytes, "live: plat_ram_bytes is the effective RAM");
#if !defined(__linux__)
    ck(!strcmp(live.limits_source, "none") && live.cpu_cores == live.host_cpu_cores &&
       live.ram_bytes == live.host_ram_bytes,
       "live: no cgroup outside Linux, so the host numbers are unchanged");
#endif

    for (int i = g_n_made - 1; i >= 0; i--) {
        if (g_is_dir[i]) test_rmdir(g_made[i]);
        else remove(g_made[i]);
    }
    if (g_fail) {
        fprintf(stderr, "capacity: FAILED\n");
        return 1;
    }
    puts("capacity ok");
    return 0;
}
