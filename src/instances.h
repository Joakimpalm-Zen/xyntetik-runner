// Instance discovery registry.
//
// Every runner process in a RUN mode (one-shot generation, --serve, --tray)
// writes one JSON record at startup so a controller — the tray, or any
// future scheduler — can see what is running on this machine without
// port-scanning or process-list guessing. Exit leaves the record alone so
// filesystem cleanup cannot stall termination. Readers and new registrations
// filter dead owners, and an independent helper sweeps them asynchronously,
// checking process-start stamps when available. Utility modes
// (--quantize, --caps, --bench-json, --version) do not register.
//
// Registration is BEST-EFFORT: a failure to write the record must never
// affect the run itself. Nothing in the inference or serve paths reads it.
#ifndef RUNNER_INSTANCES_H
#define RUNNER_INSTANCES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    long       pid;
    long long  started;      // unix seconds
    uint64_t   procstart;    // OS process-creation stamp; 0 = legacy/unknown
    char       mode[16];     // "cli" | "serve" | "tray"
    int        port;         // 0 = not serving
    char     **model_names;  // [n_models], owned
    char     **model_paths;  // [n_models], owned
    int        n_models;
    char       version[32];
} instance_rec;

// Directory the records live in: <config>/xyntetik/runner/instances
// (~/.xyntetik/runner/instances on POSIX, %APPDATA%\xyntetik\runner\instances
// on Windows). Created on first use. Returns a static buffer, or NULL if no
// home/config directory could be resolved.
const char *instances_dir(void);

// Register THIS process; overwrites any previous record for this pid.
bool instances_register(const char *mode, int port,
                        const char *const *model_names,
                        const char *const *model_paths, int n_models);

// Rewrite this process's record with a new port (a server knows its final
// port only after bind; registration happens at model load).
bool instances_set_port(int port);

// R9.5: whether this process can record itself in the registry, by writing
// and removing a probe file there. A launch does not raise a tray where it
// cannot: the tray could not record itself, the next launch would find no
// tray, and trays would stack one per launch.
bool instances_writable(void);

// R9.5: one tray per machine, as a lock held for the tray's whole life
// (<dir>/tray.lock), so two launches inside a tray's start cannot both start
// one: the check and the claim are one step. Busy: another process holds it.
// Unavailable: there is no registry directory to hold it in.
typedef enum { TRAY_LOCK_TAKEN, TRAY_LOCK_BUSY, TRAY_LOCK_UNAVAILABLE } tray_lock_state;
tray_lock_state instances_tray_lock(void);
// Whether another process holds the tray lock now (a tray is running, though
// its record may not be written yet).
bool instances_tray_lock_busy(void);
// Let go of this process's tray lock (a tray that will not run after all).
void instances_tray_unlock(void);

// Explicitly withdraw this process's record while still running. Safe to
// call more than once. Do not call at exit: filesystem removal can block.
void instances_unregister(void);

// List live records, scheduling stale cleanup without waiting for deletion.
// Returns a malloc'd array (free with instances_list_free), count in *n.
// Never returns records for pids that are not alive.
instance_rec *instances_list(int *n);
void instances_list_free(instance_rec *recs, int n);

// Internal re-exec entry point. Cleanup owns its lock and all deletion I/O
// in a separate process; callers never wait for it at startup or exit.
int instances_sweep_worker(void);

// True if a process with this pid is alive (platform-appropriate check).
bool instance_pid_alive(long pid);

// ------------------------------------------------------------ startup lease
//
// The Runner package's StartupLease (python/src/xyntetik_runner/lease.py) in
// C, record for record, so the tray, the Python client's ManagedRunner and the
// Suite never both launch a Runner on one port (R4.12.24). The lease lives at
// <state root>/leases/runner-<port>.pid, next to instances/.
typedef struct {
    char path[1100];
    char token[33];
    bool held;
} runner_lease;

// <state root>/leases/runner-<port>.pid, the leases directory created; false
// when no state root resolves.
bool runner_lease_path(int port, char *out, size_t cap);

// Claim the lease at `path` for this process. False while a live owner holds
// it (or when the claim cannot be written); a dead or reused owner's record is
// moved aside and the claim retried.
bool runner_lease_acquire(runner_lease *l, const char *path);

// Release, only while the record still carries this lease's token. Safe to
// call on a lease that is not held.
void runner_lease_release(runner_lease *l);

// The pid the record at `path` names as its owner; 0 when none.
long runner_lease_holder(const char *path);

// This platform's process start identity, exactly as lease.py reads it:
// Linux /proc field 22, Windows the creation FILETIME, macOS `ps -o lstart=`
// under LC_ALL=C and TZ=UTC. False where it cannot be read.
bool runner_lease_start_identity(long pid, char *out, size_t cap);

#endif // RUNNER_INSTANCES_H
