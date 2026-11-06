"""R3.30 step 3: what one expert's upload costs on this host, via the driver API.
Measures HtoD of one Mellum expert (6,580,224 bytes) from the mapped model file
(pageable) and from pinned memory, synchronous and on a second stream.
Usage: upload_bench.py MODEL [--rate PER_SEC --seconds S --pinned|--pageable]
With --rate it paces uploads for S seconds from the mapped file, the side load
for the interference runs, and prints what it achieved.
"""
import ctypes as C
import os
import sys
import time

EXPERT = 6580224
cu = C.WinDLL("nvcuda.dll")


def ck(rc, what):
    if rc != 0:
        raise SystemExit(f"{what}: CUresult {rc}")


ck(cu.cuInit(0), "init")
dev = C.c_int()
ck(cu.cuDeviceGet(C.byref(dev), 0), "dev")
ctx = C.c_void_p()
ck(cu.cuDevicePrimaryCtxRetain(C.byref(ctx), dev), "ctx")
ck(cu.cuCtxSetCurrent(ctx), "setctx")
cu.cuMemAlloc_v2.argtypes = [C.POINTER(C.c_uint64), C.c_size_t]
cu.cuMemcpyHtoD_v2.argtypes = [C.c_uint64, C.c_void_p, C.c_size_t]
cu.cuMemcpyHtoDAsync_v2.argtypes = [C.c_uint64, C.c_void_p, C.c_size_t, C.c_void_p]
cu.cuMemAllocHost_v2.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
cu.cuStreamCreate.argtypes = [C.POINTER(C.c_void_p), C.c_uint]
cu.cuStreamSynchronize.argtypes = [C.c_void_p]

SLOTS = 4
d = C.c_uint64()
ck(cu.cuMemAlloc_v2(C.byref(d), EXPERT * SLOTS), "alloc")
pin = C.c_void_p()
ck(cu.cuMemAllocHost_v2(C.byref(pin), EXPERT * SLOTS), "pin")
C.memset(pin, 1, EXPERT * SLOTS)
st = C.c_void_p()
ck(cu.cuStreamCreate(C.byref(st), 1), "stream")

path = sys.argv[1]
# a read-only python mmap gives no address, so map the file through the OS
k32 = C.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = C.c_void_p
k32.CreateFileW.argtypes = [C.c_wchar_p, C.c_uint32, C.c_uint32, C.c_void_p, C.c_uint32, C.c_uint32, C.c_void_p]
k32.CreateFileMappingW.restype = C.c_void_p
k32.CreateFileMappingW.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32, C.c_wchar_p]
k32.MapViewOfFile.restype = C.c_void_p
k32.MapViewOfFile.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32, C.c_size_t]
h = k32.CreateFileW(path, 0x80000000, 1, None, 3, 0, None)
hm = k32.CreateFileMappingW(h, None, 2, 0, 0, None)
view = k32.MapViewOfFile(hm, 4, 0, 0, 0)
size = os.path.getsize(path)
if not view:
    raise SystemExit("map failed")


def pageable_src(i):
    return view + (1 << 30) + (i * EXPERT) % (size - (2 << 30))


def touch(p):
    # fault the source in first, so the timing below is the copy and not the disk
    C.string_at(p, EXPERT)


def timed(label, fn, n=40):
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    med = ts[len(ts) // 2]
    print(f"{label:44} median {med:7.3f} ms  p90 {ts[int(.9*n)]:7.3f}  {EXPERT/med/1e6:6.2f} GB/s", flush=True)


if "--rate" in sys.argv:
    # the side load: replay an admission list (offset bytes, three lines an
    # expert) at RATE experts a second from the mapped file, pageable
    rate = float(sys.argv[sys.argv.index("--rate") + 1])
    secs = float(sys.argv[sys.argv.index("--seconds") + 1])
    items = [tuple(map(int, l.split())) for l in open(sys.argv[sys.argv.index("--list") + 1])]
    i = n = 0
    busy = 0.0
    end = time.perf_counter() + secs
    nxt = time.perf_counter()
    while time.perf_counter() < end:
        t0 = time.perf_counter()
        if "--from-pinned" in sys.argv:
            # the floor: one DMA of one expert from a pinned buffer, no CPU copy
            # and no pageable path; same bytes over the bus, no read of the file
            ck(cu.cuMemcpyHtoDAsync_v2(d.value, pin, EXPERT, st), "h2d")
            ck(cu.cuStreamSynchronize(st), "sync")
        else:
            for _ in range(3):
                off, b = items[i % len(items)]
                i += 1
                ck(cu.cuMemcpyHtoD_v2(d.value, view + off, b), "h2d")
        busy += time.perf_counter() - t0
        n += 1
        nxt += 1.0 / rate
        s = nxt - time.perf_counter()
        if s > 0:
            time.sleep(s)
    print(f"side load: {n} experts in {secs:.0f}s ({n/secs:.1f}/s), {busy/n*1e3:.3f} ms each", flush=True)
    sys.exit(0)

for i in range(48):
    touch(pageable_src(i))
timed("pageable HtoD (mapped file, warm pages)", lambda i: ck(cu.cuMemcpyHtoD_v2(d.value, pageable_src(i), EXPERT), "h2d"))
timed("pinned HtoD sync", lambda i: ck(cu.cuMemcpyHtoD_v2(d.value, pin, EXPERT), "h2d"))


def async_one(i):
    ck(cu.cuMemcpyHtoDAsync_v2(d.value + (i % SLOTS) * EXPERT, pin, EXPERT, st), "h2d")
    ck(cu.cuStreamSynchronize(st), "sync")


timed("pinned HtoD async, second stream", async_one)


def staged(i):
    C.memmove(pin, pageable_src(i), EXPERT)
    ck(cu.cuMemcpyHtoDAsync_v2(d.value, pin, EXPERT, st), "h2d")
    ck(cu.cuStreamSynchronize(st), "sync")


timed("staged: memcpy from mapping + pinned async", staged)
timed("memcpy from mapping to pinned alone", lambda i: C.memmove(pin, pageable_src(i), EXPERT))
