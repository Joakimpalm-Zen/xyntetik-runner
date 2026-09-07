#!/usr/bin/env python3
"""Measure how this machine's storage serves large weight reads.

An engine can get weight bytes to the GPU two ways: map the file and let the
consumer fault pages in, or read them explicitly into a buffer. Which one wins
is a property of the storage and the OS, not of the model, and the gap can be
large enough to change a design decision.

The reason is queue depth. A page fault is synchronous and serves one page at a
time, so an N-megabyte region faults in as thousands of small dependent reads.
An explicit read of the same region is one large request the device can pipeline
and prefetch. On flash this routinely costs several times the bandwidth.

This measures both paths on the same file, at the same slice size, at the same
scattered page-aligned offsets:

    python scripts/weight-io-bench.py --file models/model.gguf
    python scripts/weight-io-bench.py --file models/model.gguf \\
        --slice-bytes 4194304 --samples 300

JSON on stdout so runs can be diffed mechanically.

Caching is the thing that makes this measurement lie. On macOS the read path
sets F_NOCACHE so it measures the device rather than the buffer cache. There is
no portable equivalent for the mapped path, and neither flag can evict pages an
earlier reader already cached, so prefer a file substantially larger than RAM
and one you have not just written or read. `cached_second_pass_suspected` in the
output flags the obvious failure.
"""

import argparse
import json
import mmap
import os
import statistics
import sys
import time

DEFAULT_SLICE = 8 * 1024 * 1024
F_NOCACHE = 48  # <sys/fcntl.h>, macOS only
# Windows opens in text mode by default, which would translate CRLF inside
# binary weights and make every byte count a lie rather than an error.
O_BINARY = getattr(os, "O_BINARY", 0)


def page_size():
    """Fault granularity: what one page fault brings in."""
    return mmap.PAGESIZE


def map_alignment():
    """What a mapping offset must be a multiple of.

    The same as the page size everywhere except Windows, where a mapping
    starts on a 64 KB allocation granule. Both measured paths use offsets
    aligned to this, because the comparison is only meaningful if the two
    read the same regions.
    """
    return mmap.ALLOCATIONGRANULARITY


if hasattr(os, "pread"):
    READ_CALL = "pread"

    def positioned_read(fd, want, at):
        return os.pread(fd, want, at)
else:
    # No pread on Windows. A seek plus a read is the same bytes from the same
    # place for this single-threaded loop; it is one extra syscall per part,
    # which the output names so a cross-platform diff is not read as a
    # storage difference.
    READ_CALL = "lseek+read"

    def positioned_read(fd, want, at):
        os.lseek(fd, at, os.SEEK_SET)
        return os.read(fd, want)


def open_uncached(path):
    """Open for reading and ask the OS not to cache, where that is supported."""
    fd = os.open(path, os.O_RDONLY | O_BINARY | getattr(os, "O_CLOEXEC", 0))
    uncached = False
    if sys.platform == "darwin":
        try:
            import fcntl
            uncached = fcntl.fcntl(fd, F_NOCACHE, 1) == 0
        except OSError:
            uncached = False
    return fd, uncached


def map_slice(fd, length, offset):
    """One read-only private mapping of [offset, offset+length).

    The keyword arguments are not portable: Unix takes flags/prot, Windows
    takes access. ACCESS_READ is the private read-only mapping on both.
    """
    return mmap.mmap(fd, length, access=mmap.ACCESS_READ, offset=offset)


def offsets(size, slice_bytes, count, stride_seed):
    """Deterministic aligned offsets that do not repeat or cross EOF.

    Scattered rather than sequential, because a run of adjacent reads lets
    readahead answer most of them and measures the cache instead of the device.
    An odd stride coprime with the position count walks the whole file without
    revisiting one, and stays reproducible so a number can be re-derived.

    Aligned to map_alignment() rather than the page size: the mapped path
    cannot start anywhere else on Windows, and the two paths have to sample
    the same offsets for their ratio to mean anything.
    """
    span = size - slice_bytes
    if span <= 0:
        raise SystemExit(
            f"file is {size} B, smaller than one {slice_bytes} B slice")
    page = map_alignment()
    positions = span // page
    if positions < 1:
        raise SystemExit("file leaves no aligned slice offsets")
    step = stride_seed % positions or 1
    while positions > 1 and _gcd(step, positions) != 1:
        step += 1
        if step >= positions:
            step = 1
            break
    return [((i * step) % positions) * page for i in range(count)]


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _summarize(durations, slice_bytes, extra=None):
    per_sample = sorted(slice_bytes / d for d in durations)
    total = sum(durations)
    out = {
        "samples": len(durations),
        "total_bytes": slice_bytes * len(durations),
        "total_seconds": total,
        "mean_bytes_per_second": slice_bytes * len(durations) / total,
        "p50_bytes_per_second": statistics.median(per_sample),
        "p05_bytes_per_second": per_sample[max(0, int(0.05 * len(per_sample)) - 1)],
        "min_bytes_per_second": per_sample[0],
        "max_bytes_per_second": per_sample[-1],
        "p50_slice_ms": 1000.0 * statistics.median(durations),
        "max_slice_ms": 1000.0 * max(durations),
    }
    if extra:
        out.update(extra)
    return out


def measure_read(path, slice_bytes, samples, warmup, stride_seed, parts):
    """Explicit reads: `parts` bounded pread calls covering one slice."""
    if slice_bytes % parts:
        raise SystemExit(f"slice {slice_bytes} is not divisible into {parts} parts")
    part_bytes = slice_bytes // parts
    size = os.path.getsize(path)
    buffer = bytearray(slice_bytes)
    durations = []
    fd, uncached = open_uncached(path)
    try:
        for index, offset in enumerate(
                offsets(size, slice_bytes, samples + warmup, stride_seed)):
            start = time.perf_counter()
            got = 0
            for part in range(parts):
                want = part_bytes
                at = offset + part * part_bytes
                while want:
                    chunk = positioned_read(fd, want, at)
                    if not chunk:
                        raise SystemExit(f"short read at offset {at}")
                    buffer[got:got + len(chunk)] = chunk
                    got += len(chunk)
                    at += len(chunk)
                    want -= len(chunk)
            elapsed = time.perf_counter() - start
            if index >= warmup:
                durations.append(elapsed)
    finally:
        os.close(fd)
    return _summarize(durations, slice_bytes,
                      {"parts_per_slice": parts, "uncached": uncached,
                       "read_call": READ_CALL})


def measure_faults(path, slice_bytes, samples, warmup, stride_seed):
    """Mapped reads: touch one byte per page so the region faults in."""
    size = os.path.getsize(path)
    page = page_size()
    durations = []
    fd = os.open(path, os.O_RDONLY | O_BINARY | getattr(os, "O_CLOEXEC", 0))
    try:
        for index, offset in enumerate(
                offsets(size, slice_bytes, samples + warmup, stride_seed)):
            start = time.perf_counter()
            view = map_slice(fd, slice_bytes, offset)
            try:
                total = 0
                for at in range(0, slice_bytes, page):
                    total += view[at]
                elapsed = time.perf_counter() - start
                if hasattr(view, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                    view.madvise(mmap.MADV_DONTNEED)
            finally:
                view.close()
            if index >= warmup:
                durations.append(elapsed)
    finally:
        os.close(fd)
    return _summarize(durations, slice_bytes, {"pages_per_slice": slice_bytes // page})


def cache_probe(path, slice_bytes, parts):
    """Read one slice twice. A much faster second pass means we measured RAM."""
    first = measure_read(path, slice_bytes, 1, 0, 1, parts)
    second = measure_read(path, slice_bytes, 1, 0, 1, parts)
    a = first["total_seconds"]
    b = second["total_seconds"]
    ratio = a / b if b > 0 else float("inf")
    return {
        "first_seconds": a,
        "second_seconds": b,
        "speedup_ratio": ratio,
        "cached_second_pass_suspected": ratio > 8.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True,
                        help="any large file; a GGUF weight file is typical")
    parser.add_argument("--slice-bytes", type=int, default=DEFAULT_SLICE)
    parser.add_argument("--parts", type=int, default=1,
                        help="split each slice into N positioned reads")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--stride-seed", type=int, default=104729)
    parser.add_argument("--skip-faults", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.samples < 1 or args.warmup < 0 or args.slice_bytes < page_size():
        raise SystemExit("need samples >= 1, warmup >= 0, slice >= one page")
    if args.parts < 1:
        raise SystemExit("need parts >= 1")

    report = {
        "schema": 1,
        "file": os.path.abspath(args.file),
        "file_bytes": os.path.getsize(args.file),
        "slice_bytes": args.slice_bytes,
        "page_bytes": page_size(),
        "map_alignment_bytes": map_alignment(),
        "platform": sys.platform,
        "cache_probe": cache_probe(args.file, args.slice_bytes, args.parts),
        "read_path": measure_read(args.file, args.slice_bytes, args.samples,
                                  args.warmup, args.stride_seed, args.parts),
    }
    if not args.skip_faults:
        report["fault_path"] = measure_faults(
            args.file, args.slice_bytes, args.samples, args.warmup,
            args.stride_seed)
        fault = report["fault_path"]["p50_bytes_per_second"]
        read = report["read_path"]["p50_bytes_per_second"]
        report["read_over_fault_p50_ratio"] = read / fault if fault else None

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
