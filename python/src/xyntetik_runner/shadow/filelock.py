"""One writer at a time for the small state files shadow mode keeps under
``~/.xyntetik/shadow``: a lock file held for the duration of a
read-modify-write, and a temporary name unique to the writer for the
rename-into-place.

Two commands or two hook invocations (two sessions capturing at once) used
to race each other: a shared ``.tmp`` name meant one writer renamed the
other's half-written file, and an unlocked read-modify-write dropped the
other's update (external review and sweep, 2026-09-27).
"""
from __future__ import annotations

import contextlib
import os
import secrets
import sys
import time
from collections.abc import Iterator
from pathlib import Path

if sys.platform == "win32":
    import msvcrt

    def _lock_fd(fd: int) -> None:
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                return
            except OSError:
                time.sleep(0.2)

    def _unlock_fd(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock_fd(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _unlock_fd(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    """Hold ``<path>.lock`` (a sibling file) for the block. Not re-entrant:
    a block must not open the same lock again."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(f".{path.name}.lock")
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        _lock_fd(fd)
        yield
    finally:
        try:
            _unlock_fd(fd)
        finally:
            os.close(fd)


def replace_text(path: Path, text: str) -> None:
    """Write ``text`` through a temporary file unique to this writer, then
    rename into place, so a reader never sees a partial file and two
    writers never share a temporary name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def replace_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


__all__ = ["locked", "replace_text", "replace_bytes"]
