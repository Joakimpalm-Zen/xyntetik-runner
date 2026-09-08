"""Content identity of a workspace before and after a local attempt.

A baseline is a map of relative path to sha256 over the files of a tree.
Everything the verifier says about a patch is said against a baseline: the
no-op check is "the tree did not change", the tamper check is "a protected
or configuration file is among the changes", and the patch identity that
goes into an evidence record is a hash over the changed paths and their
new contents, so two attempts that produce the same bytes get the same id
regardless of how they got there.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

IGNORED_DIRS = frozenset({".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv",
                          ".ruff_cache", "node_modules"})


def file_sha256(path: Path) -> str:
    """Content identity with CRLF folded to LF.

    A git checkout on Windows translates line endings, so the same file has
    different bytes on two machines. A protected manifest frozen on one and
    loaded on the other must still agree, and a patch that changes nothing
    but line endings is not a change. Hashing the newline-normalized bytes
    makes both true; nothing here ever compares raw bytes.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        pending = b""
        for chunk in iter(lambda: f.read(1 << 16), b""):
            data = pending + chunk
            # a chunk boundary may split a CRLF pair; hold a trailing CR back
            pending = data[-1:] if data.endswith(b"\r") else b""
            h.update(data[: len(data) - len(pending)].replace(b"\r\n", b"\n"))
        h.update(pending)
    return h.hexdigest()


def tree_hashes(root: Path, *, ignored_dirs: Iterable[str] = IGNORED_DIRS) -> dict[str, str]:
    """Relative POSIX path to sha256 for every regular file under ``root``.

    Symlinks are skipped: a link that points outside the tree would let a
    "workspace" hash bytes it does not contain.
    """
    ignored = frozenset(ignored_dirs)
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in ignored for part in rel.parts[:-1]):
            continue
        out[rel.as_posix()] = file_sha256(path)
    return out


def tree_sha256(hashes: Mapping[str, str]) -> str:
    """One digest for a whole tree: the canonical JSON of its path->hash map."""
    canonical = json.dumps(dict(sorted(hashes.items())), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class Changes:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    modified: tuple[str, ...]

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(sorted({*self.added, *self.removed, *self.modified}))

    def __bool__(self) -> bool:
        return bool(self.added or self.removed or self.modified)


@dataclass(frozen=True)
class Baseline:
    """A captured tree: its file hashes and their combined digest."""

    files: Mapping[str, str]
    sha256: str

    @classmethod
    def capture(cls, root: Path) -> Baseline:
        hashes = tree_hashes(root)
        return cls(files=dict(hashes), sha256=tree_sha256(hashes))

    def changes(self, root: Path) -> Changes:
        now = tree_hashes(root)
        added = tuple(sorted(p for p in now if p not in self.files))
        removed = tuple(sorted(p for p in self.files if p not in now))
        modified = tuple(sorted(p for p in now if p in self.files and now[p] != self.files[p]))
        return Changes(added=added, removed=removed, modified=modified)

    def patch_sha256(self, root: Path) -> str:
        """Identity of what the attempt changed: canonical JSON of the changed
        paths with their new hashes (``None`` for a removed file)."""
        now = tree_hashes(root)
        record = {p: now.get(p) for p in self.changes(root).paths}
        canonical = json.dumps(record, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
