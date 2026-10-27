"""The SHA-256 of a file, the way every script that records one computes it:
the whole file as lowercase hex (hashlib.file_digest where the interpreter
has it, 1 MiB reads otherwise).

A script imports this from its own directory; one that a test loads by file
path puts that directory on sys.path first.
"""
import hashlib


def sha256_file(path, chunk=1 << 20):
    with open(path, "rb") as f:
        if hasattr(hashlib, "file_digest"):   # 3.11+: the readinto loop runs in C
            return hashlib.file_digest(f, "sha256").hexdigest()
        h = hashlib.sha256()
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
        return h.hexdigest()
