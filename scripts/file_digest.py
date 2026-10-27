"""The SHA-256 of a file, the way every script that records one computes it:
the whole file, in 1 MiB reads, as lowercase hex.

A script imports this from its own directory; one that a test loads by file
path puts that directory on sys.path first.
"""
import hashlib


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
