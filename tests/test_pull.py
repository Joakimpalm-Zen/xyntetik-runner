"""`--pull owner/repo[:TAG]`: fetch a model into the cache without loading it.

The transfer is `-hf`'s (same selection, cache, verification, fake hub
through RUNNER_CURL; tests/test_hf_fetch.py pins that path). What this file
pins is the mode: nothing is loaded, stdout carries the result and only the
result, and every claim about a file is anchored on the fixture's own size
and SHA-256 computed here, outside Runner.
"""
import hashlib
import json
import os
import sys

import pytest

from test_hf_fetch import (  # noqa: F401  (fixtures)
    calls, fixture_model, hub, resolve_calls, run, runner_bin, sha256)


def test_pull_fetches_into_the_cache_and_prints_one_line(runner_bin, hub):
    want = hub["dir"] / "model-Q8_0.gguf"
    r = run(runner_bin, hub["env"], "--pull", "fake/repo:Q8_0")
    assert r.returncode == 0, r.stderr
    cached = hub["cache"] / "fake--repo" / "model-Q8_0.gguf"
    assert cached.exists() and sha256(cached) == sha256(want)
    lines = r.stdout.splitlines()
    assert len(lines) == 1, r.stdout
    assert str(cached.resolve()) in lines[0] and str(want.stat().st_size) in lines[0]
    # nothing was loaded: no model banner, and the transfer notes stay on stderr
    assert "hf: fetching fake/repo/model-Q8_0.gguf" in r.stderr
    assert "loaded" not in r.stderr


def pull_json(runner_bin, env, spec):
    r = run(runner_bin, env, "--pull", spec, "--json")
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)            # stdout is exactly one object
    assert r.stdout.count("\n") == 1 and isinstance(doc, dict)
    return doc, r


def test_json_reports_the_file_against_the_fixtures_own_digest(runner_bin, hub):
    want = hub["dir"] / "model-Q8_0.gguf"
    doc, r = pull_json(runner_bin, hub["env"], "fake/repo:Q8_0")
    cached = (hub["cache"] / "fake--repo" / "model-Q8_0.gguf").resolve()
    assert doc == {
        "path": str(cached),
        "size_bytes": want.stat().st_size,
        "files": [{"path": str(cached), "size_bytes": want.stat().st_size,
                   "sha256": sha256(want), "verification": "hub_sha256"}],
    }
    # the transfer notes are on stderr, never mixed into the object
    assert "hf: fetching fake/repo/model-Q8_0.gguf" in r.stderr


def test_a_repeated_pull_transfers_no_model_bytes(runner_bin, hub):
    first, _ = pull_json(runner_bin, hub["env"], "fake/repo:Q8_0")
    second, r = pull_json(runner_bin, hub["env"], "fake/repo:Q8_0")
    assert second == first
    assert "hf: fake/repo/model-Q8_0.gguf cached" in r.stderr
    assert len(resolve_calls(hub["log"])) == 1


def test_corrupt_cache_bytes_are_fetched_again_never_reported(runner_bin, hub):
    want = hub["dir"] / "model-Q8_0.gguf"
    pull_json(runner_bin, hub["env"], "fake/repo:Q8_0")
    cached = hub["cache"] / "fake--repo" / "model-Q8_0.gguf"
    data = bytearray(cached.read_bytes())
    data[len(data) // 2] ^= 0xFF
    cached.write_bytes(bytes(data))
    doc, r = pull_json(runner_bin, hub["env"], "fake/repo:Q8_0")
    assert "does not verify" in r.stderr
    assert doc["files"][0]["sha256"] == sha256(want) == sha256(cached)
    assert len(resolve_calls(hub["log"])) == 2


def test_a_leftover_part_file_is_never_accepted(runner_bin, hub):
    want = hub["dir"] / "model-Q8_0.gguf"
    repodir = hub["cache"] / "fake--repo"
    repodir.mkdir(parents=True)
    (repodir / "model-Q8_0.gguf.part").write_bytes(want.read_bytes()[:100])
    doc, _ = pull_json(runner_bin, hub["env"], "fake/repo:Q8_0")
    assert doc["files"][0]["sha256"] == sha256(want)
    assert not (repodir / "model-Q8_0.gguf.part").exists()
    assert len(resolve_calls(hub["log"])) == 1


def listing(hub, siblings):
    (hub["dir"] / "files.json").write_text(json.dumps({"siblings": siblings}))


def test_size_only_is_labelled_hub_size_and_still_digested(runner_bin, hub):
    want = hub["dir"] / "model-Q8_0.gguf"
    listing(hub, [{"rfilename": "model-Q8_0.gguf", "size": want.stat().st_size}])
    doc, _ = pull_json(runner_bin, hub["env"], "fake/repo:Q8_0")
    assert doc["files"][0]["verification"] == "hub_size"
    assert doc["files"][0]["sha256"] == sha256(want)
    # and the cached file, verified by size alone, is digested on reuse too
    again, _ = pull_json(runner_bin, hub["env"], "fake/repo:Q8_0")
    assert again == doc


def test_neither_digest_nor_size_fails_without_a_transfer(runner_bin, hub, fixture_model):
    listing(hub, [{"rfilename": "model-Q8_0.gguf"}])
    r = run(runner_bin, hub["env"], "--pull", "fake/repo:Q8_0", "--json")
    assert r.returncode == 1 and r.stdout == ""
    assert "neither a sha256 nor a size" in r.stderr
    assert resolve_calls(hub["log"]) == []
    # -hf keeps accepting such a file, as it always has
    ok = run(runner_bin, hub["env"], "-hf", "fake/repo:Q8_0", "-p", "hi", "-n", "1", "--temp", "0")
    assert ok.returncode == 0, ok.stderr


def test_a_digest_mismatch_fails_and_writes_no_object(runner_bin, hub):
    r = run(runner_bin, hub["env"], "--pull", "fake/repo:Q4_K_M", "--json")
    assert r.returncode == 1 and r.stdout == ""
    assert "sha256" in r.stderr
    assert not (hub["cache"] / "fake--repo" / "model-Q4_K_M.gguf").exists()


def shards(hub, n, corrupt=None, missing=None):
    """n shard files of arbitrary bytes (a pull never parses them), listed with
    their digests; `corrupt` lists a wrong digest for that part, `missing`
    lists a part the hub cannot serve."""
    sib = []
    for k in range(1, n + 1):
        name = f"big-Q8_0-{k:05d}-of-{n:05d}.gguf"
        path = hub["dir"] / name
        path.write_bytes(bytes([k]) * (1000 + 37 * k))
        entry = {"rfilename": name, "size": path.stat().st_size,
                 "lfs": {"sha256": sha256(path), "size": path.stat().st_size}}
        if k == corrupt:
            entry["lfs"]["sha256"] = "f" * 64
        if k == missing:
            path.unlink()
        sib.append(entry)
    listing(hub, sib)
    return [hub["dir"] / s["rfilename"] for s in sib]


def test_multipart_lists_every_part_in_order_with_the_sum(runner_bin, hub):
    parts = shards(hub, 3)
    sizes = [p.stat().st_size for p in parts]
    doc, _ = pull_json(runner_bin, hub["env"], "fake/repo:Q8_0")
    repodir = (hub["cache"] / "fake--repo").resolve()
    assert doc["path"] == str(repodir / parts[0].name)
    assert doc["size_bytes"] == sum(sizes)
    assert [f["path"] for f in doc["files"]] == [str(repodir / p.name) for p in parts]
    assert [f["sha256"] for f in doc["files"]] == [sha256(p) for p in parts]
    assert [f["size_bytes"] for f in doc["files"]] == sizes


def test_multipart_with_a_corrupt_or_missing_part_fails(runner_bin, hub):
    for kw in ({"corrupt": 2}, {"missing": 3}):
        shards(hub, 3, **kw)
        r = run(runner_bin, hub["env"], "--pull", "fake/repo:Q8_0", "--json")
        assert r.returncode == 1 and r.stdout == "", (kw, r.stderr)


def test_pull_is_a_mode_of_its_own(runner_bin, hub, fixture_model):
    for extra in (["-m", str(fixture_model)], ["-hf", "fake/repo:Q8_0"], ["--serve"],
                  ["-p", "hi"], ["--fit", str(fixture_model)], ["-n", "4"]):
        r = run(runner_bin, hub["env"], "--pull", "fake/repo:Q8_0", *extra)
        assert r.returncode == 1, (extra, r.stderr)
        assert extra[0] in r.stderr and "--pull" in r.stderr, (extra, r.stderr)
        # the order does not matter
        r = run(runner_bin, hub["env"], *extra, "--pull", "fake/repo:Q8_0")
        assert r.returncode == 1 and extra[0] in r.stderr, (extra, r.stderr)
    assert calls(hub["log"]) == []


def test_bad_specs_name_the_flag(runner_bin, hub):
    r = run(runner_bin, hub["env"], "--pull", "no-slash-here")
    assert r.returncode == 1 and "--pull wants owner/repo" in r.stderr
    r = run(runner_bin, hub["env"], "--pull", "fake/repo")   # two candidates
    assert r.returncode == 1 and "--pull REPO:TAG" in r.stderr
    assert resolve_calls(hub["log"]) == []


SPARSE_CURL = r'''
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_HUB_LOG"], "a") as log:
    log.write(json.dumps(args) + "\n")
out, url = args[args.index("-o") + 1], args[-1]
if "/api/models/" in url:
    with open(os.environ["FAKE_HUB_FILES"]) as src, open(out, "w") as dst:
        dst.write(src.read())
    sys.exit(0)
# the model: zeros of the listed length, written sparse (no 2 GB of I/O)
with open(out, "wb") as f:
    f.truncate(int(os.environ["FAKE_HUB_SIZE"]))
sys.exit(0)
'''


@pytest.mark.skipif(sys.platform != "win32" and not os.environ.get("RUNNER_TEST_LARGE"),
                    reason="the 32-bit stat lives on Windows; elsewhere opt in with "
                           "RUNNER_TEST_LARGE=1 (hashes 2 GiB three times, ~25 s)")
def test_a_file_over_2_gib_is_sized_and_reused(runner_bin, tmp_path):
    """A real checkpoint is larger than 2 GiB. MinGW's stat() keeps the size
    in 32 bits and fails there (model.c's file identity documents it), and
    the cache check and the size check sized files with it: on Windows such
    a file was never seen as cached, and --pull could not report it. Zeros of
    a length past 2^31, whose SHA-256 is computed here, anchor the size, the
    digest and the reuse."""
    size = (1 << 31) + 4096
    h = hashlib.sha256()
    block = bytes(1 << 20)
    for _ in range(size // len(block)):
        h.update(block)
    h.update(bytes(size % len(block)))
    files = tmp_path / "files.json"
    files.write_text(json.dumps({"siblings": [{"rfilename": "big-Q8_0.gguf", "size": size}]}))
    script = tmp_path / "sparsecurl.py"
    script.write_text(SPARSE_CURL)
    if sys.platform == "win32":
        curl = tmp_path / "sparsecurl.cmd"
        curl.write_text(f'@"{sys.executable}" "{script}" %*\r\n')
    else:
        curl = tmp_path / "sparsecurl"
        curl.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
        curl.chmod(0o755)
    log = tmp_path / "curl.log"
    env = dict(os.environ, RUNNER_CURL=str(curl), RUNNER_HF_CACHE=str(tmp_path / "cache"),
               FAKE_HUB_FILES=str(files), FAKE_HUB_SIZE=str(size), FAKE_HUB_LOG=str(log))
    env.pop("HF_TOKEN", None)
    first, _ = pull_json(runner_bin, env, "fake/big:Q8_0")
    assert first["size_bytes"] == size
    assert first["files"][0]["sha256"] == h.hexdigest()
    assert first["files"][0]["verification"] == "hub_size"
    second, r = pull_json(runner_bin, env, "fake/big:Q8_0")
    assert second == first and "cached" in r.stderr
    assert len(resolve_calls(log)) == 1
