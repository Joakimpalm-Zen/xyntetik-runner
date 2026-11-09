"""Offline cache inventory. Literal bytes anchor sizes and SHA-256 externally."""
import json
import os
import subprocess


def _run(runner_bin, cache, *flags, alias=False):
    env = dict(os.environ, RUNNER_HF_CACHE=str(cache), RUNNER_CURL="inventory-must-not-run-curl")
    return subprocess.run([str(runner_bin), *(["--info", "cached"] if alias else ["cached"]),
                           *flags], env=env, capture_output=True, text=True, timeout=30)


def test_cached_reports_literal_bytes_offline(runner_bin, tmp_path):
    cache = tmp_path / "cache"
    repo = cache / "owner--repo"
    repo.mkdir(parents=True)
    model = repo / "modèle naïf.gguf"
    model.write_bytes(b"abc")
    result = _run(runner_bin, cache, "--json")
    assert result.returncode == 0, result.stderr
    doc = json.loads(result.stdout)
    assert doc["schema_version"] == "xyntetik.runner.cached.v1"
    assert doc["errors"] == []
    entry, = doc["models"]
    assert entry["complete"] is True and entry["size_bytes"] == 3
    file, = entry["files"]
    assert file["size_bytes"] == 3
    assert file["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert file["hash_status"] == "computed"
    assert "hashing" in result.stderr and "loaded" not in result.stderr
    human = _run(runner_bin, cache)
    assert human.returncode == 0 and file["sha256"] in human.stdout
    assert "3 bytes" in human.stdout and "modèle naïf.gguf" in human.stdout
    assert json.loads(_run(runner_bin, cache, "--json", alias=True).stdout) == doc


def test_multipart_groups_nested_files_and_reports_missing_parts(runner_bin, tmp_path):
    import hashlib
    cache = tmp_path / "cache"
    repo = cache / "owner--repo"
    folder = repo / "nested"
    folder.mkdir(parents=True)
    for i, data in [(3, b"ccc"), (1, b"a"), (2, b"bb")]:
        (folder / f"model-0000{i}-of-00003.gguf").write_bytes(data)
    (repo / "mmproj.gguf").write_bytes(b"image")
    (repo / "unfinished.gguf.part").write_bytes(b"ignore")
    result = _run(runner_bin, cache, "--json")
    assert result.returncode == 0, result.stderr
    models = json.loads(result.stdout)["models"]
    assert len(models) == 2
    model = next(m for m in models if not m["auxiliary"])
    assert model["complete"] is True and model["size_bytes"] == 6
    assert model["parts_expected"] == 3
    assert [f["relative_path"] for f in model["files"]] == [
        f"nested/model-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
    assert [f["sha256"] for f in model["files"]] == [
        hashlib.sha256(b).hexdigest() for b in (b"a", b"bb", b"ccc")]
    (folder / "model-00001-of-00003.gguf").unlink()
    result = _run(runner_bin, cache, "--json")
    assert result.returncode == 0
    model = next(m for m in json.loads(result.stdout)["models"] if not m["auxiliary"])
    assert model["complete"] is False and model["size_bytes"] == 5
    assert len(model["files"]) == 2


def test_sizes_only_and_missing_cache(runner_bin, tmp_path):
    cache = tmp_path / "cache"
    result = _run(runner_bin, cache, "--json")
    assert result.returncode == 0 and json.loads(result.stdout)["models"] == []
    assert not cache.exists()
    cache.mkdir()
    empty = _run(runner_bin, cache, "--json")
    assert empty.returncode == 0 and json.loads(empty.stdout)["models"] == []
    repo = cache / "a--b--c"
    repo.mkdir(parents=True)
    (repo / "tiny.gguf").write_bytes(b"abc")
    result = _run(runner_bin, cache, "--json", "--sizes-only")
    assert result.returncode == 0 and "hashing" not in result.stderr
    model, = json.loads(result.stdout)["models"]
    assert model["repo"] is None  # the flattened name does not prove an id
    file, = model["files"]
    assert file["size_bytes"] == 3 and file["sha256"] is None
    assert file["hash_status"] == "not_requested"


def test_cached_listing_is_optional_identity_and_comparison(runner_bin, tmp_path):
    cache = tmp_path / "cache"
    repo = cache / "a--b--c"
    repo.mkdir(parents=True)
    (repo / "tiny.gguf").write_bytes(b"abc")
    listing = {"id": "a--b/c", "siblings": [{"rfilename": "tiny.gguf", "size": 3,
               "lfs": {"sha256": "0" * 64}}]}
    (repo / ".files.json").write_text(json.dumps(listing))
    result = _run(runner_bin, cache, "--json")
    assert result.returncode == 0  # mismatch is a reported fact, not repair
    model, = json.loads(result.stdout)["models"]
    assert model["repo"] == "a--b/c"
    file, = model["files"]
    assert file["listing_match"] is False
    assert file["listed_sha256"] == "0" * 64 and file["listed_size_bytes"] == 3
    assert file["sha256"] != file["listed_sha256"]
    (repo / ".files.json").write_text('{invalid')
    result = _run(runner_bin, cache, "--json")
    model, = json.loads(result.stdout)["models"]
    assert model["repo"] is None and model["files"][0]["listing_match"] is None


def test_sizes_only_does_not_open_content_and_read_errors_are_reported(runner_bin, tmp_path):
    import pytest
    if os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0):
        pytest.skip("POSIX mode bits need an unprivileged process")
    cache = tmp_path / "cache"
    repo = cache / "owner--repo"
    repo.mkdir(parents=True)
    model = repo / "private.gguf"
    model.write_bytes(b"abc")
    model.chmod(0)
    try:
        fast = _run(runner_bin, cache, "--json", "--sizes-only")
        assert fast.returncode == 0
        assert json.loads(fast.stdout)["models"][0]["files"][0]["sha256"] is None
        full = _run(runner_bin, cache, "--json")
        assert full.returncode == 1
        doc = json.loads(full.stdout)
        assert len(doc["errors"]) == 1
        file = doc["models"][0]["files"][0]
        assert file["sha256"] is None and file["hash_status"] == "unreadable"
    finally:
        model.chmod(0o600)


def test_symlinks_are_not_followed(runner_bin, tmp_path):
    import pytest
    if os.name == "nt":
        pytest.skip("creating symlinks requires Windows privileges")
    cache = tmp_path / "cache"
    repo = cache / "owner--repo"
    repo.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.gguf").write_bytes(b"outside")
    (repo / "link.gguf").symlink_to(outside / "secret.gguf")
    (repo / "folder").symlink_to(outside, target_is_directory=True)
    (cache / "linked--repo").symlink_to(outside, target_is_directory=True)
    result = _run(runner_bin, cache, "--json")
    assert result.returncode == 0 and json.loads(result.stdout)["models"] == []


def test_invalid_combinations_and_non_directory_root(runner_bin, tmp_path):
    cache = tmp_path / "cache"
    for flags in (("--serve",), ("-m", "absent.gguf"), ("--info", "caps"),
                  ("--memory", "8"), ("--json", "schema.json")):
        r = _run(runner_bin, cache, *flags)
        assert r.returncode == 1 and r.stdout == "" and "not allowed" in r.stderr
    cache.write_bytes(b"not a directory")
    result = _run(runner_bin, cache, "--json")
    assert result.returncode == 1
    assert json.loads(result.stdout)["errors"]


def test_platform_default_root_and_64_bit_sizes(runner_bin, tmp_path):
    # Sparse metadata-only fixture: exercises Windows' >2 GiB stat without
    # reading or allocating a multi-GB model in the inventory process.
    cache = (tmp_path / "xyntetik-runner" / "hf" if os.name == "nt" else
             tmp_path / ".cache" / "xyntetik-runner" / "hf")
    repo = cache / "owner--repo"
    repo.mkdir(parents=True)
    size = (3 << 30) + 17
    with (repo / "large.gguf").open("wb") as f:
        f.truncate(size)
    env = dict(os.environ, RUNNER_HF_CACHE="", HOME=str(tmp_path), LOCALAPPDATA=str(tmp_path))
    result = subprocess.run([str(runner_bin), "cached", "--json", "--sizes-only"],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    doc = json.loads(result.stdout)
    assert os.path.normcase(os.path.normpath(doc["cache_root"])) == os.path.normcase(str(cache))
    model, = doc["models"]
    assert model["size_bytes"] == size and model["files"][0]["size_bytes"] == size
    assert model["files"][0]["sha256"] is None
