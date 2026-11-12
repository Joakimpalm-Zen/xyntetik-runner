"""`-hf owner/repo[:TAG]`: the transfer half, against a fake hub.

`RUNNER_CURL` names the program the runner runs for every transfer, so a
stand-in curl that answers the file-list URL with a canned `files.json` and
the `resolve/main/<file>` URL by copying a fixture is enough to exercise the
whole path offline: the cache layout, the SHA-256 check against the Hub's
LFS record, the cache hit on the second run, the tag errors, the bearer
token, and the flag's own argument rules. The selection rules themselves
are pinned in tests/test_hfhub.c.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

FAKE_CURL = r'''
import json, os, shutil, sys
args = sys.argv[1:]
with open(os.environ["FAKE_HUB_LOG"], "a") as log:
    log.write(json.dumps(args) + "\n")
if "-K" in args and args[args.index("-K") + 1] == "-":
    with open(os.environ["FAKE_HUB_LOG"] + ".stdin", "a") as log:
        log.write(json.dumps(sys.stdin.read()) + "\n")
out = args[args.index("-o") + 1]
url = args[-1]
hub = os.environ["FAKE_HUB_DIR"]
if "/api/models/" in url:
    if os.environ.get("FAKE_HUB_MISSING"):
        sys.exit(22)   # curl -f on the Hub's 404 for a repository it lacks
    shutil.copy(os.path.join(hub, "files.json"), out)
    sys.exit(0)
if "/resolve/main/" in url:
    import urllib.parse
    name = urllib.parse.unquote(url.split("/resolve/main/", 1)[1])
    src = os.path.join(hub, name)
    if not os.path.exists(src):
        sys.exit(22)
    shutil.copy(src, out)
    sys.exit(0)
sys.exit(3)
'''


@pytest.fixture(scope="module")
def fixture_model(tmp_path_factory):
    m = tmp_path_factory.mktemp("m") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT)
    return m


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture()
def hub(tmp_path, fixture_model):
    """A fake hub directory, its curl stand-in, and the environment the runner needs."""
    hubdir = tmp_path / "hub"
    hubdir.mkdir()
    good = hubdir / "model-Q8_0.gguf"
    good.write_bytes(fixture_model.read_bytes())
    other = hubdir / "model-Q4_K_M.gguf"
    other.write_bytes(fixture_model.read_bytes())
    files = {
        "siblings": [
            {"rfilename": ".gitattributes", "size": 1500},
            {"rfilename": "README.md", "size": 4000},
            {"rfilename": "model-Q8_0.gguf", "size": good.stat().st_size,
             "lfs": {"sha256": sha256(good), "size": good.stat().st_size}},
            # a wrong hash on purpose: the runner must refuse this one
            {"rfilename": "model-Q4_K_M.gguf", "size": other.stat().st_size,
             "lfs": {"sha256": "0" * 64, "size": other.stat().st_size}},
        ]
    }
    (hubdir / "files.json").write_text(json.dumps(files))
    script = tmp_path / "fakecurl.py"
    script.write_text(FAKE_CURL)
    if sys.platform == "win32":
        curl = tmp_path / "fakecurl.cmd"
        curl.write_text(f'@"{sys.executable}" "{script}" %*\r\n')
    else:
        curl = tmp_path / "fakecurl"
        curl.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
        curl.chmod(0o755)
    log = tmp_path / "curl.log"
    cache = tmp_path / "cache"
    env = dict(os.environ, RUNNER_CURL=str(curl), RUNNER_HF_CACHE=str(cache),
               FAKE_HUB_DIR=str(hubdir), FAKE_HUB_LOG=str(log))
    env.pop("HF_TOKEN", None)
    return {"dir": hubdir, "env": env, "log": log, "cache": cache}


def run(runner_bin, env, *args, timeout=120):
    return subprocess.run([str(runner_bin), *args], env=env, cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout)


def calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def resolve_calls(log):
    return [c for c in calls(log) if "/resolve/main/" in c[-1]]


def test_fetch_by_tag_downloads_verifies_and_runs(runner_bin, hub):
    r = run(runner_bin, hub["env"], "-hf", "fake/repo:Q8_0", "-p", "hi", "-n", "4", "--temp", "0")
    assert r.returncode == 0, r.stderr
    cached = hub["cache"] / "fake--repo" / "model-Q8_0.gguf"
    assert cached.exists() and sha256(cached) == sha256(hub["dir"] / "model-Q8_0.gguf")
    assert not (hub["cache"] / "fake--repo" / "model-Q8_0.gguf.part").exists()
    assert "hf: fetching fake/repo/model-Q8_0.gguf" in r.stderr
    api = [c for c in calls(hub["log"]) if "/api/models/fake/repo" in c[-1]]
    assert len(api) == 1 and "blobs=true" in api[0][-1]
    assert [c[-1] for c in resolve_calls(hub["log"])] == [
        "https://huggingface.co/fake/repo/resolve/main/model-Q8_0.gguf"]


def test_second_run_is_a_cache_hit(runner_bin, hub):
    first = run(runner_bin, hub["env"], "-hf", "fake/repo:Q8_0", "-p", "hi", "-n", "2", "--temp", "0")
    assert first.returncode == 0, first.stderr
    second = run(runner_bin, hub["env"], "-hf", "fake/repo:Q8_0", "-p", "hi", "-n", "2", "--temp", "0")
    assert second.returncode == 0, second.stderr
    assert "hf: fake/repo/model-Q8_0.gguf cached" in second.stderr
    assert len(resolve_calls(hub["log"])) == 1


def test_sha_mismatch_is_refused_and_leaves_nothing(runner_bin, hub):
    r = run(runner_bin, hub["env"], "-hf", "fake/repo:Q4_K_M", "-p", "hi", "-n", "2")
    assert r.returncode != 0
    assert "sha256" in r.stderr and "LFS" in r.stderr
    repodir = hub["cache"] / "fake--repo"
    assert not (repodir / "model-Q4_K_M.gguf").exists()
    assert not (repodir / "model-Q4_K_M.gguf.part").exists()


def test_a_failed_lookup_takes_back_the_directory_it_made(runner_bin, hub):
    """A repository the Hub does not have left an empty owner--repo directory
    behind in the cache (the pre-1.12.0 sweep). It goes again; a directory
    that was there before the failed lookup keeps everything in it."""
    env = dict(hub["env"], FAKE_HUB_MISSING="1")
    r = run(runner_bin, env, "-hf", "no-such/repo", "-p", "hi")
    assert r.returncode != 0 and "does no-such/repo exist" in r.stderr
    assert not (hub["cache"] / "no-such--repo").exists()
    kept = hub["cache"] / "kept--repo"
    kept.mkdir(parents=True)
    (kept / "note.txt").write_text("from an earlier run")
    r = run(runner_bin, env, "-hf", "kept/repo", "-p", "hi")
    assert r.returncode != 0
    assert (kept / "note.txt").read_text() == "from an earlier run"


def test_unknown_tag_names_the_files(runner_bin, hub):
    r = run(runner_bin, hub["env"], "-hf", "fake/repo:Q3_K", "-p", "hi")
    assert r.returncode != 0
    assert "Q3_K" in r.stderr and "model-Q8_0.gguf" in r.stderr and "model-Q4_K_M.gguf" in r.stderr
    assert resolve_calls(hub["log"]) == []


def test_repo_alone_with_several_files_asks_for_a_tag(runner_bin, hub):
    r = run(runner_bin, hub["env"], "-hf", "fake/repo", "-p", "hi")
    assert r.returncode != 0
    assert "-hf REPO:TAG" in r.stderr
    assert resolve_calls(hub["log"]) == []


def test_token_is_sent_as_a_bearer_header_and_never_as_an_argument(runner_bin, hub):
    """HF_TOKEN reaches curl as a config line on its standard input (-K -).
    It was an argument (-H "Authorization: Bearer <token>"), and a program's
    arguments are in the process list for every user of the machine to read
    while a download runs."""
    token = 'hf_test"to\\ken'
    env = dict(hub["env"], HF_TOKEN=token)
    r = run(runner_bin, env, "-hf", "fake/repo:Q8_0", "-p", "hi", "-n", "2", "--temp", "0")
    assert r.returncode == 0, r.stderr
    made = calls(hub["log"])
    assert made
    for c in made:
        assert not any("hf_test" in a or "Bearer" in a for a in c), c
        assert c[c.index("-K") + 1] == "-"
    fed = [json.loads(line) for line in open(str(hub["log"]) + ".stdin")]
    assert len(fed) == len(made)
    # one curl config line; the token escaped as a config string
    assert set(fed) == {'header = "Authorization: Bearer hf_test\\"to\\\\ken"\n'}, fed


def test_a_token_with_a_control_character_is_refused(runner_bin, hub):
    env = dict(hub["env"], HF_TOKEN="hf_a\nheader = \"X: y\"")
    r = run(runner_bin, env, "-hf", "fake/repo:Q8_0", "-p", "hi", "-n", "2", "--temp", "0")
    assert r.returncode != 0 and "HF_TOKEN is not a token" in r.stderr, r.stderr
    assert not calls(hub["log"])


def test_no_token_leaves_curl_its_own_arguments(runner_bin, hub):
    r = run(runner_bin, hub["env"], "-hf", "fake/repo:Q8_0", "-p", "hi", "-n", "2", "--temp", "0")
    assert r.returncode == 0, r.stderr
    for c in calls(hub["log"]):
        assert "-K" not in c and "-H" not in c, c


def test_flag_argument_rules(runner_bin, hub, fixture_model):
    both = run(runner_bin, hub["env"], "-m", str(fixture_model), "-hf", "fake/repo:Q8_0", "-p", "hi")
    assert both.returncode != 0 and "-m" in both.stderr and "-hf" in both.stderr
    bad = run(runner_bin, hub["env"], "-hf", "no-slash-here", "-p", "hi")
    assert bad.returncode != 0 and "owner/repo" in bad.stderr
    assert calls(hub["log"]) == []


def test_help_documents_the_flag(runner_bin):
    r = subprocess.run([str(runner_bin), "--help"], capture_output=True, text=True, cwd=ROOT)
    assert "-hf REPO[:TAG]" in r.stdout + r.stderr


def test_a_listing_with_a_traversing_name_is_refused_whole(runner_bin, hub, tmp_path):
    """The file list is the repository owner's data. A name with a `..`
    segment became a cache path under it and was written wherever it pointed
    (sweep, 2026-09-27); an absolute path or a drive letter did the same.
    The runner refuses the repository and fetches nothing."""
    files = json.loads((hub["dir"] / "files.json").read_text())
    good = hub["dir"] / "model-Q8_0.gguf"
    for bad in ("../escape-Q8_0.gguf", "/abs/escape-Q8_0.gguf", "sub/../../up-Q8_0.gguf",
                "C:\\escape-Q8_0.gguf", "a//b-Q8_0.gguf"):
        listing = dict(files)
        listing["siblings"] = files["siblings"] + [
            {"rfilename": bad, "size": good.stat().st_size,
             "lfs": {"sha256": sha256(good), "size": good.stat().st_size}}]
        (hub["dir"] / "files.json").write_text(json.dumps(listing))
        if hub["log"].exists():
            hub["log"].unlink()
        p = run(runner_bin, hub["env"], "-hf", "owner/repo:Q8_0", "-p", "hi", "-n", "1",
                "--gpu", "off", "-t", "2")
        assert p.returncode != 0, (bad, p.stderr)
        assert "will not hold" in p.stderr, (bad, p.stderr)
        # nothing was fetched: the listing was the only transfer
        assert resolve_calls(hub["log"]) == [], (bad, calls(hub["log"]))
        assert not (tmp_path / "escape-Q8_0.gguf").exists()


def test_a_listing_with_an_impossible_size_is_refused(runner_bin, hub):
    """A listed size is the repository owner's data too, and it is what
    verifies a file with no LFS record. It was converted with a plain cast:
    1e300 is undefined behaviour there, and on x86 it became a negative size
    that read as "unknown", so the download was accepted unverified; arm64
    saturates and refused the same listing. A size that is not a whole,
    non-negative byte count refuses the repository before anything is
    fetched, the way a traversing name does."""
    files = json.loads((hub["dir"] / "files.json").read_text())
    good = hub["dir"] / "model-Q8_0.gguf"
    for bad in (1e300, -5, 12.5):
        listing = dict(files)
        listing["siblings"] = [s for s in files["siblings"]
                               if s["rfilename"] != "model-Q8_0.gguf"] + [
            {"rfilename": "model-Q8_0.gguf", "size": bad}]
        (hub["dir"] / "files.json").write_text(json.dumps(listing))
        if hub["log"].exists():
            hub["log"].unlink()
        p = run(runner_bin, hub["env"], "-hf", "owner/repo:Q8_0", "-p", "hi", "-n", "1",
                "--gpu", "off", "-t", "2")
        assert p.returncode != 0, (bad, p.stderr)
        assert "size" in p.stderr, (bad, p.stderr)
        assert resolve_calls(hub["log"]) == [], (bad, calls(hub["log"]))
    assert good.exists()


def test_a_receipt_names_the_hub_source_not_the_cache(runner_bin, hub, tmp_path):
    # R1.17: a model fetched with -hf is named by base name, sha256 and its
    # source, the commit the listing named; the cache path stays out
    rec = tmp_path / "r.json"
    r = run(runner_bin, hub["env"], "-hf", "fake/repo:Q8_0", "-p", "hi", "-n", "2",
            "--temp", "0", "--gpu", "off", "--transcript", str(rec))
    assert r.returncode == 0, r.stderr
    m = json.loads(rec.read_text())["model"]
    assert m["name"] == "model-Q8_0.gguf" and "path" not in m
    assert m["source"] == "hf:fake/repo@main"     # a listing without a commit
    assert str(hub["cache"]) not in rec.read_text()
    files = json.loads((hub["dir"] / "files.json").read_text())
    files["sha"] = "0123456789abcdef0123456789abcdef01234567"
    (hub["dir"] / "files.json").write_text(json.dumps(files))
    r = run(runner_bin, hub["env"], "-hf", "fake/repo:Q8_0", "-p", "hi", "-n", "2",
            "--temp", "0", "--gpu", "off", "--transcript", str(rec))
    assert r.returncode == 0, r.stderr
    m = json.loads(rec.read_text())["model"]
    assert m["source"] == "hf:fake/repo@0123456789abcdef0123456789abcdef01234567"


def test_a_multipart_file_in_a_folder_keeps_its_folder(runner_bin, hub):
    """docs/model-tiers.json records such a file's `path`; every part must be
    asked for at that path, as the Hub lists it."""
    folder = hub["dir"] / "IQ3_S"
    folder.mkdir()
    siblings = []
    for k in (1, 2):
        name = f"IQ3_S/model-IQ3_S-0000{k}-of-00002.gguf"
        (hub["dir"] / name).write_bytes(b"part %d" % k)
        siblings.append({"rfilename": name, "size": 6,
                         "lfs": {"sha256": sha256(hub["dir"] / name), "size": 6}})
    (hub["dir"] / "files.json").write_text(json.dumps({"siblings": siblings}))
    r = run(runner_bin, hub["env"], "--pull", "fake/repo:IQ3_S")
    assert r.returncode == 0, r.stderr
    assert [c[-1] for c in resolve_calls(hub["log"])] == [
        f"https://huggingface.co/fake/repo/resolve/main/IQ3_S/model-IQ3_S-0000{k}-of-00002.gguf"
        for k in (1, 2)]


def test_a_file_in_a_repository_folder_is_fetched_by_its_path(runner_bin, hub):
    """R10.28: a listed file can sit in a folder of its repository (the Qwen3-4B tool-use repo keeps
    merged/merged-Q8_0.gguf beside study/ adapters, as its Hub listing showed on 2026-11-09). The
    tag matches the base name, and the transfer keeps the folder: resolve/main/merged/<file>."""
    sub = hub["dir"] / "merged"
    sub.mkdir()
    model = sub / "merged-Q8_0.gguf"
    model.write_bytes((hub["dir"] / "model-Q8_0.gguf").read_bytes())
    (hub["dir"] / "study").mkdir()
    adapter = hub["dir"] / "study" / "adapter-trained-through-Q8_0.gguf"
    adapter.write_bytes(b"not fetched")
    sib = lambda rf, p: {"rfilename": rf, "size": p.stat().st_size,
                         "lfs": {"sha256": sha256(p), "size": p.stat().st_size}}
    (hub["dir"] / "files.json").write_text(json.dumps({"siblings": [
        {"rfilename": "README.md", "size": 10},
        sib("merged/merged-Q8_0.gguf", model),
        sib("study/adapter-trained-through-Q8_0.gguf", adapter),
    ]}))
    r = run(runner_bin, hub["env"], "-hf", "fake/repo:merged-Q8_0", "-p", "hi", "-n", "2",
            "--temp", "0")
    assert r.returncode == 0, r.stderr
    assert [c[-1] for c in resolve_calls(hub["log"])] == [
        "https://huggingface.co/fake/repo/resolve/main/merged/merged-Q8_0.gguf"]
    cached = [p for p in (hub["cache"] / "fake--repo").rglob("*merged-Q8_0.gguf")]
    assert len(cached) == 1 and sha256(cached[0]) == sha256(model), cached
    # the folder's other Q8_0 file is a second match for the short tag, so it is named, not guessed
    r = run(runner_bin, hub["env"], "-hf", "fake/repo:Q8_0", "-p", "hi", "-n", "2")
    assert r.returncode != 0 and "merged-Q8_0.gguf" in r.stderr, r.stderr


def test_a_file_name_with_url_characters_is_fetched(runner_bin, hub):
    """A repository file name is put into the transfer URL, and a name is not
    a URL: curl refuses one with a space outright (exit 3), and '#' or '?'
    cut the path short. Each byte outside the unreserved set is percent
    encoded; '/' stays the folder separator."""
    sub = hub["dir"] / "my folder"
    sub.mkdir()
    name = "my folder/my model #2 (50%)?-Q8_0.gguf"
    model = hub["dir"] / name
    model.write_bytes((hub["dir"] / "model-Q8_0.gguf").read_bytes())
    (hub["dir"] / "files.json").write_text(json.dumps({"siblings": [
        {"rfilename": name, "size": model.stat().st_size,
         "lfs": {"sha256": sha256(model), "size": model.stat().st_size}}]}))
    r = run(runner_bin, hub["env"], "--pull", "fake/repo:Q8_0")
    assert r.returncode == 0, r.stderr
    assert [c[-1] for c in resolve_calls(hub["log"])] == [
        "https://huggingface.co/fake/repo/resolve/main/"
        "my%20folder/my%20model%20%232%20%2850%25%29%3F-Q8_0.gguf"]
