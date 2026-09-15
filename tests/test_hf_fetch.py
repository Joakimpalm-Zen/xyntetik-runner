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
out = args[args.index("-o") + 1]
url = args[-1]
hub = os.environ["FAKE_HUB_DIR"]
if "/api/models/" in url:
    shutil.copy(os.path.join(hub, "files.json"), out)
    sys.exit(0)
if "/resolve/main/" in url:
    name = url.split("/resolve/main/", 1)[1]
    src = os.path.join(hub, name)
    if not os.path.exists(src):
        sys.exit(22)
    shutil.copy(src, out)
    sys.exit(0)
sys.exit(3)
'''


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


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


def test_token_is_sent_as_a_bearer_header(runner_bin, hub):
    env = dict(hub["env"], HF_TOKEN="hf_test_token")
    r = run(runner_bin, env, "-hf", "fake/repo:Q8_0", "-p", "hi", "-n", "2", "--temp", "0")
    assert r.returncode == 0, r.stderr
    for c in calls(hub["log"]):
        assert "-H" in c and c[c.index("-H") + 1] == "Authorization: Bearer hf_test_token"


def test_flag_argument_rules(runner_bin, hub, fixture_model):
    both = run(runner_bin, hub["env"], "-m", str(fixture_model), "-hf", "fake/repo:Q8_0", "-p", "hi")
    assert both.returncode != 0 and "-m" in both.stderr and "-hf" in both.stderr
    bad = run(runner_bin, hub["env"], "-hf", "no-slash-here", "-p", "hi")
    assert bad.returncode != 0 and "owner/repo" in bad.stderr
    assert calls(hub["log"]) == []


def test_help_documents_the_flag(runner_bin):
    r = subprocess.run([str(runner_bin), "--help"], capture_output=True, text=True, cwd=ROOT)
    assert "-hf REPO[:TAG]" in r.stdout + r.stderr
