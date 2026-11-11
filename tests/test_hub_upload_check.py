"""scripts/hub-upload-check.py: the plan of a Hub upload, refused when it would
overwrite or delete a file nobody named (R7.1.5).

No network: a stub Hub on loopback answers the two calls the script makes (the
revision and the recursive tree listing, paged the way the Hub pages it), and
HF_ENDPOINT points the script at it, as it does for huggingface_hub. The local
hashes are anchored outside the script: a small file's blob id must equal
`git hash-object`, and an LFS file is compared by sha256 of its bytes.
"""
import hashlib
import http.server
import json
import os
import pathlib
import subprocess
import sys
import threading
import urllib.parse

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hub-upload-check.py"
REPO = "someone/some-model"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
TOKEN = "hf_secretTOKENvalue123"


def blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def lfs_entry(path, data):
    return {"type": "file", "path": path, "size": len(data), "oid": blob_id(b"pointer " + path.encode()),
            "lfs": {"oid": hashlib.sha256(data).hexdigest(), "size": len(data), "pointerSize": 133}}


def small_entry(path, data):
    return {"type": "file", "path": path, "size": len(data), "oid": blob_id(data)}


class Hub:
    """A stub of the two Hub API routes the script reads."""

    def __init__(self, entries, *, repo_type="model", exists=True, page=None):
        self.entries = entries
        self.repo_type = repo_type
        self.exists = exists
        self.page = page
        self.auth = []
        hub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                hub.auth.append(self.headers.get("Authorization"))
                url = urllib.parse.urlsplit(self.path)
                q = urllib.parse.parse_qs(url.query)
                base = f"/api/{hub.repo_type}s/{REPO}"
                if not hub.exists:
                    return self._send(404, {"error": "Repository not found"})
                if url.path == f"{base}/revision/main":
                    return self._send(200, {"id": REPO, "sha": COMMIT})
                if url.path == f"{base}/tree/main" and q.get("recursive") == ["true"]:
                    items = [{"type": "directory", "path": "sub", "oid": "d" * 40, "size": 0}] + hub.entries
                    start = int(q.get("cursor", ["0"])[0])
                    size = hub.page or len(items) or 1
                    chunk = items[start:start + size]
                    headers = {}
                    if start + size < len(items):
                        nxt = f"http://127.0.0.1:{self.server.server_port}{url.path}?recursive=true&cursor={start + size}"
                        headers["Link"] = f'<{nxt}>; rel="next"'
                    return self._send(200, chunk, headers)
                return self._send(404, {"error": f"no route {url.path}"})

            def _send(self, code, obj, headers=None):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    @property
    def endpoint(self):
        return f"http://127.0.0.1:{self.server.server_port}"


def check(hub, folder, *args, token=None):
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    env["HF_ENDPOINT"] = hub.endpoint
    if token:
        env["HF_TOKEN"] = token
    return subprocess.run([sys.executable, str(SCRIPT), REPO, str(folder), *args],
                          capture_output=True, text=True, env=env, timeout=60)


def write(folder, files):
    for rel, data in files.items():
        p = folder / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return folder


OLD_GGUF, NEW_GGUF = b"GGUF old weights" * 64, b"GGUF new weights" * 64
CARD = b"---\nlicense: apache-2.0\n---\n# A model\n"


def test_a_silent_overwrite_is_refused(tmp_path):
    remote = [lfs_entry("model.gguf", OLD_GGUF), small_entry("README.md", CARD)]
    local = write(tmp_path / "up", {"model.gguf": NEW_GGUF, "README.md": CARD})
    with Hub(remote) as hub:
        p = check(hub, local)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "replace" in p.stdout and "model.gguf" in p.stdout
    assert "REFUSED" in p.stdout
    assert "--replace model.gguf" in p.stdout


def test_a_named_replacement_is_allowed_and_shown(tmp_path):
    remote = [lfs_entry("model.gguf", OLD_GGUF)]
    local = write(tmp_path / "up", {"model.gguf": NEW_GGUF})
    with Hub(remote) as hub:
        p = check(hub, local, "--replace", "model.gguf")
    assert p.returncode == 0, p.stdout + p.stderr
    line = next(l for l in p.stdout.splitlines() if "model.gguf" in l)
    assert line.split()[0] == "replace", line
    assert hashlib.sha256(OLD_GGUF).hexdigest()[:12] in line
    assert hashlib.sha256(NEW_GGUF).hexdigest()[:12] in line


def test_an_addition_is_allowed(tmp_path):
    remote = [small_entry("README.md", CARD)]
    local = write(tmp_path / "up", {"README.md": CARD, "model-Q8_0.gguf": NEW_GGUF})
    with Hub(remote) as hub:
        p = check(hub, local)
    assert p.returncode == 0, p.stdout + p.stderr
    line = next(l for l in p.stdout.splitlines() if "model-Q8_0.gguf" in l)
    assert line.split()[0] == "add", line


def test_identical_files_are_a_no_op(tmp_path):
    remote = [lfs_entry("model.gguf", OLD_GGUF), small_entry("README.md", CARD)]
    local = write(tmp_path / "up", {"model.gguf": OLD_GGUF, "README.md": CARD})
    with Hub(remote) as hub:
        p = check(hub, local)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "2 unchanged" in p.stdout
    assert not any(l.split()[:1] in (["add"], ["replace"], ["delete"]) for l in p.stdout.splitlines())


def test_a_changed_card_is_a_replacement_by_blob_id(tmp_path):
    remote = [small_entry("README.md", CARD)]
    local = write(tmp_path / "up", {"README.md": CARD + b"\nA new line.\n"})
    with Hub(remote) as hub:
        refused = check(hub, local)
        named = check(hub, local, "--replace", "README.md")
    assert refused.returncode == 1, refused.stdout
    assert named.returncode == 0, named.stdout + named.stderr


def test_a_remote_file_missing_locally_is_a_deletion_unless_named_or_partial(tmp_path):
    remote = [small_entry("README.md", CARD), lfs_entry("old-Q4_0.gguf", OLD_GGUF)]
    local = write(tmp_path / "up", {"README.md": CARD})
    with Hub(remote) as hub:
        refused = check(hub, local)
        named = check(hub, local, "--delete", "old-Q4_0.gguf")
        partial = check(hub, local, "--partial")
    assert refused.returncode == 1, refused.stdout
    assert "delete" in refused.stdout and "--delete old-Q4_0.gguf" in refused.stdout
    assert named.returncode == 0, named.stdout + named.stderr
    assert partial.returncode == 0, partial.stdout + partial.stderr
    assert "1 remote file is not in the folder" in partial.stdout


def test_many_refusals_end_in_counts_not_a_wall_of_flags(tmp_path):
    remote = [small_entry("README.md", CARD)] + [lfs_entry(f"q{i}.gguf", OLD_GGUF) for i in range(12)]
    local = write(tmp_path / "up", {"README.md": CARD + b"edit"})
    with Hub(remote) as hub:
        p = check(hub, local)
    assert p.returncode == 1, p.stdout + p.stderr
    last = p.stdout.splitlines()[-1]
    assert last.startswith("REFUSED: 13 changes nobody named (1 replacement, 12 deletions)"), last
    assert "--partial" in last and "--delete q" not in last
    assert sum(1 for l in p.stdout.splitlines() if l.startswith("delete ")) == 12

def test_an_overwrite_on_a_later_page_is_still_caught(tmp_path):
    remote = [small_entry(f"notes/{i}.md", b"note %d" % i) for i in range(7)] + [lfs_entry("model.gguf", OLD_GGUF)]
    files = {f"notes/{i}.md": b"note %d" % i for i in range(7)}
    files["model.gguf"] = NEW_GGUF
    local = write(tmp_path / "up", files)
    with Hub(remote, page=3) as hub:
        p = check(hub, local)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "7 unchanged" in p.stdout


def test_a_name_that_matches_nothing_on_the_remote_is_an_error(tmp_path):
    local = write(tmp_path / "up", {"model.gguf": NEW_GGUF})
    with Hub([lfs_entry("model.gguf", OLD_GGUF)]) as hub:
        p = check(hub, local, "--replace", "modle.gguf")
    assert p.returncode == 2, p.stdout + p.stderr
    assert "modle.gguf" in p.stderr


def test_the_token_is_sent_and_never_printed(tmp_path):
    local = write(tmp_path / "up", {"model.gguf": NEW_GGUF})
    with Hub([lfs_entry("model.gguf", OLD_GGUF)]) as hub:
        p = check(hub, local, token=TOKEN)
        auth = list(hub.auth)
    assert auth and all(a == f"Bearer {TOKEN}" for a in auth), auth
    assert TOKEN not in p.stdout + p.stderr
    with Hub([], exists=False) as hub:
        missing = check(hub, local, token=TOKEN)
    assert missing.returncode == 2
    assert TOKEN not in missing.stdout + missing.stderr


def test_no_token_sends_no_authorization(tmp_path):
    local = write(tmp_path / "up", {"model.gguf": OLD_GGUF})
    with Hub([lfs_entry("model.gguf", OLD_GGUF)]) as hub:
        p = check(hub, local)
        auth = list(hub.auth)
    assert p.returncode == 0, p.stdout + p.stderr
    assert auth and all(a is None for a in auth), auth


def test_a_missing_repository_is_an_error_unless_declared_new(tmp_path):
    local = write(tmp_path / "up", {"model.gguf": NEW_GGUF})
    with Hub([], exists=False) as hub:
        missing = check(hub, local)
        new = check(hub, local, "--new")
    assert missing.returncode == 2, missing.stdout + missing.stderr
    assert new.returncode == 0, new.stdout + new.stderr
    assert next(l for l in new.stdout.splitlines() if "model.gguf" in l).split()[0] == "add"
    with Hub([lfs_entry("model.gguf", OLD_GGUF)]) as hub:
        exists = check(hub, local, "--new")
    assert exists.returncode == 2, exists.stdout + exists.stderr


def test_path_in_repo_and_dataset_repos(tmp_path):
    remote = [small_entry("data/a.jsonl", b"a\n"), small_entry("README.md", CARD)]
    local = write(tmp_path / "up", {"a.jsonl": b"a changed\n"})
    with Hub(remote, repo_type="dataset") as hub:
        p = check(hub, local, "--repo-type", "dataset", "--path-in-repo", "data")
    assert p.returncode == 1, p.stdout + p.stderr
    assert "data/a.jsonl" in p.stdout
    assert "README.md" not in [l.split()[-1] for l in p.stdout.splitlines() if l.startswith("delete")]


def test_git_and_hub_cache_folders_are_not_part_of_the_upload(tmp_path):
    local = write(tmp_path / "up", {"model.gguf": OLD_GGUF, ".git/config": b"x",
                                    ".cache/huggingface/upload/model.gguf.metadata": b"y"})
    with Hub([lfs_entry("model.gguf", OLD_GGUF)]) as hub:
        p = check(hub, local)
    assert p.returncode == 0, p.stdout + p.stderr
    assert ".git" not in p.stdout and ".cache" not in p.stdout


def test_the_plan_names_the_commit_it_compared_against(tmp_path):
    local = write(tmp_path / "up", {"model.gguf": OLD_GGUF})
    with Hub([lfs_entry("model.gguf", OLD_GGUF)]) as hub:
        p = check(hub, local)
    assert COMMIT in p.stdout


def test_blob_id_matches_git(tmp_path):
    """The absolute anchor: the small-file id the Hub reports is git's blob id."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("hub_upload_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    f = tmp_path / "card.md"
    f.write_bytes(CARD + b"\r\nwindows line\r\n\x00binary")
    git = subprocess.run(["git", "hash-object", "--no-filters", str(f)], capture_output=True, text=True)
    if git.returncode != 0:
        pytest.skip("git is not available")
    assert mod.file_hashes(f)["blob"] == git.stdout.strip()
    assert mod.file_hashes(f)["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
