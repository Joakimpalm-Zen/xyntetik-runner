"""The served model id is the file's whole basename, however long it is.

Single-model serve joins the registry machinery so /unload and --ttl work,
and that copy used to run through a 64-byte registry name. A basename longer
than that was TRUNCATED rather than refused, so the server listed an id no
file had and every client that named the file got 404 from a model that was
loaded and serving. Measured on the lab box (2026-09-24) with a 68-character
checkpoint whose cut landed exactly where `.gguf` began, which is why it read
as "the extension is being stripped".

The same run under `--parallel 4` skips that join and kept the full name,
which is why two servers on the same file disagreed about its id.
"""
import json
import pathlib
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import RunnerServer, find_runner  # noqa: E402

# 68 characters: the lab's checkpoint name, the case that found this
LONG_NAME = "Xyntetik-Kvist-14B-train-w14b-envelope-a5-checkpoint-00600-bf16.gguf"


@pytest.fixture(scope="module")
def long_named_model(tmp_path_factory):
    base = tmp_path_factory.mktemp("longname")
    src = base / "src.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(src)],
                   check=True, stdout=subprocess.DEVNULL)
    dst = base / LONG_NAME
    shutil.move(src, dst)
    return dst


def _ids(server):
    with urllib.request.urlopen(server.base_url + "/v1/models", timeout=30) as r:
        return [m["id"] for m in json.load(r)["data"]]


def _complete(server, name):
    req = urllib.request.Request(
        server.base_url + "/v1/completions",
        data=json.dumps({"model": name, "prompt": "hi", "max_tokens": 3,
                         "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


@pytest.mark.parametrize("parallel", [1, 2])
def test_the_served_id_is_the_whole_basename(long_named_model, parallel):
    """Both slot counts report the same id, and it is the file's own name."""
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    assert len(LONG_NAME) > 64, "the fixture must exceed the old registry field"
    with RunnerServer(exe, long_named_model, ctx=512, parallel=parallel,
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        assert _ids(srv) == [LONG_NAME]
        assert _complete(srv, LONG_NAME) == 200
        # the truncation that used to be served is not a name any more
        assert _complete(srv, LONG_NAME[:63]) == 404
