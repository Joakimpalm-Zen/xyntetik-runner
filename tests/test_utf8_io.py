"""Text crosses the process boundary as UTF-8 on every platform.

On Windows the C runtime built argv, the environment and every narrow file
name in the ANSI code page. A non-ASCII prompt given with -p reached the
tokenizer as cp1252 bytes, and as "?" for what cp1252 cannot spell, while the
same text read with -f arrived intact: on the Windows CUDA host (2026-10-23)
TEXT below was 8 tokens through -f and 10 different ones through -p on
stablelm-2-zephyr-1.6b, and the README's own prompt 822 tokens against 816
on macOS. runner.exe now carries a manifest that makes UTF-8 the process code
page (src/runner.manifest) on Windows 10 1903 and later.

The anchor is outside the runner. The synthetic fixture's vocabulary is byte
fallback and nothing else (scripts/make-test-model.py: token id = 3 + byte,
BOS = 1), so the tokens of a prompt are its UTF-8 encoding as the Unicode
standard defines it, after SentencePiece's whitespace escape: U+2581 for each
space and one prepended. EXPECTED below was derived by hand from the code
points; test_the_anchor_is_the_utf8_encoding re-derives it with Python's own
encoder, a second implementation that is not ours.

On macOS and Linux these tests assert that nothing changed: -p and -f give the
same tokens, and a file under a non-ASCII path opens.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WINDOWS = sys.platform == "win32"

# "cafe, naive" with their accents, a middle dot and an em dash between the
# words, then two ideographs: two letters cp1252 has (e acute, i diaeresis),
# two symbols it has and two ideographs it does not
TEXT = "caf\u00e9 \u00b7 na\u00efve \u2014 \u65e5\u672c"
# BOS, then 3 + each byte of the UTF-8 of TEXT with U+2581 (E2 96 81) for
# each space and one prepended:
#   U+2581 E2 96 81 -> 229 153 132   U+00E9 C3 A9 -> 198 172   U+00B7 C2 B7 -> 197 186
#   U+00EF C3 AF -> 198 178          U+2014 E2 80 94 -> 229 131 151
#   U+65E5 E6 97 A5 -> 233 154 168   U+672C E6 9C AC -> 233 159 175
EXPECTED = [1,
            229, 153, 132, 102, 100, 105, 198, 172,           # cafe
            229, 153, 132, 197, 186,                          # middle dot
            229, 153, 132, 113, 100, 198, 178, 121, 104,      # naive
            229, 153, 132, 229, 131, 151,                     # em dash
            229, 153, 132, 233, 154, 168, 233, 159, 175]      # two ideographs
# a directory and file name with a cp1252 letter, two ideographs and a code
# point outside the Basic Multilingual Plane (a UTF-16 surrogate pair)
ODD_DIR = "mod\u00e8le \u65e5\u672c \U0001F600"
ODD_FILE = "caf\u00e9 \u65e5\u672c.gguf"

FAKE_CURL = r'''
import os, shutil, sys
args = sys.argv[1:]
out = args[args.index("-o") + 1]
url = args[-1]
hub = os.environ["FAKE_HUB_DIR"]
if "/api/models/" in url:
    shutil.copy(os.path.join(hub, "files.json"), out)
    sys.exit(0)
if "/resolve/main/" in url:
    shutil.copy(os.path.join(hub, url.split("/resolve/main/", 1)[1]), out)
    sys.exit(0)
sys.exit(3)
'''


def binaries():
    """runner itself; on Windows also the build without the manifest, which
    is what an older Windows (or a stripped manifest) runs."""
    names = ["runner.exe", "runner-no-manifest.exe"] if WINDOWS else ["runner"]
    return [n for n in names if (ROOT / n).exists()] or names[:1]


@pytest.fixture(scope="module", params=binaries())
def runner_bin(request):
    exe = ROOT / request.param
    if not exe.exists():
        pytest.skip(f"{request.param} not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("m") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT)
    return m


def run(exe, *args, env=None):
    return subprocess.run([str(exe), *args], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, encoding="utf-8", errors="replace",
                          timeout=120)


def score_tokens(exe, model, *prompt_args, env=None):
    p = run(exe, "-m", str(model), "--score", "-c", "64", "--gpu", "off", *prompt_args,
            env=env)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)["tokens"]


def test_the_anchor_is_the_utf8_encoding():
    spm = "\u2581" + TEXT.replace(" ", "\u2581")
    assert EXPECTED == [1] + [3 + b for b in spm.encode("utf-8")]


def test_a_prompt_given_with_p_arrives_as_utf8(runner_bin, model):
    assert score_tokens(runner_bin, model, "-p", TEXT) == EXPECTED


def test_the_same_prompt_from_a_file_tokenises_the_same(runner_bin, model, tmp_path):
    f = tmp_path / "prompt.txt"
    f.write_bytes(TEXT.encode("utf-8"))
    assert score_tokens(runner_bin, model, "-f", str(f)) == EXPECTED


def test_a_model_and_a_prompt_file_under_a_non_ascii_path_open(runner_bin, model, tmp_path):
    d = tmp_path / ODD_DIR
    d.mkdir()
    m = d / ODD_FILE
    shutil.copyfile(model, m)
    f = d / "pr\u00f6mpt \u65e5\u672c.txt"
    f.write_bytes(TEXT.encode("utf-8"))
    assert score_tokens(runner_bin, m, "-f", str(f)) == EXPECTED


def test_a_non_ascii_hf_cache_directory_is_used_as_named(runner_bin, model, tmp_path):
    """RUNNER_HF_CACHE is an environment value and a path at once: the cache
    must land in the directory it names, not in a mis-decoded sibling."""
    import hashlib
    hub = tmp_path / "hub"
    hub.mkdir()
    shutil.copyfile(model, hub / "model-Q8_0.gguf")
    blob = (hub / "model-Q8_0.gguf").read_bytes()
    size = len(blob)
    (hub / "files.json").write_text(json.dumps({"siblings": [
        {"rfilename": "model-Q8_0.gguf", "size": size,
         "lfs": {"sha256": hashlib.sha256(blob).hexdigest(), "size": size}}]}))
    script = tmp_path / "fakecurl.py"
    script.write_text(FAKE_CURL)
    if WINDOWS:
        curl = tmp_path / "fakecurl.cmd"
        curl.write_text(f'@"{sys.executable}" "{script}" %*\r\n')
    else:
        curl = tmp_path / "fakecurl"
        curl.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
        curl.chmod(0o755)
    cache = tmp_path / ODD_DIR / "hf"
    env = dict(os.environ, RUNNER_CURL=str(curl), RUNNER_HF_CACHE=str(cache),
               FAKE_HUB_DIR=str(hub))
    env.pop("HF_TOKEN", None)
    p = run(runner_bin, "-hf", "fake/repo:Q8_0", "--score", "-c", "64", "--gpu", "off",
            "-p", TEXT, env=env)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["tokens"] == EXPECTED
    cached = cache / "fake--repo" / "model-Q8_0.gguf"
    assert cached.read_bytes() == blob


def test_caps_reports_the_code_page(runner_bin):
    p = run(runner_bin, "--caps")
    assert p.returncode == 0, p.stderr
    cp = json.loads(p.stdout)["code_page"]
    if not WINDOWS:
        assert cp is None
    elif runner_bin.name == "runner.exe":
        # 65001 is CP_UTF8 (Microsoft's code page identifiers); the hosted
        # Windows image is Server 2022, which honours the manifest
        assert cp == {"ansi": 65001, "utf8_manifest": True}
    else:
        assert cp["utf8_manifest"] is False
