"""R6.68.24: a prompt read from a file with -f is text, so its CRLF line ends are
read as LF and one file gives the same tokens on every platform.

A file saved on Windows (or by an editor set to CRLF) carried its carriage
returns into the prompt: "Line one.\\r\\nLine two." tokenized to two more
tokens than its LF twin on test.gguf, so the same prompt file scored and
generated differently depending on where it was written. Only a CR that
immediately precedes an LF is dropped; a lone CR is content and stays, and -p
text is untouched (its escapes are explicit).

The anchor is the file's bytes: the LF file's tokens are the reference, and
-p with the same text spelled by escapes gives the tokens a lone CR must keep.
--score's own "tokens" array is the observable (the CPU route, so the test
runs on every CI host).
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXE = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
MODEL = ROOT / "test.gguf"


@pytest.fixture(scope="module")
def runner_bin():
    if not EXE.exists() or not MODEL.exists():
        pytest.skip("runner binary or test.gguf not built; run `make test`")
    return EXE


def tokens(exe, tmp_path, name, *, data=None, prompt=None):
    out = tmp_path / (name + ".json")
    args = [str(exe), "-m", str(MODEL), "--gpu", "off", "--score", "--score-out", str(out)]
    if data is not None:
        f = tmp_path / (name + ".txt")
        f.write_bytes(data)
        args += ["-f", str(f)]
    else:
        args += ["-p", prompt]
    r = subprocess.run(args, capture_output=True, timeout=120)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-2000:]
    return json.loads(out.read_text(encoding="utf-8"))["tokens"]


def test_a_crlf_prompt_file_tokenizes_as_its_lf_twin(runner_bin, tmp_path):
    lf = tokens(runner_bin, tmp_path, "lf", data=b"Line one.\nLine two.\nThe end.")
    crlf = tokens(runner_bin, tmp_path, "crlf", data=b"Line one.\r\nLine two.\r\nThe end.")
    assert crlf == lf
    # a file that ends in a line end, the usual case for a saved file
    lf_end = tokens(runner_bin, tmp_path, "lf-end", data=b"Line one.\nThe end.\n")
    crlf_end = tokens(runner_bin, tmp_path, "crlf-end", data=b"Line one.\r\nThe end.\r\n")
    assert crlf_end == lf_end


def test_a_lone_cr_is_content_and_stays(runner_bin, tmp_path):
    lone = tokens(runner_bin, tmp_path, "lone", data=b"Line one.\rLine two.")
    lf = tokens(runner_bin, tmp_path, "lone-lf", data=b"Line one.\nLine two.")
    assert lone != lf
    # the same text spelled by -p's escapes: a lone CR is read exactly
    assert lone == tokens(runner_bin, tmp_path, "lone-p", prompt="Line one.\\rLine two.")
    # only the CR right before an LF goes: CR CR LF reads as CR LF
    double = tokens(runner_bin, tmp_path, "crcrlf", data=b"Line one.\r\r\nLine two.")
    assert double == tokens(runner_bin, tmp_path, "crlf-p", prompt="Line one.\\r\\nLine two.")


def test_a_crlf_plain_text_data_file_trains_its_lf_twins_adapter(runner_bin, tmp_path):
    # runner train's plain-text DATA is text too: the same rule, so one data
    # file trains the same adapter on every platform. The adapter's bytes do
    # not depend on the data file's name (two LF copies give one adapter), so
    # its bytes are the observable; training runs on the CPU here.
    lf = b"The river runs past the old mill.\nThe bridge crosses it at the market.\n" * 4
    adapters = {}
    for name, data in (("lf", lf), ("lf-copy", lf), ("crlf", lf.replace(b"\n", b"\r\n"))):
        src = tmp_path / (name + ".txt")
        src.write_bytes(data)
        out = tmp_path / (name + ".gguf")
        r = subprocess.run([str(runner_bin), "train", str(src), "-m", str(MODEL), "--steps", "3",
                            "--ctx", "32", "--out", str(out)], capture_output=True, timeout=300)
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-2000:]
        adapters[name] = out.read_bytes()
    assert adapters["lf-copy"] == adapters["lf"]
    assert adapters["crlf"] == adapters["lf"]


def test_p_text_keeps_its_explicit_crlf(runner_bin, tmp_path):
    # -p's escapes are the user's own: a spelled \r\n is not a file's line end
    p_crlf = tokens(runner_bin, tmp_path, "p-crlf", prompt="Line one.\\r\\nLine two.")
    p_lf = tokens(runner_bin, tmp_path, "p-lf", prompt="Line one.\\nLine two.")
    assert p_crlf != p_lf
