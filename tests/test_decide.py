"""Typed decisions (R13.10): POST /v1/decide and `--decide FILE`.

The readout's contract, pinned on the tiny dense fixture:
  * probabilities are the softmax over the given options' total log-probs,
    argmax agrees, every logprob is finite and non-positive;
  * an option's log-prob is the SUM of its in-context token conditionals:
    it equals what `--score` reports for the same text positions on the
    concatenation prompt + option (to the solo-versus-batched envelope the
    score test already pins, 1e-4 here since only the prompt rows differ);
  * a longer option sharing a shorter one's prefix scores no higher than the
    shorter one (each conditional is at most 0), which is the trie sharing
    made visible;
  * option order and question order do not change any number (every scored
    conditional comes from a solo one-token forward), so permuting the
    options returns the same logprob per option string, exactly;
  * the CLI mode and the server route produce the same numbers;
  * malformed requests are refused with a 400 naming the problem, never a
    silent partial answer.
"""
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    d = tmp_path_factory.mktemp("decide")
    dense = d / "dense.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(dense)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return dense


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def server(runner_bin, model):
    port = _free_port()
    proc = subprocess.Popen(
        [str(runner_bin), "-m", str(model), "--serve", "--port", str(port),
         "-c", "512", "--gpu", "off", "--no-tray", "-t", "2"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            urllib.request.urlopen(base + "/v1/models", timeout=2).read()
            break
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(proc.stderr.read().decode(errors="replace"))
            time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError("server did not come up")
    yield base
    proc.kill()
    proc.wait()


def _post(base, body, expect=200):
    req = urllib.request.Request(base + "/v1/decide", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            assert expect == 200, "expected an error"
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        assert e.code == expect, (e.code, e.read())
        return json.loads(e.read() or b"{}") if expect != 200 else None


STATE = "the quick brown fox jumps over the lazy dog"
QUESTION = "what did the fox do"


def test_shape_and_invariants(server):
    out = _post(server, {"state": STATE, "questions": [
        {"id": "q1", "question": QUESTION, "options": ["jumps", "sleeps", "jumps over"]},
    ]})
    assert out["object"] == "decide"
    assert out["usage"]["completion_tokens"] == 0
    assert out["envelope"]["rendering"] == "raw-v1"
    assert len(out["envelope"]["state_sha256"]) == 64
    d = out["decisions"][0]
    assert d["id"] == "q1"
    assert d["options"] == ["jumps", "sleeps", "jumps over"]
    lps, ps = d["logprobs"], d["probs"]
    assert all(lp <= 0 for lp in lps)
    assert abs(sum(ps) - 1.0) < 1e-6
    assert d["argmax"] == max(range(3), key=lambda i: lps[i])
    assert all(n >= 1 for n in d["n_tokens"])
    # the trie: "jumps over" extends "jumps", so it can score no higher
    assert lps[2] <= lps[0] + 1e-9
    assert d["n_tokens"][2] > d["n_tokens"][0]


def test_order_invariance(server):
    opts = ["jumps", "sleeps", "jumps over", "barks loudly"]
    a = _post(server, {"state": STATE, "questions": [{"question": QUESTION, "options": opts}]})
    b = _post(server, {"state": STATE, "questions": [{"question": QUESTION, "options": opts[::-1]}]})
    la = dict(zip(a["decisions"][0]["options"], a["decisions"][0]["logprobs"]))
    lb = dict(zip(b["decisions"][0]["options"], b["decisions"][0]["logprobs"]))
    assert la == lb, (la, lb)
    # and a second question on the same state does not disturb the first
    c = _post(server, {"state": STATE, "questions": [
        {"question": "who is lazy", "options": ["the dog", "the fox"]},
        {"question": QUESTION, "options": opts},
    ]})
    lc = dict(zip(c["decisions"][1]["options"], c["decisions"][1]["logprobs"]))
    assert lc == la, (lc, la)


def _score_positions(runner_bin, model, text):
    # the same -c as the server: the fixture trains at 256, so a different
    # context means a different YaRN factor and different logits
    p = subprocess.run([str(runner_bin), "-m", str(model), "--score", "-p", text,
                        "-t", "2", "--gpu", "off", "-c", "512"], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(p.stdout)


def test_matches_teacher_forced_score(runner_bin, model, server):
    # decide's total for an option == the sum of --score's per-position
    # log-probs over the option's positions in the concatenated text
    prompt = STATE + "\n\n" + QUESTION + "\n"
    for opt in ["jumps", "jumps over"]:
        out = _post(server, {"state": STATE, "questions": [
            {"question": QUESTION, "options": [opt, "zzz"]}]})
        d = out["decisions"][0]
        n_tok = d["n_tokens"][0]
        sc = _score_positions(runner_bin, model, prompt + opt)
        lps = sc["logprobs"]           # log P(token i+1 | tokens[:i+1])
        expected = sum(lps[-n_tok:])
        assert abs(d["logprobs"][0] - expected) < 1e-4, (d["logprobs"][0], expected, n_tok)


def test_cli_matches_server(runner_bin, model, server, tmp_path):
    req = {"state": STATE, "answer_prefix": "Answer: ",
           "questions": [{"id": "a", "question": QUESTION, "options": ["jumps", "sleeps"]},
                         {"id": "b", "question": "who is lazy", "options": ["the dog", "the fox", "nobody"]}]}
    srv = _post(server, req)
    f = tmp_path / "q.jsonl"
    f.write_text(json.dumps(req) + "\n" + json.dumps(req) + "\n")
    p = subprocess.run([str(runner_bin), "-m", str(model), "--decide", str(f), "-t", "2", "--gpu", "off", "-c", "512"],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    lines = [json.loads(l) for l in p.stdout.decode().splitlines() if l.strip()]
    assert len(lines) == 2
    for cli in lines:
        for ds, dc in zip(srv["decisions"], cli["decisions"]):
            assert ds["logprobs"] == dc["logprobs"], (ds, dc)
            assert ds["probs"] == dc["probs"]
    assert srv["envelope"]["questions_sha256"] == lines[0]["envelope"]["questions_sha256"]


def test_refusals(server):
    _post(server, {"questions": [{"question": "x", "options": ["a", "b"]}]}, expect=400)
    _post(server, {"state": STATE, "questions": []}, expect=400)
    _post(server, {"state": STATE, "questions": [{"question": "x", "options": ["only"]}]}, expect=400)
    _post(server, {"state": STATE, "questions": [{"question": "x", "options": ["a", ""]}]}, expect=400)
    _post(server, {"state": STATE, "questions": [{"question": "x", "options": ["a", "a"]}]}, expect=400)
    _post(server, {"state": STATE, "questions": [{"question": "", "options": ["a", "b"]}]}, expect=400)
