"""Per-request adapter routing (R8.6): `--adapter NAME=PATH` loads adapters
once, and `"model": "<base>:<NAME>"` selects one for a request.

Anchors:
  * a routed request answers exactly as a server started with `--lora` on the
    same adapter (the path test_lora.py already pins against the merged model);
  * a request without a suffix, and one naming the all-zero adapter, answer
    exactly as the bare base: B = 0 adds nothing, so any leak of another
    request's adapter into them shows as a difference;
  * the adapter digest in runner_telemetry is hashlib's over the file.
Interleaving adapted and bare requests with the same prompt proves the prefix
cache never serves rows computed under one adapter to a request under another.
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

PROMPT = "the quick brown fox jumps over the lazy dog"
ARGS = ["--gpu", "off", "-t", "2", "--lora-scale", "100"]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp("route")
    base = d / "base.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(base)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, ROOT / "scripts/make-test-lora.py",
                    str(base), str(d / "fx")], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return {"base": base, "tuned": d / "fx.adapter.gguf",
            "zero": d / "fx.zero.gguf"}


def _post(srv, payload, path="/v1/completions"):
    req = urllib.request.Request(srv.base_url + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as r:
        return r.code, json.load(r)


def _answer(srv, model=None):
    payload = {"prompt": PROMPT, "max_tokens": 8, "temperature": 0,
               "logprobs": 1}
    if model:
        payload["model"] = model
    st, body = _post(srv, payload)
    assert st == 200, body
    ch = body["choices"][0]
    return (ch["text"], ch["logprobs"]["token_logprobs"]), body


@pytest.fixture(scope="module")
def refs(runner_bin, fx):
    # Each reference is taken twice: cold, and again with the prompt's KV
    # reused. CPU prefill is not batch-invariant (the reused request feeds its
    # last prompt token alone, the cold one in the batch), so the two can
    # differ in a logprob's last digit, and a routed request is held to the
    # reference with the SAME history, not to whichever was taken first.
    with RunnerServer(runner_bin, fx["base"], extra_args=ARGS) as srv:
        bare = (_answer(srv)[0], _answer(srv)[0])
    with RunnerServer(runner_bin, fx["base"],
                      extra_args=[*ARGS, "--lora", str(fx["tuned"])]) as srv:
        tuned = (_answer(srv)[0], _answer(srv)[0])
    assert bare[0] != tuned[0], "the adapter must change the answer"
    assert bare[0][0] == bare[1][0] and tuned[0][0] == tuned[1][0]
    return {"bare": bare, "tuned": tuned}


def _routed(runner_bin, fx, parallel=1):
    return RunnerServer(runner_bin, fx["base"], parallel=parallel, extra_args=[
        *ARGS, "--adapter", f"tuned={fx['tuned']}",
        "--adapter", f"null={fx['zero']}"])


def test_a_routed_request_is_the_lora_server(runner_bin, fx, refs):
    with _routed(runner_bin, fx) as srv:
        # interleaved, the same prompt each time: nothing may carry over
        # an identity's first request is cold, its later ones fork the
        # prefix its first one published (the zero adapter is its own identity)
        seen = set()
        for want, ident, model in [
                ("tuned", "tuned", "base.gguf:tuned"), ("bare", "bare", None),
                ("tuned", "tuned", "base.gguf:tuned"),
                ("bare", "bare", "base.gguf"), ("bare", "null", "base.gguf:null"),
                ("tuned", "tuned", "runner:tuned")]:
            got, body = _answer(srv, model)
            assert got == refs[want][ident in seen], (model, want)
            seen.add(ident)
        _, body = _answer(srv, "base.gguf:tuned")
        a = body["runner_telemetry"]["adapter"]
        assert a == {"name": "tuned", "sha256": hashlib.sha256(
            fx["tuned"].read_bytes()).hexdigest()}
        _, body = _answer(srv)
        assert "adapter" not in body["runner_telemetry"]
        with urllib.request.urlopen(srv.base_url + "/v1/runner/provenance",
                                    timeout=30) as r:
            listed = json.load(r)["config"]["adapters"]
        assert [(x["name"], x["sha256"]) for x in listed] == [
            ("tuned", a["sha256"]),
            ("null", hashlib.sha256(fx["zero"].read_bytes()).hexdigest())]


def test_parallel_slots_route_independently(runner_bin, fx, refs):
    import concurrent.futures as cf
    with _routed(runner_bin, fx, parallel=2) as srv:
        jobs = ["base.gguf:tuned", None] * 4
        with cf.ThreadPoolExecutor(4) as ex:
            got = list(ex.map(lambda m: (m, _answer(srv, m)[0]), jobs))
        for model, answer in got:
            # which of a slot's requests found a prefix to reuse is a race
            assert answer in refs["tuned" if model else "bare"], model


def test_models_and_refusals(runner_bin, fx):
    with _routed(runner_bin, fx) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            ids = [m["id"] for m in json.load(r)["data"]]
        assert ids == ["base.gguf", "base.gguf:tuned", "base.gguf:null"], ids
        st, body = _post(srv, {"prompt": "x", "max_tokens": 1,
                               "model": "base.gguf:nope"})
        assert st == 404, body
    # startup refusals: an adapter beside --lora, and a malformed spec
    for extra, needle in [(["--lora", str(fx["tuned"]),
                            "--adapter", f"t={fx['tuned']}"], b"--adapter"),
                          (["--adapter", "noequals"], b"NAME=PATH"),
                          (["--adapter", f"bad name={fx['tuned']}"], b"NAME")]:
        p = subprocess.run([runner_bin, "-m", str(fx["base"]), "--serve",
                            "--port", "1", "--no-tray", *ARGS, *extra],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=60)
        assert p.returncode != 0 and needle in p.stderr, p.stderr[-400:]
