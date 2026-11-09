"""Speculative decoding defaults per workload (R3.39.4).

The lookup and the NextN head emit plain decoding's tokens, so a default is a
speed question, answered by the measured rows
(docs/compat-reports/r3.3.7-draft-width-defaults-2026-11-03): with no source
named, a server request with draft_hints takes the lookup and a CLI run whose
output is one JSON document takes Qwen3.5-4B's NextN head, both only with
every layer on a CUDA device. Anything else decodes plain, because the head
reads 0.88x plain on prose and the lookup has no prose row.

The decision table itself is pinned in tests/test_lookup_draft.c. This file
pins what each surface does with a request without the signal (plain, on
every backend), the explicit `--draft off`, and that a JSON run's record
replays under --verify. The positive CUDA rows need a CUDA device; the head's
positive row also needs the measured file, which no fixture is (the
geometry gate is cli_head_measured), so it is checked on the CUDA gate
with the real file, not here.
"""
import json
import os
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import RunnerServer, find_runner  # noqa: E402

# The prompt repeats, so a lookup would find a match from its first round: a
# run that stays plain here stays plain because nothing turned the lookup on.
REPEAT = ("the cat sat on the mat and the dog sat on the mat and the cat sat "
          "on the mat and the dog sat on the mat and the cat sat on the")


@pytest.fixture(scope="module")
def model():
    m = ROOT / "test.gguf"
    if not m.exists():
        pytest.skip("test.gguf fixture not built")
    return m


@pytest.fixture(scope="module")
def mtp_model(tmp_path_factory):
    out = tmp_path_factory.mktemp("dd") / "mtp.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    "--mtp-layers", "1", str(out)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return out


def _cuda(runner_bin):
    caps = subprocess.run([runner_bin, "--caps"], cwd=ROOT, stdout=subprocess.PIPE, timeout=30)
    return (json.loads(caps.stdout.decode()).get("gpu") or {}).get("backend") == "cuda"


def _gpu_args(runner_bin):
    # every layer on the device where CUDA is (the defaults' one scope);
    # elsewhere the build's own default placement
    return ["--gpu", "99"] if _cuda(runner_bin) else []


def _run(runner_bin, model, *extra):
    p = subprocess.run(
        [runner_bin, "-m", str(model), "-p", REPEAT, "-n", "16", "-c", "512",
         "--temp", "0", "-t", "2", *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
        env={**os.environ, "RUNNER_DEBUG_ROUTE": "1"})
    return p.returncode, p.stderr.decode(errors="replace")


def _plain(err):
    routes = [ln for ln in err.splitlines() if ln.startswith("route:")]
    assert routes, err
    return (not any(" draft=" in ln for ln in routes)
            and not any(ln.startswith("spec:") for ln in err.splitlines()))


@pytest.mark.parametrize("which", ["plain", "mtp"])
def test_cli_prose_decodes_plain(runner_bin, model, mtp_model, which):
    """No --json: no signal, so no source, with or without a NextN block."""
    m = model if which == "plain" else mtp_model
    rc, err = _run(runner_bin, m, *_gpu_args(runner_bin))
    assert rc == 0, err
    assert _plain(err), err


def test_cli_json_without_the_measured_head_decodes_plain(runner_bin, mtp_model):
    """--json is the tool signal, but this head has no row: plain decoding."""
    rc, err = _run(runner_bin, mtp_model, "--json", *_gpu_args(runner_bin))
    assert rc == 0, err
    assert _plain(err), err
    assert "the default for JSON output" not in err


def test_cli_draft_off_is_a_source_of_its_own(runner_bin, model):
    rc, err = _run(runner_bin, model, "--draft", "off", "--json", *_gpu_args(runner_bin))
    assert rc == 0, err
    assert _plain(err), err
    rc, err = _run(runner_bin, model, "--draft", "off", "--draft", "lookup")
    assert rc != 0 and "--draft off names no source" in err, err


def test_cli_lookup_route_line_names_the_source(runner_bin, model):
    """A named source shows on the route line, without ':default'."""
    rc, err = _run(runner_bin, model, "--gpu", "off", "--draft", "lookup")
    assert rc == 0, err
    assert any(ln.startswith("route:") and " draft=lookup " in ln
               for ln in err.splitlines()), err


def test_json_record_replays_under_verify(runner_bin, model, tmp_path):
    """A --json run's record verifies. Where the default takes the head, the
    replay decodes plain (verify is not a default's mode) and the identity
    class gives it the same tokens."""
    rec = tmp_path / "r.json"
    p = subprocess.run(
        [runner_bin, "-m", str(model), "-p", REPEAT, "-n", "16", "-c", "512",
         "--temp", "0", "-t", "2", "--json", "--record", str(rec),
         *_gpu_args(runner_bin)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    v = subprocess.run(
        [runner_bin, "-m", str(model), "--verify", str(rec), "--json", "-t", "2",
         *_gpu_args(runner_bin)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    assert v.returncode == 0, v.stderr.decode(errors="replace")


# ------------------------------------------------------------------- server

def _get(srv, path):
    with urllib.request.urlopen(srv.base_url + path, timeout=30) as r:
        return json.load(r)


def _post(srv, body):
    req = urllib.request.Request(
        srv.base_url + "/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _req(**extra):
    return {"prompt": REPEAT, "max_tokens": 16, "temperature": 0,
            "cache_prompt": False, **extra}


def test_server_request_without_hints_decodes_plain(runner_bin, model, tmp_path):
    """Prose over the server: no draft_hints, no source, before and after a
    request that carried them (the lookup is one request's, never the slot's)."""
    cuda = _cuda(runner_bin)
    log = tmp_path / "srv.log"
    with RunnerServer(find_runner(ROOT), model, ctx=1024, parallel=1, log_path=log,
                      extra_args=["-t", "2", *_gpu_args(runner_bin)],
                      env={"RUNNER_DEBUG_ROUTE": "1"}) as srv:
        draft = _get(srv, "/v1/capabilities")["draft"]
        assert draft["requested"] is False and draft["active"] is False
        assert ("default" in draft) is cuda, draft
        t = _post(srv, _req())["runner_telemetry"]
        assert t["speculative"] is False and "speculation" not in t
        if cuda:
            hinted = _post(srv, _req(draft_hints=["the cat sat on the mat"]))
            assert hinted["runner_telemetry"]["speculation"]["source"] == "lookup"
            after = _post(srv, _req())["runner_telemetry"]
            assert after["speculative"] is False and "speculation" not in after
            assert "active" in _get(srv, "/v1/capabilities")["draft"]
            assert _get(srv, "/v1/capabilities")["draft"]["active"] is False
        else:
            with pytest.raises(urllib.error.HTTPError) as ei:
                _post(srv, _req(draft_hints=["the cat sat on the mat"]))
            assert ei.value.code == 400
    text = log.read_text(errors="replace")
    if cuda:
        assert " draft=lookup:default " in text, text[-2000:]
    else:
        assert " draft=" not in text, text[-2000:]


def test_server_draft_off_refuses_hints(runner_bin, model):
    with RunnerServer(find_runner(ROOT), model, ctx=1024, parallel=1,
                      extra_args=["-t", "2", "--draft", "off",
                                  *_gpu_args(runner_bin)]) as srv:
        assert "default" not in _get(srv, "/v1/capabilities")["draft"]
        with pytest.raises(urllib.error.HTTPError) as ei:
            _post(srv, _req(draft_hints=["the cat sat on the mat"]))
        assert ei.value.code == 400
        t = _post(srv, _req())["runner_telemetry"]
        assert t["speculative"] is False


# ------------------------------------------ a defaulted run's record verifies
#
# RUNNER_TEST_DRAFT_SCOPE=1 makes the default sites treat this host and any
# single NextN head as the measured scope, so the defaults fire on fixtures.
# The replay runs WITHOUT it: a record replays the source it names, never the
# answer today's default rule would give. On a CUDA host (the CUDA gate) this
# is the case the measured file exposed: the head's walk takes no decode
# graph, so a plain replay took another route and the verify refused it.

SCOPE = {"RUNNER_TEST_DRAFT_SCOPE": "1"}


def _scope_gpu(runner_bin):
    # every layer on CUDA where it is; elsewhere the CPU, the head's route
    return ["--gpu", "99"] if _cuda(runner_bin) else ["--gpu", "off"]


def test_a_defaulted_head_records_and_verifies(runner_bin, mtp_model, tmp_path):
    rec = tmp_path / "r.json"
    p = subprocess.run(
        [runner_bin, "-m", str(mtp_model), "-p", REPEAT, "-n", "16", "-c", "512",
         "--temp", "0", "-t", "2", "--json", "--record", str(rec),
         *_scope_gpu(runner_bin)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
        env={**os.environ, **SCOPE, "RUNNER_DEBUG_ROUTE": "1"})
    err = p.stderr.decode(errors="replace")
    assert p.returncode == 0, err
    assert "the default for JSON output" in err, err
    assert " draft=mtp:default " in err, err
    spec = json.loads(rec.read_text())["speculation"]
    assert spec["source"] == "mtp" and spec["default"] is True, spec
    v = subprocess.run(
        [runner_bin, "-m", str(mtp_model), "--verify", str(rec), "--json", "-t", "2",
         *_scope_gpu(runner_bin)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
        env={**os.environ, "RUNNER_DEBUG_ROUTE": "1"})
    out = (v.stdout + v.stderr).decode(errors="replace")
    assert v.returncode == 0 and "VERIFIED" in out, out[-1500:]
    assert "the record's default source" in out and " draft=mtp:default " in out, out[-1500:]


def test_a_named_source_record_verifies_without_the_flag(runner_bin, mtp_model, tmp_path):
    """--draft mtp named at record time: the replay binds it from the record,
    with no flag, and the record carries no "default"."""
    rec = tmp_path / "r.json"
    p = subprocess.run(
        [runner_bin, "-m", str(mtp_model), "-p", REPEAT, "-n", "16", "-c", "512",
         "--temp", "0", "-t", "2", "--draft", "mtp", "--record", str(rec),
         *_scope_gpu(runner_bin)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    spec = json.loads(rec.read_text())["speculation"]
    assert spec["source"] == "mtp" and "default" not in spec, spec
    v = subprocess.run(
        [runner_bin, "-m", str(mtp_model), "--verify", str(rec), "-t", "2",
         *_scope_gpu(runner_bin)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
        env={**os.environ, "RUNNER_DEBUG_ROUTE": "1"})
    out = (v.stdout + v.stderr).decode(errors="replace")
    assert v.returncode == 0 and "VERIFIED" in out, out[-1500:]
    assert " draft=mtp " in out, out[-1500:]


def test_a_defaulted_lookup_receipt_verifies(runner_bin, model, tmp_path):
    """D1's record: a served request with draft_hints, after a plain request
    on the same slot, writes a receipt naming lookup by default, and the CLI
    replay binds the lookup and verifies it."""
    key = tmp_path / "k.json"
    subprocess.run([runner_bin, "--keygen", key], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pk = json.loads(key.read_text())["public_key"]
    d = tmp_path / "rc"
    with RunnerServer(find_runner(ROOT), model, ctx=512, parallel=1, env=SCOPE,
                      extra_args=["-t", "2", "--receipts", str(d), "--sign-key", str(key),
                                  *_scope_gpu(runner_bin)]) as srv:
        assert _get(srv, "/v1/capabilities")["draft"]["default"] == {"draft_hints": "lookup"}
        plain = _post(srv, _req())["runner_telemetry"]
        assert plain["speculative"] is False
        hinted = _post(srv, _req(draft_hints=["the cat sat on the mat"]))["runner_telemetry"]
        assert hinted["speculation"]["source"] == "lookup"
    names = sorted(p.name for p in d.iterdir())
    assert len(names) == 2, names
    first, second = (json.loads((d / n).read_text()) for n in names)
    assert "speculation" not in first
    assert second["speculation"]["source"] == "lookup"
    assert second["speculation"]["default"] is True
    for n in names:
        v = subprocess.run(
            [runner_bin, "-m", str(model), "--verify", str(d / n), "--trust-key", pk,
             "-t", "2", "-c", "512", *_scope_gpu(runner_bin)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        out = (v.stdout + v.stderr).decode(errors="replace")
        assert v.returncode == 0 and "VERIFIED" in out, (n, out[-1500:])
    assert "the record's default source (draft_hints)" in out, out[-1500:]
