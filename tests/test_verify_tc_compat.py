"""--verify judges a record's tensor-core route by what it recorded (R3.39.2).

A record now names the route in its detail (tc=taken, tc=off or tc=forced) and lists the weight types whose products
the route takes on the tensor cores (profile.route.tc_types). A record written before that named the MODE instead
("tc=promoted" whenever RUNNER_CUDA_TC was unset, the route taken or not) and had no tc_types.

- **A pre-fix record** verifies on this binary when nothing else in its route differs. Its tc word is read as "not
  recorded", and the verdict says so. Any other route token that differs still makes it UNVERIFIABLE.
- **A record that names its tensor-core types** holds the replay to them. A replay whose types differ is
  UNVERIFIABLE, and so is a replay whose label differs.

A genuine pre-fix CUDA record names its device, and its chain hash covers those bytes, so this file does not keep one.
It makes a record with this binary and writes it back the way the pre-fix writer did: the tc word set to "promoted",
tc_types dropped, the chain hash recomputed over the bytes before `,"chain"`. Records from the binary before the fix
(main 241e9d7) were verified the same way when this was written (R3.39.2's record).

The checks need a CUDA device at full offload; elsewhere the module skips.
"""
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPT = "The quick brown fox"


def _cuda(runner_bin):
    caps = subprocess.run([str(runner_bin), "--caps"], cwd=ROOT, stdout=subprocess.PIPE, timeout=30)
    return (json.loads(caps.stdout.decode()).get("gpu") or {}).get("backend") == "cuda"


def _run(runner_bin, *args, cwd, env=None):
    return subprocess.run([str(runner_bin), *args, "--gpu", "99", "--no-tray", "-t", "2"], cwd=cwd,
                          capture_output=True, text=True, errors="replace", timeout=300,
                          env={**os.environ, **(env or {})})


@pytest.fixture(scope="module")
def recs(runner_bin, tmp_path_factory):
    if not _cuda(runner_bin):
        pytest.skip("needs a CUDA device")
    tmp = tmp_path_factory.mktemp("tcverify")
    out = {}
    for name, extra in (("llama", []), ("deepseek2", ["--arch", "deepseek2", "--mla-wide"])):
        f = tmp / f"{name}.gguf"
        subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", *extra, str(f)], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
        q = f"{name}-q4k.gguf"
        subprocess.run([str(runner_bin), "-m", str(f), "--quantize", str(tmp / q), "--quant", "q4_k"], check=True,
                       cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p = _run(runner_bin, "-m", q, "-p", PROMPT, "-n", "8", "--temp", "0", "--record", f"{name}.json", cwd=tmp)
        assert p.returncode == 0, p.stderr[-1500:]
        out[name] = (tmp, q, tmp / f"{name}.json")
    return out


def _rechain(raw: bytes) -> bytes:
    cpos = raw.rfind(b',"chain"')
    head = raw[:cpos]
    want = hashlib.sha256(head).hexdigest().encode()
    tail = re.sub(rb'"hash":"[0-9a-f]{64}"', b'"hash":"' + want + b'"', raw[cpos:], count=1)
    return head + tail


def _pre_fix(raw: bytes) -> bytes:
    """The record as the writer before tc_types wrote it: the mode's word, no type list."""
    out, n = re.subn(rb"\btc=(taken|off)\b", b"tc=promoted", raw, count=1)
    assert n == 1, raw[:400]
    out, n = re.subn(rb',"tc_types":"[^"]*"', b"", out, count=1)
    assert n == 1, raw[:400]
    return _rechain(out)


def _verify(runner_bin, tmp, model, rec, env=None):
    p = _run(runner_bin, "-m", model, "--verify", str(rec), cwd=tmp, env=env)
    return p.returncode, p.stdout + p.stderr


@pytest.mark.parametrize("name, label, types", [("llama", "taken", "Q5_0"), ("deepseek2", "off", "")])
def test_a_record_names_its_route_and_tensor_core_types(runner_bin, recs, name, label, types):
    tmp, q, rec = recs[name]
    route = json.loads(rec.read_text())["profile"]["route"]
    assert f" tc={label} " in f" {route['backend_detail']} ", route
    assert route["tc_types"] == types, route
    rc, out = _verify(runner_bin, tmp, q, rec)
    assert rc == 0 and "VERIFIED" in out and "not recorded" not in out, out[-1500:]


@pytest.mark.parametrize("name", ["llama", "deepseek2"])
def test_a_pre_fix_record_verifies_and_says_its_tensor_cores_are_not_recorded(runner_bin, recs, name):
    tmp, q, rec = recs[name]
    pre = tmp / f"{name}-pre.json"
    pre.write_bytes(_pre_fix(rec.read_bytes()))
    rc, out = _verify(runner_bin, tmp, q, pre)
    assert rc == 0 and "VERIFIED" in out, out[-1500:]
    assert "tensor-core types not recorded" in out, out[-1500:]


def test_a_pre_fix_record_still_fails_when_another_route_token_differs(runner_bin, recs):
    tmp, q, rec = recs["llama"]
    pre = tmp / "llama-pre2.json"
    pre.write_bytes(_pre_fix(rec.read_bytes()))
    rc, out = _verify(runner_bin, tmp, q, pre, env={"RUNNER_CUDA_Q8X": "1"})
    assert rc == 3 and "did not take the recorded route" in out, out[-1500:]


def test_recorded_tensor_core_types_hold_the_replay(runner_bin, recs):
    tmp, q, rec = recs["llama"]
    alt = tmp / "llama-types.json"
    raw, n = re.subn(rb'"tc_types":"Q5_0"', b'"tc_types":""', rec.read_bytes(), count=1)
    assert n == 1
    alt.write_bytes(_rechain(raw))
    rc, out = _verify(runner_bin, tmp, q, alt)
    assert rc == 3 and "tensor-core types differ" in out, out[-1500:]
    rc, out = _verify(runner_bin, tmp, q, rec, env={"RUNNER_CUDA_TC": "0"})
    assert rc == 3 and "did not take the recorded route" in out, out[-1500:]
