"""Tulu 3's chat template on served files (R4.50.11): detected by the file's EOS, the file's own EOS written.

allenai/Llama-3.1-Tulu-3-8B's template (revision 66694379; 70B, 405B and allenai/OLMo-2-0325-32B-Instruct ship the
same bytes) writes OLMo 2 Instruct's turns with no bos_token, and the eos_token after an assistant turn. The eos is the
file's, so runner has one family per vocabulary: tulu3 writes Tulu 3's <|end_of_text|>, tulu3-olmo2 OLMo 2's
<|endoftext|>. scripts/template-conformance.py holds both renders against the publishers' templates, byte for byte
and token for token, and tests/test_template.c's test_tulu3_shape pins their branches offline. This file holds, on
tiny files carrying the publisher's template:
- detection: Tulu 3's EOS reads tulu3, OLMo 2's reads tulu3-olmo2, and any other EOS (the fixture's </s>) keeps the
  former zephyr reading;
- the served prompt (its ids, from a --record receipt): the assistant turn closed by the file's own EOS as one control
  token, then a newline, and no </s> anywhere; the BOS once, the tokenizer's (the template writes none);
- the OpenAI tool protocol refused by name with a 400 on chat, /v1/responses and /v1/messages, and --info tools
  naming no tool protocol.
"""
import glob
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

# make-test-olmo2.py's vocabulary: <unk> <s> </s>, the 256 byte tokens (0x0A at 13, "A" at 68), then the specials
NEWLINE, LETTER_A, SLASH_S = 13, 68, 2
TULU3_BOS, TULU3_EOS = 259, 260   # --tulu3-specials: <|begin_of_text|>, <|end_of_text|>
OLMO_BOS_EOS = 259                # --olmo-specials: <|endoftext|>


def _make(tmp_path_factory, name, *opts):
    path = tmp_path_factory.mktemp("tulu3") / name
    subprocess.run([sys.executable, ROOT / "scripts" / "make-test-olmo2.py", str(path), "--tulu3-template", *opts],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


@pytest.fixture(scope="module")
def tulu3_file(tmp_path_factory):
    return _make(tmp_path_factory, "tulu3.gguf", "--tulu3-specials")


@pytest.fixture(scope="module")
def olmo_file(tmp_path_factory):
    return _make(tmp_path_factory, "tulu3-olmo2.gguf", "--olmo-specials")


@pytest.fixture(scope="module")
def other_file(tmp_path_factory):
    return _make(tmp_path_factory, "tulu3-other-eos.gguf")


def _info(runner_bin, model):
    p = subprocess.run([str(runner_bin), "-m", str(model), "--info", "tools", "--gpu", "off"], cwd=ROOT,
                       capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(p.stdout)


def _post(srv, path, body):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _message(doc):
    err = doc.get("error")
    return err.get("message", "") if isinstance(err, dict) else str(err)


HISTORY = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "A"}, {"role": "user", "content": "Q"}]


def _served_ids(runner_bin, model, rec):
    rec.mkdir()
    with RunnerServer(runner_bin, model, ctx=512, extra_args=["--gpu", "off", "-t", "2", "--record", str(rec)]) as srv:
        status, doc = _post(srv, "/v1/chat/completions", {"messages": HISTORY, "max_tokens": 1, "temperature": 0})
    assert status == 200, doc
    (receipt,) = glob.glob(str(rec / "*.json"))
    return json.loads(Path(receipt).read_text())["prompt"]["tokens"]


def test_detection_picks_the_family_by_the_files_eos(runner_bin, tulu3_file, olmo_file, other_file):
    got = {name: _info(runner_bin, f)["template"]
           for name, f in (("tulu3", tulu3_file), ("olmo", olmo_file), ("other", other_file))}
    assert got == {"tulu3": "tulu3", "olmo": "tulu3-olmo2", "other": "zephyr"}, got


def test_tulu3_closes_the_assistant_turn_with_its_own_eos(runner_bin, tulu3_file, tmp_path):
    ids = _served_ids(runner_bin, tulu3_file, tmp_path / "rec")
    assert ids[0] == TULU3_BOS and ids.count(TULU3_BOS) == 1, ids[:4]   # the tokenizer's BOS, once
    assert ids.count(TULU3_EOS) == 1 and SLASH_S not in ids, ids
    at = ids.index(TULU3_EOS)
    assert (ids[at - 1], ids[at + 1]) == (LETTER_A, NEWLINE), ids[at - 2:at + 3]


def test_tulu3_olmo2_closes_it_with_olmo2s(runner_bin, olmo_file, tmp_path):
    ids = _served_ids(runner_bin, olmo_file, tmp_path / "rec")
    # the fixture sets add_bos_token, so the tokenizer's BOS opens the prompt; the template writes none, and its
    # one other <|endoftext|> is the assistant turn's eos
    assert ids[0] == OLMO_BOS_EOS and ids[1] != OLMO_BOS_EOS and ids.count(OLMO_BOS_EOS) == 2, ids
    assert SLASH_S not in ids, ids
    at = ids.index(OLMO_BOS_EOS, 1)
    assert (ids[at - 1], ids[at + 1]) == (LETTER_A, NEWLINE), ids[at - 2:at + 3]


TOOLS = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]


@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS}),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"},
                                           {"role": "tool", "content": "20", "tool_call_id": "c1"}]}),
    ("/v1/responses", {"input": "hi", "tools": [{"type": "function", "name": "weather",
                                                 "parameters": {"type": "object"}}]}),
    ("/v1/messages", {"messages": [{"role": "user", "content": "hi"}],
                      "tools": [{"name": "weather", "input_schema": {"type": "object"}}]}),
])
@pytest.mark.parametrize("which", ["tulu3", "tulu3-olmo2"])
def test_the_tool_protocol_is_refused_by_name(runner_bin, tulu3_file, olmo_file, which, path, body):
    model = tulu3_file if which == "tulu3" else olmo_file
    with RunnerServer(runner_bin, model, ctx=512, extra_args=["--gpu", "off", "-t", "2"]) as srv:
        status, doc = _post(srv, path, {**body, "max_tokens": 1})
    msg = _message(doc)
    assert status == 400 and f"the {which} chat template has no tool protocol" in msg, (path, doc)
    assert msg.rstrip().endswith("as nothing"), msg


def test_info_tools_names_no_tool_protocol(runner_bin, tulu3_file, olmo_file):
    for f in (tulu3_file, olmo_file):
        info = _info(runner_bin, f)
        assert (info["tool_family"], info["native_tool_protocol"]) == ("none", False), info
