"""Clef (Cloudflare/clef): the "clef" architecture and POST /v1/systemone.

The fixture is scripts/make-test-clef.py: the Ornith-sized qwen35 backbone
under the clef prefix with a random joint schema head, and a byte-level
vocabulary, so a record's token ids decode back to its prompt and the
encoder's rendering can be held against python's own json.dumps here. The
gold comparison with the publisher's joint_schema_model.py needs torch and a
real checkpoint; scripts/clef-gold.py runs it, and docs/clef-admission-2026-10-16.md
has the numbers.
"""
import contextlib
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
RUNNER = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
RECORDS = ROOT / "tests" / "fixtures" / "clef-records.jsonl"
SYSTEM_PROMPT = ("Read the complete state and schema. Decide every field jointly. "
                 "Each answer must be exactly one of that field's allowed options.")
NOUL_TRUE = "The proposition is true or the answer is yes."
NOUL_FALSE = "The proposition is false or the answer is no."


@pytest.fixture(scope="module")
def runner_bin():
    if not RUNNER.exists():
        pytest.skip("runner binary not built")
    return RUNNER


@pytest.fixture(scope="module")
def clef_model(tmp_path_factory):
    m = tmp_path_factory.mktemp("clef") / "test-clef.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-clef.py", str(m)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return m


def records():
    # split on "\n" only: a record holds U+2028, which splitlines() breaks at
    text = RECORDS.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def record(rid):
    return next(r for r in records() if r["id"] == rid)


def systemone(runner_bin, model, reqs, *extra, env=None):
    lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs)
    p = subprocess.run(
        [str(runner_bin), "-m", str(model), "--gpu", "off", "-c", "4096", *extra,
         "--systemone", "-"],
        input=lines.encode("utf-8"), cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=120, env=env)
    out = [json.loads(line) for line in p.stdout.decode("utf-8").split("\n") if line.strip()]
    return p, out


def traced(r):
    r = dict(r)
    r["runner_logits"] = True
    return r


# ------------------------------------------------- encode_record's text, here
def render(v):
    return v if isinstance(v, str) else json.dumps(
        v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(q):
    if q["type"] == "noul":
        c = {"true": NOUL_TRUE, "false": NOUL_FALSE}
        c.update(q.get("criteria") or {})
        return [(k, c[k]) for k in ("true", "false")]
    if q["type"] == "choice":
        return sorted((str(k), v) for k, v in q["criteria"].items())
    return [(str(i), v) for i, v in enumerate(q["criteria"])]


def expected_prompt(r):
    """The text encode_record tokenizes, piece by piece, and the text of
    every span it records (joint_schema_model.py, encode_record)."""
    schema, spans = ["\n\nSCHEMA FIELDS:\n"], []
    for i, (qid, q) in enumerate(r["questions"].items()):
        schema.append(f"\nFIELD {i + 1}\nID: {qid}\nTYPE: {q['type']}\nINSTRUCTION: ")
        ins = q.get("instructions")
        qtext = render(qid if ins is None or ins == "" else ins)
        schema += [qtext, "\nALLOWED OPTIONS:\n"]
        opts = []
        for k, (oid, d) in enumerate(question_options(q)):
            sem = {"option_id": oid}
            if d is not None:
                sem["description"] = d
            opts.append(render(sem))
            schema += [f"OPTION {k + 1}: ", opts[-1], "\n"]
        schema.append("END FIELD\n")
        spans.append((qtext, opts))
    prefix = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    suffix = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
    return prefix + render(r["state"]) + "".join(schema) + suffix, spans


def decode(ids):
    # the fixture vocabulary is <unk> <s> </s> then the 256 byte tokens, and
    # its SentencePiece encoding writes a space as U+2581
    assert all(i >= 3 for i in ids)
    return bytes(i - 3 for i in ids).decode("utf-8").replace("▁", " ")


def test_clef_is_an_admitted_architecture(runner_bin):
    caps = json.loads(subprocess.run([str(runner_bin), "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, check=True).stdout)
    assert "clef" in caps["architectures"]


RENDER_CASES = ["card-invoice", "card-systemone", "noul-criteria", "choice-keys",
                "python-spellings", "instructions-types", "special-text",
                "many-questions", "whitespace-state", "empty-state", "odd-ids",
                "number-state", "null-state", "combining-marks"]


def test_prompt_and_spans_are_encode_records(runner_bin, clef_model):
    p, out = systemone(runner_bin, clef_model, [traced(record(i)) for i in RENDER_CASES])
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    for rid, resp in zip(RENDER_CASES, out):
        text, spans = expected_prompt(record(rid))
        tr = resp["runner_logits"]
        ids = tr["input_ids"]
        assert decode(ids) == text, rid
        assert resp["usage"] == {"input_tokens": len(ids), "output_tokens": 0}
        for (qtext, opts), q in zip(spans, tr["questions"]):
            a, b = q["question_span"]
            assert decode(ids[a:b]) == qtext, (rid, q["id"])
            assert [decode(ids[a:b]) for a, b in q["option_spans"]] == opts, (rid, q["id"])


def test_answers_follow_systemone(runner_bin, clef_model):
    r = record("many-questions")
    p, (resp,) = systemone(runner_bin, clef_model, [traced(r)])
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert resp["model"] == "clef"
    assert list(resp["answers"]) == list(r["questions"])
    for q, tq in zip(r["questions"].items(), resp["runner_logits"]["questions"]):
        qid, spec = q
        ans = resp["answers"][qid]
        assert ans["type"] == spec["type"]
        if spec["type"] == "noul":
            assert set(ans) == {"type", "noul"} and 0 <= ans["noul"] <= 1
        elif spec["type"] == "choice":
            probs = ans["probabilities"]
            # the caller's order, and the first maximum is the choice
            assert list(probs) == list(spec["criteria"])
            best = max(probs, key=probs.get)
            assert ans["choice"] == best and ans["confidence"] == probs[best]
            assert abs(sum(probs.values()) - 1) <= 1e-4 * len(probs)
        else:
            probs = ans["probabilities"]
            assert list(probs) == [str(i) for i in range(len(spec["criteria"]))]
            assert ans["legend"] == {str(i): d for i, d in enumerate(spec["criteria"])}
            exp = sum(i * p for i, p in enumerate(probs.values()))
            assert abs(ans["score"] - exp) <= 1e-3
            assert ans["confidence"] == max(probs.values())
        # every number is python's round(x, 4)
        for v in ([ans.get("noul")] if spec["type"] == "noul" else list(ans["probabilities"].values())):
            assert v == round(v, 4)
    env = resp["envelope"]
    assert env["encoding"] == "joint_schema_model.encode_record"
    assert env["max_length"] == 4096 and env["state_truncated"] is False
    assert len(env["input_ids_sha256"]) == 64


def test_score_legend_keeps_integers_a_double_cannot_hold(runner_bin, clef_model):
    """The legend is the caller's criteria handed back, and the prompt is
    built from the same criteria, so both must keep every digit. 2**53 + 1 is
    the first integer a double cannot hold: it came back as ...992 (the
    2026-10-16 review), nested ones too, while the prompt kept ...993, so the
    legend no longer named what the model had scored. Python's json keeps
    integers exact, which makes it the anchor here."""
    big = 2 ** 53 + 1
    criteria = [big, {"n": -big, "ids": [big + 2, 12345678901234567890]}, "small"]
    r = {"id": "big-integers", "model": "clef", "state": {"order": big},
         "questions": {"size": {"type": "score", "criteria": criteria}}}
    p, (resp,) = systemone(runner_bin, clef_model, [traced(r)])
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert resp["answers"]["size"]["legend"] == {str(i): c for i, c in enumerate(criteria)}
    assert decode(resp["runner_logits"]["input_ids"]) == expected_prompt(r)[0]


def test_trace_only_on_request(runner_bin, clef_model):
    p, (resp,) = systemone(runner_bin, clef_model, [record("card-invoice")])
    assert p.returncode == 0
    assert "runner_logits" not in resp


@pytest.mark.parametrize("req,message", [
    ({"state": "s", "questions": {"q": {"type": "noul"}}}, "model and state are required"),
    ({"model": 3, "state": "s", "questions": {"q": {"type": "noul"}}}, "model and state are required"),
    ({"model": "m", "questions": {"q": {"type": "noul"}}}, "model and state are required"),
    ({"model": "m", "state": "s"}, "at least one question is required"),
    ({"model": "m", "state": "s", "questions": {}}, "at least one question is required"),
    ({"model": "m", "state": "s", "questions": [1]}, "at least one question is required"),
    ({"model": "m", "state": "s", "questions": {"q": "noul"}}, "q: a question must be an object"),
    ({"model": "m", "state": "s", "questions": {"q": {"type": "yes"}}},
     "q: type must be noul, choice, or score"),
    ({"model": "m", "state": "s", "questions": {"q": {"type": "choice"}}},
     "q: criteria must not be empty"),
    ({"model": "m", "state": "s", "questions": {"q": {"type": "score", "criteria": []}}},
     "q: criteria must not be empty"),
    ({"model": "m", "state": "s", "questions": {"q": {"type": "choice", "criteria": ["a"]}}},
     "q: choice criteria must be an object of option ids to descriptions"),
    ({"model": "m", "state": "s", "questions": {"q": {"type": "score", "criteria": {"a": 1}}}},
     "q: score criteria must be an array of level descriptions"),
    ({"model": "m", "state": "s", "questions": {"q": {"type": "noul", "criteria": ["x"]}}},
     "q: noul criteria must be an object with true and/or false"),
    ({"model": "m", "state": "s", "images": ["x"], "questions": {"q": {"type": "noul"}}},
     "images and videos are not supported yet"),
    ({"model": "m", "state": "s", "questions": {"": {"type": "noul"}}},
     ": the instructions encode to no tokens"),
    ({"model": "m", "state": "s", "questions": {"a\"b": {"type": "x"}}},
     "a\"b: type must be noul, choice, or score"),
])
def test_requests_the_reference_refuses(runner_bin, clef_model, req, message):
    p, (resp,) = systemone(runner_bin, clef_model, [req])
    assert p.returncode == 1
    assert resp["error"]["status"] == 400
    assert resp["error"]["message"].startswith(message), resp


def test_schema_longer_than_the_bound_is_refused(runner_bin, clef_model):
    p, (resp,) = systemone(runner_bin, clef_model, [record("many-questions")], "-c", "600")
    assert resp["error"]["status"] == 400
    assert resp["error"]["message"].startswith("schema requires ")
    assert resp["error"]["message"].endswith("tokens before state; maximum is 600")


def test_a_long_state_is_cut_to_the_bound(runner_bin, clef_model):
    r = dict(record("card-systemone"))
    r["state"] = "the queue is long. " * 200
    p, (resp,) = systemone(runner_bin, clef_model, [traced(r)], "-c", "2000")
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert resp["envelope"]["state_truncated"] is True
    assert resp["envelope"]["max_length"] == 2000
    assert resp["usage"]["input_tokens"] == 2000
    # the cut is taken from the state alone; the schema survives whole
    text, _ = expected_prompt(r)
    tail = text[text.index("\n\nSCHEMA FIELDS:\n"):]
    assert decode(resp["runner_logits"]["input_ids"]).endswith(tail)


def scores(resp):
    return [x for q in resp["runner_logits"]["questions"] for x in q["logits"]]


def test_prompt_chunking_does_not_move_the_scores(runner_bin, clef_model):
    reqs = [traced(record("card-invoice")), traced(record("many-questions"))]
    _, wide = systemone(runner_bin, clef_model, reqs)
    _, narrow = systemone(runner_bin, clef_model, reqs, "-b", "16")
    for a, b in zip(wide, narrow):
        assert a["runner_logits"]["input_ids"] == b["runner_logits"]["input_ids"]
        assert max(abs(x - y) for x, y in zip(scores(a), scores(b))) <= 1e-5


def test_pinned_scores(runner_bin, clef_model):
    # This build's head on the fixture, after it matched the publisher's code
    # on a real checkpoint (docs/clef-admission-2026-10-16.md): a regression
    # pin on the head's wiring, not a reference value. Re-pinned for R8.18:
    # the DeltaNet q/k L2 norm takes the publisher's 1e-6, and the fixture's
    # RMS epsilon is 1e-5 (a real checkpoint's is 1e-6, so its scores stay).
    _, (resp,) = systemone(runner_bin, clef_model, [traced(record("card-invoice"))])
    got = scores(resp)
    want = [1.85890114, 1.49842787, 1.94177055, 2.14295936, 2.29865026]
    assert max(abs(x - y) for x, y in zip(got, want)) <= 1e-4, got


def test_gpu_matches_cpu(runner_bin, clef_model):
    caps = json.loads(subprocess.run([str(runner_bin), "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, check=True).stdout)
    if not caps.get("gpu"):
        pytest.skip("no GPU backend: this test asserts a device/host match")
    reqs = [traced(record("card-invoice")), traced(record("odd-ids"))]
    _, cpu = systemone(runner_bin, clef_model, reqs)
    lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs)
    p = subprocess.run([str(runner_bin), "-m", str(clef_model), "-c", "4096",
                        "--systemone", "-"], input=lines.encode(), cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    gpu = [json.loads(line) for line in p.stdout.decode().split("\n") if line.strip()]
    for a, b in zip(cpu, gpu):
        assert max(abs(x - y) for x, y in zip(scores(a), scores(b))) <= 1e-4


def test_a_head_missing_a_tensor_is_refused(runner_bin, tmp_path):
    m = tmp_path / "broken-clef.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-clef.py", str(m)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL,
                   env={**os.environ, "CLEF_TEST_DROP": "decision.scales"})
    p, out = systemone(runner_bin, m, [record("card-invoice")])
    assert p.returncode != 0 and out == []
    assert b"the decision head is missing decision.scales" in p.stderr


# ------------------------------------------------------------------ server
def _free_port():
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _serve(runner_bin, model):
    port = _free_port()
    proc = subprocess.Popen(
        [str(runner_bin), "-m", str(model), "--serve", "--port", str(port),
         "-c", "4096", "--gpu", "off", "--no-tray"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 60
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError("server exited: " + proc.stderr.read().decode(errors="replace"))
            try:
                with urllib.request.urlopen(base + "/health", timeout=1):
                    break
            except (urllib.error.URLError, OSError):
                time.sleep(0.1)
        else:
            raise AssertionError("server never answered /health")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)


def _post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _served_id(base):
    with urllib.request.urlopen(base + "/v1/models", timeout=10) as resp:
        return json.loads(resp.read())["data"][0]["id"]


def test_server_route_answers_what_the_cli_answers(runner_bin, clef_model):
    with _serve(runner_bin, clef_model) as base:
        # the server routes on `model` like every route; the CLI echoes it
        r = traced(record("card-systemone"))
        r["model"] = _served_id(base)
        lines = json.dumps(r, ensure_ascii=False) + "\n"
        cli = subprocess.run([str(runner_bin), "-m", str(clef_model), "--gpu", "off",
                              "-c", "4096", "--systemone", "-"], input=lines.encode(),
                             cwd=ROOT, stdout=subprocess.PIPE, timeout=120).stdout.strip()
        status, body = _post(base, "/v1/systemone", r)
        assert status == 200, body
        assert body.strip() == cli
        # the slot's KV was overwritten from position 0; a second request on
        # it answers the same
        assert _post(base, "/v1/systemone", r)[1].strip() == cli
        with urllib.request.urlopen(base + "/v1/capabilities", timeout=10) as resp:
            caps = json.loads(resp.read())
        assert caps["decision_head"] == {"type": "clef", "endpoint": "/v1/systemone"}


def test_a_model_without_a_head_is_refused(runner_bin, tmp_path):
    m = tmp_path / "ornith.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", str(m)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    with _serve(runner_bin, m) as base:
        r = record("card-invoice")
        r["model"] = _served_id(base)
        status, body = _post(base, "/v1/systemone", r)
        assert status == 400
        assert b"this model has no decision head" in body
        with urllib.request.urlopen(base + "/v1/capabilities", timeout=10) as resp:
            assert json.loads(resp.read())["decision_head"] is None


def test_clef_gold_scores_any_servers_decisions(tmp_path):
    """clef-gold.py compare --decisions reads a /v1/systemone reply (llama.cpp's
    shape as much as Runner's) back into the reference's option order: a
    choice by criteria key, a noul as true/false, a score by index. It counts
    the decisions that agree, the largest |dprob|, each flip with the
    reference's margin, and refuses an answer it cannot read."""
    import math
    def logits(ps):
        return [math.log(p) for p in ps]
    ref = [{"reference": "joint_schema_model.py", "code_sha256": "x", "dtype": "float32", "device": "cpu"},
           {"id": "r1", "questions": [
               {"id": "c", "option_ids": ["a", "b", "c"], "logits": logits([0.6, 0.3, 0.1])},
               {"id": "n", "option_ids": ["true", "false"], "logits": logits([0.8, 0.2])}]},
           {"id": "r2", "questions": [
               {"id": "s", "option_ids": ["0", "1", "2"], "logits": logits([0.2, 0.45, 0.35])}]}]
    rep = [{"answers": {"c": {"type": "choice", "choice": "a", "probabilities": {"a": 0.55, "b": 0.35, "c": 0.10}},
                        "n": {"type": "noul", "noul": 0.75}}},
           {"answers": {"s": {"type": "score", "score": 1.2, "probabilities": {"0": 0.2, "1": 0.35, "2": 0.45}}}}]
    rp, np_, out = tmp_path / "ref.jsonl", tmp_path / "rep.jsonl", tmp_path / "out.json"
    rp.write_text("".join(json.dumps(r) + "\n" for r in ref))
    np_.write_text("".join(json.dumps(r) + "\n" for r in rep))
    p = subprocess.run([sys.executable, str(ROOT / "scripts/clef-gold.py"), "compare", "--decisions",
                        "--ref", str(rp), "--runner", str(np_), "--json", str(out)],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    got = json.loads(out.read_text())
    assert got["questions"] == 3 and got["decisions_agree"] == 2
    assert abs(got["max_abs_dprob"] - 0.10) < 1e-6
    flips = [d for d in got["questions_detail"] if not d["agree"]]
    assert [(d["record"], d["question"]) for d in flips] == [("r2", "s")]
    assert abs(flips[0]["reference_margin"] - 0.10) < 1e-6
    assert "flip r2/s" in p.stdout and "no id, span or logit check" in p.stdout
    # an answer without its probabilities is reported, not guessed
    rep[1]["answers"]["s"].pop("probabilities")
    np_.write_text("".join(json.dumps(r) + "\n" for r in rep))
    p = subprocess.run([sys.executable, str(ROOT / "scripts/clef-gold.py"), "compare", "--decisions",
                        "--ref", str(rp), "--runner", str(np_)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 1 and "unreadable answer: r2/s" in p.stdout
