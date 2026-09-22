"""Public-interface gates for scripts/decide-calibrate.py, the calibration
instrument for POST /v1/decide (R13.10, the `decide` contract).

No network: every test drives a tiny in-process http.server standing in for
the real /v1/decide endpoint, with a caller-controlled readout (`_Server`
below), so the test suite runs in well under a second and never depends on
a built runner binary or a real model.
"""

import hashlib
import importlib.util
import json
import math
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "decide-calibrate.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dc = _load("decide_calibrate", SCRIPT)


# ------------------------------------------------------------- fixtures ---

class _Server:
    """A minimal /v1/decide server backed by a caller-supplied `scorer`.

    `scorer(options)` returns unnormalized weights aligned to `options` in
    the order the REQUEST gave them; this handler renormalizes them into a
    softmax-shaped `probs` vector so the same scorer can be either
    order-invariant (a function of each option string alone) or
    position-dependent (a function of index) depending on what the test
    wants to check. `received` keeps every request body, verbatim, so a
    test can assert on exactly what was sent over the wire.
    """

    def __init__(self, scorer, served=("test",), not_found=False):
        self.served = list(served)
        self.not_found = not_found
        self.scorer = scorer
        self.received = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                # the instrument checks GET /v1/models before scoring
                if self.path.rstrip("/") == "/v1/models":
                    out = json.dumps({"object": "list",
                                      "data": [{"id": m} for m in outer.served]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                else:
                    self.send_response(404); self.end_headers()

            def do_POST(self):
                if outer.not_found:
                    out = json.dumps({"error": {"message": "unknown model (see /v1/models)",
                                                "type": "invalid_request_error",
                                                "param": "model", "code": "model_not_found"}}).encode()
                    outer.received.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                outer.received.append(body)
                decisions = []
                for q in body["questions"]:
                    opts = q["options"]
                    weights = outer.scorer(opts)
                    total = sum(weights)
                    probs = [w / total for w in weights]
                    logprobs = [math.log(max(p, 1e-12)) for p in probs]
                    argmax = max(range(len(probs)), key=lambda i: probs[i])
                    decisions.append({
                        "id": q["id"], "options": opts, "logprobs": logprobs,
                        "probs": probs, "argmax": argmax,
                        "n_tokens": [1] * len(opts),
                    })
                resp = {
                    "id": "decide-test", "model": body.get("model", "test"),
                    "created": 0, "decisions": decisions,
                    "usage": {"prompt_tokens": 1, "completion_tokens": 0},
                    "envelope": {},
                }
                data = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def endpoint(self):
        _, port = self.httpd.server_address
        return f"http://127.0.0.1:{port}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def make_server():
    servers = []

    def _make(scorer, **kw):
        s = _Server(scorer, **kw)
        servers.append(s)
        return s

    yield _make
    for s in servers:
        s.close()


def hash_scorer(options):
    """Weight depends only on the option STRING, never on its position, so
    reordering a request's options cannot move mass between strings."""
    return [(int(hashlib.sha256(o.encode()).hexdigest(), 16) % 997) + 1
            for o in options]


def position_scorer(options):
    """Weight depends only on POSITION (first option always heaviest), so
    reordering a request's options moves mass between strings."""
    n = len(options)
    return [float(n - i) for i in range(n)]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# An 8-row dataset covering: a reordered-option permutation_group (g1), two
# different permutation_groups sharing one `state` (g2/g3, exercising
# --batch-state), a distribution-typed answer (row 3), and a second
# reordered-option group (g6) whose two variants disagree on the label
# index because the true answer ("m") sits at a different index in each
# variant's option list -- the same shape the real ledger data uses.
# A permutation variant shares its STATE with the other variants of its
# group (reordered options, rephrased question); the instrument keys
# invariance by (permutation_group, state) because the question set files a chosen
# and a rejected edit, two states with opposite labels, under one group id.
ROWS = [
    {"state": "sA", "question": "is it fresh (v0)", "options": ["yes", "no"],
     "answer": 0, "source": "synthetic", "permutation_group": "g1", "variant": 0},
    {"state": "sA", "question": "is it fresh (v1)", "options": ["no", "yes"],
     "answer": 1, "source": "synthetic", "permutation_group": "g1", "variant": 1},
    {"state": "sB", "question": "which animal (a)", "options": ["cat", "dog", "bird"],
     "answer": 0, "source": "nli", "permutation_group": "g2", "variant": 0},
    {"state": "sB", "question": "which animal (b)", "options": ["cat", "dog", "bird"],
     "answer": [0.2, 0.3, 0.5], "source": "nli", "permutation_group": "g3", "variant": 0},
    {"state": "sC", "question": "q3", "options": ["x", "y"],
     "answer": 1, "source": "ledger", "permutation_group": "g4", "variant": 0},
    {"state": "sD", "question": "q4", "options": ["p", "q", "r"],
     "answer": 0, "source": "ledger", "permutation_group": "g5", "variant": 0},
    {"state": "sE", "question": "letter (v0)", "options": ["m", "n"],
     "answer": 0, "source": "synthetic", "permutation_group": "g6", "variant": 0},
    {"state": "sE", "question": "letter (v1)", "options": ["n", "m"],
     "answer": 1, "source": "synthetic", "permutation_group": "g6", "variant": 1},
]
ALL_GROUPS = ["g1", "g2", "g3", "g4", "g5", "g6"]


# -------------------------------------------------- (a) split determinism ---

def test_split_is_by_group_and_deterministic_for_a_seed():
    a = dc.assign_holdout_groups(ALL_GROUPS, seed=42, holdout_frac=0.5)
    b = dc.assign_holdout_groups(ALL_GROUPS, seed=42, holdout_frac=0.5)
    assert a == b
    # Order of the input list must not matter: the same set of groups, in a
    # different order, still yields the same split (groups are sorted before
    # the seeded shuffle).
    shuffled = list(reversed(ALL_GROUPS))
    c = dc.assign_holdout_groups(shuffled, seed=42, holdout_frac=0.5)
    assert a == c
    assert 0 < len(a) < len(ALL_GROUPS)
    # A different seed is allowed to (and here does) choose a different split.
    d = dc.assign_holdout_groups(ALL_GROUPS, seed=7, holdout_frac=0.5)
    assert d != a


def test_no_permutation_group_straddles_both_splits(tmp_path, make_server):
    server = make_server(hash_scorer)
    qfile = tmp_path / "questions.jsonl"
    write_jsonl(qfile, ROWS)
    out = tmp_path / "result.json"
    holdout_file = tmp_path / "holdout.jsonl"

    code = dc.main([
        "--questions", str(qfile), "--endpoint", server.endpoint,
        "--model-name", "test", "--seed", "42", "--holdout-frac", "0.5",
        "--out", str(out), "--emit-holdout", str(holdout_file),
    ])
    assert code == 0

    holdout_groups = dc.assign_holdout_groups(ALL_GROUPS, seed=42, holdout_frac=0.5)
    emitted = [json.loads(line) for line in holdout_file.read_text().splitlines()]
    emitted_groups = {r["permutation_group"] for r in emitted}
    train_groups = set(ALL_GROUPS) - holdout_groups

    # Every row emitted as held-out belongs to a group the split chose as
    # held-out, and no held-out group's rows leak into "the rest" -- i.e. no
    # group appears on both sides.
    assert emitted_groups <= holdout_groups
    assert emitted_groups.isdisjoint(train_groups)
    result = json.loads(out.read_text())
    assert (result["provenance"]["n_train_rows"]
            + result["provenance"]["n_holdout_rows"]) == len(ROWS)


# --------------------------------------------- (b) hand-computed metrics ---

def test_metrics_match_hand_computation_on_six_questions():
    # Six binary questions, each row a dict of exactly what compute_metrics
    # reads: `probs` (aligned to a 2-option vector) and `label_dist` (the
    # one-hot label). No HTTP involved -- this checks the arithmetic alone.
    #
    #   row  probs        label(idx)  correct?  confidence  bin (10 bins)
    #   1    [0.90,0.10]   0            yes       0.90        9  [0.90,1.00)
    #   2    [0.80,0.20]   0            yes       0.80        8  [0.80,0.90)
    #   3    [0.70,0.30]   1            no        0.70        7  [0.70,0.80)
    #   4    [0.60,0.40]   0            yes       0.60        6  [0.60,0.70)
    #   5    [0.55,0.45]   1            no        0.55        5  [0.50,0.60)
    #   6    [0.95,0.05]   0            yes       0.95        9  [0.90,1.00)
    #
    # accuracy = 4/6 (rows 3 and 5 are wrong) = 0.666666...
    #
    # Brier = mean_i sum_j (p_ij - y_ij)^2:
    #   row1 (0.9-1)^2+(0.1-0)^2 = 0.01+0.01   = 0.02
    #   row2 (0.8-1)^2+(0.2-0)^2 = 0.04+0.04   = 0.08
    #   row3 (0.7-0)^2+(0.3-1)^2 = 0.49+0.49   = 0.98
    #   row4 (0.6-1)^2+(0.4-0)^2 = 0.16+0.16   = 0.32
    #   row5 (0.55-0)^2+(0.45-1)^2 = 0.3025*2  = 0.605
    #   row6 (0.95-1)^2+(0.05-0)^2 = 0.0025*2  = 0.005
    #   sum = 2.01, mean = 2.01/6 = 0.335
    #
    # log loss = mean_i -log(p at the true index):
    #   -ln(0.90) + -ln(0.80) + -ln(0.30) + -ln(0.60) + -ln(0.45) + -ln(0.95)
    #   summed and divided by 6 (computed exactly below with math.log so the
    #   literal isn't hand-rounded).
    #
    # ECE (10 equal-width bins, weight n_b/N times |acc_b - mean_conf_b|):
    #   bin9 [0.90,1.00): rows 1,6, both correct -> acc=1.0, mean_conf=0.925,
    #        contributes 2*|1.0-0.925| = 0.15
    #   bin8 [0.80,0.90): row2, correct -> acc=1.0, conf=0.80,
    #        contributes 1*|1.0-0.80| = 0.20
    #   bin7 [0.70,0.80): row3, WRONG -> acc=0.0, conf=0.70,
    #        contributes 1*|0.0-0.70| = 0.70
    #   bin6 [0.60,0.70): row4, correct -> acc=1.0, conf=0.60,
    #        contributes 1*|1.0-0.60| = 0.40
    #   bin5 [0.50,0.60): row5, WRONG -> acc=0.0, conf=0.55,
    #        contributes 1*|0.0-0.55| = 0.55
    #   ECE = (0.15+0.20+0.70+0.40+0.55) / 6 = 2.00/6 = 0.333333...
    rows = [
        {"probs": [0.90, 0.10], "label_dist": [1.0, 0.0]},
        {"probs": [0.80, 0.20], "label_dist": [1.0, 0.0]},
        {"probs": [0.70, 0.30], "label_dist": [0.0, 1.0]},
        {"probs": [0.60, 0.40], "label_dist": [1.0, 0.0]},
        {"probs": [0.55, 0.45], "label_dist": [0.0, 1.0]},
        {"probs": [0.95, 0.05], "label_dist": [1.0, 0.0]},
    ]
    expected_logloss = (
        -math.log(0.90) - math.log(0.80) - math.log(0.30)
        - math.log(0.60) - math.log(0.45) - math.log(0.95)
    ) / 6

    m = dc.compute_metrics(rows, bins=10)

    assert m["n"] == 6
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["brier"] == pytest.approx(2.01 / 6)
    assert m["log_loss"] == pytest.approx(expected_logloss)
    assert m["ece"] == pytest.approx(1 / 3)

    # Reliability curve: bin 9 holds both row1 and row6.
    bin9 = m["reliability"][9]
    assert bin9["n"] == 2
    assert bin9["confidence"] == pytest.approx(0.925)
    assert bin9["accuracy"] == pytest.approx(1.0)
    bin7 = m["reliability"][7]
    assert bin7["n"] == 1
    assert bin7["accuracy"] == pytest.approx(0.0)


# ------------------------------------------ (c) permutation invariance ---

def test_invariance_is_zero_for_an_order_invariant_readout(tmp_path, make_server):
    server = make_server(hash_scorer)
    qfile = tmp_path / "questions.jsonl"
    write_jsonl(qfile, ROWS)
    out = tmp_path / "result.json"

    code = dc.main([
        "--questions", str(qfile), "--endpoint", server.endpoint,
        "--model-name", "test", "--seed", "42", "--holdout-frac", "0.5",
        "--out", str(out),
    ])
    assert code == 0
    result = json.loads(out.read_text())

    for split in ("train", "holdout"):
        inv = result[split]["overall"]["invariance"]
        if inv["groups_matched"]:
            assert inv["mean_max_tvd"] == pytest.approx(0.0, abs=1e-9)
            assert inv["max_max_tvd"] == pytest.approx(0.0, abs=1e-9)


def test_invariance_is_positive_for_a_position_dependent_readout(tmp_path, make_server):
    server = make_server(position_scorer)
    qfile = tmp_path / "questions.jsonl"
    # Just the reordered pair (g1): options ["yes","no"] then ["no","yes"].
    # position_scorer gives weights [2,1] in request order every time, so
    # "yes" carries 2/3 in variant 0 but only 1/3 in variant 1 -- the
    # opposite of order-invariance.
    write_jsonl(qfile, ROWS[:2])
    out = tmp_path / "result.json"

    code = dc.main([
        "--questions", str(qfile), "--endpoint", server.endpoint,
        "--model-name", "test", "--seed", "1", "--holdout-frac", "0.0",
        "--out", str(out),
    ])
    assert code == 0
    result = json.loads(out.read_text())
    inv = result["train"]["overall"]["invariance"]
    assert inv["groups_matched"] == 1
    assert inv["mean_max_tvd"] == pytest.approx(1 / 3)
    assert inv["argmax_change_frac"] == pytest.approx(1.0)


# ------------------------------------- (d) a distribution-typed answer ---

def test_label_to_distribution_handles_index_and_distribution():
    assert dc.label_to_distribution(0, 3) == [1.0, 0.0, 0.0]
    assert dc.label_to_distribution(2, 3) == [0.0, 0.0, 1.0]
    assert dc.label_to_distribution([0.2, 0.3, 0.5], 3) == [0.2, 0.3, 0.5]
    with pytest.raises(ValueError):
        dc.label_to_distribution([0.5, 0.5], 3)  # wrong length
    with pytest.raises(ValueError):
        dc.label_to_distribution(5, 3)  # out of range


def test_distribution_answer_scores_without_error(tmp_path, make_server):
    server = make_server(hash_scorer)
    qfile = tmp_path / "questions.jsonl"
    write_jsonl(qfile, ROWS)
    out = tmp_path / "result.json"

    code = dc.main([
        "--questions", str(qfile), "--endpoint", server.endpoint,
        "--model-name", "test", "--seed", "42", "--holdout-frac", "0.5",
        "--out", str(out),
    ])
    assert code == 0
    assert json.loads(out.read_text())["provenance"]["n_failed"] == 0


# ------------------------------------------------ (e) emitted hold-out ---

def test_emitted_holdout_rows_are_unchanged_and_exactly_the_holdout_set(tmp_path, make_server):
    server = make_server(hash_scorer)
    qfile = tmp_path / "questions.jsonl"
    write_jsonl(qfile, ROWS)
    holdout_file = tmp_path / "holdout.jsonl"

    code = dc.main([
        "--questions", str(qfile), "--endpoint", server.endpoint,
        "--model-name", "test", "--seed", "42", "--holdout-frac", "0.5",
        "--emit-holdout", str(holdout_file),
    ])
    assert code == 0

    holdout_groups = dc.assign_holdout_groups(ALL_GROUPS, seed=42, holdout_frac=0.5)
    expected = [r for r in ROWS if r["permutation_group"] in holdout_groups]
    emitted = [json.loads(line) for line in holdout_file.read_text().splitlines()]

    assert emitted == expected  # same rows, same order, every field unchanged


# --------------------------------------------- (f) verbatim request bodies ---

def test_options_are_sent_verbatim_and_in_order(tmp_path, make_server):
    server = make_server(hash_scorer)
    qfile = tmp_path / "questions.jsonl"
    write_jsonl(qfile, ROWS)

    code = dc.main([
        "--questions", str(qfile), "--endpoint", server.endpoint,
        "--model-name", "test", "--seed", "42", "--holdout-frac", "0.5",
    ])
    assert code == 0
    assert server.received, "the fake server received no requests"

    sent_by_id = {}
    for body in server.received:
        for q in body["questions"]:
            sent_by_id[q["id"]] = q["options"]

    for line_no, row in enumerate(ROWS):
        assert sent_by_id[f"q{line_no}"] == row["options"], (
            "options must reach the server verbatim and in the given order, "
            "never relabeled")

    # Rows 2 and 3 (indices 2 and 3) share `state` "sB" -- with --batch-state
    # on by default they must arrive in ONE request carrying both questions.
    shared_state_bodies = [b for b in server.received if b["state"] == "sB"]
    assert len(shared_state_bodies) == 1
    assert {q["id"] for q in shared_state_bodies[0]["questions"]} == {"q2", "q3"}


def test_batch_state_off_sends_one_request_per_question(tmp_path, make_server):
    server = make_server(hash_scorer)
    qfile = tmp_path / "questions.jsonl"
    write_jsonl(qfile, ROWS)

    code = dc.main([
        "--questions", str(qfile), "--endpoint", server.endpoint,
        "--model-name", "test", "--seed", "42", "--holdout-frac", "0.5",
        "--no-batch-state",
    ])
    assert code == 0
    assert len(server.received) == len(ROWS)
    assert all(len(b["questions"]) == 1 for b in server.received)


def test_split_group_keeps_a_pair_together():
    # two permutation groups that share a split_group (a DPO pair's chosen
    # and rejected halves) land on the same side, whatever the seed
    rows = [
        {"permutation_group": "gA", "split_group": "pair1"},
        {"permutation_group": "gB", "split_group": "pair1"},
        {"permutation_group": "gC", "split_group": "gC"},
        {"permutation_group": "gD"},
    ]
    keys = [dc.split_key(r) for r in rows]
    assert keys == ["pair1", "pair1", "gC", "gD"]
    for seed in range(20):
        held = dc.assign_holdout_groups(keys, seed=seed, holdout_frac=0.5)
        assert (("pair1" in held) == ("pair1" in held))  # one key, one side
        sides = {dc.split_key(r) in held for r in rows[:2]}
        assert len(sides) == 1


def test_distribution_label_matching_prediction_is_calibrated():
    # a prediction equal to a distribution-valued target has zero calibration
    # error: the hit is the label's mass on the predicted option, not a
    # 0/1 against the label's argmax
    rows = [{"probs": [0.8, 0.2], "label_dist": [0.8, 0.2], "options": ["a", "b"]}] * 10
    m = dc.compute_metrics(rows, bins=10)
    assert m["ece"] == pytest.approx(0.0, abs=1e-12)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["brier"] == pytest.approx(0.0, abs=1e-12)
    # and a one-hot label still counts a plain hit
    rows = [{"probs": [0.8, 0.2], "label_dist": [1.0, 0.0], "options": ["a", "b"]}] * 10
    m = dc.compute_metrics(rows, bins=10)
    assert m["ece"] == pytest.approx(0.2)
    assert m["accuracy"] == pytest.approx(1.0)


def test_unknown_model_name_is_refused_before_any_request(tmp_path, make_server):
    # a wrong --model-name is a one-line error naming the served ids, not a
    # full-length run of 404s with an empty report (Blackwell, 2026-09-22)
    server = make_server(hash_scorer)
    qfile = tmp_path / "questions.jsonl"
    write_jsonl(qfile, ROWS)
    code = dc.main(["--questions", str(qfile), "--endpoint", server.endpoint,
                    "--model-name", "not-the-served-id", "--seed", "1"])
    assert code == 2
    assert server.received == []


def test_model_not_found_from_decide_is_not_retried(tmp_path, make_server, capsys):
    # the server lists the id but 404s the decide (a swap-set server between
    # loads, say): fatal per group, no retry budget spent, non-zero exit
    server = make_server(hash_scorer, not_found=True)
    qfile = tmp_path / "questions.jsonl"
    write_jsonl(qfile, ROWS)
    out = tmp_path / "r.json"
    code = dc.main(["--questions", str(qfile), "--endpoint", server.endpoint,
                    "--model-name", "test", "--seed", "1", "--out", str(out),
                    "--report", str(tmp_path / "r.md")])
    assert code == 1
    groups = len({(r["state"], r.get("split_group") or r["permutation_group"]) for r in ROWS})
    assert len(server.received) <= groups + 2   # each group sent once, never retried
    assert "NO RESULT" in (tmp_path / "r.md").read_text()
