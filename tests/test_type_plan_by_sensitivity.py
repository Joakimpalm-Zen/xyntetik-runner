"""scripts/type-plan-by-sensitivity.py (R4.7.3): a --quant-error report in, a
--type-plan out. Groups are demoted cheapest first (added error per saved
byte), to a count or a byte target; mixed weighted and unweighted errors,
a missing type, and an unreachable target are refused."""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "type-plan-by-sensitivity.py"


def report(tmp_path, blocks, weighted=True, missing=None):
    """Two expert tensors per block; block b adds error cost[b] per byte."""
    lines = []
    for b, cost in blocks.items():
        for part in ("ffn_up_exps", "ffn_down_exps"):
            t = f"blk.{b}.{part}.weight"
            types = {"q4_0": {"bytes": 1000, "error": 1.0},
                     "q3_k": {"bytes": 800, "error": 1.0 + 200 * cost}}
            if missing == t:
                types["q3_k"] = None
            lines.append({"tensor": t, "source_type": "q8_0", "rows": 8, "cols": 256,
                          "weighted": weighted if b != 99 else not weighted, "types": types})
    lines.append({"tensor": "blk.0.attn_q.weight", "source_type": "q8_0", "rows": 8,
                  "cols": 256, "weighted": weighted,
                  "types": {"q4_0": {"bytes": 1000, "error": 0.1},
                            "q3_k": {"bytes": 800, "error": 0.2}}})
    p = tmp_path / "report.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return p


def plan(path, *args):
    p = subprocess.run([sys.executable, str(SCRIPT), str(path), "--high", "q4_0",
                        "--low", "q3_k", *args], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.returncode == 0 else None), p.stderr


def test_the_cheapest_blocks_are_demoted_first(tmp_path):
    rep = report(tmp_path, {0: 5.0, 1: 0.5, 2: 3.0, 3: 0.1})
    rc, pl, err = plan(rep, "--match", "_exps", "--groups", "2")
    assert rc == 0, err
    demoted = [r["match"] for r in pl["rules"] if r["type"] == "q3_k"]
    assert demoted == ["blk.3.ffn_down_exps.weight", "blk.3.ffn_up_exps.weight",
                       "blk.1.ffn_down_exps.weight", "blk.1.ffn_up_exps.weight"]
    assert pl["rules"][-1] == {"match": "_exps", "type": "q4_0"}
    assert pl["default"] == "keep"
    assert "blk.3, blk.1" in err
    # a byte target: each block saves 400 bytes, so 1,000 needs three blocks
    rc, pl, _ = plan(rep, "--match", "_exps", "--save-bytes", "1000")
    assert rc == 0
    assert len([r for r in pl["rules"] if r["type"] == "q3_k"]) == 6
    # tensor by tensor
    rc, pl, _ = plan(rep, "--match", "_exps", "--group", "tensor", "--groups", "1")
    assert rc == 0 and pl["rules"][0]["match"].startswith("blk.3.")


def test_without_match_every_costed_tensor_is_a_candidate(tmp_path):
    rep = report(tmp_path, {0: 5.0, 1: 0.5})
    rc, pl, _ = plan(rep, "--groups", "1")
    assert rc == 0
    assert pl["default"] == "q4_0"
    # blk.0's attention tensor joins blk.0's group; blk.1 is still cheaper
    assert all(r["match"].startswith("blk.1.") for r in pl["rules"])


def test_refusals(tmp_path):
    rep = report(tmp_path, {0: 1.0, 99: 1.0})
    rc, _, err = plan(rep, "--match", "_exps", "--groups", "1")
    assert rc != 0 and "imatrix-weighted and some are not" in err
    assert plan(rep, "--match", "_exps", "--groups", "1", "--allow-mixed")[0] == 0
    rep = report(tmp_path, {0: 1.0}, missing="blk.0.ffn_up_exps.weight")
    rc, _, err = plan(rep, "--match", "_exps", "--groups", "1")
    assert rc != 0 and "has no q3_k entry" in err
    rep = report(tmp_path, {0: 1.0, 1: 2.0})
    rc, _, err = plan(rep, "--match", "_exps", "--groups", "3")
    assert rc != 0 and "only 2 groups" in err
    rc, _, err = plan(rep, "--match", "nothing", "--groups", "1")
    assert rc != 0 and "matches 'nothing'" in err


def ladder_report(tmp_path):
    """MD-001's shape (R4.34.1): an expert bank that is large and cheap to take
    one rung down but dear to take two, and an output tensor that is small and
    cheap to take all the way. With two rungs the cheap-per-byte output goes to
    the lowest type and the bank stays at the top; with a middle rung the bank
    takes the cheap first step, and nothing reaches the lowest type."""
    def row(t, b8, e8, b6, e6, b4, e4):
        return {"tensor": t, "source_type": "bf16", "rows": 8, "cols": 256, "weighted": False,
                "types": {"q8_0": {"bytes": b8, "error": e8}, "q6_k": {"bytes": b6, "error": e6},
                          "q4_k": {"bytes": b4, "error": e4}}}
    lines = [row("blk.0.ffn_up_exps.weight", 1000, 0.0, 750, 10.0, 500, 1000.0),
             row("output.weight", 200, 0.0, 150, 5.0, 100, 20.0)]
    p = tmp_path / "ladder.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return p


def run_plan(path, *args):
    p = subprocess.run([sys.executable, str(SCRIPT), str(path), *args], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.returncode == 0 else None), p.stderr


def test_a_middle_rung_takes_the_large_tensor_down_one_step_first(tmp_path):
    rep = ladder_report(tmp_path)
    # two rungs at a 100-byte target: the output (0.2 error a byte) before the bank (2.0)
    rc, two, err = run_plan(rep, "--high", "q8_0", "--low", "q4_k", "--save-bytes", "100")
    assert rc == 0, err
    assert two["rules"] == [{"match": "output.weight", "type": "q4_k"}] and two["default"] == "q8_0"
    # the ladder at the same target: the bank's first step (0.04 a byte) is the cheapest
    rc, lad, err = run_plan(rep, "--ladder", "q8_0,q6_k,q4_k", "--save-bytes", "100")
    assert rc == 0, err
    assert lad["rules"] == [{"match": "blk.0.ffn_up_exps.weight", "type": "q6_k"}]
    assert not any(r["type"] == "q4_k" for r in lad["rules"])
    assert "ladder q8_0,q6_k,q4_k: demoted 1 of 2 groups (q6_k 1, q4_k 0)" in err
    # --mid is the same ladder
    rc, mid, _ = run_plan(rep, "--high", "q8_0", "--mid", "q6_k", "--low", "q4_k", "--save-bytes", "100")
    assert rc == 0 and mid == lad
    # a larger target walks further, one rung at a time: the output's two steps
    # (0.1, then 0.3 a byte) come before the bank's second (3.96)
    rc, far, _ = run_plan(rep, "--ladder", "q8_0,q6_k,q4_k", "--save-bytes", "350")
    assert rc == 0
    assert far["rules"] == [{"match": "blk.0.ffn_up_exps.weight", "type": "q6_k"},
                            {"match": "output.weight", "type": "q4_k"}]


def test_a_two_rung_ladder_is_the_rung_less_plan(tmp_path):
    rep = report(tmp_path, {0: 5.0, 1: 0.5, 2: 3.0, 3: 0.1})
    for target in (["--groups", "2"], ["--save-bytes", "1000"]):
        rc, a, _ = plan(rep, "--match", "_exps", *target)
        rc2, b, _ = run_plan(rep, "--ladder", "q4_0,q3_k", "--match", "_exps", *target)
        assert rc == rc2 == 0 and a == b


def test_a_dominated_rung_is_skipped_and_named(tmp_path):
    # q6_k costs more per byte than going straight to q4_k: off the hull
    line = {"tensor": "blk.0.attn_v.weight", "source_type": "bf16", "rows": 8, "cols": 256,
            "weighted": False, "types": {"q8_0": {"bytes": 1000, "error": 0.0},
                                         "q6_k": {"bytes": 900, "error": 50.0},
                                         "q4_k": {"bytes": 500, "error": 60.0}}}
    rep = tmp_path / "dom.jsonl"
    rep.write_text(json.dumps(line) + "\n")
    rc, pl, err = run_plan(rep, "--ladder", "q8_0,q6_k,q4_k", "--save-bytes", "1")
    assert rc == 0, err
    assert pl["rules"] == [{"match": "blk.0.attn_v.weight", "type": "q4_k"}]
    assert "1 rung(s) skipped as dominated" in err and "blk.0 q8_0->q4_k (skipped q6_k)" in err


def test_ladder_refusals(tmp_path):
    rep = ladder_report(tmp_path)
    rc, _, err = run_plan(rep, "--ladder", "q8_0,q6_k,q4_k", "--high", "q8_0", "--save-bytes", "1")
    assert rc != 0 and "not both" in err
    rc, _, err = run_plan(rep, "--ladder", "q8_0,q8_0", "--save-bytes", "1")
    assert rc != 0 and "two or more different types" in err
    rc, _, err = run_plan(rep, "--ladder", "q8_0,q6_k,q4_k", "--save-bytes", "100000")
    assert rc != 0 and "asked" in err
