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
