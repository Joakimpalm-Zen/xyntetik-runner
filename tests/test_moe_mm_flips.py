"""The flip harness's arithmetic, anchored on hand-computed values.

scripts/moe-mm-flips.py turns two routing traces into a flip-and-margin
account. Its numbers feed a promotion argument, so the arithmetic gets an
absolute anchor: a synthetic trace pair whose flips, margins, and logit
deltas are computed by hand here, not by the code under test.
"""
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "moe_mm_flips", ROOT / "scripts/moe-mm-flips.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def rec(pos, layer, experts, logits):
    return {"pos": pos, "layer": layer, "experts": experts,
            "gates": [0.5] * len(experts), "logits": logits}


def as_map(rows):
    return {(r["pos"], r["layer"]): r for r in rows}


def test_flip_and_margin_arithmetic():
    # Four experts, top-2 routing. Hand-computed:
    #  (0,0): off picks [3,1] from logits [.1,.4,.2,.9]; sorted desc
    #         .9,.4,.2,.1 -> boundary margin = .4 - .2 = .2 (exactly).
    #         on picks the same set in the same order: no flip.
    #  (1,0): off [.5,.49,.48,.0] picks [0,1], margin .49-.48 = .01;
    #         on picks [0,2]: a SET flip at a .01 near-tie.
    #  (2,0): off [.9,.8,.1,.0] picks [0,1]; on picks [1,0]:
    #         ORDER-only flip, margin .8-.1 = .7.
    off = as_map([
        rec(0, 0, [3, 1], [0.1, 0.4, 0.2, 0.9]),
        rec(1, 0, [0, 1], [0.5, 0.49, 0.48, 0.0]),
        rec(2, 0, [0, 1], [0.9, 0.8, 0.1, 0.0]),
    ])
    on = as_map([
        rec(0, 0, [3, 1], [0.1, 0.4, 0.2, 0.9]),
        rec(1, 0, [0, 2], [0.5, 0.48, 0.49, 0.0]),
        rec(2, 0, [1, 0], [0.9, 0.8, 0.1, 0.0]),
    ])
    out, flips = mod.compare(off, on)
    assert out["records"] == 3
    assert out["set_flips"] == 1 and out["order_only_flips"] == 1
    assert abs(out["set_flip_rate"] - 1 / 3) < 1e-12
    # the one flip's margin (.01) lands in the [.001,.01) bucket? No: .01 is
    # the bucket EDGE and belongs to [.01,.05) — hand-check the boundary.
    assert out["flip_margin_histogram"]["[0.01,0.05)"] == 1
    assert sum(out["flip_margin_histogram"].values()) == 1
    assert abs(out["flip_margin_max"] - 0.01) < 1e-9
    # the one set flip happened at (1,0) with the hand-computed margin
    assert flips == [((1, 0), out["flip_margin_max"])]
    # mean |Δlogit|: only (1,0) differs, by .01 at two of four entries,
    # over 12 total logit comparisons -> .02/12
    assert abs(out["router_logit_mean_abs_delta"] - 0.02 / 12) < 1e-12
    assert abs(out["router_logit_max_abs_delta"] - 0.01) < 1e-9


def test_boundary_margin_is_kth_minus_next():
    assert abs(mod.boundary_margin([0.1, 0.4, 0.2, 0.9], 2) - 0.2) < 1e-12
    assert abs(mod.boundary_margin([1.0, 0.0], 1) - 1.0) < 1e-12
