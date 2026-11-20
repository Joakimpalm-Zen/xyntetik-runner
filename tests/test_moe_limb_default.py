"""R3.83: moe-limb is the default on CUDA for a gemma-4 MoE file whose routed expert products are all Q4_0.

Through the CLI and the server on generated fixtures (scripts/make-test-moe.py at its default widths): what a load
chose is its record's profile.gpu_route; a replay takes the record's route (route_pinned decides before the switch
and the default); RUNNER_MOE_LIMB=0 and the reference route hold it off; the files it does not take keep today's route
and a gemma-4 MoE says why once; the route is part of the identity that keys stored rows (a preamble snapshot written
under one is not found under the other); and the split plan charges its scratch, a hand-computed number of bytes a
window column. The products' bits are the C gate's (make test-cuda-moe-limb-fixture, which runs the default, the
switch at 0 and the reference route). Every case needs a CUDA device and is skipped without one.
"""
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
sys.path.insert(0, str(ROOT / "tests"))
from harness import RunnerServer  # noqa: E402
import test_kv_snapshots as kvs  # noqa: E402  (the preamble snapshot helpers)

WHY = "gpu: moe-limb not taken, the experts run today's route: "


def _env(**extra):
    """the test's own environment: no RUNNER_ switch from outside, only the ones a case sets"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("RUNNER_")}
    env.update(extra)
    return env


@pytest.fixture(scope="module")
def fx(runner_bin, tmp_path_factory):
    caps = json.loads(subprocess.run([runner_bin, "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
                                     check=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("needs a CUDA device: moe-limb is a CUDA route")
    d = tmp_path_factory.mktemp("moelimb")
    env = _env()
    env.pop("MOE_TEST_DIMS", None)
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(d / "fx")], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL, env=env)
    out = {}
    for name, src, quant in (("g4q40", "fx.gemma4-moe.gguf", "q4_0"), ("g4q8", "fx.gemma4-moe.gguf", "q8_0"),
                             ("moe1q40", "fx.moe1.gguf", "q4_0")):
        q = subprocess.run([runner_bin, "-m", str(d / src), "--quantize", str(d / (name + ".gguf")),
                            "--quant", quant], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           timeout=120, env=env)
        assert q.returncode == 0, q.stderr.decode(errors="replace")
        out[name] = d / (name + ".gguf")
    out["d"] = d
    return out


def _run(runner_bin, model, *args, env=None):
    return subprocess.run([runner_bin, "-m", str(model), "-p", "the runner trains the", "-n", "12", "--temp", "0",
                           "-s", "7", "-t", "2", "--no-tray", *map(str, args)], cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=120, env=env or _env())


def _record(runner_bin, model, rec, *args, env=None):
    """a transcript of a 12-token greedy run: (its gpu_route as a list, stderr)"""
    p = _run(runner_bin, model, "--transcript", rec, *args, env=env)
    err = p.stderr.decode(errors="replace")
    assert p.returncode == 0, err
    r = json.loads(pathlib.Path(rec).read_bytes())
    if not r["profile"]["gpu"]:
        pytest.skip("the fixture did not run on the device")
    return r["profile"]["gpu_route"].split(","), err


def _verify(runner_bin, model, rec, env=None):
    v = subprocess.run([runner_bin, "-m", str(model), "--verify", str(rec), "-t", "2", "--no-tray"], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, env=env or _env())
    assert v.returncode == 0, v.stderr.decode(errors="replace")
    return json.loads(v.stdout)["verdict"]


def test_the_default_takes_it_and_its_record_replays(runner_bin, fx, tmp_path):
    """1 and 4: no switch set, a gemma-4 MoE with Q4_0 experts takes moe-limb, its record names it, and the
    record replays VERIFIED (with it, since the record's route decides)."""
    rec = tmp_path / "default.json"
    routes, err = _record(runner_bin, fx["g4q40"], rec)
    assert "moe-limb" in routes, routes
    assert "moe-grouped" not in routes, routes     # the route replaces the grouped prefill on such an instance
    assert WHY not in err, err
    assert _verify(runner_bin, fx["g4q40"], rec) == "VERIFIED"
    # the record's route decides before the switch: held off in the replay's environment, it still replays
    assert _verify(runner_bin, fx["g4q40"], rec, env=_env(RUNNER_MOE_LIMB="0")) == "VERIFIED"


def test_the_kill_switch_gives_todays_route_and_its_record_replays_without_it(runner_bin, fx, tmp_path):
    """2 and 4: RUNNER_MOE_LIMB=0 is today's route (every other route as the default has it), says why once, and
    its record replays without moe-limb, also where the replay's environment would take it."""
    on, _ = _record(runner_bin, fx["g4q40"], tmp_path / "on.json")
    rec = tmp_path / "off.json"
    off, err = _record(runner_bin, fx["g4q40"], rec, env=_env(RUNNER_MOE_LIMB="0"))
    assert "moe-limb" not in off, off
    assert [r for r in on if r != "moe-limb"] == off, (on, off)
    assert err.count(WHY) == 1 and WHY + "RUNNER_MOE_LIMB=0" in err, err
    assert _verify(runner_bin, fx["g4q40"], rec) == "VERIFIED"
    assert _verify(runner_bin, fx["g4q40"], rec, env=_env(RUNNER_MOE_LIMB="1")) == "VERIFIED"


def test_the_reference_route_holds_it_off(runner_bin, fx, tmp_path):
    """3: RUNNER_CUDA_ROUTE=reference (the CPU-against-CUDA harness's setting) and --gpu-route reference run
    none of the routes, moe-limb included: the default must not leak into the reference route."""
    for args, env, name in (((), _env(RUNNER_CUDA_ROUTE="reference"), "env"),
                            (("--gpu-route", "reference"), None, "pin")):
        routes, err = _record(runner_bin, fx["g4q40"], tmp_path / (name + ".json"), *args, env=env)
        assert routes == ["reference"], (name, routes)
        assert WHY + "the route pin leaves it out" in err, (name, err)


def test_outside_its_scope_today_s_route_runs_and_a_gemma4_moe_says_why(runner_bin, fx, tmp_path):
    """5: a gemma-4 MoE with Q8_0 experts, a non-gemma MoE with Q4_0 experts and a partial offload of the Q4_0
    gemma-4 file keep today's route, model-wide; the gemma-4 files say why, once. An explicit request still takes
    it on the device's layers of a partial offload, as the opt-in did."""
    routes, err = _record(runner_bin, fx["g4q8"], tmp_path / "q8.json")
    assert "moe-limb" not in routes, routes
    assert err.count(WHY) == 1 and "are Q8_0, and the route takes Q4_0 only" in err, err
    routes, err = _record(runner_bin, fx["moe1q40"], tmp_path / "moe1.json")
    assert "moe-limb" not in routes, routes
    assert WHY not in err, err                     # not a gemma-4 MoE: nothing to explain
    routes, err = _record(runner_bin, fx["g4q40"], tmp_path / "partial.json", "--gpu", "1")
    assert "moe-limb" not in routes, routes
    assert err.count(WHY) == 1 and "a partial offload: layer 1 runs on the host" in err, err
    routes, err = _record(runner_bin, fx["g4q40"], tmp_path / "partial-asked.json", "--gpu", "1",
                          env=_env(RUNNER_MOE_LIMB="1"))
    assert "moe-limb" in routes, routes
    assert WHY not in err, err


def test_a_preamble_snapshot_of_the_other_route_is_not_found(runner_bin, fx, tmp_path):
    """1: the route is part of the identity whose name a preamble snapshot carries (engine_prompt_route, which also
    keys the prefix store and a slot's rows): one written with the route off is a miss with it on, and the reverse,
    whether the switch holds it off or a pin naming every other route of the default does (no switch then names
    the difference). The live controls: the default twice, and a pin naming the default's whole set, are found."""
    on_routes, _ = _record(runner_bin, fx["g4q40"], tmp_path / "on.json")
    others = ",".join(r for r in on_routes if r != "moe-limb") or "reference"
    # the fixture's context is 256 byte tokens: a preamble over the 64-token threshold that leaves room
    system = "You are a coding agent. Read before you edit and run the tests after a change."

    def srv(snaps, run):
        env, args = run
        return RunnerServer(runner_bin, fx["g4q40"], ctx=256, env=env, extra_args=[
            "--gpu", "auto", "-t", "2", "--chat-template", "chatml", "--kv-snapshots", str(snaps),
            "--kv-snapshots-auto", "64", *args])
    on, off = (_env(), ()), (_env(RUNNER_MOE_LIMB="0"), ())
    pin_off, pin_on = (_env(), ("--gpu-route", others)), (_env(), ("--gpu-route", others + ",moe-limb"))
    cases = (("switch off -> on", off, on, False), ("on -> switch off", on, off, False),
             ("pin without it -> on", pin_off, on, False), ("on -> pin without it", on, pin_off, False),
             ("pin with it -> on", pin_on, on, True), ("on -> on", on, on, True))
    for i, (name, write, read, found) in enumerate(cases):
        snaps = tmp_path / ("snaps-%d" % i)
        with srv(snaps, write) as a:
            first = kvs._chat(a, system, "List the files.")
            assert first["reuse"] == "none", (name, first)
            assert len(kvs._wait_files(snaps, 1)) == 1, (name, list(snaps.iterdir()) if snaps.exists() else snaps)
        with srv(snaps, read) as b:
            warm = kvs._chat(b, system, "Which file is largest?")
        assert (warm["reuse"] == "preamble_snapshot") == found, (name, warm)


def test_the_split_plan_charges_its_scratch(runner_bin, fx):
    """7: the default charges the route's scratch to every window column (--fit's and the split's plan): a column's
    limbs (3 bytes an element), block scales (4 bytes a 32-element block), flag and zero expert entry (4 bytes
    each), at the wider expert input. Hand-computed for the fixture: max(n_embd 32, n_ff_exp 64) = 64, so
    3 * 64 + 4 * 2 + 8 = 208 bytes a column more than with the route held off (8,808 on Gemma-4-26B-A4B,
    n_embd 2816)."""
    assert _plan(runner_bin, fx)[1] - _plan(runner_bin, fx, env=_env(RUNNER_MOE_LIMB="0"))[1] == 208


def _plan(runner_bin, fx, *args, env=None):
    """the split's layer count and the window's bytes a column, from the plan's two gpu-split lines"""
    p = _run(runner_bin, fx["g4q40"], *args, env=env)
    err = p.stderr.decode(errors="replace")
    assert p.returncode == 0, err
    g = re.search(r"gpu-split: budget=\S+ fixed=\S+ G=(\d+)/\d+", err)
    m = re.search(r"gpu-split: prompt window \d+ columns \([^)]*?(\d+) bytes a column", err)
    assert g and m, err[-600:]
    return int(g.group(1)), int(m.group(1))


def test_the_plan_charges_its_scratch_only_where_the_route_runs(runner_bin, fx):
    """7 (the second reader of #976): a partial offload the default declines pays nothing for the route's scratch,
    the same bytes a window column as with the route held off; asked for there (RUNNER_MOE_LIMB=1) the route runs
    on the device's layers and the window pays its 208 bytes a column. The split is the same in every case: it was
    planned from the narrowest window's generic charge, which never held the route's scratch, so the route can cost
    window columns and never a layer (the 14 GB Gemma-4-26B file on a 12 GB card splits 23 of 30 layers with and
    without the charge: 409,388 against 400,580 bytes a column, the same 320 columns)."""
    off = _plan(runner_bin, fx, "--gpu", "1", env=_env(RUNNER_MOE_LIMB="0"))
    declined = _plan(runner_bin, fx, "--gpu", "1")
    asked = _plan(runner_bin, fx, "--gpu", "1", env=_env(RUNNER_MOE_LIMB="1"))
    assert off[0] == declined[0] == asked[0] == 1, (off, declined, asked)
    assert declined[1] == off[1], "a declined partial offload's window still pays for the route's scratch"
    assert asked[1] - off[1] == 208, (asked, off)
    full_off, full_on = _plan(runner_bin, fx, env=_env(RUNNER_MOE_LIMB="0")), _plan(runner_bin, fx)
    assert full_off[0] == full_on[0] == 2, (full_off, full_on)
