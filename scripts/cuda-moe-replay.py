#!/usr/bin/env python3
"""RUNNER_MOE_TRACE and RUNNER_MOE_REPLAY on the CUDA routers (R3.21.2).

The CPU routers recorded and replayed expert choices since #499; the device
routers did neither, so a GPU run under RUNNER_MOE_REPLAY routed as usual and
a GPU run under RUNNER_MOE_TRACE wrote nothing. For each model and each MoE
arm (the fused default, the eager host-routed path, the grouped and the
by-expert prefill), scored solo (one row a forward) and chunked (64-row
forwards, so a prefill path routes whole tiles):

  * a traced run scores the same as a plain one, and its trace holds the
    experts the CPU routers choose at every (position, layer), scored the
    same way;
  * replaying that trace scores the same, forcing every slot and counting
    no differing choice and no uncovered pair;
  * a trace with every expert id moved to the next one (a bijection, so each
    record stays distinct experts) is followed: the traced choice under it
    is the perturbed trace, the run counts the overrides, its scores move
    (unless the model routes every expert, where moving only reorders), and
    they agree with the CPU routers replaying the same trace.

Usage: cuda-moe-replay.py RUNNER MODEL [MODEL ...]
Exit status 0 when every case passes; prints a skip line and exits 0 when the
runner has no CUDA device.
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_fixture import read_header  # noqa: E402

# about 140 tokens on the fixtures' byte tokenizer (their context is 256):
# a solo pass of that many forwards, and three 64-row chunks
PROMPT = ("The quick brown fox jumps over the lazy dog while the five boxing "
          "wizards jump quickly and the sphinx of black quartz judges my vow.")

ARMS = {
    "fused": {"RUNNER_MOE_GROUPED": "0", "RUNNER_MOE_TWIN": "0"},
    "eager": {"RUNNER_MOE_EAGER": "1"},
    "grouped": {"RUNNER_MOE_GROUPED": "1", "RUNNER_MOE_TWIN": "0"},
    "twin": {"RUNNER_MOE_GROUPED": "0", "RUNNER_MOE_TWIN": "1"},
}
MODES = {"solo": {}, "chunked": {"RUNNER_SCORE_CHUNKED": "1", "RUNNER_SPEC_BATCH": "64"}}
CLEAR = ("RUNNER_MOE_TRACE", "RUNNER_MOE_REPLAY", "RUNNER_MOE_EAGER", "RUNNER_MOE_GROUPED",
         "RUNNER_MOE_TWIN", "RUNNER_SCORE_CHUNKED", "RUNNER_SPEC_BATCH", "RUNNER_CUDA_ROUTE")
# the CPU against CUDA logprob distance under the moved trace may exceed the
# unpinned distance in the same mode (arithmetic only: 1e-6 solo, up to 2e-3
# for gemma-4's chunked fixture) by this factor plus a floor; a wrong weight
# or renormalisation is off by ~0.1
CPU_FACTOR, CPU_FLOOR = 4.0, 1e-4


def score(runner, model, gpu, env):
    e = {k: v for k, v in os.environ.items() if k not in CLEAR}
    e.update(env)
    e["RUNNER_GPU_FALLBACK"] = "fail"
    p = subprocess.run([runner, "-m", model, "--score", "-p", PROMPT, "--no-tray",
                        "--gpu", "auto" if gpu else "off"], env=e,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    err = p.stderr.decode(errors="replace")
    if p.returncode != 0:
        sys.exit("FAIL %s: exit %d\n%s" % (model, p.returncode, err[-2000:]))
    route = None
    for ln in err.splitlines():
        if ln.startswith("score: route "):
            route = json.loads(ln[len("score: route "):])
    return json.loads(p.stdout)["logprobs"], route, err


def expert_count(model):
    """The model's declared expert count: a trace shows only the experts
    chosen, and a top-1 fixture may choose expert 0 at every routing."""
    with open(model, "rb") as fh:
        kv = read_header(fh)[0]
    counts = [v for k, v in kv.items() if k.endswith(".expert_count")]
    if not counts:
        sys.exit("FAIL %s: no expert_count in the header" % model)
    return int(counts[0])


def trace_of(path):
    recs = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            r = json.loads(ln)
            key = (r["pos"], r["layer"])
            if key in recs:
                sys.exit("FAIL: the trace records %s twice" % (key,))
            recs[key] = r["experts"]
    return recs


def write_trace(path, recs):
    with open(path, "w", encoding="utf-8") as fh:
        for (pos, layer), ex in sorted(recs.items()):
            fh.write(json.dumps({"pos": pos, "layer": layer, "experts": ex}) + "\n")


def counts(err):
    for ln in err.splitlines():
        if ln.startswith("moe-replay: ") and "forced" in ln:
            nums = [int(w) for w in ln.replace(",", " ").split() if w.isdigit()]
            return nums[0], nums[1], nums[2]
    sys.exit("FAIL: no moe-replay count line\n" + err[-1500:])


def check(runner, model, arm, mode, tmp):
    tag = "%s %s %s" % (os.path.basename(model), arm, mode)
    env = dict(ARMS[arm], **MODES[mode])
    lp0, route, err = score(runner, model, True, env)
    if (route or {}).get("backend") != "cuda":
        sys.exit("FAIL %s: did not run on the device" % tag)
    twin = (route.get("kernels") or {}).get("cuda_moe_twin") or 0
    if arm == "twin" and mode == "chunked" and not twin:
        sys.exit("FAIL %s: the by-expert prefill never ran" % tag)

    t_gpu, t_cpu = os.path.join(tmp, "gpu.jsonl"), os.path.join(tmp, "cpu.jsonl")
    for t in (t_gpu, t_cpu):
        if os.path.exists(t):
            os.remove(t)
    lp, _, _ = score(runner, model, True, dict(env, RUNNER_MOE_TRACE=t_gpu))
    if lp != lp0:
        sys.exit("FAIL %s: tracing changed the scores" % tag)
    lpc0, _, _ = score(runner, model, False, dict(MODES[mode], RUNNER_MOE_TRACE=t_cpu))
    d0 = max(abs(a - b) for a, b in zip(lp0, lpc0))
    rg, rc = trace_of(t_gpu), trace_of(t_cpu)
    if not rg or rg != rc:
        diff = [k for k in set(rg) | set(rc) if rg.get(k) != rc.get(k)]
        sys.exit("FAIL %s: the device trace (%d records) is not the CPU's (%d); "
                 "%d pairs differ, first %s" % (tag, len(rg), len(rc), len(diff),
                                                sorted(diff)[:3]))
    used = len(next(iter(rg.values())))

    lp, _, err = score(runner, model, True, dict(env, RUNNER_MOE_REPLAY=t_gpu))
    forced, differed, missing = counts(err)
    if lp != lp0 or forced != len(rg) * used or differed or missing:
        sys.exit("FAIL %s: replaying its own trace: scores %s, forced %d of %d, "
                 "differed %d, uncovered %d" % (tag, "same" if lp == lp0 else "CHANGED",
                                                forced, len(rg) * used, differed, missing))

    ne = expert_count(model)
    moved = {k: [(e + 1) % ne for e in ex] for k, ex in rg.items()}
    t_moved, t_pin, t_cpin = (os.path.join(tmp, n) for n in
                              ("moved.jsonl", "pin.jsonl", "cpin.jsonl"))
    write_trace(t_moved, moved)
    for t in (t_pin, t_cpin):
        if os.path.exists(t):
            os.remove(t)
    lp1, _, err = score(runner, model, True,
                        dict(env, RUNNER_MOE_REPLAY=t_moved, RUNNER_MOE_TRACE=t_pin))
    forced, differed, missing = counts(err)
    if trace_of(t_pin) != moved:
        sys.exit("FAIL %s: the device did not follow the moved trace" % tag)
    # a model that routes every expert (gemma-4's fixture, 2 of 2) can only
    # have its experts reordered: the pin must still take, but the scores
    # cannot move
    reorder = all(sorted(moved[k]) == sorted(ex) for k, ex in rg.items())
    if (forced != len(rg) * used or not differed or missing or
            (lp1 == lp0 and not reorder)):
        sys.exit("FAIL %s: the moved trace: forced %d of %d, differed %d, uncovered %d, "
                 "scores %s" % (tag, forced, len(rg) * used, differed, missing,
                                "moved" if lp1 != lp0 else "UNCHANGED"))
    lpc, _, _ = score(runner, model, False, dict(MODES[mode], RUNNER_MOE_REPLAY=t_moved,
                                                RUNNER_MOE_TRACE=t_cpin))
    if trace_of(t_cpin) != moved:
        sys.exit("FAIL %s: the CPU did not follow the moved trace" % tag)
    d = max(abs(a - b) for a, b in zip(lp1, lpc))
    bar = CPU_FACTOR * d0 + CPU_FLOOR
    if len(lp1) != len(lpc) or d > bar:
        sys.exit("FAIL %s: under the moved trace the device and the CPU differ by %.3g "
                 "(unpinned %.3g, bar %.2g)" % (tag, d, d0, bar))
    print("moe replay %s: %d routings x %d traced as the CPU's, replayed with no "
          "override, moved trace followed (%d overrides%s, CPU distance %.1e, unpinned "
          "%.1e)%s" %
          (tag, len(rg), used, differed,
           ", every expert routed: a reorder" if reorder else "", d, d0,
           ", by-expert tiles %d" % twin if twin else ""))


def main():
    runner, models = sys.argv[1], sys.argv[2:]
    caps = json.loads(subprocess.run([runner, "--caps"], stdout=subprocess.PIPE,
                                     check=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        print("moe replay: skipped (no CUDA device)")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        for model in models:
            for arm in ARMS:
                for mode in MODES:
                    check(runner, model, arm, mode, tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
