"""RUNNER_METAL_BATCH_TILES: the decode microbatch on the GEMM tiles (R3.24 (b)).

The Metal decode microbatch runs its weight products as multi-column identity
matvecs, each column its solo decode's bits. RUNNER_METAL_BATCH_TILES=1 sends
them to the prefill GEMM tiles instead, from MB_TILES_MIN (6) columns: faster
at 6 to 8 slots, a different summation order, so tolerance-class and opt-in.

RUNNER_SCORE_MICRO=N scores every position as one column of an N-wide
microbatch, which is how the route gets its D1 rows. The anchors here:

- an identity microbatch scores byte for byte what solo decode scores, so the
  score mode itself adds nothing;
- under the tiles the score differs from solo (the tiles ran: the route line
  counts GEMM dispatches) with no top-1 flip, and the route detail says
  micro=tiles;
- below MB_TILES_MIN the tiles stay off and the bytes are solo's;
- a microbatch the backend declines is refused, never scored solo.
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
# about 140 tokens on the fixture's byte-level vocabulary, inside -c 256
TEXT = ("The quick brown fox jumps over the lazy dog while the five boxing wizards "
        "jump quickly, and a sphinx of black quartz judges my vow.")


@pytest.fixture(scope="module")
def runner_bin():
    if sys.platform == "darwin":
        pressure = int(subprocess.check_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]))
        if pressure >= 2:
            pytest.skip("Metal model runs wait for memory pressure below 2")
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    caps = json.loads(subprocess.run([str(exe), "--caps"], capture_output=True, text=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "metal":
        pytest.skip("no Metal device")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    # 256-wide q8_0 rows: every projection is a whole number of tile k-steps,
    # so the tiles take each weight the microbatch hands them
    f = tmp_path_factory.mktemp("mbtiles") / "wide-q8.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--wide",
                    "--quant", "q8_0", str(f)], check=True, cwd=ROOT, capture_output=True)
    return f


def _score(runner_bin, model, env, *extra):
    full = {k: v for k, v in os.environ.items()
            if k not in ("RUNNER_METAL_BATCH_TILES", "RUNNER_SCORE_MICRO", "RUNNER_SCORE_CHUNKED")}
    full.update(env)
    r = subprocess.run([str(runner_bin), "-m", str(model), "--score", "--score-top", "8",
                        "-p", TEXT, "-c", "256", *extra],
                       capture_output=True, text=True, env=full, timeout=300)
    return r


def _ok(r):
    assert r.returncode == 0, r.stderr[-1500:]
    doc = json.loads(r.stdout)
    m = re.search(r"score: route (\{.*\})", r.stderr)
    assert m, r.stderr[-1500:]
    return doc, json.loads(m.group(1))


def _gemm(route):
    return sum((route.get("kernels") or {}).get("gemm_by_type", {}).values())


def test_identity_microbatch_scores_solo_bytes(runner_bin, model):
    solo, sr = _ok(_score(runner_bin, model, {}))
    micro, mr = _ok(_score(runner_bin, model, {"RUNNER_SCORE_MICRO": "8"}))
    assert micro["logprobs"] == solo["logprobs"]
    assert micro["top"] == solo["top"]
    assert micro["score_route"]["paths"] == {"micro": {"forwards": solo["n_scored"], "rows_max": 8}}
    assert "micro=tiles" not in micro["score_route"]["backend_detail"]
    assert _gemm(mr) == 0


def test_tiles_differ_without_a_flip_and_say_so(runner_bin, model):
    solo, _ = _ok(_score(runner_bin, model, {}))
    tiles, tr = _ok(_score(runner_bin, model, {"RUNNER_SCORE_MICRO": "8",
                                               "RUNNER_METAL_BATCH_TILES": "1"}))
    assert _gemm(tr) > 0, "the tiles never ran"
    assert tiles["logprobs"] != solo["logprobs"]
    assert tiles["argmax"] == solo["argmax"]
    worst = max(abs(a - b) for a, b in zip(tiles["logprobs"], solo["logprobs"]))
    assert worst < 0.05, worst
    assert tiles["score_route"]["backend_detail"].endswith(" micro=tiles")
    assert "RUNNER_METAL_BATCH_TILES=1" in tr["tuning"]


def test_tiles_start_at_six_columns(runner_bin, model):
    solo, _ = _ok(_score(runner_bin, model, {}))
    for width, tiled in (("5", False), ("6", True)):
        doc, route = _ok(_score(runner_bin, model, {"RUNNER_SCORE_MICRO": width,
                                                    "RUNNER_METAL_BATCH_TILES": "1"}))
        assert (_gemm(route) > 0) == tiled, width
        assert (doc["logprobs"] == solo["logprobs"]) == (not tiled), width


def test_a_declined_microbatch_is_refused(runner_bin, model):
    r = _score(runner_bin, model, {"RUNNER_SCORE_MICRO": "8"}, "--gpu", "off")
    assert r.returncode != 0
    assert "declined the microbatch" in r.stderr
    r = _score(runner_bin, model, {"RUNNER_SCORE_MICRO": "8", "RUNNER_SCORE_CHUNKED": "1"})
    assert r.returncode != 0
    assert "set one" in r.stderr


def _rehash(path, record):
    """The record written again with its public chain hash recomputed (tests/test_transcript.py's
    _forge_record): the hash covers the bytes before ',"chain":'."""
    import hashlib
    body_record = dict(record)
    chain = dict(body_record.pop("chain"))
    body = json.dumps(body_record, separators=(",", ":"), ensure_ascii=False).encode()[:-1]
    chain["hash"] = hashlib.sha256(body).hexdigest()
    path.write_bytes(body + b',"chain":' + json.dumps(chain, separators=(",", ":")).encode() + b"}\n")


@pytest.mark.parametrize("parallel", [1, 8])
def test_receipt_names_actual_tiles_and_refuses_missing_schedule(runner_bin, model, tmp_path,
                                                               parallel):
    sys.path.insert(0, str(ROOT / "tests" / "conformance"))
    from harness import RunnerServer

    directory = tmp_path / "receipts"
    env = dict(os.environ, RUNNER_METAL_BATCH_TILES="1", RUNNER_METAL_BATCH="1",
               RUNNER_METAL_TIMING="0")
    markers = [f"request_private_{i}" for i in range(parallel)]
    # 32 decoded tokens a turn: enough for the eight to overlap at 6 or more columns, and a
    # replay that costs a quarter of the 96 this used to take (283 s on the hosted runner's
    # paravirtual GPU for this case, the most of a 30-minute job)
    ntok = 32
    with RunnerServer(runner_bin, model, ctx=512, parallel=parallel, env=env,
                      extra_args=["-t", "2", "--receipts", str(directory)]) as srv:
        def generate(i, n=ntok):
            data = json.dumps({"prompt": markers[i], "max_tokens": n,
                               "temperature": 0, "ignore_eos": True}).encode()
            req = urllib.request.Request(srv.base_url + "/v1/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as response:
                assert response.status == 200
                return json.load(response)
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            list(pool.map(generate, range(parallel)))
        if parallel > 1:
            markers.append("solo_after_tiled_requests")
            generate(len(markers) - 1, 8)
    records = [(p, json.loads(p.read_bytes())) for p in directory.glob("receipt-*.json")]
    assert len(records) == len(markers)
    tiled = []
    for path, record in records:
        route = record["profile"]["route"]
        has_tiles = route.get("micro") == "tiles"
        assert ("micro=tiles" in route["backend_detail"]) == has_tiles
        text = json.dumps({"prompt": record["prompt"], "profile": record["profile"]})
        own = record["prompt"]["text"]
        assert own in markers
        assert all(marker == own or marker not in text for marker in markers)
        if own == "solo_after_tiled_requests":
            assert not has_tiles, "a previous request's marker leaked into a solo generation"
        if has_tiles:
            tiled.append(path)
            # R3.72: the turn's decode schedule, the tiled steps as runs of
            # [first position, count, width], widths from MB_TILES_MIN (6) to
            # the microbatch's 8, increasing, within the turn's decoded tokens
            steps = route.get("micro_steps")
            assert steps and not route.get("micro_steps_truncated"), route
            end = -1
            for first, count, width in steps:
                assert first >= end and count >= 1 and 6 <= width <= 8, steps
                end = first + count
            assert sum(c for _, c, _ in steps) <= ntok, steps
        else:
            assert "micro_steps" not in route, route
    if parallel == 1:
        assert not tiled, "an enabled switch is not an actual tiled forward"
    else:
        assert tiled, "concurrent generation never took a tiled step"
        # R3.72: a tiled turn replays, its listed steps as column 0 of a microbatch of the
        # recorded width beside stand-ins, the others solo; main before this refused it as
        # "not replayable yet". One turn is replayed, the one with the most tiled steps: each
        # replay loads seven stand-ins, and the other turns' schedules are checked above.
        env = {k: v for k, v in os.environ.items() if k != "RUNNER_METAL_BATCH_TILES"}

        def verify(path, extra_env=None):
            return subprocess.run([runner_bin, "-m", str(model), "--verify", str(path),
                                   "--ignore-eos", "--no-tray"], capture_output=True, text=True,
                                  timeout=120, env=dict(env, **(extra_env or {})))
        def tiled_steps(path):
            return sum(c for _, c, _ in json.loads(path.read_bytes())["profile"]["route"]["micro_steps"])
        tiled.sort(key=tiled_steps, reverse=True)
        replay = verify(tiled[0])
        assert replay.returncode == 0, replay.stdout + replay.stderr
        assert json.loads(replay.stdout)["verdict"] == "VERIFIED", replay.stdout
        assert "tiled run" in replay.stderr and "stand-ins" in replay.stderr, replay.stderr
        # live controls: the tiles switched off in the verifier is a route the replay did not take,
        # and a recorded run the turn never reached is a step where the tiles did not run
        off = verify(tiled[0], {"RUNNER_METAL_BATCH_TILES": "0"})
        assert off.returncode == 3 and "did not take the recorded route" in off.stderr, off.stderr
        # memory, before the stand-ins load: a machine that cannot hold the turn's whole KV
        # cache beside them is told so with the numbers, not swapped or failed mid-load
        low = verify(tiled[0], {"RUNNER_TEST_RAM_AVAILABLE_MB": "1"})
        assert low.returncode == 3, low.stdout + low.stderr
        assert "maps the turn's whole" in low.stderr and "of RAM is available" in low.stderr, low.stderr
        assert "stand-ins" in low.stderr and "tiled run" not in low.stderr, low.stderr
        record = json.loads(tiled[0].read_bytes())
        assert "signature" not in record, "an unsigned server's receipt is re-hashed below"
        steps = record["profile"]["route"]["micro_steps"]
        end = steps[-1][0] + steps[-1][1]
        record["profile"]["route"]["micro_steps"] = steps + [[end + 1000, 1, 8]]
        forged = tmp_path / "forged.json"
        _rehash(forged, record)
        bad = verify(forged)
        assert bad.returncode == 3, bad.stdout + bad.stderr
        assert "did not run at the recorded steps" in bad.stderr, bad.stderr
