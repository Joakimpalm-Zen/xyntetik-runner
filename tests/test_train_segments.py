"""R17.9 step 4: a --save-every run's record commits to its checkpoints, so an auditor can re-run one segment.

The record lists every numbered checkpoint the run wrote ({step, cursor, adapter_sha256, optim_sha256}) and a chain
over them, bound to the record itself:
  h_0 = sha256("xyntetik.runner.train.chain.v1|" + sha256(the record's bytes before the chain fields, plus "}"))
  h_i = sha256("h_{i-1}|step|cursor|adapter_sha256|optim_sha256"), each as lowercase hex text
so a chain cannot be moved to another run: any change to the base's or the data's hash, the recipe, the overrides or
the environment changes h_0. The gate under it: in a T3 build, a run resumed from a checkpoint is the uninterrupted
run byte for byte at every later checkpoint (adapter and .optim), measured at several cut points.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROWS = [{"prompt": "question %d about item %d" % (i, i * 7 % 5), "completion": " answer %d" % (i * 3)}
        for i in range(5)]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def _env():
    return {k: v for k, v in os.environ.items() if not k.startswith("RUNNER_") and not k.startswith("XR_")}


def _run(exe, args):
    p = subprocess.run([str(exe), *map(str, args)], cwd=ROOT, capture_output=True, text=True, timeout=600,
                       env=_env())
    assert p.returncode == 0, p.stderr[-800:]
    return p


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _ck(out, k):
    return pathlib.Path("%s.step-%05d" % (out, k))


@pytest.fixture
def fixture(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), str(tmp_path / "base.gguf")],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    return tmp_path


def _recipe(d, steps, every):
    return ["-m", d / "base.gguf", "--train", d / "data.jsonl", "--train-steps", steps, "--save-every", every,
            "--lr", "3e-3", "--train-batch", 2, "--gpu", "off"]


def chain_of(text, rec):
    body = text[:text.index(',"checkpoints":')] + "}"
    h = hashlib.sha256(("xyntetik.runner.train.chain.v1|" +
                        hashlib.sha256(body.encode()).hexdigest()).encode()).hexdigest()
    h0 = h
    for c in rec["checkpoints"]:
        h = hashlib.sha256(("%s|%d|%d|%s|%s" % (h, c["step"], c["cursor"], c["adapter_sha256"],
                                                 c["optim_sha256"])).encode()).hexdigest()
    return h0, h


def test_a_save_every_run_records_its_checkpoints_and_a_chain_bound_to_the_record(runner_bin, fixture):
    out = fixture / "a.gguf"
    _run(runner_bin, [*_recipe(fixture, 6, 2), "-t", 2, "--train-out", out])
    text = pathlib.Path(str(out) + ".train.json").read_text()
    rec = json.loads(text)
    cks = rec.get("checkpoints")
    assert [c["step"] for c in cks or []] == [2, 4, 6], rec.keys()
    for c in cks:
        assert c["cursor"] == 2 * c["step"], c                        # batch 2: rows consumed
        assert c["adapter_sha256"] == _sha(_ck(out, c["step"])), c
        assert c["optim_sha256"] == _sha(str(_ck(out, c["step"])) + ".optim"), c
    h0, head = chain_of(text, rec)
    assert rec["checkpoint_chain"] == {"basis": "xyntetik.runner.train.chain.v1", "h0": h0, "head": head}
    # the binding's live control: the same chain over a record whose lr differs is another h_0
    moved = text.replace('"lr":', '"lr":1', 1)
    assert chain_of(moved, json.loads(moved))[0] != h0


def test_a_run_without_checkpoints_records_none(runner_bin, fixture):
    out = fixture / "b.gguf"
    _run(runner_bin, ["-m", fixture / "base.gguf", "--train", fixture / "data.jsonl", "--train-steps", 2,
                      "--lr", "3e-3", "-t", 2, "--gpu", "off", "--train-out", out])
    rec = json.loads(pathlib.Path(str(out) + ".train.json").read_text())
    assert "checkpoints" not in rec and "checkpoint_chain" not in rec


def test_resumed_runs_are_the_straight_run_byte_for_byte_in_a_t3_build(runner_bin, fixture):
    """The kill switch of segment replay, as a gate: if a resumed run were not byte-exact, segments could not be
    chained. Straight at -t 2; resumed at -t 4 from steps 1, 3 and 6 of 8, with warmup and a cosine schedule."""
    v = subprocess.run([str(runner_bin), "--version"], capture_output=True, text=True, timeout=20).stdout
    if "(t3)" not in v:
        pytest.skip("a T3 build's gate (make T3=1): a default build is exact on its own binary only")
    sched = ["--lr-warmup", 2, "--lr-schedule", "cosine", "--lr-floor", "0.1"]
    straight = fixture / "s.gguf"
    _run(runner_bin, [*_recipe(fixture, 8, 1), *sched, "-t", 2, "--train-out", straight])
    for cut in (1, 3, 6):
        out = fixture / ("r%d.gguf" % cut)
        _run(runner_bin, [*_recipe(fixture, 8, 1), *sched, "-t", 4, "--train-resume", _ck(straight, cut),
                          "--train-out", out])
        for k in range(cut + 1, 9):
            assert _sha(_ck(out, k)) == _sha(_ck(straight, k)), (cut, k, "adapter")
            assert _sha(str(_ck(out, k)) + ".optim") == _sha(str(_ck(straight, k)) + ".optim"), (cut, k, "optim")
        assert _sha(out) == _sha(straight), (cut, "final")
    ctl = fixture / "c.gguf"            # the live control: another lr is another adapter
    _run(runner_bin, [*[x if x != "3e-3" else "2e-3" for x in _recipe(fixture, 8, 1)], *sched, "-t", 2,
                      "--train-out", ctl])
    assert _sha(ctl) != _sha(straight)


# ---- attest replay RECORD --segment A..B: re-run steps A+1..B from the step-A checkpoint beside the record ----

def _attest(exe, args):
    return subprocess.run([str(exe), "attest", "replay", *map(str, args)], cwd=ROOT, capture_output=True, text=True,
                          timeout=600, env=_env())


def _same_binary(rec):
    return ["--allow-same-binary"] if rec.get("determinism") == "cpu-same-binary" else []


@pytest.fixture
def run6(runner_bin, fixture):
    out = fixture / "a.gguf"
    _run(runner_bin, [*_recipe(fixture, 6, 2), "-t", 2, "--train-out", out])
    rec_path = pathlib.Path(str(out) + ".train.json")
    return fixture, out, rec_path, json.loads(rec_path.read_text())


def test_a_segment_replays_from_its_checkpoint_and_verifies(runner_bin, run6):
    d, out, rec_path, rec = run6
    r = _attest(runner_bin, [rec_path, "--segment", "2..4", "-t", 2, *_same_binary(rec)])
    assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
    assert "VERIFIED segment 2..4" in r.stdout, r.stdout
    r = _attest(runner_bin, [rec_path, "--segment", "0..2", "-t", 2, *_same_binary(rec)])   # from the run's start
    assert r.returncode == 0 and "VERIFIED segment 0..2" in r.stdout, (r.stdout + r.stderr)[-800:]


def test_a_start_file_that_is_not_the_recorded_checkpoint_is_refused_before_training(runner_bin, run6):
    d, out, rec_path, rec = run6
    start = _ck(out, 2)
    start.write_bytes(start.read_bytes() + b"\0")
    r = _attest(runner_bin, [rec_path, "--segment", "2..4", "-t", 2, *_same_binary(rec)])
    assert r.returncode == 2 and "is not the checkpoint the record names" in r.stdout + r.stderr, r.stdout + r.stderr
    assert "train:" not in r.stderr                    # refused before any step ran


def test_an_edited_entry_or_a_moved_chain_is_refused(runner_bin, run6):
    d, out, rec_path, rec = run6
    text = rec_path.read_text()
    bad = json.loads(text)
    bad["checkpoints"][1]["adapter_sha256"] = "0" * 64
    rec_path.write_text(json.dumps(bad))
    r = _attest(runner_bin, [rec_path, "--segment", "2..4", "-t", 2, *_same_binary(rec)])
    assert r.returncode == 2 and "checkpoint_chain" in r.stdout + r.stderr, r.stdout + r.stderr
    rec_path.write_text(text.replace('"lr":', '"lr":1', 1))   # the chain moved to another recipe: h_0 differs
    r = _attest(runner_bin, [rec_path, "--segment", "2..4", "-t", 2, *_same_binary(rec)])
    assert r.returncode == 2 and "h0" in r.stdout + r.stderr, r.stdout + r.stderr


def test_a_segment_that_is_not_two_recorded_checkpoints_is_refused_by_name(runner_bin, run6):
    d, out, rec_path, rec = run6
    for seg, why in (("2..3", "not a checkpoint"), ("4..2", "ends before it starts"), ("2", "A..B")):
        r = _attest(runner_bin, [rec_path, "--segment", seg, "-t", 2, *_same_binary(rec)])
        assert r.returncode != 0 and why in r.stdout + r.stderr, (seg, r.stdout + r.stderr)


def test_a_teacher_dpo_or_full_record_is_refused_for_segments_by_name(runner_bin, run6):
    d, out, rec_path, rec = run6
    text = rec_path.read_text()
    for field, value, name in (("teacher", {"path": "t"}, "a teacher"), ("dpo", {"path": "p"}, "DPO"),
                               ("variant", {"path": "v", "sha256": "0" * 64}, "--train-full")):
        r0 = json.loads(text)
        r0[field] = value
        rec_path.write_text(json.dumps(r0))
        r = _attest(runner_bin, [rec_path, "--segment", "2..4", "-t", 2, *_same_binary(rec)])
        assert r.returncode == 2 and name in r.stdout + r.stderr, (field, r.stdout + r.stderr)


# ---- from the read of #942: the start files are hostile input, and so is every number in the record ----

@pytest.fixture(scope="module")
def hostile_bin():
    """The sanitizer build when it is fresh (make debug: ASan and UBSan), else the plain one, as
    tests/test_hostile_records.py chooses."""
    dbg = ROOT / ("runner-debug.exe" if sys.platform == "win32" else "runner-debug")
    if dbg.exists():
        newest = max((f.stat().st_mtime for f in (ROOT / "src").rglob("*.c")), default=0)
        if dbg.stat().st_mtime >= newest:
            return dbg
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture
def run6h(hostile_bin, fixture):
    """run6 made by the binary the hostile cases attest with: a same-binary record replays only on its own."""
    out = fixture / "a.gguf"
    _run(hostile_bin, [*_recipe(fixture, 6, 2), "-t", 2, "--train-out", out])
    rec_path = pathlib.Path(str(out) + ".train.json")
    return fixture, out, rec_path, json.loads(rec_path.read_text())


def test_a_symlinked_start_file_is_refused(hostile_bin, run6h):
    """The step-A files are opened as the record's other cited files are: a regular file by its base name, never
    a symlink, even to the right bytes."""
    d, out, rec_path, rec = run6h
    start = _ck(out, 2)
    elsewhere = d / "outside"
    elsewhere.mkdir()
    real = elsewhere / "real.step2"
    real.write_bytes(start.read_bytes())
    start.unlink()
    try:
        os.symlink(real, start)
    except (OSError, NotImplementedError):
        pytest.skip("no symlinks here")
    r = _attest(hostile_bin, [rec_path, "--segment", "2..4", "-t", 2, *_same_binary(rec)])
    assert r.returncode == 2 and "VERIFIED" not in r.stdout, (r.returncode, r.stdout + r.stderr)
    assert "not a regular file" in r.stderr, r.stderr


def _with_checkpoints(text, steps, entries):
    """The record's text with another steps count and checkpoint list, its chain recomputed as the runner does."""
    body = text[:text.index(',"checkpoints":')]
    body = body.replace('"steps":%d' % json.loads(text)["steps"], '"steps":%d' % steps, 1)
    h0 = hashlib.sha256(("xyntetik.runner.train.chain.v1|" +
                         hashlib.sha256((body + "}").encode()).hexdigest()).encode()).hexdigest()
    h = h0
    for c in entries:
        h = hashlib.sha256(("%s|%d|%d|%s|%s" % (h, c["step"], c["cursor"], c["adapter_sha256"],
                                                 c["optim_sha256"])).encode()).hexdigest()
    return (body + ',"checkpoints":' + json.dumps(entries, separators=(",", ":")) +
            ',"checkpoint_chain":{"basis":"xyntetik.runner.train.chain.v1","h0":"%s","head":"%s"}}\n' % (h0, h))


def test_a_hostile_checkpoint_interval_is_refused_without_overflow(hostile_bin, run6h):
    """Steps near INT_MAX: the interval check must not overflow int (UBSan aborts on it). A record whose second
    checkpoint is not twice its first is refused."""
    d, out, rec_path, rec = run6h
    c0 = rec["checkpoints"][0]
    entries = [dict(c0, step=1073741824), dict(c0, step=2147483646)]
    rec_path.write_text(_with_checkpoints(rec_path.read_text(), 2147483647, entries))
    r = _attest(hostile_bin, [rec_path, "--segment", "0..1073741824", "-t", 2, *_same_binary(rec)])
    assert r.returncode == 2, (r.returncode, (r.stdout + r.stderr)[-600:])
    assert "not every K steps" in r.stderr and "runtime error" not in r.stderr, r.stderr[-600:]


def test_the_record_is_checked_before_its_start_files(runner_bin, run6):
    """The record's own checks (schema, signature, determinism) come before the segment's: an unsigned record
    replayed with --trust-key is refused for its signature, never by the state of the files beside it. That holds
    in every build, exact or same-binary, so CI's T3 job, which takes no skips, runs it. Where the record is
    same-binary, its determinism refusal also comes first."""
    d, out, rec_path, rec = run6
    _ck(out, 2).unlink()
    r = _attest(runner_bin, [rec_path, "--segment", "2..4", "-t", 2, *_same_binary(rec), "--trust-key", "a" * 64])
    assert r.returncode == 2 and "MISMATCH signature" in r.stdout, (r.stdout + r.stderr)[-600:]
    assert "is not the checkpoint" not in r.stderr and "not a regular file" not in r.stderr, r.stderr
    if rec.get("determinism") == "cpu-same-binary":
        r = _attest(runner_bin, [rec_path, "--segment", "2..4", "-t", 2])
        assert r.returncode == 2 and "MISMATCH determinism" in r.stdout, (r.stdout + r.stderr)[-600:]


# ---- attest replay RECORD --sample s --seed S: s random segments, drawn from the auditor's seed ----

SAMPLE_ALGO = "xyntetik.runner.train.sample.v1"
SEED = "0123456789abcdef0123456789abcdef"


def _draw(seed, head, m, s):
    """The draw as the docs define it: segment j ranked by sha256(algo|seed|head|j), the s lowest, ascending."""
    rank = sorted(range(m), key=lambda j: hashlib.sha256(("%s|%s|%s|%d" % (SAMPLE_ALGO, seed, head, j)).encode())
                  .hexdigest())
    return sorted(rank[:s])


@pytest.fixture
def run8(runner_bin, fixture):
    out = fixture / "a.gguf"
    _run(runner_bin, [*_recipe(fixture, 8, 2), "-t", 2, "--train-out", out])
    rec_path = pathlib.Path(str(out) + ".train.json")
    rec = json.loads(rec_path.read_text())
    bounds = [0] + [c["step"] for c in rec["checkpoints"]]
    return fixture, out, rec_path, rec, ["%d..%d" % (bounds[j], bounds[j + 1]) for j in range(len(bounds) - 1)]


def _sample(exe, rec_path, rec, *args):
    return _attest(exe, [rec_path, "-t", 2, *_same_binary(rec), *args])


def test_a_sample_is_drawn_from_the_seed_and_the_record_and_verifies(runner_bin, run8):
    d, out, rec_path, rec, segs = run8
    head = rec["checkpoint_chain"]["head"]
    r = _sample(runner_bin, rec_path, rec, "--sample", 2, "--seed", SEED)
    assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
    want = " ".join(segs[j] for j in _draw(SEED, head, 4, 2))
    assert "sample: %s, seed %s, record head %s, 4 segments, 2 drawn" % (SAMPLE_ALGO, SEED, head) in r.stdout
    assert "sample: drawn " + want + "\n" in r.stdout, r.stdout
    assert r.stdout.count("VERIFIED segment ") == 2 and "VERIFIED sample: 2 of 4 segments" in r.stdout, r.stdout
    # the bound, worded as one: random draws, nothing proved about the rest
    assert "missed with probability 0.5" in r.stdout and "proves nothing about the segments not drawn" in r.stdout
    assert "always picks the same segment proves that segment only" in r.stdout


def test_another_seed_is_another_draw_and_os_prints_its_seed(runner_bin, run8):
    d, out, rec_path, rec, segs = run8
    head = rec["checkpoint_chain"]["head"]
    other = "f" * 32
    assert _draw(SEED, head, 4, 1) != _draw(other, head, 4, 1) or _draw(SEED, head, 4, 2) != _draw(other, head, 4, 2)
    r = _sample(runner_bin, rec_path, rec, "--sample", 1, "--seed", other)
    assert r.returncode == 0 and "sample: drawn %s\n" % segs[_draw(other, head, 4, 1)[0]] in r.stdout, r.stdout
    r = _sample(runner_bin, rec_path, rec, "--sample", 1, "--seed", "os")
    assert r.returncode == 0, (r.stdout + r.stderr)[-600:]
    seed = r.stdout.split(", seed ")[1].split(",")[0]
    assert len(seed) == 64 and int(seed, 16) >= 0, seed
    assert "sample: drawn %s\n" % segs[_draw(seed, head, 4, 1)[0]] in r.stdout, r.stdout


def test_a_draw_of_the_first_interval_replays_it_from_the_run_start(runner_bin, run8):
    d, out, rec_path, rec, segs = run8
    head = rec["checkpoint_chain"]["head"]
    seed = next("%032x" % i for i in range(1000) if _draw("%032x" % i, head, 4, 1) == [0])   # forced
    r = _sample(runner_bin, rec_path, rec, "--sample", 1, "--seed", seed)
    assert r.returncode == 0 and "VERIFIED segment 0..2" in r.stdout, (r.stdout + r.stderr)[-600:]


def test_a_sample_of_every_segment_replays_them_all_and_writes_a_report(runner_bin, run8):
    d, out, rec_path, rec, segs = run8
    rep = d / "audit.json"
    r = _sample(runner_bin, rec_path, rec, "--sample", 4, "--seed", SEED, "--report", rep)
    assert r.returncode == 0 and r.stdout.count("VERIFIED segment ") == 4, (r.stdout + r.stderr)[-600:]
    j = json.loads(rep.read_text())
    assert j["schema_version"] == "xyntetik.runner.sample.v1" and j["algorithm"] == SAMPLE_ALGO
    assert j["seed"] == SEED and j["record_head"] == rec["checkpoint_chain"]["head"]
    assert j["segments"] == 4 and j["drawn"] == [{"segment": s, "verdict": "VERIFIED"} for s in segs]
    assert j["verdict"] == "VERIFIED" and j["one_false_missed"] == 0


def test_a_sample_refuses_by_name(runner_bin, run8, fixture):
    d, out, rec_path, rec, segs = run8
    for args, why in (((), "--sample needs --seed"),
                      (("--seed", "abcd"), "at least 128 bits"),
                      (("--seed", "z" * 32), "hex"),
                      (("--seed", SEED, "--segment", "2..4"), "not both"),):
        r = _sample(runner_bin, rec_path, rec, "--sample", 1, *args)
        assert r.returncode != 0 and why in r.stdout + r.stderr, (args, r.stdout + r.stderr)
    for n in (0, 5):
        r = _sample(runner_bin, rec_path, rec, "--sample", n, "--seed", SEED)
        assert r.returncode != 0 and "between 1 and 4" in r.stdout + r.stderr, (n, r.stdout + r.stderr)
    r = _sample(runner_bin, rec_path, rec, "--seed", SEED, "--segment", "2..4")
    assert r.returncode != 0 and "--seed is the auditor's seed of --sample" in r.stderr, r.stderr
