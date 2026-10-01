"""Session images: suspend, resume and fork a generation (R1.3.1-.3).

`--session-out FILE` writes a generation's state to one file: the tokens so
far, the KV of every position and the recurrent fold, the sampler's knobs and
rng state, the constraint, and the next token's logits, bound by a SHA-256
trailer. `--suspend-after N` stops the step loop after N generated tokens and
images it there; `--resume FILE` continues it; `--resume FILE --fork-seed S`
continues it under a new rng seed.

The gate is byte identity: a generation run straight through to its budget
and imaged at the end, and the same generation suspended half way, resumed
and imaged at the end, must write the SAME file. The image carries no
timestamp, so anything the resume failed to restore -- a KV row, the fold,
the rng, the penalty window, the constraint's validator -- shows up as a
differing byte in the tokens, the state or the logits. The gate runs seeded
sampling with a repeat penalty, --json, --json-schema, greedy, and a Mamba-2
hybrid whose fold is recurrent state no KV row holds.

Anchors outside the runner: hashlib for the trailer and the model digest in
the header, the fixture's byte vocabulary (token t is byte t-3) for the
tokens the image records against the bytes the run printed, and `cmp`-style
byte equality of two files the runner wrote in two different processes.
"""
import hashlib
import json
import pathlib
import struct
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAGIC = b"runner.session.1"
N = 50
HALF = 25
SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}},
          "required": ["text"], "additionalProperties": False}


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp("session")
    model, other = d / "model.gguf", d / "other.gguf"
    for path, extra in ((model, []), (other, ["--qk-norm"])):
        subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                        str(path), *extra], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
    hyb = d / "hybrid"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-hybrid.py",
                    str(hyb), "--dense"], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    schema = d / "schema.json"
    schema.write_text(json.dumps(SCHEMA))
    other_schema = d / "other-schema.json"
    other_schema.write_text(json.dumps({**SCHEMA, "required": []}))
    return {"d": d, "model": model, "other": other,
            "hybrid": hyb.with_suffix(".gguf"), "schema": schema,
            "other_schema": other_schema}


def _run(runner_bin, *args):
    r = subprocess.run([runner_bin, "-t", "2", *map(str, args)], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    return r.returncode, r.stdout, r.stderr.decode(errors="replace")


def _image(path):
    """Parses an image by its documented layout (session.h)."""
    b = pathlib.Path(path).read_bytes()
    assert b[:16] == MAGIC
    assert hashlib.sha256(b[:-32]).digest() == b[-32:], "trailer is not the SHA-256 of the bytes"
    off = 16
    (hl,) = struct.unpack_from("<I", b, off); off += 4
    header = json.loads(b[off:off + hl]); off += hl
    (nt,) = struct.unpack_from("<I", b, off); off += 4
    tokens = list(struct.unpack_from(f"<{nt}i", b, off)); off += 4 * nt
    (sl,) = struct.unpack_from("<Q", b, off); off += 8
    state = b[off:off + sl]; off += sl
    (nv,) = struct.unpack_from("<I", b, off); off += 4
    logits = struct.unpack_from(f"<{nv}f", b, off); off += 4 * nv
    assert off == len(b) - 32, "bytes after the last section"
    return {"header": header, "tokens": tokens, "state": state, "logits": logits,
            "sha256": hashlib.sha256(b[:-32]).hexdigest()}


# (id, model key, generation args): each a generation the gate runs twice
CASES = [
    ("seeded-penalty", "model", ["-p", "hello", "-s", "7", "--temp", "0.9",
                                 "--repeat-penalty", "1.3", "--ignore-eos"]),
    ("greedy", "model", ["-p", "hello", "--temp", "0", "--ignore-eos"]),
    ("json", "model", ["-p", "hello", "-s", "3", "--temp", "0.8", "--json"]),
    ("json-schema", "model", ["-p", "hello", "-s", "5", "--temp", "0.8",
                              "--json-schema", "{schema}"]),
    ("recurrent", "hybrid", ["-p", "hello", "-s", "5", "--temp", "0.9",
                             "--repeat-penalty", "1.2", "--ignore-eos"]),
]


def _args(fx, gen):
    return [a.format(schema=fx["schema"]) for a in gen]


def _resume_extra(gen):
    # a schema is not in the image, only its digest: the resume names it again
    return gen[gen.index("--json-schema"):gen.index("--json-schema") + 2] \
        if "--json-schema" in gen else []


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_suspend_and_resume_write_the_straight_runs_image(runner_bin, fx, case):
    name, mkey, gen = case
    model, gen = fx[mkey], _args(fx, gen)
    d = fx["d"] / name
    d.mkdir()
    rc, out_a, err = _run(runner_bin, "-m", model, "-n", N, *gen,
                          "--session-out", d / "a.img")
    assert rc == 0, err[-800:]
    rc, out_b1, err = _run(runner_bin, "-m", model, "-n", N, *gen,
                           "--suspend-after", HALF, "--session-out", d / "b1.img")
    assert rc == 0, err[-800:]
    assert f"{HALF} of {N} generated" in err
    rc, out_b2, err = _run(runner_bin, "-m", model, "--resume", d / "b1.img",
                           *_resume_extra(gen), "--session-out", d / "b2.img")
    assert rc == 0, err[-800:]

    a, b2 = (d / "a.img").read_bytes(), (d / "b2.img").read_bytes()
    assert a == b2, f"{name}: a suspended and resumed generation imaged differently"
    # what was printed agrees too: the suspended half, then the resumed half
    assert out_a == out_b1[:-1] + out_b2

    img, half = _image(d / "a.img"), _image(d / "b1.img")
    h = img["header"]
    assert h["schema_version"] == "xyntetik.runner.session.v1"
    assert h["model"]["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert h["generated"] == N and h["max_new"] == N
    assert h["n_tokens"] == h["n_prompt"] + N == len(img["tokens"])
    assert half["header"]["generated"] == HALF
    assert half["tokens"] == img["tokens"][:len(half["tokens"])]
    assert f"sha256 {img['sha256']}" in err
    assert "timestamp" not in json.dumps(h) and "time" not in h
    # the schema's digest is the transcript's: sha256 of its compact JSON
    assert h["constraint"]["json_schema_sha256"] == (
        hashlib.sha256(json.dumps(SCHEMA, separators=(",", ":")).encode()).hexdigest()
        if "--json-schema" in gen else None)
    assert h["constraint"]["json_mode"] is ("--json" in gen)
    if mkey == "model" and "--json" not in gen and "--json-schema" not in gen:
        # the fixture's byte vocabulary: what the image says was generated is
        # what the run printed, after the echoed prompt
        want = bytes(t - 3 for t in img["tokens"][h["n_prompt"]:] if t >= 3)
        assert out_a == b"hello" + want + b"\n"


def test_a_chain_of_suspensions_lands_on_the_same_image(runner_bin, fx):
    d = fx["d"] / "chain"
    d.mkdir()
    gen = ["-m", fx["hybrid"], "-p", "hello", "-n", N, "-s", "9", "--temp", "1.0",
           "--repeat-penalty", "1.1", "--ignore-eos"]
    assert _run(runner_bin, *gen, "--session-out", d / "a.img")[0] == 0
    assert _run(runner_bin, *gen, "--suspend-after", 10, "--session-out", d / "s10.img")[0] == 0
    rc, _, err = _run(runner_bin, "-m", fx["hybrid"], "--resume", d / "s10.img",
                      "--suspend-after", 30, "--session-out", d / "s30.img")
    assert rc == 0 and "30 of 50 generated" in err, err[-600:]
    rc, _, err = _run(runner_bin, "-m", fx["hybrid"], "--resume", d / "s30.img",
                      "--session-out", d / "end.img")
    assert rc == 0, err[-600:]
    assert (d / "a.img").read_bytes() == (d / "end.img").read_bytes()


def test_fork_is_reproducible_and_diverges_by_seed(runner_bin, fx):
    d = fx["d"] / "fork"
    d.mkdir()
    m = fx["model"]
    gen = ["-m", m, "-p", "hello", "-n", N, "-s", "21", "--temp", "1.0", "--ignore-eos"]
    assert _run(runner_bin, *gen, "--suspend-after", HALF, "--session-out", d / "s.img")[0] == 0
    outs = {}
    for tag, extra in (("r1", []), ("r2", []), ("f11", ["--fork-seed", 11]),
                       ("f11b", ["--fork-seed", 11]), ("f12", ["--fork-seed", 12])):
        rc, out, err = _run(runner_bin, "-m", m, "--resume", d / "s.img", *extra,
                            "--session-out", d / f"{tag}.img")
        assert rc == 0, err[-600:]
        if extra:
            assert "forked" in err
        outs[tag] = out
    # a resume without a fork seed continues the image's own rng: the same
    # continuation every time, and the one the straight run would have made
    assert outs["r1"] == outs["r2"]
    assert (d / "r1.img").read_bytes() == (d / "r2.img").read_bytes()
    # a fork seed is a new, reproducible continuation of the same prefix
    assert outs["f11"] == outs["f11b"]
    assert outs["f11"] != outs["f12"] and outs["f11"] != outs["r1"]
    base = _image(d / "s.img")["tokens"]
    for tag in ("r1", "f11", "f12"):
        assert _image(d / f"{tag}.img")["tokens"][:len(base)] == base


def test_resume_refuses_what_it_cannot_continue_exactly(runner_bin, fx):
    d = fx["d"] / "refuse"
    d.mkdir()
    m = fx["model"]
    gen = ["-m", m, "-p", "hello", "-n", N, "-s", "4", "--ignore-eos"]
    assert _run(runner_bin, *gen, "--suspend-after", HALF, "--session-out", d / "s.img")[0] == 0
    good = (d / "s.img").read_bytes()

    rc, _, err = _run(runner_bin, "-m", fx["other"], "--resume", d / "s.img")
    assert rc == 1 and "not the model the image was made with" in err, err[-400:]

    for at in (20, len(good) // 2, len(good) - 40, len(good) - 1):
        t = bytearray(good)
        t[at] ^= 1
        (d / f"t{at}.img").write_bytes(bytes(t))
        rc, out, err = _run(runner_bin, "-m", m, "--resume", d / f"t{at}.img")
        assert rc == 1 and "SHA-256 trailer does not match" in err, err[-400:]
        assert out == b""

    (d / "trunc.img").write_bytes(good[:-100])
    rc, _, err = _run(runner_bin, "-m", m, "--resume", d / "trunc.img")
    assert rc == 1 and "error: session:" in err

    rc, _, err = _run(runner_bin, "-m", m, "--resume", d / "missing.img")
    assert rc == 1 and "cannot read" in err


def test_an_image_is_never_overwritten(runner_bin, fx):
    d = fx["d"] / "exists"
    d.mkdir()
    out = d / "taken.img"
    out.write_bytes(b"precious")
    rc, stdout, err = _run(runner_bin, "-m", fx["model"], "-p", "hello", "-n", 10,
                           "--suspend-after", 5, "--session-out", out)
    assert rc == 1 and "already exists" in err, err[-400:]
    assert stdout == b"", "refused after generating, not before"
    assert out.read_bytes() == b"precious"


def test_a_schema_image_resumes_only_under_its_schema(runner_bin, fx):
    d = fx["d"] / "schema-refuse"
    d.mkdir()
    m = fx["model"]
    assert _run(runner_bin, "-m", m, "-p", "hello", "-n", N, "-s", "5", "--temp",
                "0.8", "--json-schema", fx["schema"], "--suspend-after", HALF,
                "--session-out", d / "s.img")[0] == 0
    rc, _, err = _run(runner_bin, "-m", m, "--resume", d / "s.img")
    assert rc == 1 and "give --json-schema naming that schema" in err, err[-400:]
    rc, _, err = _run(runner_bin, "-m", m, "--resume", d / "s.img",
                      "--json-schema", fx["other_schema"])
    assert rc == 1 and "give --json-schema naming that schema" in err, err[-400:]

    assert _run(runner_bin, "-m", m, "-p", "hello", "-n", N, "--ignore-eos",
                "--suspend-after", HALF, "--session-out", d / "plain.img")[0] == 0
    rc, _, err = _run(runner_bin, "-m", m, "--resume", d / "plain.img",
                      "--json-schema", fx["schema"])
    assert rc == 1 and "was not generated under a schema" in err, err[-400:]


def test_a_generation_that_ends_early_writes_no_image_and_says_so(runner_bin, fx):
    d = fx["d"] / "ended"
    d.mkdir()
    # this seed samples end-of-text well inside the budget on this fixture;
    # an ended generation has nothing to resume
    rc, _, err = _run(runner_bin, "-m", fx["model"], "-p", "hello", "-n", 300,
                      "-s", "3", "--temp", "1.0", "--session-out", d / "x.img")
    assert "[end of text]" in err, "the fixture no longer ends early on this seed"
    assert rc == 1 and "ended before the image point" in err, err[-400:]
    assert not (d / "x.img").exists()


@pytest.mark.parametrize("args,why", [
    (["-p", "hi", "-n", 10, "--suspend-after", 5], "needs --session-out"),
    (["-p", "hi", "-n", 10, "--fork-seed", 3], "needs --resume"),
    (["-p", "hi", "-n", 10, "--suspend-after", 10, "--session-out", "{d}/x.img"],
     "must stop before the -n budget"),
    (["-p", "hi", "-n", -1, "--session-out", "{d}/x.img"], "finite generation budget"),
    (["-p", "hi", "-n", 10, "--session-out", "{d}/x.img", "--transcript", "{d}/t.json"],
     "one-shot runs"),
    (["-p", "hi", "-n", 10, "--session-out", "{d}/x.img", "--draft-lookup"],
     "solo step loop"),
    (["--resume", "{d}/x.img", "-p", "hi"], "give no -p"),
    (["-p", "hi", "-n", 10, "--session-out", "{d}/x.img", "--watermark", "{d}/k.json"],
     "no --watermark"),
])
def test_session_flags_refuse_combinations_they_cannot_honour(runner_bin, fx, tmp_path, args, why):
    if "--watermark" in args:
        assert _run(runner_bin, "--watermark-keygen", tmp_path / "k.json")[0] == 0
    rc, out, err = _run(runner_bin, "-m", fx["model"],
                        *[str(a).format(d=tmp_path) for a in args])
    assert rc == 1 and why in err, err[-400:]
    assert out == b""
    assert not (tmp_path / "x.img").exists()


@pytest.mark.parametrize("flag", [["-s", "5"], ["-n", "60"], ["-c", "128"], ["--kv", "f16"],
                                  ["--temp", "0.5"], ["--top-k", "3"], ["--top-p", "0.5"],
                                  ["--min-p", "0.2"], ["--repeat-penalty", "1.5"]],
                         ids=lambda f: f[0])
def test_resume_refuses_a_flag_the_image_fixes(runner_bin, fx, flag):
    d = fx["d"] / ("fixed" + flag[0])
    d.mkdir()
    m = fx["model"]
    assert _run(runner_bin, "-m", m, "-p", "hello", "-n", 20, "--ignore-eos",
                "--suspend-after", 10, "--session-out", d / "s.img")[0] == 0
    rc, out, err = _run(runner_bin, "-m", m, "--resume", d / "s.img", *flag)
    assert rc == 1 and f"{flag[0]} would change them" in err, err[-400:]
    assert out == b""
    if flag[0] == "-s":
        assert "--fork-seed" in err


def test_resume_refuses_a_constraint_the_image_was_not_made_under(runner_bin, fx):
    d = fx["d"] / "constraint"
    d.mkdir()
    m = fx["model"]
    assert _run(runner_bin, "-m", m, "-p", "hello", "-n", 20, "-s", "2",
                "--suspend-after", 10, "--session-out", d / "s.img")[0] == 0
    for flag in ("--json", "--ignore-eos"):
        rc, out, err = _run(runner_bin, "-m", m, "--resume", d / "s.img", flag)
        assert rc == 1 and f"not generated under {flag}" in err, err[-400:]


def test_a_greedy_image_does_not_carry_the_clock(runner_bin, fx):
    """An unseeded run's rng is the wall clock, and the image recorded it even
    for a greedy generation that never draws from it: two images of the same
    greedy state differed whenever the runs straddled a second, which made
    the byte-identity gate above pass or fail by timing."""
    import time
    d = fx["d"] / "clock"
    d.mkdir()
    gen = ["-p", "hello", "--temp", "0", "--ignore-eos"]
    rc, _, err = _run(runner_bin, "-m", fx["model"], "-n", N, *gen,
                      "--session-out", d / "a.img")
    assert rc == 0, err[-800:]
    time.sleep(1.1)
    rc, _, err = _run(runner_bin, "-m", fx["model"], "-n", N, *gen,
                      "--suspend-after", HALF, "--session-out", d / "b1.img")
    assert rc == 0, err[-800:]
    time.sleep(1.1)
    rc, _, err = _run(runner_bin, "-m", fx["model"], "--resume", d / "b1.img",
                      "--session-out", d / "b2.img")
    assert rc == 0, err[-800:]
    assert (d / "a.img").read_bytes() == (d / "b2.img").read_bytes()
