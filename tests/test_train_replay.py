"""R17.9(b): runner attest replay verifies .train.json runs before and after replay.

A verifier takes a training record (.train.json), validates schema, signature
(--trust-key), runner version, build compatibility (T3=1 for cpu-t3-exact, or
exact matching binary with --allow-same-binary for cpu-same-binary) and every
cited input hash before loading weights or training, then replays on the CPU
and compares the resulting artifact sha256 byte for byte.
"""
import hashlib
import json
import math
import os
import pathlib
import platform
import shutil
import struct
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = "the runner trains the adapter and the adapter learns the corpus. " * 3


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def _flavor(runner_bin):
    p = subprocess.run([str(runner_bin), "--version"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, check=True, text=True, timeout=30)
    return "t3" if "(t3)" in p.stdout else "default"


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _flip_last_byte(path):
    b = bytearray(pathlib.Path(path).read_bytes())
    b[-1] ^= 1
    pathlib.Path(path).write_bytes(bytes(b))


def _write_npy(path, descr, shape, payload):
    dims = ", ".join(str(s) for s in shape)
    hdr = "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }" % (descr, dims)
    hdr += " " * ((64 - (10 + len(hdr) + 1) % 64) % 64) + "\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr)) + hdr.encode() + payload)


def _write_teacher(d, lines, k=4, pos=5, mass=0.8):
    d.mkdir(parents=True, exist_ok=True)
    ids, lps = [], []
    for l in lines:
        for p in range(pos):
            if p < len(l["completion_tokens"]):
                t = l["completion_tokens"][p]
                row = [t] + [3 + (t * (j + 3)) % 250 for j in range(k - 1)]
                w = [2.0 ** -j for j in range(k)]
                lp = [math.log(mass * x / sum(w)) for x in w]
            else:
                row, lp = [0] * k, [0.0] * k
            ids += row
            lps += lp
    shape = (len(lines), pos, k)
    _write_npy(d / "topk_idx.npy", "<i4", shape, struct.pack("<%di" % len(ids), *ids))
    _write_npy(d / "topk_logp.npy", "<f4", shape, struct.pack("<%df" % len(lps), *lps))


def _run(exe, args, cwd=ROOT):
    return subprocess.run([str(exe), *map(str, args)], cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, timeout=600)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp("replay")
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), str(d / "base.gguf")],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    (d / "corpus.txt").write_text(CORPUS, encoding="utf-8")
    (d / "rows.jsonl").write_text(
        json.dumps({"prompt": "the runner trains", "completion": " the adapter", "weight": 1.0}) + "\n" +
        json.dumps({"prompt": "the adapter learns", "completion": " the corpus", "weight": 0.5}) + "\n",
        encoding="utf-8")
    id_rows = [
        {"prompt_tokens": [1, 5, 9], "completion_tokens": [11, 17, 23, 29, 31]},
        {"prompt_tokens": [1, 7], "completion_tokens": [13, 19, 21, 25]},
    ]
    (d / "id_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in id_rows), encoding="utf-8")
    _write_teacher(d / "teacher", id_rows)
    (d / "pairs.jsonl").write_text(
        json.dumps({"prompt": "the runner trains", "chosen": " the adapter", "rejected": " the corpus"}) + "\n" +
        json.dumps({"prompt": "the adapter learns", "chosen": " the corpus", "rejected": " the runner"}) + "\n",
        encoding="utf-8")
    return d


def _replay_flags(rec_path):
    rec = json.loads(pathlib.Path(rec_path).read_text(encoding="utf-8"))
    return ["--allow-same-binary"] if rec.get("determinism") == "cpu-same-binary" else []


def test_replay_verifies_relocated_dense_ce_and_requires_same_binary_on_default(runner_bin, fx, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(fx / "base.gguf", work / "base.gguf")
    shutil.copy(fx / "corpus.txt", work / "corpus.txt")
    out = work / "ad.gguf"
    p = _run(runner_bin, ["-m", work / "base.gguf", "--train", work / "corpus.txt",
                          "--train-steps", "4", "--lr", "3e-3", "--train-out", out, "-t", "2", "--gpu", "off"])
    assert p.returncode == 0, p.stderr
    want_sha = _sha(out)
    rec_path = pathlib.Path(f"{out}.train.json")
    rec = json.loads(rec_path.read_text(encoding="utf-8"))

    # Copy only record, base and data into a fresh directory (no adapter on disk)
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy(work / "base.gguf", moved / "base.gguf")
    shutil.copy(work / "corpus.txt", moved / "corpus.txt")
    shutil.copy(rec_path, moved / "ad.gguf.train.json")

    # R17.9: only a clean, stamped T3 build writes cpu-t3-exact; any other record replays as same-binary
    if rec["determinism"] != "cpu-t3-exact":
        assert rec["determinism"] == "cpu-same-binary"
        refused = _run(runner_bin, ["attest", "replay", moved / "ad.gguf.train.json"])
        assert refused.returncode == 2
        assert "MISMATCH determinism" in refused.stdout
        verdict = _run(runner_bin, ["attest", "replay", moved / "ad.gguf.train.json", "--allow-same-binary", "-t", "1"])
    else:
        assert _flavor(runner_bin) == "t3", rec
        verdict = _run(runner_bin, ["attest", "replay", moved / "ad.gguf.train.json", "-t", "1"])

    assert verdict.returncode == 0, verdict.stdout + verdict.stderr
    assert f"VERIFIED ad.gguf sha256 {want_sha}" in verdict.stdout
    assert not list(moved.glob("*.tmp")) and not (moved / "ad.gguf").exists()


def test_replay_covers_options_resume_dpo_teacher_and_full_weight(runner_bin, fx, tmp_path):
    d = tmp_path / "modes"
    d.mkdir()
    for name in ("base.gguf", "rows.jsonl", "id_rows.jsonl", "pairs.jsonl"):
        shutil.copy(fx / name, d / name)
    shutil.copytree(fx / "teacher", d / "teacher")

    # 1. --no-bos + --kv-f32 + --train-batch 1 + cosine schedule (tests position-mean when batch=1)
    m1 = d / "nobos-cos.gguf"
    p = _run(runner_bin, ["-m", d / "base.gguf", "--train", d / "rows.jsonl", "--no-bos", "--train-kv-f32",
                          "--train-batch", "1", "--lr-schedule", "cosine", "--lr-warmup", "1",
                          "--train-steps", "3", "--lr", "3e-3", "--train-out", m1, "--gpu", "off"])
    assert p.returncode == 0, p.stderr
    r = _run(runner_bin, ["attest", "replay", f"{m1}.train.json", *_replay_flags(f"{m1}.train.json")])
    assert r.returncode == 0 and f"VERIFIED nobos-cos.gguf sha256 {_sha(m1)}" in r.stdout

    # 2. Continued --lora with --lora-scale
    m2 = d / "scaled.gguf"
    p = _run(runner_bin, ["-m", d / "base.gguf", "--train", d / "rows.jsonl", "--lora", m1, "--lora-scale", "0.5",
                          "--train-steps", "2", "--lr", "3e-3", "--train-out", m2, "--gpu", "off"])
    assert p.returncode == 0, p.stderr
    r = _run(runner_bin, ["attest", "replay", f"{m2}.train.json", *_replay_flags(f"{m2}.train.json")])
    assert r.returncode == 0 and f"VERIFIED scaled.gguf sha256 {_sha(m2)}" in r.stdout

    # 3. Resumed checkpoint run (--save-every + --train-resume)
    ckpt = d / "ckpt.gguf"
    p = _run(runner_bin, ["-m", d / "base.gguf", "--train", d / "rows.jsonl", "--save-every", "2",
                          "--train-steps", "2", "--lr", "3e-3", "--train-out", ckpt, "--gpu", "off"])
    assert p.returncode == 0, p.stderr
    resumed = d / "resumed.gguf"
    p = _run(runner_bin, ["-m", d / "base.gguf", "--train", d / "rows.jsonl", "--train-resume", f"{ckpt}.step-00002",
                          "--train-steps", "4", "--lr", "3e-3", "--train-out", resumed, "--gpu", "off"])
    assert p.returncode == 0, p.stderr
    r = _run(runner_bin, ["attest", "replay", f"{resumed}.train.json", *_replay_flags(f"{resumed}.train.json")])
    assert r.returncode == 0 and f"VERIFIED resumed.gguf sha256 {_sha(resumed)}" in r.stdout

    # 4. --train-dpo
    dpo = d / "dpo.gguf"
    p = _run(runner_bin, ["-m", d / "base.gguf", "--train-dpo", d / "pairs.jsonl", "--dpo-beta", "0.2",
                          "--train-steps", "2", "--lr", "3e-3", "--train-out", dpo, "--gpu", "off"])
    assert p.returncode == 0, p.stderr
    r = _run(runner_bin, ["attest", "replay", f"{dpo}.train.json", "--allow-same-binary"])
    assert r.returncode == 0 and f"VERIFIED dpo.gguf sha256 {_sha(dpo)}" in r.stdout

    # 5. --train-teacher (override dir with --teacher when files sit in subfolder)
    kd = d / "kd.gguf"
    p = _run(runner_bin, ["-m", d / "base.gguf", "--train", d / "id_rows.jsonl", "--train-teacher", d / "teacher",
                          "--train-steps", "2", "--lr", "3e-3", "--train-out", kd, "--gpu", "off"])
    assert p.returncode == 0, p.stderr
    r = _run(runner_bin, ["attest", "replay", f"{kd}.train.json", "--teacher", d / "teacher", "--allow-same-binary"])
    assert r.returncode == 0 and f"VERIFIED kd.gguf sha256 {_sha(kd)}" in r.stdout

    # 6. --train-full variant
    full = d / "variant.gguf"
    p = _run(runner_bin, ["-m", d / "base.gguf", "--train", d / "rows.jsonl", "--train-full",
                          "--train-steps", "2", "--lr", "1e-3", "--train-out", full, "--gpu", "off"])
    assert p.returncode == 0, p.stderr
    r = _run(runner_bin, ["attest", "replay", f"{full}.train.json", "--allow-same-binary"])
    assert r.returncode == 0 and f"VERIFIED variant.gguf sha256 {_sha(full)}" in r.stdout


def test_replay_enforces_signatures_preflight_and_output_mismatch(runner_bin, fx, tmp_path):
    d = tmp_path / "preflight"
    d.mkdir()
    for name in ("base.gguf", "corpus.txt", "rows.jsonl", "id_rows.jsonl"):
        shutil.copy(fx / name, d / name)
    shutil.copytree(fx / "teacher", d / "teacher")

    assert _run(runner_bin, ["attest", "keygen", d / "k1.json"]).returncode == 0
    pub1 = _run(runner_bin, ["attest", "keygen", d / "k2.json"]).stdout.strip()
    assert pub1
    pub1 = json.loads((d / "k1.json").read_text())["public_key"]
    pub2 = json.loads((d / "k2.json").read_text())["public_key"]

    ad = d / "signed.gguf"
    p = _run(runner_bin, ["-m", d / "base.gguf", "--train", d / "corpus.txt", "--train-steps", "2",
                          "--lr", "3e-3", "--train-out", ad, "--sign-key", d / "k1.json", "--gpu", "off"])
    assert p.returncode == 0, p.stderr
    rec_file = pathlib.Path(f"{ad}.train.json")
    flags = _replay_flags(rec_file)

    ok = _run(runner_bin, ["attest", "replay", rec_file, "--trust-key", pub1, *flags])
    assert ok.returncode == 0 and "VERIFIED" in ok.stdout

    wrong_key = _run(runner_bin, ["attest", "replay", rec_file, "--trust-key", pub2, *flags])
    assert wrong_key.returncode == 2 and "MISMATCH signature" in wrong_key.stdout

    # Mutate signed JSON -> signature fails before replay
    bad_sig = d / "bad-sig.train.json"
    bad_sig.write_text(rec_file.read_text().replace('"steps":2', '"steps":3'), encoding="utf-8")
    tampered = _run(runner_bin, ["attest", "replay", bad_sig, *flags])
    assert tampered.returncode == 2 and "MISMATCH signature" in tampered.stdout

    # Unsigned record fails --trust-key
    plain = d / "plain.gguf"
    assert _run(runner_bin, ["-m", d / "base.gguf", "--train", d / "corpus.txt", "--train-steps", "2",
                             "--lr", "3e-3", "--train-out", plain, "--gpu", "off"]).returncode == 0
    plain_rec = pathlib.Path(f"{plain}.train.json")
    raw = json.loads(plain_rec.read_text(encoding="utf-8"))
    uns = _run(runner_bin, ["attest", "replay", plain_rec, "--trust-key", pub1, *flags])
    assert uns.returncode == 2 and "MISMATCH signature" in uns.stdout

    def check_mut(mutator, extra_args, want_prefix):
        mut = json.loads(json.dumps(raw))
        mutator(mut)
        pfile = d / "mut.train.json"
        pfile.write_text(json.dumps(mut), encoding="utf-8")
        res = _run(runner_bin, ["attest", "replay", pfile, *extra_args])
        assert res.returncode == 2 and want_prefix in res.stdout, (want_prefix, res.stdout, res.stderr)

    check_mut(lambda m: m.update({"schema_version": "xyntetik.runner.train.v99"}), flags, "MISMATCH schema_version")
    check_mut(lambda m: m.update({"determinism": "gpu-within-tolerance"}), flags, "MISMATCH determinism")
    check_mut(lambda m: m.update({"runner": "0.0.0"}), flags, "MISMATCH runner")
    check_mut(lambda m: m.update({"determinism": "cpu-t3-exact"}) or m["build"].update({"flavor": "bogus"}),
              [], "MISMATCH build.flavor")
    check_mut(lambda m: m.update({"determinism": "cpu-same-binary"}) or m["build"].update({"binary_sha256": "0" * 64}),
              ["--allow-same-binary"], "MISMATCH build.binary_sha256")
    if raw["determinism"] == "cpu-t3-exact":   # a clean, stamped T3 build (R17.9)
        check_mut(lambda m: m["build"].update({"commit": "0" * 40}), flags, "MISMATCH build.commit")
    check_mut(lambda m: m["base"].update({"sha256": "f" * 64}), flags, "MISMATCH base.sha256")
    check_mut(lambda m: m["data"].update({"sha256": "f" * 64}), flags, "MISMATCH data.sha256")
    check_mut(lambda m: m["adapter"].update({"sha256": "f" * 64}), flags, "MISMATCH adapter.sha256")

    # Resumed checkpoint optim mismatch & teacher mismatch fail pre-flight
    ckpt = d / "c.gguf"
    assert _run(runner_bin, ["-m", d / "base.gguf", "--train", d / "rows.jsonl", "--save-every", "1",
                             "--train-steps", "1", "--lr", "3e-3", "--train-out", ckpt, "--gpu", "off"]).returncode == 0
    res2 = d / "r2.gguf"
    assert _run(runner_bin, ["-m", d / "base.gguf", "--train", d / "rows.jsonl", "--train-resume", f"{ckpt}.step-00001",
                             "--train-steps", "2", "--lr", "3e-3", "--train-out", res2, "--gpu", "off"]).returncode == 0
    _flip_last_byte(f"{ckpt}.step-00001.optim")
    r_opt = _run(runner_bin, ["attest", "replay", f"{res2}.train.json", *_replay_flags(f"{res2}.train.json")])
    assert r_opt.returncode == 2 and "MISMATCH resume_optim_sha256" in r_opt.stdout

    kd = d / "kd2.gguf"
    assert _run(runner_bin, ["-m", d / "base.gguf", "--train", d / "id_rows.jsonl", "--train-teacher", d / "teacher",
                             "--train-steps", "1", "--lr", "3e-3", "--train-out", kd, "--gpu", "off"]).returncode == 0
    _flip_last_byte(d / "teacher" / "topk_idx.npy")
    r_kd = _run(runner_bin, ["attest", "replay", f"{kd}.train.json", "--teacher", d / "teacher", "--allow-same-binary"])
    assert r_kd.returncode == 2 and "MISMATCH teacher.topk_idx.sha256" in r_kd.stdout

    # P2/P3 alignment: tampered fresh lora_scale or tampered cpu-t3-exact on full_weight / DPO / canon-gap fail
    check_mut(lambda m: m.update({"lora_scale": 0.5}), flags, "MISMATCH recipe")
    full = d / "full-check.gguf"
    assert _run(runner_bin, ["-m", d / "base.gguf", "--train", d / "rows.jsonl", "--train-full",
                             "--train-steps", "1", "--lr", "1e-3", "--train-out", full, "--gpu", "off"]).returncode == 0
    full_mut = json.loads(pathlib.Path(f"{full}.train.json").read_text(encoding="utf-8"))
    if raw["determinism"] == "cpu-t3-exact":   # a clean, stamped T3 build (R17.9)
        full_mut["determinism"] = "cpu-t3-exact"
        fm_path = d / "full-t3.train.json"
        fm_path.write_text(json.dumps(full_mut), encoding="utf-8")
        r_ft3 = _run(runner_bin, ["attest", "replay", fm_path])
        assert r_ft3.returncode == 2 and "MISMATCH determinism" in r_ft3.stdout
        bf16 = ROOT / "test-bf16.gguf"
        if bf16.exists():
            bf_out = d / "bf-out.gguf"
            assert _run(runner_bin, ["-m", bf16, "--train", d / "corpus.txt", "--train-steps", "1",
                                     "--lr", "3e-3", "--train-out", bf_out, "--gpu", "off"]).returncode == 0
            bf_mut = json.loads(pathlib.Path(f"{bf_out}.train.json").read_text(encoding="utf-8"))
            bf_mut["determinism"] = "cpu-t3-exact"
            bf_path = d / "bf-t3.train.json"
            bf_path.write_text(json.dumps(bf_mut), encoding="utf-8")
            r_bf = _run(runner_bin, ["attest", "replay", bf_path, "-m", bf16])
            assert r_bf.returncode == 2 and "MISMATCH determinism" in r_bf.stdout


# R17.9, the second read: the replay takes the recipe and every setting that decides the bytes from the record
# alone. A model flag or RUNNER_ switch of the caller's that differs is a refusal before training, naming the field.
def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("RUNNER_") and not k.startswith("XR_")}
    env.update(extra)
    return env


def _run_env(exe, args, env):
    return subprocess.run([str(exe), *map(str, args)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=600, env=env)


def _trained(runner_bin, fx, d, name, extra=(), env=None):
    shutil.copy(fx / "base.gguf", d / "base.gguf")
    shutil.copy(fx / "corpus.txt", d / "corpus.txt")
    out = d / f"{name}.gguf"
    p = _run_env(runner_bin, ["-m", d / "base.gguf", "--train", d / "corpus.txt", "--train-steps", "3", "--lr", "3e-3",
                              "--train-out", out, "-t", "2", "--gpu", "off", *extra], env or _clean_env())
    assert p.returncode == 0, p.stderr
    return pathlib.Path(f"{out}.train.json"), _sha(out)


def _same_binary(rec_file):
    rec = json.loads(pathlib.Path(rec_file).read_text(encoding="utf-8"))
    return [] if rec["determinism"] == "cpu-t3-exact" else ["--allow-same-binary"]


def test_replay_takes_the_model_overrides_from_the_record_alone(runner_bin, fx, tmp_path):
    """The P1 of the second read: a run trained with --rope-base 50000 replayed to MISMATCH from its record
    alone, and to VERIFIED with --rope-base 50000 on the replay's command line, a recipe the record did not
    hold. Now the record holds it, the replay applies it, and a caller's value that differs is refused."""
    rec, sha = _trained(runner_bin, fx, tmp_path, "rope", extra=("--rope-base", "50000"))
    alone = _run_env(runner_bin, ["attest", "replay", rec, "-t", "1", *_same_binary(rec)], _clean_env())
    assert alone.returncode == 0 and f"VERIFIED rope.gguf sha256 {sha}" in alone.stdout, alone.stdout + alone.stderr
    other = _run_env(runner_bin, ["attest", "replay", rec, "--rope-base", "40000", *_same_binary(rec)], _clean_env())
    assert other.returncode == 2 and "MISMATCH model_overrides.rope_base" in other.stdout, other.stdout + other.stderr
    assert "train:" not in other.stderr, "refused only after training"


def test_replay_takes_the_environment_from_the_record_alone(runner_bin, fx, tmp_path):
    """The P2: a byte-moving switch (RUNNER_CPU_I8_PREFILL on arm64, RUNNER_CPU_I8K on AVX2) changes a default
    build's bytes, and a same-binary replay verified only when the replayer happened to set it too. Now the record
    names it, and a replay whose RUNNER_ switches differ from the record's is refused before training, whichever
    side has one."""
    sw = "RUNNER_CPU_I8_PREFILL" if platform.machine().lower() in ("arm64", "aarch64") else "RUNNER_CPU_I8K"
    rec, sha = _trained(runner_bin, fx, tmp_path, "i8", env=_clean_env(**{sw: "1"}))
    flags = ["attest", "replay", rec, "--allow-same-binary", "-t", "1"]
    missing = _run_env(runner_bin, flags, _clean_env())
    assert missing.returncode == 2 and "MISMATCH environment" in missing.stdout, missing.stdout + missing.stderr
    assert "train:" not in missing.stderr, "refused only after training"
    same = _run_env(runner_bin, flags, _clean_env(**{sw: "1"}))
    assert same.returncode == 0 and f"VERIFIED i8.gguf sha256 {sha}" in same.stdout, same.stdout + same.stderr
    other_dir = tmp_path / "plain"
    other_dir.mkdir()
    plain, _ = _trained(runner_bin, fx, other_dir, "plain")
    added = _run_env(runner_bin, ["attest", "replay", plain, "-t", "1", *_same_binary(plain)], _clean_env(**{sw: "1"}))
    assert added.returncode == 2 and "MISMATCH environment" in added.stdout, added.stdout + added.stderr


def test_replay_of_an_exact_record_refuses_a_dirty_build_on_either_side(runner_bin, fx, tmp_path):
    """A dirty build's source is not a commit: a record that says cpu-t3-exact and build.dirty true is not one a
    clean T3 run writes, and a dirty replaying binary is not the commit's. Both are refused before training."""
    if _flavor(runner_bin) != "t3":
        pytest.skip("a default build writes no cpu-t3-exact record")
    rec, _ = _trained(runner_bin, fx, tmp_path, "exact")
    raw = json.loads(rec.read_text(encoding="utf-8"))
    if raw["determinism"] != "cpu-t3-exact":
        pytest.skip("this T3 build is dirty or unstamped, so its records are not exact")
    raw["build"]["dirty"] = True
    edited = tmp_path / "dirty.train.json"
    edited.write_text(json.dumps(raw), encoding="utf-8")
    r = _run_env(runner_bin, ["attest", "replay", edited, "-m", tmp_path / "base.gguf", "--data", tmp_path / "corpus.txt"],
                 _clean_env())
    assert r.returncode == 2 and "MISMATCH build.dirty" in r.stdout, r.stdout + r.stderr
    marked = _run_env(runner_bin, ["attest", "replay", rec], _clean_env(RUNNER_TEST_BUILD_DIRTY="1"))
    assert marked.returncode == 2 and "MISMATCH build.dirty" in marked.stdout, marked.stdout + marked.stderr


def test_a_changed_data_byte_that_passes_preflight_is_a_mismatch_after_training(runner_bin, fx, tmp_path):
    """The trained live control: VERIFIED on the record as written; then one data byte flipped and the record's
    data hash edited to match, so the pre-flight passes and only training can tell. MISMATCH adapter.sha256."""
    rec, sha = _trained(runner_bin, fx, tmp_path, "live")
    ok = _run_env(runner_bin, ["attest", "replay", rec, "-t", "1", *_same_binary(rec)], _clean_env())
    assert ok.returncode == 0 and f"VERIFIED live.gguf sha256 {sha}" in ok.stdout, ok.stdout + ok.stderr
    data = bytearray((tmp_path / "corpus.txt").read_bytes())
    data[10] ^= 1
    (tmp_path / "corpus.txt").write_bytes(bytes(data))
    raw = json.loads(rec.read_text(encoding="utf-8"))
    raw["data"]["sha256"] = hashlib.sha256(bytes(data)).hexdigest()
    rec.write_text(json.dumps(raw), encoding="utf-8")
    bad = _run_env(runner_bin, ["attest", "replay", rec, "-t", "1", *_same_binary(rec)], _clean_env())
    assert bad.returncode == 2 and "MISMATCH adapter.sha256" in bad.stdout, bad.stdout + bad.stderr
    assert "train:" in bad.stderr, "the mismatch must come from training"


def test_replay_holds_the_ssm_switches_as_it_holds_the_runner_ones(runner_bin, fx, tmp_path):
    """XR_SSM_CHUNK and XR_SSM_SERIAL are read outside the RUNNER_ prefix and reach the recurrent scan's
    arithmetic; the record names them (#925). A replayer that sets one the record lacks is refused before
    training, as for a RUNNER_ switch."""
    rec, _ = _trained(runner_bin, fx, tmp_path, "plain-ssm")
    r = _run_env(runner_bin, ["attest", "replay", rec, "-t", "1", *_same_binary(rec)],
                 _clean_env(XR_SSM_SERIAL="1"))
    assert r.returncode == 2 and "MISMATCH environment" in r.stdout, r.stdout + r.stderr
    assert "train:" not in r.stderr, "refused only after training"


def test_replay_preserves_nonzero_train_seed(runner_bin, fx, tmp_path):
    """R17.10: -s N sets smp.rng and cli_resolve_inputs overwrites o->seed from smp.rng;
    cli_prepare_replay_train must copy the record's seed into smp.rng too so non-zero
    seed training records replay VERIFIED."""
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(fx / "base.gguf", work / "base.gguf")
    shutil.copy(fx / "corpus.txt", work / "corpus.txt")
    out = work / "seeded.gguf"
    p = _run(runner_bin, ["-m", work / "base.gguf", "--train", work / "corpus.txt",
                          "--train-steps", "4", "--lr", "3e-3", "-s", "7",
                          "--train-out", out, "-t", "2", "--gpu", "off"])
    assert p.returncode == 0, p.stderr
    want_sha = _sha(out)
    rec_path = pathlib.Path(f"{out}.train.json")
    assert json.loads(rec_path.read_text(encoding="utf-8"))["seed"] == 7

    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy(work / "base.gguf", moved / "base.gguf")
    shutil.copy(work / "corpus.txt", moved / "corpus.txt")
    shutil.copy(rec_path, moved / "seeded.gguf.train.json")
    verdict = _run(runner_bin, ["attest", "replay", moved / "seeded.gguf.train.json",
                                *_same_binary(moved / "seeded.gguf.train.json"), "-t", "1"])
    assert verdict.returncode == 0, verdict.stdout + verdict.stderr
    assert f"VERIFIED seeded.gguf sha256 {want_sha}" in verdict.stdout


