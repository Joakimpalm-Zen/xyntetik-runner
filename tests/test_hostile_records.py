"""R6.94: hostile-record sweep of `runner attest replay` and `runner attest unlearn`.

Every file an auditor verifies — the training record, parent record, data
file, drop list, and cited sidecars — is untrusted input. Each test crafts a
concrete hostile input targeting a distinct failure mechanism and must fail
cleanly without memory corruption, UB, path escape, silent buffer truncation,
false VERIFIED/exit-0 verdicts, or multi-hundred-MB preflight reads.
"""
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

ROWS = [
    {"prompt": "the runner trains", "completion": " the adapter"},
    {"prompt": "the adapter learns", "completion": " the corpus"},
    {"prompt": "the corpus teaches", "completion": " the model"},
]
DROP = [ROWS[1]]
RECIPE = ("--train-steps", "2", "--lr", "3e-3", "-t", "1", "--gpu", "off")


@pytest.fixture(scope="module")
def runner_bin():
    env_exe = os.environ.get("RUNNER_EXE")
    if env_exe:
        p = pathlib.Path(env_exe)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return p
    dbg = ROOT / ("runner-debug.exe" if sys.platform == "win32" else "runner-debug")
    if dbg.exists():
        newest = max((f.stat().st_mtime for f in (ROOT / "src").rglob("*.c")), default=0)
        if dbg.stat().st_mtime >= newest:
            return dbg
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def _run(exe, args, cwd=ROOT, env=None):
    return subprocess.run(
        [str(exe), *map(str, args)],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
        env=env,
    )


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _clean_env(**extra):
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("RUNNER_") and not k.startswith("XR_")
    }
    env.update(extra)
    return env


def _replay_flags(rec_path):
    rec = json.loads(pathlib.Path(rec_path).read_text(encoding="utf-8"))
    return ["--allow-same-binary"] if rec.get("determinism") == "cpu-same-binary" else []


def _write_rows(path, rows):
    pathlib.Path(path).write_bytes(
        "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")
    )


def _assert_no_sanitizer_or_crash(res):
    assert res.returncode >= 0, (res.returncode, res.stderr)
    assert "runtime error:" not in res.stderr, res.stderr
    assert "AddressSanitizer" not in res.stderr, res.stderr


@pytest.fixture(scope="module")
def fx(runner_bin, tmp_path_factory):
    """Real base model, parent CE record, replayable CE record, and unlearn record."""
    d = tmp_path_factory.mktemp("hostile_fx")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/make-test-model.py"), str(d / "base.gguf")],
        check=True,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
    )
    _write_rows(d / "data1.jsonl", ROWS)
    _write_rows(d / "drop.jsonl", DROP)
    (d / "corpus.txt").write_text("the runner trains cleanly. " * 4, encoding="utf-8")

    # Parent CE run
    p = _run(
        runner_bin,
        ["-m", d / "base.gguf", "--train", d / "data1.jsonl", "--train-out", d / "parent.gguf", *RECIPE],
        env=_clean_env(),
    )
    assert p.returncode == 0, p.stderr

    # Genuine unlearn run
    u = _run(
        runner_bin,
        ["--unlearn", d / "parent.gguf.train.json", "--drop", d / "drop.jsonl",
         "-m", d / "base.gguf", "--train", d / "data1.jsonl", "--train-out", d / "unlearned.gguf",
         "-t", "1", "--gpu", "off"],
        env=_clean_env(),
    )
    assert u.returncode == 0, u.stderr

    # Plain corpus replay run
    c = _run(
        runner_bin,
        ["-m", d / "base.gguf", "--train", d / "corpus.txt", "--train-out", d / "plain.gguf", *RECIPE],
        env=_clean_env(),
    )
    assert c.returncode == 0, c.stderr
    return d


def test_unlearn_dropped_count_ubsan_overflow_and_fractional_float(runner_bin, fx, tmp_path):
    """dropped_count=1e12 triggered UBSan float-cast-overflow; 1.9 truncated to 1 and VERIFIED."""
    d = tmp_path / "num"
    shutil.copytree(fx, d)
    rec = d / "unlearned.gguf.train.json"
    raw = json.loads(rec.read_text(encoding="utf-8"))
    flags = _replay_flags(rec)

    # 1. Float above INT_MAX (1e12 <= 1e12 limit): must refuse cleanly, never UB-crash
    big = json.loads(json.dumps(raw))
    big["unlearn"]["dropped_count"] = 1e12
    rec.write_text(json.dumps(big), encoding="utf-8")
    r_big = _run(runner_bin, ["attest", "unlearn", rec, "--parent-data", d / "data1.jsonl",
                              "--dropped", d / "drop.jsonl", *flags], env=_clean_env())
    _assert_no_sanitizer_or_crash(r_big)
    assert r_big.returncode == 2, (r_big.stdout, r_big.stderr)
    assert "VERIFIED" not in r_big.stdout

    # 2. Fractional float whose (int) truncation equals the true integer count
    frac = json.loads(json.dumps(raw))
    frac["unlearn"]["dropped_count"] = 1.9
    frac["unlearn"]["dropped_distinct"] = 1.5
    rec.write_text(json.dumps(frac), encoding="utf-8")
    r_frac = _run(runner_bin, ["attest", "unlearn", rec, "--parent-data", d / "data1.jsonl",
                               "--dropped", d / "drop.jsonl", *flags], env=_clean_env())
    _assert_no_sanitizer_or_crash(r_frac)
    assert r_frac.returncode == 2, (r_frac.stdout, r_frac.stderr)
    assert "VERIFIED" not in r_frac.stdout


def test_replay_rejects_float_where_integer_is_read(runner_bin, fx, tmp_path):
    """JSON float literals like 2.0 or 16.0 must not pass integer record fields."""
    d = tmp_path / "int_float"
    shutil.copytree(fx, d)
    src = (d / "plain.gguf.train.json").read_text(encoding="utf-8")
    raw = json.loads(src)
    flags = _replay_flags(d / "plain.gguf.train.json")

    replacements = [
        ("steps", f'"steps":{raw["steps"]}', f'"steps":{raw["steps"]}.0'),
        ("ctx", f'"ctx":{raw["ctx"]}', f'"ctx":{raw["ctx"]}.0'),
        ("lora_rank", f'"lora_rank":{raw["lora_rank"]}', f'"lora_rank":{raw["lora_rank"]}.0'),
        ("seed", f'"seed":{raw["seed"]}', f'"seed":{raw["seed"]}.0'),
        ("batch", f'"batch":{raw["recipe"]["batch"]}', f'"batch":{raw["recipe"]["batch"]}.0'),
        ("warmup", f'"warmup":{raw["recipe"]["warmup"]}', f'"warmup":{raw["recipe"]["warmup"]}.0'),
    ]
    for name, old, new in replacements:
        assert old in src, (name, old)
        mut = d / f"{name}.train.json"
        mut.write_text(src.replace(old, new, 1), encoding="utf-8")
        r = _run(runner_bin, ["attest", "replay", mut, *flags], env=_clean_env())
        _assert_no_sanitizer_or_crash(r)
        assert r.returncode == 2, (name, r.returncode, r.stdout, r.stderr)
        assert "VERIFIED" not in r.stdout


def test_unlearn_embedded_nul_byte_cannot_alias_rows(runner_bin, fx, tmp_path):
    """Embedded NUL in DATA1/D stopped strcmp at NUL, dropping unrelated rows."""
    d = tmp_path / "nul_row"
    d.mkdir()
    shutil.copy(fx / "base.gguf", d / "base.gguf")
    row_pub = b'{"prompt":"the runner","completion":" a\x00PUBLIC"}\n'
    row_sec = b'{"prompt":"the runner","completion":" a\x00SECRET"}\n'
    row_ok = b'{"prompt":"the runner trains","completion":" the adapter"}\n'
    row_drop = b'{"prompt":"the runner","completion":" a\x00OTHER"}\n'
    (d / "data1.jsonl").write_bytes(row_pub + row_sec + row_ok)
    (d / "drop.jsonl").write_bytes(row_drop)
    (d / "ok_only.jsonl").write_bytes(row_ok)

    # Craft a valid parent record pointing at NUL-containing data1, and an
    # unlearned child claiming the NUL rows were dropped cleanly via row_drop.
    p = _run(
        runner_bin,
        ["-m", d / "base.gguf", "--train", d / "ok_only.jsonl", "--train-out", d / "parent_tpl.gguf", *RECIPE],
        env=_clean_env(),
    )
    assert p.returncode == 0, p.stderr
    parent_raw = json.loads((d / "parent_tpl.gguf.train.json").read_text(encoding="utf-8"))
    parent_raw["data"]["path"] = "data1.jsonl"
    parent_raw["data"]["sha256"] = _sha(d / "data1.jsonl")
    (d / "parent.train.json").write_text(json.dumps(parent_raw), encoding="utf-8")

    # 1. --unlearn must refuse NUL bytes in data/drop and write nothing
    u = _run(
        runner_bin,
        ["--unlearn", d / "parent.train.json", "--drop", d / "drop.jsonl",
         "-m", d / "base.gguf", "--train", d / "data1.jsonl", "--train-out", d / "u.gguf",
         "-t", "1", "--gpu", "off"],
        env=_clean_env(),
    )
    _assert_no_sanitizer_or_crash(u)
    assert u.returncode != 0, (u.stdout, u.stderr)
    assert not (d / "u.gguf.retained.jsonl").exists()
    assert not (d / "u.gguf").exists()

    # 2. Even if someone replays/audits a forged record matching only row_ok,
    # attest unlearn must refuse NUL bytes rather than aliasing via strcmp.
    shutil.copy(d / "ok_only.jsonl", d / "u.gguf.retained.jsonl")
    child_raw = json.loads((d / "parent_tpl.gguf.train.json").read_text(encoding="utf-8"))
    child_raw["data"]["path"] = "u.gguf.retained.jsonl"
    child_raw["data"]["sha256"] = _sha(d / "ok_only.jsonl")
    child_raw["unlearn"] = {
        "parent_record": {"path": "parent.train.json", "sha256": _sha(d / "parent.train.json")},
        "dropped_sha256": _sha(d / "drop.jsonl"),
        "dropped_count": 2,
        "dropped_distinct": 1,
    }
    child_rec = d / "forged_nul.train.json"
    child_rec.write_text(json.dumps(child_raw), encoding="utf-8")
    a = _run(
        runner_bin,
        ["attest", "unlearn", child_rec, "--parent-data", d / "data1.jsonl",
         "--dropped", d / "drop.jsonl", *_replay_flags(child_rec)],
        env=_clean_env(),
    )
    _assert_no_sanitizer_or_crash(a)
    assert a.returncode == 2, (a.stdout, a.stderr)
    assert "VERIFIED" not in a.stdout


def test_replay_and_unlearn_reject_path_traversal_absolute_newlines_symlinks(runner_bin, fx, tmp_path):
    """Cited record paths must stay local basenames without control bytes or symlinks."""
    outside = tmp_path / "outside"
    shutil.copytree(fx, outside)

    audit = tmp_path / "audit"
    audit.mkdir()
    rec_src = json.loads((outside / "plain.gguf.train.json").read_text(encoding="utf-8"))
    flags = _replay_flags(outside / "plain.gguf.train.json")

    # 1. Path traversal (../outside/...) and absolute path escaping audit/
    for bad_base, bad_data in [
        ("../outside/base.gguf", "../outside/corpus.txt"),
        (str(outside / "base.gguf"), str(outside / "corpus.txt")),
    ]:
        mut = json.loads(json.dumps(rec_src))
        mut["base"]["path"] = bad_base
        mut["data"]["path"] = bad_data
        p = audit / "escape.train.json"
        p.write_text(json.dumps(mut), encoding="utf-8")
        r = _run(runner_bin, ["attest", "replay", p, *flags], env=_clean_env())
        _assert_no_sanitizer_or_crash(r)
        assert r.returncode == 2, (bad_base, r.stdout, r.stderr)
        assert "VERIFIED" not in r.stdout

    # 2. Newline / control character in adapter.path (stdout line injection)
    shutil.copy(outside / "base.gguf", audit / "base.gguf")
    shutil.copy(outside / "corpus.txt", audit / "corpus.txt")
    inj = json.loads(json.dumps(rec_src))
    inj["adapter"]["path"] = "plain.gguf\nVERIFIED spoofed sha256 " + "0" * 64
    p_inj = audit / "inj.train.json"
    p_inj.write_text(json.dumps(inj), encoding="utf-8")
    r_inj = _run(runner_bin, ["attest", "replay", p_inj, *flags], env=_clean_env())
    _assert_no_sanitizer_or_crash(r_inj)
    assert r_inj.returncode == 2, (r_inj.stdout, r_inj.stderr)
    lines = r_inj.stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith("MISMATCH adapter.path:"), r_inj.stdout

    # 3. Symlinks inside the audited folder must not be followed
    sym_dir = tmp_path / "symlinked"
    sym_dir.mkdir()
    try:
        (sym_dir / "base.gguf").symlink_to(outside / "base.gguf")
        (sym_dir / "corpus.txt").symlink_to(outside / "corpus.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    (sym_dir / "sym.train.json").write_text(json.dumps(rec_src), encoding="utf-8")
    r_sym = _run(runner_bin, ["attest", "replay", sym_dir / "sym.train.json", *flags], env=_clean_env())
    _assert_no_sanitizer_or_crash(r_sym)
    assert r_sym.returncode == 2, (r_sym.stdout, r_sym.stderr)
    assert "VERIFIED" not in r_sym.stdout

    # Symlink to unlearn parent_record must also be refused
    sym_u = tmp_path / "symlinked_unlearn"
    sym_u.mkdir()
    shutil.copy(outside / "base.gguf", sym_u / "base.gguf")
    shutil.copy(outside / "unlearned.gguf.retained.jsonl", sym_u / "unlearned.gguf.retained.jsonl")
    shutil.copy(outside / "data1.jsonl", sym_u / "data1.jsonl")
    shutil.copy(outside / "drop.jsonl", sym_u / "drop.jsonl")
    (sym_u / "parent.gguf.train.json").symlink_to(outside / "parent.gguf.train.json")
    u_rec = json.loads((outside / "unlearned.gguf.train.json").read_text(encoding="utf-8"))
    (sym_u / "u.train.json").write_text(json.dumps(u_rec), encoding="utf-8")
    r_usym = _run(
        runner_bin,
        ["attest", "unlearn", sym_u / "u.train.json", "--parent-data", sym_u / "data1.jsonl",
         "--dropped", sym_u / "drop.jsonl", *_replay_flags(sym_u / "u.train.json")],
        env=_clean_env(),
    )
    _assert_no_sanitizer_or_crash(r_usym)
    assert r_usym.returncode == 2, (r_usym.stdout, r_usym.stderr)
    assert "VERIFIED" not in r_usym.stdout


def test_silent_truncation_of_parent_data_sha_and_overlong_adapter_path(runner_bin, fx, tmp_path):
    """Fixed buffers truncated >64-char sha and >512-char adapter paths instead of refusing."""
    d = tmp_path / "trunc"
    shutil.copytree(fx, d)

    # 1. parent.data.sha256 with trailing suffix past 64 chars was truncated to 64 hex by snprintf
    p_path = d / "parent.gguf.train.json"
    p_raw = json.loads(p_path.read_text(encoding="utf-8"))
    p_raw["data"]["sha256"] = p_raw["data"]["sha256"] + "EXTRA"
    p_path.write_text(json.dumps(p_raw), encoding="utf-8")

    u_path = d / "unlearned.gguf.train.json"
    u_raw = json.loads(u_path.read_text(encoding="utf-8"))
    u_raw["unlearn"]["parent_record"]["sha256"] = _sha(p_path)
    u_path.write_text(json.dumps(u_raw), encoding="utf-8")

    r_sha = _run(
        runner_bin,
        ["attest", "unlearn", u_path, "--parent-data", d / "data1.jsonl",
         "--dropped", d / "drop.jsonl", *_replay_flags(u_path)],
        env=_clean_env(),
    )
    _assert_no_sanitizer_or_crash(r_sha)
    assert r_sha.returncode == 2, (r_sha.stdout, r_sha.stderr)
    assert "VERIFIED" not in r_sha.stdout

    # 2. Overlong adapter.path (>512 bytes) was silently truncated in VERIFIED output
    long_rec = d / "long_adapter.train.json"
    c_raw = json.loads((d / "plain.gguf.train.json").read_text(encoding="utf-8"))
    c_raw["adapter"]["path"] = ("a" * 520) + ".gguf"
    long_rec.write_text(json.dumps(c_raw), encoding="utf-8")
    r_long = _run(runner_bin, ["attest", "replay", long_rec, *_replay_flags(long_rec)], env=_clean_env())
    _assert_no_sanitizer_or_crash(r_long)
    assert r_long.returncode == 2, (r_long.stdout, r_long.stderr)
    assert "VERIFIED" not in r_long.stdout


def test_environment_block_rejects_arbitrary_vars_and_non_secret_wildcard(runner_bin, fx, tmp_path):
    """environment block must only hold tracked training switches and never wildcard non-secrets with (set)."""
    d = tmp_path / "env"
    shutil.copytree(fx, d)
    rec = d / "plain.gguf.train.json"
    raw = json.loads(rec.read_text(encoding="utf-8"))
    raw["determinism"] = "cpu-same-binary"

    # 1. Arbitrary process env var ("PATH") and inert variable ("RUNNER_EXE")
    for env_block in ({"PATH": "(set)"}, {"RUNNER_EXE": "(set)"}):
        mut = json.loads(json.dumps(raw))
        mut["environment"] = env_block
        p = d / "env_arb.train.json"
        p.write_text(json.dumps(mut), encoding="utf-8")
        r = _run(
            runner_bin,
            ["attest", "replay", p, "--allow-same-binary"],
            env=_clean_env(PATH=os.environ.get("PATH", "/usr/bin:/bin"), RUNNER_EXE=str(runner_bin)),
        )
        _assert_no_sanitizer_or_crash(r)
        assert r.returncode == 2, (env_block, r.stdout, r.stderr)
        assert "VERIFIED" not in r.stdout

    # 2. "(set)" wildcard on a non-credential training switch
    sw = "RUNNER_CPU_I8_PREFILL" if platform.machine().lower() in ("arm64", "aarch64") else "RUNNER_CPU_I8K"
    wild = json.loads(json.dumps(raw))
    wild["environment"] = {sw: "(set)"}
    p_wild = d / "env_wild.train.json"
    p_wild.write_text(json.dumps(wild), encoding="utf-8")
    r_wild = _run(
        runner_bin,
        ["attest", "replay", p_wild, "--allow-same-binary"],
        env=_clean_env(**{sw: "1"}),
    )
    _assert_no_sanitizer_or_crash(r_wild)
    assert r_wild.returncode == 2, (r_wild.stdout, r_wild.stderr)
    assert "VERIFIED" not in r_wild.stdout


def test_oversized_record_and_preflight_order_before_large_allocation(runner_bin, fx, tmp_path):
    """Oversized records (>16 MiB) and invalid unlearn records must fail before reading large inputs."""
    d = tmp_path / "preflight"
    shutil.copytree(fx, d)

    # 1. >16 MiB hostile record refused before JSON AST allocation
    huge = d / "huge.train.json"
    with open(huge, "wb") as f:
        f.write(b'{"schema_version":"bad","pad":"')
        chunk = b"A" * (1024 * 1024)
        for _ in range(17):
            f.write(chunk)
        f.write(b'"}\n')
    r_huge = _run(runner_bin, ["attest", "replay", huge], env=_clean_env())
    _assert_no_sanitizer_or_crash(r_huge)
    assert r_huge.returncode in (1, 2)
    assert "too large" in r_huge.stderr.lower() or "16 mib" in r_huge.stderr.lower(), (
        r_huge.stdout, r_huge.stderr
    )

    # 2. Invalid scalar field (steps = -1) must be refused BEFORE reading/splitting parent DATA1 & D
    u_rec = d / "unlearned.gguf.train.json"
    u_raw = json.loads(u_rec.read_text(encoding="utf-8"))
    (d / "plain.txt").write_text("plain non jsonl text\n", encoding="utf-8")
    u_raw["unlearn"]["parent_data_not_read"] = True
    u_raw["steps"] = -1
    bad_p = json.loads((d / "parent.gguf.train.json").read_text(encoding="utf-8"))
    bad_p["steps"] = -1
    bad_p["data"]["sha256"] = _sha(d / "plain.txt")
    (d / "parent.gguf.train.json").write_text(json.dumps(bad_p), encoding="utf-8")
    u_raw["unlearn"]["parent_record"]["sha256"] = _sha(d / "parent.gguf.train.json")
    del u_raw["unlearn"]["parent_data_not_read"]
    u_rec.write_text(json.dumps(u_raw), encoding="utf-8")
    r_ord = _run(
        runner_bin,
        ["attest", "unlearn", u_rec, "--parent-data", d / "plain.txt",
         "--dropped", d / "drop.jsonl", *_replay_flags(u_rec)],
        env=_clean_env(),
    )
    _assert_no_sanitizer_or_crash(r_ord)
    assert r_ord.returncode == 2, (r_ord.stdout, r_ord.stderr)
    assert "MISMATCH recipe" in r_ord.stdout, (r_ord.stdout, r_ord.stderr)

    # 3. A D that drops every row of DATA1 must fail preflight BEFORE loading the model
    shutil.copytree(fx, d / "alldrop", dirs_exist_ok=True)
    ad = d / "alldrop"
    _write_rows(ad / "all_drop.jsonl", ROWS)
    ad_u = json.loads((ad / "unlearned.gguf.train.json").read_text(encoding="utf-8"))
    ad_u["data"]["sha256"] = hashlib.sha256(b"").hexdigest()
    ad_u["unlearn"]["dropped_sha256"] = _sha(ad / "all_drop.jsonl")
    ad_u["unlearn"]["dropped_count"] = len(ROWS)
    ad_u["unlearn"]["dropped_distinct"] = len(ROWS)
    (ad / "alldrop.train.json").write_text(json.dumps(ad_u), encoding="utf-8")
    r_ad = _run(
        runner_bin,
        ["attest", "unlearn", ad / "alldrop.train.json", "--parent-data", ad / "data1.jsonl",
         "--dropped", ad / "all_drop.jsonl", *_replay_flags(ad / "alldrop.train.json")],
        env=_clean_env(),
    )
    _assert_no_sanitizer_or_crash(r_ad)
    assert r_ad.returncode == 2, (r_ad.stdout, r_ad.stderr)
    assert "load:" not in r_ad.stderr, (r_ad.stdout, r_ad.stderr)


def test_type_confusion_eval_kill_eligibility_and_signature_nul(runner_bin, fx, tmp_path):
    """Type confusion, kill/eval syntax, ineligible unlearn modes, and embedded NUL before signature."""
    d = tmp_path / "types"
    shutil.copytree(fx, d)
    raw = json.loads((d / "plain.gguf.train.json").read_text(encoding="utf-8"))
    flags = _replay_flags(d / "plain.gguf.train.json")

    # 1. full_weight: null must fail type validation instead of acting like full_weight=true
    fw_null = json.loads(json.dumps(raw))
    fw_null["full_weight"] = None
    (d / "fw_null.train.json").write_text(json.dumps(fw_null), encoding="utf-8")
    r_fw = _run(runner_bin, ["attest", "replay", d / "fw_null.train.json", *flags], env=_clean_env())
    _assert_no_sanitizer_or_crash(r_fw)
    assert r_fw.returncode == 2 and "MISMATCH full_weight" in r_fw.stdout, (r_fw.stdout, r_fw.stderr)

    # 2. Unknown kill.metric must fail in preflight (exit 2) before loading the model
    km = json.loads(json.dumps(raw))
    km["evals"] = [{"label": "val", "rows": {"path": "corpus.txt", "sha256": _sha(d / "corpus.txt")}}]
    km["kill"] = {"at": 1, "metric": "nonexistent", "min_rel": 0.0}
    (d / "kill_bad.train.json").write_text(json.dumps(km), encoding="utf-8")
    r_km = _run(runner_bin, ["attest", "replay", d / "kill_bad.train.json", *flags], env=_clean_env())
    _assert_no_sanitizer_or_crash(r_km)
    assert r_km.returncode == 2 and "MISMATCH kill" in r_km.stdout, (r_km.stdout, r_km.stderr)
    assert "load:" not in r_km.stderr

    # 3. attest unlearn must refuse ineligible parent/child modes (e.g. DPO on both)
    pairs = [
        {"prompt": "the runner", "chosen": " a1", "rejected": " a2"},
        {"prompt": "the adapter", "chosen": " b1", "rejected": " b2"},
        {"prompt": "the corpus", "chosen": " c1", "rejected": " c2"},
    ]
    _write_rows(d / "pairs.jsonl", pairs)
    _write_rows(d / "drop_pair.jsonl", pairs[1:2])
    _write_rows(d / "kept_pair.jsonl", [pairs[0], pairs[2]])
    assert _run(
        runner_bin,
        ["-m", d / "base.gguf", "--train-dpo", d / "pairs.jsonl", "--dpo-beta", "0.2",
         "--train-out", d / "dpo_p.gguf", *RECIPE],
        env=_clean_env(),
    ).returncode == 0
    assert _run(
        runner_bin,
        ["-m", d / "base.gguf", "--train-dpo", d / "kept_pair.jsonl", "--dpo-beta", "0.2",
         "--train-out", d / "dpo_u.gguf", *RECIPE],
        env=_clean_env(),
    ).returncode == 0
    dpo_u = json.loads((d / "dpo_u.gguf.train.json").read_text(encoding="utf-8"))
    dpo_u["unlearn"] = {
        "parent_record": {"path": "dpo_p.gguf.train.json", "sha256": _sha(d / "dpo_p.gguf.train.json")},
        "dropped_sha256": _sha(d / "drop_pair.jsonl"),
        "dropped_count": 1,
        "dropped_distinct": 1,
    }
    (d / "dpo_u.gguf.train.json").write_text(json.dumps(dpo_u), encoding="utf-8")
    r_dpo = _run(
        runner_bin,
        ["attest", "unlearn", d / "dpo_u.gguf.train.json", "--parent-data", d / "pairs.jsonl",
         "--dropped", d / "drop_pair.jsonl", "--allow-same-binary"],
        env=_clean_env(),
    )
    _assert_no_sanitizer_or_crash(r_dpo)
    assert r_dpo.returncode == 2, (r_dpo.stdout, r_dpo.stderr)
    assert "VERIFIED" not in r_dpo.stdout

    # 4. attest check on a record with embedded NUL before ,"signature" must report BAD SIGNATURE (2), not UNSIGNED (1)
    nul_sig = d / "nul_sig.train.json"
    nul_sig.write_bytes(
        b'{"schema_version":"xyntetik.runner.train.v1"}\x00,"signature":{"algo":"ed25519","public_key":"'
        + b"0" * 64 + b'","sig":"' + b"0" * 128 + b'"}}'
    )
    r_ns = _run(runner_bin, ["attest", "check", nul_sig], env=_clean_env())
    _assert_no_sanitizer_or_crash(r_ns)
    assert r_ns.returncode == 2 and "UNSIGNED" not in r_ns.stdout, (r_ns.stdout, r_ns.stderr)


def test_lineage_rejects_relocate_traversal_and_symlinks(runner_bin, fx, tmp_path):
    """attest lineage must not escape the record directory via ../, absolute paths, or symlinks."""
    outside = tmp_path / "outside"
    shutil.copytree(fx, outside)
    raw = json.loads((outside / "plain.gguf.train.json").read_text(encoding="utf-8"))

    audit = tmp_path / "audit"
    audit.mkdir()
    for bad_base in (str(outside / "base.gguf"), "../outside/base.gguf"):
        mut = json.loads(json.dumps(raw))
        mut["base"]["path"] = bad_base
        mut["data"]["path"] = str(outside / "corpus.txt")
        rec = audit / "esc.train.json"
        rec.write_text(json.dumps(mut), encoding="utf-8")
        r = _run(runner_bin, ["attest", "lineage", rec], env=_clean_env())
        _assert_no_sanitizer_or_crash(r)
        assert r.returncode == 2 and "BROKEN" in r.stdout, (bad_base, r.returncode, r.stdout)

    sym_dir = tmp_path / "sym"
    sym_dir.mkdir()
    try:
        (sym_dir / "base.gguf").symlink_to(outside / "base.gguf")
        (sym_dir / "corpus.txt").symlink_to(outside / "corpus.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    (sym_dir / "sym.train.json").write_text(json.dumps(raw), encoding="utf-8")
    r_sym = _run(runner_bin, ["attest", "lineage", sym_dir / "sym.train.json"], env=_clean_env())
    _assert_no_sanitizer_or_crash(r_sym)
    assert r_sym.returncode == 2 and "BROKEN" in r_sym.stdout, (r_sym.returncode, r_sym.stdout)


def test_lineage_diamond_dag_is_not_rewalked_exponentially(runner_bin, tmp_path):
    """Shared ancestors in a multi-parent DAG must be memoized, not walked O(2^depth) times."""
    origin_bytes = b"origin"
    origin_sha = hashlib.sha256(origin_bytes).hexdigest()
    (tmp_path / "origin.gguf").write_bytes(origin_bytes)

    levels = 11
    prev_recs = []
    # Build bottom-up DAG of merge-like records linking two child records ("base" and "adapter")
    for lvl in range(levels, 0, -1):
        cur = []
        for idx in range(2):
            name = f"L{lvl}_{idx}.json"
            out_bytes = f"out_{lvl}_{idx}".encode()
            out_sha = hashlib.sha256(out_bytes).hexdigest()
            rec = {
                "schema_version": "xyntetik.runner.merge.v1",
                "runner": "test",
                "merged": {"path": f"m_{lvl}_{idx}.gguf", "sha256": out_sha},
            }
            if lvl == levels:
                rec["base"] = {"path": "origin.gguf", "sha256": origin_sha}
            else:
                rec["base"] = {
                    "path": prev_recs[0][1], "sha256": prev_recs[0][2],
                    "record": {"path": prev_recs[0][0], "sha256": prev_recs[0][3]},
                }
                rec["adapter"] = {
                    "path": prev_recs[1][1], "sha256": prev_recs[1][2],
                    "record": {"path": prev_recs[1][0], "sha256": prev_recs[1][3]},
                }
            raw = json.dumps(rec).encode()
            (tmp_path / name).write_bytes(raw)
            cur.append((name, f"m_{lvl}_{idx}.gguf", out_sha, hashlib.sha256(raw).hexdigest()))
            if lvl == 1:
                break
        prev_recs = cur

    r = _run(runner_bin, ["attest", "lineage", tmp_path / "L1_0.json"], env=_clean_env())
    _assert_no_sanitizer_or_crash(r)
    assert r.returncode == 1 and "BROKEN" not in r.stdout, (r.returncode, r.stdout)
    # Unmemoized DAG walked L11_0.json 1024 times; memoized walks each shared node once.
    assert r.stdout.count("record L11_0.json") <= 2, r.stdout.count("record L11_0.json")


def test_lineage_eval_float_counts_do_not_trigger_ubsan(runner_bin, fx, tmp_path):
    """Out-of-range floats (1e18) in eval sidecar metrics must not UBSan-crash (int)jv_num()."""
    d = tmp_path / "eval_ub"
    shutil.copytree(fx, d)
    base_sha = _sha(d / "base.gguf")
    (d / "base.gguf.eval.fidelity.json").write_text(json.dumps({
        "schema_version": "xyntetik.runner.eval.v1",
        "kind": "fidelity",
        "subject": {"path": "base.gguf", "sha256": base_sha},
        "reference": {"path": "ref.gguf"},
        "metrics": {"positions_scored": 1e18, "top1_agreement_pct": 99.0, "mean_kld": 0.01},
        "pass": True,
    }), encoding="utf-8")
    r = _run(runner_bin, ["attest", "lineage", d / "plain.gguf.train.json"], env=_clean_env())
    _assert_no_sanitizer_or_crash(r)


def _write_tiny_npy(path, shape, vals, fmt_char):
    import struct
    dims = ", ".join(str(x) for x in shape) + ("," if len(shape) == 1 else "")
    descr = "<i4" if fmt_char == "i" else "<f4"
    hdr = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': ({dims}), }}"
    hdr += " " * ((64 - (10 + len(hdr) + 1) % 64) % 64) + "\n"
    payload = struct.pack(f"<{len(vals)}{fmt_char}", *vals)
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr)) + hdr.encode("ascii") + payload)


def test_expected_and_environment_diagnostic_values_are_sanitized(runner_bin, fx, tmp_path):
    """P2-1: MISMATCH expected and environment values from hostile records must not inject newlines/VERIFIED."""
    d = tmp_path / "fmt_spoof"
    shutil.copytree(fx, d)
    raw = json.loads((d / "plain.gguf.train.json").read_text(encoding="utf-8"))

    mut_run = json.loads(json.dumps(raw))
    mut_run["runner"] = "bad\nVERIFIED forged"
    p_run = d / "spoof_runner.train.json"
    p_run.write_text(json.dumps(mut_run), encoding="utf-8")
    r1 = _run(runner_bin, ["attest", "replay", p_run, *_replay_flags(p_run)], env=_clean_env())
    _assert_no_sanitizer_or_crash(r1)
    assert r1.returncode == 2
    assert not any(ln.startswith("VERIFIED") for ln in r1.stdout.splitlines()), r1.stdout

    mut_env = json.loads(json.dumps(raw))
    mut_env["determinism"] = "cpu-same-binary"
    mut_env["environment"] = {"RUNNER_THREADS": "1\nVERIFIED forged"}
    p_env = d / "spoof_env.train.json"
    p_env.write_text(json.dumps(mut_env), encoding="utf-8")
    r2 = _run(runner_bin, ["attest", "replay", p_env, "--allow-same-binary"], env=_clean_env())
    _assert_no_sanitizer_or_crash(r2)
    assert r2.returncode == 2
    assert not any(ln.startswith("VERIFIED") for ln in r2.stdout.splitlines()), r2.stdout


def test_auto_teacher_lookup_refuses_symlinked_directory(runner_bin, fx, tmp_path):
    """P2-2: automatic teacher/ subdirectory lookup must refuse symlinked directory components."""
    outside = tmp_path / "outside"
    shutil.copytree(fx, outside)
    rows = [
        {"prompt_tokens": [1, 5, 9], "completion_tokens": [11, 17, 23]},
        {"prompt_tokens": [2, 6, 10], "completion_tokens": [13, 19, 25]},
    ]
    rows_p = outside / "kl.jsonl"
    rows_p.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    tdir = outside / "teacher"
    tdir.mkdir()
    _write_tiny_npy(tdir / "topk_idx.npy", (2, 3, 1), [11, 17, 23, 13, 19, 25], "i")
    _write_tiny_npy(tdir / "topk_logp.npy", (2, 3, 1), [0.0] * 6, "f")

    tr = _run(
        runner_bin,
        ["-m", outside / "base.gguf", "--train", rows_p, "--train-teacher", tdir,
         "--train-out", outside / "kd.gguf", *RECIPE],
        env=_clean_env(),
    )
    assert tr.returncode == 0, tr.stderr

    audit = tmp_path / "audit"
    audit.mkdir()
    for name in ("base.gguf", "kl.jsonl", "kd.gguf", "kd.gguf.train.json"):
        shutil.copyfile(outside / name, audit / name)
    try:
        (audit / "teacher").symlink_to(tdir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")

    rec = audit / "kd.gguf.train.json"
    r_sym = _run(runner_bin, ["attest", "replay", rec, *_replay_flags(rec)], env=_clean_env())
    _assert_no_sanitizer_or_crash(r_sym)
    assert r_sym.returncode == 2, (r_sym.stdout, r_sym.stderr)
    assert "VERIFIED" not in r_sym.stdout, r_sym.stdout

    # Explicit caller-supplied --teacher override must remain distinct and still work.
    r_exp = _run(
        runner_bin,
        ["attest", "replay", rec, "--teacher", tdir, *_replay_flags(rec)],
        env=_clean_env(),
    )
    _assert_no_sanitizer_or_crash(r_exp)
    assert r_exp.returncode == 0 and "VERIFIED" in r_exp.stdout, (r_exp.stdout, r_exp.stderr)


def test_unlearn_producer_validates_recipe_before_splitting_or_writing_retained(runner_bin, fx, tmp_path):
    """P2-3: --unlearn must refuse invalid parent recipe before reading DATA1/D or writing .retained.jsonl."""
    d = tmp_path / "unlearn_ord"
    shutil.copytree(fx, d)
    p_rec = d / "bad_parent.train.json"
    p_raw = json.loads((d / "parent.gguf.train.json").read_text(encoding="utf-8"))
    p_raw["steps"] = -1
    p_rec.write_text(json.dumps(p_raw), encoding="utf-8")

    out_model = d / "retrained.gguf"
    retained = d / "retrained.gguf.retained.jsonl"
    r = _run(
        runner_bin,
        ["--unlearn", p_rec, "--drop", d / "drop.jsonl",
         "-m", d / "base.gguf", "--train", d / "data1.jsonl", "--train-out", out_model],
        env=_clean_env(),
    )
    _assert_no_sanitizer_or_crash(r)
    assert r.returncode == 2 and "MISMATCH recipe" in r.stdout, (r.stdout, r.stderr)
    assert not retained.exists(), "refused --unlearn wrote retained file before recipe validation"
    assert not out_model.exists()


def test_lineage_memo_exhaustion_fails_closed_instead_of_rewalking(runner_bin, tmp_path):
    """P2-4: exhausting completed-node memo capacity must abort as BROKEN, not fall back to O(2^d)."""
    d = tmp_path / "memo_cap"
    d.mkdir()
    orig_sha = hashlib.sha256(b"origin").hexdigest()
    (d / "origin.gguf").write_bytes(b"origin")

    def mk_rec(name, out_name, base_obj, ad_obj=None):
        rec = {
            "schema_version": "xyntetik.runner.merge.v1",
            "runner": "test",
            "merged": {"path": out_name, "sha256": hashlib.sha256(out_name.encode()).hexdigest()},
            "base": base_obj,
        }
        if ad_obj:
            rec["adapter"] = ad_obj
        raw = json.dumps(rec).encode()
        (d / name).write_bytes(raw)
        return out_name, hashlib.sha256(out_name.encode()).hexdigest(), hashlib.sha256(raw).hexdigest()

    # Branch A: 4095 unique binary-tree nodes (depth 12 < 32), exceeding the 4096 memo capacity.
    def build_tree(prefix, depth):
        name = f"{prefix}.json"
        out_name = f"{prefix}.gguf"
        if depth == 0:
            return (name, *mk_rec(name, out_name, {"path": "origin.gguf", "sha256": orig_sha}))
        ln, lo, lso, lsr = build_tree(prefix + "L", depth - 1)
        rn, ro, rso, rsr = build_tree(prefix + "R", depth - 1)
        b = {"path": lo, "sha256": lso, "record": {"path": ln, "sha256": lsr}}
        a = {"path": ro, "sha256": rso, "record": {"path": rn, "sha256": rsr}}
        return (name, *mk_rec(name, out_name, b, a))

    # Build depth-11 binary tree = 4095 unique nodes.
    tn, to, tso, tsr = build_tree("T", 11)

    # Branch B: 14-level shared two-parent diamond after the 4095-node tree.
    prev = []
    for lvl in range(14, 0, -1):
        cur = []
        for idx in range(2):
            name = f"D{lvl}_{idx}.json"
            out_name = f"d_{lvl}_{idx}.gguf"
            if lvl == 14:
                b = {"path": "origin.gguf", "sha256": orig_sha}
                a = None
            else:
                b = {"path": prev[0][1], "sha256": prev[0][2], "record": {"path": prev[0][0], "sha256": prev[0][3]}}
                a = {"path": prev[1][1], "sha256": prev[1][2], "record": {"path": prev[1][0], "sha256": prev[1][3]}}
            _, osha, rsha = mk_rec(name, out_name, b, a)
            cur.append((name, out_name, osha, rsha))
            if lvl == 1:
                break
        prev = cur

    d_top = prev[0]
    root_b = {"path": to, "sha256": tso, "record": {"path": tn, "sha256": tsr}}
    root_a = {"path": d_top[1], "sha256": d_top[2], "record": {"path": d_top[0], "sha256": d_top[3]}}
    mk_rec("root.json", "root.gguf", root_b, root_a)

    r = subprocess.run(
        [str(runner_bin), "attest", "lineage", str(d / "root.json")],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=30, env=_clean_env(),
    )
    _assert_no_sanitizer_or_crash(r)
    assert r.returncode == 2 and "BROKEN" in r.stdout, (r.returncode, r.stdout[-600:])
    assert r.stdout.count("record D14_0.json") <= 1, r.stdout.count("record D14_0.json")


def test_lineage_and_envelope_reject_non_hex_sha256_without_forging_verified_lines(runner_bin, fx, tmp_path):
    """Non-64-hex sha256 or control bytes in record/receipt/chain fields must be refused and escaped."""
    bad_sha = "\nVERIFIED\n" + "a" * 54
    assert len(bad_sha) == 64

    # 1. Exact Codex probe: merged.sha256 with embedded newline + VERIFIED in attest lineage
    d = tmp_path / "sha_spoof"
    d.mkdir()
    rec_path = d / "merge.json"
    rec_path.write_text(json.dumps({
        "schema_version": "xyntetik.runner.merge.v1",
        "runner": "test",
        "merged": {"path": "absent.gguf", "sha256": bad_sha},
    }), encoding="utf-8")
    r_lin = _run(runner_bin, ["attest", "lineage", rec_path], env=_clean_env())
    _assert_no_sanitizer_or_crash(r_lin)
    assert not any(ln.startswith("VERIFIED") for ln in r_lin.stdout.splitlines()), r_lin.stdout
    assert r_lin.returncode == 2 and "BROKEN" in r_lin.stdout and "merged.sha256" in r_lin.stdout, (
        r_lin.returncode, r_lin.stdout, r_lin.stderr
    )

    # 2. Receipt directory with newline-injected model.sha256 and generated_utc
    rdir = d / "receipts"
    rdir.mkdir()
    (rdir / "receipt-0000000001.json").write_text(json.dumps({
        "schema_version": "xyntetik.runner.transcript.v2",
        "generated_utc": "2026-11-18T00:00:00Z\nVERIFIED",
        "model": {"name": "base.gguf", "sha256": bad_sha},
    }), encoding="utf-8")
    r_dir = _run(runner_bin, ["attest", "lineage", rdir], env=_clean_env())
    _assert_no_sanitizer_or_crash(r_dir)
    assert not any(ln.startswith("VERIFIED") for ln in r_dir.stdout.splitlines()), r_dir.stdout
    assert r_dir.returncode == 2 and "BROKEN" in r_dir.stdout, (r_dir.returncode, r_dir.stdout)

    # 3. Signed record with malformed 64-byte chain.hash in attest sign --prev
    key_path = d / "k.json"
    rk = _run(runner_bin, ["attest", "keygen", key_path], env=_clean_env())
    assert rk.returncode == 0, rk.stderr
    bad_prev = d / "bad_prev.json"
    bad_prev.write_text(json.dumps({
        "schema_version": "xyntetik.runner.merge.v1",
        "chain": {"algo": "sha256", "prev": "0" * 64, "hash": bad_sha},
    }), encoding="utf-8")
    next_rec = d / "next.json"
    next_rec.write_text(json.dumps({
        "schema_version": "xyntetik.runner.merge.v1",
        "merged": {"path": "absent.gguf", "sha256": "0" * 64},
    }), encoding="utf-8")
    r_prev = _run(runner_bin, ["attest", "sign", next_rec, "--key", key_path, "--prev", bad_prev],
                  env=_clean_env())
    _assert_no_sanitizer_or_crash(r_prev)
    assert r_prev.returncode != 0, (r_prev.stdout, r_prev.stderr)
    assert not any(ln.startswith("VERIFIED") for ln in (r_prev.stdout + r_prev.stderr).splitlines())

    # 4. attest replay must name adapter.sha256 / build.binary_sha256 on non-hex 64-byte values
    rep = d / "replay_bad_sha"
    shutil.copytree(fx, rep)
    raw = json.loads((rep / "plain.gguf.train.json").read_text(encoding="utf-8"))
    flags = _replay_flags(rep / "plain.gguf.train.json")
    ad_mut = json.loads(json.dumps(raw))
    ad_mut["adapter"]["sha256"] = bad_sha
    (rep / "bad_ad_sha.train.json").write_text(json.dumps(ad_mut), encoding="utf-8")
    r_ad = _run(runner_bin, ["attest", "replay", rep / "bad_ad_sha.train.json", *flags], env=_clean_env())
    _assert_no_sanitizer_or_crash(r_ad)
    assert not any(ln.startswith("VERIFIED") for ln in r_ad.stdout.splitlines()), r_ad.stdout
    assert r_ad.returncode == 2 and "MISMATCH adapter.sha256:" in r_ad.stdout, (r_ad.stdout, r_ad.stderr)

    bld_mut = json.loads(json.dumps(raw))
    bld_mut["build"]["binary_sha256"] = bad_sha
    (rep / "bad_bld_sha.train.json").write_text(json.dumps(bld_mut), encoding="utf-8")
    r_bld = _run(runner_bin, ["attest", "replay", rep / "bad_bld_sha.train.json", *flags], env=_clean_env())
    _assert_no_sanitizer_or_crash(r_bld)
    assert not any(ln.startswith("VERIFIED") for ln in r_bld.stdout.splitlines()), r_bld.stdout
    assert r_bld.returncode == 2 and "MISMATCH build.binary_sha256:" in r_bld.stdout, (r_bld.stdout, r_bld.stderr)


def test_diagnostic_sanitizers_escape_c1_bidi_and_raw_high_bytes(runner_bin, fx, tmp_path):
    """Bytes outside printable ASCII (0x20..0x7e) — C1 controls, bidi overrides, lone 0xff — escape as \\xNN."""
    d = tmp_path / "esc_high"
    shutil.copytree(fx, d)

    def run_bytes(args):
        return subprocess.run(
            [str(runner_bin), *map(str, args)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            env=_clean_env(),
        )

    # 1. Claude's C1 CSI probe (\u009b2J -> C2 9B 32 4A) in --unlearn drop line and attest lineage base.path
    drop_c1 = d / "drop_c1.jsonl"
    drop_c1.write_bytes(b'{"prompt":"\xc2\x9b2J\xc2\x9b31mFAKE"}\n')
    r_c1 = run_bytes([
        "--unlearn", d / "parent.gguf.train.json", "--drop", drop_c1,
        "-m", d / "base.gguf", "--train", d / "data1.jsonl", "--train-out", d / "u_c1.gguf",
        "-t", "1", "--gpu", "off",
    ])
    assert r_c1.returncode != 0
    assert b"\xc2\x9b2J" not in r_c1.stderr and b"\xc2\x9b" not in r_c1.stdout, (r_c1.stdout, r_c1.stderr)
    assert b"\\xc2\\x9b" in r_c1.stderr, r_c1.stderr

    lin_c1 = d / "lin_c1.json"
    lin_c1.write_bytes(json.dumps({
        "schema_version": "xyntetik.runner.merge.v1",
        "runner": "test",
        "merged": {"path": "absent.gguf", "sha256": "0" * 64},
        "base": {"path": "\u009b2Jbad", "sha256": "1" * 64},
    }).encode("utf-8"))
    r_lc1 = run_bytes(["attest", "lineage", lin_c1])
    assert r_lc1.returncode == 1
    assert b"\xc2\x9b2J" not in r_lc1.stdout and b"\xc2\x9b" not in r_lc1.stderr, (r_lc1.stdout, r_lc1.stderr)
    assert b"\\xc2\\x9b" in r_lc1.stdout, r_lc1.stdout

    # 2. Bidi override (U+202E -> E2 80 AE) in record field & envelope sidecar schema
    rec = json.loads((d / "plain.gguf.train.json").read_text(encoding="utf-8"))
    rec["runner"] = "safe\u202eevil"
    p_bidi = d / "bidi.train.json"
    p_bidi.write_text(json.dumps(rec), encoding="utf-8")
    r_bidi = run_bytes(["attest", "replay", p_bidi, *_replay_flags(p_bidi)])
    assert r_bidi.returncode == 2
    assert b"\xe2\x80\xae" not in r_bidi.stdout and b"\xe2\x80\xae" not in r_bidi.stderr
    assert b"\\xe2\\x80\\xae" in r_bidi.stdout, r_bidi.stdout

    env_mdl = d / "env_probe.gguf"
    shutil.copyfile(d / "base.gguf", env_mdl)
    (d / "env_probe.gguf.envelope.json").write_bytes(
        b'{"schema_version":"bad\\n\xc2\x9b2J\xe2\x80\xae"}'
    )
    r_env = run_bytes(["-m", env_mdl, "-p", "hi", "-n", "1", "--gpu", "off"])
    assert b"\xc2\x9b" not in r_env.stderr and b"\xe2\x80\xae" not in r_env.stderr, r_env.stderr

    # 3. Lone invalid UTF-8 byte 0xff in --unlearn drop line (and backslash doubling)
    drop_ff = d / "drop_ff.jsonl"
    drop_ff.write_bytes(b'{"prompt":"x\\y\xff"}\n')
    r_ff = run_bytes([
        "--unlearn", d / "parent.gguf.train.json", "--drop", drop_ff,
        "-m", d / "base.gguf", "--train", d / "data1.jsonl", "--train-out", d / "u_ff.gguf",
        "-t", "1", "--gpu", "off",
    ])
    assert r_ff.returncode != 0
    assert b"\xff" not in r_ff.stderr and b"\xff" not in r_ff.stdout, (r_ff.stdout, r_ff.stderr)
    assert b"\\xff" in r_ff.stderr and b"\\\\" in r_ff.stderr, r_ff.stderr



