"""--train: the D4 loop under the D5 gates.

The headline property is DETERMINISTIC TRAINING: same data + same seed ->
byte-identical adapter GGUF, twice over. Everything else is the loop doing
its job: loss falls on an overfit corpus, the saved adapter loads back
through --lora and measurably improves --score on the trained text, the
JSONL prompt/completion mode masks the prompt from the loss, and a different
seed produces a different adapter (the determinism is seeded, not vacuous).
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ("the runner trains the adapter and the adapter learns the corpus. "
          * 3)
SENT = "the runner trains the adapter and the adapter learns the corpus."


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("train")
    b = d / "base.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(b)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    (d / "corpus.txt").write_text(CORPUS)
    (d / "data.jsonl").write_text(
        json.dumps({"prompt": "the runner trains",
                    "completion": " the adapter and the adapter learns",
                    "weight": 1.0}) + "\n" +
        json.dumps({"prompt": "the corpus",
                    "completion": " is learned by the adapter",
                    "weight": 0.5}) + "\n")
    return d


def _train(runner_bin, base_dir, out, data="corpus.txt", steps=20, extra=()):
    cmd = [runner_bin, "-m", str(base_dir / "base.gguf"),
           "--train", str(base_dir / data), "--train-steps", str(steps),
           "--lr", "3e-3", "--train-out", str(out), "-t", "2", *extra]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=600)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    losses = [json.loads(l)["loss"] for l in p.stdout.splitlines() if l]
    return losses


def _run_train(runner_bin, base_dir, out, data, extra=()):
    cmd = [runner_bin, "-m", str(base_dir / "base.gguf"),
           "--train", str(base_dir / data), "--train-steps", "1",
           "--lr", "3e-3", "--train-out", str(out), "-t", "2", *extra]
    return subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=600)


def _score(runner_bin, base_dir, lora=None):
    cmd = [runner_bin, "-m", str(base_dir / "base.gguf"), "--score",
           "-p", SENT, "-t", "2", "--gpu", "off"]
    if lora:
        cmd += ["--lora", str(lora)]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(p.stdout)["nll_mean"]


def test_training_is_byte_deterministic(runner_bin, base, tmp_path):
    a1, a2 = tmp_path / "a1.gguf", tmp_path / "a2.gguf"
    l1 = _train(runner_bin, base, a1)
    l2 = _train(runner_bin, base, a2)
    assert l1 == l2, "loss trajectories differ between identical runs"
    assert a1.read_bytes() == a2.read_bytes(), \
        "adapter files differ between identical runs"
    # D7: the provenance records agree on the adapter sha (and carry the
    # base/data shas that make the reproducibility claim checkable)
    r1 = json.loads((tmp_path / "a1.gguf.train.json").read_text())
    r2 = json.loads((tmp_path / "a2.gguf.train.json").read_text())
    assert r1["adapter"]["sha256"] == r2["adapter"]["sha256"]
    assert len(r1["base"]["sha256"]) == 64 and len(r1["data"]["sha256"]) == 64
    # build identity: same executable must be distinguishable from same
    # source in any reproduction report (external rebuild study, 2026-08)
    assert len(r1["build"]["binary_sha256"]) == 64
    assert r1["build"]["compiler"] and r1["build"]["os"] and r1["build"]["arch"]


# Every family the trainer admits trains reproducibly, not only the plain
# fixture above: the backward pass branches by family (qk-norm, Gemma's norms
# and softcap, the E-series per-layer embedding, Muse, Granite's multipliers,
# a quantized base, sparse-MoE attention adapters with and without router
# renormalization), and a family whose two identical runs differ would make
# every comparison of its adapters meaningless.
TRAIN_FAMILIES = {
    "qk-norm": ("make-test-model.py", ["--qk-norm"], None),
    "gemma3": ("make-test-model.py", ["--gemma3"], None),
    "gemma4": ("make-test-model.py", ["--arch", "gemma4"], None),
    "e-series": ("make-test-model.py", ["--eseries", "3,8", "--drop-kv", "shared"], None),
    "muse": ("make-test-model.py", ["--muse-glimmer"], None),
    "granite": ("make-test-model.py", ["--granite"], None),
    "q8_0": ("make-test-model.py", ["--quant", "q8_0"], None),
    "partial-rope": ("make-test-model.py", ["--arch", "qwen3", "--rope-dim", "8"], None),
    "qwen35": ("make-test-ornith.py", [], None),
    "clef": ("make-test-clef.py", [], None),
    "moe-renorm": ("make-test-moe.py", [], "moe4-train"),
    "moe-no-renorm": ("make-test-moe.py", [], "moe4-train-nonorm"),
}


def _family_fixture(tmp_path, family):
    gen, args, member = TRAIN_FAMILIES[family]
    d = tmp_path / "fx"
    d.mkdir()
    if member:
        subprocess.run([sys.executable, ROOT / "scripts" / gen, str(d / "m")],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        (d / f"m.{member}.gguf").rename(d / "base.gguf")
    else:
        subprocess.run([sys.executable, ROOT / "scripts" / gen, *args, str(d / "base.gguf")],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    (d / "corpus.txt").write_text(CORPUS)
    return d


@pytest.mark.parametrize("family", sorted(TRAIN_FAMILIES))
def test_every_trainable_family_is_byte_deterministic(runner_bin, base, tmp_path, family):
    d = _family_fixture(tmp_path, family)
    a1, a2 = tmp_path / "a1.gguf", tmp_path / "a2.gguf"
    l1 = _train(runner_bin, d, a1, steps=4)
    l2 = _train(runner_bin, d, a2, steps=4)
    assert l1 == l2, (family, "loss trajectories differ between identical runs")
    assert a1.read_bytes() == a2.read_bytes(), (family, "adapter files differ")


# R8.19: the taped forward runs in rounds of 16 positions, every product
# column a solo decode step's bits; RUNNER_TRAIN_ROWS=N sets the width, 1 a
# position at a time. Every width must train the same adapter byte for byte
# on every family, with an adapter in the forward from step 2 on.
@pytest.mark.parametrize("family", ["base"] + sorted(TRAIN_FAMILIES))
def test_batched_taped_forward_is_the_solo_bytes(runner_bin, base, tmp_path, family):
    d = base if family == "base" else _family_fixture(tmp_path, family)
    runs = []
    for rows in ("1", "5", None):
        out = tmp_path / ("rows-%s.gguf" % (rows or "default"))
        env = dict(os.environ)
        if rows:
            os.environ["RUNNER_TRAIN_ROWS"] = rows
        try:
            losses = _train(runner_bin, d, out, steps=4)
        finally:
            os.environ.clear()
            os.environ.update(env)
        runs.append((rows, losses, out.read_bytes()))
    for rows, losses, adapter in runs[1:]:
        assert losses == runs[0][1], (family, rows, runs[0][1], losses)
        assert adapter == runs[0][2], (family, rows, "adapter bytes differ")


# The manual's `-t N` row: "the adapter bytes do not depend on it". Every
# pooled pass of a step (the products, the adapter's own rank-r math since
# R8.19, the DeltaNet heads) keeps each element's chain whatever the split.
@pytest.mark.parametrize("family", ["base", "gemma3", "e-series", "qwen35", "clef", "moe-renorm"])
def test_adapter_bytes_do_not_depend_on_the_thread_count(runner_bin, base, tmp_path, family):
    d = base if family == "base" else _family_fixture(tmp_path, family)
    a1, a3 = tmp_path / "t1.gguf", tmp_path / "t3.gguf"
    l1 = _train(runner_bin, d, a1, steps=4, extra=("-t", "1"))
    l3 = _train(runner_bin, d, a3, steps=4, extra=("-t", "3"))
    assert l1 == l3, (family, l1, l3)
    assert a1.read_bytes() == a3.read_bytes(), (family, "adapter bytes depend on -t")


# R8.19 --train-gpu: every weight product of a step on CUDA. Not the CPU's
# bytes (the device products add in another order) but the same training
# within rounding: 4 steps a family against the CPU run, every step's loss
# within 1e-4 relative, and the record names the device. Skips where no CUDA
# device is usable (the error it gives there is checked below).
@pytest.mark.parametrize("family", ["base", "q8_0", "gemma4", "qwen35", "clef", "moe-renorm"])
def test_train_gpu_matches_cpu_within_rounding(runner_bin, base, tmp_path, family):
    d = base if family == "base" else _family_fixture(tmp_path, family)
    runs = {}
    for arm in ("cpu", "gpu"):
        out = tmp_path / ("%s.gguf" % arm)
        p = _run_train(runner_bin, d, out, "corpus.txt",
                       extra=("--train-steps", "4") + (("--train-gpu",) if arm == "gpu" else ()))
        err = p.stderr.decode(errors="replace")
        if arm == "gpu" and p.returncode != 0 and "needs a CUDA build" in err:
            pytest.skip("no usable CUDA device")
        assert p.returncode == 0, err
        runs[arm] = ([json.loads(l)["loss"] for l in p.stdout.splitlines() if l], out)
    (lc, _), (lg, ag) = runs["cpu"], runs["gpu"]
    assert len(lc) == len(lg) == 4
    for a, b in zip(lc, lg):
        assert abs(a - b) <= 1e-4 * max(abs(a), 1e-6), (family, lc, lg)
    rec = json.loads(pathlib.Path(str(ag) + ".train.json").read_text())
    assert rec["products"]["device"] == "cuda", rec.get("products")


def test_train_gpu_is_refused_without_training_or_a_device(runner_bin, base, tmp_path):
    p = subprocess.run([runner_bin, "-m", str(base / "base.gguf"), "--train-gpu",
                        "-p", "hi", "-n", "1"], cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    assert p.returncode != 0
    assert b"--train-gpu runs a --train or --train-dpo step" in p.stderr
    out = tmp_path / "ad.gguf"
    p = _run_train(runner_bin, base, out, "corpus.txt", extra=("--train-gpu",))
    if p.returncode == 0:
        pytest.skip("a CUDA device is usable here")
    assert b"--train-gpu needs a CUDA build and an NVIDIA GPU" in p.stderr
    assert not out.exists(), "a refused run wrote an adapter"


def test_loss_falls_and_adapter_improves_score(runner_bin, base, tmp_path):
    out = tmp_path / "ad.gguf"
    losses = _train(runner_bin, base, out, steps=30)
    assert losses[-1] < losses[0] * 0.95, (losses[0], losses[-1])
    base_nll = _score(runner_bin, base)
    tuned_nll = _score(runner_bin, base, lora=out)
    assert tuned_nll < base_nll - 0.2, (base_nll, tuned_nll)


# R8.16: training through the Gated DeltaNet recurrence learns, end to end:
# the loss falls on the corpus and the saved adapter, loaded back through
# --lora (the serving hooks on all five DeltaNet projections, the fused Q/gate
# and the FFN), scores the trained text better than the bare base.
@pytest.mark.parametrize("family", ["qwen35", "clef"])
def test_deltanet_training_learns_and_serves(runner_bin, tmp_path, family):
    d = _family_fixture(tmp_path, family)
    out = tmp_path / "ad.gguf"
    losses = _train(runner_bin, d, out, steps=30)
    assert losses[-1] < losses[0] * 0.95, (losses[0], losses[-1])
    assert _score(runner_bin, d, lora=out) < _score(runner_bin, d) - 0.2


# --train-dpo end to end: run_train's DPO step loop, its per-step fields, the
# save and the record naming the objective. The objective and its gradient
# are gated numerically by tests/test_dpo_grad.c; this pins the CLI around
# them, which nothing else ran.
DPO_FIELDS = ("step", "pair", "tokens", "loss", "margin", "acc", "coeff",
              "logp_w", "logp_l", "logp_ref_w", "logp_ref_l")


def _train_dpo(runner_bin, base_dir, data, out, steps=4):
    cmd = [runner_bin, "-m", str(base_dir / "base.gguf"),
           "--train-dpo", str(data), "--train-steps", str(steps),
           "--lr", "3e-3", "--train-out", str(out), "-t", "2", "--gpu", "off"]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=600)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return [json.loads(l) for l in p.stdout.splitlines() if l]


def test_dpo_training_is_byte_deterministic_and_recorded(runner_bin, base, tmp_path):
    data = tmp_path / "pairs.jsonl"
    data.write_text(
        json.dumps({"prompt": "the runner trains", "chosen": " the adapter",
                    "rejected": " the corpus"}) + "\n" +
        json.dumps({"prompt": "the adapter learns", "chosen": " the corpus",
                    "rejected": " the runner"}) + "\n")
    a1, a2 = tmp_path / "d1.gguf", tmp_path / "d2.gguf"
    s1 = _train_dpo(runner_bin, base, data, a1)
    s2 = _train_dpo(runner_bin, base, data, a2)
    assert [{k: s[k] for k in DPO_FIELDS} for s in s1] == \
        [{k: s[k] for k in DPO_FIELDS} for s in s2]
    assert a1.read_bytes() == a2.read_bytes()
    assert [s["step"] for s in s1] == [1, 2, 3, 4]
    assert [s["pair"] for s in s1] == [0, 1, 0, 1]
    # the first step's adapter is the base (B = 0): the policy is exactly
    # the reference the bypass measures
    assert s1[0]["margin"] == 0.0
    assert s1[0]["logp_w"] == s1[0]["logp_ref_w"]
    assert s1[0]["logp_l"] == s1[0]["logp_ref_l"]
    rec = json.loads((tmp_path / "d1.gguf.train.json").read_text())
    assert rec["objective"] == "dpo"
    assert rec["dpo_beta"] == pytest.approx(0.1)
    assert rec["steps"] == 4
    assert rec["adamw"] == {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
                            "weight_decay": 0.01}
    assert rec["loss_first"] == s1[0]["loss"]
    assert rec["loss_last"] == s1[-1]["loss"]
    assert rec["adapter"]["sha256"] == hashlib.sha256(a1.read_bytes()).hexdigest()
    # and the adapter serves
    _score(runner_bin, base, lora=a1)


def test_jsonl_mode_trains_with_prompt_masking(runner_bin, base, tmp_path):
    out = tmp_path / "aj.gguf"
    losses = _train(runner_bin, base, out, data="data.jsonl", steps=12)
    assert len(losses) == 12
    assert losses[-1] < losses[0]
    # round-trips through --lora
    _score(runner_bin, base, lora=out)


def test_provenance_record_escapes_data_and_output_paths(
        runner_bin, base, tmp_path):
    # Quotes force JSON escaping on POSIX; Windows forbids them in file names,
    # but its path separators exercise the backslash escaping instead.
    data_name = ('corpus-windows.txt' if os.name == "nt"
                 else 'corpus"quoted.txt')
    (base / data_name).write_text(CORPUS)
    out_name = ('adapter-windows.gguf' if os.name == "nt"
                else 'adapter"quoted.gguf')
    out = tmp_path / out_name
    _train(runner_bin, base, out, data=data_name, steps=1)
    record = pathlib.Path(f"{out}.train.json")
    rec = json.loads(record.read_text())
    assert rec["data"]["path"] == str(base / data_name)
    assert rec["adapter"]["path"] == str(out)


def test_provenance_hashes_binary_when_invoked_through_path(
        runner_bin, base, tmp_path):
    out = tmp_path / "path-invoked.gguf"
    env = dict(os.environ)
    env["PATH"] = str(runner_bin.parent) + os.pathsep + env.get("PATH", "")
    cmd = [runner_bin.name, "-m", str(base / "base.gguf"),
           "--train", str(base / "corpus.txt"), "--train-steps", "1",
           "--lr", "3e-3", "--train-out", str(out), "-t", "2"]
    p = subprocess.run(cmd, cwd=tmp_path, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=600, env=env)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    rec = json.loads(pathlib.Path(f"{out}.train.json").read_text())
    assert rec["build"]["binary_sha256"] == hashlib.sha256(
        runner_bin.read_bytes()).hexdigest()


@pytest.mark.parametrize("record", [
    '{"prompt":"p","completion":"c","weight":"heavy"}\n',
    "{\n",
])
def test_jsonl_rejects_malformed_examples(runner_bin, base, tmp_path, record):
    (base / "invalid.jsonl").write_text(record)
    out = tmp_path / "invalid.gguf"
    p = _run_train(runner_bin, base, out, "invalid.jsonl")
    assert p.returncode != 0
    assert not out.exists()


def test_jsonl_honors_train_context_without_truncating(
        runner_bin, base, tmp_path):
    record = {"prompt": "the runner", "completion": " trains the adapter " * 8}
    (base / "too-long.jsonl").write_text(json.dumps(record) + "\n")
    out = tmp_path / "too-long.gguf"
    p = _run_train(runner_bin, base, out, "too-long.jsonl",
                   extra=("--train-ctx", "4"))
    assert p.returncode != 0
    assert b"--train-ctx" in p.stderr
    assert not out.exists()


def test_failed_checkpoint_install_preserves_previous_adapter(
        runner_bin, base, tmp_path):
    out = tmp_path / "checkpoint.gguf"
    sentinel = b"previous valid checkpoint"
    out.write_bytes(sentinel)
    env = dict(os.environ, RUNNER_LORA_INSTALL_FAIL="1")
    cmd = [runner_bin, "-m", str(base / "base.gguf"),
           "--train", str(base / "corpus.txt"), "--train-steps", "1",
           "--lr", "3e-3", "--train-out", str(out), "-t", "2"]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=600, env=env)
    assert p.returncode != 0
    assert out.read_bytes() == sentinel
    assert not pathlib.Path(f"{out}.partial").exists()


def test_optimizer_allocation_failure_does_not_publish_adapter(
        runner_bin, base, tmp_path):
    out = tmp_path / "optimizer-oom.gguf"
    env = dict(os.environ, RUNNER_LORA_ADAM_ALLOC_FAIL="1")
    cmd = [runner_bin, "-m", str(base / "base.gguf"),
           "--train", str(base / "corpus.txt"), "--train-steps", "1",
           "--lr", "3e-3", "--train-out", str(out), "-t", "2"]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=600, env=env)
    assert p.returncode != 0
    assert b"optimizer state" in p.stderr
    assert not out.exists()
    assert not pathlib.Path(f"{out}.train.json").exists()


def test_gpu_training_matches_cpu_bytes(runner_bin, base, tmp_path):
    """D8 slice 2: RUNNER_TRAIN_GPU=1 must not change a single adapter byte.

    The device transposed-matvec is gated bit-identical to the CPU chain
    (test_mvt.c), so the whole training run must be too. Skips unless a
    CUDA device actually engaged (the runner prints its train-gpu banner).
    """
    import os
    a_cpu, a_gpu = tmp_path / "cpu.gguf", tmp_path / "gpu.gguf"
    _train(runner_bin, base, a_cpu, steps=6)
    env = dict(os.environ, RUNNER_TRAIN_GPU="1")
    cmd = [runner_bin, "-m", str(base / "base.gguf"),
           "--train", str(base / "corpus.txt"), "--train-steps", "6",
           "--lr", "3e-3", "--train-out", str(a_gpu), "-t", "2"]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=600, env=env)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    if b"train-gpu:" not in p.stderr or b"no usable CUDA" in p.stderr:
        pytest.skip("no CUDA device engaged")
    assert a_cpu.read_bytes() == a_gpu.read_bytes(), \
        "GPU-assisted training changed the adapter bytes"


# The MoE adapters' backward runs every routed expert's down projection
# through the transposed matvec, each expert as a view of the fused expert
# tensor. With RUNNER_TRAIN_GPU=1 those views go to the device, so the device
# must tell one expert's bytes from another's: a cache that knew a tensor by
# its descriptor's address reused the first expert's weights for every later
# one, and the adapter came out different from the CPU's.
@pytest.mark.parametrize("family", ["moe-renorm", "moe-no-renorm"])
def test_gpu_training_matches_cpu_bytes_on_moe_experts(runner_bin, base, tmp_path, family):
    import os
    d = _family_fixture(tmp_path, family)
    a_cpu, a_gpu = tmp_path / "cpu.gguf", tmp_path / "gpu.gguf"
    _train(runner_bin, d, a_cpu, steps=3, extra=("--gpu", "off"))
    env = dict(os.environ, RUNNER_TRAIN_GPU="1")
    cmd = [runner_bin, "-m", str(d / "base.gguf"),
           "--train", str(d / "corpus.txt"), "--train-steps", "3",
           "--lr", "3e-3", "--train-out", str(a_gpu), "-t", "2", "--gpu", "off"]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=600, env=env)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    if b"train-gpu:" not in p.stderr or b"no usable CUDA" in p.stderr:
        pytest.skip("no CUDA device engaged")
    assert a_cpu.read_bytes() == a_gpu.read_bytes(), \
        (family, "GPU-assisted training changed the MoE adapter's bytes")


# R8.16: the backward through Gated DeltaNet runs the transposed products of
# the five DeltaNet projections (and the fused Q/gate one) through the same
# matvec_t the assist moves to the device: the adapter must stay the CPU's.
@pytest.mark.parametrize("family", ["qwen35", "clef"])
def test_gpu_training_matches_cpu_bytes_through_deltanet(runner_bin, base, tmp_path, family):
    import os
    d = _family_fixture(tmp_path, family)
    a_cpu, a_gpu = tmp_path / "cpu.gguf", tmp_path / "gpu.gguf"
    _train(runner_bin, d, a_cpu, steps=3, extra=("--gpu", "off"))
    cmd = [runner_bin, "-m", str(d / "base.gguf"),
           "--train", str(d / "corpus.txt"), "--train-steps", "3",
           "--lr", "3e-3", "--train-out", str(a_gpu), "-t", "2", "--gpu", "off"]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=600, env=dict(os.environ, RUNNER_TRAIN_GPU="1"))
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    if b"train-gpu:" not in p.stderr or b"no usable CUDA" in p.stderr:
        pytest.skip("no CUDA device engaged")
    assert a_cpu.read_bytes() == a_gpu.read_bytes(), \
        (family, "GPU-assisted training changed the adapter's bytes")


def test_seed_changes_the_adapter(runner_bin, base, tmp_path):
    a1, a2 = tmp_path / "s1.gguf", tmp_path / "s2.gguf"
    _train(runner_bin, base, a1, steps=4, extra=("-s", "7"))
    _train(runner_bin, base, a2, steps=4, extra=("-s", "8"))
    assert a1.read_bytes() != a2.read_bytes()


def test_end_of_turn_token_is_opt_in_and_recorded(runner_bin, base, tmp_path):
    """--train-eot appends the family's turn terminator (the llama fixture
    declares no ChatML/Llama-3 terminator, so it is the EOS) as one more
    completion target on every line; a jsonl line can ask for it alone;
    the provenance record says which id was used."""
    def tokens_of(extra, data="data.jsonl"):
        p = _run_train(runner_bin, base, tmp_path / "eot.gguf", data, extra=extra)
        assert p.returncode == 0, p.stderr.decode(errors="replace")
        return [json.loads(l)["tokens"] for l in p.stdout.splitlines() if l.startswith(b"{")]
    plain = tokens_of(())
    with_eot = tokens_of(("--train-eot",))
    assert with_eot[0] == plain[0] + 1
    rec = json.loads((tmp_path / "eot.gguf.train.json").read_text(encoding="utf-8"))
    assert rec["end_of_turn"] is True and isinstance(rec["eot_id"], int) and rec["eot_id"] >= 0
    (base / "eot-line.jsonl").write_text(
        json.dumps({"prompt": "the runner trains", "completion": " the adapter", "end_of_turn": True}) + "\n",
        encoding="utf-8")
    (base / "plain-line.jsonl").write_text(
        json.dumps({"prompt": "the runner trains", "completion": " the adapter"}) + "\n", encoding="utf-8")
    assert tokens_of((), "eot-line.jsonl")[0] == tokens_of((), "plain-line.jsonl")[0] + 1
    rec = json.loads((tmp_path / "eot.gguf.train.json").read_text(encoding="utf-8"))
    assert rec["end_of_turn"] is False, "the flag was not given; the line asked"
    (base / "bad-line.jsonl").write_text(
        json.dumps({"prompt": "x", "completion": " y", "end_of_turn": "yes"}) + "\n", encoding="utf-8")
    p = _run_train(runner_bin, base, tmp_path / "bad.gguf", "bad-line.jsonl")
    assert p.returncode != 0 and b"end_of_turn must be true or false" in p.stderr


def test_text_corpus_is_not_truncated_by_its_byte_count(runner_bin, base, tmp_path):
    """The fixture's vocabulary spells a space as three byte-fallback tokens,
    so a text of one-letter words tokenizes to about twice its byte count.
    The text branch sized its buffer at bytes + 8 and tok_encode stops
    silently at its cap, so most of such a corpus never reached the
    optimizer. The jsonl branch regrows until the encoding fits; the text
    branch must too. --score on the same text is the independent count."""
    text = " ".join("abcdefghijklmnopqrst")
    (base / "short-words.txt").write_text(text, encoding="utf-8")
    p = subprocess.run([runner_bin, "-m", str(base / "base.gguf"), "--score",
                        "-p", text, "-t", "2", "--gpu", "off"], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    n_tokens = json.loads(p.stdout)["n_tokens"]
    assert n_tokens > len(text.encode()) + 8, "the probe must exceed the old cap"
    q = _run_train(runner_bin, base, tmp_path / "words.gguf", "short-words.txt",
                   extra=("--train-ctx", "128"))
    assert q.returncode == 0, q.stderr.decode(errors="replace")
    steps = [json.loads(l)["tokens"] for l in q.stdout.splitlines() if l.startswith(b"{")]
    assert steps[0] == n_tokens, (steps[0], n_tokens)


def test_small_one_answer_data_warns_before_training(runner_bin, base, tmp_path):
    """The shape of the 2026-10-16 report: a handful of examples that all
    share one answer, many passes, a raised lr, no end-of-turn token. Each
    is named before the first step, and training still runs."""
    # short, as a one-line answer is (the fixture's byte-level vocabulary
    # spells a word as several tokens)
    lines = [json.dumps({"prompt": f"question {i}?", "completion": " seven"})
             for i in range(4)]
    (base / "one-answer.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [runner_bin, "-m", str(base / "base.gguf"), "--train", str(base / "one-answer.jsonl"),
           "--train-steps", "13", "--lr", "5e-4", "--train-out", str(tmp_path / "w.gguf"), "-t", "2"]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    err = p.stderr.decode()
    assert "13 steps over 4 examples is 3.2 passes over the data" in err
    assert "4 of 4 completions are the same text" in err
    assert "do not end with an end-of-turn token" in err
    assert "--lr 0.0005 is 5x the default 0.0001" in err
    assert (tmp_path / "w.gguf").exists()


def test_documented_use_trains_without_a_warning(runner_bin, base, tmp_path):
    """Distinct answers, one pass, the default lr and an end-of-turn token:
    nothing to warn about."""
    lines = [json.dumps({"prompt": f"question {i}?", "completion": f" answer number {i}"})
             for i in range(8)]
    (base / "varied.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [runner_bin, "-m", str(base / "base.gguf"), "--train", str(base / "varied.jsonl"),
           "--train-steps", "8", "--train-eot", "--train-out", str(tmp_path / "q.gguf"), "-t", "2"]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert b"train: warning" not in p.stderr, p.stderr.decode(errors="replace")


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def test_continued_training_records_the_adapter_it_started_from(runner_bin, base, tmp_path):
    """--train with --lora continues an adapter (train-grpo-lite.py does it
    every round). The record names that adapter as an input, with its own
    record, so --lineage walks back through every earlier run, and gives the
    rank of the adapter actually trained, not the --lora-rank default."""
    first, second = tmp_path / "first.gguf", tmp_path / "second.gguf"
    _train(runner_bin, base, first, steps=2, extra=("--lora-rank", "4"))
    fresh = json.loads((tmp_path / "first.gguf.train.json").read_text())
    assert "from_adapter" not in fresh and fresh["lora_rank"] == 4
    _train(runner_bin, base, second, steps=2, extra=("--lora", str(first)))
    rec = json.loads((tmp_path / "second.gguf.train.json").read_text())
    assert rec["lora_rank"] == 4 and rec["alpha"] == 8
    src = rec["from_adapter"]
    assert src["path"] == str(first) and src["sha256"] == _sha(first)
    assert src["record"]["path"] == str(first) + ".train.json"
    assert src["record"]["sha256"] == _sha(str(first) + ".train.json")
    walk = subprocess.run([runner_bin, "--lineage", str(second)], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    out = walk.stdout.decode()
    assert walk.returncode == 1, out   # consistent, unsigned
    assert "from_adapter: " + str(first) in out
    assert out.count("train -> ") == 2, out


def test_training_in_place_records_the_adapter_as_it_was(runner_bin, base, tmp_path):
    """--lora and --train-out naming one file: the starting adapter is hashed
    before the first step, so the record names what the run started from."""
    ad = tmp_path / "inplace.gguf"
    _train(runner_bin, base, ad, steps=2)
    before = _sha(ad)
    _train(runner_bin, base, ad, steps=2, extra=("--lora", str(ad)))
    rec = json.loads((tmp_path / "inplace.gguf.train.json").read_text())
    assert rec["from_adapter"]["sha256"] == before
    assert rec["adapter"]["sha256"] == _sha(ad) != before


def test_dpo_trains_without_naming_the_cpu(runner_bin, base, tmp_path):
    """--train-dpo is the CPU path like --train. It used to load onto the GPU
    of a Metal or CUDA build and then refuse, naming --train, unless --gpu off
    was given."""
    data = tmp_path / "pairs.jsonl"
    data.write_text(json.dumps({"prompt": "the runner", "chosen": " trains",
                                "rejected": " sleeps"}) + "\n")
    out = tmp_path / "dpo-auto.gguf"
    p = subprocess.run([runner_bin, "-m", str(base / "base.gguf"), "--train-dpo",
                        str(data), "--train-steps", "1", "--train-out", str(out),
                        "-t", "2"], cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=600)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert out.exists()
