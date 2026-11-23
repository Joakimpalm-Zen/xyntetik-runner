"""R4.48.10: End-to-end LoRA training, serving, --info adapter, determinism,
resume replay, and scope refusals for Inkling 2 Mini (inkling2_loop).
"""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
import test_inkling2_hardening as h  # noqa: E402
import score_record  # noqa: E402

TEXT = "the quick brown fox jumps over the lazy dog and keeps on running.\n"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def tiny_ink2(tmp_path_factory):
    d = tmp_path_factory.mktemp("ink2_train_fx")
    m = d / "tiny.gguf"
    h._write(m, h._kvs(), h._tensors())
    return m


def _run(exe, args, cwd, expect_ok=True, env=None):
    p = subprocess.run([str(exe), *map(str, args)], cwd=cwd, capture_output=True,
                       text=True, timeout=180, env=env)
    if expect_ok:
        assert p.returncode == 0, f"cmd={args} rc={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
    return p


def test_info_adapter_admits_inkling2(runner_bin, tiny_ink2, tmp_path):
    p = _run(runner_bin, ["-m", tiny_ink2, "--info", "adapter"], tmp_path)
    info = json.loads(p.stdout)
    assert info["architecture"] == "inkling2_loop"
    assert info["adapter_serving"]["supported"] is True
    assert info["adapter_serving"]["reason"] is None
    assert info["adapter_serving"]["device_kernels"] is None
    assert info["training"]["supported"] is True
    assert info["training"]["reason"] is None
    assert info["training"]["host"] == "cpu"


def test_train_loss_drops_determinism_resume_and_serve(runner_bin, tiny_ink2, tmp_path):
    model = tmp_path / "tiny.gguf"
    shutil.copy(tiny_ink2, model)
    corpus = tmp_path / "data.txt"
    corpus.write_text(TEXT * 4)

    base_args = ["-m", model, "--train", corpus, "--train-steps", "4",
                 "--lora-rank", "4", "--lr", "3e-3", "-s", "7", "--gpu", "off"]

    _run(runner_bin, [*base_args, "-t", "2", "--train-out", tmp_path / "a1.gguf"], tmp_path)
    _run(runner_bin, [*base_args, "-t", "2", "--train-out", tmp_path / "a2.gguf"], tmp_path)
    _run(runner_bin, [*base_args, "-t", "1", "--train-out", tmp_path / "at1.gguf"], tmp_path)
    _run(runner_bin, [*base_args, "-t", "4", "--train-out", tmp_path / "at4.gguf"], tmp_path)

    sha = lambda name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    assert sha("a1.gguf") == sha("a2.gguf")
    assert sha("a1.gguf") == sha("at1.gguf")
    assert sha("a1.gguf") == sha("at4.gguf")

    rec = json.loads((tmp_path / "a1.gguf.train.json").read_text())
    assert rec["loss_last"] < rec["loss_first"]
    assert rec["lora_rank"] == 4
    assert rec["steps"] == 4

    # Checkpoint + resume must reproduce the uninterrupted adapter bytes cleanly.
    _run(runner_bin, ["-m", model, "--train", corpus, "--train-steps", "2",
                      "--save-every", "2", "--lora-rank", "4", "--lr", "3e-3",
                      "-s", "7", "--gpu", "off", "-t", "2", "--train-out", tmp_path / "r.gguf"],
         tmp_path)
    assert (tmp_path / "r.gguf.optim").exists()
    _run(runner_bin, ["-m", model, "--train", corpus,
                      "--train-resume", tmp_path / "r.gguf", "--train-steps", "4", "--lora-rank", "4",
                      "--lr", "3e-3", "-s", "7", "--gpu", "off", "-t", "2",
                      "--train-out", tmp_path / "r_resumed.gguf"],
         tmp_path)
    assert sha("r_resumed.gguf") == sha("a1.gguf")

    # Training record replay reproduces a1.gguf byte for byte.
    replay_flags = ["--allow-same-binary"] if rec.get("determinism") == "cpu-same-binary" else []
    _run(runner_bin, ["attest", "replay", tmp_path / "a1.gguf.train.json", *replay_flags, "-t", "1"], tmp_path)

    # Serving: zero-scale adapter scores byte-identical to bare base; trained adapter shifts logits.
    base_score = _run(runner_bin, ["-m", model, "--score", "-p", TEXT.strip(), "-t", "2", "--gpu", "off"],
                      tmp_path).stdout.encode()
    zero_score = _run(runner_bin, ["-m", model, "--lora", tmp_path / "a1.gguf", "--lora-scale", "0",
                                   "--score", "-p", TEXT.strip(), "-t", "2", "--gpu", "off"],
                      tmp_path).stdout.encode()
    assert score_record.body_bytes(zero_score) == score_record.body_bytes(base_score)

    adapted_1 = _run(runner_bin, ["-m", model, "--lora", tmp_path / "a1.gguf", "--score",
                                  "-p", TEXT.strip(), "-t", "2", "--gpu", "off"],
                     tmp_path).stdout.encode()
    adapted_2 = _run(runner_bin, ["-m", model, "--lora", tmp_path / "a1.gguf", "--score",
                                  "-p", TEXT.strip(), "-t", "1", "--gpu", "off"],
                     tmp_path).stdout.encode()
    assert score_record.body_bytes(adapted_1) != score_record.body_bytes(base_score)
    assert score_record.body_bytes(adapted_1) == score_record.body_bytes(adapted_2)


def test_out_of_scope_modes_refused_with_sentence(runner_bin, tiny_ink2, tmp_path):
    corpus = tmp_path / "data.txt"
    corpus.write_text(TEXT * 2)
    jcorpus = tmp_path / "data.jsonl"
    jcorpus.write_text(json.dumps({"text": TEXT}) + "\n")

    # 1. --train-full refused
    p_full = _run(runner_bin, ["-m", tiny_ink2, "--train", str(corpus), "--train-full",
                               "--train-steps", "1", "--gpu", "off", "--train-out", "full.gguf"],
                  tmp_path, expect_ok=False)
    assert p_full.returncode != 0
    assert "full-weight training does not cover inkling2_loop" in p_full.stderr

    # 2. --train-dpo refused
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({"prompt": "hi", "chosen": "yes", "rejected": "no"}) + "\n")
    p_dpo = _run(runner_bin, ["-m", tiny_ink2, "--train-dpo", str(pairs),
                              "--train-steps", "1", "--gpu", "off", "--train-out", "dpo.gguf"],
                 tmp_path, expect_ok=False)
    assert p_dpo.returncode != 0
    assert "DPO training does not cover inkling2_loop" in p_dpo.stderr

    # 3. --train-teacher (KD) refused
    tdir = tmp_path / "teacher"
    tdir.mkdir()
    p_kd = _run(runner_bin, ["-m", tiny_ink2, "--train", str(jcorpus), "--train-teacher", str(tdir),
                             "--train-steps", "1", "--gpu", "off", "--train-out", "kd.gguf"],
                tmp_path, expect_ok=False)
    assert p_kd.returncode != 0
    assert "distillation" in p_kd.stderr and "inkling2_loop" in p_kd.stderr


def test_admission_limits_and_oom_propagation(runner_bin, tiny_ink2, tmp_path):
    corpus = tmp_path / "data.txt"
    corpus.write_text(TEXT * 2)

    # LoRA rank > 256 refused cleanly at training init with a sentence.
    p_rank = _run(runner_bin, ["-m", tiny_ink2, "--train", str(corpus), "--train-steps", "1",
                               "--lora-rank", "257", "--gpu", "off", "--train-out", "r257.gguf"],
                  tmp_path, expect_ok=False)
    assert p_rank.returncode != 0
    assert "256" in p_rank.stderr and "inkling2_loop" in p_rank.stderr
    assert not (tmp_path / "r257.gguf").exists()

    # head_dim > 128 refused at model bind/load with a sentence.
    kvs = [(k, t, 256 if k in ("inkling2_loop.attention.key_length", "inkling2_loop.attention.value_length") else v)
           for k, t, v in h._kvs()]
    bad_hd = tmp_path / "bad_hd.gguf"
    h._write(bad_hd, kvs, h._tensors())
    p_hd = _run(runner_bin, ["-m", bad_hd, "-p", "hello", "-n", "1", "--gpu", "off"],
                tmp_path, expect_ok=False)
    assert p_hd.returncode != 0
    assert "128" in p_hd.stderr and "head_dim" in p_hd.stderr

    # Injected allocation failures in backward propagate cleanly with a stderr sentence and write no adapter.
    bwd_sites = ("lora_proj", "f_lora_apply", "ckpt", "bscr", "layer", "msg_fwd", "msg_bwd",
                 "scaled_clip", "depth_attn", "kv_conv", "attn_fwd", "attn_bwd")
    for site in bwd_sites:
        env_bwd_oom = dict(os.environ, RUNNER_TEST_INK2_ALLOC_FAIL=site)
        out_oom = tmp_path / f"oom_bwd_{site}.gguf"
        p_oom = _run(runner_bin, ["-m", tiny_ink2, "--train", str(corpus), "--train-steps", "1",
                                  "--lora-rank", "4", "--gpu", "off", "--train-out", out_oom],
                     tmp_path, expect_ok=False, env=env_bwd_oom)
        assert p_oom.returncode != 0, f"OOM site {site} succeeded"
        assert "error: inkling2_loop" in p_oom.stderr and "out of memory" in p_oom.stderr.lower(), (
            f"OOM site {site} missing error sentence in stderr: {p_oom.stderr!r}")
        assert not out_oom.exists()

    # Injected allocation failure in serving LoRA forward fails loudly rather than dropping deltas.
    good_a = tmp_path / "good_a.gguf"
    _run(runner_bin, ["-m", tiny_ink2, "--train", str(corpus), "--train-steps", "1",
                      "--lora-rank", "4", "--gpu", "off", "--train-out", good_a],
         tmp_path)
    env_fwd_oom = dict(os.environ, RUNNER_TEST_INK2_ALLOC_FAIL="lora_apply")
    p_foom = _run(runner_bin, ["-m", tiny_ink2, "--lora", good_a, "--score", "-p", TEXT.strip(),
                               "--gpu", "off"],
                  tmp_path, expect_ok=False, env=env_fwd_oom)
    assert p_foom.returncode != 0
    assert "error: inkling2_loop" in p_foom.stderr and "out of memory" in p_foom.stderr.lower()
