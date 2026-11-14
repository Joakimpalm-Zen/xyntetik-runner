"""The Apple GPU maps only the KV rows a context uses, and grows them as it fills (R3.68).

Metal allocated the KV cache for the whole context and every layer up front, and the GPU made it
resident in full at first use: 1.2 GB for Qwen3-4B at -c 8192 with a short prompt, and the whole
offload refused for Phi-3.5-mini at -c 8192 on an 8 GB Mac. Now the KV is one host allocation, paged in
as it is used (the CPU path's own), and the GPU sees each layer's K and V rows through a no-copy buffer
over the rows in use. At the top of a forward the buffers grow in fixed steps of rows; a grow that
would not fit in RAM continues on the CPU, which reads the same rows from the same memory.

RUNNER_TEST_METAL_KV_STEP sets the step (a gate arm: a small step crosses many boundaries),
RUNNER_TEST_METAL_KV_SKIP_GROW skips the next grow (the host check before every encode must catch it),
and RUNNER_TEST_METAL_KV_GROW_FAILS_AT makes a grow to that many rows not fit. No check here reads
phys_footprint: it does not count the pages the GPU wires behind a no-copy buffer.
"""
import json
import os
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL = ROOT / "test.gguf"
TEXT = ("The quick brown fox jumps over the lazy dog while the five boxing wizards jump quickly, and a "
        "sphinx of black quartz judges my vow. ") * 3


def _metal_or_skip(runner_bin):
    caps = json.loads(subprocess.run([str(runner_bin), "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
                                     timeout=60).stdout)
    if (caps.get("gpu") or {}).get("backend") != "metal":
        pytest.skip("the Metal backend's KV mapping")


def _env(**seams):
    env = {k: v for k, v in os.environ.items() if not k.startswith("RUNNER_TEST_METAL_KV_")}
    env.update({k: str(v) for k, v in seams.items()})
    return env


def _bench(runner_bin, ctx, **seams):
    p = subprocess.run([str(runner_bin), "-m", str(MODEL), "--bench-json", "-n", "2", "-c", str(ctx),
                        "--no-tray"], cwd=ROOT, env=_env(**seams), capture_output=True, text=True,
                       timeout=300)
    assert p.returncode == 0, p.stderr[-800:]
    return json.loads(p.stdout), p.stderr


def _generate(runner_bin, prompt, **seams):
    # bytes: a toy model's tokens need not decode as UTF-8, and identity is over the bytes
    p = subprocess.run([str(runner_bin), "-m", str(MODEL), "-p", prompt, "-n", "48", "-c", "1024",
                        "--temp", "0", "--no-tray"], cwd=ROOT, env=_env(**seams), capture_output=True,
                       timeout=300)
    p.stderr = p.stderr.decode("utf-8", "replace")
    return p


def test_a_short_prompt_maps_only_the_rows_it_uses(runner_bin):
    _metal_or_skip(runner_bin)
    d, err = _bench(runner_bin, 4096)
    rows = d["route"]["kv_rows"]
    used = d["prompt_tokens"] + d["generated_tokens"]
    assert rows["step"] == 256 and rows["of"] == 4096, rows
    assert rows["mapped"] == -(-used // 256) * 256 < 4096, (rows, used)
    assert "the KV cache grows with the context" in err, err[-800:]
    # the control: a step of the whole context maps all of it, as before
    d, _ = _bench(runner_bin, 4096, RUNNER_TEST_METAL_KV_STEP=4096)
    assert d["route"]["kv_rows"]["mapped"] == 4096, d["route"]["kv_rows"]


def test_a_small_grow_step_gives_the_same_tokens(runner_bin):
    """Bit identity across many grows: 16-row steps against one mapping of the whole context. The live
    control: a changed first word changes the output, so the context reaches the logits."""
    _metal_or_skip(runner_bin)
    small = _generate(runner_bin, TEXT, RUNNER_TEST_METAL_KV_STEP=16)
    whole = _generate(runner_bin, TEXT, RUNNER_TEST_METAL_KV_STEP=1024)
    assert small.returncode == 0 and whole.returncode == 0, small.stderr[-400:] + whole.stderr[-400:]
    assert small.stdout and small.stdout == whole.stdout
    other = _generate(runner_bin, "A" + TEXT[3:], RUNNER_TEST_METAL_KV_STEP=16)
    assert other.returncode == 0 and other.stdout != small.stdout, "the context does not reach the logits"


def test_a_missed_grow_is_caught_before_the_encode(runner_bin):
    _metal_or_skip(runner_bin)
    p = _generate(runner_bin, TEXT, RUNNER_TEST_METAL_KV_STEP=16, RUNNER_TEST_METAL_KV_SKIP_GROW=1,
                  RUNNER_GPU_FALLBACK="fail")
    assert p.returncode == 3, (p.returncode, p.stderr[-800:])
    assert "are not mapped" in p.stderr, p.stderr[-800:]


def test_a_grow_that_does_not_fit_continues_on_the_cpu(runner_bin):
    _metal_or_skip(runner_bin)
    d, err = _bench(runner_bin, 4096, RUNNER_TEST_METAL_KV_GROW_FAILS_AT=512)
    assert "does not fit" in err and "continuing on the CPU" in err, err[-800:]
    assert d["generated_tokens"] == 2, d
    assert d["route"]["backend"] == "cpu" and "KV" in d["route"]["fallback"], d["route"]
