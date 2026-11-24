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
import re
import subprocess
import sys

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


def _gpu_lines(err):
    """Every test-seam and gpu: line from stderr, never buried by \\rprompt: progress."""
    lines = [ln.strip() for ln in err.splitlines()
             if "test seam:" in ln or "gpu:" in ln or "error:" in ln]
    return "\n".join(lines) if lines else err[-800:]


def _bench(runner_bin, ctx, **seams):
    p = subprocess.run([str(runner_bin), "-m", str(MODEL), "--bench-json", "-n", "2", "-c", str(ctx),
                        "--no-tray"], cwd=ROOT, env=_env(**seams), capture_output=True, text=True,
                       timeout=300)
    assert p.returncode == 0, _gpu_lines(p.stderr)
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
    assert "the KV cache grows with the context" in err, _gpu_lines(err)
    # the control: a step of the whole context maps all of it, as before
    d, _ = _bench(runner_bin, 4096, RUNNER_TEST_METAL_KV_STEP=4096)
    assert d["route"]["kv_rows"]["mapped"] == 4096, d["route"]["kv_rows"]


def _grew(stderr):
    """The engine's own count of KV grows in a run, from its end-of-run line."""
    m = re.search(r"gpu: the KV cache grew (\d+) times?, to (\d+) of (\d+) rows", stderr)
    return (int(m.group(1)), int(m.group(2))) if m else None


def test_a_small_grow_step_gives_the_same_tokens(runner_bin):
    """Bit identity across many grows: 16-row steps against one mapping of the whole context, both on
    the Apple GPU (RUNNER_GPU_FALLBACK=fail: a fallback would compare CPU bytes), and the small step
    having grown at least once per 16 rows the run wrote. The live control: a changed first word
    changes the output, so the context reaches the logits."""
    _metal_or_skip(runner_bin)
    small = _generate(runner_bin, TEXT, RUNNER_TEST_METAL_KV_STEP=16, RUNNER_GPU_FALLBACK="fail")
    whole = _generate(runner_bin, TEXT, RUNNER_TEST_METAL_KV_STEP=1024, RUNNER_GPU_FALLBACK="fail")
    assert small.returncode == 0 and whole.returncode == 0, (
        _gpu_lines(small.stderr) + "\n---\n" + _gpu_lines(whole.stderr))
    for p in (small, whole):
        assert "gpu: Metal backend" in p.stderr and "continuing on the CPU" not in p.stderr, _gpu_lines(p.stderr)
    # a forward grows once to cover all of its rows: the prompt's batches grow at least once past the
    # first step, and each decode step that crosses a 16-row boundary grows again
    prompt = int(re.search(r"prompt: (\d+) tok", small.stderr).group(1))
    grows, rows = _grew(small.stderr) or (0, 0)
    crossings = -(-(prompt + 47) // 16) - (-(-prompt // 16))
    assert grows >= 1 + crossings and rows >= prompt + 47 and rows % 16 == 0, \
        f"grows={grows} rows={rows} prompt={prompt}\n{_gpu_lines(small.stderr)}"
    assert _grew(whole.stderr) in (None, (0, 1024)), _gpu_lines(whole.stderr)
    assert small.stdout and small.stdout == whole.stdout
    other = _generate(runner_bin, "A" + TEXT[3:], RUNNER_TEST_METAL_KV_STEP=16)
    assert other.returncode == 0 and other.stdout != small.stdout, "the context does not reach the logits"


def test_a_multipart_model_grows_against_all_of_its_weights(runner_bin, tmp_path):
    """codex's read of PR G: the grow check charged the weights of a multi-part file's first part
    only (metal_choose_split's wlen), so a part set admitted against the working set could grow as
    if its weights were one part's. The same weights whole and in two parts must reach the same
    refusal at the same row count under the same working set."""
    _metal_or_skip(runner_bin)
    whole = tmp_path / "wide8.gguf"
    subprocess.run([sys.executable, ROOT / "scripts" / "make-test-model.py", "--wide", "--n-ff", "2048",
                    "--layers", "8", str(whole)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, ROOT / "scripts" / "gguf-split.py", str(whole), str(tmp_path / "parts"),
                    "2"], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    parts = sorted(tmp_path.glob("parts-0000?-of-00002.gguf"))
    assert len(parts) == 2, list(tmp_path.iterdir())
    prompt = tmp_path / "long.txt"
    prompt.write_text("the river city market " * 70, encoding="utf-8")   # about 2,100 tokens

    def run(model):
        p = subprocess.run([str(runner_bin), "-m", str(model), "-f", str(prompt), "-n", "1", "-c", "4096",
                            "--temp", "0", "--no-tray"], cwd=ROOT, capture_output=True, timeout=600,
                           env=_env(RUNNER_TEST_METAL_WORKING_SET_MB=WS_MB))
        err = p.stderr.decode("utf-8", "replace")
        assert p.returncode == 0, _gpu_lines(err)
        m = re.search(r"the KV cache for (\d+) rows does not fit", err)
        return int(m.group(1)) if m else None, err

    one, err_one = run(whole)
    split, err_split = run(parts[0])
    assert "partial offload" not in err_one, "whole must be admitted full-offload\n" + _gpu_lines(err_one)
    assert "partial offload" not in err_split, "parts[0] must be admitted full-offload\n" + _gpu_lines(err_split)
    assert one is not None, "the calibration: the whole file must refuse a grow here\n" + _gpu_lines(err_one)
    assert split == one, f"one={one} split={split}\n{_gpu_lines(err_split)}"


# the working set at which the 8-layer fixture (54.3 MiB whole; 25.3 + 29.0 MiB split) is admitted
# full-offload (past the 0.94 admission factor at 571 MiB) and refuses a grow at 1,536 rows while a
# first-part-only weight charge would grow past the entire 2,100-token prompt (512 MiB allowance +
# weights + 4 KiB/row KV)
WS_MB = 572


def test_a_runtime_failure_keeps_the_backend_error_past_cpu_fallback(runner_bin, tmp_path):
    """R6.98: when a Metal forward fails at runtime and the run finishes thousands of prompt tokens
    on CPU, the end-of-stderr gpu_lost_note and route.fallback must still carry the backend's
    runtime error (command buffer error domain/code/description, scratch, or wrap detail) rather
    than dropping it behind \\rprompt: progress updates."""
    _metal_or_skip(runner_bin)
    prompt = tmp_path / "long.txt"
    prompt.write_text("the river city market " * 500, encoding="utf-8")   # about 15,000 tokens
    p = subprocess.run([str(runner_bin), "-m", str(MODEL), "-f", str(prompt), "-n", "1", "-c", "16384",
                        "--temp", "0", "--no-tray"], cwd=ROOT, capture_output=True, timeout=300,
                       env=_env(RUNNER_METAL_INJECT_FAILURE=3))
    err = p.stderr.decode("utf-8", "replace")
    assert p.returncode == 0, _gpu_lines(err)
    # confirm the failure happened early enough that the raw tail of stderr no longer holds the
    # first command-buffer line (the exact shape that hid R6.98 sighting 2)
    assert "gpu: command buffer failed" not in err[-800:], err[-800:]
    lost = [ln for ln in err.splitlines() if "finished on the CPU" in ln]
    assert lost and "forward failed at runtime" in lost[-1] and "command buffer failed" in lost[-1], \
        f"gpu_lost_note lost the backend error:\n{_gpu_lines(err)}"
    d, berr = _bench(runner_bin, 1024, RUNNER_METAL_INJECT_FAILURE=1)
    assert d["route"]["backend"] == "cpu", d["route"]
    assert "forward failed at runtime" in d["route"]["fallback"] and \
        "command buffer failed" in d["route"]["fallback"], (d["route"], _gpu_lines(berr))


def test_a_missed_grow_is_caught_before_the_encode(runner_bin):
    _metal_or_skip(runner_bin)
    p = _generate(runner_bin, TEXT, RUNNER_TEST_METAL_KV_STEP=16, RUNNER_TEST_METAL_KV_SKIP_GROW=1,
                  RUNNER_GPU_FALLBACK="fail")
    assert p.returncode == 3, f"rc={p.returncode}\n{_gpu_lines(p.stderr)}"
    assert "are not mapped" in p.stderr, _gpu_lines(p.stderr)


def test_a_grow_that_does_not_fit_continues_on_the_cpu(runner_bin):
    _metal_or_skip(runner_bin)
    d, err = _bench(runner_bin, 4096, RUNNER_TEST_METAL_KV_GROW_FAILS_AT=512)
    assert "rows does not fit" in err and "continuing on the CPU" in err, _gpu_lines(err)
    assert d["generated_tokens"] == 2, d
    assert d["route"]["backend"] == "cpu" and "KV" in d["route"]["fallback"], d["route"]


def _wide8(tmp_path):
    p = tmp_path / "wide8.gguf"
    subprocess.run([sys.executable, ROOT / "scripts" / "make-test-model.py", "--wide", "--n-ff", "2048",
                    "--layers", "8", str(p)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return p


def test_a_grow_charges_the_weights_the_device_really_holds(runner_bin, tmp_path):
    """codex's second read of PR G: a split whose no-copy wrap fails copies the WHOLE file into one
    shared buffer (metal_place_weights), but its grows were priced against the first layers' prefix.
    RUNNER_METAL_MAX_BUF=1 makes the wrap fail; at a 560 MiB working set the copied 57 MB file and the
    fixed allowance leave no room for a grow, while the wrapped prefix (the control) does."""
    _metal_or_skip(runner_bin)
    model = _wide8(tmp_path)

    def run(**seams):
        p = subprocess.run([str(runner_bin), "-m", str(model), "-p", TEXT, "-n", "8", "-c", "4096",
                            "--gpu-layers", "1", "--temp", "0", "--no-tray"], cwd=ROOT, capture_output=True,
                           timeout=300, env=_env(RUNNER_TEST_METAL_WORKING_SET_MB=560,
                                                 RUNNER_TEST_METAL_KV_STEP=16, **seams))
        err = p.stderr.decode("utf-8", "replace")
        assert p.returncode == 0, _gpu_lines(err)
        return err

    copied = run(RUNNER_METAL_MAX_BUF=1)
    assert "partial offload" in copied, _gpu_lines(copied)
    assert "rows does not fit" in copied and "continuing on the CPU" in copied, _gpu_lines(copied)
    wrapped = run()
    assert "partial offload" in wrapped and "rows does not fit" not in wrapped, _gpu_lines(wrapped)


def test_the_end_of_run_line_keeps_the_high_water_mark_after_a_refusal(runner_bin):
    """codex's second read of PR G: after a refused grow the end-of-run line read "to 0 of C rows",
    since the refusal releases the region buffers. It keeps the rows the cache reached, and says the
    run ended on the CPU."""
    _metal_or_skip(runner_bin)
    p = _generate(runner_bin, TEXT, RUNNER_TEST_METAL_KV_STEP=16, RUNNER_TEST_METAL_KV_GROW_FAILS_AT=600)
    assert p.returncode == 0, _gpu_lines(p.stderr)
    grew = _grew(p.stderr)
    assert grew is not None and grew[0] >= 1 and 550 <= grew[1] < 600, f"grew={grew}\n{_gpu_lines(p.stderr)}"
    assert re.search(r"a grow to 6\d\d rows did not fit, so the run ended on the CPU", p.stderr), _gpu_lines(p.stderr)


def test_a_wrap_that_fails_part_way_releases_what_it_made(runner_bin, tmp_path):
    """codex's third read of PR G: metal_wrap_weights could fail after making some wraps (here a 2 MiB
    FFN tensor over a 1 MiB per-buffer cap, after the small early tensors wrapped), and the copy
    fallback overwrote them unreleased. RUNNER_TEST_METAL_HELD_AFTER_FREE prints the bytes this
    process's Metal device still holds once the model is freed (no other process moves it); a split
    whose wrap failed part-way must hold what a split that wrapped holds."""
    _metal_or_skip(runner_bin)
    model = _wide8(tmp_path)

    def held(**seams):
        p = subprocess.run([str(runner_bin), "-m", str(model), "-p", "hello there", "-n", "4", "-c", "1024",
                            "--gpu-layers", "1", "--temp", "0", "--no-tray"], cwd=ROOT, capture_output=True,
                           timeout=300, env=_env(RUNNER_TEST_METAL_HELD_AFTER_FREE=1, **seams))
        err = p.stderr.decode("utf-8", "replace")
        assert p.returncode == 0, _gpu_lines(err)
        return int(re.search(r"the Metal device holds (\d+) bytes after the model is freed", err).group(1)), err

    wrapped, _ = held()
    partway, err = held(RUNNER_METAL_MAX_BUF=1048576)
    assert "not wrapping" in err and "weights copied" in err, _gpu_lines(err)
    assert partway == wrapped, (partway, wrapped)
