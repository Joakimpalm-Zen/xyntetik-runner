"""`--fit`: what a GGUF would cost on this machine, from its header alone.

The feature exists to answer "should I download this" before the bandwidth is
spent, so the load-bearing property is that it works on a file whose data
section is ABSENT — and, just as importantly, that making it work did not
weaken the normal loader, which refuses a file whose data section does not
cover the tensors it declares. That refusal is a safety property: it is what
stops a truncated or crafted file from pointing a tensor outside the mapping.
Both halves are asserted here.
"""
import json
import os
import pathlib
import re
import struct
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "test.gguf"
sys.path.insert(0, str(ROOT / "scripts"))
from gguf_fixture import ku, ks, s  # noqa: E402


def run(runner_bin, *args):
    return subprocess.run([runner_bin, *args], cwd=ROOT, timeout=180,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@pytest.fixture(scope="module")
def header_only(tmp_path_factory):
    """The first 1 MiB of the toy fixture: header present, data section not."""
    if not FIXTURE.exists():
        pytest.skip("test.gguf not generated")
    whole = FIXTURE.read_bytes()
    cut = tmp_path_factory.mktemp("fit") / "head.gguf"
    # Enough for the toy fixture's metadata and tensor descriptors, far less
    # than its data. If this ever stops being a truncation, the test is void.
    n = min(len(whole) - 1, 1 << 20)
    cut.write_bytes(whole[:n])
    assert cut.stat().st_size < FIXTURE.stat().st_size
    return cut


def test_fit_reports_on_a_whole_file(runner_bin):
    p = run(runner_bin, "--fit", str(FIXTURE))
    out = p.stdout.decode()
    assert p.returncode == 0, p.stderr.decode()
    assert "weights" in out and "kv cache" in out and "verdict" in out


def test_fit_works_without_a_data_section(runner_bin, header_only):
    """The whole point: an answer from a header-only download."""
    p = run(runner_bin, "--fit", str(header_only))
    assert p.returncode == 0, p.stderr.decode()
    assert "verdict" in p.stdout.decode()


def test_header_only_sizes_match_the_whole_file(runner_bin, header_only):
    """A truncated file must not report a truncated model.

    `weights` comes from the tensor descriptors, not from how many bytes are
    actually present — so a 1 MiB slice of a larger file must still report the
    larger file's size. Getting this wrong would make every truncated header
    look like it fits.
    """
    def weights(path):
        out = run(runner_bin, "--fit", str(path)).stdout.decode()
        m = re.search(r"weights\s+([\d.]+) GiB", out)
        assert m, out
        return m.group(1)

    assert weights(header_only) == weights(FIXTURE)


def test_normal_load_still_refuses_the_truncated_file(runner_bin, header_only):
    """The safety property the header-tolerant path must not have weakened.

    gguf_open() rejects a file whose data section does not cover its declared
    tensors. `--fit` reads through a SEPARATE path; if it had instead relaxed
    the shared one, this would start loading and the refusal would be gone.
    """
    p = run(runner_bin, "-m", str(header_only), "-p", "hi", "-n", "1",
            "--temp", "0")
    assert p.returncode != 0, "the loader accepted a file with no data section"
    assert b"error" in p.stderr.lower()


def test_context_changes_the_kv_line(runner_bin):
    """-c feeds the estimate, so the answer tracks how it would be run."""
    small = run(runner_bin, "--fit", str(FIXTURE), "-c", "512").stdout.decode()
    large = run(runner_bin, "--fit", str(FIXTURE), "-c", "8192").stdout.decode()
    assert "ctx 512" in small and "ctx 8192" in large
    assert small != large


def test_parallel_slots_each_pay_a_cache(runner_bin):
    """--parallel N: the weights once, a KV cache per slot, as the server's
    load bills it; the cache line is all of them and says so."""
    def kv(*extra):
        # a context long enough that the toy cache prints above 0.00 GiB
        p = run(runner_bin, "--fit", str(FIXTURE), "-c", "4000000", *extra)
        assert p.returncode == 0, p.stderr.decode()
        out = p.stdout.decode()
        w = re.search(r"weights\s+([\d.]+) GiB", out).group(1)
        return out, w, float(re.search(r"kv cache\s+([\d.]+) GiB", out).group(1))
    one, w1, k1 = kv()
    three, w3, k3 = kv("--parallel", "3")
    assert "slots" not in one and "3 (--parallel)" in three
    assert w1 == w3 and k1 > 0.5
    assert k3 == pytest.approx(3 * k1, abs=0.011)


def test_fit_does_not_fetch(runner_bin):
    """A URL is not a path, and the runner does not speak HTTP.

    Recorded as a test because "just add a download" is the obvious next
    request and the answer is deliberately no — the README documents a ranged
    read instead.
    """
    p = run(runner_bin, "--fit", "https://example.invalid/model.gguf")
    assert p.returncode != 0
    assert b"http" not in p.stdout.lower()


def _ple_header(path, arch="gemma4", ple_width=512, ffn=True):
    """Only descriptors, never weights: 8.5 GiB of Q8_0 PLE rows.

    Absolute byte anchor: Q8_0 stores 32 values in 34 bytes. The ordinary
    tied token/output table and BF16 model projection remain in the hot set.
    Every tensor a load binds is declared (R6.68.45: the fit's plan runs the
    loader's tensor binders, dry, and refuses a header they would refuse): the
    output norm, each block's norms, attention and FFN, and for gemma4 with a
    per-layer input width its projection norm and each block's PLE gate,
    projection and norm.
    """
    kv = [ks("general.architecture", arch),
          ku(f"{arch}.block_count", 64),
          ku(f"{arch}.embedding_length", 256),
          ku(f"{arch}.embedding_length_per_layer_input", ple_width),
          ku(f"{arch}.attention.head_count", 4),
          ku(f"{arch}.attention.head_count_kv", 1),
          # what a header the loader can plan declares (R6.68.29: the fit is
          # the loader's plan): the FFN width, and the head width gemma4
          # would otherwise take as 512, not the 64 these cases price
          *([ku(f"{arch}.feed_forward_length", 1024)] if ffn else []),
          ku(f"{arch}.attention.key_length", 64),
          ku(f"{arch}.context_length", 4096)]
    E, H, KV, HD, FF = 256, 4, 1, 64, 1024
    F32, BF16 = (0, 4), (30, 2)
    tensors = [("per_layer_token_embd.weight", 32768, 262144, 8, 34 / 32),
               ("token_embd.weight", E, 262144, 8, 34 / 32),
               ("per_layer_model_proj.weight", E, 32768, *BF16),
               ("output_norm.weight", E, 1, *F32)]
    ple = arch == "gemma4" and ple_width > 0
    if ple:
        tensors.append(("per_layer_proj_norm.weight", ple_width, 1, *F32))
    for i in range(64):
        b = f"blk.{i}."
        tensors += [(b + "attn_norm.weight", E, 1, *F32), (b + "ffn_norm.weight", E, 1, *F32),
                    (b + "attn_q.weight", E, H * HD, *BF16), (b + "attn_k.weight", E, KV * HD, *BF16),
                    (b + "attn_v.weight", E, KV * HD, *BF16),
                    (b + "attn_output.weight", H * HD, E, *BF16),
                    (b + "ffn_gate.weight", E, FF, *BF16), (b + "ffn_up.weight", E, FF, *BF16),
                    (b + "ffn_down.weight", FF, E, *BF16)]
        if ple:
            tensors += [(b + "inp_gate.weight", E, ple_width, *BF16),
                        (b + "proj.weight", ple_width, E, *BF16),
                        (b + "post_norm.weight", E, 1, *F32)]
    directory, offset = b"", 0
    for name, width, rows, kind, bytes_per_value in tensors:
        dims = struct.pack("<IQ", 1, width) if rows == 1 else struct.pack("<IQQ", 2, width, rows)
        directory += s(name) + dims + struct.pack("<IQ", kind, offset)
        offset += int(width * rows * bytes_per_value)
    head = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kv))
    head += b"".join(kv) + directory
    head += b"\0" * (-len(head) % 32)
    path.write_bytes(head)
    cold = 32768 * 262144 // 32 * 34
    return len(head) + offset, cold


def test_eseries_fit_excludes_only_the_row_gathered_table(runner_bin, tmp_path):
    path = tmp_path / "ple-header.gguf"
    total, cold = _ple_header(path)
    p = run(runner_bin, "-m", str(path), "--info", "fit", "--gpu", "off", "-c", "512")
    assert p.returncode == 0, p.stderr.decode()
    out = p.stdout.decode()
    assert path.stat().st_size < 1 << 16   # descriptors: kilobytes for 8.7 GiB of tensors
    gib = 2 ** 30
    assert re.search(rf"weights\s+{total / gib:.2f} GiB", out), out
    assert re.search(rf"hot set\s+{(total - cold) / gib:.2f} GiB", out), out
    assert re.search(rf"row-gathered\s+{cold / gib:.2f} GiB", out), out
    assert "per-layer token embedding" in out
    available = re.search(r"available RAM\s+([\d.]+) GiB", out)
    if available and float(available[1]) > 0.2:
        # Each of 64 layers owns 1 KV head of 64 f16 channels on both sides.
        # Rounding the live available-RAM and printed slack costs <0.01 GiB.
        kv = 64 * 64 * 2 * 2 * 512
        slack = re.search(r"verdict\s+FITS[^\n]*?([\d.]+) GiB to spare", out)
        assert slack, out
        assert abs(float(available[1]) - float(slack[1])
                   - (total - cold + kv) / gib) < 0.011, out


@pytest.mark.parametrize("arch,ple_width", [("llama", 512), ("gemma4", 0)])
def test_fit_does_not_discount_an_unrecognized_ple_table(runner_bin, tmp_path, arch, ple_width):
    path = tmp_path / "not-ple.gguf"
    _ple_header(path, arch, ple_width)
    p = run(runner_bin, "-m", str(path), "--info", "fit", "--gpu", "off")
    assert p.returncode == 0, p.stderr.decode()
    assert "hot set" not in p.stdout.decode()
    assert "row-gathered" not in p.stdout.decode()


def test_a_ple_table_the_loader_refuses_is_not_priced(runner_bin, tmp_path):
    """A gemma4 header declaring a per-layer input width (256) its table does
    not carry (32768 = 64 x 512): the loader refuses the table's shape, so the
    fit does too (R6.68.45), and prices no row-gathered set for it."""
    path = tmp_path / "not-ple.gguf"
    _ple_header(path, "gemma4", 256)
    p = run(runner_bin, "-m", str(path), "--info", "fit", "--gpu", "off")
    assert p.returncode == 1, p
    out = p.stdout.decode()
    assert re.search(r"verdict\s+WILL NOT LOAD .* the loader refuses this file: tensor "
                     r"per_layer_token_embd\.weight in blk\.0 has shape \[32768,262144\], "
                     r"expected \[16384,>=262144\]", out), out
    assert "row-gathered" not in out


# R3.48: deepseek2's KV as the loader lays it out. The absorbed form, the default, keeps one row a token and layer,
# kv_lora_rank + rope wide, which V reads too; RUNNER_MLA_NAIVE=1 keeps the header's head_count_kv x key_length, K and
# V each. The fit took the header's shape whatever the form: 10.7x the absorbed cache on DeepSeek-V2-Lite's geometry.
# Anchor: make-test-model.py's declared shapes (layers, heads = head_count_kv, key_length, kv_lora_rank, rope).
MLA_SHAPES = {
    "default": ([], 2, 4, 16, 16, 8),
    "v2-lite-attention": (["--mla-wide"], 2, 16, 192, 512, 64),
}


@pytest.mark.parametrize("naive", [False, True], ids=["absorbed", "naive"])
@pytest.mark.parametrize("shape", sorted(MLA_SHAPES))
def test_fit_and_the_loader_charge_deepseek2_kv_alike(runner_bin, tmp_path, shape, naive):
    args, layers, heads, key, lora, rope = MLA_SHAPES[shape]
    model = tmp_path / "ds.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--arch", "deepseek2", *args, str(model)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    per_tok = layers * 2 * heads * key * 2 if naive else layers * (lora + rope) * 2
    env = {k: v for k, v in os.environ.items() if k != "RUNNER_MLA_NAIVE"}
    if naive:
        env["RUNNER_MLA_NAIVE"] = "1"
    big, mid = 1 << 22, 1 << 14   # the fit allocates nothing; the load does
    fit = subprocess.run([runner_bin, "--fit", str(model), "-c", str(big), "--gpu", "off"], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, env=env)
    assert fit.returncode == 0, fit.stderr.decode(errors="replace")
    fit_gib = float(re.search(r"kv cache\s+([\d.]+) GiB", fit.stdout.decode()).group(1))
    assert abs(fit_gib - per_tok * big / (1 << 30)) <= 0.006, (shape, naive, fit_gib)
    load = subprocess.run([runner_bin, "-m", str(model), "-v", "-p", "hi", "-n", "1", "-c", str(mid), "--gpu", "off",
                           "-t", "2"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, env=env)
    err = load.stderr.decode(errors="replace")
    assert load.returncode == 0, err[-800:]
    load_mb = float(re.search(r"kv cache\s+([\d.]+) MB", err).group(1))
    assert abs(load_mb - per_tok * mid / 1e6) <= 0.06, (shape, naive, load_mb)


# R6.68.29: --info fit is the loader's own plan of the file (model_plan), not a
# second reading of its header. Three consequences, each held here.
G4_GEOM_FIT = re.compile(r"fit: \(geometry: .*?kv f16 (\d+)")
G4_GEOM_LOAD = re.compile(r"load: \(geometry: .*?kv (\d+) per token")


def _gemma4_without_key_length(tmp_path):
    model = tmp_path / "g4-nokey.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--arch", "gemma4",
                    "--head-dim", "512", "--omit-head-keys", str(model)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return model


def test_a_gemma4_header_without_key_length_fits_at_the_loaders_head_width(runner_bin, tmp_path):
    """gemma4's loader takes a 512-wide head when attention.key_length is absent
    (bind_meta_gemma4); the fit took n_embd / n_head, 16 here, and priced 256 B of
    KV a token for a load that lays out 8,192. Anchor: make-test-model.py's shape,
    2 layers x 2 KV heads x 512 x K and V x 2 bytes."""
    model = _gemma4_without_key_length(tmp_path)
    env = dict(os.environ, RUNNER_DEBUG_VRAM="1")
    fit = subprocess.run([runner_bin, "-m", str(model), "--info", "fit", "--gpu", "off", "-c", "256"],
                         cwd=ROOT, capture_output=True, timeout=120, env=env)
    load = subprocess.run([runner_bin, "-m", str(model), "-p", "hi", "-n", "1", "--gpu", "off",
                           "-t", "1", "-c", "256"], cwd=ROOT, capture_output=True, timeout=120, env=env)
    assert fit.returncode == 0 and load.returncode == 0, (fit.stderr, load.stderr)
    fit_kv = int(G4_GEOM_FIT.search(fit.stderr.decode()).group(1))
    load_kv = int(G4_GEOM_LOAD.search(load.stderr.decode()).group(1))
    assert fit_kv == load_kv == 2 * 2 * 512 * 2 * 2, (fit_kv, load_kv)


def test_a_header_the_loader_refuses_is_the_fits_answer(runner_bin, tmp_path):
    """A header the loader cannot plan will not load, and the fit says so as its
    verdict, with the loader's own reason, and exits 1 (as for any file it cannot
    clear for a load). This one declares no FFN width."""
    path = tmp_path / "no-ffn.gguf"
    _ple_header(path, "llama", 512, ffn=False)
    p = run(runner_bin, "-m", str(path), "--info", "fit", "--gpu", "off")
    assert p.returncode == 1, p
    out = p.stdout.decode()
    assert re.search(r"verdict\s+WILL NOT LOAD .* the loader refuses this file: "
                     r"missing model hyperparameters for arch 'llama'", out), out


def test_the_plans_quiet_lines_stay_quiet_only_for_the_fit(runner_bin, tmp_path):
    """The fit plans quietly (no family notes on its output); a load of the same
    file still prints them, because only the fit's plan may be quiet."""
    model = _gemma4_without_key_length(tmp_path)
    fit = run(runner_bin, "-m", str(model), "--info", "fit", "--gpu", "off", "-c", "256")
    load = run(runner_bin, "-m", str(model), "-p", "hi", "-n", "1", "--gpu", "off", "-t", "1",
               "-c", "256")
    assert b"gemma4:" not in fit.stdout + fit.stderr
    assert b"gemma4:" in load.stderr, load.stderr


# R6.68.45: the fit's plan runs the loader's tensor binders too, dry: every
# tensor the load would find, type and shape-check, none read (the fit opens
# the header alone, and no tensor has data through that handle).
def _pair(tmp_path, gen):
    """A recurrent hybrid fixture and its `.missing-ssm_d` twin."""
    prefix = tmp_path / "h"
    subprocess.run([sys.executable, ROOT / "scripts" / gen, str(prefix)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return tmp_path / "h.gguf", tmp_path / "h.missing-ssm_d.gguf"


@pytest.mark.parametrize("gen", ["make-test-hybrid.py", "make-test-nemotron.py"],
                         ids=["granitehybrid", "nemotron_h"])
def test_a_file_the_tensor_binders_refuse_is_the_fits_answer(runner_bin, tmp_path, gen):
    """The fit said FITS for test-hybrid-batch.missing-ssm_d and
    test-bridge-nemo.missing-ssm_d while their load refused "missing tensor
    blk.0.ssm_d": its plan ran the metadata binders and never the tensor ones.
    The verdict is now the load's refusal, in the load's own words."""
    whole, broken = _pair(tmp_path, gen)
    load = run(runner_bin, "-m", str(broken), "-p", "hi", "-n", "1", "--gpu", "off", "-t", "1",
               "-c", "64")
    assert load.returncode != 0, load
    errors = [ln for ln in load.stderr.decode().splitlines() if ln.startswith("error: ")]
    assert errors and errors[0] == "error: missing tensor blk.0.ssm_d", load.stderr
    fit = run(runner_bin, "-m", str(broken), "--info", "fit", "--gpu", "off")
    assert fit.returncode == 1, fit
    out = fit.stdout.decode()
    assert re.search(r"verdict\s+WILL NOT LOAD .* the loader refuses this file: "
                     r"missing tensor blk\.0\.ssm_d", out), out
    assert run(runner_bin, "-m", str(whole), "--info", "fit", "--gpu", "off").returncode == 0


NO_READ = {
    # bind_layer reads each block's stored output scale (0.91 + 0.02 i here)
    "gemma4-output-scale": ("make-test-model.py", ["--gemma4-hetero"]),
    # bind_deltanet_mixer dequantizes the conv taps once
    "qwen35-conv-taps": ("make-test-ornith.py", []),
    # model_clef_bind refuses decision.scales by their values
    "clef-scales": ("make-test-clef.py", []),
}


@pytest.mark.parametrize("case", sorted(NO_READ))
def test_the_fits_dry_bind_reads_no_tensor(runner_bin, tmp_path, case):
    """Each file's load reads tensor values while it binds. Through the fit's
    header-only handle every tensor's data is NULL, so a read in the dry bind
    faults; the fit answers each one, and plans the same geometry the load
    lays out."""
    gen, args = NO_READ[case]
    model = tmp_path / "m.gguf"
    subprocess.run([sys.executable, ROOT / "scripts" / gen, *args, str(model)], check=True,
                   cwd=ROOT, stdout=subprocess.DEVNULL)
    env = dict(os.environ, RUNNER_DEBUG_VRAM="1")
    fit = subprocess.run([runner_bin, "-m", str(model), "--info", "fit", "--gpu", "off", "-c", "256"],
                         cwd=ROOT, capture_output=True, timeout=120, env=env)
    assert fit.returncode == 0, (fit.returncode, fit.stderr)
    assert b"WILL NOT LOAD" not in fit.stdout, fit.stdout
    load = subprocess.run([runner_bin, "-m", str(model), "-p", "hi", "-n", "1", "--gpu", "off",
                           "-t", "1", "-c", "256"], cwd=ROOT, capture_output=True, timeout=120, env=env)
    assert load.returncode == 0, load.stderr
    fit_kv = int(G4_GEOM_FIT.search(fit.stderr.decode()).group(1))
    load_kv = int(G4_GEOM_LOAD.search(load.stderr.decode()).group(1))
    assert fit_kv == load_kv, (fit_kv, load_kv)


# R6.68.50: the agent-profile check is the plan's first step. It ran in
# model_bind_weights ahead of the plan, so the fit priced a file whose profile
# the load refuses.
PROFILE_REFUSALS = {
    "unknown-feature": ["--agent-feature", "future_attention_v99"],
    "mtp-feature": ["--agent-feature", "mtp"],
    "bad-digest": ["--agent-digest", "z" * 64],
}


def _profile_model(tmp_path, flags):
    model = tmp_path / "profile.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", *flags, str(model)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return model


@pytest.mark.parametrize("case", sorted(PROFILE_REFUSALS))
def test_an_agent_profile_the_load_refuses_is_the_fits_answer(runner_bin, tmp_path, case):
    """The fit's reason is the load's first error line, word for word."""
    model = _profile_model(tmp_path, PROFILE_REFUSALS[case])
    load = run(runner_bin, "-m", str(model), "-p", "hi", "-n", "1", "--gpu", "off", "-t", "1",
               "-c", "64")
    assert load.returncode != 0, load
    errors = [ln for ln in load.stderr.decode().splitlines() if ln.startswith("error: ")]
    assert errors, load.stderr
    fit = run(runner_bin, "-m", str(model), "--info", "fit", "--gpu", "off")
    assert fit.returncode == 1, fit
    verdict = [ln for ln in fit.stdout.decode().splitlines() if "verdict" in ln]
    assert verdict and "WILL NOT LOAD" in verdict[0], fit.stdout
    reason = verdict[0].split("the loader refuses this file: ", 1)[1]
    assert errors[0] == "error: " + reason, (errors[0], reason)


def test_a_valid_agent_profile_still_fits(runner_bin, tmp_path):
    model = _profile_model(tmp_path, ["--agent-profile"])
    fit = run(runner_bin, "-m", str(model), "--info", "fit", "--gpu", "off")
    assert fit.returncode == 0, fit
    assert b"WILL NOT LOAD" not in fit.stdout


def test_a_split_expert_moe_is_priced_as_sparse(runner_bin, tmp_path):
    """R6.68.53: the legacy layout of one tensor per expert (older Mixtral
    GGUFs) is a sparse MoE to the load, whose hot set discounts its banks
    (model_hot_set_bytes). The fit found routed banks by the name
    `_exps.weight`, which that layout never has, and priced the file as dense:
    no hot set, the whole file. make-test-moe.py's moe3 routes 1 of 2 experts
    in that layout."""
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(tmp_path / "m")],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    p = run(runner_bin, "-m", str(tmp_path / "m.moe3.gguf"), "--info", "fit", "--gpu", "off")
    assert p.returncode == 0, p
    out = p.stdout.decode()
    assert "MoE 2 experts, 1 used" in out, out
    assert "only the routed experts a token actually uses" in out, out


# R6.68.55: qwen4exp's context cap. This build runs the sparse-attention
# indexer dense, which is exact only up to its top_k (2048 in the fixture, as
# in the real Qwen3.8-Flash-Next file). The check sat in the bind, which runs
# before any context is set, and never fired: a default load took 4096.
def _qwen4exp(tmp_path, train_ctx):
    model = tmp_path / "q.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-qwen4exp.py", str(model)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, QWEN4EXP_TEST_CTX=str(train_ctx)))
    return model


def test_qwen4exp_refuses_a_context_past_its_indexer_top_k(runner_bin, tmp_path):
    model = _qwen4exp(tmp_path, 8192)
    load = run(runner_bin, "-m", str(model), "-p", "hi", "-n", "1", "--gpu", "off", "-t", "1",
               "-c", "4096")
    assert load.returncode != 0, load
    assert b"admits contexts up to 2048 tokens (asked 4096)" in load.stderr, load.stderr
    fit = run(runner_bin, "-m", str(model), "--info", "fit", "--gpu", "off", "-c", "4096")
    assert fit.returncode == 1, fit
    assert re.search(r"WILL NOT LOAD .* admits contexts up to 2048 tokens \(asked 4096\)",
                     fit.stdout.decode()), fit.stdout


def test_qwen4exp_holds_a_default_context_to_its_indexer_top_k(runner_bin, tmp_path):
    model = _qwen4exp(tmp_path, 8192)
    load = run(runner_bin, "-m", str(model), "-p", "hi", "-n", "1", "--gpu", "off", "-t", "1")
    assert load.returncode == 0, load
    assert b"| ctx 2048 |" in load.stdout + load.stderr, load.stderr
    fit = run(runner_bin, "-m", str(model), "--info", "fit", "--gpu", "off")
    assert fit.returncode == 0, fit
    assert b"at ctx 2048" in fit.stdout, fit.stdout


def test_a_router_past_the_device_caps_runs_on_the_cpu_and_the_fit_says_so(runner_bin, tmp_path):
    """R6.68.67: a device bank's router takes at most so many experts and picks
    a token (CUDA 512 and 256, Metal 256 and 256). Every CUDA route clamped past
    them and nothing refused, so such a model was routed among a subset of its
    experts on the device alone (the class of 2026-10-07, when the cap was 256).
    The load now runs it on the CPU and says why, and --info fit says the same
    words. make-test-moe.py's wide file routes 2 of 520."""
    gpu = json.loads(run(runner_bin, "--caps").stdout).get("gpu") or {}
    if not gpu.get("name"):
        pytest.skip("no GPU device in this build or on this machine")
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(tmp_path / "m")],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, MOE_TEST_WIDE="520"))
    model = tmp_path / "m.wide.gguf"
    load = run(runner_bin, "-m", str(model), "-p", "hi", "-n", "1", "-t", "1", "-c", "64")
    assert load.returncode == 0, load
    said = [ln for ln in load.stderr.decode().splitlines() if ln.startswith("gpu: this model routes")]
    assert said, load.stderr
    fit = run(runner_bin, "-m", str(model), "--info", "fit")
    assert fit.returncode == 0, fit
    marker = "present, but the load runs on the CPU: "
    lines = [ln for ln in fit.stdout.decode().splitlines() if marker in ln]
    assert lines, fit.stdout
    reason = lines[0].split(marker, 1)[1]
    assert "routes 2 of 520 experts" in reason, reason
    assert said[0].startswith("gpu: " + reason + ","), (said[0], reason)


def _wide_moe(tmp_path, name, experts, used):
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(tmp_path / name)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, MOE_TEST_WIDE=str(experts),
                            MOE_TEST_WIDE_USED=str(used)))
    return tmp_path / f"{name}.wide.gguf"


def test_a_router_picking_past_the_host_selection_is_refused_not_run(runner_bin, tmp_path):
    """R6.68.89: the host keeps a token's picked experts in arrays of 256.
    The MoE binder held expert_used_count to expert_count (up to 1024) only,
    so a file picking 257 loaded and the first token wrote past those arrays
    (SIGSEGV). The bind refuses it now, for every MoE binder, and --info fit
    says the same words. 256 still runs."""
    model = _wide_moe(tmp_path, "over", 258, 257)
    load = run(runner_bin, "-m", str(model), "-p", "hi", "-n", "1", "--gpu", "off",
               "-t", "1", "-c", "64")
    assert load.returncode == 1, load
    said = [ln for ln in load.stderr.decode().splitlines()
            if "picks 257 experts a token" in ln]
    assert said, load.stderr
    fit = run(runner_bin, "-m", str(model), "--info", "fit", "--gpu", "off")
    assert said[0].removeprefix("error: ") in (fit.stdout + fit.stderr).decode(), fit
    edge = _wide_moe(tmp_path, "edge", 258, 256)
    ok = run(runner_bin, "-m", str(edge), "-p", "hi", "-n", "1", "--gpu", "off",
             "-t", "1", "-c", "64")
    assert ok.returncode == 0, ok
