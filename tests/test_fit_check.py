"""`--fit`: what a GGUF would cost on this machine, from its header alone.

The feature exists to answer "should I download this" before the bandwidth is
spent, so the load-bearing property is that it works on a file whose data
section is ABSENT — and, just as importantly, that making it work did not
weaken the normal loader, which refuses a file whose data section does not
cover the tensors it declares. That refusal is a safety property: it is what
stops a truncated or crafted file from pointing a tensor outside the mapping.
Both halves are asserted here.
"""
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


def _ple_header(path, arch="gemma4", ple_width=512):
    """Only descriptors, never weights: 8.5 GiB of Q8_0 PLE rows.

    Absolute byte anchor: Q8_0 stores 32 values in 34 bytes. The ordinary
    tied token/output table and BF16 model projection remain in the hot set.
    """
    kv = [ks("general.architecture", arch),
          ku(f"{arch}.block_count", 64),
          ku(f"{arch}.embedding_length", 256),
          ku(f"{arch}.embedding_length_per_layer_input", ple_width),
          ku(f"{arch}.attention.head_count", 4),
          ku(f"{arch}.attention.head_count_kv", 1),
          ku(f"{arch}.context_length", 4096)]
    tensors = [("per_layer_token_embd.weight", 32768, 262144, 8, 34 / 32),
               ("token_embd.weight", 256, 262144, 8, 34 / 32),
               ("per_layer_model_proj.weight", 256, 32768, 30, 2)]
    directory, offset = b"", 0
    for name, width, rows, kind, bytes_per_value in tensors:
        directory += s(name) + struct.pack("<IQQIQ", 2, width, rows, kind, offset)
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
    assert path.stat().st_size < 1024
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


@pytest.mark.parametrize("arch,ple_width", [("llama", 512), ("gemma4", 0), ("gemma4", 256)])
def test_fit_does_not_discount_an_unrecognized_ple_table(runner_bin, tmp_path, arch, ple_width):
    path = tmp_path / "not-ple.gguf"
    _ple_header(path, arch, ple_width)
    p = run(runner_bin, "-m", str(path), "--info", "fit", "--gpu", "off")
    assert p.returncode == 0, p.stderr.decode()
    assert "hot set" not in p.stdout.decode()
    assert "row-gathered" not in p.stdout.decode()
