"""The serve soak, run small against the fixture model.

Two things are held here. The instrument: every phase of scripts/soak-serve.py
runs and reports through its public interface, the command line and the JSON
report. And the server, on the paths the soak walks: a greedy reply repeats,
a cancel in each of the four ways leaves the server idle and answering the
same, the context limit is refused with a 400 that names it, and concurrent
clients get the replies they got alone.

No absolute anchor of its own: the replies of a random-weight fixture are
compared with themselves. The anchors are the HTTP statuses the manual
documents (400 context_length_exceeded, 409 for a cancelled request).
"""
import importlib.util
import json
import pathlib
import socket
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
MODEL = ROOT / "test.gguf"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def soak(tmp_path, *extra):
    cmd = [sys.executable, str(ROOT / "scripts" / "soak-serve.py"),
           "--runner", str(RUNNER), "--model", str(MODEL), "--port", str(free_port()),
           "--workdir", str(tmp_path), "--filler-base", "3", "--filler-vocab", "20",
           "--cycles", "8", "--cancels", "2", "--clients", "4",
           "--concurrent-seconds", "2", "--cancel-prompt-tokens", "96",
           # the fixture prefills in microseconds: a small batch and the test
           # hooks give a cancel something to land in
           "--server-arg=-b", "--server-arg=8",
           "--server-env", "RUNNER_TEST_PREFILL_DELAY_MS=40",
           "--server-env", "RUNNER_TEST_STEP_DELAY_MS=5", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    report = json.loads((tmp_path / "report.json").read_text())
    return p, report


@pytest.mark.skipif(not RUNNER.exists() or not MODEL.exists(),
                    reason="runner or test.gguf not built")
def test_the_soak_runs_every_phase_clean_on_the_fixture(tmp_path):
    p, r = soak(tmp_path)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]
    assert r["findings"] == []
    ph = r["phases"]
    assert ph["identity"]["cycles"] == 8 and ph["identity"]["mismatches"] == 0
    c = ph["cancel"]["counts"]
    # each kind was sent, and cancels by id landed while the request ran
    assert c["close_prefill"] == 2 and c["close_decode"] == 2
    assert c["id_prefill"] >= 1 and c["id_decode"] >= 1
    limit = {row["case"]: row for row in ph["limit"]}
    assert limit["one token over"]["status"] == 400
    assert limit["one token over"]["code"] == "context_length_exceeded"
    assert limit["one under, one token asked"]["status"] == 200
    run_in = limit["generation runs into the window"]
    assert run_in["status"] == 200 and run_in["completion_tokens"] <= 8
    conc = ph["concurrent"]
    assert conc["requests"] > 0 and conc["mismatch"] == 0 and conc["invalid"] == 0
    assert set(conc["status"]) == {"200"}


@pytest.mark.skipif(not RUNNER.exists() or not MODEL.exists(),
                    reason="runner or test.gguf not built")
def test_a_cancel_phase_that_lands_nothing_is_a_finding(tmp_path):
    """The negative control: a soak that cannot fail proves nothing. With a
    delay longer than the request, no cancel by id can land mid-prefill, and
    the soak must say the phase measured nothing instead of passing."""
    p, r = soak(tmp_path, "--skip", "identity", "--skip", "limit", "--skip", "concurrent",
                "--cancels", "1", "--cancel-delay", "6")
    assert p.returncode == 1
    assert r["phases"]["cancel"]["counts"]["id_prefill"] == 0
    assert any("measured nothing" in f["what"] for f in r["findings"])


def test_device_memory_is_read_from_the_plain_table_when_the_queries_refuse():
    """A MIG slice answers both nvidia-smi queries "[Insufficient
    Permissions]", and the soak then compared no device memory and said
    nothing (found on a 1g.24gb slice before its first run there). The plain
    table still carries the numbers; the rows below are its shape."""
    spec = importlib.util.spec_from_file_location("soak_serve", ROOT / "scripts" / "soak-serve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    table = """
|   0  NVIDIA GPU                     On  |   00000000:01:00.0 Off |                   On |
|=========================================+========================+======================|
|  0    3   0   0  |              81MiB / 24192MiB    | 14      0 |  1   0    0    0    0 |
|                  |              12MiB /  8317MiB    |           |                       |
| Processes:                                                                              |
|    0    3    0             4242      C   ./runner                                 118MiB |
"""
    assert mod.parse_smi_table(table, "4242") == 118      # the server's own row
    assert mod.parse_smi_table(table, "999") == 81        # else the device row
    assert mod.parse_smi_table(table) == 81
    assert mod.parse_smi_table("[Insufficient Permissions], [Insufficient Permissions]") is None


@pytest.mark.skipif(not RUNNER.exists() or not MODEL.exists(),
                    reason="runner or test.gguf not built")
def test_the_swap_phase_holds_each_model_to_its_fresh_reply(tmp_path):
    """Two models alternated: every reply equals the one a fresh process gave
    for that model, and memory is compared between readings that hold the
    same model (the first run on a real pair reported the two models' size
    difference as growth)."""
    second = tmp_path / "second.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make-test-model.py"), str(second)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    p, r = soak(tmp_path, "--model2", str(second), "--swaps", "12",
                "--skip", "identity", "--skip", "cancel", "--skip", "limit",
                "--skip", "concurrent")
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-1000:]
    mem = r["phases"]["swap"]["memory"]
    assert set(mem) == {"a", "b"}
    # each model was resident for half the swaps, and each has its own series
    assert len(mem["a"]) == 6 and len(mem["b"]) == 6
    assert not set(mem["a"]) & set(mem["b"])
