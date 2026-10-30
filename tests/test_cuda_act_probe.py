"""scripts/cuda-act-probe.py and scripts/release-cuda-gate.py without a device:
the PTX the probe loads is the one the binary embeds and holds the three
kernels with the parameter lists the probe passes; its comparison refuses a
NaN; the gate writes a report's home directory as ~/."""
import importlib.util
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_probe_reads_the_embedded_ptx_and_its_three_kernels():
    probe = load("cuda-act-probe")
    ptx = probe.ptx_from_header(os.path.join(ROOT, "src", "kernels_ptx.h"))
    assert ptx.lstrip().startswith("//") and ".target sm_75" in ptx
    want = {"k_q35_attn_gate": ["u64", "u64", "u32", "u32", "u32"],
            "k_silu_mul": ["u64", "u64", "u32"],
            "k_moe_actmul": ["u64", "u64", "u32", "u32", "u32", "u32", "u64", "u64", "u64", "u64", "u64"]}
    for kernel, params in want.items():
        m = re.search(r"\.entry " + kernel + r"\(\s*(.*?)\)", ptx, re.S)
        assert m, kernel
        assert re.findall(r"\.param \.(\w+) ", m.group(1)) == params, kernel


def test_the_comparison_refuses_nan_and_takes_exact_zero():
    probe = load("cuda-act-probe")
    assert probe.close(0.0, probe.sigmoid(-269.7))
    assert not probe.close(float("nan"), 0.0)
    assert not probe.close(float("inf"), 1.0)
    assert probe.close(0.9900481, probe.sigmoid(4.6))


def test_the_gate_writes_the_home_directory_as_tilde(tmp_path, monkeypatch):
    gate = load("release-cuda-gate")
    monkeypatch.setenv("HOME", "/home/someone")
    monkeypatch.setenv("USERPROFILE", "/home/someone")
    report = tmp_path / "r.json"
    report.write_text('{"binary": "/home/someone/runner/runner", "model": "/home/someone/m.gguf"}\n')
    gate.redact_home(str(report))
    assert report.read_text() == '{"binary": "~/runner/runner", "model": "~/m.gguf"}\n'


def test_the_probe_writes_its_report_with_lf_on_every_host(tmp_path, monkeypatch):
    """The report is opened with newline="\\n": a Windows run wrote CRLF until
    it was, and the release gate's JSONs are committed byte for byte."""
    probe = load("cuda-act-probe")
    xs = probe.XS
    want = [probe.sigmoid(x) for x in xs]
    monkeypatch.setattr(probe, "run", lambda ptx: ("a card", 13020, {"k_q35_attn_gate": (list(want), want)}))
    out = tmp_path / "probe.json"
    monkeypatch.setattr("sys.argv", ["cuda-act-probe.py", "--out", str(out)])
    assert probe.main() == 0
    data = out.read_bytes()
    assert b"\r" not in data and data.endswith(b"}\n")
