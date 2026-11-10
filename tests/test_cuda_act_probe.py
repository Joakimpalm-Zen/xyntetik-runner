"""scripts/cuda-act-probe.py and scripts/release-cuda-gate.py without a device:
the PTX the probe loads is the one the binary embeds and holds the three
kernels with the parameter lists the probe passes; its comparison refuses a
NaN; the gate names the model and the binary without their trees, writes any
other home path in a report as ~/, and names the NVFP4 prefill read's prompt
with its sha256."""
import hashlib
import importlib.util
import os
import re
import types

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


def test_the_gate_names_the_model_and_binary_without_their_trees(tmp_path):
    """R6.44: a model given by an absolute path outside the checkout put
    ~/<dir>/qwen3-0.6b-q8_0.gguf in the 1.16.0 smoke report (home redaction
    keeps the tree), and the commit hook refused the report. The model is now
    named by its file name, and a file inside the checkout by its path in it."""
    gate = load("release-cuda-gate")
    model = tmp_path / "models" / "m.gguf"
    model.parent.mkdir()
    model.write_bytes(b"x")
    binary = os.path.join(ROOT, "runner")
    report = tmp_path / "r.json"
    report.write_text('{"generation_output": "loaded %s | llama", "binary": "%s"}\n' % (model, binary))
    gate.redact_paths(str(report), [str(model), binary])
    text = report.read_text()
    assert text == '{"generation_output": "loaded m.gguf | llama", "binary": "runner"}\n', text
    assert str(tmp_path) not in text


def test_the_gate_redacts_the_reports_it_writes(tmp_path, monkeypatch):
    """main() runs the redaction on each report the smoke and probe write."""
    gate = load("release-cuda-gate")
    model = tmp_path / "far" / "m.gguf"
    model.parent.mkdir()
    model.write_bytes(b"x")
    written = []

    def fake_run(argv):
        out = argv[argv.index("--out") + 1]
        with open(out, "w", encoding="utf-8") as f:
            f.write('{"generation_output": "loaded %s"}\n' % model)
        written.append(out)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    repo = tmp_path / "repo"   # a scratch checkout: the reports and the prompt fixture live under it
    (repo / "docs" / "compat-reports").mkdir(parents=True)
    (repo / "tests" / "fixtures").mkdir(parents=True)
    (repo / "tests" / "fixtures" / "nvfp4-prefill-prompt.txt").write_bytes(b"p")
    monkeypatch.setattr(gate, "ROOT", str(repo))
    monkeypatch.setattr("sys.argv", ["release-cuda-gate.py", "--binary", "r", "--model", str(model),
                                     "--version", "0.0.0", "--date", "2000-01-01"])
    assert gate.main() == 0
    assert len(written) == 2
    for out in written:
        assert open(out, encoding="utf-8").read() == '{"generation_output": "loaded m.gguf"}\n'


def test_the_nvfp4_prefill_prompt_is_a_named_fixture(monkeypatch, capsys):
    """A read names its prompt by bytes: 1.13.0's was named by length alone and
    its text is gone. Anchors from outside the gate: the sha256 is the one the
    1.14.0 report recorded, and the fixture is the shortest head of the corpus
    Qwen3.5-4B's own tokenizer.json encodes to 1,971 tokens."""
    name = "tests/fixtures/nvfp4-prefill-prompt.txt"
    sha = "a25ab76dfd7ba46f22a8e2c94a7ee0ad91dcf997336bc77786ae23a825136f3d"
    with open(os.path.join(ROOT, name), "rb") as f:
        data = f.read()
    assert hashlib.sha256(data).hexdigest() == sha and len(data) == 7598
    # text mode on purpose: the corpus is not -text, so a Windows checkout holds CRLF
    with open(os.path.join(ROOT, "tests", "fixtures", "gold-corpus-2k.txt"), encoding="utf-8") as f:
        corpus = f.read()
    assert data.decode("utf-8") == corpus[:7555]
    gate = load("release-cuda-gate")
    monkeypatch.setattr(gate.subprocess, "run", lambda argv: types.SimpleNamespace(returncode=0))
    monkeypatch.setattr("sys.argv", ["release-cuda-gate.py", "--binary", "r", "--model", "m",
                                     "--version", "0.0.0", "--date", "2000-01-01"])
    assert gate.main() == 0
    out = capsys.readouterr().out
    assert name in out and sha in out and "-f " in out and "RUNNER_CUDA_TC_NVFP4=0" in out


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
