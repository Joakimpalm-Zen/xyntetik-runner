"""`runner --doctor` prints one diagnostic report (R16.6.2).

The report is what a user attaches to a question, so the tests hold the three
things that make it safe and useful to share: it is valid JSON on stdout with
nothing else there, it names what the process actually did (template, backend,
sampler), and it carries no prompt or reply text, and no path, unless asked.
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import find_runner  # noqa: E402


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    m = tmp_path_factory.mktemp("doctor") / "private-dir-name" / "model.gguf"
    m.parent.mkdir()
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return exe, m


def _doctor(exe, m, *extra):
    p = subprocess.run([exe, "-m", str(m), "--gpu", "off", "--no-tray", *extra],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    return p.returncode, json.loads(p.stdout), p.stderr.decode(errors="replace")


def test_report_names_what_ran_and_holds_no_content(model):
    exe, m = model
    rc, d, _ = _doctor(exe, m, "--doctor", "--chat-template", "chatml")
    assert d["schema"] == "xyntetik.runner.doctor.v1"
    assert d["runner"]["version"]
    assert d["model"]["file"] == "model.gguf"
    assert d["placement"]["requested"] == "off" and d["placement"]["backend"] == "cpu"
    assert d["placement"]["gpu_layers"] == 0
    assert d["template"]["name"] == "chatml" and d["template"]["recognised"] is True
    assert d["sampling"]
    assert d["probe"]["ran"] is True and d["timings"]["generated_tokens"] >= 0
    assert d["verdict"] in ("ok", "degraded", "broken")
    assert rc == (2 if d["verdict"] == "broken" else 0)
    # nothing a user typed, nothing the model said, and no directory name
    text = json.dumps(d)
    assert "question" not in d["probe"] and "reply" not in d["probe"]
    assert "What is 2+2" not in text and "private-dir-name" not in text
    for f in d["findings"]:
        assert f["severity"] in ("degraded", "broken")
        assert f["code"] and f["detail"] and f["next_step"]


def test_text_is_included_only_when_asked(model):
    exe, m = model
    _, d, _ = _doctor(exe, m, "--doctor-include-text", "--chat-template", "chatml")
    assert d["probe"]["question"].startswith("What is 2+2")
    assert "reply" in d["probe"]
    assert "read before sharing" in d["shareable"]


def test_an_unrecognised_template_is_a_broken_verdict(model):
    exe, m = model
    rc, d, _ = _doctor(exe, m, "--doctor")
    if d["template"]["recognised"]:
        pytest.skip("the fixture's template is recognised on this build")
    assert rc == 2 and d["verdict"] == "broken"
    assert "template_not_recognised" in [f["code"] for f in d["findings"]]
