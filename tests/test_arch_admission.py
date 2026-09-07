"""Architecture admission is an allowlist (RNR-004).

Tensor-name compatibility is not proof of mathematical compatibility, so an
unknown `general.architecture` must be refused rather than run through
llama-style math (which would emit plausible but silently wrong output). The
only escape is an explicit, clearly-labelled opt-in.
"""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _make_model(path, arch):
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-model.py", "--arch", arch, str(path)],
        check=True, cwd=ROOT,
    )


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def test_unknown_arch_is_refused(runner_bin, tmp_path):
    model = tmp_path / "mystery.gguf"
    _make_model(model, "mystery")
    proc = subprocess.run(
        [runner_bin, "-m", model, "-p", "hi", "-n", "1", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
    )
    assert proc.returncode != 0, "an unknown architecture must not load by default"
    err = proc.stderr.decode(errors="replace")
    assert "unsupported architecture 'mystery'" in err
    assert "refusing" in err


def test_opt_in_bypasses_admission_with_a_loud_warning(runner_bin, tmp_path):
    model = tmp_path / "mystery.gguf"
    _make_model(model, "mystery")
    proc = subprocess.run(
        [runner_bin, "-m", model, "-p", "hi", "-n", "1", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
        env={**__import__("os").environ, "RUNNER_ALLOW_UNKNOWN_ARCH": "1"},
    )
    # The opt-in gets past admission (the "mystery.*" geometry keys exist, so it
    # actually loads and runs); what matters is the refusal is gone and the
    # warning is unmistakable.
    err = proc.stderr.decode(errors="replace")
    assert "is UNSUPPORTED" in err
    assert "silently wrong" in err
    assert "refusing" not in err


def test_known_arch_still_loads(runner_bin, tmp_path):
    model = tmp_path / "llama.gguf"
    _make_model(model, "llama")
    proc = subprocess.run(
        [runner_bin, "-m", model, "-p", "hi", "-n", "1", "--gpu", "off", "--temp", "0"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")


# ---------------------------------------------------------------- gate audit
#
# The golden pass of 2026-09-07 found the runner numerically wrong on
# `stablelm`: it normalises with LayerNorm and the runner was applying
# RMSNorm, and its `stablelm2` pre-tokenizer had no row so a third of the
# corpus tokenized differently. Neither defect was exotic. Both would have
# been caught on the first day by the checks this repository already runs.
#
# They survived because `stablelm` was named in model_supported_archs() and
# in the README's architecture table while having NO row in
# tests/compatibility/models.json. No pinned file means no tokenizer
# differential, no greedy reference, no device identity check: the
# architecture was claimed and never gated.
#
# This test makes that structural gap visible instead of leaving it to be
# rediscovered. An architecture the runner admits either has a manifest row
# or is listed below with the reason it does not.

# Architectures admitted by name but deliberately not pinned, each with the
# reason. Shrinking this list is the work; adding to it needs a reason that
# survives being read out loud.
ARCH_GATE_EXEMPTIONS = {
    # Exercised through a pinned file that declares a different
    # general.architecture: every Mistral and SmolLM2 GGUF in the manifest
    # converts to arch "llama", so the code path is covered even though the
    # string is not. Keep until a file actually ships with these strings.
    "mistral": "covered by llama-arch files (mistral-7b-instruct-v0.3-q4_k_m)",
    "smollm": "covered by llama-arch files; no GGUF ships arch=smollm",
    # Genuinely ungated. These are claims with no pinned file behind them,
    # which is exactly the shape of the stablelm defect. Each needs a file
    # pinned in the manifest or removal from the admitted list.
    "stablelm": "UNGATED: fixed 2026-09-07 (R4.23), still needs a pinned file",
    "granitehybrid": "UNGATED: needs a pinned file (R4.23.3)",
    "nemotron_h": "UNGATED: needs a pinned file (R4.23.3)",
    "nemotron_h_moe": "UNGATED: needs a pinned file (R4.23.3)",
}


def _admitted_architectures():
    import re
    src = (ROOT / "src/model.c").read_text()
    m = re.search(r"static const char \*const arches\[\] = \{(.*?)\};", src, re.S)
    assert m, "could not find the admitted-architecture list in src/model.c"
    return re.findall(r'"([^"]+)"', m.group(1))


def _manifest_architectures():
    import json
    man = json.loads((ROOT / "tests/compatibility/models.json").read_text())
    return {row["architecture"] for row in man["models"]}


def test_every_admitted_architecture_is_gated_or_named_as_ungated():
    """No architecture is claimed in silence.

    Either a pinned file exercises it, or it appears in ARCH_GATE_EXEMPTIONS
    with the reason. A new architecture added to model_supported_archs()
    without either fails here, which is the check `stablelm` never had.
    """
    admitted = _admitted_architectures()
    gated = _manifest_architectures()
    unaccounted = [a for a in admitted
                   if a not in gated and a not in ARCH_GATE_EXEMPTIONS]
    assert not unaccounted, (
        "these architectures are admitted by model_supported_archs() but have "
        "no row in tests/compatibility/models.json and no entry in "
        "ARCH_GATE_EXEMPTIONS: " + ", ".join(unaccounted) + ". Pin a file for "
        "each, or add it to the exemption map with the reason it is safe.")


def test_exemption_list_does_not_outlive_its_reason():
    """An exemption for an architecture that later got a pinned file is stale."""
    gated = _manifest_architectures()
    stale = sorted(gated & set(ARCH_GATE_EXEMPTIONS))
    assert not stale, (
        "these architectures now have a manifest row, so their entry in "
        "ARCH_GATE_EXEMPTIONS is stale and should be deleted: "
        + ", ".join(stale))
