"""Intentional API refusals cannot hide an unavailable or accepted render."""
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("template_refusal", ROOT / "scripts/template-conformance.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_only_the_named_api_error_with_a_real_oracle_is_accepted():
    dev = SimpleNamespace(runner_error="only text is accepted")
    assert MOD.refusal_matches(dev, "image placeholder", ("runner", "400"),
                               "content type image: only text is accepted")
    assert not MOD.refusal_matches(dev, None, ("reference", "template error"),
                                   "only text is accepted")
    assert not MOD.refusal_matches(dev, "image placeholder", None, None)
    assert not MOD.refusal_matches(dev, "image placeholder", ("runner", "400"),
                                   "unrelated invalid tool arguments")


def test_refusal_entries_cannot_match_any_error_via_an_empty_needle():
    with pytest.raises(SystemExit, match="requires runner_error"):
        MOD.Deviation({"id": "inkling2/image", "family": "inkling2",
                       "kind": "refusal", "case": "image"})
