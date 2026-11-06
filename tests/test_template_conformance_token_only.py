"""The template-conformance gate's token-only deviation kind (R2.4.11).

A case may render byte-identically on both sides yet tokenize differently on
purpose: Runner reads a control token only in the template's own bytes, so one
the caller spelled in its content is text (tests/test_prompt_marks.c), where
the reference tokenizer reads it as the control token. The kind accounts for
exactly that and nothing else; these hold it to its three conditions."""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
_spec = importlib.util.spec_from_file_location(
    "template_conformance", os.path.join(ROOT, "scripts", "template-conformance.py"))
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)

# ids 0, 1, 4, 5 are text; 2 and 3 are added tokens (USER_DEFINED, as Qwen's
# <think> pair is stored), 6 a CONTROL token
TOKENS = ["a", "b", "<think>", "</think>", "<th", "ink>", "<|im_start|>"]
TYPES = [1, 1, 4, 4, 1, 1, 3]
CALLER = "<think>b</think> and the rest of what the caller sent"


def test_a_caller_spelled_control_token_read_as_text_is_accepted():
    # the reference reads <think> and </think> as one token each; Runner's
    # side carries their spellings as text around the same text token
    ok, why = tc.token_only_verdict(True, [0, 2, 1, 3], [0, 4, 5, 1, 4, 5],
                                    TOKENS, TYPES, CALLER)
    assert ok, why
    assert "<think>" in why and "</think>" in why


def test_the_kind_is_refused_when_the_text_differs():
    ok, why = tc.token_only_verdict(False, [0, 2, 1, 3], [0, 4, 5, 1, 4, 5],
                                    TOKENS, TYPES, CALLER)
    assert not ok and "not byte-identical" in why


def test_the_kind_is_refused_when_a_differing_reference_id_is_not_a_control_token():
    ok, why = tc.token_only_verdict(True, [0, 1, 3], [0, 4, 3],
                                    TOKENS, TYPES, CALLER)
    assert not ok and "is not a control token" in why


def test_the_kind_is_refused_when_the_caller_never_spelled_the_token():
    # <|im_start|> is a control token, but this caller's content never spells it
    ok, why = tc.token_only_verdict(True, [0, 6, 1], [0, 4, 5, 1],
                                    TOKENS, TYPES, CALLER)
    assert not ok and "not spelled in the caller's content" in why


def test_the_kind_is_refused_when_runner_has_ids_the_reference_lacks():
    ok, why = tc.token_only_verdict(True, [0, 1], [0, 4, 1],
                                    TOKENS, TYPES, CALLER)
    assert not ok and "no reference counterpart" in why


def test_a_token_only_entry_names_a_case_and_carries_no_patch():
    base = {"id": "fam/x", "family": "fam", "kind": "token-only", "why": "w",
            "cite": {"file": "tests/test_prompt_marks.c", "lines": "1",
                     "anchor": "x"}}
    assert tc.Deviation(dict(base, case="c")).case == "c"
    with pytest.raises(SystemExit):
        tc.Deviation(base)
    with pytest.raises(SystemExit):
        tc.Deviation(dict(base, case="c", patch={"op": "builtin", "name": "n"}))


def test_a_token_only_entry_must_cite_the_prompt_marks_rule():
    d = tc.Deviation({"id": "fam/x", "family": "fam", "kind": "token-only",
                      "case": "c", "why": "w",
                      "cite": {"file": "src/template.c", "lines": "1",
                               "anchor": "x"}})
    rot = tc.check_citations({"fam": [d]}, ["fam"])
    assert any("must cite tests/test_prompt_marks.c" in r for r in rot)


def test_caller_text_is_every_string_the_case_carries():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "p1"}]},
            {"role": "assistant", "content": "c", "reasoning_content": "r",
             "tool_calls": [{"function": {"arguments": "{\"k\": \"v\"}"}}]}]
    t = tc.caller_text(msgs)
    for s in ("p1", "c", "r", "{\"k\": \"v\"}"):
        assert s in t
