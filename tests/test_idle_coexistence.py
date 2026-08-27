"""The reproducer behind a published claim, so its arithmetic is gated too.

README cites this script's numbers for idle coexistence and first-token
latency across engines. A TTFT that measures the wrong frame flatters whichever
engine happens to announce its turn earliest, which is exactly the kind of
comparison this project refuses to publish.
"""
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "idle_coexistence", ROOT / "scripts" / "idle_coexistence.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _frame(payload):
    return b"data: " + json.dumps(payload).encode() + b"\n"


def test_the_role_only_opening_frame_is_not_a_first_token():
    """Runner emits this one right after the response headers, before prefill
    (src/completion.c: "Emitting it up front ... keeps the contract independent
    of what the model generates"). Timing it measures connection setup."""
    opening = _frame({"choices": [{"index": 0, "delta": {"role": "assistant"},
                                   "finish_reason": None}]})
    assert not MOD.frame_carries_output(opening)


def test_a_content_delta_is_a_first_token():
    delta = _frame({"choices": [{"index": 0, "delta": {"content": "2"},
                                 "finish_reason": None}]})
    assert MOD.frame_carries_output(delta)
    reasoning = _frame({"choices": [{"index": 0,
                                     "delta": {"reasoning_content": "hm"}}]})
    assert MOD.frame_carries_output(reasoning)
    text = _frame({"choices": [{"index": 0, "text": "2"}]})
    assert MOD.frame_carries_output(text)


def test_terminators_and_noise_are_not_first_tokens():
    assert not MOD.frame_carries_output(b"data: [DONE]\n")
    assert not MOD.frame_carries_output(b": keepalive\n")
    assert not MOD.frame_carries_output(b"\n")
    assert not MOD.frame_carries_output(b"data: {not json\n")
    empty = _frame({"choices": [{"index": 0, "delta": {"content": ""},
                                 "finish_reason": "stop"}]})
    assert not MOD.frame_carries_output(empty)
