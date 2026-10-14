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


def test_proc_stat_counts_fields_after_the_command_name():
    """utime and stime are fields 14 and 15 of /proc/PID/stat, counted after
    the parenthesised command name, which may itself hold spaces and ')'."""
    line = ("4242 (run (ner) x) S 17 4242 4242 0 -1 4194304 1 0 0 0 "
            "731 102 0 0 20 0 9 0 12345 0 0\n")
    assert MOD.proc_stat_fields(line) == (17, 833)


def test_filetime_is_hundred_nanosecond_units():
    # 2**32 units of 100 ns is 429.4967296 s
    assert MOD.filetime_seconds(1, 0) == 429.4967296
    assert MOD.filetime_seconds(0, 10_000_000) == 1.0


def test_the_platform_sampler_is_chosen_and_names_itself():
    assert MOD.ps_stat is not None and MOD.ACCOUNTING
    import os
    s = MOD.ps_stat(os.getpid())
    assert s and s["cputime_s"] > 0 and s["rss_mb"] > 0 and s["procs"] >= 1
