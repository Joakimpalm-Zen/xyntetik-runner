"""Known chat_template_kwargs keys on the families that ignored them (R2.14).

A family's reference reacts to these keys, and Runner's renderer did not: the
client got the default shape and no word of it. Each is honoured here or refused.

The expected bytes are the published templates', by hand, not Runner's output:
  lfm2: `keep_thinking = preserve_thinking or ...` writes the block on past turns
  qwen38: a thought is kept when `preserve_thinking is undefined or ... true or`
    after the last query, so an explicit false drops the earlier ones
  apertus: `{{ developer_token + 'Deliberation: ' }}` then enabled or disabled
  granite42: `content + '\\n\\n{reasoning effort: low}'` on the last user turn

Rendered through the real request path by template-conformance-render.
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / ("template-conformance-render.exe" if sys.platform == "win32" else "template-conformance-render")
HISTORY = [
    {"role": "user", "content": "First?"},
    {"role": "assistant", "content": "One.", "reasoning_content": "first thought"},
    {"role": "user", "content": "Second?"},
    {"role": "assistant", "content": "Two.", "reasoning_content": "second thought"},
    {"role": "user", "content": "Third?"},
]


def render(template, kwargs=None, chat_template=None, messages=HISTORY):
    if not BIN.exists():
        pytest.skip("make template-conformance-render first")
    req = {"model": "x", "messages": messages}
    if kwargs is not None:
        req["chat_template_kwargs"] = kwargs
    case = {"id": "c", "template": template, "request": req}
    if chat_template:
        case["chat_template"] = chat_template
    p = subprocess.run([str(BIN)], input=json.dumps({"cases": [case]}), capture_output=True, text=True,
                       encoding="utf-8", cwd=ROOT)
    assert p.returncode == 0, p.stderr
    r = json.loads(p.stdout)["results"][0]
    return r["prompt"], r.get("error")


@pytest.mark.parametrize("template", ["lfm2", "lfm2-24b"])
def test_lfm2_keeps_past_thoughts_when_asked(template):
    off, _ = render(template)
    on, err = render(template, {"preserve_thinking": True})
    assert err is None
    assert "first thought" not in off
    assert "<think>first thought</think>" in on


def test_lfm2_false_is_its_default():
    assert render("lfm2", {"preserve_thinking": False}) == render("lfm2")


def test_qwen38_keeps_thoughts_by_default_and_false_drops_them():
    base, _ = render("qwen38")
    assert "first thought" in base and "second thought" in base
    assert render("qwen38", {"preserve_thinking": True}) == render("qwen38")
    off, err = render("qwen38", {"preserve_thinking": False})
    assert err is None
    assert "first thought" not in off and "second thought" not in off
    # the reference writes no block at all on those turns, not an empty one
    assert "<|im_start|>assistant\nOne.<|im_end|>" in off


def test_qwen38_false_keeps_a_thought_after_the_last_query():
    msgs = HISTORY + [{"role": "assistant", "content": "Three.", "reasoning_content": "third thought"}]
    off, _ = render("qwen38", {"preserve_thinking": False}, messages=msgs)
    assert "first thought" not in off and "third thought" in off


def test_apertus_deliberation():
    assert "Deliberation: disabled\n" in render("apertus")[0]
    assert "Deliberation: disabled\n" in render("apertus", {"enable_thinking": False})[0]
    assert "Deliberation: enabled\n" in render("apertus", {"enable_thinking": True})[0]


@pytest.mark.parametrize("messages", [HISTORY, [{"role": "system", "content": "S"}] + HISTORY])
def test_granite42_low_effort_goes_on_the_last_user_turn(messages):
    low, err = render("granite42", {"reasoning_effort": "low"}, messages=messages)
    assert err is None
    assert low.count("{reasoning effort: low}") == 1
    assert "Third?\n\n{reasoning effort: low}<|im_end|>" in low
    assert render("granite42", {"reasoning_effort": "high"}, messages=messages) == render("granite42", messages=messages)


def test_spark_refuses_xhigh_as_its_template_does():
    _, err = render("spark", {"reasoning_effort": "xhigh"})
    assert err and "reasoning_effort" in err


def test_generation_prompt_false_is_refused_where_the_template_reads_it():
    tpl = "{% if add_generation_prompt %}x{% endif %}"
    _, err = render("chatml", {"add_generation_prompt": False}, chat_template=tpl)
    assert err and "add_generation_prompt" in err
    assert render("chatml", {"add_generation_prompt": True}, chat_template=tpl)[1] is None


def test_generation_prompt_false_is_ignored_where_the_template_does_not_read_it():
    # llama2's and mistral's templates have no such switch, so the reference ignores it
    assert render("chatml", {"add_generation_prompt": False}, chat_template="{{ messages }}")[1] is None


@pytest.mark.parametrize("template,key,value", [
    ("chatml", "enable_thinking", True),
    ("chatml", "preserve_thinking", True),
    ("chatml", "reasoning_effort", "low"),
    ("chatml", "reasoning_strength", "low"),
    ("qwen36", "reasoning_effort", "low"),
    ("lfm2-8b", "preserve_thinking", True),
])
def test_a_known_key_this_renderer_ignores_is_refused_where_the_template_reads_it(template, key, value):
    _, err = render(template, {key: value}, chat_template="{% if " + key + " %}x{% endif %}")
    assert err and key in err


@pytest.mark.parametrize("template,key,value", [
    ("chatml", "enable_thinking", True),
    ("chatml", "reasoning_effort", "low"),
    ("llama3", "enable_thinking", False),
    ("qwen36", "reasoning_effort", "low"),
])
def test_a_known_key_is_ignored_where_the_template_does_not_read_it(template, key, value):
    assert render(template, {key: value}, chat_template="{{ messages }}")[1] is None


@pytest.mark.parametrize("template,key,value", [
    ("chatml-think", "enable_thinking", True),
    ("qwen36", "preserve_thinking", True),
    ("qwen38", "preserve_thinking", False),
    ("qwen38", "reasoning_effort", "low"),
    ("muse", "reasoning_strength", "low"),
    ("spark", "reasoning_effort", "low"),
    ("granite42", "reasoning_effort", "low"),
    ("apertus", "enable_thinking", True),
    ("lfm2", "preserve_thinking", True),
])
def test_a_key_the_renderer_honours_is_served(template, key, value):
    assert render(template, {key: value}, chat_template="{% if " + key + " %}x{% endif %}")[1] is None
