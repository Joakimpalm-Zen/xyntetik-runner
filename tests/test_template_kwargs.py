"""chat_template_kwargs a renderer cannot honour (R2.13).

Runner renders each family with hand-written code, not by evaluating the file's
Jinja, so a key it does not know has nowhere to go. A reference gives a key to the
template as a variable. So:

  * the template reads the key: the client asked for something Runner will not
    render, and it is told, with a 400 naming the key, not served the off shape;
  * the template does not read it: the reference ignores it too, and so does Runner.

"Reads" means as a variable. The first case below is the shape of Sage-A1.5's
switch. The others are places a name appears without being read: a comment, plain
text, a string, an attribute of something else, and the template's own assignment.
"""
import json
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from server_probe import free_port  # noqa: E402

EXE = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
TAIL = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
        "<|im_start|>assistant\n")

READS = {
    "is_defined": "{%- set on = verification is defined and verification %}",
    "bare_if": "{% if verification %}x{% endif %}",
    "output": "{{ verification }}",
    "filter": "{{ verification|tojson }}",
    "whitespace_control": "{%-if   verification-%}x{%-endif-%}",
    "after_a_string": "{% if 'a' == 'a' and verification %}x{% endif %}",
    "escaped_quote_before": "{% if 'it\\'s' and verification %}x{% endif %}",
}
DOES_NOT_READ = {
    "absent": "",
    "comment": '{#- turned on with chat_template_kwargs={"verification": true} #}',
    "plain_text": "Say verification out loud.",
    "string": "{{ 'verification must be a boolean' }}",
    "double_quoted": '{{ "verification" }}',
    "attribute": "{% if message.verification %}x{% endif %}",
    "subscript": "{% if message['verification'] %}x{% endif %}",
    "longer_name": "{{ verification_mode }}{{ pre_verification }}",
    "own_set": "{% set verification = 1 %}",
    "namespace_field": "{% set ns = namespace(verification=0) %}",
}


def make_model(tmp, name, template):
    tpl = tmp / f"{name}.jinja"
    tpl.write_text(template + TAIL, encoding="utf-8")
    out = tmp / f"{name}.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), "--chat-template-file", str(tpl),
                    str(out)], check=True, cwd=ROOT, stdout=subprocess.PIPE)
    return out


class Server:
    def __init__(self, model, tmp):
        self.port = free_port()
        self.log = open(tmp / "server.log", "wb")
        self.p = subprocess.Popen([str(EXE), "-m", str(model), "--serve", "--port", str(self.port), "--gpu", "off",
                                   "-c", "512", "--no-tray"], cwd=ROOT, stdout=self.log, stderr=self.log)
        import time
        for _ in range(300):
            if self.p.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1).read()
                return
            except OSError:
                time.sleep(0.05)
        raise AssertionError((tmp / "server.log").read_text(errors="replace")[-600:])

    def post(self, path, body):
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(r, timeout=60) as f:
                return f.status, json.loads(f.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def close(self):
        self.p.kill()
        self.p.wait(timeout=30)
        self.log.close()


@pytest.fixture
def serve(tmp_path):
    if not EXE.exists():
        pytest.skip("runner binary not built")
    made = []

    def start(template):
        s = Server(make_model(tmp_path, "m", template), tmp_path)
        made.append(s)
        return s
    yield start
    for s in made:
        s.close()


def chat(kwargs):
    return {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 2, "temperature": 0,
            "chat_template_kwargs": kwargs}


@pytest.mark.parametrize("shape", READS)
def test_a_key_the_template_reads_is_refused_by_name(serve, shape):
    s = serve(READS[shape])
    code, body = s.post("/v1/chat/completions", chat({"verification": True}))
    assert code == 400, body
    assert "verification" in body["error"]["message"], body
    assert body["error"]["code"] == "unsupported_template_kwarg", body
    assert body["error"]["param"] == "chat_template_kwargs", body


@pytest.mark.parametrize("shape", DOES_NOT_READ)
def test_a_key_the_template_does_not_read_is_ignored(serve, shape):
    s = serve(DOES_NOT_READ[shape])
    code, with_key = s.post("/v1/chat/completions", chat({"verification": True}))
    assert code == 200, with_key
    code, without = s.post("/v1/chat/completions", chat({}))
    assert code == 200, without
    assert with_key["choices"][0]["message"] == without["choices"][0]["message"]


@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions", chat({"verification": True})),
    ("/v1/responses", {"input": "hello", "max_output_tokens": 2, "chat_template_kwargs": {"verification": True}}),
    ("/v1/messages", {"model": "m.gguf", "max_tokens": 2, "messages": [{"role": "user", "content": "hello"}],
                      "chat_template_kwargs": {"verification": True}}),
    ("/v1/messages/count_tokens", {"model": "m.gguf", "messages": [{"role": "user", "content": "hello"}],
                                   "chat_template_kwargs": {"verification": True}}),
    ("/v1/runner/contexts", {"id": "c1", "messages": [{"role": "user", "content": "hello"}],
                             "chat_template_kwargs": {"verification": True}}),
])
def test_every_surface_that_renders_refuses_it(serve, path, body):
    s = serve(READS["is_defined"])
    code, out = s.post(path, body)
    assert code == 400, out
    assert "verification" in json.dumps(out), out


def test_null_is_absent_as_it_is_for_every_known_key(serve):
    s = serve(READS["is_defined"])
    code, out = s.post("/v1/chat/completions", chat({"verification": None}))
    assert code == 200, out


@pytest.mark.parametrize("key,value", [
    ("enable_thinking", True),
    ("preserve_thinking", True),
    ("reasoning_effort", "low"),
    ("reasoning_strength", "low"),
    ("bare_recipients", True),
])
def test_a_key_runner_renders_is_not_refused_though_the_template_reads_it(serve, key, value):
    s = serve("{% if " + key + " %}x{% endif %}")
    code, out = s.post("/v1/chat/completions", chat({key: value}))
    assert code == 200, out


@pytest.mark.parametrize("key,value", [
    ("add_generation_prompt", True),
    ("messages", []),
    ("tools", []),
    ("bos_token", "x"),
    ("eos_token", "x"),
])
def test_a_value_the_renderer_itself_supplies_is_not_refused(serve, key, value):
    # nearly every template reads these; they are the render's inputs, not switches
    # a family adds, and a client that sends one was ignored before and still is
    s = serve("{{ bos_token }}{{ eos_token }}{% if add_generation_prompt %}x{% endif %}{{ tools }}")
    code, out = s.post("/v1/chat/completions", chat({key: value}))
    assert code == 200, out


def test_the_unknown_key_is_found_among_known_ones(serve):
    s = serve(READS["is_defined"])
    code, out = s.post("/v1/chat/completions", chat({"enable_thinking": False, "verification": True}))
    assert code == 400 and "verification" in out["error"]["message"], out
