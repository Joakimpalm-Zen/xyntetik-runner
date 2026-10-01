#!/usr/bin/env python3
"""Does every admitted model family still load and work on this build?

`stress-models.py` answers load / CPU==GPU / best settings for a shelf of
files. This sweep asks the question a batch of engine changes raises: with a
REAL model of each family, does every surface still behave, on the usual
request and on the edges? Fixtures cannot answer that: their vocabulary is
bytes and their weights are tiny, and several defects found on 2026-10-01
only exist on real models.

Per model, on each backend asked for (cpu, gpu):

  CLI     load and greedy generation; GPU text equals CPU text (prefix);
          a seeded sampled run is reproducible; a transcript replays VERIFIED;
          a suspended and resumed generation images byte-identically (CPU);
          hostile prompts (empty, control bytes, long unicode) do not crash.
  serve   chat, completions, Responses and Messages, buffered and streamed
          (streamed text equals buffered at temperature 0, valid UTF-8, no
          U+FFFD the buffered answer lacks); max_tokens 1; stop sequences;
          JSON mode and a strict schema; a required tool call, buffered and
          streamed, with a non-ASCII argument; a tool result turn; a prompt
          near the context limit; a prompt over it (a 400, not a crash); a
          repeated request (cached, same text); concurrent requests on two
          slots; logprobs, echo and prompt_logprobs; named context, rerank,
          decide, provenance, embeddings; malformed requests; a client that
          hangs up mid-stream; a served receipt that replays VERIFIED (CPU).

Verdicts: PASS, FAIL (the engine did something wrong), SKIP (refused with a
stated reason, or not applicable), NOTE (well-formed, but the MODEL's answer
is off: not an engine verdict). One JSON per model under --out-dir, and a
summary table on stdout. --skip-done resumes an interrupted pass.

  family-sweep.py --roster roster.json --out-dir out --backends cpu,gpu
  family-sweep.py --model a.gguf --model b.gguf --backends cpu --quick

A roster is a JSON list of {"family": "...", "path": "...", "ctx": 4096,
"backends": ["cpu","gpu"], "note": "..."}; only `path` is required.
"""

import argparse
import concurrent.futures as cf
import hashlib
import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer, find_runner  # noqa: E402

UNI = "Åsa gick ut på isen, 日本 🙂 naïve"
REFUSALS = ("unsupported architecture", "needs sink-aware attention",
            "cannot be imaged", "does not cover", "refus", "not supported",
            "unsupported")
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "unit": {"type": "string",
                                              "enum": ["C", "F"]}},
        "required": ["city"]}}}
SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"}, "age": {"type": "integer"},
    "tags": {"type": "array", "items": {"type": "string"}}},
    "required": ["name", "age"], "additionalProperties": False}


class Skip(Exception):
    pass


class Note(Exception):
    pass


def sh(cmd, timeout, env=None):
    t0 = time.time()
    try:
        p = subprocess.run([str(c) for c in cmd], capture_output=True,
                           timeout=timeout, env=env)
        return p.returncode, p.stdout, p.stderr.decode("utf-8", "replace"), \
            time.time() - t0
    except subprocess.TimeoutExpired as e:
        return None, e.stdout or b"", (e.stderr or b"").decode("utf-8", "replace") \
            + "\n[timed out]", time.time() - t0


def refused(err):
    low = err.lower()
    return next((r for r in REFUSALS if r in low), None)


class Bench:
    """One model on one backend: collects verdicts."""

    def __init__(self, runner, model, ctx, backend, threads, timeout, quick):
        self.runner, self.model, self.ctx = runner, model, ctx
        self.backend, self.threads = backend, threads
        self.timeout, self.quick = timeout, quick
        self.rows = []
        self.gpu_args = ["--gpu", "off"] if backend == "cpu" else ["--gpu", "auto"]
        self.common = [*self.gpu_args, "-c", str(ctx)]
        if threads:
            self.common += ["-t", str(threads)]

    def check(self, name, fn):
        t0 = time.time()
        row = {"check": name, "backend": self.backend}
        try:
            detail = fn()
            row.update(verdict="PASS", detail=detail)
        except Skip as e:
            row.update(verdict="SKIP", detail=str(e))
        except Note as e:
            row.update(verdict="NOTE", detail=str(e))
        except AssertionError as e:
            row.update(verdict="FAIL", detail=str(e)[:1500])
        except Exception as e:   # the harness or the server died: that is a finding
            row.update(verdict="FAIL", detail=(type(e).__name__ + ": " + str(e)
                                               + "\n" + traceback.format_exc()[-600:])[:1500])
        row["seconds"] = round(time.time() - t0, 1)
        self.rows.append(row)
        print(f"    {row['verdict']:4} {self.backend:3} {name}"
              + ("" if row["verdict"] == "PASS" else f": {str(row['detail'])[:200]}"),
              flush=True)
        return row["verdict"] == "PASS"

    # ------------------------------------------------------------- CLI
    def cli(self, *args, timeout=None):
        return sh([self.runner, "-m", self.model, *self.common, *args],
                  timeout or self.timeout)

    def cli_greedy(self):
        rc, out, err, _ = self.cli("-p", "The capital of France is", "-n", "24",
                                   "--temp", "0")
        if rc != 0:
            r = refused(err)
            if r:
                raise Skip(f"refused at load ({r}): " + err.strip()[-300:])
            assert False, f"exit {rc}: {err[-800:]}"
        assert out.strip(), "no output"
        out.decode("utf-8")   # valid UTF-8 or this raises
        for bad in ("falling back to CPU", "forward failed", "launch failed"):
            assert bad not in err, f"{bad}: {err[-600:]}"
        self.greedy_text = out
        return out.decode()[:120]

    def cli_seeded(self):
        a = self.cli("-p", "Once upon a time", "-n", "24", "--temp", "0.8", "-s", "7")
        b = self.cli("-p", "Once upon a time", "-n", "24", "--temp", "0.8", "-s", "7")
        assert a[0] == 0 and b[0] == 0, a[2][-400:] + b[2][-400:]
        assert a[1] == b[1], "the same seed gave two different texts"
        c = self.cli("-p", "Once upon a time", "-n", "24", "--temp", "0.8", "-s", "8")
        if c[1] == a[1]:
            raise Note("another seed gave the same text (a near-greedy model?)")

    def cli_transcript(self, tmp):
        rec = tmp / f"t-{self.backend}.json"
        rc, _, err, _ = self.cli("-p", "The river rose in the night and", "-n", "24",
                                 "--temp", "0.7", "-s", "11", "--transcript", rec)
        assert rc == 0, f"recording: exit {rc}: {err[-600:]}"
        r = json.loads(rec.read_text())
        o = r["output"]
        assert len(o["tokens"]) == o["n"], f"n {o['n']} but {len(o['tokens'])} tokens"
        if o["n"] == 0:
            raise Note("the model ended the turn at once; nothing to replay")
        rc, _, err, _ = sh([self.runner, "-m", self.model, "--verify", rec,
                            *(["-t", str(self.threads)] if self.threads else [])],
                           self.timeout)
        assert rc == 0 and "VERIFIED" in err, f"exit {rc}: {err[-600:]}"
        return err.strip().splitlines()[-1][:160]

    def cli_session(self, tmp):
        a, b1, b2 = (tmp / f"{n}-{self.backend}.img" for n in ("a", "b1", "b2"))
        gen = ["-p", "The river", "-s", "5", "--temp", "0.8", "--ignore-eos"]
        thr = ["-t", str(self.threads)] if self.threads else []
        res = [self.runner, "-m", self.model, *thr]   # the image fixes -c
        base = [*res, "-c", str(self.ctx)]
        rc, _, err, _ = sh([*base, *gen, "-n", "16", "--session-out", a], self.timeout)
        if rc != 0:
            r = refused(err)
            if r:
                raise Skip(err.strip()[-200:])
            assert False, err[-600:]
        rc, _, err, _ = sh([*base, *gen, "-n", "16", "--suspend-after", "8",
                            "--session-out", b1], self.timeout)
        assert rc == 0, err[-600:]
        rc, _, err, _ = sh([*res, "--resume", b1, "--session-out", b2], self.timeout)
        assert rc == 0, err[-600:]
        assert a.read_bytes() == b2.read_bytes(), \
            "a suspended and resumed generation imaged differently"
        for f in (a, b1, b2):
            f.unlink()

    def cli_hostile(self):
        prompts = {"empty": "", "controls": "a\tb\x01\x02\x7f c",
                   "unicode": (UNI + " ") * 40, "only-space": "   \n\n  "}
        notes = []
        for name, p in prompts.items():
            rc, out, err, _ = self.cli("-p", p, "-n", "4", "--temp", "0")
            assert rc is not None, f"{name}: timed out"
            assert rc in (0, 1), f"{name}: exit {rc} (a signal or an abort): {err[-400:]}"
            if rc == 1:
                assert "error" in err.lower(), f"{name}: exit 1 without saying why"
                notes.append(f"{name}: refused")
            else:
                out.decode("utf-8", "replace")
        return "; ".join(notes) or None

    # ----------------------------------------------------------- serve
    def req(self, srv, method, path, payload=None, timeout=None, raw=False):
        c = http.client.HTTPConnection("127.0.0.1", srv.port,
                                       timeout=timeout or self.timeout)
        body = None if payload is None else (
            payload if isinstance(payload, bytes) else json.dumps(payload).encode())
        c.request(method, path, body=body,
                  headers={"Content-Type": "application/json"} if body else {})
        r = c.getresponse()
        data = r.read()
        c.close()
        if raw:
            return r.status, data
        try:
            return r.status, json.loads(data)
        except ValueError:
            return r.status, {"_raw": data[:400].decode("utf-8", "replace")}

    def sse(self, srv, path, payload):
        st, data = self.req(srv, "POST", path, {**payload, "stream": True}, raw=True)
        assert st == 200, f"stream status {st}: {data[:300]}"
        text = data.decode("utf-8")   # the wire is UTF-8 or this raises
        ev = []
        for line in text.splitlines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                ev.append(json.loads(line[6:]))
        return ev

    def chat(self, srv, content, **kw):
        st, d = self.req(srv, "POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": 48, **kw})
        assert st == 200, f"{st}: {json.dumps(d)[:400]}"
        return d

    def s_basic(self, srv):
        d = self.chat(srv, "Say hello in five words.")
        ch = d["choices"][0]
        assert ch["finish_reason"] in ("stop", "length"), ch["finish_reason"]
        u = d["usage"]
        assert u["prompt_tokens"] > 0 and u["completion_tokens"] > 0
        assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
        assert (ch["message"].get("content") or "").strip() or \
            ch["message"].get("reasoning_content"), "an empty turn"
        return (ch["message"].get("content") or "")[:80]

    def s_stream_equals_buffered(self, srv):
        p = {"messages": [{"role": "user", "content":
                           f"Repeat exactly: {UNI}"}],
             "temperature": 0, "max_tokens": 64, "cache_prompt": False}
        st, d = self.req(srv, "POST", "/v1/chat/completions", p)
        assert st == 200, d
        buf = d["choices"][0]["message"].get("content") or ""
        ev = self.sse(srv, "/v1/chat/completions", p)
        got = "".join(c["delta"].get("content") or "" for e in ev
                      for c in e.get("choices", []))
        assert got == buf, f"streamed {got!r} != buffered {buf!r}"
        if "�" in got:
            raise Note("the model itself produced U+FFFD (buffered agrees)")
        if UNI not in buf:
            raise Note(f"did not repeat the text: {buf[:80]!r}")

    def s_limits(self, srv):
        d = self.chat(srv, "Count from one to fifty.", max_tokens=1)
        assert d["usage"]["completion_tokens"] <= 1, d["usage"]
        assert d["choices"][0]["finish_reason"] in ("length", "stop")
        st, d = self.req(srv, "POST", "/v1/completions", {
            "prompt": "one two three four five six seven", "max_tokens": 32,
            "temperature": 0, "stop": ["nine", " ten"]})
        assert st == 200, d
        t = d["choices"][0]["text"]
        assert "nine" not in t and " ten" not in t, f"a stop sequence was emitted: {t!r}"

    def s_json(self, srv):
        d = self.chat(srv, "Return a JSON object describing a cat.",
                      response_format={"type": "json_object"}, max_tokens=200)
        txt = d["choices"][0]["message"]["content"]
        try:
            json.loads(txt)   # a legal document, complete or closed at the ceiling
        except (ValueError, TypeError):
            assert False, "json_object content is not a JSON document: " + \
                json.dumps(d["choices"][0], ensure_ascii=False)[:700]
        d = self.chat(srv, "Describe a person named Ada who is 36.", max_tokens=200,
                      response_format={"type": "json_schema", "json_schema": {
                          "name": "person", "strict": True, "schema": SCHEMA}})
        try:
            v = json.loads(d["choices"][0]["message"]["content"])
        except (ValueError, TypeError):
            assert False, "json_schema content is not a JSON document: " + \
                json.dumps(d["choices"][0], ensure_ascii=False)[:700]
        assert isinstance(v, dict) and isinstance(v.get("name"), str) and \
            isinstance(v.get("age"), int) and not isinstance(v.get("age"), bool), v
        assert set(v) <= {"name", "age", "tags"}, v
        return json.dumps(v, ensure_ascii=False)[:120]

    def _tool_args(self, d):
        tc = d["choices"][0]["message"].get("tool_calls")
        assert tc, f"no tool call: {json.dumps(d['choices'][0])[:300]}"
        assert tc[0]["function"]["name"] == "get_weather", tc
        return json.loads(tc[0]["function"]["arguments"])

    def s_tools(self, srv):
        p = {"messages": [{"role": "user", "content":
                           "What is the weather in Åsa-Östersund, in C?"}],
             "tools": [WEATHER], "tool_choice": "required", "temperature": 0,
             "max_tokens": 200, "cache_prompt": False}
        st, d = self.req(srv, "POST", "/v1/chat/completions", p)
        assert st == 200, d
        a = self._tool_args(d)
        assert isinstance(a.get("city"), str), a
        ev = self.sse(srv, "/v1/chat/completions", p)
        args = "".join(t["function"].get("arguments", "") for e in ev
                       for c in e.get("choices", [])
                       for t in (c["delta"].get("tool_calls") or []))
        sa = json.loads(args)
        assert sa == a, f"streamed arguments {sa} != buffered {a}"
        if "�" in args and "�" not in json.dumps(a, ensure_ascii=False):
            assert False, "U+FFFD only on the stream"
        if "Å" not in a["city"] and "sa" in a["city"]:
            raise Note(f"the model wrote the city as {a['city']!r}")
        # the tool result turn
        st, d2 = self.req(srv, "POST", "/v1/chat/completions", {
            "messages": [p["messages"][0],
                         {"role": "assistant", "content": None, "tool_calls": [
                             {"id": "call_1", "type": "function", "function": {
                                 "name": "get_weather",
                                 "arguments": json.dumps(a, ensure_ascii=False)}}]},
                         {"role": "tool", "tool_call_id": "call_1",
                          "content": "{\"temp\": -3, \"unit\": \"C\"}"}],
            "tools": [WEATHER], "temperature": 0, "max_tokens": 64})
        assert st == 200, d2
        st, d3 = self.req(srv, "POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "tools": [WEATHER], "tool_choice": "auto", "temperature": 0,
            "max_tokens": 64})
        assert st == 200, d3
        return json.dumps(a, ensure_ascii=False)

    def _count(self, srv, text):
        st, d = self.req(srv, "POST", "/v1/completions", {
            "prompt": text, "max_tokens": 1, "temperature": 0})
        return st, d

    def s_context(self, srv):
        word = " river"
        st, d = self._count(srv, "The" + word * 50)
        assert st == 200, d
        per = max(1, (d["usage"]["prompt_tokens"] - 1) // 50)
        near = int(self.ctx * 0.8 / per)
        st, d = self._count(srv, "The" + word * near)
        assert st == 200, f"a prompt at 80% of the context: {st} {json.dumps(d)[:300]}"
        n_near = d["usage"]["prompt_tokens"]
        assert n_near < self.ctx
        st, d = self._count(srv, "The" + word * int(self.ctx * 1.3 / per + 50))
        assert st == 400, f"a prompt over the context answered {st}, not 400"
        assert "context" in json.dumps(d).lower(), d
        st, h = self.req(srv, "GET", "/health")
        assert st == 200, "the server did not survive an oversized prompt"
        return f"{n_near} of {self.ctx} tokens prefilled; overflow refused"

    def s_reuse(self, srv):
        p = {"prompt": "A short history of the printing press. " * 6,
             "max_tokens": 16, "temperature": 0}
        st, a = self.req(srv, "POST", "/v1/completions", p)
        st2, b = self.req(srv, "POST", "/v1/completions", p)
        assert st == 200 and st2 == 200, (a, b)
        cached = b["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        assert cached > 0, (f"the repeated prompt reused nothing: {b['usage']}; "
                            f"reuse {b['runner_telemetry'].get('prompt_reuse')}")
        if a["choices"][0]["text"] != b["choices"][0]["text"]:
            raise Note("the repeated request's greedy text differs from the cold "
                       "one (a near-tie moved by the reused KV's last bits): "
                       f"{a['choices'][0]['text']!r} vs {b['choices'][0]['text']!r}")
        return f"{cached} cached"

    def s_concurrent(self, srv):
        def one(i):
            return self.req(srv, "POST", "/v1/chat/completions", {
                "messages": [{"role": "user", "content": f"Name {i + 2} fruits."}],
                "temperature": 0, "max_tokens": 24})
        with cf.ThreadPoolExecutor(4) as ex:
            res = list(ex.map(one, range(4)))
        bad = [(s, json.dumps(d)[:200]) for s, d in res if s != 200]
        assert not bad, bad
        again = [one(i) for i in range(4)]
        diff = [i for i in range(4) if
                again[i][1]["choices"][0]["message"].get("content") !=
                res[i][1]["choices"][0]["message"].get("content")]
        if diff:
            raise Note(f"requests {diff} answered differently alone than "
                       "alongside others (greedy near-tie)")

    def s_logprobs(self, srv):
        st, d = self.req(srv, "POST", "/v1/completions", {
            "prompt": "The capital of France is", "max_tokens": 4,
            "temperature": 0, "logprobs": 3})
        assert st == 200, d
        lp = d["choices"][0]["logprobs"]
        n = len(lp["tokens"])
        assert n == len(lp["token_logprobs"]) == len(lp["top_logprobs"]) >= 1
        assert all(x <= 1e-6 for x in lp["token_logprobs"]), lp["token_logprobs"]
        st, d = self.req(srv, "POST", "/v1/completions", {
            "prompt": "The capital of France is", "max_tokens": 0, "echo": True,
            "logprobs": 2, "temperature": 0})
        assert st == 200, d
        e = d["choices"][0]["logprobs"]
        assert e["token_logprobs"][0] is None and len(e["tokens"]) >= 3, e
        st, d = self.req(srv, "POST", "/v1/completions", {
            "prompt": "The capital of France is", "max_tokens": 1,
            "prompt_logprobs": 2, "temperature": 0})
        assert st == 200, d
        pl = d["choices"][0]["prompt_logprobs"]
        assert pl[0] is None and all(isinstance(x, dict) for x in pl[1:]), pl[:3]
        st, c = self.req(srv, "POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4,
            "temperature": 0, "logprobs": True, "top_logprobs": 2})
        assert st == 200 and c["choices"][0]["logprobs"]["content"], c

    def s_surfaces(self, srv):
        st, d = self.req(srv, "POST", "/v1/responses", {
            "input": "Say hi.", "max_output_tokens": 32, "temperature": 0})
        assert st == 200 and d["object"] == "response" and d["output"], d
        buf = "".join(c["text"] for o in d["output"] if o["type"] == "message"
                      for c in o["content"])
        ev = self.sse(srv, "/v1/responses", {
            "input": "Say hi.", "max_output_tokens": 32, "temperature": 0})
        got = "".join(e["delta"] for e in ev
                      if e.get("type") == "response.output_text.delta")
        assert got == buf, f"responses stream {got!r} != buffered {buf!r}"
        assert any(e.get("type") == "response.completed" for e in ev) or \
            any(e.get("type") == "response.incomplete" for e in ev), "no terminal event"
        st, d = self.req(srv, "POST", "/v1/messages", {
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 32, "temperature": 0})
        assert st == 200 and d["type"] == "message" and d["content"], d
        buf = "".join(b["text"] for b in d["content"] if b["type"] == "text")
        ev = self.sse(srv, "/v1/messages", {
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 32, "temperature": 0})
        got = "".join(e["delta"]["text"] for e in ev
                      if e.get("type") == "content_block_delta"
                      and e["delta"].get("type") == "text_delta")
        assert got == buf, f"messages stream {got!r} != buffered {buf!r}"
        assert any(e.get("type") == "message_stop" for e in ev), "no message_stop"
        st, d = self.req(srv, "POST", "/v1/messages", {
            "messages": [{"role": "user", "content": "Weather in Oslo?"}],
            "max_tokens": 200, "temperature": 0, "tool_choice": {"type": "any"},
            "tools": [{"name": "get_weather", "input_schema":
                       WEATHER["function"]["parameters"]}]})
        assert st == 200, d
        tu = [b for b in d["content"] if b["type"] == "tool_use"]
        assert tu and isinstance(tu[0]["input"].get("city"), str), d["content"]

    def s_runner_routes(self, srv):
        for path in ("/health", "/v1/models", "/v1/capabilities", "/metrics",
                     "/v1/runner/prefix-cache", "/v1/runner/provenance"):
            st, _ = self.req(srv, "GET", path, raw=True)
            assert st == 200, f"{path}: {st}"
        # A named context in its designed use: the system turn pinned once,
        # then chat requests that start with it. A template that folds the
        # system turn into the first user turn has no such prefix: said, not
        # failed.
        notes = None
        sysm = {"role": "system", "content":
                "You are a terse assistant for a river-survey team. " * 4}
        st, pin = self.req(srv, "POST", "/v1/runner/contexts",
                           {"id": "sweep", "messages": [sysm]})
        if st in (400, 409):
            notes = f"named context not pinned ({st}): " + json.dumps(pin)[:200]
        else:
            assert st == 200, pin
            st, d = self.req(srv, "POST", "/v1/chat/completions", {
                "messages": [sysm, {"role": "user", "content": "Where is the ford?"}],
                "max_tokens": 8, "temperature": 0, "context_id": "sweep"})
            if st == 409:
                notes = "this template does not render the system turn as a " \
                        "strict prefix: " + json.dumps(d)[:200]
            else:
                assert st == 200, d
                assert d["runner_telemetry"]["context"]["id"] == "sweep"
                assert d["runner_telemetry"]["prompt_cached_tokens"] >= pin["tokens"]
            st, d = self.req(srv, "POST", "/v1/chat/completions", {
                "messages": [{"role": "user", "content": "Something else entirely."}],
                "max_tokens": 2, "context_id": "sweep"})
            assert st == 409, f"a prompt that does not start with the context: {st}"
            assert self.req(srv, "DELETE", "/v1/runner/contexts/sweep")[0] == 200
        docs = ["The cat sat on the mat.", "Quarterly revenue rose four percent.",
                "A kitten slept on a rug."]
        st, a = self.req(srv, "POST", "/v1/rerank",
                         {"query": "cats resting", "documents": docs})
        assert st == 200 and len(a["results"]) == 3, a
        st, b = self.req(srv, "POST", "/v1/rerank",
                         {"query": "cats resting", "documents": docs[::-1]})
        sa = {docs[r["index"]]: r["logit"] for r in a["results"]}
        sb = {docs[::-1][r["index"]]: r["logit"] for r in b["results"]}
        assert sa == sb, f"a rerank score moved with document order: {sa} vs {sb}"
        st, d = self.req(srv, "POST", "/v1/decide", {
            "state": "The sky is clear and it is noon.",
            "questions": [{"question": "Is the sky blue?",
                           "options": ["yes", "no"]}]})
        assert st == 200, d
        st, e = self.req(srv, "POST", "/v1/embeddings", {"input": "a river"})
        if st != 200:
            notes = ((notes + "; ") if notes else "") + \
                f"embeddings {st}: {json.dumps(e)[:120]}"
        else:
            v = e["data"][0]["embedding"]
            assert len(v) > 8 and abs(sum(x * x for x in v) - 1) < 1e-3, "not unit length"
        st, prov = self.req(srv, "GET", "/v1/runner/provenance")
        deadline = time.time() + 600
        while prov["model"]["sha256_state"] == "hashing" and time.time() < deadline:
            time.sleep(2)
            st, prov = self.req(srv, "GET", "/v1/runner/provenance")
        assert prov["model"]["sha256_state"] == "done", prov["model"]
        if Path(self.model).stat().st_size < 12 << 30:
            h = hashlib.sha256()
            with open(self.model, "rb") as f:
                for blk in iter(lambda: f.read(1 << 24), b""):
                    h.update(blk)
            assert prov["model"]["sha256"] == h.hexdigest(), "provenance digest is wrong"
        return notes

    def s_malformed(self, srv):
        cases = [
            ("not json", b"{nope"),
            ("wrong types", {"messages": "hi"}),
            ("negative max_tokens", {"messages": [{"role": "user", "content": "x"}],
                                     "max_tokens": -5}),
            ("huge max_tokens", {"messages": [{"role": "user", "content": "x"}],
                                 "max_tokens": 10 ** 12, "stop": ["x"] * 9}),
            ("bad role", {"messages": [{"role": "wizard", "content": "x"}]}),
            ("empty messages", {"messages": []}),
            ("nan temperature", b'{"messages":[{"role":"user","content":"x"}],'
                                b'"temperature":NaN}'),
            ("deep nesting", ("[" * 5000 + "]" * 5000).encode()),
            ("bad tool schema", {"messages": [{"role": "user", "content": "x"}],
                                 "tools": [{"type": "function", "function": {
                                     "name": "f", "parameters": {"type": "nonsense"}}}],
                                 "tool_choice": "required"}),
            ("invalid utf8", b'{"messages":[{"role":"user","content":"\xff\xfe"}]}'),
        ]
        for name, body in cases:
            st, d = self.req(srv, "POST", "/v1/chat/completions", body)
            assert 400 <= st < 500, f"{name}: answered {st} {json.dumps(d)[:600]}"
        st, d = self.req(srv, "POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": ""}], "max_tokens": 4})
        assert st in (200, 400), f"empty content: {st}"
        assert self.req(srv, "GET", "/health")[0] == 200, "dead after malformed requests"

    def s_disconnect(self, srv):
        s = socket.create_connection(("127.0.0.1", srv.port), timeout=30)
        body = json.dumps({"messages": [{"role": "user", "content":
                                         "Write a very long story."}],
                           "max_tokens": 400, "stream": True}).encode()
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                  b"Content-Type: application/json\r\nContent-Length: "
                  + str(len(body)).encode() + b"\r\n\r\n" + body)
        s.recv(2048)
        s.close()
        t0 = time.time()
        d = self.chat(srv, "Say ok.", max_tokens=4)
        assert d["usage"]["completion_tokens"] >= 1
        return f"next request answered {time.time() - t0:.1f}s after the hang-up"

    def s_receipt(self, srv, rdir):
        st, d = self.req(srv, "POST", "/v1/completions", {
            "prompt": "The printing press was invented by", "max_tokens": 16,
            "temperature": 0.7, "seed": 3, "cache_prompt": False})
        assert st == 200, d
        r = d["runner_telemetry"].get("receipt") or {}
        assert r.get("file"), f"no receipt: {r}"
        rc, _, err, _ = sh([self.runner, "-m", self.model, "--verify",
                            Path(rdir) / r["file"],
                            *(["-t", str(self.threads)] if self.threads else [])],
                           self.timeout)
        assert rc == 0 and "VERIFIED" in err, f"exit {rc}: {err[-500:]}"

    def run(self, tmp):
        print(f"  [{self.backend}]", flush=True)
        if not self.check("cli greedy", self.cli_greedy):
            return self.rows
        self.check("cli seeded sampling is reproducible", self.cli_seeded)
        self.check("cli transcript replays VERIFIED", lambda: self.cli_transcript(tmp))
        if self.backend == "cpu":
            self.check("cli session suspend/resume byte identity",
                       lambda: self.cli_session(tmp))
        if not self.quick:
            self.check("cli hostile prompts", self.cli_hostile)
        rdir = tmp / f"receipts-{self.backend}"
        extra = [*self.gpu_args, *(["-t", str(self.threads)] if self.threads else [])]
        if self.backend == "cpu":
            extra += ["--receipts", str(rdir)]
        log = tmp / f"server-{self.backend}.log"
        try:
            with RunnerServer(self.runner, self.model, ctx=self.ctx, parallel=2,
                              extra_args=extra, start_timeout=900,
                              log_path=str(log)) as srv:
                self.srv_ctx = self.ctx
                for name, fn in [
                        ("chat basic", self.s_basic),
                        ("stream equals buffered, unicode", self.s_stream_equals_buffered),
                        ("max_tokens 1 and stop sequences", self.s_limits),
                        ("json mode and strict schema", self.s_json),
                        ("tool call required, streamed, tool result, auto", self.s_tools),
                        ("context near limit and over it", self.s_context),
                        ("repeated request reuses KV", self.s_reuse),
                        ("concurrent requests on two slots", self.s_concurrent),
                        ("logprobs, echo, prompt_logprobs", self.s_logprobs),
                        ("responses and messages surfaces", self.s_surfaces),
                        ("runner routes: context, rerank, decide, provenance",
                         self.s_runner_routes),
                        ("malformed requests", self.s_malformed),
                        ("client hang-up mid-stream", self.s_disconnect)]:
                    if self.quick and name.split()[0] in ("concurrent", "runner",
                                                          "malformed", "client"):
                        continue
                    self.check("serve " + name, lambda fn=fn: fn(srv))
                    if srv.proc.poll() is not None:
                        self.rows.append({"check": "server alive", "backend": self.backend,
                                          "verdict": "FAIL", "detail":
                                          f"the server died during '{name}' (exit "
                                          f"{srv.proc.returncode}); log tail: "
                                          + log.read_text(errors="replace")[-800:]})
                        print("    FAIL server died", flush=True)
                        return self.rows
                if self.backend == "cpu":
                    self.check("serve receipt replays VERIFIED",
                               lambda: self.s_receipt(srv, rdir))
        except Exception as e:
            tail = log.read_text(errors="replace")[-800:] if log.exists() else ""
            r = refused(tail)
            self.rows.append({"check": "server lifecycle", "backend": self.backend,
                              "verdict": "SKIP" if r else "FAIL",
                              "detail": f"{type(e).__name__}: {e}; log tail: {tail}"})
            print(f"    {'SKIP' if r else 'FAIL'} server lifecycle: {e}", flush=True)
        return self.rows


def sweep_model(args, runner, entry, out_dir):
    path = Path(entry["path"])
    name = entry.get("family") or path.stem
    out = out_dir / (path.stem + ".json")
    if args.skip_done and out.exists():
        return json.loads(out.read_text())
    print(f"\n== {name}: {path.name} ({path.stat().st_size / 2**30:.1f} GB)", flush=True)
    ctx = int(entry.get("ctx") or args.ctx)
    rec = {"family": name, "path": str(path), "bytes": path.stat().st_size,
           "ctx": ctx, "note": entry.get("note"), "rows": [],
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    texts = {}
    with tempfile.TemporaryDirectory(prefix="family-sweep-", dir=args.tmp) as td:
        for backend in entry.get("backends") or args.backends.split(","):
            b = Bench(runner, path, ctx, backend, args.threads, args.timeout, args.quick)
            rec["rows"] += b.run(Path(td))
            texts[backend] = getattr(b, "greedy_text", None)
    if texts.get("cpu") and texts.get("gpu"):
        a, g = texts["cpu"], texts["gpu"]
        same = a.startswith(g) or g.startswith(a)
        rec["rows"].append({"check": "cli greedy GPU equals CPU", "backend": "both",
                            "verdict": "PASS" if same else "NOTE",
                            "detail": None if same else
                            f"cpu {a[:160]!r} vs gpu {g[:160]!r} (a near-tie can "
                            "separate two correct backends; judge the text)"})
        print(f"    {'PASS' if same else 'NOTE'} both cli greedy GPU equals CPU", flush=True)
    rec["verdicts"] = {v: sum(r["verdict"] == v for r in rec["rows"])
                       for v in ("PASS", "FAIL", "NOTE", "SKIP")}
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--roster", help="JSON list of models (see the module docstring)")
    ap.add_argument("--model", action="append", default=[], help="a GGUF (repeatable)")
    ap.add_argument("--runner", help="runner binary (default: the build in the repo)")
    ap.add_argument("--out-dir", default="family-sweep-out")
    ap.add_argument("--backends", default="cpu", help="cpu, gpu, or cpu,gpu")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=0, help="-t for every run")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per command")
    ap.add_argument("--tmp", default=None, help="scratch directory")
    ap.add_argument("--quick", action="store_true", help="the core checks only")
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()
    runner = Path(args.runner) if args.runner else find_runner(ROOT)
    roster = json.loads(Path(args.roster).read_text()) if args.roster else []
    roster += [{"path": m} for m in args.model]
    if not roster:
        ap.error("give --roster or --model")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for entry in roster:
        if not Path(entry["path"]).exists():
            print(f"\n== {entry.get('family', entry['path'])}: MISSING {entry['path']}")
            recs.append({"family": entry.get("family"), "path": entry["path"],
                         "rows": [], "verdicts": {"MISSING": 1}})
            continue
        try:
            recs.append(sweep_model(args, runner, entry, out_dir))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"   harness error: {e}")
            recs.append({"family": entry.get("family"), "path": entry["path"],
                         "rows": [], "verdicts": {"HARNESS": 1}, "error": str(e)})
    print("\n| family | file | PASS | FAIL | NOTE | SKIP |\n|---|---|---|---|---|---|")
    fails = 0
    for r in recs:
        v = r["verdicts"]
        fails += v.get("FAIL", 0) + v.get("HARNESS", 0)
        print(f"| {r.get('family')} | {Path(r['path']).name} | {v.get('PASS', 0)} | "
              f"{v.get('FAIL', 0)} | {v.get('NOTE', 0)} | "
              f"{v.get('SKIP', 0) + v.get('MISSING', 0)} |")
    for r in recs:
        for row in r.get("rows", []):
            if row["verdict"] == "FAIL":
                print(f"\nFAIL {r.get('family')} [{row['backend']}] {row['check']}:\n"
                      f"  {row['detail']}")
    (out_dir / "summary.json").write_text(json.dumps(recs, indent=1, ensure_ascii=False))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
