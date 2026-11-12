#!/usr/bin/env python3
"""Soak a `runner --serve` process: the questions a unit test cannot ask.

A test binary proves one path once. A served process answers thousands of
requests, is cancelled in the middle of them, swaps models and runs into its
context limit, and the failures that matter there are slow ones: a reply that
changes on the 140th repeat, memory that never comes back after a cancel, a
slot that stays busy. This script asks those questions of a real model and
writes down what it saw.

Phases (each can be switched off with --skip):

  identity    the same greedy request N times: every reply must be
              byte-identical to the first, and the process's memory must not
              keep growing once it has warmed up.
  cancel      a request is cancelled mid-prefill and mid-decode, by closing
              its socket and by its request id. A prefill cancel is sent when
              /health shows the request in its prefill phase. After each
              cancel the server must go idle and answer the reference request
              exactly.
  swap        with --model2: alternate between two models. Each model's
              reply must equal the one a fresh process gave, every time.
  limit       prompts at the context window, one token over it and a few
              tokens under it. The contract is 200 or a 400 that names the
              context; a 5xx, a dropped connection or a hang is a finding.
  concurrent  several clients at once for a while (plain, streamed, JSON
              schema, tool call). Every client's greedy reply must equal the
              one it got alone, so a crossed or corrupted reply shows.

The instrument has one absolute anchor of its own: a reply that parses as
the JSON schema the request named, and HTTP status codes read off the wire.
Everything else is self-agreement (this build against itself), which proves
stability, not correctness; correctness is what the fidelity gates are for.

Exit status 0 with no finding, 1 with findings, 2 when the soak could not
run. The report (--out) is JSON; findings are printed as they happen.
"""

import argparse
import http.client
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

LOG_ALARMS = ("AddressSanitizer", "ThreadSanitizer", "runtime error:",
              "Assertion failed", "assertion failed", "Segmentation fault",
              "stack smashing", "double free", "heap-use-after-free")


def parse_smi_table(text, pid=None):
    """MiB in use from nvidia-smi's plain table: the process row of `pid`
    when it is there (`|  0  N/A  N/A  4242  C  ./runner  118MiB |`), else
    the device rows' "81MiB / 24192MiB" summed; None when neither parses.
    A device row starts with its index; the row under it, which starts with
    no number, is the BAR1 mapping and is not memory in use."""
    devices = []
    for line in text.splitlines():
        cells = line.replace("|", " ").split()
        if pid and pid in cells and cells and cells[-1].endswith("MiB") and "/" not in cells:
            if cells[-1][:-3].isdigit():
                return int(cells[-1][:-3])
        for i, c in enumerate(cells):
            if not cells[0].isdigit():
                break
            if c == "/" and 0 < i < len(cells) - 1 and cells[i - 1].endswith("MiB") \
                    and cells[i + 1].endswith("MiB") and cells[i - 1][:-3].isdigit():
                devices.append(int(cells[i - 1][:-3]))
                break
    return sum(devices) if devices else None


class Soak:
    def __init__(self, a):
        self.a = a
        self.host, self.port = a.host, a.port
        self.findings = []
        self.report = {"phases": {}, "findings": self.findings}
        self.proc = None
        self.log_path = None

    # ---- plumbing ---------------------------------------------------------

    def finding(self, phase, what, **detail):
        row = {"phase": phase, "what": what, **detail}
        self.findings.append(row)
        print(f"FINDING [{phase}] {what} {json.dumps(detail)[:400]}", flush=True)

    def conn(self, timeout=None):
        return http.client.HTTPConnection(self.host, self.port,
                                          timeout=timeout or self.a.timeout)

    def call(self, method, path, body=None, headers=None, timeout=None):
        """One request. Returns (status, parsed JSON or None, raw text)."""
        c = self.conn(timeout)
        try:
            data = json.dumps(body).encode() if body is not None else None
            h = {"Content-Type": "application/json"} if data is not None else {}
            h.update(headers or {})
            c.request(method, path, body=data, headers=h)
            r = c.getresponse()
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw), raw
            except ValueError:
                return r.status, None, raw
        finally:
            c.close()

    def health(self):
        try:
            s, j, _ = self.call("GET", "/health", timeout=10)
            return j if s == 200 else None
        except (OSError, http.client.HTTPException):
            return None

    def wait_idle(self, phase, deadline_s=None):
        """The server must drain to zero active requests; a slot that stays
        busy after its client is gone is the hang this soak exists to find."""
        end = time.time() + (deadline_s or self.a.idle_timeout)
        while time.time() < end:
            h = self.health()
            if h is None:
                if self.proc is not None and self.proc.poll() is not None:
                    self.finding(phase, "the server process ended",
                                 exit_code=self.proc.returncode)
                    return False
            elif h.get("active_requests", 0) == 0:
                return True
            time.sleep(0.2)
        self.finding(phase, "the server did not go idle", seconds=deadline_s or self.a.idle_timeout)
        return False

    def wait_phase(self, phase):
        """Wait until /health lists a request in `phase`, then --cancel-delay
        more. The cancel phase sends one request at a time, so the request
        listed is the one to cancel. False when none shows up: the request
        was too quick to be caught there."""
        end = time.time() + self.a.phase_timeout
        while time.time() < end:
            for r in (self.health() or {}).get("requests") or []:
                if r.get("phase") == phase:
                    time.sleep(self.a.cancel_delay)
                    return True
            time.sleep(0.01)
        return False

    def gpu_mem_mb(self):
        """Device memory of the server process where nvidia-smi can say it,
        else the device's total in use; None without nvidia-smi or when it
        gives no number. A query can answer "[Insufficient Permissions]" (a
        MIG slice does), so the plain table is read as well: its process row
        for the server's pid, else its device row."""
        smi = shutil.which("nvidia-smi")
        if not smi:
            return None

        def out(*args):
            try:
                return subprocess.run([smi, *args], capture_output=True, text=True,
                                      timeout=20).stdout
            except (OSError, subprocess.SubprocessError):
                return ""
        pid = str(self.proc.pid) if self.proc is not None else None
        if pid:
            for line in out("--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits").splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) == 2 and parts[0] == pid and parts[1].isdigit():
                    return int(parts[1])
        vals = [int(v) for v in out("--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits").split() if v.isdigit()]
        if vals:
            return sum(vals)
        return parse_smi_table(out(), pid)

    def memory(self):
        """rss_mb is the process's resident set. On macOS the server also
        reports its physical footprint, which counts Metal's buffers where
        the resident set does not (13 MB resident at a 638 MB footprint was
        measured), so the growth rule reads footprint_mb when it is there."""
        h = self.health() or {}
        m = {"rss_mb": round(h.get("rss_bytes", 0) / 2**20, 1),
             "gpu_mb": self.gpu_mem_mb(), "t": round(time.time(), 1)}
        if "phys_footprint_bytes" in h:
            m["footprint_mb"] = round(h["phys_footprint_bytes"] / 2**20, 1)
        return m

    def grew(self, phase, warm, end, label):
        """Growth past the warm reading by more than the tolerance."""
        # where the footprint is reported it replaces the resident set, which
        # there falls as pages are compressed and never sees device memory
        keys = ("footprint_mb", "gpu_mb") if "footprint_mb" in warm else ("rss_mb", "gpu_mb")
        for key in keys:
            w, e = warm.get(key), end.get(key)
            if not w or not e:
                continue
            if e - w > max(self.a.growth_mb, w * self.a.growth_frac):
                self.finding(phase, f"{key} kept growing {label}",
                             warm=w, end=e, grew_mb=round(e - w, 1))

    # ---- the server -------------------------------------------------------

    def start(self, model_spec, tag):
        self.log_path = Path(self.a.workdir) / f"server-{tag}.log"
        cmd = [self.a.runner, "--serve", "--port", str(self.port), "-m", model_spec]
        cmd += self.a.server_args
        log = open(self.log_path, "wb")
        env = dict(os.environ)
        for kv in self.a.server_env:
            k, _, v = kv.partition("=")
            env[k] = v
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.close()
        end = time.time() + self.a.load_timeout
        while time.time() < end:
            if self.proc.poll() is not None:
                raise SystemExit(f"soak: the server ended during startup "
                                 f"(exit {self.proc.returncode}); see {self.log_path}")
            if self.health() is not None:
                return
            time.sleep(0.5)
        self.stop("startup")
        raise SystemExit(f"soak: no /health within {self.a.load_timeout} s; see {self.log_path}")

    def stop(self, phase):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            if os.name == "nt":
                self.proc.terminate()   # no SIGTERM drain to observe on Windows
            else:
                self.proc.send_signal(signal.SIGTERM)
            try:
                rc = self.proc.wait(timeout=60)
                if os.name != "nt" and rc != 0:
                    self.finding(phase, "SIGTERM did not end the server with exit 0",
                                 exit_code=rc)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
                self.finding(phase, "the server did not exit within 60 s of SIGTERM")
        else:
            self.finding(phase, "the server process had already ended",
                         exit_code=self.proc.returncode)
        self.scan_log(phase)
        self.proc = None

    def scan_log(self, phase):
        try:
            text = self.log_path.read_text("utf-8", "replace")
        except OSError:
            return
        for line in text.splitlines():
            if any(k in line for k in LOG_ALARMS):
                self.finding(phase, "alarm line in the server log", line=line[:300],
                             log=str(self.log_path))
                break

    # ---- requests ---------------------------------------------------------

    def chat_body(self, text, model=None, max_tokens=None, **extra):
        b = {"messages": [{"role": "user", "content": text}], "temperature": 0,
             "seed": 1, "max_tokens": max_tokens or self.a.max_tokens}
        if model:
            b["model"] = model
        b.update(extra)
        return b

    @staticmethod
    def reply_key(j):
        """What must be identical between two greedy replies."""
        try:
            ch = j["choices"][0]
        except (KeyError, IndexError, TypeError):
            return None
        m = ch.get("message") or {}
        calls = [(c.get("function", {}).get("name"), c.get("function", {}).get("arguments"))
                 for c in (m.get("tool_calls") or [])]
        return json.dumps([m.get("content"), m.get("reasoning_content"), calls,
                           ch.get("text"), ch.get("finish_reason")], sort_keys=True)

    def ask(self, body, path="/v1/chat/completions", headers=None):
        try:
            return self.call("POST", path, body, headers)
        except (OSError, http.client.HTTPException) as e:
            return -1, None, f"{type(e).__name__}: {e}"

    def stream(self, body, stop_after=None, on_first=None):
        """A streamed chat. Returns (status, text, n_chunks, ended_cleanly).
        stop_after closes the socket after that many data chunks; on_first is
        called with the first chunk (it carries the response id)."""
        b = dict(body, stream=True)
        c = self.conn()
        text, n, clean = [], 0, False
        try:
            c.request("POST", "/v1/chat/completions", body=json.dumps(b).encode(),
                      headers={"Content-Type": "application/json"})
            r = c.getresponse()
            if r.status != 200:
                return r.status, r.read().decode("utf-8", "replace"), 0, False
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    clean = True
                    break
                try:
                    j = json.loads(payload)
                except ValueError:
                    return 200, "unparseable chunk: " + payload[:200].decode("utf-8", "replace"), n, False
                n += 1
                if n == 1 and on_first:
                    on_first(j)
                for ch in j.get("choices") or []:
                    d = ch.get("delta") or {}
                    text.append((d.get("reasoning_content") or "") + (d.get("content") or ""))
                if stop_after and n >= stop_after:
                    break
            return 200, "".join(text), n, clean
        except (OSError, http.client.HTTPException) as e:
            return -1, f"{type(e).__name__}: {e}", n, False
        finally:
            c.close()

    def reference(self, phase, model=None):
        s, j, raw = self.ask(self.chat_body(self.a.prompt, model))
        key = self.reply_key(j) if s == 200 else None
        if key is None:
            self.finding(phase, "the reference request failed", status=s, body=raw[:300])
        return key

    def same_as(self, phase, ref, what, model=None):
        if ref is None:
            return False
        s, j, raw = self.ask(self.chat_body(self.a.prompt, model))
        key = self.reply_key(j) if s == 200 else None
        if key != ref:
            self.finding(phase, f"the reference reply changed {what}", status=s,
                         expected=ref[:300], got=(key or raw)[:300])
            return False
        return True

    def filler(self, n, salt=0):
        """n prompt token ids. The salt changes the first tokens, so a repeat
        of the phase does not fork the previous prompt out of the cache."""
        v = self.a.filler_vocab
        return [self.a.filler_base + ((i * 7 + salt * 131) % v) for i in range(n)]

    # ---- phases -----------------------------------------------------------

    def phase_identity(self, ref):
        n = self.a.cycles
        marks = sorted({1, max(1, n // 4), max(1, n // 2), n})
        mem, bad = {}, 0
        t0 = time.time()
        for i in range(1, n + 1):
            if not self.same_as("identity", ref, f"at cycle {i}"):
                bad += 1
                if bad >= 3:
                    break
            if i in marks:
                mem[i] = self.memory()
                print(f"identity {i}/{n} {mem[i]}", flush=True)
        if len(mem) >= 2:
            ks = sorted(mem)
            warm = mem[ks[1]] if len(ks) > 2 else mem[ks[0]]
            self.grew("identity", warm, mem[ks[-1]], f"between cycle {ks[1] if len(ks) > 2 else ks[0]} and {ks[-1]}")
        self.report["phases"]["identity"] = {"cycles": n, "mismatches": bad, "memory": mem,
                                             "seconds": round(time.time() - t0, 1)}

    def phase_cancel(self, ref):
        n, ctx = self.a.cancels, self.ctx
        long_n = max(64, min(ctx - 64, self.a.cancel_prompt_tokens))
        counts = {"close_prefill": 0, "close_prefill_landed": 0, "id_prefill": 0, "close_decode": 0, "id_decode": 0,
                  "id_prefill_too_late": 0, "id_decode_too_late": 0}
        warm = None
        for k in range(n):
            # 1. the socket closes while the prompt is being processed
            body = {"prompt": self.filler(long_n, 4 * k + 1), "max_tokens": 8,
                    "temperature": 0, "stream": True}
            c = self.conn()
            try:
                c.request("POST", "/v1/completions", body=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
                if self.wait_phase("prefill"):
                    counts["close_prefill_landed"] += 1
            except OSError as e:
                self.finding("cancel", "could not send the request to cancel", error=str(e))
            finally:
                c.close()
            counts["close_prefill"] += 1
            if not (self.wait_idle("cancel") and self.same_as("cancel", ref, "after a socket close mid-prefill")):
                break

            # 2. cancel by request id while the prompt is being processed
            rid = f"soak-{os.getpid()}-{k}"
            got = {}

            def run(rid=rid, k=k, got=got):
                got["r"] = self.ask({"prompt": self.filler(long_n, 4 * k + 2), "max_tokens": 8,
                                     "temperature": 0}, "/v1/completions",
                                    {"X-Request-Id": rid})
            t = threading.Thread(target=run)
            t.start()
            self.wait_phase("prefill")
            cs, cj, craw = self.ask(None, f"/v1/runner/requests/{rid}/cancel")
            t.join(self.a.timeout)
            if t.is_alive():
                self.finding("cancel", "a request cancelled by id never returned", request_id=rid)
                break
            rs = got.get("r", (-1, None, ""))
            if cs == 200:
                counts["id_prefill"] += 1
                if rs[0] != 409:
                    self.finding("cancel", "a request cancelled by id did not end with 409",
                                 status=rs[0], body=rs[2][:300], cancel=craw[:200])
            elif cs == 404 and rs[0] == 200:
                counts["id_prefill_too_late"] += 1    # it finished first: not a finding
            else:
                self.finding("cancel", "cancel by id answered unexpectedly", cancel_status=cs,
                             cancel=craw[:300], request_status=rs[0], body=rs[2][:200])
            if not (self.wait_idle("cancel") and self.same_as("cancel", ref, "after a cancel by id mid-prefill")):
                break

            # 3. the socket closes while tokens are streaming
            s, text, chunks, _ = self.stream(
                self.chat_body(self.a.long_prompt, max_tokens=self.a.decode_tokens), stop_after=3)
            if s != 200:
                self.finding("cancel", "the streamed request to cancel failed", status=s, body=text[:300])
            counts["close_decode"] += 1
            if not (self.wait_idle("cancel") and self.same_as("cancel", ref, "after a socket close mid-decode")):
                break

            # 4. cancel by response id while tokens are streaming
            seen = {}

            def first(j, seen=seen):
                seen["id"] = j.get("id")
                seen["cancel"] = self.ask(None, f"/v1/runner/requests/{j.get('id')}/cancel")
            s, text, chunks, clean = self.stream(
                self.chat_body(self.a.long_prompt, max_tokens=self.a.decode_tokens), on_first=first)
            cs = seen.get("cancel", (-1, None, ""))[0]
            if cs == 200:
                counts["id_decode"] += 1
            elif cs == 404:
                counts["id_decode_too_late"] += 1
            else:
                self.finding("cancel", "cancel by response id answered unexpectedly",
                             cancel_status=cs, cancel=seen.get("cancel", (0, 0, ""))[2][:300],
                             response_id=seen.get("id"))
            if not (self.wait_idle("cancel") and self.same_as("cancel", ref, "after a cancel by id mid-decode")):
                break
            if k == max(0, n // 4):
                warm = self.memory()
            if (k + 1) % 10 == 0:
                print(f"cancel {k + 1}/{n} {counts}", flush=True)
        end = self.memory()
        if warm:
            self.grew("cancel", warm, end, "across cancels")
        if n and counts["id_prefill"] == 0:
            self.finding("cancel", "no cancel by id landed mid-prefill: the phase measured nothing "
                         "(raise --cancel-prompt-tokens or lower --cancel-delay)", counts=counts)
        self.report["phases"]["cancel"] = {"counts": counts, "warm": warm, "end": end}

    def phase_swap(self, refs):
        # Memory is compared between two readings that hold the SAME model:
        # the two models differ in size, so a reading with one resident
        # against a reading with the other is the models' difference, not
        # growth (the first Windows run reported exactly that as a finding).
        names = list(refs)
        mem = {name: {} for name in names}
        for i in range(1, self.a.swaps + 1):
            name = names[i % 2]
            if not self.same_as("swap", refs[name], f"for model {name} at swap {i}", model=name):
                break
            mem[name][i] = self.memory()
            if i % 10 == 0 or i == self.a.swaps:
                print(f"swap {i}/{self.a.swaps} model {name} {mem[name][i]}", flush=True)
        for name in names:
            ks = sorted(mem[name])
            if len(ks) >= 4:
                warm = ks[len(ks) // 4]            # past the first loads of each model
                self.grew("swap", mem[name][warm], mem[name][ks[-1]],
                          f"with model {name} resident, between swap {warm} and {ks[-1]}")
        self.report["phases"]["swap"] = {"swaps": self.a.swaps, "memory": mem}

    def phase_limit(self, ref):
        ctx, rows = self.ctx, []
        cases = [("at the window", ctx, 4, (200, 400)),
                 ("one token over", ctx + 1, 4, (400,)),
                 ("one under, one token asked", ctx - 1, 1, (200, 400)),
                 ("generation runs into the window", ctx - 8, 64, (200, 400))]
        for label, n, max_new, allowed in cases:
            s, j, raw = self.ask({"prompt": self.filler(n, 900 + n % 7), "max_tokens": max_new,
                                  "temperature": 0}, "/v1/completions")
            row = {"case": label, "prompt_tokens": n, "max_tokens": max_new, "status": s}
            if s == 200 and j:
                u = j.get("usage") or {}
                row["completion_tokens"] = u.get("completion_tokens")
                row["finish_reason"] = (j.get("choices") or [{}])[0].get("finish_reason")
                if u.get("prompt_tokens", 0) + (u.get("completion_tokens") or 0) > ctx:
                    self.finding("limit", "a reply used more tokens than the context window",
                                 case=label, usage=u, context_window=ctx)
            elif s == 400 and j:
                row["code"] = (j.get("error") or {}).get("code")
            if s not in allowed or j is None:
                self.finding("limit", "the context limit was not answered cleanly", case=label,
                             status=s, body=raw[:300], allowed=list(allowed))
            rows.append(row)
            print(f"limit {row}", flush=True)
            if not (self.wait_idle("limit") and self.same_as("limit", ref, f"after the case '{label}'")):
                break
        self.report["phases"]["limit"] = rows

    def phase_concurrent(self):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}},
                  "required": ["answer"], "additionalProperties": False}
        tool = {"type": "function", "function": {
            "name": "get_weather", "description": "Current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}}
        kinds = [
            ("plain", self.chat_body("List four fruits, one per line.")),
            ("stream", self.chat_body("Count from one to twelve in words.")),
            ("schema", self.chat_body("Answer in JSON: what is the capital of France?",
                                      response_format={"type": "json_schema", "json_schema": {
                                          "name": "a", "strict": True, "schema": schema}})),
            ("tool", self.chat_body("What is the weather in Oslo?", tools=[tool])),
        ]
        kinds = [kinds[i % len(kinds)] for i in range(self.a.clients)]
        # each client's reply alone, before any contention
        alone, skipped, kept = [], {}, []
        for kind, body in kinds:
            raw = ""
            if kind in skipped:
                continue
            if kind == "stream":
                s, text, _, clean = self.stream(body)
                got = text if s == 200 and clean else None
            else:
                s, j, raw = self.ask(body)
                got = self.reply_key(j) if s == 200 else None
            if got is None and s == 400 and kind in ("schema", "tool"):
                # a model whose template has no tools, or a build that refuses
                # the constraint, says so with a 400: recorded, not soaked
                skipped[kind] = raw[:300]
                continue
            if got is None:
                self.finding("concurrent", f"the '{kind}' request failed alone", status=s,
                             body=raw[:300])
            kept.append((kind, body))
            alone.append(got)
        kinds = kept
        if skipped:
            print(f"concurrent: not soaked, refused alone with 400: {skipped}", flush=True)
        stats = {"requests": 0, "status": {}, "mismatch": 0, "invalid": 0,
                 "kinds": [k for k, _ in kinds], "refused_alone": skipped}
        lock = threading.Lock()
        end = time.time() + self.a.concurrent_seconds

        def client(i):
            kind, body = kinds[i]
            while time.time() < end:
                if kind == "stream":
                    s, text, _, clean = self.stream(body)
                    got = text if s == 200 and clean else None
                    j = None
                else:
                    s, j, raw = self.ask(body)
                    got = self.reply_key(j) if s == 200 else None
                with lock:
                    stats["requests"] += 1
                    stats["status"][str(s)] = stats["status"].get(str(s), 0) + 1
                    if s != 200:
                        if stats["status"][str(s)] <= 3:
                            self.finding("concurrent", f"a '{kind}' request failed under load",
                                         status=s, client=i)
                        continue
                    if alone[i] is not None and got != alone[i]:
                        stats["mismatch"] += 1
                        if stats["mismatch"] <= 3:
                            self.finding("concurrent", f"a '{kind}' reply under load differs from the "
                                         "reply alone", client=i, alone=str(alone[i])[:300],
                                         got=str(got)[:300])
                    if kind == "schema" and j:
                        try:
                            doc = json.loads(j["choices"][0]["message"]["content"])
                            assert isinstance(doc, dict) and isinstance(doc.get("answer"), str)
                            assert set(doc) == {"answer"}
                        except (ValueError, KeyError, TypeError, AssertionError):
                            stats["invalid"] += 1
                            if stats["invalid"] <= 3:
                                self.finding("concurrent", "a JSON-schema reply does not satisfy "
                                             "its schema", body=json.dumps(j)[:400])
                    if kind == "tool" and j:
                        for c in (j["choices"][0].get("message") or {}).get("tool_calls") or []:
                            try:
                                json.loads(c["function"]["arguments"])
                            except (ValueError, KeyError, TypeError):
                                stats["invalid"] += 1
                                self.finding("concurrent", "a tool call's arguments are not JSON",
                                             call=json.dumps(c)[:300])
        warm = self.memory()
        ts = [threading.Thread(target=client, args=(i,)) for i in range(len(kinds))]
        for t in ts:
            t.start()
        for t in ts:
            t.join(self.a.concurrent_seconds + self.a.timeout)
            if t.is_alive():
                self.finding("concurrent", "a client never returned")
        self.wait_idle("concurrent")
        stats["warm"], stats["end"] = warm, self.memory()
        print(f"concurrent {stats}", flush=True)
        self.report["phases"]["concurrent"] = stats

    # ---- the run ----------------------------------------------------------

    def run(self):
        a = self.a
        own = not a.attach
        refs = {}
        if own and a.model2 and "swap" not in a.skip:
            # each model's reply from a process that has served nothing else
            for name, path in (("b", a.model2),):
                self.start(path, f"fresh-{name}")
                refs[name] = self.reference("swap")
                self.stop("swap")
        if own:
            spec = f"a={a.model},b={a.model2}" if a.model2 and "swap" not in a.skip else a.model
            self.start(spec, "soak")
        elif self.health() is None:
            raise SystemExit(f"soak: no server answers at {self.host}:{self.port}")
        model = "a" if refs else None
        s, j, _ = self.call("GET", "/v1/models")
        entries = (j or {}).get("data") or []
        self.report["models"] = entries
        ref = self.reference("start", model)
        if ref is None:
            self.stop("start")
            return 2
        refs["a"] = ref
        # the window is read after the first load: before it a swap set has
        # no resident model to report a window for
        s, j, _ = self.call("GET", "/v1/models")
        ctxs = [e.get("context_window") for e in (j or {}).get("data") or []
                if e.get("context_window")]
        self.ctx = a.ctx or (min(ctxs) if ctxs else 0)
        self.report["context_window"] = self.ctx
        self.report["start"] = self.memory()
        self.report["gpu_memory"] = ("measured" if self.report["start"]["gpu_mb"] is not None
                                     else "unmeasured: nvidia-smi is absent or gave no number")
        print(f"soak: reference reply {ref[:160]}", flush=True)
        try:
            if "identity" not in a.skip:
                self.phase_identity(ref)
            if "cancel" not in a.skip and self.ctx:
                self.phase_cancel(ref)
            if "limit" not in a.skip and self.ctx:
                self.phase_limit(ref)
            if "concurrent" not in a.skip:
                self.phase_concurrent()
            if "b" in refs and "swap" not in a.skip:
                self.phase_swap(refs)
            self.same_as("end", ref, "at the end of the soak", model)
            h = self.health()
            if not h or h.get("status") != "ok":
                self.finding("end", "the server is not healthy at the end", health=h)
            self.report["end"] = self.memory()
        finally:
            if own:
                self.stop("end")
        return 1 if self.findings else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner", default="./runner", help="the runner binary to serve with")
    ap.add_argument("--model", help="GGUF to serve")
    ap.add_argument("--model2", help="a second GGUF: enables the swap phase")
    ap.add_argument("--attach", action="store_true",
                    help="soak a server that is already running on --host/--port")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18431)
    ap.add_argument("--server-arg", dest="server_args", action="append", default=[],
                    help="one extra argument for the server, repeatable (--server-arg=--kv --server-arg=q8)")
    ap.add_argument("--server-env", action="append", default=[], metavar="NAME=VALUE",
                    help="one environment variable for the server, repeatable")
    ap.add_argument("--skip", action="append", default=[],
                    choices=["identity", "cancel", "swap", "limit", "concurrent"])
    ap.add_argument("--cycles", type=int, default=200)
    ap.add_argument("--cancels", type=int, default=50, help="cancels of each of the four kinds")
    ap.add_argument("--swaps", type=int, default=30)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--concurrent-seconds", type=float, default=300)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=0, help="context window, default: read from /v1/models")
    ap.add_argument("--cancel-prompt-tokens", type=int, default=2048)
    ap.add_argument("--cancel-delay", type=float, default=0.05,
                    help="seconds a cancel waits after its request shows up in its phase")
    ap.add_argument("--phase-timeout", type=float, default=20,
                    help="seconds to wait for a request to show up in the phase to cancel it in")
    ap.add_argument("--filler-base", type=int, default=1000,
                    help="lowest token id used for filler prompts")
    ap.add_argument("--filler-vocab", type=int, default=500,
                    help="how many token ids from --filler-base on the filler prompts use")
    ap.add_argument("--growth-mb", type=float, default=64,
                    help="memory growth after warm-up that counts as a finding")
    ap.add_argument("--growth-frac", type=float, default=0.03)
    ap.add_argument("--timeout", type=float, default=900, help="per-request timeout, seconds")
    ap.add_argument("--idle-timeout", type=float, default=120)
    ap.add_argument("--load-timeout", type=float, default=900)
    ap.add_argument("--prompt", default="Name three primary colours, then stop.")
    ap.add_argument("--long-prompt", default="Write a long story about a lighthouse keeper.")
    ap.add_argument("--workdir", default="soak-out")
    ap.add_argument("--out", help="JSON report path, default <workdir>/report.json")
    a = ap.parse_args()
    if not a.attach and not a.model:
        ap.error("--model is required unless --attach is given")
    if a.port == 8080 and not a.attach:
        ap.error("the soak does not start a server on 8080, the default port of a real one")
    Path(a.workdir).mkdir(parents=True, exist_ok=True)
    socket.setdefaulttimeout(a.timeout)
    soak = Soak(a)
    t0 = time.time()
    rc = soak.run()
    soak.report["seconds"] = round(time.time() - t0, 1)
    soak.report["argv"] = sys.argv[1:]
    out = Path(a.out) if a.out else Path(a.workdir) / "report.json"
    out.write_text(json.dumps(soak.report, indent=1) + "\n")
    print(f"soak: {len(soak.findings)} finding(s) in {soak.report['seconds']} s; "
          f"device memory {soak.report.get('gpu_memory', 'unmeasured')}; report {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
