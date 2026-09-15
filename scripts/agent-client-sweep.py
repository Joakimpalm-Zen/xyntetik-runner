#!/usr/bin/env python3
"""Real coding-agent clients against one served model, end to end.

The README's coding-agent evidence is a dated executable observation:
which client, which version, completed a tool -> execution -> result ->
answer loop against the runner. This script is that observation as a
harness. For every client it can find it writes a fixture directory with
a sentinel generated per run (so a pass cannot come from anything the model
already knew), runs the client's non-interactive mode with the one task
"read the file and reply with the sentinel", and judges the run by the
sentinel coming back. Each row records the client's version, the runner's
version and binary hash, the served model and its hash, the wall time, the
client's output tail and the runner's request log for that window, and is
written to the JSON as soon as it completes.

Rows: opencode (Chat Completions via @ai-sdk/openai-compatible), claude
(Anthropic Messages), codex (Responses), continue (`cn`, Chat Completions),
cline (Chat Completions), pi (Chat Completions), aider (Chat Completions,
`--dry-run`: transport and inference only, the edit-format row the README
declares). A client that is not installed is reported NOT RUN, never as a
pass; `--require CLIENT,...` turns NOT RUN into a failing exit for the
rows a release needs.

Usage:
  agent-client-sweep.py --base-url http://127.0.0.1:8080 --model-file M.gguf
                        [--runner ./runner] [--clients a,b,c] [--require a,b]
                        [--deadline S] --out sweep.json
"""
import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return json.load(r)


def version_of(cmd, args=("--version",)):
    exe = shutil.which(cmd)
    if not exe:
        return None, None
    try:
        out = subprocess.run([exe, *args], capture_output=True, text=True,
                             timeout=120).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    return exe, out.splitlines()[0] if out else "unknown"


def run(cmd, cwd, env, deadline, stdin_text=None):
    t0 = time.monotonic()
    # a shell updates PWD on cd and some clients read it (opencode found its
    # project config through it, and without this fell back to the user's
    # global provider of the same name); subprocess's cwd does not
    env = dict(env, PWD=cwd)
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, input=stdin_text,
                           capture_output=True, text=True, timeout=deadline)
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = -1, (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), "deadline %ss" % deadline
    return rc, out, err, round(time.monotonic() - t0, 1)


PROMPT = ("Read the file sentinel.txt in the current directory with your "
          "tools and reply with the code word it contains, in one sentence.")


class Sweep:
    def __init__(self, a):
        self.a = a
        self.base = a.base_url
        self.port = self.base.rsplit(":", 1)[-1].rstrip("/")
        self.model = a.model or get(self.base, "/v1/models")["data"][0]["id"]
        self.caps = get(self.base, "/v1/capabilities")
        self.rows = []

    # ---- fixtures --------------------------------------------------------
    def fixture(self, name):
        d = tempfile.mkdtemp(prefix="agent-sweep-%s-" % name)
        sentinel = "XYNTETIK-%04X-%04X" % (random.getrandbits(16), random.getrandbits(16))
        with open(os.path.join(d, "sentinel.txt"), "w") as f:
            f.write("The code word is %s.\n" % sentinel)
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write("# fixture\n")
        os.makedirs(os.path.join(d, "src"))
        with open(os.path.join(d, "src", "main.py"), "w") as f:
            f.write("print('hello')\n")
        home = os.path.join(d, "home")
        os.makedirs(home)
        return d, sentinel, home

    def env(self, home, **extra):
        e = dict(os.environ)
        e["HOME"] = home
        e.update(extra)
        return e

    # ---- clients ---------------------------------------------------------
    def opencode(self, d, sentinel, home):
        exe, ver = version_of("opencode")
        if not exe:
            return None, ver, None
        with open(os.path.join(d, "opencode.json"), "w") as f:
            json.dump({"$schema": "https://opencode.ai/config.json", "provider": {
                "runner": {"npm": "@ai-sdk/openai-compatible", "name": "Xyntetik Runner",
                           "options": {"baseURL": self.base + "/v1"},
                           "models": {self.model: {"name": "runner",
                                                   "limit": {"context": self.caps["context"],
                                                             "output": 4096}}}}}}, f)
        # opencode's own server needs its real state directories (a redirected
        # HOME made it fail with "Unexpected server error" before a request was
        # made); the provider comes from the project-level opencode.json above
        env = dict(os.environ, OPENCODE_PERMISSION='{"*":"allow"}')
        return [exe, "run", "--model", "runner/" + self.model, PROMPT], ver, env

    def claude(self, d, sentinel, home):
        exe, ver = version_of("claude")
        if not exe:
            return None, ver, None
        env = self.env(home, ANTHROPIC_BASE_URL=self.base, ANTHROPIC_API_KEY="not-used",
                       ANTHROPIC_MODEL=self.model, CLAUDE_CONFIG_DIR=os.path.join(home, "claude"))
        return [exe, "-p", "--model", self.model, PROMPT, "--allowedTools", "Read"], ver, env

    def codex(self, d, sentinel, home):
        exe, ver = version_of("codex")
        if not exe:
            return None, ver, None
        cfg = os.path.join(home, ".codex")
        os.makedirs(cfg)
        with open(os.path.join(cfg, "config.toml"), "w") as f:
            # hosted web search is declared as a live tool the runner refuses
            # by name; the README's Codex recipe disables it for a local
            # session (both spellings, for the versions that read either)
            f.write('model = "%s"\nmodel_provider = "runner"\n'
                    'web_search = "disabled"\n'
                    '[model_providers.runner]\nname = "Xyntetik Runner"\n'
                    'base_url = "%s/v1"\nwire_api = "responses"\nenv_key = "RUNNER_API_KEY"\n'
                    '[features]\nweb_search_request = false\n'
                    % (self.model, self.base))
        env = self.env(home, RUNNER_API_KEY="none", CODEX_HOME=cfg)
        # read-only is enough for the loop (the file is read by the sandboxed
        # shell; 0.154 dropped --full-auto from `exec`); a container without
        # unprivileged user namespaces cannot start Codex's bubblewrap sandbox
        # at all, and --codex-no-sandbox runs it unsandboxed THERE, where the
        # container is the sandbox
        sandbox = ["--dangerously-bypass-approvals-and-sandbox"] \
            if self.a.codex_no_sandbox else ["-s", "read-only"]
        return [exe, "exec", *sandbox, "--skip-git-repo-check", "-C", d,
                PROMPT], ver, env

    def continue_cli(self, d, sentinel, home):
        exe, ver = version_of("cn")
        if not exe:
            return None, ver, None
        cfg = os.path.join(home, ".continue")
        os.makedirs(cfg)
        path = os.path.join(cfg, "config.yaml")
        with open(path, "w") as f:
            f.write("name: runner-sweep\nversion: 0.0.1\nschema: v1\nmodels:\n"
                    "  - name: runner\n    provider: openai\n    model: %s\n"
                    "    apiBase: %s/v1\n    apiKey: none\n"
                    "    capabilities:\n      - tool_use\n"
                    "    roles:\n      - chat\n      - edit\n      - apply\n"
                    % (self.model, self.base))
        env = self.env(home, CONTINUE_GLOBAL_DIR=cfg)
        return [exe, "-p", "--auto", "--config", path, PROMPT], ver, env

    def cline(self, d, sentinel, home):
        exe, ver = version_of("cline")
        if not exe:
            return None, ver, None
        env = self.env(home)
        cfg = os.path.join(home, ".cline")
        # the provider is configured once, isolated under the fixture home
        subprocess.run([exe, "auth", "-p", self.a.cline_provider, "-k", "none",
                        "-m", self.model, "-b", self.base + "/v1",
                        "--config", cfg, "--data-dir", os.path.join(cfg, "data")],
                       cwd=d, env=env, capture_output=True, text=True, timeout=120)
        return [exe, PROMPT, "--config", cfg, "--data-dir", os.path.join(cfg, "data"),
                "--auto-approve", "true", "--thinking", "none", "-c", d,
                "-t", str(self.a.deadline)], ver, env

    def pi(self, d, sentinel, home):
        exe, ver = version_of("pi")
        if not exe:
            return None, ver, None
        agent = os.path.join(home, ".pi", "agent")
        os.makedirs(agent)
        with open(os.path.join(agent, "models.json"), "w") as f:
            json.dump({"providers": {"runner": {
                "baseUrl": self.base + "/v1", "api": "openai-completions",
                "apiKey": "none",
                "models": [{"id": self.model, "name": "runner", "reasoning": False,
                            "input": ["text"], "contextWindow": self.caps["context"],
                            "maxTokens": 4096, "cost": {"input": 0, "output": 0,
                                                        "cacheRead": 0, "cacheWrite": 0}}]}}}, f)
        env = self.env(home)
        return [exe, "-p", "--no-session", "--no-context-files", "--provider", "runner",
                "--model", self.model, PROMPT], ver, env

    def aider(self, d, sentinel, home):
        exe, ver = version_of("aider")
        if not exe:
            return None, ver, None
        env = self.env(home, OPENAI_API_KEY="none", OPENAI_API_BASE=self.base + "/v1")
        return [exe, "--model", "openai/" + self.model, "--no-git", "--yes-always",
                "--dry-run", "--no-show-model-warnings", "--no-check-update",
                "--message", "Read sentinel.txt and reply with the code word it "
                "contains, in one sentence.", "sentinel.txt"], ver, env

    CLIENTS = {"opencode": opencode, "claude": claude, "codex": codex,
               "continue": continue_cli, "cline": cline, "pi": pi, "aider": aider}

    # ---- the loop ----------------------------------------------------------
    def row(self, name):
        d, sentinel, home = self.fixture(name)
        cmd, ver, env = self.CLIENTS[name](self, d, sentinel, home)
        entry = {"client": name, "version": ver, "sentinel": sentinel,
                 "result": "NOT RUN", "reason": None, "wall_s": None}
        if not cmd:
            entry["reason"] = "not installed"
            shutil.rmtree(d, ignore_errors=True)
            return entry
        entry["command"] = cmd
        rc, out, err, wall = run(cmd, d, env, self.a.deadline)
        entry["wall_s"] = wall
        entry["exit"] = rc
        entry["stdout_tail"] = out[-1500:]
        entry["stderr_tail"] = err[-800:]
        entry["result"] = "PASS" if sentinel in out else "FAIL"
        if entry["result"] == "FAIL":
            entry["reason"] = "the sentinel did not come back" + (
                " (deadline)" if rc == -1 else "")
        if not self.a.keep:
            shutil.rmtree(d, ignore_errors=True)
        else:
            entry["fixture"] = d
        return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model")
    ap.add_argument("--model-file")
    ap.add_argument("--runner")
    ap.add_argument("--clients", default=",".join(Sweep.CLIENTS))
    ap.add_argument("--require", default="")
    ap.add_argument("--deadline", type=int, default=1800)
    ap.add_argument("--codex-no-sandbox", action="store_true",
                    help="run Codex without its bubblewrap sandbox (a container "
                         "that forbids user namespaces cannot start it)")
    ap.add_argument("--cline-provider", default="openai",
                    help="Cline's provider id for an OpenAI-compatible server")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = Sweep(a)
    runner_ver = None
    if a.runner:
        runner_ver = subprocess.run([a.runner, "--version"], capture_output=True,
                                    text=True).stdout.strip()
    record = {"gate": "agent-client-sweep", "label": a.label,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "host": platform.node(), "platform": platform.platform(),
              "base_url": a.base_url, "model": s.model,
              "model_file": a.model_file,
              "model_sha256": sha256(a.model_file) if a.model_file else None,
              "runner": a.runner, "runner_version": runner_ver,
              "runner_sha256": sha256(a.runner) if a.runner else None,
              "sampling_served": s.caps.get("sampling"),
              "template": s.caps.get("template"),
              "tool_protocol": s.caps.get("tool_protocol"),
              "context": s.caps.get("context"), "prompt": PROMPT,
              "rows": [], "verdict": "NOT RUN"}

    def write():
        with open(a.out, "w") as f:
            json.dump(record, f, indent=1)

    write()
    required = set(x for x in a.require.split(",") if x)
    ok = True
    for name in a.clients.split(","):
        e = s.row(name)
        record["rows"].append(e)
        print("%-9s %-8s %-10s %s%s" % (name, e["result"], e.get("version") or "-",
                                         "%ss" % e["wall_s"] if e["wall_s"] else "",
                                         " " + e["reason"] if e.get("reason") else ""),
              flush=True)
        if e["result"] == "FAIL" or (e["result"] == "NOT RUN" and name in required):
            ok = False
        write()
    record["verdict"] = "PASS" if ok else "FAIL"
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
