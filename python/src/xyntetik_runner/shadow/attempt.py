"""A bounded local attempt: the model, a scratch copy, four tools, a budget.

The attempt sees the pre-state tree, the user's request and the names of the
test files that existed at the pre-state. It gets four tools: list files,
read a file, write a file, run the visible tests; and ``finish``. There is
no shell. Every path is resolved inside the scratch copy and a path that
escapes it is refused, so the attempt cannot read or write the original
repository. The budget bounds turns, tool calls, wall clock and test runs;
running out of any of them ends the attempt with that reason and the
verifier then judges whatever is in the tree.

Not a sandbox: ``run_tests`` executes the tests, and therefore the code the
model wrote, as the calling user with the network reachable. That is the
same exposure the verifier has and is the subject of R14.3.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from xyntetik_runner.shadow.baseline import IGNORED_DIRS
from xyntetik_runner.shadow.scaffold import BASE_SYSTEM, Scaffold

ChatFn = Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]
"""``chat(messages, tools) -> assistant message`` (``content``, optional
``tool_calls`` in the OpenAI shape, optional ``usage``)."""

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List files under the workspace matching a glob "
        "(default: everything). Directories like .git and __pycache__ are hidden.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}},
                       "required": []}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a UTF-8 text file from the workspace. "
        "Long files are returned in windows: pass offset (line, 0-based) to continue.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "offset": {"type": "integer"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write the complete new content of a file in the "
        "workspace, creating it if needed.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace one exact, unique occurrence of old_text in "
        "a workspace file with new_text. Use this for large files instead of rewriting them; "
        "old_text must match exactly once.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_text": {"type": "string"},
            "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the visible test files with pytest and return "
        "the tail of the output.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "finish", "description": "Declare the task done. Call this when the tests pass "
        "or when you cannot make further progress.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}},
                       "required": []}}},
]

SYSTEM_PROMPT = BASE_SYSTEM  # the base scaffold; tools are listed in TOOLS


@dataclass(frozen=True)
class Budget:
    max_turns: int = 12
    max_tool_calls: int = 40
    wall_s: float = 900.0
    max_tokens: int = 1500
    test_runs: int = 4
    test_timeout_s: float = 120.0
    read_lines: int = 200
    read_chars: int = 12000
    list_entries: int = 200
    output_chars: int = 3000


@dataclass(frozen=True)
class AttemptResult:
    turns: int
    tool_calls: int
    test_runs: int
    prompt_tokens: int
    completion_tokens: int
    wall_s: float
    stop_reason: str
    finished: bool
    summary: str = ""
    tool_names: tuple[str, ...] = ()
    transcript: tuple[dict[str, Any], ...] = ()
    """Every message of the attempt, the local model's own words and the
    tool results it saw. Kept as evidence: it is what the record explains."""


class Workspace:
    """Path confinement and the four tools over one directory."""

    def __init__(self, root: Path, *, visible_tests: Sequence[str], pythonpath: Sequence[str],
                 python: str, budget: Budget):
        self.root = root.resolve()
        self.visible_tests = tuple(visible_tests)
        self.pythonpath = tuple(pythonpath)
        self.python = python
        self.budget = budget
        self.test_runs = 0

    def resolve(self, rel: str) -> Path:
        if not isinstance(rel, str) or not rel or rel.startswith(("/", "\\")) or ":" in rel[:3]:
            raise ValueError("path must be relative to the workspace")
        target = (self.root / rel).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError("path escapes the workspace")
        return target

    def list_files(self, pattern: str = "**/*") -> str:
        pattern = pattern or "**/*"
        if pattern.startswith(("/", "\\")) or ".." in pattern:
            return "error: pattern must be relative and may not contain .."
        out: list[str] = []
        for p in sorted(self.root.glob(pattern)):
            rel = p.relative_to(self.root)
            if any(part in IGNORED_DIRS for part in rel.parts):
                continue
            if p.is_file():
                out.append(rel.as_posix())
            if len(out) >= self.budget.list_entries:
                out.append(f"... (truncated at {self.budget.list_entries} entries)")
                break
        return "\n".join(out) if out else "(no files match)"

    def read_file(self, path: str, offset: int = 0) -> str:
        try:
            target = self.resolve(path)
        except ValueError as e:
            return f"error: {e}"
        if not target.is_file():
            return "error: no such file"
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as e:
            return f"error: {e}"
        offset = max(0, int(offset or 0))
        window = lines[offset:offset + self.budget.read_lines]
        text = "\n".join(f"{offset + i + 1}: {line}" for i, line in enumerate(window))
        if len(text) > self.budget.read_chars:
            text = text[:self.budget.read_chars] + "\n... (truncated)"
        remaining = len(lines) - (offset + len(window))
        if remaining > 0:
            text += f"\n... {remaining} more line(s); call again with offset={offset + len(window)}"
        return text or "(empty file)"

    def write_file(self, path: str, content: str) -> str:
        try:
            target = self.resolve(path)
        except ValueError as e:
            return f"error: {e}"
        if not isinstance(content, str):
            return "error: content must be a string"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as e:
            return f"error: {e}"
        return f"wrote {len(content)} chars to {path}"

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        try:
            target = self.resolve(path)
        except ValueError as e:
            return f"error: {e}"
        if not target.is_file():
            return "error: no such file"
        if not isinstance(old_text, str) or not old_text:
            return "error: old_text must be a non-empty string"
        if not isinstance(new_text, str):
            return "error: new_text must be a string"
        try:
            text = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            return f"error: {e}"
        n = text.count(old_text)
        if n == 0:
            # the anchor was invented (measured: 17 of 52 edit answers quote text
            # that is not in the file); name the file's nearest lines so the
            # next call copies one instead of guessing again
            import difflib
            want = next((x.strip() for x in old_text.split("\n") if x.strip()), old_text.strip())
            near = difflib.get_close_matches(want, [x.strip() for x in text.split("\n") if x.strip()], n=3, cutoff=0.5)
            hint = ("; the nearest lines in the file are: " + " | ".join(repr(x) for x in near)) if near else ""
            return f"error: old_text not found; read the file and copy the exact text{hint}"
        if n > 1:
            return f"error: old_text occurs {n} times; include more surrounding lines"
        try:
            target.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        except OSError as e:
            return f"error: {e}"
        return f"edited {path}: replaced {len(old_text)} chars with {len(new_text)}"

    def run_tests(self) -> str:
        if self.test_runs >= self.budget.test_runs:
            return f"error: test-run budget of {self.budget.test_runs} exhausted"
        self.test_runs += 1
        targets = [t for t in self.visible_tests if (self.root / t).is_file()]
        if not targets:
            targets = ["tests"] if (self.root / "tests").is_dir() else ["."]
        env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "SystemRoot", "TEMP", "TMP")
               if k in os.environ}
        env.update({"HOME": str(self.root), "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": os.pathsep.join([str(self.root)] +
                                                  [str(self.root / r) for r in self.pythonpath])})
        cmd = [self.python, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *targets]
        proc = subprocess.Popen(cmd, cwd=self.root, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=(os.name == "posix"))
        try:
            out, _ = proc.communicate(timeout=self.budget.test_timeout_s)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except OSError:
                pass
            out, _ = proc.communicate()
            return f"error: tests timed out after {self.budget.test_timeout_s:g}s"
        text = (out or b"").decode("utf-8", errors="replace")
        if len(text) > self.budget.output_chars:
            text = "...\n" + text[-self.budget.output_chars:]
        return f"exit code {proc.returncode}\n{text}"


def _fn(call: dict[str, Any]) -> dict[str, Any]:
    fn = call.get("function")
    return fn if isinstance(fn, dict) else call


def _args(call: dict[str, Any]) -> dict[str, Any]:
    raw = _fn(call).get("arguments", {})
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _name(call: dict[str, Any]) -> str:
    return str(_fn(call).get("name") or "")


def budget_with(scaffold: Scaffold, budget: Budget) -> Budget:
    """The budget with the scaffold's overrides, bounded above by the caller's
    so a scaffold can spend less, never more."""
    fields = {k: v for k, v in scaffold.budget.items() if hasattr(budget, k)}
    kw: dict[str, Any] = {}
    for k, v in fields.items():
        cur = getattr(budget, k)
        kw[k] = type(cur)(min(v, cur))
    return replace(budget, **kw)


def attempt(request: str, workspace: Workspace, chat: ChatFn, *, budget: Budget | None = None,
            context: Sequence[str] = (), scaffold: Scaffold | None = None) -> AttemptResult:
    scaffold = scaffold or Scaffold.base()
    budget = budget_with(scaffold, budget or workspace.budget)
    workspace.budget = budget
    tools = scaffold.apply_tools(TOOLS)
    listing = workspace.list_files("*")
    earlier = ""
    if context:
        joined = "\n\n".join(f"- {c.strip()}" for c in context)
        earlier = f"Earlier requests in this session, oldest first:\n{joined}\n\n"
    user = (f"{earlier}Task:\n{request.strip()}\n\nVisible test files: "
            f"{', '.join(workspace.visible_tests) or '(none at this state; look for tests/)'}\n\n"
            f"Top-level files:\n{listing}")
    messages: list[dict[str, Any]] = [{"role": "system", "content": scaffold.system_text()},
                                      {"role": "user", "content": user}]
    t0 = time.monotonic()
    turns = calls = ptoks = ctoks = 0
    nudged = False
    names: list[str] = []
    while True:
        if turns >= budget.max_turns:
            return _done(turns, calls, workspace, ptoks, ctoks, t0, "turn budget", False, names,
                         messages=messages)
        if time.monotonic() - t0 > budget.wall_s:
            return _done(turns, calls, workspace, ptoks, ctoks, t0, "wall clock", False, names,
                         messages=messages)
        turns += 1
        try:
            reply = chat(messages, tools)
        except Exception as e:  # the model side failed; the tree is still judged
            return _done(turns, calls, workspace, ptoks, ctoks, t0,
                         f"model error: {type(e).__name__}: {str(e)[:240]}", False, names,
                         messages=messages)
        usage_raw = reply.get("usage")
        usage: dict[str, Any] = usage_raw if isinstance(usage_raw, dict) else {}
        ptoks += int(usage.get("prompt_tokens", 0) or 0)
        ctoks += int(usage.get("completion_tokens", 0) or 0)
        calls_raw = reply.get("tool_calls")
        tool_calls: list[dict[str, Any]] = [c for c in calls_raw if isinstance(c, dict)] \
            if isinstance(calls_raw, list) else []
        assistant: dict[str, Any] = {"role": "assistant", "content": reply.get("content") or ""}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        messages.append(assistant)
        if not tool_calls:
            if nudged:
                return _done(turns, calls, workspace, ptoks, ctoks, t0, "no tool call", False, names,
                             messages=messages)
            nudged = True
            messages.append({"role": "user", "content": "Use the tools to make the change, run "
                             "the tests, then call finish."})
            continue
        for call in tool_calls:
            calls += 1
            if calls > budget.max_tool_calls:
                return _done(turns, calls, workspace, ptoks, ctoks, t0, "tool-call budget", False,
                             names, messages=messages)
            name, args = _name(call), _args(call)
            names.append(name)
            if name == "finish":
                return _done(turns, calls, workspace, ptoks, ctoks, t0, "finish", True, names,
                             str(args.get("summary") or ""), messages=messages)
            if name == "list_files":
                result = workspace.list_files(str(args.get("pattern") or "**/*"))
            elif name == "read_file":
                result = workspace.read_file(str(args.get("path") or ""), int(args.get("offset") or 0))
            elif name == "write_file":
                result = workspace.write_file(str(args.get("path") or ""), args.get("content"))  # type: ignore[arg-type]
            elif name == "edit_file":
                result = workspace.edit_file(str(args.get("path") or ""), args.get("old_text"),  # type: ignore[arg-type]
                                             args.get("new_text"))  # type: ignore[arg-type]
            elif name == "run_tests":
                result = workspace.run_tests()
            else:
                result = f"error: unknown tool {name!r}"
            messages.append({"role": "tool", "tool_call_id": str(call.get("id") or f"call_{calls}"),
                             "name": name, "content": result})


def _done(turns: int, calls: int, ws: Workspace, ptoks: int, ctoks: int, t0: float, reason: str,
          finished: bool, names: list[str], summary: str = "",
          messages: Sequence[dict[str, Any]] = ()) -> AttemptResult:
    return AttemptResult(turns=turns, tool_calls=calls, test_runs=ws.test_runs,
                         prompt_tokens=ptoks, completion_tokens=ctoks,
                         wall_s=round(time.monotonic() - t0, 3), stop_reason=reason,
                         finished=finished, summary=summary, tool_names=tuple(sorted(set(names))),
                         transcript=tuple(messages))


def probe_speed(post_json: Callable[..., dict[str, Any]], model: str, *, tokens: int = 64
                ) -> float:
    """Decode speed in tokens per second from one short generation: the
    fit-first rule measured rather than assumed. A model whose weights or
    context spilled the device shows up here as a crawl."""
    payload = {"model": model, "messages": [{"role": "user", "content":
               "Write the numbers from one to two hundred as words, separated by commas."}],
               "max_tokens": tokens, "temperature": 0}
    t0 = time.monotonic()
    data = post_json("/v1/chat/completions", payload)
    wall = max(time.monotonic() - t0, 1e-6)
    usage_raw = data.get("usage")
    usage: dict[str, Any] = usage_raw if isinstance(usage_raw, dict) else {}
    generated = int(usage.get("completion_tokens", 0) or 0)
    return generated / wall if generated else 0.0


def runner_chat(post_json: Callable[..., dict[str, Any]], model: str, *, max_tokens: int,
                temperature: float = 0.0) -> ChatFn:
    """Adapt ``RunnerEndpoint.post_json`` to the ``ChatFn`` shape."""
    def chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        # "required" constrains every turn to a real tool call through the
        # runner's schema-driven decoding; finish is a tool, so the loop can
        # still end. Without it Qwen2.5-Coder wrote its calls as JSON code
        # blocks in the text and the runner rightly returned no tool call.
        payload = {"model": model, "messages": messages, "tools": tools,
                   "tool_choice": "required", "max_tokens": max_tokens,
                   "temperature": temperature}
        data = post_json("/v1/chat/completions", payload)
        choice = data["choices"][0]
        msg = dict(choice["message"])
        if isinstance(data.get("usage"), dict):
            msg["usage"] = data["usage"]
        return msg
    return chat

