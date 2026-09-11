"""What recurs in the user's work, by what it produced (R15.13).

Measured on a real ledger before this module was written: by wording, a
developer's requests are a long tail, 1,580 of 1,644 clusters singletons
and the hundred largest covering 15% of the traffic. By what the work
*produced* they concentrate hard, the ten largest kinds covering about half
of it. So this groups episodes by their effect on the repository and the
machine, never by how the request was phrased.

Two rules the grouping follows, both of them corrections to a first attempt
that got them wrong:

- **The label is the effect, not the mechanism.** "Edited a file with the
  edit tool" and "rewrote it with a shell heredoc" are the same work done
  two ways, and a grouping that separates them is describing the assistant's
  habits rather than the user's work. Delegation to sub-agents is recorded
  as an attribute for the same reason: it is how the work was carried out,
  not what came of it.
- **Harness text is not a request.** Task notifications, slash commands and
  local-command output made up 26% of the first ledger measured. They are
  excluded here, as the importer now excludes them at capture.

This is a *retrospective* inventory and the docstring says so because the
distinction is load-bearing: a companion measurement found that the kind of
work a request becomes is **not predictable from the request**, with a 1.5B
model, a trained classifier and the previous episode's kind all scoring
below the majority-class baseline. Nothing here may be used to route a
request before the work happens. What it is for is choosing which recurring
kinds of work deserve a schema and a typed procedure, and showing the user
where their time actually goes.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

# Shell plumbing appears in every kind of work and separates none of it.
_SRC = re.compile(r"\.(c|h|py|sh|js|ts|tsx|go|rs|metal|cu|toml|cfg|ini|yml|yaml|json|sql)$|(^|/)Makefile$")
_DOC = re.compile(r"\.(md|txt|rst|adoc)$")
_WRITE = re.compile(r"(?:^|[;&|]\s*)(?:cat|tee)\s*>\s*(\S+)"
                    r"|>\s*(\S+\.(?:py|sh|c|h|md|json|txt|yml|yaml|toml))\b"
                    r"|\bsed\s+-i[^|;&]*\s(\S+)")
_TEST = re.compile(r"\b(pytest|make(?:\s|$)|npm\s+(?:test|run)|cargo\s+test|go\s+test|ctest)\b|\./test-")
_REMOTE = re.compile(r"\b(ssh|scp|rsync|bwexec|schtasks|taskkill)\b")
_WEBCMD = re.compile(r"\bcurl\s+[^|]*https?://|\bwget\s+https?://")
_COMMIT = re.compile(r"\bgit\s+(commit|push)\b")
_HARNESS = ("<", "/", "[Image", "Caveat:")

KINDS = ("code_change", "doc_change", "execution", "web_read", "inspection")
KIND_TEXT = {
    "code_change": "changed source",
    "doc_change": "changed documents",
    "execution": "ran tests, a build or a remote job",
    "web_read": "read the web",
    "inspection": "read and answered, changed nothing",
}


@dataclass
class Effects:
    """What one episode did. Counts, so a caller can weigh as well as group."""
    code: int = 0
    docs: int = 0
    tests: int = 0
    remote: int = 0
    web: int = 0
    reads: int = 0
    delegated: int = 0
    commit: int = 0

    @property
    def touched(self) -> bool:
        return bool(self.code or self.docs)

    def kind(self) -> str:
        """The single label: what the episode produced, in the order a reader
        would call most consequential. Delegation is deliberately not a kind."""
        if self.code:
            return "code_change"
        if self.docs:
            return "doc_change"
        if self.tests or self.remote:
            return "execution"
        if self.web:
            return "web_read"
        return "inspection"

    def empty(self) -> bool:
        return not any((self.code, self.docs, self.tests, self.remote, self.web,
                        self.reads, self.delegated))


@dataclass(frozen=True)
class Observation:
    request: str
    kind: str
    delegated: bool
    commit: bool
    at: str
    session: str
    project: str = ""   # the directory a harness keeps one project's sessions in


@dataclass
class Cluster:
    kind: str
    size: int = 0
    delegated: int = 0
    commits: int = 0
    first_seen: str = ""
    last_seen: str = ""
    examples: list[str] = field(default_factory=list)


def is_request(text: Any) -> bool:
    """A user's own words, not the harness's. The same rule the importer
    applies at capture; applied again here because a ledger written before
    that fix still carries them."""
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t or t.startswith(_HARNESS):
        return False
    head = t[:200]
    return "<task-notification>" not in head and "<local-command" not in head


def effects_of(tool_calls: Iterable[tuple[str, dict[str, Any]]]) -> Effects:
    """Fold one episode's tool calls into what they did."""
    e = Effects()
    for name, inp in tool_calls:
        if name in ("Edit", "Write", "NotebookEdit"):
            path = str(inp.get("file_path") or "")
            if _SRC.search(path):
                e.code += 1
            elif path:
                e.docs += 1
        elif name in ("WebSearch", "WebFetch"):
            e.web += 1
        elif name == "Agent":
            e.delegated += 1
        elif name in ("Read", "Glob", "Grep"):
            e.reads += 1
        elif name == "Bash":
            cmd = str(inp.get("command") or "")
            for m in _WRITE.finditer(cmd):
                target = next((g for g in m.groups() if g), "")
                if _SRC.search(target):
                    e.code += 1
                elif _DOC.search(target):
                    e.docs += 1
            if _TEST.search(cmd):
                e.tests += 1
            if _REMOTE.search(cmd):
                e.remote += 1
            if _WEBCMD.search(cmd):
                e.web += 1
            if _COMMIT.search(cmd):
                e.commit += 1
            e.reads += 1
    return e


def _lines(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    yield rec
    except OSError:
        return


def observe_session(path: Path) -> list[Observation]:
    """One session file: each of the user's requests, and what followed it
    until the next one."""
    out: list[Observation] = []
    sid, project = path.stem, path.parent.name
    req: str | None = None
    at = ""
    calls: list[tuple[str, dict[str, Any]]] = []

    def close() -> None:
        nonlocal req, calls, at
        if req is not None and calls:
            e = effects_of(calls)
            if not e.empty():
                out.append(Observation(req, e.kind(), bool(e.delegated), bool(e.commit), at, sid, project))
        req, calls, at = None, [], ""

    for rec in _lines(path):
        if rec.get("type") == "user" and not rec.get("isMeta") and not rec.get("isCompactSummary"):
            content = (rec.get("message") or {}).get("content") if isinstance(rec.get("message"), dict) else None
            if is_request(content):
                close()
                req = str(content).strip()
                at = str(rec.get("timestamp") or "")
                continue
        if req is None or rec.get("type") != "assistant":
            continue
        content = (rec.get("message") or {}).get("content") if isinstance(rec.get("message"), dict) else None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    calls.append((str(block.get("name") or ""), block.get("input") or {}))
    close()
    return out


def observe(roots: Iterable[Path], min_chars: int = 60) -> list[Observation]:
    """Every session under ``roots``. ``min_chars`` drops continuations like
    "yes" and "go ahead", which inherit their kind from the conversation and
    describe no work of their own."""
    out: list[Observation] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            out += [o for o in observe_session(path) if len(o.request) >= min_chars]
    return out


def cluster(observations: Iterable[Observation]) -> list[Cluster]:
    by: dict[str, Cluster] = {}
    for o in observations:
        c = by.setdefault(o.kind, Cluster(kind=o.kind))
        c.size += 1
        c.delegated += o.delegated
        c.commits += o.commit
        if o.at:
            c.first_seen = min(c.first_seen or o.at, o.at)
            c.last_seen = max(c.last_seen, o.at)
        if len(c.examples) < 3:
            c.examples.append(re.sub(r"\s+", " ", o.request)[:96])
    return sorted(by.values(), key=lambda c: -c.size)


MIN_FOR_SHARES = 30


def render(observations: list[Observation], *, sessions: int = 0) -> str:
    """Counts before rates, and no share at all below the same floor the
    capability report uses."""
    n = len(observations)
    if not n:
        return ("no recurring work yet: no session carried a request with actions after it.\n"
                "Work through a few tasks and run this again.")
    clusters = cluster(observations)
    show_shares = n >= MIN_FOR_SHARES
    head = f"{n} episodes with actions"
    if sessions:
        head += f" across {sessions} sessions"
    lines = [head + ("" if show_shares else
                     f" (shares withheld below {MIN_FOR_SHARES} episodes)"), ""]
    lines.append("| kind | episodes |" + (" share |" if show_shares else "") +
                 " ended in a commit | delegated |")
    lines.append("|---|---:|" + ("---:|" if show_shares else "") + "---:|---:|")
    for c in clusters:
        row = f"| {KIND_TEXT[c.kind]} | {c.size} |"
        if show_shares:
            row += f" {100 * c.size / n:.0f}% |"
        lines.append(row + f" {c.commits} | {c.delegated} |")
    lines.append("")
    for c in clusters:
        lines.append(f"{KIND_TEXT[c.kind]} ({c.size}):")
        for ex in c.examples:
            lines.append(f"  {ex}")
    lines += ["",
              "This is what your work HAS been, not a prediction of what a request will become:",
              "the kind of work a request turns into was measured to be unpredictable from the",
              "request itself. Use it to choose which recurring work is worth a typed procedure."]
    return "\n".join(lines)


def default_roots(home: Path | None = None) -> list[Path]:
    home = home or Path.home()
    return [home / ".claude" / "projects", home / ".codex" / "sessions"]
