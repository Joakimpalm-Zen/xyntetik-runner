"""Import episodes from the frontier tools' own traces, without their answers.

Codex and Claude Code both write session logs to the user's home directory.
An episode is one task boundary in those logs: for Codex the
``task_started`` / ``task_complete`` events of a session; for Claude Code
one user prompt up to the next. The importer reads the working directory,
the timestamps and the user's own request text, and for Codex the NAMES of
the tools the session called. It never reads an assistant message, a
reasoning item, a tool argument or a tool result: those carry the
frontier's answer, and the whole point of the instrument is that the local
attempt does not get it.

Formats are undocumented and will move. A file the parser cannot read is
skipped and counted, never guessed at.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

CLAUDE_TURN_CAP = timedelta(hours=1)


@dataclass(frozen=True)
class Episode:
    source: str
    session_id: str
    turn: int
    cwd: str
    started_at: str
    ended_at: str
    request: str
    request_sha256: str
    tool_names: tuple[str, ...] = ()
    trace_path: str = ""
    head_start: str = ""
    head_end: str = ""
    """Repository HEAD when the request was made and when the turn ended,
    known only for prospectively captured episodes (``source == "capture"``);
    with both, the task's commit range is exact instead of a time window."""
    heads_start: tuple[tuple[str, str], ...] = ()
    heads_end: tuple[tuple[str, str], ...] = ()
    """(repository path, HEAD) for every repository at or under the working
    directory at the two ends, so a session started from a parent directory
    still gets exact ranges for the repositories inside it."""
    context: tuple[str, ...] = ()
    """Earlier requests of the same session, the user's own words, oldest
    first, for the attempt to read beside the request (R14.4.2)."""

    @property
    def episode_id(self) -> str:
        return f"{self.source}:{self.session_id[:12]}:{self.turn}"

    @property
    def is_command(self) -> bool:
        """A harness-injected turn (a slash command, a system block), not a
        request a person typed."""
        return self.request.lstrip().startswith("<")


@dataclass(frozen=True)
class ScanReport:
    episodes: tuple[Episode, ...]
    files_read: int
    files_skipped: tuple[str, ...]


def _iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _fmt(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _records(path: Path) -> Iterator[dict[str, object]]:
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


def scan_codex(root: Path) -> ScanReport:
    """``~/.codex/sessions/**/*.jsonl``: one session per file."""
    episodes: list[Episode] = []
    skipped: list[str] = []
    files = sorted(root.rglob("*.jsonl")) if root.is_dir() else []
    for path in files:
        try:
            episodes.extend(_codex_file(path))
        except (OSError, ValueError, KeyError, TypeError):
            skipped.append(str(path))
    return ScanReport(tuple(episodes), len(files) - len(skipped), tuple(skipped))


CONTEXT_TURNS = 3
CONTEXT_CHARS = 4000


def _context(prior: list[str]) -> tuple[str, ...]:
    """The last few earlier requests, newest last, within a character cap."""
    out: list[str] = []
    total = 0
    for text in reversed(prior[-CONTEXT_TURNS:]):
        if not text.strip() or text.lstrip().startswith("<"):
            continue
        if total + len(text) > CONTEXT_CHARS:
            break
        out.append(text)
        total += len(text)
    return tuple(reversed(out))


def _codex_file(path: Path) -> list[Episode]:
    out: list[Episode] = []
    cwd = session_id = None
    started: str | None = None
    request = ""
    tools: list[str] = []
    prior: list[str] = []
    turn = 0
    for rec in _records(path):
        kind = rec.get("type")
        payload = rec.get("payload")
        if not isinstance(payload, dict):
            continue
        if kind == "session_meta":
            cwd = str(payload.get("cwd") or "")
            session_id = str(payload.get("id") or payload.get("session_id") or path.stem)
        elif kind == "event_msg":
            ev = payload.get("type")
            if ev == "task_started":
                started = str(rec.get("timestamp") or "")
                request, tools = "", []
            elif ev == "user_message":
                # The user's own words, the one message body the importer reads.
                msg = payload.get("message")
                if isinstance(msg, str):
                    request = msg
            elif ev == "task_complete" and started and cwd and session_id:
                turn += 1
                out.append(Episode(
                    source="codex", session_id=session_id, turn=turn, cwd=cwd,
                    started_at=_fmt(_iso(started)), ended_at=_fmt(_iso(str(rec["timestamp"]))),
                    request=request, request_sha256=_sha(request),
                    tool_names=tuple(sorted(set(tools))), trace_path=str(path),
                    context=_context(prior)))
                prior.append(request)
                started = None
        elif kind == "response_item" and payload.get("type") == "function_call":
            name = payload.get("name")
            if isinstance(name, str):
                tools.append(name)
    return out


def scan_claude_code(root: Path) -> ScanReport:
    """``~/.claude/projects/*/*.jsonl``: one session per file, a turn per
    user prompt; the turn ends at the next prompt or after an hour."""
    episodes: list[Episode] = []
    skipped: list[str] = []
    files = sorted(root.glob("*/*.jsonl")) if root.is_dir() else []
    for path in files:
        try:
            episodes.extend(_claude_file(path))
        except (OSError, ValueError, KeyError, TypeError):
            skipped.append(str(path))
    return ScanReport(tuple(episodes), len(files) - len(skipped), tuple(skipped))


def _claude_file(path: Path) -> list[Episode]:
    turns: list[tuple[str, str, datetime, str]] = []
    for rec in _records(path):
        if rec.get("type") != "user":
            continue
        msg = rec.get("message")
        content = msg.get("content") if isinstance(msg, dict) else None
        stamp = rec.get("timestamp")
        cwd = rec.get("cwd")
        if not (isinstance(content, str) and isinstance(stamp, str) and isinstance(cwd, str)):
            continue  # tool results and structured turns are not requests
        turns.append((str(rec.get("sessionId") or path.stem), cwd, _iso(stamp), content))
    out: list[Episode] = []
    for i, (sid, cwd, start, text) in enumerate(turns):
        end = turns[i + 1][2] if i + 1 < len(turns) else start + CLAUDE_TURN_CAP
        out.append(Episode(
            source="claude_code", session_id=sid, turn=i + 1, cwd=cwd,
            started_at=_fmt(start), ended_at=_fmt(end), request=text,
            request_sha256=_sha(text), trace_path=str(path),
            context=_context([t[3] for t in turns[:i]])))
    return out


CAPTURE_FILE = Path(".xyntetik") / "shadow" / "capture.jsonl"


def scan_capture(path: Path) -> ScanReport:
    """The prospective path: lines written by ``shadow capture`` from a
    prompt hook and a stop hook. A ``prompt`` line carries the request and
    the repository HEAD at that moment; the next ``stop`` line for the same
    session carries HEAD at the end. Only the user's request is recorded."""
    if not path.is_file():
        return ScanReport((), 0, ())
    open_turns: dict[str, dict[str, object]] = {}
    prior: dict[str, list[str]] = {}
    turns: dict[str, int] = {}
    episodes: list[Episode] = []
    try:
        for rec in _records(path):
            sid = str(rec.get("session_id") or "")
            event = rec.get("event")
            if not sid:
                continue
            if event == "prompt":
                # A prompt while one is open closes the earlier one at this time.
                if sid in open_turns:
                    episodes.append(_close(open_turns.pop(sid), rec, path, prior.get(sid, [])))
                    prior.setdefault(sid, []).append(str(episodes[-1].request))
                open_turns[sid] = rec
            elif event == "stop" and sid in open_turns:
                episodes.append(_close(open_turns.pop(sid), rec, path, prior.get(sid, [])))
                prior.setdefault(sid, []).append(str(episodes[-1].request))
    except (OSError, ValueError, KeyError, TypeError):
        return ScanReport(tuple(episodes), 0, (str(path),))
    for sid, rec in open_turns.items():
        started = _iso(str(rec["timestamp"]))
        episodes.append(_close(rec, {"timestamp": _fmt(started + CLAUDE_TURN_CAP), "head": ""},
                               path, prior.get(sid, [])))
    for i, e in enumerate(episodes):
        turns[e.session_id] = turns.get(e.session_id, 0) + 1
        episodes[i] = Episode(**{**e.__dict__, "turn": turns[e.session_id]})
    return ScanReport(tuple(episodes), 1, ())


def _heads(rec: dict[str, object]) -> tuple[tuple[str, str], ...]:
    raw = rec.get("heads")
    if not isinstance(raw, dict):
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in raw.items() if v))


def _close(start: dict[str, object], end: dict[str, object], path: Path,
           prior: list[str]) -> Episode:
    request = str(start.get("prompt") or "")
    return Episode(
        source="capture", session_id=str(start["session_id"]), turn=0,
        cwd=str(start.get("cwd") or ""), started_at=_fmt(_iso(str(start["timestamp"]))),
        ended_at=_fmt(_iso(str(end["timestamp"]))), request=request,
        request_sha256=_sha(request), trace_path=str(path),
        head_start=str(start.get("head") or ""), head_end=str(end.get("head") or ""),
        heads_start=_heads(start), heads_end=_heads(end), context=_context(prior))


def scan_all(home: Path | None = None) -> ScanReport:
    home = home or Path.home()
    a = scan_codex(home / ".codex" / "sessions")
    b = scan_claude_code(home / ".claude" / "projects")
    c = scan_capture(home / CAPTURE_FILE)
    return ScanReport(a.episodes + b.episodes + c.episodes,
                      a.files_read + b.files_read + c.files_read,
                      a.files_skipped + b.files_skipped + c.files_skipped)


def write_episodes(episodes: Iterable[Episode], path: Path) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for e in episodes:
            f.write(json.dumps(e.__dict__, sort_keys=True) + "\n")
            n += 1
    return n


def read_episodes(path: Path) -> list[Episode]:
    out: list[Episode] = []
    for rec in _records(path):
        rec["tool_names"] = tuple(rec.get("tool_names") or ())  # type: ignore[arg-type]
        rec["context"] = tuple(rec.get("context") or ())  # type: ignore[arg-type]
        for key in ("heads_start", "heads_end"):
            raw = rec.get(key)
            pairs = raw if isinstance(raw, list) else []
            rec[key] = tuple((str(a), str(b)) for a, b in pairs)
        out.append(Episode(**rec))  # type: ignore[arg-type]
    return out
