"""One explicit opt-in that wires shadow mode into the harnesses.

Nothing here runs implicitly: ``shadow install`` is a command a person
types, it says what it wrote, and ``shadow uninstall`` removes exactly
that. Claude Code gets two hooks (prompt and stop) merged into the user
settings beside whatever is already there, and a ``/shadow`` skill that
shows the ledger. Codex gets a ``/shadow`` prompt; it has no prompt-time
hook, so its capture stays on the session files ``import`` already reads.
The hook commands never block a prompt: they redirect their errors away
and end in ``|| true``.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

MARK = "xyntetik_runner.shadow capture"
SKILL_NAME = "shadow"


@dataclass(frozen=True)
class Installed:
    settings: Path | None
    claude_skill: Path | None
    codex_prompt: Path | None
    hooks_added: int


def hook_command(event: str, python: str, pythonpath: str | None) -> str:
    prefix = f"PYTHONPATH={pythonpath} " if pythonpath else ""
    return (f"{prefix}{python} -m xyntetik_runner.shadow capture --event {event} "
            f"2>/dev/null || true")


def _hook_entry(command: str) -> dict[str, object]:
    return {"hooks": [{"type": "command", "timeout": 10, "command": command}]}


def install_claude_hooks(settings: Path, *, python: str, pythonpath: str | None) -> int:
    """Merge the two hooks into ``settings`` (created if absent); idempotent.
    Returns how many hooks were added. A backup sits beside the file."""
    data: dict[str, object] = {}
    if settings.is_file():
        data = json.loads(settings.read_text(encoding="utf-8") or "{}")
        shutil.copyfile(settings, settings.with_suffix(".json.bak-shadow"))
    hooks = data.setdefault("hooks", {})
    assert isinstance(hooks, dict)
    added = 0
    for event, name in (("prompt", "UserPromptSubmit"), ("stop", "Stop")):
        entries = hooks.setdefault(name, [])
        assert isinstance(entries, list)
        present = any(MARK in json.dumps(e) for e in entries)
        if not present:
            entries.append(_hook_entry(hook_command(event, python, pythonpath)))
            added += 1
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return added


def uninstall_claude_hooks(settings: Path) -> int:
    if not settings.is_file():
        return 0
    data = json.loads(settings.read_text(encoding="utf-8") or "{}")
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        return 0
    removed = 0
    for name in ("UserPromptSubmit", "Stop"):
        entries = hooks.get(name)
        if not isinstance(entries, list):
            continue
        kept = [e for e in entries if MARK not in json.dumps(e)]
        removed += len(entries) - len(kept)
        if kept:
            hooks[name] = kept
        else:
            hooks.pop(name, None)
    if not hooks:
        data.pop("hooks", None)
    settings.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return removed


def skill_text(python: str, pythonpath: str | None, out: str) -> str:
    prefix = f"PYTHONPATH={pythonpath} " if pythonpath else ""
    return f"""---
name: {SKILL_NAME}
description: Shadow-mode status from inside the session - the ledger's counts and both denominators, per admitted task with every attempt, and the capture file's size. Read-only; never starts a replay. Use when the user asks /shadow, "shadow status", "what did my local model manage", or "how many episodes are captured".
---

# /shadow: shadow-mode status without leaving the session

Run this and show the output verbatim in a fenced block, then one or two
sentences of reading. Counts before rates; never invent a percentage the
report withholds.

```
{prefix}{python} -m xyntetik_runner.shadow report --out {out} --tasks
```

Then the capture that accumulates from the prompt and stop hooks (only the
user's own requests, directories, times and commit ids):

```
{prefix}{python} -m xyntetik_runner.shadow capture --summary
```

Never run `replay`, `optimize`, `bank` or `import` from this command; they
are long-running and belong to a deliberate session. If the user wants one,
name the command and its cost and let them start it.
"""


def codex_prompt_text(python: str, pythonpath: str | None, out: str) -> str:
    prefix = f"PYTHONPATH={pythonpath} " if pythonpath else ""
    return f"""Shadow-mode status, read-only, without leaving this session.

Run and show verbatim:
{prefix}{python} -m xyntetik_runner.shadow report --out {out} --tasks
{prefix}{python} -m xyntetik_runner.shadow capture --summary

Counts before rates; never invent a percentage the report withholds. Do not
run replay, optimize, bank or import from here; they are long-running and
belong to a deliberate session.
"""


def install(home: Path, *, python: str = sys.executable, pythonpath: str | None = None,
            out: str = "~/.xyntetik/shadow", claude: bool = True, codex: bool = True) -> Installed:
    settings = skill = prompt = None
    added = 0
    if claude:
        settings = home / ".claude" / "settings.json"
        added = install_claude_hooks(settings, python=python, pythonpath=pythonpath)
        skill = home / ".claude" / "skills" / SKILL_NAME / "SKILL.md"
        skill.parent.mkdir(parents=True, exist_ok=True)
        skill.write_text(skill_text(python, pythonpath, out), encoding="utf-8")
    if codex:
        prompt = home / ".codex" / "prompts" / f"{SKILL_NAME}.md"
        prompt.parent.mkdir(parents=True, exist_ok=True)
        prompt.write_text(codex_prompt_text(python, pythonpath, out), encoding="utf-8")
    return Installed(settings=settings, claude_skill=skill, codex_prompt=prompt, hooks_added=added)


def uninstall(home: Path) -> Installed:
    settings = home / ".claude" / "settings.json"
    removed = uninstall_claude_hooks(settings)
    skill = home / ".claude" / "skills" / SKILL_NAME / "SKILL.md"
    prompt = home / ".codex" / "prompts" / f"{SKILL_NAME}.md"
    for p in (skill, prompt):
        if p.is_file():
            p.unlink()
    if skill.parent.is_dir() and not any(skill.parent.iterdir()):
        skill.parent.rmdir()
    return Installed(settings=settings if settings.is_file() else None, claude_skill=None,
                     codex_prompt=None, hooks_added=-removed)
