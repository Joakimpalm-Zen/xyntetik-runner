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


CONFIG_REL = Path(".xyntetik") / "shadow" / "config.json"


@dataclass(frozen=True)
class Installed:
    settings: Path | None
    claude_skill: Path | None
    codex_prompt: Path | None
    hooks_added: int
    config: Path | None = None


def write_config(home: Path, *, model: str, runner: str, ctx: int, gpu: str,
                 threads: int, out: str) -> Path:
    """What `delegate` needs to start the runner by itself: the model the
    user chose at `runner --shadow-mode -m`, the runner executable, and
    the serving knobs. Plain JSON the user can read and edit."""
    path = home / CONFIG_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"model": model, "runner": runner, "ctx": ctx, "gpu": gpu, "threads": threads,
            "out": out}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def read_config(home: Path) -> dict[str, object]:
    path = home / CONFIG_REL
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def harness_present(home: Path) -> tuple[bool, bool]:
    """(Claude Code, Codex) as judged by their home directories."""
    return (home / ".claude").is_dir(), (home / ".codex").is_dir()


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
description: Shadow mode from inside the session - status (the ledger's counts and both denominators, every attempt per task, the capture size), and offloading a task to the local model where the evidence shows verified successes. Use when the user asks /shadow, "shadow status", "what did my local model manage", "offload this to the local model", or "can the local model do this".
---

# /shadow: shadow mode without leaving the session

## Status (default)

Run this and show the output verbatim in a fenced block, then one or two
sentences of reading. Counts before rates; never invent a percentage the
report withholds.

```
{prefix}{python} -m xyntetik_runner.shadow report --out {out} --tasks
{prefix}{python} -m xyntetik_runner.shadow capture --summary
```

## Offload a task to the local model (when the user asks for it)

The harness routes; the runner proves. Offloading is explicit, evidence-gated
and never applied silently.

1. Read where the local model has verified successes:
   ```
   {prefix}{python} -m xyntetik_runner.shadow routes --out {out}
   ```
   It prints, per task class, attempted and verified counts and whether the
   class qualifies. If no class qualifies, say so plainly and offer the bench
   (`shadow bench --repo . --models ...`), which is long-running and the
   user's to start.
2. If the task's class qualifies, delegate it on a scratch worktree of the
   current repository (the working tree is never touched):
   ```
   {prefix}{python} -m xyntetik_runner.shadow delegate --repo . --request "<the user's request, verbatim>"
   ```
   It starts the runner with the configured model if none is running,
   runs the bounded attempt, runs the repository's tests on the scratch
   copy, and prints the verdict, the diff and a patch path.
3. Show the diff and the test verdict. If the tests passed, offer
   `git apply <patch>`; the user applies it, you do not. If they failed,
   say so and continue with the frontier model as usual.

Never run `replay`, `optimize`, `bank` or `import` from this command; they
are long-running and belong to a deliberate session.
"""


def codex_prompt_text(python: str, pythonpath: str | None, out: str) -> str:
    prefix = f"PYTHONPATH={pythonpath} " if pythonpath else ""
    return f"""Shadow mode without leaving this session.

Status (default): run and show verbatim, counts before rates, never a
percentage the report withholds:
{prefix}{python} -m xyntetik_runner.shadow report --out {out} --tasks
{prefix}{python} -m xyntetik_runner.shadow capture --summary

Offload a task to the local model (only when the user asks): first
{prefix}{python} -m xyntetik_runner.shadow routes --out {out}
which says per task class whether the local model has verified successes.
If the task's class qualifies:
{prefix}{python} -m xyntetik_runner.shadow delegate --repo . --request "<the request verbatim>"
runs a bounded attempt on a scratch worktree (the working tree is never
touched), runs the repository's tests there, and prints the verdict, the
diff and a patch path. Show both; if the tests passed offer `git apply
<patch>` and let the user apply it; if they failed, say so and continue
with the frontier model. If no class qualifies, say so and offer the bench.

Do not run replay, optimize, bank or import from here; they are long-running
and belong to a deliberate session.
"""


def install(home: Path, *, python: str = sys.executable, pythonpath: str | None = None,
            out: str = "~/.xyntetik/shadow", claude: bool = True, codex: bool = True,
            model: str = "", runner: str = "runner", ctx: int = 8192, gpu: str = "auto",
            threads: int = 0) -> Installed:
    settings = skill = prompt = config = None
    added = 0
    if model:
        config = write_config(home, model=model, runner=runner, ctx=ctx, gpu=gpu,
                              threads=threads, out=out)
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
    return Installed(settings=settings, claude_skill=skill, codex_prompt=prompt, hooks_added=added,
                     config=config)


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
