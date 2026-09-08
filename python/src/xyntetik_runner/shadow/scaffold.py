"""The scaffold artifact (R15.0): what sits between the model and the harness.

A scaffold is a versioned document the attempt harness consumes: the
system text, the procedure the model is told to follow, the tool
descriptions, the budget it may override, and exemplars. The model stays
untouched, so one scaffold runs on any family the runner serves, and its
sha256 sits in the evidence identity beside the model hash, so a report
row names "this model with this scaffold". It is text: a diff is readable,
promotion is a file copy, rollback is the previous file.

Exemplars are content. The optimizer never writes them (v1); a person may,
from verified local attempts or their own words, never from frontier
output. ``Scaffold.base()`` reproduces the harness's original prompt so
the base scaffold is the control arm of every comparison.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCAFFOLD_SCHEMA = "xyntetik.shadow.scaffold.v1"

BASE_SYSTEM = (
    "You are a software engineer fixing a repository checked out at the workspace root. "
    "Work only through the tools. Read before you edit, change only what the task needs, "
    "write complete files, run the tests to check your work, and call finish when the "
    "tests pass or you cannot make further progress. Never ask the user questions."
)


@dataclass(frozen=True)
class Scaffold:
    name: str
    system: str
    procedure: tuple[str, ...] = ()
    tool_descriptions: Mapping[str, str] = field(default_factory=dict)
    exemplars: tuple[str, ...] = ()
    budget: Mapping[str, float] = field(default_factory=dict)
    notes: str = ""
    parent_sha256: str = ""
    schema_version: str = SCAFFOLD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SCAFFOLD_SCHEMA:
            raise ValueError(f"unknown scaffold schema {self.schema_version!r}")
        if not self.system.strip():
            raise ValueError("a scaffold needs system text")

    @classmethod
    def base(cls) -> Scaffold:
        return cls(name="base", system=BASE_SYSTEM)

    def canonical(self) -> str:
        data = asdict(self)
        data["tool_descriptions"] = dict(sorted(self.tool_descriptions.items()))
        data["budget"] = dict(sorted(self.budget.items()))
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    def system_text(self) -> str:
        """The system message the harness sends: system text, then the
        numbered procedure, then exemplars, in that order."""
        parts = [self.system.strip()]
        if self.procedure:
            steps = "\n".join(f"{i + 1}. {p.strip()}" for i, p in enumerate(self.procedure))
            parts.append(f"Procedure:\n{steps}")
        if self.exemplars:
            parts.append("Examples of good work:\n" + "\n\n".join(e.strip() for e in self.exemplars))
        return "\n\n".join(parts)

    def apply_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The tool list with this scaffold's descriptions where it has one."""
        out: list[dict[str, Any]] = []
        for tool in tools:
            fn = dict(tool.get("function", {}))
            name = str(fn.get("name", ""))
            if name in self.tool_descriptions:
                fn["description"] = self.tool_descriptions[name]
            out.append({**tool, "function": fn})
        return out

    def to_json(self) -> str:
        data = asdict(self)
        data["tool_descriptions"] = dict(self.tool_descriptions)
        data["budget"] = dict(self.budget)
        return json.dumps(data, sort_keys=True, indent=2) + "\n"

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return self.sha256

    @classmethod
    def from_json(cls, text: str) -> Scaffold:
        data: dict[str, Any] = json.loads(text)
        return cls(
            name=str(data["name"]), system=str(data["system"]),
            procedure=tuple(str(p) for p in data.get("procedure") or ()),
            tool_descriptions={str(k): str(v) for k, v in (data.get("tool_descriptions") or {}).items()},
            exemplars=tuple(str(e) for e in data.get("exemplars") or ()),
            budget={str(k): float(v) for k, v in (data.get("budget") or {}).items()},
            notes=str(data.get("notes") or ""), parent_sha256=str(data.get("parent_sha256") or ""),
            schema_version=str(data.get("schema_version") or SCAFFOLD_SCHEMA))

    @classmethod
    def load(cls, path: Path) -> Scaffold:
        return cls.from_json(path.read_text(encoding="utf-8"))
