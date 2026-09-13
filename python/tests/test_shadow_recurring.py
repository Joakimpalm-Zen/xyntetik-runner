"""The recurring-work inventory: grouped by effect, never by mechanism."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xyntetik_runner.shadow import recurring


def session(tmp_path: Path, records: list[dict[str, Any]], name: str = "s.jsonl",
            project: str = "proj") -> Path:
    d = tmp_path / project
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return p


def ask(text: str, at: str = "2026-09-01T10:00:00Z") -> dict[str, Any]:
    return {"type": "user", "message": {"content": text}, "timestamp": at}


def did(*calls: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    return {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": n, "input": i} for n, i in calls]}}


def test_the_same_work_done_two_ways_is_one_kind() -> None:
    """A file edited with the edit tool and a file written by a shell heredoc
    are the same work; a grouping that separates them describes the
    assistant's habits rather than the user's work."""
    with_tool = recurring.effects_of([("Edit", {"file_path": "/r/src/engine.c"})])
    with_shell = recurring.effects_of([("Bash", {"command": "cat > /r/src/engine.c <<'EOF'\nx\nEOF"})])
    assert with_tool.kind() == with_shell.kind() == "code_change"
    docs_tool = recurring.effects_of([("Write", {"file_path": "/r/docs/plan.md"})])
    docs_shell = recurring.effects_of([("Bash", {"command": "sed -i s/a/b/ /r/docs/plan.md"})])
    assert docs_tool.kind() == docs_shell.kind() == "doc_change"


def test_kinds_follow_what_was_produced() -> None:
    assert recurring.effects_of([("Bash", {"command": "pytest -q tests/"})]).kind() == "execution"
    assert recurring.effects_of([("Bash", {"command": "ssh box 'uptime'"})]).kind() == "execution"
    assert recurring.effects_of([("WebSearch", {"query": "x"})]).kind() == "web_read"
    assert recurring.effects_of([("Bash", {"command": "grep -rn thing src/"})]).kind() == "inspection"
    # source wins over documents when a turn touched both: it is the consequential half
    both = recurring.effects_of([("Edit", {"file_path": "a.md"}), ("Edit", {"file_path": "b.py"})])
    assert both.kind() == "code_change" and both.docs == 1 and both.code == 1


def test_delegation_and_commits_are_attributes_not_kinds() -> None:
    e = recurring.effects_of([("Agent", {"prompt": "go"}), ("Edit", {"file_path": "x.py"}),
                              ("Bash", {"command": "git commit -m x && git push"})])
    assert e.kind() == "code_change", "how the work was carried out is not what it produced"
    assert e.delegated == 1 and e.commit == 1
    # delegation alone, with nothing produced, still is not its own kind
    assert recurring.effects_of([("Agent", {"prompt": "go"})]).kind() == "inspection"


def test_harness_text_is_not_a_request() -> None:
    for bad in ("<task-notification>\n<task-id>x</task-id>", "/exit", "[Image: 100x100]",
                "Caveat: the messages below were generated", "<local-command-stdout>ok", "   "):
        assert not recurring.is_request(bad), bad
    for good in ("fix the parser in engine.c", "why did the CUDA build fail?"):
        assert recurring.is_request(good)
    assert not recurring.is_request(None) and not recurring.is_request(12)


def test_an_episode_is_a_request_and_what_followed_until_the_next_one(tmp_path: Path) -> None:
    p = session(tmp_path, [
        ask("change the tokenizer so byte fallback stops truncating", "2026-09-01T10:00:00Z"),
        did(("Read", {"file_path": "t.c"}), ("Edit", {"file_path": "src/tokenizer.c"})),
        did(("Bash", {"command": "git commit -m fix"})),
        ask("now explain what that changed for the reader", "2026-09-01T11:00:00Z"),
        did(("Grep", {"pattern": "x"})),
        {"type": "user", "isMeta": True, "message": {"content": "Stop hook feedback: keep going"}},
        ask("/exit"),
    ])
    obs = recurring.observe_session(p)
    assert [o.kind for o in obs] == ["code_change", "inspection"]
    assert obs[0].commit is True and obs[1].commit is False
    assert obs[0].at == "2026-09-01T10:00:00Z" and obs[0].project == "proj"
    assert obs[0].request.startswith("change the tokenizer")


def test_requests_with_no_actions_after_them_are_not_episodes(tmp_path: Path) -> None:
    p = session(tmp_path, [ask("thanks, that is all"), ask("and one more thing, rebuild it"),
                           did(("Bash", {"command": "make runner"}))])
    obs = recurring.observe_session(p)
    assert len(obs) == 1 and obs[0].kind == "execution"


def test_continuations_are_dropped_by_the_length_floor(tmp_path: Path) -> None:
    session(tmp_path, [ask("go ahead"), did(("Edit", {"file_path": "a.py"})),
                       ask("now change the loader so it grows its buffer instead of truncating"),
                       did(("Edit", {"file_path": "b.py"}))])
    kept = recurring.observe([tmp_path], min_chars=60)
    assert len(kept) == 1, "a continuation inherits its kind from the conversation"
    assert len(recurring.observe([tmp_path], min_chars=0)) == 2


def test_render_counts_before_rates_and_says_it_is_retrospective(tmp_path: Path) -> None:
    few = [recurring.Observation("r" * 70, "code_change", False, True, "", "s", "p")] * 5
    text = recurring.render(few)
    assert "shares withheld below 30" in text and "| 5 |" in text and "%" not in text.split("\n")[0]
    many = [recurring.Observation("r" * 70, "code_change", False, True, "", "s", "p")] * 40
    text = recurring.render(many, sessions=3)
    assert "across 3 sessions" in text and "100%" in text
    assert "not a prediction" in text, "the inventory must not be read as a router"
    assert "no recurring work yet" in recurring.render([])


def test_command_reports_the_project_and_survives_a_missing_directory(tmp_path: Path, capsys: Any) -> None:
    from xyntetik_runner.shadow.cli import main
    session(tmp_path, [ask("rewrite the verifier so a shim in the workspace cannot run it"),
                       did(("Edit", {"file_path": "verifier.py"}))], project="mine")
    rc = main(["recurring", "--sessions", f"{tmp_path},{tmp_path / 'nope'}", "--project", "mine"])
    out = capsys.readouterr()
    assert rc == 0 and "changed source" in out.out and "1 episodes" in out.out
    assert "nothing was read from" in out.err
    rc = main(["recurring", "--sessions", str(tmp_path), "--project", "other"])
    assert rc == 0 and "no recurring work yet" in capsys.readouterr().out
