"""Capture v2: the evidence a learning method needs, or an explicit account of
what was missed.

The test that matters most is the last one: a record plus the blob store must
rebuild the pre-state byte for byte. Everything else in this file exists to keep
that property true.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from xyntetik_runner.shadow import capture as C


def _repo(tmp: Path) -> Path:
    r = tmp / "repo"
    (r / "pkg").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    subprocess.run(["git", "-C", str(r), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(r), "config", "user.name", "t"], check=True)
    (r / "pkg" / "a.py").write_text("def f():\n    return 1\n", newline="\n")
    (r / "keep.txt").write_text("x\n", newline="\n")
    subprocess.run(["git", "-C", str(r), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(r), "commit", "-qm", "init"], check=True)
    return r


# ----------------------------------------------------------------- provenance

def test_every_event_is_labelled_and_none_is_dropped() -> None:
    """Filtering harness injections does not recover their originating tasks, so
    nothing is filtered. Each event keeps its raw text and gains a label."""
    cases = [
        ("<task-notification>\n<task-id>b1</task-id>\n", C.PROV_NOTIFICATION),
        ("<system-reminder>\nbe careful\n</system-reminder>", C.PROV_REMINDER),
        ("This session is being continued from a previous conversation.", C.PROV_CONTINUATION),
        ("<command-name>/compact</command-name>", C.PROV_REMINDER),
        ("make parse_amount accept separators", C.PROV_USER),
        ("", C.PROV_UNKNOWN),
    ]
    for text, want in cases:
        assert C.classify(text).kind == want, text[:40]


def test_a_request_with_a_reminder_stapled_to_its_end_is_still_a_request() -> None:
    """The governing label is the envelope at the START, because that is where
    the harness puts its own framing. A request does not stop being one because
    a reminder was appended to it."""
    t = "fix the validation guard\n\n<system-reminder>\nnote\n</system-reminder>"
    assert C.classify(t).kind == C.PROV_USER


def test_a_parent_in_the_payload_beats_any_text_heuristic() -> None:
    p = C.classify("looks like a plain request", payload={"parent_session_id": "parent-1"})
    assert p.kind == C.PROV_WORKER
    p2 = C.classify("x", payload={"subagent_type": "Explore"})
    assert p2.kind == C.PROV_WORKER


def test_notification_task_ids_are_recorded_beside_the_link_not_as_it() -> None:
    """A <task-id> names the background JOB, not the request that caused it. It
    is evidence, and using it as the task link would attribute a change set to a
    notification instead of to the work the user asked for."""
    ids = C.notification_task_ids("<task-id>bqou7ne9n</task-id> x <task-id>b2</task-id>")
    assert ids == ["bqou7ne9n", "b2"]


# --------------------------------------------------------------------- links

def test_notifications_and_continuations_link_to_the_real_request(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    req = C.link(home, session_id="s1", event_id="e1",
                 prov=C.classify("implement the guard"))
    assert req.kind == C.PROV_USER and req.task_id == "e1" and req.originating_event_id == "e1"

    note = C.link(home, session_id="s1", event_id="e2",
                  prov=C.classify("<task-notification>\n<task-id>b9</task-id>\n"))
    assert note.kind == C.PROV_NOTIFICATION
    assert note.task_id == "e1", "a notification must point at the request that caused the work"
    assert note.originating_event_id == "e1"

    cont = C.link(home, session_id="s1", event_id="e3",
                  prov=C.classify("This session is being continued from a previous conversation."))
    assert cont.task_id == "e1"

    # a second real request opens a new thread
    req2 = C.link(home, session_id="s1", event_id="e4", prov=C.classify("now do the other thing"))
    assert req2.task_id == "e4"
    note2 = C.link(home, session_id="s1", event_id="e5",
                   prov=C.classify("<task-notification>x"))
    assert note2.task_id == "e4"


def test_a_worker_opens_its_own_thread_and_keeps_the_parent_explicit(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    C.link(home, session_id="parent", event_id="p1", prov=C.classify("parent asks"))
    w = C.link(home, session_id="worker", event_id="w1",
               prov=C.classify("go and search", payload={"parent_session_id": "parent"}))
    assert w.kind == C.PROV_WORKER
    assert w.task_id == "w1", "a worker's own thread, not flattened into the parent's"
    ident = C.identity({"parent_session_id": "parent", "subagent_type": "Explore"},
                       session_id="worker")
    assert ident["parent_session_id"] == "parent" and ident["worker"] is True
    assert ident["agent"] == "Explore"


def test_identity_never_invents_a_parent() -> None:
    """A guessed parent is worse than a null one: it would silently reattribute a
    worker's change set to a session that did not ask for it."""
    ident = C.identity({}, session_id="s")
    assert ident["parent_session_id"] is None and ident["worker"] is False


# ------------------------------------------------------------------ snapshot

def test_the_snapshot_is_content_not_a_commit(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    home = tmp_path / "home"
    (r / "pkg" / "a.py").write_text("def f():\n    return 2\n", newline="\n")   # dirty, uncommitted
    (r / "new.py").write_text("fresh\n", newline="\n")
    s = C.snapshot(r, home)
    d = s.to_dict()
    assert d["clean"] is False
    assert set(d["files"]) == {"pkg/a.py", "new.py"}
    assert d["files"]["pkg/a.py"]["status"] == "M"
    assert d["files"]["new.py"]["status"] == "??"
    # the head is kept and labelled for what it is
    assert d["head"] and d["head_role"] == "storage_reference"
    # and the CONTENT is recoverable, which a head cannot do for dirty work
    assert C.get_blob(home, d["files"]["pkg/a.py"]["sha256"]) == b"def f():\n    return 2\n"


def test_snapshot_keeps_a_non_ascii_path_and_its_content(tmp_path: Path) -> None:
    """git status quotes any path outside ASCII as C escapes ("caf\\303\\251.txt"),
    and a parser that only strips the quotes looks the escaped spelling up
    on disk, finds nothing, and records the file as deleted: its content,
    the whole point of the snapshot, is lost. The NUL-separated form has no
    quoting, and a round trip through rebuild proves the content survives."""
    from xyntetik_runner.shadow import capture_report as R
    r = _repo(tmp_path)
    home = tmp_path / "home"
    name = "caf\u00e9 na\u00efve.txt"
    (r / name).write_text("SENTINEL\n", encoding="utf-8", newline="\n")
    s = C.snapshot(r, home)
    d = s.to_dict()
    assert name in d["files"], d["files"]
    entry = d["files"][name]
    assert not entry.get("deleted") and entry.get("blob") is True
    assert C.get_blob(home, entry["sha256"]) == b"SENTINEL\n"
    rb = R.rebuild(home, C.snapshot_all(r, home), tmp_path / "rebuilt")
    try:
        assert rb["unrestorable"] == []
        tree = Path(next(iter(rb["repos"].values()))["worktree"])
        assert (tree / name).read_text(encoding="utf-8") == "SENTINEL\n"
    finally:
        R.rebuild_cleanup(rb)


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need a privilege on Windows")
def test_snapshot_records_a_symlink_without_following_it(tmp_path: Path) -> None:
    """A link is a path, and git stores it as one. Reading through it would
    copy whatever it points at, inside the repository or not, into the blob
    store: an untracked link to a key file is not workspace content. The
    target is recorded, the link is rebuilt as a link, no blob is stored."""
    from xyntetik_runner.shadow import capture_report as R
    r = _repo(tmp_path)
    home = tmp_path / "home"
    outside = tmp_path / "outside.secret"
    outside.write_bytes(b"NOT WORKSPACE CONTENT\n")
    (r / "link.txt").symlink_to(outside)
    s = C.snapshot(r, home)
    entry = s.to_dict()["files"]["link.txt"]
    assert entry.get("symlink") == str(outside) and entry.get("blob") is False
    import hashlib
    assert C.get_blob(home, hashlib.sha256(b"NOT WORKSPACE CONTENT\n").hexdigest()) is None
    rb = R.rebuild(home, C.snapshot_all(r, home), tmp_path / "rebuilt")
    try:
        assert rb["unrestorable"] == []
        tree = Path(next(iter(rb["repos"].values()))["worktree"])
        assert (tree / "link.txt").is_symlink()
        assert os.readlink(tree / "link.txt") == str(outside)
    finally:
        R.rebuild_cleanup(rb)


def test_snapshot_uses_native_paths_when_git_reports_a_foreign_absolute_root(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MSYS Git can report /drive/... to a native Windows Python process."""
    r = _repo(tmp_path)
    wanted = b"def f():\n    return 2\n"
    (r / "pkg" / "a.py").write_bytes(wanted)
    run = subprocess.run

    def foreign_root(argv: list[str], **kwargs: Any) -> Any:
        result = run(argv, **kwargs)
        if "--show-toplevel" in argv and result.returncode == 0:
            result.stdout = "/unmapped-git-root/repo\n"
        return result

    monkeypatch.setattr(subprocess, "run", foreign_root)
    home = tmp_path / "home"
    snap = C.snapshot(r / "pkg", home)
    assert snap.repo is not None and Path(snap.repo).samefile(r)
    assert C.get_blob(home, snap.files["pkg/a.py"]["sha256"]) == wanted


def test_gitignored_files_stay_out(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    home = tmp_path / "home"
    (r / ".gitignore").write_text("junk.log\n", newline="\n")
    (r / "junk.log").write_text("noise\n", newline="\n")
    s = C.snapshot(r, home)
    assert "junk.log" not in s.files
    assert ".gitignore" in s.files


def test_an_oversized_file_is_hashed_not_stored_and_says_so(tmp_path: Path) -> None:
    """A bound that bites silently turns 'the workspace was small' and 'we
    stopped looking' into the same record."""
    r = _repo(tmp_path)
    home = tmp_path / "home"
    (r / "big.txt").write_bytes(b"a" * (C.MAX_FILE_BYTES + 10))
    s = C.snapshot(r, home)
    e = s.files["big.txt"]
    assert e["sha256"] and e["blob"] is False
    assert any("over" in k["why"] and k["path"] == "big.txt" for k in s.skipped)
    assert C.get_blob(home, e["sha256"]) is None


def test_a_spent_deadline_marks_the_snapshot_truncated(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    home = tmp_path / "home"
    for i in range(8):
        (r / f"f{i}.txt").write_text(f"{i}\n", newline="\n")
    s = C.snapshot(r, home, deadline_s=0.0)
    assert s.truncated is True
    assert any("deadline" in k["why"] for k in s.skipped)


def test_a_binary_file_is_hashed_not_stored(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    home = tmp_path / "home"
    (r / "b.bin").write_bytes(b"\x00\x01\x02" * 100)
    s = C.snapshot(r, home)
    assert s.files["b.bin"]["binary"] is True and s.files["b.bin"]["blob"] is False


def test_a_deletion_is_recorded(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    home = tmp_path / "home"
    (r / "keep.txt").unlink()
    s = C.snapshot(r, home)
    assert s.files["keep.txt"]["deleted"] is True


# ---------------------------------------------------------------- change sets

def test_the_change_set_covers_dirty_to_dirty(tmp_path: Path) -> None:
    """A commit range cannot express this: both states are uncommitted."""
    r = _repo(tmp_path)
    home = tmp_path / "home"
    (r / "pkg" / "a.py").write_text("v2\n", newline="\n")
    (r / "gone.py").write_text("temp\n", newline="\n")
    before = C.snapshot(r, home)
    (r / "pkg" / "a.py").write_text("v3\n", newline="\n")
    (r / "added.py").write_text("new\n", newline="\n")
    (r / "gone.py").unlink()
    after = C.snapshot(r, home)
    cs = C.change_set(before, after)
    assert cs["modified"] == ["pkg/a.py"]
    assert cs["added"] == ["added.py"]
    assert cs["removed"] == ["gone.py"]
    assert cs["count"] == 3
    # and the head never moved, so a commit-based recovery would see nothing
    assert before.head == after.head


def test_nested_repositories_are_all_snapshotted(tmp_path: Path) -> None:
    """Sessions run from an umbrella directory with the real projects as
    untracked checkouts inside it. A single-repo snapshot records none of the
    work done in them."""
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    subprocess.run(["git", "-C", str(outer), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(outer), "config", "user.name", "t"], check=True)
    inner = _repo(outer)
    home = tmp_path / "home"
    (inner / "pkg" / "a.py").write_text("changed inside the nested repo\n", newline="\n")
    sa = C.snapshot_all(outer, home)
    assert sa["repo_count"] >= 2
    keys = {Path(k).name for k in sa["repos"]}
    assert "outer" in keys and "repo" in keys
    inner_snap = sa["repos"][str(C_canonical(inner))]
    assert "pkg/a.py" in inner_snap["files"]


def C_canonical(p: Path) -> Path:
    from xyntetik_runner.shadow.tasks import canonical
    return canonical(p)


def test_change_set_all_reports_per_repository(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    subprocess.run(["git", "-C", str(outer), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(outer), "config", "user.name", "t"], check=True)
    inner = _repo(outer)
    home = tmp_path / "home"
    before = C.snapshot_all(outer, home)
    (inner / "pkg" / "a.py").write_text("moved\n", newline="\n")
    after = C.snapshot_all(outer, home)
    cs = C.change_set_all(before, after)
    assert cs["count"] == 1
    assert len(cs["repos_touched"]) == 1
    assert cs["paths"] and cs["paths"][0].endswith("pkg/a.py")


# -------------------------------------------------------------- verification

def test_verification_is_bound_to_the_state_it_ran_against() -> None:
    """Without the binding, a recorded exit code says only that some tests passed
    at some point in the session, which is not evidence about any change set."""
    v = C.verification_record(command="pytest -q", exit_code=0, output_tail="2 passed",
                              manifest_sha256="abc123", cwd="/x")
    assert v["verdict"] == "passed" and v["bound_to_manifest_sha256"] == "abc123"
    assert C.verification_record(command="pytest", exit_code=1, output_tail="",
                                 manifest_sha256="d", cwd="/x")["verdict"] == "failed"
    # an instrument failure is not a verdict
    assert C.verification_record(command="pytest", exit_code=2, output_tail="",
                                 manifest_sha256="d", cwd="/x")["verdict"] is None


def test_only_real_test_runners_count_as_verification() -> None:
    for c in ("pytest -q", "make test", "python3 -m pytest x", "cargo test", "npm test", "tox"):
        assert C.looks_like_verification(c), c
    for c in ("ls", "git status", "echo pytest-ish", "make build", "./runner --serve"):
        assert not C.looks_like_verification(c), c


# --------------------------------------------------------------- completeness

def test_completeness_is_measured_per_record(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    home = tmp_path / "home"
    (r / "pkg" / "a.py").write_text("dirty\n", newline="\n")
    snap = C.snapshot_all(r, home)
    rec = C.build("prompt", payload={"prompt": "do the thing", "transcript_path": "/t.jsonl"},
                  cwd=r, home=home, tool="claude_code", session_id="s1", snap=snap)
    c = rec["completeness"]
    assert c["raw_payload"] and c["provenance_labelled"] and c["linked_to_task"]
    assert c["pre_or_post_state"] and c["state_contents_stored"] and c["state_complete"]
    assert c["model_interaction"] is True
    assert c["verification"] is False, "a prompt record has no verification yet"
    # the v1 shape would fail most of these, which is the point
    v1 = {"event": "prompt", "cwd": "/x", "head": "abc", "prompt": "p",
          "session_id": "s", "timestamp": "t", "tool": "claude_code"}
    c1 = C.completeness(v1)
    assert not any([c1["raw_payload"], c1["pre_or_post_state"], c1["verification"],
                    c1["linked_to_task"], c1["state_contents_stored"]])


def test_a_large_payload_is_preserved_in_the_blob_store_not_truncated_away(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    big = "x" * (C.MAX_RAW_BYTES + 5000)
    rec = C.build("prompt", payload={"prompt": big}, cwd=tmp_path, home=home, session_id="s")
    raw = rec["raw"]
    assert raw["_truncated"] is True and raw["_blob_sha256"]
    restored = json.loads(C.get_blob(home, raw["_blob_sha256"]).decode())
    assert restored["prompt"] == big, "the payload must be recoverable in full"


# ------------------------------------------------- the acceptance test

def test_a_record_plus_the_blob_store_rebuilds_the_pre_state(tmp_path: Path) -> None:
    """The property the whole module exists for.

    A commit head cannot do this: the work was never committed, and at the moment
    of capture the tree held content that exists nowhere else.
    """
    r = _repo(tmp_path)
    home = tmp_path / "home"
    wanted = {
        "pkg/a.py": "def f():\n    return 'the state at prompt time'\n",
        "pkg/b.py": "SENTINEL = 1\n",
    }
    for rel, body in wanted.items():
        (r / rel).write_text(body, newline="\n")
    snap = C.snapshot_all(r, home)
    rec = C.build("prompt", payload={"prompt": "fix f"}, cwd=r, home=home,
                  session_id="s1", snap=snap)
    C.append(home, rec, path=home / "cap.jsonl")

    # the work continues and overwrites the captured state entirely
    for rel in wanted:
        (r / rel).write_text("LATER, and the earlier content is gone from disk\n", newline="\n")

    # now rebuild from the record alone
    on_disk = json.loads((home / "cap.jsonl").read_text().splitlines()[0])
    repos = on_disk["state"]["workspace"]["repos"]
    rebuilt: dict[str, str] = {}
    for repo_path, snapd in repos.items():
        for rel, entry in (snapd.get("files") or {}).items():
            if entry.get("blob"):
                data = C.get_blob(home, entry["sha256"])
                assert data is not None, rel
                rebuilt[rel] = data.decode()
    for rel, body in wanted.items():
        assert rebuilt[rel] == body, rel
    # and the state the record claims is the state we rebuilt
    assert on_disk["state"]["workspace"]["combined_sha256"] == snap["combined_sha256"]


def test_a_stop_event_is_a_session_event_not_an_unknown(tmp_path: Path) -> None:
    """A stop or verify event legitimately has no request text. Labelling that
    `unknown` would put it in the same bucket as a prompt whose text we failed to
    read, and those are a normal shape and a capture defect respectively."""
    assert C.classify("", event="stop").kind == C.PROV_SESSION
    assert C.classify("", event="verify").kind == C.PROV_SESSION
    assert C.classify("", event="prompt").kind == C.PROV_UNKNOWN, \
        "an empty PROMPT is still a defect and must stay visible as one"


def test_a_session_event_inherits_the_thread_and_never_opens_one(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    C.link(home, session_id="s", event_id="r1", prov=C.classify("the real request"))
    stop = C.link(home, session_id="s", event_id="s1", prov=C.classify("", event="stop"))
    assert stop.task_id == "r1" and stop.originating_event_id == "r1"
    later = C.link(home, session_id="s", event_id="n1", prov=C.classify("<task-notification>x"))
    assert later.task_id == "r1", "a stop must not have closed or moved the thread"


# ------------------------------------------- the three questions, kept separate

def _fire(home: Path, repo: Path, event: str, payload: dict[str, Any]) -> dict[str, Any]:
    snap = C.snapshot_all(repo, home)
    rec = C.build(event, payload=payload, cwd=repo, home=home, tool="t",
                  session_id=str(payload.get("session_id") or ""), snap=snap,
                  verification=payload.get("_verification"))
    C.append(home, rec)
    return rec


def test_the_report_keeps_completeness_replay_and_eligibility_separate(tmp_path: Path) -> None:
    """They fail for different reasons and one number hides which. Completeness is
    the recorder's property, replay the record-plus-store's, eligibility the
    method's -- and eligibility can fail while the other two are perfect."""
    from xyntetik_runner.shadow import capture_report as R
    r = _repo(tmp_path)
    home = tmp_path / "home"
    (r / "pkg" / "a.py").write_text("def f():\n    return 1  # before\n", newline="\n")
    _fire(home, r, "prompt", {"prompt": "fix f", "session_id": "s1",
                              "transcript_path": "/t.jsonl"})
    (r / "pkg" / "a.py").write_text("def f():\n    return 2  # after\n", newline="\n")
    _fire(home, r, "verify", {"session_id": "s1", "_verification":
                              C.verification_record(command="pytest -q", exit_code=0,
                                                    output_tail="1 passed",
                                                    manifest_sha256="m", cwd=str(r))})
    _fire(home, r, "stop", {"session_id": "s1"})

    rep = R.report(home)
    # 1. completeness: a property of the recorder
    assert rep["completeness"]["records"] == 3
    assert rep["completeness"]["share"]["pre_or_post_state"] == 1.0
    assert rep["completeness"]["share"]["linked_to_task"] == 1.0
    # 2. replay: a property of the record plus the blob store
    assert rep["workspace_reconstruction"]["by_outcome"].get("reconstructed", 0) >= 1
    assert "eligibility" not in rep, "learning policy is owned by Shade"



def test_replay_calls_a_missing_blob_broken_not_partial(tmp_path: Path) -> None:
    """`partial` means the record said it did not store something. `broken` means
    the record claims content the store cannot produce. Collapsing them would
    hide an evicted blob behind a deliberate bound."""
    from xyntetik_runner.shadow import capture_report as R
    r = _repo(tmp_path)
    home = tmp_path / "home"
    (r / "pkg" / "a.py").write_text("content that will be lost\n", newline="\n")
    rec = _fire(home, r, "prompt", {"prompt": "x", "session_id": "s"})
    assert R.replay(home, rec)["outcome"] in ("reconstructed", "partial")
    # evict every blob
    for b in (home / C.BLOBS_DIR).rglob("*"):
        if b.is_file():
            b.unlink()
    out = R.replay(home, rec)
    assert out["outcome"] == "broken" and out["blobs_missing"] >= 1


def test_an_unlinked_record_is_reported_not_dropped(tmp_path: Path) -> None:
    from xyntetik_runner.shadow import capture_report as R
    r = _repo(tmp_path)
    home = tmp_path / "home"
    # a verify with no preceding prompt: no thread to inherit
    _fire(home, r, "verify", {"session_id": "orphan"})
    rep = R.report(home)
    assert rep["threads"]["unlinked_records"] == 1
    assert rep["completeness"]["share"]["linked_to_task"] == 0.0






# --------------------- reconstruction vs reproduced verification, and attribution

def test_a_job_first_seen_under_an_older_task_keeps_both_candidates(tmp_path: Path) -> None:
    """The case that makes inheritance a heuristic: a background job announced by a
    notification may have been STARTED under an earlier task while the session has
    moved on. Assigning it to the currently open thread would attribute a change
    set to work that did not cause it, so uncertainty is retained instead."""
    home = tmp_path / "home"
    home.mkdir()
    C.link(home, session_id="s", event_id="T1", prov=C.classify("first request"))
    n1 = C.link(home, session_id="s", event_id="E1",
                prov=C.classify("<task-notification><task-id>bJOB</task-id>"),
                job_ids=["bJOB"])
    assert n1.attribution == C.ATTR_HEURISTIC and n1.ownership_uncertain is False
    assert n1.task_id == "T1"

    C.link(home, session_id="s", event_id="T2", prov=C.classify("second request"))
    n2 = C.link(home, session_id="s", event_id="E2",
                prov=C.classify("<task-notification><task-id>bJOB</task-id>"),
                job_ids=["bJOB"])
    assert n2.ownership_uncertain is True
    assert n2.candidate_task_ids == ["T1", "T2"], "both candidates retained"
    assert "not established" in n2.attribution_basis


def test_a_user_request_attribution_is_established_not_a_heuristic(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    p = C.link(home, session_id="s", event_id="T1", prov=C.classify("do the thing"))
    assert p.attribution == C.ATTR_SELF and p.ownership_uncertain is False
    rec = {"provenance": p.to_dict()}
    assert C.completeness(rec)["attribution_established"] is True


def test_reconstruction_and_reproduction_are_reported_separately(tmp_path: Path) -> None:
    """They can disagree in both directions, so they are two functions and two
    columns. Here reconstruction succeeds and the verification genuinely re-runs."""
    from xyntetik_runner.shadow import capture_report as R
    r = _repo(tmp_path)
    home = tmp_path / "home"
    # a test that passes only with the dirty content in place
    (r / "check.sh").write_text("#!/bin/sh\ngrep -q SENTINEL pkg/a.py\n", newline="\n")
    (r / "pkg" / "a.py").write_text("SENTINEL = 1\n", newline="\n")
    _fire(home, r, "prompt", {"prompt": "add the sentinel", "session_id": "s1"})
    v = C.verification_record(command="sh check.sh", exit_code=0, output_tail="",
                              manifest_sha256=C.snapshot_all(r, home)["combined_sha256"], cwd=str(r))
    _fire(home, r, "verify", {"session_id": "s1", "_verification": v})

    rows = R.load(home / C.CAPTURE2_FILE)
    th = R.threads(rows)
    tid = next(k for k in th if k)
    # 1. reconstruction: a property of the record
    recon = [R.replay(home, x) for x in th[tid]]
    assert any(x["outcome"] == "reconstructed" for x in recon)
    # 2. reproduction: a property of the world, measured by re-running
    rep = R.reproduce_verification(home, th[tid])
    assert len(rep) == 1
    assert rep[0]["outcome"] == "reproduced", rep[0]
    assert rep[0]["replay_verdict"] == "passed"
    assert rep[0]["reconstruction"]["restored"] >= 1


def test_reproduction_requires_the_bound_state_and_uses_its_recorded_directory(tmp_path: Path) -> None:
    from xyntetik_runner.shadow import capture_report as R
    r = _repo(tmp_path)
    home = tmp_path / "home"
    (r / "pkg" / "check.sh").write_text("grep -q 'return 1' a.py\n", newline="\n")
    before = _fire(home, r, "prompt", {"prompt": "change the return value", "session_id": "s"})
    v = C.verification_record(command="sh check.sh", exit_code=0, output_tail="",
                              manifest_sha256=before["state"]["workspace"]["combined_sha256"],
                              cwd=str(r / "pkg"))
    (r / "pkg" / "a.py").write_text("def f():\n    return 2\n", newline="\n")
    after = _fire(home, r, "verify", {"session_id": "s", "_verification": v})
    result = R.reproduce_verification(home, [before, after])
    assert result[0]["outcome"] == "reproduced", result
    after["verification"]["bound_to_manifest_sha256"] = "not-captured"
    result = R.reproduce_verification(home, [before, after])
    assert result[0]["outcome"] == "not_attempted", result


def test_reproduction_can_fail_while_reconstruction_succeeds(tmp_path: Path) -> None:
    """The direction that matters: a perfectly reconstructable record whose
    recorded verdict does not come back. Calling that a replay success would be
    the exact conflation the owner asked to avoid."""
    from xyntetik_runner.shadow import capture_report as R
    r = _repo(tmp_path)
    home = tmp_path / "home"
    (r / "check.sh").write_text("#!/bin/sh\ngrep -q SENTINEL pkg/a.py\n", newline="\n")
    (r / "pkg" / "a.py").write_text("no sentinel here\n", newline="\n")
    _fire(home, r, "prompt", {"prompt": "x", "session_id": "s1"})
    # the record CLAIMS it passed, which the reconstruction will contradict
    v = C.verification_record(command="sh check.sh", exit_code=0, output_tail="",
                              manifest_sha256=C.snapshot_all(r, home)["combined_sha256"], cwd=str(r))
    _fire(home, r, "verify", {"session_id": "s1", "_verification": v})
    rows = R.load(home / C.CAPTURE2_FILE)
    th = R.threads(rows)
    tid = next(k for k in th if k)
    assert any(R.replay(home, x)["outcome"] == "reconstructed" for x in th[tid])
    rep = R.reproduce_verification(home, th[tid])
    assert rep[0]["outcome"] == "not_reproduced", rep[0]
    assert rep[0]["recorded_verdict"] == "passed" and rep[0]["replay_verdict"] == "failed"


def test_the_binding_names_which_state_and_flags_a_mismatch(tmp_path: Path) -> None:
    from xyntetik_runner.shadow import capture_report as R
    r = _repo(tmp_path)
    home = tmp_path / "home"
    (r / "pkg" / "a.py").write_text("one\n", newline="\n")
    pre = _fire(home, r, "prompt", {"prompt": "x", "session_id": "s1"})
    pre_m = pre["state"]["workspace"]["combined_sha256"]
    # the workspace MOVES after the prompt, so pre and post are different states
    # and a verification still bound to `pre` is a genuine before/after mismatch.
    # Without this edit pre == post and there is nothing to flag, which is the
    # correct behaviour and was the bug in the first version of this test.
    (r / "pkg" / "a.py").write_text("two\n", newline="\n")
    v = C.verification_record(command="pytest -q", exit_code=0, output_tail="",
                              manifest_sha256=pre_m, cwd=str(r))
    _fire(home, r, "verify", {"session_id": "s1", "_verification": v})
    rows = R.load(home / C.CAPTURE2_FILE)
    th = R.threads(rows)
    b = R.binding(th[next(k for k in th if k)])
    assert len(b) == 1
    assert b[0]["bound_to_which_state"] == "pre", \
        "a verification bound to the starting state means nothing had been edited"
    assert b[0]["before_after_mismatch"] is True


def test_costs_keep_partial_records_in_the_denominator(tmp_path: Path) -> None:
    """A capture that is fast because it gave up is not a fast capture."""
    from xyntetik_runner.shadow import capture_report as R
    r = _repo(tmp_path)
    home = tmp_path / "home"
    (r / "big.txt").write_bytes(b"a" * (C.MAX_FILE_BYTES + 1))
    _fire(home, r, "prompt", {"prompt": "x", "session_id": "s1"})
    rows = R.load(home / C.CAPTURE2_FILE)
    co = R.costs(rows, home)
    assert co["snapshot_latency_s"]["n"] == len(rows), "every record counted"
    assert "partial ones included" in co["denominator_note"]
    assert co["blob_store"]["objects"] >= 0
