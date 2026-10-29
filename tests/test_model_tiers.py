"""The model support roster, docs/model-tiers.json.

These test the validator's teeth and the committed roster's shape. The
absolute anchors are facts from outside the code under test: the Hugging Face
listing of the maintainer's repositories and the LFS sha256 of the published
gemma-4-E2B file (both read from the Hub API on 2026-10-23 and written here
by hand), and the owner's 2026-10-22 tier decisions.
"""

import copy
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "model-tiers.py"
ROSTER = ROOT / "docs" / "model-tiers.json"

spec = importlib.util.spec_from_file_location("model_tiers", SCRIPT)
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

SHA_A = "a" * 64
DEVICES = {
    "schema": "xyntetik.runner.device-evidence.v1",
    "policy": {"default_max_age_days": 45},
    "classes": [
        {"id": "macos-arm64-metal",
         "expect": {"os": "macos", "arch": "arm64", "gpu_backend": "metal"}},
        {"id": "linux-x86_64-cuda",
         "expect": {"os": "linux", "arch": "x86_64", "gpu_backend": "cuda"}},
    ],
}


def tier1_row(**over):
    row = {
        "id": "toy-q8_0",
        "model": "Toy",
        "family": "Toy",
        "tier": "tier1",
        "claim": "A toy row for the validator's tests.",
        "architecture": "llama",
        "upstream": {"checkpoint": "toy/toy", "revision": None},
        "files": [{"name": "toy-Q8_0.gguf", "repo": "toy/toy-GGUF",
                   "revision": None, "sha256": SHA_A, "bytes": 1234}],
        "quantization": "Q8_0",
        "published": False,
        "envelope": {
            "features": ["text"],
            "fp32_reference": "measured",
            "anchor": "toy float32 reference",
            "context": None,
            "limitations": None,
            "evidence": ["docs/model-evidence.md"],
            "missing": None,
        },
        "smoke": {"device_class": "macos-arm64-metal", "backend": "metal"},
        "backends": {"cpu": None, "cuda": None, "metal": None},
        "lora": None,
        "transitions": [{"tier": "tier1", "effective": "2026-10-22",
                       "reason": "test", "last_validated_release": None}],
    }
    row.update(over)
    return row


def roster(*rows):
    return {
        "schema": mt.SCHEMA,
        "policy": {"backends": ["cpu", "cuda", "metal"]},
        "exempt": [],
        "rows": list(rows) or [tier1_row()],
    }


def errors(doc):
    return mt.validate(doc, DEVICES)


def test_a_complete_tier1_row_validates():
    assert errors(roster()) == []


def test_a_tier1_row_without_a_file_digest_is_refused():
    row = tier1_row()
    del row["files"][0]["sha256"]
    assert any("sha256" in e for e in errors(roster(row)))


def test_a_cell_that_is_neither_a_gate_nor_a_limit_is_refused():
    """Three states only: a recorded gate, a declared limit, or nothing.
    A bare `true` (or a typed checkmark) is a mark nobody measured."""
    row = tier1_row()
    row["backends"]["cuda"] = True
    assert any("cuda" in e for e in errors(roster(row)))


def test_the_smoke_backend_cannot_be_a_declared_limit():
    row = tier1_row()
    row["backends"]["metal"] = {"kind": "limit", "reason": "no Metal path",
                                "matrix": "MANUAL.md#support-matrix"}
    assert any("smoke" in e for e in errors(roster(row)))


def test_the_smoke_device_class_must_be_a_known_one():
    row = tier1_row(smoke={"device_class": "a-desk-somewhere",
                           "backend": "metal"})
    assert any("device_class" in e for e in errors(roster(row)))


def test_no_fp32_reference_must_be_said_in_the_workorders_words():
    """Section 4.1: where no float32 reference is feasible the row says
    'anchor: agreement with another engine, not fidelity', never left to be
    inferred."""
    row = tier1_row()
    row["envelope"]["fp32_reference"] = "infeasible"
    row["envelope"]["anchor"] = "llama.cpp agrees"
    assert any("another engine" in e for e in errors(roster(row)))
    row["envelope"]["anchor"] = ("anchor: agreement with another engine, "
                                 "not fidelity. llama.cpp agrees.")
    assert errors(roster(row)) == []


def test_a_published_file_below_tier1_needs_a_stated_exception():
    """Owner decision 1: a published file is Tier 1 by definition; the one
    exception is decided and written down, never implied."""
    row = tier1_row(tier="tier2", published=True)
    row["transitions"][-1]["tier"] = "tier2"
    assert any("published" in e for e in errors(roster(row)))
    row["published_exception"] = "card leads with its measured shortfall"
    assert errors(roster(row)) == []


def test_the_latest_transition_must_lead_to_the_rows_tier():
    row = tier1_row()
    row["transitions"][-1]["tier"] = "tier2"
    assert any("transition" in e for e in errors(roster(row)))


def test_transitions_are_a_history_newest_last():
    """The transition log is rendered from these, so a move appends an entry
    and the migration's entry stays: a log that kept only the latest move
    would lose the 2026-10-22 row the first time a model changed tier."""
    row = tier1_row()
    row["transitions"] = []
    assert any("transitions" in e for e in errors(roster(row)))
    row = tier1_row(tier="tier2")
    row["transitions"].append({"tier": "tier2", "effective": "2026-12-01",
                               "reason": "moved", "last_validated_release": "1.5.1"})
    assert errors(roster(row)) == []
    row["transitions"][-1]["effective"] = "2026-10-01"
    assert any("order" in e for e in errors(roster(row)))


def test_an_upcoming_row_says_what_is_missing():
    row = {"id": "later", "model": "Later", "tier": "upcoming",
           "claim": "Not admitted.", "architecture": "later", "missing": "",
           "transitions": [{"tier": "upcoming", "effective": "2026-10-22",
                          "reason": "test", "last_validated_release": None}]}
    assert any("missing" in e for e in errors(roster(tier1_row(), row)))
    row["missing"] = "an admission"
    assert errors(roster(tier1_row(), row)) == []


def test_a_tier1_or_tier2_row_names_its_family_and_an_upcoming_row_need_not():
    """docs/model-support.md lists a model under its family,
    so a gated or maintained row without one has no place in either."""
    row = tier1_row()
    del row["family"]
    assert any("family" in e for e in errors(roster(row)))
    t2 = tier1_row(tier="tier2", family=" ")
    t2["transitions"][-1]["tier"] = "tier2"
    assert any("family" in e for e in errors(roster(t2)))
    up = {"id": "later", "model": "Later", "tier": "upcoming",
          "claim": "Not admitted.", "architecture": "later", "missing": "an admission",
          "transitions": [{"tier": "upcoming", "effective": "2026-10-22",
                           "reason": "test", "last_validated_release": None}]}
    assert errors(roster(tier1_row(), up)) == []


def test_a_model_is_listed_under_one_family():
    a = tier1_row(id="toy-a")
    b = tier1_row(id="toy-b", family="Other")
    b["files"] = [{"name": "toy-b.gguf", "sha256": "e" * 64, "bytes": 1}]
    assert any("one family" in e for e in errors(roster(a, b)))
    b["family"] = "Toy"
    assert errors(roster(a, b)) == []


def test_an_experimental_row_leads_with_the_word():
    """Owner decision 4: "experimental" is the first word of the row."""
    row = tier1_row(experimental=True, claim="A toy, experimental.")
    assert any("Experimental" in e for e in errors(roster(row)))
    row["claim"] = "Experimental: a toy."
    assert errors(roster(row)) == []


def test_a_vision_models_row_leads_with_text_path_only():
    row = tier1_row(vision_upstream=True)
    assert any("Text path only" in e for e in errors(roster(row)))
    row["claim"] = "Text path only: the toy's vision encoder is not loaded."
    assert errors(roster(row)) == []


def lora(train=(True, None), serve=(True, None)):
    return {"training": {"supported": train[0], "reason": train[1]},
            "adapter_serving": {"supported": serve[0], "reason": serve[1]},
            "runner_version": "1.5.1", "date": "2026-10-23"}


def test_a_recorded_lora_answer_validates():
    """Owner decision 6: the file's --adapt-info answer, both halves."""
    row = tier1_row(lora=lora(train=(False, "MoE shared experts")))
    assert errors(roster(row)) == []


def test_a_lora_refusal_carries_its_reason_and_a_yes_carries_none():
    """The overview's Train cell prints the refusal verbatim, so a refusal
    without one would render as `no: ` and a yes with one would be two
    answers at once."""
    assert any("training" in e for e in
               errors(roster(tier1_row(lora=lora(train=(False, ""))))))
    assert any("adapter_serving" in e for e in
               errors(roster(tier1_row(lora=lora(serve=(True, "but"))))))


def test_a_lora_answer_names_the_build_and_day_it_came_from():
    answer = lora()
    del answer["runner_version"]
    assert any("lora" in e for e in errors(roster(tier1_row(lora=answer))))


def head_row(**over):
    row = tier1_row(id="toy-head", file_kind="mtp-head",
                    runs_with={"row": "toy-q8_0", "as": "mtp-file"})
    row["files"] = [{"name": "toy-mtp.gguf", "sha256": "e" * 64, "bytes": 99}]
    row.update(over)
    return row


def test_a_tier1_file_the_smoke_cannot_run_is_refused():
    """A head or an adapter does not run on its own. Without the row it runs
    with, the per-release smoke can never gate it, and the release gate,
    which refuses never-run rows, could never pass."""
    head = head_row()
    del head["runs_with"]
    assert any("cannot run" in e for e in errors(roster(tier1_row(), head)))


def test_a_head_runs_with_a_trunk_row_the_roster_names():
    assert errors(roster(tier1_row(), head_row())) == []
    assert any("runs_with" in e for e in errors(roster(
        tier1_row(), head_row(runs_with={"row": "no-such-row", "as": "mtp-file"}))))


def test_a_head_is_loaded_as_a_head_and_an_adapter_as_an_adapter():
    assert any("runs_with" in e for e in errors(roster(
        tier1_row(), head_row(runs_with={"row": "toy-q8_0", "as": "lora"}))))
    adapter = head_row(id="toy-lora", file_kind="adapter",
                       runs_with={"row": "toy-q8_0", "as": "lora"})
    assert errors(roster(tier1_row(), adapter)) == []


def test_a_companion_runs_with_a_model_not_with_another_companion():
    second = head_row(id="toy-head-2", runs_with={"row": "toy-head", "as": "mtp-file"})
    assert any("runs_with" in e for e in errors(roster(tier1_row(), head_row(), second)))


def test_a_companion_cannot_be_gated_where_its_base_has_no_path():
    base = tier1_row(smoke={"device_class": "macos-arm64-metal", "backend": "cpu"})
    base["backends"]["metal"] = {"kind": "limit", "reason": "no Metal path",
                                 "matrix": "MANUAL.md#support-matrix"}
    assert any("runs_with" in e for e in errors(roster(base, head_row())))


def test_a_model_file_does_not_name_a_row_to_run_with():
    row = tier1_row(runs_with={"row": "toy-q8_0", "as": "lora"})
    assert any("runs_with" in e for e in errors(roster(row)))


def test_duplicate_row_ids_are_refused():
    assert any("duplicate" in e for e in errors(roster(tier1_row(), tier1_row())))


def test_a_wrong_schema_is_refused(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"schema": "something.else", "rows": []}),
                 encoding="utf-8")
    with pytest.raises(SystemExit):
        mt.load(p)


# --- the committed roster ---------------------------------------------------

# GET https://huggingface.co/api/models?author=Joakimpalm-Zen, 2026-10-23.
PUBLISHED_REPOS = {
    "gpt-oss-20b-keep30-MXFP4-GGUF", "gemma-4-E2B-it-Q4_0-GGUF",
    "Qwen3-Coder-30B-A3B-Instruct-keep120-Q4_K_M-GGUF",
    "Qwen3-30B-A3B-selective-attnQ8_0-expQ4_0-GGUF",
    "NVIDIA-Nemotron-Nano-9B-v2-Q8_0-GGUF", "Qwen3-4B-Runner-ToolUse-Q4_K_M",
    "Muse-Glimmer-30B-Surgical-Q4_K-GGUF", "Muse-Glimmer-30B-Surgical-Q8_0-GGUF",
    "gemma-4-31B-it-attn48-removed-Q4_0-GGUF",
    "Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered-GGUF",
    "Muse-Glimmer-30B-Surgical-6p34-Q4_K-recovered-GGUF", "Xyntetik-Kvist-14B",
    "Xyntetik-Kvist-14B-GGUF", "Xyntetik-Kvist-14B-research-checkpoints",
    "Xyntetik-Blad-Genesis-0.7M", "Qwen3.8-Flash-Next-MTP-GGUF",
    "Cloudflare-clef-IQ3_S-recovered-GGUF",
}


def committed():
    return mt.load(ROSTER)


def test_the_committed_roster_validates():
    r = subprocess.run([sys.executable, str(SCRIPT), "--check"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True)
    assert r.returncode == 0, r.stdout


def test_every_published_repository_is_a_row_or_exempt():
    doc = committed()
    seen = {f["repo"].split("/", 1)[1] for row in doc["rows"]
            for f in row.get("files", [])
            if f.get("repo", "").startswith("Joakimpalm-Zen/")}
    seen |= {e["repo"].split("/", 1)[1] for e in doc["exempt"]}
    assert PUBLISHED_REPOS <= seen, sorted(PUBLISHED_REPOS - seen)


def test_the_published_e2b_row_names_the_hubs_digest():
    doc = committed()
    shas = {f["sha256"] for row in doc["rows"] for f in row.get("files", [])
            if f["name"] == "gemma-4-E2B-it-Q4_K_M-Q4_0-mix.gguf"}
    assert shas == {"16ffaa2ea568e7a0c9b5c6ddbd39a376879f9ac007861334974de39f4cc52e3f"}


def test_the_owner_decisions_of_2026_10_22_are_in_the_roster():
    rows = {row["id"]: row for row in committed()["rows"]}
    by_file = {f["name"]: row for row in rows.values()
               for f in row.get("files", [])}
    assert by_file["gpt-oss-20b-keep30-MXFP4.gguf"]["tier"] == "tier2"
    # Tier 1 until 2026-10-27, when Tier 1 began to follow pull
    assert by_file["Trinity-Nano-Preview-Q8_0.gguf"]["tier"] == "tier2"
    flash = by_file["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00001-of-00004.gguf"]
    assert flash["tier"] == "tier1"
    assert mt.first_word(flash["claim"]) == "Experimental"
    assert not [r for r in rows.values() if r["tier"] == "tier3"]


def test_the_committed_families_are_the_owners_list():
    """Since 2026-10-27 Tier 1 follows pull: the families whose repositories
    showed at least 1,000 downloads in 30 days, plus the publisher files the
    owner named for active validation on 2026-10-29 (Bolt E4B, Mellum2.1);
    the README lists them in the order of each one's first row."""
    families = [f for f, _ in mt.tier1_families(committed())]
    assert set(families) == {"Gemma 4", "Mellum2.1", "Qwen3.8", "Xyntetik-Kvist-14B"}
    first = {}
    for r in committed()["rows"]:
        if r["tier"] == "tier1":
            first.setdefault(r["family"], len(first))
    assert families == sorted(first, key=first.get)


def test_no_cell_in_the_committed_roster_was_typed():
    """A gate cell names the run that wrote it: a commit, a caps fingerprint
    and an output digest. A cell without them was not written by --record."""
    for row in committed()["rows"]:
        for backend, cell in (row.get("backends") or {}).items():
            if cell and cell.get("kind") == "gate":
                for key in ("commit", "caps", "output_sha256", "ran"):
                    assert cell.get(key), (row["id"], backend, key)


# --- --record: a cell comes from a smoke run's result file, never by hand ---

RUN_DAY = "2026-10-23"
RUN_VERSION = "1.5.1"


def smoke_result(**over):
    """What scripts/tier-smoke.py writes after a compared run on the Metal
    laptop: the machine's --caps fingerprint, the file digests it hashed, the
    compared output's digest and the file's --adapt-info answer."""
    res = {
        "schema": mt.RESULT_SCHEMA,
        "row": "toy-q8_0",
        "backend": "metal",
        "device_class": "macos-arm64-metal",
        "generated_utc": RUN_DAY + "T10:00:00Z",
        "commit": "b" * 40,
        "ran": "runner -m toy-Q8_0.gguf -p 'The capital of France is' -n 32 --temp 0",
        "caps": {"version": RUN_VERSION, "os": "macos", "arch": "arm64",
                 "cpu_cores": 8, "gpu_backend": "metal", "gpu_name": "Apple GPU"},
        "banner": "gpu: Metal backend on Apple GPU (zero-copy weights)",
        "files": [{"sha256": SHA_A, "bytes": 1234}],
        "status": "match",
        "output": {"sha256": "c" * 64, "bytes": 40},
        "pin": {"path": "docs/model-evidence.md", "sha256": "c" * 64},
        "expect": {"substring": "Paris", "found": True},
        "adapt_info": {"architecture": "llama",
                       "adapter_serving": {"supported": True, "reason": None,
                                           "device_kernels": "metal"},
                       "training": {"supported": False,
                                    "reason": "MoE shared experts", "host": "cpu"}},
    }
    res.update(over)
    return res


def record(res, row=None):
    doc = roster(row or tier1_row())
    mt.record(doc, res["row"], res, DEVICES)
    return doc


def test_record_writes_the_cell_and_the_lora_answer():
    doc = record(smoke_result())
    row = doc["rows"][0]
    cell = row["backends"]["metal"]
    assert cell["kind"] == "gate"
    assert (cell["runner_version"], cell["date"], cell["result"]) == \
        (RUN_VERSION, RUN_DAY, "pass")
    assert cell["device_class"] == "macos-arm64-metal"
    assert row["lora"] == {
        "training": {"supported": False, "reason": "MoE shared experts"},
        "adapter_serving": {"supported": True, "reason": None},
        "runner_version": RUN_VERSION, "date": RUN_DAY}
    assert row["backends"]["cpu"] is None and row["backends"]["cuda"] is None
    assert errors(doc) == []


def test_record_refuses_a_run_on_a_different_file():
    with pytest.raises(SystemExit, match="sha256"):
        record(smoke_result(files=[{"sha256": "d" * 64, "bytes": 1234}]))


def test_record_refuses_caps_from_the_wrong_machine():
    caps = dict(smoke_result()["caps"], os="linux", arch="x86_64")
    with pytest.raises(SystemExit, match="wrong machine"):
        record(smoke_result(caps=caps))


def test_record_refuses_a_gpu_cell_from_a_run_that_did_not_offload():
    """A Metal build that fell back to the CPU prints no backend banner; its
    output is a CPU result and cannot fill the Metal cell."""
    with pytest.raises(SystemExit, match="banner"):
        record(smoke_result(banner=None))


def test_record_refuses_a_cpu_cell_from_a_run_on_the_gpu():
    with pytest.raises(SystemExit, match="banner"):
        record(smoke_result(backend="cpu"))


def test_record_refuses_the_run_that_wrote_the_pin():
    """A pinning run compared nothing; only a later run against the pin is
    a gate."""
    with pytest.raises(SystemExit, match="pin"):
        record(smoke_result(status="pinned"))


def test_record_refuses_a_declared_limit_backend():
    row = tier1_row(smoke={"device_class": "macos-arm64-metal", "backend": "cpu"})
    row["backends"]["metal"] = {"kind": "limit", "reason": "no Metal path",
                                "matrix": "MANUAL.md#support-matrix"}
    with pytest.raises(SystemExit, match="limit"):
        record(smoke_result(), row)


def test_record_writes_a_tier2_cell_as_dated_evidence():
    """A Tier 2 file a run includes is recorded like a Tier 1 file (a new
    admission at Tier 2 records the pass that admitted it); the gate still
    reads only Tier 1 rows."""
    row = tier1_row(tier="tier2")
    row["transitions"][-1]["tier"] = "tier2"
    doc = record(smoke_result(), row)
    cell = doc["rows"][0]["backends"]["metal"]
    assert cell["result"] == "pass" and cell["runner_version"] == "1.5.1"
    assert mt.gate(doc, TODAY) == []


def test_record_refuses_a_row_below_tier2():
    row = tier1_row(tier="tier3")
    row["transitions"][-1]["tier"] = "tier3"
    with pytest.raises(SystemExit, match="Tier 1 or Tier 2"):
        record(smoke_result(), row)


def test_a_mismatch_against_the_pin_records_a_fail():
    doc = record(smoke_result(status="mismatch"))
    assert doc["rows"][0]["backends"]["metal"]["result"] == "fail"


def test_a_missing_outside_fact_records_a_fail():
    """The absolute anchor: the continuation must contain the fact the
    prompt asks for. Matching our own pin while losing the fact is a fail."""
    doc = record(smoke_result(expect={"substring": "Paris", "found": False}))
    assert doc["rows"][0]["backends"]["metal"]["result"] == "fail"


def test_the_cli_records_into_the_roster_file(tmp_path):
    rpath = tmp_path / "tiers.json"
    dpath = tmp_path / "devices.json"
    res = tmp_path / "result.json"
    mt.save(roster(), rpath)
    dpath.write_text(json.dumps(DEVICES), encoding="utf-8")
    res.write_text(json.dumps(smoke_result()), encoding="utf-8")
    r = subprocess.run([sys.executable, str(SCRIPT), "--json", str(rpath),
                        "--device-evidence", str(dpath), "--record", "toy-q8_0",
                        "--result", str(res)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert r.returncode == 0, r.stdout
    row = mt.load(rpath)["rows"][0]
    assert row["backends"]["metal"]["date"] == RUN_DAY


# --- the gate: what scripts/check-release.py refuses a tag on ----------------
# Policy decision 2026-10-28: a record is never re-run by age. The gate asks
# only for a recorded pass on the backend the row names and no recorded
# failure on any; the printed age is information for the paper trail.

import datetime  # noqa: E402

TODAY = datetime.date(2026, 10, 23)


def gate(day, result="pass", device_class="macos-arm64-metal", version="1.5.0"):
    return {"kind": "gate", "runner_version": version, "date": day,
            "device_class": device_class, "result": result, "commit": "b" * 40,
            "caps": {"version": version}, "output_sha256": "c" * 64,
            "ran": "tier-smoke", "evidence": "docs/model-evidence.md"}


def gated(**cells):
    row = tier1_row()
    row["backends"].update(cells)
    return roster(row)


def test_a_recorded_pass_on_the_named_backend_passes_the_gate():
    assert mt.gate(gated(metal=gate("2026-10-22")), TODAY) == []


def test_a_tier1_row_whose_smoke_never_ran_is_refused():
    """The device ledger's NEVER: a row nobody validated has no pass to show."""
    assert mt.gate(gated(), TODAY) == ["toy-q8_0"]


def test_an_old_record_is_not_refused_and_its_age_is_printed(capsys):
    """No re-testing by age: a pass from months ago stands until the next
    validation run, and the report says how old it is."""
    assert mt.gate(gated(metal=gate("2026-01-01")), TODAY) == []
    out = capsys.readouterr().out
    assert out.startswith("ok") and "295 days ago" in out and "1.5.0 on 2026-01-01" in out


def test_the_roster_carries_no_age_policy():
    """The age limit left with the decision: neither the policy nor a row
    holds one, and the validator asks for none."""
    doc = roster()
    assert "default_max_age_days" not in doc["policy"]
    assert errors(doc) == []
    committed_policy = mt.load(ROSTER)["policy"]
    assert "default_max_age_days" not in committed_policy
    assert not any("max_age_days" in r for r in mt.load(ROSTER)["rows"])


def test_a_failing_cell_on_any_backend_refuses():
    """Recent and red is worse than absent: a recorded failure on a backend
    the row is not gated on still says this file broke."""
    doc = gated(metal=gate("2026-10-22"), cpu=gate("2026-10-22", result="fail"))
    assert mt.gate(doc, TODAY) == ["toy-q8_0"]


def test_a_pass_from_another_device_class_counts_and_is_named(capsys):
    """A recorded pass stands wherever it ran; the report names the class
    it ran on beside the one the row names for its next run."""
    doc = gated(metal=gate("2026-10-22", device_class="linux-x86_64-cuda"))
    assert mt.gate(doc, TODAY) == []
    assert "metal on linux-x86_64-cuda" in capsys.readouterr().out


def test_rows_below_tier1_are_not_gated():
    row = tier1_row(tier="tier2")
    row["transitions"][-1]["tier"] = "tier2"
    assert mt.gate(roster(row), TODAY) == []


# Borrowed hardware (owner decision 2026-10-27): a row whose smoke class is
# a lent machine is reported like the others, but a missing record on it
# never holds a release; a recorded failure does on any class.

LENT = frozenset({"macos-arm64-metal"})


def test_a_never_gated_row_on_borrowed_hardware_does_not_refuse(capsys):
    assert mt.gate(gated(), TODAY, borrowed=LENT) == []
    out = capsys.readouterr().out
    assert out.startswith("NEVER") and "borrowed hardware" in out


def test_a_failing_cell_on_borrowed_hardware_still_refuses():
    doc = gated(metal=gate("2026-10-22", result="fail"))
    assert mt.gate(doc, TODAY, borrowed=LENT) == ["toy-q8_0"]


def test_a_never_gated_row_on_owned_hardware_refuses_whatever_else_is_borrowed():
    assert mt.gate(gated(), TODAY, borrowed=frozenset({"linux-x86_64-cuda"})) == ["toy-q8_0"]


def test_borrowed_classes_come_from_the_ledgers_ownership_field():
    devices = {"classes": [{"id": "a", "ownership": "borrowed"},
                           {"id": "b", "ownership": "owned"}, {"id": "c"}]}
    assert mt.borrowed_classes(devices) == frozenset({"a"})


def test_the_committed_ledger_marks_the_lent_linux_host_borrowed():
    """The decision this encodes: the Linux CUDA host is lent with no terms."""
    assert "linux-x86_64-cuda" in mt.borrowed_classes()
    assert "windows-x86_64-cuda" not in mt.borrowed_classes()


def test_an_overview_mark_on_borrowed_hardware_carries_the_superscript():
    cell = gate("2026-10-22")
    assert "<sup>b</sup>" in mt.overview_mark(cell, LENT)
    assert "<sup>b</sup>" not in mt.overview_mark(cell, frozenset())
    # the legend always explains the superscript; the cells carry it only
    # when their class is borrowed
    assert "<sup>b</sup>" in mt.OVERVIEW_LEGEND
    cells = lambda text: text.split("<details>", 1)[1]
    assert "<sup>b</sup>" in cells(mt.render_overview(gated(metal=cell), LENT))
    assert "<sup>b</sup>" not in cells(mt.render_overview(gated(metal=cell), frozenset()))


def test_the_support_doc_names_borrowed_hardware_on_the_row():
    text = mt.render_doc(gated(metal=gate("2026-10-22")), LENT)
    assert "(borrowed hardware" in text
    assert "(borrowed hardware" not in mt.render_doc(gated(metal=gate("2026-10-22")),
                                                     frozenset())


def test_the_gate_report_prints_every_tier1_row(capsys):
    mt.gate(gated(metal=gate("2026-10-22")), TODAY)
    assert "toy-q8_0" in capsys.readouterr().out


def test_the_gate_summary_follows_the_rows_in_one_pipe(tmp_path):
    """check-release reads this check with stdout and stderr merged into
    one pipe; the per-row lines must come before the summary under them."""
    rpath = tmp_path / "tiers.json"
    dpath = tmp_path / "devices.json"
    mt.save(gated(), rpath)
    dpath.write_text(json.dumps(DEVICES), encoding="utf-8")
    r = subprocess.run([sys.executable, str(SCRIPT), "--gate", "--json", str(rpath),
                        "--device-evidence", str(dpath), "--today", TODAY.isoformat()],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert r.returncode == 1
    assert r.stdout.index("NEVER") < r.stdout.index("need")


def test_record_refuses_a_companion_run_on_another_base():
    doc = roster(tier1_row(), head_row())
    res = smoke_result(row="toy-head", files=[{"sha256": "e" * 64, "bytes": 99}],
                       **{"with": {"row": "toy-q8_0",
                                   "files": [{"sha256": "f" * 64, "bytes": 1}]}})
    with pytest.raises(SystemExit, match="runs with"):
        mt.record(doc, "toy-head", res, DEVICES)
    res["with"]["files"] = [{"sha256": SHA_A, "bytes": 1234}]
    mt.record(doc, "toy-head", res, DEVICES)
    assert doc["rows"][1]["backends"]["metal"]["result"] == "pass"


# --- the overview (docs/model-support.md): a render of the roster, never typed --
#
# One collapsible block per Tier 1 family: a plain summary line (the bold
# family name, its model count, a mark per backend and for Train over all
# its files), then per model its line (the bold name, its label in plain
# words) and a table with one row per roster file. The marks below are
# written out by hand from the owner's decisions of 2026-10-23 and
# 2026-10-24: a recorded cell is the version and date of its gate, not run
# is a blank cell, a limit is n/a linked to the manual; a family's mark is
# ✅, 🟨 k of n, a hyphen or n/a.

def overview_families(doc):
    """(summary, models) per family block of the render, in order; a model
    is (its line, its file rows) and a file row is its list of <td>
    elements, attributes and all."""
    out = []
    for block in re.findall(r"<details>\n(.*?)\n</details>", mt.render_overview(doc), re.S):
        summary = re.match(r"<summary>(.*)</summary>\n", block).group(1)
        models = []
        for line, table in re.findall(r"<p>(.*?)</p>\n(<table>.*?</table>)", block, re.S):
            rows = [re.findall(r"<td[^>]*>.*?</td>", tr)
                    for tr in re.findall(r"<tr>(.*?)</tr>", table) if "<td" in tr]
            models.append((line, rows))
        out.append((summary, models))
    return out


def file_row(doc, name):
    return [r for _, models in overview_families(doc) for _, rows in models
            for r in rows if name in r[0]][0]


def marks(summary):
    """The marks of a family's summary line, label to mark."""
    m = re.match(r"<b>.*?</b>, \d+ models?: (.*)$", summary)
    return dict(x.split(" ", 1) for x in m.group(1).split(" · "))


def test_the_readme_links_to_the_overview_and_carries_no_render():
    """Since 2026-10-27 the marks are rendered once, in docs/model-support.md:
    the README's section says what the check is and links to the overview,
    and types no table, legend or Upcoming line of its own."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/model-support.md#overview" in text
    assert "model-tiers:begin" not in text and "<summary><b>" not in text
    assert "**Upcoming**" not in text and mt.OVERVIEW_LEGEND not in text


def test_the_doc_block_opens_with_the_overview():
    """The rendered block of docs/model-support.md starts with the validation
    runs at a glance (its anchor, heading and legend), before the tiers."""
    doc = committed()
    text = mt.render_doc(doc).lstrip("\n")
    assert text.startswith('<a id="overview"></a>\n## Validation runs at a glance\n\n'
                           + mt.OVERVIEW_LEGEND)
    assert text.index("</details>") < text.index('<a id="tier1"></a>')


def test_a_recorded_cell_renders_its_version_and_date_and_not_run_is_blank():
    """The version over the date, in a cell that does not wrap: GitHub's
    table layout otherwise splits a date at its hyphens when a neighbour
    (a refusal sentence) claims the width."""
    doc = gated(metal=gate("2026-10-23", version="1.5.1"))
    assert file_row(doc, "toy-Q8_0.gguf")[1:] == [
        "<td></td>", "<td></td>", "<td nowrap>✅ 1.5.1<br>2026-10-23</td>", "<td></td>"]


def test_a_declared_limit_renders_n_a_linked_to_the_matrix():
    doc = gated(metal=gate("2026-10-23"),
                cuda={"kind": "limit", "reason": "no CUDA path",
                      "matrix": "MANUAL.md#support-matrix"})
    # the link is written from docs/, where the overview lives
    assert file_row(doc, "toy-Q8_0.gguf")[2] == '<td><a href="../MANUAL.md#support-matrix">n/a</a></td>'


def test_a_measurement_renders_after_the_cells_and_only_when_present():
    """A dated deployment measurement sits beside the gate cells it was
    taken next to, one bullet each, and a row without one renders as before."""
    plain = tier1_row()
    lines = mt._row_doc(plain, {plain["id"]: plain})
    measured = tier1_row(measurements=["**Fit (Metal, 2026-10-29):** 4.7 s.",
                                       "**Second:** two."])
    with_m = mt._row_doc(measured, {measured["id"]: measured})
    metal = next(i for i, l in enumerate(with_m) if l.startswith("- **Metal:**"))
    assert with_m[metal + 1:metal + 3] == ["- **Fit (Metal, 2026-10-29):** 4.7 s.",
                                           "- **Second:** two."]
    assert [l for l in with_m if l not in lines] == with_m[metal + 1:metal + 3]


def test_a_failed_gate_never_renders_a_check_mark():
    doc = gated(metal=gate("2026-10-23", result="fail"))
    assert file_row(doc, "toy-Q8_0.gguf")[3] == "<td nowrap>failed 1.5.0<br>2026-10-23</td>"
    (summary, _), = overview_families(doc)
    assert marks(summary)["Metal"] == "failed" and "✅" not in summary


def test_the_train_column_is_the_recorded_adapt_info_answer():
    doc = gated(metal=gate("2026-10-23"))
    doc["rows"][0]["lora"] = lora(train=(False, "MoE shared experts | <router> & more"))
    # the refusal is the engine's sentence, escaped as HTML, and it may wrap
    assert file_row(doc, "toy-Q8_0.gguf")[4] == \
        "<td>no: MoE shared experts | &lt;router&gt; &amp; more</td>"
    doc["rows"][0]["lora"] = lora()
    assert file_row(doc, "toy-Q8_0.gguf")[4] == "<td nowrap>✅ 1.5.1<br>2026-10-23</td>"


def test_every_tier1_file_gets_its_row_even_unrun():
    """Every roster file is one the owner committed to validate, so it is
    listed, its cells blank until a gate records them; one shape for every
    model, run or not."""
    (summary, [(line, rows)]), = overview_families(roster())
    assert line == "<b>Toy</b>"
    assert rows == [['<td><a href="#toy-q8_0"><code>toy-Q8_0.gguf'
                     '</code></a></td>'] + ["<td></td>"] * 4]
    assert marks(summary) == {"CPU": "-", "CUDA": "-", "Metal": "-", "Train": "-"}


def test_each_family_is_one_collapsed_block_of_html_with_no_blank_line():
    """The block is one HTML block in GitHub's markdown: a blank line inside
    it would end the block and print the rest as text. Each model of the
    family is its line and its table, one <tr> per file."""
    other = tier1_row(id="toy-b", model="Toy B")
    other["files"] = [{"name": "toy-b.gguf", "sha256": "e" * 64, "bytes": 1}]
    text = mt.render_overview(roster(tier1_row(), other))
    assert text.count("<details>") == 1
    block = text.split("<details>\n", 1)[1].split("</details>", 1)[0]
    lines = block.split("\n")
    head = ["<table>", "<thead><tr><th>File</th><th>CPU</th><th>CUDA</th>"
            "<th>Metal</th><th>Train</th></tr></thead>", "<tbody>"]
    assert lines[0] == ("<summary><b>Toy</b>, 2 models: CPU - · CUDA - · Metal - · "
                        "Train -</summary>")
    assert lines[1] == "<p><b>Toy</b></p>" and lines[2:5] == head
    assert lines[5].startswith("<tr><td>") and "toy-Q8_0.gguf" in lines[5]
    assert lines[6:8] == ["</tbody>", "</table>"]
    assert lines[8] == "<p><b>Toy B</b></p>" and lines[9:12] == head
    assert lines[12].startswith("<tr><td>") and "toy-b.gguf" in lines[12]
    assert lines[13:] == ["</tbody>", "</table>", ""]
    assert "\n\n" not in block


def test_the_summary_line_is_plain_text_and_the_label_follows_the_model_name():
    """No inline code pills on the family's line, and the label (text path
    only, experimental) belongs to the model it qualifies."""
    doc = gated(metal=gate("2026-10-23"))
    doc["rows"][0].update(experimental=True, claim="Experimental: a toy.")
    (summary, [(line, _)]), = overview_families(doc)
    assert line == "<b>Toy</b>, experimental"
    assert summary == "<b>Toy</b>, 1 model: CPU - · CUDA - · Metal ✅ · Train -"
    doc["rows"][0].update(vision_upstream=True)
    assert overview_families(doc)[0][1][0][0] == "<b>Toy</b>, experimental, text path only"
    text = mt.render_overview(doc)
    assert "Text path only:" not in text and "<code>CPU" not in text
    assert "&nbsp;" not in text


def test_the_summary_is_html_so_family_and_model_names_are_escaped():
    doc = roster(tier1_row(model="Toy <R&D>", family="Fam <R&D>"))
    (summary, [(line, _)]), = overview_families(doc)
    assert summary.startswith("<b>Fam &lt;R&amp;D&gt;</b>, 1 model: ")
    assert line == "<b>Toy &lt;R&amp;D&gt;</b>"


def test_the_file_name_links_its_envelope_and_a_sharded_set_counts_its_parts():
    row = tier1_row()
    row["files"] = [{"name": f"toy-0000{i}-of-00003.gguf", "sha256": SHA_A, "bytes": 1}
                    for i in (1, 2, 3)]
    (_, [(_, rows)]), = overview_families(roster(row))
    assert rows[0][0] == ('<td><a href="#toy-q8_0"><code>'
                          'toy-00001-of-00003.gguf</code></a> (3 parts)</td>')


def test_the_summary_sums_up_each_backend_over_every_file_of_the_family():
    """A family's mark per backend and for Train, written out by hand for
    files of two models in mixed states: n/a when every file has the limit,
    failed when any recorded a failure, ✅ when every file is recorded, a
    hyphen when none is, 🟨 k of n otherwise, where a limit beside recorded
    files counts as not run. Train counts admitted files, and a refusal is
    the engine's limit."""
    limit = {"kind": "limit", "reason": "no path", "matrix": "MANUAL.md#support-matrix"}
    a = tier1_row(id="mix-a", model="Mix", family="Mixed")
    b = tier1_row(id="mix-b", model="Mix", family="Mixed")
    c = tier1_row(id="mix-c", model="Mix Two", family="Mixed")
    for r, name in ((b, "mix-b.gguf"), (c, "mix-c.gguf")):
        r["files"] = [{"name": name, "sha256": "e" * 64, "bytes": 1}]
    a["backends"] = {"cpu": gate("2026-10-23"), "cuda": gate("2026-10-23"), "metal": limit}
    b["backends"] = {"cpu": gate("2026-10-23"), "cuda": None, "metal": limit}
    c["backends"] = {"cpu": limit, "cuda": None, "metal": limit}
    a["lora"], b["lora"] = lora(), lora(train=(False, "MoE"))
    d = tier1_row(id="odd-a", model="Odd", family="Odd")
    e = tier1_row(id="odd-b", model="Odd", family="Odd")
    e["files"] = [{"name": "odd-b.gguf", "sha256": "f" * 64, "bytes": 1}]
    d["backends"] = {"cpu": gate("2026-10-23"), "cuda": limit, "metal": None}
    e["backends"] = {"cpu": gate("2026-10-23", result="fail"), "cuda": None, "metal": None}
    d["lora"] = e["lora"] = lora(train=(False, "recurrent"))
    f = tier1_row(id="all-a", model="All", family="All")
    f["backends"] = {"cpu": gate("2026-10-23"), "cuda": gate("2026-10-23"),
                     "metal": gate("2026-10-23")}
    f["lora"] = lora()
    blocks = overview_families(roster(a, b, c, d, e, f))
    assert [marks(s) for s, _ in blocks] == [
        {"CPU": "🟨 2 of 3", "CUDA": "🟨 1 of 3", "Metal": "n/a", "Train": "🟨 1 of 3"},
        {"CPU": "failed", "CUDA": "-", "Metal": "-", "Train": "n/a"},
        {"CPU": "✅", "CUDA": "✅", "Metal": "✅", "Train": "✅"},
    ]
    assert [s.split(":")[0] for s, _ in blocks] == [
        "<b>Mixed</b>, 2 models", "<b>Odd</b>, 1 model", "<b>All</b>, 1 model"]
    assert [[len(rows) for _, rows in models] for _, models in blocks] == [[2, 1], [2], [1]]


def test_families_keep_roster_order_and_models_sort_by_name_within_one():
    """A family is listed where its first row stands in the roster; within
    it, models sort by name (the label aside) and files keep their roster
    order, so adding a row to a family does not move the family."""
    zeta = tier1_row(id="zeta-a", model="Zeta", family="Last")
    zeta2 = tier1_row(id="zeta-b", model="Zeta", family="Last")
    zeta2["files"] = [{"name": "zeta-b.gguf", "sha256": "e" * 64, "bytes": 1}]
    alpha = tier1_row(id="alpha", model="alpha", family="Last", experimental=True,
                      claim="Experimental: a toy.")
    first = tier1_row(id="first", model="First", family="First")
    first["files"] = [{"name": "first.gguf", "sha256": "f" * 64, "bytes": 1}]
    blocks = overview_families(roster(zeta, first, zeta2, alpha))
    assert [s.split("</b>")[0] for s, _ in blocks] == ["<b>Last", "<b>First"]
    assert [line.split("</b>")[0] for line, _ in blocks[0][1]] == ["<b>alpha", "<b>Zeta"]
    assert [re.search(r"<code>(.*?)</code>", r[0]).group(1) for r in blocks[0][1][1][1]] == \
        ["toy-Q8_0.gguf", "zeta-b.gguf"]
    assert mt.render_overview(roster(zeta, first, alpha, zeta2)) == \
        mt.render_overview(roster(zeta, first, zeta2, alpha))
    assert [m for _, m, _ in mt.tier1_groups(roster(zeta, first, zeta2, alpha))] == \
        ["alpha", "Zeta", "First"]


def test_a_declared_limit_links_the_manuals_support_matrix():
    """Every n/a points at the same place a reader checks a limit: the
    manual's support matrix. Other evidence belongs in the row's envelope."""
    row = tier1_row()
    row["backends"]["cuda"] = {"kind": "limit", "reason": "no CUDA path",
                               "matrix": "tests/compatibility/coverage.json"}
    assert any("matrix" in e for e in errors(roster(row)))


def test_the_legend_comes_first_and_names_every_mark():
    text = mt.render_overview(roster())
    legend = text.strip().split("\n")[0]
    assert text.index(legend) < text.index("<details>")
    for word in ("✅", "blank", "n/a", "failed", "`--info adapter`", "🟨 *k of n*",
                 "a hyphen: none run", "family", "envelope"):
        assert word in legend, word
    # no en or em dash in the prose or the marks
    assert "\u2013" not in text and "\u2014" not in text


def test_upcoming_models_are_one_line_under_the_blocks_and_tier2_has_its_own_line():
    """Since 2026-10-28 the overview is the record of every file's last
    validation run: Tier 1 families first, Tier 2 families under their own
    line (dated evidence, not gated), Upcoming models as one line last."""
    up = {"id": "later", "model": "Later VL", "tier": "upcoming",
          "claim": "Text path only: not admitted.", "architecture": "later",
          "missing": "an admission", "vision_upstream": True,
          "transitions": [{"tier": "upcoming", "effective": "2026-10-22",
                           "reason": "test", "last_validated_release": None}]}
    up2 = dict(up, id="sooner", model="Sooner", vision_upstream=False)
    t2 = tier1_row(id="old", model="Old", family="Elders", tier="tier2")
    t2["transitions"][-1]["tier"] = "tier2"
    text = mt.render_overview(roster(tier1_row(), t2, up, up2))
    tail = text.rsplit("</details>", 1)[1]
    assert ("**Upcoming** (planned, not supported yet): Later VL (text path only), "
            "Sooner.") in tail.splitlines()
    assert "Later" not in text.rsplit("</details>", 1)[0]
    t1_at = text.index("**Tier 1, actively validated**")
    t2_at = text.index("**Tier 2, maintained on report**")
    assert t1_at < text.index("<b>Toy</b>") < t2_at < text.index("<b>Elders</b>")
    # the committed roster renders its Tier 2 families the same way
    committed_text = mt.render_overview(committed())
    assert "**Tier 2, maintained on report**" in committed_text
    assert "<summary><b>Gemma 4</b>" in committed_text


# --- docs/model-support.md: envelopes and the transition log, rendered ----------

def test_the_support_doc_block_is_the_render_of_the_roster():
    text = (ROOT / "docs" / "model-support.md").read_text(encoding="utf-8")
    assert text.count(mt.DOC_BEGIN) == 1 and text.count(mt.DOC_END) == 1
    block = text.split(mt.DOC_BEGIN, 1)[1].split(mt.DOC_END, 1)[0]
    assert block == mt.render_doc(committed())


def test_the_support_doc_lists_models_under_their_family_and_upcoming_alone():
    """docs/model-support.md groups a tier the overview's way (a family's
    heading, its models' headings, each file's envelope under its model);
    an Upcoming model keeps its own heading, in the order of admission."""
    t2 = tier1_row(id="old", model="Old", family="Elders", tier="tier2")
    t2["transitions"][-1]["tier"] = "tier2"
    up = {"id": "later", "model": "Later", "tier": "upcoming",
          "claim": "Not admitted.", "architecture": "later", "missing": "an admission",
          "transitions": [{"tier": "upcoming", "effective": "2026-10-22",
                           "reason": "test", "last_validated_release": None}]}
    other = tier1_row(id="toy-b", model="Toy B")
    other["files"] = [{"name": "toy-b.gguf", "sha256": "e" * 64, "bytes": 1}]
    text = mt.render_doc(roster(tier1_row(), t2, up, other))
    heads = [l for l in text.splitlines() if l.startswith("#")]
    assert heads[0] == "## Validation runs at a glance"
    heads = heads[1:]
    assert heads[:7] == ["## Tier 1, actively validated", "### Toy", "#### Toy",
                         "##### `toy-Q8_0.gguf`", "#### Toy B", "##### `toy-b.gguf`",
                         "## Tier 2, maintained on report"]
    assert heads[7:10] == ["### Elders", "#### Old", "##### `toy-Q8_0.gguf`"]
    up_at = heads.index("## Upcoming, planned admission")
    assert heads[up_at + 1:up_at + 3] == ["### Later", "#### Later"]


def test_every_envelope_link_in_the_overview_lands_on_a_row():
    import re
    doc = committed()
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', mt.render_doc(doc)))
    links = set(re.findall(r'href="#([A-Za-z0-9._-]+)"', mt.render_overview(doc)))
    assert links and links <= anchors, sorted(links - anchors)


def test_an_envelope_carries_the_lora_answer_and_the_tier_history():
    doc = gated(metal=gate("2026-10-23"))
    doc["rows"][0]["lora"] = lora(train=(False, "MoE shared experts"))
    text = mt.render_doc(doc)
    assert "no: MoE shared experts" in text
    assert "adapter serving" in text.lower()
    assert "2026-10-22" in text and "Tier 1" in text


def test_the_transition_log_lists_every_entry_of_every_row():
    row = tier1_row(tier="tier2")
    row["transitions"] = [
        {"tier": "tier1", "effective": "2026-10-22", "reason": "migration",
         "last_validated_release": None},
        {"tier": "tier2", "effective": "2026-12-01", "reason": "moved down",
         "last_validated_release": "1.5.1"}]
    log = mt.render_doc(roster(row)).split("Transition log", 1)[1]
    assert "migration" in log and "moved down" in log and "1.5.1" in log
    assert log.index("migration") < log.index("moved down")


def test_links_in_the_support_doc_resolve_from_docs():
    """The roster stores repository paths; the document lives in docs/, so
    every rendered link is rewritten relative to it and must exist."""
    import re
    text = mt.render_doc(committed())
    for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text):
        if target.startswith("http"):
            continue
        assert (ROOT / "docs" / target).resolve().exists(), target
