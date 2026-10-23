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
        "policy": {"default_max_age_days": 45,
                   "backends": ["cpu", "cuda", "metal"]},
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
    """The README's Train cell prints the refusal verbatim, so a refusal
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
    assert by_file["Trinity-Nano-Preview-Q8_0.gguf"]["tier"] == "tier1"
    flash = by_file["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00001-of-00004.gguf"]
    assert flash["tier"] == "tier1"
    assert mt.first_word(flash["claim"]) == "Experimental"
    assert not [r for r in rows.values() if r["tier"] == "tier3"]


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


def test_record_refuses_a_row_below_tier1():
    row = tier1_row(tier="tier2")
    row["transitions"][-1]["tier"] = "tier2"
    with pytest.raises(SystemExit, match="Tier 1"):
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


# --- staleness: what scripts/check-release.py refuses a tag on ---------------

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


def test_a_smoke_cell_inside_the_max_age_is_fresh():
    assert mt.stale(gated(metal=gate("2026-10-22")), TODAY) == []


def test_a_tier1_row_whose_smoke_never_ran_is_refused():
    """The device ledger's NEVER: a row nobody gated is not current."""
    assert mt.stale(gated(), TODAY) == ["toy-q8_0"]


def test_a_smoke_cell_past_the_max_age_is_stale():
    # 2026-09-07 is 46 days before 2026-10-23; the policy allows 45
    assert mt.stale(gated(metal=gate("2026-09-07")), TODAY) == ["toy-q8_0"]


def test_the_boundary_day_is_not_stale():
    assert mt.stale(gated(metal=gate("2026-09-08")), TODAY) == []


def test_a_row_can_hold_a_tighter_max_age():
    doc = gated(metal=gate("2026-10-12"))
    doc["rows"][0]["max_age_days"] = 10
    assert mt.stale(doc, TODAY) == ["toy-q8_0"]


def test_a_failing_cell_on_any_backend_refuses():
    """Recent and red is worse than absent: a recorded failure on a backend
    the row is not gated on still says this file broke."""
    doc = gated(metal=gate("2026-10-22"), cpu=gate("2026-10-22", result="fail"))
    assert mt.stale(doc, TODAY) == ["toy-q8_0"]


def test_a_smoke_from_another_device_class_does_not_count():
    """Owner decision 3: the smoke runs on the class named in the row."""
    doc = gated(metal=gate("2026-10-22", device_class="linux-x86_64-cuda"))
    assert mt.stale(doc, TODAY) == ["toy-q8_0"]


def test_rows_below_tier1_are_not_gated():
    row = tier1_row(tier="tier2")
    row["transitions"][-1]["tier"] = "tier2"
    assert mt.stale(roster(row), TODAY) == []


def test_the_stale_report_prints_every_tier1_row(capsys):
    mt.stale(gated(metal=gate("2026-10-22")), TODAY)
    assert "toy-q8_0" in capsys.readouterr().out


def test_the_stale_summary_follows_the_rows_in_one_pipe(tmp_path):
    """check-release reads this check with stdout and stderr merged into
    one pipe; the per-row lines must come before the summary under them."""
    rpath = tmp_path / "tiers.json"
    dpath = tmp_path / "devices.json"
    mt.save(gated(), rpath)
    dpath.write_text(json.dumps(DEVICES), encoding="utf-8")
    r = subprocess.run([sys.executable, str(SCRIPT), "--stale", "--json", str(rpath),
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


# --- the README section: a render of the roster, never typed --------------------
#
# One collapsible block per Tier 1 model: a summary line (the bold name, its
# label in plain words, a status chip per backend), then a table with one
# row per roster file. The marks below are written out by hand from the
# owner's decisions of 2026-10-23: a recorded cell is the version and date
# of its gate, not run is a blank cell, a limit is n/a linked to the manual.

def readme_blocks(doc):
    """(summary, file rows) per model block of the render, in order; a
    file row is its list of cells."""
    out = []
    for block in re.findall(r"<details>\n(.*?)\n</details>", mt.render_readme(doc), re.S):
        summary = re.match(r"<summary>(.*)</summary>\n", block).group(1)
        rows = [[c.strip() for c in line.strip()[1:-1].split(" | ")]
                for line in block.splitlines()[4:] if line.startswith("| ")]
        out.append((summary, rows))
    return out


def file_row(doc, name):
    return [r for _, rows in readme_blocks(doc) for r in rows if name in r[0]][0]


def chips(summary):
    """The status chips of a summary line, backend name to status."""
    return dict(c.split("&nbsp;") for c in re.findall(r"<code>(.*?)</code>", summary))


def test_the_readme_block_is_the_render_of_the_roster():
    """The help-parity pattern: the README's section between the two markers
    is exactly what the roster renders, so it cannot drift from it.
    After --record, `scripts/model-tiers.py --write-docs` rewrites it."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.count(mt.README_BEGIN) == 1 and text.count(mt.README_END) == 1
    block = text.split(mt.README_BEGIN, 1)[1].split(mt.README_END, 1)[0]
    assert block == mt.render_readme(committed())


def test_a_recorded_cell_renders_its_version_and_date_and_not_run_is_blank():
    doc = gated(metal=gate("2026-10-23", version="1.5.1"))
    assert file_row(doc, "toy-Q8_0.gguf")[1:] == ["", "", "✅ 1.5.1<br>2026-10-23", ""]


def test_a_declared_limit_renders_n_a_linked_to_the_matrix():
    doc = gated(metal=gate("2026-10-23"),
                cuda={"kind": "limit", "reason": "no CUDA path",
                      "matrix": "MANUAL.md#support-matrix"})
    assert file_row(doc, "toy-Q8_0.gguf")[2] == "[n/a](MANUAL.md#support-matrix)"


def test_a_failed_gate_never_renders_a_check_mark():
    doc = gated(metal=gate("2026-10-23", result="fail"))
    assert file_row(doc, "toy-Q8_0.gguf")[3] == "failed 1.5.0<br>2026-10-23"
    (summary, _), = readme_blocks(doc)
    assert chips(summary)["Metal"] == "failed" and "✅" not in summary


def test_the_train_column_is_the_recorded_adapt_info_answer():
    doc = gated(metal=gate("2026-10-23"))
    doc["rows"][0]["lora"] = lora(train=(False, "MoE shared experts | router"))
    # a pipe in the refusal sentence would split the cell
    assert file_row(doc, "toy-Q8_0.gguf")[4] == "no: MoE shared experts \\| router"
    doc["rows"][0]["lora"] = lora()
    assert file_row(doc, "toy-Q8_0.gguf")[4] == "✅ 1.5.1<br>2026-10-23"


def test_every_tier1_file_gets_its_row_even_unrun():
    """Every roster file is one the owner committed to validate, so it is
    listed, its cells blank until a gate records them; one shape for every
    model, run or not."""
    (summary, rows), = readme_blocks(roster())
    assert rows == [["[`toy-Q8_0.gguf`](docs/model-support.md#toy-q8_0)", "", "", "", ""]]
    assert chips(summary) == {"CPU": "–", "CUDA": "–", "Metal": "–"}


def test_each_model_is_one_collapsed_block_github_renders_as_a_table():
    """GitHub renders markdown inside <details> only after a blank line,
    and the HTML block must end with one before </details>."""
    text = mt.render_readme(gated(metal=gate("2026-10-23")))
    block = text.split("<details>\n", 1)[1].split("</details>", 1)[0]
    lines = block.split("\n")
    assert lines[0].startswith("<summary><b>Toy</b>") and lines[0].endswith("</summary>")
    assert lines[1] == ""
    assert lines[2] == "| File | CPU | CUDA | Metal | Train |"
    assert lines[3] == "|---|---|---|---|---|"
    assert lines[-2:] == ["", ""]


def test_the_label_follows_the_bold_name_in_plain_words():
    doc = gated(metal=gate("2026-10-23"))
    doc["rows"][0].update(experimental=True, claim="Experimental: a toy.")
    assert readme_blocks(doc)[0][0].startswith("<b>Toy</b>, experimental <code>")
    doc["rows"][0].update(vision_upstream=True)
    assert readme_blocks(doc)[0][0].startswith(
        "<b>Toy</b>, experimental, text path only <code>")
    assert "Text path only:" not in mt.render_readme(doc)


def test_the_summary_is_html_so_a_model_name_is_escaped():
    doc = roster(tier1_row(model="Toy <R&D>"))
    assert readme_blocks(doc)[0][0].startswith("<b>Toy &lt;R&amp;D&gt;</b> <code>")


def test_the_file_name_links_its_envelope_and_a_sharded_set_counts_its_parts():
    row = tier1_row()
    row["files"] = [{"name": f"toy-0000{i}-of-00003.gguf", "sha256": SHA_A, "bytes": 1}
                    for i in (1, 2, 3)]
    (_, rows), = readme_blocks(roster(row))
    assert rows[0][0] == ("[`toy-00001-of-00003.gguf`](docs/model-support.md#toy-q8_0)"
                          " (3 parts)")


def test_the_summary_sums_up_each_backend_over_the_models_files():
    """A model's chip per backend, written out by hand for files in mixed
    states: n/a when every file has the limit, failed when any recorded a
    failure, ✅ when every file without the limit is recorded, – when none
    is, partial otherwise. A limit never counts as a missing run."""
    limit = {"kind": "limit", "reason": "no path", "matrix": "MANUAL.md#support-matrix"}
    a = tier1_row(id="mix-a", model="Mix")
    b = tier1_row(id="mix-b", model="Mix")
    c = tier1_row(id="mix-c", model="Mix")
    for r, name in ((b, "mix-b.gguf"), (c, "mix-c.gguf")):
        r["files"] = [{"name": name, "sha256": "e" * 64, "bytes": 1}]
    a["backends"] = {"cpu": gate("2026-10-23"), "cuda": gate("2026-10-23"), "metal": limit}
    b["backends"] = {"cpu": gate("2026-10-23"), "cuda": None, "metal": limit}
    c["backends"] = {"cpu": limit, "cuda": None, "metal": limit}
    d = tier1_row(id="odd-a", model="Odd")
    e = tier1_row(id="odd-b", model="Odd")
    e["files"] = [{"name": "odd-b.gguf", "sha256": "f" * 64, "bytes": 1}]
    d["backends"] = {"cpu": gate("2026-10-23"), "cuda": limit, "metal": None}
    e["backends"] = {"cpu": gate("2026-10-23", result="fail"), "cuda": None, "metal": None}
    blocks = readme_blocks(roster(a, b, c, d, e))
    assert [chips(s) for s, _ in blocks] == [
        {"CPU": "✅", "CUDA": "partial", "Metal": "n/a"},
        {"CPU": "failed", "CUDA": "–", "Metal": "–"},
    ]
    assert [len(rows) for _, rows in blocks] == [3, 2]


def test_models_are_sorted_by_name_and_files_keep_the_roster_order():
    """The render does not depend on the order rows were added in: model
    blocks sort by name (the label aside), files within one keep their
    roster order."""
    zeta = tier1_row(id="zeta-a", model="Zeta")
    zeta2 = tier1_row(id="zeta-b", model="Zeta")
    zeta2["files"] = [{"name": "zeta-b.gguf", "sha256": "e" * 64, "bytes": 1}]
    alpha = tier1_row(id="alpha", model="alpha", experimental=True,
                      claim="Experimental: a toy.")
    blocks = readme_blocks(roster(zeta, zeta2, alpha))
    assert [s.split("</b>")[0] for s, _ in blocks] == ["<b>alpha", "<b>Zeta"]
    assert [r[0].split("`")[1] for r in blocks[1][1]] == ["toy-Q8_0.gguf", "zeta-b.gguf"]
    assert mt.render_readme(roster(alpha, zeta, zeta2)) == \
        mt.render_readme(roster(zeta, zeta2, alpha))


def test_a_declared_limit_links_the_manuals_support_matrix():
    """Every n/a points at the same place a reader checks a limit: the
    manual's support matrix. Other evidence belongs in the row's envelope."""
    row = tier1_row()
    row["backends"]["cuda"] = {"kind": "limit", "reason": "no CUDA path",
                               "matrix": "tests/compatibility/coverage.json"}
    assert any("matrix" in e for e in errors(roster(row)))


def test_the_legend_comes_first_and_names_every_mark():
    text = mt.render_readme(roster())
    legend = text.strip().split("\n")[0]
    assert text.index(legend) < text.index("<details>")
    for word in ("✅", "blank", "n/a", "failed", "`--adapt-info`", "partial", "–", "envelope"):
        assert word in legend, word


def test_upcoming_models_are_one_line_under_the_blocks_and_tier2_is_absent():
    up = {"id": "later", "model": "Later VL", "tier": "upcoming",
          "claim": "Text path only: not admitted.", "architecture": "later",
          "missing": "an admission", "vision_upstream": True,
          "transitions": [{"tier": "upcoming", "effective": "2026-10-22",
                           "reason": "test", "last_validated_release": None}]}
    up2 = dict(up, id="sooner", model="Sooner", vision_upstream=False)
    t2 = tier1_row(id="old", model="Old", tier="tier2")
    t2["transitions"][-1]["tier"] = "tier2"
    text = mt.render_readme(roster(tier1_row(), t2, up, up2))
    tail = text.rsplit("</details>", 1)[1]
    assert ("**Upcoming** (planned, not supported yet): Later VL (text path only), "
            "Sooner.") in tail.splitlines()
    assert "Later" not in text.rsplit("</details>", 1)[0]
    assert "Old" not in text


# --- docs/model-support.md: envelopes and the transition log, rendered ----------

def test_the_support_doc_block_is_the_render_of_the_roster():
    text = (ROOT / "docs" / "model-support.md").read_text(encoding="utf-8")
    assert text.count(mt.DOC_BEGIN) == 1 and text.count(mt.DOC_END) == 1
    block = text.split(mt.DOC_BEGIN, 1)[1].split(mt.DOC_END, 1)[0]
    assert block == mt.render_doc(committed())


def test_every_envelope_link_in_the_readme_lands_on_a_row():
    import re
    doc = committed()
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', mt.render_doc(doc)))
    links = set(re.findall(r"docs/model-support\.md#([A-Za-z0-9._-]+)",
                           mt.render_readme(doc)))
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
