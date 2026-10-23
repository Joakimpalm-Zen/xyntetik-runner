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
        "transition": {"tier": "tier1", "effective": "2026-10-22",
                       "reason": "test", "last_validated_release": None},
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
    row["transition"]["tier"] = "tier2"
    assert any("published" in e for e in errors(roster(row)))
    row["published_exception"] = "card leads with its measured shortfall"
    assert errors(roster(row)) == []


def test_the_latest_transition_must_lead_to_the_rows_tier():
    row = tier1_row()
    row["transition"]["tier"] = "tier2"
    assert any("transition" in e for e in errors(roster(row)))


def test_an_upcoming_row_says_what_is_missing():
    row = {"id": "later", "model": "Later", "tier": "upcoming",
           "claim": "Not admitted.", "architecture": "later", "missing": "",
           "transition": {"tier": "upcoming", "effective": "2026-10-22",
                          "reason": "test", "last_validated_release": None}}
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


def test_a_head_files_trunk_is_a_tier1_row():
    """An MTP head runs beside a trunk; the trunk it is gated with must be
    a row the roster validates, not a name."""
    assert any("smoke_trunk" in e for e in
               errors(roster(tier1_row(smoke_trunk="no-such-row"))))
    head = tier1_row(id="toy-head", smoke_trunk="toy-q8_0")
    assert errors(roster(tier1_row(), head)) == []


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
                 "cpu_cores": 8, "gpu_backend": "metal", "gpu_name": "Apple M1"},
        "banner": "gpu: Metal backend on Apple M1 (zero-copy weights)",
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
    row["transition"]["tier"] = "tier2"
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
    row["transition"]["tier"] = "tier2"
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
