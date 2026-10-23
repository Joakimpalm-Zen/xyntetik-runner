"""scripts/tier-smoke.py, the per-release smoke over the model roster.

The script is driven through its command line. The runner it calls is a
stand-in process (tests/fixtures/tier_smoke_fake_runner.py) that answers
--caps, --adapt-info and a one-shot prompt the way the real binary does, so
these tests hold the smoke's own decisions (hash the file first, pin once,
compare bytes, require the backend's banner, the outside fact and a pin long
enough to catch a change) without a
model on the CI host. The real binary's run on a real Tier 1 file is the
committed pin and the roster cells it recorded.

Absolute anchors: the sha256 of the fixture file is computed here with
hashlib, outside the code under test, and the fact the prompt asks for
("Mercury" is the planet closest to the Sun) is a fact about the world, not
about Runner.
"""

import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE = ROOT / "scripts" / "tier-smoke.py"
FAKE = ROOT / "tests" / "fixtures" / "tier_smoke_fake_runner.py"

spec = importlib.util.spec_from_file_location("model_tiers", ROOT / "scripts" / "model-tiers.py")
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

DEVICES = {
    "schema": "xyntetik.runner.device-evidence.v1",
    "policy": {"default_max_age_days": 45},
    "classes": [{"id": "macos-arm64-metal",
                 "expect": {"os": "macos", "arch": "arm64", "gpu_backend": "metal"}}],
}


PROMPT = ("List the eight planets of the Solar System in order from the Sun, "
          "one per line, then describe each in one sentence.")
# what the stand-in runner continues PROMPT with unless told otherwise
LONG = ("\n\n1. Mercury: the smallest planet and the closest to the Sun.\n"
        "2. Venus: the second planet from the Sun and the hottest.")


@pytest.fixture
def world(tmp_path):
    model = tmp_path / "toy-Q8_0.gguf"
    model.write_bytes(b"GGUF toy weights, not a model")
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    roster = {
        "schema": mt.SCHEMA,
        "policy": {"default_max_age_days": 45, "backends": ["cpu", "cuda", "metal"]},
        "exempt": [],
        "rows": [{
            "id": "toy-q8_0", "model": "Toy", "tier": "tier1",
            "claim": "A toy Tier 1 row.", "architecture": "llama",
            "upstream": {"checkpoint": "toy/toy", "revision": None},
            "files": [{"name": "toy-Q8_0.gguf", "sha256": sha,
                       "bytes": model.stat().st_size}],
            "envelope": {"features": ["text"], "fp32_reference": "measured",
                         "anchor": "toy", "evidence": ["docs/model-evidence.md"]},
            "smoke": {"device_class": "macos-arm64-metal", "backend": "metal"},
            "backends": {"cpu": None, "cuda": None, "metal": None},
            "lora": None,
            "transitions": [{"tier": "tier1", "effective": "2026-10-22",
                           "reason": "test", "last_validated_release": None}],
        }],
    }
    paths = {
        "roster": tmp_path / "model-tiers.json",
        "devices": tmp_path / "device-evidence.json",
        "manifest": tmp_path / "tier-smoke.json",
        "pins": tmp_path / "pins",
        "out": tmp_path / "result.json",
        "model": model,
    }
    mt.save(roster, paths["roster"])
    paths["devices"].write_text(json.dumps(DEVICES), encoding="utf-8")
    paths["manifest"].write_text(json.dumps({
        "schema": "xyntetik.runner.tier-smoke.v1",
        "defaults": {"prompt": PROMPT, "n_predict": 48, "expect": "Mercury",
                     "min_generated_bytes": 96,
                     "args": ["--temp", "0", "--no-tray"]},
        "rows": {},
    }), encoding="utf-8")
    return paths


def smoke(world, *extra, env=None):
    cmd = [sys.executable, str(SMOKE), "--runner", str(FAKE),
           "--roster", str(world["roster"]),
           "--device-evidence", str(world["devices"]),
           "--manifest", str(world["manifest"]), "--pins", str(world["pins"]),
           "--row", "toy-q8_0", "--model", str(world["model"]),
           "--out", str(world["out"])] + list(extra)
    e = dict(os.environ, **(env or {}))
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, env=e)


def result(world):
    return json.loads(world["out"].read_text(encoding="utf-8"))


def test_the_first_run_pins_and_the_second_compares(world):
    r = smoke(world, "--backend", "metal", "--pin")
    assert r.returncode == 0, r.stdout
    assert result(world)["status"] == "pinned"
    pin = world["pins"] / "toy-q8_0.macos-arm64-metal.metal.txt"
    # the one-shot CLI echoes the prompt, then the continuation
    assert pin.read_bytes().startswith(PROMPT.encode())
    r = smoke(world, "--backend", "metal")
    assert r.returncode == 0, r.stdout
    res = result(world)
    assert res["status"] == "match"
    assert res["expect"] == {"substring": "Mercury", "found": True}
    assert res["output"]["sha256"] == hashlib.sha256(pin.read_bytes()).hexdigest()
    assert res["banner"].startswith("gpu: Metal backend on")
    assert res["adapt_info"]["training"]["supported"] is True


def test_record_writes_the_cell_and_the_lora_answer(world):
    assert smoke(world, "--backend", "metal", "--pin").returncode == 0
    r = smoke(world, "--backend", "metal", "--record")
    assert r.returncode == 0, r.stdout
    row = mt.load(world["roster"])["rows"][0]
    assert row["backends"]["metal"]["result"] == "pass"
    assert row["backends"]["metal"]["runner_version"] == "9.8.7"
    assert row["lora"]["training"] == {"supported": True, "reason": None}
    assert row["backends"]["cpu"] is None


def test_a_cpu_run_is_pinned_and_recorded_on_its_own(world):
    """A backend's pin is its own: the CPU never inherits the Metal pin."""
    assert smoke(world, "--backend", "metal", "--pin").returncode == 0
    r = smoke(world, "--backend", "cpu")
    assert r.returncode != 0
    assert "no pin" in r.stdout
    assert smoke(world, "--backend", "cpu", "--pin").returncode == 0
    assert smoke(world, "--backend", "cpu", "--record").returncode == 0
    row = mt.load(world["roster"])["rows"][0]
    assert row["backends"]["cpu"]["result"] == "pass"
    assert row["backends"]["metal"] is None


def test_a_changed_output_is_a_mismatch_and_records_a_fail(world):
    assert smoke(world, "--backend", "metal", "--pin").returncode == 0
    r = smoke(world, "--backend", "metal", "--record",
              env={"FAKE_RUNNER_TEXT": LONG.replace("closest", "nearest")})
    assert r.returncode != 0
    assert result(world)["status"] == "mismatch"
    assert mt.load(world["roster"])["rows"][0]["backends"]["metal"]["result"] == "fail"


def test_a_file_that_is_not_the_rows_is_refused_before_it_runs(world):
    world["model"].write_bytes(b"some other file")
    r = smoke(world, "--backend", "metal", "--pin")
    assert r.returncode != 0
    assert "sha256" in r.stdout
    assert not (world["pins"] / "toy-q8_0.macos-arm64-metal.metal.txt").exists()


def test_an_existing_pin_is_never_overwritten(world):
    """Re-pinning is a decision with a reason in a commit: delete the pin
    and pin again. The script does not do it quietly."""
    assert smoke(world, "--backend", "metal", "--pin").returncode == 0
    r = smoke(world, "--backend", "metal", "--pin")
    assert r.returncode != 0
    assert "already pinned" in r.stdout


def test_a_metal_run_that_fell_back_to_the_cpu_cannot_record(world):
    assert smoke(world, "--backend", "metal", "--pin").returncode == 0
    r = smoke(world, "--backend", "metal", "--record",
              env={"FAKE_RUNNER_FALLBACK": "1"})
    assert r.returncode != 0
    assert "banner" in r.stdout
    assert mt.load(world["roster"])["rows"][0]["backends"]["metal"] is None


def test_the_pin_manifest_records_what_was_pinned(world):
    assert smoke(world, "--backend", "metal", "--pin").returncode == 0
    man = json.loads(world["manifest"].read_text(encoding="utf-8"))
    pin = man["rows"]["toy-q8_0"]["pins"]["macos-arm64-metal/metal"]
    data = (world["pins"] / "toy-q8_0.macos-arm64-metal.metal.txt").read_bytes()
    assert pin["sha256"] == hashlib.sha256(data).hexdigest()
    assert pin["bytes"] == len(data)
    assert pin["pinned"]["runner_version"] == "9.8.7"


def manifest_of(world):
    return json.loads(world["manifest"].read_text(encoding="utf-8"))


def test_a_pin_shorter_than_the_minimum_is_refused(world):
    """A pin of a two-token answer catches almost no change: the first E2B
    pins were 7 generated bytes (" Paris." and end of text)."""
    r = smoke(world, "--backend", "metal", "--pin",
              env={"FAKE_RUNNER_TEXT": " Mercury."})
    assert r.returncode != 0
    assert "toy-q8_0" in r.stdout and "9 bytes" in r.stdout and "96" in r.stdout
    assert not (world["pins"] / "toy-q8_0.macos-arm64-metal.metal.txt").exists()
    assert "toy-q8_0" not in manifest_of(world)["rows"]


def test_a_row_can_set_its_own_minimum(world):
    man = manifest_of(world)
    man["rows"]["toy-q8_0"] = {"min_generated_bytes": 8}
    world["manifest"].write_text(json.dumps(man), encoding="utf-8")
    r = smoke(world, "--backend", "metal", "--pin",
              env={"FAKE_RUNNER_TEXT": " Mercury."})
    assert r.returncode == 0, r.stdout


def test_a_pin_under_the_minimum_is_not_compared_against(world):
    """Raising the minimum retires the short pins it no longer accepts: the
    compare refuses them instead of passing on them."""
    man = manifest_of(world)
    man["rows"]["toy-q8_0"] = {"min_generated_bytes": 8}
    world["manifest"].write_text(json.dumps(man), encoding="utf-8")
    assert smoke(world, "--backend", "metal", "--pin",
                 env={"FAKE_RUNNER_TEXT": " Mercury."}).returncode == 0
    man = manifest_of(world)
    del man["rows"]["toy-q8_0"]["min_generated_bytes"]
    world["manifest"].write_text(json.dumps(man), encoding="utf-8")
    r = smoke(world, "--backend", "metal", env={"FAKE_RUNNER_TEXT": " Mercury."})
    assert r.returncode != 0
    assert "retire" in r.stdout


def test_retire_takes_the_pin_its_entry_and_the_cell_that_cites_it(world):
    """Re-pinning after an intended token change: the pin, its manifest
    entry and the roster cell whose evidence it is go together, so the roster
    never cites a pin that is gone and no cell is edited by hand."""
    assert smoke(world, "--backend", "metal", "--pin").returncode == 0
    assert smoke(world, "--backend", "metal", "--record").returncode == 0
    r = subprocess.run([sys.executable, str(SMOKE), "--retire",
                        "--roster", str(world["roster"]),
                        "--device-evidence", str(world["devices"]),
                        "--manifest", str(world["manifest"]),
                        "--pins", str(world["pins"]), "--row", "toy-q8_0",
                        "--backend", "metal", "--device-class", "macos-arm64-metal"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert r.returncode == 0, r.stdout
    assert not (world["pins"] / "toy-q8_0.macos-arm64-metal.metal.txt").exists()
    assert "macos-arm64-metal/metal" not in \
        manifest_of(world).get("rows", {}).get("toy-q8_0", {}).get("pins", {})
    row = mt.load(world["roster"])["rows"][0]
    assert row["backends"]["metal"] is None
    # the file's --adapt-info answer is not the pin's evidence and stays
    assert row["lora"]["training"]["supported"] is True
    assert mt.validate(mt.load(world["roster"]), DEVICES) == []
    assert smoke(world, "--backend", "metal", "--pin").returncode == 0


def test_a_pin_made_with_another_prompt_is_not_compared_against(world):
    """A pin holds what one prompt produced. After the manifest's prompt or
    settings change, comparing a new run with it would report a mismatch
    whose cause is the manifest, so the compare refuses and names it."""
    assert smoke(world, "--backend", "metal", "--pin").returncode == 0
    pin = manifest_of(world)["rows"]["toy-q8_0"]["pins"]["macos-arm64-metal/metal"]
    assert pin["spec"]["prompt"] == PROMPT and pin["spec"]["n_predict"] == 48
    man = manifest_of(world)
    man["defaults"]["n_predict"] = 64
    world["manifest"].write_text(json.dumps(man), encoding="utf-8")
    r = smoke(world, "--backend", "metal")
    assert r.returncode != 0
    assert "different" in r.stdout


# --- a companion runs with the row it names -----------------------------------

def add_companion(world, kind, how, name):
    comp = world["model"].parent / name
    comp.write_bytes(b"GGUF companion bytes for " + kind.encode())
    doc = mt.load(world["roster"])
    row = json.loads(json.dumps(doc["rows"][0]))
    row.update(id="toy-" + kind, file_kind=kind,
               runs_with={"row": "toy-q8_0", "as": how},
               files=[{"name": name, "sha256": hashlib.sha256(comp.read_bytes()).hexdigest(),
                       "bytes": comp.stat().st_size}])
    doc["rows"].append(row)
    mt.save(doc, world["roster"])
    return comp


def companion_smoke(world, comp, row, *extra, env=None):
    cmd = [sys.executable, str(SMOKE), "--runner", str(FAKE),
           "--roster", str(world["roster"]),
           "--device-evidence", str(world["devices"]),
           "--manifest", str(world["manifest"]), "--pins", str(world["pins"]),
           "--row", row, "--model", str(comp), "--with-model", str(world["model"]),
           "--backend", "metal", "--out", str(world["out"])] + list(extra)
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, env=dict(os.environ, **(env or {})))


def test_a_head_runs_beside_its_trunk_and_pins_under_its_own_row(world):
    head = add_companion(world, "mtp-head", "mtp-file", "toy-mtp.gguf")
    r = companion_smoke(world, head, "toy-mtp-head", "--pin")
    assert r.returncode == 0, r.stdout
    res = result(world)
    assert "-m toy-Q8_0.gguf" in res["ran"] and "--mtp-file toy-mtp.gguf" in res["ran"]
    assert res["with"]["row"] == "toy-q8_0"
    assert (world["pins"] / "toy-mtp-head.macos-arm64-metal.metal.txt").exists()
    assert companion_smoke(world, head, "toy-mtp-head", "--record").returncode == 0
    rows = {r["id"]: r for r in mt.load(world["roster"])["rows"]}
    assert rows["toy-mtp-head"]["backends"]["metal"]["result"] == "pass"
    assert rows["toy-q8_0"]["backends"]["metal"] is None


def test_an_adapter_runs_on_its_base(world):
    lora = add_companion(world, "adapter", "lora", "toy-lora.gguf")
    r = companion_smoke(world, lora, "toy-adapter", "--pin")
    assert r.returncode == 0, r.stdout
    assert "--lora toy-lora.gguf" in result(world)["ran"]
    # the stand-in marks a run that applied an adapter, so this is the
    # composed run's output and not the base's
    pin = world["pins"] / "toy-adapter.macos-arm64-metal.metal.txt"
    assert b"(adapter applied)" in pin.read_bytes()


def test_a_trunk_that_is_not_the_named_rows_is_refused(world):
    head = add_companion(world, "mtp-head", "mtp-file", "toy-mtp.gguf")
    world["model"].write_bytes(b"another trunk")
    r = companion_smoke(world, head, "toy-mtp-head", "--pin")
    assert r.returncode != 0
    assert "sha256" in r.stdout
