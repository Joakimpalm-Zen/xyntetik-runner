"""runner recommend (--info recommend): the listed model files that fit this machine (R10.25).

The rule is Dojo's (tools/runner-start.sh), so the two never disagree: a listed text file, one .gguf,
no larger than the RAM, fits when its resident weights are within a share of the RAM (half by
default). Resident means the whole file, or the fit report's hot set for a file whose token table is
read by row (`resident_bytes` in the roster). The command reads no heads and makes no network
request: a listing command stays offline.

Absolute anchor: Dojo's own implementation (tools/runner-start.sh, which fetches each file's first
16 MiB from the Hub and asks `runner --fit`) listed these 16 files on an 8 GiB machine at half its RAM,
largest resident first, on 2026-11-09, with R10.28's fix (a file in a repository folder fetched by its
path). The committed roster, read offline, must give the same list.
"""
import json
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GIB = 1 << 30
DOJO_8GIB = ["qwen3-4b-runner-tooluse-merged-q8_0", "phi-3.5-mini-instruct-q8_0",
             "ministral-3-3b-instruct-2512-q8_0", "spark-x2.6-4b-q4_k_m",
             "nemotron-3.5-bolt-e4b-q4_k_m", "minicpm5-2b-q8_0", "spark-x2.5-4b-q4_k_m",
             "qwen3-4b-q4_k_m-bartowski", "phi-3.5-mini-instruct-q4_k_m",
             "ministral-3-3b-instruct-2512-q4_k_m", "qwen2.5-1.5b-instruct-q8_0",
             "smollm2-1.7b-instruct-q8_0", "qwen2.5-1.5b-instruct-q8_0-bartowski",
             "minicpm5-2b-q4_k_m", "gemma-4-e2b-it-q4_k_m-q4_0-mix", "lfm2.5-1.2b-instruct-q8_0",
             "k2-horizon-0.9b-q8_0", "smollm2-1.7b-instruct-q4_k_m", "lfm2.5-1.2b-instruct-q4_k_m",
             "k2-horizon-0.9b-q4_k_m", "lfm2.5-350m-q8_0", "lfm2.5-350m-q4_k_m", "smollm2-135m-instruct-q8_0"]


def _file(name, gib, **extra):
    return {"name": name, "repo": "org/" + name.split(".")[0], "bytes": int(gib * GIB),
            "sha256": "0" * 64, **extra}


def _row(rid, files, tier="tier2", features=("text",), ref="measured",
         anchor="This file against the publisher's float32 reference. More detail follows."):
    return {"id": rid, "model": rid, "tier": tier, "claim": "c", "architecture": "llama",
            "transitions": [], "files": files,
            "envelope": {"features": list(features), "fp32_reference": ref, "anchor": anchor}}


ROSTER = {"schema": "xyntetik.runner.model-tiers.v1", "rows": [
    _row("small-measured", [_file("small.gguf", 1.0)]),
    _row("mid-owed", [_file("mid.gguf", 3.0)], ref="owed"),
    # a 6 GiB file whose token table is read by row: 2.5 GiB stay resident
    _row("gathered", [_file("gathered.gguf", 6.0, resident_bytes=int(2.5 * GIB))], tier="tier1"),
    _row("big", [_file("big.gguf", 7.0)]),
    _row("split", [_file("split-00001-of-00002.gguf", 1.6), _file("split-00002-of-00002.gguf", 1.6)]),
    _row("planned", [_file("planned.gguf", 0.5)], tier="upcoming"),
    _row("vision-only", [_file("vis.gguf", 0.5)], features=("vision",)),
    _row("nested", [_file("sub/nested.gguf", 0.5)]),
    _row("in-a-folder", [_file("folder.gguf", 0.5, path="merged/folder.gguf")]),
    _row("not-gguf", [_file("weights.safetensors", 0.5)]),
    {"id": "no-files", "model": "m", "tier": "tier2", "claim": "c", "architecture": "llama",
     "transitions": []},
]}


@pytest.fixture()
def roster(tmp_path):
    p = tmp_path / "roster.json"
    p.write_text(json.dumps(ROSTER))
    return p


def _recommend(runner_bin, *args, sub=False):
    cmd = [runner_bin, "recommend", *args] if sub else [runner_bin, "--info", "recommend", *args]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    return p.returncode, p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace")


def _entries(out):
    """id -> the entry's text: its line ("  <GiB> GiB  <id>  ...") and the lines indented below it."""
    got, cur = {}, None
    for ln in out.splitlines():
        w = ln.split()
        if ln.startswith("  ") and not ln.startswith("   ") and len(w) > 2 and w[1] == "GiB":
            cur = w[2]
            got[cur] = ln
        elif cur and ln.startswith("    "):
            got[cur] += "\n" + ln
        else:
            cur = None
    return got


def _ids(out):
    return list(_entries(out))


def test_a_memory_cap_changes_the_list(runner_bin, roster):
    rc, out8, err = _recommend(runner_bin, "--roster", str(roster), "--memory", "8")
    assert rc == 0, err
    # 4 GiB is half of 8: the 7 GiB file is over it; the row-gathered one counts its hot set
    assert _ids(out8) == ["split", "mid-owed", "gathered", "small-measured", "in-a-folder"], out8
    rc, out4, err = _recommend(runner_bin, "--roster", str(roster), "--memory", "4")
    assert rc == 0, err
    # 2 GiB: only the 1 and 0.5 GiB files are left; the 6 and 7 GiB files are larger than the RAM
    assert _ids(out4) == ["small-measured", "in-a-folder"], out4
    assert "of 8.00 GiB" in out8 and "of 4.00 GiB" in out4


def test_the_share_moves_the_bar(runner_bin, roster):
    rc, out, err = _recommend(runner_bin, "--roster", str(roster), "--memory", "8",
                              "--ram-share", "1")
    assert rc == 0, err
    assert _ids(out)[0] == "big", out
    rc, _, err = _recommend(runner_bin, "--roster", str(roster), "--ram-share", "2")
    assert rc != 0 and "--ram-share" in err


def test_what_the_entries_say(runner_bin, roster):
    rc, out, err = _recommend(runner_bin, "--roster", str(roster), "--memory", "8")
    assert rc == 0, err
    for gone in ("planned", "vision-only", "nested", "not-gguf", "no-files", "big"):
        assert gone not in _ids(out), (gone, out)
    e = _entries(out)
    # R10.28: a file in a folder of its repository is listed, by the spelling -hf resolves
    assert "-hf org/folder:folder" in e["in-a-folder"], e["in-a-folder"]
    owed = e["mid-owed"]
    assert "unmeasured" in owed, owed
    measured = e["small-measured"]
    # the anchor's first sentence, not the rest of it
    assert "measured: This file against the publisher's float32 reference." in measured
    assert "More detail" not in measured
    gathered = e["gathered"]
    assert "hot set" in gathered and "6.00 GiB file" in gathered, gathered
    assert "-hf org/small:small" in measured
    assert "docs/model-support.md#small-measured" in measured
    assert "tier" not in out.lower()


def test_the_subcommand_is_the_mode(runner_bin, roster):
    a = _recommend(runner_bin, "--roster", str(roster), "--memory", "8")
    b = _recommend(runner_bin, "--roster", str(roster), "--memory", "8", sub=True)
    assert a == b


def test_the_embedded_roster_gives_dojos_list(runner_bin):
    """Without --roster the binary reads its own snapshot: at 8 GiB the list Dojo's run gave, with
    the rows admitted since, and the same entries --roster reads from the committed file."""
    rc, out, err = _recommend(runner_bin, "--memory", "8")
    assert rc == 0, err
    assert _ids(out) == DOJO_8GIB, out
    rc2, out2, _ = _recommend(runner_bin, "--memory", "8",
                              "--roster", str(ROOT / "docs" / "model-tiers.json"))
    assert rc2 == 0 and _ids(out2) == _ids(out), (out, out2)


def test_a_roster_that_does_not_parse_is_refused(runner_bin, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")
    rc, _, err = _recommend(runner_bin, "--roster", str(bad))
    assert rc != 0 and "roster" in err
