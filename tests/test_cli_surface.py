"""The shape of the command line: one verb per run, the merged spellings, the
subcommands, and the defaults a mode implies.

Every case here goes through the public interface (argv in, exit status and
stderr out). The anchor is the documented contract in MANUAL.md's
command-line reference: a refusal names the two verbs or the flag it is
about, an older spelling behaves exactly as the spelling that replaced it,
and a subcommand is a rewrite of the flags it stands for, so the flags'
own tests keep covering what runs behind it.

Most cases use a model path that does not exist. A refusal that still wins
proves it ran BEFORE the load; a run that reaches "cannot open" proves the
line was accepted.
"""
import json
import re
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "test.gguf"


def run(runner_bin, *args, stdin=b""):
    return subprocess.run([str(runner_bin), *args], cwd=ROOT, input=stdin,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=120)


def refused(runner_bin, *args):
    p = run(runner_bin, *args)
    err = p.stderr.decode(errors="replace")
    assert p.returncode != 0, err[-400:]
    assert "cannot open" not in err, err[-400:]
    return err


def accepted_up_to_the_load(runner_bin, *args):
    """The line parsed and passed every argument check: the only thing left
    to fail was opening the model that is not there."""
    p = run(runner_bin, "-m", "no-such-model.gguf", *args)
    err = p.stderr.decode(errors="replace")
    assert p.returncode != 0
    assert "cannot open" in err or "no such file" in err.lower(), err[-400:]
    return err


# ----------------------------------------------------------------- one verb


@pytest.mark.parametrize("a,b", [
    (["-p", "hi"], ["--serve"]),
    (["-i"], ["--serve"]),
    (["--score", "-p", "hi"], ["--decide", "x.jsonl"]),
    (["--info", "doctor"], ["--bench-json"]),
    (["--rewrite", "out.gguf"], ["--serve"]),
    (["--verify", "r.json"], ["--train", "d.txt"]),
])
def test_two_verbs_are_refused_by_name(runner_bin, a, b):
    """Two modes on one line used to resolve by the order main() dispatches
    them: `-p hi --serve` started the server and dropped the prompt."""
    err = refused(runner_bin, "-m", "no-such-model.gguf", *a, *b, "--no-tray")
    assert "one mode per run" in err, err[-400:]
    assert a[0] in err and b[0] in err, err[-400:]


@pytest.mark.parametrize("verb", [
    ["--serve"], ["--info", "tools"], ["--decide", "x.jsonl"],
    ["--verify", "r.json"], ["--train", "d.txt"],
])
def test_a_prompt_beside_a_verb_that_never_reads_it_is_refused(runner_bin, verb):
    err = refused(runner_bin, "-m", "no-such-model.gguf", "-p", "hi", *verb,
                  "--no-tray")
    assert "-p" in err and verb[0] in err, err[-400:]


def test_a_prompt_file_that_is_missing_is_still_reported_as_such(runner_bin):
    """The prompt-versus-verb rule runs after the inputs resolve, so the
    older, more useful refusal keeps its place."""
    p = run(runner_bin, "-m", "no-such-model.gguf", "-i", "-f",
            "no-such-prompt.txt", "--no-tray")
    assert b"cannot read no-such-prompt.txt" in p.stderr


# ------------------------------------------------------------ merged flags


def test_json_takes_an_optional_schema_file(runner_bin, tmp_path):
    schema = tmp_path / "s.json"
    schema.write_text('{"type":"object"}')
    accepted_up_to_the_load(runner_bin, "-p", "x", "--json", str(schema))
    accepted_up_to_the_load(runner_bin, "-p", "x", "--json")
    accepted_up_to_the_load(runner_bin, "-p", "x", "--json-schema", str(schema))


@pytest.mark.parametrize("args", [
    ["--think"], ["--think", "on"], ["--think", "off"], ["--no-think"],
])
def test_think_values_are_chat_only_like_before(runner_bin, args):
    err = refused(runner_bin, "-m", "no-such-model.gguf", "-p", "x", *args)
    assert "--think" in err or "--no-think" in err


def test_think_auto_is_the_default_and_valid_anywhere(runner_bin):
    accepted_up_to_the_load(runner_bin, "-p", "x", "--think", "auto")


def test_tray_off_is_the_old_no_tray(runner_bin):
    accepted_up_to_the_load(runner_bin, "-p", "x", "--tray", "off")
    accepted_up_to_the_load(runner_bin, "-p", "x", "--tray", "auto")
    err = refused(runner_bin, "-m", "no-such-model.gguf", "-p", "x", "--tray",
                  "sometimes")
    assert "--tray" in err


def test_gpu_takes_a_layer_count(runner_bin):
    accepted_up_to_the_load(runner_bin, "-p", "x", "--gpu", "12")
    accepted_up_to_the_load(runner_bin, "-p", "x", "--gpu", "0")
    accepted_up_to_the_load(runner_bin, "-p", "x", "--gpu-layers", "12")
    err = refused(runner_bin, "-m", "no-such-model.gguf", "-p", "x", "--gpu",
                  "most")
    assert "--gpu expects auto, off or a layer count" in err


@pytest.mark.parametrize("spec", ["50", "vram=40,ram=60", "cpu=50", "ram=30,cpu=25"])
def test_reserve_takes_one_percentage_or_named_budgets(runner_bin, spec):
    accepted_up_to_the_load(runner_bin, "-p", "x", "--reserve", spec)


@pytest.mark.parametrize("spec", ["vram=", "disk=10", "ram=200", "40,50"])
def test_reserve_refuses_what_is_not_a_budget(runner_bin, spec):
    err = refused(runner_bin, "-m", "no-such-model.gguf", "-p", "x",
                  "--reserve", spec)
    assert "--reserve" in err


def test_record_is_the_transcript_in_one_shot(runner_bin, tmp_path):
    if not FIXTURE.exists():
        pytest.skip("fixture test.gguf not built")
    out = tmp_path / "run.json"
    p = run(runner_bin, "-m", str(FIXTURE), "-p", "hi", "-n", "2", "-s", "1",
            "--record", str(out), "--gpu", "off", "--no-tray")
    assert p.returncode == 0, p.stderr.decode(errors="replace")[-400:]
    rec = json.loads(out.read_text())
    assert rec["schema_version"] == "xyntetik.runner.transcript.v1"
    v = run(runner_bin, "-m", str(FIXTURE), "--verify", str(out), "--gpu",
            "off", "--no-tray")
    assert v.returncode == 0, v.stderr.decode(errors="replace")[-400:]


def test_record_is_the_receipts_directory_when_serving(runner_bin, tmp_path):
    """The receipts store announces itself on stderr when it is configured,
    which happens before the model is opened: that line naming the --record
    path proves --record went to the receipts side under --serve."""
    d = tmp_path / "receipts"
    err = accepted_up_to_the_load(runner_bin, "--serve", "--record", str(d),
                                  "--no-tray")
    assert "receipts: " + str(d) in err, err[-400:]


def test_record_keep_needs_a_record_directory(runner_bin):
    err = refused(runner_bin, "-m", "no-such-model.gguf", "--serve",
                  "--record-keep", "3", "--no-tray")
    assert "--record-keep" in err


def test_score_takes_an_optional_top_n(runner_bin):
    accepted_up_to_the_load(runner_bin, "-p", "x", "--score", "5")
    accepted_up_to_the_load(runner_bin, "-p", "x", "--score")
    accepted_up_to_the_load(runner_bin, "-p", "x", "--score", "--score-top", "5")
    err = refused(runner_bin, "-m", "no-such-model.gguf", "-p", "x", "--score",
                  "300")
    assert "--score" in err


@pytest.mark.parametrize("spec,needs", [
    ("mtp", "--gpu off"),
    ("lookup", None),
])
def test_draft_names_its_source(runner_bin, spec, needs):
    accepted_up_to_the_load(runner_bin, "-p", "x", "--draft", spec)


def test_draft_mtp_with_a_companion_file(runner_bin, tmp_path):
    head = tmp_path / "head.gguf"
    head.write_bytes(b"")
    accepted_up_to_the_load(runner_bin, "-p", "x", "--draft", "mtp:" + str(head))


def test_one_draft_source_per_run_still_holds(runner_bin):
    err = refused(runner_bin, "-m", "no-such-model.gguf", "-p", "x", "--draft",
                  "lookup", "--mtp")
    assert "one draft source" in err


@pytest.mark.parametrize("spec", ["linear=2", "base=10000", "yarn=4", "yarn=4,base=1e6"])
def test_rope_takes_named_settings(runner_bin, spec):
    accepted_up_to_the_load(runner_bin, "-p", "x", "--rope", spec)


def test_rope_refuses_linear_beside_yarn_as_before(runner_bin):
    err = refused(runner_bin, "-m", "no-such-model.gguf", "-p", "x", "--rope",
                  "linear=2,yarn=4")
    assert "cannot be used together" in err


@pytest.mark.parametrize("kind", ["tools", "adapter", "doctor"])
def test_info_loads_the_model_for_its_kind(runner_bin, kind):
    accepted_up_to_the_load(runner_bin, "--info", kind)


def test_info_caps_needs_no_model(runner_bin):
    p = run(runner_bin, "--info", "caps")
    assert p.returncode == 0, p.stderr[-300:]
    json.loads(p.stdout)


def _without_live_ram(report):
    """The fit report quotes the machine's free RAM at the moment it ran
    ("available RAM 3.03 GiB right now", "3.03 GiB to spare"); two runs a
    moment apart straddle other processes' allocations, which the parallel
    suite (pytest-xdist) made routine. The verdict and every size that comes
    from the file stay in the comparison."""
    return re.sub(rb"\d+\.\d+ GiB (right now|to spare)", rb"<live> GiB \1", report)


def test_info_fit_reads_the_header_of_m(runner_bin):
    if not FIXTURE.exists():
        pytest.skip("fixture test.gguf not built")
    p = run(runner_bin, "-m", str(FIXTURE), "--info", "fit", "--gpu", "off")
    assert p.returncode == 0, p.stderr.decode(errors="replace")[-400:]
    old = run(runner_bin, "--fit", str(FIXTURE), "--gpu", "off")
    assert _without_live_ram(old.stdout) == _without_live_ram(p.stdout)
    assert b"available RAM" in p.stdout and b"verdict" in p.stdout


def test_info_refuses_an_unknown_kind(runner_bin):
    err = refused(runner_bin, "--info", "everything")
    assert "--info" in err


# ------------------------------------------------- the rewrite pass: one verb


def test_rewrite_is_quantize_by_default(runner_bin):
    """A missing model under --rewrite fails the way --quantize fails: at the
    open, after the rewrite flags were accepted."""
    accepted_up_to_the_load(runner_bin, "--rewrite", "out.gguf", "--quant", "q8_0")


def test_rewrite_merge_trigger_needs_an_adapter_like_merge_lora_did(runner_bin):
    err = refused(runner_bin, "-m", "no-such-model.gguf", "--rewrite", "out.gguf",
                  "--merge-lora")
    assert "--lora" in err


def test_rewrite_context_trigger_needs_target_and_factor(runner_bin):
    err = refused(runner_bin, "-m", "no-such-model.gguf", "--rewrite", "out.gguf",
                  "--context-surgery")
    assert "-c TARGET" in err


def test_a_rewrite_trigger_without_rewrite_is_refused(runner_bin):
    err = refused(runner_bin, "-m", "no-such-model.gguf", "--merge-lora", "-p", "x")
    assert "--rewrite OUT" in err


def test_legacy_rewrite_verbs_still_take_their_output(runner_bin):
    accepted_up_to_the_load(runner_bin, "--quantize", "out.gguf")
    err = refused(runner_bin, "-m", "no-such-model.gguf", "--merge-lora", "out.gguf")
    assert "--lora" in err


# ------------------------------------------------------ serve-mode defaults


def test_wait_for_vram_accepts_zero_as_fail_fast(runner_bin):
    accepted_up_to_the_load(runner_bin, "--serve", "--wait-for-vram", "0", "--no-tray")


@pytest.mark.parametrize("value", ["on", "off"])
def test_yield_on_request_takes_on_or_off(runner_bin, value):
    accepted_up_to_the_load(runner_bin, "--serve", "--yield-on-request", value,
                            "--no-tray")


@pytest.mark.parametrize("flag", ["--port", "--ttl"])
def test_serve_only_knobs_are_refused_without_serve(runner_bin, flag):
    err = refused(runner_bin, "-m", "no-such-model.gguf", "-p", "x", flag, "9",
                  "--no-tray")
    assert flag in err and "--serve" in err


def test_parallel_is_valid_with_fit(runner_bin):
    if not FIXTURE.exists():
        pytest.skip("fixture test.gguf not built")
    p = run(runner_bin, "-m", str(FIXTURE), "--info", "fit", "--parallel", "2",
            "--gpu", "off")
    assert p.returncode == 0, p.stderr.decode(errors="replace")[-400:]


# ------------------------------------------------------------ trust inputs


def test_trust_key_accepts_the_keygen_output_file(runner_bin, tmp_path):
    key = tmp_path / "k.json"
    p = run(runner_bin, "attest", "keygen", str(key))
    assert p.returncode == 0, p.stderr[-300:]
    pub = json.loads(p.stdout)["public_key"]
    pubfile = tmp_path / "k.pub"
    pubfile.write_bytes(p.stdout)
    rec = tmp_path / "r.json"
    rec.write_text('{"a":1}\n')
    s = run(runner_bin, "attest", "sign", str(rec), "--key", str(key))
    assert s.returncode == 0, s.stderr[-300:]
    by_hex = run(runner_bin, "attest", "check", str(rec), "--trust-key", pub)
    by_file = run(runner_bin, "attest", "check", str(rec), "--trust-key",
                  str(pubfile))
    assert by_hex.returncode == 0, by_hex.stderr[-300:]
    assert by_file.returncode == by_hex.returncode
    other = tmp_path / "o.json"
    run(runner_bin, "attest", "keygen", str(other))
    wrong = run(runner_bin, "attest", "check", str(rec), "--trust-key",
                str(other))
    assert wrong.returncode != 0


# ------------------------------------------------------------- subcommands


def test_attest_help_and_no_verb(runner_bin):
    p = run(runner_bin, "attest", "--help")
    assert p.returncode == 0 and b"runner attest" in p.stdout
    p = run(runner_bin, "attest")
    assert p.returncode != 0 and b"runner attest" in p.stderr
    p = run(runner_bin, "attest", "frobnicate", "x")
    assert p.returncode != 0 and b"frobnicate" in p.stderr


def test_attest_bundle_and_pack_take_positionals(runner_bin, tmp_path):
    """The positional pair stands for the --export-* and --*-out flags; a
    missing receipt fails inside the export, after the pair was accepted."""
    out = tmp_path / "bundle"
    p = run(runner_bin, "attest", "bundle", str(tmp_path / "missing.json"), str(out))
    assert p.returncode != 0
    assert b"go together" not in p.stderr, p.stderr[-300:]
    p = run(runner_bin, "attest", "pack", str(tmp_path / "none"), str(out))
    assert p.returncode != 0
    assert b"go together" not in p.stderr, p.stderr[-300:]
    p = run(runner_bin, "attest", "bundle", str(tmp_path / "missing.json"))
    assert p.returncode != 0 and b"OUT_DIR" in p.stderr


def test_attest_watermark_keygen_and_detect(runner_bin, tmp_path):
    key = tmp_path / "wm.json"
    p = run(runner_bin, "attest", "watermark-keygen", str(key))
    assert p.returncode == 0, p.stderr[-300:]
    rec = tmp_path / "r.json"
    rec.write_text("{}")
    p = run(runner_bin, "attest", "detect", str(rec), "--key", str(key))
    # not a record and no -m: refused for that reason, which proves the
    # key reached --watermark and the file reached --detect-watermark
    assert p.returncode != 0 and b"needs -m MODEL" in p.stderr, p.stderr[-300:]


def test_attest_sign_model_maps_its_key(runner_bin, tmp_path):
    p = run(runner_bin, "attest", "sign-model", str(tmp_path / "m.gguf"))
    assert p.returncode != 0 and b"--model-key" in p.stderr, p.stderr[-300:]


def test_train_help_and_mapping(runner_bin, tmp_path):
    p = run(runner_bin, "train", "--help")
    assert p.returncode == 0 and b"runner train" in p.stdout
    p = run(runner_bin, "train")
    assert p.returncode != 0 and b"runner train" in p.stderr
    data = tmp_path / "d.txt"
    data.write_text("hello\n")
    # the short option names reach the training flags: --steps is refused
    # for the value --train-steps refuses, before any model is opened
    err = refused(runner_bin, "train", str(data), "-m", "no-such-model.gguf",
                  "--steps", "0")
    assert "--train-steps" in err or "--steps" in err
    for extra in (["--steps", "1", "--out", str(tmp_path / "a.gguf")],
                  ["--dpo", "--dpo-beta", "0.2"]):
        p = run(runner_bin, "train", str(data), "-m", "no-such-model.gguf",
                *extra, "--no-tray")
        err = p.stderr.decode(errors="replace")
        assert p.returncode != 0 and "cannot open" in err, err[-400:]


def test_train_refuses_a_second_data_file_as_unknown(runner_bin, tmp_path):
    err = refused(runner_bin, "train", "a.txt", "b.txt", "-m", "x.gguf")
    assert "b.txt" in err


# ------------------------------------------------------------------- help


def test_help_lists_the_subcommands_and_not_the_legacy_spellings(runner_bin):
    p = run(runner_bin, "--help")
    text = p.stdout.decode()
    assert p.returncode == 0
    assert "runner attest" in text and "runner train" in text
    declared = [l for l in text.splitlines() if l.startswith("  -")]
    for old in ("--gpu-layers", "--no-tray", "--tool-info", "--score-top",
                "--transcript ", "--receipts ", "--export-pack", "--keygen",
                "--train-steps", "--draft-lookup", "--mtp-file"):
        assert not any(old in l for l in declared), old
    assert "--adapt-info" not in text.split("older spellings")[0]
    assert "as one JSON line, then exit" not in text
