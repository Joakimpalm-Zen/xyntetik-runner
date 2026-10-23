"""What the CLI decides from its arguments alone, and in which order.

main() parses every flag, refuses a combination it cannot honour, runs the
utility modes that need no model, and only then loads one. A refusal is the
behaviour under test here: each names the flag and the fix, and each must
arrive BEFORE the work it guards. The order is pinned by pointing -m at a file
that does not exist: an argument refusal that still wins proves the check ran
before the load, and a load error proves it did not.

Absolute anchors: the TCP port range (a port is a 16-bit number and 0 is not
a listening port), and the seed rule (xorshift64 maps state 0 to 0, so -s 0
cannot be honoured). The other expected values are this CLI's documented
refusals (MANUAL.md, `runner --help`), pinned so a restructured main() keeps
them and keeps their order.
"""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NO_MODEL = "no-such-model.gguf"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("m") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT)
    return m


def run(runner_bin, *args, cwd=None):
    return subprocess.run([str(runner_bin), *map(str, args)], cwd=cwd or ROOT,
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, errors="replace",
                          timeout=120)


# ---------------------------------------------------------------- parse rules

def test_help_is_an_answer_on_stdout(runner_bin):
    p = run(runner_bin, "--help")
    assert p.returncode == 0
    assert p.stdout.startswith("usage:") and p.stderr == ""


def test_an_unknown_option_is_refused_with_usage_on_stderr(runner_bin):
    p = run(runner_bin, "--bogus")
    assert p.returncode == 1
    assert "unknown option --bogus" in p.stderr
    assert "usage:" in p.stderr and p.stdout == ""


def test_a_flag_missing_its_value_prints_usage_and_fails(runner_bin):
    p = run(runner_bin, "-m")
    assert p.returncode == 1
    assert "usage:" in p.stderr and p.stdout == ""


def test_an_early_exit_ends_parsing_where_it_stands(runner_bin):
    """--version answers at its position: flags before it are parsed (and can
    fail), flags after it are never read."""
    after = run(runner_bin, "--version", "--bogus")
    assert after.returncode == 0 and after.stdout.startswith("runner ")
    before = run(runner_bin, "--bogus", "--version")
    assert before.returncode == 1 and "unknown option --bogus" in before.stderr


@pytest.mark.parametrize("value,ok", [("0", False), ("1", True), ("65535", True),
                                      ("65536", False)])
def test_port_domain_is_the_tcp_port_range(runner_bin, value, ok):
    p = run(runner_bin, "--port", value, "--caps")
    assert (p.returncode == 0) == ok, p.stderr
    if not ok:
        assert "--port expects an integer in [1, 65535]" in p.stderr


@pytest.mark.parametrize("args,consumed", [
    (["--cpu-moe", "--caps"], False),
    (["--cpu-moe", "auto", "--caps"], True),
    (["--cpu-moe", "3", "--caps"], True),
    (["--wait-for-vram", "--caps"], False),
    (["--wait-for-vram", "10", "--caps"], True),
])
def test_optional_values_are_taken_only_when_they_are_ours(runner_bin, args, consumed):
    """--cpu-moe takes "auto" or a count, --wait-for-vram a count; anything else
    is the next flag and is parsed as one, so --caps still answers."""
    p = run(runner_bin, *args)
    assert p.returncode == 0, p.stderr
    assert p.stdout.startswith('{"version":')


def test_a_word_after_cpu_moe_is_parsed_as_its_own_option(runner_bin):
    p = run(runner_bin, "--cpu-moe", "x", "--caps")
    assert p.returncode == 1 and "unknown option x" in p.stderr


@pytest.mark.parametrize("value", ["0", "86401"])
def test_wait_for_vram_bounds_are_checked_when_given(runner_bin, value):
    p = run(runner_bin, "--wait-for-vram", value, "--caps")
    assert p.returncode == 1
    assert "--wait-for-vram expects an integer in [1, 86400]" in p.stderr


@pytest.mark.parametrize("spec,why", [
    ("x", "--adapter wants NAME=PATH"),
    ("=p", "--adapter wants NAME=PATH"),
    ("bad name=p", "--adapter NAME must be 1 to 63 characters"),
])
def test_adapter_specs_are_checked_while_parsing(runner_bin, spec, why):
    p = run(runner_bin, "--adapter", spec, "--caps")
    assert p.returncode == 1 and why in p.stderr


def test_seventeen_adapters_are_refused(runner_bin):
    args = sum((["--adapter", "a%d=p" % i] for i in range(17)), [])
    p = run(runner_bin, *args, "--caps")
    assert p.returncode == 1 and "at most 16 --adapter" in p.stderr


# ------------------------------------------- refused before any model is loaded

BEFORE_LOAD = [
    # (arguments, exit status, the reason the refusal names)
    (["--system", "s", "-p", "x"], 1, "--system is only valid with interactive chat"),
    (["--think", "-p", "x"], 1, "--think is only valid with interactive chat"),
    (["--sign-record", "r.json"], 1, "--sign-record needs --sign-key FILE"),
    (["--export-pack", "d"], 1, "--export-pack RECEIPTS_DIR and --pack-out DIR go together"),
    (["--pack-envelope", "e", "-p", "x"], 1, "--pack-envelope and --pack-attach go with --export-pack"),
    (["--sign-model", "x.gguf"], 1, "--sign-model MODEL and --model-key KEY.pem go together"),
    (["--export-bundle", "r"], 1, "--export-bundle RECEIPT and --bundle-out DIR go together"),
    (["--detect-watermark", "x.txt"], 1, "--detect-watermark needs --watermark KEY"),
    (["--keygen-algo", "rsa", "--keygen", "k.key"], 1, "--keygen-algo: unknown algorithm rsa"),
    (["--draft-required", "-p", "x"], 1, "--draft-required needs --draft PATH"),
    (["--draft-required", "--draft", "d.gguf", "--serve"], 1, "--draft-required is a local CLI guard"),
    (["--draft-lookup", "--mtp", "-p", "x"], 1, "--draft-lookup: one draft source per run"),
    (["-hf", "owner/repo"], 1, "use -m or -hf, not both"),
    ([], 1, "need -p PROMPT, -i, or --serve"),
    (["-f", "no-such-prompt.txt"], 1, "cannot read no-such-prompt.txt"),
    (["--suspend-after", "2", "-p", "x"], 1, "--suspend-after N needs --session-out FILE"),
    (["--fork-seed", "3", "-p", "x"], 1, "--fork-seed forks a session and needs --resume FILE"),
    (["--resume", "img", "-p", "x"], 1, "--resume continues an image's prompt; give no -p"),
    (["--session-out", "o", "--serve"], 1, "session images are for one-shot runs"),
    (["--session-out", "o", "-p", "x", "--draft-lookup"], 1, "session images run the solo step loop"),
    (["--session-out", "o", "-p", "x", "-n", "-1"], 1, "session images need a finite generation budget"),
    (["--session-out", "o", "-p", "x", "-n", "4", "--suspend-after", "4"], 1,
     "--suspend-after N must stop before the -n budget is spent"),
    (["--resume", "img", "-s", "5"], 1, "--resume continues under the image's own settings; -s"),
    (["--verify", "no-such-record.json"], 3, "UNVERIFIABLE: cannot read no-such-record.json"),
    (["--yarn-factor", "1", "-p", "x"], 1, "--yarn-factor must be greater than 1"),
    (["--yarn-factor", "2", "--rope-scale", "2", "-p", "x"], 1,
     "--yarn-factor and --rope-scale cannot be used together"),
    (["--chat-template", "nope", "-p", "x"], 1, "unknown --chat-template 'nope'"),
    (["--context-surgery", "o.gguf", "--quant", "q8_0"], 1, "--context-surgery does not combine"),
    (["--context-surgery", "o.gguf"], 1, "--context-surgery requires -c TARGET and --yarn-factor FACTOR"),
    (["--merge-lora", "o.gguf"], 1, "--merge-lora requires --lora ADAPTER"),
    (["--merge-lora", "o.gguf", "--lora", "a.gguf", "--type-plan", "p.json"], 1,
     "--merge-lora does not combine"),
    (["--merge-lora", "o.gguf", "--lora", "a.gguf", "--quant", "nope"], 1,
     "--quant 'nope' is not a writable type"),
    (["--quantize", "o.gguf", "--type-plan-strict"], 1, "--type-plan-strict needs --type-plan"),
    (["--quantize", "o.gguf", "--quant", "q8_0", "--type-plan", "p.json"], 1,
     "--quant and --type-plan cannot be used together"),
    (["--quantize", "o.gguf", "--quant", "nope"], 1, "--quant 'nope' is not a writable type"),
    (["--receipts", "d", "-p", "x"], 1, "--receipts writes one receipt per served request"),
    (["--receipts-keep", "3", "--serve"], 1, "--receipts-keep needs --receipts DIR"),
    (["--kv-snapshots", "d", "-p", "x"], 1, "--kv-snapshots saves and loads named contexts"),
    (["--sessions", "d", "-p", "x"], 1, "--sessions keeps the server's session images"),
    (["--adapter", "a=a.gguf", "-p", "x"], 1, "--adapter routes adapters per request and needs --serve"),
    (["--adapter", "a=a.gguf", "--lora", "a.gguf", "--serve"], 1, "--adapter and --lora are exclusive"),
    (["--prune-experts", "p.json", "-p", "x"], 1, "--prune-experts requires --quantize OUT"),
    (["--remove-sublayer", "attn:1", "-p", "x"], 1, "--remove-sublayer requires --quantize OUT"),
    (["--type-plan", "p.json", "-p", "x"], 1, "--type-plan requires --quantize OUT"),
    (["--quant", "q8_0", "-p", "x"], 1, "--quant requires --quantize OUT"),
    (["--merge-allow-erased", "-p", "x"], 1, "--merge-allow-erased requires --merge-lora OUT"),
]


@pytest.mark.parametrize("args,rc,why", BEFORE_LOAD,
                         ids=[" ".join(a) or "<no mode>" for a, _, _ in BEFORE_LOAD])
def test_refused_before_the_model_is_opened(runner_bin, tmp_path, args, rc, why):
    """-m names a file that does not exist, so a refusal that still wins was
    decided from the arguments alone, before any load. -hf is the exception
    that proves the order: it is refused for coming with -m at all."""
    p = run(runner_bin, "-m", NO_MODEL, *args, cwd=tmp_path)
    assert p.returncode == rc, (p.returncode, p.stderr)
    assert why in p.stderr, p.stderr
    assert NO_MODEL not in p.stderr.replace(why, ""), "the model was opened first"


def test_the_load_is_what_fails_once_the_arguments_are_valid(runner_bin, tmp_path):
    """The control for the table above: valid arguments reach the load."""
    p = run(runner_bin, "-m", NO_MODEL, "-p", "x", cwd=tmp_path)
    assert p.returncode == 1
    assert NO_MODEL in p.stderr


def test_signing_utilities_ignore_the_generation_flags(runner_bin, tmp_path):
    """--sign-record runs before generation is even considered, so a missing
    key is the only complaint even beside a prompt and a model."""
    p = run(runner_bin, "--sign-record", "r.json", "-m", NO_MODEL, "-p", "x", cwd=tmp_path)
    assert p.returncode == 1
    assert "--sign-record needs --sign-key FILE" in p.stderr


def test_fit_answers_before_caps(runner_bin, model):
    """Two answers asked for in one run: the fit check is the one given,
    whichever order the flags came in."""
    for args in (["--fit", model, "--caps"], ["--caps", "--fit", model]):
        p = run(runner_bin, *args)
        assert p.returncode == 0, p.stderr
        assert p.stdout.startswith("fit: "), p.stdout[:80]


def test_a_utility_mode_needs_no_prompt(runner_bin, model):
    """Loading modes that are not generation (--tool-info here) run without
    -p, -i or --serve; generation needs one of the three."""
    p = run(runner_bin, "-m", model, "--gpu", "off", "--tool-info")
    assert p.returncode == 0, p.stderr
    assert p.stdout.startswith('{"tool_family":')
