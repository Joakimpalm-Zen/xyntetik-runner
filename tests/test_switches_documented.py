"""Every RUNNER_* switch the engine reads is documented (R18.8.4, R18.8.5).

`RUNNER_*` variables are tuning and diagnostic switches outside the compatibility promise
(docs/versioning.md), but one nobody can find is one nobody can use, review or retire: the November
deprecation review (docs/deprecation-review-2026-11.md) found switches a test read and diagnostics that
were named nowhere. This holds the line its method drew. A switch is a `RUNNER_...` name passed to
`getenv` or to one of the helpers that read the environment, in `src/`; it is documented when MANUAL.md,
README.md or a page under docs/ names it. The pages that only record history do not count, and neither
does the capability inventory, which is generated from the source.

Test seams are exempt: `RUNNER_TEST_*`, the injected failures (`*INJECT*`), and the failure injectors
below, which were named before the `RUNNER_TEST_` convention.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

READERS = r"(?:getenv|env_i64|env_f64|env_u64|plat_env_file_get|metal_env_on|metal_env_default_on)"
SWITCH = re.compile(READERS + r'\(\s*"(RUNNER_[A-Z0-9_]+)"')
# records of what was, not documentation of what is
HISTORY = {"capability-inventory.md", "changelog-archive.md", "deprecation-review-2026-11.md"}
# failure injectors named before RUNNER_TEST_: each makes one failure path reachable from a test
INJECTORS = {
    "RUNNER_FILE_RMW_WRITE_FAIL", "RUNNER_LORA_ADAM_ALLOC_FAIL", "RUNNER_LORA_INSTALL_FAIL",
    "RUNNER_QUANTIZE_INSTALL_FAIL", "RUNNER_TURN_GUARD_SELFTEST",
}


def _switches():
    names = set()
    for p in (ROOT / "src").iterdir():
        if p.suffix in {".c", ".h", ".m", ".cu", ".cuh", ".metal"}:
            names |= set(SWITCH.findall(p.read_text(encoding="utf-8", errors="replace")))
    return names


def _seam(name):
    return name.startswith("RUNNER_TEST_") or "INJECT" in name or name in INJECTORS


def _documentation():
    pages = [ROOT / "MANUAL.md", ROOT / "README.md"]
    pages += [p for p in (ROOT / "docs").rglob("*.md")
              if p.name not in HISTORY and "compat-reports" not in p.parts]
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in pages)


def test_the_extraction_finds_the_switches():
    # a regex that matched nothing would pass the check below vacuously
    names = _switches()
    assert len(names) > 100, len(names)
    assert {"RUNNER_ATTN_BATCH", "RUNNER_METAL_MV2", "RUNNER_VRAM_PRIORITY"} <= names


def test_every_switch_is_documented():
    text = _documentation()
    missing = sorted(n for n in _switches()
                     if not _seam(n) and not re.search(rf"\b{n}\b", text))
    assert not missing, f"read in src/ and named on no page: {missing}"


def test_every_named_injector_is_still_read():
    # an injector that left the source is dead weight in the exemption
    assert INJECTORS <= _switches(), INJECTORS - _switches()
