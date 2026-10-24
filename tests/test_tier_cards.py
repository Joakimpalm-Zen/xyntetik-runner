"""The "last validated" block on the Hugging Face cards, rendered from
docs/model-tiers.json by scripts/model-tiers.py --cards.

Offline: every card here is fixture text and the Hub is replaced at the
function boundary (fetch_card, upload_card). The absolute anchors are facts
from outside the code under test: the head of the published gemma-4-E2B
card as the Hub served it on 2026-10-23 (copied here verbatim), and marks
written out by hand for fixture cells whose version and date are known.
"""

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "model-tiers.py"

spec = importlib.util.spec_from_file_location("model_tiers", SCRIPT)
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

ROW_URL = ("https://github.com/Joakimpalm-Zen/xyntetik-runner/blob/main/"
           "docs/model-support.md#")

# https://huggingface.co/Joakimpalm-Zen/gemma-4-E2B-it-Q4_0-GGUF/raw/main/README.md
# as the Hub served it on 2026-10-23: lines 1 to 25 (the frontmatter through
# the marker after the status block), then line 124, where the card names
# its file
E2B_CARD = """\
---
license: apache-2.0
base_model: google/gemma-4-E2B-it
tags:
- gguf
- gemma4
- e-series
- per-layer-embeddings
- quantized
- xyntetik-runner
pipeline_tag: text-generation
---

# gemma-4-E2B-it, Q4_0/Q4_K mixed (GGUF)

<!-- BEGIN xyntetik-status -->
> **Release: built to be downloaded and served with [Xyntetik Runner](https://github.com/Joakimpalm-Zen/xyntetik-runner).**
> - **Measured status against the parent:** fails the fidelity bar (the dual-column numbers are on this card); published as the quickstart file for an 8 GB machine, not as a fidelity claim.
> - **Parent model:** [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
> - **Collection:** [Runner Releases](https://huggingface.co/collections/Joakimpalm-Zen/runner-releases-6aa98baaed03bba0e8a561ae)
> - **Evidence:** the tables on this card
> - **Runner compatibility:** the file the Runner README quickstart names.
<!-- END xyntetik-status -->

<!-- BEGIN xyntetik-actions -->
- File: `gemma-4-E2B-it-Q4_K_M-Q4_0-mix.gguf` (2,631,015,904 bytes)
"""
E2B_REPO = "Joakimpalm-Zen/gemma-4-E2B-it-Q4_0-GGUF"
E2B_FILE = "gemma-4-E2B-it-Q4_K_M-Q4_0-mix.gguf"


def gate(day, version, result="pass"):
    return {"kind": "gate", "runner_version": version, "date": day,
            "device_class": "macos-arm64-metal", "result": result,
            "commit": "b" * 40, "caps": {"version": version},
            "output_sha256": "c" * 64, "ran": "tier-smoke",
            "evidence": "docs/model-evidence.md"}


def row(rid, name, repo, tier="tier1", parts=1, **over):
    files = [{"name": name if i == 0 else f"{name}.part{i + 1}", "repo": repo,
              "sha256": "a" * 64, "bytes": 1} for i in range(parts)]
    r = {"id": rid, "model": "Toy", "tier": tier, "published": True,
         "claim": "A toy row.", "architecture": "llama", "files": files,
         "smoke": {"device_class": "macos-arm64-metal", "backend": "metal"},
         "backends": {"cpu": None, "cuda": None, "metal": None}, "lora": None,
         "transitions": [{"tier": tier, "effective": "2026-10-22",
                          "reason": "test", "last_validated_release": None}]}
    r.update(over)
    return r


def e2b_roster():
    """The published E2B row's marks as the roster held them on 2026-10-23:
    CPU and Metal gated at 1.5.1, CUDA not run, training admitted."""
    r = row("gemma-4-e2b-it-q4_k_m-q4_0-mix", E2B_FILE, E2B_REPO)
    r["backends"].update(cpu=gate("2026-10-23", "1.5.1"),
                         metal=gate("2026-10-23", "1.5.1"))
    r["lora"] = {"training": {"supported": True, "reason": None},
                 "adapter_serving": {"supported": True, "reason": None},
                 "runner_version": "1.5.1", "date": "2026-10-23"}
    return {"schema": mt.SCHEMA, "rows": [r], "exempt": []}


def rendered(doc, repo, card):
    return mt.place_card_block(card, mt.render_card_block(doc, repo),
                               mt.card_file_names(doc, repo))


# --- the block ------------------------------------------------------------------

def test_a_tier1_file_line_carries_the_readmes_marks_and_its_row_link():
    block = mt.render_card_block(e2b_roster(), E2B_REPO)
    assert block[0] == "<!-- BEGIN xyntetik-validated -->"
    assert block[-1] == "<!-- END xyntetik-validated -->"
    assert block[1] == "- **Last validated (Runner support tiers):**"
    assert block[2] == (
        "  - `gemma-4-E2B-it-Q4_K_M-Q4_0-mix.gguf`: CPU ✅ v1.5.1 2026-10-23"
        " · CUDA ☐ not run · Metal ✅ v1.5.1 2026-10-23"
        " · Train ✅ v1.5.1 2026-10-23"
        f" ([roster row]({ROW_URL}gemma-4-e2b-it-q4_k_m-q4_0-mix))")
    # the closing sentence: what a mark is, and what it is not
    assert "says nothing about answer quality" in block[3]
    assert '"not run" means nobody has run it yet' in block[3]


# --- insertion ------------------------------------------------------------------

def test_the_block_is_inserted_as_the_last_bullets_of_the_status_quote():
    out = rendered(e2b_roster(), E2B_REPO, E2B_CARD)
    lines = out.splitlines()
    end = lines.index("<!-- END xyntetik-status -->")
    assert lines[end - 1] == "> <!-- END xyntetik-validated -->"
    begin = lines.index("> <!-- BEGIN xyntetik-validated -->")
    assert lines[begin - 1].startswith("> - **Runner compatibility:**")
    assert all(l.startswith("> ") for l in lines[begin:end])
    # nothing outside the managed block moved: removing it gives the card back
    assert "\n".join(lines[:begin] + lines[end:]) + "\n" == E2B_CARD


# --- replacement ----------------------------------------------------------------

def test_an_existing_block_is_replaced_in_place_and_rendering_is_idempotent():
    doc = e2b_roster()
    once = rendered(doc, E2B_REPO, E2B_CARD)
    assert rendered(doc, E2B_REPO, once) == once
    # a roster change rewrites the block and only the block
    doc["rows"][0]["backends"]["cuda"] = gate("2026-11-02", "1.5.2")
    twice = rendered(doc, E2B_REPO, once)
    assert "CUDA ✅ v1.5.2 2026-11-02" in twice
    assert "CUDA ☐ not run" not in twice
    assert mt.strip_card_block(twice) == mt.strip_card_block(once) == E2B_CARD


def test_a_hand_edit_inside_the_block_is_overwritten_by_the_render():
    once = rendered(e2b_roster(), E2B_REPO, E2B_CARD)
    edited = once.replace("CUDA ☐ not run", "CUDA ✅ trust me")
    assert rendered(e2b_roster(), E2B_REPO, edited) == once


# --- what a block says ------------------------------------------------------------

def test_every_roster_file_of_the_repo_gets_its_own_line_in_roster_order():
    repo = "Joakimpalm-Zen/Toy-GGUF"
    doc = {"schema": mt.SCHEMA, "exempt": [], "rows": [
        row("toy-bf16", "Toy-BF16.gguf", repo),
        row("toy-q8_0", "Toy-Q8_0.gguf", repo,
            backends={"cpu": None, "cuda": {"kind": "limit", "reason": "no path",
                                            "matrix": "MANUAL.md#support-matrix"},
                      "metal": None}),
        row("other", "Other.gguf", "Joakimpalm-Zen/Other-GGUF")]}
    block = mt.render_card_block(doc, repo)
    files = [l for l in block if l.startswith("  - `")]
    assert [l.split("`")[1] for l in files] == ["Toy-BF16.gguf", "Toy-Q8_0.gguf"]
    assert files[0].count("☐ not run") == 4
    # n/a links the manual's support matrix by an absolute URL: a card is
    # not a page of this repository
    assert ("CUDA [n/a](https://github.com/Joakimpalm-Zen/xyntetik-runner/blob/"
            "main/MANUAL.md#support-matrix)") in files[1]


def test_a_sharded_file_is_one_line_named_by_its_first_part():
    repo = "Joakimpalm-Zen/Toy-GGUF"
    doc = {"schema": mt.SCHEMA, "exempt": [],
           "rows": [row("toy-split", "Toy-00001-of-00004.gguf", repo, parts=4)]}
    files = [l for l in mt.render_card_block(doc, repo) if l.startswith("  - `")]
    assert len(files) == 1 and "`Toy-00001-of-00004.gguf` (4 parts):" in files[0]
    assert mt.card_file_names(doc, repo) == ["Toy-00001-of-00004.gguf"]


def test_a_refused_training_answer_prints_its_reason():
    doc = e2b_roster()
    doc["rows"][0]["lora"]["training"] = {"supported": False,
                                         "reason": "MoE shared experts | not admitted"}
    line = mt.render_card_block(doc, E2B_REPO)[2]
    # a card line is not a table cell: the sentence is printed as recorded
    assert "Train no: MoE shared experts | not admitted (" in line


def test_a_failed_gate_never_renders_a_check_mark_on_a_card():
    doc = e2b_roster()
    doc["rows"][0]["backends"]["metal"] = gate("2026-10-23", "1.5.1", result="fail")
    line = mt.render_card_block(doc, E2B_REPO)[2]
    assert "Metal failed v1.5.1 2026-10-23" in line


def test_a_tier2_repo_says_its_tier_and_reason_and_carries_no_marks():
    repo = "Joakimpalm-Zen/gpt-oss-20b-keep30-MXFP4-GGUF"
    r = row("gpt-oss-20b-keep30-mxfp4", "gpt-oss-20b-keep30-MXFP4.gguf", repo,
            tier="tier2", backends=None,
            published_exception="Owner decision 2026-10-22: a published file "
                                "may be Tier 2 when its card already leads "
                                "with its measured shortfall.")
    block = mt.render_card_block({"schema": mt.SCHEMA, "exempt": [], "rows": [r]}, repo)
    assert block[2] == (
        "  - `gpt-oss-20b-keep30-MXFP4.gguf`: Tier 2, maintained on report, not "
        "gated on every release, so it carries no backend marks. Owner decision "
        "2026-10-22: a published file may be Tier 2 when its card already leads "
        "with its measured shortfall. "
        f"([roster row]({ROW_URL}gpt-oss-20b-keep30-mxfp4))")
    text = "\n".join(block)
    assert "✅" not in text and "☐" not in text and "CPU" not in text
    # the sentence about marks belongs to a block that shows marks
    assert "greedy smoke" not in text


def test_a_card_block_is_per_file_and_ignores_the_family():
    """The README lists models by family; a card covers one repository's
    files and names no family, so moving a row between families leaves
    every card block byte for byte as it was."""
    doc = e2b_roster()
    doc["rows"][0]["family"] = "Gemma 4"
    before = mt.render_card_block(doc, E2B_REPO)
    doc["rows"][0]["family"] = "Another family"
    assert mt.render_card_block(doc, E2B_REPO) == before
    assert "family" not in "\n".join(before).lower()


def test_the_cards_are_the_repositories_of_published_rows():
    doc = {"schema": mt.SCHEMA, "exempt": [], "rows": [
        row("a", "A.gguf", "Joakimpalm-Zen/b-GGUF"),
        row("b", "B.gguf", "Joakimpalm-Zen/A-GGUF"),
        row("c", "C.gguf", "Joakimpalm-Zen/A-GGUF"),
        row("d", "D.gguf", "someone/else-GGUF", published=False)]}
    assert mt.card_repos(doc) == ["Joakimpalm-Zen/A-GGUF", "Joakimpalm-Zen/b-GGUF"]


# --- refusals -------------------------------------------------------------------

def refusal(card, doc=None):
    with pytest.raises(mt.CardRefused) as e:
        rendered(doc or e2b_roster(), E2B_REPO, card)
    return str(e.value)


# https://huggingface.co/Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF/raw/main/README.md
# as the Hub served it on 2026-10-23, lines 1 to 18: a card with no status
# block, whose status prose is the owner's to write
FLASH_CARD = """\
---
license: other
license_name: qwen-community-license-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- speculative-decoding
- mtp
- qwen4exp
- xyntetik
- xyntetik-runner
---

# Qwen3.8-Flash-Next MTP head, GGUF companion

**What this is and is not.** This is the multi-token-prediction (MTP, "NextN") draft head of
| `Qwen3.8-Flash-Next-MTP-Q8_0.gguf` | 4,165,605,056 bytes |
"""
FLASH_REPO = "Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF"


def flash_roster():
    return {"schema": mt.SCHEMA, "exempt": [], "rows": [
        row("qwen3.8-flash-next-mtp-q8_0", "Qwen3.8-Flash-Next-MTP-Q8_0.gguf", FLASH_REPO)]}


def test_a_card_without_a_status_block_gets_a_quote_of_its_own_after_the_h1():
    """The status block is the owner's prose and is never written here; a
    card without one carries the block as its own quote, right after the H1."""
    out = rendered(flash_roster(), FLASH_REPO, FLASH_CARD)
    lines = out.splitlines()
    h1 = lines.index("# Qwen3.8-Flash-Next MTP head, GGUF companion")
    assert lines[h1 + 1] == ""
    assert lines[h1 + 2] == "> <!-- BEGIN xyntetik-validated -->"
    end = lines.index("> <!-- END xyntetik-validated -->")
    assert all(l.startswith("> ") for l in lines[h1 + 2:end + 1])
    assert lines[end + 1] == "" and lines[end + 2].startswith("**What this is and is not.**")
    assert "> - **Last validated (Runner support tiers):**" in lines
    assert "xyntetik-status" not in out
    # removing the managed lines gives the card back, byte for byte
    assert mt.strip_card_block(out) == FLASH_CARD


def test_a_quote_of_its_own_is_replaced_in_place():
    once = rendered(flash_roster(), FLASH_REPO, FLASH_CARD)
    assert rendered(flash_roster(), FLASH_REPO, once) == once
    doc = flash_roster()
    doc["rows"][0]["backends"]["cuda"] = gate("2026-11-02", "1.5.2")
    twice = rendered(doc, FLASH_REPO, once)
    assert "CUDA ✅ v1.5.2 2026-11-02" in twice
    assert mt.strip_card_block(twice) == FLASH_CARD


def test_a_card_with_no_status_block_and_no_h1_is_refused():
    card = FLASH_CARD.replace("# Qwen3.8-Flash-Next MTP head", "Qwen3.8-Flash-Next MTP head")
    with pytest.raises(mt.CardRefused) as e:
        rendered(flash_roster(), FLASH_REPO, card)
    assert "no xyntetik-status block and no H1" in str(e.value)


def test_a_heading_inside_the_frontmatter_is_not_the_h1():
    card = FLASH_CARD.replace("tags:\n", "# a YAML comment\ntags:\n", 1)
    lines = rendered(flash_roster(), FLASH_REPO, card).splitlines()
    assert lines[lines.index("# a YAML comment") + 1] == "tags:"
    h1 = lines.index("# Qwen3.8-Flash-Next MTP head, GGUF companion")
    assert lines[h1 + 2] == "> <!-- BEGIN xyntetik-validated -->"


def test_a_validated_block_away_from_the_h1_and_outside_a_status_block_is_refused():
    once = rendered(flash_roster(), FLASH_REPO, FLASH_CARD)
    block = [l for l in once.splitlines(keepends=True) if l.startswith("> ")]
    moved = FLASH_CARD + "".join(block)
    with pytest.raises(mt.CardRefused) as e:
        rendered(flash_roster(), FLASH_REPO, moved)
    assert "malformed" in str(e.value)


def test_a_status_less_card_must_still_mention_its_roster_files():
    card = FLASH_CARD.replace("| `Qwen3.8-Flash-Next-MTP-Q8_0.gguf` | 4,165,605,056 bytes |\n", "")
    with pytest.raises(mt.CardRefused) as e:
        rendered(flash_roster(), FLASH_REPO, card)
    assert "Qwen3.8-Flash-Next-MTP-Q8_0.gguf" in str(e.value)


@pytest.mark.parametrize("card", [
    # an END without its BEGIN
    E2B_CARD.replace("<!-- BEGIN xyntetik-status -->\n", ""),
    # two status blocks
    E2B_CARD + "<!-- BEGIN xyntetik-status -->\n<!-- END xyntetik-status -->\n",
    # END before BEGIN
    E2B_CARD.replace("<!-- BEGIN xyntetik-status -->", "<!-- TMP -->")
            .replace("<!-- END xyntetik-status -->", "<!-- BEGIN xyntetik-status -->")
            .replace("<!-- TMP -->", "<!-- END xyntetik-status -->"),
    # a marker sharing its line with text
    E2B_CARD.replace("<!-- END xyntetik-status -->",
                     "the end <!-- END xyntetik-status -->"),
])
def test_malformed_status_markers_are_refused(card):
    assert "malformed" in refusal(card) or "shares its line" in refusal(card)


@pytest.mark.parametrize("damage", [
    # a validated BEGIN with no END
    lambda c: c.replace("> <!-- END xyntetik-validated -->\n", ""),
    # the validated block outside the status block
    lambda c: c.replace("> <!-- END xyntetik-validated -->\n", "")
               .replace("<!-- BEGIN xyntetik-actions -->",
                        "<!-- END xyntetik-validated -->"),
    # two validated blocks
    lambda c: c.replace("<!-- END xyntetik-status -->",
                        "> <!-- BEGIN xyntetik-validated -->\n"
                        "> <!-- END xyntetik-validated -->\n"
                        "<!-- END xyntetik-status -->"),
])
def test_malformed_validated_markers_are_refused(damage):
    card = damage(rendered(e2b_roster(), E2B_REPO, E2B_CARD))
    assert "xyntetik-validated markers are malformed" in refusal(card)


def test_a_card_that_does_not_mention_a_roster_file_is_refused():
    doc = e2b_roster()
    doc["rows"].append(row("e2b-extra", "gemma-4-E2B-it-extra.gguf", E2B_REPO))
    msg = refusal(E2B_CARD, doc)
    assert "gemma-4-E2B-it-extra.gguf" in msg and E2B_FILE not in msg


def test_the_block_itself_does_not_count_as_the_card_mentioning_a_file():
    """After the first render the block names every file; a card whose own
    text dropped one must still be refused, or the check goes vacuous."""
    once = rendered(e2b_roster(), E2B_REPO, E2B_CARD)
    dropped = once.replace("- File: `gemma-4-E2B-it-Q4_K_M-Q4_0-mix.gguf`", "- File:")
    assert E2B_FILE in dropped
    assert E2B_FILE in refusal(dropped)


# --- --cards, --cards-check and --cards-push, with the Hub at the boundary --------

KVIST = "Joakimpalm-Zen/Toy-Kvist-GGUF"
KVIST_CARD = """\
# Toy Kvist (GGUF)

<!-- BEGIN xyntetik-status -->
> **Release.**
> - **Runner compatibility:** any Runner release.
<!-- END xyntetik-status -->

| `Toy-Kvist-BF16.gguf` | `Toy-Kvist-Q8_0.gguf` |
"""
BARE = "Joakimpalm-Zen/Toy-Bare-GGUF"
BARE_CARD = "# Toy bare\n\nNo status block; `Toy-Bare.gguf` is the file.\n"
LOST = "Joakimpalm-Zen/Toy-Lost-GGUF"
LOST_CARD = "# Toy lost\n\nThis card names no file.\n"


def hub_roster():
    doc = e2b_roster()
    doc["rows"] += [row("kvist-bf16", "Toy-Kvist-BF16.gguf", KVIST),
                    row("kvist-q8_0", "Toy-Kvist-Q8_0.gguf", KVIST),
                    row("bare", "Toy-Bare.gguf", BARE),
                    row("lost", "Toy-Lost.gguf", LOST)]
    return doc


class Hub:
    """The Hub at the function boundary: cards by repo, a revision per
    repo, and the uploads it was asked for."""

    def __init__(self, cards):
        self.cards = dict(cards)
        self.revs = {r: "0" * 39 + "1" for r in cards}
        self.uploads = []

    def fetch(self, repo):
        return self.cards[repo], self.revs[repo]

    def upload(self, repo, data, message, parent_commit):
        assert parent_commit == self.revs[repo], "pushed on top of a moved head"
        self.uploads.append((repo, data, message))
        self.cards[repo] = data.decode("utf-8")
        self.revs[repo] = "0" * 39 + "2"
        return f"https://huggingface.co/{repo}/commit/{self.revs[repo]}"


def hub():
    return Hub({E2B_REPO: E2B_CARD, KVIST: KVIST_CARD, BARE: BARE_CARD,
                LOST: LOST_CARD})


def states(results):
    return {repo: state for repo, state, _ in results}


def test_cards_writes_each_rendered_card_under_owner_dash_dash_repo(tmp_path):
    h = hub()
    results = mt.render_cards(hub_roster(), tmp_path, h.fetch)
    assert states(results) == {E2B_REPO: "rendered", KVIST: "rendered",
                               BARE: "rendered", LOST: "refused"}
    e2b = (tmp_path / "Joakimpalm-Zen--gemma-4-E2B-it-Q4_0-GGUF" / "README.md")
    assert e2b.read_text(encoding="utf-8") == rendered(hub_roster(), E2B_REPO, E2B_CARD)
    kvist = (tmp_path / "Joakimpalm-Zen--Toy-Kvist-GGUF" / "README.md").read_text(
        encoding="utf-8")
    assert kvist.count("☐ not run") == 8
    bare = (tmp_path / "Joakimpalm-Zen--Toy-Bare-GGUF" / "README.md").read_text(
        encoding="utf-8")
    assert bare.startswith("# Toy bare\n\n> <!-- BEGIN xyntetik-validated -->\n")
    # a refused card is reported by name and nothing is written for it
    assert not (tmp_path / "Joakimpalm-Zen--Toy-Lost-GGUF").exists()
    assert "Toy-Lost.gguf" in dict((r, d) for r, _, d in results)[LOST]


def test_cards_check_reports_drift_until_the_block_is_on_the_hub():
    h = hub()
    assert states(mt.cards_check(hub_roster(), h.fetch)) == {
        E2B_REPO: "drift", KVIST: "drift", BARE: "drift", LOST: "refused"}
    h.cards[E2B_REPO] = rendered(hub_roster(), E2B_REPO, E2B_CARD)
    h.cards[BARE] = rendered(hub_roster(), BARE, BARE_CARD)
    got = states(mt.cards_check(hub_roster(), h.fetch))
    assert got[E2B_REPO] == "ok" and got[BARE] == "ok"
    # a roster change is drift again: the card is checked like the README is
    doc = hub_roster()
    doc["rows"][0]["backends"]["cuda"] = gate("2026-11-02", "1.5.2")
    assert states(mt.cards_check(doc, h.fetch))[E2B_REPO] == "drift"


def test_cards_check_is_drift_for_a_hand_edit_inside_the_block():
    h = hub()
    h.cards[E2B_REPO] = rendered(hub_roster(), E2B_REPO, E2B_CARD).replace(
        "CUDA ☐ not run", "CUDA ✅ trust me")
    assert states(mt.cards_check(hub_roster(), h.fetch))[E2B_REPO] == "drift"


def test_a_card_the_hub_cannot_serve_is_refused_by_name():
    def fetch(repo):
        raise mt.CardRefused("HTTP 404 from the Hub")
    results = mt.cards_check(hub_roster(), fetch)
    assert set(states(results).values()) == {"refused"}
    assert all("HTTP 404" in d for _, _, d in results)


def push(doc, outdir, h, yes):
    return mt.cards_push(doc, outdir, yes, h.fetch, h.upload, "1.5.1")


def test_the_push_is_a_dry_run_unless_told_yes(tmp_path):
    h = hub()
    mt.render_cards(hub_roster(), tmp_path, h.fetch)
    results = push(hub_roster(), tmp_path, h, yes=False)
    assert states(results) == {E2B_REPO: "would push", KVIST: "would push",
                               BARE: "would push", LOST: "refused"}
    assert h.uploads == []


def test_the_push_uploads_one_commit_per_card_with_the_message(tmp_path):
    h = hub()
    mt.render_cards(hub_roster(), tmp_path, h.fetch)
    results = push(hub_roster(), tmp_path, h, yes=True)
    assert states(results)[E2B_REPO] == "pushed"
    assert [u[0] for u in h.uploads] == [E2B_REPO, BARE, KVIST]
    assert {u[2] for u in h.uploads} == {
        "card: last validated line from the Runner support roster (v1.5.1)"}
    assert h.cards[E2B_REPO] == rendered(hub_roster(), E2B_REPO, E2B_CARD)
    # pushed cards check clean, and a second push has nothing to do
    assert states(mt.cards_check(hub_roster(), h.fetch))[E2B_REPO] == "ok"
    assert states(push(hub_roster(), tmp_path, h, yes=True))[E2B_REPO] == "up to date"
    assert len(h.uploads) == 3


def test_the_push_refuses_a_card_edited_on_the_hub_since_it_was_rendered(tmp_path):
    """A concurrent edit to the card is never overwritten: the bytes outside
    the managed block must be the ones the render was made from."""
    h = hub()
    mt.render_cards(hub_roster(), tmp_path, h.fetch)
    h.cards[E2B_REPO] = E2B_CARD.replace("the tables on this card",
                                         "the tables below")
    for yes in (False, True):
        results = push(hub_roster(), tmp_path, h, yes=yes)
        assert states(results)[E2B_REPO] == "refused"
        assert "changed on the Hub" in dict((r, d) for r, _, d in results)[E2B_REPO]
    assert [u[0] for u in h.uploads] == [BARE, KVIST]
    assert "the tables below" in h.cards[E2B_REPO]


def test_the_push_refuses_a_render_the_roster_has_moved_past(tmp_path):
    h = hub()
    mt.render_cards(hub_roster(), tmp_path, h.fetch)
    doc = hub_roster()
    doc["rows"][0]["backends"]["cuda"] = gate("2026-11-02", "1.5.2")
    results = push(doc, tmp_path, h, yes=True)
    assert states(results)[E2B_REPO] == "refused"
    assert "--cards" in dict((r, d) for r, _, d in results)[E2B_REPO]
    assert E2B_REPO not in [u[0] for u in h.uploads]


def test_the_push_refuses_a_repo_with_no_rendered_card(tmp_path):
    h = hub()
    results = push(hub_roster(), tmp_path, h, yes=True)
    assert set(states(results).values()) == {"refused"}
    assert h.uploads == []


# --- an unreachable Hub is not drift ----------------------------------------------

class FakeHTTPError(Exception):
    pass


@pytest.mark.parametrize("fail, state", [
    ("timeout", "unreachable"), ("refused-connection", "unreachable"),
    (503, "unreachable"), (429, "unreachable"), (404, "refused")])
def test_a_network_failure_is_unreachable_and_a_missing_card_is_refused(
        monkeypatch, fail, state):
    """The check runs on a schedule against a network service: a timeout,
    a dropped connection or a busy Hub says nothing about the cards, while a
    card the Hub says is not there is a real finding."""
    import socket
    import urllib.error
    import urllib.request

    def urlopen(req, timeout=None):
        if fail == "timeout":
            raise urllib.error.URLError(socket.timeout("timed out"))
        if fail == "refused-connection":
            raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))
        raise urllib.error.HTTPError(req.full_url, fail, "status", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    results = mt.cards_check(e2b_roster())
    assert states(results) == {E2B_REPO: state}


def test_the_check_exits_soft_on_an_unreachable_hub_and_hard_on_drift(capsys):
    ok = [(E2B_REPO, "ok", "")]
    away = ok + [(KVIST, "unreachable", "timed out")]
    drift = away + [(BARE, "drift", "")]
    hard, soft = ("drift", "refused"), ("unreachable",)
    assert mt._report(ok, hard, soft) == 0
    assert mt._report(away, hard, soft) == mt.EXIT_UNREACHABLE == 3
    assert mt._report(drift, hard, soft) == 1
    err = capsys.readouterr().err
    assert "could not be read from the Hub" in err and "not drift" in err


# --- the cards-drift workflow -----------------------------------------------------

def test_the_drift_job_runs_after_merges_and_weekly_never_on_a_pull_request():
    """The cards are pushed by hand after a merge that changes the render, so
    a pull request would always see drift: the job runs on main and on a
    schedule, is not a required check, fails on drift or a refused card and
    only warns when the Hub cannot be read."""
    import re
    wf = (ROOT / ".github" / "workflows" / "cards-drift.yml").read_text(encoding="utf-8")
    on = wf.split("\non:", 1)[1].split("\npermissions:", 1)[0]
    assert "pull_request" not in on
    assert "branches: [main]" in on and "schedule:" in on
    assert re.search(r"^  cards-drift:$", wf, re.M)
    assert "scripts/model-tiers.py --cards-check" in wf
    assert f'"$status" -eq {mt.EXIT_UNREACHABLE}' in wf and "::warning" in wf
    assert 'exit "$status"' in wf
    # reading public cards needs no token, so none is handed to the job
    assert "secrets." not in wf and "HF_TOKEN" not in wf
