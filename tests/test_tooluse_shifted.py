"""Tests for tooluse-shifted evaluation.

Tests catalog, prompt set, labels, and scorer components.
"""
import json
import hashlib
import os
import pytest


def load_catalog():
    """Load the catalog."""
    with open("evals/tooluse-shifted/catalog-v1.json") as f:
        return json.load(f)


def load_set():
    """Load the prompt set."""
    rows = []
    with open("evals/tooluse-shifted/set-v1.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_frozen_hash():
    """Load the frozen labels hash."""
    with open("evals/tooluse-shifted/LABELS.sha256") as f:
        return f.read().strip()


def compute_labels_hash():
    """Compute current labels hash."""
    rows = load_set()
    labels = []
    for row in rows:
        labels.append([row["id"], row["gold_tool"], row["gold_args"]])
    labels.sort(key=lambda x: x[0])
    canonical = "\n".join(json.dumps(label) for label in labels)
    return hashlib.sha256(canonical.encode()).hexdigest()


class TestCatalog:
    """Tests for the tool catalog."""

    def test_catalog_exists(self):
        """Catalog file must exist."""
        assert os.path.exists("evals/tooluse-shifted/catalog-v1.json")

    def test_catalog_has_tools(self):
        """Catalog must have at least 20 tools."""
        catalog = load_catalog()
        assert len(catalog) >= 20, f"Expected at least 20 tools, got {len(catalog)}"

    def test_catalog_has_none_tool(self):
        """Catalog must include a 'none' tool."""
        catalog = load_catalog()
        tool_names = [t["name"] for t in catalog]
        assert "none" in tool_names

    def test_catalog_tool_structure(self):
        """Each tool must have required fields."""
        catalog = load_catalog()
        for tool in catalog:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "args" in tool, f"Tool {tool['name']} missing 'args'"
            assert "required" in tool, f"Tool {tool['name']} missing 'required'"

    def test_catalog_tools_have_schemas(self):
        """Tools must have proper JSON schema args."""
        catalog = load_catalog()
        for tool in catalog:
            if tool["name"] != "none":
                args = tool.get("args", {})
                for arg_name, arg_spec in args.items():
                    assert "type" in arg_spec, \
                        f"Tool {tool['name']} arg {arg_name} missing 'type'"

    def test_catalog_required_fields_valid(self):
        """Required fields must match defined args."""
        catalog = load_catalog()
        for tool in catalog:
            name = tool["name"]
            args = set(tool.get("args", {}).keys())
            required = set(tool.get("required", []))
            assert required.issubset(args), \
                f"Tool {name}: required {required} not subset of args {args}"


class TestPromptSet:
    """Tests for the prompt evaluation set."""

    def test_set_exists(self):
        """Set file must exist."""
        assert os.path.exists("evals/tooluse-shifted/set-v1.jsonl")

    def test_set_has_minimum_prompts(self):
        """Set must have at least 150 prompts."""
        rows = load_set()
        assert len(rows) >= 150, f"Expected at least 150 prompts, got {len(rows)}"

    def test_all_rows_have_required_fields(self):
        """Each prompt row must have required fields."""
        rows = load_set()
        required = {"id", "prompt", "gold_tool", "gold_args", "category", "source"}
        for i, row in enumerate(rows):
            missing = required - set(row.keys())
            assert not missing, f"Row {i} ({row.get('id')}) missing {missing}"

    def test_unique_prompt_ids(self):
        """All prompt IDs must be unique."""
        rows = load_set()
        ids = [r["id"] for r in rows]
        assert len(set(ids)) == len(ids), "Duplicate prompt IDs found"

    def test_gold_tools_in_catalog(self):
        """All gold_tool values must exist in catalog."""
        catalog = load_catalog()
        catalog_tools = {t["name"] for t in catalog}
        rows = load_set()

        bad = []
        for row in rows:
            if row["gold_tool"] not in catalog_tools:
                bad.append((row["id"], row["gold_tool"]))

        assert not bad, f"Gold tools not in catalog: {bad}"

    def test_gold_args_valid(self):
        """gold_args must have valid names for the tool."""
        catalog = load_catalog()
        tools_by_name = {t["name"]: t for t in catalog}
        rows = load_set()

        bad = []
        for row in rows:
            tool_name = row["gold_tool"]
            tool = tools_by_name[tool_name]
            valid_args = set(tool.get("args", {}).keys())
            gold_args = set(row["gold_args"].keys())

            # For 'none' tool, args must be empty
            if tool_name == "none":
                if gold_args:
                    bad.append((row["id"], "none tool must have empty args"))
            else:
                # All gold args must be valid for this tool
                invalid = gold_args - valid_args
                if invalid:
                    bad.append((row["id"], f"invalid args {invalid} for {tool_name}"))

        assert not bad, f"Invalid gold args: {bad}"

    def test_category_distribution(self):
        """Each category must have at least 20 items."""
        rows = load_set()
        by_category = {}
        for row in rows:
            cat = row["category"]
            by_category.setdefault(cat, []).append(row["id"])

        expected_categories = {"paraphrase", "multi_intent", "underspecified",
                             "near_miss", "none"}
        assert set(by_category.keys()) == expected_categories, \
            f"Categories {set(by_category.keys())} != {expected_categories}"

        for cat, ids in by_category.items():
            assert len(ids) >= 20, \
                f"Category '{cat}' has {len(ids)} items, need at least 20"

    def test_source_field_valid(self):
        """source field must be 'handwritten' or 'bank-copy:...'."""
        rows = load_set()
        for row in rows:
            source = row["source"]
            valid = source == "handwritten" or source.startswith("bank-copy:")
            assert valid, f"Row {row['id']}: invalid source '{source}'"


class TestLabelsFreeze:
    """Tests for frozen labels."""

    def test_labels_frozen_file_exists(self):
        """LABELS.sha256 file must exist."""
        assert os.path.exists("evals/tooluse-shifted/LABELS.sha256")

    def test_labels_hash_matches(self):
        """Current labels must match frozen hash."""
        current = compute_labels_hash()
        frozen = load_frozen_hash()
        assert current == frozen, \
            f"Labels mismatch: current {current} != frozen {frozen}"

    def test_labels_hash_is_valid_sha256(self):
        """Frozen hash must be valid SHA256 format."""
        frozen = load_frozen_hash()
        assert len(frozen) == 64, f"Hash should be 64 chars, got {len(frozen)}"
        assert all(c in "0123456789abcdef" for c in frozen), "Invalid hex characters"


class TestScorerFunctions:
    """Tests for scorer parsing and validation functions."""

    def _extract_json(self, text):
        """Extract first valid JSON object from text (replicated from scorer)."""
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        return None
        return None

    def test_extract_json_valid(self):
        """extract_json should parse valid JSON."""
        text = 'some text {"tool": "read_file", "args": {"path": "test.txt"}} more'
        result = self._extract_json(text)
        assert result is not None
        assert result["tool"] == "read_file"

    def test_extract_json_invalid(self):
        """extract_json should return None for invalid JSON."""
        assert self._extract_json("no json here") is None
        assert self._extract_json("") is None

    def test_extract_json_nested(self):
        """extract_json should handle nested braces."""
        text = '{"tool": "write_file", "args": {"path": "x", "content": "a{b}c"}}'
        result = self._extract_json(text)
        assert result["args"]["content"] == "a{b}c"

    def test_schema_validation(self):
        """Schema validation should check required args."""
        catalog = load_catalog()
        tools_by_name = {t["name"]: t for t in catalog}

        # read_file requires 'path'
        read_tool = tools_by_name["read_file"]
        assert "path" in read_tool["required"]
        assert len(read_tool["required"]) == 1

        # search_files requires 'pattern' and 'path'
        search_tool = tools_by_name["search_files"]
        required = set(search_tool["required"])
        assert "pattern" in required
        assert "path" in required
        assert len(required) == 2


class TestLabelModificationDetection:
    """Test that modifications to labels are detected."""

    def test_frozen_hash_detects_change(self, tmp_path):
        """If a label is changed, hash should not match."""
        # Load current labels
        rows = load_set()
        labels = []
        for row in rows:
            labels.append([row["id"], row["gold_tool"], row["gold_args"]])
        labels.sort(key=lambda x: x[0])

        # Modify one label
        original_hash = compute_labels_hash()
        labels[0][1] = "modified_tool"  # Change the tool name
        modified = "\n".join(json.dumps(label) for label in labels)
        modified_hash = hashlib.sha256(modified.encode()).hexdigest()

        # Hashes should differ
        assert original_hash != modified_hash


import importlib.util
import io
import urllib.error
import urllib.request


def load_scorer():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "eval-tooluse-shifted.py")
    spec = importlib.util.spec_from_file_location("eval_tooluse_shifted", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestServerCommand:
    """The serve command the scorer actually builds."""

    def _capture(self, monkeypatch, mod):
        seen = []

        class P:
            def __init__(self, cmd, **kw):
                seen.append(cmd)

        monkeypatch.setattr(mod.subprocess, "Popen", P)
        return seen

    def test_serve_without_adapter(self, monkeypatch):
        mod = load_scorer()
        seen = self._capture(monkeypatch, mod)

        class Args:
            runner, model, threads, lora_scale = "./runner", "m.gguf", 4, 1.0

        mod.serve(Args(), None, 9000)
        assert "--lora" not in seen[0] and "--lora-scale" not in seen[0]
        assert seen[0][:3] == ["./runner", "-m", "m.gguf"] and "--gpu" in seen[0]

    def test_serve_with_adapter_passes_the_scale(self, monkeypatch):
        mod = load_scorer()
        seen = self._capture(monkeypatch, mod)

        class Args:
            runner, model, threads, lora_scale = "./runner", "m.gguf", 4, 0.5

        mod.serve(Args(), "a.gguf", 9000)
        i = seen[0].index("--lora")
        assert seen[0][i + 1] == "a.gguf"
        assert seen[0][seen[0].index("--lora-scale") + 1] == "0.5"


class TestNativeLeg:
    """The chat leg names the served model and never scores a refusal as a match."""

    CATALOG = [{"name": "read_file", "args": {"path": {"type": "string"}}, "required": ["path"]},
               {"name": "none", "args": {}, "required": []}]
    ROWS = [{"id": "a", "category": "none", "prompt": "hello", "gold": {"tool": "none", "args": {}}},
            {"id": "b", "category": "paraphrase", "prompt": "open x", "gold": {"tool": "read_file", "args": {"path": "x"}}}]

    def test_served_model_id_reads_v1_models(self, monkeypatch):
        mod = load_scorer()

        class R(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        monkeypatch.setattr(mod.urllib.request, "urlopen",
                            lambda url, timeout=0: R(json.dumps({"data": [{"id": "served-1"}]}).encode()))
        assert mod.served_model_id(1) == "served-1"

    def test_refused_requests_are_failures_not_none_matches(self, monkeypatch):
        mod = load_scorer()
        bodies = []

        def refuse(req, timeout=0):
            bodies.append(json.loads(req.data))
            raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, io.BytesIO(b'{"error":"model_not_found"}'))

        monkeypatch.setattr(mod.urllib.request, "urlopen", refuse)
        out = mod.score_native_leg(self.ROWS, 1, self.CATALOG, "served-1")
        assert all(b["model"] == "served-1" for b in bodies)
        assert out["refusals"] == 2
        assert out["tool_ok"] == 0 and out["exact_match"] == 0
        assert all(r["refusal"] for r in out["rows"])

    def test_a_none_answer_matches_only_when_the_request_succeeded(self, monkeypatch):
        mod = load_scorer()

        class R(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        monkeypatch.setattr(mod.urllib.request, "urlopen",
                            lambda req, timeout=0: R(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]}).encode()))
        out = mod.score_native_leg(self.ROWS[:1], 1, self.CATALOG, "served-1")
        assert out["refusals"] == 0 and out["tool_ok"] == 1
