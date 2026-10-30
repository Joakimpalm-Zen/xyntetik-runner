#!/usr/bin/env python3
"""Record K2-Horizon's own renders of a tool-schema torture set.

K2-Horizon's chat template (the one IFM embeds in its K2-Horizon-0.9B and
-7B GGUFs) renders tool declarations as markdown, falls back to one JSON
line per tool for the whole set when a schema uses a construct its markdown
renderer does not handle, and raises on schemas it calls malformed. Runner
ports all three (src/template.c, "K2-Horizon"). The template-conformance
matrix carries one ordinary tool, so this set exercises the rest: every
schema shape the renderer has a branch for, every fallback trigger and
every raise.

The publisher's renders are recorded into tests/fixtures/
k2-horizon-tool-schemas.json, and tests/test_template.c checks Runner's
render of each case against it (byte for byte, marks removed) and Runner's
refusal against each raise. The fixture is the artifact under test; this
script only re-records it from the template:

    python scripts/k2h-tool-schema-fixture.py models/K2-Horizon-1B-Q8_0.gguf

Each case renders `[user "Hi"]` with the tools and the generation prompt,
the template's defaults otherwise (markdown presentation, xml calls,
reasoning_effort high). The oracle's BOS is stripped, as the conformance
allowlist does (k2-horizon/bos): Runner's tokenizer adds it.
"""
import hashlib
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "fixtures", "k2-horizon-tool-schemas.json")


def harness():
    path = os.path.join(ROOT, "scripts", "template-conformance.py")
    spec = importlib.util.spec_from_file_location("template_conformance", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fn(name, parameters=None, description="Look something up.", **extra):
    f = {"name": name, "description": description}
    if parameters is not None:
        f["parameters"] = parameters
    f.update(extra)
    return {"type": "function", "function": f}


def obj(props, required=None, **extra):
    s = {"type": "object", "properties": props}
    if required is not None:
        s["required"] = required
    s.update(extra)
    return s


CASES = [
    # ---- the markdown renderer's branches
    ("scalars", [fn("lookup", obj({
        "q": {"type": "string", "description": "What to look up.\nOne line per term."},
        "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
        "ratio": {"type": "number", "default": 0.5},
        "tiny": {"type": "number", "default": 1e-07},
        "big": {"type": "number", "default": 1.5e+16},
        "whole": {"type": "number", "default": 100.0},
        "flag": {"type": "boolean", "default": False},
        "empty": {"type": "string", "default": ""},
        "none": {"type": "string", "default": None},
    }, ["q"]))]),
    ("enums", [fn("pick", obj({
        "unit": {"type": "string", "enum": ["c", "f", "it's", "a\\b", "two\nlines", ""]},
        "level": {"enum": [1, 2.5, True, None, "x"]},
        "nothing": {"type": "string", "enum": []},
    }, ["unit"]))]),
    ("nested-object", [fn("create", obj({
        "user": obj({
            "name": {"type": "string", "description": "Full name.\nAs on the passport."},
            "address": obj({
                "city": {"type": "string"},
                "zip": {"type": "string", "pattern": "^[0-9]{5}$"},
            }, ["city"], description="Where they live."),
        }, ["name"], description="The new user."),
    }, ["user"]))]),
    ("arrays", [fn("batch", obj({
        "ids": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
        "pairs": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
        "rows": {"type": "array", "items": obj({
            "k": {"type": "string"}, "v": {"type": "number"}}, ["k"])},
        "anything": {"type": "array"},
        "bare_items": {"items": {"type": "string"}},
        "true_items": {"type": "array", "items": True},
        "false_items": {"type": "array", "items": False},
    }))]),
    ("type-unions", [fn("maybe", obj({
        "a": {"type": ["string", "null"]},
        "b": {"type": ["array", "null"], "items": {"type": "integer"}},
        "c": {"type": []},
        "d": {"type": "string", "format": "date-time", "title": "When"},
        "e": {},
        "f": True,
        "n": False,
    }, ["a"]))]),
    ("combinators", [fn("route", obj({
        "target": {"oneOf": [
            {"type": "string", "description": "A host name."},
            obj({"ip": {"type": "string"}}, ["ip"]),
        ], "description": "Where to send it."},
        "mode": {"anyOf": [{"type": "integer"}, {"type": "string", "enum": ["auto"]}]},
        "label": {"type": "string", "description": "Plain, beside them."},
    }, ["target"]))]),
    ("root-oneof", [fn("either", {"oneOf": [
        obj({"a": {"type": "string"}}, ["a"]),
        obj({"b": {"type": "integer"}}, ["b"]),
    ], "description": "One of two shapes."})]),
    ("root-items", [fn("listy", {"type": "array", "items": {"type": "string"},
                                 "description": "A list root."})]),
    ("additional-properties", [fn("meta", obj({
        "tags": {"type": "object", "additionalProperties": {"type": "string"}},
        "closed": {"type": "object", "additionalProperties": False},
        "open": {"type": "object", "additionalProperties": True},
    }, additionalProperties=False))]),
    ("pattern-properties", [fn("headers", obj({
        "h": {"type": "object", "patternProperties": {
            "^x-": {"type": "string", "description": "Custom header."},
            "^n-": 5}},
        "flat": {"type": "object", "patternProperties": "anything"},
    }))]),
    ("returns", [
        fn("weather", obj({"city": {"type": "string"}}, ["city"]),
           returns=obj({"temp": {"type": "number"}, "sky": {"type": "string"}}, ["temp"],
                       description="Current conditions.")),
        fn("ping", obj({}), returns="pong"),
        fn("status", obj({}), response=obj({"ok": {"type": "boolean"}}, ["ok"])),
    ]),
    ("no-parameters-shape", [
        fn("now", obj({})),
        fn("bare", {"type": "object"}),
        fn("described", {"type": "object", "description": "Nothing to pass."}),
    ]),
    ("defaults-repr", [fn("cfg", obj({
        "opts": {"type": "object", "default": {"a": 1, "b": [1, 2], "c": "x  y\tz", "d": None}},
        "list": {"type": "array", "default": ["it's", "q\"uote", "back\\slash"]},
        "ws": {"type": "string", "default": "  spaced  out  "},
    }))]),
    ("unicode", [fn("translate", obj({
        "text": {"type": "string", "description": "Text, any script: 日本語, Ελληνικά, emoji 🌍."},
        "to": {"type": "string", "enum": ["ja", "el", "sv"], "default": "sv"},
        "pad": {"type": "string", "default": "a b　c"},
    }, ["text"]))]),
    ("defs-without-ref", [fn("withdefs", obj({
        "x": {"type": "string"},
    }, **{"$defs": {"Thing": obj({"id": {"type": "integer"}}, ["id"])}}))]),
    ("variant-required", [fn("variants", obj({
        "v": {"oneOf": [{"required": ["a"]}, {"type": "string"}]},
    }))]),
    ("multi-tool-order", [
        fn("first", obj({"a": {"type": "string"}}, ["a"]), description="First tool."),
        fn("second", obj({"b": {"type": "integer"}}), description="Second tool.\nTwo lines."),
        fn("third", obj({}), description=""),
    ]),
    # ---- the JSON fallback: one construct the markdown renderer does not
    # handle sends the WHOLE toolset to tojson
    ("fallback-ref", [
        fn("plain", obj({"a": {"type": "string"}})),
        fn("refd", obj({"t": {"$ref": "#/$defs/T"}}, **{"$defs": {"T": {"type": "string"}}})),
    ]),
    ("fallback-oneof-mapping", [fn("odd", obj({"a": {"oneOf": {"type": "string"}}}))]),
    ("fallback-root-container", [fn("odd", obj({"a": {"type": "string"}},
                                             **{"x-meta": {"owner": "ops"}}))]),
    ("fallback-function-container", [fn("odd", obj({"a": {"type": "string"}}),
                                         metadata={"team": "ops"})]),
    ("fallback-nested-mapping-extra", [fn("odd", obj({
        "a": {"type": "string", "x-ui": {"widget": {"kind": "text"}}}}))]),
    ("fallback-required-without-properties-in-variant", [fn("odd", obj({
        "a": {"anyOf": [{"type": "object", "required": ["k"]}]}}))]),
    ("fallback-numbers-tojson", [
        fn("ref2", obj({"n": {"$ref": "#/x"}, "f": {"type": "number", "default": 0.1},
                        "i": {"type": "integer", "default": 3}, "u": {"type": "string", "default": "é"}})),
    ]),
    # ---- the raises (Runner answers 400 with the template's message)
    ("raise-required-not-list", [fn("bad", obj({"a": {"type": "string"}}, "a"))]),
    ("raise-required-null", [fn("bad", {"type": "object", "properties": {"a": {"type": "string"}},
                                        "required": None})]),
    ("raise-required-unknown", [fn("bad", obj({"a": {"type": "string"}}, ["a", "b"]))]),
    ("raise-required-no-properties", [fn("bad", {"type": "object", "required": ["a"]})]),
    ("raise-nested-required", [fn("bad", obj({"a": obj({"x": {"type": "string"}}, ["y"])}))]),
    ("raise-items-required", [fn("bad", obj({"a": {"type": "array", "items": obj({"x": {"type": "string"}}, ["z"])}}))]),
    ("raise-returns-required", [fn("bad", obj({}), returns=obj({"x": {"type": "string"}}, ["q"]))]),
    ("raise-missing-parameters", [fn("bad")]),
    ("raise-parameters-null", [fn("bad", None, parameters_null=True)]),
    ("raise-arguments-not-parameters", [fn("bad", None, arguments={"type": "object"})]),
    ("raise-properties-not-object", [fn("bad", {"type": "object", "properties": ["a"]})]),
    ("raise-oneof-number", [fn("bad", obj({"a": {"oneOf": 3}}))]),
    ("raise-description-number", [fn("bad", obj({}), description=42)]),
    # lenient: a response schema (no returns) is not raised on
    ("response-lenient", [fn("ok", obj({}), response=obj({"x": {"type": "string"}}, ["nope"]))]),
]

# Schemas that are not JSON Schema at all. The template renders them (the
# markdown renderer's `any`, or the JSON fallback); Runner refuses them
# first, for every family, in the tool envelope's JSON Schema check
# (template.c tool_schema_sane), because the call grammar is compiled from
# the same schema. Recorded with the template's render all the same, so the
# divergence is on file and checked.
RUNNER_REFUSES = [
    ("invalid-items-string", [fn("odd", obj({"a": {"type": "array", "items": "string"}}))],
     "a schema must be an object"),
    ("invalid-property-string", [fn("odd", obj({"g": "just a string"}))],
     "a schema must be an object"),
    ("invalid-property-null", [fn("odd", obj({"h": None}))],
     "a schema must be an object"),
    ("invalid-type-mapping", [fn("odd", obj({"a": {"type": {"kind": "string"}}}))],
     "\"type\" is not a JSON Schema type"),
    ("invalid-enum-string", [fn("odd", obj({"a": {"type": "string", "enum": "abc"}}))],
     "\"enum\" must be an array"),
    ("invalid-parameters-list", [fn("odd", ["not", "an", "object"])],
     "a schema must be an object"),
]


def build_tools(tools):
    out = []
    for t in tools:
        f = dict(t["function"])
        if f.pop("parameters_null", False):
            f["parameters"] = None
        out.append({"type": t["type"], "function": f})
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: k2h-tool-schema-fixture.py PATH/TO/K2-Horizon-*.gguf")
    h = harness()
    kv = h.gguf_kv(sys.argv[1])
    tmpl = kv["tokenizer.chat_template"]
    toks = kv.get("tokenizer.ggml.tokens") or []
    bos = toks[kv["tokenizer.ggml.bos_token_id"]]
    meta = {"bos_token": bos, "eos_token": toks[kv["tokenizer.ggml.eos_token_id"]]}
    env = h.jinja_env()
    cases = []
    for cid, tools, refuses in ([c + (None,) for c in CASES] + RUNNER_REFUSES):
        tools = build_tools(tools)
        msgs = [{"role": "user", "content": "Hi"}]
        rec = {"id": cid, "tools": tools}
        if refuses:
            rec["runner_refuses"] = refuses
        try:
            r = h.render_reference(env, tmpl, meta, msgs, True, "default",
                                   "enable_thinking", tools)
            assert r.startswith(bos), cid
            rec["render"] = r[len(bos):]
        except Exception as e:  # the template's raise_exception, or a TypeError
            rec["raises"] = "%s: %s" % (type(e).__name__, e)
        cases.append(rec)
    doc = {
        "_README": "K2-Horizon's own renders of a tool-schema torture set, "
                   "recorded by scripts/k2h-tool-schema-fixture.py from the "
                   "template IFM embeds in K2-Horizon-0.9B-GGUF and "
                   "K2-Horizon-7B-GGUF; checked against Runner by "
                   "tests/test_template.c. Each render is [user \"Hi\"] with "
                   "the tools and the generation prompt, BOS stripped.",
        "template_sha256": hashlib.sha256(tmpl.encode()).hexdigest(),
        "template_bytes": len(tmpl.encode()),
        "cases": cases,
    }
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    n_raise = sum(1 for c in cases if "raises" in c)
    n_ref = sum(1 for c in cases if "runner_refuses" in c)
    print("%d cases (%d raise, %d refused by Runner's schema check) -> %s"
          % (len(cases), n_raise, n_ref, os.path.relpath(OUT, ROOT)))


if __name__ == "__main__":
    main()
