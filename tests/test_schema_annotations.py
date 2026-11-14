"""Annotation-only schema keywords are accepted and ignored (R2.17).

A JSON Schema validator ignores annotations; Runner refused every keyword it did not know, which
refused 4,072 of JSONSchemaBench's 8,108 schemas for keywords that constrain nothing (id, readonly,
Kubernetes' patch directives, Snowplow's self, ...). A named list of annotation-only keywords is now
accepted and ignored. Every keyword that could constrain the instance and is not on the list stays
refused: Runner never accepts a constraint it does not enforce. A vendor extension (x-*) is accepted
by its exact name only, where its vendor defines it as constraining nothing: x-kubernetes-validations
is a validation rule, and accepting it would let a model break it.

id and $id are annotations only where they change no reference resolution: at the root (a local
"#/..." pointer resolves against the document either way), or where the subtree under them holds
no $ref. A nested id over a $ref would move that reference's base, so it is refused, unless it is
only a fragment ("#a" names a place and moves no base).
"""
import json
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL = ROOT / "test.gguf"
SELF = {"vendor": "com.example", "name": "thing", "format": "jsonschema", "version": "1-0-0"}


def _compile(runner_bin, tmp_path, schema):
    f = tmp_path / "s.json"
    f.write_text(json.dumps(schema), encoding="utf-8")
    p = subprocess.run([str(runner_bin), "-m", str(MODEL), "--json-schema", str(f), "-p", "x", "-n", "1",
                        "--temp", "0", "--gpu", "off", "--no-tray"],
                       cwd=ROOT, capture_output=True, text=True, timeout=60)
    return p.returncode, p.stderr


def _obj(**props):
    return {"type": "object", "properties": props, "required": sorted(props)}


ACCEPTED = {
    "readonly, its misspelling": _obj(a={"type": "string", "readonly": True}),
    "Kubernetes' patch directives and group-version-kind": {
        **_obj(a={"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}}},
                  "x-kubernetes-patch-strategy": "merge", "x-kubernetes-patch-merge-key": "name"}),
        "x-kubernetes-group-version-kind": [{"group": "", "kind": "Pod", "version": "v1"}]},
    "Angular's x-prompt": _obj(a={"type": "string", "x-prompt": "Name?"}),
    "an unreferenced definition's $id over a $ref": {
        "type": "string",
        "$defs": {"unused": {"$id": "https://example.test/unused", "type": "object",
                             "properties": {"x": {"$ref": "#/$defs/T"}}},
                  "T": {"type": "integer"}}},
    "id at the root, with a local $ref": {"id": "http://example.com/s.json",
                                         "definitions": {"n": {"type": "string"}},
                                         **_obj(a={"$ref": "#/definitions/n"})},
    "a nested id over no $ref": _obj(a={"id": "#a", "type": "string"}),
    "a nested fragment-only $id over a $ref": {"definitions": {"n": {"type": "string"}},
                                              **_obj(a={"$id": "#a", **_obj(b={"$ref": "#/definitions/n"})})},
    # The shape of the one JSONSchemaBench schema the base compiled and the base check refused
    # (written here from its shape; the benchmark's schemas are not redistributed): an absolute root
    # $id, and nested $ids that are paths, the labels older generators wrote, over "#/definitions"
    # references. Resolved against the root, as before: a deliberate leniency (fable, 2026-11-14).
    "a path $id under an absolute root, over a $ref": {
        "$id": "http://example.test/schemas/query.json",
        "definitions": {"n": {"type": "string"}},
        **_obj(a={"$id": "/properties/a", **_obj(b={"$ref": "#/definitions/n"})})},
    "Snowplow's self at the root": {"self": SELF, **_obj(a={"type": "string"})},
}

REFUSED = {
    "a nested id over a $ref": ({"definitions": {"n": {"type": "string"}},
                                 **_obj(a={"id": "http://other.example/", **_obj(b={"$ref": "#/definitions/n"})})},
                                "id"),
    "a nested $id over a $ref": ({"$defs": {"n": {"type": "string"}},
                                  **_obj(a={"$id": "http://other.example/", **_obj(b={"$ref": "#/$defs/n"})})},
                                 "$id"),
    "a nested absolute $id under an absolute root, over a $ref": ({
        "$id": "http://example.test/schemas/query.json", "definitions": {"n": {"type": "string"}},
        **_obj(a={"$id": "http://other.example/a.json", **_obj(b={"$ref": "#/definitions/n"})})}, "$id"),
    "a path $id with no absolute root, over a $ref": ({"definitions": {"n": {"type": "string"}},
        **_obj(a={"$id": "/properties/a", **_obj(b={"$ref": "#/definitions/n"})})}, "$id"),
    "a nested $id reached through a $ref, over a $ref": ({"$defs": {
        "A": {"$id": "http://other.example/", **_obj(b={"$ref": "#/$defs/T"})}, "T": {"type": "integer"}},
        **_obj(a={"$ref": "#/$defs/A"})}, "$id"),
    "self below the root": (_obj(a={"type": "string", "self": SELF}), "self"),
    "a Kubernetes validation rule": ({"type": "integer", "x-kubernetes-validations": [{"rule": "self > 0"}]},
                                     "x-kubernetes-validations"),
    "Kubernetes' int-or-string": ({"type": "string", "x-kubernetes-int-or-string": True},
                                  "x-kubernetes-int-or-string"),
    "Kubernetes' list type": ({"type": "array", "items": {"type": "string"}, "x-kubernetes-list-type": "set"},
                              "x-kubernetes-list-type"),
    "Kubernetes' preserve-unknown-fields": (_obj(a={"type": "object", "x-kubernetes-preserve-unknown-fields": True}),
                                            "x-kubernetes-preserve-unknown-fields"),
    "a vendor extension not on the list": (_obj(a={"type": "string", "x-order": 1}), "x-order"),
    "a constraint Runner does not enforce": ({"type": "array", "items": {"type": "string"},
                                              "uniqueItems": True}, "uniqueItems"),
    "a key that is no keyword on the list": (_obj(a={"type": "string", "_format": "date-time"}), "_format"),
}


@pytest.mark.parametrize("name", sorted(ACCEPTED))
def test_an_annotation_is_accepted_and_ignored(runner_bin, tmp_path, name):
    rc, err = _compile(runner_bin, tmp_path, ACCEPTED[name])
    assert rc == 0, f"{name}: {err[-400:]}"


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_what_could_constrain_stays_refused(runner_bin, tmp_path, name):
    schema, word = REFUSED[name]
    rc, err = _compile(runner_bin, tmp_path, schema)
    assert rc != 0 and word in err, f"{name}: rc {rc}, {err[-400:]}"
