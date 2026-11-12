"""Every buffer model.c allocates for a model is released (R6.68.30).

model_t and layer_t carry about 170 separately allocated buffers. Until
R6.68.30 two hand-kept lists released them (runtime_free and
model_free_weights), and nothing tied an allocation to its release: a
LayerNorm family's norm biases and a recurrent block's second pair of norms
leaked on every load, so on every model swap in a server, until R6.68.23
found them by this same cross-check done by hand. The release now walks
three tables of rows (LAYER_OWNED, MODEL_WEIGHTS_OWNED, MODEL_RUNTIME_OWNED);
this test reads model.c and fails when a field is filled from an allocator and
is neither a row of one of them nor freed by name anywhere in the file (a
buffer released another way, a cache the GPU may own, is freed by name where
that is decided).

Read from the source, not measured: it sees `x->field = malloc(...)` and its
kin, not a buffer built in a local and stored later. The leak gate (the
sanitized suite) measures; this keeps the next missing row from being written.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOC = r"(?:malloc|calloc|realloc|tensor_to_f32|strdup)"
HOLDERS = {"l": "layer", "ly": "layer", "layer": "layer", "m": "model", "dst": "model"}


def _owned(src):
    allocated = {}
    for mt in re.finditer(r"\b(l|ly|layer|m|dst)->([a-z_0-9]+)\s*=\s*(?:\([^)]*\)\s*)?" + ALLOC + r"\(",
                          src):
        key = (HOLDERS[mt.group(1)], mt.group(2))
        allocated.setdefault(key, src.count("\n", 0, mt.start()) + 1)
    released = set()
    for mt in re.finditer(r"free\(\s*(?:\(void \*\))?\s*(l|ly|layer|m|dst|w)->([a-z_0-9]+)\s*\)", src):
        released.add((HOLDERS.get(mt.group(1), "model"), mt.group(2)))
    for table, holder in (("LAYER_OWNED", "layer"), ("MODEL_WEIGHTS_OWNED", "model"),
                          ("MODEL_RUNTIME_OWNED", "model")):
        body = re.search(r"static const size_t " + table + r"\[\] = \{(.*?)\};", src, re.S)
        assert body, f"model.c has no {table}"
        for f in re.findall(r"offsetof\((?:layer_t|model_t), ([a-z_0-9]+)\)", body.group(1)):
            released.add((holder, f))
    return allocated, released


def test_every_allocated_field_is_released():
    src = (ROOT / "src" / "model.c").read_text(encoding="utf-8")
    allocated, released = _owned(src)
    assert len(allocated) > 100, f"found only {len(allocated)} allocated fields: the reader broke"
    missing = sorted((f"{who}.{f}", line) for (who, f), line in allocated.items()
                     if (who, f) not in released)
    assert not missing, ("allocated in model.c and released nowhere (add a row to "
                         "LAYER_OWNED, MODEL_WEIGHTS_OWNED or MODEL_RUNTIME_OWNED): "
                         + ", ".join(f"{f} (model.c:{line})" for f, line in missing))


def test_a_row_removed_is_caught():
    """The reader is not vacuous: drop one row and the field is reported."""
    src = (ROOT / "src" / "model.c").read_text(encoding="utf-8")
    cut = src.replace("offsetof(layer_t, attn_norm_b),", "", 1)
    assert cut != src, "the attn_norm_b row is gone; pick another row for this check"
    allocated, released = _owned(cut)
    assert ("layer", "attn_norm_b") in allocated and ("layer", "attn_norm_b") not in released
