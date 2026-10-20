"""Every device buffer a CUDA instance allocates is freed with it.

src/cuda.c allocates an instance's scratch in its cuda_alloc_* functions
(`cu.MemAlloc(&g->NAME, ...)`) and frees it in gpu_ctx_free from a list kept
by hand. Until 2026-10-20 the list missed the hyper-connection streams, the
split-K partials and the shared-expert buffers, so every qwen4exp or
shared-expert instance leaked them into the CUDA context all models on the
device share. tests/test_cuda_teardown.c measures that on a device for the
families its fixtures load; this holds the two lists together for every
family, from the source alone, so a new buffer needs its free in the same
change. Source only: no device, no build.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

# instance buffers released by their own path rather than the list, and where
OWN_RELEASE = {
    "lora_a": "lora_dev_release (the adapter tables, rebound per adapter)",
    "lora_b": "lora_dev_release",
    "lora_t": "lora_dev_release",
}


def test_every_instance_buffer_is_freed_with_the_instance():
    src = (ROOT / "src/cuda.c").read_text()
    allocated = set(re.findall(r"MemAlloc\(&g->(\w+)", src))
    assert len(allocated) > 40, "the allocation pattern no longer matches cuda.c"
    i = src.index("static void gpu_ctx_free(model_t *m, gpu_t *g) {")
    j = src.index("CUdeviceptr bufs[] = {", i)
    k = src.index("};", j)
    freed = set(re.findall(r"g->(\w+)", src[j:k]))
    missing = sorted(allocated - freed - set(OWN_RELEASE))
    assert not missing, ("allocated by cuda_alloc_* but never freed in gpu_ctx_free: %s"
                         % ", ".join(missing))
    release = src[src.index("static void lora_dev_release(gpu_t *g) {"):]
    release = release[:release.index("\n}\n")]
    for name in OWN_RELEASE:
        assert "g->" + name in release, name + " is no longer released by lora_dev_release"


def test_every_shared_buffer_is_freed_with_the_upload():
    # the shared weights record (gpu_weights): every device pointer it
    # declares, scalar or per-layer table, is released by shared_destroy;
    # 2026-10-20 the hyper-connection gammas, the Mamba-2 tables and the MTP
    # gammas were not, nor any table's MTP slot past n_layer
    src = (ROOT / "src/cuda.c").read_text()
    i = src.index("typedef struct gpu_weights {")
    j = src.index("} gpu_weights;", i)
    fields = set()
    for line in src[i:j].splitlines():
        code = line.split("//")[0]
        if "CUdeviceptr" not in code:
            continue
        decl = code.split("CUdeviceptr", 1)[1].strip().rstrip(";")
        for name in decl.split(","):
            name = name.strip().lstrip("*").strip()
            if name.isidentifier():
                fields.add(name)
    assert len(fields) > 30, "the gpu_weights field pattern no longer matches"
    k = src.index("static void shared_destroy(gpu_weights *w) {")
    body = src[k:src.index("\n}\n", k)]
    missing = sorted(f for f in fields if "w->" + f not in body)
    assert not missing, "declared in gpu_weights but never released by shared_destroy: " + ", ".join(missing)
    assert "l < w->n_tab" in body, "shared_destroy must walk every table slot, the MTP block's included"
