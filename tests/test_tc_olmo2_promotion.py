"""R4.50.9: olmo2's default tensor-core promotion is scoped to Q8_0.

The host arm compiles the actual pure decision functions from cuda.c against
model.h, with metadata only. It tests the policy and the route-label decision,
not device dispatch or arithmetic. Its expected scope comes from the measured
Q8_0 file, not the implementation's table. It runs without CUDA.

The CUDA arm runs a chunked score on generated Q8_0 and Q4_0 files, checking
both the default's tc=taken/off label and its GEMM counter. A forced override
is not evidence that the default was promoted. Real-file numerical evidence
is recorded in docs/tc-promotion-evidence.md.
"""
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TYPES = ("Q8_0", "Q4_0", "Q4_K", "Q6_K", "Q5_K", "Q3_K", "Q2_K", "Q4_1", "Q5_0", "Q5_1",
         "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL",
         "NVFP4", "F32", "F16", "BF16")


@pytest.fixture(scope="module")
def policy(tmp_path_factory):
    """Only the CUDA-driver boundary is left out; no copy of the policy."""
    src = (ROOT / "src/cuda.c").read_text(encoding="utf-8")
    names = ("tc_codebook", "tc_kq_type", "tc_type_promotable", "tc_nvfp4_env", "tc_promoted",
             "tc_default_takes")
    functions = []
    for name in names:
        found = re.search(r"^static (?:bool|int) " + name + r"\([^;]*?\) \{\n.*?^\}", src, re.M | re.S)
        assert found, f"CUDA policy function {name} moved: update the host probe"
        functions.append(found.group())
    d = tmp_path_factory.mktemp("tc-policy")
    c = d / "policy.c"
    c.write_text('#include <stdatomic.h>\n#include <stdlib.h>\n#include "model.h"\n' +
                 "\n".join(functions) + '\nint main(int argc, char **argv) {\n'
                 ' if (argc != 2) return 2;\n model_t m = {0}; gguf_tensor t = {0};\n'
                 ' snprintf(m.arch, sizeof(m.arch), "%s", argv[1]);\n'
                 ' m.gf.n_tensors = 1; m.gf.tensors = &t;\n'
                 ' const int types[] = {' + ','.join('T_' + t for t in TYPES) + '};\n'
                 ' const char *names[] = {' + ','.join('"' + t + '"' for t in TYPES) + '};\n'
                 ' for (size_t i = 0; i < sizeof(types)/sizeof(*types); i++) {\n'
                 '  t.type = types[i]; t.n_dims = 2;\n'
                 '  int promoted = tc_promoted(&m, t.type), matrix = tc_default_takes(&m);\n'
                 '  t.n_dims = 1;\n'
                 '  printf("%s %d %d %d\\n", names[i], promoted, matrix, tc_default_takes(&m));\n'
                 ' }\n return 0;\n}\n', encoding="utf-8")
    exe = d / ("policy.exe" if sys.platform == "win32" else "policy")
    cc = shlex.split(os.environ.get("CC", "cc"))
    p = subprocess.run(cc + ["-std=gnu11", "-Wall", "-Wextra", "-Wshadow", "-Werror", "-I", str(ROOT / "src"),
                            str(c), "-o", str(exe)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr

    def query(arch):
        p = subprocess.run([str(exe), arch], capture_output=True, text=True, check=True, timeout=10)
        return {name: tuple(map(int, bits)) for name, *bits in (line.split() for line in p.stdout.splitlines())}
    return query


def test_olmo2_q8_0_is_promoted_for_matrices_only(policy):
    assert policy("olmo2")["Q8_0"] == (1, 1, 0)


def test_olmo2_unmeasured_types_stay_scalar(policy):
    got = policy("olmo2")
    assert {t: bits for t, bits in got.items() if t != "Q8_0"} == {
        t: (0, 0, 0) for t in TYPES if t != "Q8_0"}


def test_the_existing_scoped_control_is_live(policy):
    got = policy("clef")
    assert got["Q8_0"] == got["Q4_K"] == (1, 1, 0)
    assert got["Q4_0"] == (0, 0, 0)


@pytest.fixture(scope="module")
def cuda_models(runner_bin, tmp_path_factory):
    caps = subprocess.run([str(runner_bin), "--caps"], capture_output=True, check=True, text=True, timeout=30)
    if (json.loads(caps.stdout).get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("needs a CUDA device")
    d = tmp_path_factory.mktemp("tc-olmo2")
    base = d / "olmo2.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-olmo2.py", str(base), "--head-dim", "32", "--gqa"],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    out = {}
    for quant in ("q8_0", "q4_0"):
        path = d / f"olmo2-{quant}.gguf"
        subprocess.run([str(runner_bin), "-m", str(base), "--quantize", str(path), "--quant", quant],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        out[quant] = path
    return out


@pytest.mark.parametrize("quant, override, label, types", [
    ("q8_0", None, "taken", {"Q8_0"}),
    ("q8_0", "0", "off", set()),
    ("q4_0", None, "off", set()),
])
def test_cuda_label_and_dispatch_match_the_scoped_default(runner_bin, cuda_models, quant, override, label, types):
    env = dict(os.environ)
    env.pop("RUNNER_CUDA_TC", None)
    env.pop("RUNNER_CUDA_TC_NVFP4", None)
    env.update(RUNNER_SCORE_CHUNKED="1", RUNNER_DEBUG_ROUTE="1")
    if override is not None:
        env["RUNNER_CUDA_TC"] = override
    p = subprocess.run([str(runner_bin), "-m", str(cuda_models[quant]), "--score", "-f",
                        str(ROOT / "tests/fixtures/gold-corpus-2k.txt"), "--score-tokens", "120", "-c", "256",
                        "-b", "64", "--gpu", "99", "--no-tray", "-t", "2"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    route = re.search(r" tc=([a-z-]+)", p.stderr)
    gemm = re.search(r'"gemm_by_type":(\{[^}]*\})', p.stderr)
    assert route and gemm, p.stderr[-2000:]
    counts = json.loads(gemm.group(1))
    assert route.group(1) == label, (quant, route.group(1), counts)
    assert set(counts) == types and all(n > 0 for n in counts.values()), (quant, counts)
