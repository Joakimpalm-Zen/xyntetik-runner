"""R6.115: SSM_PRECISE (Mamba-2 serial and chunked SSD cores in src/model.c) must emit
zero contracted FMA instructions in default and strict T3 builds on both Clang and GCC.

Earlier probes over reachable Mamba-2 shapes (ds 4..128, conv_kernel 3..4, T 1..4) and
the nemotron/granitehybrid fixtures showed Apple Clang 21 fusing both cores identically,
so this test directly checks the generated assembly of all five SSM_PRECISE functions.
"""

import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

SSM_FUNCS = (
    "mamba2_ssd_core_serial",
    "ssm_conv1d_worker",
    "ssm_scan_worker",
    "ssm_gate_norm_worker",
    "mamba2_ssd_core_chunked",
)

FMA_RE = re.compile(
    r"\b(?:fmadd|fmsub|fnmadd|fnmsub|fmla|fmls|v?fn?m(?:add|sub)[0-9]*[ps][sd])\b"
)


def _model_compile_cmd(extra_make_args):
    p = subprocess.run(
        ["make", "-Bn", "--no-print-directory", *extra_make_args, "runner"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = p.stdout.replace("\\\n", " ").splitlines()
    cmd_line = next(x for x in lines if " -c src/model.c " in x)
    quants_line = next((x for x in lines if " -c src/quants.c " in x), "")
    return cmd_line.split(), quants_line.split()


def _symbol_body(asm_lines, fn):
    # Match either `fn:` (ELF) or `_fn:` (Mach-O), ignoring comments after `:`
    pat = re.compile(rf"^_?{re.escape(fn)}:(?:\s|$)")
    start = next((i for i, l in enumerate(asm_lines) if pat.match(l.strip())), None)
    if start is None:
        return None
    body = []
    for l in asm_lines[start + 1 :]:
        s = l.strip()
        body.append(l)
        if s == ".cfi_endproc" or s.startswith(".size "):
            break
    return body


def _count_ssm_fmas(cmd_words):
    cc = cmd_words[0]
    ver = subprocess.run([cc, "--version"], capture_output=True, text=True, check=True).stdout
    is_clang = "clang" in ver.lower()

    src = (ROOT / "src" / "model.c").read_text(encoding="utf-8")
    for fn in ("mamba2_ssd_core_serial(", "mamba2_ssd_core_chunked("):
        assert "static void " + fn in src, f"missing static void {fn} in src/model.c"
        src = src.replace("static void " + fn, "__attribute__((noinline)) static void " + fn, 1)

    with tempfile.TemporaryDirectory() as td:
        tmp_c = pathlib.Path(td) / "model_ssm.c"
        tmp_s = pathlib.Path(td) / "model_ssm.s"
        tmp_c.write_text(src, encoding="utf-8")

        args = []
        skip_next = False
        for tok in cmd_words:
            if skip_next:
                skip_next = False
                continue
            if tok == "-c":
                continue
            if tok == "-o":
                skip_next = True
                continue
            if tok == "src/model.c":
                continue
            if tok == "-I":
                args.extend(["-I", str(ROOT / "src")])
                skip_next = True
                continue
            args.append(tok)
        args.extend(["-I", str(ROOT / "src"), "-S", str(tmp_c), "-o", str(tmp_s)])
        subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True)
        asm_lines = tmp_s.read_text(encoding="utf-8", errors="replace").splitlines()

    counts = {}
    for fn in SSM_FUNCS:
        body = _symbol_body(asm_lines, fn)
        assert body is not None, f"symbol {fn} missing from assembled src/model.c"
        fmas = [l.strip() for l in body if FMA_RE.search(l)]
        counts[fn] = fmas
    return is_clang, counts


class SsmPreciseContractionTest(unittest.TestCase):
    def test_ssm_precise_uncontracted_default_and_t3(self):
        if shutil.which("make") is None:
            self.skipTest("make not available")

        for label, make_args in (
            ("default", []),
            ("T3", ["T3=1", "CFLAGS=-O3 -ffast-math"]),
        ):
            with self.subTest(build=label):
                model_cmd, quants_cmd = _model_compile_cmd(make_args)
                is_clang, counts = _count_ssm_fmas(model_cmd)

                if quants_cmd:
                    self.assertNotIn("-ffp-contract=fast-honor-pragmas", quants_cmd)
                    self.assertNotIn("-ffast-math", quants_cmd)
                if not is_clang or label == "T3":
                    self.assertNotIn("-ffp-contract=fast-honor-pragmas", model_cmd)
                    self.assertNotIn("-Wno-overriding-option", model_cmd)

                bad = {fn: fmas for fn, fmas in counts.items() if fmas}
                self.assertEqual(
                    bad,
                    {},
                    f"{label} build fused FMA inside SSM_PRECISE: "
                    + ", ".join(f"{k}={len(v)} ({v[0]})" for k, v in bad.items()),
                )


if __name__ == "__main__":
    unittest.main()
