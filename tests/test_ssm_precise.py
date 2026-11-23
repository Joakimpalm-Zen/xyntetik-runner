"""R6.115: SSM_PRECISE (Mamba-2 serial and chunked SSD cores in src/model.c) must emit
zero contracted FMA instructions in default and strict T3 builds on both Clang and GCC.

Earlier probes over reachable Mamba-2 shapes (ds 4..128, conv_kernel 3..4, T 1..4) and
the nemotron/granitehybrid fixtures showed Apple Clang 21 fusing both cores identically,
so this test directly checks the generated assembly of all five SSM_PRECISE functions.
"""

import os
import pathlib
import re
import shlex
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
    r"\b(?:fmadd|fmsub|fnmadd|fnmsub|fmla|fmls|v?fn?m(?:add|sub|addsub|subadd)[0-9]*[ps][sd])\b"
)


def _split_cmd(line):
    toks = shlex.split(line, posix=(os.name != "nt"))
    if os.name == "nt":
        toks = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] == '"' else t for t in toks]
    return toks


def _model_compile_cmd(make_cmd, extra_make_args):
    p = subprocess.run(
        [make_cmd, "-Bn", "--no-print-directory", *extra_make_args, "runner"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = p.stdout.replace("\\\n", " ").splitlines()
    cmd_line = next(x for x in lines if " -c src/model.c " in x)
    quants_line = next((x for x in lines if " -c src/quants.c " in x), "")
    return _split_cmd(cmd_line), _split_cmd(quants_line) if quants_line else []


def _symbol_body(asm_lines, fn):
    # Match Mach-O (`_fn:`), ELF (`fn:`), or Windows COFF (`fn:` / `_fn:`), ignoring inline comments
    pat = re.compile(rf"^_?{re.escape(fn)}(?:\.[A-Za-z0-9_.$]+)?:\s*(?:[#;@/].*)?$")
    start = next((i for i, l in enumerate(asm_lines) if pat.match(l.strip())), None)
    if start is None:
        return None
    body = []
    for l in asm_lines[start + 1 :]:
        s = l.strip()
        body.append(l)
        # Stop at ELF (.cfi_endproc / .size), Mach-O (.cfi_endproc / End function), or Win64 COFF (.seh_endproc)
        if s == ".cfi_endproc" or s == ".seh_endproc" or s.startswith(".size ") or s.startswith("; -- End function"):
            break
    return body


def _count_ssm_fmas(cmd_words):
    cc = cmd_words[0]
    ver = subprocess.run([cc, "--version"], capture_output=True, text=True, check=True).stdout
    is_clang = "clang" in ver.lower()

    src = (ROOT / "src" / "model.c").read_text(encoding="utf-8")
    for fn in SSM_FUNCS:
        sig = "static void " + fn + "("
        assert sig in src, f"missing {sig} in src/model.c"
        src = src.replace(sig, "__attribute__((noinline)) " + sig, 1)

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
    def test_symbol_body_stops_at_seh_endproc_and_detects_x86_fma(self):
        fake_asm = [
            "ssm_scan_worker:",
            "\tmulss\t%xmm0, %xmm1",
            "\taddss\t%xmm1, %xmm2",
            "\t.seh_endproc",
            "other_fast_fn:",
            "\tvfmadd213ss\t%xmm3, %xmm4, %xmm5",
            "\t.seh_endproc",
        ]
        body = _symbol_body(fake_asm, "ssm_scan_worker")
        self.assertIsNotNone(body)
        self.assertEqual([l.strip() for l in body if FMA_RE.search(l)], [])
        other = _symbol_body(fake_asm, "other_fast_fn")
        self.assertEqual([l.strip() for l in other if FMA_RE.search(l)], ["vfmadd213ss\t%xmm3, %xmm4, %xmm5"])

    def test_ssm_precise_uncontracted_default_and_t3(self):
        make_cmd = os.environ.get("MAKE") or shutil.which("make")
        if not make_cmd:
            self.skipTest("make not available")

        for label, make_args in (
            ("default", []),
            ("T3", ["T3=1", "CFLAGS=-O3 -ffast-math"]),
        ):
            with self.subTest(build=label):
                model_cmd, quants_cmd = _model_compile_cmd(make_cmd, make_args)
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
