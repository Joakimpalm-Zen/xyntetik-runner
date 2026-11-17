#!/usr/bin/env python3
"""The embedded PTX must use nothing above the .version its header is pinned to.

scripts/embed-ptx.py stamps each header's PTX `.version 7.8`, so that drivers
older than the toolchain still take it. nvcc 13 emits ISA 9.0, and the stamp
holds only while every line the compiler writes exists at 7.8. When one does
not, the driver's JIT rejects the WHOLE module, and every kernel in it is gone.
Found in R8.24.1 (R6.90): a 336-byte kernel argument struct read only in part
made nvcc annotate the parameter loads with `.pragma "used_bytes_mask N"`,
which needs ISA 8.3. The driver refused the module ("a PTX JIT compilation
failed"), and test-train-gpu then reported "no CUDA device, skipping".

Two checks, both need no GPU:
  - the scan: the features we know to need a later ISA, with the version the
    CUDA 13.0 ptxas names for each. It runs on every header in `make test`.
  - --ptxas PATH: assemble each header's decoded string with that ptxas, for
    the target the header names. This catches every feature, not only the
    ones listed. `make ptx` runs it with the pinned toolchain's ptxas, and so
    does the ptx-reproducible job.
--self-test plants, one at a time, a used_bytes_mask pragma and a `.reg .b128`
declaration (each scanned feature's live control) in a copy of each header's
string and requires every check to catch each. --plant OUT writes that planted copy of
one header as a header: for the gates' red run (MANUAL.md, R6.90).
"""
import os
import re
import subprocess
import sys
import tempfile

# (pattern, the ISA it needs, what it is). Versions as CUDA 13.0's ptxas
# reports them ("Feature ... requires PTX ISA .version 8.3 or later").
FEATURES = [
    (re.compile(r'\.pragma\s+"used_bytes_mask\b'), (8, 3),
     'nvcc 13\'s parameter-load annotation `.pragma "used_bytes_mask"`'),
    (re.compile(r"\.b128\b"), (8, 3), "the 128-bit type .b128"),
]
PLANT = '\t.pragma "used_bytes_mask 15";'
# the self-test's plants: one line for each feature the scan knows
PLANTS = [PLANT, '\t.reg .b128 %rq<2>;']


def decode(text):
    """The C string literal of an embed-ptx.py header: (name, the PTX text)."""
    m = re.search(r"static const char \*(\w+) =\n(.*?)\n;", text, re.S)
    if not m:
        raise ValueError("no `static const char *NAME = ...;` string in the header")
    out = []
    for ln in m.group(2).split("\n"):
        s = ln.strip()
        if not (s.startswith('"') and s.endswith('\\n"')):
            raise ValueError("an unexpected header line: " + s[:60])
        out.append(re.sub(r"\\(.)", lambda e: e.group(1), s[1:-3]))
    return m.group(1), "\n".join(out)


def version_of(ptx):
    m = re.search(r"^\.version (\d+)\.(\d+)", ptx, re.M)
    t = re.search(r"^\.target (sm_\d+)", ptx, re.M)
    if not m or not t:
        raise ValueError("no .version or .target line in the PTX")
    return (int(m.group(1)), int(m.group(2))), t.group(1)


def scan(ptx):
    """Lines that use a feature above the PTX's .version: [(line, text, why)]."""
    ver, _ = version_of(ptx)
    bad = []
    for i, ln in enumerate(ptx.split("\n"), 1):
        for pat, need, what in FEATURES:
            if need > ver and pat.search(ln):
                bad.append((i, ln.strip(), "%s needs ISA %d.%d, the header says %d.%d"
                            % (what, need[0], need[1], ver[0], ver[1])))
    return bad


def assemble(ptxas, ptx):
    """ptxas on the PTX for its .target: (ok, ptxas's output)."""
    _, target = version_of(ptx)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "embedded.ptx")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(ptx)
        r = subprocess.run([ptxas, "-arch=" + target, src, "-o", os.path.join(d, "out.cubin")],
                           capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def plant(ptx, line=PLANT):
    """The PTX with the line before its first parameter load."""
    lines = ptx.split("\n")
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("ld.param"):
            return "\n".join(lines[:i] + [line] + lines[i:])
    raise ValueError("no ld.param line to plant before")


def check(label, ptx, ptxas):
    """Problems with one PTX text, as lines; empty when it is clean."""
    out = ["%s line %d: %s (%s)" % (label, i, why, txt) for i, txt, why in scan(ptx)[:20]]
    if ptxas:
        ok, log = assemble(ptxas, ptx)
        if not ok:
            out.append("%s: ptxas rejects it:\n%s" % (label, log))
    return out


def main(argv):
    ptxas, self_test, plant_out, headers = None, False, None, []
    it = iter(argv)
    for a in it:
        if a == "--ptxas":
            ptxas = next(it)
        elif a == "--self-test":
            self_test = True
        elif a == "--plant":
            plant_out = next(it)
        else:
            headers.append(a)
    if not headers:
        print(__doc__)
        return 2
    if plant_out:
        text = open(headers[0], encoding="utf-8").read()
        lines = text.split("\n")
        for i, ln in enumerate(lines):
            s = ln.lstrip()
            if s.startswith('"') and s[1:].lstrip().startswith("ld.param"):
                # escaped as embed-ptx.py escapes (the tab stays a tab)
                esc = PLANT.replace("\\", "\\\\").replace('"', '\\"')
                lines.insert(i, '    "%s\\n"' % esc)
                break
        else:
            print("FAIL: no ld.param line in %s" % headers[0])
            return 1
        with open(plant_out, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        name, ptx = decode("\n".join(lines))
        print("wrote %s: %s with a used_bytes_mask pragma before its first parameter load"
              % (plant_out, name))
        return 0
    fails = []
    for h in headers:
        name, ptx = decode(open(h, encoding="utf-8").read())
        ver, target = version_of(ptx)
        bad = check("%s (%s)" % (h, name), ptx, ptxas)
        fails += bad
        print("ptx isa: %s: %s, .version %d.%d, .target %s: %s%s"
              % (h, name, ver[0], ver[1], target,
                 "%d problem(s)" % len(bad) if bad else "nothing above its version",
                 "; ptxas assembles it" if ptxas and not bad else ""))
        for line in PLANTS if self_test else []:
            caught = check("planted", plant(ptx, line), ptxas)
            what = line.strip()
            scanned = any("planted line" in c for c in caught)
            assembled = any("ptxas rejects" in c for c in caught)
            if not scanned or (ptxas and not assembled):
                fails.append("%s: the self-test's planted `%s` was not caught (scan %s, ptxas %s)"
                             % (h, what, "yes" if scanned else "NO",
                                ("yes" if assembled else "NO") if ptxas else "not run"))
            else:
                print("ptx isa: %s: the planted `%s` is caught by the scan%s"
                      % (h, what, " and by ptxas" if ptxas else ""))
    if fails:
        print("FAIL: the embedded PTX uses a feature its pinned .version does not have; the "
              "driver's JIT would reject the module whole (keep kernel arguments small and "
              "read whole: src/kernel_args.h):")
        for f in fails:
            print("  " + f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
