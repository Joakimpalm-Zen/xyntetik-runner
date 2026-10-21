"""Build/gate contracts that must not silently turn a red run green."""
import os
import pathlib
import shutil
import subprocess
import json
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class SanitizerBuild(unittest.TestCase):
    def test_debug_instruments_transformations_and_aborts_on_ub(self):
        p = subprocess.run(["make", "-Bn", "debug"], cwd=ROOT,
                           capture_output=True, text=True, check=True)
        line = next(x for x in p.stdout.replace("\\\n", " ").splitlines() if " -o runner-debug " in x)
        self.assertIn("-fno-sanitize-recover=undefined", line)
        self.assertIn("-fno-omit-frame-pointer", line)
        self.assertIn("src/quantize.c", line)
        self.assertNotIn("quantize.o", line)
        self.assertNotIn("quants.o", line)


class CommandGate(unittest.TestCase):
    def test_roster_refuses_an_empty_root_inventory(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            for d in ("scripts", "tests/fuzz", ".github/workflows"):
                (root / d).mkdir(parents=True)
            script = root / "scripts/check-ci-roster.py"
            script.write_bytes((ROOT / "scripts/check-ci-roster.py").read_bytes())
            (root / "tests/fuzz/fuzz_probe.c").write_text("")
            (root / "Makefile").write_text("FUZZ_TARGETS = probe\n")
            (root / ".github/workflows/ci.yml").write_text("  targets: probe\n")
            result = subprocess.run([sys.executable, str(script)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_roster_catches_an_unwired_test_and_duplicate_fuzz_group(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            for d in ("scripts", "tests/fuzz", ".github/workflows"):
                (root / d).mkdir(parents=True)
            script = root / "scripts/check-ci-roster.py"
            script.write_bytes((ROOT / "scripts/check-ci-roster.py").read_bytes())
            (root / "tests/test_new.py").write_text("")
            (root / "tests/fuzz/fuzz_probe.c").write_text("")
            workflow = root / ".github/workflows/ci.yml"
            workflow.write_text("  targets: probe\n")
            make = root / "Makefile"
            make.write_text("FUZZ_TARGETS = probe\n# tests/test_new.py is not an execution entry\n")
            def run():
                return subprocess.run([sys.executable, str(script)], capture_output=True).returncode
            self.assertNotEqual(run(), 0)
            make.write_text("FUZZ_TARGETS = probe\ntest:\n\tpython -m pytest tests/test_new.py\n")
            self.assertEqual(run(), 0)
            workflow.write_text("  targets: probe probe\n")
            self.assertNotEqual(run(), 0)

    def test_roster_catches_a_device_target_no_lane_runs(self):
        """A test-cuda-* target outside cuda-gate.sh's DEVICE_STEPS, or a
        test-metal-* target test-metal does not call, is a gate nothing runs;
        DEVICE_OPT_IN is the only other place it may be, and a lane may not
        name a target the Makefile lacks."""
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            for d in ("scripts", "tests/fuzz", ".github/workflows"):
                (root / d).mkdir(parents=True)
            script = root / "scripts/check-ci-roster.py"
            script.write_bytes((ROOT / "scripts/check-ci-roster.py").read_bytes())
            (root / "tests/fuzz/fuzz_probe.c").write_text("")
            (root / "tests/test_new.py").write_text("")
            (root / ".github/workflows/ci.yml").write_text("  targets: probe\n")
            gate = root / "scripts/cuda-gate.sh"
            make = root / "Makefile"
            base = ("FUZZ_TARGETS = probe\ntest:\n\tpython -m pytest tests/test_new.py\n"
                    "test-cuda-a:\n\ttrue\ntest-metal-b:\n\ttrue\n")
            def run(makefile, steps):
                make.write_text(makefile)
                gate.write_text('DEVICE_STEPS="%s"\n' % steps)
                return subprocess.run([sys.executable, str(script)], capture_output=True).returncode
            metal = "test-metal:\n\t$(MAKE) --no-print-directory test-metal-b\n"
            self.assertEqual(run(base + metal, "a:test-cuda-a"), 0)
            self.assertNotEqual(run(base + metal, ""), 0)          # cuda target unrun
            self.assertNotEqual(run(base + "test-metal:\n\ttrue\n", "a:test-cuda-a"), 0)
            self.assertEqual(run(base + "DEVICE_OPT_IN = test-metal-b\n", "a:test-cuda-a"), 0)
            self.assertNotEqual(run(base + metal, "a:test-cuda-a z:test-cuda-typo"), 0)

    def test_fpclass_check_refuses_a_raw_class_test(self):
        """A raw isnan()/isfinite() in a -ffast-math unit is refused (clang
        22 folds them under -ffinite-math-only); the helper, the exempt
        no-fast-math units, comments and strings are not."""
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / "scripts").mkdir()
            (root / "src").mkdir()
            script = root / "scripts/check-fpclass.py"
            script.write_bytes((ROOT / "scripts/check-fpclass.py").read_bytes())
            def run(body, name="a.c"):
                for p in (root / "src").iterdir():
                    p.unlink()
                (root / "src" / name).write_text(body)
                return subprocess.run([sys.executable, str(script)], capture_output=True).returncode
            self.assertNotEqual(run("int f(double x) { return isnan(x); }\n"), 0)
            self.assertNotEqual(run("int f(float x) { return __builtin_isfinite(x); }\n"), 0)
            self.assertEqual(run("int f(double x) { return fp_isnan(x); }\n"), 0)
            self.assertEqual(run("// isnan(x) folds\nconst char *s = \"isfinite(\";\n"), 0)
            self.assertEqual(run("int f(double x) { return isnan(x); }\n", "quants.c"), 0)

    def test_failed_command_keeps_its_status_and_timing(self):
        with tempfile.TemporaryDirectory() as td:
            p = subprocess.run([sys.executable, str(ROOT / "scripts/ci-run.py"),
                                "--out", td, "failure", "--", sys.executable,
                                "-c", "raise SystemExit(7)"], capture_output=True)
            self.assertEqual(p.returncode, 7)
            record = json.loads((pathlib.Path(td) / "failure.json").read_text())
            self.assertEqual(record["exit_code"], 7)
            self.assertFalse(record["timed_out"])
            self.assertGreater(record["seconds"], 0)

    def test_ptx_source_check_without_nvcc_fails_in_one_line(self):
        # a machine without the toolkit: a named failure, not a traceback
        with tempfile.TemporaryDirectory() as td:
            missing = str(pathlib.Path(td) / "nvcc")
            p = subprocess.run([sys.executable, str(ROOT / "scripts/check-ptx-source.py"),
                                "--nvcc", missing], capture_output=True, text=True, timeout=60)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("not found", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_timeout_is_a_failure_with_a_record(self):
        with tempfile.TemporaryDirectory() as td:
            p = subprocess.run([sys.executable, str(ROOT / "scripts/ci-run.py"),
                                "--out", td, "--timeout", "0.1", "timeout", "--",
                                sys.executable, "-c", "import time; time.sleep(30)"],
                               capture_output=True, timeout=10)
            self.assertEqual(p.returncode, 124)
            self.assertTrue(json.loads((pathlib.Path(td) / "timeout.json").read_text())["timed_out"])

    def test_required_suite_cannot_pass_empty_skipped_or_failed(self):
        with tempfile.TemporaryDirectory() as td:
            xml = pathlib.Path(td) / "tests.xml"
            for body, expected in [("", 1), ('<testcase><skipped/></testcase>', 1),
                                   ('<testcase><failure/></testcase>', 1),
                                   ('<testcase/>', 0)]:
                xml.write_text('<testsuite>' + body + '</testsuite>')
                p = subprocess.run([sys.executable, str(ROOT / "scripts/check-junit.py"),
                                    "--no-skips", str(xml)], capture_output=True)
                self.assertEqual(p.returncode, expected, body)

    def test_a_device_lane_rejects_only_skips_for_want_of_the_device(self):
        """test-cuda-gated: on a CUDA host a case skipped for want of CUDA did
        not run, while a skip for something else (or one the lane allows)
        does not make the device lane fail."""
        with tempfile.TemporaryDirectory() as td:
            xml = pathlib.Path(td) / "cuda.xml"
            def run(reason):
                xml.write_text('<testsuite><testcase name="a"/><testcase name="b">'
                               '<skipped message="%s"/></testcase></testsuite>' % reason)
                return subprocess.run([sys.executable, str(ROOT / "scripts/check-junit.py"),
                                       "--no-skips-matching", "(?i)cuda|device|gpu",
                                       "--allow-skip", "CUDA runtime is available",
                                       str(xml)], capture_output=True).returncode
            self.assertEqual(run("no CUDA device"), 1)
            self.assertEqual(run("--cpu-moe placement is CUDA-side"), 1)
            self.assertEqual(run("openssl not available"), 0)
            self.assertEqual(run("CUDA runtime is available"), 0)


class ShutdownGate(unittest.TestCase):
    def test_forced_kill_is_not_a_clean_shutdown(self):
        sys.path.insert(0, str(ROOT / "tests/conformance"))
        from _process import RunnerServer
        from _errors import TransportError
        class HungProcess:
            returncode = None
            def poll(self): return None
            def send_signal(self, sig): pass
            def wait(self, timeout):
                if self.returncode is None:
                    raise subprocess.TimeoutExpired("test-server", timeout)
                return self.returncode
            def kill(self): self.returncode = -9
        server = RunnerServer("unused", "unused")
        server.proc = HungProcess()
        server.sample_rss = lambda: None
        with self.assertRaises(TransportError):
            server.stop(strict=True)


class DeviceHarnessPins(unittest.TestCase):
    """Every harness that compares or certifies runner output computed on a
    device sets RUNNER_GPU_FALLBACK=fail: without it a device fault part-way
    through a run (the Q3_K decode fault of 2026-10-20) finishes the run on the
    CPU and the harness records host rows under the device's name."""
    HARNESSES = ("compare_llamacpp.py", "reference_compare.py", "compat_matrix.py",
                 "cpu_cuda_check.py", "cpu_cuda_delta.py", "kld-compare-raw.py")

    def test_device_harnesses_refuse_a_mid_run_fallback(self):
        for name in self.HARNESSES:
            src = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertRegex(src, r"RUNNER_GPU_FALLBACK[\"']?\]?\s*[=:]\s*[\"']fail[\"']",
                             f"{name} does not pin RUNNER_GPU_FALLBACK=fail")

    def test_the_engine_reads_the_switch_at_its_one_release_site(self):
        src = (ROOT / "src" / "model.c").read_text(encoding="utf-8")
        self.assertIn('getenv("RUNNER_GPU_FALLBACK")', src)
        self.assertEqual(src.count("gpu_disable(m);"), 1,
                         "a second runtime release site must go through gpu_runtime_failure")


# The bash on PATH, which is the one the scheduled task runs the nightly
# with. A bare "bash" is not that on Windows: CreateProcess looks in System32
# before PATH and finds WSL's launcher, which cannot see C:/ paths (the
# Windows run of these tests once failed on exactly that). Paths go to it in
# forward-slash form, which MSYS2's bash reads and its sourced env file keeps.
BASH = shutil.which("bash") or "bash"


class ScheduledEntryPoints(unittest.TestCase):
    """Scripts a lab host's scheduler runs from a copy of this repository.
    Nothing in the repository calls them, so a sweep for uncited files once
    removed cuda-nightly.sh and the nightly failed at startup for a week; the
    consumer (the Windows CUDA host's scheduled task, docs/ci.md) is the
    anchor this test stands in for."""

    def test_cuda_nightly_is_present_and_executable(self):
        p = ROOT / "scripts/cuda-nightly.sh"
        self.assertTrue(p.is_file(), "the scheduled nightly runs scripts/cuda-nightly.sh")
        # NTFS keeps no mode bits (Python reports none for a .sh there); the
        # Windows host runs the script through bash, so the bit is checked
        # where a filesystem has one
        if sys.platform != "win32":
            self.assertTrue(p.stat().st_mode & 0o111, "the nightly copy is run with bash, keep it executable")
        subprocess.run([BASH, "-n", p.as_posix()], check=True)

    def test_cuda_nightly_without_base_dir_is_a_usage_error(self):
        r = subprocess.run([BASH, (ROOT / "scripts/cuda-nightly.sh").as_posix()],
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("usage", r.stderr)

    def test_cuda_nightly_skips_a_held_lock_before_touching_the_checkout(self):
        # a lock whose first number is a live pid (this test's own) means
        # another user of the box is running: log a skip, exit 0, no git
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            lock = base / "box-gpu.lock"
            lock.write_text(f"{__import__('os').getpid()} held by the test\n")
            (base / "cuda-nightly.env").write_text(
                f"NIGHTLY_SKIP_LOCKS={lock.as_posix()}\nNIGHTLY_SRC={(base / 'no-checkout').as_posix()}\n")
            # on Windows, the minimal PATH a bare MSYS2 bash can start with:
            # no Windows directory on it, so the script must still find the
            # lock's (Windows) pid alive
            env = dict(os.environ)
            if sys.platform == "win32":
                env["PATH"] = str(pathlib.Path(BASH).parent)
            r = subprocess.run([BASH, (ROOT / "scripts/cuda-nightly.sh").as_posix(), base.as_posix()],
                               capture_output=True, text=True, timeout=60, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            hist = (base / "history.log").read_text()
            self.assertIn("skip:", hist)
            self.assertIn(lock.as_posix(), hist)
            self.assertFalse((base / "logs/fetch.log").exists())


if __name__ == "__main__":
    unittest.main()
