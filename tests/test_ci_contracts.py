"""Build/gate contracts that must not silently turn a red run green."""
import pathlib
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


if __name__ == "__main__":
    unittest.main()
