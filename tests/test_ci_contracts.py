"""Build/gate contracts that must not silently turn a red run green."""
import os
import re
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


class HostedT3Skips(unittest.TestCase):
    """The t3 job runs its test files with --no-skips on hosted runners, which
    have no CUDA device, so every case there that skips for want of one must
    be deselected by t3's -k: a CUDA-only case added to test_transcript.py
    (b72a553b) without that kept t3 red for a day."""

    def test_cuda_only_cases_are_deselected(self):
        import ast
        import re
        wf = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        m = re.search(r"(tests/test_transcript\.py tests/test_train\.py "
                      r"tests/test_session_images\.py) \\\s*\n\s*-k \"([^\"]+)\"", wf)
        self.assertIsNotNone(m, "t3's pytest line moved: update this test with it")
        terms = re.findall(r"not (\S+)", m.group(2))
        for f in m.group(1).split():
            tree = ast.parse((ROOT / f).read_text(encoding="utf-8"))
            for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name.startswith("test_")):
                cuda_skip = any(
                    isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "skip"
                    and c.args and isinstance(c.args[0], ast.Constant)
                    and "CUDA" in str(c.args[0].value)
                    for c in ast.walk(fn))
                if cuda_skip:
                    self.assertTrue(any(t in fn.name for t in terms),
                                    f"{f}::{fn.name} skips without CUDA but t3's -k keeps it")


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


class _CudaGateCopy:
    """Runs a copy of cuda-gate.sh from a scratch directory, so a box's own
    env file beside the checkout's copy is never read."""

    SETTINGS = ("GATE_MODEL", "GATE_PYTHON", "RUNNER_LLAMA_CPP_BIN", "RUNNER_IQ_FIXTURES")

    def run_gate(self, env_file=None, prepare=None, bash_args=()):
        """prepare(base) makes what a case needs in the scratch directory and
        returns environment changes (None removes a variable); the summary
        JSON the run wrote is kept in self.summary."""
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            (base / "scripts").mkdir()
            (base / "build").mkdir()
            gate = base / "scripts/cuda-gate.sh"
            shutil.copyfile(ROOT / "scripts/cuda-gate.sh", gate)
            env_path = base / "scripts/cuda-gate.env"
            if env_file is not None:
                env_path.write_bytes(env_file(base).encode())
            env = {k: v for k, v in os.environ.items()
                   if not k.upper().startswith("GATE_") and k.upper() not in self.SETTINGS}
            for name, value in (prepare(base) if prepare else {}).items():
                for k in [k for k in env if k.upper() == name.upper()]:
                    del env[k]
                if value is not None:
                    env[name] = value
            r = subprocess.run([BASH, *bash_args, gate.as_posix(), (base / "build").as_posix(),
                                (base / "out").as_posix()],
                               capture_output=True, text=True, timeout=60, env=env)
            self.summary = [json.loads(f.read_text(encoding="utf-8"))
                            for f in sorted((base / "out").glob("cuda-gate-*.json"))]
        return r, base.as_posix(), env_path.as_posix()


class CudaGateExplainsNotRun(_CudaGateCopy, unittest.TestCase):
    """cuda-gate.sh takes its settings from the environment and from
    cuda-gate.env beside the script (or the file GATE_ENV names). On the
    Windows CUDA host variables exported in a Git Bash shell did not reach
    the MSYS2 bash the gate ran under, and the gate said only "no model for
    the cpu-cuda step (set GATE_MODEL)". A gate that cannot run for want of
    a setting names the env file it looked for, whether it loaded it, which
    settings it has, and the shell that ran it."""

    def test_no_env_file_names_it_and_the_missing_settings(self):
        r, _, env_path = self.run_gate()
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        out = r.stdout
        self.assertIn("env file: %s (not found)" % env_path, out)
        self.assertIn("GATE_MODEL is not set", out)
        self.assertIn("RUNNER_LLAMA_CPP_BIN is not set", out)
        for name in ("GATE_MODEL", "GATE_PYTHON", "RUNNER_LLAMA_CPP_BIN"):
            self.assertRegex(out, r"settings:.* %s unset" % name)
        self.assertRegex(out, r"shell: \S*bash")
        self.assertIn("result: not-run", out)

    def test_a_loaded_env_file_with_a_missing_model_says_both(self):
        r, base, env_path = self.run_gate(lambda b: "GATE_MODEL=%s/no-such-model.gguf\n" % b.as_posix())
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        out = r.stdout
        model = base + "/no-such-model.gguf"
        self.assertIn("env file: %s (found and loaded)" % env_path, out)
        self.assertIn("GATE_MODEL %s does not exist" % model, out)
        self.assertRegex(out, r"settings:.* GATE_MODEL=%s" % re.escape(model))
        self.assertNotIn("GATE_MODEL is not set", out)

    @unittest.skipIf(os.name == "nt", "whether MSYS2's bash keeps the carriage return is its own")
    def test_a_crlf_env_file_is_named(self):
        r, _, _ = self.run_gate(lambda b: "GATE_MODEL=%s/m.gguf\r\n" % b.as_posix())
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("carriage return", r.stdout)


PYTHON = pathlib.Path(sys.executable).as_posix()


class CudaGateRefusesADegradedLaunch(_CudaGateCopy, unittest.TestCase):
    r"""On the Windows CUDA host a tool shell started MSYS2 bash with almost
    none of the Windows environment: no USERNAME, no TEMP or TMP. Python's
    tempfile fell back to C:\Windows\Temp (its last resort there is
    %SystemRoot%\Temp), getpass.getuser() raised (pytest's directory became
    pytest-of-unknown), and the gate ran about 27 times slower than under a
    login shell on the same host (iquants 1,607 s against 59 s), with two
    20-second subprocess timeouts that were not hangs. Before its first step
    the gate asks its own python for the temp directory and the user, refuses
    (not-run, exit 2) a degraded answer, naming each check, the values it saw
    and the remedy, and records what it saw in its summary JSON.
    GATE_ALLOW_DEGRADED_ENV=1 makes each refusal a warning. An empty user
    variable alone is a warning, not a refusal: a Linux container without
    USER still has a user python resolves, and only a gate with no user name
    at all (the variable empty and getuser() raising) is refused.

    Python's own fallback is the anchor of the POSIX case: tempfile skips a
    directory it cannot write to without a word (a read-only TMPDIR resolves
    to /tmp), so the check compares the directory the environment names with
    the one python uses."""

    @unittest.skipIf(os.name == "nt", "NTFS keeps no mode bits: a read-only TMPDIR is a POSIX fixture")
    def test_a_tmpdir_python_cannot_write_is_refused(self):
        def prepare(base):
            ro = base / "read-only"
            ro.mkdir()
            ro.chmod(0o555)
            if os.access(ro, os.W_OK):
                self.skipTest("this user writes to a read-only directory (root)")
            return {"TMPDIR": ro.as_posix(), "USER": "gate", "GATE_PYTHON": PYTHON}
        r, base, _ = self.run_gate(prepare=prepare)
        out = r.stdout
        self.assertEqual(r.returncode, 2, out + r.stderr)
        self.assertIn("not run: temp directory: TMPDIR=%s/read-only is missing or not writable"
                      % base, out)
        self.assertRegex(out, r"launch environment: temp directory /\S+, user \S+")
        self.assertIn("remedy: launch from a login shell", out)
        self.assertIn("result: not-run", out)
        self.assertEqual(self.summary[0]["env"]["degraded"], ["temp directory"])

    # every name getpass.getuser() reads before it asks the passwd database
    NO_USER = {"USER": None, "USERNAME": None, "LOGNAME": None, "LNAME": None}

    def no_passwd_entry(self, base):
        """A stand-in python: the gate's python with a sitecustomize that
        makes the passwd lookup fail, as for a container uid with no entry,
        so getuser() raises once the variables are gone. Returns the
        PYTHONPATH entry."""
        d = base / "no-passwd"
        d.mkdir(exist_ok=True)
        (d / "sitecustomize.py").write_text(
            "import pwd\n"
            "def _no_entry(uid):\n"
            "    raise KeyError('getpwuid(): uid not found: %d' % uid)\n"
            "pwd.getpwuid = _no_entry\n")
        return d.as_posix()

    def ready(self, base, **extra):
        """Everything the preflight asks for, so the next check after it (the
        stand-in runner's --caps) is where a run that goes on stops."""
        runner = base / "build/runner"
        runner.write_text("#!/bin/sh\necho '{\"gpu\": null}'\n")
        runner.chmod(0o755)
        (base / "m.gguf").write_bytes(b"")
        (base / "llama").mkdir()
        (base / "stand-in").mkdir()
        (base / "stand-in/pytest.py").write_text("")  # the preflight only imports it
        (base / "tmp").mkdir()
        env = {"GATE_PYTHON": PYTHON, "GATE_MODEL": (base / "m.gguf").as_posix(),
               "TMPDIR": (base / "tmp").as_posix(),
               "RUNNER_LLAMA_CPP_BIN": (base / "llama").as_posix(),
               "PYTHONPATH": (base / "stand-in").as_posix()}
        env.update(extra)
        return env

    @unittest.skipIf(os.name == "nt", "on Windows getuser() has nothing else to read once USERNAME is gone")
    def test_an_empty_user_variable_that_python_resolves_warns_and_goes_on(self):
        import pwd
        name = pwd.getpwuid(os.getuid()).pw_name  # the passwd database, not the gate
        r, _, _ = self.run_gate(prepare=lambda b: self.ready(b, **self.NO_USER))
        out = r.stdout
        self.assertEqual(r.returncode, 2, out + r.stderr)  # no CUDA here: --caps stops it
        self.assertIn("warning: user name: USER is empty, python's getpass.getuser() resolves %s"
                      % name, out)
        self.assertNotIn("not run: user name", out)
        self.assertIn("caps reports no CUDA device", out)
        env = self.summary[0]["env"]
        self.assertEqual(env["user"], name)
        self.assertEqual(env["degraded"], [])
        self.assertEqual(env["warned"], ["user name"])

    def test_no_user_name_at_all_is_refused(self):
        var = "USERNAME" if os.name == "nt" else "USER"
        def prepare(base):
            env = dict(self.NO_USER, GATE_PYTHON=PYTHON)
            if os.name != "nt":  # Windows python has no passwd database to fall back on
                env["PYTHONPATH"] = self.no_passwd_entry(base)
            return env
        r, _, _ = self.run_gate(prepare=prepare)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("not run: user name: %s is empty and python's getpass.getuser() raises ("
                      % var, r.stdout)
        self.assertIn("remedy: launch from a login shell", r.stdout)
        self.assertIn("user name", self.summary[0]["env"]["degraded"])
        self.assertEqual(self.summary[0]["env"]["user"], "")

    @unittest.skipIf(os.name == "nt", "the stand-in runner is a POSIX script")
    def test_the_escape_hatch_warns_and_goes_on_to_the_next_check(self):
        def prepare(base, **extra):
            env = self.ready(base, **self.NO_USER, **extra)
            env["PYTHONPATH"] += os.pathsep + self.no_passwd_entry(base)
            return env
        r, _, _ = self.run_gate(prepare=prepare)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("not run: user name: USER is empty and python's getpass.getuser() raises",
                      r.stdout)
        self.assertNotIn("caps reports", r.stdout)
        r, base, _ = self.run_gate(prepare=lambda b: prepare(b, GATE_ALLOW_DEGRADED_ENV="1"))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("warning: user name: USER is empty and python's getpass.getuser() raises",
                      r.stdout)
        self.assertNotIn("not run: user name", r.stdout)
        self.assertIn("caps reports no CUDA device", r.stdout)
        env = self.summary[0]["env"]
        self.assertEqual(env["degraded"], ["user name"])
        self.assertEqual(env["tempdir"], base + "/tmp")
        self.assertRegex(env["shell"], r"bash")
        self.assertIs(env["login_shell"], False)

    @unittest.skipIf(os.name == "nt", "stubs uname with a POSIX script; "
                     "on Windows the real uname takes this branch")
    def test_the_windows_branch(self):
        r"""The Windows directory comes from SYSTEMROOT, compared without
        regard to case or slash direction (C:\WINDOWS against python's
        C:\Windows\Temp on the host), and MSYS2 needs MSYSTEM, and HOME
        unless the shell is a login shell. The uname strings are the
        platforms' own (PlatformDetection)."""
        def prepare(base, **extra):
            stub = base / "bin/uname"
            stub.parent.mkdir()
            stub.write_text("#!/bin/sh\necho MSYS_NT-10.0-19045\n")
            stub.chmod(0o755)
            (base / "Windows/Temp").mkdir(parents=True)
            (base / "tmp").mkdir()
            env = {"PATH": (base / "bin").as_posix() + os.pathsep + os.environ["PATH"],
                   "SYSTEMROOT": (base / "WINDOWS").as_posix().replace("/", "\\").upper(),
                   "TMPDIR": (base / "Windows/Temp").as_posix(), "TEMP": None, "TMP": None,
                   "USERNAME": None, "MSYSTEM": None, "HOME": None, "GATE_PYTHON": PYTHON}
            env.update(extra)
            return env
        r, base, _ = self.run_gate(prepare=prepare)
        out = r.stdout
        self.assertEqual(r.returncode, 2, out + r.stderr)
        root = (base + "/WINDOWS").replace("/", "\\").upper()
        self.assertIn("not run: temp directory: %s/Windows/Temp is under the Windows directory %s"
                      % (base, root), out)
        # USER is still set, so python resolves a name: a warning only
        self.assertIn("warning: user name: USERNAME is empty, python's getpass.getuser() resolves", out)
        self.assertIn("not run: MSYS2 shell: MSYSTEM is not set", out)
        self.assertIn("not run: MSYS2 shell: not a login shell and HOME is not set", out)
        self.assertIn("MSYSTEM unset", out)
        self.assertIn(r"remedy: launch from a login shell (PowerShell: "
                      r"C:\msys64\usr\bin\bash.exe -lc '...') or set TEMP/TMP and USERNAME", out)
        self.assertEqual(self.summary[0]["class"], "windows-x86_64-cuda")
        self.assertEqual(self.summary[0]["env"]["degraded"], ["temp directory", "MSYS2 shell"])
        self.assertEqual(self.summary[0]["env"]["warned"], ["user name"])
        for case, home, args in (("HOME set", True, ()),
                                 ("a login shell without HOME", False, ("--noprofile", "-l"))):
            with self.subTest(case):
                r, _, _ = self.run_gate(prepare=lambda b: prepare(
                    b, TMPDIR=(b / "tmp").as_posix(), MSYSTEM="UCRT64", USERNAME="gate",
                    HOME=b.as_posix() if home else None), bash_args=args)
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                self.assertNotIn("launch environment", r.stdout)
                self.assertNotIn("remedy", r.stdout)
                self.assertEqual(self.summary[0]["env"]["degraded"], [])
                self.assertEqual(self.summary[0]["env"]["warned"], [])
                self.assertEqual(self.summary[0]["env"]["msystem"], "UCRT64")
                self.assertIs(self.summary[0]["env"]["login_shell"], bool(args))


class LlamaQuantizeByName(unittest.TestCase):
    """The CUDA gate steps that quantize fixtures run llama-quantize from
    RUNNER_LLAMA_CPP_BIN, resolved by name. A llama.cpp Windows release
    directory (b11748) holds llama-quantize-impl.dll beside
    llama-quantize.exe; under C collation "-" (0x2D) sorts before "."
    (0x2E), so a `llama-quantize*` glob's first match was the DLL and the
    1.5.2 gate on the Windows CUDA host tried to execute it. Under a UTF-8
    locale the exe sorted first, which is why the same step had passed.

    The byte order of the two names is the absolute anchor; the helper is
    held through `make llama-quantize-path`, which prints what the steps
    run."""

    QUANTIZE_TARGETS = ("test-cuda-tc-kq", "test-cuda-moe-legacy",
                        "test-cuda-batch-identity", "test-cuda-spec-identity")

    def resolve(self, names, bin_dir=True):
        env = {k: v for k, v in os.environ.items() if k != "RUNNER_LLAMA_CPP_BIN"}
        env["LC_ALL"] = "C"
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            for n in names:
                (d / n).write_text("#!/bin/sh\nexit 0\n")
                (d / n).chmod(0o755)
            if bin_dir:
                env["RUNNER_LLAMA_CPP_BIN"] = d.as_posix()
            p = subprocess.run(["make", "-s", "--no-print-directory", "llama-quantize-path"],
                               cwd=ROOT, env=env, capture_output=True, text=True)
        return p, d.as_posix()

    def test_the_exe_beside_the_impl_dll_under_c_collation(self):
        self.assertLess(b"llama-quantize-impl.dll", b"llama-quantize.exe")
        p, d = self.resolve(["llama-quantize-impl.dll", "llama-quantize.exe"])
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), d + "/llama-quantize.exe")

    @unittest.skipIf(os.name == "nt", "NTFS keeps no mode bits: whether MSYS2 calls a bare name executable is its own")
    def test_the_bare_name(self):
        p, d = self.resolve(["llama-quantize"])
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), d + "/llama-quantize")

    def test_neither_name_fails_naming_the_directory_and_both_names(self):
        p, d = self.resolve(["llama-quantize-impl.dll"])
        self.assertNotEqual(p.returncode, 0)
        self.assertEqual(p.stdout.strip(), "")
        for word in (d, "llama-quantize ", "llama-quantize.exe", "RUNNER_LLAMA_CPP_BIN"):
            self.assertIn(word, p.stderr)
        p, _ = self.resolve([], bin_dir=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("RUNNER_LLAMA_CPP_BIN is not set", p.stderr)

    def test_every_quantize_step_uses_the_helper(self):
        p = subprocess.run(["make", "-n", "--no-print-directory", *self.QUANTIZE_TARGETS],
                           cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertNotIn("llama-quantize*", p.stdout)
        self.assertEqual(p.stdout.count("neither llama-quantize nor llama-quantize.exe"),
                         len(self.QUANTIZE_TARGETS))


class PlatformDetection(unittest.TestCase):
    """The Windows branch is chosen from the shell the build runs in, not
    from an environment variable a minimal shell may not carry. A bash.exe
    started directly on the Windows CUDA host had no OS variable, took the
    POSIX branch, and failed linking `.build/.../runner` (no .exe) against
    -ldl, which MinGW does not have.

    The uname strings are the platforms' own (the absolute anchor): the MSYS2
    runtime prints MSYS_NT-<ver> with MSYSTEM unset and MINGW64_NT-<ver> or
    MINGW32_NT-<ver> under any other MSYSTEM (UCRT64 and CLANG64 included),
    Git for Windows prints MINGW64_NT-<ver>, Cygwin CYGWIN_NT-<ver>, and the
    busybox-w32 uname of w64devkit prints Windows_NT."""

    WINDOWS_UNAMES = ("MINGW64_NT-10.0-19045", "MINGW32_NT-10.0-19045",
                      "MSYS_NT-10.0-19045", "CYGWIN_NT-10.0-19045",
                      "Windows_NT")

    def make_n(self, target, uname=None, args=()):
        env = {k: v for k, v in os.environ.items() if k.upper() != "OS"}
        with tempfile.TemporaryDirectory() as td:
            if uname is not None:
                stub = pathlib.Path(td) / "uname"
                stub.write_text("#!/bin/sh\necho '%s'\n" % uname)
                stub.chmod(0o755)
                env["PATH"] = td + os.pathsep + env["PATH"]
            p = subprocess.run(["make", "-Bn", target, *args], cwd=ROOT, env=env,
                               capture_output=True, text=True, check=True)
        return p.stdout.replace("\\\n", " ").splitlines()

    def runner_link(self, uname=None, args=()):
        lines = self.make_n("runner", uname, args)
        # the line that links the executable itself, not one that merely
        # names a runner file (windres reads src/runner.rc on Windows)
        return next(x for x in lines
                    if re.search(r" -o \.build/[0-9]+/runner(\.exe)?( |$)", x))

    def assert_windows(self, line):
        self.assertRegex(line, r" -o \.build/[0-9]+/runner\.exe ")
        self.assertIn("-lws2_32", line)
        self.assertNotIn("-ldl", line.split())
        # the UTF-8 application manifest (src/runner.rc): without it the C
        # runtime reads a non-ASCII -p prompt in the ANSI code page
        self.assertRegex(line, r"\.build/[0-9]+/win_manifest\.o ")

    def assert_posix(self, line):
        self.assertRegex(line, r" -o \.build/[0-9]+/runner ")
        self.assertNotIn("-lws2_32", line)

    @unittest.skipIf(os.name == "nt", "stubs uname with a POSIX script; "
                     "the Windows job runs the real-uname case below")
    def test_windows_uname_selects_windows_without_os_variable(self):
        for uname in self.WINDOWS_UNAMES:
            with self.subTest(uname=uname):
                self.assert_windows(self.runner_link(uname))
                tsan = next(x for x in self.make_n("test-decide-race-gpu", uname)
                            if "-o test-decide-race-gpu-bin" in x)
                self.assertNotIn("-fsanitize=thread", tsan)

    @unittest.skipIf(os.name == "nt", "stubs uname with a POSIX script")
    def test_posix_uname_keeps_its_branch(self):
        linux = self.runner_link("Linux")
        self.assert_posix(linux)
        self.assertIn("-ldl", linux.split())
        darwin = self.runner_link("Darwin")
        self.assert_posix(darwin)
        self.assertIn("Metal", darwin)
        self.assertNotIn("-ldl", darwin.split())

    @unittest.skipIf(os.name == "nt", "stubs uname with a POSIX script")
    def test_posix_links_no_manifest(self):
        for uname in ("Linux", "Darwin"):
            with self.subTest(uname=uname):
                self.assertNotIn("win_manifest", self.runner_link(uname))

    @unittest.skipIf(os.name == "nt", "stubs uname with a POSIX script")
    def test_a_missing_windres_stops_the_build_with_a_reason(self):
        """A runner.exe without the manifest builds and runs, and misreads
        every non-ASCII prompt, so the missing tool is an error that says
        what it costs, never a silent fallback."""
        uname = self.WINDOWS_UNAMES[0]
        obj = next(w for x in self.make_n("runner", uname) for w in x.split()
                   if w.endswith("/win_manifest.o"))
        env = {k: v for k, v in os.environ.items() if k.upper() != "OS"}
        with tempfile.TemporaryDirectory() as td:
            stub = pathlib.Path(td) / "uname"
            stub.write_text("#!/bin/sh\necho '%s'\n" % uname)
            stub.chmod(0o755)
            env["PATH"] = td + os.pathsep + env["PATH"]
            p = subprocess.run(["make", "-B", obj, "WINDRES=definitely-not-windres"],
                               cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("'definitely-not-windres' not found", p.stderr)
        self.assertIn("UTF-8 manifest", p.stderr)
        self.assertFalse((ROOT / obj).exists())

    @unittest.skipIf(os.name == "nt", "stubs uname with a POSIX script")
    def test_os_windows_nt_still_selects_windows(self):
        # mingw32-make from cmd.exe has no uname; Windows sets OS there
        self.assert_windows(self.runner_link("Linux", ["OS=Windows_NT"]))

    @unittest.skipUnless(os.name == "nt", "the real-platform half of the "
                         "stubbed cases above; runs in the Windows job")
    def test_real_windows_shell_without_os_variable(self):
        self.assert_windows(self.runner_link())


if __name__ == "__main__":
    unittest.main()
