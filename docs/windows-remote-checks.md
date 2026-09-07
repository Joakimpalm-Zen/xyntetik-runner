# Windows remote-check protocol (learned the hard way, 2026-08-10)

`ssh zen@192.168.1.123` reaches ZEN-GAMING and runs cmd.exe, but **stderr from
the remote process is not forwarded**. `gcc --version` (stdout) comes back;
compiler diagnostics (stderr) do not, and neither does `> file 2>&1` reliably
when chained.

Silence therefore means nothing. A file with `return zzz;` produced exactly the
same empty output as a clean compile.

Use the exit code, which is trustworthy:

    gcc ... & if errorlevel 1 (echo FAILED) else (echo CLEAN)

`%errorlevel%` is expanded at parse time in a one-liner and is NOT a substitute.

Two further traps:
  - `;` is not a command separator in cmd; use `&` (or `&&`).
  - Single-file `-fsyntax-only` checks are under-specified for this project:
    the Windows build needs the Makefile's flags and backend defines. Baseline
    `cuda.c` from before any local change also fails a bare check, so a bare
    pass/fail says nothing about a diff. Compare baseline against changed with
    identical flags, or build properly.

The earlier note that `make` was absent is stale. Rechecked 2026-08-13:
`C:\msys64\usr\bin\make.exe` is present (GNU Make 4.4.1), and a fresh detached
Runner checkout completed the native MinGW/UCRT gate at `aa359e2` with
`make OS=Windows_NT -j2 test` (exit 0). The run passed the native, CUDA, and
tray-core gates; Python totals were client 30 passed/1 skipped, main 162/17,
MoE 28/0, and prune 9/0. No validation runner or tray process survived it.

That run found two Windows-only test-harness defects before it went green:
`d302a7a` makes the compatibility test's executable stub native on Windows,
and `aa359e2` stops the MTP admission test from raising a detached tray that
locks `runner.exe` before a later relink. Both were reproduced red on Windows,
then verified by targeted tests and the clean full gate. Do not install or
change the machine based on the old absence claim; verify the current tool
directly before diagnosing the setup.

The headless tray seams are also verified: menu dumping passed, and the idle,
loaded, and running BMPs have distinct hashes and pixel counts. Pixel analysis
confirmed that loaded preserves the idle icon's two opposing sweeps while
filling the core, and running draws the four-segment ring. This proves the GDI
raster states, not visibility or click behavior in the live Windows taskbar.

## The CUDA smoke gate (added 2026-08-29, after it was needed)

This box is a full CUDA build-and-run machine: `C:\msys64\ucrt64\bin\gcc.exe`,
`C:\msys64\usr\bin\make.exe`, CUDA Toolkit v13.3 with `nvcc` and `ptxas`,
Python 3.12, git, and GGUF models already in `C:\Users\zen\`. A fresh clone
builds with `make OS=Windows_NT -j2 runner`.

That matters because CI has no GPU, so `src/cuda.c` is compiled on three
platforms and run by none. v0.4.2 shipped an integrated-device probe reading
attribute 17 (`KERNEL_EXEC_TIMEOUT`, which is 1 whenever a display is attached)
instead of 18 (`INTEGRATED`), and reported every desktop NVIDIA card as unified
memory. **A headless datacenter GPU reads 17 as 0 and looks correct, so the
Blackwell would have passed it.** A display-attached consumer card is a test
surface nothing else in the lab has.

Run the gate before tagging any release:

    scripts/cuda-smoke-remote.sh \
        --binary C:/Users/zen/cuda-smoke/runner.exe \
        --model  C:/Users/zen/qwen3-0.6b-q8_0.gguf \
        --expect-version X.Y.Z \
        --report docs/compat-reports/cuda-smoke-X.Y.Z-<date>-rtx3070.json

It exits nonzero on failure and the report is committed with the release, the
same way the compatibility report is. The checks are invariants rather than
expected constants, so the same script is valid on a discrete card, a headless
card, and a unified-memory device: the central one is that `unified_memory`
must agree with the VRAM and RAM sizes actually reported, which is the v0.4.2
bug regardless of which attribute the driver was asked for.

Two protocol notes the driver script already handles, repeated because they
cost time every time they are rediscovered: the remote run redirects into a
file as the SOLE command of its ssh invocation, because redirects do not
survive `&` chaining and stderr is not forwarded; and `mkdir` in cmd.exe
rejects forward slashes even though scp and python accept them.

## Detached runs and building from cmd (added 2026-09-01)

A run longer than one ssh call's patience has to be detached, and two ways
that look right do not work: a PowerShell `Start-Process` launched over ssh
dies with the session (zero-byte logs, no process left), and there is no
`nohup` in cmd.exe. What works is a scheduled task. Write a `.bat` that
`cd`s and runs the command with its own `> log 2>&1` redirect, then:

    schtasks /create /tn <name> /tr C:\path\run.bat /sc once /st 00:00 /f & schtasks /run /tn <name>

The `/ST is earlier than current time` warning is harmless; `/run` starts
it now. Poll the log with `type` in separate calls and `schtasks /delete
/tn <name> /f` when done. Use the full interpreter path in the `.bat`
(`C:\Users\zen\AppData\Local\Programs\Python\Python312\python.exe -u`);
the bare `python` is a Store alias that resolves for cmd but not reliably
for a task.

`make` invoked straight from cmd cannot find `uname` or `mkdir` and stops
at the first object. Build from msys bash instead
(`C:\msys64\usr\bin\bash.exe C:/path/build.sh`) with
`PATH=/ucrt64/bin:/usr/bin:$PATH` exported inside the script, redirecting
`make` output to a file there, and echo `$?` so the exit code comes back
on stdout. A fresh checkout at `4731dc0` built this way in under a minute
with `make OS=Windows_NT -j4 runner`.

## The suite on this box, 2026-09-07

Ran again for the first time since 2026-08-13 and it did not build: `setenv`
is not in MinGW, so `tests/test_gpu_identity.c` and `tests/test_moe_mm_ab.c`
stopped `make test` at a compile error. Past that, the Python half read
**1107 passed, 8 failed, 1 error** on `main`, every one of them the harness
rather than the engine:

| what | why it is Windows-only |
|---|---|
| `test_weight_io_bench.py` x5 | `os.pread` does not exist; `mmap`'s `flags`/`prot` are Unix-only; a mapping offset must be 64 KB-aligned, not page-aligned; `os.open` needs `O_BINARY` or a weight file is newline-translated |
| `test_kv_quality.py` | `os.killpg` does not exist, and it raises `AttributeError`, which the `OSError` fallback beside it did not catch |
| `test_tool_choice_boundary.py` | the record hashes the template TEXT, the test hashed the file BYTES, and `write_text` lands CRLF |
| `conformance/test_prefill_deadline.py` | a test-supplied `env` replaced the inherited one; without `SystemRoot` a process cannot initialise Winsock and exits 1 before printing anything |
| `test_caps.py` | `--caps` has advertised NVFP4 on CUDA since 2026-09-06 and the expected list had not admitted it (the check only runs where a backend exists, and CI has no GPU) |

All fixed the same day; the suite reads **1130 passed, 67 skipped, 0 failed**
here now. The general lesson is the one worth keeping: **CI builds Windows and
never runs the suite there**, so a Windows-only harness defect is invisible
until somebody uses this box, and this box is the only CUDA device in the lab.
Whether that becomes a CI job or a scheduled run on the box is open (suite
plan R8.7.4).

Two operational notes from the same session. Running the suite under
`schtasks` fails wholesale: the task's session cannot create
`%LOCALAPPDATA%\Temp\pytest-of-zen`, so every `tmp_path` test errors
(363 of them) for a reason that has nothing to do with the code. Run it from
the interactive ssh session instead, and detach only what is genuinely long.
And check for a leftover `runner.exe` before trusting a conformance failure:
an interrupted run leaves a server holding its port, and the next run reports
`runner exited during startup` with an empty log.
