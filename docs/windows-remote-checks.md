# Windows remote verification

Use MSYS2 UCRT64 for native builds. The full Windows suite runs in the
scheduled/manually dispatched CI lane; native platform smokes also run on
pull requests. CUDA verification still needs an actual NVIDIA device.
See [CI execution and device evidence](ci.md) for the required lanes.

```sh
make OS=Windows_NT CC=gcc
make OS=Windows_NT -j2 test
```

## Remote CUDA helper

`scripts/cuda-smoke-remote.sh` drives the smoke on an explicitly supplied
host/binary/model and retrieves the report. Read its `--help`; the expected
version comes from the candidate's `runner --version`, not an old example.
A successful command must provide a nonzero set of executed device checks
and the correct candidate/model/hardware identities.

Some Windows SSH sessions use cmd.exe and do not forward remote stderr
reliably. Redirect the remote command into a file as a single invocation,
retrieve that file, and check the exit status. Silence is not a successful
compile or test. In cmd.exe, `if errorlevel 1` checks the executed command;
`%errorlevel%` in a compound one-liner may have been expanded earlier.

Use explicit argument quoting for cmd.exe, PowerShell, or the MSYS2 shell;
they are not interchangeable. The release workflow's Windows shell is
MSYS2. Preserve `SystemRoot` and the inherited environment when overriding
test variables so Winsock can initialize. Use UTF-8 explicitly for fixture
text and compare logical text versus bytes according to the gate's contract.

## Before trusting a failure

Ensure the test process can create its temporary directory. A scheduled
service/task context can have different permissions from an interactive
SSH session. Stop only test-owned processes; a leftover server can retain
its port or executable, making a later run fail before it tests anything.
Tests that must prove graceful shutdown fail if they need a forced kill.

Record actual native/compiler flags, executable/model hashes, driver/device,
commands, report paths, and executed/skipped checks. Update the device ledger
from the verified run using `scripts/device-evidence.py`; do not manufacture
fresh evidence by changing an old report's version or date.
