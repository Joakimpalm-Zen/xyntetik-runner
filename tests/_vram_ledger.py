"""The VRAM ledger a load writes, found and never assumed (R6.84).

A RUNNER_TEST_VRAM_FREE_MB claim goes into the real device's ledger whenever the
host has a device with an id, and into "test-device" only without one
(vram_ledger_device, src/vramreg.c). So the file's name is xyntetik-vram-<the
GPU's id>.reg on a CUDA host and xyntetik-vram-test-device.reg on a host with
no device id (hosted CI, the Mac's Metal). A test that names the file itself
passes on the second kind of host and fails on the first. The registry
directory is each test's own tmp directory, so the one ledger in it is the
runner's.
"""
import pathlib
import subprocess
import time

PATTERN = "xyntetik-vram-*.reg"


def ledgers(directory):
    """The ledgers in a registry directory (a yield sentinel, *.yield.<pid>, is not one)."""
    return sorted(pathlib.Path(directory).glob(PATTERN))


def the_ledger(directory, timeout=20):
    """The one ledger in the directory, waiting up to timeout seconds for it to appear.
    More than one is a failure: the test could not tell which is the runner's."""
    deadline = time.monotonic() + timeout
    while True:
        found = ledgers(directory)
        assert len(found) <= 1, f"more than one ledger in {directory}: {[p.name for p in found]}"
        if found:
            return found[0]
        if time.monotonic() >= deadline:
            raise AssertionError(f"no ledger ({PATTERN}) appeared in {directory}")
        time.sleep(0.05)


def probe_ledger_name(runner, model, directory, env, cwd):
    """The ledger name a load on this host writes: one short load into an empty registry directory.
    For a test that must write a claim before its own load, when there is no file to find yet."""
    assert not ledgers(directory), f"the probe needs an empty registry directory: {directory}"
    r = subprocess.run([str(runner), "-m", str(model), "-p", "hi", "-n", "1", "--temp", "0", "--no-tray"],
                       cwd=cwd, env=dict(env, RUNNER_VRAM_REGISTRY_DIR=str(directory)),
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
    assert r.returncode == 0, r.stderr.decode(errors="replace")[-800:]
    return the_ledger(directory, timeout=0).name
