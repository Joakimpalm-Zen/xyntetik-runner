# The shadow fixtures carry real pytest files on purpose (a frozen repair
# task and its protected tests); the verifier runs those in a scratch copy.
# They are not this suite's tests, so collection must not descend into them.
collect_ignore_glob = ["fixtures/*"]


# The suite exercises code whose whole job is to write into a user's home:
# hooks, configs, ledgers, delegation patches. Every one of those call sites
# takes a home or an --out, and every one of them has a default that is the
# real thing. One missing argument in one test is invisible until you look
# in your own ~/.xyntetik and find 127 fixture patches there, which is
# exactly what happened on 2026-09-11.
#
# So the suite does not run with the developer's home. HOME is redirected
# for the whole session, which is what `Path.home()` and `expanduser`
# resolve through, and a test that means to read the real one must say so.
import os
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def _home_is_not_yours() -> object:
    box = tempfile.mkdtemp(prefix="xyntetik-tests-home-")
    keep = {k: os.environ.get(k) for k in ("HOME", "USERPROFILE")}
    os.environ["HOME"] = box
    if os.name == "nt":
        os.environ["USERPROFILE"] = box
    try:
        yield box
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
