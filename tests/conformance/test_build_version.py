"""The build that answers, named on the wire (R4.12.21).

A supervisor that records which Runner produced a result, or refuses a build
below its floor, needs the version from the running process. Inferring it from
the feature set breaks as soon as two releases advertise the same features, and
asking the binary on disk answers for the file, not for the process listening
on the port. Both routes carry the exact `--version` string, and agree.
"""

import json
import re
import subprocess


def _get(client, path, name):
    r = client.get(path, name=name)
    assert r.status == 200, r.status
    return json.loads(r.body)


def _binary_version(server):
    out = subprocess.run([server.exe, "--version"], capture_output=True,
                         text=True, timeout=30, check=True).stdout.strip()
    # "runner 0.5.7" or, on a T3 build, "runner 0.5.7 (t3)"
    m = re.fullmatch(r"runner (\S+)(?: \((\S+)\))?", out)
    assert m, out
    return m.group(1), m.group(2)


def test_health_and_capabilities_name_the_build(client, server):
    version, flavor = _binary_version(server)
    health = _get(client, "/health", "version-health")
    caps = _get(client, "/v1/capabilities", "version-caps")
    for doc in (health, caps):
        assert doc.get("version") == version, doc
        # the flavor field exists only on a flavored build, as in receipts
        assert doc.get("build_flavor") == flavor, doc
