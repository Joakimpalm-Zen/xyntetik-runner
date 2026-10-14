"""Every template family names the fixture that reaches its code path.

A per-family behaviour test is only as good as its fixture: the plain
fixture under `--chat-template harmony` is refused Messages' `thinking`
(Harmony's reasoning is a channel only gpt-oss's architecture has), so a
test of Harmony's reasoning run on it checks nothing, and passed. Program B's
fixture test covered five think-tag families and missed Harmony and Muse
(register item 23). tests/family_fixtures.json maps every family to the
fixture that reaches it; this test holds the map to the families the runner
reports, and checks that each channel family's fixture is admitted where the
plain one is not.
"""
import json
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer, find_runner  # noqa: E402

TABLE = json.loads((ROOT / "tests" / "family_fixtures.json").read_text(encoding="utf-8"))


def _template_ids():
    src = (ROOT / "src" / "template.c").read_text(encoding="utf-8")
    i = src.index("const char *template_name(int t) {")
    return set(re.findall(r'return "([a-z0-9-]+)"', src[i:i + 3000].split("\n}\n")[0]))


def test_every_family_the_runner_reports_has_a_fixture():
    ids = _template_ids()
    assert "chatml" in ids and "harmony" in ids, ids
    fams = set(TABLE["families"])
    assert ids - fams == set(), f"families with no fixture: {sorted(ids - fams)}"
    assert fams - ids == set(), f"rows for no family: {sorted(fams - ids)}"
    for name, row in TABLE["families"].items():
        assert row["fixture"] in TABLE["fixtures"], (name, row)
        assert row["reasoning"] in ("none", "tags", "channel"), (name, row)
        if row["reasoning"] == "channel":
            assert row["fixture"] != "plain", (name, "a channel family needs its architecture")
    for name, fx in TABLE["fixtures"].items():
        assert (ROOT / fx["make"][0]).exists(), (name, fx)


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    out = tmp_path_factory.mktemp("family-fixtures")
    built = {}

    def get(name):
        if name not in built:
            fx = TABLE["fixtures"][name]
            target = out / fx.get("prefix", fx["file"])
            subprocess.run([sys.executable, ROOT / fx["make"][0], *fx["make"][1:], str(target)],
                           check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
            built[name] = out / fx["file"]
        return built[name]
    return get


def _messages_thinking(exe, model, family):
    with RunnerServer(exe, model, ctx=2048, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2",
                                  "--chat-template", family]) as srv:
        mid = json.load(urllib.request.urlopen(srv.base_url + "/v1/models"))["data"][0]["id"]
        body = {"model": mid, "max_tokens": 4, "temperature": 0,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "messages": [{"role": "user", "content": "Hello"}]}
        req = urllib.request.Request(srv.base_url + "/v1/messages", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code


CHANNEL = sorted(n for n, r in TABLE["families"].items() if r["reasoning"] == "channel")


@pytest.mark.parametrize("family", CHANNEL)
def test_a_channel_family_is_reached_on_its_fixture_and_not_on_the_plain_one(made, family):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    own = made(TABLE["families"][family]["fixture"])
    assert _messages_thinking(exe, own, family) == 200, family
    # the blind spot the table exists for
    assert _messages_thinking(exe, made("plain"), family) == 400, family
