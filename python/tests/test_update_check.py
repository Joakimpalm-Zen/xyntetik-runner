"""The update check must be honest in both directions and never guess.

No test here touches the network: the fetcher is injected. The check's
contract is (a) a newer numbered tag reports newer, (b) same or older
reports up to date, (c) failure raises instead of silently reporting
"up to date" while offline, (d) pre-release suffixes do not confuse
the ordering of the numeric part.
"""
import pytest

from xyntetik_runner.update_check import check_latest, _version_tuple


def _fake(tag):
    return lambda: {"tag_name": tag, "html_url": f"https://example.test/{tag}"}


def test_newer_release_is_reported():
    st = check_latest("0.3.0", fetch=_fake("v0.4.0"))
    assert st.newer_available
    assert st.latest == "v0.4.0"


def test_same_version_is_up_to_date():
    assert not check_latest("0.3.0", fetch=_fake("v0.3.0")).newer_available


def test_older_published_tag_is_not_an_update():
    # a rollback or a stale API cache must not tell users to "upgrade" down
    assert not check_latest("0.3.0", fetch=_fake("v0.2.0")).newer_available


def test_prerelease_suffixes_order_by_numbers():
    assert _version_tuple("v0.1.20-alpha") == (0, 1, 20)
    assert check_latest("0.1.20-alpha", fetch=_fake("v0.3.0")).newer_available


def test_fetch_failure_raises_instead_of_lying():
    def down():
        raise OSError("no network")
    with pytest.raises(OSError):
        check_latest("0.3.0", fetch=down)


def test_garbage_tag_raises():
    with pytest.raises(ValueError):
        check_latest("0.3.0", fetch=_fake("release-tuesday"))
