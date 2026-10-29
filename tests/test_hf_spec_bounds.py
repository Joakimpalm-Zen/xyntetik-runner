"""-hf with an over-long repository name is refused before any download
(the owner--repo expansion used to overflow a 512-byte stack buffer)."""
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("spec,refusal", [
    ("a/" + "b" * 509, "too long"),            # fits the spec buffer, not the owner--repo expansion
    ("a/b/c/d/" + "e" * 400 + "/" * 100, "wants owner/repo"),   # refused by the format check first
])
def test_overlong_hf_spec_is_refused(runner_bin, spec, refusal, tmp_path):
    p = subprocess.run([str(runner_bin), "-hf", spec, "-p", "x", "-n", "1", "--gpu", "off"],
                       cwd=ROOT, env={"RUNNER_HF_CACHE": str(tmp_path), "PATH": "/usr/bin:/bin"},
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert p.returncode != 0
    err = p.stderr.decode(errors="replace")
    assert refusal in err, err   # the named refusal, not some later failure
    assert "AddressSanitizer" not in err


@pytest.mark.parametrize("spec", [
    "a b/c",            # a space ends up inside the API URL
    "a/b c",
    "a\\b/c",           # a backslash is a path separator on Windows and a literal here
    "a/b:c:d",          # '/' and ':' are all the format check looks at; the ':' stays in the repo name
    "own:er/repo",      # a drive-style colon (and an NTFS stream separator) in the cache directory
    "a/b?x=1",          # a query string in the repository name
    "a/b#frag",
    "a/b%2e",
    "a/é",         # non-ASCII is not a Hub repository name
    "./repo",
    "../repo",
    "owner/..",
    "owner/.",
])
def test_hf_spec_outside_the_hub_name_alphabet_is_refused(runner_bin, spec, tmp_path):
    """A Hub owner or repository name is letters, digits, '-', '_' and '.'.
    Anything else was passed into the request URL and used to name a cache
    directory; it is refused by name before either happens."""
    p = subprocess.run([str(runner_bin), "-hf", spec, "-p", "x", "-n", "1", "--gpu", "off"],
                       cwd=ROOT, env={"RUNNER_HF_CACHE": str(tmp_path), "PATH": "/usr/bin:/bin"},
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert p.returncode != 0
    err = p.stderr.decode(errors="replace")
    assert "wants owner/repo" in err, err
    assert list(tmp_path.iterdir()) == [], "a refused spec must not create a cache directory"
