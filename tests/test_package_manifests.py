"""The scoop and winget manifests for the released Windows binary (R7.8.2, R7.8.3).

Both are rendered by scripts/package-manifests.py from the release's own SHA256SUMS,
so each names the asset the release published, by the digest the release published.
These tests render them from a stand-in SHA256SUMS, refuse one that lacks the
Windows asset, catch a manifest edited away from its render, and hold the committed
pair to one version and one digest.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package-manifests.py"
EXE_SHA = "ab" * 32
# the Windows lines carry the binary-mode '*' the release's sha256sum writes
SUMS = (f"{'11' * 32}  runner-linux-x86_64\n"
        f"{EXE_SHA} *runner-windows-x86_64.exe\n"
        f"{'22' * 32} *runner-windows-x86_64.exe.zip\n")


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True, timeout=60)


def _render(tmp_path, version="9.9.9"):
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(SUMS, encoding="utf-8")
    r = _run("--write", version, "--sums", sums, "--date", "2026-01-02", "--out", tmp_path)
    assert r.returncode == 0, r.stderr
    return tmp_path


def test_both_manifests_render_from_the_release_sums(tmp_path):
    root = _render(tmp_path)
    scoop = json.loads((root / "bucket" / "xyntetik-runner.json").read_text())
    url = "https://github.com/Joakimpalm-Zen/xyntetik-runner/releases/download/v9.9.9/runner-windows-x86_64.exe"
    assert scoop["version"] == "9.9.9"
    assert scoop["architecture"]["64bit"] == {"url": url, "hash": EXE_SHA}
    assert scoop["bin"] == [["runner-windows-x86_64.exe", "runner"]]
    vdir = root / "packaging/winget/manifests/x/Xyntetik/Runner/9.9.9"
    inst = (vdir / "Xyntetik.Runner.installer.yaml").read_text()
    assert f"InstallerUrl: {url}" in inst and f"InstallerSha256: {EXE_SHA.upper()}" in inst
    assert "InstallerType: portable" in inst and "- runner" in inst and "ReleaseDate: 2026-01-02" in inst
    assert {p.name for p in vdir.iterdir()} == {
        "Xyntetik.Runner.yaml", "Xyntetik.Runner.installer.yaml", "Xyntetik.Runner.locale.en-US.yaml"}
    assert _run("--check", "--out", root).returncode == 0


def test_a_new_version_replaces_the_winget_folder(tmp_path):
    _render(tmp_path, "9.9.9")
    root = _render(tmp_path, "9.9.10")
    names = [p.name for p in (root / "packaging/winget/manifests/x/Xyntetik/Runner").iterdir()]
    assert names == ["9.9.10"]
    assert _run("--check", "--out", root).returncode == 0


def test_sums_without_the_windows_asset_are_refused(tmp_path):
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{'11' * 32}  runner-linux-x86_64\n", encoding="utf-8")
    r = _run("--write", "9.9.9", "--sums", sums, "--date", "2026-01-02", "--out", tmp_path)
    assert r.returncode != 0 and "no line for runner-windows-x86_64.exe" in r.stderr
    assert not (tmp_path / "bucket").exists()


def test_a_manifest_edited_away_from_its_render_fails_the_check(tmp_path):
    root = _render(tmp_path)
    inst = root / "packaging/winget/manifests/x/Xyntetik/Runner/9.9.9/Xyntetik.Runner.installer.yaml"
    inst.write_text(inst.read_text().replace(EXE_SHA.upper(), "CD" * 32))
    r = _run("--check", "--out", root)
    assert r.returncode == 1 and "different sha256s" in r.stderr


def test_the_committed_manifests_agree():
    r = _run("--check")
    assert r.returncode == 0, r.stderr
