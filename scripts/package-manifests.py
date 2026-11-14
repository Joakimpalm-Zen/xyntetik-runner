#!/usr/bin/env python3
"""Package-manager manifests for the released Windows binary (R7.8.2, R7.8.3).

    scripts/package-manifests.py --write X.Y.Z [--sums FILE] [--date YYYY-MM-DD] [--out DIR]
    scripts/package-manifests.py --check [--out DIR]

--write renders both manifests from the release's own SHA256SUMS, downloaded
with `gh release download` unless --sums names a copy:

- scoop: bucket/xyntetik-runner.json. The bucket is this repository, so
  `scoop bucket add xyntetik https://github.com/Joakimpalm-Zen/xyntetik-runner`
  then `scoop install xyntetik/xyntetik-runner` installs the release asset with
  a `runner` shim on PATH. scoop checks the sha256 the manifest carries.
- winget: packaging/winget/manifests/x/Xyntetik/Runner/X.Y.Z/, the three files
  of a portable-exe package laid out as microsoft/winget-pkgs wants them, so a
  submission copies the folder. winget checks the InstallerSha256.

--check reads them back, offline: one version, the same sha256 in both, and the
release URL that version publishes. --out writes or reads under another root
(the tests).
"""
import argparse
import datetime
import json
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = "Joakimpalm-Zen/xyntetik-runner"
ASSET = "runner-windows-x86_64.exe"
WINGET_ID = "Xyntetik.Runner"
SCOOP = pathlib.Path("bucket") / "xyntetik-runner.json"
WINGET = pathlib.Path("packaging") / "winget" / "manifests" / "x" / "Xyntetik" / "Runner"
SCHEMA = "1.9.0"
DESCRIPTION = "Local LLM inference engine for GGUF models, with signed, replayable records of what ran."


def release_url(version, asset=ASSET):
    return f"https://github.com/{REPO}/releases/download/v{version}/{asset}"


def parse_sums(text):
    """SHA256SUMS lines: a 64-hex digest, then the name after two spaces or a
    space and the binary-mode '*' (the Windows lines carry it)."""
    out = {}
    for line in text.splitlines():
        m = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](\S+)", line.strip())
        if m:
            out[m.group(2)] = m.group(1).lower()
    return out


def scoop_manifest(version, sha):
    return {
        "version": version,
        "description": DESCRIPTION,
        "homepage": f"https://github.com/{REPO}",
        "license": "Apache-2.0",
        "architecture": {"64bit": {"url": release_url(version), "hash": sha}},
        "bin": [[ASSET, "runner"]],
        "checkver": {"github": f"https://github.com/{REPO}"},
        "autoupdate": {"architecture": {"64bit": {
            "url": release_url("$version"),
            "hash": {"url": f"https://github.com/{REPO}/releases/download/v$version/SHA256SUMS"}}}},
    }


def _yaml(kind, lines):
    head = f"# yaml-language-server: $schema=https://aka.ms/winget-manifest.{kind}.{SCHEMA}.schema.json\n\n"
    return head + "\n".join(lines) + f"\nManifestType: {kind}\nManifestVersion: {SCHEMA}\n"


def winget_manifests(version, sha, date):
    ident = [f"PackageIdentifier: {WINGET_ID}", f"PackageVersion: {version}"]
    return {
        f"{WINGET_ID}.yaml": _yaml("version", ident + ["DefaultLocale: en-US"]),
        f"{WINGET_ID}.installer.yaml": _yaml("installer", ident + [
            "InstallerType: portable",
            "Commands:",
            "- runner",
            f"ReleaseDate: {date}",
            "Installers:",
            "- Architecture: x64",
            f"  InstallerUrl: {release_url(version)}",
            f"  InstallerSha256: {sha.upper()}",
        ]),
        f"{WINGET_ID}.locale.en-US.yaml": _yaml("defaultLocale", ident + [
            "PackageLocale: en-US",
            "Publisher: Xyntetik",
            "PublisherUrl: https://xyntetik.com",
            "PackageName: Xyntetik Runner",
            f"PackageUrl: https://github.com/{REPO}",
            "License: Apache-2.0",
            f"LicenseUrl: https://github.com/{REPO}/blob/main/LICENSE",
            f"ShortDescription: {DESCRIPTION}",
            f"ReleaseNotesUrl: https://github.com/{REPO}/releases/tag/v{version}",
            "Tags:",
            "- gguf",
            "- inference",
            "- llm",
        ]),
    }


def _fetch_sums(version):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["gh", "release", "download", f"v{version}", "-R", REPO, "-p", "SHA256SUMS",
                        "-D", d], check=True)
        return (pathlib.Path(d) / "SHA256SUMS").read_text(encoding="utf-8")


def write(version, sums_text, date, root):
    sha = parse_sums(sums_text).get(ASSET)
    if not sha:
        raise SystemExit(f"error: SHA256SUMS has no line for {ASSET}")
    scoop = root / SCOOP
    scoop.parent.mkdir(parents=True, exist_ok=True)
    scoop.write_text(json.dumps(scoop_manifest(version, sha), indent=4) + "\n", encoding="utf-8")
    vdir = root / WINGET / version
    for old in sorted((root / WINGET).glob("*")) if (root / WINGET).exists() else []:
        if old.is_dir() and old.name != version:   # winget-pkgs keeps history; this folder holds the current one
            for f in old.iterdir():
                f.unlink()
            old.rmdir()
    vdir.mkdir(parents=True, exist_ok=True)
    for name, text in winget_manifests(version, sha, date).items():
        (vdir / name).write_text(text, encoding="utf-8")
    print(f"wrote {SCOOP} and {WINGET / version} for {version}, {ASSET} sha256 {sha}")


def _field(text, key):
    m = re.search(rf"^\s*-?\s*{key}: (.+)$", text, re.M)
    return m.group(1).strip() if m else None


def check(root):
    errs = []
    scoop = json.loads((root / SCOOP).read_text(encoding="utf-8"))
    version = scoop["version"]
    arch = scoop["architecture"]["64bit"]
    if arch["url"] != release_url(version):
        errs.append(f"scoop url {arch['url']} is not the {version} release asset")
    if not re.fullmatch(r"[0-9a-f]{64}", arch["hash"]):
        errs.append(f"scoop hash {arch['hash']!r} is not a lowercase sha256")
    if scoop != scoop_manifest(version, arch["hash"]):
        errs.append("scoop manifest differs from what --write renders for its version and hash")
    dirs = [d for d in (root / WINGET).iterdir() if d.is_dir()] if (root / WINGET).exists() else []
    if [d.name for d in dirs] != [version]:
        errs.append(f"winget folder holds {sorted(d.name for d in dirs)}, want [{version!r}]")
    else:
        inst = (dirs[0] / f"{WINGET_ID}.installer.yaml").read_text(encoding="utf-8")
        date = _field(inst, "ReleaseDate")
        want = winget_manifests(version, arch["hash"], date)
        for name, text in want.items():
            got = (dirs[0] / name).read_text(encoding="utf-8")
            if got != text:
                errs.append(f"winget {name} differs from what --write renders")
        if _field(inst, "InstallerSha256") != arch["hash"].upper():
            errs.append("winget and scoop name different sha256s")
    for e in errs:
        print(e, file=sys.stderr)
    if not errs:
        print(f"package manifests ok: {version}, {ASSET} {arch['hash']}")
    return 1 if errs else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", metavar="X.Y.Z")
    ap.add_argument("--sums", help="a copy of the release's SHA256SUMS (default: gh release download)")
    ap.add_argument("--date", help="the release date for winget's ReleaseDate (default: the release's)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=str(ROOT))
    args = ap.parse_args()
    root = pathlib.Path(args.out)
    if args.write:
        if not re.fullmatch(r"\d+\.\d+\.\d+", args.write):
            raise SystemExit("error: --write takes a release version, X.Y.Z")
        sums = pathlib.Path(args.sums).read_text(encoding="utf-8") if args.sums else _fetch_sums(args.write)
        date = args.date
        if not date:
            out = subprocess.run(["gh", "release", "view", f"v{args.write}", "-R", REPO, "--json",
                                  "publishedAt", "--jq", ".publishedAt"],
                                 check=True, capture_output=True, text=True).stdout.strip()
            date = out[:10]
        datetime.date.fromisoformat(date)
        write(args.write, sums, date, root)
        return 0
    if args.check:
        return check(root)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
