#!/usr/bin/env python3
"""The plan of a Hugging Face Hub upload, refused when it would overwrite or
delete a file nobody named (R7.1.5).

    hub-upload-check.py REPO_ID FOLDER [--repo-type model|dataset|space]
        [--revision main] [--path-in-repo PREFIX]
        [--replace PATH ...] [--delete PATH ...] [--partial] [--new]

FOLDER is what the upload will send, laid out as it will sit in the repo
(under PREFIX when --path-in-repo is given). Every file in it is compared with
the remote file at the same path, by content: an LFS file by the sha256 the
Hub records for it, a small file by its git blob id. The plan lists each
change:

    add      a file the remote does not have
    replace  a file the remote has with other content
    delete   a remote file under PREFIX that the folder does not have

and counts the files that are identical, which the upload leaves alone. A
replacement or deletion passes only when its repo path is named with
--replace or --delete. --partial declares an upload that deletes nothing (a
card on its own, say): remote files missing from the folder are then counted
and left as they are, not refused. --new declares a repository that does not
exist yet; without it a missing repository is an error, since a private one
the token cannot see answers the same way and every file would look like an
addition.

It only reads: nothing is uploaded or changed. A public repository needs no
token; for a private one the token is read from HF_TOKEN (or
HUGGING_FACE_HUB_TOKEN) and is never printed. HF_ENDPOINT points it at
another Hub, as it does for huggingface_hub. The plan ends with the remote
commit it was compared against; passing that commit as the upload's
parent_commit makes the upload fail if the remote moved after the check.

Exit 0: the plan holds. 1: refused (an unnamed replacement or deletion).
2: the check could not be made (a missing repository, a name that matches
nothing on the remote, a network or HTTP error).
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

# The paths huggingface_hub's upload_folder never sends.
IGNORED_DIRS = {".git"}
IGNORED_PREFIXES = (".cache/huggingface/",)


def file_hashes(path):
    """sha256 of the bytes and git's blob id (sha1 of "blob <size>\\0" + bytes), in one read."""
    size = os.path.getsize(path)
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
            blob.update(chunk)
    return {"sha256": sha256.hexdigest(), "blob": blob.hexdigest(), "size": size}


def local_files(folder, prefix):
    out = {}
    root = pathlib.Path(folder)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for name in sorted(filenames):
            p = pathlib.Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            if rel.startswith(IGNORED_PREFIXES):
                continue
            out[f"{prefix}/{rel}" if prefix else rel] = p
    return out


class HubError(Exception):
    def __init__(self, code, url):
        super().__init__(f"HTTP {code} from {url}")
        self.code = code


def get(url, token):
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "hub-upload-check"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r), r.headers.get("Link", "")
    except urllib.error.HTTPError as e:
        raise HubError(e.code, url) from None


def next_link(link):
    m = re.search(r'<([^>]+)>\s*;\s*rel="next"', link or "")
    return m.group(1) if m else None


def remote_listing(endpoint, repo_type, repo, revision, token):
    """(commit, {path: entry}) for every file at the revision, all pages read."""
    base = f"{endpoint}/api/{repo_type}s/{repo}"
    rev = urllib.parse.quote(revision, safe="")
    info, _ = get(f"{base}/revision/{rev}", token)
    files = {}
    url = f"{base}/tree/{rev}?recursive=true"
    while url:
        page, link = get(url, token)
        for e in page:
            if e.get("type") == "file":
                files[e["path"]] = e
        url = next_link(link)
    return info.get("sha", "unknown"), files


def same(entry, hashes):
    lfs = entry.get("lfs")
    if lfs and lfs.get("oid"):
        return lfs["oid"] == hashes["sha256"], lfs["oid"], hashes["sha256"]
    return entry.get("oid") == hashes["blob"], entry.get("oid", ""), hashes["blob"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repo_id")
    ap.add_argument("folder")
    ap.add_argument("--repo-type", choices=("model", "dataset", "space"), default="model")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--path-in-repo", default="", help="where FOLDER sits in the repo")
    ap.add_argument("--replace", action="append", default=[], metavar="PATH",
                    help="a repo path the upload may replace (repeatable)")
    ap.add_argument("--delete", action="append", default=[], metavar="PATH",
                    help="a repo path the upload may delete (repeatable)")
    ap.add_argument("--partial", action="store_true",
                    help="the upload deletes nothing; remote files missing from FOLDER stay")
    ap.add_argument("--new", action="store_true", help="the repository does not exist yet")
    a = ap.parse_args(argv)

    folder = pathlib.Path(a.folder)
    if not folder.is_dir():
        print(f"hub-upload-check: {folder} is not a folder", file=sys.stderr)
        return 2
    prefix = a.path_in_repo.strip("/")
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    where = f"{a.repo_type} {a.repo_id} at {a.revision}"

    try:
        commit, remote = remote_listing(endpoint, a.repo_type, a.repo_id, a.revision, token)
        if a.new:
            print(f"hub-upload-check: --new, but {where} exists (commit {commit})", file=sys.stderr)
            return 2
    except HubError as e:
        if e.code in (401, 403, 404) and a.new:
            commit, remote = None, {}
        elif e.code in (401, 403, 404):
            print(f"hub-upload-check: {where} is not found or not visible{' with the token' if token else ''} "
                  f"({e}); pass --new for a repository that does not exist yet", file=sys.stderr)
            return 2
        else:
            print(f"hub-upload-check: could not list {where}: {e}", file=sys.stderr)
            return 2
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"hub-upload-check: could not list {where}: {e}", file=sys.stderr)
        return 2

    under = {p: e for p, e in remote.items() if not prefix or p == prefix or p.startswith(prefix + "/")}
    unknown = [n for n in a.replace + a.delete if n not in under]
    if unknown:
        print(f"hub-upload-check: named but not on the remote {where}: {', '.join(unknown)}", file=sys.stderr)
        return 2

    local = local_files(folder, prefix)
    plan, unchanged, refused = [], 0, []
    for path in sorted(local):
        h = file_hashes(local[path])
        entry = under.get(path)
        if entry is None:
            plan.append(f"add      {path}  ({h['size']:,} bytes)")
            continue
        ok, was, now = same(entry, h)
        if ok:
            unchanged += 1
            continue
        named = path in a.replace
        plan.append(f"replace  {path}  {was[:12]} -> {now[:12]}  ({h['size']:,} bytes)"
                    + ("  named" if named else "  REFUSED, not named"))
        if not named:
            refused.append(f"--replace {path}")
    missing = sorted(set(under) - set(local))
    kept = 0
    for path in missing:
        if path in a.delete:
            plan.append(f"delete   {path}  named")
        elif a.partial:
            kept += 1
        else:
            plan.append(f"delete   {path}  REFUSED, not named")
            refused.append(f"--delete {path}")

    print(f"hub-upload-check: {where}" + (f" (commit {commit})" if commit else " (new repository)")
          + f" against {folder}" + (f" as {prefix}/" if prefix else ""))
    for line in plan:
        print(line)
    print(f"{unchanged} unchanged")
    if kept:
        print(f"1 remote file is not in the folder and stays as it is (--partial)" if kept == 1 else
              f"{kept} remote files are not in the folder and stay as they are (--partial)")
    if refused:
        deletions = sum(1 for r in refused if r.startswith("--delete"))
        hint = "; if this upload deletes nothing, pass --partial" if deletions else ""
        if len(refused) <= 8:
            print(f"REFUSED: {len(refused)} change{'s' if len(refused) != 1 else ''} nobody named. "
                  f"If each is intended, run again with: {' '.join(refused)}{hint}")
        else:
            reps = len(refused) - deletions
            print(f"REFUSED: {len(refused)} changes nobody named ({reps} replacement{'s' if reps != 1 else ''}, "
                  f"{deletions} deletion{'s' if deletions != 1 else ''}). "
                  f"Name each one intended with --replace or --delete{hint}")
        return 1
    print("OK: the plan holds" + (f"; upload with parent_commit={commit}" if commit else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
