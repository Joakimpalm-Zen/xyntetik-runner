"""A delegation's verdict as a signed receipt (R14.6).

The body is an in-toto Statement (v1): the subjects are the patch the
attempt produced and the commit it was produced against, each by digest;
the predicate is what shadow mode knows about the attempt, the request by
hash, the model by hash, the verdict, the budget and the runner build.
The runner signs it in place with its own key (`runner --sign-record`),
appending the same chain and signature objects a notarized transcript
carries, so `runner --check-record` verifies a delegation receipt with the
code path that verifies a notarized run, and each receipt links to the
previous one by hash.

The signature never covers the user's text: the request appears only as a
sha256, the diff only as a sha256 and a path. A receipt says "this runner
build, with this model, produced this patch against this commit, and the
repository's tests said this", and anyone holding the public key can check
that nothing in it was changed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
PREDICATE_TYPE = "https://xyntetik.com/shadow/delegation/v1"
RECEIPTS_REL = Path(".xyntetik") / "shadow" / "receipts"
SIGNKEY_REL = Path(".xyntetik") / "shadow" / "signkey.json"


@dataclass(frozen=True)
class Signed:
    path: Path
    chain_hash: str
    public_key: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def statement(*, repo: str, head: str, request: str, patch_path: str, patch_sha256: str,
              changed_paths: tuple[str, ...], verdict: str, tests_exit: int | None,
              task_class: str, model: str, model_sha256: str, adapter_sha256: str,
              runner_build: str, backend: str, budget_turns: int, budget_wall_s: float,
              turns: int, tool_calls: int, wall_s: float, stop_reason: str,
              observed_at: str) -> dict[str, Any]:
    """The receipt body before signing: an in-toto Statement whose subjects
    are the patch and the commit, matched by digest."""
    return {
        "_type": STATEMENT_TYPE,
        "subject": [
            {"name": Path(patch_path).name, "digest": {"sha256": patch_sha256}},
            {"name": Path(repo).name, "digest": {"gitCommit": head}},
        ],
        "predicateType": PREDICATE_TYPE,
        "predicate": {
            "request_sha256": sha256_text(request),
            "repository": Path(repo).name,
            "changed_paths": list(changed_paths),
            "task_class": task_class,
            "verdict": verdict,
            "tests_exit": tests_exit,
            "model": Path(model).name,
            "model_sha256": model_sha256,
            "adapter_sha256": adapter_sha256,
            "runner_build": runner_build,
            "backend": backend,
            "budget": {"turns": budget_turns, "wall_s": budget_wall_s},
            "attempt": {"turns": turns, "tool_calls": tool_calls, "wall_s": wall_s,
                        "stop_reason": stop_reason},
            "observed_at": observed_at,
        },
    }


def receipts_dir(home: Path) -> Path:
    return home / RECEIPTS_REL


def latest(home: Path) -> Path | None:
    """The newest signed receipt, the one the next receipt links to."""
    d = receipts_dir(home)
    if not d.is_dir():
        return None
    files = sorted(p for p in d.glob("*.json") if not p.name.startswith("."))
    return files[-1] if files else None


def keygen(home: Path, runner: str) -> Path:
    """One signing key per shadow installation, made by the runner."""
    key = home / SIGNKEY_REL
    key.parent.mkdir(parents=True, exist_ok=True)
    if key.is_file():
        return key
    proc = subprocess.run([runner, "--keygen", str(key)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=60)
    if proc.returncode != 0 or not key.is_file():
        raise RuntimeError(f"runner --keygen failed: {(proc.stderr or proc.stdout).strip()[:200]}")
    return key


def sign_with_runner(runner: str, path: Path, key: Path, prev: Path | None) -> Signed:
    """Sign a record in place through the runner and read back the chain
    hash and the public key it wrote."""
    args = [runner, "--sign-record", str(path), "--sign-key", str(key)]
    if prev is not None:
        args += ["--record-prev", str(prev)]
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"runner --sign-record failed: {(proc.stderr or proc.stdout).strip()[:200]}")
    rec = json.loads(path.read_text(encoding="utf-8"))
    return Signed(path, str(rec["chain"]["hash"]), str(rec["signature"]["public_key"]))


def write_receipt(home: Path, body: dict[str, Any], *, runner: str, key: Path,
                  signer: Any = None) -> Signed:
    """Write the statement under the receipts directory and sign it, linked to
    the previous receipt. ``signer`` stands in for the runner in tests."""
    d = receipts_dir(home)
    d.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    n = 0
    path = d / f"{stamp}.json"
    while path.exists():
        n += 1
        path = d / f"{stamp}-{n}.json"
    prev = latest(home)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sign = signer or sign_with_runner
    try:
        return sign(runner, path, key, prev)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def check_with_runner(runner: str, path: Path, trust_key: str = "") -> tuple[int, str]:
    args = [runner, "--check-record", str(path)]
    if trust_key:
        args += ["--trust-key", trust_key]
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=60)
    return proc.returncode, (proc.stdout or proc.stderr).strip()


def render_receipts(home: Path, *, runner: str = "", check: bool = False, checker: Any = None) -> str:
    d = receipts_dir(home)
    files = sorted(d.glob("*.json")) if d.is_dir() else []
    if not files:
        return "no receipts yet (a signing key makes every delegation write one: 'shadow keygen')"
    lines = ["| receipt | repository | verdict | class | chain | prev |" + (" check |" if check else ""),
             "|---|---|---|---|---|---|" + ("---|" if check else "")]
    for p in files[-30:]:
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            lines.append(f"| {p.name} | unreadable | | | | |" + (" |" if check else ""))
            continue
        pred = r.get("predicate", {})
        chain = r.get("chain", {})
        row = (f"| {p.name} | {pred.get('repository', '')} | {pred.get('verdict', '')} | "
               f"{pred.get('task_class', '')} | {str(chain.get('hash', ''))[:12]} | {str(chain.get('prev', ''))[:12]} |")
        if check:
            rc, text = (checker or check_with_runner)(runner, p)
            row += f" {text.split(':')[0] if text else rc} |"
        lines.append(row)
    return "\n".join(lines)
