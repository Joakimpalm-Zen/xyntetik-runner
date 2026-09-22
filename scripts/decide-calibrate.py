#!/usr/bin/env python3
"""Calibration instrument for POST /v1/decide (R13.10, the `decide` contract).

`/v1/decide` scores a fixed set of verbatim-option questions against a
prefilled state with zero sampling: each option's log-probability is the
product of its in-context token conditionals, and `probs` is the softmax
over the given options only. This script asks the question that decides
whether that readout is USABLE: when the model puts 0.8 on an answer, is it
right 80% of the time, does the score survive rephrasing and re-ordering the
options, and does it hold on questions the run never trained a threshold on?

Input: the lab's question shape, one JSON object per line (contract:
`docs/decide-contract.md`, shade `research/evaluation/typed-decisions/`):

    {"state": "...", "question": "...", "options": ["...", "..."],
     "answer": 0, "source": "ledger", "permutation_group": "abc123",
     "variant": 0}

`answer` is either an index into `options` or a distribution over them
(same length and order as `options`). The file carries no per-question id;
one is assigned here (`q<line-number>`, 0-based) to correlate the request
and the response.

Held-out split: BY permutation_group, never by row, deterministic from
--seed. A permutation_group's rows can span more than one underlying
`state` (a rephrasing AND a chosen/rejected pair sharing one group, seen in
the real slice-1 data) -- the split still keeps the whole group on one side,
because that is what the contract's invariance question is checked against.

Metrics are computed twice, once per split, each overall and per source:
accuracy, multi-class Brier score, log loss (cross-entropy against the
label distribution, one-hot when `answer` is an index), expected
calibration error with a reliability curve, and a permutation-invariance
score over each permutation_group's variants. The held-out numbers are the
headline; the report leads with the unflattering metric (permutation
invariance), then ECE, then accuracy, per the house rule of publishing the
unflattering metric first.

No third-party imports -- usable anywhere the runner builds.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCHEMA_VERSION = "xyntetik.runner.decide-calibrate.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SEED = 20260922
DEFAULT_HOLDOUT_FRAC = 0.2
DEFAULT_BINS = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF = 0.5


# --------------------------------------------------------------- input ---

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_questions(path, max_questions=None):
    """Parse the question file into row dicts. Each row keeps its original
    parsed object under `raw` (so --emit-holdout can re-emit it unchanged)
    plus derived fields: `id`, `line`, `label_dist`, `split` (filled later).

    A line that fails to parse, or is missing a required field, or whose
    `answer` cannot be turned into a distribution, is skipped and counted
    in `skipped` rather than aborting the run -- the same "do not abort"
    posture the HTTP layer uses for a bad response.
    """
    required = ("state", "question", "options", "answer", "source",
                "permutation_group", "variant")
    rows = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if max_questions is not None and len(rows) >= max_questions:
                break
            try:
                obj = json.loads(line)
                for key in required:
                    if key not in obj:
                        raise KeyError(key)
                options = obj["options"]
                if not isinstance(options, list) or len(options) < 2:
                    raise ValueError("options must be a list of at least 2 strings")
                if len(set(options)) != len(options):
                    raise ValueError("duplicate option string")
                label_dist = label_to_distribution(obj["answer"], len(options))
            except (ValueError, KeyError, TypeError) as e:
                print(f"line {line_no}: skipped ({e})", file=sys.stderr)
                skipped += 1
                continue
            rows.append({
                "id": f"q{line_no}",
                "line": line_no,
                "raw": obj,
                "state": obj["state"],
                "question": obj["question"],
                "options": list(options),
                "source": obj["source"],
                "permutation_group": obj["permutation_group"],
                "split_group": obj.get("split_group"),
                "variant": obj["variant"],
                "label_dist": label_dist,
                "split": None,
                "probs": None,
                "logprobs": None,
                "argmax": None,
                "n_tokens": None,
                "failed": False,
                "error": None,
            })
    return rows, skipped


def label_to_distribution(answer, n_options):
    """An index becomes a one-hot vector; a list is used as-is (verbatim,
    same order as `options`) -- the contract does not ask for renormalization
    and silently renormalizing a malformed distribution would hide it."""
    if isinstance(answer, list):
        if len(answer) != n_options:
            raise ValueError(
                f"answer distribution has {len(answer)} entries for "
                f"{n_options} options")
        return [float(x) for x in answer]
    idx = int(answer)
    if not 0 <= idx < n_options:
        raise ValueError(f"answer index {idx} out of range for {n_options} options")
    return [1.0 if i == idx else 0.0 for i in range(n_options)]


# ---------------------------------------------------------------- split ---

def split_key(row):
    """The id the held-out split goes by: `split_group` when the file carries
    it (slice 1 rev f69ed87b: a DPO pair's chosen and rejected halves share
    it so they land on one side), else `permutation_group`."""
    return row.get("split_group") or row["permutation_group"]


def assign_holdout_groups(group_ids, seed, holdout_frac):
    """Deterministically choose the held-out permutation_group ids.

    Sorted first so the result depends only on the SET of group ids and the
    seed, not on the order rows happen to appear in the file (two files
    holding the same groups in different orders must split identically).
    """
    ordered = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_holdout = round(len(ordered) * holdout_frac)
    return set(ordered[:n_holdout])


# ----------------------------------------------------------------- HTTP ---

def build_payload(state, model_name, answer_prefix, group_rows):
    payload = {
        "state": state,
        "answer_prefix": answer_prefix,
        "questions": [
            {"id": r["id"], "question": r["question"], "options": list(r["options"])}
            for r in group_rows
        ],
    }
    if model_name:
        payload["model"] = model_name
    return payload


def post_decide(endpoint, payload, timeout=120,
                 max_retries=DEFAULT_MAX_RETRIES, backoff=DEFAULT_BACKOFF):
    """POST one /v1/decide request. Returns (ok, body_or_None, error_or_None).

    A 400 is a client error about THIS request's content (bad options,
    per the contract), not a transient failure -- retrying the identical
    payload cannot help, so it returns immediately without spending the
    retry budget. Anything else (connection failure, timeout, 5xx) is
    retried up to `max_retries` times with a linear backoff before being
    recorded as failed.
    """
    url = endpoint.rstrip("/") + "/v1/decide"
    data = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return True, body, None
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            if e.code == 400:
                # The runner error envelope is {"error": {"message": ...,
                # "type": ..., "param": ..., "code": ...}} (src/http.c,
                # send_error_detail); fall back to the raw body for a server
                # that does not use that shape.
                message = body_text[:500]
                try:
                    parsed = json.loads(body_text)
                    message = parsed.get("error", {}).get("message", message)
                except (ValueError, AttributeError):
                    pass
                return False, None, f"HTTP 400: {message}"
            last_err = f"HTTP {e.code}: {body_text[:500]}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = str(e)
        if attempt < max_retries - 1:
            time.sleep(backoff * (attempt + 1))
    return False, None, last_err or "unknown error"


def apply_decisions(group_rows, body):
    by_id = {d["id"]: d for d in body.get("decisions", [])}
    for r in group_rows:
        d = by_id.get(r["id"])
        if d is None:
            r["failed"] = True
            r["error"] = "response carried no decision for this question id"
            continue
        if list(d.get("options", [])) != list(r["options"]):
            r["failed"] = True
            r["error"] = "echoed options do not match the options sent"
            continue
        r["probs"] = list(d["probs"])
        r["logprobs"] = list(d["logprobs"])
        r["argmax"] = d["argmax"]
        r["n_tokens"] = list(d["n_tokens"])


def send_group(endpoint, model_name, answer_prefix, group_rows,
               max_retries=DEFAULT_MAX_RETRIES, backoff=DEFAULT_BACKOFF):
    """Send one group of rows (sharing one `state`) as a single request.
    A 400 on a multi-question group is isolated by falling back to one
    request per question, so one bad question in a batch does not fail
    every question that shares its state prefill."""
    payload = build_payload(group_rows[0]["state"], model_name, answer_prefix, group_rows)
    ok, body, err = post_decide(endpoint, payload, max_retries=max_retries, backoff=backoff)
    if ok:
        apply_decisions(group_rows, body)
        return
    if err and err.startswith("HTTP 400") and len(group_rows) > 1:
        for r in group_rows:
            send_group(endpoint, model_name, answer_prefix, [r],
                       max_retries=max_retries, backoff=backoff)
        return
    for r in group_rows:
        r["failed"] = True
        r["error"] = err


def group_requests(rows, batch_state):
    """Rows sharing the same `state` string batch into one request (the
    contract's KV-reuse case: the state is prefilled once, each question a
    suffix). Grouping ignores split and permutation_group -- both are
    attributed from the response's per-question ids after scoring, so
    batching by state alone is always safe."""
    if not batch_state:
        return [[r] for r in rows]
    groups = {}
    order = []
    for r in rows:
        key = r["state"]
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)
    return [groups[k] for k in order]


# -------------------------------------------------------------- metrics ---

def compute_metrics(rows, bins=DEFAULT_BINS):
    """Accuracy, Brier, log loss, ECE and a reliability curve over `rows`
    (any rows without a `probs` are excluded -- there is no prediction to
    score)."""
    scored = [r for r in rows if r["probs"] is not None]
    n = len(scored)
    if n == 0:
        return {"n": 0, "accuracy": None, "brier": None, "log_loss": None,
                "ece": None, "reliability": []}

    correct = 0
    brier_sum = 0.0
    logloss_sum = 0.0
    bin_n = [0] * bins
    bin_hits = [0] * bins
    bin_conf = [0.0] * bins

    for r in scored:
        probs = r["probs"]
        y = r["label_dist"]
        pred_idx = max(range(len(probs)), key=lambda i: probs[i])
        # the label's mass on the predicted option: 0 or 1 for an index
        # label, the expected correctness for a distribution label. Collapsing
        # the distribution to its argmax reported ECE 0.20 for a prediction
        # that matched the target exactly (found 2026-09-22).
        hit = y[pred_idx]
        correct += hit
        brier_sum += sum((p - yy) ** 2 for p, yy in zip(probs, y))
        logloss_sum += -sum(yy * math.log(max(p, 1e-12)) for p, yy in zip(probs, y))
        conf = probs[pred_idx]
        b = min(int(conf * bins), bins - 1)
        bin_n[b] += 1
        bin_hits[b] += hit
        bin_conf[b] += conf

    ece = 0.0
    reliability = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        c = bin_n[i]
        if c:
            acc_b = bin_hits[i] / c
            conf_b = bin_conf[i] / c
            ece += (c / n) * abs(acc_b - conf_b)
        else:
            acc_b = None
            conf_b = None
        reliability.append({"bin": [lo, hi], "n": c, "confidence": conf_b,
                            "accuracy": acc_b})

    return {
        "n": n,
        "accuracy": correct / n,
        "brier": brier_sum / n,
        "log_loss": logloss_sum / n,
        "ece": ece,
        "reliability": reliability,
    }


def compute_invariance(rows):
    """Permutation-invariance over `rows`' permutation_group values.

    A group here is a (permutation_group, state) pair: the variants of one
    item that share the state text. Each variant's probability vector is
    remapped, by exact option-string match, back to the option order of
    that group's first row (file order).
    A group whose variants do not share the same SET of option strings is
    "unmatched" and excluded from the score, per the contract. A group left
    with fewer than two scored rows (one row, or every other row in the
    group failed) has no pair to compare and is reported separately as
    "single" rather than folded into either bucket.
    """
    scored = [r for r in rows if r["probs"] is not None]
    # A permutation variant is the SAME state asked again with the options
    # reordered or the question rephrased. Slice 1 also files a chosen and
    # a rejected edit (two states, opposite labels) under one group id, and
    # those are different questions, so the unit of invariance is the
    # (permutation_group, state) pair, never the group id alone.
    by_group = {}
    for r in scored:
        by_group.setdefault((r["permutation_group"], r["state"]), []).append(r)

    unmatched = 0
    single = 0
    max_tvds = []
    argmax_changes = 0
    matched = 0

    for grp in by_group.values():
        if len(grp) < 2:
            single += 1
            continue
        canon = list(grp[0]["options"])
        canon_set = set(canon)
        if any(set(r["options"]) != canon_set for r in grp):
            unmatched += 1
            continue
        vectors = []
        argmaxes = []
        for r in grp:
            m = dict(zip(r["options"], r["probs"]))
            vec = [m[o] for o in canon]
            vectors.append(vec)
            argmaxes.append(canon[max(range(len(vec)), key=lambda i: vec[i])])
        group_max_tvd = 0.0
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                tvd = 0.5 * sum(abs(a - b) for a, b in zip(vectors[i], vectors[j]))
                group_max_tvd = max(group_max_tvd, tvd)
        max_tvds.append(group_max_tvd)
        matched += 1
        if len(set(argmaxes)) > 1:
            argmax_changes += 1

    return {
        "groups_total": len(by_group),
        "groups_matched": matched,
        "groups_unmatched": unmatched,
        "groups_single": single,
        "mean_max_tvd": statistics.fmean(max_tvds) if max_tvds else None,
        "max_max_tvd": max(max_tvds) if max_tvds else None,
        "argmax_change_frac": (argmax_changes / matched) if matched else None,
    }


def split_report(rows, bins):
    """One split's (train or holdout) overall + per-source metrics and
    invariance, keyed the same way the markdown renderer expects."""
    sources = sorted({r["source"] for r in rows})
    out = {
        "n_rows": len(rows),
        "n_failed": sum(1 for r in rows if r["failed"]),
        "overall": {
            "metrics": compute_metrics(rows, bins),
            "invariance": compute_invariance(rows),
        },
        "by_source": {},
    }
    for s in sources:
        sub = [r for r in rows if r["source"] == s]
        out["by_source"][s] = {
            "n_rows": len(sub),
            "n_failed": sum(1 for r in sub if r["failed"]),
            "metrics": compute_metrics(sub, bins),
            "invariance": compute_invariance(sub),
        }
    return out


# ---------------------------------------------------------------- misc ---

def git_describe(repo_root):
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=repo_root, capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def fmt_pct(x):
    return "-" if x is None else f"{100.0 * x:.2f}%"


def fmt_num(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def render_invariance_table(split_data):
    lines = [
        "| scope | groups | matched | unmatched | single | mean max TVD | "
        "max max TVD | argmax changes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    inv = split_data["overall"]["invariance"]
    lines.append(
        f"| overall | {inv['groups_total']} | {inv['groups_matched']} | "
        f"{inv['groups_unmatched']} | {inv['groups_single']} | "
        f"{fmt_num(inv['mean_max_tvd'])} | {fmt_num(inv['max_max_tvd'])} | "
        f"{fmt_pct(inv['argmax_change_frac'])} |")
    for s, sub in sorted(split_data["by_source"].items()):
        inv = sub["invariance"]
        lines.append(
            f"| {s} | {inv['groups_total']} | {inv['groups_matched']} | "
            f"{inv['groups_unmatched']} | {inv['groups_single']} | "
            f"{fmt_num(inv['mean_max_tvd'])} | {fmt_num(inv['max_max_tvd'])} | "
            f"{fmt_pct(inv['argmax_change_frac'])} |")
    return "\n".join(lines)


def render_reliability_table(metrics, bins):
    lines = [
        "| bin | n | confidence | accuracy | gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics["reliability"]:
        lo, hi = row["bin"]
        if row["n"] == 0:
            lines.append(f"| {lo:.2f}-{hi:.2f} | 0 | - | - | - |")
            continue
        gap = row["accuracy"] - row["confidence"]
        lines.append(
            f"| {lo:.2f}-{hi:.2f} | {row['n']} | {row['confidence']:.4f} | "
            f"{row['accuracy']:.4f} | {gap:+.4f} |")
    return "\n".join(lines)


def render_ece_table(split_data):
    lines = ["| scope | n | ECE |", "|---|---:|---:|"]
    m = split_data["overall"]["metrics"]
    lines.append(f"| overall | {m['n']} | {fmt_num(m['ece'])} |")
    for s, sub in sorted(split_data["by_source"].items()):
        m = sub["metrics"]
        lines.append(f"| {s} | {m['n']} | {fmt_num(m['ece'])} |")
    return "\n".join(lines)


def render_accuracy_table(split_data):
    lines = [
        "| scope | n | accuracy | Brier | log loss |",
        "|---|---:|---:|---:|---:|",
    ]
    m = split_data["overall"]["metrics"]
    lines.append(f"| overall | {m['n']} | {fmt_pct(m['accuracy'])} | "
                 f"{fmt_num(m['brier'])} | {fmt_num(m['log_loss'])} |")
    for s, sub in sorted(split_data["by_source"].items()):
        m = sub["metrics"]
        lines.append(f"| {s} | {m['n']} | {fmt_pct(m['accuracy'])} | "
                     f"{fmt_num(m['brier'])} | {fmt_num(m['log_loss'])} |")
    return "\n".join(lines)


def render_report(result):
    p = result["provenance"]
    lines = [
        "# decide-calibrate report",
        "",
        f"questions file: `{p['questions_file']}` (sha256 `{p['questions_sha256']}`)",
        f"endpoint: {p['endpoint']}  model: {p['model_name']}",
        f"seed: {p['seed']}  holdout_frac: {p['holdout_frac']}  bins: {p['bins']}",
        f"runner: {p['runner_git_describe'] or '(git describe unavailable)'}",
        f"rows: {p['n_rows']} used, {p['n_skipped']} skipped at parse, "
        f"{p['n_failed']} failed at the endpoint",
        f"split: {p['n_train_rows']} train rows / {p['n_train_groups']} groups, "
        f"{p['n_holdout_rows']} holdout rows / {p['n_holdout_groups']} groups",
        "",
        "## Permutation invariance (the unflattering metric, reported first)",
        "",
        "Mean and max, over each split's matched permutation_groups, of the "
        "largest total-variation distance between any two variants' "
        "remapped probability vectors; and the fraction of matched groups "
        "whose argmax changes across variants.",
        "",
        "### held-out (headline)",
        "",
        render_invariance_table(result["holdout"]),
        "",
        "### train",
        "",
        render_invariance_table(result["train"]),
        "",
        "## Expected calibration error",
        "",
        "### held-out (headline)",
        "",
        render_ece_table(result["holdout"]),
        "",
        f"Reliability curve, held-out, overall ({p['bins']} bins):",
        "",
        render_reliability_table(result["holdout"]["overall"]["metrics"], p["bins"]),
        "",
        "### train",
        "",
        render_ece_table(result["train"]),
        "",
        "## Accuracy, Brier score, log loss",
        "",
        "### held-out (headline)",
        "",
        render_accuracy_table(result["holdout"]),
        "",
        "### train",
        "",
        render_accuracy_table(result["train"]),
        "",
    ]
    if result["failures"]:
        lines.append("## Failures")
        lines.append("")
        lines.append(f"{len(result['failures'])} question(s) never scored:")
        lines.append("")
        for f in result["failures"][:50]:
            lines.append(f"- `{f['id']}` (line {f['line']}, source {f['source']}): "
                         f"{f['error']}")
        if len(result["failures"]) > 50:
            lines.append(f"- ... and {len(result['failures']) - 50} more")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------- main ---

def run(args):
    rows, n_skipped = read_questions(args.questions, args.max_questions)
    if not rows:
        print("error: no usable questions parsed", file=sys.stderr)
        return None, 1

    group_ids = [split_key(r) for r in rows]
    holdout_groups = assign_holdout_groups(group_ids, args.seed, args.holdout_frac)
    for r in rows:
        r["split"] = "holdout" if split_key(r) in holdout_groups else "train"

    request_groups = group_requests(rows, args.batch_state)
    total = len(rows)
    done = 0
    last_reported = 0
    for group in request_groups:
        send_group(args.endpoint, args.model_name, args.answer_prefix, group,
                   max_retries=DEFAULT_MAX_RETRIES, backoff=DEFAULT_BACKOFF)
        done += len(group)
        if done - last_reported >= 50 or done == total:
            print(f"progress: {done}/{total} questions", file=sys.stderr)
            last_reported = done

    train_rows = [r for r in rows if r["split"] == "train"]
    holdout_rows = [r for r in rows if r["split"] == "holdout"]
    train_groups = {split_key(r) for r in train_rows}
    holdout_groups_seen = {r["permutation_group"] for r in holdout_rows}

    failures = [{"id": r["id"], "line": r["line"], "source": r["source"],
                "error": r["error"]} for r in rows if r["failed"]]

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "questions_file": str(args.questions),
        "questions_sha256": sha256_file(args.questions),
        "endpoint": args.endpoint,
        "model_name": args.model_name,
        "answer_prefix": args.answer_prefix,
        "seed": args.seed,
        "holdout_frac": args.holdout_frac,
        "bins": args.bins,
        "batch_state": args.batch_state,
        "runner_git_describe": git_describe(REPO_ROOT),
        "n_rows": total,
        "n_skipped": n_skipped,
        "n_failed": len(failures),
        "n_train_rows": len(train_rows),
        "n_train_groups": len(train_groups),
        "n_holdout_rows": len(holdout_rows),
        "n_holdout_groups": len(holdout_groups_seen),
        "per_source_counts": {
            s: sum(1 for r in rows if r["source"] == s)
            for s in sorted({r["source"] for r in rows})
        },
    }

    result = {
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance,
        "train": split_report(train_rows, args.bins),
        "holdout": split_report(holdout_rows, args.bins),
        "failures": failures,
    }
    return (result, rows), 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True,
                    help="input JSONL, the agreed question shape")
    ap.add_argument("--endpoint", required=True,
                    help="base URL of the /v1/decide server")
    ap.add_argument("--model-name", required=True,
                    help="model field sent with every request")
    ap.add_argument("--answer-prefix", default="",
                    help="text placed between the question and every option")
    ap.add_argument("--max-questions", type=int, default=None,
                    help="use only the first N questions in the file")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"holdout split seed (default {DEFAULT_SEED})")
    ap.add_argument("--holdout-frac", type=float, default=DEFAULT_HOLDOUT_FRAC,
                    help=f"fraction of permutation_groups held out "
                         f"(default {DEFAULT_HOLDOUT_FRAC})")
    ap.add_argument("--bins", type=int, default=DEFAULT_BINS,
                    help=f"equal-width ECE/reliability bins (default {DEFAULT_BINS})")
    ap.add_argument("--batch-state", action=argparse.BooleanOptionalAction, default=True,
                    help="group questions sharing a state into one request (default on)")
    ap.add_argument("--out", default=None, help="write the full result as JSON")
    ap.add_argument("--emit-holdout", default=None,
                    help="re-emit the held-out rows, unchanged, as JSONL")
    ap.add_argument("--report", default=None, help="write the markdown report")
    args = ap.parse_args(argv)

    payload, code = run(args)
    if payload is None:
        return code
    result, rows = payload

    print(json.dumps({
        "n_rows": result["provenance"]["n_rows"],
        "n_failed": result["provenance"]["n_failed"],
        "train_n": result["train"]["n_rows"],
        "holdout_n": result["holdout"]["n_rows"],
        "holdout_accuracy": result["holdout"]["overall"]["metrics"]["accuracy"],
        "holdout_ece": result["holdout"]["overall"]["metrics"]["ece"],
        "holdout_mean_max_tvd": result["holdout"]["overall"]["invariance"]["mean_max_tvd"],
    }, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if args.emit_holdout:
        with open(args.emit_holdout, "w", encoding="utf-8") as f:
            for r in rows:
                if r["split"] == "holdout":
                    f.write(json.dumps(r["raw"]) + "\n")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(render_report(result))

    if result["provenance"]["n_rows"] == result["provenance"]["n_failed"]:
        print("error: every question failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
