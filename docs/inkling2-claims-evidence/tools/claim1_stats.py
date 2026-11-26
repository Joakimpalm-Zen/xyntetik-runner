#!/usr/bin/env python3
"""R4.48.12 PREREG 0.2 / 1.3 / 1.4 Claim 1 pre-registered analysis script.

Reads tools/verify_tf.py (version 2) OUT.json output file(s) and produces:
- per-engine k / n (k / 100) and per-category k_c / 25 with 95% Wilson score CIs;
- exact-tie position counts and failure breakdowns (encoding, incomplete, diff);
- per-engine first_diff distribution (min, p25, median, p75, max, buckets j=0,
  1-15, 16-63, 64-255) and ref_margin_at_first_diff median and IQR;
- Erratum 2 (ERRATUM-2026-11-24.md) head-shape sensitivity flags per engine;
- for each engine pair (Runner vs mainline llama.cpp; Runner vs Unsloth per
  ADDENDUM-2026-11-24-unsloth.md), paired 2x2 table (n_11, n_10, n_01, n_00),
  Newcombe (1998) Method 10 paired 95% CI, exact two-sided McNemar p-value,
  outcome classification, and PREREG 1.4 wording.

Pure Python standard library only (math, statistics, fractions, json, argparse,
pathlib, sys); no third-party statistics packages. Deterministic output.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

# Exact z_0.975 constant pinned in PREREG.md Section 0.2 line 43.
Z_95 = 1.959963984540054
PRIMARY_CATEGORIES = ("chat", "code", "arithmetic", "short_tool_use")


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score confidence interval (PREREG 0.2)."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"k must satisfy 0 <= k <= n, got k={k}, n={n}")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


def mcnemar_exact_p(n_10: int, n_01: int) -> float:
    """McNemar's exact two-sided p-value (PREREG 0.2 line 49)."""
    if n_10 < 0 or n_01 < 0:
        raise ValueError(f"discordant counts must be non-negative: n_10={n_10}, n_01={n_01}")
    m = n_10 + n_01
    if m == 0:
        return 1.0
    k = min(n_10, n_01)
    tail = sum(math.comb(m, i) for i in range(0, k + 1))
    # Exact rational 2 * tail / 2^m avoids intermediate underflow for large m.
    return min(1.0, float(Fraction(2 * tail, 1 << m)))


def paired_newcombe_method10(
    n_11: int, n_10: int, n_01: int, n_00: int, z: float = Z_95
) -> dict[str, Any]:
    """Newcombe (1998) Method 10 paired 95% CI, McNemar p, and outcome (PREREG 0.2)."""
    counts = (n_11, n_10, n_01, n_00)
    if any(c < 0 for c in counts):
        raise ValueError(f"cell counts must be non-negative, got {counts}")
    n = n_11 + n_10 + n_01 + n_00
    if n <= 0:
        raise ValueError("total paired count n must be positive")

    k_a = n_11 + n_10
    k_b = n_11 + n_01
    p_a = k_a / n
    p_b = k_b / n
    delta = (n_10 - n_01) / n

    l_a, u_a = wilson_interval(k_a, n, z=z)
    l_b, u_b = wilson_interval(k_b, n, z=z)

    d_corr = n_11 * n_00 - n_10 * n_01
    m_prod = (n_11 + n_10) * (n_01 + n_00) * (n_11 + n_01) * (n_10 + n_00)

    if m_prod == 0 or (0 <= 2 * d_corr <= n):
        phi = 0.0
    elif 2 * d_corr > n:
        phi = (d_corr - n / 2.0) / math.sqrt(m_prod)
    else:
        phi = d_corr / math.sqrt(m_prod)

    lo_rad = (p_a - l_a) ** 2 - 2.0 * phi * (p_a - l_a) * (u_b - p_b) + (u_b - p_b) ** 2
    hi_rad = (u_a - p_a) ** 2 - 2.0 * phi * (u_a - p_a) * (p_b - l_b) + (p_b - l_b) ** 2
    delta_lo = max(-1.0, delta - math.sqrt(max(0.0, lo_rad)))
    delta_hi = min(1.0, delta + math.sqrt(max(0.0, hi_rad)))

    p_val = mcnemar_exact_p(n_10, n_01)
    if delta > 0.0 and delta_lo > 0.0:
        outcome = "higher"
    elif delta < 0.0 and delta_hi < 0.0:
        outcome = "lower"
    else:
        outcome = "tie"

    return {
        "n": n,
        "table": {"n_11": n_11, "n_10": n_10, "n_01": n_01, "n_00": n_00},
        "m_discordant": n_10 + n_01,
        "k_a": k_a,
        "k_b": k_b,
        "p_a": p_a,
        "p_b": p_b,
        "wilson_a": [l_a, u_a],
        "wilson_b": [l_b, u_b],
        "delta": delta,
        "D": d_corr,
        "M": m_prod,
        "phi": phi,
        "delta_lo": delta_lo,
        "delta_hi": delta_hi,
        "mcnemar_p": p_val,
        "outcome": outcome,
    }


def quartiles_iqr(xs: list[float]) -> dict[str, Any] | None:
    """Return count, min, p25, median, p75, max, and iqr matching verify_tf.py."""
    if not xs:
        return None
    vals = sorted(xs)
    if len(vals) == 1:
        v = vals[0]
        return {
            "count": 1,
            "min": v,
            "p25": v,
            "median": v,
            "p75": v,
            "max": v,
            "iqr": 0 if isinstance(v, int) else 0.0,
        }
    q = statistics.quantiles(vals, n=4, method="inclusive")
    return {
        "count": len(vals),
        "min": vals[0],
        "p25": q[0],
        "median": q[1],
        "p75": q[2],
        "max": vals[-1],
        "iqr": q[2] - q[0],
    }


def ordered_categories(seen: set[str]) -> list[str]:
    known = [c for c in PRIMARY_CATEGORIES if c in seen]
    extra = sorted(c for c in seen if c not in PRIMARY_CATEGORIES)
    return known + extra


def summarize_engine(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Compute per-engine Claim 1 metrics directly from verify_tf.py verdicts."""
    if "verdicts" not in payload or not isinstance(payload["verdicts"], list):
        raise ValueError(f"engine '{name}' missing 'verdicts' list")

    verdicts = sorted(payload["verdicts"], key=lambda v: str(v["id"]))
    n = len(verdicts)
    if n == 0:
        raise ValueError(f"engine '{name}' has 0 verdicts")

    by_id: dict[str, dict[str, Any]] = {}
    cat_verdicts: dict[str, list[dict[str, Any]]] = {}
    for v in verdicts:
        pid = str(v["id"])
        if pid in by_id:
            raise ValueError(f"engine '{name}' has duplicate verdict id '{pid}'")
        by_id[pid] = v
        cat = str(v.get("category") or "uncategorized")
        cat_verdicts.setdefault(cat, []).append(v)

    k = sum(1 for v in verdicts if v["identical"])
    lo, hi = wilson_interval(k, n)

    per_category: dict[str, dict[str, Any]] = {}
    for cat in ordered_categories(set(cat_verdicts)):
        cvs = cat_verdicts[cat]
        nc = len(cvs)
        kc = sum(1 for v in cvs if v["identical"])
        clo, chi = wilson_interval(kc, nc)
        per_category[cat] = {
            "identical": kc,
            "n": nc,
            "rate": kc / nc,
            "wilson_95": [clo, chi],
        }

    enc_bad = [v["id"] for v in verdicts if not v["encoding_equal"]]
    incomplete = [v["id"] for v in verdicts if not v["complete"]]
    disagreed = [v for v in verdicts if not v["identical"] and v["first_diff"] is not None]
    non_id_no_diff = [v["id"] for v in verdicts if not v["identical"] and v["first_diff"] is None]

    fds = [int(v["first_diff"]) for v in disagreed]
    margins = [float(v["ref_margin_at_first_diff"]) for v in disagreed]
    buckets = {
        "j=0": sum(1 for f in fds if f == 0),
        "1-15": sum(1 for f in fds if 1 <= f <= 15),
        "16-63": sum(1 for f in fds if 16 <= f <= 63),
        "64-255": sum(1 for f in fds if 64 <= f <= 255),
    }

    exact_ties = sum(int(v["n_exact_ties"]) for v in verdicts)
    prompts_with_ties = [v["id"] for v in verdicts if int(v["n_exact_ties"]) > 0]

    flagged_positions = sum(len(v.get("head_shape_flagged", [])) for v in verdicts)
    prompts_flagged = [v["id"] for v in verdicts if v.get("head_shape_flagged")]
    depends_flagged = [v["id"] for v in verdicts if v.get("verdict_depends_on_flagged")]
    depends_identical = [
        v["id"] for v in verdicts if v["identical"] and v.get("verdict_depends_on_flagged")
    ]
    depends_non_identical = [
        v["id"] for v in verdicts if (not v["identical"]) and v.get("verdict_depends_on_flagged")
    ]

    return {
        "engine": name,
        "n": n,
        "identical": k,
        "rate": k / n,
        "wilson_95": [lo, hi],
        "per_category": per_category,
        "exact_tie_positions": exact_ties,
        "prompts_with_exact_ties": prompts_with_ties,
        "encoding_discrepancies": enc_bad,
        "incomplete": incomplete,
        "non_identical_without_first_diff": non_id_no_diff,
        "first_diff": quartiles_iqr(fds),
        "margin_at_first_diff": quartiles_iqr(margins),
        "first_diff_buckets": buckets,
        "head_shape": {
            "flagged_positions": flagged_positions,
            "prompts_with_flagged_positions": prompts_flagged,
            "verdicts_depending_on_flagged": depends_flagged,
            "identical_depending_on_flagged": depends_identical,
            "non_identical_depending_on_flagged": depends_non_identical,
        },
        "_by_id": by_id,
    }


def _fmt_median(q: dict[str, Any] | None) -> str:
    if q is None:
        return "none"
    m = q["median"]
    if isinstance(m, int) or (isinstance(m, float) and m.is_integer()):
        return str(int(m))
    return f"{m:.1f}"


def _fmt_p(p: float) -> str:
    if p == 1.0:
        return "1.0000"
    if p < 1e-4:
        return f"{p:.3e}"
    return f"{p:.4f}"


def _display_engine_name(key: str) -> str:
    low = key.lower()
    if low in ("llama", "llama.cpp", "llamacpp", "mainline"):
        return "llama.cpp"
    if low == "unsloth":
        return "Unsloth llama.cpp"
    if low in ("runner", "runner-ref", "runner-fast"):
        return "Runner"
    return key


def render_claim1_wording(
    stats_a: dict[str, Any],
    stats_b: dict[str, Any],
    pair: dict[str, Any],
    *,
    name_a: str | None = None,
    name_b: str | None = None,
    quantized_file: str | None = None,
) -> str:
    """Format PREREG 1.4 Claim 1 wording for the observed paired outcome."""
    disp_a = name_a or _display_engine_name(stats_a["engine"])
    disp_b = name_b or _display_engine_name(stats_b["engine"])
    n = pair["n"]
    k_a = stats_a["identical"]
    k_b = stats_b["identical"]
    lo_a, hi_a = stats_a["wilson_95"]
    lo_b, hi_b = stats_b["wilson_95"]
    delta = pair["delta"]
    delta_lo = pair["delta_lo"]
    delta_hi = pair["delta_hi"]
    p_str = _fmt_p(pair["mcnemar_p"])
    t_e = f"{disp_a} {stats_a['exact_tie_positions']}, {disp_b} {stats_b['exact_tie_positions']}"
    m_a = _fmt_median(stats_a["first_diff"])
    m_b = _fmt_median(stats_b["first_diff"])

    file_higher = quantized_file if quantized_file else "the exact UD-Q8_K_XL GGUF"
    file_other = quantized_file if quantized_file else "UD-Q8_K_XL"

    outcome = pair["outcome"]
    if outcome == "higher":
        text = (
            f"On {n} pre-registered everyday prompts (greedy, max 256 tokens, "
            f"reasoning_effort=low) using {file_higher} verified by teacher-forced forward "
            f"against the float32 transformers reference, {disp_a} matched the reference's "
            f"complete greedy generation on more prompts than {disp_b}: {k_a}/{n} "
            f"(95% Wilson [{lo_a:.3f}, {hi_a:.3f}]) vs {k_b}/{n} "
            f"(95% Wilson [{lo_b:.3f}, {hi_b:.3f}]; paired difference {delta:+.3f} "
            f"[Newcombe 95% CI {delta_lo:+.3f}, {delta_hi:+.3f}], McNemar p={p_str}, "
            f"exact ties {t_e}); first token divergence fell at median index {m_a} vs {m_b}."
        )
    elif outcome == "tie":
        text = (
            f"On {n} pre-registered everyday prompts (greedy, max 256 tokens, "
            f"reasoning_effort=low) using {file_other} verified by teacher-forced forward "
            f"against the float32 reference, {disp_a} matched the reference on {k_a}/{n} "
            f"(95% Wilson [{lo_a:.3f}, {hi_a:.3f}]) compared with {disp_b} at {k_b}/{n} "
            f"(95% Wilson [{lo_b:.3f}, {hi_b:.3f}]; paired difference {delta:+.3f} "
            f"[Newcombe 95% CI {delta_lo:+.3f}, {delta_hi:+.3f}], McNemar p={p_str}, "
            f"no statistically distinguishable difference, exact ties {t_e}); "
            f"first token divergence fell at median index {m_a} vs {m_b}."
        )
    else:
        text = (
            f"On {n} pre-registered everyday prompts (greedy, max 256 tokens, "
            f"reasoning_effort=low) using {file_other} verified by teacher-forced forward "
            f"against the float32 reference, {disp_a} matched the float32 reference on "
            f"fewer prompts than {disp_b}: {k_a}/{n} (95% Wilson [{lo_a:.3f}, {hi_a:.3f}]) "
            f"vs {k_b}/{n} (95% Wilson [{lo_b:.3f}, {hi_b:.3f}]; paired difference "
            f"{delta:+.3f} [Newcombe 95% CI {delta_lo:+.3f}, {delta_hi:+.3f}], "
            f"McNemar p={p_str}, exact ties {t_e}); first divergence occurred at median "
            f"token index {m_a} ({disp_a}) vs {m_b} ({disp_b})."
        )

    if quantized_file:
        text += (
            " (Quantized file: measured against the unquantized float32 reference "
            "to show quantization step-down cost)"
        )
    return text


def compare_pair(stats_a: dict[str, Any], stats_b: dict[str, Any]) -> dict[str, Any]:
    """Build overall and per-category paired tables for engine A minus engine B."""
    by_a = stats_a["_by_id"]
    by_b = stats_b["_by_id"]
    if set(by_a) != set(by_b):
        only_a = sorted(set(by_a) - set(by_b))
        only_b = sorted(set(by_b) - set(by_a))
        raise ValueError(
            f"prompt ID mismatch between '{stats_a['engine']}' and '{stats_b['engine']}': "
            f"only_a={only_a[:5]}, only_b={only_b[:5]}"
        )
    ids = sorted(by_a)
    n11 = n10 = n01 = n00 = 0
    cat_cells: dict[str, list[int]] = {}
    for pid in ids:
        va = by_a[pid]
        vb = by_b[pid]
        ca = str(va.get("category") or "uncategorized")
        cb = str(vb.get("category") or "uncategorized")
        if ca != cb:
            raise ValueError(f"category mismatch on '{pid}': {ca} vs {cb}")
        cells = cat_cells.setdefault(ca, [0, 0, 0, 0])
        a_ok = bool(va["identical"])
        b_ok = bool(vb["identical"])
        if a_ok and b_ok:
            n11 += 1
            cells[0] += 1
        elif a_ok and not b_ok:
            n10 += 1
            cells[1] += 1
        elif (not a_ok) and b_ok:
            n01 += 1
            cells[2] += 1
        else:
            n00 += 1
            cells[3] += 1

    overall = paired_newcombe_method10(n11, n10, n01, n00)
    per_category = {
        cat: paired_newcombe_method10(*cat_cells[cat])
        for cat in ordered_categories(set(cat_cells))
    }
    return {
        "engine_a": stats_a["engine"],
        "engine_b": stats_b["engine"],
        "overall": overall,
        "per_category": per_category,
    }


def _baseline_sort_key(name: str) -> tuple[int, str]:
    low = name.lower()
    if "llama" in low and "unsloth" not in low:
        return (0, low)
    if "unsloth" in low:
        return (1, low)
    return (2, low)


def load_verify_outputs(paths: list[Path]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load and merge engines from one or more verify_tf.py v2 OUT.json files."""
    if not paths:
        raise ValueError("at least one OUT.json path is required")
    merged_engines: dict[str, dict[str, Any]] = {}
    meta: list[dict[str, Any]] = []
    head_shape_bound: float | None = None

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        info = data.get("info", {})
        if info.get("tool") != "verify_tf.py" or info.get("version") != 2:
            raise ValueError(
                f"{path}: expected verify_tf.py version 2 output, got "
                f"tool={info.get('tool')!r}, version={info.get('version')!r}"
            )
        if head_shape_bound is None:
            head_shape_bound = float(info.get("head_shape_bound", 1e-4))
        engines = data.get("engines")
        if not isinstance(engines, dict) or not engines:
            raise ValueError(f"{path}: missing or empty 'engines' object")
        for ename, epayload in engines.items():
            if ename in merged_engines:
                raise ValueError(f"duplicate engine key '{ename}' across input files")
            merged_engines[ename] = epayload
        meta.append({"path": str(path), "info": info})

    return {"sources": meta, "head_shape_bound": head_shape_bound}, merged_engines


def analyze(
    paths: list[Path],
    *,
    runner_name: str | None = None,
    compare_names: list[str] | None = None,
    quantized_file: str | None = None,
) -> dict[str, Any]:
    """Compute the full pre-registered Claim 1 statistical report."""
    meta, raw_engines = load_verify_outputs(paths)
    engine_stats = {
        ename: summarize_engine(ename, payload) for ename, payload in raw_engines.items()
    }

    if runner_name is None:
        runner_candidates = [k for k in engine_stats if "runner" in k.lower()]
        if len(runner_candidates) == 1:
            runner_key = runner_candidates[0]
        elif len(engine_stats) >= 1 and "runner" in engine_stats:
            runner_key = "runner"
        else:
            raise ValueError(
                f"could not uniquely identify Runner engine in {list(engine_stats)}; pass --runner"
            )
    else:
        if runner_name not in engine_stats:
            raise ValueError(f"--runner '{runner_name}' not in engines {list(engine_stats)}")
        runner_key = runner_name

    if compare_names:
        for cname in compare_names:
            if cname not in engine_stats:
                raise ValueError(f"--compare '{cname}' not in engines {list(engine_stats)}")
        comp_keys = list(compare_names)
    else:
        comp_keys = sorted(
            (k for k in engine_stats if k != runner_key), key=_baseline_sort_key
        )

    pairs = []
    for ckey in comp_keys:
        pair_res = compare_pair(engine_stats[runner_key], engine_stats[ckey])
        pair_res["claim_wording"] = render_claim1_wording(
            engine_stats[runner_key],
            engine_stats[ckey],
            pair_res["overall"],
            quantized_file=quantized_file,
        )
        pairs.append(pair_res)

    public_engines = {
        k: {key: val for key, val in st.items() if key != "_by_id"}
        for k, st in engine_stats.items()
    }

    ordered_keys = [runner_key] + [k for k in comp_keys if k != runner_key]
    return {
        "tool": "claim1_stats.py",
        "prereg": "PREREG 0.2, 1.3, 1.4; ERRATUM-2026-11-24; ADDENDUM-2026-11-24-unsloth",
        "implementation": {
            "pure_python_stdlib": True,
            "third_party_statistics_packages": [],
            "z_95": Z_95,
            "quantile_method": "statistics.quantiles(..., n=4, method='inclusive')",
        },
        "sources": meta["sources"],
        "head_shape_bound": meta["head_shape_bound"],
        "quantized_file": quantized_file,
        "engine_order": ordered_keys,
        "engines": public_engines,
        "paired": pairs,
    }


def format_report(res: dict[str, Any]) -> str:
    """Format the Claim 1 report as clean deterministic Markdown."""
    lines: list[str] = []
    lines.append("# Claim 1 Statistical Report (Pre-Registered Analysis)")
    lines.append("")
    lines.append(
        "Method: PREREG 0.2, 1.3, 1.4 (95% Wilson score CI with z=1.959963984540054; "
        "Newcombe 1998 Method 10 paired 95% CI; exact two-sided McNemar p; "
        "Erratum 2 head-shape bound "
        f"{res['head_shape_bound']}); pure Python stdlib, no third-party statistics packages."
    )
    if res["quantized_file"]:
        lines.append(
            f"Quantized file label ({res['quantized_file']}): measured against the unquantized "
            "float32 reference to show quantization step-down cost."
        )
    lines.append("")

    lines.append("## Per-Engine Summary")
    lines.append("")
    lines.append(
        "| Engine | Identical | 95% Wilson CI | Exact ties | Encoding diff | Incomplete | "
        "Flagged pos (prompts) | Verdicts depending on flagged |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for ename in res["engine_order"]:
        e = res["engines"][ename]
        lo, hi = e["wilson_95"]
        hs = e["head_shape"]
        lines.append(
            f"| {ename} | {e['identical']}/{e['n']} | [{lo:.4f}, {hi:.4f}] | "
            f"{e['exact_tie_positions']} | {len(e['encoding_discrepancies'])} | "
            f"{len(e['incomplete'])} | "
            f"{hs['flagged_positions']} ({len(hs['prompts_with_flagged_positions'])}) | "
            f"{len(hs['verdicts_depending_on_flagged'])} |"
        )
    lines.append("")

    lines.append("## Per-Category Breakdown")
    lines.append("")
    lines.append("| Engine | Category | Identical | 95% Wilson CI |")
    lines.append("|---|---|---:|---|")
    for ename in res["engine_order"]:
        e = res["engines"][ename]
        for cat, cs in e["per_category"].items():
            lo, hi = cs["wilson_95"]
            lines.append(
                f"| {ename} | {cat} | {cs['identical']}/{cs['n']} | [{lo:.4f}, {hi:.4f}] |"
            )
    lines.append("")

    lines.append("## First Differing Token Distribution and Margin")
    lines.append("")
    lines.append(
        "| Engine | Non-id w/ diff | first_diff (min / p25 / med / p75 / max) | "
        "margin med (IQR) | j=0 | 1-15 | 16-63 | 64-255 |"
    )
    lines.append("|---|---:|---|---|---:|---:|---:|---:|")
    for ename in res["engine_order"]:
        e = res["engines"][ename]
        fd = e["first_diff"]
        mg = e["margin_at_first_diff"]
        bk = e["first_diff_buckets"]
        if fd is None:
            fd_str = "none"
            cnt = 0
        else:
            cnt = fd["count"]
            fd_str = (
                f"{fd['min']} / {fd['p25']:.2f} / {fd['median']:.2f} "
                f"/ {fd['p75']:.2f} / {fd['max']}"
            )
        mg_str = "none" if mg is None else f"{mg['median']:.3e} ({mg['iqr']:.3e})"
        lines.append(
            f"| {ename} | {cnt} | {fd_str} | {mg_str} | "
            f"{bk['j=0']} | {bk['1-15']} | {bk['16-63']} | {bk['64-255']} |"
        )
    lines.append("")

    lines.append("## Paired Comparisons and Claim Wording")
    lines.append("")
    for pair in res["paired"]:
        ov = pair["overall"]
        t = ov["table"]
        lines.append(f"### {pair['engine_a']} vs {pair['engine_b']}")
        lines.append(
            f"- Paired 2x2 (n={ov['n']}): n_11={t['n_11']}, n_10={t['n_10']}, "
            f"n_01={t['n_01']}, n_00={t['n_00']} (m={ov['m_discordant']})"
        )
        lines.append(
            f"- Difference delta = {ov['delta']:+.4f}, phi = {ov['phi']:+.6f}, "
            f"Newcombe Method 10 95% CI = [{ov['delta_lo']:+.4f}, {ov['delta_hi']:+.4f}], "
            f"exact McNemar p = {_fmt_p(ov['mcnemar_p'])}, outcome = {ov['outcome']}"
        )
        lines.append(f"- Claim wording: {pair['claim_wording']}")
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Pre-registered R4.48.12 Claim 1 statistical report from verify_tf.py v2 OUT.json "
            "(pure Python stdlib, no third-party statistics packages)."
        )
    )
    ap.add_argument("inputs", nargs="+", type=Path, help="verify_tf.py version 2 OUT.json file(s)")
    ap.add_argument(
        "--runner",
        default=None,
        help="Engine key for Runner (auto-detected if omitted)",
    )
    ap.add_argument(
        "--compare",
        action="append",
        default=None,
        help="Baseline engine key(s) to compare against Runner (repeatable)",
    )
    ap.add_argument(
        "--quantized-file",
        default=None,
        help="Quantized filename label for secondary quantized-step runs (PREREG 1.4)",
    )
    ap.add_argument("--json-out", type=Path, default=None, help="Write full JSON output to PATH")
    ap.add_argument("--md-out", type=Path, default=None, help="Write Markdown report to PATH")
    args = ap.parse_args(argv)

    res = analyze(
        args.inputs,
        runner_name=args.runner,
        compare_names=args.compare,
        quantized_file=args.quantized_file,
    )
    report = format_report(res)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    if args.md_out is not None:
        args.md_out.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
