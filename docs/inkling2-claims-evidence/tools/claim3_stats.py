#!/usr/bin/env python3
"""R4.48.12 PREREG 0.2 / 3.3 / 3.4: the Claim 3 pre-registered analysis.

usage: claim3_stats.py CLAIM3_DIR PROMPTS_JSONL GGUF OUT_PREFIX

Reads the five pre-registered arms' records (runner-off, runner-lg, runner-lg-rb, llama-off, llama-rb; 40 primaries
each) and writes OUT_PREFIX.json and OUT_PREFIX.txt:
- per arm, overall and by subset: k_loop / 40 (primary loop = span_repeat_loop(R) OR unclosed_thinking_exhaustion)
  and k_usable / 40, each with its 95% Wilson interval; the secondary loop diagnostics (whole_turn_span_loop,
  general_periodic_loop(R), Runner's loop-guard interventions and loop-ended turns); the tool_debug
  with_truncation_recovery companion row (Runner arms, as PREREG 3.3 2 states);
- tokens spent (PREREG 3.3 3): total, mean, median, IQR, p90, max of completion_tokens, split into reasoning / output;
- runner-lg-rb paired against runner-off, llama-off and llama-rb on k_usable: the 2 x 2 table, Newcombe (1998)
  Method 10 95% CI, exact two-sided McNemar p, and the PREREG 3.4 outcome and wording.
The Wilson, Newcombe and McNemar code is claim1_stats.py's (imported; its sha256 is checked).

Text is decoded from the recorded generated ids with the GGUF's own vocabulary (gpt2 byte-level), so Runner's and
llama.cpp's turns are judged on the same bytes; for llama.cpp's turns the decode is checked against its recorded
raw_detokenized. Definitions this script had to make where the PREREG names a field the records do not carry are
listed under 'interpretations' in both outputs. Standard library + gguf-py (for the vocabulary) only.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C1 = HERE / "claim1_stats.py"
C1_SHA = "6632e4f99ca3e15f"
if hashlib.sha256(C1.read_bytes()).hexdigest()[:16] != C1_SHA:
    sys.exit("claim1_stats.py is not %s" % C1_SHA)
sys.path.insert(0, str(HERE))
from claim1_stats import wilson_interval, paired_newcombe_method10, quartiles_iqr  # noqa: E402

ARMS = ("runner-off", "runner-lg", "runner-lg-rb", "llama-off", "llama-rb")
COMPARATORS = ("runner-off", "llama-off", "llama-rb")
GUARDED = "runner-lg-rb"
# Unsloth's column (ADDENDUM-2026-11-24-unsloth.md; its prebuilt, ADDENDUM-2026-11-25-unsloth-prebuilt.md): the same
# summaries, and runner-lg-rb paired against each Unsloth arm exactly as against mainline's, reported as an ADDITIONAL
# column; the pre-registered COMPARATORS, PREREG 3.4 outcome and wording are computed as before and unchanged.
UNSLOTH_ARMS = ("unsloth-off", "unsloth-rb")


def llama_format(arm):
    """the arms whose records are llama-server's (claim_client_llama.py): mainline's and Unsloth's"""
    return arm.startswith(("llama", "unsloth"))


def pair_table(a_by_id, b_by_id, ids):
    t = {"n_11": 0, "n_10": 0, "n_01": 0, "n_00": 0}
    for i in ids:
        t["n_%d%d" % (a_by_id[i]["usable"], b_by_id[i]["usable"])] += 1
    return t
SUBSETS = ("plain_debug", "tool_debug")
T_THINK, T_END, T_TEXT, T_STOP = 200008, 200010, 200004, 200006
CAP = 2048

INTERPRETATIONS = [
    "finish_reason is the response's choices[0].finish_reason (Runner's record field 'finish' is the client's).",
    "R = the generated ids strictly between each <|content_thinking|> (200008) and the next <|end_message|> (200010),"
    " or to the turn's end if never closed, concatenated in order.",
    "unclosed_thinking_exhaustion: completion_tokens == 2048 AND finish_reason == 'length' AND the last"
    " <|content_thinking|> has no later <|end_message|>.",
    "runner_lg_interventions: PREREG 3.3 names runner_telemetry.loop_guard.closes; the records carry"
    " loop_guard.interventions (and ended_turn), so closes > 0 is read as interventions > 0, and a loop-ended turn as"
    " finish_reason == 'loop' OR loop_guard.ended_turn.",
    "plain answer: a <|content_text|> ... <|end_message|> segment outside thinking (decoded text) that holds"
    " 'def <entry_point>(' (inside a ```python block or as plain text); 'stubbed pass only' = the def's body, by ast"
    " when the code parses (docstrings dropped), else by its indented lines, is only 'pass' / '...'.",
    "tool path: the engine's parsed tool_calls (choices[0].message.tool_calls) holds a submit_patch whose arguments"
    " parse to a JSON object with non-empty string fixed_code containing 'def <entry_point>(' and non-empty string"
    " root_cause; primary only with finish_reason in {tool_calls, stop}; finish_reason == 'length' with such a call is"
    " the with_truncation_recovery companion (Runner arms, as PREREG 3.3 2.1 states).",
    "completion_tokens = the GENERATED ID COUNT len(gen_ids) (ruled 2026-11-25, CLAIM3-INTERPRETATIONS-2026-11-25.md 7,"
    " to the earlier addendum); usage.completion_tokens is reported beside it wherever the two differ."
    " reasoning_tokens = len(R) (the records carry no reasoning_tokens field); output_tokens = completion_tokens -"
    " reasoning_tokens.",
    "unit_test_pass (secondary, PREREG 3.3 2): NOT computed for this report (ruled 2026-11-25, item 8): it executes"
    " model-written code, and this shared, borrowed host offers no isolation. Disclosed; nothing fetched.",
    "PREREG 3.4's tie wording's F (the loop guard fired on F/40 turns) = runner-lg-rb turns with interventions > 0;"
    " the worse wording's F (truncated/closed before a usable fix) = runner-lg-rb turns not usable that ended at"
    " 'length' or 'loop', or had a loop-guard intervention or a forced reasoning-budget close.",
]


def load_vocab(gguf_path):
    sys.path.insert(0, str(Path.home() / "r448v/src/llama.cpp/gguf-py"))
    import gguf
    r = gguf.GGUFReader(gguf_path)
    f = r.fields["tokenizer.ggml.tokens"]
    toks = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    inv = {chr(c): b for b, c in zip(bs, cs)}
    return toks, inv


def decode(ids, toks, inv):
    out = bytearray()
    for i in ids:
        t = toks[i]
        if i >= 200000 and t.startswith("<|") and t.endswith("|>"):
            out += t.encode()
        else:
            out += bytes(inv[ch] for ch in t)
    return out.decode("utf-8", errors="replace")


def span_repeat_loop(x, span=8, reps=3):
    for i in range(0, len(x) - span * reps + 1):
        if all(x[i:i + span] == x[i + k * span:i + (k + 1) * span] for k in range(1, reps)):
            return True
    return False


def general_periodic_loop(x):
    n = len(x)
    for p in range(1, 129):
        need = p * (max(3, math.ceil(24 / p)) - 1)  # consecutive j with x[j] == x[j+p]
        run = 0
        for j in range(0, n - p):
            run = run + 1 if x[j] == x[j + p] else 0
            if run >= need:
                return True
    return False


def thinking_split(g):
    """R (ids inside thinking), outside (ids outside thinking), unclosed (last thinking block never closed)."""
    r, out, inside = [], [], False
    for t in g:
        if not inside and t == T_THINK:
            inside = True
            continue
        if inside and t == T_END:
            inside = False
            continue
        (r if inside else out).append(t)
    return r, out, inside


def text_segments(outside_ids, toks, inv):
    segs, cur = [], None
    for t in outside_ids:
        if t == T_TEXT:
            cur = []
        elif t == T_END or (t >= 200000 and cur is not None and toks[t].startswith("<|message")):
            if cur is not None:
                segs.append(decode(cur, toks, inv))
            cur = None
        elif cur is not None:
            cur.append(t)
    if cur is not None:  # an unterminated text segment is not '<|content_text|>...<|end_message|>'
        pass
    return segs


def body_is_stub(code, ep):
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == ep:
                body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant) and isinstance(s.value.value, str))]
                return all(isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant) and s.value.value is Ellipsis) for s in body)
    except (SyntaxError, ValueError):
        pass
    lines = code.splitlines()
    m = [k for k, l in enumerate(lines) if re.match(r"\s*def\s+%s\s*\(" % re.escape(ep), l)]
    if not m:
        return True
    k = m[0]
    ind = len(lines[k]) - len(lines[k].lstrip())
    body = []
    for l in lines[k + 1:]:
        if l.strip() and len(l) - len(l.lstrip()) <= ind:
            break
        s = l.strip()
        if s and not s.startswith("#"):
            body.append(s)
    return all(s in ("pass", "...") for s in body)


def code_answer(segs, ep):
    """A usable 'def <ep>(' in a ```python block or as plain text, not a pass-only stub."""
    pat = re.compile(r"def\s+%s\s*\(" % re.escape(ep))
    for s in segs:
        if not s.strip():
            continue
        for blk in re.findall(r"```python[^\n]*\n(.*?)(?:```|\Z)", s, flags=re.S):
            if pat.search(blk) and not body_is_stub(blk, ep):
                return True
        if pat.search(s):
            start = pat.search(s).start()
            if not body_is_stub(s[s.rfind("\n", 0, start) + 1:], ep):
                return True
    return False


def tool_ok(msg, ep):
    pat = re.compile(r"def\s+%s\s*\(" % re.escape(ep))
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        if fn.get("name") != "submit_patch":
            continue
        try:
            a = json.loads(fn.get("arguments") or "")
        except (TypeError, ValueError):
            continue
        if isinstance(a, dict) and isinstance(a.get("fixed_code"), str) and a["fixed_code"].strip() and pat.search(a["fixed_code"]) \
                and isinstance(a.get("root_cause"), str) and a["root_cause"].strip():
            return True
    return False


def judge(rec, prompt, toks, inv, arm):
    g = rec["gen_ids"]
    ch = rec["response"]["choices"][0]
    fr = ch.get("finish_reason")
    msg = ch.get("message") or {}
    ct = len(g)  # the generated id count (interpretation 7, ruled)
    uct = rec["response"]["usage"]["completion_tokens"]
    ep = prompt["entry_point"]
    r, outside, unclosed = thinking_split(g)
    segs = text_segments(outside, toks, inv)
    ex = ct == CAP and fr == "length" and unclosed
    srl = span_repeat_loop(r)
    lg = ((rec["response"].get("runner_telemetry") or {}).get("loop_guard")) or {}
    rb = ((rec["response"].get("runner_telemetry") or {}).get("reasoning_budget")) or {}
    clean = fr == "stop"
    plain = clean and code_answer(segs, ep)
    if prompt["subset"] == "plain_debug":
        usable, path, trunc = plain, ("text" if plain else None), False
    else:
        t_ok = tool_ok(msg, ep)
        tool_primary = t_ok and fr in ("tool_calls", "stop")
        usable = tool_primary or plain
        path = "tool" if tool_primary else ("text" if plain else None)
        trunc = arm.startswith("runner") and (not usable) and t_ok and fr == "length"
    return {
        "id": rec["id"], "subset": prompt["subset"], "finish_reason": fr, "completion_tokens": ct,
        "usage_completion_tokens": uct, "usage_differs": uct != ct,
        "stop_token_last": bool(g) and g[-1] == T_STOP,
        "reasoning_tokens": len(r), "output_tokens": ct - len(r),
        "span_repeat_loop": srl, "unclosed_thinking_exhaustion": ex, "loop": srl or ex,
        "whole_turn_span_loop": span_repeat_loop(g), "general_periodic_loop": general_periodic_loop(r),
        "lg_interventions": lg.get("interventions") if lg else None, "lg_ended_turn": bool(lg.get("ended_turn")) if lg else None,
        "rb_forced_close": bool(rb.get("forced_close")) if rb else None,
        "usable": usable, "usable_path": path, "with_truncation_recovery": trunc,
    }


def share(k, n):
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "wilson_95": [lo, hi]}


def tok_stats(xs):
    q = quartiles_iqr(xs)
    return {"total": sum(xs), "mean": sum(xs) / len(xs), "median": q["median"], "p25": q["p25"], "p75": q["p75"],
            "iqr": q["iqr"], "p90": statistics.quantiles(sorted(xs), n=10, method="inclusive")[8], "max": max(xs)}


def summarize(rows):
    n = len(rows)
    s = {"n": n, "loop": share(sum(r["loop"] for r in rows), n), "usable": share(sum(r["usable"] for r in rows), n),
         "span_repeat_loop": sum(r["span_repeat_loop"] for r in rows),
         "unclosed_thinking_exhaustion": sum(r["unclosed_thinking_exhaustion"] for r in rows),
         "whole_turn_span_loop": sum(r["whole_turn_span_loop"] for r in rows),
         "general_periodic_loop": sum(r["general_periodic_loop"] for r in rows),
         "finish_reasons": {}, "tokens": {k: tok_stats([r[k] for r in rows]) for k in ("completion_tokens", "reasoning_tokens", "output_tokens")}}
    dif = [r for r in rows if r["usage_differs"]]
    s["usage_completion_tokens"] = {"turns_differing": len(dif), "diffs": sorted({r["completion_tokens"] - r["usage_completion_tokens"] for r in dif}),
                                    "finish_reasons_differing": sorted({r["finish_reason"] for r in dif})}
    if dif:
        s["usage_completion_tokens"]["stats"] = tok_stats([r["usage_completion_tokens"] for r in rows])
    for r in rows:
        s["finish_reasons"][r["finish_reason"]] = s["finish_reasons"].get(r["finish_reason"], 0) + 1
    if rows and rows[0]["lg_interventions"] is not None:
        s["runner_lg_interventions"] = {"turns_with_interventions": sum((r["lg_interventions"] or 0) > 0 for r in rows),
                                        "turns_loop_ended": sum(r["finish_reason"] == "loop" or bool(r["lg_ended_turn"]) for r in rows)}
    if rows and rows[0]["rb_forced_close"] is not None:
        s["reasoning_budget_forced_close"] = sum(bool(r["rb_forced_close"]) for r in rows)
    tool = [r for r in rows if r["subset"] == "tool_debug"]
    if tool:
        s["tool_paths"] = {"tool": sum(r["usable_path"] == "tool" for r in tool), "text": sum(r["usable_path"] == "text" for r in tool)}
        s["with_truncation_recovery"] = sum(r["with_truncation_recovery"] for r in tool)
    return s


def main():
    d, prompts_path, gguf_path, outp = sys.argv[1:5]
    prompts = {p["id"]: p for p in map(json.loads, open(prompts_path)) if p["split"] == "primary"}
    if len(prompts) != 40:
        sys.exit("expected 40 primaries, got %d" % len(prompts))
    toks, inv = load_vocab(gguf_path)
    per, inputs, checks = {}, {}, {}
    have_u = all((Path(d) / ("%s.jsonl" % a)).exists() for a in UNSLOTH_ARMS)
    arms = ARMS + (UNSLOTH_ARMS if have_u else ())
    for arm in arms:
        p = Path(d) / ("%s.jsonl" % arm)
        raw = p.read_bytes()
        inputs[arm] = hashlib.sha256(raw).hexdigest()
        recs = [json.loads(l) for l in raw.decode().splitlines()]
        ids = [r["id"] for r in recs]
        if sorted(ids) != sorted(prompts) or len(set(ids)) != 40:
            sys.exit("%s: the records are not the 40 primaries once each" % arm)
        per[arm] = [judge(r, prompts[r["id"]], toks, inv, arm) for r in recs]
        if llama_format(arm):
            eq = sum(decode(r["gen_ids"], toks, inv) == r.get("raw_detokenized") for r in recs)
            checks[arm] = "decode(gen_ids) == raw_detokenized on %d / 40" % eq
        checks[arm + " stop"] = "finish_reason 'stop' turns ending in 200006: %d / %d" % (
            sum(x["stop_token_last"] for x in per[arm] if x["finish_reason"] == "stop"), sum(x["finish_reason"] == "stop" for x in per[arm]))
    summ = {arm: {"all": summarize(per[arm]), **{s: summarize([r for r in per[arm] if r["subset"] == s]) for s in SUBSETS}} for arm in arms}
    gid = {r["id"]: r for r in per[GUARDED]}
    pairs = {}
    for b in COMPARATORS + (UNSLOTH_ARMS if have_u else ()):
        t = pair_table(gid, {r["id"]: r for r in per[b]}, prompts)
        pairs[b] = paired_newcombe_method10(t["n_11"], t["n_10"], t["n_01"], t["n_00"])
    oc = {b: pairs[b]["outcome"] for b in COMPARATORS}
    if all(o == "higher" for o in oc.values()):
        outcome = "beats"
    elif any(o == "lower" for o in oc.values()):
        outcome = "worse"
    else:
        outcome = "tie"
    A = lambda arm: summ[arm]["all"]
    K = lambda arm: A(arm)["usable"]["k"]
    L = lambda arm: A(arm)["loop"]["k"]
    T = lambda arm: A(arm)["tokens"]["completion_tokens"]["median"]
    lo_g, hi_g = A(GUARDED)["usable"]["wilson_95"]
    fmt_t = lambda v: ("%g" % v)
    if outcome == "beats":
        wording = ("On 40 code-debugging prompts (greedy, max 2048 tokens, reasoning_effort=high, UD-Q8_K_XL), Runner with --loop-guard and"
                   " --reasoning-budget 1024 delivered more usable answers or parseable tool calls than every comparator: %d/40 turns (95%% Wilson"
                   " [%.3f, %.3f], loop rate %d/40, median %s tokens), vs %d/40 (loop %d/40, median %s tokens) for unguarded Runner, %d/40 (loop"
                   " %d/40, median %s tokens) for unrestricted llama.cpp, and %d/40 (loop %d/40, median %s tokens) for budgeted llama.cpp (every"
                   " pairwise Newcombe 95%% CI excludes 0; McNemar p <= %.4g)." % (
                       K(GUARDED), lo_g, hi_g, L(GUARDED), fmt_t(T(GUARDED)), K("runner-off"), L("runner-off"), fmt_t(T("runner-off")),
                       K("llama-off"), L("llama-off"), fmt_t(T("llama-off")), K("llama-rb"), L("llama-rb"), fmt_t(T("llama-rb")),
                       max(pairs[b]["mcnemar_p"] for b in COMPARATORS)))
    elif outcome == "tie":
        f = A(GUARDED)["runner_lg_interventions"]["turns_with_interventions"]
        names = ", ".join(b for b in COMPARATORS if oc[b] == "tie")
        wording = ("On 40 code-debugging prompts (UD-Q8_K_XL, greedy, reasoning_effort=high), guarded Runner yielded %d/40 usable turns (95%%"
                   " Wilson [%.3f, %.3f], loop %d/40, median %s tokens) vs unguarded Runner %d/40 (%d/40, %s tokens) and llama.cpp %d/40"
                   " unrestricted / %d/40 budgeted, with no statistically distinguishable paired difference on usable turns against"
                   " comparator(s) %s (paired Newcombe 95%% CI includes 0); the loop guard fired on %d/40 turns." % (
                       K(GUARDED), lo_g, hi_g, L(GUARDED), fmt_t(T(GUARDED)), K("runner-off"), L("runner-off"), fmt_t(T("runner-off")),
                       K("llama-off"), K("llama-rb"), names, f))
    else:
        worse = [b for b in COMPARATORS if oc[b] == "lower"]
        b0 = worse[0]
        f = sum((not r["usable"]) and (r["finish_reason"] in ("length", "loop") or (r["lg_interventions"] or 0) > 0 or bool(r["rb_forced_close"])) for r in per[GUARDED])
        wording = ("On 40 code-debugging prompts (UD-Q8_K_XL, greedy, reasoning_effort=high), Runner with --loop-guard and --reasoning-budget"
                   " 1024 finished fewer usable turns than %s: %d/40 (95%% Wilson [%.3f, %.3f], loop %d/40, median %s tokens) vs unguarded"
                   " Runner %d/40 and llama.cpp %d/40 unrestricted / %d/40 budgeted (Newcombe 95%% CI [%.3f, %.3f] excludes 0, McNemar p=%.4g);"
                   " the guarded arms truncated/closed %d/40 turns before reaching a usable fix." % (
                       ", ".join(worse), K(GUARDED), lo_g, hi_g, L(GUARDED), fmt_t(T(GUARDED)), K("runner-off"), K("llama-off"), K("llama-rb"),
                       pairs[b0]["delta_lo"], pairs[b0]["delta_hi"], pairs[b0]["mcnemar_p"], f))
    status = ("FINAL: the five pre-registered arms and Unsloth's two (an additional column)" if have_u
              else "INFORMATION until the Unsloth arms are settled (the supervisor 2026-11-25 ~18:00Z)")
    out = {"script": "claim3_stats.py", "status": status,
           "inputs_sha256": inputs, "prompts_sha256": hashlib.sha256(Path(prompts_path).read_bytes()).hexdigest(),
           "claim1_stats_sha256_16": C1_SHA, "checks": checks, "interpretations": INTERPRETATIONS,
           "arms": summ, "paired_usable": {"%s vs %s" % (GUARDED, b): pairs[b] for b in COMPARATORS},
           "outcome": outcome, "wording": wording, "turns": per}
    if have_u:
        out["unsloth_column"] = {
            "basis": "ADDENDUM-2026-11-24-unsloth.md: Runner against Unsloth's build computed exactly as against mainline,"
                     " reported as an additional column; the pre-registered outcome and wording above are unchanged",
            "paired_usable": {"%s vs %s" % (GUARDED, b): pairs[b] for b in UNSLOTH_ARMS}}
    Path(outp + ".json").write_text(json.dumps(out, indent=1, sort_keys=False) + "\n")
    L_ = []
    w = L_.append
    w("CLAIM 3 (PREREG 3.3 / 3.4), %s" % out["status"])
    w("inputs: " + ", ".join("%s %s" % (a, inputs[a][:16]) for a in arms))
    w("checks: " + "; ".join("%s: %s" % kv for kv in checks.items()))
    w("")
    w("%-13s %-11s %-22s %-22s %4s %4s %4s %4s  %s" % ("arm", "subset", "usable k/n [Wilson]", "loop k/n [Wilson]", "srl", "exh", "wtl", "gpl", "finish"))
    for arm in arms:
        for s in ("all",) + SUBSETS:
            x = summ[arm][s]
            u, l = x["usable"], x["loop"]
            w("%-13s %-11s %2d/%-2d [%.3f, %.3f]   %2d/%-2d [%.3f, %.3f]   %4d %4d %4d %4d  %s" % (
                arm if s == "all" else "", s, u["k"], u["n"], u["wilson_95"][0], u["wilson_95"][1], l["k"], l["n"], l["wilson_95"][0], l["wilson_95"][1],
                x["span_repeat_loop"], x["unclosed_thinking_exhaustion"], x["whole_turn_span_loop"], x["general_periodic_loop"],
                ", ".join("%s %d" % kv for kv in sorted(x["finish_reasons"].items(), key=lambda kv: -kv[1]))))
    w("  srl = span_repeat_loop(R), exh = unclosed_thinking_exhaustion (loop = srl OR exh); wtl = whole_turn_span_loop, gpl = general_periodic_loop(R) (secondary)")
    w("")
    for arm in arms:
        x = summ[arm]["all"]
        extra = []
        if "runner_lg_interventions" in x:
            extra.append("loop guard: interventions on %(turns_with_interventions)d turns, loop-ended %(turns_loop_ended)d" % x["runner_lg_interventions"])
        if "reasoning_budget_forced_close" in x:
            extra.append("reasoning budget forced close on %d turns" % x["reasoning_budget_forced_close"])
        tp = summ[arm]["tool_debug"]
        extra.append("tool_debug usable via tool %d / text %d" % (tp["tool_paths"]["tool"], tp["tool_paths"]["text"]))
        if arm.startswith("runner"):
            extra.append("with_truncation_recovery (companion, not primary) +%d" % tp["with_truncation_recovery"])
        w("%-13s %s" % (arm, "; ".join(extra)))
    w("")
    w("tokens (PREREG 3.3 3): completion / reasoning / output, per arm over 40 (and by subset)")
    for arm in arms:
        for s in ("all",) + SUBSETS:
            parts = []
            for k in ("completion_tokens", "reasoning_tokens", "output_tokens"):
                t = summ[arm][s]["tokens"][k]
                parts.append("%s total %d mean %.1f median %g IQR %g..%g p90 %g max %d" % (k.split("_")[0], t["total"], t["mean"], t["median"], t["p25"], t["p75"], t["p90"], t["max"]))
            w("%-13s %-11s %s" % (arm if s == "all" else "", s, " | ".join(parts)))
            uc = summ[arm][s]["usage_completion_tokens"]
            if uc["turns_differing"]:
                t = uc["stats"]
                w("%-13s %-11s   beside: usage.completion_tokens differs on %d turns (generated ids - usage = %s; finish %s): total %d mean %.1f median %g IQR %g..%g p90 %g max %d" % (
                    "", "", uc["turns_differing"], uc["diffs"], ", ".join(uc["finish_reasons_differing"]), t["total"], t["mean"], t["median"], t["p25"], t["p75"], t["p90"], t["max"]))
    w("  completion = the generated id count (interpretation 7); usage.completion_tokens is shown beside only where it differs")
    w("")
    w("paired on k_usable (A = %s)" % GUARDED)
    for b in COMPARATORS:
        p = pairs[b]
        w("  vs %-10s n_11 %2d n_10 %2d n_01 %2d n_00 %2d  delta %+.3f  Newcombe 95%% [%+.3f, %+.3f]  McNemar p %.4g  -> %s" % (
            b, p["table"]["n_11"], p["table"]["n_10"], p["table"]["n_01"], p["table"]["n_00"], p["delta"], p["delta_lo"], p["delta_hi"], p["mcnemar_p"], p["outcome"]))
    w("")
    w("PREREG 3.4 outcome: %s" % outcome)
    w("wording: " + wording)
    if have_u:
        w("")
        w("Unsloth's column (ADDENDUM-2026-11-24-unsloth.md; its published prebuilt, ADDENDUM-2026-11-25-unsloth-prebuilt.md),"
          " paired exactly as against mainline; ADDITIONAL, the outcome and wording above unchanged (A = %s)" % GUARDED)
        for b in UNSLOTH_ARMS:
            p = pairs[b]
            w("  vs %-10s n_11 %2d n_10 %2d n_01 %2d n_00 %2d  delta %+.3f  Newcombe 95%% [%+.3f, %+.3f]  McNemar p %.4g  -> %s" % (
                b, p["table"]["n_11"], p["table"]["n_10"], p["table"]["n_01"], p["table"]["n_00"], p["delta"], p["delta_lo"], p["delta_hi"], p["mcnemar_p"], p["outcome"]))
    w("")
    w("interpretations (where the PREREG names a field the records do not carry):")
    for i, s in enumerate(INTERPRETATIONS, 1):
        w("  %d. %s" % (i, s))
    Path(outp + ".txt").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    main()
