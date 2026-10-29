#!/usr/bin/env python3
"""Gold logit check: the model's own reference implementation (transformers,
float32, CPU) against one or two serving endpoints on the same text.

Every cross-engine gate in this repository compares the runner with
llama.cpp, and llama.cpp is a second implementation, not the truth: it has
its own attention precision (its CPU flash-attention kernel accumulates in
f16) and its own quantization of activations. When the two disagree, this
script asks the question the gates cannot: which one is closer to what the
model's authors ship. It scores the reference implementation in float32
against each endpoint, position by position, on a fixed corpus, and reports
the KL divergence and top-1 agreement of each endpoint against the gold
distribution. Run it on an unquantized (bf16/f16) GGUF so weight rounding
is the same on every side and only the arithmetic differs.

Serve both sides at a context the model was TRAINED for. The runner
applies YaRN automatically when `-c` exceeds the training context and says
so on stderr ("requested ctx N > training ctx M"); llama.cpp does not. A
run that crosses that line measures the scaling, not the arithmetic:
measured 2026-09-07 on StableLM 2 1.6B (training context 4096), the same
binary on the same file read mean KL 0.1056 at `-c 8192` and 0.0001 at
`-c 4096`. Cap `--max-positions * --stride` below the training context too,
for the same reason.

    gold-logits.py --hf /path/to/hf-model --corpus tests/fixtures/mixed-corpus.txt \\
        --endpoint-a http://127.0.0.1:58631 --model-name-a X.gguf \\
        --endpoint-b http://127.0.0.1:58632 --model-name-b X.gguf \\
        --max-positions 100 --out gold.json

Endpoints speak /v1/completions with `logprobs` (llama-server and the runner
both do). A server reuses the KV of the prefix it scored one position earlier,
so each position is one decoded token; `--fresh-prompt` sends
`cache_prompt: false`, and every position is then a full prefill of its
prefix. A side can instead be a runner score file (`--score-a FILE`, from
`runner --score --score-top N -f CORPUS`): one pass over the text gives every
position, and with `RUNNER_SCORE_CHUNKED=1` each comes from the batched
prefill route, the arithmetic a prompt takes. The file's token ids must equal
the reference's tokenization of the corpus (BOS included), or the run stops.
The tokenizer is the HF one; the prefix at each position is the decoded
token prefix, so every side sees the same text.

Tokens are matched by id wherever the endpoint gives ids (the runner's
`top_token_ids`, llama-server's `id`), and by spelling otherwise. Spelling
alone cannot match a stop or control token: it decodes to no bytes, so a
server spells it "" while the reference spells it `<|endoftext|>`, and every
such token in a JSON map is the same "" key. Matched by spelling, the gold
mass on it was scored at a floor whatever the engine computed: measured by the
lab on 2026-10-17 on a qwen35 BF16 file, one position read KL 0.112 where the
19 shared tokens agree to 8.2e-9, because the gold top-20 held
`<|endoftext|>` at 2.8%. Where an endpoint gives no ids, a reference special
token it did not list is dropped, the gold renormalised over the rest, and the
dropped mass reported per row, instead of a floor.
"""
import argparse
import json
import math
import sys
import urllib.request


class _Obj(dict):
    """A JSON object that also keeps its members in order, duplicates
    included. The runner spells every control token "" in a top_logprobs
    map, so two of them in one map are two "" keys, and a plain dict keeps
    only the last; the parallel `top_token_ids` array needs every member."""

    def __init__(self, pairs):
        super().__init__(pairs)
        self.pairs = list(pairs)


class Top:
    """One endpoint's answer at one position: logprob by spelling, by id
    (None when the endpoint gives no ids), each id's spelling as the
    endpoint wrote it, and its top-1."""

    def __init__(self, by_spell, by_id, spelled, top1_spell, top1_id):
        self.by_spell, self.by_id, self.spelled = by_spell, by_id, spelled
        self.top1_spell, self.top1_id = top1_spell, top1_id


def _runner_top(pairs, ids, chosen=None):
    """The runner's parallel shape: (spelling, logprob) pairs in order, the
    ids in the same order, and the chosen (spelling, id, logprob) if any."""
    by_spell = dict(pairs)
    by_id = spelled = None
    if ids is not None and len(ids) == len(pairs):
        by_id = {i: v for i, (_, v) in zip(ids, pairs)}
        spelled = {i: t for i, (t, _) in zip(ids, pairs)}
    entries = [(v, t, ids[k] if by_id is not None else None)
               for k, (t, v) in enumerate(pairs)]
    if chosen:
        t, i, v = chosen
        by_spell[t] = v
        if by_id is not None and i is not None:
            by_id[i], spelled[i] = v, t
        entries.append((v, t, i if by_id is not None else None))
    v, t, i = max(entries, key=lambda e: e[0]) if entries else (None, None, None)
    return Top(by_spell, by_id, spelled, t, i)


def parse_top(d):
    """The distribution a /v1/completions response gives for its one
    generated position, or None when it gives none."""
    ch = d["choices"][0]
    lp = ch.get("logprobs")
    if lp and "content" in lp:  # llama.cpp's OpenAI-style schema
        if not lp["content"]:
            return None
        step = lp["content"][0]
        alts = step["top_logprobs"]
        by_spell = {e["token"]: e["logprob"] for e in alts}
        by_spell[step["token"]] = step["logprob"]
        by_id = spelled = None
        if "id" in step and all("id" in e for e in alts):
            by_id = {e["id"]: e["logprob"] for e in alts}
            spelled = {e["id"]: e["token"] for e in alts}
            by_id[step["id"]], spelled[step["id"]] = step["logprob"], step["token"]
        # The top-1 is the distribution's argmax, as on the runner's side, not
        # the emitted token: llama-server applies the model file's sampling
        # defaults before a temperature-0 pick (Qwen2.5 ships repeat_penalty
        # 1.1 over 64 tokens), so its pick missed its own argmax on 19 of 100
        # Qwen2.5-0.5B positions, measured 2026-10-22.
        best = max(list(alts) + [step], key=lambda e: e["logprob"])
        return Top(by_spell, by_id, spelled, best["token"], best.get("id"))
    if lp and lp.get("tokens"):  # the runner's schema: parallel arrays
        alts = lp["top_logprobs"][0]
        pairs = alts.pairs if isinstance(alts, _Obj) else list(alts.items())
        ids = (lp.get("top_token_ids") or [None])[0]
        tid = (lp.get("token_ids") or [None])[0]
        return _runner_top(pairs, ids, (lp["tokens"][0], tid, lp["token_logprobs"][0]))
    # The greedy pick was a stop: the runner's logprob arrays align with the
    # emitted text and stay empty, and the decision rides in stop_logprobs.
    # Skipping these positions would drop a one-sided stop from the score.
    sl = ch.get("stop_logprobs")
    if sl:
        alts = sl.get("top_logprobs") or {}
        pairs = alts.pairs if isinstance(alts, _Obj) else list(alts.items())
        return _runner_top(pairs, sl.get("top_token_ids"),
                           ("", ch.get("stop_token_id"), sl["logprob"]))
    # A server may answer a position with no logprobs at all (measured
    # 2026-09-07 on SmolLM2: the request whose next token is the end of
    # text comes back without the block). That is a position this gate
    # cannot score, not a reason to lose the whole family.
    return None


def reference_logp(model, span, rows):
    """The reference's log-probabilities at the scored rows only.

    One forward over the whole span, as before, but the vocabulary
    projection runs on the rows that are scored: `logits_to_keep` with a
    tensor of positions (transformers' generation models take it). The full
    logit tensor is rows x vocabulary in float32, and its log-softmax is a
    second copy: on MiMo-V2.6-Distill-Qwen-9B (4,509 positions, 248,320
    tokens) that is 4.5 GB twice on top of 38 GB of float32 weights, and
    the lab container stopped answering under it (2026-10-29). 300 scored
    rows are 0.3 GB. A model whose forward refuses the argument, or returns
    every row anyway, falls back to the full tensor. Returns (logp, index,
    path): logp[index[row]] is the row, and path says which way it went."""
    import torch
    keep = sorted(set(rows))
    ids = torch.tensor([span])
    with torch.no_grad():
        try:
            out = model(ids, logits_to_keep=torch.tensor(keep))
            lg = out.logits[0]
            if lg.shape[0] != len(keep):
                raise TypeError("logits_to_keep ignored")
            index = {r: i for i, r in enumerate(keep)}
            path = "scored rows only"
        except TypeError:
            out = model(ids)
            lg = out.logits[0]
            index = {r: r for r in keep}
            path = "full tensor"
        logp = torch.log_softmax(lg.float(), dim=-1)
    return logp, index, path


def query(endpoint, model_name, prompt, top_n=20, fresh=False):
    payload = {"model": model_name, "prompt": prompt, "max_tokens": 1,
               "temperature": 0, "logprobs": top_n}
    if fresh:
        payload["cache_prompt"] = False
    req = urllib.request.Request(
        f"{endpoint}/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read(), object_pairs_hook=_Obj)
    return parse_top(d)


def load_score(path, span, top_n):
    """A runner --score --score-top report as a side: per reference row, the
    Top the endpoint would have answered. Refuses a file whose tokens are not
    the reference's span, or whose head is narrower than --top-n."""
    d = json.load(open(path))
    toks = d.get("tokens") or []
    if "top" not in d:
        sys.exit(f"{path}: no 'top' array; score with --score-top N")
    if toks[:len(span)] != list(span):
        first = next((i for i in range(min(len(toks), len(span)))
                      if toks[i] != span[i]), min(len(toks), len(span)))
        sys.exit(f"{path}: token ids differ from the reference at index {first} "
                 f"(file {toks[first:first + 4]}, reference {list(span)[first:first + 4]}); "
                 "score the same corpus with the same BOS handling (--force-bos)")
    if d["top"] and len(d["top"][0]) < min(top_n, d.get("n_vocab", top_n)):
        sys.exit(f"{path}: --score-top {len(d['top'][0])} is narrower than --top-n {top_n}")
    return d["top"]


def score_side(rows_top, row, spell):
    """Top for one reference row from a score file's head."""
    head = rows_top[row]
    by_id = {int(i): float(v) for i, v in head}
    i0, v0 = head[0]
    return Top({}, by_id, None, spell(int(i0)), int(i0))


def token_speller(tok):
    """How a server spells one token in its logprobs: the text the token adds
    inside a sentence.

    `decode([id])` is not that on the families whose tokenizer prepends a
    space (SentencePiece with add_space_prefix: Mistral v0.3, Phi-3.5, Llama
    2). Their decoder strips the leading space of the first piece of every
    decode, so a lone "\u2581new" reads "new" while both servers report " new".
    KL here matches tokens by spelling, so almost every word token went
    unmatched: measured 2026-09-07, Mistral v0.3 read mean KL 1.6114 on both
    engines to four decimals and Phi-3.5 about 1.9, with top-1 agreeing only on
    punctuation. The prefixes the servers scored were the reference's own ids
    (checked 2026-09-30 against the runner's tokenizer at all 100 positions of
    both families); the spelling was the fault. Decoding each token after an
    anchor token and keeping what it added gives the in-sentence spelling on
    every family, and the same spelling as before on byte-level BPE ones
    (Qwen3, Granite). No cleanup either: a server does not strip the space
    before punctuation.
    """
    anchor = tok("a", add_special_tokens=False)["input_ids"][-1:]
    base = tok.decode(anchor, clean_up_tokenization_spaces=False) if anchor else ""
    cache = {}

    def spell(i):
        if i not in cache:
            s = tok.decode(anchor + [i], clean_up_tokenization_spaces=False)
            cache[i] = (s[len(base):] if anchor and s.startswith(base)
                        else tok.decode([i], clean_up_tokenization_spaces=False))
        return cache[i]
    return spell


def kld(gold, other):
    """KL(gold || other) over gold's top tokens, with `other`'s mass renormalised
    over the same support; tokens missing from `other` get its floor."""
    floor = min(other.values()) - 2.0 if other else -20.0
    keys = list(gold.keys())
    g = [gold[k] for k in keys]
    o = [other.get(k, floor) for k in keys]
    gz = math.log(sum(math.exp(x) for x in g))
    oz = math.log(sum(math.exp(x) for x in o))
    return sum(math.exp(x - gz) * ((x - gz) - (y - oz)) for x, y in zip(g, o))


def compare(gold_rows, top, unlistable=frozenset()):
    """KL(gold || endpoint) over gold's top tokens, matched by id when the
    endpoint gives ids and by spelling otherwise. `gold_rows` is
    [(id, spelling, logprob)]. Matched by spelling, a gold token in
    `unlistable` (the reference's special tokens) that the endpoint did not
    list is dropped, not floored, and its share of the gold top-N mass is
    returned; any other token the endpoint did not list gets the floor, as
    kld() gives it. Returns (kld, dropped_gold_mass, match)."""
    match = "id" if top.by_id is not None else "spelling"
    gz = math.log(sum(math.exp(lp) for _, _, lp in gold_rows))
    kept, dropped = {}, 0.0
    other = {}
    for i, sp, lp in gold_rows:
        v = top.by_id.get(i) if top.by_id is not None else top.by_spell.get(sp)
        if v is None and match == "spelling" and i in unlistable:
            dropped += math.exp(lp - gz)
            continue
        kept[i] = lp
        if v is not None:
            other[i] = v
    listed = list(top.by_id.values()) if top.by_id is not None else list(top.by_spell.values())
    floor = min(listed) - 2.0 if listed else -20.0
    k = kld(kept, {i: other.get(i, floor) for i in kept}) if kept else 0.0
    return k, dropped, match


def id_spelling_mismatches(top, spell):
    """Ids whose spelling at the endpoint differs from the reference's: the
    sign of an endpoint whose vocabulary is numbered differently, where id
    matching would pair the wrong tokens. A token the endpoint spells "" (a
    control token) or with U+FFFD (a partial character) says nothing."""
    if top.spelled is None:
        return 0
    return sum(1 for i, t in top.spelled.items()
               if t and "\ufffd" not in t and t != spell(i))


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on discordant pairs.

    b = positions the first side got right and the second wrong, c the
    reverse. Concordant positions carry no information. Reported because a
    percentage-point gap at 100 positions is one or two tokens: measured
    2026-09-07, every margin-qualified difference in the golden pass except
    stablelm's had p >= 0.25, our own wins included.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail / 2.0 ** n)


def wilcoxon_signed_rank(diffs):
    """Two-sided normal-approximation Wilcoxon on paired per-position KL.

    `diffs` are (side b) - (side a), so a positive statistic means side a is
    closer to the reference. This uses every position rather than only the
    ones where the argmax flips, which measured 2026-09-07 gives it roughly
    ten times the power of the top-1 comparison at the same corpus size.
    Returns (z, p, n_nonzero) or None when there is too little to test.
    """
    d = [x for x in diffs if x != 0.0]
    n = len(d)
    if n < 5:
        return None
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    w = sum(ranks[i] for i in range(n) if d[i] > 0)
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd == 0:
        return None
    z = (w - mu) / sd
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return z, p, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="HF model directory")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--endpoint-a")
    ap.add_argument("--model-name-a")
    ap.add_argument("--endpoint-b")
    ap.add_argument("--model-name-b")
    ap.add_argument("--score-a", help="side a from a runner --score --score-top file")
    ap.add_argument("--score-b", help="side b from a runner --score --score-top file")
    ap.add_argument("--fresh-prompt", action="store_true",
                    help="send cache_prompt:false, so every position is a full prefill")
    ap.add_argument("--max-positions", type=int, default=100)
    ap.add_argument("--min-prefix", type=int, default=8)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="the publisher ships its own modeling code")
    ap.add_argument("--force-bos", action="store_true",
                    help="prepend the tokenizer's BOS to the reference even "
                         "when the HF tokenizer does not add one. Needed "
                         "where the SERVED artifact declares add_bos_token "
                         "but the publisher's loaded tokenizer does not add "
                         "it (measured 2026-09-07 on Gemma 4 E2B): the "
                         "endpoints follow the artifact, so the reference "
                         "must see the same prefix or the two are compared "
                         "across a one-token offset.")
    args = ap.parse_args()
    if bool(args.endpoint_a) == bool(args.score_a):
        ap.error("side a needs exactly one of --endpoint-a/--model-name-a or --score-a")
    if args.endpoint_a and not args.model_name_a:
        ap.error("--endpoint-a needs --model-name-a")
    if args.endpoint_b and args.score_b:
        ap.error("side b takes --endpoint-b or --score-b, not both")
    if args.endpoint_b and not args.model_name_b:
        ap.error("--endpoint-b needs --model-name-b")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=args.trust_remote_code)
    # A publisher whose top-level class is multimodal (Qwen3.5, Gemma 4,
    # Muse Glimmer) still answers text through the same language model;
    # the causal-LM auto class refuses those configs, so fall back to the
    # image-text class and drive it with text only. The config name is
    # printed so the report says which class produced the gold.
    kw = dict(dtype=torch.float32, trust_remote_code=args.trust_remote_code)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.hf, **kw)
    except (ValueError, KeyError, OSError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.hf, **kw)
    print("reference class:", type(model).__name__, file=sys.stderr)
    model.eval()

    text = open(args.corpus, encoding="utf-8").read()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    body = ids[: args.min_prefix + args.max_positions * args.stride + 1]
    # The serving endpoints tokenize the prefix TEXT, so a family whose
    # tokenizer prepends BOS gives the model one more token than a reference
    # scored on the bare ids. Measured 2026-09-07: without this the gemma-3,
    # gemma-4 and Phi-3.5 rows read mean KL around 2 to 5 with both engines
    # agreeing with each other and disagreeing with the reference, which is
    # the signature of a broken instrument rather than a broken engine
    # (qwen3 and granite, whose tokenizers add nothing, read 0.0000). The
    # special prefix is taken from the tokenizer itself and prepended once;
    # position `pos` of the body is then row len(special) + pos - 1.
    # Taken by DIFFERENCE on a real word, not from the empty string: some
    # tokenizers add nothing to "" yet prepend BOS to any real input, and
    # measured 2026-09-07 that is exactly what Phi-3.5 and Gemma 4 do, which
    # left them reading mean KL around 2 and 5 after the first version of
    # this fix had already repaired Gemma 3.
    _probe = "word"
    _with = tok(_probe, add_special_tokens=True)["input_ids"]
    _without = tok(_probe, add_special_tokens=False)["input_ids"]
    n_special = len(_with) - len(_without)
    special = _with[:n_special] if n_special > 0 else []
    if args.force_bos and not special and tok.bos_token_id is not None:
        special = [tok.bos_token_id]
    span = list(special) + list(body)
    scored = [len(special) + args.min_prefix + n * args.stride - 1
              for n in range(args.max_positions)]
    logp, logp_row, logits_path = reference_logp(model, span, scored)
    print("reference logits:", logits_path, file=sys.stderr)

    spell = token_speller(tok)
    # the reference's special tokens: what a server spells "" or not at all
    unlistable = set(getattr(tok, "all_special_ids", []) or [])
    for i, t in (getattr(tok, "added_tokens_decoder", None) or {}).items():
        if getattr(t, "special", False):
            unlistable.add(int(i))
    rows = []
    skipped = 0
    files = {}
    for name, path in (("a", args.score_a), ("b", args.score_b)):
        if path:
            files[name] = load_score(path, span, args.top_n)
    sides = [("a", args.endpoint_a or args.score_a, args.model_name_a or "score file")]
    if args.endpoint_b or args.score_b:
        sides.append(("b", args.endpoint_b or args.score_b, args.model_name_b or "score file"))
    for n in range(args.max_positions):
        pos = args.min_prefix + n * args.stride
        prefix = tok.decode(body[:pos])
        # the decoded prefix must re-tokenize to the same ids, or the sides
        # would be scored on different contexts
        if tok(prefix, add_special_tokens=False)["input_ids"] != body[:pos]:
            continue
        top = torch.topk(logp[logp_row[len(special) + pos - 1]], args.top_n)
        gold_rows = [(int(i), spell(int(i)), float(v))
                     for v, i in zip(top.values, top.indices)]
        gold_top1_id = gold_rows[0][0]
        gold_top1 = gold_rows[0][1]
        row = {"pos": pos, "gold_top1": gold_top1, "gold_margin":
               float(top.values[0] - top.values[1])}
        got = {name: (score_side(files[name], len(special) + pos - 1, spell)
                      if name in files else
                      query(ep, mn, prefix, args.top_n, args.fresh_prompt))
               for name, ep, mn in sides}
        if any(v is None for v in got.values()):
            skipped += 1
            continue
        for name, ep, mn in sides:
            other = got[name]
            k, dropped, match = compare(gold_rows, other, unlistable)
            agree = (other.top1_id == gold_top1_id if match == "id"
                     else other.top1_spell == gold_top1)
            row[name] = {"kld": k, "top1": other.top1_spell, "agree": agree,
                         "match": match, "dropped_gold_mass": dropped,
                         "id_spelling_mismatches": id_spelling_mismatches(other, spell)}
        rows.append(row)
        print(f"pos {pos:4d} " + " ".join(
            f"{k}: kld {row[k]['kld']:.4f} {'ok' if row[k]['agree'] else 'DIFF'}"
            for k in row if k in ("a", "b")), file=sys.stderr)

    summary = {"positions": len(rows), "hf": args.hf, "corpus": args.corpus,
               "reference_class": type(model).__name__,
               "reference_logits": logits_path,
               "special_prefix_tokens": len(special),
               "special_prefix_forced": bool(args.force_bos and special),
               "token_spelling": "in-sentence (anchored decode)",
               "token_match": "by id where the endpoint gives ids, else by "
                              "spelling with unlisted special tokens dropped",
               "positions_skipped_no_logprobs": skipped,
               "fresh_prompt": bool(args.fresh_prompt),
               "reference_dtype": "float32", "reference_device": "cpu",
               "sides": {}}
    for name, ep, mn in sides:
        ks = [r[name]["kld"] for r in rows]
        summary["sides"][name] = {
            "endpoint": ep, "model": mn,
            "mean_kld_vs_gold": sum(ks) / max(1, len(ks)),
            "max_kld_vs_gold": max(ks, default=0.0),
            "top1_agreement_pct": 100.0 * sum(r[name]["agree"] for r in rows) / max(1, len(rows)),
            "margin_qualified_top1_pct": 100.0 * sum(
                r[name]["agree"] for r in rows if r["gold_margin"] > 0.5)
            / max(1, sum(1 for r in rows if r["gold_margin"] > 0.5)),
            "matched_by_id_positions": sum(r[name]["match"] == "id" for r in rows),
            "positions_with_dropped_mass": sum(r[name]["dropped_gold_mass"] > 0 for r in rows),
            "max_dropped_gold_mass": max((r[name]["dropped_gold_mass"] for r in rows), default=0.0),
            "id_spelling_mismatches": sum(r[name]["id_spelling_mismatches"] for r in rows),
        }
        if summary["sides"][name]["id_spelling_mismatches"]:
            print(f"warning: side {name}: token ids whose spelling differs from the "
                  "reference's; the endpoint may number its vocabulary differently, "
                  "and id matching would pair the wrong tokens", file=sys.stderr)
    # Whether any of the above is distinguishable from noise. Both sides
    # see the same positions, so the comparison is paired.
    if len(sides) == 2:
        qual = [r for r in rows if r["gold_margin"] > 0.5]
        a_only = sum(1 for r in qual if r["a"]["agree"] and not r["b"]["agree"])
        b_only = sum(1 for r in qual if r["b"]["agree"] and not r["a"]["agree"])
        w = wilcoxon_signed_rank([r["b"]["kld"] - r["a"]["kld"] for r in rows])
        summary["significance"] = {
            "qualified_positions": len(qual),
            "margin_qualified_hits_a": sum(1 for r in qual if r["a"]["agree"]),
            "margin_qualified_hits_b": sum(1 for r in qual if r["b"]["agree"]),
            "discordant_a_only": a_only,
            "discordant_b_only": b_only,
            "mcnemar_exact_p": mcnemar_exact(a_only, b_only),
            "kld_wilcoxon_z": None if w is None else w[0],
            "kld_wilcoxon_p": None if w is None else w[1],
            "kld_wilcoxon_n": None if w is None else w[2],
            "verdict": (
                "not distinguishable" if w is None or w[1] >= 0.05
                else ("a closer" if w[0] > 0 else "b closer")),
        }
    # v3: tokens are spelled as a server spells them (token_speller); a v2
    # report on an add_space_prefix family compared mismatched spellings.
    # v4: tokens are matched by id where the endpoint gives ids; a v3 report
    # scored a gold stop or control token at a floor (the server spells it
    # ""), and skipped the positions whose greedy pick was a stop.
    report = {"schema_version": "xyntetik.runner.gold-logits.v4",
              "summary": summary, "rows": rows}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
