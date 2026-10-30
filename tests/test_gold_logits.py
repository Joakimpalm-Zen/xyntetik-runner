"""The gold harness spells a token the way a server does (R6.7.6).

On a SentencePiece tokenizer that prepends a space (Mistral v0.3, Phi-3.5), a
lone decode strips the leading space of its first piece, so the reference said
"new" where both servers said " new". The harness matches tokens by spelling,
so those families read mean KL around 1.6 to 2.0 with the two engines agreeing
to four decimals. These fakes reproduce both decoder shapes without a model.
"""

import importlib.util
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("gold_logits", ROOT / "scripts" / "gold-logits.py")
gold = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gold)


class _Tok:
    """A vocabulary of pieces; `strip_first` is SentencePiece's decoder, which
    drops the dummy-prefix space at the start of every decode."""

    def __init__(self, pieces, strip_first):
        self.pieces = pieces
        self.ids = {p: i for i, p in enumerate(pieces)}
        self.strip_first = strip_first

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [self.ids["▁a" if self.strip_first else "a"]]}

    def decode(self, ids, clean_up_tokenization_spaces=True):
        s = "".join(self.pieces[i] for i in ids).replace("▁", " ")
        if self.strip_first and s.startswith(" "):
            s = s[1:]
        if clean_up_tokenization_spaces:
            s = s.replace(" ,", ",")
        return s


SPM = _Tok(["▁a", "▁new", ",", "▁", "▁model"], strip_first=True)
BPE = _Tok(["a", " new", ",", " ", " model"], strip_first=False)


def test_the_reference_projects_only_the_scored_rows():
    """reference_logp asks the model for the scored rows' logits only, and
    falls back to the full tensor for a forward that refuses the argument or
    ignores it, with the same values at every scored row either way."""
    torch = __import__("pytest").importorskip("torch")
    vocab, n = 7, 12
    full = torch.randn(1, n, vocab)

    class Out:
        def __init__(self, logits): self.logits = logits

    class Keeps:
        def __call__(self, ids, logits_to_keep=None):
            assert ids.shape == (1, n)
            return Out(full if logits_to_keep is None else full[:, logits_to_keep, :])

    class Refuses:
        def __call__(self, ids):
            return Out(full)

    class Ignores:
        def __call__(self, ids, logits_to_keep=None):
            return Out(full)

    rows = [9, 2, 5, 2]
    want = torch.log_softmax(full[0], dim=-1)
    for model, path in ((Keeps(), "scored rows only"), (Refuses(), "full tensor"),
                        (Ignores(), "full tensor")):
        logp, index, got_path = gold.reference_logp(model, list(range(n)), rows)
        assert got_path == path
        if path == "scored rows only":
            assert logp.shape == (3, vocab)
        for r in rows:
            assert torch.allclose(logp[index[r]], want[r])


def test_a_space_prefixed_token_keeps_its_space():
    spell = gold.token_speller(SPM)
    assert spell(1) == " new"
    assert spell(4) == " model"
    assert spell(3) == " "
    assert spell(2) == ","
    # the lone decode is what went wrong
    assert SPM.decode([1]) == "new"


def test_byte_level_spelling_is_unchanged():
    spell = gold.token_speller(BPE)
    for i in range(len(BPE.pieces)):
        assert spell(i) == BPE.decode([i], clean_up_tokenization_spaces=False)


def test_identical_distributions_read_zero_once_spelled_alike():
    lp = {1: math.log(0.6), 4: math.log(0.3), 2: math.log(0.1)}
    server = {" new": math.log(0.6), " model": math.log(0.3), ",": math.log(0.1)}
    spell = gold.token_speller(SPM)
    assert abs(gold.kld({spell(i): v for i, v in lp.items()}, server)) < 1e-12
    # spelled by lone decode, the same two distributions read far apart
    assert gold.kld({SPM.decode([i]): v for i, v in lp.items()}, server) > 0.5


# The lab's position (2026-10-17, qwen35 BF16, position 122): the float32 gold
# top tokens, with the stop token <|endoftext|> at -3.5751 (2.8% of the mass).
# A server spells a stop token "" (it decodes to no bytes) and gives its id in
# top_token_ids; the reference spells it <|endoftext|>.
STOP = 151643
GOLD = [(16, "1", -1.2223), (17, "2", -1.9458), (20, "5", -1.9494),
        (18, "3", -2.4348), (19, "4", -3.1237), (311, " to", -3.2378),
        (STOP, "<|endoftext|>", -3.5751), (24, "9", -3.8922)]
SPELL = {i: sp for i, sp, _ in GOLD}


def _runner_json(entries, chosen):
    """A runner /v1/completions body: top_logprobs as a JSON map written by
    hand, so repeated "" keys stay repeated as they do on the wire."""
    top = "{" + ",".join(f"{json.dumps(sp)}:{lp!r}" for _, sp, lp in entries) + "}"
    ids = json.dumps([i for i, _, _ in entries])
    ci, csp, clp = chosen
    return ('{"choices":[{"text":"","logprobs":{"tokens":[%s],"token_ids":[%d],'
            '"token_logprobs":[%r],"top_logprobs":[%s],"top_token_ids":[%s]}}]}'
            % (json.dumps(csp), ci, clp, top, ids))


def _parse(body):
    return gold.parse_top(json.loads(body, object_pairs_hook=gold._Obj))


def _served(rows):
    # the server lists the same tokens, the stop spelled "", agreeing to ~1e-6
    return [(i, "" if i == STOP else sp, lp + (1e-6 if k == 3 else 0.0))
            for k, (i, sp, lp) in enumerate(rows)]


def test_a_stop_token_spelled_empty_matches_by_id():
    top = _parse(_runner_json(_served(GOLD), (16, "1", -1.2223)))
    k, dropped, match = gold.compare(GOLD, top)
    assert match == "id" and dropped == 0.0
    assert k < 1e-9, k
    # matched by spelling, the stop's gold mass met the floor: the defect
    # (0.049 on these eight tokens; the lab's full top-20 read 0.112)
    floored = gold.kld({sp: lp for _, sp, lp in GOLD}, top.by_spell)
    assert floored > 0.02, floored


def test_two_control_tokens_both_survive_the_json_map():
    rows = GOLD[:6] + [(STOP, "<|endoftext|>", -3.5751), (151645, "<|im_end|>", -4.0)]
    served = [(i, "" if i in (STOP, 151645) else sp, lp) for i, sp, lp in rows]
    top = _parse(_runner_json(served, (16, "1", -1.2223)))
    assert top.by_id[STOP] == -3.5751 and top.by_id[151645] == -4.0
    assert len(top.by_spell) < len(top.by_id)   # the plain map lost one


def test_without_ids_an_unlisted_special_token_is_dropped_and_reported():
    """An endpoint that gives no ids and does not list the stop: its gold
    mass leaves the comparison and is reported, instead of a floor."""
    listed = [r for r in GOLD if r[0] != STOP]
    body = {"choices": [{"logprobs": {"content": [{
        "token": "1", "logprob": -1.2223,
        "top_logprobs": [{"token": sp, "logprob": lp} for _, sp, lp in listed]}]}}]}
    top = gold.parse_top(body)
    k, dropped, match = gold.compare(GOLD, top, unlistable={STOP})
    assert match == "spelling" and k < 1e-12
    share = math.exp(-3.5751) / sum(math.exp(lp) for _, _, lp in GOLD)
    assert abs(dropped - share) < 1e-12
    k_floor, _, _ = gold.compare(GOLD, top)   # not known to be unlistable
    assert k_floor > 0.02


def test_llama_server_ids_are_used():
    body = {"choices": [{"logprobs": {"content": [{
        "id": 16, "token": "1", "logprob": -1.2223,
        "top_logprobs": [{"id": i, "token": "" if i == STOP else sp, "logprob": lp}
                         for i, sp, lp in GOLD]}]}}]}
    k, _, match = gold.compare(GOLD, gold.parse_top(body))
    assert match == "id" and k < 1e-12


def test_llama_server_top1_is_its_argmax_not_its_pick():
    """llama-server's emitted token comes after the model file's penalties,
    so it can sit below the top of the distribution the server reports; the
    top-1 compared against the reference is that top on both engines."""
    alts = [(16, "1", -1.212), (17, "2", -1.69), (18, "3", -1.919)]
    body = {"choices": [{"logprobs": {"content": [{
        "id": 18, "token": "3", "logprob": -1.919,
        "top_logprobs": [{"id": i, "token": sp, "logprob": lp} for i, sp, lp in alts]}]}}]}
    top = gold.parse_top(body)
    assert (top.top1_spell, top.top1_id) == ("1", 16)
    runner = _parse(_runner_json(alts, (18, "3", -1.919)))
    assert (runner.top1_spell, runner.top1_id) == ("1", 16)


def test_a_stop_as_the_greedy_pick_is_read_from_stop_logprobs():
    """The runner's logprob arrays align with emitted text, so a greedy stop
    leaves them empty and rides in stop_logprobs; the position used to be
    skipped, dropping a one-sided stop from the score."""
    rows = [(STOP, "<|endoftext|>", -0.2)] + GOLD[:4]
    alts = "{" + ",".join(f"{json.dumps('' if i == STOP else sp)}:{lp!r}"
                          for i, sp, lp in rows) + "}"
    body = ('{"choices":[{"text":"","finish_reason":"stop","stop_token_id":%d,'
            '"stop_logprobs":{"logprob":-0.2,"top_logprobs":%s,"top_token_ids":%s},'
            '"logprobs":{"tokens":[],"token_ids":[],"token_logprobs":[],'
            '"top_logprobs":[],"top_token_ids":[]}}]}'
            % (STOP, alts, json.dumps([i for i, _, _ in rows])))
    top = _parse(body)
    assert top is not None and top.top1_id == STOP
    k, _, match = gold.compare(rows, top)
    assert match == "id" and k < 1e-12


def test_an_id_spelled_differently_is_counted():
    top = _parse(_runner_json(_served(GOLD), (16, "1", -1.2223)))
    assert gold.id_spelling_mismatches(top, SPELL.get) == 0
    top.spelled[16] = "one"            # numbered differently at the endpoint
    top.spelled[20] = "�"         # a partial character says nothing
    assert gold.id_spelling_mismatches(top, SPELL.get) == 1
