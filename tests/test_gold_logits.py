"""The gold harness spells a token the way a server does (R6.7.6).

On a SentencePiece tokenizer that prepends a space (Mistral v0.3, Phi-3.5), a
lone decode strips the leading space of its first piece, so the reference said
"new" where both servers said " new". The harness matches tokens by spelling,
so those families read mean KL around 1.6 to 2.0 with the two engines agreeing
to four decimals. These fakes reproduce both decoder shapes without a model.
"""

import importlib.util
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
