import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_capture_cli_accepts_an_immutable_reference_revision():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "difftok.py"), "--help"],
        text=True, capture_output=True, check=True)
    assert "--ref-revision" in proc.stdout


import hashlib  # noqa: E402
import json  # noqa: E402

import pytest  # noqa: E402

REFS = ROOT / "tests" / "compatibility" / "tokenizer-references"
CORPUS = ROOT / "tests" / "fixtures" / "tokenizer-corpus.txt"


def _corpus():
    return [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l]


@pytest.mark.parametrize("name", ["mistral-7b-instruct-v0.3", "phi-3.5-mini-instruct",
                                  "salamandra-7b-instruct"])
def test_sentencepiece_references_are_what_they_say(name):
    """R6.2.2 (owner 2026-10-03): these three rows gate on the publisher's
    SentencePiece model; the tokenizer.json capture is kept beside it as the
    informational second reference. The marker rows are exactly the strings
    that spell one of the capture's own listed markers."""
    cap = json.loads((REFS / f"{name}.json").read_text())
    assert cap["source"] == "sentencepiece"
    assert len(cap["model_sha256"]) == 64 and len(cap["ref_revision"]) == 40
    corpus = _corpus()
    h = hashlib.sha256()
    for s in corpus:
        h.update(s.encode("utf-8"))
        h.update(b"\0")
    assert cap["corpus_sha256"] == h.hexdigest()
    assert len(cap["ids"]) == len(corpus)
    want = [i for i, s in enumerate(corpus) if any(m in s for m in cap["special_markers"])]
    assert cap["special_marker_rows"] == want
    assert "<s>" in cap["special_markers"]
    info = json.loads((REFS / f"{name}.tokenizer-json.json").read_text())
    assert info["corpus_sha256"] == cap["corpus_sha256"] and "source" not in info


import os  # noqa: E402

EXE = ROOT / ("difftok.exe" if os.name == "nt" else "difftok")


def _built():
    """The harness binary, built on demand the way difftok.py builds it."""
    if not EXE.exists():
        r = subprocess.run([os.environ.get("MAKE", "make"), "-C", str(ROOT), EXE.name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0 or not EXE.exists():
            pytest.skip("difftok could not be built here")
    if not (ROOT / "test.gguf").exists():
        pytest.skip("test.gguf not built")
    return EXE


def _difftok(capture, tmp_path, *extra):
    gguf = ROOT / "test.gguf"
    _built()
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "difftok.py"),
                           "--gguf", str(gguf), "--ref-ids", str(capture),
                           "--no-build", "--show", "0", *extra],
                          text=True, capture_output=True)


def test_marker_rows_are_reported_and_not_gated(tmp_path):
    out = subprocess.run([str(_built()), str(ROOT / "test.gguf"), str(CORPUS)],
                         text=True, capture_output=True)
    if out.returncode != 0:
        pytest.skip("difftok cannot read test.gguf")
    ids = [[int(x) for x in l.split()] for l in out.stdout.splitlines()]
    corpus = _corpus()
    h = hashlib.sha256()
    for s in corpus:
        h.update(s.encode("utf-8"))
        h.update(b"\0")
    def write(rows, changed):
        theirs = [list(r) for r in ids]
        for i in changed:
            theirs[i] = theirs[i] + [0]
        p = tmp_path / "cap.json"
        p.write_text(json.dumps({"corpus_sha256": h.hexdigest(), "ids": theirs,
                                 "special_marker_rows": rows}))
        return p
    ok = _difftok(write([3], [3]), tmp_path)
    assert ok.returncode == 0 and "1 more differ on a spelled special marker" in ok.stdout, ok.stdout
    bad = _difftok(write([3], [4]), tmp_path)
    assert bad.returncode == 1 and "1/" in bad.stdout, bad.stdout
    info = _difftok(write([], [4, 5]), tmp_path, "--report-only")
    assert info.returncode == 0 and "2/" in info.stdout, info.stdout


def test_the_vocabulary_corpus_is_every_whole_model_entry():
    """--vocab's strings: each model-vocabulary entry decoded to text, leaving
    out added tokens and a lone byte of a longer character; U+0000 is
    compared, the harness reading each string by its length."""
    tokenizers = pytest.importorskip("tokenizers")
    import importlib.util
    spec = importlib.util.spec_from_file_location("difftok", ROOT / "scripts/difftok.py")
    difftok = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(difftok)
    # byte-level alphabet: U+00C3 is the lone byte 0xC3, U+0100 is byte 0x00
    vocab = {"h": 0, "o": 1, "ho": 2, "\u00c3": 3, "\u0100": 4, "\u0120": 5, "\u0120ho": 6,
             "<|x|>": 7}
    tok = tokenizers.Tokenizer(tokenizers.models.BPE(vocab, [("h", "o"), ("\u0120", "ho")]))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = tokenizers.decoders.ByteLevel()
    tok.add_special_tokens(["<|x|>"])
    strings, skipped = difftok.vocab_corpus(tok)
    assert strings == ["h", "o", "ho", "\0", " ", " ho"]
    assert skipped == {"added": 1, "partial_utf8": 1, "empty": 0}


def test_vocab_corpus_keeps_genuine_ufffd_and_skips_partial_utf8():
    """R4.50.13: a genuine U+FFFD BPE token (0xEF 0xBF 0xBD = 'ï¿½') round-trips
    back to [id] and must NOT be skipped as partial_utf8, while lone lead bytes
    like 0xEF still skip as partial_utf8."""
    tokenizers = pytest.importorskip("tokenizers")
    import importlib.util
    spec = importlib.util.spec_from_file_location("difftok", ROOT / "scripts/difftok.py")
    difftok = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(difftok)
    vocab = {"\u00ef": 0, "\u00bf": 1, "\u00bd": 2, "\u00ef\u00bf": 3, "\u00ef\u00bf\u00bd": 4}
    merges = [("\u00ef", "\u00bf"), ("\u00ef\u00bf", "\u00bd")]
    tok = tokenizers.Tokenizer(tokenizers.models.BPE(vocab, merges))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = tokenizers.decoders.ByteLevel()
    strings, skipped = difftok.vocab_corpus(tok)
    assert strings == ["\ufffd"]
    assert skipped == {"added": 0, "partial_utf8": 4, "empty": 0}


def test_check_vocab_types_classes_and_defects(tmp_path):
    """R4.50.13 Gate (A): 3-class publisher taxonomy catches unsloth OLMo-3 id 5809
    and LiquidAI LFM2.5-350M 'Mathias'/'python', reports <think>/</think> typed
    CONTROL, and passes a clean USER_DEFINED file."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import gguf_fixture as gf
    import importlib.util
    spec = importlib.util.spec_from_file_location("cvt", ROOT / "scripts/check-vocab-types.py")
    cvt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cvt)

    tokens_ref = ["<|endoftext|>", "<think>", "</think>", "p", "y", "t", "h", "o", "n",
                  "py", "pyt", "pyth", "pytho", "python", "\u00ef", "\u00bf", "\u00bd",
                  "\u00ef\u00bf", "\u00ef\u00bf\u00bd"]
    merges = ["p y", "py t", "pyt h", "pyth o", "pytho n", "\u00ef \u00bf", "\u00ef\u00bf \u00bd"]
    tj = {
        "model": {"type": "BPE", "vocab": {t: i for i, t in enumerate(tokens_ref)},
                  "merges": [m.split(" ") for m in merges]},
        "added_tokens": [
            {"id": 0, "content": "<|endoftext|>", "special": True, "normalized": False},
            {"id": 1, "content": "<think>", "special": False, "normalized": False},
            {"id": 2, "content": "</think>", "special": False, "normalized": False},
            {"id": 13, "content": "python", "special": False, "normalized": True},
        ],
    }

    def make_gguf(name, toks, ttype):
        p = tmp_path / name
        kvs = [
            gf.ks("general.architecture", "llama"),
            gf.ks("tokenizer.ggml.model", "gpt2"),
            gf.ks("tokenizer.ggml.pre", "lfm2"),
            gf.kas("tokenizer.ggml.tokens", toks),
            gf.kas("tokenizer.ggml.merges", merges),
            gf.kai("tokenizer.ggml.token_type", ttype),
            gf.ku("tokenizer.ggml.bos_token_id", 0),
            gf.ku("tokenizer.ggml.eos_token_id", 0),
        ]
        gf.write_gguf(str(p), [], kvs, announce=False)
        return p

    # 1. Clean (1.2B pattern): special=3, added_nonspecial=4, plain=1 -> 0 failures, 0 reports
    tt_clean = [1] * len(tokens_ref)
    tt_clean[0] = 3; tt_clean[1] = 4; tt_clean[2] = 4; tt_clean[13] = 4
    res_ok = cvt.check_vocab_types(cvt.read_local_gguf_kv(make_gguf("clean.gguf", tokens_ref, tt_clean)), tj)
    assert res_ok["failures"] == [] and res_ok["reports"] == []

    # 2. LFM2.5-350M pattern: python (13) and <think>/<\/think> (1, 2) typed CONTROL -> 1 failure, 2 reports
    tt_350m = list(tt_clean)
    tt_350m[1] = 3; tt_350m[2] = 3; tt_350m[13] = 3
    res_350m = cvt.check_vocab_types(cvt.read_local_gguf_kv(make_gguf("lfm350m.gguf", tokens_ref, tt_350m)), tj)
    assert [f["id"] for f in res_350m["failures"]] == [13]
    assert [r["id"] for r in res_350m["reports"]] == [1, 2]
    res_strict = cvt.check_vocab_types(cvt.read_local_gguf_kv(tmp_path / "lfm350m.gguf"), tj, strict_added=True)
    assert [f["id"] for f in res_strict["failures"]] == [1, 2, 13]

    # 3. Unsloth OLMo-3 id 5809 pattern: plain vocab id 18 rewritten to U+FFFD + CONTROL -> failure
    toks_5809 = list(tokens_ref); toks_5809[18] = "\ufffd"
    tt_5809 = list(tt_clean); tt_5809[18] = 3
    res_5809 = cvt.check_vocab_types(cvt.read_local_gguf_kv(make_gguf("olmo5809.gguf", toks_5809, tt_5809)), tj)
    assert [f["id"] for f in res_5809["failures"]] == [18] and res_5809["failures"][0]["class"] == "plain_vocab"


def test_difftok_no_special_and_decode_all_and_opt_out(tmp_path):
    """R4.50.13: difftok --decode-all and --no-special reflect B1 repair and
    RUNNER_TOK_REPAIR=0 opt-out."""
    exe = _built()
    sys.path.insert(0, str(ROOT / "scripts"))
    import gguf_fixture as gf

    toks = ["<|endoftext|>", "p", "y", "t", "h", "o", "n", "py", "pyt", "pyth", "pytho", "python"]
    merges = ["p y", "py t", "pyt h", "pyth o", "pytho n"]
    ttype = [3] + [1] * 10 + [3]  # python (id 11) mistyped CONTROL
    p = tmp_path / "m.gguf"
    kvs = [
        gf.ks("general.architecture", "llama"),
        gf.ks("tokenizer.ggml.model", "gpt2"),
        gf.ks("tokenizer.ggml.pre", "lfm2"),
        gf.kas("tokenizer.ggml.tokens", toks),
        gf.kas("tokenizer.ggml.merges", merges),
        gf.kai("tokenizer.ggml.token_type", ttype),
        gf.ku("tokenizer.ggml.bos_token_id", 0),
        gf.ku("tokenizer.ggml.eos_token_id", 0),
    ]
    gf.write_gguf(str(p), [], kvs, announce=False)

    env_on = dict(os.environ); env_on.pop("RUNNER_TOK_REPAIR", None)
    env_off = dict(os.environ, RUNNER_TOK_REPAIR="0")

    dec_on = subprocess.run([str(exe), "--decode-all", str(p)], text=True, capture_output=True, check=True, env=env_on).stdout.splitlines()
    dec_off = subprocess.run([str(exe), "--decode-all", str(p)], text=True, capture_output=True, check=True, env=env_off).stdout.splitlines()
    assert dec_on[0] == "" and bytes.fromhex(dec_on[11]) == b"python"
    assert dec_off[0] == "" and dec_off[11] == ""

    cfile = tmp_path / "c.jsonl"
    cfile.write_text('"python"\n', encoding="utf-8")
    ns = subprocess.run([str(exe), "--no-special", str(p), str(cfile)], text=True, capture_output=True, check=True, env=env_on).stdout.strip()
    assert ns == "11"


def _runner_bin():
    for cand in [ROOT / "runner", ROOT / "runner.exe"]:
        if cand.exists():
            return cand
    bdir = ROOT / ".build"
    if bdir.exists():
        for name in ("runner", "runner.exe"):
            hits = list(bdir.rglob(name))
            if hits:
                return hits[0]
    return None


def test_tok_repair_receipt_and_verify(tmp_path):
    """R4.50.13: profile.route.tok_repair records demoted ids; --verify pins the
    replay to the record so pre-fix (1.22.0) receipts verify without repair and
    repaired receipts verify even when RUNNER_TOK_REPAIR=0 is set."""
    runner = _runner_bin()
    if runner is None:
        pytest.skip("runner binary not built")
    sys.path.insert(0, str(ROOT / "scripts"))
    import gguf_fixture as gf

    # Minimal 268-token GPT-2-BPE 1-layer llama fixture generating '[' -> 'python' -> ']' -> EOS
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    b2u = dict(zip(bs, [chr(c) for c in cs]))
    toks = [b2u[b] for b in range(256)] + ["py", "pyt", "pyth", "pytho", "python", "<|endoftext|>"]
    merges = ["p y", "py t", "pyt h", "pyth o", "pytho n"]
    id_py, id_eos = 260, 261
    ttype = [1] * len(toks)
    ttype[id_py] = 3    # mistyped merge-output CONTROL
    ttype[id_eos] = 3   # genuine EOS CONTROL

    n_vocab, n_embd, n_head, n_kv, n_ff = len(toks), 32, 2, 2, 64
    embd = [0.0] * (n_vocab * n_embd)
    outw = [0.0] * (n_vocab * n_embd)
    for prev_id, next_id, dim in [(ord("A"), ord("["), 0), (ord("["), id_py, 1), (id_py, ord("]"), 2), (ord("]"), id_eos, 3)]:
        embd[prev_id * n_embd + dim] = 1.0
        outw[next_id * n_embd + dim] = 10.0

    gguf = tmp_path / "fixture.gguf"
    kvs = [
        gf.ks("general.architecture", "llama"),
        gf.ks("general.name", "r45013-receipt-fixture"),
        gf.ku("llama.block_count", 1),
        gf.ku("llama.context_length", 64),
        gf.ku("llama.embedding_length", n_embd),
        gf.ku("llama.feed_forward_length", n_ff),
        gf.ku("llama.attention.head_count", n_head),
        gf.ku("llama.attention.head_count_kv", n_kv),
        gf.kf("llama.attention.layer_norm_rms_epsilon", 1e-5),
        gf.kf("llama.rope.freq_base", 10000.0),
        gf.ks("tokenizer.ggml.model", "gpt2"),
        gf.ks("tokenizer.ggml.pre", "lfm2"),
        gf.kas("tokenizer.ggml.tokens", toks),
        gf.kas("tokenizer.ggml.merges", merges),
        gf.kai("tokenizer.ggml.token_type", ttype),
        gf.ku("tokenizer.ggml.bos_token_id", id_eos),
        gf.ku("tokenizer.ggml.eos_token_id", id_eos),
        gf.kb("tokenizer.ggml.add_bos_token", False),
    ]
    z_a, z_f, ones = [0.0] * (n_embd * n_embd), [0.0] * (n_embd * n_ff), [1.0] * n_embd
    tensors = [
        ("token_embd.weight", [n_embd, n_vocab], gf.pack(embd)),
        ("blk.0.attn_norm.weight", [n_embd], gf.pack(ones)),
        ("blk.0.attn_q.weight", [n_embd, n_embd], gf.pack(z_a)),
        ("blk.0.attn_k.weight", [n_embd, n_embd], gf.pack(z_a)),
        ("blk.0.attn_v.weight", [n_embd, n_embd], gf.pack(z_a)),
        ("blk.0.attn_output.weight", [n_embd, n_embd], gf.pack(z_a)),
        ("blk.0.ffn_norm.weight", [n_embd], gf.pack(ones)),
        ("blk.0.ffn_gate.weight", [n_embd, n_ff], gf.pack(z_f)),
        ("blk.0.ffn_up.weight", [n_embd, n_ff], gf.pack(z_f)),
        ("blk.0.ffn_down.weight", [n_ff, n_embd], gf.pack(z_f)),
        ("output_norm.weight", [n_embd], gf.pack(ones)),
        ("output.weight", [n_embd, n_vocab], gf.pack(outw)),
    ]
    gf.write_gguf(str(gguf), tensors, kvs, announce=False)

    env_on = dict(os.environ); env_on.pop("RUNNER_TOK_REPAIR", None)
    env_off = dict(os.environ, RUNNER_TOK_REPAIR="0")
    rec_old = tmp_path / "old.json"
    rec_new = tmp_path / "new.json"

    p_old = subprocess.run([str(runner), "-m", str(gguf), "--gpu", "off", "--temp", "0", "-s", "1",
                            "-n", "8", "-p", "A", "--record", str(rec_old)],
                           text=True, capture_output=True, check=True, env=env_off)
    p_new = subprocess.run([str(runner), "-m", str(gguf), "--gpu", "off", "--temp", "0", "-s", "1",
                            "-n", "8", "-p", "A", "--record", str(rec_new)],
                           text=True, capture_output=True, check=True, env=env_on)
    assert '260 "python"' in p_new.stderr and "RUNNER_TOK_REPAIR=0" in p_new.stderr
    assert "tokenizer: demoted" not in p_old.stderr
    ro = json.loads(rec_old.read_text(encoding="utf-8"))
    rn = json.loads(rec_new.read_text(encoding="utf-8"))
    assert ro["output"]["text"] == "[]" and "tok_repair" not in ro["profile"]["route"]
    assert rn["output"]["text"] == "[python]" and rn["profile"]["route"]["tok_repair"] == [id_py]

    # Verify old receipt under default (repair ON) -> VERIFIED (replays without repair)
    v_old = subprocess.run([str(runner), "--verify", str(rec_old), "-m", str(gguf)],
                           text=True, capture_output=True, env=env_on)
    assert v_old.returncode == 0, v_old.stderr
    # Verify new receipt under default and under RUNNER_TOK_REPAIR=0 -> VERIFIED (record wins)
    for env in (env_on, env_off):
        v_new = subprocess.run([str(runner), "--verify", str(rec_new), "-m", str(gguf)],
                               text=True, capture_output=True, env=env)
        assert v_new.returncode == 0, v_new.stderr

    # Mismatched tok_repair (with valid recomputed chain hash) -> UNVERIFIABLE (rc=3)
    raw = rec_new.read_bytes()
    cut = raw.rfind(b',"chain":')
    assert cut > 0
    body = raw[:cut].replace(b'"tok_repair":[260]', b'"tok_repair":[999]')
    assert body != raw[:cut]
    chain = dict(json.loads(rec_new.read_text(encoding="utf-8"))["chain"])
    chain["hash"] = hashlib.sha256(body).hexdigest()
    rec_bad = tmp_path / "bad.json"
    rec_bad.write_bytes(body + b',"chain":' + json.dumps(chain, separators=(",", ":")).encode() + b"}\n")
    v_bad = subprocess.run([str(runner), "--verify", str(rec_bad), "-m", str(gguf)],
                           text=True, capture_output=True, env=env_on)
    assert v_bad.returncode == 3 and "tok_repair" in v_bad.stderr, v_bad.stderr
