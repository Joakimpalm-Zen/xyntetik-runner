#!/usr/bin/env python3
"""Headers-only GGUF tokenizer.ggml.token_type and vocabulary check against a
publisher's tokenizer.json (R4.50.13).

    scripts/check-vocab-types.py --gguf FILE.gguf --ref PATH_OR_HF_REPO [--ref-revision REV]
    scripts/check-vocab-types.py --hf-gguf REPO@REV/FILE.gguf --ref HF_REPO [--ref-revision REV]

Classifies every vocabulary id the way the publisher's tokenizer treats it:
  - special          (added_tokens with special=true): matched first, hidden on decode
                     -> GGUF CONTROL (3) (or UNKNOWN (2) for <unk>).
  - added non-special(added_tokens with special=false): matched first, RENDERED on decode
                     -> GGUF USER_DEFINED (4).
                     Fails if a normalized or BPE merge-output added non-special token is
                     typed CONTROL/UNUSED (hiding ordinary words like "python" / "Mathias")
                     or NORMAL (dropping match-first priority).
                     Reports structural non-normalized added non-special tokens typed CONTROL
                     (e.g. LFM2.5-350M <think>/</think>, FIM markers, <|extra_id_*>);
                     fails on those too when --strict-added is given.
  - plain vocab      (in model.vocab, not in added_tokens): BPE / SPM, rendered on decode
                     -> GGUF NORMAL (1) (or BYTE (6)/UNKNOWN (2) for SPM), with exact
                     BPE piece text equality against model.vocab[id].
                     Fails if typed CONTROL/USER_DEFINED/UNUSED or if BPE text differs
                     (catches unsloth Olmo-3-7B-Think id 5809 rewritten from 'ï¿½' to U+FFFD).

Exit status: 0 if no hard failures, 1 if any hard failure, 2 on usage/fetch error.
"""
import argparse
import json
import os
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request

GGUF_TYPE_FMT = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
TYPE_NAMES = {1: "NORMAL", 2: "UNKNOWN", 3: "CONTROL", 4: "USER_DEFINED", 5: "UNUSED", 6: "BYTE"}
WANT_KEYS = {
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.padding_token_id",
}


def parse_gguf_kv(buf, want_keys=WANT_KEYS):
    if len(buf) < 24:
        raise EOFError("short header")
    if buf[0:4] != b"GGUF":
        raise ValueError(f"bad GGUF magic {buf[0:4]!r}")
    _ver, _n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
    off = 24

    def read_str():
        nonlocal off
        if off + 8 > len(buf):
            raise EOFError("str len")
        (n,) = struct.unpack_from("<Q", buf, off)
        off += 8
        if off + n > len(buf):
            raise EOFError("str data")
        s = buf[off:off + n].decode("utf-8", "replace")
        off += n
        return s

    def skip_val(vt):
        nonlocal off
        if vt == 8:
            if off + 8 > len(buf):
                raise EOFError("str len")
            (n,) = struct.unpack_from("<Q", buf, off)
            off += 8
            if off + n > len(buf):
                raise EOFError("str data")
            off += n
        elif vt == 9:
            if off + 12 > len(buf):
                raise EOFError("arr hdr")
            et, cnt = struct.unpack_from("<IQ", buf, off)
            off += 12
            if et in GGUF_TYPE_FMT:
                sz = GGUF_TYPE_FMT[et][1] * cnt
                if off + sz > len(buf):
                    raise EOFError("arr data")
                off += sz
            else:
                for _ in range(cnt):
                    skip_val(et)
        else:
            sz = GGUF_TYPE_FMT[vt][1]
            if off + sz > len(buf):
                raise EOFError("scalar")
            off += sz

    def read_val(vt):
        nonlocal off
        if vt == 8:
            return read_str()
        if vt == 9:
            if off + 12 > len(buf):
                raise EOFError("arr hdr")
            et, cnt = struct.unpack_from("<IQ", buf, off)
            off += 12
            if et in GGUF_TYPE_FMT:
                fmt, sz = GGUF_TYPE_FMT[et]
                need = sz * cnt
                if off + need > len(buf):
                    raise EOFError("arr data")
                res = list(struct.unpack_from("<" + fmt[1] * cnt, buf, off))
                off += need
                return res
            if et == 8:
                return [read_str() for _ in range(cnt)]
            return [read_val(et) for _ in range(cnt)]
        fmt, sz = GGUF_TYPE_FMT[vt]
        if off + sz > len(buf):
            raise EOFError("scalar")
        (v,) = struct.unpack_from(fmt, buf, off)
        off += sz
        return v

    out = {}
    for _ in range(n_kv):
        k = read_str()
        if off + 4 > len(buf):
            raise EOFError("vtype")
        (vt,) = struct.unpack_from("<I", buf, off)
        off += 4
        if k in want_keys:
            out[k] = read_val(vt)
        else:
            skip_val(vt)
    return out


def _hf_token():
    for env in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(env):
            return os.environ[env]
    p = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read().strip()
    return None


def _fetch_url(url, byte_end=None, timeout=45):
    headers = {"User-Agent": "runner-check-vocab-types/1.0"}
    if byte_end is not None:
        headers["Range"] = f"bytes=0-{byte_end - 1}"
    tok = _hf_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def read_local_gguf_kv(path):
    with open(path, "rb") as f:
        for mb in (4, 16, 48, 128):
            f.seek(0)
            buf = f.read(mb * 1024 * 1024)
            try:
                return parse_gguf_kv(buf)
            except EOFError:
                if len(buf) < mb * 1024 * 1024:
                    raise
    raise RuntimeError(f"GGUF header exceeded 128 MB: {path}")


def read_hf_gguf_kv(spec):
    if "@" in spec:
        repo_part, rest = spec.split("@", 1)
        rev, fname = rest.split("/", 1)
    else:
        parts = spec.split("/", 2)
        if len(parts) < 3:
            raise ValueError(f"--hf-gguf expects REPO@REV/FILE or ORG/REPO/FILE, got {spec!r}")
        repo_part = f"{parts[0]}/{parts[1]}"
        rev = "main"
        fname = parts[2]
    qname = urllib.parse.quote(fname, safe="/")
    url = f"https://huggingface.co/{repo_part}/resolve/{rev}/{qname}"
    for mb in (4, 12, 24, 48):
        buf = _fetch_url(url, byte_end=mb * 1024 * 1024)
        try:
            return parse_gguf_kv(buf)
        except EOFError:
            continue
    raise RuntimeError(f"GGUF header exceeded 48 MB: {url}")


def load_publisher_tokenizer_json(ref, revision=None):
    if os.path.isdir(ref):
        return json.loads(open(os.path.join(ref, "tokenizer.json"), encoding="utf-8").read())
    if os.path.isfile(ref):
        return json.loads(open(ref, encoding="utf-8").read())
    rev = revision or "main"
    url = f"https://huggingface.co/{ref}/resolve/{rev}/tokenizer.json"
    return json.loads(_fetch_url(url).decode("utf-8"))


def check_vocab_types(kv, tj, strict_added=False):
    """Compare GGUF header metadata `kv` against publisher `tokenizer.json` dict `tj`.

    Returns a dict with:
      failures: list of hard failure dicts
      reports:  list of informational/warning dicts (e.g. structural added non-special
                typed CONTROL such as <think>/</think> or FIM markers, or special typed
                USER_DEFINED)
      summary:  counts per class
    """
    toks = kv.get("tokenizer.ggml.tokens") or []
    ttypes = kv.get("tokenizer.ggml.token_type")
    tok_model = kv.get("tokenizer.ggml.model", "llama")
    is_bpe = tok_model in ("gpt2", "gemma4")

    model_obj = tj.get("model") or {}
    raw_vocab = model_obj.get("vocab")
    if isinstance(raw_vocab, dict):
        id2vocab = {v: k for k, v in raw_vocab.items()}
    elif isinstance(raw_vocab, list):
        id2vocab = {i: item[0] for i, item in enumerate(raw_vocab)}
    else:
        id2vocab = {}

    merges = model_obj.get("merges") or []
    merges_out = {
        "".join(m.split(" ")) if isinstance(m, str) else "".join(m)
        for m in merges
    }
    added = {a["id"]: a for a in tj.get("added_tokens") or []}

    failures = []
    reports = []
    counts = {"plain_vocab": 0, "added_nonspecial": 0, "special": 0}

    if ttypes is None:
        reports.append({"kind": "no_token_type", "detail": "GGUF carries no tokenizer.ggml.token_type array"})
        return {"failures": failures, "reports": reports, "summary": counts}

    n = min(len(toks), len(ttypes))
    for i in range(n):
        tt = ttypes[i]
        tname = TYPE_NAMES.get(tt, str(tt))
        gstr = toks[i]
        if i in added:
            a = added[i]
            astr = a.get("content", "")
            is_spec = bool(a.get("special", False))
            is_norm = bool(a.get("normalized", False))
            if is_spec:
                counts["special"] += 1
                if tt == 1:
                    failures.append({
                        "class": "special", "id": i, "gguf_str": gstr, "pub_str": astr,
                        "gguf_type": tname,
                        "why": "publisher special=true token typed NORMAL in GGUF (not matched first / rendered)",
                    })
                elif tt not in (3, 2):
                    reports.append({
                        "class": "special", "id": i, "gguf_str": gstr, "pub_str": astr,
                        "gguf_type": tname,
                        "why": f"publisher special=true token typed {tname} instead of CONTROL (rendered on decode)",
                    })
            else:
                counts["added_nonspecial"] += 1
                is_word_like = is_norm or (astr in merges_out) or not (
                    (astr.startswith("<") and astr.endswith(">"))
                    or (astr.startswith("[") and astr.endswith("]"))
                    or (astr.startswith("|||") and astr.endswith("|||"))
                )
                if tt in (3, 5):
                    entry = {
                        "class": "added_nonspecial", "id": i, "gguf_str": gstr, "pub_str": astr,
                        "gguf_type": tname, "normalized": is_norm, "in_merges_out": astr in merges_out,
                        "why": (
                            "publisher added non-special token (matched first and rendered) typed "
                            f"{tname} (hidden on decode and masked in JSON/schema mode)"
                        ),
                    }
                    if is_word_like or strict_added:
                        failures.append(entry)
                    else:
                        reports.append(entry)
                elif tt == 1:
                    failures.append({
                        "class": "added_nonspecial", "id": i, "gguf_str": gstr, "pub_str": astr,
                        "gguf_type": tname,
                        "why": "publisher added non-special token typed NORMAL (not matched before BPE)",
                    })
        elif i in id2vocab:
            counts["plain_vocab"] += 1
            pstr = id2vocab[i]
            allowed = (1,) if is_bpe else (1, 2, 6)
            if tt not in allowed:
                failures.append({
                    "class": "plain_vocab", "id": i, "gguf_str": gstr, "pub_str": pstr,
                    "gguf_type": tname,
                    "why": f"plain vocab token (not in added_tokens) typed {tname} instead of NORMAL",
                })
            elif is_bpe and gstr != pstr:
                failures.append({
                    "class": "plain_vocab", "id": i, "gguf_str": gstr, "pub_str": pstr,
                    "gguf_type": tname,
                    "why": "plain vocab BPE token spelling differs from publisher model.vocab",
                })

    return {"failures": failures, "reports": reports, "summary": counts}


def format_report(res, show=20):
    lines = []
    s = res["summary"]
    lines.append(
        f"vocab-types: plain={s['plain_vocab']} added_nonspecial={s['added_nonspecial']} "
        f"special={s['special']} | failures={len(res['failures'])} reports={len(res['reports'])}"
    )
    for f in res["failures"][:show]:
        lines.append(
            f"  FAIL [{f['class']}] id={f['id']} gguf={f['gguf_str']!r} "
            f"pub={f['pub_str']!r} type={f['gguf_type']}: {f['why']}"
        )
    if len(res["failures"]) > show:
        lines.append(f"  ... {len(res['failures']) - show} more failure(s)")
    for r in res["reports"][:show]:
        if "id" in r:
            lines.append(
                f"  REPORT [{r['class']}] id={r['id']} gguf={r['gguf_str']!r} "
                f"type={r['gguf_type']}: {r['why']}"
            )
        else:
            lines.append(f"  REPORT {r['kind']}: {r['detail']}")
    if len(res["reports"]) > show:
        lines.append(f"  ... {len(res['reports']) - show} more report(s)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Headers-only GGUF token_type check vs publisher tokenizer.json")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gguf", help="local GGUF path (only header bytes are read)")
    src.add_argument("--hf-gguf", help="remote HuggingFace GGUF spec REPO@REV/FILE (HTTP Range only)")
    ap.add_argument("--ref", required=True, help="publisher tokenizer.json path, directory, or HF repo id")
    ap.add_argument("--ref-revision", help="immutable HuggingFace revision for --ref")
    ap.add_argument("--strict-added", action="store_true",
                    help="also fail on structural non-normalized added non-special tokens typed CONTROL")
    ap.add_argument("--show", type=int, default=20, help="max failures/reports to print")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    try:
        kv = read_local_gguf_kv(args.gguf) if args.gguf else read_hf_gguf_kv(args.hf_gguf)
        tj = load_publisher_tokenizer_json(args.ref, args.ref_revision)
    except Exception as exc:
        sys.stderr.write(f"check-vocab-types: error: {exc}\n")
        return 2

    res = check_vocab_types(kv, tj, strict_added=args.strict_added)
    if args.json:
        print(json.dumps(res, indent=1, ensure_ascii=False))
    else:
        print(format_report(res, show=args.show))
    return 1 if res["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
