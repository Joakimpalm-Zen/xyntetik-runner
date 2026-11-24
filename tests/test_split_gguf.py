"""Native multi-part GGUF loading through the public runner surface."""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
N_SHARDS = 3


@pytest.fixture(scope="module")
def whole(tmp_path_factory):
    model = tmp_path_factory.mktemp("split") / "whole.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(model)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return model


@pytest.fixture(scope="module")
def shards(whole):
    subprocess.run(
        [sys.executable, ROOT / "scripts/gguf-split.py", str(whole),
         str(whole.parent / "part"), str(N_SHARDS)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    found = sorted(whole.parent.glob("part-*-of-*.gguf"))
    assert len(found) == N_SHARDS, found
    return found


@pytest.fixture(scope="module")
def sparse_shards(whole):
    """llama.cpp-style sets keep model metadata only in the first part."""
    subprocess.run(
        [sys.executable, ROOT / "scripts/gguf-split.py", str(whole),
         str(whole.parent / "sparse"), str(N_SHARDS), "--sparse-metadata"],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    found = sorted(whole.parent.glob("sparse-*-of-*.gguf"))
    assert len(found) == N_SHARDS, found
    return found


def _run(runner_bin, model):
    return subprocess.run(
        [runner_bin, "-m", str(model), "-p", "hi", "-n", "4", "--temp", "0",
         "--gpu", "off"], cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=120)


@pytest.mark.parametrize("which", range(N_SHARDS))
def test_every_part_loads_the_complete_model(runner_bin, whole, shards, which):
    expected = _run(runner_bin, whole)
    actual = _run(runner_bin, shards[which])
    assert expected.returncode == 0, expected.stderr.decode(errors="replace")
    assert actual.returncode == 0, actual.stderr.decode(errors="replace")
    assert actual.stdout == expected.stdout


def test_missing_part_is_refused_loudly(runner_bin, shards):
    missing = shards[1]
    parked = missing.with_suffix(".parked")
    missing.rename(parked)
    try:
        result = _run(runner_bin, shards[0])
    finally:
        parked.rename(missing)
    error = result.stderr.decode(errors="replace")
    assert result.returncode != 0
    assert "missing or unreadable split GGUF part 2/3" in error
    assert str(missing) in error


def test_parts_are_real_ggufs(shards):
    for shard in shards:
        assert shard.read_bytes()[:4] == b"GGUF"


def test_standard_sparse_metadata_parts_load_the_complete_model(
        runner_bin, whole, sparse_shards):
    expected = _run(runner_bin, whole)
    actual = _run(runner_bin, sparse_shards[0])
    assert expected.returncode == 0, expected.stderr.decode(errors="replace")
    assert actual.returncode == 0, actual.stderr.decode(errors="replace")
    assert actual.stdout == expected.stdout


def _quantize(runner_bin, src, dst):
    return subprocess.run(
        [runner_bin, "-m", str(src), "--quantize", str(dst), "--quant", "q8_0"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)


def test_quantizing_a_split_source_yields_a_loadable_single_file(
        runner_bin, whole, shards, tmp_path):
    """The quantizer emits ONE file. gguf_open merged the parts before it ran,
    so an output that still carries the source's split.* keys announces a
    multi-part set that does not exist -- and the loader refuses it outright,
    after the quantize exited 0 saying it had converted every tensor."""
    from_split = tmp_path / "from-split.gguf"
    from_whole = tmp_path / "from-whole.gguf"
    for src, dst in ((shards[0], from_split), (whole, from_whole)):
        done = _quantize(runner_bin, src, dst)
        assert done.returncode == 0, done.stderr.decode(errors="replace")

    split_run = _run(runner_bin, from_split)
    whole_run = _run(runner_bin, from_whole)
    assert whole_run.returncode == 0, whole_run.stderr.decode(errors="replace")
    assert split_run.returncode == 0, split_run.stderr.decode(errors="replace")
    assert split_run.stdout == whole_run.stdout


# R19.17.1: a part from ANOTHER SET of the same model (another quant, split the same way) carries the same
# split.count and split.tensors.count, so the split keys alone cannot tell it. A full-metadata part states the set's
# own keys too; where it states one the first part states and contradicts it, the set is refused by name. A sparse
# part (llama.cpp's standard layout: the three split keys only) states nothing to compare, so it is accepted, as
# llama.cpp accepts it: the format carries no set identity.
def _quant_set(runner_bin, whole, quant, prefix, *flags):
    q = whole.parent / f"{prefix}.gguf"
    done = subprocess.run([runner_bin, "-m", str(whole), "--quantize", str(q), "--quant", quant], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    assert done.returncode == 0, done.stderr.decode(errors="replace")
    subprocess.run([sys.executable, ROOT / "scripts/gguf-split.py", str(q), str(whole.parent / prefix),
                    str(N_SHARDS), *flags], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    found = sorted(whole.parent.glob(f"{prefix}-*-of-*.gguf"))
    assert len(found) == N_SHARDS, found
    return found


def test_a_full_metadata_part_from_another_quant_set_is_refused_by_name(runner_bin, whole):
    q8 = _quant_set(runner_bin, whole, "q8_0", "q8")
    q4 = _quant_set(runner_bin, whole, "q4_0", "q4")
    q8[1].write_bytes(q4[1].read_bytes())   # the Q4_0 set's part 2 in the Q8_0 set's place
    for entry in (q8[0], q8[1], q8[2]):
        result = _run(runner_bin, entry)
        error = result.stderr.decode(errors="replace")
        assert result.returncode != 0, (entry.name, error)
        assert "split GGUF part 2/3 is from another set: general.file_type 2 against 7 in part 1" in error, error
        assert str(q8[1]) in error, error


# The general key loop, kind by kind: the F32 fixture carries no general.file_type, so a refusal of these sets can
# only come from the loop over every key part 2 states. One key of each stored kind is changed in part 2 of a
# full-metadata set, in place and at its own length, so nothing else about the part moves.
_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_STR, _ARR = 8, 9


def _kv_span(raw, key):
    """(type, start, end) of `key`'s value in a GGUF file's header."""
    import struct

    def skip(t, p):
        if t == _STR:
            return p + 8 + struct.unpack_from("<Q", raw, p)[0]
        if t == _ARR:
            at, n = struct.unpack_from("<IQ", raw, p)
            p += 12
            if at == _STR:
                for _ in range(n):
                    p = skip(_STR, p)
                return p
            return p + _SCALAR[at] * n
        return p + _SCALAR[t]

    p, n_kv = 24, struct.unpack_from("<Q", raw, 16)[0]
    for _ in range(n_kv):
        klen = struct.unpack_from("<Q", raw, p)[0]
        name = raw[p + 8:p + 8 + klen].decode()
        p += 8 + klen
        t = struct.unpack_from("<I", raw, p)[0]
        end = skip(t, p + 4)
        if name == key:
            return t, p + 4, end
        p = end
    raise KeyError(key)


def _element(raw, start, k):
    """(start, end) of string element k of the string array whose value starts at `start`."""
    import struct
    p = start + 12
    for _ in range(k):
        p += 8 + struct.unpack_from("<Q", raw, p)[0]
    return p + 8, p + 8 + struct.unpack_from("<Q", raw, p)[0]


def _patched(raw, key, kind):
    import struct
    t, start, end = _kv_span(raw, key)
    raw = bytearray(raw)
    if kind == "u32":
        assert t == 4, t
        raw[start:end] = struct.pack("<I", struct.unpack_from("<I", raw, start)[0] + 1)
    elif kind == "f32":
        assert t == 6, t
        raw[start:end] = struct.pack("<f", struct.unpack_from("<f", raw, start)[0] * 2)
    elif kind == "bool":
        assert t == 7, t
        raw[start:end] = bytes([raw[start] ^ 1])
    elif kind == "str":
        assert t == _STR, t
        raw[end - 1] ^= 1   # the last character, at the string's own length
    elif kind == "str-array":
        assert t == _ARR and struct.unpack_from("<I", raw, start)[0] == _STR, t
        _s, e = _element(raw, start, 3)
        raw[e - 1] ^= 1
    elif kind == "f32-array":
        assert t == _ARR and struct.unpack_from("<I", raw, start)[0] == 6, t
        at = start + 12 + 4 * 5
        raw[at:at + 4] = struct.pack("<f", 1.0 + struct.unpack_from("<f", raw, at)[0])
    elif kind == "i32-array":
        assert t == _ARR and struct.unpack_from("<I", raw, start)[0] == 5, t
        at = start + 12 + 4 * 4
        raw[at:at + 4] = struct.pack("<i", struct.unpack_from("<i", raw, at)[0] + 1)
    return bytes(raw)


@pytest.mark.parametrize("key,kind,said", [
    ("llama.context_length", "u32", "llama.context_length 257 against 256 in part 1"),
    ("llama.rope.freq_base", "f32", "llama.rope.freq_base 20000 against 10000 in part 1"),
    ("tokenizer.ggml.add_bos_token", "bool", "tokenizer.ggml.add_bos_token false against true in part 1"),
    ("tokenizer.ggml.model", "str", 'tokenizer.ggml.model "llam`" against "llama" in part 1'),
    ("tokenizer.ggml.tokens", "str-array", "tokenizer.ggml.tokens, an array of "),
    ("tokenizer.ggml.scores", "f32-array", "tokenizer.ggml.scores, an array of "),
    ("tokenizer.ggml.token_type", "i32-array", "tokenizer.ggml.token_type, an array of "),
])
def test_a_full_metadata_part_contradicting_any_key_is_refused_by_name(runner_bin, whole, tmp_path, key, kind, said):
    subprocess.run([sys.executable, ROOT / "scripts/gguf-split.py", str(whole), str(tmp_path / "p"), str(N_SHARDS)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    parts = sorted(tmp_path.glob("p-*-of-*.gguf"))
    assert len(parts) == N_SHARDS, parts
    raw = parts[1].read_bytes()
    with pytest.raises(KeyError):   # no general.file_type: the loop is the only path here
        _kv_span(raw, "general.file_type")
    parts[1].write_bytes(_patched(raw, key, kind))
    result = _run(runner_bin, parts[0])
    error = result.stderr.decode(errors="replace")
    assert result.returncode != 0, error
    assert "split GGUF part 2/3 is from another set: " + said in error, error
    if kind.endswith("-array"):   # two arrays of one length: the sentence names the element where they part
        at = {"str-array": 3, "f32-array": 5, "i32-array": 4}[kind]
        assert f", differs from part 1's at element {at}: " in error, error
    assert str(parts[1]) in error, error


def test_a_sparse_part_from_another_quant_set_is_accepted_as_the_format_allows(runner_bin, whole):
    # the limit, held so a change to it is seen: a sparse part states only the split keys
    q8 = _quant_set(runner_bin, whole, "q8_0", "sq8", "--sparse-metadata")
    q4 = _quant_set(runner_bin, whole, "q4_0", "sq4", "--sparse-metadata")
    q8[1].write_bytes(q4[1].read_bytes())
    result = _run(runner_bin, q8[0])
    assert result.returncode == 0, result.stderr.decode(errors="replace")


# R19.17.1: a split set is named as the set whichever part opens it: the first part's file name (the name the set's
# other parts are derived from, and the one llama.cpp and the Hub's listings lead with) and the whole set's bytes.
# A single file keeps its own name and size.
def _serve_names(runner_bin, model, home):
    """What a server started from `model` names: /v1/models' id, the capabilities, its load line, its discovery
    record (under a test HOME) and its provenance model block, once the digest is taken."""
    sys.path.insert(0, str(ROOT / "tests" / "conformance"))
    import json
    import os
    import time
    import urllib.request
    from harness import RunnerServer
    log = model.parent / f"serve-{model.name}.log"
    home.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HOME=str(home), APPDATA=str(home))
    records = home / ("xyntetik" if os.name == "nt" else ".xyntetik") / "runner" / "instances"
    with RunnerServer(runner_bin, model, ctx=256, log_path=str(log), env=env,
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        models = json.load(urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30))
        caps = json.load(urllib.request.urlopen(srv.base_url + "/v1/capabilities", timeout=30))
        record = json.loads((records / f"{srv.proc.pid}.json").read_text())
        deadline = time.monotonic() + 30
        while True:
            prov = json.load(urllib.request.urlopen(srv.base_url + "/v1/runner/provenance", timeout=30))["model"]
            if prov["sha256_state"] != "hashing" or time.monotonic() > deadline:
                break
            time.sleep(0.05)
    loaded = [ln for ln in log.read_text(errors="replace").splitlines() if ln.startswith("loaded ")]
    return models["data"][0]["id"], caps, loaded, record["models"], prov


def _cli_names(runner_bin, model, tmp_path):
    """What the one-shot surfaces name: --info doctor's model block, --bench-json's model, --decide's model."""
    import json
    run = dict(cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    doctor = subprocess.run([runner_bin, "-m", str(model), "--info", "doctor", "--gpu", "off"], **run)
    # 2: the report's verdict is "broken", as a random fixture's probe reply is; the report is printed either way
    assert doctor.returncode in (0, 2), doctor.stderr.decode(errors="replace")
    bench = subprocess.run([runner_bin, "-m", str(model), "--bench-json", "-p", "hi", "-n", "2", "--gpu", "off"], **run)
    assert bench.returncode == 0, bench.stderr.decode(errors="replace")
    req = tmp_path / "decide.jsonl"
    req.write_text(json.dumps({"state": "the fox", "questions": [{"id": "a", "question": "which",
                                                                "options": ["jumps", "sleeps"]}]}) + "\n")
    decide = subprocess.run([runner_bin, "-m", str(model), "--decide", str(req), "-t", "2", "--gpu", "off"], **run)
    assert decide.returncode == 0, decide.stderr.decode(errors="replace")
    d = json.loads(doctor.stdout)["model"]
    return ({"file": d["file"], "bytes": d["bytes"]}, json.loads(bench.stdout.splitlines()[-1])["model"],
            json.loads(decide.stdout.splitlines()[0])["model"])


def _score_model(runner_bin, model):
    import json
    done = subprocess.run([runner_bin, "-m", str(model), "--score", "-p", "the warehouse keeps item 42 in aisle seven",
                           "--gpu", "off", "-t", "2"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=120)
    assert done.returncode == 0, done.stderr.decode(errors="replace")
    return json.loads(done.stdout)["score_route"]["model"]


def _caps_names(caps):
    import json
    text = json.dumps(caps)
    return sorted({n for n in ("part-00001-of-00003.gguf", "part-00002-of-00003.gguf", "part-00003-of-00003.gguf")
                   if n in text})


# Every surface that names THE MODEL names the set; provenance names one FILE beside that file's digest, so for a set
# it is the part the server was started from, as a receipt names it.
def test_a_split_set_is_named_as_the_set_from_every_part(runner_bin, whole, shards, tmp_path):
    import hashlib
    set_bytes = sum(p.stat().st_size for p in shards)
    for entry in shards:
        mid, caps, loaded, record, prov = _serve_names(runner_bin, entry, tmp_path / f"home-{entry.name}")
        assert mid == shards[0].name, (entry.name, mid)
        assert _caps_names(caps) == [shards[0].name], (entry.name, _caps_names(caps))
        assert loaded and loaded[0].startswith(f"loaded {shards[0]} | "), (entry.name, loaded)
        assert loaded[0].endswith(" | a split set of 3 parts"), (entry.name, loaded)
        assert [(r["name"], pathlib.Path(r["path"]).name) for r in record] == [(shards[0].name, shards[0].name)], \
            (entry.name, record)
        assert (prov["id"], pathlib.Path(prov["path"]).name, prov["size"]) == \
            (entry.name, entry.name, entry.stat().st_size), (entry.name, prov)
        assert prov["sha256"] == hashlib.sha256(entry.read_bytes()).hexdigest(), (entry.name, prov)
        assert _score_model(runner_bin, entry) == {"file": shards[0].name, "bytes": set_bytes}, entry.name
        doctor, bench, decided = _cli_names(runner_bin, entry, tmp_path)
        assert doctor == {"file": shards[0].name, "bytes": set_bytes}, (entry.name, doctor)
        assert pathlib.Path(bench) == shards[0], (entry.name, bench)
        assert decided == shards[0].name, (entry.name, decided)


def test_a_single_file_keeps_its_own_name(runner_bin, whole, tmp_path):
    mid, caps, loaded, record, prov = _serve_names(runner_bin, whole, tmp_path / "home")
    assert mid == whole.name, mid
    assert loaded and loaded[0].startswith(f"loaded {whole} | ") and "split set" not in loaded[0], loaded
    assert [r["name"] for r in record] == [whole.name] and prov["id"] == whole.name, (record, prov)
    assert _score_model(runner_bin, whole) == {"file": whole.name, "bytes": whole.stat().st_size}
    doctor, bench, decided = _cli_names(runner_bin, whole, tmp_path)
    assert doctor == {"file": whole.name, "bytes": whole.stat().st_size} and pathlib.Path(bench) == whole, (doctor, bench)
    assert decided == whole.name, decided


# A single file with an --mtp-file companion maps two files, and is not a split set: the load line names the file
# and says nothing of a set (it said "a split set of 2 parts" when the count of mapped files decided).
def test_a_single_file_with_an_mtp_companion_is_not_a_split_set(runner_bin, whole):
    companion = whole.parent / "mtp-companion.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(companion), "--mtp-layers", "1"],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    result = subprocess.run([runner_bin, "-m", str(whole), "--mtp-file", str(companion), "-p", "hi", "-n", "4",
                             "--temp", "0", "--gpu", "off"], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=120)
    error = result.stderr.decode(errors="replace")
    assert result.returncode == 0, error
    assert "mtp: 1 predictor block(s) declared" in error, error   # the companion did attach
    loaded = [ln for ln in error.splitlines() if ln.startswith("loaded ")]
    assert loaded and loaded[0].startswith(f"loaded {whole} | ") and "split set" not in loaded[0], loaded
