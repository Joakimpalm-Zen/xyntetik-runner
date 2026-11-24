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
def _serve_names(runner_bin, model):
    sys.path.insert(0, str(ROOT / "tests" / "conformance"))
    import json
    import urllib.request
    from harness import RunnerServer
    log = model.parent / f"serve-{model.name}.log"
    with RunnerServer(runner_bin, model, ctx=256, log_path=str(log), extra_args=["--gpu", "off", "-t", "2"]) as srv:
        models = json.load(urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30))
        caps = json.load(urllib.request.urlopen(srv.base_url + "/v1/capabilities", timeout=30))
    loaded = [ln for ln in log.read_text(errors="replace").splitlines() if ln.startswith("loaded ")]
    return models["data"][0]["id"], caps, loaded


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


def test_a_split_set_is_named_as_the_set_from_every_part(runner_bin, whole, shards):
    set_bytes = sum(p.stat().st_size for p in shards)
    for entry in shards:
        mid, caps, loaded = _serve_names(runner_bin, entry)
        assert mid == shards[0].name, (entry.name, mid)
        assert _caps_names(caps) == [shards[0].name], (entry.name, _caps_names(caps))
        assert loaded and loaded[0].endswith(" | a split set of 3 parts"), (entry.name, loaded)
        assert _score_model(runner_bin, entry) == {"file": shards[0].name, "bytes": set_bytes}, entry.name


def test_a_single_file_keeps_its_own_name(runner_bin, whole):
    mid, caps, loaded = _serve_names(runner_bin, whole)
    assert mid == whole.name, mid
    assert loaded and "split set" not in loaded[0], loaded
    assert _score_model(runner_bin, whole) == {"file": whole.name, "bytes": whole.stat().st_size}
