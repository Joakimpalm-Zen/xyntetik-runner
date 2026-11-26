#!/usr/bin/env python3
"""R4.48.8.3: Produce or wrap pinned pre-computed reference logits
(`xyntetik.runner.ref-logits.v1`) for `runner derive --reference logits:MANIFEST.json`
and `runner --score --score-ref MANIFEST.json`.

Subcommands:
  wrap     Wrap an existing raw float32 (`*.logits.f32`) or `.npy` dump
           (such as `ref_firstlight.py read` output) into a pinned `.ref.json`
           manifest with mandatory SHA-256 verification and O(1) RAM.
  plan     Check memory requirements against an explicit `--ram-cap-gib <G>`
           budget. Never silently falls back to `patched-stream`: the default
           `--mode patched` refuses if patched exceeds 90% of the cap, because
           `patched-stream` is a new execution mode that must first pass the
           bit-for-bit `control` check against `patched` on a short read.
  produce  Run the publisher float32 forward under `--ram-cap-gib <G>` in either
           `--mode patched` (in-memory `Store` + `Patch`) or explicit
           `--mode patched-stream` (per-layer `safe_open` pre/post hooks),
           refuse any non-finite logits, and write `.logits.npy` + `.ref.json`.
  control  Run BOTH `patched` and `patched-stream` on a short prefix
           (`--positions N`, default 256) and assert bit-for-bit identical
           float32 logits (`0` differing bits, equal SHA-256).
"""
import argparse
import hashlib
import json
import os
import pathlib
import struct
import sys

SCHEMA = "xyntetik.runner.ref-logits.v1"
HEADROOM = 0.90


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_npy_f32_2d(path: pathlib.Path, rows: int, cols: int, raw_f32: bytes) -> None:
    expected = rows * cols * 4
    if len(raw_f32) != expected:
        raise ValueError(f"raw float32 length {len(raw_f32)} != {rows}*{cols}*4 ({expected})")
    descr = f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}"
    hdr_b = descr.encode("ascii")
    pad = (64 - ((10 + len(hdr_b) + 1) % 64)) % 64
    hdr = hdr_b + (b" " * pad) + b"\n"
    with path.open("wb") as f:
        f.write(b"\x93NUMPY\x01\x00")
        f.write(struct.pack("<H", len(hdr)))
        f.write(hdr)
        f.write(raw_f32)


def parse_window(spec: str):
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"--window must be START:STRIDE:COUNT (got {spec!r})")
    s, t, c = (int(x) for x in parts)
    if s < 1 or t < 1 or c < 1:
        raise ValueError(f"--window values must be >= 1 (got {spec!r})")
    return s, t, c


def estimate_hf_gib(hf_dir: pathlib.Path):
    """Estimate memory tiers (plain f32, patched Store, layer-streaming patched)
    from config.json and safetensors headers only (< 50 MB RAM)."""
    cfg_path = hf_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    tcfg = cfg.get("text_config", cfg)

    idx_path = hf_dir / "model.safetensors.index.json"
    shards = []
    if idx_path.exists():
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        shards = sorted({hf_dir / v for v in idx.get("weight_map", {}).values()})
    else:
        single = hf_dir / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"missing safetensors weights in {hf_dir}")
        shards = [single]

    dtype_bytes = {
        "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
        "F8_E4M3": 1, "F8_E5M2": 1, "U8": 1, "I8": 1,
        "I32": 4, "I64": 8, "U32": 4, "U64": 8,
    }
    plain_f32 = 0
    patched_store = 0
    non_expert = 0
    layer_f32 = {}

    for sp in shards:
        with sp.open("rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen).decode("utf-8"))
        for k, meta in hdr.items():
            if k == "__metadata__":
                continue
            shape = meta["shape"]
            numel = 1
            for d in shape:
                numel *= int(d)
            dt = meta["dtype"]
            # Packed NVFP4 (U8) holds 2 nibbles per byte -> 2 elements per U8
            eff_numel = numel * 2 if (dt == "U8" and "experts" in k and k.endswith(".weight")) else numel
            is_scale = k.endswith("_scale") or k.endswith(".scale") or "scale" in k
            f32_b = 0 if is_scale else eff_numel * 4
            plain_f32 += f32_b
            # Patched store holds BF16 (2 B/elem) for experts/bf16 weights, F32 for small/f32 tensors
            st_b = 0 if is_scale else (eff_numel * 2 if eff_numel >= 4096 else eff_numel * 4)
            patched_store += st_b
            if ".layers." in k and ".experts." in k:
                # Extract physical layer index
                after = k.split(".layers.", 1)[1]
                lid = after.split(".", 1)[0]
                layer_f32[lid] = layer_f32.get(lid, 0) + f32_b
            else:
                non_expert += st_b

    max_layer = max(layer_f32.values(), default=0)
    # Add ~4.5 GiB activation/window overhead for patched, ~2.5 GiB for layer-streaming
    gib = float(1024 ** 3)
    mem_plain = plain_f32 / gib
    mem_patched = (patched_store / gib) + 4.5
    mem_stream = ((non_expert + max_layer) / gib) + 2.5
    return {
        "arch": tcfg.get("model_type", "unknown"),
        "mem_plain_f32_gib": round(mem_plain, 2),
        "mem_patched_gib": round(mem_patched, 2),
        "mem_stream_gib": round(mem_stream, 2),
    }


def evaluate_plan(mem_patched_gib: float, mem_stream_gib: float, ram_cap_gib: float, mode: str):
    if ram_cap_gib <= 0:
        return False, "error: --ram-cap-gib must be > 0"
    budget = ram_cap_gib * HEADROOM
    if mode == "patched":
        if mem_patched_gib <= budget:
            return True, (
                f"plan: mode=patched fits ({mem_patched_gib:.2f} GiB <= "
                f"{budget:.2f} GiB = 90% of {ram_cap_gib:.2f} GiB cap)"
            )
        return False, (
            f"error: mode=patched requires ~{mem_patched_gib:.2f} GiB, which exceeds "
            f"{budget:.2f} GiB (90% of --ram-cap-gib {ram_cap_gib:.2f}). "
            f"Layer-streaming mode (~{mem_stream_gib:.2f} GiB) is NOT auto-selected "
            f"because it is a separate execution mode: run `make-ref-logits.py control` "
            f"against `patched` on a short read to confirm bit-for-bit parity first, "
            f"then pass `--mode patched-stream` explicitly."
        )
    if mode == "patched-stream":
        if mem_stream_gib <= budget:
            return True, (
                f"plan: mode=patched-stream fits ({mem_stream_gib:.2f} GiB <= "
                f"{budget:.2f} GiB = 90% of {ram_cap_gib:.2f} GiB cap); "
                f"ensure `make-ref-logits.py control` has confirmed bit-for-bit parity "
                f"with `patched` on this model before relying on its output"
            )
        return False, (
            f"error: mode=patched-stream requires ~{mem_stream_gib:.2f} GiB, which exceeds "
            f"{budget:.2f} GiB (90% of --ram-cap-gib {ram_cap_gib:.2f}; "
            f"patched would need ~{mem_patched_gib:.2f} GiB)"
        )
    return False, f"error: unknown mode {mode!r} (expected 'patched' or 'patched-stream')"


def cmd_plan(args) -> int:
    if args.hf:
        est = estimate_hf_gib(pathlib.Path(args.hf))
        mp, ms = est["mem_patched_gib"], est["mem_stream_gib"]
    else:
        if args.mem_patched_gib is None or args.mem_stream_gib is None:
            print("error: plan requires --hf DIR or both --mem-patched-gib and --mem-stream-gib", file=sys.stderr)
            return 1
        mp, ms = args.mem_patched_gib, args.mem_stream_gib
    ok, msg = evaluate_plan(mp, ms, args.ram_cap_gib, args.mode)
    print(msg, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 2


def cmd_wrap(args) -> int:
    logits_path = pathlib.Path(args.logits).resolve()
    out_path = pathlib.Path(args.out).resolve()
    if not logits_path.is_file():
        print(f"error: logits file not found: {logits_path}", file=sys.stderr)
        return 1
    if logits_path.parent != out_path.parent:
        try:
            rel = logits_path.relative_to(out_path.parent)
            rel_str = rel.as_posix()
        except ValueError:
            print(
                f"error: logits file {logits_path} must live in or under manifest directory {out_path.parent}",
                file=sys.stderr,
            )
            return 1
    else:
        rel_str = logits_path.name

    tokens = None
    n_special = args.n_special
    n_vocab = args.n_vocab
    ref_class = args.ref_class
    ref_dtype = args.ref_dtype or "float32"
    ref_device = args.device or "cpu"
    producer_mode = args.producer_mode or "wrap-raw-f32"

    if args.tokens_json:
        td = json.loads(pathlib.Path(args.tokens_json).read_text(encoding="utf-8"))
        tokens = [int(x) for x in td["tokens"]]
        if n_special is None:
            n_special = int(td.get("n_special", 0))
        if n_vocab is None and "n_vocab" in td:
            n_vocab = int(td["n_vocab"])
    elif args.safetensors:
        from safetensors import safe_open
        with safe_open(args.safetensors, framework="pt") as sf:
            tokens = [int(x) for x in sf.get_tensor("ids").tolist()]

    if args.info:
        info = json.loads(pathlib.Path(args.info).read_text(encoding="utf-8"))
        if n_vocab is None and "V" in info:
            n_vocab = int(info["V"])
        if not args.producer_mode and info.get("mode"):
            producer_mode = f"wrap-{info['mode']}"

    if not tokens or len(tokens) < 2:
        print("error: wrap requires --tokens-json or --safetensors with >= 2 token IDs", file=sys.stderr)
        return 1
    if n_vocab is None or n_vocab < 2:
        print("error: wrap requires --n-vocab (or n_vocab in --tokens-json / V in --info)", file=sys.stderr)
        return 1
    if n_special is None:
        n_special = 0

    fmt = args.format
    if not fmt:
        fmt = "npy" if logits_path.suffix == ".npy" else "raw-f32"

    fsize = logits_path.stat().st_size
    if fmt == "raw-f32":
        row_bytes = n_vocab * 4
        if fsize == 0 or (fsize % row_bytes) != 0:
            print(f"error: raw-f32 file size {fsize} is not a multiple of n_vocab*4 ({row_bytes})", file=sys.stderr)
            return 1
        n_rows = fsize // row_bytes
    elif fmt == "npy":
        import ast
        with logits_path.open("rb") as f:
            magic = f.read(6)
            if magic != b"\x93NUMPY":
                print(f"error: {logits_path} is not a valid .npy file", file=sys.stderr)
                return 1
            ver = struct.unpack("<BB", f.read(2))
            hlen = struct.unpack("<H" if ver[0] == 1 else "<I", f.read(2 if ver[0] == 1 else 4))[0]
            hdr = ast.literal_eval(f.read(hlen).decode("latin1"))
            if hdr.get("descr") not in ("<f4", "=f4") or hdr.get("fortran_order"):
                print(f"error: .npy must be C-order <f4 (got {hdr})", file=sys.stderr)
                return 1
            shape = hdr.get("shape")
            if not isinstance(shape, tuple) or len(shape) != 2 or int(shape[1]) != n_vocab:
                print(f"error: .npy shape {shape} does not match n_vocab {n_vocab}", file=sys.stderr)
                return 1
            n_rows = int(shape[0])
    else:
        print(f"error: unsupported format {fmt!r}", file=sys.stderr)
        return 1

    lsha = sha256_file(logits_path)
    sha_sidecar = pathlib.Path(str(logits_path) + ".sha256")
    if sha_sidecar.is_file():
        expected = sha_sidecar.read_text(encoding="utf-8").strip().split()[0].lower()
        if expected != lsha:
            print(f"error: {sha_sidecar} expects {expected}, actual sha256 is {lsha}", file=sys.stderr)
            return 1

    manifest = {
        "schema_version": SCHEMA,
        "logits": {
            "path": rel_str,
            "format": fmt,
            "dtype": "float32",
            "shape": [n_rows, n_vocab],
            "sha256": lsha,
            "row_indices": None,
        },
        "tokens": tokens,
        "n_special": n_special,
        "n_vocab": n_vocab,
        "reference": {
            "repo": args.repo,
            "revision": args.revision,
            "class": ref_class,
            "dtype": ref_dtype,
            "device": ref_device,
            "producer": "scripts/make-ref-logits.py",
            "producer_mode": producer_mode,
            "weights_sha256": args.weights_sha256,
        },
    }
    if args.corpus:
        cp = pathlib.Path(args.corpus)
        manifest["corpus"] = {
            "path": cp.name,
            "sha256": sha256_file(cp),
            "bytes": cp.stat().st_size,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrap: wrote {out_path} ({fmt} [{n_rows}, {n_vocab}], sha256={lsha[:12]}..., mode={producer_mode})")
    return 0


def _run_forward_logits(hf_dir: pathlib.Path, ids, mode: str, threads: int, ref_firstlight_path: pathlib.Path | None):
    """Execute the publisher float32 forward in `patched` or `patched-stream` mode.
    Returns (logits_f32_ndarray [T, V], model_class_name)."""
    import importlib.util
    import numpy as np
    import torch

    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)

    if mode == "patched":
        if ref_firstlight_path is None:
            raise RuntimeError("mode=patched requires --ref-firstlight PATH pointing to ref_firstlight.py")
        spec = importlib.util.spec_from_file_location("ref_firstlight", str(ref_firstlight_path))
        rf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rf)
        ns = argparse.Namespace(source="real", dir=str(hf_dir), mode="patched", threads=threads)
        info = {}
        model, _patch, _cfg = rf.setup_model(ns, info)
        if model is None:
            raise RuntimeError("ref_firstlight.setup_model returned None")
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False)
        logits = out.logits[0].float().contiguous()
        if int((~torch.isfinite(logits)).sum()) != 0:
            raise RuntimeError("non-finite logits produced in mode=patched")
        return logits.numpy().astype(np.float32, copy=False), type(model).__name__

    if mode == "patched-stream":
        if ref_firstlight_path is None:
            raise RuntimeError("mode=patched-stream requires --ref-firstlight PATH pointing to ref_firstlight.py")
        spec = importlib.util.spec_from_file_location("ref_firstlight", str(ref_firstlight_path))
        rf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rf)
        from safetensors import safe_open

        cfg = rf.load_config("real", str(hf_dir))
        model = rf.meta_model(cfg)
        # Resolve shard map for safe_open streaming per physical layer.
        idx_path = hf_dir / "model.safetensors.index.json"
        if idx_path.exists():
            wmap = json.loads(idx_path.read_text(encoding="utf-8"))["weight_map"]
        else:
            with safe_open(str(hf_dir / "model.safetensors"), framework="pt") as sf:
                wmap = {k: "model.safetensors" for k in sf.keys()}

        # Materialize non-expert tensors once (dense/embeddings/norms/head),
        # leaving MoE expert stacks on meta until each physical layer executes.
        loaders = {}
        for shard_name in sorted(set(wmap.values())):
            loaders[shard_name] = rf.Loader(str(hf_dir / shard_name), model, cfg)

        def _find_loader(target_name):
            for ld in loaders.values():
                if target_name in ld.sources:
                    return ld
            return None

        expert_targets_by_layer = {}
        for pname, param in list(model.named_parameters()):
            if ".layers." in pname and ".experts." in pname:
                lid = int(pname.split(".layers.", 1)[1].split(".", 1)[0])
                expert_targets_by_layer.setdefault(lid, []).append(pname)
            else:
                ld = _find_loader(pname)
                if ld is None:
                    raise RuntimeError(f"missing source tensor for {pname}")
                t = ld.reference(pname)
                parts = pname.split(".")
                mod = model
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                setattr(mod, parts[-1], torch.nn.Parameter(t, requires_grad=False))

        handles = []
        for lid, layer in enumerate(model.model.layers):
            targets = expert_targets_by_layer.get(lid, [])
            if not targets:
                continue

            def _make_hooks(tlist):
                def _pre(_mod, _args):
                    for pname in tlist:
                        ld = _find_loader(pname)
                        t = ld.reference(pname)
                        parts = pname.split(".")
                        m = model
                        for p in parts[:-1]:
                            m = getattr(m, p)
                        setattr(m, parts[-1], torch.nn.Parameter(t, requires_grad=False))

                def _post(_mod, _args, _out):
                    for pname in tlist:
                        parts = pname.split(".")
                        m = model
                        for p in parts[:-1]:
                            m = getattr(m, p)
                        old = getattr(m, parts[-1])
                        setattr(
                            m,
                            parts[-1],
                            torch.nn.Parameter(
                                torch.empty(old.shape, dtype=old.dtype, device="meta"),
                                requires_grad=False,
                            ),
                        )

                return _pre, _post

            pre_h, post_h = _make_hooks(targets)
            handles.append(layer.register_forward_pre_hook(pre_h))
            handles.append(layer.register_forward_hook(post_h))

        try:
            with torch.no_grad():
                out = model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False)
            logits = out.logits[0].float().contiguous()
        finally:
            for h in handles:
                h.remove()
        if int((~torch.isfinite(logits)).sum()) != 0:
            raise RuntimeError("non-finite logits produced in mode=patched-stream")
        return logits.numpy().astype(np.float32, copy=False), type(model).__name__

    raise ValueError(f"unsupported mode {mode!r}")


def cmd_produce(args) -> int:
    hf_dir = pathlib.Path(args.hf)
    corpus_path = pathlib.Path(args.corpus)
    out_prefix = pathlib.Path(args.out_prefix)

    est = estimate_hf_gib(hf_dir)
    ok, msg = evaluate_plan(est["mem_patched_gib"], est["mem_stream_gib"], args.ram_cap_gib, args.mode)
    if not ok:
        print(msg, file=sys.stderr)
        return 2
    print(msg)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    text = corpus_path.read_text(encoding="utf-8")
    ids = tok.encode(text, add_special_tokens=True)
    n_special = 1 if (tok.bos_token_id is not None and ids and ids[0] == tok.bos_token_id) else 0
    if args.max_tokens and len(ids) > args.max_tokens:
        ids = ids[: args.max_tokens]

    rf_path = pathlib.Path(args.ref_firstlight) if args.ref_firstlight else None
    arr, cls_name = _run_forward_logits(hf_dir, ids, args.mode, args.threads, rf_path)
    n_rows, n_vocab = int(arr.shape[0]), int(arr.shape[1])

    row_indices = None
    if args.window and args.sparse:
        s, st, cnt = parse_window(args.window)
        row_indices = [n_special + s + i * st - 1 for i in range(cnt)]
        arr = arr[row_indices].copy()
        n_rows = len(row_indices)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    npy_path = pathlib.Path(str(out_prefix) + ".logits.npy")
    write_npy_f32_2d(npy_path, n_rows, n_vocab, arr.tobytes(order="C"))
    lsha = sha256_file(npy_path)
    pathlib.Path(str(npy_path) + ".sha256").write_text(f"{lsha}  {npy_path.name}\n", encoding="utf-8")

    man_path = pathlib.Path(str(out_prefix) + ".ref.json")
    manifest = {
        "schema_version": SCHEMA,
        "logits": {
            "path": npy_path.name,
            "format": "npy",
            "dtype": "float32",
            "shape": [n_rows, n_vocab],
            "sha256": lsha,
            "row_indices": row_indices,
        },
        "tokens": [int(x) for x in ids],
        "n_special": n_special,
        "n_vocab": n_vocab,
        "corpus": {
            "path": corpus_path.name,
            "sha256": sha256_file(corpus_path),
            "bytes": corpus_path.stat().st_size,
        },
        "reference": {
            "repo": args.repo or hf_dir.name,
            "revision": args.revision,
            "class": cls_name,
            "dtype": "float32",
            "device": "cpu",
            "producer": "scripts/make-ref-logits.py",
            "producer_mode": args.mode,
            "weights_sha256": args.weights_sha256,
        },
    }
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"produce: wrote {man_path} + {npy_path} ([{n_rows}, {n_vocab}], sha256={lsha[:12]}..., mode={args.mode})")
    return 0


def cmd_control(args) -> int:
    """Bit-for-bit parity control between `patched` and `patched-stream` on a short read."""
    import numpy as np
    from transformers import AutoTokenizer

    hf_dir = pathlib.Path(args.hf)
    corpus_path = pathlib.Path(args.corpus)
    rf_path = pathlib.Path(args.ref_firstlight)

    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    ids = tok.encode(corpus_path.read_text(encoding="utf-8"), add_special_tokens=True)
    ids = ids[: args.positions]

    a, cls_a = _run_forward_logits(hf_dir, ids, "patched", args.threads, rf_path)
    b, cls_b = _run_forward_logits(hf_dir, ids, "patched-stream", args.threads, rf_path)

    ab = a.tobytes(order="C")
    bb = b.tobytes(order="C")
    sha_a = hashlib.sha256(ab).hexdigest()
    sha_b = hashlib.sha256(bb).hexdigest()
    diff_bits = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
    max_abs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    report = {
        "positions": len(ids),
        "shape": list(a.shape),
        "class_patched": cls_a,
        "class_patched_stream": cls_b,
        "sha256_patched": sha_a,
        "sha256_patched_stream": sha_b,
        "differing_elements": diff_bits,
        "max_abs": max_abs,
        "bit_exact": (diff_bits == 0 and sha_a == sha_b),
    }
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["bit_exact"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Produce or wrap pinned ref-logits manifests (R4.48.8.3).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Check RAM budget against --ram-cap-gib")
    p_plan.add_argument("--hf", help="HF checkpoint directory (estimate from safetensors headers)")
    p_plan.add_argument("--mem-patched-gib", type=float, help="Override patched GiB estimate (for testing)")
    p_plan.add_argument("--mem-stream-gib", type=float, help="Override patched-stream GiB estimate (for testing)")
    p_plan.add_argument("--ram-cap-gib", type=float, required=True, help="Explicit host/container RAM cap in GiB")
    p_plan.add_argument("--mode", default="patched", choices=["patched", "patched-stream"])

    p_wrap = sub.add_parser("wrap", help="Wrap existing *.logits.f32 or .npy into a pinned .ref.json manifest")
    p_wrap.add_argument("--logits", required=True, help="Existing logits file (.logits.f32 or .npy)")
    p_wrap.add_argument("--format", choices=["raw-f32", "npy"], help="Format (default inferred from extension)")
    p_wrap.add_argument("--tokens-json", help="JSON containing tokens[] (and optional n_special, n_vocab)")
    p_wrap.add_argument("--safetensors", help="ref_firstlight.py .safetensors containing ids tensor")
    p_wrap.add_argument("--info", help="ref_firstlight.py .json sidecar")
    p_wrap.add_argument("--n-vocab", type=int, help="Vocabulary size V")
    p_wrap.add_argument("--n-special", type=int, help="Number of leading special tokens (e.g. BOS)")
    p_wrap.add_argument("--corpus", help="Corpus file to pin by sha256")
    p_wrap.add_argument("--repo", help="Publisher repository name")
    p_wrap.add_argument("--revision", help="Publisher revision SHA")
    p_wrap.add_argument("--ref-class", help="Model class name")
    p_wrap.add_argument("--ref-dtype", default="float32")
    p_wrap.add_argument("--device", default="cpu")
    p_wrap.add_argument("--producer-mode", help="Producer mode tag recorded in manifest")
    p_wrap.add_argument("--weights-sha256", help="Publisher weights sha256")
    p_wrap.add_argument("--out", required=True, help="Output manifest .ref.json path")

    p_prod = sub.add_parser("produce", help="Produce pinned reference logits under --ram-cap-gib")
    p_prod.add_argument("--hf", required=True)
    p_prod.add_argument("--corpus", required=True)
    p_prod.add_argument("--out-prefix", required=True)
    p_prod.add_argument("--ram-cap-gib", type=float, required=True)
    p_prod.add_argument("--mode", default="patched", choices=["patched", "patched-stream"])
    p_prod.add_argument("--ref-firstlight", required=True, help="Path to ref_firstlight.py")
    p_prod.add_argument("--threads", type=int, default=4)
    p_prod.add_argument("--max-tokens", type=int, default=0)
    p_prod.add_argument("--window", help="Optional START:STRIDE:COUNT window")
    p_prod.add_argument("--sparse", action="store_true", help="Store only --window rows with row_indices")
    p_prod.add_argument("--repo")
    p_prod.add_argument("--revision")
    p_prod.add_argument("--weights-sha256")

    p_ctl = sub.add_parser("control", help="Verify bit-for-bit parity of patched vs patched-stream")
    p_ctl.add_argument("--hf", required=True)
    p_ctl.add_argument("--corpus", required=True)
    p_ctl.add_argument("--ref-firstlight", required=True)
    p_ctl.add_argument("--positions", type=int, default=256)
    p_ctl.add_argument("--threads", type=int, default=4)
    p_ctl.add_argument("--out", help="Optional control receipt JSON path")

    args = ap.parse_args()
    if args.cmd == "plan":
        return cmd_plan(args)
    if args.cmd == "wrap":
        return cmd_wrap(args)
    if args.cmd == "produce":
        return cmd_produce(args)
    if args.cmd == "control":
        return cmd_control(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
