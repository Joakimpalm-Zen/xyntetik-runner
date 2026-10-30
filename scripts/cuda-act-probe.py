#!/usr/bin/env python3
"""Runner's own CUDA sigmoid and SiLU kernels on inputs down to -270.

Under -ffast-math the CPU's plain 1/(1+expf(-x)) and x/(1+expf(-x)) return
NaN on x86-64 for every x below about -88 (src/act.h, #473). The CUDA kernels
keep the plain forms, and the argument that they are safe is the build's: the
PTX comes from nvcc without --use_fast_math, so division is div.rn.f32 (IEEE)
and 1/(1+inf) is exactly 0; the three rcp.approx sites (the GELU and MoE
activation kernels) return +0 for +inf. This checks that argument on a
device, with the kernels the binary embeds:

    python scripts/cuda-act-probe.py [--ptx src/kernels_ptx.h] [--out probe.json]

It needs the NVIDIA driver only (nvcuda.dll on Windows, libcuda.so.1 on
Linux): no toolkit, no nvcc, no numpy. The PTX is read from the header the
binary embeds and loaded through the driver API; three kernels run over the
same inputs, and each value is compared with a float64 reference:

    k_q35_attn_gate   x *= 1/(1+expf(-gate))        (the sigmoid)
    k_silu_mul        g = (g/(1+expf(-g))) * u      (SiLU, the dense FFN)
    k_moe_actmul      SiLU(g) * u * selw * dscale   (the MoE expert activation)

Exit 0 when every value is finite and within 1e-6 relative (1e-30 absolute)
of the reference, 1 when one is not, 2 when no device could be used.
"""
import argparse
import ctypes
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XS = [-269.7, -166.3, -100.0, -90.0, -89.0, -88.5, -88.0, -87.0, -80.0, -50.0,
      -1.0, 0.0, 4.6, 80.0, 89.0, 100.0]


def ptx_from_header(path):
    """The k_ptx_src string literal of kernels_ptx.h, unescaped."""
    text = open(path, encoding="utf-8").read()
    start = text.index("k_ptx_src")
    body = text[start:text.index(";\n", start)]
    out = []
    for lit in re.findall(r'"((?:\\.|[^"\\])*)"', body):
        out.append(re.sub(r'\\(.)', lambda m: {"n": "\n", "t": "\t", "r": "\r", "0": "\0"}.get(m.group(1), m.group(1)), lit))
    return "".join(out)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def close(got, want):
    return math.isfinite(got) and abs(got - want) <= 1e-6 * max(1.0, abs(want)) + 1e-30


class Cuda:
    def __init__(self):
        name = "nvcuda.dll" if os.name == "nt" else "libcuda.so.1"
        self.lib = ctypes.WinDLL(name) if os.name == "nt" else ctypes.CDLL(name)

    def __call__(self, fn, *args):
        rc = getattr(self.lib, fn)(*args)
        if rc != 0:
            raise RuntimeError(f"{fn} returned CUDA error {rc}")


def run(ptx):
    cu = Cuda()
    cu("cuInit", 0)
    dev = ctypes.c_int()
    cu("cuDeviceGet", ctypes.byref(dev), 0)
    name = ctypes.create_string_buffer(256)
    cu("cuDeviceGetName", name, 256, dev)
    drv = ctypes.c_int()
    cu("cuDriverGetVersion", ctypes.byref(drv))
    # the primary context, as cuda.c takes it
    ctx = ctypes.c_void_p()
    cu("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev)
    cu("cuCtxSetCurrent", ctx)
    mod = ctypes.c_void_p()
    src = ctypes.create_string_buffer(ptx.encode("utf-8"))
    cu("cuModuleLoadData", ctypes.byref(mod), src)

    n = len(XS)
    floats = ctypes.c_float * n

    def upload(values, ctype=ctypes.c_float):
        arr = (ctype * len(values))(*values)
        ptr = ctypes.c_uint64()
        cu("cuMemAlloc_v2", ctypes.byref(ptr), ctypes.c_size_t(ctypes.sizeof(arr)))
        cu("cuMemcpyHtoD_v2", ptr, arr, ctypes.c_size_t(ctypes.sizeof(arr)))
        return ptr

    def download(ptr):
        arr = floats()
        cu("cuMemcpyDtoH_v2", arr, ptr, ctypes.c_size_t(ctypes.sizeof(arr)))
        return [float(v) for v in arr]

    def launch(kernel, args):
        fn = ctypes.c_void_p()
        cu("cuModuleGetFunction", ctypes.byref(fn), mod, kernel.encode())
        params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
        cu("cuLaunchKernel", fn, 1, 1, 1, 32, 1, 1, 0, None, params, None)
        cu("cuCtxSynchronize")

    null = ctypes.c_uint64(0)
    results = {}
    # the sigmoid: x (ones) *= 1/(1+expf(-gate)), one row
    x = upload([1.0] * n)
    launch("k_q35_attn_gate", [x, upload(XS), ctypes.c_int(n), ctypes.c_int(n), ctypes.c_int(n)])
    results["k_q35_attn_gate"] = (download(x), [sigmoid(v) for v in XS])
    # SiLU(g) * u with u = 2
    g = upload(XS)
    launch("k_silu_mul", [g, upload([2.0] * n), ctypes.c_int(n)])
    results["k_silu_mul"] = (download(g), [2.0 * v * sigmoid(v) for v in XS])
    # the MoE expert activation, act 0 (SiLU), one selected expert, weight 1
    g2 = upload(XS)
    launch("k_moe_actmul", [g2, upload([2.0] * n), ctypes.c_int(n), ctypes.c_int(n), ctypes.c_int(n),
                            ctypes.c_int(0), upload([1.0]), upload([1.0]), upload([0], ctypes.c_int),
                            null, null])
    results["k_moe_actmul"] = (download(g2), [2.0 * v * sigmoid(v) for v in XS])
    cu("cuDevicePrimaryCtxRelease_v2", dev)
    return name.value.decode(errors="replace"), drv.value, results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ptx", default=os.path.join(ROOT, "src", "kernels_ptx.h"))
    ap.add_argument("--out")
    a = ap.parse_args()
    ptx = ptx_from_header(a.ptx)
    try:
        device, driver, results = run(ptx)
    except (OSError, RuntimeError, AttributeError) as e:
        print(f"cuda-act-probe: no device used: {e}", file=sys.stderr)
        return 2
    bad = 0
    report = {"device": device, "driver": driver, "inputs": XS, "kernels": {}}
    print(f"device {device}, driver {driver // 1000}.{driver % 1000 // 10}")
    for kernel, (got, want) in results.items():
        wrong = [(x, g, w) for x, g, w in zip(XS, got, want) if not close(g, w)]
        bad += len(wrong)
        report["kernels"][kernel] = {"got": got, "want": want, "wrong": len(wrong)}
        print(f"  {kernel:16s} {len(XS) - len(wrong)}/{len(XS)} finite and within float rounding"
              + "".join(f"\n      x={x:g}: got {g!r}, want {w:.6g}" for x, g, w in wrong))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
            f.write("\n")
    print("cuda-act-probe: " + ("every value finite and exact" if not bad else f"{bad} wrong values"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
