#!/bin/sh
# The one way to regenerate src/kernels_ptx.h and src/kernels_ptx80.h.
#
# ptx-linux-64.lock pins every package of the toolchain (nvcc, crt, cccl, the
# runtime headers and the host gcc, by version and build) for linux-64; with
# it, make ptx reproduces the committed headers byte for byte (versions.txt
# records the versions and the check). nvcc needs no GPU. Another nvcc, other
# runtime headers or another host compiler change kernels (MSVC as the host
# compiler regenerated 20 of 477 compute_75 kernels and all 38 compute_80
# ones), so PTX built any other way is not committed.
#
#   tools/ptx-toolchain/make-ptx.sh [ENV_DIR]
#
# creates the environment at ENV_DIR (default .ptx-toolchain in the checkout)
# from the lockfile with micromamba, mamba or conda, adds the nvvm link the
# packages leave out (nvcc.profile looks for cicc under targets/), and runs
# make ptx with it. The ptx-reproducible workflow runs it and fails on any
# byte difference. ccbuild-conda-explicit.txt and ccbuild-pip-freeze.txt
# record the environment the committed PTX was first built in.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
env=${1:-$root/.ptx-toolchain}
lock=$here/ptx-linux-64.lock
if [ ! -x "$env/bin/nvcc" ]; then
    if command -v micromamba >/dev/null 2>&1; then
        micromamba create -y -q -p "$env" --file "$lock"
    elif command -v mamba >/dev/null 2>&1; then
        mamba create -y -q -p "$env" --file "$lock"
    else
        conda create -y -q -p "$env" --file "$lock"
    fi
fi
[ -e "$env/targets/x86_64-linux/nvvm" ] || ln -s ../../nvvm "$env/targets/x86_64-linux/nvvm"
cd "$root"
PATH="$env/bin:/usr/bin:/bin" CUDA_VISIBLE_DEVICES=-1 \
    make ptx NVCC="$env/bin/nvcc" NVCC_CCBIN="$env/bin/x86_64-conda-linux-gnu-gcc"
