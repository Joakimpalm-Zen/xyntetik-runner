#!/usr/bin/env python3
"""Reproduce build probes in two disposable, clean tag checkouts; CPU only.

Example: python3 reproduce.py --source150 /tmp/src150 --source160 /tmp/src160
  --clang /path/to/clang --qwen /models/Qwen3-0.6B-bf16.gguf
  --trinity /models/Trinity-Nano-Preview-Q8_0.gguf
  --corpus /src/tests/fixtures/gold-corpus-2k.txt --output /tmp/new-results

This takes the first 6000 corpus bytes, matching the recorded probes. Both
source directories must be clean checkouts of the named tags. Output must
not exist. The original Conda-config comparison is optional.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
for k in ('source150', 'source160', 'clang', 'qwen', 'trinity', 'corpus', 'output'):
    p.add_argument('--' + k, required=True)
p.add_argument('--gcc', default='/usr/bin/gcc')
p.add_argument('--conda-clang-binary')
a = p.parse_args()
out = Path(a.output).resolve()
out.mkdir()  # fail before overwriting existing evidence
corpus = out / 'corpus-6k.txt'
corpus.write_bytes(Path(a.corpus).read_bytes()[:6000])
flags = '-O3 -ffast-math -std=gnu11 -Wall -Wextra -Wno-unused-parameter -march='
cc = a.clang + ' --no-default-config --sysroot=/'
configs = [('g150-v3', a.source150, a.gcc, 'x86-64-v3'),
           ('g160-v3', a.source160, a.gcc, 'x86-64-v3'),
           ('c150-v3', a.source150, cc, 'x86-64-v3'),
           ('c150-native', a.source150, cc, 'native'),
           ('g150-native', a.source150, a.gcc, 'native')]
manifest = {}
for n, src, compiler, arch in configs:
    expected = 'f913dcaf037a287e161621021f95915ba1ae2e04' if '150' in n else '869d72e89a74e2d147ceadf5400e10d9fe9e877c'
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=src, text=True).strip()
    assert actual == expected, (src, actual, expected)
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=src)
    command = ['make', '-j8', 'runner', 'CC=' + compiler, 'CFLAGS=' + flags + arch, 'PYTHON=python3']
    with (out / (n + '-build.log')).open('w') as log:
        # make does not track compiler/flag changes; each probe needs fresh objects.
        subprocess.run(['make', 'clean'], cwd=src, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(command, cwd=src, stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copy2(Path(src) / 'runner', out / n)
    manifest[n] = {'source': actual, 'command': command,
                   'sha256': hashlib.sha256((out / n).read_bytes()).hexdigest()}
(out / 'builds.json').write_text(json.dumps(manifest, indent=2) + '\n')
binaries = {n: str(out / n) for n, *_ in configs}
if a.conda_clang_binary:
    binaries['c150-conda-native'] = a.conda_clang_binary
for model, path in [('qwen3-0.6b-bf16', a.qwen), ('trinity-served', a.trinity)]:
    for name, binary in binaries.items():
        target = out / f'score-{model}-{name}.json'
        with target.open('w') as log, target.with_suffix('.err').open('w') as err:
            subprocess.run([binary, '-m', path, '--gpu', 'off', '-t', '16', '-c', '4096',
                            '--score', '--score-top', '8', '-f', str(corpus)], stdout=log, stderr=err,
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
        json.loads(target.read_text())
        print(model, name, hashlib.sha256(target.read_bytes()).hexdigest(), flush=True)
