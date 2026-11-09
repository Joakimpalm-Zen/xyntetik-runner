#!/usr/bin/env python3
"""Select extra CI checks from NUL-separated diff paths.

Sources and gate/fixture helpers are broad by design: a newly added C gate
or a new shared source must not need a second hand-maintained target roster.
Pure prose, site pages, the Python client and Python-only tests stay fast.
"""
import argparse
import os
from pathlib import PurePosixPath
import sys

CONTROL = {'Makefile', 'MANUAL.md', 'AGENTS.md', 'CLAUDE.md',
           'docs/model-tiers.json', 'tests/test_ci_contracts.py'}
PROXY_INPUTS = {'examples/reverse-proxy/compose.yaml', 'examples/reverse-proxy/Caddyfile',
                'examples/reverse-proxy/start.sh', 'docs/reverse-proxy.md',
                'scripts/test-reverse-proxy.py'}
C_INPUTS = {'.c', '.h', '.m', '.metal', '.cc', '.cpp', '.rc', '.manifest', '.inc'}


def needs_core(path):
    return (path in CONTROL
            or path.startswith(('src/', 'scripts/', '.github/workflows/', 'tests/fixtures/'))
            or (path.startswith('tests/') and PurePosixPath(path).suffix in C_INPUTS))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scope', choices=('gates-core', 'reverse-proxy'), default='gates-core')
    ap.add_argument('--nul-stdin', action='store_true', help='read git diff --name-only -z on stdin')
    ap.add_argument('paths', nargs='*')
    args = ap.parse_args()
    paths = args.paths
    if args.nul_stdin:
        paths += [os.fsdecode(p) for p in sys.stdin.buffer.read().split(b'\0') if p]
    selected = any(needs_core(p) for p in paths) if args.scope == 'gates-core' else any(p in PROXY_INPUTS for p in paths)
    print('true' if selected else 'false')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
