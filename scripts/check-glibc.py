#!/usr/bin/env python3
"""Refuse a Linux release whose imported GLIBC versions exceed its ABI floor."""
import argparse
import re
import subprocess
import sys


def check(binary, maximum):
    result = subprocess.run(['objdump', '-T', binary], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise ValueError(result.stderr.strip() or 'objdump failed')
    imports = [line for line in result.stdout.splitlines() if '*UND*' in line]
    versions = {tuple(map(int, v.split('.'))) for line in imports
                for v in re.findall(r'\bGLIBC_([0-9]+(?:\.[0-9]+)+)\b', line)}
    if not versions:
        raise ValueError('no imported GLIBC versions found; expected a dynamic Linux ELF')
    if any('GLIBC_PRIVATE' in line for line in imports):
        raise ValueError('imports GLIBC_PRIVATE')
    limit = tuple(map(int, maximum.split('.')))
    highest = max(versions)
    print(f'{binary}: highest imported GLIBC_{".".join(map(str, highest))}; limit GLIBC_{maximum}')
    bad = [line for line in imports if any(tuple(map(int, v.split('.'))) > limit
           for v in re.findall(r'\bGLIBC_([0-9]+(?:\.[0-9]+)+)\b', line))]
    if bad:
        raise ValueError('GLIBC compatibility regression:\n' + '\n'.join(bad))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary')
    p.add_argument('--max-version', default='2.38', help='previous Linux release ABI floor (default: 2.38)')
    args = p.parse_args()
    try:
        check(args.binary, args.max_version)
    except (OSError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
