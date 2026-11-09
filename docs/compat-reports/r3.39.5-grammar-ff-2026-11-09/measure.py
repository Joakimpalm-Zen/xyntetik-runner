import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

exe, model, destination = map(pathlib.Path, sys.argv[1:])
destination.mkdir(parents=True, exist_ok=True)
schemas = {
    'object': {'type': 'object', 'properties': {
        'first_field': {'type': 'string', 'enum': ['aa', 'ab']},
        'second_field': {'type': 'integer'}}, 'required': ['first_field', 'second_field']},
    'pinned': {'type': 'object', 'properties': {'answer': {'type': 'string', 'enum': [
        'A fixed long answer with no choice of words or spelling.']}}, 'required': ['answer']},
}
summary = []
deadline = time.monotonic() + 180
for backend in ('cpu', 'metal'):
    for case, schema in schemas.items():
        sf = destination / f'{case}.schema.json'
        sf.write_text(json.dumps(schema))
        expected = None
        for round_no in range(3):
            for enabled in ((0, 1) if round_no % 2 == 0 else (1, 0)):
                if time.monotonic() > deadline:
                    raise RuntimeError('180 second bound reached')
                name = f'{backend}-{case}-{round_no}-{enabled}'
                record = destination / f'{name}.json'
                env = dict(os.environ, RUNNER_GRAMMAR_FF=str(enabled), RUNNER_SPEC_STATS='1')
                proc = subprocess.run([str(exe.resolve()), '-m', str(model), '-p',
                    'Return an object matching the schema.', '--json-schema', str(sf),
                    '-n', '96', '-c', '256', '-b', '32', '-t', '2', '--temp', '0',
                    '--repeat-penalty', '1', '-s', '1', '--record', str(record),
                    '--gpu', 'off' if backend == 'cpu' else 'auto', '--no-tray'],
                    env=env, capture_output=True, timeout=min(40, max(0.1, deadline-time.monotonic())))
                (destination / f'{name}.stderr').write_bytes(proc.stderr)
                assert proc.returncode == 0, proc.stderr[-2000:]
                doc = json.loads(record.read_bytes())
                actual = (doc['output']['tokens'], doc['output']['bytes_hex'])
                if expected is None:
                    expected = actual
                assert actual == expected, name
                assert doc['profile']['gpu'] == (backend == 'metal'), doc['profile']
                parsed = json.loads(bytes.fromhex(doc['output']['bytes_hex']))
                assert set(schema['required']).issubset(parsed)
                match = re.search(rb'gen: (\d+) tok, ([0-9.]+) tok/s', proc.stderr)
                assert match, proc.stderr[-1000:]
                spec = re.search(rb'grammar (\d+)/(\d+)', proc.stderr)
                row = {'backend': backend, 'case': case, 'round': round_no, 'ff': enabled,
                    'tokens': int(match[1]), 'tok_s': float(match[2]),
                    'accepted': int(spec[1]) if spec else 0,
                    'drafted': int(spec[2]) if spec else 0,
                    'output_sha256': hashlib.sha256(bytes.fromhex(actual[1])).hexdigest(),
                    'token_sha256': hashlib.sha256(json.dumps(actual[0]).encode()).hexdigest()}
                summary.append(row)
                (destination / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
                print(json.dumps(row), flush=True)
