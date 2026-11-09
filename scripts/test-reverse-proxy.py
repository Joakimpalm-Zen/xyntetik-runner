#!/usr/bin/env python3
"""R7.7: run the shipped TLS/key proxy against a real loopback fixture server.

Default mode requires Docker Compose on Linux. --native-caddy runs the same
Caddyfile and startup guard directly for a lightweight local cross-check.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / 'examples/reverse-proxy'


def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def run(cmd, **kw):
    return subprocess.run([str(x) for x in cmd], check=True, capture_output=True,
                          text=True, encoding='utf-8', **kw)


def request(url, *, key=None, data=None, context=None, origin=None):
    headers = {'Content-Type': 'application/json'}
    if key is not None:
        headers['Authorization'] = 'Bearer ' + key
    if origin:
        headers['Origin'] = origin
    r = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(r, context=context, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as response:
        return response.code, response.read()


def loopback_listener(pid, expected_port):
    if sys.platform.startswith('linux'):
        found = []
        for name in ('tcp', 'tcp6'):
            for line in Path('/proc/net/' + name).read_text().splitlines()[1:]:
                fields = line.split()
                address, number = fields[1].split(':')
                if fields[3] == '0A' and int(number, 16) == expected_port:
                    found.append((name, address))
        assert found == [('tcp', '0100007F')], found
    elif sys.platform == 'darwin':
        lines = run(['lsof', '-nP', '-a', '-p', str(pid), '-iTCP', '-sTCP:LISTEN', '-Fn']).stdout.splitlines()
        addresses = [line[1:] for line in lines if line.startswith('n')]
        assert addresses == [f'127.0.0.1:{expected_port}'], addresses
    else:
        raise RuntimeError('listener check supports Linux or macOS')


def wait_for(url, proc=None, **kwargs):
    end = time.monotonic() + 30
    while time.monotonic() < end:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError('service exited before readiness')
        try:
            return request(url, **kwargs)
        except (OSError, urllib.error.URLError):
            time.sleep(.05)
    raise RuntimeError('service readiness timeout')


def stop(p):
    if p is None:
        return
    p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(timeout=5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runner', type=Path, default=ROOT / 'runner')
    ap.add_argument('--native-caddy', type=Path)
    args = ap.parse_args()
    assert (RECIPE / 'start.sh').is_file(), 'the proxy startup recipe is missing'
    runner = args.runner.resolve()
    assert runner.is_file(), 'build runner first'
    if not args.native_caddy and not sys.platform.startswith('linux'):
        raise RuntimeError('the Compose recipe test requires Linux host networking')
    if not args.native_caddy and not shutil.which('docker'):
        raise RuntimeError('Docker Compose is required; use --native-caddy only for a native cross-check')
    with tempfile.TemporaryDirectory(prefix='runner-proxy-test-') as tmp:
        box = Path(tmp)
        certs = box / 'certs'
        certs.mkdir()
        run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
             '-subj', '/CN=localhost', '-addext', 'subjectAltName=DNS:localhost,IP:127.0.0.1',
             '-keyout', certs / 'tls.key', '-out', certs / 'tls.crt'])
        fixture = box / 'fixture.gguf'
        run([sys.executable, ROOT / 'scripts/make-test-model.py', fixture])
        upstream, front = port(), port()
        while front == upstream:
            front = port()
        key = secrets.token_hex(32)
        env = dict(os.environ, HOME=str(box), XDG_CONFIG_HOME=str(box / 'config'),
                   XDG_DATA_HOME=str(box / 'data'), RUNNER_PROXY_API_KEY=key,
                   RUNNER_PORT=str(upstream), RUNNER_PROXY_HTTPS_PORT=str(front),
                   RUNNER_PROXY_HOST='localhost', RUNNER_PROXY_CERT_DIR=str(certs),
                   RUNNER_PROXY_UID=str(os.getuid()), RUNNER_PROXY_GID=str(os.getgid()))
        compose = ['docker', 'compose', '--project-name', 'runner-proxy-test-' + str(os.getpid()),
                   '--file', str(RECIPE / 'compose.yaml')]
        proc = proxy = None
        log = proxylog = None
        compose_used = False
        try:
            log = (box / 'runner.log').open('w')
            proc = subprocess.Popen([str(runner), '-m', str(fixture), '--serve', '--no-tray',
                                     '--gpu', 'off', '--draft', 'off', '-t', '1', '-c', '128',
                                     '--port', str(upstream)], cwd=ROOT, env=env,
                                    stdout=subprocess.DEVNULL, stderr=log)
            base = f'http://127.0.0.1:{upstream}'
            assert wait_for(base + '/health', proc)[0] == 200
            loopback_listener(proc.pid, upstream)
            if args.native_caddy:
                env.update(CADDY=str(args.native_caddy.resolve()), RUNNER_PROXY_CONFIG=str(RECIPE / 'Caddyfile'),
                           RUNNER_PROXY_TLS_CERT=str(certs / 'tls.crt'), RUNNER_PROXY_TLS_KEY=str(certs / 'tls.key'))
                bad = subprocess.run(['/bin/sh', str(RECIPE / 'start.sh')],
                                     env=dict(env, RUNNER_PROXY_API_KEY='bad'), capture_output=True)
                assert bad.returncode != 0 and b'64 hexadecimal' in bad.stderr, 'invalid key configuration must fail closed'
                proxylog = (box / 'proxy.log').open('w')
                proxy = subprocess.Popen(['/bin/sh', str(RECIPE / 'start.sh')], env=env,
                                         stdout=proxylog, stderr=proxylog)
            else:
                run(['docker', 'info'])
                run(compose + ['config', '--quiet'], env=env)
                compose_used = True
                bad = subprocess.run(compose + ['run', '--rm', '--no-deps', '-e',
                                     'RUNNER_PROXY_API_KEY=bad', 'proxy'], env=env, capture_output=True)
                assert bad.returncode != 0 and b'64 hexadecimal' in bad.stderr, 'invalid key configuration must fail closed'
                run(compose + ['up', '--detach', '--wait'], env=env)
            tls = ssl.create_default_context(cafile=str(certs / 'tls.crt'))
            public = f'https://localhost:{front}'
            assert wait_for(public + '/health', proxy, context=tls)[0] == 401
            assert request(public + '/health', key='0' * 64, context=tls)[0] == 401
            status, body = request(public + '/v1/models', key=key, context=tls,
                                   origin='https://client.example.test')
            assert status == 200 and json.loads(body)['data'], (status, body)
            payload = json.dumps({'prompt': 'x', 'max_tokens': 4, 'temperature': 0}).encode()
            assert request(public + '/v1/completions', data=payload, context=tls)[0] == 401
            status, body = request(public + '/v1/completions', key=key, data=payload, context=tls)
            assert status == 200 and json.loads(body)['choices'], (status, body)
            stream = json.loads(payload)
            stream['stream'] = True
            status, body = request(public + '/v1/completions', key=key,
                                   data=json.dumps(stream).encode(), context=tls)
            assert status == 200 and b'data: [DONE]' in body, (status, body)
            loopback_listener(proc.pid, upstream)
            assert request(base + '/health')[0] == 200  # auth stays at the proxy
            assert request(base + '/health', origin='https://client.example.test')[0] == 403
            print(('native' if args.native_caddy else 'compose') +
                  ': TLS verified; absent/wrong keys rejected; valid key reaches fixture and SSE; engine listener 127.0.0.1 only')
        except Exception:
            if compose_used:
                logs = subprocess.run(compose + ['logs', '--no-color'], env=env, capture_output=True, text=True)
                print((logs.stdout + logs.stderr).replace(key, '<redacted>'), file=sys.stderr)
            else:
                plog = box / 'proxy.log'
                if plog.exists():
                    print(plog.read_text().replace(key, '<redacted>'), file=sys.stderr)
            raise
        finally:
            if compose_used:
                subprocess.run(compose + ['down', '--volumes', '--remove-orphans'], env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            stop(proxy)
            stop(proc)
            for handle in (log, proxylog):
                if handle is not None:
                    handle.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
