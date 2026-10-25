"""The release gate accepts the old ABI and rejects a newer imported version."""
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform != 'linux' or not shutil.which('gcc')
                                or not shutil.which('objdump'), reason='Linux ELF tools required')


@pytest.mark.parametrize('version,limit,ok', [('2.38', '2.38', True),
                                           ('2.43', '2.38', False),
                                           ('2.9', '2.38', True),
                                           ('2.3.4', '2.38', True),
                                           ('PRIVATE', '2.38', False)])
def test_imported_version(tmp_path, version, limit, ok):
    # An actual ELF import carrying a linker version, not a string planted
    # in the executable. The version is an external linker specification.
    (tmp_path / 'lib.c').write_text('int abi_probe(void) { return 0; }\n')
    (tmp_path / 'main.c').write_text('extern int abi_probe(void); int main(void) { return abi_probe(); }\n')
    (tmp_path / 'versions').write_text(f'GLIBC_{version} {{ global: abi_probe; }};\n')
    subprocess.run(['gcc', '-shared', '-fPIC', 'lib.c', '-Wl,--version-script=versions',
                    '-o', 'libprobe.so'], cwd=tmp_path, check=True)
    subprocess.run(['gcc', 'main.c', '-L.', '-lprobe', '-o', 'probe'], cwd=tmp_path, check=True)
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/check-glibc.py'),
                             str(tmp_path / 'probe'), '--max-version', limit], capture_output=True, text=True)
    assert (result.returncode == 0) == ok, result.stdout + result.stderr


def test_invalid_binary_fails_closed(tmp_path):
    bad = tmp_path / 'not-elf'
    bad.write_text('GLIBC_2.38\n')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/check-glibc.py'), str(bad)],
                            capture_output=True, text=True)
    assert result.returncode != 0
