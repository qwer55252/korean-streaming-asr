"""Run with the project virtual environment's Python. No administrator access needed."""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = '4979e04f5dcaccb36057e059bbaed8a2f5288315'  # whisper.cpp v1.8.2

HASHES = {
    'tiny': 'bd577a113a864445d4c299885e0cb97d4ba92b5f',
    'base': '465707469ff3a37a2b9b8d8f89f2f99de7299dac',
    'small': '55356645c2b361a969dfd0ef2c5a50d530afd8d5',
}

def verify(path, model):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha1').hexdigest()
    if digest != HASHES[model]:
        raise SystemExit(f'Model checksum mismatch: {path}. Remove it and retry.')

def run(*args):
    subprocess.run([str(a) for a in args], check=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['tiny', 'base', 'small'], default='base')
    args = p.parse_args()
    source = ROOT / '.runtime' / 'whisper.cpp'
    source.parent.mkdir(exist_ok=True)
    if not source.exists():
        run('git', 'clone', '--depth', '1', '--branch', 'v1.8.2',
            'https://github.com/ggml-org/whisper.cpp.git', source)
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != REVISION:
        raise SystemExit('Unexpected whisper.cpp revision; remove .runtime/whisper.cpp and retry.')
    cmake = Path(sys.executable).parent / ('cmake.exe' if os.name == 'nt' else 'cmake')
    config = ['-DWHISPER_BUILD_SERVER=ON', '-DWHISPER_BUILD_TESTS=OFF', '-DCMAKE_BUILD_TYPE=Release']
    if platform.system() == 'Darwin':
        sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
        config += ['-DGGML_METAL=ON', f'-DCMAKE_OSX_SYSROOT={sdk}']
        # Some Command Line Tools installations omit libc++ from clang's search path.
        probe = subprocess.run(['c++', '-isysroot', sdk, '-x', 'c++', '-fsyntax-only', '-'],
                               input='#include <cstdio>\n', text=True, capture_output=True)
        if probe.returncode and (Path(sdk) / 'usr/include/c++/v1/cstdio').exists():
            config.append(f'-DCMAKE_CXX_FLAGS=-isystem {sdk}/usr/include/c++/v1')
    run(cmake, '-S', source, '-B', source / 'build', *config)
    run(cmake, '--build', source / 'build', '--config', 'Release', '-j', '4')
    model = source / 'models' / f'ggml-{args.model}.bin'
    if not model.exists():
        temporary = model.with_suffix('.part')
        print(f'Downloading {args.model} model...', flush=True)
        urllib.request.urlretrieve(
            f'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{model.name}', temporary)
        verify(temporary, args.model)
        temporary.replace(model)
    verify(model, args.model)
    print(f'Ready: {model}\nStart: python scripts/server.py --model {args.model}')

if __name__ == '__main__':
    main()
