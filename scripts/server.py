import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description='Start local-only Whisper server; Ctrl+C to stop')
p.add_argument('--model', choices=['tiny', 'base', 'small'], default='base')
p.add_argument('--port', type=int, default=8080)
p.add_argument('--cpu', action='store_true')
args = p.parse_args()
source = ROOT / '.runtime' / 'whisper.cpp'
candidates = [source / 'build' / 'bin' / name for name in
              ['whisper-server', 'Release/whisper-server.exe', 'whisper-server.exe']]
exe = next((path for path in candidates if path.exists()), None)
model = source / 'models' / f'ggml-{args.model}.bin'
if exe is None or not model.exists():
    raise SystemExit(f'Run python scripts/setup.py --model {args.model} first.')
command = [str(exe), '-m', str(model), '-l', 'ko', '--host', '127.0.0.1',
           '--port', str(args.port), '-t', '4', '-bs', '1', '-bo', '1', '-nf', '-nc']
if args.cpu:
    command.append('-ng')
try:
    raise SystemExit(subprocess.call(command, cwd=source))
except KeyboardInterrupt:
    pass
