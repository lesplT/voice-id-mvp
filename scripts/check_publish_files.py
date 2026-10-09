"""Fail closed if git's proposed upload includes private runtime data or tokens."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
for name in filter(None, paths):
    path = Path(name)
    if path.parts[0] in {'data','models','.venv','.supergoal'} or path.name.startswith('.env') and path.name != '.env.example':
        raise SystemExit(f'Private file detected: {name}')
    if path.suffix.lower() in {'.wav','.mp3','.m4a','.ckpt','.bin','.sqlite3','.zip','.log'}:
        raise SystemExit(f'Private or binary artifact detected: {name}')
    content = (ROOT / path).read_text(encoding='utf-8')
    if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)', content):
        raise SystemExit(f'Possible credential detected: {name}')
print(f'Publish manifest checked: {len(list(filter(None,paths)))} text files, no runtime recordings/models/.env/tokens')
