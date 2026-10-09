"""Download official pretrained weights, with checksum and atomic replace."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'models/ecapa/embedding_model.ckpt'
SHA256 = '0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2'
URL = 'https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt'

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

if DEST.exists():
    if digest(DEST) != SHA256:
        raise RuntimeError('Existing ECAPA checkpoint checksum differs; preserved, not overwritten')
else:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = DEST.with_suffix('.part')
    urllib.request.urlretrieve(URL, temporary)
    if digest(temporary) != SHA256:
        raise RuntimeError('Downloaded ECAPA checksum mismatch; checkpoint not installed')
    temporary.replace(DEST)
print(f'ECAPA checkpoint verified: {DEST}')
