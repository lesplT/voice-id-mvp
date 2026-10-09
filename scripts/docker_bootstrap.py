"""Prepare pretrained weights and wait for Qdrant; no training, no personal data."""
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def unpack_model(name, url):
    target = ROOT / 'models' / name
    if target.is_dir():
        return
    with tempfile.TemporaryDirectory(dir=ROOT / 'models', prefix='download-') as temporary:
        archive = Path(temporary) / 'model.zip'
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as bundle:
            for item in bundle.infolist():
                resolved = (Path(temporary) / item.filename).resolve()
                if not resolved.is_relative_to(Path(temporary).resolve()):
                    raise RuntimeError('Unsafe model archive path')
            bundle.extractall(temporary)
        extracted = Path(temporary) / name
        if not extracted.is_dir():
            raise RuntimeError('Expected model folder is missing')
        extracted.rename(target)


def main():
    (ROOT / 'models').mkdir(exist_ok=True)
    for name in ('vosk-model-spk-0.4', 'vosk-model-small-ru-0.22'):
        print('Preparing pretrained model:', name, flush=True)
        unpack_model(name, f'https://alphacephei.com/vosk/models/{name}.zip')
    runpy.run_path(str(ROOT / 'scripts/download_ecapa.py'), run_name='__main__')
    from voice_id_mvp.asr import GigaAmAsr
    print('Preparing GigaAM v3 CPU (first download can take several minutes)', flush=True)
    GigaAmAsr._get_model()
    endpoint = os.environ['QDRANT_URL'].rstrip('/') + '/readyz'
    for _ in range(60):
        try:
            with urllib.request.urlopen(endpoint, timeout=3) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(2)
    else:
        raise RuntimeError('Qdrant did not become ready')
    from voice_id_mvp.repository import VoiceRepository
    VoiceRepository().ensure_collection()
    print('Pretrained models verified; Qdrant ready', flush=True)


if __name__ == '__main__':
    main()
