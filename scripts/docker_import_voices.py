"""Import private points from stdin into an empty Docker collection only."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voice_id_mvp.repository import VoiceRepository
from qdrant_client.http import models

data = json.load(sys.stdin)
repository = VoiceRepository()
if data['collection'] != repository.settings.qdrant_collection:
    raise RuntimeError('Collection names differ; no data imported')
repository.ensure_collection()
with repository._client() as client:
    if client.count(repository.settings.qdrant_collection, exact=True).count:
        raise RuntimeError('Target collection is not empty; existing profiles preserved')
    points = [models.PointStruct(**p) for p in data['points']]
    if points:
        client.upsert(repository.settings.qdrant_collection, points, wait=True)
print(f'Imported {len(points)} private voice points into Docker; native data unchanged.')
