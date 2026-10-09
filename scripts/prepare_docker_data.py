"""Copy native private data to an EMPTY Docker workspace; never overwrite either."""
import json
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    target = ROOT / 'data/docker/workspace'
    export = ROOT / 'data/docker/voice-migration.json'
    if target.exists() and any(target.iterdir()):
        raise RuntimeError('Docker data already exists. Nothing was overwritten. Migrate before first startup.')
    if export.exists():
        raise RuntimeError('Private voice export already exists. Preserved; inspect before retrying.')
    target.parent.mkdir(parents=True, exist_ok=True)
    from voice_id_mvp.repository import VoiceRepository
    repository = VoiceRepository()
    # Explicitly read the native embedded store, not the new Docker server.
    repository.settings = repository.settings.model_copy(update={'qdrant_url':'local'})
    points = []
    with repository._client() as client:
        if client.collection_exists(repository.settings.qdrant_collection):
            offset = None
            while True:
                batch, offset = client.scroll(repository.settings.qdrant_collection,
                    limit=256, offset=offset, with_vectors=True, with_payload=True)
                points.extend({'id':str(p.id), 'vector':p.vector, 'payload':p.payload} for p in batch)
                if offset is None:
                    break
    with tempfile.TemporaryDirectory(dir=target.parent, prefix='migration-') as temporary:
        staging = Path(temporary) / 'workspace'
        staging.mkdir()
        for name in ('journal', 'dictionary'):
            source = ROOT / 'data' / name
            if not source.exists():
                continue
            destination = staging / name
            destination.mkdir()
            # Snapshot SQLite rather than copying an open DB file.
            databases = list(source.glob('*.sqlite3')) + list(source.glob('*.db'))
            for db in databases:
                with closing(sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True)) as connection:
                    if name == 'journal' and connection.execute("SELECT COUNT(*) FROM recordings WHERE status IN ('queued','running')").fetchone()[0]:
                        raise RuntimeError('Wait for native recording jobs to finish before migration.')
                    with closing(sqlite3.connect(destination / db.name)) as backup:
                        connection.backup(backup)
            for path in source.iterdir():
                if path.is_dir():
                    shutil.copytree(path, destination / path.name)
        if target.exists():
            # Only the verified empty target directory can be removed.
            target.rmdir()
        staging.rename(target)
    with export.open('x', encoding='utf-8') as stream:
        json.dump({'collection':repository.settings.qdrant_collection, 'points':points}, stream, ensure_ascii=False)
    print(f'Private local snapshot created; exported {len(points)} voice vectors. No original data changed.')
    print('Start Docker, then pipe data/docker/voice-migration.json to scripts/docker_import_voices.py inside enrollment.')


if __name__ == '__main__':
    main()
