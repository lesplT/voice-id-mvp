"""Local, durable recording journal. No automatic retention expiry."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from .config import get_settings


class JournalBusy(ValueError):
    pass


class JournalStore:
    def __init__(self, root=None):
        self.root = Path(root or get_settings().journal_path).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'audio').mkdir(exist_ok=True)
        self.db = self.root / 'journal.sqlite3'
        with self.connect() as connection:
            connection.execute('''CREATE TABLE IF NOT EXISTS recordings (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, operation TEXT NOT NULL,
                filename TEXT NOT NULL, bytes INTEGER NOT NULL, use_dictionary INTEGER NOT NULL,
                status TEXT NOT NULL, result TEXT, error TEXT, preview TEXT NOT NULL DEFAULT '',
                search TEXT NOT NULL, participants TEXT NOT NULL DEFAULT '[]')''')

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def audio_path(self, recording_id):
        if not re.fullmatch(r'[a-f0-9]{32}', recording_id):
            raise KeyError(recording_id)
        path = self.root / 'audio' / (recording_id + '.bin')
        if path.resolve().parent != (self.root / 'audio').resolve() or path.is_symlink():
            raise KeyError(recording_id)
        return path

    def create(self, operation, filename, data, use_dictionary=True):
        if operation not in {'dialogue', 'transcribe'} or not data:
            raise ValueError('Journal accepts nonempty dialogue/transcription recordings only')
        recording_id = uuid4().hex
        filename = Path(filename.replace('\\', '/')).name[:255] or 'audio'
        path = self.audio_path(recording_id)
        try:
            with path.open('xb') as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            with self.connect() as connection:
                connection.execute('INSERT INTO recordings (id,created_at,operation,filename,bytes,use_dictionary,status,search) VALUES (?,?,?,?,?,?,?,?)',
                    (recording_id, datetime.now(timezone.utc).isoformat(), operation, filename,
                     len(data), int(use_dictionary), 'queued', filename.casefold()))
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return self.get(recording_id)

    def get(self, recording_id):
        self.audio_path(recording_id)
        with self.connect() as connection:
            row = connection.execute('SELECT * FROM recordings WHERE id=?', (recording_id,)).fetchone()
        if row is None:
            raise KeyError(recording_id)
        entry = dict(row)
        entry['result'] = json.loads(entry['result']) if entry['result'] is not None else None
        entry['participants'] = json.loads(entry['participants'])
        entry['use_dictionary'] = bool(entry['use_dictionary'])
        entry.pop('search')
        return entry

    def update(self, recording_id, status, result=None, error=None):
        if status not in {'running', 'done', 'failed'}:
            raise ValueError('Invalid journal status')
        text = (result or {}).get('text', '')
        participants = sorted({item.get('speaker_id', 'UNKNOWN') for item in (result or {}).get('segments', [])})
        with self.connect() as connection:
            row = connection.execute('SELECT filename FROM recordings WHERE id=?', (recording_id,)).fetchone()
            if row is None:
                raise KeyError(recording_id)
            connection.execute('UPDATE recordings SET status=?,result=?,error=?,preview=?,search=?,participants=? WHERE id=?',
                (status, json.dumps(result, ensure_ascii=False) if result is not None else None,
                 error, text[:400], (row['filename'] + '\n' + text).casefold(),
                 json.dumps(participants, ensure_ascii=False), recording_id))

    def list(self, query='', limit=30, offset=0):
        with self.connect() as connection:
            count = connection.execute('SELECT COUNT(*), COALESCE(SUM(bytes),0) FROM recordings').fetchone()
            total = connection.execute('SELECT COUNT(*) FROM recordings WHERE instr(search,?)>0', (query.casefold(),)).fetchone()[0]
            rows = connection.execute('''SELECT id,created_at,operation,filename,bytes,status,error,preview,participants
                FROM recordings WHERE instr(search,?)>0 ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?''',
                (query.casefold(), limit, offset)).fetchall()
        entries = [dict(row) for row in rows]
        for entry in entries:
            entry['participants'] = json.loads(entry['participants'])
        return {'entries': entries, 'total': total, 'all_total': count[0], 'total_bytes': count[1]}

    def recover_interrupted(self):
        with self.connect() as connection:
            connection.execute("UPDATE recordings SET status='interrupted', error=? WHERE status IN ('queued','running')",
                ('Обработка прервана перезапуском. Исходная запись сохранена; для обработки загрузите её снова.',))

    def delete(self, recording_id):
        path = self.audio_path(recording_id)
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT status FROM recordings WHERE id=?', (recording_id,)).fetchone()
            if row is None:
                raise KeyError(recording_id)
            if row['status'] in {'queued','running'}:
                raise JournalBusy('Дождитесь окончания обработки перед удалением записи.')
            path.unlink(missing_ok=True)
            connection.execute('DELETE FROM recordings WHERE id=?', (recording_id,))


def audio_media_type(filename):
    return {'.mp3':'audio/mpeg', '.wav':'audio/wav', '.m4a':'audio/mp4', '.mp4':'audio/mp4',
            '.ogg':'audio/ogg', '.flac':'audio/flac', '.webm':'audio/webm', '.aac':'audio/aac'}.get(Path(filename).suffix.lower(), 'application/octet-stream')
