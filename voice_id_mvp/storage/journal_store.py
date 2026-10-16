"""Local, durable recording journal. No automatic retention expiry."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from voice_id_mvp.core.config import get_settings


class JournalBusy(ValueError):
    """Purpose:
        Reject deletion of a recording that is still queued or processing.

    Ingress:
        A human-readable conflict description.

    Egress:
        An exception translated into an HTTP conflict response.
    """
    pass


class JournalStore:
    """Purpose:
        Persist original recordings and results until explicit manual deletion.

    Ingress:
        An optional journal directory or configured journal path.

    Egress:
        SQLite metadata and UUID-confined original audio files; there is no automatic expiry.
    """
    def __init__(self, root=None):
        """Purpose:
            Initialize journal directories and schema without replacing saved recordings.

        Ingress:
            An optional journal root directory.

        Egress:
            A store ready to read or persist recordings.
        """
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
        """Purpose:
            Provide a committed-or-rolled-back SQLite transaction and close the connection.

        Ingress:
            The journal database path held by the store.

        Egress:
            Yields a row-based connection; exceptions roll back the transaction.
        """
        connection = sqlite3.connect(self.db, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def audio_path(self, recording_id):
        """Purpose:
            Validate a recording ID and confine its audio path to the journal directory.

        Ingress:
            A lowercase 32-character hexadecimal recording identifier.

        Egress:
            A safe .bin audio path; raises KeyError for invalid IDs or symbolic links.
        """
        if not re.fullmatch(r'[a-f0-9]{32}', recording_id):
            raise KeyError(recording_id)
        path = self.root / 'audio' / (recording_id + '.bin')
        if path.resolve().parent != (self.root / 'audio').resolve() or path.is_symlink():
            raise KeyError(recording_id)
        return path

    def create(self, operation, filename, data, use_dictionary=True):
        """Purpose:
            Durably save original audio before queuing recognition.

        Ingress:
            Operation, original filename, nonempty audio bytes, and dictionary preference.

        Egress:
            The new journal entry; file persistence precedes metadata insertion.
        """
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
        """Purpose:
            Read a saved recording and decode its result and participant metadata.

        Ingress:
            A recording identifier.

        Egress:
            The journal entry dictionary; raises KeyError if absent or invalid.
        """
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
        """Purpose:
            Persist processing status, result, preview, search text, and participants.

        Ingress:
            Recording ID, allowed processing status, optional result, and optional error.

        Egress:
            None; the existing row is updated without removing its original audio.
        """
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
        """Purpose:
            Search and paginate saved recordings while reporting total audio storage.

        Ingress:
            Optional query text, page limit, and offset.

        Egress:
            Entry summaries, matching/overall counts, and total retained audio bytes.
        """
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
        """Purpose:
            Mark unfinished jobs interrupted after a workspace process restart.

        Ingress:
            Existing queued/running journal rows.

        Egress:
            None; audio is retained and interrupted rows receive a recovery explanation.
        """
        with self.connect() as connection:
            connection.execute("UPDATE recordings SET status='interrupted', error=? WHERE status IN ('queued','running')",
                ('Обработка прервана перезапуском. Исходная запись сохранена; для обработки загрузите её снова.',))

    def delete(self, recording_id):
        """Purpose:
            Delete one completed/interrupted recording and its original audio explicitly.

        Ingress:
            An existing recording identifier.

        Egress:
            None; raises JournalBusy for active work and KeyError for absent entries.
        """
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
    """Purpose:
        Select a safe audio MIME type without treating uploaded content as HTML.

    Ingress:
        An original audio filename.

    Egress:
        A supported audio MIME type or application/octet-stream.
    """
    return {'.mp3':'audio/mpeg', '.wav':'audio/wav', '.m4a':'audio/mp4', '.mp4':'audio/mp4',
            '.ogg':'audio/ogg', '.flac':'audio/flac', '.webm':'audio/webm', '.aac':'audio/aac'}.get(Path(filename).suffix.lower(), 'application/octet-stream')
