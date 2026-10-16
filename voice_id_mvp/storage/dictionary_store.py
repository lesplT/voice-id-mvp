"""Local curated dictionary. Examples are labelled recordings, not model training."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import unicodedata
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator
from voice_id_mvp.core.config import get_settings


def word_key(text):
    """Purpose:
        Create a Unicode-normalized, case-insensitive dictionary lookup key.

    Ingress:
        A word, phrase, or search string.

    Egress:
        The stripped NFC-normalized casefolded string.
    """
    return unicodedata.normalize('NFC',text).strip().casefold()


class DictionaryError(ValueError):
    """Purpose:
        Represent dictionary conflicts or invalid storage operations.

    Ingress:
        A user-facing explanation of the dictionary error.

    Egress:
        An exception handled by dictionary HTTP routes.
    """
    pass


class WordInput(BaseModel):
    """Purpose:
        Validate editable dictionary entries before persistence.

    Ingress:
        A canonical word, aliases, optional category/note, and active flag.

    Egress:
        A validated dictionary payload with normalized words and aliases.
    """
    word: str = Field(min_length=1,max_length=120)
    category: str = Field(default='',max_length=50)
    description: str = Field(default='',max_length=1000)
    aliases: list[str] = Field(default_factory=list,max_length=30)
    active: bool = True

    @field_validator('word')
    @classmethod
    def clean_word(cls,value):
        """Purpose:
            Normalize a canonical dictionary word and require alphabetic content.

        Ingress:
            The proposed word or phrase.

        Egress:
            The normalized nonempty word; raises ValueError for invalid content.
        """
        value=unicodedata.normalize('NFC',value).strip()
        if not value or not any(c.isalpha() for c in value):
            raise ValueError('Введите слово или фразу.')
        return value

    @field_validator('aliases')
    @classmethod
    def clean_aliases(cls,values):
        """Purpose:
            Normalize and deduplicate explicit mistaken spellings.

        Ingress:
            A list of alias strings supplied with a dictionary entry.

        Egress:
            Distinct validated aliases; raises ValueError for oversized or nonalphabetic variants.
        """
        result=[]
        for value in values:
            value=unicodedata.normalize('NFC',value).strip()
            if len(value)>120:
                raise ValueError('Вариант ошибки должен быть короче 120 символов.')
            if value and not any(c.isalpha() for c in value):
                raise ValueError('Вариант ошибки должен содержать буквы.')
            if value and word_key(value) not in {word_key(v) for v in result}:
                result.append(value)
        return result


class DictionaryStore:
    """Purpose:
        Persist curated words and labelled audio examples without training models.

    Ingress:
        An optional storage directory or the configured dictionary path.

    Egress:
        SQLite-backed entry operations and paths to retained WAV examples.
    """
    def __init__(self,folder: Path | None=None):
        """Purpose:
            Initialize the dictionary directories and schema without replacing existing data.

        Ingress:
            An optional dictionary storage directory.

        Egress:
            An initialized store; missing directories and tables are created.
        """
        self.folder = Path(folder or get_settings().dictionary_path)
        self.folder.mkdir(parents=True,exist_ok=True)
        self.audio_folder=self.folder/'audio'
        self.audio_folder.mkdir(exist_ok=True)
        self.path=self.folder/'dictionary.sqlite3'
        with self._db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS words(
                    id TEXT PRIMARY KEY, word TEXT NOT NULL, word_key TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL, description TEXT NOT NULL, aliases TEXT NOT NULL,
                    active INTEGER NOT NULL, archived INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS examples(
                    id TEXT PRIMARY KEY, word_id TEXT NOT NULL REFERENCES words(id),
                    filename TEXT NOT NULL, transcript TEXT NOT NULL, duration REAL NOT NULL,
                    created_at TEXT NOT NULL);
            ''')

    @contextmanager
    def _db(self):
        """Purpose:
            Provide a transactional SQLite connection with foreign-key enforcement.

        Ingress:
            The store's configured database path.

        Egress:
            Yields a row-based connection and always closes it on context exit.
        """
        db=sqlite3.connect(self.path,timeout=30)
        db.row_factory=sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _decode(row):
        """Purpose:
            Convert a SQLite dictionary row into application-level field types.

        Ingress:
            A dictionary row containing JSON aliases and integer flags.

        Egress:
            A dictionary with decoded aliases and Boolean flags.
        """
        result=dict(row)
        result['aliases']=json.loads(result['aliases'])
        result['active']=bool(result['active'])
        result['archived']=bool(result['archived'])
        return result

    def list(self,query=''):
        """Purpose:
            List nonarchived words and their examples, optionally filtering by text.

        Ingress:
            An optional normalized search query.

        Egress:
            Dictionary entry dictionaries, ordered by canonical lookup key.
        """
        with self._db() as db:
            rows=db.execute('SELECT * FROM words WHERE archived=0 ORDER BY word_key').fetchall()
            result=[]
            for row in rows:
                item=self._decode(row)
                if query and word_key(query) not in word_key(item['word']+' '+item['category']+' '+item['description']):
                    continue
                item['examples']=[dict(e) for e in db.execute('SELECT * FROM examples WHERE word_id=? ORDER BY created_at',(item['id'],))]
                result.append(item)
            return result

    def get(self,word_id):
        """Purpose:
            Read one nonarchived dictionary entry and its audio metadata.

        Ingress:
            A word identifier.

        Egress:
            The entry dictionary; raises KeyError when the word is absent or archived.
        """
        with self._db() as db:
            row=db.execute('SELECT * FROM words WHERE id=? AND archived=0',(word_id,)).fetchone()
            if row is None:
                raise KeyError(word_id)
            result=self._decode(row)
            result['examples']=[dict(e) for e in db.execute('SELECT * FROM examples WHERE word_id=? ORDER BY created_at',(word_id,))]
            return result

    def save(self,entry: WordInput,word_id=None):
        """Purpose:
            Create or update a word while rejecting conflicting active words and aliases.

        Ingress:
            Validated WordInput and an optional existing word identifier.

        Egress:
            The persisted entry; raises DictionaryError on conflicting names or aliases.
        """
        now=datetime.now(timezone.utc).isoformat()
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            if word_id and db.execute('SELECT id FROM words WHERE id=? AND archived=0',(word_id,)).fetchone() is None:
                raise KeyError(word_id)
            rows=db.execute('SELECT * FROM words WHERE archived=0 AND active=1').fetchall()
            own_keys={word_key(entry.word),*(word_key(a) for a in entry.aliases)}
            if entry.active:
                for row in rows:
                    if row['id']==word_id:
                        continue
                    other_keys={row['word_key'],*(word_key(a) for a in json.loads(row['aliases']))}
                    if own_keys & other_keys:
                        raise DictionaryError('Слово или вариант ошибки уже используется в другой активной карточке.')
            values=(entry.word,word_key(entry.word),entry.category.strip(),entry.description.strip(),json.dumps(entry.aliases,ensure_ascii=False),int(entry.active))
            try:
                if word_id:
                    db.execute('UPDATE words SET word=?,word_key=?,category=?,description=?,aliases=?,active=?,updated_at=? WHERE id=?',(*values,now,word_id))
                else:
                    word_id=uuid4().hex
                    db.execute('INSERT INTO words(id,word,word_key,category,description,aliases,active,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)',(word_id,*values,now,now))
            except sqlite3.IntegrityError as exc:
                raise DictionaryError('Такое слово уже есть в словаре или архиве. Измените существующую карточку.') from exc
        return self.get(word_id)

    def archive(self,word_id):
        """Purpose:
            Hide a word and disable its corrections without deleting audio examples.

        Ingress:
            An existing nonarchived word identifier.

        Egress:
            None; the word is archived and its files remain on disk.
        """
        self.get(word_id)
        with self._db() as db:
            db.execute('UPDATE words SET archived=1,active=0 WHERE id=?',(word_id,))

    def add_example(self,word_id,wav_path: Path,filename,transcript,duration):
        """Purpose:
            Attach a labelled WAV example to an existing dictionary word.

        Ingress:
            Word ID, WAV path, original filename, transcript, and duration.

        Egress:
            The updated entry; raises on an archived word or the per-word example limit.
        """
        self.get(word_id)
        example_id=uuid4().hex
        destination=self.audio_folder/(example_id+'.wav')
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT archived FROM words WHERE id=?',(word_id,)).fetchone()['archived']:
                raise KeyError(word_id)
            if db.execute('SELECT COUNT(*) FROM examples WHERE word_id=?',(word_id,)).fetchone()[0]>=20:
                raise DictionaryError('Максимум 20 примеров на слово.')
            import shutil
            shutil.copyfile(wav_path,destination)
            db.execute('INSERT INTO examples VALUES(?,?,?,?,?,?)',(example_id,word_id,Path(filename).name[:200],transcript.strip(),duration,datetime.now(timezone.utc).isoformat()))
        return self.get(word_id)

    def example_path(self,example_id):
        """Purpose:
            Resolve a retained example only when its parent word is visible.

        Ingress:
            A hexadecimal example identifier.

        Egress:
            The existing WAV path; raises KeyError for invalid, absent, or archived examples.
        """
        if len(example_id)!=32 or any(c not in '0123456789abcdef' for c in example_id):
            raise KeyError(example_id)
        with self._db() as db:
            row=db.execute('SELECT e.id FROM examples e JOIN words w ON w.id=e.word_id WHERE e.id=? AND w.archived=0',(example_id,)).fetchone()
        path=self.audio_folder/(example_id+'.wav')
        if row is None or not path.is_file():
            raise KeyError(example_id)
        return path
