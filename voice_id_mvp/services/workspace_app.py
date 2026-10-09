from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ..config import ROOT, get_settings
from ..dictionary import DictionaryStore, DictionaryError, WordInput
from ..audio import normalized_upload, wav_duration, AudioError
from ..segmentation import has_speech
from ..dictionary_corrections import apply_dictionary_result
from ..workspace_jobs import JobManager, QueueFull
from ..journal import JournalStore, JournalBusy, audio_media_type

WEB = ROOT / 'voice_id_mvp/web'
MAX_UPLOAD = 100 * 1024 * 1024


def dictionary_store():
    return DictionaryStore()


def journal_store():
    return JournalStore()


@asynccontextmanager
async def lifespan(app):
    journal_store().recover_interrupted()
    app.state.jobs = JobManager()
    yield
    app.state.jobs.shutdown()


app = FastAPI(title='Voice ID — единый интерфейс', lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1','localhost','testserver'])


@app.middleware('http')
async def local_origin(request, call_next):
    from fastapi.responses import JSONResponse
    origin = request.headers.get('origin')
    if request.method not in {'GET','HEAD','OPTIONS'} and origin and origin != str(request.base_url).rstrip('/'):
        return JSONResponse({'detail':'Запрос разрешён только из локального интерфейса.'},status_code=403)
    length = request.headers.get('content-length','')
    if length.isdigit() and int(length) > MAX_UPLOAD + 2*1024*1024:
        return JSONResponse({'detail':'Файл слишком большой. Максимум — 100 МБ.'},status_code=413)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'same-origin'
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'"
    return response


@app.get('/health')
def health():
    return {'status':'ok','service':'workspace'}


@app.get('/')
def home():
    return FileResponse(WEB/'index.html')


async def limited_upload(audio: UploadFile, limit=None):
    limit = MAX_UPLOAD if limit is None else limit
    data = bytearray()
    while chunk := await audio.read(1024*1024):
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(413, 'Файл слишком большой. Максимум — '+str(limit//(1024*1024))+' МБ.')
    if not data:
        raise HTTPException(400,'Выберите непустой аудиофайл.')
    return bytes(data)


def service_request(method, port, path, **kwargs):
    try:
        with httpx.Client(timeout=600,trust_env=False) as client:
            settings = get_settings()
            base = {8001:settings.enrollment_service_url, 8002:settings.identification_service_url,
                    8003:settings.transcription_service_url}[port]
            response = client.request(method,base.rstrip('/')+path,**kwargs)
        if response.status_code >= 400:
            if response.status_code == 400:
                raise RuntimeError('Аудиофайл не удалось прочитать. Проверьте запись или выберите другой файл.')
            try:
                detail = response.json().get('detail','Запрос не выполнен')
            except ValueError:
                detail = 'Сервис вернул ошибку'
            raise RuntimeError(f'Не удалось обработать запись: {detail}')
        return response.json()
    except httpx.ConnectError as exc:
        raise RuntimeError('Сервис обработки не запущен. Запустите scripts/start_all.ps1 и повторите.') from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError('Обработка заняла слишком много времени. Попробуйте более короткую запись.') from exc


@app.post('/api/jobs',status_code=202)
async def create_job(operation: Literal['dialogue','transcribe','identify','enroll'] = Form(...),
                     audio: UploadFile = File(...), speaker_id: str = Form(''), use_dictionary: bool = Form(True)):
    if operation == 'enroll' and (not speaker_id.strip() or len(speaker_id) > 80 or '/' in speaker_id or '\\' in speaker_id):
        raise HTTPException(422,'Введите имя говорящего, до 80 символов, без слешей.')
    data = await limited_upload(audio)
    filename = Path(audio.filename or 'audio.wav').name
    port, path = {'dialogue':(8003,'/dialogue'),'transcribe':(8003,'/transcribe'),
                  'identify':(8002,'/identify'),'enroll':(8001,f'/speakers/{quote(speaker_id.strip(),safe="")}/enroll')}[operation]
    journal = journal_store() if operation in {'dialogue','transcribe'} else None
    try:
        entry = journal.create(operation,filename,data,use_dictionary) if journal else None
    except OSError as exc:
        raise HTTPException(507,'Не удалось сохранить запись на диск. Проверьте свободное место; обработка не начата.') from exc
    def process():
        try:
            if entry:
                journal.update(entry['id'],'running')
            result = service_request('POST',port,path,files={'audio':(filename,data,'application/octet-stream')})
            if operation in {'dialogue','transcribe'}:
                result = apply_dictionary_result(result,enabled=use_dictionary,
                    entries=dictionary_store().list() if use_dictionary else [])
                result['journal_id'] = entry['id']
                journal.update(entry['id'],'done',result=result)
            return result
        except Exception as exc:
            if entry:
                journal.update(entry['id'],'failed',error=str(exc))
            raise
    try:
        job = app.state.jobs.submit(operation,process)
        if entry:
            job['journal_id'] = entry['id']
        return job
    except QueueFull as exc:
        if entry:
            journal.update(entry['id'],'failed',error=str(exc))
        raise HTTPException(429,str(exc)) from exc


@app.get('/api/journal')
def list_journal(q: str = Query('',max_length=200), limit: int = Query(30,ge=1,le=100), offset: int = Query(0,ge=0)):
    return journal_store().list(q,limit,offset)


@app.get('/api/journal/{recording_id}')
def journal_entry(recording_id: str):
    try:
        return journal_store().get(recording_id)
    except KeyError as exc:
        raise HTTPException(404,'Запись не найдена или удалена.') from exc


@app.get('/api/journal/{recording_id}/audio')
def journal_audio(recording_id: str, download: bool = False):
    entry = journal_entry(recording_id)
    path = journal_store().audio_path(recording_id)
    if not path.is_file():
        raise HTTPException(404,'Аудиофайл отсутствует на диске.')
    return FileResponse(path,media_type=audio_media_type(entry['filename']),
                        filename=entry['filename'],content_disposition_type='attachment' if download else 'inline')


@app.delete('/api/journal/{recording_id}')
def delete_journal(recording_id: str):
    try:
        journal_store().delete(recording_id)
    except KeyError as exc:
        raise HTTPException(404,'Запись не найдена или удалена.') from exc
    except JournalBusy as exc:
        raise HTTPException(409,str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503,'Не удалось удалить аудиофайл. Закройте воспроизведение и повторите.') from exc
    app.state.jobs.forget_journal(recording_id)
    return {'deleted':True}


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    try:
        return app.state.jobs.get(job_id)
    except KeyError as exc:
        raise HTTPException(404,'Результат уже недоступен. Повторите обработку записи.') from exc


@app.get('/api/speakers')
def speakers():
    try:
        return service_request('GET',8001,'/speakers')
    except RuntimeError as exc:
        raise HTTPException(503,str(exc)) from exc


if WEB.exists():
    app.mount('/static',StaticFiles(directory=WEB),name='static')


@app.get('/api/dictionary')
def list_dictionary(q: str=''):
    return {'words':dictionary_store().list(q[:200])}


@app.post('/api/dictionary',status_code=201)
def create_word(entry: WordInput):
    try:
        return dictionary_store().save(entry)
    except DictionaryError as exc:
        raise HTTPException(409,str(exc)) from exc


@app.put('/api/dictionary/{word_id}')
def update_word(word_id: str,entry: WordInput):
    try:
        return dictionary_store().save(entry,word_id)
    except KeyError as exc:
        raise HTTPException(404,'Слово не найдено.') from exc
    except DictionaryError as exc:
        raise HTTPException(409,str(exc)) from exc


@app.delete('/api/dictionary/{word_id}')
def archive_word(word_id: str):
    try:
        dictionary_store().archive(word_id)
        return {'archived':word_id}
    except KeyError as exc:
        raise HTTPException(404,'Слово не найдено.') from exc


@app.post('/api/dictionary/{word_id}/examples',status_code=201)
async def add_example(word_id: str,audio: UploadFile=File(...),transcript: str=Form('')):
    from starlette.concurrency import run_in_threadpool
    if len(transcript)>1000:
        raise HTTPException(422,'Текст примера должен быть короче 1000 символов.')
    data=await limited_upload(audio,limit=15*1024*1024)
    def save():
        try:
            store=dictionary_store()
            word=store.get(word_id)
            suffix=Path(audio.filename or 'audio.wav').suffix
            with normalized_upload(data,suffix) as wav:
                duration=wav_duration(wav)
                if duration<.35 or duration>30:
                    raise HTTPException(422,'Пример должен длиться от 0,35 до 30 секунд.')
                if not has_speech(wav):
                    raise HTTPException(422,'В примере не найдена речь. Выберите другую запись.')
                return store.add_example(word_id,wav,audio.filename or 'audio',transcript or word['word'],duration)
        except KeyError as exc:
            raise HTTPException(404,'Слово не найдено.') from exc
        except (DictionaryError,AudioError,ValueError) as exc:
            raise HTTPException(400,str(exc)) from exc
    return await run_in_threadpool(save)


@app.get('/api/dictionary/examples/{example_id}/audio')
def example_audio(example_id: str):
    try:
        return FileResponse(dictionary_store().example_path(example_id),media_type='audio/wav')
    except KeyError as exc:
        raise HTTPException(404,'Аудиопример не найден.') from exc
