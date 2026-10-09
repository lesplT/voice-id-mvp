import time
import pytest
from fastapi.testclient import TestClient

from voice_id_mvp.journal import JournalStore, JournalBusy
from voice_id_mvp.services import workspace_app as module


def wait_job(client, job_id):
    for _ in range(100):
        job = client.get('/api/jobs/' + job_id).json()
        if job['status'] in {'done', 'failed'}:
            return job
        time.sleep(.01)
    raise AssertionError('Job did not finish')


def test_storage_persistence_unknown_search_delete(tmp_path):
    store = JournalStore(tmp_path)
    entry = store.create('dialogue', '../покупатель.mp3', b'original-audio')
    result = {'text': 'Здравствуйте, покупатель!', 'segments': [
        {'start': .1, 'end': 1.2, 'speaker_id': 'UNKNOWN', 'text': 'Здравствуйте, покупатель!'}]}
    store.update(entry['id'], 'done', result=result)
    reopened = JournalStore(tmp_path)
    loaded = reopened.get(entry['id'])
    assert loaded['result'] == result
    assert loaded['filename'] == 'покупатель.mp3'
    assert loaded['participants'] == ['UNKNOWN']
    assert reopened.audio_path(entry['id']).read_bytes() == b'original-audio'
    assert reopened.list('ПОКУПАТЕЛЬ')['total'] == 1
    assert reopened.list('no match')['total'] == 0
    assert reopened.list()['total_bytes'] == len(b'original-audio')
    reopened.delete(entry['id'])
    assert not reopened.audio_path(entry['id']).exists()
    with pytest.raises(KeyError):
        reopened.get(entry['id'])


def test_interrupted_and_busy(tmp_path):
    store = JournalStore(tmp_path)
    queued = store.create('transcribe', 'a.wav', b'abc')
    running = store.create('dialogue', 'b.wav', b'def')
    store.update(running['id'], 'running')
    with pytest.raises(JournalBusy):
        store.delete(running['id'])
    assert store.audio_path(running['id']).exists()
    reopened = JournalStore(tmp_path)
    reopened.recover_interrupted()
    for entry in (queued, running):
        assert reopened.get(entry['id'])['status'] == 'interrupted'
        assert reopened.audio_path(entry['id']).exists()
        reopened.delete(entry['id'])
    assert reopened.list()['total'] == 0


def test_path_guard_and_pagination(tmp_path):
    store = JournalStore(tmp_path)
    for _ in range(4):
        entry = store.create('transcribe', 'файл.wav', b'abc')
        store.update(entry['id'], 'done', {'text': 'text'})
    one = store.list(limit=2)
    two = store.list(limit=2, offset=2)
    assert one['total'] == 4 and len(one['entries']) == 2
    assert not {e['id'] for e in one['entries']} & {e['id'] for e in two['entries']}
    for unsafe in ('../secret', 'x', 'a' * 33, 'A' * 32):
        with pytest.raises(KeyError):
            store.audio_path(unsafe)


def test_gateway_journal_restart_audio_and_unknown(monkeypatch):
    result = {'text': 'Текст незнакомого покупателя', 'segments': [
        {'start': 0, 'end': 1, 'speaker_id': 'UNKNOWN', 'reason': 'below_threshold', 'text': 'Здравствуйте'}]}
    monkeypatch.setattr(module, 'service_request', lambda *a, **k: result.copy())
    with TestClient(module.app) as client:
        response = client.post('/api/jobs', data={'operation': 'dialogue', 'use_dictionary': 'false'},
            files={'audio': ('customer.mp3', b'original-mp3')})
        assert response.status_code == 202
        job = response.json()
        recording_id = job['journal_id']
        assert wait_job(client, job['id'])['status'] == 'done'
        entry = client.get('/api/journal/' + recording_id).json()
        assert entry['result']['segments'][0]['speaker_id'] == 'UNKNOWN'
        assert entry['result']['original_text'] == result['text']
        assert entry['result']['segments'][0]['text'] == 'Здравствуйте'
    with TestClient(module.app) as client:
        assert client.get('/api/jobs/' + job['id']).status_code == 404
        assert client.get('/api/journal/' + recording_id).json()['status'] == 'done'
        audio = client.get('/api/journal/' + recording_id + '/audio')
        assert audio.content == b'original-mp3'
        assert audio.headers['content-type'] == 'audio/mpeg'
        assert 'inline' in audio.headers['content-disposition']
        assert 'attachment' in client.get('/api/journal/' + recording_id + '/audio?download=true').headers['content-disposition']
        assert client.delete('/api/journal/' + recording_id, headers={'origin': 'https://outside.invalid'}).status_code == 403
        assert client.delete('/api/journal/' + recording_id).status_code == 200
        assert client.get('/api/journal/' + recording_id).status_code == 404
        assert client.get('/api/journal/' + recording_id + '/audio').status_code == 404
        assert client.get('/api/journal?limit=0').status_code == 422


def test_failure_keeps_audio_and_nonjournal_operations(monkeypatch):
    def fail(*a, **k):
        raise RuntimeError('Не удалось прочитать аудио')
    monkeypatch.setattr(module, 'service_request', fail)
    with TestClient(module.app) as client:
        job = client.post('/api/jobs', data={'operation': 'transcribe'}, files={'audio': ('broken.mp3', b'broken')}).json()
        assert wait_job(client, job['id'])['status'] == 'failed'
        entry = client.get('/api/journal/' + job['journal_id']).json()
        assert entry['status'] == 'failed' and entry['error']
        assert client.get('/api/journal/' + entry['id'] + '/audio').content == b'broken'
        job = client.post('/api/jobs', data={'operation': 'identify'}, files={'audio': ('a.mp3', b'abc')}).json()
        wait_job(client, job['id'])
        assert client.get('/api/journal').json()['total'] == 1


def test_queue_full_still_keeps_recording(monkeypatch):
    from voice_id_mvp.workspace_jobs import QueueFull
    with TestClient(module.app) as client:
        def full(*a):
            raise QueueFull('Очередь заполнена')
        monkeypatch.setattr(module.app.state.jobs, 'submit', full)
        response = client.post('/api/jobs', data={'operation':'dialogue'}, files={'audio':('a.wav', b'audio')})
        assert response.status_code == 429
        entry = client.get('/api/journal').json()['entries'][0]
        assert entry['status'] == 'failed'
        assert client.get('/api/journal/' + entry['id'] + '/audio').content == b'audio'
