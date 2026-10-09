import threading
import time
import pytest
from fastapi.testclient import TestClient
from voice_id_mvp.services import workspace_app as module
from voice_id_mvp.workspace_jobs import JobManager, QueueFull


def test_queue_and_failure():
    manager=JobManager(capacity=1)
    event=threading.Event()
    try:
        job=manager.submit('test',lambda:event.wait(2))
        assert manager.get(job['id'])['status'] in {'queued','running'}
        with pytest.raises(QueueFull):
            manager.submit('other',lambda:1)
        event.set()
    finally:
        manager.shutdown()
    assert manager.get(job['id'])['status']=='done'
    other=JobManager()
    def broken():
        raise ValueError('bad audio')
    job=other.submit('test',broken)
    other.shutdown()
    assert other.get(job['id'])['status']=='failed'


def test_gateway_contract(monkeypatch,tmp_path):
    from voice_id_mvp.dictionary import DictionaryStore
    monkeypatch.setattr(module,'dictionary_store',lambda:DictionaryStore(tmp_path))
    monkeypatch.setattr(module,'service_request',lambda *a,**k:{'text':'Русский текст'})
    with TestClient(module.app) as client:
        assert client.get('/health').status_code==200
        result=client.post('/api/jobs',data={'operation':'transcribe'},files={'audio':('a.wav',b'abc')})
        assert result.status_code==202
        job_id=result.json()['id']
        for _ in range(50):
            job=client.get('/api/jobs/'+job_id).json()
            if job['status']=='done':
                break
            time.sleep(.01)
        assert job['result']['text']=='Русский текст'
        assert client.get('/api/jobs/not-found').status_code==404
        assert client.post('/api/jobs',data={'operation':'other'},files={'audio':('a',b'abc')}).status_code==422
        assert client.post('/api/jobs',data={'operation':'identify'},files={'audio':('a',b'')}).status_code==400


def test_origin_and_size(monkeypatch):
    monkeypatch.setattr(module,'MAX_UPLOAD',2)
    with TestClient(module.app) as client:
        assert client.post('/api/jobs',headers={'Origin':'https://outside.invalid'},data={'operation':'identify'},files={'audio':('a',b'abc')}).status_code==403
        assert client.post('/api/jobs',data={'operation':'identify'},files={'audio':('a',b'abc')}).status_code==413
        # Explicit limit exercises the streaming upload bound.
        import asyncio
        from starlette.datastructures import UploadFile
        from io import BytesIO
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            asyncio.run(module.limited_upload(UploadFile(BytesIO(b'abc')),limit=2))


def test_upstream_failure_is_a_failed_job(monkeypatch):
    def fail(*args,**kwargs):
        raise RuntimeError('Сервис не запущен')
    monkeypatch.setattr(module,'service_request',fail)
    with TestClient(module.app) as client:
        job=client.post('/api/jobs',data={'operation':'identify'},files={'audio':('a.wav',b'abc')}).json()
        for _ in range(50):
            result=client.get('/api/jobs/'+job['id']).json()
            if result['status']=='failed':
                break
            time.sleep(.01)
        assert result['status']=='failed'
        assert result['error']=='Сервис не запущен'
