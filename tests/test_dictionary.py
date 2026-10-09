import wave
import pytest
from fastapi.testclient import TestClient
from voice_id_mvp.dictionary import DictionaryStore, DictionaryError, WordInput
from voice_id_mvp.services import workspace_app as module


def test_dictionary_persistence_and_conflicts(tmp_path):
    store=DictionaryStore(tmp_path)
    word=store.save(WordInput(word='Чиабатта',aliases=['чааббат']))
    assert DictionaryStore(tmp_path).list()[0]['word']=='Чиабатта'
    assert store.list('чиаб')[0]['id']==word['id']
    with pytest.raises(DictionaryError):
        store.save(WordInput(word='другая',aliases=['чааббат']))
    store.save(WordInput(word='Чиабатта',aliases=['чааббат'],active=False),word['id'])
    assert not store.get(word['id'])['active']
    store.archive(word['id'])
    assert store.list()==[]


def test_audio_storage_and_archive(tmp_path):
    store=DictionaryStore(tmp_path/'db')
    word=store.save(WordInput(word='Круассан'))
    wav=tmp_path/'a.wav'
    with wave.open(str(wav),'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000);stream.writeframes(b'\0\0'*16000)
    result=store.add_example(word['id'],wav,'../outside.wav','круассан',1)
    example=result['examples'][0]
    assert store.example_path(example['id']).parent==store.audio_folder
    result=store.add_example(word['id'],wav,'second.wav','круассан',1)
    assert len(result['examples'])==2
    with pytest.raises(KeyError):
        store.example_path('../outside')
    store.archive(word['id'])
    with pytest.raises(KeyError):
        store.example_path(example['id'])
    assert (store.audio_folder/(example['id']+'.wav')).exists()


def test_dictionary_http_validation(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'dictionary_store',lambda:DictionaryStore(tmp_path))
    with TestClient(module.app) as client:
        assert client.post('/api/dictionary',json={'word':' '}).status_code==422
        word=client.post('/api/dictionary',json={'word':'чиабатта','aliases':['чааббат']}).json()
        assert client.post('/api/dictionary',json={'word':'ЧИАБАТТА'}).status_code==409
        assert client.get('/api/dictionary').json()['words'][0]['id']==word['id']
        assert client.post(f"/api/dictionary/{word['id']}/examples",files={'audio':('broken.mp3',b'broken')}).status_code==400
        assert client.get('/api/dictionary/examples/nope/audio').status_code==404
        assert client.delete('/api/dictionary/'+word['id']).status_code==200
        assert client.get('/api/dictionary').json()['words']==[]
