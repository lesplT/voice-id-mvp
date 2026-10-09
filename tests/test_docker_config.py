from types import SimpleNamespace
from voice_id_mvp.services import workspace_app


def test_gateway_uses_configured_service_address(monkeypatch):
    observed = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs['trust_env'] is False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def request(self, method, url, **kwargs):
            observed.append(url)
            return SimpleNamespace(status_code=200, json=lambda:{'ok':True})
    monkeypatch.setattr(workspace_app.httpx, 'Client', Client)
    monkeypatch.setattr(workspace_app, 'get_settings', lambda:SimpleNamespace(
        enrollment_service_url='http://enrollment:8001',
        identification_service_url='http://identification:8002',
        transcription_service_url='http://transcription:8003/'))
    assert workspace_app.service_request('GET',8001,'/speakers') == {'ok':True}
    assert workspace_app.service_request('POST',8003,'/dialogue') == {'ok':True}
    assert observed == ['http://enrollment:8001/speakers','http://transcription:8003/dialogue']
