from fastapi.testclient import TestClient

from voice_id_mvp.services.enrollment_app import app as enrollment
from voice_id_mvp.services.identification_app import app as identification
from voice_id_mvp.services.transcription_app import app as transcription


def test_health_endpoints() -> None:
    for app, service in (
        (enrollment, "enrollment"),
        (identification, "identification"),
        (transcription, "transcription"),
    ):
        response = TestClient(app).get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": service}

