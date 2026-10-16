from fastapi import FastAPI, File, UploadFile

from voice_id_mvp.services.api_utils import api_error, read_upload
from voice_id_mvp.audio.processing import normalized_upload
from voice_id_mvp.recognition.identification import identify_wav


app = FastAPI(title="Voice ID Identification", version="0.1.0")


@app.get("/health")
def health() -> dict:
    """Purpose:
        Report HTTP service availability without loading recognition models.

    Ingress:
        A GET /health request.

    Egress:
        A JSON dictionary identifying the service and its status.
    """
    return {"status": "ok", "service": "identification"}


@app.post("/identify")
async def identify(
    audio: UploadFile = File(...), top_k: int | None = None
) -> dict:
    """Purpose:
        Normalize a single-speaker upload and apply conservative voice identification.

    Ingress:
        Uploaded audio and the endpoint's optional identification parameters.

    Egress:
        A speaker/UNKNOWN decision with duration, scores, and candidates.
    """
    data, suffix = await read_upload(audio)
    try:
        with normalized_upload(data, suffix) as wav:
            return identify_wav(wav, top_k=top_k)
    except Exception as exc:
        raise api_error(exc) from exc

