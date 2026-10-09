from fastapi import FastAPI, File, UploadFile

from ..api_utils import api_error, read_upload
from ..audio import normalized_upload
from ..identification import identify_wav


app = FastAPI(title="Voice ID Identification", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "identification"}


@app.post("/identify")
async def identify(
    audio: UploadFile = File(...), top_k: int | None = None
) -> dict:
    data, suffix = await read_upload(audio)
    try:
        with normalized_upload(data, suffix) as wav:
            return identify_wav(wav, top_k=top_k)
    except Exception as exc:
        raise api_error(exc) from exc

