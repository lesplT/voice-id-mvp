from fastapi import FastAPI, File, HTTPException, UploadFile

from ..api_utils import api_error, read_upload
from ..audio import normalized_upload, wav_duration
from ..embeddings import get_embedder, normalized_centroid
from ..repository import VoiceRepository


app = FastAPI(title="Voice ID Enrollment", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "enrollment"}


@app.post("/speakers/{speaker_id}/enroll")
async def enroll(speaker_id: str, audio: UploadFile = File(...)) -> dict:
    if not speaker_id.strip():
        raise HTTPException(status_code=422, detail="speaker_id is required")
    data, suffix = await read_upload(audio)
    try:
        with normalized_upload(data, suffix) as wav:
            embeddings = get_embedder().extract_all(wav)
            if not embeddings:
                raise HTTPException(status_code=422, detail="no_embedding")
            centroid = normalized_centroid(embeddings)
            enrollment_id = VoiceRepository().add_enrollment(
                speaker_id.strip(), embeddings, centroid, audio.filename or "upload"
            )
            return {
                "speaker_id": speaker_id.strip(),
                "enrollment_id": enrollment_id,
                "embedding_count": len(embeddings),
                "duration_seconds": wav_duration(wav),
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise api_error(exc) from exc


@app.get("/speakers")
def speakers() -> dict:
    try:
        return {"speakers": VoiceRepository().list_speakers()}
    except Exception as exc:
        raise api_error(exc) from exc


@app.delete("/speakers/{speaker_id}")
def delete_speaker(speaker_id: str) -> dict:
    try:
        VoiceRepository().delete_speaker(speaker_id)
        return {"deleted": speaker_id}
    except Exception as exc:
        raise api_error(exc) from exc
