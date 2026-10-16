from fastapi import FastAPI, File, HTTPException, UploadFile

from voice_id_mvp.services.api_utils import api_error, read_upload
from voice_id_mvp.audio.processing import normalized_upload, wav_duration
from voice_id_mvp.recognition.embeddings import get_embedder, normalized_centroid
from voice_id_mvp.storage.voice_repository import VoiceRepository


app = FastAPI(title="Voice ID Enrollment", version="0.1.0")


@app.get("/health")
def health() -> dict:
    """Purpose:
        Report HTTP service availability without loading recognition models.

    Ingress:
        A GET /health request.

    Egress:
        A JSON dictionary identifying the service and its status.
    """
    return {"status": "ok", "service": "enrollment"}


@app.post("/speakers/{speaker_id}/enroll")
async def enroll(speaker_id: str, audio: UploadFile = File(...)) -> dict:
    """Purpose:
        Register normalized embeddings and a centroid for one person's recording.

    Ingress:
        A speaker identifier and uploaded single-speaker audio.

    Egress:
        Enrollment ID, embedding count, and duration; existing references are retained.
    """
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
    """Purpose:
        List saved speaker profiles through the appropriate repository/backend.

    Ingress:
        A GET request for registered speakers.

    Egress:
        A dictionary of speaker summaries; backend failures are translated into HTTP errors.
    """
    try:
        return {"speakers": VoiceRepository().list_speakers()}
    except Exception as exc:
        raise api_error(exc) from exc


@app.delete("/speakers/{speaker_id}")
def delete_speaker(speaker_id: str) -> dict:
    """Purpose:
        Delete only the explicitly selected speaker's stored references and centroid.

    Ingress:
        A speaker identifier from a DELETE request.

    Egress:
        A deletion acknowledgement, or a translated storage error.
    """
    try:
        VoiceRepository().delete_speaker(speaker_id)
        return {"deleted": speaker_id}
    except Exception as exc:
        raise api_error(exc) from exc
