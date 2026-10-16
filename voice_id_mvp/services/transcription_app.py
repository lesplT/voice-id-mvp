from fastapi import FastAPI, File, UploadFile

from voice_id_mvp.services.api_utils import api_error, read_upload
from voice_id_mvp.recognition.asr import transcribe_wav
from voice_id_mvp.audio.processing import normalized_upload
from voice_id_mvp.application.dialogue import dialogue_from_wav
from voice_id_mvp.application.dictionary_corrections import apply_dictionary_result


app = FastAPI(title="Voice ID Transcription", version="0.1.0")


@app.get("/health")
def health() -> dict:
    """Purpose:
        Report HTTP service availability without loading recognition models.

    Ingress:
        A GET /health request.

    Egress:
        A JSON dictionary identifying the service and its status.
    """
    return {"status": "ok", "service": "transcription"}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), use_dictionary: bool = False) -> dict:
    """Purpose:
        Normalize an upload and recognize Russian text, optionally applying explicit dictionary rules.

    Ingress:
        Uploaded audio and the use_dictionary query flag.

    Egress:
        A transcription dictionary; original text/correction metadata are included when enabled.
    """
    data, suffix = await read_upload(audio)
    try:
        with normalized_upload(data, suffix) as wav:
            result = transcribe_wav(wav)
            return apply_dictionary_result(result) if use_dictionary else result
    except Exception as exc:
        raise api_error(exc) from exc


@app.post("/dialogue")
async def dialogue(audio: UploadFile = File(...), use_dictionary: bool = False) -> dict:
    """Purpose:
        Normalize a conversation and combine acoustic speaker ID with Russian text.

    Ingress:
        Uploaded conversation audio and optional dictionary flag.

    Egress:
        Timestamped dialogue segments, including UNKNOWN participants and their recognized text.
    """
    data, suffix = await read_upload(audio)
    try:
        with normalized_upload(data, suffix) as wav:
            result = dialogue_from_wav(wav)
            return apply_dictionary_result(result) if use_dictionary else result
    except Exception as exc:
        raise api_error(exc) from exc
