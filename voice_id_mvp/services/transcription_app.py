from typing import Literal

from fastapi import FastAPI, File, UploadFile
from starlette.concurrency import run_in_threadpool

from voice_id_mvp.services.api_utils import api_error, read_upload
from voice_id_mvp.recognition.asr import transcribe_wav
from voice_id_mvp.audio.processing import normalized_upload
from voice_id_mvp.application.dialogue import dialogue_from_wav
from voice_id_mvp.application.dictionary_corrections import apply_dictionary_result
from voice_id_mvp.recognition.nemotron import diarize_nemotron


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
async def dialogue(audio: UploadFile = File(...), use_dictionary: bool = False,
                   diarization_engine: Literal["legacy", "nemotron"] | None = None) -> dict:
    """Purpose:
        Normalize a conversation and combine acoustic speaker ID with Russian text.

    Ingress:
        Uploaded conversation audio, optional dictionary flag and optional diarizer override.

    Egress:
        Timestamped dialogue segments, including UNKNOWN participants and their recognized text.
    """
    data, suffix = await read_upload(audio)
    try:
        with normalized_upload(data, suffix) as wav:
            result = await run_in_threadpool(dialogue_from_wav, wav, diarization_engine)
            return apply_dictionary_result(result) if use_dictionary else result
    except Exception as exc:
        raise api_error(exc) from exc


@app.post("/diarize")
async def diarize(audio: UploadFile = File(...)) -> dict:
    """Purpose: Return NVIDIA speaker tracks without text or identity lookup.
    Ingress: A multipart recording upload.
    Egress: Anonymous timestamped segments, including simultaneous speakers.
    """
    data, suffix = await read_upload(audio)
    try:
        with normalized_upload(data, suffix) as wav:
            turns = await run_in_threadpool(diarize_nemotron, wav)
            return {"engine": "nemotron", "segments": [
                {"start": part.start, "end": part.end, "speaker": label} for part, label in turns
            ]}
    except Exception as exc:
        raise api_error(exc) from exc
