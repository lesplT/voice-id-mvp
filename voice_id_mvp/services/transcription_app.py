from fastapi import FastAPI, File, UploadFile

from ..api_utils import api_error, read_upload
from ..asr import transcribe_wav
from ..audio import normalized_upload
from ..dialogue import dialogue_from_wav
from ..dictionary_corrections import apply_dictionary_result


app = FastAPI(title="Voice ID Transcription", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "transcription"}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), use_dictionary: bool = False) -> dict:
    data, suffix = await read_upload(audio)
    try:
        with normalized_upload(data, suffix) as wav:
            result = transcribe_wav(wav)
            return apply_dictionary_result(result) if use_dictionary else result
    except Exception as exc:
        raise api_error(exc) from exc


@app.post("/dialogue")
async def dialogue(audio: UploadFile = File(...), use_dictionary: bool = False) -> dict:
    data, suffix = await read_upload(audio)
    try:
        with normalized_upload(data, suffix) as wav:
            result = dialogue_from_wav(wav)
            return apply_dictionary_result(result) if use_dictionary else result
    except Exception as exc:
        raise api_error(exc) from exc
