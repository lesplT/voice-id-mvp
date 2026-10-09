from pathlib import Path

from fastapi import HTTPException, UploadFile

from .audio import AudioError, normalized_upload


async def read_upload(upload: UploadFile) -> tuple[bytes, str]:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    suffix = Path(upload.filename or "audio.bin").suffix or ".bin"
    return data, suffix


def api_error(exc: Exception) -> HTTPException:
    status = 400 if isinstance(exc, (AudioError, ValueError)) else 503
    return HTTPException(status_code=status, detail=str(exc))

