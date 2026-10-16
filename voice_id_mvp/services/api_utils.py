from pathlib import Path

from fastapi import HTTPException, UploadFile

from voice_id_mvp.audio.processing import AudioError, normalized_upload


async def read_upload(upload: UploadFile) -> tuple[bytes, str]:
    """Purpose:
        Read and bound a multipart audio upload before processing.

    Ingress:
        An UploadFile supplied by an HTTP client.

    Egress:
        The uploaded bytes and safe filename suffix; raises an HTTP error for invalid input.
    """
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    suffix = Path(upload.filename or "audio.bin").suffix or ".bin"
    return data, suffix


def api_error(exc: Exception) -> HTTPException:
    """Purpose:
        Translate processing exceptions into consistent HTTP errors.

    Ingress:
        An exception raised by audio, recognition, or storage code.

    Egress:
        An HTTPException that preserves the appropriate client/server failure category.
    """
    status = 400 if isinstance(exc, (AudioError, ValueError)) else 503
    return HTTPException(status_code=status, detail=str(exc))
