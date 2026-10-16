from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from voice_id_mvp.core.config import get_settings


class AudioError(RuntimeError):
    """Purpose:
        Represent audio decoding, normalization, or validation failures.

    Ingress:
        A message describing an unsupported or invalid recording.

    Egress:
        An exception translated by the HTTP layer into an actionable response.
    """
    pass


def normalize_audio(source: Path, destination: Path) -> Path:
    """Purpose:
        Convert an input recording into mono 16 kHz PCM16 WAV using FFmpeg.

    Ingress:
        Source recording path and destination WAV path.

    Egress:
        The validated destination path; raises AudioError if conversion fails.
    """
    settings = get_settings()
    ffmpeg = shutil.which(settings.ffmpeg_bin)
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = settings.ffmpeg_bin
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioError(f"FFmpeg failed to start: {exc}") from exc
    if completed.returncode != 0:
        raise AudioError(completed.stderr.strip() or "FFmpeg normalization failed")
    validate_wav(destination)
    return destination


def validate_wav(path: Path) -> None:
    """Purpose:
        Check the channel count, sample width, and rate required by recognition backends.

    Ingress:
        A readable WAV path.

    Egress:
        None on success; raises AudioError for an incompatible WAV format.
    """
    try:
        with wave.open(str(path), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                raise AudioError("Expected mono PCM s16le WAV at 16 kHz")
    except wave.Error as exc:
        raise AudioError(f"Invalid WAV: {exc}") from exc


def wav_duration(path: Path) -> float:
    """Purpose:
        Measure a WAV recording from its frame count and sample rate.

    Ingress:
        A readable WAV path.

    Egress:
        Recording duration in seconds as a float.
    """
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


@contextmanager
def normalized_upload(data: bytes, suffix: str = ".bin") -> Iterator[Path]:
    """Purpose:
        Provide an isolated, temporary normalized WAV for uploaded audio.

    Ingress:
        Uploaded audio bytes and the filename suffix.

    Egress:
        Yields a normalized WAV path; temporary files are removed on context exit.
    """
    with tempfile.TemporaryDirectory(prefix="voice-id-") as tmp:
        src = Path(tmp) / f"input{suffix}"
        dst = Path(tmp) / "normalized.wav"
        src.write_bytes(data)
        normalize_audio(src, dst)
        yield dst


def write_wav_slice(source: Path, destination: Path, start: float, end: float) -> Path:
    """Purpose:
        Extract a time interval while preserving the source WAV format and frame accuracy.

    Ingress:
        Source/destination WAV paths and start/end times in seconds.

    Egress:
        The destination path containing the selected PCM frames.
    """
    with wave.open(str(source), "rb") as src:
        rate = src.getframerate()
        channels = src.getnchannels()
        sample_width = src.getsampwidth()
        first_frame = max(0, round(start * rate))
        last_frame = max(first_frame, round(end * rate))
        src.setpos(first_frame)
        frames = src.readframes(last_frame - first_frame)
    with wave.open(str(destination), "wb") as dst:
        dst.setnchannels(channels)
        dst.setsampwidth(sample_width)
        dst.setframerate(rate)
        dst.writeframes(frames)
    return destination
