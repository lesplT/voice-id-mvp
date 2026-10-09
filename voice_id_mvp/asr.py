from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import wave
from pathlib import Path

from .config import get_settings
from .audio import wav_duration, write_wav_slice


class AsrUnavailable(RuntimeError):
    pass


class VoskAsr:
    _model = None
    _lock = threading.Lock()

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    from vosk import Model

                    path = get_settings().vosk_asr_model_path
                    if not path.exists():
                        raise AsrUnavailable(f"Vosk ASR model is missing: {path}")
                    cls._model = Model(str(path))
        return cls._model

    def transcribe(self, wav_path: Path) -> dict:
        from vosk import KaldiRecognizer

        words: list[dict] = []
        texts: list[str] = []
        with wave.open(str(wav_path), "rb") as wav:
            recognizer = KaldiRecognizer(self._get_model(), wav.getframerate())
            recognizer.SetWords(True)
            while True:
                chunk = wav.readframes(4000)
                if not chunk:
                    break
                if recognizer.AcceptWaveform(chunk):
                    self._consume(recognizer.Result(), texts, words)
            self._consume(recognizer.FinalResult(), texts, words)
        return {"engine": "vosk", "text": " ".join(texts).strip(), "words": words}

    @staticmethod
    def _consume(raw: str, texts: list[str], words: list[dict]) -> None:
        data = json.loads(raw)
        if data.get("text"):
            texts.append(data["text"])
        words.extend(data.get("result", []))


class GigaAmAsr:
    _model = None
    _lock = threading.Lock()

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    try:
                        import gigaam
                        import torch
                    except ImportError as exc:
                        raise AsrUnavailable(
                            "GigaAM is not installed; install requirements-gigaam.txt"
                        ) from exc
                    settings = get_settings()
                    torch.set_num_threads(4)
                    # Upstream invokes the literal 'ffmpeg' executable.
                    if not shutil.which("ffmpeg"):
                        import imageio_ffmpeg
                        tools = settings.gigaam_cache_path.parent / "tools"
                        tools.mkdir(parents=True, exist_ok=True)
                        binary = tools / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
                        if not binary.exists():
                            shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), binary)
                        os.environ["PATH"] = str(tools) + os.pathsep + os.environ.get("PATH", "")
                    cls._model = gigaam.load_model(
                        get_settings().gigaam_model_name,
                        fp16_encoder=False,
                        device="cpu",
                        download_root=str(settings.gigaam_cache_path),
                    )
        return cls._model

    def transcribe(self, wav_path: Path) -> dict:
        try:
            model = self._get_model()
            duration = wav_duration(wav_path)
            if duration <= 24:
                result = model.transcribe(str(wav_path))
            else:
                # Upstream short-form API accepts at most 25 seconds.
                parts = []
                with tempfile.TemporaryDirectory(prefix="gigaam-parts-") as tmp:
                    cursor = 0.0
                    while cursor < duration:
                        clip = Path(tmp) / "part.wav"
                        end = min(cursor + 20, duration)
                        write_wav_slice(wav_path, clip, cursor, end)
                        parts.append(model.transcribe(str(clip)))
                        cursor = end
                result = " ".join(parts)
        except Exception as exc:
            raise AsrUnavailable(f"GigaAM failed: {exc}") from exc
        text = result if isinstance(result, str) else str(result)
        return {"engine": "gigaam", "text": text.strip(), "words": []}


def transcribe_wav(wav_path: Path) -> dict:
    settings = get_settings()
    engine = settings.asr_engine.lower()
    if engine == "vosk":
        return VoskAsr().transcribe(wav_path)
    if engine == "gigaam":
        try:
            return GigaAmAsr().transcribe(wav_path)
        except AsrUnavailable as exc:
            if not settings.asr_fallback_to_vosk:
                raise
            result = VoskAsr().transcribe(wav_path)
            result["fallback_reason"] = str(exc)
            return result
    raise AsrUnavailable(f"Unknown ASR_ENGINE: {settings.asr_engine}")
