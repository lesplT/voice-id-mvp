from __future__ import annotations

import json
import threading
import wave
from pathlib import Path

import numpy as np

from .config import get_settings


class EmbeddingError(RuntimeError):
    pass


def get_embedder():
    engine = get_settings().speaker_engine
    if engine == "ecapa":
        from .ecapa import EcapaSpeakerEmbedder
        return EcapaSpeakerEmbedder()
    if engine == "vosk":
        return VoskSpeakerEmbedder()
    raise EmbeddingError(f"Unknown speaker engine: {engine}")


def l2_normalize(vector: list[float] | np.ndarray) -> list[float]:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot normalize an empty/zero embedding")
    return (array / norm).astype(float).tolist()


def normalized_centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("At least one embedding is required")
    normalized = np.asarray([l2_normalize(v) for v in vectors], dtype=np.float32)
    return l2_normalize(normalized.mean(axis=0))


class VoskSpeakerEmbedder:
    _asr_model = None
    _spk_model = None
    _lock = threading.Lock()

    @classmethod
    def _get_models(cls):
        if cls._asr_model is None or cls._spk_model is None:
            with cls._lock:
                if cls._asr_model is None or cls._spk_model is None:
                    from vosk import Model, SpkModel

                    settings = get_settings()
                    if not settings.vosk_asr_model_path.exists():
                        raise EmbeddingError(f"Vosk ASR model is missing: {settings.vosk_asr_model_path}")
                    if not settings.vosk_spk_model_path.exists():
                        raise EmbeddingError(f"Speaker model is missing: {settings.vosk_spk_model_path}")
                    cls._asr_model = Model(str(settings.vosk_asr_model_path))
                    cls._spk_model = SpkModel(str(settings.vosk_spk_model_path))
        return cls._asr_model, cls._spk_model

    def extract_all(self, wav_path: Path) -> list[list[float]]:
        from vosk import KaldiRecognizer

        asr_model, spk_model = self._get_models()
        results: list[list[float]] = []
        with wave.open(str(wav_path), "rb") as wav:
            recognizer = KaldiRecognizer(asr_model, wav.getframerate())
            recognizer.SetSpkModel(spk_model)
            while True:
                chunk = wav.readframes(4000)
                if not chunk:
                    break
                if recognizer.AcceptWaveform(chunk):
                    self._append_result(results, recognizer.Result())
            self._append_result(results, recognizer.FinalResult())
        return results

    @staticmethod
    def _append_result(results: list[list[float]], raw: str) -> None:
        try:
            data = json.loads(raw)
            vector = data.get("spk")
            if vector:
                results.append(l2_normalize(vector))
        except (json.JSONDecodeError, TypeError, ValueError):
            return
