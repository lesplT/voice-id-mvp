from __future__ import annotations

import json
import threading
import wave
from pathlib import Path

import numpy as np

from voice_id_mvp.core.config import get_settings


class EmbeddingError(RuntimeError):
    """Purpose:
        Represent invalid speaker input, missing weights, or an unsupported engine.

    Ingress:
        An explanation of the embedding failure.

    Egress:
        An exception that recognition/API layers can report or classify.
    """
    pass


def get_embedder():
    """Purpose:
        Instantiate the configured pretrained speaker-embedding adapter.

    Ingress:
        The speaker_engine setting.

    Egress:
        An ECAPA or Vosk adapter; raises EmbeddingError for unknown engines.
    """
    engine = get_settings().speaker_engine
    if engine == "ecapa":
        from voice_id_mvp.recognition.ecapa import EcapaSpeakerEmbedder
        return EcapaSpeakerEmbedder()
    if engine == "vosk":
        return VoskSpeakerEmbedder()
    raise EmbeddingError(f"Unknown speaker engine: {engine}")


def l2_normalize(vector: list[float] | np.ndarray) -> list[float]:
    """Purpose:
        Normalize a finite nonzero embedding to unit Euclidean length.

    Ingress:
        A numeric vector as a list or NumPy array.

    Egress:
        A unit-length list of floats; raises ValueError for a zero or invalid norm.
    """
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot normalize an empty/zero embedding")
    return (array / norm).astype(float).tolist()


def normalized_centroid(vectors: list[list[float]]) -> list[float]:
    """Purpose:
        Compute a normalized mean from normalized reference embeddings.

    Ingress:
        A nonempty list of speaker vectors.

    Egress:
        The unit-length centroid; raises ValueError when no vectors are supplied.
    """
    if not vectors:
        raise ValueError("At least one embedding is required")
    normalized = np.asarray([l2_normalize(v) for v in vectors], dtype=np.float32)
    return l2_normalize(normalized.mean(axis=0))


class VoskSpeakerEmbedder:
    """Purpose:
        Extract pretrained Vosk Speaker ID vectors from normalized recordings.

    Ingress:
        Normalized WAV paths and configured Vosk ASR/speaker model directories.

    Egress:
        L2-normalized speaker embeddings extracted from decoder results.
    """
    _asr_model = None
    _spk_model = None
    _lock = threading.Lock()

    @classmethod
    def _get_models(cls):
        """Purpose:
            Load shared Vosk ASR and speaker models once under a lock.

        Ingress:
            Model directories from application settings.

        Egress:
            The cached ASR/speaker model pair; raises EmbeddingError for missing files.
        """
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
        """Purpose:
            Collect all usable speaker vectors emitted while decoding a WAV.

        Ingress:
            A normalized WAV path.

        Egress:
            A list of normalized Vosk speaker embeddings.
        """
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
        """Purpose:
            Append a usable Vosk speaker vector while ignoring malformed decoder output.

        Ingress:
            A mutable embedding accumulator and raw Vosk JSON.

        Egress:
            None; a valid normalized speaker vector may be appended in place.
        """
        try:
            data = json.loads(raw)
            vector = data.get("spk")
            if vector:
                results.append(l2_normalize(vector))
        except (json.JSONDecodeError, TypeError, ValueError):
            return
