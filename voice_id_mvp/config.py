from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    asr_engine: str = "gigaam"
    speaker_engine: Literal["ecapa", "vosk"] = "ecapa"
    ecapa_model_path: Path = ROOT / "models/ecapa/embedding_model.ckpt"
    asr_fallback_to_vosk: bool = True
    gigaam_model_name: str = "v3_e2e_rnnt"
    gigaam_cache_path: Path = ROOT / "models/gigaam"
    vosk_spk_model_path: Path = ROOT / "models/vosk-model-spk-0.4"
    vosk_asr_model_path: Path = ROOT / "models/vosk-model-small-ru-0.22"
    ffmpeg_bin: str = "ffmpeg"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_local_path: Path = ROOT / "data/qdrant_local"
    qdrant_collection: str = "voice_references_ecapa_v1"
    dictionary_path: Path = ROOT / "data/dictionary"
    journal_path: Path = ROOT / "data/journal"
    enrollment_service_url: str = "http://127.0.0.1:8001"
    identification_service_url: str = "http://127.0.0.1:8002"
    transcription_service_url: str = "http://127.0.0.1:8003"
    voice_id_threshold: float = 0.45
    voice_id_margin: float = 0.12
    voice_id_min_seconds: float = 0.35
    voice_id_top_k: int = 20
    vad_frame_ms: int = 30
    vad_min_segment_seconds: float = 0.35
    vad_max_segment_seconds: float = 20.0
    vad_padding_ms: int = 240
    vad_min_dbfs: float = -46.0
    vad_noise_margin_db: float = 9.0

    def model_post_init(self, __context) -> None:
        for name in ("vosk_spk_model_path", "vosk_asr_model_path", "qdrant_local_path", "gigaam_cache_path", "ecapa_model_path", "dictionary_path", "journal_path"):
            p = Path(getattr(self, name))
            if not p.is_absolute():
                setattr(self, name, (ROOT / p).resolve())


@lru_cache
def get_settings() -> Settings:
    return Settings()
