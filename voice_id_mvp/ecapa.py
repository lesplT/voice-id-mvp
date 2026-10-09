"""Pretrained SpeechBrain ECAPA, loaded without remote code or HF symlinks."""
from pathlib import Path
import threading
import wave
import numpy as np
from .config import get_settings
from .embeddings import EmbeddingError, l2_normalize


class EcapaSpeakerEmbedder:
    _modules = None
    _lock = threading.RLock()

    @classmethod
    def _load(cls):
        with cls._lock:
            if cls._modules is None:
                import torch, torchaudio
                # SpeechBrain 1.0.3 checks an API removed by recent torchaudio.
                # Audio is read with wave; this shim is only a capability check.
                if not hasattr(torchaudio, "list_audio_backends"):
                    torchaudio.list_audio_backends = lambda: ["soundfile"]
                from speechbrain.lobes.features import Fbank
                from speechbrain.processing.features import InputNormalization
                from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
                checkpoint = get_settings().ecapa_model_path
                if not checkpoint.exists():
                    raise EmbeddingError(f"ECAPA checkpoint missing: {checkpoint}. Run scripts/download_ecapa.py")
                torch.set_num_threads(4)
                model = ECAPA_TDNN(input_size=80, channels=[1024,1024,1024,1024,3072],
                    kernel_sizes=[5,3,3,3,1], dilations=[1,2,3,4,1],
                    attention_channels=128, lin_neurons=192).eval()
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                cls._modules = (Fbank(n_mels=80).eval(),
                    InputNormalization(norm_type="sentence", std_norm=False).eval(), model)
            return cls._modules

    def extract_samples(self, samples: np.ndarray) -> list[float]:
        import torch
        if len(samples) < 5600:
            raise EmbeddingError("ECAPA needs at least 0.35 seconds")
        features, normalization, model = self._load()
        with self._lock, torch.inference_mode():
            audio = torch.from_numpy(np.asarray(samples, dtype=np.float32).copy()).unsqueeze(0)
            lengths = torch.ones(1)
            vector = model(normalization(features(audio), lengths), lengths).reshape(-1).numpy()
        return l2_normalize(vector)

    def extract_all(self, wav_path: Path) -> list[list[float]]:
        with wave.open(str(wav_path), "rb") as wav:
            if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise EmbeddingError("Expected 16 kHz mono PCM16 WAV")
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)/32768
        if len(samples) < 5600:
            return []
        # Bound memory for long uploads; enrollment keeps several independent samples.
        chunks = [samples[i:i+960000] for i in range(0, len(samples), 960000)]
        return [self.extract_samples(chunk) for chunk in chunks if len(chunk) >= 5600]
