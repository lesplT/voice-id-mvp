import wave
from pathlib import Path

from voice_id_mvp.asr import GigaAmAsr


def test_long_recording_is_split_below_upstream_limit(tmp_path, monkeypatch):
    source = tmp_path / "long.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(bytes(46 * 32000))
    lengths = []

    class Model:
        def transcribe(self, path):
            with wave.open(path, "rb") as wav:
                lengths.append(wav.getnframes() / wav.getframerate())
            return "fragment"

    monkeypatch.setattr(GigaAmAsr, "_get_model", classmethod(lambda cls: Model()))
    result = GigaAmAsr().transcribe(Path(source))
    assert lengths == [20, 20, 6]
    assert result["text"] == "fragment fragment fragment"
