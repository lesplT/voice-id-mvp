import wave

from voice_id_mvp.dialogue import merge_dialogue_segments
from voice_id_mvp.identification import identify_wav
from voice_id_mvp.segmentation import detect_speech_segments
from voice_id_mvp.audio import write_wav_slice, wav_duration


def test_too_short_does_not_call_models(tmp_path) -> None:
    path = tmp_path / "short.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 3200)
    result = identify_wav(path)
    assert result["speaker_id"] == "UNKNOWN"
    assert result["reason"] == "too_short"


def test_merge_adjacent_same_speaker() -> None:
    merged = merge_dialogue_segments(
        [
            {"start": 0.0, "end": 1.0, "speaker_id": "a", "text": "привет", "score": 0.9},
            {"start": 1.2, "end": 2.0, "speaker_id": "a", "text": "мир", "score": 0.8},
            {"start": 3.0, "end": 4.0, "speaker_id": "b", "text": "ответ", "score": 0.85},
        ]
    )
    assert len(merged) == 2
    assert merged[0]["text"] == "привет мир"
    assert merged[0]["parts"] == 2


def test_silence_is_no_speech(tmp_path) -> None:
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 32000)
    assert detect_speech_segments(path) == []


def test_exact_minimum_slice_does_not_lose_a_frame(tmp_path):
    source, target = tmp_path / "source.wav", tmp_path / "target.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 320000)
    write_wav_slice(source, target, 14.35, 14.70)
    assert wav_duration(target) == .35


def test_slice_preserves_source_sample_rate(tmp_path):
    source,target=tmp_path/'source.wav',tmp_path/'slice.wav'
    with wave.open(str(source),'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(22050)
        wav.writeframes(b'\0\0'*44100)
    write_wav_slice(source,target,.5,1.5)
    assert wav_duration(target)==1
    with wave.open(str(target),'rb') as wav:
        assert wav.getframerate()==22050
