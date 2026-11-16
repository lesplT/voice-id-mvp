import wave

import pytest

from voice_id_mvp.benchmark.metrics import diarization_errors, identity_errors, word_errors
from voice_id_mvp.benchmark.runner import run_folder
from voice_id_mvp.application.dialogue import clean_speaker_interval
from voice_id_mvp.audio.segmentation import TimeSegment


def test_word_edits_and_normalization():
    row = word_errors("Ёлка, красная была!", "ёлка синяя была вчера")
    assert row == {"wer": 2/3, "reference_words": 3, "substitutions": 1, "deletions": 0, "insertions": 1}
    assert word_errors("две булки", "")["deletions"] == 2


def test_der_counts_overlap_and_false_speech():
    ref = [(0, 3, "A"), (1, 2, "B")]
    hyp = [(0, 3, "s9"), (3, 4, "s9")]
    row = diarization_errors(ref, hyp, 4)
    assert row["mapping"] == {"s9": "A"}
    assert row["missed_speaker_seconds"] == 1
    assert row["false_alarm_speaker_seconds"] == 1
    assert row["der"] == 0.5
    assert diarization_errors(ref, hyp, 4, skip_overlap=True)["der"] == 0.5
    assert diarization_errors(ref, [(0, 3, "different"), (1, 2, "also")], 4)["der"] == 0


def test_named_identity_is_not_permuted():
    ref = [(0, 2, "A"), (2, 3, "B")]
    hyp = [(0, 1, "A"), (1, 2, "UNKNOWN"), (2, 3, "A")]
    scores = identity_errors(ref, hyp, 3, {"A"})
    assert scores["named_accuracy"] == 0.5
    assert scores["unknown_seconds"] == 1
    assert scores["false_accept_seconds"] == 1


def test_name_lookup_has_unmixed_audio_even_when_turns_overlap():
    interval, overlapping = clean_speaker_interval(
        TimeSegment(1, 8), [TimeSegment(0, 2), TimeSegment(5, 6), TimeSegment(5.5, 9)])
    assert overlapping
    assert interval == TimeSegment(2, 5)
    assert clean_speaker_interval(TimeSegment(2, 3), [TimeSegment(1, 4)]) == (None, True)


def test_folder_counts_without_labels_and_reports_explicit_model_errors(tmp_path, monkeypatch):
    import json
    import shutil
    import voice_id_mvp.benchmark.runner as runner

    source = tmp_path / "input"
    source.mkdir()
    with wave.open(str(source / "a.wav"), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", ""))
        wav.writeframes(b"\0\0" * 16000)
    monkeypatch.setattr(runner, "normalize_audio", shutil.copyfile)
    monkeypatch.setattr(runner.VoskAsr, "transcribe", lambda *_: {"text": "булка готова"})
    def not_installed(*_):
        raise RuntimeError("missing model")
    monkeypatch.setattr(runner, "transcribe_nemotron_directory", not_installed)
    report = tmp_path / "output"
    summary = run_folder(source, report, ("vosk", "nemotron-asr"))
    rows = json.loads((report / "recordings/000001.json").read_text())["models"]
    assert rows["vosk"]["words"] == 2
    assert "metrics" not in rows["vosk"]
    assert summary["asr"]["vosk"]["common_scored_files"] == 0
    assert rows["nemotron-asr"]["error"] == "missing model"
    assert (report / "comparison.csv").is_file()
    with pytest.raises(ValueError, match="empty results folder"):
        run_folder(source, report, ("vosk",))
