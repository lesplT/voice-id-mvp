from __future__ import annotations

import tempfile
from pathlib import Path

from voice_id_mvp.recognition.asr import transcribe_wav
from voice_id_mvp.audio.processing import write_wav_slice
from voice_id_mvp.recognition.identification import identify_wav
from voice_id_mvp.storage.voice_repository import VoiceRepository
from voice_id_mvp.audio.segmentation import TimeSegment, detect_speech_segments
from voice_id_mvp.core.config import get_settings
from voice_id_mvp.recognition.nemotron import diarize_nemotron


def merge_dialogue_segments(segments: list[dict], max_gap: float = 0.5) -> list[dict]:
    """Purpose:
        Merge nearby turns only when the same known speaker is identified.

    Ingress:
        Ordered segment dictionaries and the maximum permitted inter-turn gap.

    Egress:
        A new list of merged turns with combined text, span, part count, and score.
    """
    merged: list[dict] = []
    for segment in segments:
        if (
            merged
            and segment["speaker_id"] != "UNKNOWN"
            and merged[-1]["speaker_id"] == segment["speaker_id"]
            and segment["start"] - merged[-1]["end"] <= max_gap
        ):
            merged[-1]["end"] = segment["end"]
            merged[-1]["text"] = " ".join(
                p for p in (merged[-1]["text"], segment["text"]) if p
            )
            merged[-1]["parts"] += 1
            merged[-1]["score"] = max(
                [x for x in (merged[-1].get("score"), segment.get("score")) if x is not None],
                default=None,
            )
        else:
            merged.append({**segment, "parts": 1})
    return merged


def legacy_timings(wav_path: Path, repository) -> list[TimeSegment]:
    """Purpose: Reuse the production speech and speaker-change detection in evaluations.
    Ingress: Normalized audio and a store of enrolled reference vectors.
    Egress: Legacy intervals before identification and transcription.
    """
    intervals = detect_speech_segments(wav_path)
    if intervals and get_settings().speaker_engine == "ecapa":
        from voice_id_mvp.recognition.speaker_turns import split_speaker_turns
        return split_speaker_turns(wav_path, intervals, repository.reference_vectors())
    return intervals


def clean_speaker_interval(timing: TimeSegment, other_turns: list[TimeSegment]) -> tuple[TimeSegment | None, bool]:
    """Purpose: Find the largest portion of a turn without competing speakers.
    Ingress: One diarized turn and the other speakers' intervals.
    Egress: The longest unmixed interval and whether any overlap was present.
    """
    candidates = []
    overlap = False
    cursor = timing.start
    for other in sorted(other_turns, key=lambda turn: (turn.start, turn.end)):
        if other.end <= timing.start or other.start >= timing.end:
            continue
        overlap = True
        if other.start > cursor:
            candidates.append(TimeSegment(cursor, other.start))
        cursor = max(cursor, other.end)
    if cursor < timing.end:
        candidates.append(TimeSegment(cursor, timing.end))
    return max(candidates, key=lambda interval: interval.duration, default=None), overlap


def dialogue_from_wav(wav_path: Path, diarization_engine: str | None = None) -> dict:
    """Purpose:
        Combine speech segmentation, acoustic speaker ID, and Russian transcription.

    Ingress:
        A normalized WAV recording and an optional override for the diarization engine.

    Egress:
        A dialogue dictionary containing timestamped speaker/text segments; UNKNOWN turns remain included.
    """
    settings = get_settings()
    engine = diarization_engine or settings.diarization_engine
    repository = VoiceRepository()
    if engine == "nemotron":
        turns = diarize_nemotron(wav_path)
    elif engine == "legacy":
        turns = [(timing, None) for timing in legacy_timings(wav_path, repository)]
    else:
        raise ValueError(f"Unknown diarization engine: {engine}")
    if not turns:
        return {"segments": [], "text": "", "reason": "no_speech", "diarization_engine": engine}
    tracks: dict[str, list[TimeSegment]] = {}
    for interval, track in turns:
        if track is not None:
            tracks.setdefault(track, []).append(interval)
    competing = {name: [interval for other, intervals in tracks.items() if other != name for interval in intervals]
                 for name in tracks}
    rendered: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="voice-dialogue-") as tmp:
        for timing, track in turns:
            part = Path(tmp) / "segment.wav"
            write_wav_slice(wav_path, part, timing.start, timing.end)
            clean, overlapping = clean_speaker_interval(timing, competing.get(track, []))
            if not overlapping:
                identity = identify_wav(part, repository=repository)
            elif clean is not None and round(clean.duration, 5) >= settings.voice_id_min_seconds:
                clean_path = write_wav_slice(wav_path, Path(tmp) / "clean.wav", clean.start, clean.end)
                identity = identify_wav(clean_path, repository=repository)
            else:
                identity = {"speaker_id": "UNKNOWN", "reason": "overlapping_speech", "score": None}
            transcript = transcribe_wav(part)
            rendered.append(
                {
                    "start": round(timing.start, 3),
                    "end": round(timing.end, 3),
                    "speaker_id": identity["speaker_id"],
                    "reason": identity.get("reason"),
                    "score": identity.get("score"),
                    "text": transcript["text"],
                    "asr_engine": transcript["engine"],
                    **({"diarization_label": track, "overlapping_speech": overlapping} if track else {}),
                }
            )
    merged = (merge_dialogue_segments(rendered) if engine == "legacy"
              else [{**item, "parts": 1} for item in rendered])
    lines = [
        f"[{item['start']:.2f}–{item['end']:.2f}] "
        f"{item.get('diarization_label', 'UNKNOWN') if item['speaker_id'] == 'UNKNOWN' else item['speaker_id']}: {item['text']}"
        for item in merged
    ]
    return {"segments": merged, "text": "\n".join(lines), "reason": None,
            "speaker_engine": settings.speaker_engine, "diarization_engine": engine}
