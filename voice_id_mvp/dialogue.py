from __future__ import annotations

import tempfile
from pathlib import Path

from .asr import transcribe_wav
from .audio import write_wav_slice
from .identification import identify_wav
from .repository import VoiceRepository
from .segmentation import detect_speech_segments
from .config import get_settings


def merge_dialogue_segments(segments: list[dict], max_gap: float = 0.5) -> list[dict]:
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


def dialogue_from_wav(wav_path: Path) -> dict:
    time_segments = detect_speech_segments(wav_path)
    if not time_segments:
        return {"segments": [], "text": "", "reason": "no_speech"}
    repository = VoiceRepository()
    if get_settings().speaker_engine == "ecapa":
        from .speaker_turns import split_speaker_turns
        time_segments = split_speaker_turns(wav_path, time_segments, repository.reference_vectors())
    rendered: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="voice-dialogue-") as tmp:
        for index, timing in enumerate(time_segments):
            part = Path(tmp) / f"segment-{index:04d}.wav"
            write_wav_slice(wav_path, part, timing.start, timing.end)
            identity = identify_wav(part, repository=repository)
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
                }
            )
    merged = merge_dialogue_segments(rendered)
    lines = [
        f"[{item['start']:.2f}–{item['end']:.2f}] {item['speaker_id']}: {item['text']}"
        for item in merged
    ]
    return {"segments": merged, "text": "\n".join(lines), "reason": None,
            "speaker_engine": get_settings().speaker_engine}
