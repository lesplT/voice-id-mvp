from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import get_settings


@dataclass(frozen=True)
class TimeSegment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def _dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    return 20.0 * math.log10(max(rms, 1.0) / 32768.0)


def detect_speech_segments(wav_path: Path) -> list[TimeSegment]:
    settings = get_settings()
    with wave.open(str(wav_path), "rb") as wav:
        rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    frame_size = max(1, int(rate * settings.vad_frame_ms / 1000))
    levels = np.asarray(
        [_dbfs(samples[i : i + frame_size]) for i in range(0, len(samples), frame_size)]
    )
    if levels.size == 0:
        return []
    noise_floor = float(np.percentile(levels, 25))
    threshold = max(settings.vad_min_dbfs, noise_floor + settings.vad_noise_margin_db)
    active = levels >= threshold
    if not bool(active.any()):
        return []

    pad_frames = max(1, math.ceil(settings.vad_padding_ms / settings.vad_frame_ms))
    expanded = active.copy()
    active_indexes = np.flatnonzero(active)
    for index in active_indexes:
        expanded[max(0, index - pad_frames) : min(len(active), index + pad_frames + 1)] = True

    raw: list[TimeSegment] = []
    start = None
    for index, is_active in enumerate(np.append(expanded, False)):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            begin = start * frame_size / rate
            end = min(len(samples) / rate, index * frame_size / rate)
            if end - begin >= settings.vad_min_segment_seconds:
                raw.append(TimeSegment(begin, end))
            start = None

    split: list[TimeSegment] = []
    maximum = settings.vad_max_segment_seconds
    for segment in raw:
        cursor = segment.start
        while segment.end - cursor > maximum:
            split.append(TimeSegment(cursor, cursor + maximum))
            cursor += maximum
        if segment.end - cursor >= settings.vad_min_segment_seconds:
            split.append(TimeSegment(cursor, segment.end))
    return split


def has_speech(wav_path: Path) -> bool:
    return bool(detect_speech_segments(wav_path))

