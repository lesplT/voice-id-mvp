"""Acoustic speaker-change proposals; never infer identity from transcript text."""
import wave
from pathlib import Path
import numpy as np
from voice_id_mvp.recognition.ecapa import EcapaSpeakerEmbedder
from voice_id_mvp.audio.segmentation import TimeSegment


def stable_boundaries(observations: list[tuple[float, str | None]]) -> list[float]:
    """Purpose:
        Propose speaker-change times only after two consecutive confident observations.

    Ingress:
        Time/label observations; None denotes an uncertain acoustic window.

    Egress:
        Boundary timestamps between stable known-speaker runs; text is never used.
    """
    current = None
    last_time = None
    pending = None
    pending_time = None
    count = 0
    boundaries = []
    for center, speaker in observations:
        if speaker is None:
            pending, count = None, 0
            continue
        if speaker == current:
            last_time = center
            pending, count = None, 0
        else:
            if pending != speaker:
                pending, pending_time, count = speaker, center, 1
            else:
                count += 1
            if count >= 2:
                if current is not None:
                    boundaries.append((last_time + pending_time)/2)
                current, last_time = speaker, center
                pending, count = None, 0
    return boundaries


def split_speaker_turns(wav_path: Path, segments: list[TimeSegment], references: dict) -> list[TimeSegment]:
    """Purpose:
        Split detected speech using acoustic reference comparisons, never transcript meaning.

    Ingress:
        A normalized WAV path, speech intervals, and registered reference vectors.

    Egress:
        TimeSegment intervals with conservative speaker-change proposals.
    """
    if len(references) < 2:
        return segments
    with wave.open(str(wav_path), "rb") as wav:
        rate = wav.getframerate()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)/32768
    names = list(references)
    matrices = [np.asarray(references[name]) for name in names]
    embedder = EcapaSpeakerEmbedder()
    def scores(vector):
        # Same top-three aggregation as identification.
        """Purpose:
            Compare one voice vector with each registered speaker's strongest references.

        Ingress:
            An embedding and the enclosing per-speaker reference matrices.

        Egress:
            A NumPy array of top-three mean similarity scores.
        """
        return np.asarray([np.sort(matrix @ vector)[-3:].mean() for matrix in matrices])
    output = []
    for segment in segments:
        if segment.duration < 1.4:
            output.append(segment)
            continue
        observations = []
        for start in np.arange(segment.start, segment.end-.8+.001, .2):
            vector = np.asarray(embedder.extract_samples(audio[round(start*rate):round((start+.8)*rate)]))
            values = scores(vector)
            order = np.argsort(values)[::-1]
            # Lower threshold ONLY proposes a boundary, never accepts an identity.
            label = names[order[0]] if values[order[0]] >= .30 and values[order[0]]-values[order[1]] >= .12 else None
            observations.append((float(start+.4), label))
        boundaries = stable_boundaries(observations)
        points = [segment.start] + [b for b in boundaries if segment.start+.35 <= b <= segment.end-.35] + [segment.end]
        output.extend(TimeSegment(a,b) for a,b in zip(points, points[1:]) if b-a >= .35)
    return output
