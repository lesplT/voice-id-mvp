"""Measure Vosk speaker extraction without changing application thresholds."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voice_id_mvp.audio import normalize_audio, write_wav_slice, wav_duration
from voice_id_mvp.embeddings import VoskSpeakerEmbedder, normalized_centroid
from voice_id_mvp.segmentation import detect_speech_segments

folder = ROOT / "data/short-evaluation"
folder.mkdir(parents=True, exist_ok=True)
embedder = VoskSpeakerEmbedder()
references = {}
for speaker, filename in [("platon", "test1sound.mp3"), ("prokhor", "test3sound.mp3")]:
    output = folder / (speaker + ".wav")
    normalize_audio(Path("C:/Users/PC/Downloads") / filename, output)
    references[speaker] = normalized_centroid(embedder.extract_all(output))
dialogue = folder / "dialogue.wav"
normalize_audio(Path("C:/Users/PC/Downloads/test4sound.mp3"), dialogue)
rows = []
for timing in detect_speech_segments(dialogue):
    output = folder / "probe.wav"
    write_wav_slice(dialogue, output, timing.start, timing.end)
    vectors = embedder.extract_all(output)
    scores = {}
    if vectors:
        query = normalized_centroid(vectors)
        scores = {key: float(np.dot(query, ref)) for key, ref in references.items()}
    rows.append(dict(start=timing.start, end=timing.end, duration=timing.duration,
                     embedding_count=len(vectors), scores=scores))
report = dict(dialogue_seconds=wav_duration(dialogue), segments=rows)
(folder / "vosk-baseline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
