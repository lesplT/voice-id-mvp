from __future__ import annotations

from pathlib import Path

from voice_id_mvp.audio.processing import wav_duration
from voice_id_mvp.core.config import get_settings
from voice_id_mvp.recognition.embeddings import VoskSpeakerEmbedder, normalized_centroid, get_embedder
from voice_id_mvp.storage.voice_repository import VoiceRepository
from voice_id_mvp.audio.segmentation import has_speech


UNKNOWN = "UNKNOWN"


def aggregate_hits(hits: list[dict]) -> list[dict]:
    """Purpose:
        Rank speakers using their best reference scores without counting duplicate centroids.

    Ingress:
        Vector-search hits containing speaker IDs, kinds, and cosine scores.

    Egress:
        Candidate dictionaries sorted by mean top-three reference score.
    """
    grouped: dict[str, list[float]] = {}
    for hit in hits:
        # Centroids duplicate reference evidence; rank by reference points only.
        if hit.get("kind") == "centroid":
            continue
        grouped.setdefault(hit["speaker_id"], []).append(float(hit["score"]))
    candidates = []
    for speaker, scores in grouped.items():
        selected = sorted(scores, reverse=True)[:3]
        candidates.append({
            "speaker_id": speaker,
            "score": sum(selected) / len(selected),
            "best_score": max(scores),
            "hit_count": len(scores),
        })
    return sorted(candidates, key=lambda x: x["score"], reverse=True)


def decide_identity(
    candidates: list[dict], threshold: float, margin: float
) -> dict:
    """Purpose:
        Accept a candidate only when similarity and separation thresholds are satisfied.

    Ingress:
        Ranked speaker candidates, minimum similarity, and minimum runner-up margin.

    Egress:
        A speaker decision or UNKNOWN with an explanatory reason and score.
    """
    if not candidates:
        return {"speaker_id": UNKNOWN, "reason": "no_embedding", "score": None}
    best = candidates[0]
    best = {**best, "best_score": best.get("score", best["best_score"])}
    if best["best_score"] < threshold:
        return {
            "speaker_id": UNKNOWN,
            "reason": "below_threshold",
            "score": best["best_score"],
        }
    if len(candidates) > 1 and best["best_score"] - candidates[1].get("score", candidates[1]["best_score"]) < margin:
        return {
            "speaker_id": UNKNOWN,
            "reason": "ambiguous",
            "score": best["best_score"],
        }
    return {
        "speaker_id": best["speaker_id"],
        "reason": None,
        "score": best["best_score"],
    }


def identify_wav(
    wav_path: Path,
    repository: VoiceRepository | None = None,
    embedder: VoskSpeakerEmbedder | None = None,
    top_k: int | None = None,
) -> dict:
    """Purpose:
        Identify a speaker conservatively, rejecting insufficient speech before embedding.

    Ingress:
        A normalized WAV path and optional repository, embedder, and search-limit overrides.

    Egress:
        A decision dictionary with duration, ranked candidates, and embedding metadata.
    """
    settings = get_settings()
    duration = wav_duration(wav_path)
    if duration < settings.voice_id_min_seconds:
        return {
            "speaker_id": UNKNOWN, "reason": "too_short", "score": None,
            "duration_seconds": duration, "candidates": [],
        }
    if not has_speech(wav_path):
        return {
            "speaker_id": UNKNOWN, "reason": "no_speech", "score": None,
            "duration_seconds": duration, "candidates": [],
        }
    embeddings = (embedder or get_embedder()).extract_all(wav_path)
    if not embeddings:
        return {
            "speaker_id": UNKNOWN, "reason": "no_embedding", "score": None,
            "duration_seconds": duration, "candidates": [],
        }
    query = normalized_centroid(embeddings)
    hits = (repository or VoiceRepository()).search(
        query, limit=top_k or settings.voice_id_top_k
    )
    candidates = aggregate_hits(hits)
    decision = decide_identity(
        candidates, settings.voice_id_threshold, settings.voice_id_margin
    )
    return {
        **decision,
        "duration_seconds": duration,
        "embedding_count": len(embeddings),
        "candidates": candidates,
    }
