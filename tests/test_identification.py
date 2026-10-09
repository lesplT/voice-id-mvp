from voice_id_mvp.identification import UNKNOWN, aggregate_hits, decide_identity


def test_top_k_aggregation_uses_mean_score_per_speaker() -> None:
    result = aggregate_hits(
        [
            {"speaker_id": "platon", "score": 0.81},
            {"speaker_id": "anna", "score": 0.78},
            {"speaker_id": "platon", "score": 0.91},
        ]
    )
    assert result[0]["speaker_id"] == "platon"
    assert abs(result[0]["score"] - 0.86) < 1e-6
    assert result[0]["best_score"] == 0.91
    assert result[0]["hit_count"] == 2


def test_below_threshold_is_unknown() -> None:
    decision = decide_identity(
        [{"speaker_id": "platon", "best_score": 0.69, "hit_count": 1}], 0.72, 0.05
    )
    assert decision["speaker_id"] == UNKNOWN
    assert decision["reason"] == "below_threshold"


def test_ambiguous_is_unknown() -> None:
    decision = decide_identity(
        [
            {"speaker_id": "platon", "best_score": 0.84, "hit_count": 1},
            {"speaker_id": "anna", "best_score": 0.81, "hit_count": 1},
        ],
        0.72,
        0.05,
    )
    assert decision["speaker_id"] == UNKNOWN
    assert decision["reason"] == "ambiguous"


def test_no_hits_is_no_embedding() -> None:
    decision = decide_identity([], 0.72, 0.05)
    assert decision["speaker_id"] == UNKNOWN
    assert decision["reason"] == "no_embedding"

