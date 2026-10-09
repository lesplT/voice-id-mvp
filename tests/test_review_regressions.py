from voice_id_mvp.identification import aggregate_hits, decide_identity
from voice_id_mvp.dialogue import merge_dialogue_segments

def test_one_accidental_high_match_does_not_win():
    hits = [{"speaker_id": "outlier", "score": x} for x in [.99, .5, .4]]
    hits += [{"speaker_id": "consistent", "score": x} for x in [.86, .85, .84]]
    candidates = aggregate_hits(hits)
    assert candidates[0]["speaker_id"] == "consistent"
    assert decide_identity(candidates, .72, .05)["speaker_id"] == "consistent"

def test_centroid_cannot_duplicate_reference_evidence():
    hits = [{"speaker_id": "a", "score": .6, "kind": "reference"},
            {"speaker_id": "a", "score": .99, "kind": "centroid"}]
    result = aggregate_hits(hits)
    assert len(result) == 1
    assert result[0]["score"] == .6
    assert decide_identity(result, .72, .05)["reason"] == "below_threshold"

def test_unknown_segments_do_not_imply_same_speaker():
    rows = [dict(start=0, end=1, speaker_id="UNKNOWN", text="a", score=None),
            dict(start=1.1, end=2, speaker_id="UNKNOWN", text="b", score=None)]
    assert len(merge_dialogue_segments(rows)) == 2
