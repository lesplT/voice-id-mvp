from voice_id_mvp.speaker_turns import stable_boundaries

def test_stable_acoustic_change():
    rows=[(0.4,'a'),(.6,'a'),(.8,'a'),(1.0,None),(1.2,'b'),(1.4,'b')]
    assert stable_boundaries(rows) == [1.0]

def test_single_outlier_is_not_a_turn():
    rows=[(.4,'a'),(.6,'a'),(.8,'b'),(1.0,'a'),(1.2,'a')]
    assert stable_boundaries(rows) == []

def test_uncertainty_does_not_assign_speaker():
    assert stable_boundaries([(.4,None),(.6,None),(.8,'a')]) == []
