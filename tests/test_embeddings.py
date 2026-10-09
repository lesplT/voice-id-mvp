import numpy as np
import pytest

from voice_id_mvp.embeddings import l2_normalize, normalized_centroid


def test_l2_normalize() -> None:
    vector = l2_normalize([3.0, 4.0])
    assert vector == pytest.approx([0.6, 0.8])
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_centroid_normalizes_every_input_and_output() -> None:
    centroid = normalized_centroid([[10.0, 0.0], [0.0, 2.0]])
    assert centroid == pytest.approx([2**-0.5, 2**-0.5])
    assert np.linalg.norm(centroid) == pytest.approx(1.0)


def test_zero_vector_rejected() -> None:
    with pytest.raises(ValueError):
        l2_normalize([0.0, 0.0])

