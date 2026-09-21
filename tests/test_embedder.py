import numpy as np
import pytest

from face_pipeline.embedder import normalize_embedding


def test_normalize_embedding_returns_flat_unit_vector() -> None:
    embedding = np.array([[3.0, 4.0]], dtype=np.float32)

    normalized = normalize_embedding(embedding)

    assert normalized.shape == (2,)
    assert normalized.dtype == np.float32
    assert normalized.tolist() == pytest.approx([0.6, 0.8])
    assert np.linalg.norm(normalized) == pytest.approx(1.0)


def test_normalize_embedding_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        normalize_embedding(np.zeros(8, dtype=np.float32))


def test_normalize_embedding_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite values"):
        normalize_embedding(np.array([1.0, np.nan], dtype=np.float32))

