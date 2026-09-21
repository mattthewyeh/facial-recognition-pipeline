from pathlib import Path

import numpy as np
import pytest

from face_pipeline.matcher import FaceMatcher, cosine_similarity
from face_pipeline.profiles import LoadedProfile, Profile


def loaded_profile(
    profile_id: str,
    name: str,
    embeddings: list[list[float]],
) -> LoadedProfile:
    matrix = np.asarray(embeddings, dtype=np.float32)
    return LoadedProfile(
        profile=Profile(
            profile_id=profile_id,
            name=name,
            created_at="2026-09-20T00:00:00+00:00",
            embedding_model="SFace 2021-12",
            sample_count=len(embeddings),
            directory=Path("/tmp") / profile_id,
        ),
        embeddings=matrix,
    )


def test_cosine_similarity_for_same_direction() -> None:
    score = cosine_similarity(
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([2.0, 0.0], dtype=np.float32),
    )
    assert score == pytest.approx(1.0)


def test_cosine_similarity_rejects_different_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions must match"):
        cosine_similarity(
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )


def test_matcher_selects_highest_similarity_profile() -> None:
    matcher = FaceMatcher(
        profiles=[
            loaded_profile("one", "Matthew", [[1.0, 0.0], [0.9, 0.1]]),
            loaded_profile("two", "Other", [[0.0, 1.0]]),
        ],
        threshold=0.5,
    )

    result = matcher.match(np.array([0.95, 0.05], dtype=np.float32))

    assert result.recognized is True
    assert result.profile_id == "one"
    assert result.name == "Matthew"
    assert result.similarity is not None and result.similarity > 0.99


def test_matcher_labels_best_candidate_unknown_below_threshold() -> None:
    matcher = FaceMatcher(
        profiles=[loaded_profile("one", "Matthew", [[1.0, 0.0]])],
        threshold=0.8,
    )

    result = matcher.match(np.array([0.5, 0.866], dtype=np.float32))

    assert result.recognized is False
    assert result.profile_id is None
    assert result.name == "Unknown"
    assert result.similarity == pytest.approx(0.5, abs=0.001)


def test_matcher_returns_unknown_without_compatible_profiles() -> None:
    matcher = FaceMatcher(
        profiles=[loaded_profile("one", "Old model", [[1.0, 0.0, 0.0]])],
        threshold=0.3,
    )

    result = matcher.match(np.array([1.0, 0.0], dtype=np.float32))

    assert result.recognized is False
    assert result.name == "Unknown"
    assert result.similarity is None


def test_matcher_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="between -1 and 1"):
        FaceMatcher([], threshold=1.5)

