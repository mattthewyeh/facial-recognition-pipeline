from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from face_pipeline.embedder import normalize_embedding
from face_pipeline.profiles import LoadedProfile, ProfileStore


DEFAULT_COSINE_THRESHOLD = 0.363


@dataclass(frozen=True)
class MatchResult:
    profile_id: str | None
    name: str
    similarity: float | None
    recognized: bool


def cosine_similarity(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
) -> float:
    left_normalized = normalize_embedding(left)
    right_normalized = normalize_embedding(right)
    if left_normalized.shape != right_normalized.shape:
        raise ValueError(
            "Embedding dimensions must match: "
            f"{left_normalized.size} != {right_normalized.size}"
        )
    return float(np.dot(left_normalized, right_normalized))


class FaceMatcher:
    def __init__(
        self,
        profiles: list[LoadedProfile],
        threshold: float = DEFAULT_COSINE_THRESHOLD,
    ) -> None:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("Cosine threshold must be between -1 and 1")
        self.profiles = profiles
        self.threshold = threshold

    @classmethod
    def from_store(
        cls,
        store: ProfileStore,
        threshold: float = DEFAULT_COSINE_THRESHOLD,
    ) -> FaceMatcher:
        return cls(store.load_all(), threshold)

    def match(self, query: NDArray[np.floating]) -> MatchResult:
        normalized_query = normalize_embedding(query)
        best_profile: LoadedProfile | None = None
        best_similarity: float | None = None

        for profile in self.profiles:
            centroid = profile.centroid
            if centroid.shape != normalized_query.shape:
                continue
            similarity = float(np.dot(normalized_query, centroid))
            if best_similarity is None or similarity > best_similarity:
                best_profile = profile
                best_similarity = similarity

        if best_profile is None or best_similarity is None:
            return MatchResult(
                profile_id=None,
                name="Unknown",
                similarity=None,
                recognized=False,
            )

        recognized = best_similarity >= self.threshold
        return MatchResult(
            profile_id=best_profile.profile.profile_id if recognized else None,
            name=best_profile.profile.name if recognized else "Unknown",
            similarity=best_similarity,
            recognized=recognized,
        )

