from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from numpy.typing import NDArray

from face_pipeline.matcher import cosine_similarity


@dataclass(frozen=True)
class EmbeddingRecord:
    identity: str
    image_path: Path
    embedding: NDArray[np.float32]
    latency_ms: float


@dataclass(frozen=True)
class PairScore:
    first_image: str
    second_image: str
    first_identity: str
    second_identity: str
    same_identity: bool
    similarity: float


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    true_accepts: int
    false_rejects: int
    true_rejects: int
    false_accepts: int
    accuracy: float
    balanced_accuracy: float
    false_accept_rate: float
    false_reject_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def build_pair_scores(records: list[EmbeddingRecord]) -> list[PairScore]:
    scores: list[PairScore] = []
    for first_index, first in enumerate(records):
        for second in records[first_index + 1 :]:
            scores.append(
                PairScore(
                    first_image=str(first.image_path),
                    second_image=str(second.image_path),
                    first_identity=first.identity,
                    second_identity=second.identity,
                    same_identity=first.identity == second.identity,
                    similarity=cosine_similarity(first.embedding, second.embedding),
                )
            )
    return scores


def calculate_threshold_metrics(
    pairs: list[PairScore],
    threshold: float,
) -> ThresholdMetrics:
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("Cosine threshold must be between -1 and 1")

    genuine = [pair for pair in pairs if pair.same_identity]
    impostor = [pair for pair in pairs if not pair.same_identity]
    if not genuine or not impostor:
        raise ValueError(
            "Evaluation requires same-person and different-person image pairs"
        )

    true_accepts = sum(pair.similarity >= threshold for pair in genuine)
    false_rejects = len(genuine) - true_accepts
    false_accepts = sum(pair.similarity >= threshold for pair in impostor)
    true_rejects = len(impostor) - false_accepts
    total = len(genuine) + len(impostor)
    true_accept_rate = true_accepts / len(genuine)
    true_reject_rate = true_rejects / len(impostor)

    return ThresholdMetrics(
        threshold=threshold,
        true_accepts=true_accepts,
        false_rejects=false_rejects,
        true_rejects=true_rejects,
        false_accepts=false_accepts,
        accuracy=(true_accepts + true_rejects) / total,
        balanced_accuracy=(true_accept_rate + true_reject_rate) / 2.0,
        false_accept_rate=false_accepts / len(impostor),
        false_reject_rate=false_rejects / len(genuine),
    )


def recommend_threshold(pairs: list[PairScore]) -> ThresholdMetrics:
    similarities = sorted({pair.similarity for pair in pairs})
    if not similarities:
        raise ValueError("At least one comparison pair is required")

    candidates = {-1.0, 1.0}
    candidates.update(
        (left + right) / 2.0
        for left, right in zip(similarities, similarities[1:])
    )
    evaluated = [
        calculate_threshold_metrics(pairs, threshold)
        for threshold in sorted(candidates)
    ]
    return max(
        evaluated,
        key=lambda metrics: (
            metrics.balanced_accuracy,
            metrics.accuracy,
            -metrics.false_accept_rate,
            metrics.threshold,
        ),
    )


def summarize_scores(pairs: list[PairScore], same_identity: bool) -> dict[str, float | int]:
    scores = [pair.similarity for pair in pairs if pair.same_identity == same_identity]
    if not scores:
        raise ValueError("No matching comparison pairs were found")
    return {
        "count": len(scores),
        "minimum": min(scores),
        "maximum": max(scores),
        "mean": mean(scores),
    }
