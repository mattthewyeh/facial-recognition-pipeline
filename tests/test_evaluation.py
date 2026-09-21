from pathlib import Path

import numpy as np
import pytest

from face_pipeline.evaluation import (
    EmbeddingRecord,
    build_pair_scores,
    calculate_threshold_metrics,
    recommend_threshold,
)


def record(identity: str, filename: str, values: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(
        identity=identity,
        image_path=Path(filename),
        embedding=np.asarray(values, dtype=np.float32),
        latency_ms=10.0,
    )


def separated_records() -> list[EmbeddingRecord]:
    return [
        record("person_a", "a1.jpg", [1.0, 0.0]),
        record("person_a", "a2.jpg", [0.99, 0.05]),
        record("person_b", "b1.jpg", [0.0, 1.0]),
        record("person_b", "b2.jpg", [0.05, 0.99]),
    ]


def test_build_pair_scores_creates_genuine_and_impostor_pairs() -> None:
    pairs = build_pair_scores(separated_records())

    assert len(pairs) == 6
    assert sum(pair.same_identity for pair in pairs) == 2
    assert sum(not pair.same_identity for pair in pairs) == 4


def test_threshold_metrics_count_perfect_separation() -> None:
    pairs = build_pair_scores(separated_records())

    metrics = calculate_threshold_metrics(pairs, threshold=0.8)

    assert metrics.true_accepts == 2
    assert metrics.false_rejects == 0
    assert metrics.true_rejects == 4
    assert metrics.false_accepts == 0
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.balanced_accuracy == pytest.approx(1.0)


def test_recommended_threshold_separates_sample_scores() -> None:
    pairs = build_pair_scores(separated_records())

    recommended = recommend_threshold(pairs)
    highest_impostor = max(
        pair.similarity for pair in pairs if not pair.same_identity
    )
    lowest_genuine = min(pair.similarity for pair in pairs if pair.same_identity)

    assert recommended.balanced_accuracy == pytest.approx(1.0)
    assert recommended.false_accepts == 0
    assert recommended.false_rejects == 0
    assert highest_impostor < recommended.threshold < lowest_genuine


def test_threshold_metrics_require_both_pair_types() -> None:
    pairs = build_pair_scores(
        [
            record("person_a", "a1.jpg", [1.0, 0.0]),
            record("person_a", "a2.jpg", [0.9, 0.1]),
        ]
    )

    with pytest.raises(ValueError, match="same-person and different-person"):
        calculate_threshold_metrics(pairs, threshold=0.5)
