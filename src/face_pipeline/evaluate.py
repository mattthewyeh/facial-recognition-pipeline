from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import cv2 as cv

from face_pipeline.detector import DEFAULT_DETECTOR_MODEL, YuNetFaceDetector
from face_pipeline.embedder import DEFAULT_EMBEDDING_MODEL, SFaceEmbedder
from face_pipeline.evaluation import (
    EmbeddingRecord,
    build_pair_scores,
    calculate_threshold_metrics,
    recommend_threshold,
    summarize_scores,
)
from face_pipeline.matcher import DEFAULT_COSINE_THRESHOLD


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate recognition scores using identity-named image directories."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Directory containing one subdirectory per identity",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_COSINE_THRESHOLD,
        help=f"Threshold to evaluate (default: {DEFAULT_COSINE_THRESHOLD})",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=DEFAULT_DETECTOR_MODEL,
    )
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=DEFAULT_EMBEDDING_MODEL,
    )
    return parser.parse_args()


def discover_images(dataset: Path) -> list[tuple[str, Path]]:
    if not dataset.is_dir():
        raise SystemExit(f"Dataset directory not found: {dataset}")

    images: list[tuple[str, Path]] = []
    for identity_dir in sorted(dataset.iterdir()):
        if not identity_dir.is_dir() or identity_dir.name.startswith("."):
            continue
        identity_images = sorted(
            path
            for path in identity_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        if not identity_images:
            continue
        images.extend((identity_dir.name, image_path) for image_path in identity_images)

    identities = {identity for identity, _ in images}
    if len(identities) < 2:
        raise SystemExit("Evaluation requires images for at least two identities")
    if not any(
        sum(identity == other_identity for other_identity, _ in images) >= 2
        for identity in identities
    ):
        raise SystemExit("Evaluation requires at least two images of one identity")
    return images


def extract_records(
    images: list[tuple[str, Path]],
    detector: YuNetFaceDetector,
    embedder: SFaceEmbedder,
) -> list[EmbeddingRecord]:
    records: list[EmbeddingRecord] = []
    for identity, image_path in images:
        frame = cv.imread(str(image_path))
        if frame is None:
            raise SystemExit(f"Could not read image: {image_path}")

        started_at = time.perf_counter()
        detections = detector.detect(frame)
        if len(detections) != 1:
            raise SystemExit(
                f"{image_path}: expected exactly one face, found {len(detections)}"
            )
        embedding = embedder.extract(frame, detections[0])
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        records.append(
            EmbeddingRecord(
                identity=identity,
                image_path=image_path,
                embedding=embedding,
                latency_ms=latency_ms,
            )
        )
        print(f"Processed {identity}/{image_path.name}: {latency_ms:.2f} ms")
    return records


def build_report(
    dataset: Path,
    records: list[EmbeddingRecord],
    threshold: float,
) -> dict:
    pairs = build_pair_scores(records)
    configured_metrics = calculate_threshold_metrics(pairs, threshold)
    recommended_metrics = recommend_threshold(pairs)
    latencies = [record.latency_ms for record in records]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "identity_count": len({record.identity for record in records}),
        "image_count": len(records),
        "pipeline_performance": {
            "mean_latency_ms": mean(latencies),
            "median_latency_ms": median(latencies),
            "estimated_fps": 1000.0 / mean(latencies),
        },
        "similarity_scores": {
            "genuine_pairs": summarize_scores(pairs, same_identity=True),
            "impostor_pairs": summarize_scores(pairs, same_identity=False),
        },
        "configured_threshold": configured_metrics.to_dict(),
        "recommended_threshold": recommended_metrics.to_dict(),
    }


def print_report(report: dict) -> None:
    performance = report["pipeline_performance"]
    genuine = report["similarity_scores"]["genuine_pairs"]
    impostor = report["similarity_scores"]["impostor_pairs"]
    configured = report["configured_threshold"]
    recommended = report["recommended_threshold"]
    print("\nEvaluation summary")
    print(
        f"  identities={report['identity_count']} images={report['image_count']} "
        f"mean_latency={performance['mean_latency_ms']:.2f} ms "
        f"estimated_fps={performance['estimated_fps']:.2f}"
    )
    print(
        f"  genuine pairs={genuine['count']} mean={genuine['mean']:.4f} "
        f"range=[{genuine['minimum']:.4f}, {genuine['maximum']:.4f}]"
    )
    print(
        f"  impostor pairs={impostor['count']} mean={impostor['mean']:.4f} "
        f"range=[{impostor['minimum']:.4f}, {impostor['maximum']:.4f}]"
    )
    print(
        f"  threshold={configured['threshold']:.4f} "
        f"false_accepts={configured['false_accepts']} "
        f"false_rejects={configured['false_rejects']} "
        f"balanced_accuracy={configured['balanced_accuracy']:.4f}"
    )
    print(
        f"  suggested_threshold={recommended['threshold']:.4f} "
        f"balanced_accuracy={recommended['balanced_accuracy']:.4f}"
    )


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    images = discover_images(dataset)
    detector = YuNetFaceDetector(args.detector_model.expanduser().resolve())
    embedder = SFaceEmbedder(args.embedding_model.expanduser().resolve())
    records = extract_records(images, detector, embedder)
    try:
        report = build_report(dataset, records, args.threshold)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print_report(report)

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved evaluation report: {output_path}")


if __name__ == "__main__":
    main()

