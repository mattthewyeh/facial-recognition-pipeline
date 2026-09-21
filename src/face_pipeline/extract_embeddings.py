from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

from face_pipeline.detector import DEFAULT_DETECTOR_MODEL, YuNetFaceDetector
from face_pipeline.embedder import DEFAULT_EMBEDDING_MODEL, SFaceEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align detected faces and extract normalized SFace embeddings."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=DEFAULT_DETECTOR_MODEL,
        help=f"YuNet model path (default: {DEFAULT_DETECTOR_MODEL})",
    )
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"SFace model path (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional .npz path for embeddings, boxes, and confidence scores",
    )
    parser.add_argument(
        "--aligned-dir",
        type=Path,
        help="Optional directory for aligned face images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    frame = cv.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path}")

    detector = YuNetFaceDetector(args.detector_model.expanduser().resolve())
    embedder = SFaceEmbedder(args.embedding_model.expanduser().resolve())
    detections = detector.detect(frame)
    if not detections:
        raise SystemExit("No faces detected; no embeddings were created")

    embeddings: list[np.ndarray] = []
    boxes: list[tuple[int, int, int, int]] = []
    scores: list[float] = []

    aligned_dir = args.aligned_dir.expanduser().resolve() if args.aligned_dir else None
    if aligned_dir is not None:
        aligned_dir.mkdir(parents=True, exist_ok=True)

    for index, detection in enumerate(detections, start=1):
        aligned = embedder.align(frame, detection)
        embedding = embedder.extract_from_aligned(aligned)
        embeddings.append(embedding)
        boxes.append(detection.box)
        scores.append(detection.score)
        print(
            f"Face {index}: dimensions={embedding.size} "
            f"norm={np.linalg.norm(embedding):.4f} "
            f"confidence={detection.score:.4f}"
        )

        if aligned_dir is not None:
            aligned_path = aligned_dir / f"face_{index}.jpg"
            if not cv.imwrite(str(aligned_path), aligned):
                raise SystemExit(f"Could not save aligned face: {aligned_path}")
            print(f"Saved aligned face: {aligned_path}")

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        if output_path.suffix.lower() != ".npz":
            raise SystemExit("Embedding output must use the .npz extension")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            embeddings=np.stack(embeddings),
            boxes=np.asarray(boxes, dtype=np.int32),
            scores=np.asarray(scores, dtype=np.float32),
        )
        print(f"Saved embedding data: {output_path}")


if __name__ == "__main__":
    main()

