from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from face_pipeline.detector import (
    DEFAULT_DETECTOR_MODEL,
    FaceDetection,
    YuNetFaceDetector,
    draw_detections,
)
from face_pipeline.embedder import DEFAULT_EMBEDDING_MODEL, SFaceEmbedder
from face_pipeline.profiles import DEFAULT_PROFILE_DIR, ProfileStore


@dataclass(frozen=True)
class EnrollmentSample:
    aligned_face: NDArray[np.uint8]
    embedding: NDArray[np.float32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enroll a named person from images or a webcam."
    )
    parser.add_argument("--name", required=True, help="Person's display name")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--image",
        action="append",
        type=Path,
        help="Image containing exactly one face; repeat for multiple samples",
    )
    source.add_argument("--camera", type=int, help="Webcam index, usually 0")
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Webcam samples to capture (default: 5)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_PROFILE_DIR,
        help=f"Profile storage directory (default: {DEFAULT_PROFILE_DIR})",
    )
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


def extract_single_sample(
    frame: NDArray[np.uint8],
    detector: YuNetFaceDetector,
    embedder: SFaceEmbedder,
) -> EnrollmentSample:
    detections = detector.detect(frame)
    if len(detections) != 1:
        raise ValueError(
            f"Enrollment requires exactly one detected face; found {len(detections)}"
        )
    return sample_from_detection(frame, detections[0], embedder)


def sample_from_detection(
    frame: NDArray[np.uint8],
    detection: FaceDetection,
    embedder: SFaceEmbedder,
) -> EnrollmentSample:
    aligned_face = embedder.align(frame, detection)
    embedding = embedder.extract_from_aligned(aligned_face)
    return EnrollmentSample(aligned_face=aligned_face, embedding=embedding)


def samples_from_images(
    image_paths: list[Path],
    detector: YuNetFaceDetector,
    embedder: SFaceEmbedder,
) -> list[EnrollmentSample]:
    samples: list[EnrollmentSample] = []
    for image_path in image_paths:
        resolved_path = image_path.expanduser().resolve()
        frame = cv.imread(str(resolved_path))
        if frame is None:
            raise SystemExit(f"Could not read image: {resolved_path}")
        try:
            sample = extract_single_sample(frame, detector, embedder)
        except ValueError as error:
            raise SystemExit(f"{resolved_path}: {error}") from error
        samples.append(sample)
        print(f"Prepared enrollment sample: {resolved_path}")
    return samples


def samples_from_camera(
    camera_index: int,
    sample_count: int,
    detector: YuNetFaceDetector,
    embedder: SFaceEmbedder,
) -> list[EnrollmentSample]:
    if sample_count < 1:
        raise SystemExit("--samples must be at least 1")
    camera = cv.VideoCapture(camera_index)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera index {camera_index}")

    samples: list[EnrollmentSample] = []
    print("Press SPACE to capture a sample and q to cancel.")
    try:
        while len(samples) < sample_count:
            success, frame = camera.read()
            if not success:
                raise SystemExit("Could not read a frame from the camera")
            frame = cv.flip(frame, 1)
            detections = detector.detect(frame)
            annotated = draw_detections(frame, detections)
            instruction = (
                f"samples: {len(samples)}/{sample_count} | "
                "show exactly one face | SPACE capture | q cancel"
            )
            cv.putText(
                annotated,
                instruction,
                (12, 28),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )
            cv.imshow("Face enrollment", annotated)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                raise SystemExit("Enrollment cancelled; no profile was saved")
            if key == ord(" "):
                if len(detections) != 1:
                    print(f"Capture skipped: detected {len(detections)} faces")
                    continue
                samples.append(sample_from_detection(frame, detections[0], embedder))
                print(f"Captured sample {len(samples)}/{sample_count}")
    finally:
        camera.release()
        cv.destroyAllWindows()
    return samples


def main() -> None:
    args = parse_args()
    detector = YuNetFaceDetector(args.detector_model.expanduser().resolve())
    embedder = SFaceEmbedder(args.embedding_model.expanduser().resolve())

    if args.image:
        samples = samples_from_images(args.image, detector, embedder)
    else:
        samples = samples_from_camera(
            args.camera,
            args.samples,
            detector,
            embedder,
        )

    store = ProfileStore(args.data_dir)
    try:
        loaded_profile = store.create(
            name=args.name,
            embeddings=[sample.embedding for sample in samples],
            aligned_faces=[sample.aligned_face for sample in samples],
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    profile = loaded_profile.profile
    print(
        f"Enrolled {profile.name!r} with {profile.sample_count} sample(s) "
        f"at {profile.directory}"
    )


if __name__ == "__main__":
    main()

