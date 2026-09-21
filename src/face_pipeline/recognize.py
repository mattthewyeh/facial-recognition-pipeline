from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from face_pipeline.detector import (
    DEFAULT_DETECTOR_MODEL,
    FaceDetection,
    YuNetFaceDetector,
    validate_frame,
)
from face_pipeline.embedder import DEFAULT_EMBEDDING_MODEL, SFaceEmbedder
from face_pipeline.matcher import DEFAULT_COSINE_THRESHOLD, FaceMatcher, MatchResult
from face_pipeline.profiles import DEFAULT_PROFILE_DIR, ProfileStore


@dataclass(frozen=True)
class Recognition:
    detection: FaceDetection
    match: MatchResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize enrolled faces using SFace cosine similarity."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Path to an input image")
    source.add_argument("--camera", type=int, help="Webcam index, usually 0")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_PROFILE_DIR,
        help=f"Profile storage directory (default: {DEFAULT_PROFILE_DIR})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_COSINE_THRESHOLD,
        help=(
            "Minimum cosine similarity for a known identity "
            f"(default: {DEFAULT_COSINE_THRESHOLD})"
        ),
    )
    parser.add_argument("--output", type=Path, help="Optional annotated image path")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open an image preview window",
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


def recognize_frame(
    frame: NDArray[np.uint8],
    detector: YuNetFaceDetector,
    embedder: SFaceEmbedder,
    matcher: FaceMatcher,
) -> list[Recognition]:
    detections = detector.detect(frame)
    return [
        Recognition(
            detection=detection,
            match=matcher.match(embedder.extract(frame, detection)),
        )
        for detection in detections
    ]


def draw_recognitions(
    frame: NDArray[np.uint8],
    recognitions: list[Recognition],
) -> NDArray[np.uint8]:
    validate_frame(frame)
    annotated = frame.copy()
    for recognition in recognitions:
        detection = recognition.detection
        match = recognition.match
        color = (0, 200, 0) if match.recognized else (0, 0, 255)
        cv.rectangle(
            annotated,
            (detection.x, detection.y),
            (detection.x + detection.width, detection.y + detection.height),
            color,
            2,
        )
        score = "n/a" if match.similarity is None else f"{match.similarity:.3f}"
        label_y = max(20, detection.y - 8)
        cv.putText(
            annotated,
            f"{match.name} {score}",
            (detection.x, label_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv.LINE_AA,
        )
    return annotated


def print_recognitions(recognitions: list[Recognition]) -> None:
    print(f"Detected faces: {len(recognitions)}")
    for index, recognition in enumerate(recognitions, start=1):
        match = recognition.match
        score = "n/a" if match.similarity is None else f"{match.similarity:.4f}"
        print(
            f"  {index}: label={match.name!r} similarity={score} "
            f"box={recognition.detection.box}"
        )


def recognize_image(
    image_path: Path,
    output_path: Path | None,
    display: bool,
    detector: YuNetFaceDetector,
    embedder: SFaceEmbedder,
    matcher: FaceMatcher,
) -> None:
    frame = cv.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path}")
    recognitions = recognize_frame(frame, detector, embedder, matcher)
    print_recognitions(recognitions)
    annotated = draw_recognitions(frame, recognitions)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv.imwrite(str(output_path), annotated):
            raise SystemExit(f"Could not write output image: {output_path}")
        print(f"Saved annotated image: {output_path}")
    if display:
        cv.imshow("SFace recognition", annotated)
        cv.waitKey(0)
        cv.destroyAllWindows()


def recognize_camera(
    camera_index: int,
    detector: YuNetFaceDetector,
    embedder: SFaceEmbedder,
    matcher: FaceMatcher,
) -> None:
    camera = cv.VideoCapture(camera_index)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera index {camera_index}")

    previous_time = time.perf_counter()
    smoothed_fps = 0.0
    print("Press q in the camera window to quit.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise SystemExit("Could not read a frame from the camera")
            frame = cv.flip(frame, 1)
            recognitions = recognize_frame(frame, detector, embedder, matcher)
            annotated = draw_recognitions(frame, recognitions)

            current_time = time.perf_counter()
            elapsed = max(current_time - previous_time, 1e-9)
            current_fps = 1.0 / elapsed
            smoothed_fps = current_fps if smoothed_fps == 0.0 else (
                0.9 * smoothed_fps + 0.1 * current_fps
            )
            previous_time = current_time
            cv.putText(
                annotated,
                f"faces: {len(recognitions)}  fps: {smoothed_fps:.1f}",
                (12, 28),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )
            cv.imshow("SFace recognition", annotated)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.camera is not None and args.no_display:
        raise SystemExit("--no-display can only be used with --image")
    if args.output is not None and args.image is None:
        raise SystemExit("--output can only be used with --image")

    store = ProfileStore(args.data_dir)
    matcher = FaceMatcher.from_store(store, threshold=args.threshold)
    if not matcher.profiles:
        raise SystemExit("No profiles found. Run enroll-face before recognition.")

    detector = YuNetFaceDetector(args.detector_model.expanduser().resolve())
    embedder = SFaceEmbedder(args.embedding_model.expanduser().resolve())
    if args.image is not None:
        recognize_image(
            image_path=args.image.expanduser().resolve(),
            output_path=args.output.expanduser().resolve() if args.output else None,
            display=not args.no_display,
            detector=detector,
            embedder=embedder,
            matcher=matcher,
        )
    else:
        recognize_camera(args.camera, detector, embedder, matcher)


if __name__ == "__main__":
    main()

