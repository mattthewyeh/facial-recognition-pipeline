from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2 as cv

from face_pipeline.camera import CameraError, open_camera, read_camera_frame
from face_pipeline.detector import (
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_SCORE_THRESHOLD,
    FaceDetection,
    YuNetFaceDetector,
    draw_detections,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect faces in an image or webcam feed with YuNet."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Path to an input image")
    source.add_argument("--camera", type=int, help="Webcam index, usually 0")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_DETECTOR_MODEL,
        help=f"YuNet ONNX model path (default: {DEFAULT_DETECTOR_MODEL})",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help=(
            "Minimum YuNet detection confidence "
            f"(default: {DEFAULT_SCORE_THRESHOLD})"
        ),
    )
    parser.add_argument("--output", type=Path, help="Optional annotated image path")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open an image preview window",
    )
    return parser.parse_args()


def print_detections(detections: list[FaceDetection]) -> None:
    print(f"Detected faces: {len(detections)}")
    for index, detection in enumerate(detections, start=1):
        print(
            f"  {index}: box={detection.box} "
            f"confidence={detection.score:.4f}"
        )


def detect_image(
    detector: YuNetFaceDetector,
    image_path: Path,
    output_path: Path | None,
    display: bool,
) -> None:
    frame = cv.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path}")

    detections = detector.detect(frame)
    print_detections(detections)
    annotated = draw_detections(frame, detections)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv.imwrite(str(output_path), annotated):
            raise SystemExit(f"Could not write output image: {output_path}")
        print(f"Saved annotated image: {output_path}")

    if display:
        cv.imshow("YuNet face detection", annotated)
        cv.waitKey(0)
        cv.destroyAllWindows()


def detect_camera(detector: YuNetFaceDetector, camera_index: int) -> None:
    try:
        camera = open_camera(camera_index)
    except CameraError as error:
        raise SystemExit(str(error)) from error

    previous_time = time.perf_counter()
    smoothed_fps = 0.0
    print("Press q in the camera window to quit.")

    try:
        while True:
            try:
                frame = read_camera_frame(camera)
            except CameraError as error:
                raise SystemExit(str(error)) from error

            detections = detector.detect(frame)
            annotated = draw_detections(frame, detections)

            current_time = time.perf_counter()
            elapsed = max(current_time - previous_time, 1e-9)
            current_fps = 1.0 / elapsed
            smoothed_fps = current_fps if smoothed_fps == 0.0 else (
                0.9 * smoothed_fps + 0.1 * current_fps
            )
            previous_time = current_time

            cv.putText(
                annotated,
                f"faces: {len(detections)}  fps: {smoothed_fps:.1f}",
                (12, 28),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )
            cv.imshow("YuNet face detection", annotated)
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

    detector = YuNetFaceDetector(
        model_path=args.model.expanduser().resolve(),
        score_threshold=args.score_threshold,
    )

    if args.image is not None:
        detect_image(
            detector,
            args.image.expanduser().resolve(),
            args.output.expanduser().resolve() if args.output else None,
            display=not args.no_display,
        )
    else:
        detect_camera(detector, args.camera)


if __name__ == "__main__":
    main()
