"""Capture full webcam frames for held-out recognition evaluation."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2 as cv

from face_pipeline.camera import CameraError, open_camera, read_camera_frame
from face_pipeline.detector import (
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_SCORE_THRESHOLD,
    YuNetFaceDetector,
    draw_detections,
)


DEFAULT_OUTPUT_ROOT = Path("data/evaluation")
VALID_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture original webcam frames for an evaluation split."
    )
    parser.add_argument("--identity", required=True, help="Pseudonymous identity label")
    parser.add_argument(
        "--split",
        required=True,
        choices=("enrollment", "validation", "test"),
    )
    parser.add_argument("--session", required=True, help="Unique capture-session label")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--samples", type=int, default=8, help="Frames to capture (default: 8)")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Dataset directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument("--detector-model", type=Path, default=DEFAULT_DETECTOR_MODEL)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
    )
    return parser.parse_args()


def validate_component(value: str, field: str) -> str:
    if not VALID_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{field} must start with a letter or number and contain only "
            "letters, numbers, underscores or hyphens"
        )
    return value


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Existing manifest is not a JSON array: {path}")
    return rows


def append_manifest_rows(
    manifest_path: Path,
    rows: list[dict[str, str]],
) -> None:
    existing = load_manifest(manifest_path)
    existing_paths = {row.get("path") for row in existing if isinstance(row, dict)}
    duplicates = [row["path"] for row in rows if row["path"] in existing_paths]
    if duplicates:
        raise ValueError(f"Manifest already contains: {duplicates[0]}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps([*existing, *rows], indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)


def next_sample_number(directory: Path) -> int:
    numbers = []
    for path in directory.glob("frame_*.jpg"):
        suffix = path.stem.removeprefix("frame_")
        if suffix.isdigit():
            numbers.append(int(suffix))
    return max(numbers, default=0) + 1


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    try:
        identity = validate_component(args.identity, "--identity")
        session = validate_component(args.session, "--session")
    except ValueError as error:
        raise SystemExit(str(error)) from error

    output_root = args.output_root.expanduser().resolve()
    capture_dir = output_root / args.split / identity / session
    capture_dir.mkdir(parents=True, exist_ok=True)
    first_number = next_sample_number(capture_dir)

    detector = YuNetFaceDetector(
        args.detector_model.expanduser().resolve(),
        score_threshold=args.score_threshold,
    )
    try:
        camera = open_camera(args.camera)
    except CameraError as error:
        raise SystemExit(str(error)) from error

    captured: list[tuple[Path, object]] = []
    print("Press SPACE to save a full frame and q to finish without saving pending frames.")
    try:
        while len(captured) < args.samples:
            try:
                frame = read_camera_frame(camera)
            except CameraError as error:
                raise SystemExit(str(error)) from error
            frame = cv.flip(frame, 1)
            detections = detector.detect(frame)
            annotated = draw_detections(frame, detections)
            instruction = (
                f"{args.split} | {identity} | {len(captured)}/{args.samples} | "
                "one face | SPACE capture | q cancel"
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
            cv.imshow("Evaluation capture", annotated)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                raise SystemExit("Capture cancelled; no new frames were saved")
            if key == ord(" "):
                if len(detections) != 1:
                    print(f"Capture skipped: detected {len(detections)} faces")
                    continue
                number = first_number + len(captured)
                destination = capture_dir / f"frame_{number:03d}.jpg"
                captured.append((destination, frame.copy()))
                print(f"Captured frame {len(captured)}/{args.samples}")
    finally:
        camera.release()
        cv.destroyAllWindows()

    written_paths: list[Path] = []
    try:
        for destination, frame in captured:
            if not cv.imwrite(str(destination), frame, [cv.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Could not write image: {destination}")
            written_paths.append(destination)
        rows = [
            {
                "path": destination.relative_to(output_root).as_posix(),
                "identity": identity,
                "split": args.split,
                "session": session,
            }
            for destination in written_paths
        ]
        append_manifest_rows(output_root / "manifest.json", rows)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        for path in written_paths:
            path.unlink(missing_ok=True)
        raise SystemExit(str(error)) from error

    print(f"Saved {len(written_paths)} full frames to {capture_dir}")
    print(f"Updated {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
