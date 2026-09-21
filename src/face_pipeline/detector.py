from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETECTOR_MODEL = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
EXPECTED_YUNET_VALUES = 15


@dataclass(frozen=True)
class FaceDetection:
    x: int
    y: int
    width: int
    height: int
    landmarks: NDArray[np.float32]
    score: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    @classmethod
    def from_yunet_row(
        cls,
        row: NDArray[np.float32],
        frame_width: int,
        frame_height: int,
    ) -> FaceDetection:
        values = np.asarray(row, dtype=np.float32).reshape(-1)
        if values.size != EXPECTED_YUNET_VALUES:
            raise ValueError(
                f"Expected {EXPECTED_YUNET_VALUES} YuNet values, received {values.size}"
            )

        raw_x, raw_y, raw_width, raw_height = values[:4]
        left = max(0, int(round(float(raw_x))))
        top = max(0, int(round(float(raw_y))))
        right = min(frame_width, int(round(float(raw_x + raw_width))))
        bottom = min(frame_height, int(round(float(raw_y + raw_height))))

        return cls(
            x=left,
            y=top,
            width=max(0, right - left),
            height=max(0, bottom - top),
            landmarks=values[4:14].reshape(5, 2).copy(),
            score=float(values[14]),
        )


class YuNetFaceDetector:
    def __init__(
        self,
        model_path: Path = DEFAULT_DETECTOR_MODEL,
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"YuNet model not found at {model_path}. Run download-face-models first."
            )
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")

        self._detector = cv.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )

    def detect(self, frame: NDArray[np.uint8]) -> list[FaceDetection]:
        validate_frame(frame)
        frame_height, frame_width = frame.shape[:2]
        self._detector.setInputSize((frame_width, frame_height))
        _, faces = self._detector.detect(frame)

        if faces is None:
            return []

        detections = [
            FaceDetection.from_yunet_row(face, frame_width, frame_height)
            for face in faces
        ]
        return [detection for detection in detections if detection.width and detection.height]


def validate_frame(frame: NDArray[np.uint8]) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError("Frame must be a NumPy array")
    if frame.dtype != np.uint8:
        raise ValueError("Frame must use uint8 pixels")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must be a BGR color image with three channels")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("Frame dimensions must be greater than zero")


def draw_detections(
    frame: NDArray[np.uint8],
    detections: list[FaceDetection],
) -> NDArray[np.uint8]:
    validate_frame(frame)
    annotated = frame.copy()

    for detection in detections:
        top_left = (detection.x, detection.y)
        bottom_right = (
            detection.x + detection.width,
            detection.y + detection.height,
        )
        cv.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 2)
        for landmark in detection.landmarks:
            point = tuple(np.rint(landmark).astype(int))
            cv.circle(annotated, point, 2, (0, 255, 255), -1)

        label_y = max(20, detection.y - 8)
        cv.putText(
            annotated,
            f"face {detection.score:.2f}",
            (detection.x, label_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )

    return annotated

