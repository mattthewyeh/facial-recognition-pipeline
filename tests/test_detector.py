from pathlib import Path

import numpy as np
import pytest

from face_pipeline.detector import (
    DEFAULT_DETECTOR_MODEL,
    FaceDetection,
    YuNetFaceDetector,
    draw_detections,
    validate_frame,
)


def yunet_row() -> np.ndarray:
    return np.array(
        [
            -4.0,
            5.0,
            30.0,
            40.0,
            8.0,
            15.0,
            18.0,
            15.0,
            13.0,
            22.0,
            9.0,
            32.0,
            17.0,
            32.0,
            0.96,
        ],
        dtype=np.float32,
    )


def test_detection_parses_landmarks_and_clamps_box() -> None:
    detection = FaceDetection.from_yunet_row(
        yunet_row(),
        frame_width=20,
        frame_height=30,
    )

    assert detection.box == (0, 5, 20, 25)
    assert detection.landmarks.shape == (5, 2)
    assert detection.score == pytest.approx(0.96)


def test_detection_rejects_unexpected_model_output() -> None:
    with pytest.raises(ValueError, match="Expected 15 YuNet values"):
        FaceDetection.from_yunet_row(
            np.zeros(14, dtype=np.float32),
            frame_width=100,
            frame_height=100,
        )


def test_validate_frame_rejects_grayscale_image() -> None:
    with pytest.raises(ValueError, match="BGR color image"):
        validate_frame(np.zeros((40, 40), dtype=np.uint8))


def test_draw_detections_does_not_modify_input() -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    original = frame.copy()
    detection = FaceDetection.from_yunet_row(
        yunet_row(),
        frame_width=50,
        frame_height=50,
    )

    annotated = draw_detections(frame, [detection])

    assert np.array_equal(frame, original)
    assert not np.array_equal(annotated, original)


def test_detector_rescales_model_coordinates_to_original_frame() -> None:
    class FakeDetector:
        def __init__(self) -> None:
            self.input_size: tuple[int, int] | None = None

        def setInputSize(self, input_size: tuple[int, int]) -> None:
            self.input_size = input_size

        def detect(self, frame: np.ndarray) -> tuple[None, np.ndarray]:
            assert frame.shape == (360, 640, 3)
            return None, np.array(
                [[0, 5, 20, 40, 8, 15, 18, 15, 13, 22, 9, 32, 17, 32, 0.96]],
                dtype=np.float32,
            )

    detector = YuNetFaceDetector.__new__(YuNetFaceDetector)
    detector._max_input_dimension = 640
    detector._detector = FakeDetector()

    detections = detector.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))

    assert detector._detector.input_size == (640, 360)
    assert detections[0].box == (0, 15, 60, 120)
    assert detections[0].landmarks[0] == pytest.approx([24, 45])


@pytest.mark.skipif(
    not DEFAULT_DETECTOR_MODEL.exists(),
    reason="YuNet model has not been downloaded",
)
def test_detector_processes_blank_frame() -> None:
    detector = YuNetFaceDetector(model_path=Path(DEFAULT_DETECTOR_MODEL))
    detections = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
    assert detections == []
