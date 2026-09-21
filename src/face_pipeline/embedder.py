from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from face_pipeline.detector import FaceDetection, validate_frame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING_MODEL = (
    PROJECT_ROOT / "models" / "face_recognition_sface_2021dec.onnx"
)


def normalize_embedding(embedding: NDArray[np.floating]) -> NDArray[np.float32]:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        raise ValueError("Embedding cannot be empty")
    if not np.isfinite(vector).all():
        raise ValueError("Embedding must contain only finite values")

    magnitude = float(np.linalg.norm(vector))
    if magnitude == 0.0:
        raise ValueError("Embedding magnitude must be greater than zero")
    return vector / magnitude


class SFaceEmbedder:
    def __init__(self, model_path: Path = DEFAULT_EMBEDDING_MODEL) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"SFace model not found at {model_path}. Run download-face-models first."
            )
        self._recognizer = cv.FaceRecognizerSF.create(
            model=str(model_path),
            config="",
        )

    def align(
        self,
        frame: NDArray[np.uint8],
        detection: FaceDetection,
    ) -> NDArray[np.uint8]:
        validate_frame(frame)
        aligned = self._recognizer.alignCrop(frame, detection.model_output)
        if aligned is None or aligned.size == 0:
            raise RuntimeError("SFace could not align the detected face")
        return aligned

    def extract_from_aligned(
        self,
        aligned_face: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        validate_frame(aligned_face)
        features = self._recognizer.feature(aligned_face)
        if features is None:
            raise RuntimeError("SFace did not return an embedding")
        return normalize_embedding(features)

    def extract(
        self,
        frame: NDArray[np.uint8],
        detection: FaceDetection,
    ) -> NDArray[np.float32]:
        return self.extract_from_aligned(self.align(frame, detection))

