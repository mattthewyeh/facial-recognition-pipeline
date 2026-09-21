from __future__ import annotations

import sys
import time

import cv2 as cv
from numpy.typing import NDArray
import numpy as np


class CameraError(RuntimeError):
    pass


def open_camera(camera_index: int) -> cv.VideoCapture:
    backend = cv.CAP_AVFOUNDATION if sys.platform == "darwin" else cv.CAP_ANY
    camera = cv.VideoCapture(camera_index, backend)
    if not camera.isOpened():
        camera.release()
        raise CameraError(f"Could not open camera index {camera_index}")
    return camera


def read_camera_frame(
    camera: cv.VideoCapture,
    attempts: int = 30,
    retry_delay_seconds: float = 0.1,
) -> NDArray[np.uint8]:
    for attempt in range(attempts):
        success, frame = camera.read()
        if success and frame is not None:
            return frame
        if attempt + 1 < attempts:
            time.sleep(retry_delay_seconds)
    raise CameraError("Could not read a frame from the camera")

