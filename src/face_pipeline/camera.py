from __future__ import annotations

import json
import subprocess
import sys
import time

import cv2 as cv
from numpy.typing import NDArray
import numpy as np


class CameraError(RuntimeError):
    pass


def list_macos_cameras() -> tuple[str, ...]:
    """Return camera names reported by macOS, or an empty tuple if unavailable."""
    try:
        completed = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return ()
    devices = payload.get("SPCameraDataType", [])
    if not isinstance(devices, list):
        return ()
    return tuple(
        name
        for device in devices
        if isinstance(device, dict)
        and isinstance((name := device.get("_name")), str)
        and name.strip()
    )


def require_single_macos_webcam(camera_names: tuple[str, ...]) -> None:
    """Reject Continuity Camera and ambiguous external-camera configurations."""
    if not camera_names:
        return
    lowered = tuple(name.casefold() for name in camera_names)
    if any("iphone" in name or "continuity" in name for name in lowered):
        raise CameraError(
            "iPhone Continuity Camera is active. On the iPhone, turn off "
            "Settings > General > AirPlay & Continuity > Continuity Camera, "
            "then run this command again. This project only uses the built-in webcam."
        )
    built_in = [
        name
        for name in lowered
        if "facetime" in name or "built-in" in name or "built in" in name
    ]
    if not built_in:
        raise CameraError(
            "The built-in Mac webcam was not found. Keep the MacBook lid open and "
            "close other apps that may be using the camera."
        )
    if len(camera_names) != 1:
        raise CameraError(
            "Multiple cameras are connected. Disconnect external cameras so the "
            "built-in Mac webcam is the only available camera."
        )


def open_webcam() -> cv.VideoCapture:
    """Open the single local webcam; on macOS require the built-in camera."""
    backend = cv.CAP_AVFOUNDATION if sys.platform == "darwin" else cv.CAP_ANY
    if sys.platform == "darwin":
        require_single_macos_webcam(list_macos_cameras())
    camera = cv.VideoCapture(0, backend)
    if not camera.isOpened():
        camera.release()
        raise CameraError(
            "Could not open the built-in webcam. Close other camera apps, confirm "
            "camera permission for this terminal and keep the MacBook lid open."
        )
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
