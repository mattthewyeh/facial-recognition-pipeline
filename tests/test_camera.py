import numpy as np
import pytest

import face_pipeline.camera as camera_module
from face_pipeline.camera import (
    CameraError,
    open_webcam,
    read_camera_frame,
    require_single_macos_webcam,
)


class FakeCamera:
    def __init__(self, opened: bool, reads: list[tuple[bool, object]]) -> None:
        self.opened = opened
        self.reads = iter(reads)
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        return next(self.reads)

    def release(self) -> None:
        self.released = True


def test_open_webcam_uses_only_index_zero_on_macos(monkeypatch) -> None:
    fake = FakeCamera(opened=True, reads=[])
    requested: dict[str, int] = {}

    def fake_video_capture(index: int, backend: int) -> FakeCamera:
        requested["index"] = index
        requested["backend"] = backend
        return fake

    monkeypatch.setattr(camera_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        camera_module,
        "list_macos_cameras",
        lambda: ("FaceTime HD Camera",),
    )
    monkeypatch.setattr(camera_module.cv, "VideoCapture", fake_video_capture)

    result = open_webcam()

    assert result is fake
    assert requested == {"index": 0, "backend": camera_module.cv.CAP_AVFOUNDATION}


def test_open_webcam_releases_failed_capture(monkeypatch) -> None:
    fake = FakeCamera(opened=False, reads=[])
    monkeypatch.setattr(camera_module.sys, "platform", "linux")
    monkeypatch.setattr(
        camera_module.cv,
        "VideoCapture",
        lambda _index, _backend: fake,
    )

    with pytest.raises(CameraError, match="Could not open"):
        open_webcam()

    assert fake.released is True


def test_macos_webcam_rejects_continuity_camera() -> None:
    with pytest.raises(CameraError, match="Continuity Camera is active"):
        require_single_macos_webcam(("FaceTime HD Camera", "iPhone Camera"))


def test_macos_webcam_rejects_external_camera() -> None:
    with pytest.raises(CameraError, match="Multiple cameras"):
        require_single_macos_webcam(("FaceTime HD Camera", "USB Camera"))


def test_macos_webcam_requires_built_in_camera() -> None:
    with pytest.raises(CameraError, match="built-in Mac webcam was not found"):
        require_single_macos_webcam(("USB Camera",))


def test_macos_webcam_accepts_only_facetime_camera() -> None:
    require_single_macos_webcam(("FaceTime HD Camera",))


def test_read_camera_frame_retries_startup_failure() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    fake = FakeCamera(opened=True, reads=[(False, None), (True, frame)])

    result = read_camera_frame(fake, attempts=2, retry_delay_seconds=0.0)

    assert result is frame


def test_read_camera_frame_raises_after_attempts() -> None:
    fake = FakeCamera(opened=True, reads=[(False, None), (False, None)])

    with pytest.raises(CameraError, match="Could not read"):
        read_camera_frame(fake, attempts=2, retry_delay_seconds=0.0)
