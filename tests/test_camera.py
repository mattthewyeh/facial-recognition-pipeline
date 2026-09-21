import numpy as np
import pytest

import face_pipeline.camera as camera_module
from face_pipeline.camera import CameraError, open_camera, read_camera_frame


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


def test_open_camera_uses_avfoundation_on_macos(monkeypatch) -> None:
    fake = FakeCamera(opened=True, reads=[])
    requested: dict[str, int] = {}

    def fake_video_capture(index: int, backend: int) -> FakeCamera:
        requested["index"] = index
        requested["backend"] = backend
        return fake

    monkeypatch.setattr(camera_module.sys, "platform", "darwin")
    monkeypatch.setattr(camera_module.cv, "VideoCapture", fake_video_capture)

    result = open_camera(2)

    assert result is fake
    assert requested == {"index": 2, "backend": camera_module.cv.CAP_AVFOUNDATION}


def test_open_camera_releases_failed_capture(monkeypatch) -> None:
    fake = FakeCamera(opened=False, reads=[])
    monkeypatch.setattr(
        camera_module.cv,
        "VideoCapture",
        lambda _index, _backend: fake,
    )

    with pytest.raises(CameraError, match="Could not open"):
        open_camera(0)

    assert fake.released is True


def test_read_camera_frame_retries_startup_failure() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    fake = FakeCamera(opened=True, reads=[(False, None), (True, frame)])

    result = read_camera_frame(fake, attempts=2, retry_delay_seconds=0.0)

    assert result is frame


def test_read_camera_frame_raises_after_attempts() -> None:
    fake = FakeCamera(opened=True, reads=[(False, None), (False, None)])

    with pytest.raises(CameraError, match="Could not read"):
        read_camera_frame(fake, attempts=2, retry_delay_seconds=0.0)

