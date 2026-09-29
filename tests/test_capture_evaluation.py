import json
from pathlib import Path

import pytest

from face_pipeline.capture_evaluation import (
    append_manifest_rows,
    next_sample_number,
    validate_component,
)


def test_validate_component_accepts_safe_labels() -> None:
    assert validate_component("matthew_1", "identity") == "matthew_1"
    assert validate_component("day-1", "session") == "day-1"


@pytest.mark.parametrize("value", ["", "../matthew", "matthew yeh", "/tmp/a"])
def test_validate_component_rejects_unsafe_labels(value: str) -> None:
    with pytest.raises(ValueError):
        validate_component(value, "identity")


def test_append_manifest_rows_preserves_existing_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "path": "enrollment/a/frame_001.jpg",
                    "identity": "a",
                    "split": "enrollment",
                    "session": "day1",
                }
            ]
        ),
        encoding="utf-8",
    )
    new_row = {
        "path": "validation/a/frame_001.jpg",
        "identity": "a",
        "split": "validation",
        "session": "day2",
    }

    append_manifest_rows(manifest, [new_row])

    assert json.loads(manifest.read_text(encoding="utf-8"))[-1] == new_row


def test_append_manifest_rows_rejects_duplicate_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    row = {
        "path": "enrollment/a/frame_001.jpg",
        "identity": "a",
        "split": "enrollment",
        "session": "day1",
    }
    append_manifest_rows(manifest, [row])

    with pytest.raises(ValueError, match="already contains"):
        append_manifest_rows(manifest, [row])


def test_next_sample_number_continues_sequence(tmp_path: Path) -> None:
    (tmp_path / "frame_002.jpg").touch()
    (tmp_path / "frame_009.jpg").touch()
    (tmp_path / "other.jpg").touch()

    assert next_sample_number(tmp_path) == 10
