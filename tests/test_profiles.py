import json
from pathlib import Path

import numpy as np
import pytest

from face_pipeline.profiles import ProfileStore, normalize_name


def aligned_face(color: int) -> np.ndarray:
    return np.full((112, 112, 3), color, dtype=np.uint8)


def test_profile_store_round_trip(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profiles")

    created = store.create(
        name="  Matthew   Yeh  ",
        embeddings=[
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.8, 0.2], dtype=np.float32),
        ],
        aligned_faces=[aligned_face(20), aligned_face(40)],
    )

    assert created.profile.name == "Matthew Yeh"
    assert created.profile.sample_count == 2
    assert created.embeddings.shape == (2, 2)
    assert np.linalg.norm(created.centroid) == pytest.approx(1.0)
    assert (created.profile.directory / "profile.json").is_file()
    assert (created.profile.directory / "embeddings.npy").is_file()
    assert len(list((created.profile.directory / "samples").glob("*.jpg"))) == 2

    metadata = json.loads(
        (created.profile.directory / "profile.json").read_text(encoding="utf-8")
    )
    assert metadata["name"] == "Matthew Yeh"
    assert store.load_all()[0].profile.profile_id == created.profile.profile_id


def test_profile_store_rejects_duplicate_names(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profiles")
    store.create(
        "Matthew Yeh",
        [np.array([1.0, 0.0], dtype=np.float32)],
        [aligned_face(20)],
    )

    with pytest.raises(ValueError, match="already exists"):
        store.create(
            "  MATTHEW yeh ",
            [np.array([0.9, 0.1], dtype=np.float32)],
            [aligned_face(40)],
        )


def test_profile_store_requires_matching_samples(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profiles")

    with pytest.raises(ValueError, match="one aligned face"):
        store.create(
            "Matthew",
            [np.array([1.0, 0.0], dtype=np.float32)],
            [],
        )


def test_normalize_name_rejects_blank_name() -> None:
    with pytest.raises(ValueError, match="cannot be blank"):
        normalize_name("   ")

