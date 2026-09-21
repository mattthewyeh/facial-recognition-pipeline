from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from face_pipeline.detector import validate_frame
from face_pipeline.embedder import normalize_embedding


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_DIR = PROJECT_ROOT / "data" / "profiles"
EMBEDDING_MODEL_NAME = "SFace 2021-12"


@dataclass(frozen=True)
class Profile:
    profile_id: str
    name: str
    created_at: str
    embedding_model: str
    sample_count: int
    directory: Path


@dataclass(frozen=True)
class LoadedProfile:
    profile: Profile
    embeddings: NDArray[np.float32]

    @property
    def centroid(self) -> NDArray[np.float32]:
        return normalize_embedding(np.mean(self.embeddings, axis=0))


class ProfileStore:
    def __init__(self, root: Path = DEFAULT_PROFILE_DIR) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        name: str,
        embeddings: list[NDArray[np.floating]],
        aligned_faces: list[NDArray[np.uint8]],
    ) -> LoadedProfile:
        clean_name = normalize_name(name)
        if any(profile.profile.name.casefold() == clean_name.casefold() for profile in self.load_all()):
            raise ValueError(f"A profile named {clean_name!r} already exists")
        if not embeddings:
            raise ValueError("At least one embedding is required")
        if len(embeddings) != len(aligned_faces):
            raise ValueError("Each embedding must have one aligned face image")

        normalized_embeddings = np.stack(
            [normalize_embedding(embedding) for embedding in embeddings]
        ).astype(np.float32)
        if len({embedding.size for embedding in normalized_embeddings}) != 1:
            raise ValueError("All embeddings must have the same dimensions")
        for aligned_face in aligned_faces:
            validate_frame(aligned_face)

        profile_id = uuid.uuid4().hex
        profile_dir = self.root / profile_id
        temporary_dir = self.root / f".{profile_id}.tmp"
        samples_dir = temporary_dir / "samples"
        created_at = datetime.now(timezone.utc).isoformat()
        metadata = {
            "profile_id": profile_id,
            "name": clean_name,
            "created_at": created_at,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "sample_count": len(embeddings),
        }

        try:
            samples_dir.mkdir(parents=True)
            np.save(temporary_dir / "embeddings.npy", normalized_embeddings)
            (temporary_dir / "profile.json").write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )
            for index, aligned_face in enumerate(aligned_faces, start=1):
                sample_path = samples_dir / f"sample_{index:02d}.jpg"
                if not cv.imwrite(str(sample_path), aligned_face):
                    raise OSError(f"Could not save enrollment image: {sample_path}")
            temporary_dir.rename(profile_dir)
        except Exception:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise

        return self.load(profile_dir)

    def load(self, profile_dir: Path) -> LoadedProfile:
        metadata_path = profile_dir / "profile.json"
        embeddings_path = profile_dir / "embeddings.npy"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            embeddings = np.load(embeddings_path, allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid profile directory: {profile_dir}") from error

        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(f"Profile embeddings must be a non-empty matrix: {profile_dir}")
        normalized_embeddings = np.stack(
            [normalize_embedding(embedding) for embedding in embeddings]
        ).astype(np.float32)

        required_fields = {
            "profile_id",
            "name",
            "created_at",
            "embedding_model",
            "sample_count",
        }
        if not required_fields.issubset(metadata):
            raise ValueError(f"Profile metadata is missing required fields: {profile_dir}")
        if int(metadata["sample_count"]) != normalized_embeddings.shape[0]:
            raise ValueError(f"Profile sample count does not match embeddings: {profile_dir}")

        profile = Profile(
            profile_id=str(metadata["profile_id"]),
            name=normalize_name(str(metadata["name"])),
            created_at=str(metadata["created_at"]),
            embedding_model=str(metadata["embedding_model"]),
            sample_count=int(metadata["sample_count"]),
            directory=profile_dir,
        )
        return LoadedProfile(profile=profile, embeddings=normalized_embeddings)

    def load_all(self) -> list[LoadedProfile]:
        profiles = [
            self.load(path)
            for path in self.root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]
        return sorted(profiles, key=lambda item: item.profile.name.casefold())


def normalize_name(name: str) -> str:
    clean_name = " ".join(name.split())
    if not clean_name:
        raise ValueError("Profile name cannot be blank")
    if len(clean_name) > 100:
        raise ValueError("Profile name cannot exceed 100 characters")
    return clean_name

